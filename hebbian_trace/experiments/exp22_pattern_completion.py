"""Exp 22: Pattern Completion via T_auto (CA3 autoassociative trace).

Tests two-step retrieval: Q → T_auto → Q_corrected → T_v → V.

T_auto stores template-driven Q→Q pairs (e.g., Q("I")→Q("name"))
so that paraphrased questions ("Who am I?") can retrieve facts stored
under the canonical concept word ("name").

Protocol:
1. Extract Q→Q pairs from exp12's RAW_QUESTION_VARIANTS
2. Write T_auto pairs once (static template knowledge)
3. Per episode: write facts to T_v, query with variant questions
4. Compare: no T_auto (standard) vs with T_auto (completion channel)

Expected: misaligned queries ("Who am I?") should improve dramatically.
Aligned queries ("What is my name?") should stay the same or improve.
"""

import random
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types,
    get_linking_bpe_ids,
    GPT2FactType,
    _get_all_entity_ids,
    tokenize_fact,
)
from .exp12_realistic_benchmarks import (
    RAW_QUESTION_VARIANTS,
    build_question_variants,
    QuestionVariant,
)


# ── Concept word mapping ──────────────────────────────────────────────
# These are the Q keys stored in T_v via shift-1 mechanism.
# Determined by fact template structure:
#   "My name is {X}." → linking "is" at pos 3 → Q_store = Q[pos 2] = Q("name")
#   "I live in {X}."  → linking "in" at pos 2 → Q_store = Q[pos 1] = Q("live")
#   etc.

CONCEPT_WORDS: dict[str, str] = {
    "name": "name",
    "city": "live",
    "company": "work",
    "color": "color",
    "food": "food",
    "pet": "pet",
    "country": "country",
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── T_auto pair extraction ────────────────────────────────────────────

@dataclass
class AutoPair:
    """One Q→Q pair for T_auto."""
    variant_token: str    # e.g., " I"
    variant_id: int       # BPE ID of variant token
    concept_token: str    # e.g., " name"
    concept_id: int       # BPE ID of concept token
    fact_type: str        # e.g., "name"
    category: str         # "misaligned" or "semantic"
    question: str         # source question text


def extract_auto_pairs(
    tokenizer: GPT2Tokenizer,
) -> list[AutoPair]:
    """Extract Q→Q pairs from RAW_QUESTION_VARIANTS.

    For each misaligned/semantic variant, maps:
        Q_addr_token (what appears at pos[-2]) → concept_word (what T_v expects)

    Skips aligned variants (they already have the correct Q_addr).
    Warns about collisions (same variant token → different concepts).
    """
    variants = build_question_variants(tokenizer)
    pairs: list[AutoPair] = []
    seen: dict[int, list[str]] = {}  # variant_id → [concept_words]

    for fact_type, type_variants in variants.items():
        concept_word = CONCEPT_WORDS[fact_type]
        # Get concept BPE ID
        concept_ids = tokenizer.encode(" " + concept_word,
                                       add_special_tokens=False)
        if len(concept_ids) != 1:
            print(f"  WARNING: concept '{concept_word}' is multi-token, "
                  f"skipping {fact_type}")
            continue
        concept_id = concept_ids[0]

        for v in type_variants:
            if v.category == "aligned":
                continue  # skip — already has correct Q_addr

            # Check if q_addr_token is single-token
            if v.q_addr_id < 0:
                continue

            # Track collisions
            if v.q_addr_id in seen:
                existing = seen[v.q_addr_id]
                if concept_word not in existing:
                    existing.append(concept_word)
            else:
                seen[v.q_addr_id] = [concept_word]

            pairs.append(AutoPair(
                variant_token=v.q_addr_token,
                variant_id=v.q_addr_id,
                concept_token=" " + concept_word,
                concept_id=concept_id,
                fact_type=fact_type,
                category=v.category,
                question=v.text,
            ))

    # Report collisions
    for vid, concepts in seen.items():
        if len(concepts) > 1:
            token = tokenizer.decode([vid])
            print(f"  COLLISION: Q('{token}') → {concepts}")

    return pairs


# ── Evaluation ────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    """Result for one question variant."""
    question: str
    category: str
    fact_type: str
    accuracy: float
    n_correct: int
    n_total: int


def run_variant_eval(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    n_facts: int,
    n_episodes: int,
    auto_pairs: list[AutoPair] | None,
    completion_alpha: float = 1.0,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[VariantResult]:
    """Evaluate cross-context retrieval with question variants.

    For each episode:
    1. Reset T_v (keep T_auto if pre-written)
    2. Write N facts to T_v
    3. Query each fact with its aligned question AND all variant questions
    4. Track accuracy per variant

    Args:
        model: GPT2WithTrace instance
        tokenizer: GPT-2 tokenizer
        fact_types: available fact types
        n_facts: facts per episode
        n_episodes: number of episodes
        auto_pairs: if not None, enable T_auto with these pairs
        completion_alpha: weight for completion channel
        alpha: trace alpha
        seed: random seed

    Returns:
        list of VariantResult, one per question variant
    """
    rng = random.Random(seed)
    model.eval()
    device = next(model.parameters()).device

    # Get all entity IDs for answer restriction
    entity_ids = _get_all_entity_ids(fact_types)

    # Build question variants
    variants = build_question_variants(tokenizer)

    # Set alpha
    model.trace.alpha = alpha

    # Write T_auto once if provided
    if auto_pairs is not None:
        model.trace.autoassociative_traces.zero_()
        pair_tuples = [(p.variant_id, p.concept_id) for p in auto_pairs]
        model.write_auto_pairs(pair_tuples)
        model.set_auto_mode(True, completion_alpha)
    else:
        model.set_auto_mode(False)

    # Track results per (fact_type, question_text)
    # key: (fact_type, question_text) → [correct, total]
    results_map: dict[tuple[str, str, str], list[int]] = {}

    for ep_idx in range(n_episodes):
        # Reset T_v only (T_auto persists)
        model.trace.value_traces.zero_()

        # Select fact types for this episode
        if n_facts <= len(fact_types):
            selected = rng.sample(fact_types, n_facts)
        else:
            selected = [rng.choice(fact_types) for _ in range(n_facts)]

        # Select entities
        episode_facts: list[tuple[GPT2FactType, str, int]] = []
        for ft in selected:
            entity_name, entity_id = rng.choice(ft.entities)
            episode_facts.append((ft, entity_name, entity_id))

        # Write phase: encode facts individually
        model.set_trace_mode(use=False, update=True)
        for ft, entity_name, entity_id in episode_facts:
            template = ft.fact_templates[0]
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            input_tensor = torch.tensor(
                [fact_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)

        # Read phase: query with all variants
        model.set_trace_mode(use=True, update=False)
        for ft, entity_name, entity_id in episode_facts:
            if ft.name not in variants:
                continue

            for v in variants[ft.name]:
                q_ids = v.bpe_ids
                input_tensor = torch.tensor(
                    [q_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(input_tensor)

                pred_logits = logits[0, -1, :]
                entity_logits = pred_logits[entity_ids]
                pred_id = entity_ids[entity_logits.argmax().item()]
                correct = int(pred_id == entity_id)

                key = (ft.name, v.category, v.text)
                if key not in results_map:
                    results_map[key] = [0, 0]
                results_map[key][0] += correct
                results_map[key][1] += 1

    # Build results
    results = []
    for (fact_type, category, question), (n_correct, n_total) in \
            sorted(results_map.items()):
        results.append(VariantResult(
            question=question,
            category=category,
            fact_type=fact_type,
            accuracy=n_correct / max(n_total, 1),
            n_correct=n_correct,
            n_total=n_total,
        ))

    return results


def print_results(results: list[VariantResult], label: str):
    """Pretty-print variant results grouped by category."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    # Group by category
    by_cat: dict[str, list[VariantResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat in ["aligned", "misaligned", "semantic"]:
        if cat not in by_cat:
            continue
        cat_results = by_cat[cat]
        cat_acc = sum(r.n_correct for r in cat_results) / \
                  max(sum(r.n_total for r in cat_results), 1)

        print(f"\n  [{cat.upper()}] overall: {cat_acc:.1%}")
        for r in cat_results:
            print(f"    {r.fact_type:>8} | {r.question:<35} | "
                  f"{r.accuracy:.0%} ({r.n_correct}/{r.n_total})")

    # Overall
    total_correct = sum(r.n_correct for r in results)
    total = sum(r.n_total for r in results)
    print(f"\n  OVERALL: {total_correct / max(total, 1):.1%} "
          f"({total_correct}/{total})")


# ── Diagnostic: T_auto round-trip ─────────────────────────────────────

def diagnose_auto_roundtrip(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    auto_pairs: list[AutoPair],
):
    """Check if T_auto correctly maps variant Q → concept Q.

    For each pair, computes:
    1. Q_variant_exp → T_auto → Q_retrieved (base space)
    2. cosine(Q_retrieved, Q_concept) — should be high
    3. cosine(Q_retrieved, Q_other_concepts) — should be low
    """
    import torch.nn.functional as F

    print("\n" + "=" * 70)
    print("  T_auto Round-Trip Diagnostic")
    print("=" * 70)

    # Collect all concept Q vectors
    concept_qs: dict[str, torch.Tensor] = {}
    for fact_type, concept_word in CONCEPT_WORDS.items():
        cid = tokenizer.encode(" " + concept_word,
                               add_special_tokens=False)
        if len(cid) == 1:
            q = model.trace.compute_q_for_token(model._wte, cid[0])
            concept_qs[fact_type] = q  # (H, d_trace)

    for pair in auto_pairs:
        # Compute Q_variant
        Q_var = model.trace.compute_q_for_token(
            model._wte, pair.variant_id)  # (H, d_trace)

        # Expand for T_auto addressing
        if model.trace._pattern_sep_enabled:
            Q_var_exp = model.trace._sparse_expand(
                Q_var.unsqueeze(0).unsqueeze(2))
            Q_var_exp = Q_var_exp.squeeze(0).squeeze(1)  # (H, expanded)
        else:
            Q_var_exp = Q_var

        # Read T_auto: Q_var_exp @ T_auto → Q_retrieved
        Q_retrieved = torch.einsum(
            'hp,hpq->hq',
            Q_var_exp, model.trace.autoassociative_traces)  # (H, d_trace)

        # Cosine with target concept
        Q_target = concept_qs[pair.fact_type]
        cos_target = F.cosine_similarity(
            Q_retrieved.flatten().unsqueeze(0),
            Q_target.flatten().unsqueeze(0),
        ).item()

        # Cosine with all other concepts
        best_other = -1.0
        best_other_name = ""
        for other_type, Q_other in concept_qs.items():
            if other_type == pair.fact_type:
                continue
            cos = F.cosine_similarity(
                Q_retrieved.flatten().unsqueeze(0),
                Q_other.flatten().unsqueeze(0),
            ).item()
            if cos > best_other:
                best_other = cos
                best_other_name = other_type

        margin = cos_target - best_other
        verdict = "OK" if margin > 0.1 else ("weak" if margin > 0 else "FAIL")

        print(f"  Q('{pair.variant_token}')→Q('{pair.concept_token}'): "
              f"cos_target={cos_target:.3f}, "
              f"best_other={best_other:.3f} ({best_other_name}), "
              f"margin={margin:.3f} [{verdict}]")


# ── Main experiment ───────────────────────────────────────────────────

def run_experiment(
    n_facts_list: list[int] = [1, 3, 5, 7],
    n_episodes: int = 30,
    alpha: float = 0.5,
    completion_alpha: float = 1.0,
    seed: int = 42,
):
    """Run pattern completion experiment.

    Compares standard cross-context (no T_auto) vs T_auto-augmented
    across aligned, misaligned, and semantic question variants.
    """
    device = get_device()
    print(f"Device: {device}")

    # Load model
    print("Loading GPT-2...")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
    )
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    # Enable pattern separation
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    print(f"\nFact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"  {ft.name}: {len(ft.entities)} entities")

    # Extract T_auto pairs
    print("\nExtracting T_auto pairs...")
    auto_pairs = extract_auto_pairs(tokenizer)
    print(f"  Total pairs: {len(auto_pairs)}")
    for p in auto_pairs:
        print(f"    [{p.category}] {p.fact_type}: "
              f"Q('{p.variant_token}') → Q('{p.concept_token}') "
              f"  ({p.question})")

    # Diagnostic: write T_auto and check round-trip
    model.trace.autoassociative_traces.zero_()
    pair_tuples = [(p.variant_id, p.concept_id) for p in auto_pairs]
    model.write_auto_pairs(pair_tuples)
    diagnose_auto_roundtrip(model, tokenizer, auto_pairs)

    # Run for each n_facts
    for n_facts in n_facts_list:
        print(f"\n{'#' * 70}")
        print(f"  n_facts = {n_facts}")
        print(f"{'#' * 70}")

        # Phase 1: Standard (no T_auto)
        results_std = run_variant_eval(
            model, tokenizer, fact_types,
            n_facts=n_facts, n_episodes=n_episodes,
            auto_pairs=None,  # no completion
            alpha=alpha, seed=seed,
        )
        print_results(results_std, f"STANDARD (no T_auto), n={n_facts}")

        # Phase 2: With T_auto
        results_auto = run_variant_eval(
            model, tokenizer, fact_types,
            n_facts=n_facts, n_episodes=n_episodes,
            auto_pairs=auto_pairs,
            completion_alpha=completion_alpha,
            alpha=alpha, seed=seed,
        )
        print_results(results_auto, f"T_AUTO (completion), n={n_facts}")

        # Summary comparison
        print(f"\n  COMPARISON (n={n_facts}):")
        for cat in ["aligned", "misaligned", "semantic"]:
            std_r = [r for r in results_std if r.category == cat]
            auto_r = [r for r in results_auto if r.category == cat]
            if not std_r:
                continue
            std_acc = sum(r.n_correct for r in std_r) / \
                      max(sum(r.n_total for r in std_r), 1)
            auto_acc = sum(r.n_correct for r in auto_r) / \
                       max(sum(r.n_total for r in auto_r), 1)
            delta = auto_acc - std_acc
            sign = "+" if delta >= 0 else ""
            print(f"    {cat:>12}: {std_acc:.1%} → {auto_acc:.1%} "
                  f"({sign}{delta:.1%})")


def run_alpha_sweep(
    n_facts: int = 5,
    n_episodes: int = 20,
    alpha: float = 0.5,
    completion_alphas: list[float] = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
    seed: int = 42,
):
    """Sweep completion_alpha to find optimal weight."""
    device = get_device()
    print(f"Device: {device}")

    print("Loading GPT-2...")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
    )
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    auto_pairs = extract_auto_pairs(tokenizer)
    print(f"T_auto pairs: {len(auto_pairs)}")

    # Baseline (no T_auto)
    results_std = run_variant_eval(
        model, tokenizer, fact_types,
        n_facts=n_facts, n_episodes=n_episodes,
        auto_pairs=None, alpha=alpha, seed=seed,
    )

    print(f"\n{'=' * 70}")
    print(f"  Completion Alpha Sweep (n={n_facts}, alpha={alpha})")
    print(f"{'=' * 70}")
    print(f"  {'c_alpha':>8} | {'aligned':>8} | {'misaligned':>10} | "
          f"{'semantic':>8} | {'overall':>8}")
    print(f"  {'-' * 8}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}")

    # Print baseline
    for cat_label, cat_name in [("aligned", "aligned"),
                                 ("misaligned", "misaligned"),
                                 ("semantic", "semantic")]:
        pass
    cat_accs = {}
    for cat in ["aligned", "misaligned", "semantic"]:
        cat_r = [r for r in results_std if r.category == cat]
        if cat_r:
            cat_accs[cat] = sum(r.n_correct for r in cat_r) / \
                            max(sum(r.n_total for r in cat_r), 1)
        else:
            cat_accs[cat] = 0
    overall = sum(r.n_correct for r in results_std) / \
              max(sum(r.n_total for r in results_std), 1)
    print(f"  {'none':>8} | {cat_accs['aligned']:>7.1%} | "
          f"{cat_accs['misaligned']:>9.1%} | "
          f"{cat_accs['semantic']:>7.1%} | {overall:>7.1%}")

    for c_alpha in completion_alphas:
        results = run_variant_eval(
            model, tokenizer, fact_types,
            n_facts=n_facts, n_episodes=n_episodes,
            auto_pairs=auto_pairs,
            completion_alpha=c_alpha,
            alpha=alpha, seed=seed,
        )
        cat_accs = {}
        for cat in ["aligned", "misaligned", "semantic"]:
            cat_r = [r for r in results if r.category == cat]
            if cat_r:
                cat_accs[cat] = sum(r.n_correct for r in cat_r) / \
                                max(sum(r.n_total for r in cat_r), 1)
            else:
                cat_accs[cat] = 0
        overall = sum(r.n_correct for r in results) / \
                  max(sum(r.n_total for r in results), 1)
        print(f"  {c_alpha:>8.1f} | {cat_accs['aligned']:>7.1%} | "
              f"{cat_accs['misaligned']:>9.1%} | "
              f"{cat_accs['semantic']:>7.1%} | {overall:>7.1%}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 22: Pattern Completion via T_auto")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: n_episodes=10, n_facts=[3,5]")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--n-facts", type=int, nargs="+",
                        default=[1, 3, 5, 7])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--completion-alpha", type=float, default=1.0)
    parser.add_argument("--sweep-alpha", action="store_true",
                        help="Sweep completion_alpha values")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.sweep_alpha:
        run_alpha_sweep(
            n_facts=args.n_facts[0] if len(args.n_facts) == 1 else 5,
            n_episodes=args.n_episodes if not args.quick else 10,
            alpha=args.alpha,
            seed=args.seed,
        )
    else:
        if args.quick:
            args.n_episodes = 10
            args.n_facts = [3, 5]

        run_experiment(
            n_facts_list=args.n_facts,
            n_episodes=args.n_episodes,
            alpha=args.alpha,
            completion_alpha=args.completion_alpha,
            seed=args.seed,
        )
