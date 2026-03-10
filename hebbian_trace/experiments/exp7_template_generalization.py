"""Experiment 7: Template Generalization via Concept Injection.

Tests whether Tier 2 templates ("I am {X}", "Call me {X}") can work
with the Hebbian trace by injecting the concept word's Q at the
correct storage position.

Discovery: addressing is already concept-specific. The linking-token
mask has a shift-1 offset: when "is" is at position 3, mask activates
Q_store at position 2 = Q("name"). So Tier 1 naturally stores
Q("name") → V("Andrey").

Tier 2 fails because:
  - "am" / "me" weren't linking tokens (no mask activation, no storage)
  - Even if they were, Q("I") or Q("Call") would be stored (shift-1),
    not Q("name")

Solution: concept injection — override store_Q at the position before
the linking token with the concept word's Q. Then "am" triggers
Q("name") → V("Andrey") storage.

Also fixes country bug: template "I come from {X}" stores Q("come")
but question "Where do I come from ?" retrieves Q("from"). Fix: inject
Q("from") at storage position via concept_word="from".

Phases:
  1. Diagnostic: verify Q addressing for each fact type
  2. Country fix: test new country templates (no injection needed)
  3. Tier 2 with injection: "I am {X}" → inject Q("name")
  4. Composition: injection + pattern sep + adaptive alpha
"""

import argparse
import time

import torch

from ..model import MiniGPT
from ..nlp_tasks import (
    NLP_VOCAB, FACT_TYPES, FactTemplate,
    make_nlp_eval_episodes, compute_concept_injection,
)
from ..nlp_evaluate import (
    NLPEvalResults,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
    evaluate_hebbian,
    evaluate_baseline,
)
from .exp2_nlp_facts import get_device, load_nlp_model, pretrain_nlp


# ── Constants ──

N_EVAL = 200
ALPHA = 0.1
TRACE_LR = 0.1
FACT_COUNTS = [1, 3, 5, 10]


# ── Helpers ──

def setup_model(
    model: MiniGPT,
    use_pattern_sep: bool = True,
    use_adaptive: bool = False,
    norm_target: float = 5.0,
):
    """Configure model for evaluation."""
    for attn in model.get_attention_layers():
        attn.trace_lr = TRACE_LR
        attn.alpha = ALPHA
    model.set_per_head_decay([0.99] * model.n_heads)
    if use_pattern_sep:
        model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    else:
        model.disable_pattern_separation()
    model.set_adaptive_alpha(use_adaptive, norm_target, score_only=True)


def cleanup_model(model: MiniGPT):
    """Reset model to default state."""
    model.disable_pattern_separation()
    model.set_adaptive_alpha(False, 1.0)
    model.set_per_head_decay([0.99] * model.n_heads)


def run_eval_suite(
    model: MiniGPT,
    n_facts_list: list[int],
    n_eval: int,
    seed: int,
    tier: int | None = None,
    use_injection: bool = False,
    use_pattern_sep: bool = True,
    use_adaptive: bool = False,
    norm_target: float = 5.0,
    label: str = "",
) -> dict[int, dict]:
    """Run cross-context eval for multiple n_facts values.

    Returns:
        {n_facts: {cross_acc, cross_bl_acc, tier1_acc, tier2_acc}}
    """
    setup_model(model, use_pattern_sep, use_adaptive, norm_target)

    results = {}
    for n_facts in n_facts_list:
        episodes = make_nlp_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            seed=seed + n_facts * 1000, tier=tier)

        cross = evaluate_cross_context(
            model, episodes, use_injection=use_injection)

        results[n_facts] = {
            'cross_acc': cross.accuracy,
            'tier1_acc': cross.tier1_accuracy,
            'tier2_acc': cross.tier2_accuracy,
            'n_total': cross.n_total,
        }

    cleanup_model(model)
    return results


def print_results_table(
    all_results: dict[str, dict[int, dict]],
    n_facts_list: list[int],
    title: str,
):
    """Print comparison table."""
    print(f"\n{title}")
    cond_names = list(all_results.keys())

    header = f"{'n_facts':>7} │"
    for name in cond_names:
        header += f" {name:>20}"
    print(header)
    print("─" * (10 + 21 * len(cond_names)))

    for n in n_facts_list:
        row = f"{n:>7d} │"
        for name in cond_names:
            if n in all_results[name]:
                acc = all_results[name][n]['cross_acc']
                row += f" {acc:>19.1%}"
            else:
                row += f" {'N/A':>19}"
        print(row)

    # Tier breakdown if available
    has_tiers = any(
        all_results[name][n].get('tier1_acc') is not None
        for name in cond_names for n in n_facts_list
        if n in all_results[name]
    )
    if has_tiers:
        print(f"\nTier 1 breakdown:")
        for n in n_facts_list:
            row = f"{n:>7d} │"
            for name in cond_names:
                if n in all_results[name]:
                    t1 = all_results[name][n].get('tier1_acc')
                    row += f" {t1:>19.1%}" if t1 is not None else f" {'N/A':>19}"
                else:
                    row += f" {'N/A':>19}"
            print(row)

        print(f"\nTier 2 breakdown:")
        for n in n_facts_list:
            row = f"{n:>7d} │"
            for name in cond_names:
                if n in all_results[name]:
                    t2 = all_results[name][n].get('tier2_acc')
                    row += f" {t2:>19.1%}" if t2 is not None else f" {'N/A':>19}"
                else:
                    row += f" {'N/A':>19}"
            print(row)


# ── Phase 1: Diagnostic ──

def run_phase1_diagnostic(model: MiniGPT, seed: int = 42):
    """Verify Q addressing: what Q is stored/retrieved for each fact type."""
    print(f"\n{'=' * 70}")
    print("PHASE 1: Q Addressing Diagnostic")
    print(f"{'=' * 70}")
    print("Verify which Q is stored and retrieved for each fact type.\n")

    vocab = NLP_VOCAB
    device = next(model.parameters()).device
    d_k = model.d_model // model.n_heads

    # For each fact type, take first Tier 1 template and check
    for ft in FACT_TYPES:
        t1_templates = [t for t in ft.fact_templates if t.tier == 1]
        if not t1_templates:
            continue
        tmpl = t1_templates[0]
        value = ft.values[0]
        words = [value if w == "{X}" else w for w in tmpl.words]

        # Build sequence
        seq_words = ["<bos>"] + words + ["<eos>"]
        seq_indices = vocab.encode(seq_words)
        input_tensor = torch.tensor(
            [seq_indices], dtype=torch.long, device=device)

        # Compute store_Q via same pipeline as model
        with torch.no_grad():
            tok = model.token_embed(input_tensor)
            store_Q = model.blocks[0].attn.W_q(
                model.blocks[0].ln1(tok))
            store_Q = store_Q.view(1, len(seq_indices), model.n_heads, d_k)
            store_Q = store_Q.transpose(1, 2)  # (1, H, S, d_k)

        # Find linking token position
        linking_ids = set(vocab.linking_tokens)
        linking_pos = None
        for i, idx in enumerate(seq_indices):
            if idx in linking_ids:
                linking_pos = i
                break

        if linking_pos is None:
            print(f"  {ft.name:>10}: NO linking token found in {seq_words}")
            continue

        # Q_store uses shift: Q_store = Q[:, :, :-2, :] → positions 0..S-3
        # mask checks token_ids[:, 1:-1] → positions 1..S-2
        # When linking at position p, mask_index = p-1, Q_store[p-1] = Q[p-1]
        stored_word_pos = linking_pos - 1
        stored_word = seq_words[stored_word_pos] if stored_word_pos >= 0 else "?"

        # Retrieval: Q_addr uses Q from previous position
        # Question: "What is my <concept> ?"
        q_tmpl = ft.question_templates[0]
        q_words = ["<bos>"] + q_tmpl.words
        q_indices = vocab.encode(q_words)

        # Q_addr at last position (?) = Q[?-1] = Q of word before ?
        last_q_word = q_words[-2] if len(q_words) > 1 else "?"

        print(f"  {ft.name:>10}: "
              f"store Q(\"{stored_word}\") at pos {stored_word_pos} | "
              f"retrieve Q(\"{last_q_word}\") | "
              f"match={'✓' if stored_word == last_q_word else '✗'} | "
              f"template: {' '.join(words)}")

    # Check Tier 2 templates
    print(f"\n  Tier 2 templates (require injection):")
    for ft in FACT_TYPES:
        t2_templates = [t for t in ft.fact_templates if t.tier == 2]
        for tmpl in t2_templates:
            value = ft.values[0]
            words = [value if w == "{X}" else w for w in tmpl.words]
            # Find what would be stored without injection
            seq_words = ["<bos>"] + words + ["<eos>"]
            seq_indices = vocab.encode(seq_words)
            linking_ids = set(vocab.linking_tokens)
            linking_pos = None
            for i, idx in enumerate(seq_indices):
                if idx in linking_ids:
                    linking_pos = i
                    break
            if linking_pos:
                stored_word = seq_words[linking_pos - 1]
                print(f"  {ft.name:>10}: "
                      f"would store Q(\"{stored_word}\") → "
                      f"inject Q(\"{tmpl.concept_word}\") | "
                      f"template: {' '.join(words)}")
            else:
                print(f"  {ft.name:>10}: "
                      f"no linking token | "
                      f"template: {' '.join(words)}")


# ── Phase 2: Country Fix ──

def run_phase2_country_fix(
    model: MiniGPT, n_eval: int, seed: int,
    n_facts_list: list[int],
):
    """Test country fix via concept injection."""
    print(f"\n{'=' * 70}")
    print("PHASE 2: Country Fix via Concept Injection")
    print(f"{'=' * 70}")
    print("Bug: 'I come from {X}' stores Q('come'), retrieves Q('from')")
    print("Fix: inject Q('from') at storage pos → Q('from')→V('{X}')")
    print()

    all_results = {}

    # Tier 1 without injection (country stores Q("come"), retrieves Q("from") = mismatch)
    all_results['T1 no_inj'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=1,
        use_injection=False, label="Tier 1 no injection")

    # Tier 1 with injection (country gets Q("from") injected → match)
    all_results['T1 + inject'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=1,
        use_injection=True, label="Tier 1 with injection")

    # Per-type breakdown for largest n
    max_n = max(n_facts_list)
    if max_n >= 5:
        vocab = NLP_VOCAB
        device = next(model.parameters()).device

        for inj_mode, inj_label in [(False, "no injection"),
                                     (True, "with injection")]:
            print(f"\nPer-type accuracy at n={max_n} (Tier 1, {inj_label}):")
            setup_model(model)
            episodes = make_nlp_eval_episodes(
                n_episodes=n_eval, n_facts=max_n,
                seed=seed + max_n * 1000, tier=1)

            type_correct = {}
            type_total = {}

            for episode in episodes:
                model.reset_traces()
                model.set_trace_mode(use=False, update=True)
                for seq_idx, train_seq in enumerate(episode.train_sequences):
                    inj = None
                    if inj_mode and hasattr(episode, 'fact_templates_used'):
                        n_f = seq_idx + 1
                        fw = [f[2] for f in episode.facts[:n_f]]
                        inj = compute_concept_injection(
                            fw, episode.fact_templates_used[:n_f], vocab)
                    inp = torch.tensor(
                        [train_seq], dtype=torch.long, device=device)
                    with torch.no_grad():
                        _ = model(inp, concept_injection=inj)
                model.set_trace_mode(use=True, update=False)

                for query_indices, answer_idx, ft_name, tier \
                        in episode.test_queries:
                    inp = torch.tensor(
                        [query_indices], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = model(inp)
                    pred_logits = logits[0, -1, :]
                    entity_logits = pred_logits[vocab.entity_indices]
                    pred_idx = vocab.entity_indices[
                        entity_logits.argmax().item()]

                    if ft_name not in type_correct:
                        type_correct[ft_name] = 0
                        type_total[ft_name] = 0
                    type_total[ft_name] += 1
                    if pred_idx == answer_idx:
                        type_correct[ft_name] += 1

            for name in sorted(type_correct.keys()):
                acc = type_correct[name] / max(type_total[name], 1)
                n = type_total[name]
                print(f"  {name:>10}: {acc:.1%} ({type_correct[name]}/{n})")

            cleanup_model(model)

    print_results_table(all_results, n_facts_list, "PHASE 2 RESULTS:")


# ── Phase 3: Tier 2 with Injection ──

def run_phase3_injection(
    model: MiniGPT, n_eval: int, seed: int,
    n_facts_list: list[int],
):
    """Test Tier 2 templates with and without concept injection."""
    print(f"\n{'=' * 70}")
    print("PHASE 3: Tier 2 Templates + Concept Injection")
    print(f"{'=' * 70}")
    print("Tier 2: 'I am {X}', 'Call me {X}', 'I come from {X}'")
    print("Injection: override Q at storage pos with concept word Q")
    print()

    all_results = {}

    # Tier 1 baseline (reference)
    all_results['Tier1 (reference)'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=1,
        use_injection=False, label="Tier 1 reference")

    # Tier 2 without injection (expected: ~random for name, ~0 for country)
    all_results['Tier2 no_inj'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=2,
        use_injection=False, label="Tier 2 no injection")

    # Tier 2 with injection
    all_results['Tier2 + inject'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=2,
        use_injection=True, label="Tier 2 with injection")

    # Mixed tiers without injection
    all_results['Mixed no_inj'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=None,
        use_injection=False, label="Mixed tiers no injection")

    # Mixed tiers with injection
    all_results['Mixed + inject'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=None,
        use_injection=True, label="Mixed tiers with injection")

    print_results_table(all_results, n_facts_list, "PHASE 3 RESULTS:")


# ── Phase 4: Injection + Best Mechanisms ──

def run_phase4_composition(
    model: MiniGPT, n_eval: int, seed: int,
    n_facts_list: list[int],
    norm_target: float = 5.0,
):
    """Test injection composition with pattern sep + adaptive alpha."""
    print(f"\n{'=' * 70}")
    print("PHASE 4: Composition — Injection + Pattern Sep + Adaptive Alpha")
    print(f"{'=' * 70}")

    all_results = {}

    # Tier 1 + PS + adaptive (current best, reference)
    all_results['T1 PS+adapt'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=1,
        use_injection=False, use_pattern_sep=True,
        use_adaptive=True, norm_target=norm_target)

    # Mixed + injection + PS + adaptive
    all_results['Mix inj+PS+adapt'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=None,
        use_injection=True, use_pattern_sep=True,
        use_adaptive=True, norm_target=norm_target)

    # Tier 2 + injection + PS + adaptive
    all_results['T2 inj+PS+adapt'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=2,
        use_injection=True, use_pattern_sep=True,
        use_adaptive=True, norm_target=norm_target)

    # No PS (ablation)
    all_results['Mix inj, no PS'] = run_eval_suite(
        model, n_facts_list, n_eval, seed, tier=None,
        use_injection=True, use_pattern_sep=False,
        use_adaptive=False)

    print_results_table(all_results, n_facts_list, "PHASE 4 RESULTS:")


# ── Calibration ──

def calibrate_norm_target(model: MiniGPT, seed: int = 99) -> float:
    """Measure score trace norm after 1 fact for adaptive alpha."""
    device = next(model.parameters()).device

    episodes = make_nlp_eval_episodes(
        n_episodes=1, n_facts=1, seed=seed, tier=1)
    ep = episodes[0]

    setup_model(model, use_pattern_sep=True, use_adaptive=False)
    model.reset_traces()
    model.set_trace_mode(use=False, update=True)

    for train_seq in ep.train_sequences:
        inp = torch.tensor([train_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(inp)

    score_norms = []
    for attn in model.get_attention_layers():
        score_norms.append(attn.traces.norm(dim=(1, 2)))
    mean_norm = torch.stack(score_norms).mean().item()

    model.reset_traces()
    model.set_trace_mode(use=False, update=False)
    cleanup_model(model)
    return mean_norm


# ── Main ──

def run(
    load_path: str | None = None,
    n_eval: int = N_EVAL,
    n_facts_list: list[int] | None = None,
    seed: int = 42,
    device_name: str | None = None,
    verbose: bool = True,
):
    """Run template generalization experiment."""
    if n_facts_list is None:
        n_facts_list = FACT_COUNTS

    device = get_device(device_name)
    t_start = time.time()

    print("=" * 70)
    print("EXPERIMENT 7: Template Generalization via Concept Injection")
    print("=" * 70)

    # ── Step 1: Get model ──
    if load_path:
        print(f"\nLoading model from {load_path}...")
        model = load_nlp_model(load_path, device_name)
        print("  Model loaded.")
    else:
        print("\nPretraining...")
        model, train_stats = pretrain_nlp(
            n_sequences=20000, max_facts=5, batch_size=64, epochs=60,
            lr=1e-3, d_model=256, n_heads=8, n_layers=8,
            max_seq_len=128, dropout=0.1,
            alpha=ALPHA, trace_lr=TRACE_LR, trace_decay=0.99,
            use_raw_embed=True, use_key_q=True,
            seed=seed, device=device, verbose=verbose,
        )
        print(f"  Final accuracy: {train_stats['epoch_acc'][-1]:.1%}")
        vocab = NLP_VOCAB
        model.set_linking_token_ids(vocab.linking_tokens)

    print(f"  d_model={model.d_model}, n_heads={model.n_heads}, "
          f"d_k={model.d_model // model.n_heads}")

    # Update linking tokens (includes new "am", "me")
    vocab = NLP_VOCAB
    model.set_linking_token_ids(vocab.linking_tokens)
    print(f"  Linking tokens: {[vocab.idx2tok[i] for i in vocab.linking_tokens]}")

    # ── Phase 1: Diagnostic ──
    run_phase1_diagnostic(model, seed)

    # ── Phase 2: Country fix ──
    run_phase2_country_fix(model, n_eval, seed, n_facts_list)

    # ── Calibrate adaptive alpha ──
    norm_1 = calibrate_norm_target(model, seed=99)
    norm_target = norm_1 * 3
    print(f"\n  Adaptive alpha: 1-fact norm={norm_1:.4f}, "
          f"target={norm_target:.4f}")

    # ── Phase 3: Tier 2 + injection ──
    run_phase3_injection(model, n_eval, seed, n_facts_list)

    # ── Phase 4: Composition ──
    run_phase4_composition(
        model, n_eval, seed, n_facts_list, norm_target)

    # ── Final Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 7 COMPLETE (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 70}")


def run_quick(device_name: str | None = None, load_path: str | None = None):
    """Quick test: fewer episodes, smaller n_facts."""
    run(
        load_path=load_path,
        n_eval=50,
        n_facts_list=[1, 3, 5],
        device_name=device_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 7: Template Generalization via Concept Injection")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (~5 min)")
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained model (e.g., models/nlp_full.pt)")
    args = parser.parse_args()

    if args.quick:
        run_quick(device_name=args.device, load_path=args.load)
    else:
        run(
            load_path=args.load,
            n_eval=args.n_eval,
            seed=args.seed,
            device_name=args.device,
        )
