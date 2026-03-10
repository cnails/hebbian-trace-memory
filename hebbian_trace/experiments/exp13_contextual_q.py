"""Experiment 13: Context-Enriched Q — β Sweep.

Exp 12 identified three architectural walls rooted in context-free Q:
  - Shift-1 brittleness: aligned 90.6% vs misaligned 21.2%
  - Distractor confusion: 70% confusion rate
  - Q-collision: -28.6pp cost

Solution: Q = Q_base + β * Q_ctx, where Q_ctx comes from GPT-2 hidden states.
Sweep β to find the Pareto frontier: cross-context stability vs disambiguation.

Phase 1: Setup & backward compat check (β=0 reproduces exp12)
Phase 2: β sweep with three metrics (cross-ctx, paraphrasing, distractors)
Phase 3: Layer sweep (fix best β, vary GPT-2 layer)
Phase 4: Pareto frontier summary

Usage:
    python -m hebbian_trace.experiments.exp13_contextual_q --quick
    python -m hebbian_trace.experiments.exp13_contextual_q --n-eval 50
"""

import argparse
import random
import time
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    GPT2EvalEpisode, GPT2FactType,
    _get_all_entity_ids, tokenize_fact,
)
from .exp12_realistic_benchmarks import (
    build_question_variants, QuestionVariant,
    build_distractor_episodes, validate_distractor_persons,
    DISTRACTOR_PERSONS,
)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Evaluation functions that pass beta through ──────────────────────

def _predict_answer_beta(model, query_ids: list[int],
                         entity_ids: list[int],
                         beta: float | torch.Tensor = 0.0,
                         context_layer: int = -1) -> int:
    """Run model on query with contextual Q and return predicted entity."""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor, beta=beta, context_layer=context_layer)

    pred_logits = logits[0, -1, :]
    entity_logits = pred_logits[entity_ids]
    best_pos = entity_logits.argmax().item()
    return entity_ids[best_pos]


def evaluate_cross_context_beta(
    model, episodes: list[GPT2EvalEpisode],
    fact_types: list[GPT2FactType],
    beta: float | torch.Tensor = 0.0,
    context_layer: int = -1,
) -> float:
    """Cross-context eval with contextual Q.

    Write facts to trace, query with question-only.
    Returns accuracy.
    """
    model.eval()
    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device
    total_correct = 0
    total_queries = 0

    for episode in episodes:
        model.reset_traces()

        # Write phase
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor, beta=beta, context_layer=context_layer)

        # Read phase
        model.set_trace_mode(use=True, update=False)
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer_beta(
                model, query_ids, entity_ids, beta, context_layer)
            if pred_id == answer_id:
                total_correct += 1
            total_queries += 1

    return total_correct / max(total_queries, 1)


def evaluate_paraphrasing_beta(
    model, tokenizer, fact_types, variants,
    n_eval, n_facts, beta: float | torch.Tensor = 0.0,
    context_layer=-1, seed=42,
) -> dict[str, tuple[int, int]]:
    """Paraphrasing eval with contextual Q.

    Returns {category: (n_correct, n_total)}.
    """
    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device

    episodes = make_gpt2_eval_episodes(
        n_episodes=n_eval, n_facts=n_facts,
        tokenizer=tokenizer, fact_types=fact_types, seed=seed)

    category_totals: dict[str, list] = {
        "aligned": [0, 0], "misaligned": [0, 0], "semantic": [0, 0],
    }

    for episode in episodes:
        model.reset_traces()

        # Write phase
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor, beta=beta, context_layer=context_layer)

        # Read phase: test all variants
        model.set_trace_mode(use=True, update=False)
        for (type_name, entity_name, entity_id, _) in episode.facts:
            if type_name not in variants:
                continue
            for v in variants[type_name]:
                pred_id = _predict_answer_beta(
                    model, v.bpe_ids, entity_ids, beta, context_layer)
                category_totals[v.category][1] += 1
                if pred_id == entity_id:
                    category_totals[v.category][0] += 1

    return category_totals


def evaluate_distractors_beta(
    model, tokenizer, fact_types, n_eval, n_my_facts,
    n_distractors_per_type, beta: float | torch.Tensor = 0.0,
    context_layer=-1, seed=42,
) -> tuple[float, float]:
    """Distractor eval with contextual Q.

    Returns (my_accuracy, confusion_rate).
    """
    # Validate persons
    all_entity_names = set()
    for ft in fact_types:
        for name, _ in ft.entities:
            all_entity_names.add(name)
    persons = validate_distractor_persons(tokenizer, all_entity_names)

    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device

    episodes = build_distractor_episodes(
        n_episodes=n_eval,
        n_my_facts=n_my_facts,
        n_distractors_per_type=n_distractors_per_type,
        tokenizer=tokenizer,
        fact_types=fact_types,
        distractor_persons=persons,
        seed=seed,
    )

    total_correct = 0
    total_confused = 0
    total_queries = 0

    for episode in episodes:
        model.reset_traces()

        # Write phase
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor, beta=beta, context_layer=context_layer)

        # Read phase
        model.set_trace_mode(use=True, update=False)
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer_beta(
                model, query_ids, entity_ids, beta, context_layer)
            total_queries += 1
            if pred_id == answer_id:
                total_correct += 1
            else:
                ft = next(f for f in fact_types if f.name == type_name)
                ft_entity_ids = {eid for _, eid in ft.entities}
                if pred_id in ft_entity_ids:
                    total_confused += 1

    acc = total_correct / max(total_queries, 1)
    confusion = total_confused / max(total_queries, 1)
    return acc, confusion


# ── Phase 1: Setup ────────────────────────────────────────────────────

def run_phase1_setup(device_str=None):
    """Load model and verify backward compatibility."""
    print("=" * 65)
    print("EXP 13: Context-Enriched Q — β Sweep (Pareto Frontier)")
    print("=" * 65)

    device = get_device(device_str)
    print(f"\nDevice: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    ).to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model.eval()

    print(f"Fact types: {len(fact_types)}")
    print(f"Trace params: {sum(p.numel() for p in model.trace.parameters()):,}")

    # Backward compat check: β=0 should give same results as exp12
    print(f"\n{'─' * 65}")
    print("PHASE 1: Backward Compatibility Check (β=0)")
    print(f"{'─' * 65}")

    episodes = make_gpt2_eval_episodes(
        n_episodes=10, n_facts=3,
        tokenizer=tokenizer, fact_types=fact_types, seed=42)
    acc = evaluate_cross_context_beta(model, episodes, fact_types, beta=0.0)
    print(f"  Cross-context (n=3, β=0): {acc:.1%}")
    print(f"  ✓ Backward compat OK" if acc > 0.5 else
          f"  ⚠ WARNING: low baseline")

    return model, tokenizer, fact_types, linking_ids, device


# ── Phase 2: β Sweep ─────────────────────────────────────────────────

@dataclass
class BetaResult:
    beta: float
    context_layer: int
    cross_ctx_n3: float
    cross_ctx_n5: float
    aligned: float
    misaligned: float
    semantic: float
    dist_accuracy: float
    dist_confusion: float


def run_phase2_beta_sweep(model, tokenizer, fact_types, n_eval,
                          n_facts_para, betas, context_layer=-1, seed=42):
    """Sweep β across all three diagnostic metrics."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 2: β Sweep (layer={context_layer}, n_eval={n_eval})")
    print(f"{'─' * 65}")

    variants = build_question_variants(tokenizer)
    results: list[BetaResult] = []

    for beta in betas:
        t0 = time.time()
        print(f"\n  β = {beta:.2f}...")

        # Test A: Cross-context (n=3, n=5)
        ep3 = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=3,
            tokenizer=tokenizer, fact_types=fact_types, seed=seed)
        acc3 = evaluate_cross_context_beta(
            model, ep3, fact_types, beta=beta, context_layer=context_layer)

        ep5 = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=5,
            tokenizer=tokenizer, fact_types=fact_types, seed=seed + 5000)
        acc5 = evaluate_cross_context_beta(
            model, ep5, fact_types, beta=beta, context_layer=context_layer)

        # Test B: Paraphrasing
        para = evaluate_paraphrasing_beta(
            model, tokenizer, fact_types, variants,
            n_eval=n_eval, n_facts=n_facts_para,
            beta=beta, context_layer=context_layer, seed=seed)

        aligned = para["aligned"][0] / max(para["aligned"][1], 1)
        misaligned = para["misaligned"][0] / max(para["misaligned"][1], 1)
        semantic = para["semantic"][0] / max(para["semantic"][1], 1)

        # Test C: Distractors (1 per type)
        dist_acc, dist_conf = evaluate_distractors_beta(
            model, tokenizer, fact_types, n_eval=n_eval,
            n_my_facts=n_facts_para, n_distractors_per_type=1,
            beta=beta, context_layer=context_layer, seed=seed)

        elapsed = time.time() - t0
        r = BetaResult(
            beta=beta, context_layer=context_layer,
            cross_ctx_n3=acc3, cross_ctx_n5=acc5,
            aligned=aligned, misaligned=misaligned, semantic=semantic,
            dist_accuracy=dist_acc, dist_confusion=dist_conf,
        )
        results.append(r)

        print(f"    cross-ctx: n=3 {acc3:.1%}, n=5 {acc5:.1%}")
        print(f"    paraphrase: aligned {aligned:.1%}, "
              f"misaligned {misaligned:.1%}, semantic {semantic:.1%}")
        print(f"    distractors: acc {dist_acc:.1%}, "
              f"confusion {dist_conf:.1%}")
        print(f"    ({elapsed:.0f}s)")

    return results


# ── Phase 3: Layer Sweep ──────────────────────────────────────────────

def run_phase3_layer_sweep(model, tokenizer, fact_types, n_eval,
                           n_facts_para, best_beta, layers, seed=42):
    """Fix best β, sweep GPT-2 layers."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 3: Layer Sweep (β={best_beta:.2f})")
    print(f"{'─' * 65}")

    variants = build_question_variants(tokenizer)
    results: list[BetaResult] = []

    # GPT-2 Small has 13 hidden states: [embed, layer0, ..., layer11]
    # Index 0 = embedding, 1-12 = transformer layers
    for layer_idx in layers:
        t0 = time.time()
        layer_name = f"embed" if layer_idx == 0 else f"layer_{layer_idx - 1}"
        print(f"\n  Layer {layer_idx} ({layer_name})...")

        # Cross-context n=5
        ep5 = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=5,
            tokenizer=tokenizer, fact_types=fact_types, seed=seed + 5000)
        acc5 = evaluate_cross_context_beta(
            model, ep5, fact_types, beta=best_beta, context_layer=layer_idx)

        # Paraphrasing
        para = evaluate_paraphrasing_beta(
            model, tokenizer, fact_types, variants,
            n_eval=n_eval, n_facts=n_facts_para,
            beta=best_beta, context_layer=layer_idx, seed=seed)

        aligned = para["aligned"][0] / max(para["aligned"][1], 1)
        misaligned = para["misaligned"][0] / max(para["misaligned"][1], 1)
        semantic = para["semantic"][0] / max(para["semantic"][1], 1)

        # Distractors
        dist_acc, dist_conf = evaluate_distractors_beta(
            model, tokenizer, fact_types, n_eval=n_eval,
            n_my_facts=n_facts_para, n_distractors_per_type=1,
            beta=best_beta, context_layer=layer_idx, seed=seed)

        elapsed = time.time() - t0
        r = BetaResult(
            beta=best_beta, context_layer=layer_idx,
            cross_ctx_n3=0, cross_ctx_n5=acc5,
            aligned=aligned, misaligned=misaligned, semantic=semantic,
            dist_accuracy=dist_acc, dist_confusion=dist_conf,
        )
        results.append(r)

        print(f"    cross-ctx n=5: {acc5:.1%}, "
              f"aligned: {aligned:.1%}, misaligned: {misaligned:.1%}")
        print(f"    distractors: acc {dist_acc:.1%}, "
              f"confusion {dist_conf:.1%}")
        print(f"    ({elapsed:.0f}s)")

    return results


# ── Phase 4: Pareto Summary ──────────────────────────────────────────

def run_phase4_pareto(beta_results: list[BetaResult],
                      layer_results: list[BetaResult] | None = None):
    """Print Pareto frontier summary."""
    print(f"\n{'═' * 75}")
    print("PARETO FRONTIER: Cross-Context Stability vs Disambiguation")
    print(f"{'═' * 75}")

    print(f"\n  β SWEEP (layer=-1, last GPT-2 layer):")
    print(f"  {'β':>5s} │ {'Cross n=3':>9s} │ {'Cross n=5':>9s} │ "
          f"{'Aligned':>8s} │ {'Misalig':>8s} │ {'Semantic':>8s} │ "
          f"{'Dist Acc':>8s} │ {'Confusn':>8s}")
    print(f"  {'─' * 73}")

    for r in beta_results:
        print(f"  {r.beta:5.2f} │ {r.cross_ctx_n3:>8.1%} │ "
              f"{r.cross_ctx_n5:>8.1%} │ {r.aligned:>7.1%} │ "
              f"{r.misaligned:>7.1%} │ {r.semantic:>7.1%} │ "
              f"{r.dist_accuracy:>7.1%} │ {r.dist_confusion:>7.1%}")

    if layer_results:
        best_beta = layer_results[0].beta
        print(f"\n  LAYER SWEEP (β={best_beta:.2f}):")
        print(f"  {'Layer':>5s} │ {'Cross n=5':>9s} │ "
              f"{'Aligned':>8s} │ {'Misalig':>8s} │ "
              f"{'Dist Acc':>8s} │ {'Confusn':>8s}")
        print(f"  {'─' * 55}")

        for r in layer_results:
            layer_name = (f"embed" if r.context_layer == 0
                          else f"L{r.context_layer - 1}")
            print(f"  {layer_name:>5s} │ {r.cross_ctx_n5:>8.1%} │ "
                  f"{r.aligned:>7.1%} │ {r.misaligned:>7.1%} │ "
                  f"{r.dist_accuracy:>7.1%} │ {r.dist_confusion:>7.1%}")

    # Find sweet spot
    print(f"\n  ANALYSIS:")
    baseline = beta_results[0]  # β=0
    best_trade = baseline
    best_score = 0

    for r in beta_results:
        # Score: improvement in confusion reduction vs aligned loss
        confusion_gain = baseline.dist_confusion - r.dist_confusion
        aligned_loss = baseline.aligned - r.aligned
        # Good tradeoff: big confusion reduction, small aligned loss
        score = confusion_gain - 2 * max(aligned_loss, 0)
        if score > best_score:
            best_score = score
            best_trade = r

    if best_trade.beta > 0:
        print(f"    Sweet spot: β={best_trade.beta:.2f}")
        print(f"      Confusion: {baseline.dist_confusion:.1%} → "
              f"{best_trade.dist_confusion:.1%} "
              f"({best_trade.dist_confusion - baseline.dist_confusion:+.1%})")
        print(f"      Aligned:   {baseline.aligned:.1%} → "
              f"{best_trade.aligned:.1%} "
              f"({best_trade.aligned - baseline.aligned:+.1%})")
        print(f"      Cross n=5: {baseline.cross_ctx_n5:.1%} → "
              f"{best_trade.cross_ctx_n5:.1%} "
              f"({best_trade.cross_ctx_n5 - baseline.cross_ctx_n5:+.1%})")
    else:
        print("    No β > 0 improved the tradeoff.")
        print("    Context-free Q may be locally optimal for random projections.")
        print("    Next step: train W_ctx via contrastive loss (approach 3).")

    print(f"{'═' * 75}")


# ── Entry points ──────────────────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick sweep (~5-10 min)."""
    t0 = time.time()
    model, tokenizer, fact_types, linking_ids, dev = run_phase1_setup(device)

    n_eval = 20
    n_facts_para = 3
    betas = [0.0, 0.1, 0.3, 0.5, 1.0]

    beta_results = run_phase2_beta_sweep(
        model, tokenizer, fact_types, n_eval, n_facts_para,
        betas=betas, context_layer=-1, seed=seed)

    # Find best β for layer sweep
    baseline_conf = beta_results[0].dist_confusion
    best_beta = 0.0
    best_improvement = 0
    for r in beta_results:
        improvement = baseline_conf - r.dist_confusion
        if improvement > best_improvement and r.aligned > 0.3:
            best_improvement = improvement
            best_beta = r.beta

    layer_results = None
    if best_beta > 0:
        layer_results = run_phase3_layer_sweep(
            model, tokenizer, fact_types, n_eval, n_facts_para,
            best_beta=best_beta, layers=[1, 4, 7, 10, 12], seed=seed)

    run_phase4_pareto(beta_results, layer_results)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_full(device=None, seed=42, n_eval=50):
    """Full sweep (~25-40 min)."""
    t0 = time.time()
    model, tokenizer, fact_types, linking_ids, dev = run_phase1_setup(device)

    n_facts_para = 3
    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    beta_results = run_phase2_beta_sweep(
        model, tokenizer, fact_types, n_eval, n_facts_para,
        betas=betas, context_layer=-1, seed=seed)

    # Find best β
    baseline_conf = beta_results[0].dist_confusion
    best_beta = 0.0
    best_improvement = 0
    for r in beta_results:
        improvement = baseline_conf - r.dist_confusion
        if improvement > best_improvement and r.aligned > 0.3:
            best_improvement = improvement
            best_beta = r.beta

    layer_results = None
    if best_beta > 0:
        layer_results = run_phase3_layer_sweep(
            model, tokenizer, fact_types, n_eval, n_facts_para,
            best_beta=best_beta, layers=[1, 4, 7, 10, 12], seed=seed)

    run_phase4_pareto(beta_results, layer_results)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 13: Context-Enriched Q — β Sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 episodes, fewer betas)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    if args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run_full(device=args.device, seed=args.seed, n_eval=args.n_eval)
