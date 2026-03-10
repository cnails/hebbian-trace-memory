#!/usr/bin/env python3
"""Reproduce flagship evaluation results.

Runs cross-context evaluation at multiple fact counts (n=1,3,5,7)
and compares trace-based retrieval against RAG baselines.

Two evaluation modes:
  - 7 base types (97 entity candidates): easy retrieval regime
  - 24 extended types (229 entity candidates): harder, realistic regime

Usage:
    python evaluate.py              # 50 episodes (quick)
    python evaluate.py --n-eval 100 # 100 episodes (paper results)
"""

import argparse
import time

import numpy as np
import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_fact_types,
    build_extended_fact_types,
    get_linking_bpe_ids,
    get_all_entity_ids,
    make_eval_episodes,
    evaluate_baseline,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)
from hebbian_trace.rag_baselines import (
    OracleRAGStore,
    TFIDFRAGStore,
    EmbeddingRAGStore,
    evaluate_rag,
    evaluate_retrieval_accuracy,
    evaluate_knn,
)


def bootstrap_ci(per_episode_acc: list[float], n_boot: int = 10000,
                  ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval for mean accuracy.

    Args:
        per_episode_acc: per-episode accuracy values.
        n_boot: number of bootstrap resamples.
        ci: confidence level (default 0.95).
        seed: random seed for reproducibility.

    Returns:
        (lo, hi) bounds of the CI.
    """
    arr = np.array(per_episode_acc)
    n = len(arr)
    if n <= 1:
        return (arr[0] if n == 1 else 0.0, arr[0] if n == 1 else 0.0)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = arr[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return (lo, hi)


def _fmt(result) -> str:
    """Format EvalResults as 'acc [lo, hi]' with 95% bootstrap CI."""
    lo, hi = bootstrap_ci(result.per_episode_acc)
    return f"{result.accuracy * 100:5.1f} [{lo * 100:4.1f},{hi * 100:5.1f}]"


def _run_table(model, fact_types, tokenizer, wte_weight,
               n_eval, n_facts_list, label):
    """Run evaluation table for a set of fact types."""
    entity_ids = get_all_entity_ids(fact_types)

    oracle_store = OracleRAGStore(tokenizer)
    tfidf_store = TFIDFRAGStore(tokenizer)
    embed_store = EmbeddingRAGStore(tokenizer, wte_weight)

    print(f"  {label}: {len(fact_types)} types, "
          f"{len(entity_ids)} entity candidates, "
          f"{n_eval} episodes")
    print()
    w = 17  # column width for 'acc [lo, hi]'
    sep = "-" * (6 + 7 * (w + 2))
    print(sep)
    print(f"  {'n':>3}  {'Trace':>{w}}  {'kNN-LM':>{w}}  {'In-ctx':>{w}}  "
          f"{'RAG-Orc':>{w}}  {'RAG-Emb':>{w}}  {'RAG-TF':>{w}}  "
          f"{'No-trace':>{w}}")
    print(sep)

    for n_facts in n_facts_list:
        if n_facts > len(fact_types):
            continue

        episodes = make_eval_episodes(
            n_eval, n_facts, tokenizer, fact_types, seed=42)

        t0 = time.time()

        cross = evaluate_cross_context(
            model, episodes, fact_types, verbose=False)
        baseline = evaluate_baseline(
            model, episodes, fact_types, tokenizer, verbose=False)
        cross_bl = evaluate_cross_context_baseline(
            model, episodes, fact_types)

        rag_oracle = evaluate_rag(
            model, episodes, fact_types, oracle_store, top_k=1)
        rag_embed = evaluate_rag(
            model, episodes, fact_types, embed_store, top_k=1)
        rag_tfidf = evaluate_rag(
            model, episodes, fact_types, tfidf_store, top_k=1)

        knn = evaluate_knn(model, episodes, fact_types,
                           k=32, temperature=10.0, lam=0.25)

        dt = time.time() - t0

        print(f"  {n_facts:>3}  {_fmt(cross):>{w}}  "
              f"{_fmt(knn):>{w}}  "
              f"{_fmt(baseline):>{w}}  "
              f"{_fmt(rag_oracle):>{w}}  "
              f"{_fmt(rag_embed):>{w}}  "
              f"{_fmt(rag_tfidf):>{w}}  "
              f"{_fmt(cross_bl):>{w}}")

    print(sep)
    print("  (values: accuracy% [95% bootstrap CI], 10,000 resamples)")
    print()


def run_evaluation(n_eval: int = 50,
                   weights_path: str = "weights/trace_module.pt"):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 90)
    print("  Hebbian Trace Memory — Evaluation with RAG Baselines")
    print("=" * 90)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print()

    # Setup
    print("Loading GPT-2 + trace module...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    linking_ids = get_linking_bpe_ids(tokenizer)

    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.trace.load_state_dict(state, strict=False)
        print(f"Loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found, using random projections")

    trace_params = sum(p.numel() for p in model.trace.parameters())
    gpt2_params = sum(p.numel() for p in model.gpt2.parameters())
    print(f"  GPT-2 params:  {gpt2_params:,} (frozen)")
    print(f"  Trace params:  {trace_params:,} (external module)")
    print()

    wte_weight = model.gpt2.transformer.wte.weight.detach().cpu()

    # --- Table 1: 7 base types (easy retrieval) ---
    base_types = build_fact_types(tokenizer)
    _run_table(model, base_types, tokenizer, wte_weight,
               n_eval, [1, 3, 5, 7], "Table 1 — Base types")

    # --- Table 2: 24 extended types (harder, realistic) ---
    ext_types = build_extended_fact_types(tokenizer)
    _run_table(model, ext_types, tokenizer, wte_weight,
               n_eval, [1, 3, 5, 7, 12, 18, 24], "Table 2 — Extended types")

    # Legend
    print("  Trace:     Hebbian trace retrieval (question-only input)")
    print("  kNN-LM:    kNN over GPT-2 hidden states (Khandelwal+ 2020)")
    print("  In-ctx:    All facts + question in one pass (GPT-2 native)")
    print("  RAG-Orc:   RAG with perfect retrieval (upper bound)")
    print("  RAG-Emb:   RAG with GPT-2 embedding similarity retrieval")
    print("  RAG-TF:    RAG with TF-IDF keyword retrieval")
    print("  No-trace:  Question-only, no memory (expected ~random)")
    print()
    print("  kNN-LM stores (hidden_state, next_token) pairs from fact passages")
    print("  and retrieves by nearest-neighbor search in hidden-state space.")
    print("  Unlike RAG, it does not prepend facts in-context. Like the trace,")
    print("  it operates at the logit/probability level. However, it uses")
    print("  contextual hidden states (context-dependent) vs the trace's")
    print("  context-free Q addressing (context-independent).")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Hebbian Trace Memory — Evaluation")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--weights", type=str, default="weights/trace_module.pt",
                        help="Path to trace module weights")
    args = parser.parse_args()

    run_evaluation(n_eval=args.n_eval, weights_path=args.weights)


if __name__ == "__main__":
    main()
