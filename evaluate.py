#!/usr/bin/env python3
"""Reproduce flagship evaluation results.

Runs cross-context evaluation at multiple fact counts (n=1,3,5,7)
and compares trace-based retrieval against baselines.

Usage:
    python evaluate.py              # 50 episodes (quick)
    python evaluate.py --n-eval 100 # 100 episodes (paper results)
"""

import argparse
import time

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_fact_types,
    get_linking_bpe_ids,
    make_eval_episodes,
    evaluate_baseline,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)


def run_evaluation(n_eval: int = 50,
                   weights_path: str = "weights/trace_module.pt"):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("  Hebbian Trace Memory — Evaluation")
    print("=" * 60)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print()

    # Setup
    print("Loading GPT-2 + trace module...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    # Load weights
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.trace.load_state_dict(state, strict=False)
        print(f"Loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found, using random projections")

    # Count parameters
    trace_params = sum(p.numel() for p in model.trace.parameters())
    gpt2_params = sum(p.numel() for p in model.gpt2.parameters())
    print(f"  GPT-2 params:  {gpt2_params:,} (frozen)")
    print(f"  Trace params:  {trace_params:,} (external module)")
    print()

    # Evaluate at multiple fact counts
    n_facts_list = [1, 3, 5, 7]

    print("-" * 60)
    print(f"  {'n_facts':>7}  {'Cross-ctx':>10}  {'Baseline':>10}  "
          f"{'Cross BL':>10}  {'Gap':>8}")
    print("-" * 60)

    for n_facts in n_facts_list:
        episodes = make_eval_episodes(
            n_eval, n_facts, tokenizer, fact_types, seed=42)

        t0 = time.time()

        # Cross-context with trace (THE REAL TEST)
        cross = evaluate_cross_context(
            model, episodes, fact_types, verbose=False)

        # In-context baseline
        baseline = evaluate_baseline(
            model, episodes, fact_types, tokenizer, verbose=False)

        # Cross-context without trace (lower bound)
        cross_bl = evaluate_cross_context_baseline(
            model, episodes, fact_types)

        dt = time.time() - t0
        gap = cross.accuracy - cross_bl.accuracy

        print(f"  {n_facts:>7}  {cross.accuracy:>9.1%}  "
              f"{baseline.accuracy:>9.1%}  "
              f"{cross_bl.accuracy:>9.1%}  "
              f"{'+' if gap > 0 else ''}{gap:>6.1%}")

    print("-" * 60)
    print()
    print("  Cross-ctx:   Trace-based retrieval (question-only, no in-context facts)")
    print("  Baseline:    In-context (all facts + question in one pass)")
    print("  Cross BL:    No trace, question-only (expected ~random)")
    print("  Gap:         Cross-ctx minus Cross BL")
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
