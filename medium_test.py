#!/usr/bin/env python3
"""GPT-2 Medium (355M) transfer test.

Verifies that the Hebbian trace mechanism generalizes from GPT-2 Small
(124M, d=768) to GPT-2 Medium (355M, d=1024) without any retraining.

The trace module auto-scales projections to match d_model:
  - GPT-2 Small:  W_proj (512, 768),  ~1.1M trace params
  - GPT-2 Medium: W_proj (512, 1024), ~1.6M trace params

Usage:
    python medium_test.py                    # 50 episodes
    python medium_test.py --n-eval 100       # 100 episodes
    python medium_test.py --model gpt2       # run on Small for comparison
"""

import argparse
import time

import numpy as np
import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_fact_types,
    get_linking_bpe_ids,
    get_all_entity_ids,
    make_eval_episodes,
    evaluate_baseline,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_test(model_name: str = 'gpt2-medium', n_eval: int = 50,
             alpha: float | None = None):
    device = get_device()

    # Auto-select alpha based on model (logit scales differ)
    if alpha is None:
        alpha = 1.0 if 'medium' in model_name else 0.5

    print("=" * 70)
    print(f"  Hebbian Trace — {model_name} Transfer Test")
    print("=" * 70)
    print(f"  Device:    {device}")
    print(f"  Alpha:     {alpha}")
    print(f"  Episodes:  {n_eval}")
    print()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    print(f"Loading {model_name}...")
    t0 = time.time()
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
        model_name=model_name, device=device,
    )
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    dt = time.time() - t0

    config = model.gpt2.config
    trace_params = sum(p.numel() for p in model.trace.parameters())
    gpt2_params = sum(p.numel() for p in model.gpt2.parameters())

    print(f"  Loaded in {dt:.1f}s")
    print(f"  d_model:     {config.n_embd}")
    print(f"  n_layers:    {config.n_layer}")
    print(f"  GPT-2 params:{gpt2_params:,} (frozen)")
    print(f"  Trace params:{trace_params:,}")
    print()

    # Evaluate
    n_facts_list = [1, 3, 5, 7]

    def _ci(per_ep):
        arr = np.array(per_ep)
        if len(arr) <= 1:
            return arr[0], arr[0]
        rng = np.random.RandomState(0)
        boots = np.array([arr[rng.randint(0, len(arr), len(arr))].mean()
                          for _ in range(10000)])
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    def _fmt(r):
        lo, hi = _ci(r.per_episode_acc)
        return f"{r.accuracy * 100:5.1f} [{lo * 100:4.1f},{hi * 100:5.1f}]"

    w = 17
    sep = "-" * (6 + 3 * (w + 2) + 10 + 8)
    print(sep)
    print(f"  {'n':>3}  {'Cross+Trace':>{w}}  {'In-context':>{w}}  "
          f"{'No-trace':>{w}}  {'Gap':>8}  {'Time':>6}")
    print(sep)

    for n_facts in n_facts_list:
        episodes = make_eval_episodes(
            n_eval, n_facts, tokenizer, fact_types, seed=42)

        t0 = time.time()

        cross = evaluate_cross_context(
            model, episodes, fact_types, verbose=False)
        baseline = evaluate_baseline(
            model, episodes, fact_types, tokenizer, verbose=False)
        cross_bl = evaluate_cross_context_baseline(
            model, episodes, fact_types)

        dt = time.time() - t0
        gap = cross.accuracy - cross_bl.accuracy

        print(f"  {n_facts:>3}  {_fmt(cross):>{w}}  "
              f"{_fmt(baseline):>{w}}  "
              f"{_fmt(cross_bl):>{w}}  "
              f"{gap:>+7.1%}  {dt:>5.0f}s")

    print(sep)
    print("  (values: accuracy% [95% bootstrap CI], 10,000 resamples)")
    print()
    print(f"  Model: {model_name} ({gpt2_params/1e6:.0f}M params)")
    print(f"  Trace: {trace_params:,} params ({trace_params/gpt2_params*100:.1f}% "
          f"of model)")
    print(f"  Config: PS 8x_k16, alpha={alpha}, "
          f"trace_lr=1.0, decay=0.99")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Medium transfer test")
    parser.add_argument("--model", type=str, default="gpt2-medium",
                        help="Model name (gpt2, gpt2-medium)")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Logit injection strength (auto if not set)")
    args = parser.parse_args()

    run_test(model_name=args.model, n_eval=args.n_eval, alpha=args.alpha)


if __name__ == "__main__":
    main()
