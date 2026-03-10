#!/usr/bin/env python3
"""Experiment 18: GPT-2 Medium (355M) transfer test.

Does the Hebbian trace mechanism generalize from GPT-2 Small (124M, d=768)
to GPT-2 Medium (355M, d=1024)? Uses the same evaluation protocol as exp8.

The trace module auto-scales projections to match d_model:
  - GPT-2 Small:  W_proj (512, 768),  ~1.1M trace params
  - GPT-2 Medium: W_proj (512, 1024), ~1.6M trace params

Key question: does the logit injection mechanism work at larger scale?

Phase 1: Proof of concept (alpha=0.5, no PS) — n=1,3,5,7
Phase 2: Alpha sweep — find optimal injection strength for Medium
Phase 3: Pattern separation — 8x_k16 at best alpha
Phase 4: Head-to-head comparison — Small vs Medium at matched config

Usage:
    python -m hebbian_trace.experiments.exp18_gpt2_medium --quick
    python -m hebbian_trace.experiments.exp18_gpt2_medium --n-eval 50
    python -m hebbian_trace.experiments.exp18_gpt2_medium --n-eval 100
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    evaluate_gpt2_baseline, evaluate_gpt2_cross_context,
    evaluate_gpt2_cross_context_baseline,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str,
               alpha: float = 0.5, n_trace_heads: int = 8,
               d_trace: int = 64) -> GPT2WithTrace:
    """Load GPT-2 + trace model."""
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    model = GPT2WithTrace(
        n_trace_heads=n_trace_heads, d_trace=d_trace,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
        model_name=model_name, device=device,
    )
    dt = time.time() - t0

    config = model.gpt2.config
    print(f"  Loaded in {dt:.1f}s")
    print(f"  Model:       {model_name}")
    print(f"  d_model:     {config.n_embd}")
    print(f"  n_layers:    {config.n_layer}")
    print(f"  n_heads:     {config.n_head}")
    print(f"  inject_layer:{model.inject_layer} (mid-depth)")
    print(f"  GPT-2 params:{sum(p.numel() for p in model.gpt2.parameters()):,}")
    print(f"  Trace params:{sum(p.numel() for p in model.trace.parameters()):,}")
    print(f"  Trace d_model: {model.trace.d_model}")
    print(f"  W_proj shape:  {tuple(model.trace.W_proj.weight.shape)}")
    print(f"  W_val shape:   {tuple(model.trace.W_val.weight.shape)}")
    print(f"  W_out shape:   {tuple(model.trace.W_out.weight.shape)}")

    return model


def bootstrap_ci(per_ep, n_boot=10000, seed=0):
    """95% bootstrap CI for mean accuracy."""
    arr = np.array(per_ep)
    n = len(arr)
    if n <= 1:
        v = arr[0] if n == 1 else 0.0
        return v, v
    rng = np.random.RandomState(seed)
    boots = np.array([arr[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_eval_suite(model, tokenizer, fact_types, n_eval, n_facts_list,
                   label="", verbose=False):
    """Run cross-context eval at multiple n_facts values."""
    results = {}

    for n_facts in n_facts_list:
        episodes = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)

        t0 = time.time()

        cc_bl = evaluate_gpt2_cross_context_baseline(
            model, episodes, fact_types, verbose=False)
        cc = evaluate_gpt2_cross_context(
            model, episodes, fact_types, verbose=False)
        bl = evaluate_gpt2_baseline(
            model, episodes, fact_types, tokenizer, verbose=False)

        dt = time.time() - t0

        cc_lo, cc_hi = bootstrap_ci(cc.per_episode_acc)
        bl_lo, bl_hi = bootstrap_ci(bl.per_episode_acc)

        results[n_facts] = {
            'baseline': bl.accuracy,
            'cross_ctx': cc.accuracy,
            'cross_bl': cc_bl.accuracy,
            'gap': cc.accuracy - cc_bl.accuracy,
            'cross_ci': [cc_lo, cc_hi],
            'baseline_ci': [bl_lo, bl_hi],
            'time': dt,
        }

    return results


def print_results_table(results, label=""):
    """Print formatted results table with 95% CI."""
    if label:
        print(f"\n  {label}")
    print(f"  {'n':>3}  {'Baseline':>10}  {'Cross+Trace':>18}  "
          f"{'Cross BL':>10}  {'Gap':>8}  {'Time':>6}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*18}  {'─'*10}  {'─'*8}  {'─'*6}")

    for n_facts in sorted(results.keys()):
        r = results[n_facts]
        ci = r.get('cross_ci')
        if ci:
            cross_str = f"{r['cross_ctx']:.1%} [{ci[0]:.1%},{ci[1]:.1%}]"
        else:
            cross_str = f"{r['cross_ctx']:.1%}"
        print(f"  {n_facts:3d}  {r['baseline']:>10.1%}  "
              f"{cross_str:>18}  "
              f"{r['cross_bl']:>10.1%}  "
              f"{r['gap']:>+7.1%}  "
              f"{r['time']:>5.0f}s")


# ── Phase 1: Proof of Concept ────────────────────────────────────────

def run_phase1(model, tokenizer, fact_types, n_eval, n_facts_list):
    """Cross-context eval on GPT-2 Medium."""
    print(f"\n{'─' * 70}")
    print("PHASE 1: Proof of Concept (GPT-2 Medium, alpha=0.5, no PS)")
    print(f"{'─' * 70}")

    results = run_eval_suite(
        model, tokenizer, fact_types, n_eval, n_facts_list)
    print_results_table(results)
    return results


# ── Phase 2: Alpha Sweep ─────────────────────────────────────────────

def run_phase2(model, tokenizer, fact_types, n_eval):
    """Alpha sweep on Medium — logit scale may differ from Small."""
    print(f"\n{'─' * 70}")
    print("PHASE 2: Alpha Sweep (GPT-2 Medium)")
    print(f"{'─' * 70}")

    alphas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    test_ns = [1, 5]
    original_alpha = model.trace.alpha

    results = {}

    for alpha in alphas:
        model.trace.alpha = alpha
        r = run_eval_suite(model, tokenizer, fact_types, n_eval, test_ns)
        results[alpha] = r
        print(f"  alpha={alpha:.1f}: n=1 {r[1]['cross_ctx']:.1%}, "
              f"n=5 {r[5]['cross_ctx']:.1%}")

    # Summary
    print(f"\n  {'Alpha':>6}  {'n=1 Cross':>10}  {'n=1 BL':>8}  "
          f"{'n=5 Cross':>10}  {'n=5 BL':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}")

    best_alpha = None
    best_score = -1

    for alpha in alphas:
        r = results[alpha]
        cc1 = r[1]['cross_ctx']
        bl1 = r[1]['baseline']
        cc5 = r[5]['cross_ctx']
        bl5 = r[5]['baseline']
        avg = (cc1 + cc5) / 2
        marker = ""
        if avg > best_score:
            best_score = avg
            best_alpha = alpha
            marker = " <-- best"
        print(f"  {alpha:6.1f}  {cc1:>10.1%}  {bl1:>8.1%}  "
              f"{cc5:>10.1%}  {bl5:>8.1%}{marker}")

    print(f"\n  Best alpha: {best_alpha} (avg cross: {best_score:.1%})")
    model.trace.alpha = best_alpha

    return results, best_alpha


# ── Phase 3: Pattern Separation ──────────────────────────────────────

def run_phase3(model, tokenizer, fact_types, n_eval, n_facts_list):
    """Pattern separation on Medium."""
    print(f"\n{'─' * 70}")
    print("PHASE 3: Pattern Separation (8x_k16 on GPT-2 Medium)")
    print(f"{'─' * 70}")

    # Without PS
    model.disable_pattern_separation()
    results_no_ps = run_eval_suite(
        model, tokenizer, fact_types, n_eval, n_facts_list)

    # With PS
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    results_ps = run_eval_suite(
        model, tokenizer, fact_types, n_eval, n_facts_list)

    # Summary
    print(f"\n  {'n':>3}  {'No PS':>8}  {'8x_k16':>8}  {'Diff':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}")
    for n in sorted(results_no_ps.keys()):
        no = results_no_ps[n]['cross_ctx']
        ps = results_ps[n]['cross_ctx']
        print(f"  {n:3d}  {no:>8.1%}  {ps:>8.1%}  {ps-no:>+7.1%}")

    return results_no_ps, results_ps


# ── Phase 4: Head-to-Head Comparison ─────────────────────────────────

def run_phase4(model_medium, model_small, tokenizer, fact_types,
               n_eval, n_facts_list):
    """Compare Small vs Medium at same config."""
    print(f"\n{'─' * 70}")
    print("PHASE 4: Head-to-Head — GPT-2 Small vs Medium")
    print(f"{'─' * 70}")

    # Both with PS
    model_small.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model_medium.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    print(f"\n  Alpha: Small={model_small.trace.alpha}, "
          f"Medium={model_medium.trace.alpha}")

    r_small = run_eval_suite(
        model_small, tokenizer, fact_types, n_eval, n_facts_list,
        label="GPT-2 Small")
    r_medium = run_eval_suite(
        model_medium, tokenizer, fact_types, n_eval, n_facts_list,
        label="GPT-2 Medium")

    # Comparison table
    print(f"\n  {'n':>3}  {'Small Cross':>13}  {'Medium Cross':>14}  "
          f"{'Delta':>8}  {'Small BL':>10}  {'Medium BL':>11}")
    print(f"  {'─'*3}  {'─'*13}  {'─'*14}  {'─'*8}  {'─'*10}  {'─'*11}")

    for n in sorted(r_small.keys()):
        sc = r_small[n]['cross_ctx']
        mc = r_medium[n]['cross_ctx']
        sb = r_small[n]['baseline']
        mb = r_medium[n]['baseline']
        delta = mc - sc
        print(f"  {n:3d}  {sc:>13.1%}  {mc:>14.1%}  "
              f"{delta:>+7.1%}  {sb:>10.1%}  {mb:>11.1%}")

    return r_small, r_medium


# ── Main ─────────────────────────────────────────────────────────────

def run_experiment(n_eval: int = 50, quick: bool = False, seed: int = 42):
    device = get_device()

    print("=" * 70)
    print("  Exp 18: GPT-2 Medium (355M) Transfer Test")
    print("=" * 70)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    n_facts_list = [1, 3, 5, 7] if not quick else [1, 3, 5]

    # -- Load Medium --
    model_medium = load_model('gpt2-medium', device, alpha=0.5)
    model_medium.set_linking_token_ids(linking_ids)

    # Phase 1: Proof of concept
    phase1 = run_phase1(model_medium, tokenizer, fact_types,
                        n_eval, n_facts_list)

    # Phase 2: Alpha sweep
    phase2, best_alpha = run_phase2(model_medium, tokenizer, fact_types,
                                    n_eval)

    # Phase 3: Pattern separation (at best alpha)
    phase3_no_ps, phase3_ps = run_phase3(
        model_medium, tokenizer, fact_types, n_eval, n_facts_list)

    # Phase 4: Head-to-head (load Small for comparison)
    if not quick:
        model_small = load_model('gpt2', device, alpha=0.5)
        model_small.set_linking_token_ids(linking_ids)
        phase4_small, phase4_medium = run_phase4(
            model_medium, model_small, tokenizer, fact_types,
            n_eval, n_facts_list)
    else:
        phase4_small = phase4_medium = None

    # -- Save results --
    output = {
        "config": {
            "model": "gpt2-medium",
            "d_model": 1024,
            "n_layers": 24,
            "n_eval": n_eval,
            "seed": seed,
            "best_alpha": best_alpha,
            "n_trace_heads": 8,
            "d_trace": 64,
        },
        "phase1_proof_of_concept": {
            str(k): v for k, v in phase1.items()
        },
        "phase2_alpha_sweep": {
            str(alpha): {str(n): r for n, r in rs.items()}
            for alpha, rs in phase2.items()
        },
        "phase3_pattern_separation": {
            "no_ps": {str(k): v for k, v in phase3_no_ps.items()},
            "ps_8x_k16": {str(k): v for k, v in phase3_ps.items()},
        },
    }
    if phase4_small is not None:
        output["phase4_comparison"] = {
            "small": {str(k): v for k, v in phase4_small.items()},
            "medium": {str(k): v for k, v in phase4_medium.items()},
        }

    os.makedirs("results", exist_ok=True)
    path = "results/exp18_gpt2_medium.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")

    # -- Summary --
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 18 SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Best alpha for Medium: {best_alpha}")
    print(f"\n  Cross-context accuracy (PS 8x_k16, alpha={best_alpha}):")
    for n in sorted(phase3_ps.keys()):
        ps = phase3_ps[n]['cross_ctx']
        gap = phase3_ps[n]['gap']
        print(f"    n={n}: {ps:.1%} (gap: {gap:+.1%})")

    if phase4_small is not None:
        print(f"\n  Small vs Medium (PS 8x_k16):")
        for n in sorted(phase4_small.keys()):
            sc = phase4_small[n]['cross_ctx']
            mc = phase4_medium[n]['cross_ctx']
            print(f"    n={n}: Small {sc:.1%} → Medium {mc:.1%} "
                  f"({mc-sc:+.1%})")

    print(f"\n  Trace params: Small ~1.1M → Medium ~1.6M (+45%)")
    print(f"  Model params: Small 124M → Medium 355M (+186%)")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 18: GPT-2 Medium transfer test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 episodes, skip phase 4)")
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_eval = args.n_eval or (20 if args.quick else 50)

    run_experiment(n_eval=n_eval, quick=args.quick, seed=args.seed)
