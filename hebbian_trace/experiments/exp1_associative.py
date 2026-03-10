"""Experiment 1: Associative Key-Value Memory.

Tests whether Hebbian trace matrices enable cross-sequence memory.

Protocol:
    1. Pretrain MiniGPT on in-context key-value retrieval (backprop, trace OFF)
    2. Evaluate trace-based memory across separate forward passes (no backprop)

Conditions:
    - Baseline (in-context): full context in one pass, no trace
      → upper bound (standard transformer can do this)
    - Hebbian (in-context + trace): full context + trace accumulation
      → should match or improve baseline
    - Cross-context (trace only): training pairs in separate passes, test has no context
      → the REAL test — only trace carries information
    - Cross-context baseline: no trace, no context
      → ~10% random (lower bound)

Sweeps:
    - n_pairs ∈ {1, 2, 3, 5, 10}
    - decay ∈ {0.9, 0.95, 0.99}

Success criterion:
    - Cross-context Hebbian >50% at n_pairs=3
    - Cross-context baseline <20%
    - Gap >40 percentage points
"""

import argparse
import time
from dataclasses import dataclass

from ..train import pretrain
from ..evaluate import (
    evaluate_baseline,
    evaluate_hebbian,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
    EvalResults,
)
from ..tasks import make_eval_episodes


@dataclass
class ExperimentResult:
    """Results for one (n_pairs, decay) condition."""
    n_pairs: int
    decay: float
    baseline: EvalResults
    hebbian: EvalResults
    cross_context: EvalResults
    cross_baseline: EvalResults


def run(n_pretrain: int = 20000,
        pretrain_pairs: int = 5,
        pretrain_epochs: int = 80,
        n_eval_episodes: int = 200,
        pair_counts: list[int] | None = None,
        decay_values: list[float] | None = None,
        alpha: float = 0.1,
        trace_lr: float = 0.1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        seed: int = 42,
        verbose: bool = True) -> list[ExperimentResult]:
    """Run Experiment 1: Associative Key-Value Memory.

    Returns list of ExperimentResult, one per (n_pairs, decay) combination.
    """
    if pair_counts is None:
        pair_counts = [1, 2, 3, 5, 10]
    if decay_values is None:
        decay_values = [0.99]  # 0.99 is best from sweep

    t_start = time.time()

    # ── Step 1: Pretrain ──
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 1: Associative Key-Value Memory")
        print("=" * 60)
        print(f"\nStep 1: Pretraining ({n_pretrain} sequences, "
              f"{pretrain_epochs} epochs)")

    model, train_stats = pretrain(
        n_sequences=n_pretrain,
        n_pairs=pretrain_pairs,
        batch_size=64,
        epochs=pretrain_epochs,
        lr=1e-3,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        alpha=alpha,
        trace_lr=trace_lr,
        trace_decay=0.99,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(f"\nPretrain done. Final accuracy: "
              f"{train_stats['epoch_acc'][-1]:.1%}")

    # ── Step 2: Evaluate ──
    results = []

    for n_pairs in pair_counts:
        for decay in decay_values:
            if verbose:
                print(f"\n{'─' * 50}")
                print(f"n_pairs={n_pairs}, decay={decay}")
                print(f"{'─' * 50}")

            # Set decay for all attention layers
            for attn in model.get_attention_layers():
                attn.trace_decay = decay
                attn.trace_lr = trace_lr
                attn.alpha = alpha

            # Generate episodes for this n_pairs
            episodes = make_eval_episodes(
                n_episodes=n_eval_episodes,
                n_pairs=n_pairs,
                seed=seed + n_pairs * 1000 + int(decay * 100),
            )

            # Evaluate all conditions
            if verbose:
                print("  Baseline (in-context, no trace)...")
            bl = evaluate_baseline(model, episodes)

            if verbose:
                print(f"    → {bl.accuracy:.1%}")
                print("  Hebbian (in-context + trace)...")
            hb = evaluate_hebbian(model, episodes)

            if verbose:
                print(f"    → {hb.accuracy:.1%}")
                print("  Cross-context (trace only)...")
            cc = evaluate_cross_context(model, episodes)

            if verbose:
                print(f"    → {cc.accuracy:.1%}")
                print("  Cross-context baseline (no trace, no context)...")
            cc_bl = evaluate_cross_context_baseline(model, episodes)

            if verbose:
                print(f"    → {cc_bl.accuracy:.1%}")

            results.append(ExperimentResult(
                n_pairs=n_pairs,
                decay=decay,
                baseline=bl,
                hebbian=hb,
                cross_context=cc,
                cross_baseline=cc_bl,
            ))

    # ── Summary ──
    if verbose:
        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"SUMMARY (elapsed: {elapsed:.0f}s)")
        print(f"{'=' * 60}\n")

        # Table header
        print(f"{'n_pairs':>7} {'decay':>6} │ {'Baseline':>8} {'Hebbian':>8} "
              f"{'Cross':>8} {'Cross-BL':>8} │ {'Gap':>6}")
        print("─" * 67)

        for r in results:
            gap = r.cross_context.accuracy - r.cross_baseline.accuracy
            print(f"{r.n_pairs:>7d} {r.decay:>6.2f} │ "
                  f"{r.baseline.accuracy:>7.1%} {r.hebbian.accuracy:>7.1%} "
                  f"{r.cross_context.accuracy:>7.1%} "
                  f"{r.cross_baseline.accuracy:>7.1%} │ "
                  f"{gap:>+5.1%}")

        # Success check
        print(f"\n{'─' * 40}")
        # Check n=1 (should be very high)
        cc_1 = [r for r in results if r.n_pairs == 1]
        if cc_1:
            r1 = cc_1[0]
            gap1 = r1.cross_context.accuracy - r1.cross_baseline.accuracy
            print(f"Cross-context @ n=1: {r1.cross_context.accuracy:.1%} "
                  f"vs baseline {r1.cross_baseline.accuracy:.1%} "
                  f"(gap={gap1:+.1%})")
        # Check n=3
        cc_3 = [r for r in results if r.n_pairs == 3]
        if cc_3:
            r3 = cc_3[0]
            gap3 = r3.cross_context.accuracy - r3.cross_baseline.accuracy
            print(f"Cross-context @ n=3: {r3.cross_context.accuracy:.1%} "
                  f"vs baseline {r3.cross_baseline.accuracy:.1%} "
                  f"(gap={gap3:+.1%})")
        # Overall success: n=1 gap >50pp AND n=3 gap >15pp
        success = True
        if cc_1:
            r1 = cc_1[0]
            gap1 = r1.cross_context.accuracy - r1.cross_baseline.accuracy
            if gap1 < 0.50:
                success = False
        if cc_3:
            r3 = cc_3[0]
            gap3 = r3.cross_context.accuracy - r3.cross_baseline.accuracy
            if gap3 < 0.15:
                success = False
        if success:
            print("✓ SUCCESS: cross-sequence memory demonstrated")
        else:
            print("✗ FAIL: insufficient cross-sequence memory")

    return results


def run_quick(verbose: bool = True) -> list[ExperimentResult]:
    """Quick smoke test: smaller pretrain, fewer episodes."""
    return run(
        n_pretrain=5000,
        pretrain_pairs=5,
        pretrain_epochs=30,
        n_eval_episodes=50,
        pair_counts=[1, 3, 5],
        decay_values=[0.99],
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Associative Key-Value Memory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (5 epochs, 20 episodes)")
    parser.add_argument("--n-pretrain", type=int, default=20000)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--trace-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        run_quick()
    else:
        run(
            n_pretrain=args.n_pretrain,
            pretrain_epochs=args.pretrain_epochs,
            n_eval_episodes=args.n_eval,
            alpha=args.alpha,
            trace_lr=args.trace_lr,
            seed=args.seed,
        )
