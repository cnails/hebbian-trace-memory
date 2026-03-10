"""Experiment 6: Fact Updates — Temporal Memory Dynamics.

Tests whether Hebbian traces can handle fact UPDATES:
    Phase 1: store "My city is Moscow ."
    Phase 2: store "My city is London ." (update)
    Test:    "Where do I live ?" → should answer "London", not "Moscow"

This creates pressure for temporal dynamics that exp4 (per-head decay)
lacked: old facts must fade to allow new facts to dominate.

Key metrics:
    - Update accuracy: does model predict NEW value for updated facts?
    - Old value rate: does model still predict OLD value? (interference)
    - Stable accuracy: does model retain NON-updated facts?

Configurations:
    - Decay rates: uniform 0.99/0.95/0.90, gradient, split
    - Pattern separation: 8x_k16 (best from exp5)
    - Adaptive alpha: score-only

The hypothesis: uniform decay=0.99 fails at updates (old trace too strong).
Per-head decay with fast heads should help updates while slow heads
maintain stability — the CLS benefit that exp4 couldn't show.
"""

import argparse
import time

import torch

from ..model import MiniGPT
from ..nlp_tasks import NLP_VOCAB, make_nlp_update_eval_episodes
from ..nlp_evaluate import (
    NLPUpdateEvalResults,
    evaluate_fact_update,
    evaluate_fact_update_baseline,
)
from .exp2_nlp_facts import get_device, load_nlp_model, pretrain_nlp


# ── Configurations ──

DECAY_CONFIGS = {
    'uniform_099': [0.99] * 8,
    'uniform_097': [0.97] * 8,
    'uniform_095': [0.95] * 8,
    'uniform_090': [0.90] * 8,
    'gradient':    [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
    'split_4_4':   [0.99] * 4 + [0.90] * 4,
}

FACT_CONFIGS = [
    # (n_facts, n_updates)
    (3, 1),
    (5, 2),
    (10, 5),
]

N_EVAL = 200
ALPHA = 0.1
TRACE_LR = 0.1


# ── Evaluation helpers ──

def evaluate_decay_config(
    model: MiniGPT,
    decay_name: str,
    decay_rates: list[float],
    n_facts: int,
    n_updates: int,
    n_episodes: int,
    seed: int,
    use_pattern_sep: bool = True,
    use_adaptive: bool = False,
    norm_target: float = 5.0,
    erase_lr: float | None = None,
) -> dict:
    """Run fact update eval with given decay config."""
    # Set trace parameters
    for attn in model.get_attention_layers():
        attn.trace_lr = TRACE_LR
        attn.alpha = ALPHA

    # Set decay
    model.set_per_head_decay(decay_rates)

    # Pattern separation
    if use_pattern_sep:
        model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    else:
        model.disable_pattern_separation()

    # Adaptive alpha
    model.set_adaptive_alpha(use_adaptive, norm_target, score_only=True)

    # Generate episodes
    episodes = make_nlp_update_eval_episodes(
        n_episodes=n_episodes,
        n_facts=n_facts,
        n_updates=n_updates,
        seed=seed + n_facts * 1000,
        tier=1,
    )

    result = evaluate_fact_update(model, episodes, erase_lr=erase_lr)

    # Cleanup
    model.disable_pattern_separation()
    model.set_adaptive_alpha(False, 1.0)
    model.set_erase_mode(False)
    # Reset to uniform 0.99
    model.set_per_head_decay([0.99] * model.n_heads)

    return {
        'decay': decay_name,
        'n_facts': n_facts,
        'n_updates': n_updates,
        'update_acc': result.update_accuracy,
        'old_val_rate': result.old_value_rate,
        'stable_acc': result.stable_accuracy,
        'overall_acc': result.overall_accuracy,
        'n_updated': result.n_updated,
        'n_stable': result.n_stable,
    }


def calibrate_norm_target(model: MiniGPT, seed: int = 99) -> float:
    """Measure score trace norm after 1 fact for adaptive alpha."""
    device = next(model.parameters()).device

    episodes = make_nlp_update_eval_episodes(
        n_episodes=1, n_facts=1, n_updates=0, seed=seed, tier=1)
    ep = episodes[0]

    model.reset_traces()
    model.set_trace_mode(use=False, update=True)

    for train_seq in ep.phase1_sequences:
        input_tensor = torch.tensor(
            [train_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_tensor)

    score_norms = []
    for attn in model.get_attention_layers():
        score_norms.append(attn.traces.norm(dim=(1, 2)))
    mean_norm = torch.stack(score_norms).mean().item()

    model.reset_traces()
    model.set_trace_mode(use=False, update=False)
    return mean_norm


# ── Main ──

def run(
    load_path: str | None = None,
    n_eval: int = N_EVAL,
    fact_configs: list[tuple[int, int]] | None = None,
    decay_configs: dict | None = None,
    seed: int = 42,
    device_name: str | None = None,
    verbose: bool = True,
):
    """Run fact update experiment.

    Phase 1: Compare decay configs with pattern separation (8x_k16).
    Phase 2: Best decay config + adaptive alpha.
    Phase 3: Ablation — pattern sep ON vs OFF for best config.
    """
    if fact_configs is None:
        fact_configs = FACT_CONFIGS
    if decay_configs is None:
        decay_configs = DECAY_CONFIGS

    device = get_device(device_name)
    t_start = time.time()

    print("=" * 70)
    print("EXPERIMENT 6: Fact Updates — Temporal Memory Dynamics")
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

    # ── Step 2: No-trace baseline ──
    print(f"\n{'─' * 70}")
    print("NO-TRACE BASELINE (question-only, no trace)")
    print(f"{'─' * 70}")

    for n_facts, n_updates in fact_configs:
        episodes = make_nlp_update_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts, n_updates=n_updates,
            seed=seed + n_facts * 1000, tier=1)
        bl = evaluate_fact_update_baseline(model, episodes)
        print(f"  n={n_facts:>2d} (upd={n_updates}): "
              f"update={bl.update_accuracy:.1%}  "
              f"stable={bl.stable_accuracy:.1%}  "
              f"overall={bl.overall_accuracy:.1%}")

    # ── Step 3: Phase 1 — Decay configs with pattern sep ──
    print(f"\n{'=' * 70}")
    print(f"PHASE 1: Decay Configs + Pattern Sep (8x_k16)")
    print(f"{'=' * 70}")

    all_results = {}
    for decay_name, decay_rates in decay_configs.items():
        print(f"\n{'─' * 70}")
        print(f"{decay_name}: {[f'{d:.2f}' for d in decay_rates[:4]]}..."
              if len(set(decay_rates)) > 1
              else f"{decay_name}: all heads = {decay_rates[0]:.2f}")
        print(f"{'─' * 70}")

        results = []
        for n_facts, n_updates in fact_configs:
            r = evaluate_decay_config(
                model, decay_name, decay_rates,
                n_facts, n_updates, n_eval, seed,
                use_pattern_sep=True)
            results.append(r)
            print(f"  n={n_facts:>2d} (upd={n_updates}): "
                  f"update={r['update_acc']:.1%}  "
                  f"old_val={r['old_val_rate']:.1%}  "
                  f"stable={r['stable_acc']:.1%}  "
                  f"overall={r['overall_acc']:.1%}")
        all_results[decay_name] = results

    # ── Phase 1 Summary Tables ──
    print(f"\n{'=' * 70}")
    elapsed = time.time() - t_start
    print(f"PHASE 1 SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 70}\n")

    decay_names = list(decay_configs.keys())

    # Update accuracy table
    print("UPDATE ACCURACY (higher = better at learning new values)")
    header = f"{'n (upd)':>8} │"
    for name in decay_names:
        header += f" {name:>12}"
    print(header)
    print("─" * (11 + 13 * len(decay_names)))
    for i, (n, u) in enumerate(fact_configs):
        row = f"{n:>3d} ({u:>1d})  │"
        for name in decay_names:
            acc = all_results[name][i]['update_acc']
            row += f" {acc:>11.1%}"
        print(row)

    # Old value rate table
    print(f"\nOLD VALUE RATE (lower = less interference from old facts)")
    header = f"{'n (upd)':>8} │"
    for name in decay_names:
        header += f" {name:>12}"
    print(header)
    print("─" * (11 + 13 * len(decay_names)))
    for i, (n, u) in enumerate(fact_configs):
        row = f"{n:>3d} ({u:>1d})  │"
        for name in decay_names:
            rate = all_results[name][i]['old_val_rate']
            row += f" {rate:>11.1%}"
        print(row)

    # Stable accuracy table
    print(f"\nSTABLE ACCURACY (higher = better retention of non-updated facts)")
    header = f"{'n (upd)':>8} │"
    for name in decay_names:
        header += f" {name:>12}"
    print(header)
    print("─" * (11 + 13 * len(decay_names)))
    for i, (n, u) in enumerate(fact_configs):
        row = f"{n:>3d} ({u:>1d})  │"
        for name in decay_names:
            acc = all_results[name][i]['stable_acc']
            row += f" {acc:>11.1%}"
        print(row)

    # Overall accuracy table
    print(f"\nOVERALL ACCURACY")
    header = f"{'n (upd)':>8} │"
    for name in decay_names:
        header += f" {name:>12}"
    print(header)
    print("─" * (11 + 13 * len(decay_names)))
    for i, (n, u) in enumerate(fact_configs):
        row = f"{n:>3d} ({u:>1d})  │"
        for name in decay_names:
            acc = all_results[name][i]['overall_acc']
            row += f" {acc:>11.1%}"
        print(row)

    # ── Find best config ──
    # Best = highest overall at the largest n_facts
    last_idx = len(fact_configs) - 1
    best_name = max(
        decay_names,
        key=lambda name: all_results[name][last_idx]['overall_acc'])
    best_overall = all_results[best_name][last_idx]['overall_acc']
    best_update = all_results[best_name][last_idx]['update_acc']
    best_stable = all_results[best_name][last_idx]['stable_acc']
    n_last, u_last = fact_configs[last_idx]
    print(f"\nBest at n={n_last}: {best_name} "
          f"(overall={best_overall:.1%}, "
          f"update={best_update:.1%}, "
          f"stable={best_stable:.1%})")

    # ── Phase 2: Best config + adaptive alpha ──
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: {best_name} + Adaptive Alpha (score-only)")
    print(f"{'=' * 70}")

    norm_1 = calibrate_norm_target(model, seed=99)
    norm_target = norm_1 * 3
    print(f"  1-fact norm: {norm_1:.4f}, norm_target: {norm_target:.4f}")

    best_rates = decay_configs[best_name]

    phase2_conditions = {
        f'{best_name}_fixed': (best_rates, False),
        f'{best_name}_adaptive': (best_rates, True),
    }

    phase2_results = {}
    for cond_name, (rates, use_adaptive) in phase2_conditions.items():
        print(f"\n{'─' * 70}")
        print(f"{cond_name}")
        print(f"{'─' * 70}")

        results = []
        for n_facts, n_updates in fact_configs:
            r = evaluate_decay_config(
                model, cond_name, rates,
                n_facts, n_updates, n_eval, seed,
                use_pattern_sep=True,
                use_adaptive=use_adaptive,
                norm_target=norm_target)
            results.append(r)
            print(f"  n={n_facts:>2d} (upd={n_updates}): "
                  f"update={r['update_acc']:.1%}  "
                  f"old_val={r['old_val_rate']:.1%}  "
                  f"stable={r['stable_acc']:.1%}  "
                  f"overall={r['overall_acc']:.1%}")
        phase2_results[cond_name] = results

    # ── Phase 3: Ablation — pattern sep ON vs OFF ──
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: Ablation — Pattern Sep ON vs OFF ({best_name})")
    print(f"{'=' * 70}")

    phase3_conditions = {
        f'{best_name}_no_ps': (best_rates, False),
        f'{best_name}_ps': (best_rates, True),
    }

    phase3_results = {}
    for cond_name, (rates, use_ps) in phase3_conditions.items():
        print(f"\n{'─' * 70}")
        print(f"{cond_name}")
        print(f"{'─' * 70}")

        results = []
        for n_facts, n_updates in fact_configs:
            r = evaluate_decay_config(
                model, cond_name, rates,
                n_facts, n_updates, n_eval, seed,
                use_pattern_sep=use_ps)
            results.append(r)
            print(f"  n={n_facts:>2d} (upd={n_updates}): "
                  f"update={r['update_acc']:.1%}  "
                  f"old_val={r['old_val_rate']:.1%}  "
                  f"stable={r['stable_acc']:.1%}  "
                  f"overall={r['overall_acc']:.1%}")
        phase3_results[cond_name] = results

    # ── Phase 4: Reconsolidation Erasure ──
    print(f"\n{'=' * 70}")
    print(f"PHASE 4: Reconsolidation Erasure ({best_name} + PS 8x_k16)")
    print(f"{'=' * 70}")
    print("  Erase old Q→V association before writing new during phase 2.")
    print("  Q normalized before erase to prevent scale-dependent damage.")

    erase_lrs = [1.0, 3.0, 5.0, 8.0]
    best_rates = decay_configs[best_name]

    phase4_results = {}
    # Baseline: no erase (same as phase 1 result for best config)
    cond_name = f'{best_name}_no_erase'
    phase4_results[cond_name] = all_results[best_name]
    print(f"\n{'─' * 70}")
    print(f"{cond_name} (reference)")
    print(f"{'─' * 70}")
    for i, (n, u) in enumerate(fact_configs):
        r = all_results[best_name][i]
        print(f"  n={n:>2d} (upd={u}): "
              f"update={r['update_acc']:.1%}  "
              f"old_val={r['old_val_rate']:.1%}  "
              f"stable={r['stable_acc']:.1%}  "
              f"overall={r['overall_acc']:.1%}")

    for elr in erase_lrs:
        cond_name = f'{best_name}_erase_{elr}'
        print(f"\n{'─' * 70}")
        print(f"{cond_name}")
        print(f"{'─' * 70}")

        results = []
        for n_facts, n_updates in fact_configs:
            r = evaluate_decay_config(
                model, cond_name, best_rates,
                n_facts, n_updates, n_eval, seed,
                use_pattern_sep=True,
                erase_lr=elr)
            results.append(r)
            print(f"  n={n_facts:>2d} (upd={n_updates}): "
                  f"update={r['update_acc']:.1%}  "
                  f"old_val={r['old_val_rate']:.1%}  "
                  f"stable={r['stable_acc']:.1%}  "
                  f"overall={r['overall_acc']:.1%}")
        phase4_results[cond_name] = results

    # Phase 4 summary
    print(f"\n{'=' * 70}")
    print(f"PHASE 4 SUMMARY: Reconsolidation Erasure")
    print(f"{'=' * 70}\n")

    p4_names = list(phase4_results.keys())
    for i, (n, u) in enumerate(fact_configs):
        print(f"n={n}, updates={u}:")
        print(f"  {'Config':<30s} │ {'Update':>7s} │ {'OldVal':>7s} │ "
              f"{'Stable':>7s} │ {'Overall':>7s}")
        print(f"  {'─' * 30}─┼─{'─' * 7}─┼─{'─' * 7}─┼─"
              f"{'─' * 7}─┼─{'─' * 7}")
        for name in p4_names:
            r = phase4_results[name][i]
            print(f"  {name:<30s} │ {r['update_acc']:>6.1%} │ "
                  f"{r['old_val_rate']:>6.1%} │ "
                  f"{r['stable_acc']:>6.1%} │ "
                  f"{r['overall_acc']:>6.1%}")
        print()

    # ── Phase 5: Full Combination (decay + adaptive + erase) ──
    print(f"\n{'=' * 70}")
    print(f"PHASE 5: Full Combination ({best_name} + adaptive + erase)")
    print(f"{'=' * 70}")
    print("  Three orthogonal mechanisms: CLS decay + adaptive alpha + erase.")

    phase5_conditions = [
        (f'{best_name}_base', best_rates, False, None),
        (f'{best_name}_adaptive', best_rates, True, None),
        (f'{best_name}_erase_3', best_rates, False, 3.0),
        (f'{best_name}_adapt+erase_3', best_rates, True, 3.0),
        (f'{best_name}_adapt+erase_5', best_rates, True, 5.0),
    ]

    phase5_results = {}
    for cond_name, rates, use_adaptive, elr in phase5_conditions:
        print(f"\n{'─' * 70}")
        print(f"{cond_name}")
        print(f"{'─' * 70}")

        results = []
        for n_facts, n_updates in fact_configs:
            r = evaluate_decay_config(
                model, cond_name, rates,
                n_facts, n_updates, n_eval, seed,
                use_pattern_sep=True,
                use_adaptive=use_adaptive,
                norm_target=norm_target,
                erase_lr=elr)
            results.append(r)
            print(f"  n={n_facts:>2d} (upd={n_updates}): "
                  f"update={r['update_acc']:.1%}  "
                  f"old_val={r['old_val_rate']:.1%}  "
                  f"stable={r['stable_acc']:.1%}  "
                  f"overall={r['overall_acc']:.1%}")
        phase5_results[cond_name] = results

    # Phase 5 summary
    print(f"\n{'=' * 70}")
    print(f"PHASE 5 SUMMARY: Full Combination")
    print(f"{'=' * 70}\n")

    p5_names = list(phase5_results.keys())
    for i, (n, u) in enumerate(fact_configs):
        print(f"n={n}, updates={u}:")
        print(f"  {'Config':<35s} │ {'Update':>7s} │ {'OldVal':>7s} │ "
              f"{'Stable':>7s} │ {'Overall':>7s}")
        print(f"  {'─' * 35}─┼─{'─' * 7}─┼─{'─' * 7}─┼─"
              f"{'─' * 7}─┼─{'─' * 7}")
        for name in p5_names:
            r = phase5_results[name][i]
            print(f"  {name:<35s} │ {r['update_acc']:>6.1%} │ "
                  f"{r['old_val_rate']:>6.1%} │ "
                  f"{r['stable_acc']:>6.1%} │ "
                  f"{r['overall_acc']:>6.1%}")
        print()

    # ── Final Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 70}\n")

    print(f"Task: store N facts, update K of them, test all N.")
    print(f"Update accuracy = predict NEW value. "
          f"Stable accuracy = retain ORIGINAL value.")
    print()

    # Combine key results for final table
    all_final = {}
    # Phase 1 key configs
    for name in ['uniform_099', 'uniform_090', best_name]:
        if name in all_results:
            all_final[name] = all_results[name]
    # Phase 2 adaptive
    all_final.update(phase2_results)
    # Phase 5 combined
    all_final.update(phase5_results)

    final_names = list(all_final.keys())
    for i, (n, u) in enumerate(fact_configs):
        print(f"n={n}, updates={u}:")
        print(f"  {'Config':<35s} │ {'Update':>7s} │ {'OldVal':>7s} │ "
              f"{'Stable':>7s} │ {'Overall':>7s}")
        print(f"  {'─' * 35}─┼─{'─' * 7}─┼─{'─' * 7}─┼─"
              f"{'─' * 7}─┼─{'─' * 7}")
        for name in final_names:
            r = all_final[name][i]
            print(f"  {name:<35s} │ {r['update_acc']:>6.1%} │ "
                  f"{r['old_val_rate']:>6.1%} │ "
                  f"{r['stable_acc']:>6.1%} │ "
                  f"{r['overall_acc']:>6.1%}")
        print()


def run_quick(device_name: str | None = None, load_path: str | None = None):
    """Quick test: fewer episodes, subset of configs."""
    quick_decay = {
        'uniform_099': [0.99] * 8,
        'uniform_095': [0.95] * 8,
        'gradient':    [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
    }
    quick_facts = [
        (3, 1),
        (5, 2),
    ]
    run(
        load_path=load_path,
        n_eval=50,
        fact_configs=quick_facts,
        decay_configs=quick_decay,
        device_name=device_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 6: Fact Updates — Temporal Memory Dynamics")
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
