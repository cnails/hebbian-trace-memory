"""Experiment 3: Adaptive Alpha — prevent trace from killing in-context attention.

Problem: at n=10 facts, trace norm grows large and overwhelms vanilla attention,
dropping Hebbian in-context accuracy from 73.5% to 29.4%.

Solution: alpha_eff = alpha / (1 + ||T_h|| / norm_target)
- Small trace (few facts): alpha_eff ≈ alpha (full strength)
- Large trace (many facts): alpha_eff decreases, preserving in-context

This experiment pretrains once, then compares fixed vs adaptive alpha
across n_facts=[1,2,3,5,10] and several norm_target values.
"""

import argparse
import time

import torch

from ..model import MiniGPT
from ..nlp_tasks import NLP_VOCAB, make_nlp_eval_episodes
from ..nlp_evaluate import (
    evaluate_baseline,
    evaluate_hebbian,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)
from .exp2_nlp_facts import pretrain_nlp, get_device, save_nlp_model, load_nlp_model


def measure_trace_norms(model: MiniGPT) -> dict:
    """Measure current trace norms across all layers/heads."""
    score_norms = []
    value_norms = []
    for attn in model.get_attention_layers():
        # Per-head Frobenius norms
        s_norms = attn.traces.norm(dim=(1, 2))  # (H,)
        v_norms = attn.value_traces.norm(dim=(1, 2))  # (H,)
        score_norms.append(s_norms)
        value_norms.append(v_norms)
    return {
        'score_per_layer': [n.tolist() for n in score_norms],
        'value_per_layer': [n.tolist() for n in value_norms],
        'score_mean': torch.stack(score_norms).mean().item(),
        'value_mean': torch.stack(value_norms).mean().item(),
        'score_max': torch.stack(score_norms).max().item(),
        'value_max': torch.stack(value_norms).max().item(),
    }


def calibrate_norm_target(model: MiniGPT, n_calibration_facts: int = 1,
                          seed: int = 99) -> float:
    """Measure trace norm after storing n_calibration_facts facts.

    Returns a norm_target calibrated so that alpha stays near full
    strength for up to ~n_calibration_facts facts.
    """
    device = next(model.parameters()).device
    vocab = NLP_VOCAB

    # Generate a calibration episode
    episodes = make_nlp_eval_episodes(
        n_episodes=1, n_facts=n_calibration_facts, seed=seed, tier=1)
    ep = episodes[0]

    model.reset_traces()
    model.set_trace_mode(use=False, update=True)  # ACh high: write only

    # Process all training sequences
    for train_seq in ep.train_sequences:
        input_tensor = torch.tensor([train_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_tensor)

    norms = measure_trace_norms(model)
    model.reset_traces()
    model.set_trace_mode(use=False, update=False)

    # Use max of score and value norms as target
    return max(norms['score_mean'], norms['value_mean'])


def evaluate_condition(model: MiniGPT, n_facts: int, n_episodes: int,
                       seed: int, alpha: float, trace_lr: float,
                       adaptive: bool, norm_target: float,
                       score_only: bool = False) -> dict:
    """Evaluate one condition (fixed or adaptive alpha at given n_facts)."""
    # Set trace parameters
    for attn in model.get_attention_layers():
        attn.trace_decay = 0.99
        attn.trace_lr = trace_lr
        attn.alpha = alpha

    model.set_adaptive_alpha(adaptive, norm_target, score_only=score_only)

    episodes = make_nlp_eval_episodes(
        n_episodes=n_episodes, n_facts=n_facts,
        seed=seed + n_facts * 1000, tier=1)

    bl = evaluate_baseline(model, episodes)
    hb = evaluate_hebbian(model, episodes)
    cc = evaluate_cross_context(model, episodes)

    # Measure trace norms BEFORE cross-baseline resets them
    norms = measure_trace_norms(model)

    cc_bl = evaluate_cross_context_baseline(model, episodes)

    return {
        'n_facts': n_facts,
        'baseline': bl.accuracy,
        'hebbian': hb.accuracy,
        'cross_ctx': cc.accuracy,
        'cross_bl': cc_bl.accuracy,
        'score_norm': norms['score_mean'],
        'value_norm': norms['value_mean'],
    }


def run(
    n_pretrain: int = 20000,
    pretrain_epochs: int = 60,
    n_eval_episodes: int = 200,
    fact_counts: list[int] | None = None,
    alpha: float = 0.1,
    trace_lr: float = 0.1,
    norm_targets: list[float] | None = None,
    auto_calibrate: bool = True,
    seed: int = 42,
    device_name: str | None = None,
    load_path: str | None = None,
    save_path: str | None = None,
    verbose: bool = True,
):
    """Run adaptive alpha experiment.

    Pretrains once (or loads model), then evaluates:
    1. Fixed alpha (baseline)
    2. Adaptive alpha at several norm_target values
    """
    if fact_counts is None:
        fact_counts = [1, 2, 3, 5, 10]

    device = get_device(device_name)
    t_start = time.time()

    print("=" * 65)
    print("EXPERIMENT 3: Adaptive Alpha")
    print("=" * 65)

    # ── Step 1: Get model ──
    if load_path:
        print(f"\nLoading model from {load_path}...")
        model = load_nlp_model(load_path, device_name)
        print("  Model loaded.")
    else:
        print(f"\nStep 1: Pretraining ({n_pretrain} sequences, "
              f"{pretrain_epochs} epochs)")
        model, train_stats = pretrain_nlp(
            n_sequences=n_pretrain,
            max_facts=5,
            batch_size=64,
            epochs=pretrain_epochs,
            lr=1e-3,
            d_model=256,
            n_heads=8,
            n_layers=8,
            max_seq_len=128,
            dropout=0.1,
            alpha=alpha,
            trace_lr=trace_lr,
            trace_decay=0.99,
            use_raw_embed=True,
            use_key_q=True,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        print(f"  Final accuracy: {train_stats['epoch_acc'][-1]:.1%}")

        vocab = NLP_VOCAB
        model.set_linking_token_ids(vocab.linking_tokens)

        if save_path:
            save_nlp_model(model, save_path)
            print(f"  Model saved to {save_path}")

    # ── Step 2: Calibrate norm_target ──
    if auto_calibrate:
        print("\nCalibrating norm_target...")
        # Measure trace norm after 1, 3, and 5 facts
        for n_cal in [1, 3, 5]:
            cal_norm = calibrate_norm_target(model, n_calibration_facts=n_cal, seed=99)
            print(f"  After {n_cal} fact(s): mean trace norm = {cal_norm:.4f}")

        norm_1 = calibrate_norm_target(model, n_calibration_facts=1, seed=99)
        # Default targets: scale relative to single-fact norm
        if norm_targets is None:
            norm_targets = [
                norm_1 * 3,    # starts reducing at ~3 facts
                norm_1 * 5,    # starts reducing at ~5 facts
                norm_1 * 10,   # starts reducing at ~10 facts
            ]
            print(f"\n  Auto norm_targets: {[f'{t:.4f}' for t in norm_targets]}")
            print(f"  (based on 1-fact norm = {norm_1:.4f})")
    else:
        if norm_targets is None:
            norm_targets = [0.5, 1.0, 2.0]

    # ── Step 3: Evaluate ──
    print(f"\nStep 2: Evaluating ({n_eval_episodes} episodes per condition)")

    # Fixed alpha baseline
    print(f"\n{'─' * 65}")
    print(f"FIXED alpha={alpha}")
    print(f"{'─' * 65}")
    fixed_results = []
    for n_facts in fact_counts:
        r = evaluate_condition(
            model, n_facts, n_eval_episodes, seed, alpha, trace_lr,
            adaptive=False, norm_target=1.0)
        fixed_results.append(r)
        print(f"  n={n_facts:>2d}: BL={r['baseline']:.1%}  "
              f"Hebb={r['hebbian']:.1%}  "
              f"Cross={r['cross_ctx']:.1%}  "
              f"CrossBL={r['cross_bl']:.1%}  "
              f"||T_s||={r['score_norm']:.3f}  ||T_v||={r['value_norm']:.3f}")

    # Adaptive alpha (both traces) at each norm_target
    adaptive_both = {}
    for nt in norm_targets:
        print(f"\n{'─' * 65}")
        print(f"ADAPTIVE BOTH alpha={alpha}, norm_target={nt:.4f}")
        print(f"{'─' * 65}")
        results = []
        for n_facts in fact_counts:
            r = evaluate_condition(
                model, n_facts, n_eval_episodes, seed, alpha, trace_lr,
                adaptive=True, norm_target=nt, score_only=False)
            results.append(r)
            print(f"  n={n_facts:>2d}: BL={r['baseline']:.1%}  "
                  f"Hebb={r['hebbian']:.1%}  "
                  f"Cross={r['cross_ctx']:.1%}  "
                  f"CrossBL={r['cross_bl']:.1%}  "
                  f"||T_s||={r['score_norm']:.3f}  ||T_v||={r['value_norm']:.3f}")
        adaptive_both[nt] = results

    # Adaptive alpha (score trace only — value trace keeps fixed alpha)
    adaptive_score = {}
    for nt in norm_targets:
        print(f"\n{'─' * 65}")
        print(f"ADAPTIVE SCORE-ONLY alpha={alpha}, norm_target={nt:.4f}")
        print(f"{'─' * 65}")
        results = []
        for n_facts in fact_counts:
            r = evaluate_condition(
                model, n_facts, n_eval_episodes, seed, alpha, trace_lr,
                adaptive=True, norm_target=nt, score_only=True)
            results.append(r)
            print(f"  n={n_facts:>2d}: BL={r['baseline']:.1%}  "
                  f"Hebb={r['hebbian']:.1%}  "
                  f"Cross={r['cross_ctx']:.1%}  "
                  f"CrossBL={r['cross_bl']:.1%}  "
                  f"||T_s||={r['score_norm']:.3f}  ||T_v||={r['value_norm']:.3f}")
        adaptive_score[nt] = results

    # ── Summary comparison ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"COMPARISON TABLE (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 65}\n")

    # Use best norm_target (first one = 3x single-fact norm)
    best_nt = norm_targets[0]

    # Combined table: Fixed vs Adaptive-Both vs Adaptive-Score-Only
    print("HEBBIAN (in-context + trace)")
    print(f"{'n':>3} │ {'Fixed':>7} {'Both':>7} {'ScoreOnly':>10} │ {'BL':>7}")
    print("─" * 45)
    for i, n_facts in enumerate(fact_counts):
        f_h = fixed_results[i]['hebbian']
        b_h = adaptive_both[best_nt][i]['hebbian']
        s_h = adaptive_score[best_nt][i]['hebbian']
        bl = fixed_results[i]['baseline']
        print(f"{n_facts:>3d} │ {f_h:>6.1%} {b_h:>6.1%} {s_h:>9.1%} │ {bl:>6.1%}")

    print(f"\nCROSS-CONTEXT (trace only, question only)")
    print(f"{'n':>3} │ {'Fixed':>7} {'Both':>7} {'ScoreOnly':>10} │ {'Random':>7}")
    print("─" * 45)
    for i, n_facts in enumerate(fact_counts):
        f_c = fixed_results[i]['cross_ctx']
        b_c = adaptive_both[best_nt][i]['cross_ctx']
        s_c = adaptive_score[best_nt][i]['cross_ctx']
        rnd = fixed_results[i]['cross_bl']
        print(f"{n_facts:>3d} │ {f_c:>6.1%} {b_c:>6.1%} {s_c:>9.1%} │ {rnd:>6.1%}")

    # All norm_targets detail for score-only
    print(f"\nSCORE-ONLY detail across norm_targets:")
    print(f"{'n':>3} │ {'Fixed':>7}", end="")
    for nt in norm_targets:
        print(f"  {'NT='+f'{nt:.1f}':>8}", end="")
    print()
    print("─" * (15 + 10 * len(norm_targets)))
    print("Hebbian:")
    for i, n_facts in enumerate(fact_counts):
        f_h = fixed_results[i]['hebbian']
        print(f"{n_facts:>3d} │ {f_h:>6.1%}", end="")
        for nt in norm_targets:
            s_h = adaptive_score[nt][i]['hebbian']
            print(f"  {s_h:>7.1%}", end="")
        print()
    print("Cross-ctx:")
    for i, n_facts in enumerate(fact_counts):
        f_c = fixed_results[i]['cross_ctx']
        print(f"{n_facts:>3d} │ {f_c:>6.1%}", end="")
        for nt in norm_targets:
            s_c = adaptive_score[nt][i]['cross_ctx']
            print(f"  {s_c:>7.1%}", end="")
        print()

    # Final verdict
    print(f"\n{'─' * 40}")
    n10_idx = next((i for i, n in enumerate(fact_counts) if n == 10), None)
    if n10_idx is not None:
        f_h = fixed_results[n10_idx]['hebbian']
        f_c = fixed_results[n10_idx]['cross_ctx']
        bl = fixed_results[n10_idx]['baseline']
        s_h = adaptive_score[best_nt][n10_idx]['hebbian']
        s_c = adaptive_score[best_nt][n10_idx]['cross_ctx']
        print(f"n=10 summary (NT={best_nt:.1f}):")
        print(f"  Hebbian:    {f_h:.1%} → {s_h:.1%} (BL={bl:.1%})")
        print(f"  Cross-ctx:  {f_c:.1%} → {s_c:.1%}")
        if s_h > f_h and s_c >= f_c * 0.95:
            print("  SCORE-ONLY WINS: Hebbian improved, cross-ctx preserved")
        elif s_h > f_h:
            print(f"  TRADE-OFF: Hebbian +{s_h-f_h:.1%}, "
                  f"cross-ctx {s_c-f_c:+.1%}")


def run_quick(device_name: str | None = None):
    """Quick test (~8 min): small pretrain, fewer episodes."""
    run(
        n_pretrain=5000,
        pretrain_epochs=30,
        n_eval_episodes=50,
        fact_counts=[1, 3, 5, 10],
        device_name=device_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3: Adaptive Alpha")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (~8 min)")
    parser.add_argument("--n-pretrain", type=int, default=20000)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--trace-lr", type=float, default=0.1)
    parser.add_argument("--norm-targets", type=float, nargs="+", default=None,
                        help="Explicit norm_target values to test")
    parser.add_argument("--no-auto-calibrate", action="store_true",
                        help="Disable auto-calibration of norm_target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained model instead of training")
    parser.add_argument("--save", type=str, default=None,
                        help="Save model after pretraining")
    args = parser.parse_args()

    if args.quick:
        run_quick(device_name=args.device)
    else:
        run(
            n_pretrain=args.n_pretrain,
            pretrain_epochs=args.pretrain_epochs,
            n_eval_episodes=args.n_eval,
            alpha=args.alpha,
            trace_lr=args.trace_lr,
            norm_targets=args.norm_targets,
            auto_calibrate=not args.no_auto_calibrate,
            seed=args.seed,
            device_name=args.device,
            load_path=args.load,
            save_path=args.save,
        )
