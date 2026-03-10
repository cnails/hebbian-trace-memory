"""Experiment 4: Multi-Head Decay Specialization.

Different attention heads get different trace decay rates, creating
complementary learning systems within a transformer:
- High-decay heads (0.99): retain long-term facts across many sequences
- Low-decay heads (0.80): focus on recent facts, forget old ones quickly

This is the CLS (Complementary Learning Systems) hypothesis tested inside
a transformer's Hebbian trace mechanism. During pretrain, all heads are
identical (traces OFF). Per-head decay is set only at evaluation time.

Phase 1: Pure per-head decay (fixed alpha) — isolate decay effect
Phase 2: Best per-head decay + adaptive alpha (score-only) — compose both
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
from .exp3_adaptive_alpha import measure_trace_norms, calibrate_norm_target


# ── Decay configurations (8 heads) ──

DECAY_CONFIGS = {
    'uniform_099': [0.99] * 8,
    'uniform_080': [0.80] * 8,
    'split_4_4':   [0.99] * 4 + [0.80] * 4,
    'gradient':    [0.80 + i * (0.99 - 0.80) / 7 for i in range(8)],
    'extreme':     [0.999] * 2 + [0.95] * 4 + [0.70] * 2,
}


def measure_per_head_norms(model: MiniGPT) -> list[dict]:
    """Return per-layer, per-head trace norms."""
    result = []
    for layer_idx, attn in enumerate(model.get_attention_layers()):
        s_norms = attn.traces.norm(dim=(1, 2)).tolist()
        v_norms = attn.value_traces.norm(dim=(1, 2)).tolist()
        result.append({
            'layer': layer_idx,
            'score_norms': s_norms,
            'value_norms': v_norms,
        })
    return result


def evaluate_decay_config(model: MiniGPT, decay_rates: list[float],
                          n_facts: int, n_episodes: int, seed: int,
                          alpha: float, trace_lr: float) -> dict:
    """Evaluate one decay configuration at given n_facts.

    Phase 1: fixed alpha, no adaptive — isolates per-head decay effect.
    """
    for attn in model.get_attention_layers():
        attn.trace_lr = trace_lr
        attn.alpha = alpha

    model.set_per_head_decay(decay_rates)
    model.set_adaptive_alpha(False, norm_target=1.0)

    episodes = make_nlp_eval_episodes(
        n_episodes=n_episodes, n_facts=n_facts,
        seed=seed + n_facts * 1000, tier=1)

    bl = evaluate_baseline(model, episodes)
    hb = evaluate_hebbian(model, episodes)
    cc = evaluate_cross_context(model, episodes)
    norms = measure_trace_norms(model)
    per_head = measure_per_head_norms(model)
    cc_bl = evaluate_cross_context_baseline(model, episodes)

    return {
        'n_facts': n_facts,
        'baseline': bl.accuracy,
        'hebbian': hb.accuracy,
        'cross_ctx': cc.accuracy,
        'cross_bl': cc_bl.accuracy,
        'score_norm': norms['score_mean'],
        'value_norm': norms['value_mean'],
        'per_head': per_head,
    }


def evaluate_decay_with_adaptive(model: MiniGPT, decay_rates: list[float],
                                 n_facts: int, n_episodes: int, seed: int,
                                 alpha: float, trace_lr: float,
                                 norm_target: float) -> dict:
    """Phase 2: per-head decay + adaptive alpha (score-only)."""
    for attn in model.get_attention_layers():
        attn.trace_lr = trace_lr
        attn.alpha = alpha

    model.set_per_head_decay(decay_rates)
    model.set_adaptive_alpha(True, norm_target, score_only=True)

    episodes = make_nlp_eval_episodes(
        n_episodes=n_episodes, n_facts=n_facts,
        seed=seed + n_facts * 1000, tier=1)

    bl = evaluate_baseline(model, episodes)
    hb = evaluate_hebbian(model, episodes)
    cc = evaluate_cross_context(model, episodes)
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


def measure_per_head_contribution(model: MiniGPT, n_facts: int,
                                  n_episodes: int, seed: int,
                                  decay_rates: list[float],
                                  alpha: float, trace_lr: float) -> dict:
    """Ablation: zero out each head's traces, measure accuracy drop.

    Inlines the cross-context evaluation loop so we can ablate heads
    BETWEEN the training phase (trace accumulation) and test phase
    (query answering). evaluate_cross_context can't be used because
    it resets and re-trains traces on each call.
    """
    from ..nlp_tasks import NLP_VOCAB
    from ..nlp_evaluate import _predict_answer

    n_heads = model.n_heads
    device = next(model.parameters()).device
    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices
    attns = model.get_attention_layers()

    # Set up config
    for attn in attns:
        attn.trace_lr = trace_lr
        attn.alpha = alpha
    model.set_per_head_decay(decay_rates)
    model.set_adaptive_alpha(False, norm_target=1.0)

    episodes = make_nlp_eval_episodes(
        n_episodes=n_episodes, n_facts=n_facts,
        seed=seed + n_facts * 1000, tier=1)

    # Accumulators: full + per-head-ablated
    full_correct = 0
    ablated_correct = [0] * n_heads
    total_queries = 0

    for episode in episodes:
        # ── Training phase: accumulate traces ──
        model.reset_traces()
        model.set_trace_mode(use=True, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor)

        # Save trained traces
        saved_traces = []
        for attn in attns:
            saved_traces.append((
                attn.traces.clone(),
                attn.value_traces.clone(),
            ))

        # ── Test phase: full (no ablation) ──
        model.set_trace_mode(use=True, update=False)
        for query_indices, answer_idx, _, _ in episode.test_queries:
            pred = _predict_answer(model, query_indices, entity_indices)
            if pred == answer_idx:
                full_correct += 1
            total_queries += 1

        # ── Test phase: per-head ablation ──
        for h in range(n_heads):
            # Restore full traces, then zero head h
            for attn, (st, vt) in zip(attns, saved_traces):
                attn.traces.copy_(st)
                attn.value_traces.copy_(vt)
                attn.traces[h].zero_()
                attn.value_traces[h].zero_()

            for query_indices, answer_idx, _, _ in episode.test_queries:
                pred = _predict_answer(model, query_indices, entity_indices)
                if pred == answer_idx:
                    ablated_correct[h] += 1

    # Restore traces
    for attn, (st, vt) in zip(attns, saved_traces):
        attn.traces.copy_(st)
        attn.value_traces.copy_(vt)

    full_acc = full_correct / max(total_queries, 1)
    contributions = []
    for h in range(n_heads):
        abl_acc = ablated_correct[h] / max(total_queries, 1)
        contributions.append({
            'head': h,
            'decay': decay_rates[h],
            'ablated_acc': abl_acc,
            'drop': full_acc - abl_acc,
        })

    return {'full_acc': full_acc, 'contributions': contributions}


def run(
    n_pretrain: int = 20000,
    pretrain_epochs: int = 60,
    n_eval_episodes: int = 200,
    fact_counts: list[int] | None = None,
    alpha: float = 0.1,
    trace_lr: float = 0.1,
    seed: int = 42,
    device_name: str | None = None,
    load_path: str | None = None,
    save_path: str | None = None,
    configs: list[str] | None = None,
    verbose: bool = True,
):
    """Run multi-head decay specialization experiment.

    Phase 1: Pure per-head decay (fixed alpha)
    Phase 2: Best config + adaptive alpha (score-only)
    """
    if fact_counts is None:
        fact_counts = [1, 3, 5, 10]
    if configs is None:
        configs = list(DECAY_CONFIGS.keys())

    device = get_device(device_name)
    t_start = time.time()

    print("=" * 70)
    print("EXPERIMENT 4: Multi-Head Decay Specialization")
    print("=" * 70)

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

    # ── Step 2: Phase 1 — per-head decay, fixed alpha ──
    print(f"\nPhase 1: Per-head decay, fixed alpha={alpha}")
    print(f"  ({n_eval_episodes} episodes per condition)\n")

    # Print config table
    print("Decay configurations:")
    for name in configs:
        rates = DECAY_CONFIGS[name]
        rates_str = " ".join(f"{r:.3f}" for r in rates)
        print(f"  {name:<14s}: [{rates_str}]")
    print()

    # Evaluate all configs
    all_results = {}
    for name in configs:
        rates = DECAY_CONFIGS[name]
        print(f"{'─' * 70}")
        print(f"{name} (decay: {min(rates):.3f}–{max(rates):.3f})")
        print(f"{'─' * 70}")

        results = []
        for n_facts in fact_counts:
            r = evaluate_decay_config(
                model, rates, n_facts, n_eval_episodes, seed,
                alpha, trace_lr)
            results.append(r)
            print(f"  n={n_facts:>2d}: BL={r['baseline']:.1%}  "
                  f"Hebb={r['hebbian']:.1%}  "
                  f"Cross={r['cross_ctx']:.1%}  "
                  f"CrossBL={r['cross_bl']:.1%}  "
                  f"||T_s||={r['score_norm']:.3f}  ||T_v||={r['value_norm']:.3f}")
        all_results[name] = results

    # ── Phase 1 summary table ──
    elapsed_p1 = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"PHASE 1 SUMMARY — Per-head decay, fixed alpha (elapsed: {elapsed_p1:.0f}s)")
    print(f"{'=' * 70}\n")

    # Cross-context accuracy table
    print("CROSS-CONTEXT ACCURACY (trace only, question only)")
    header = f"{'n':>3} │"
    for name in configs:
        header += f" {name:>12s}"
    print(header)
    print("─" * (6 + 13 * len(configs)))
    for i, n_facts in enumerate(fact_counts):
        row = f"{n_facts:>3d} │"
        for name in configs:
            acc = all_results[name][i]['cross_ctx']
            row += f" {acc:>11.1%}"
        print(row)

    # Hebbian accuracy table
    print(f"\nHEBBIAN ACCURACY (in-context + trace)")
    header = f"{'n':>3} │"
    for name in configs:
        header += f" {name:>12s}"
    print(header)
    print("─" * (6 + 13 * len(configs)))
    for i, n_facts in enumerate(fact_counts):
        row = f"{n_facts:>3d} │"
        for name in configs:
            acc = all_results[name][i]['hebbian']
            row += f" {acc:>11.1%}"
        print(row)

    # ── Find best config for n=10 ──
    n10_idx = next((i for i, n in enumerate(fact_counts) if n == 10), None)
    best_config = None
    best_n10_cross = -1.0

    if n10_idx is not None:
        for name in configs:
            cc = all_results[name][n10_idx]['cross_ctx']
            if cc > best_n10_cross:
                best_n10_cross = cc
                best_config = name

        uni_cross = all_results['uniform_099'][n10_idx]['cross_ctx']
        print(f"\nn=10 best: {best_config} ({best_n10_cross:.1%}) "
              f"vs uniform_099 ({uni_cross:.1%})")

        if best_n10_cross > uni_cross:
            print(f"  Per-head decay IMPROVES n=10 by "
                  f"+{(best_n10_cross - uni_cross)*100:.1f}pp")
        else:
            print(f"  No improvement over uniform decay")

    # ── Per-head contribution ablation ──
    # Always ablate gradient (shows timescale spectrum) + best if different
    ablation_n = 10 if n10_idx is not None else fact_counts[-1]
    ablation_configs = ['gradient']
    if best_config and best_config != 'gradient':
        ablation_configs.append(best_config)

    for ablation_config in ablation_configs:
        ablation_rates = DECAY_CONFIGS[ablation_config]

        print(f"\n{'=' * 70}")
        print(f"PER-HEAD CONTRIBUTION ABLATION ({ablation_config}, n={ablation_n})")
        print(f"{'=' * 70}")

        contrib = measure_per_head_contribution(
            model, ablation_n, min(n_eval_episodes, 100), seed,
            ablation_rates, alpha, trace_lr)

        print(f"\nFull cross-context accuracy: {contrib['full_acc']:.1%}")
        print(f"{'Head':>6} {'Decay':>7} {'Ablated':>9} {'Drop':>7}")
        print("─" * 32)
        for c in contrib['contributions']:
            print(f"  {c['head']:>4d} {c['decay']:>7.3f} {c['ablated_acc']:>8.1%} "
                  f"{c['drop']:>+6.1%}")

        # Analyze: do high-decay heads contribute more?
        high_decay_drops = [c['drop'] for c in contrib['contributions']
                            if c['decay'] >= 0.95]
        low_decay_drops = [c['drop'] for c in contrib['contributions']
                           if c['decay'] <= 0.85]
        if high_decay_drops and low_decay_drops:
            avg_high = sum(high_decay_drops) / len(high_decay_drops)
            avg_low = sum(low_decay_drops) / len(low_decay_drops)
            print(f"\nAvg drop when ablating:")
            print(f"  High-decay heads (≥0.95): {avg_high:+.1%}")
            print(f"  Low-decay heads  (≤0.85): {avg_low:+.1%}")
            if avg_high > avg_low * 1.5:
                print("  → High-decay heads dominate retrieval (long-term memory)")
            elif avg_low > avg_high * 1.5:
                print("  → Low-decay heads dominate (recent facts more important)")
            else:
                print("  → Both contribute roughly equally (complementary)")

    # ── Phase 2: uniform_099 + adaptive vs gradient + adaptive ──
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Per-head decay + adaptive alpha (score-only)")
    print(f"{'=' * 70}")

    # Calibrate norm_target
    norm_1 = calibrate_norm_target(model, n_calibration_facts=1, seed=99)
    norm_target = norm_1 * 3
    print(f"  norm_target = {norm_target:.4f} (3× single-fact norm)\n")

    phase2_configs = ['uniform_099', 'gradient']
    phase2_all = {}
    for p2_name in phase2_configs:
        p2_rates = DECAY_CONFIGS[p2_name]
        print(f"  {p2_name} + adaptive score-only:")
        results = []
        for n_facts in fact_counts:
            r = evaluate_decay_with_adaptive(
                model, p2_rates, n_facts,
                n_eval_episodes, seed, alpha, trace_lr, norm_target)
            results.append(r)
            print(f"    n={n_facts:>2d}: BL={r['baseline']:.1%}  "
                  f"Hebb={r['hebbian']:.1%}  "
                  f"Cross={r['cross_ctx']:.1%}  "
                  f"CrossBL={r['cross_bl']:.1%}")
        phase2_all[p2_name] = results

    # Phase 2 comparison table
    print(f"\nPHASE 2 SUMMARY — adaptive alpha (score-only)")
    print(f"{'':>18} {'CROSS-CONTEXT':>28} {'│':>2} {'HEBBIAN':>28}")
    print(f"{'n':>3} │ {'Uni+ada':>10} {'Grad+ada':>10} {'Δ':>6} │ "
          f"{'Uni+ada':>10} {'Grad+ada':>10} {'Δ':>6}")
    print("─" * 65)
    for i, n_facts in enumerate(fact_counts):
        uc = phase2_all['uniform_099'][i]['cross_ctx']
        gc = phase2_all['gradient'][i]['cross_ctx']
        dc = gc - uc
        uh = phase2_all['uniform_099'][i]['hebbian']
        gh = phase2_all['gradient'][i]['hebbian']
        dh = gh - uh
        print(f"{n_facts:>3d} │ {uc:>9.1%} {gc:>9.1%} {dc:>+5.1%} │ "
              f"{uh:>9.1%} {gh:>9.1%} {dh:>+5.1%}")

    # Compare Phase 1 (no adaptive) vs Phase 2 (with adaptive)
    print(f"\nPHASE 1 vs PHASE 2 — uniform_099")
    print(f"{'n':>3} │ {'Cross P1':>10} {'Cross P2':>10} {'Δ':>6} │ "
          f"{'Hebb P1':>10} {'Hebb P2':>10} {'Δ':>6}")
    print("─" * 60)
    for i, n_facts in enumerate(fact_counts):
        p1c = all_results['uniform_099'][i]['cross_ctx']
        p2c = phase2_all['uniform_099'][i]['cross_ctx']
        p1h = all_results['uniform_099'][i]['hebbian']
        p2h = phase2_all['uniform_099'][i]['hebbian']
        print(f"{n_facts:>3d} │ {p1c:>9.1%} {p2c:>9.1%} {p2c-p1c:>+5.1%} │ "
              f"{p1h:>9.1%} {p2h:>9.1%} {p2h-p1h:>+5.1%}")

    # ── Final timing ──
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")


def run_quick(device_name: str | None = None, load_path: str | None = None):
    """Quick test: fewer episodes, subset of configs."""
    run(
        n_pretrain=5000,
        pretrain_epochs=30,
        n_eval_episodes=50,
        fact_counts=[1, 5, 10],
        device_name=device_name,
        load_path=load_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 4: Multi-Head Decay Specialization")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (~10 min)")
    parser.add_argument("--n-pretrain", type=int, default=20000)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--trace-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained model instead of training")
    parser.add_argument("--save", type=str, default=None,
                        help="Save model after pretraining")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific configs to test (default: all)")
    args = parser.parse_args()

    if args.quick:
        run_quick(device_name=args.device, load_path=args.load)
    else:
        run(
            n_pretrain=args.n_pretrain,
            pretrain_epochs=args.pretrain_epochs,
            n_eval_episodes=args.n_eval,
            alpha=args.alpha,
            trace_lr=args.trace_lr,
            seed=args.seed,
            device_name=args.device,
            load_path=args.load,
            save_path=args.save,
            configs=args.configs,
        )
