"""Experiment 5: Pattern Separation — Dentate Gyrus-Inspired Sparse Expansion.

Problem: At n>=5 facts, Q vectors for different concept words overlap
in the d_k=32 space, causing interference in the value trace.
Cross-context accuracy drops from 62% (n=1) to 44% (n=10).

Solution: Sparse random expansion before trace storage/retrieval:
    1. Frozen random projection: W_expand (d_k -> d_k * expand_factor)
    2. ReLU non-linearity
    3. Top-k sparsification (only k dimensions active)

This creates near-unique addressing codes for different concept words.
Biologically analogous to dentate gyrus pattern separation in hippocampus.

Key properties:
    - W_expand is frozen random (Johnson-Lindenstrauss preserves distances)
    - Deterministic: same word -> same sparse code (via fixed W_q(LN(tok)))
    - Only affects value trace (score trace unchanged)
    - No retraining needed: traces are not learned parameters

Configs tested:
    - baseline:  no expansion (standard d_k x d_k value trace)
    - 4x_k8:    d_k=32 -> 128, top-8   (6.3% sparsity)
    - 4x_k16:   d_k=32 -> 128, top-16  (12.5% sparsity)
    - 8x_k8:    d_k=32 -> 256, top-8   (3.1% sparsity)
    - 8x_k16:   d_k=32 -> 256, top-16  (6.3% sparsity)
    - 8x_k32:   d_k=32 -> 256, top-32  (12.5% sparsity)

Phase 2: best expansion config + adaptive alpha (score-only).
"""

import argparse
import time

import torch
import torch.nn.functional as F

from ..model import MiniGPT
from ..nlp_tasks import NLP_VOCAB, make_nlp_eval_episodes
from ..nlp_evaluate import (
    evaluate_baseline,
    evaluate_hebbian,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)
from .exp2_nlp_facts import pretrain_nlp, get_device, load_nlp_model


# ── Configurations ──

PATTERN_SEP_CONFIGS = {
    'baseline': dict(expand_factor=0, top_k=0),
    '4x_k8':   dict(expand_factor=4, top_k=8),
    '4x_k16':  dict(expand_factor=4, top_k=16),
    '8x_k8':   dict(expand_factor=8, top_k=8),
    '8x_k16':  dict(expand_factor=8, top_k=16),
    '8x_k32':  dict(expand_factor=8, top_k=32),
}

FACT_COUNTS = [1, 3, 5, 10]
N_EVAL = 200
ALPHA = 0.1
TRACE_LR = 0.1


# ── Diagnostics ──

def measure_code_overlap(model: MiniGPT, expand_factor: int, top_k: int,
                         seed: int = 0) -> dict:
    """Measure pairwise overlap of sparse codes for concept words.

    Returns cosine similarity and support overlap (IoU of active dims)
    between all pairs of concept words.
    """
    device = next(model.parameters()).device
    vocab = NLP_VOCAB
    first_attn = model.blocks[0].attn
    ln1 = model.blocks[0].ln1

    concept_words = ['name', 'city', 'company', 'color', 'food',
                     'pet', 'hobby', 'language', 'age', 'country']

    # Temporarily enable pattern separation on first layer
    first_attn.enable_pattern_separation(expand_factor, top_k, seed=seed)

    codes = {}
    with torch.no_grad():
        for word in concept_words:
            idx = vocab.tok2idx.get(word)
            if idx is None:
                continue
            tok_embed = model.token_embed(
                torch.tensor([[idx]], device=device))  # (1, 1, d_model)
            Q_raw = first_attn.W_q(ln1(tok_embed))  # (1, 1, d_model)
            d_k = model.d_model // model.n_heads
            Q_raw = Q_raw.view(1, 1, model.n_heads, d_k).transpose(1, 2)
            # (1, H, 1, d_k)
            sparse_code = first_attn._sparse_expand(Q_raw)
            # (1, H, 1, expanded_dim)
            codes[word] = sparse_code.squeeze(0).squeeze(-2)  # (H, expanded_dim)

    first_attn.disable_pattern_separation()

    words = list(codes.keys())
    n = len(words)

    # Pairwise cosine similarity (averaged across heads)
    similarities = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cos = F.cosine_similarity(codes[words[i]], codes[words[j]], dim=-1)
            similarities[i, j] = cos.mean()

    # Support overlap: IoU of non-zero positions
    support_overlaps = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            si = (codes[words[i]] > 0).float()
            sj = (codes[words[j]] > 0).float()
            intersection = (si * sj).sum(dim=-1)
            union = ((si + sj) > 0).float().sum(dim=-1)
            iou = intersection / union.clamp(min=1)
            support_overlaps[i, j] = iou.mean()

    offdiag = ~torch.eye(n, dtype=bool)
    return {
        'words': words,
        'cosine_similarity': similarities,
        'support_overlap': support_overlaps,
        'mean_cosine': similarities[offdiag].mean().item(),
        'max_cosine': similarities[offdiag].max().item(),
        'mean_support_overlap': support_overlaps[offdiag].mean().item(),
        'max_support_overlap': support_overlaps[offdiag].max().item(),
    }


def measure_raw_q_overlap(model: MiniGPT) -> dict:
    """Measure pairwise cosine similarity of raw Q vectors (no expansion).

    Baseline: shows how much concept words overlap in the original d_k space.
    """
    device = next(model.parameters()).device
    vocab = NLP_VOCAB
    first_attn = model.blocks[0].attn
    ln1 = model.blocks[0].ln1

    concept_words = ['name', 'city', 'company', 'color', 'food',
                     'pet', 'hobby', 'language', 'age', 'country']

    codes = {}
    with torch.no_grad():
        for word in concept_words:
            idx = vocab.tok2idx.get(word)
            if idx is None:
                continue
            tok_embed = model.token_embed(
                torch.tensor([[idx]], device=device))
            Q_raw = first_attn.W_q(ln1(tok_embed))
            d_k = model.d_model // model.n_heads
            Q_raw = Q_raw.view(1, 1, model.n_heads, d_k).transpose(1, 2)
            codes[word] = Q_raw.squeeze(0).squeeze(-2)  # (H, d_k)

    words = list(codes.keys())
    n = len(words)

    similarities = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cos = F.cosine_similarity(codes[words[i]], codes[words[j]], dim=-1)
            similarities[i, j] = cos.mean()

    offdiag = ~torch.eye(n, dtype=bool)
    return {
        'words': words,
        'cosine_similarity': similarities,
        'mean_cosine': similarities[offdiag].mean().item(),
        'max_cosine': similarities[offdiag].max().item(),
    }


# ── Evaluation ──

def evaluate_config(model: MiniGPT, config_name: str, config: dict,
                    n_facts: int, n_episodes: int, seed: int,
                    alpha: float = ALPHA, trace_lr: float = TRACE_LR,
                    ) -> dict:
    """Evaluate one pattern separation config at one n_facts.

    Enables/disables pattern separation, sets trace params, runs 4-mode eval.
    """
    # Set trace parameters
    for attn in model.get_attention_layers():
        attn.trace_decay = 0.99
        attn.trace_lr = trace_lr
        attn.alpha = alpha

    model.set_adaptive_alpha(False, 1.0)  # fixed alpha for Phase 1

    # Enable/disable pattern separation
    if config['expand_factor'] > 0:
        model.enable_pattern_separation(
            config['expand_factor'], config['top_k'], seed=0)
    else:
        model.disable_pattern_separation()

    episodes = make_nlp_eval_episodes(
        n_episodes=n_episodes, n_facts=n_facts,
        seed=seed + n_facts * 1000, tier=1)

    bl = evaluate_baseline(model, episodes)
    hb = evaluate_hebbian(model, episodes)
    cc = evaluate_cross_context(model, episodes)
    cc_bl = evaluate_cross_context_baseline(model, episodes)

    # Clean up
    model.disable_pattern_separation()

    return {
        'config': config_name,
        'n_facts': n_facts,
        'baseline': bl.accuracy,
        'hebbian': hb.accuracy,
        'cross_ctx': cc.accuracy,
        'cross_bl': cc_bl.accuracy,
        'trace_benefit': cc.accuracy - cc_bl.accuracy,
    }


def evaluate_with_adaptive(model: MiniGPT, config_name: str, config: dict,
                           n_facts: int, n_episodes: int, seed: int,
                           norm_target: float,
                           alpha: float = ALPHA, trace_lr: float = TRACE_LR,
                           ) -> dict:
    """Phase 2: pattern separation + adaptive alpha (score-only)."""
    for attn in model.get_attention_layers():
        attn.trace_decay = 0.99
        attn.trace_lr = trace_lr
        attn.alpha = alpha

    model.set_adaptive_alpha(True, norm_target, score_only=True)

    if config['expand_factor'] > 0:
        model.enable_pattern_separation(
            config['expand_factor'], config['top_k'], seed=0)
    else:
        model.disable_pattern_separation()

    episodes = make_nlp_eval_episodes(
        n_episodes=n_episodes, n_facts=n_facts,
        seed=seed + n_facts * 1000, tier=1)

    hb = evaluate_hebbian(model, episodes)
    cc = evaluate_cross_context(model, episodes)
    cc_bl = evaluate_cross_context_baseline(model, episodes)

    model.disable_pattern_separation()
    model.set_adaptive_alpha(False, 1.0)

    return {
        'config': config_name,
        'n_facts': n_facts,
        'hebbian': hb.accuracy,
        'cross_ctx': cc.accuracy,
        'cross_bl': cc_bl.accuracy,
        'trace_benefit': cc.accuracy - cc_bl.accuracy,
    }


def calibrate_norm_target(model: MiniGPT, seed: int = 99) -> float:
    """Measure trace norm after 1 fact for adaptive alpha calibration."""
    device = next(model.parameters()).device
    vocab = NLP_VOCAB

    episodes = make_nlp_eval_episodes(
        n_episodes=1, n_facts=1, seed=seed, tier=1)
    ep = episodes[0]

    model.reset_traces()
    model.set_trace_mode(use=False, update=True)  # ACh high: write only

    for train_seq in ep.train_sequences:
        input_tensor = torch.tensor([train_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_tensor)

    # Measure score trace norm (used for adaptive alpha)
    score_norms = []
    for attn in model.get_attention_layers():
        s_norms = attn.traces.norm(dim=(1, 2))
        score_norms.append(s_norms)
    mean_norm = torch.stack(score_norms).mean().item()

    model.reset_traces()
    model.set_trace_mode(use=False, update=False)
    return mean_norm


# ── Main ──

def run(
    load_path: str | None = None,
    n_eval: int = N_EVAL,
    fact_counts: list[int] | None = None,
    configs: dict | None = None,
    seed: int = 42,
    device_name: str | None = None,
    verbose: bool = True,
):
    """Run pattern separation experiment.

    Phase 1: Compare all expansion configs with fixed alpha.
    Phase 2: Best expansion + adaptive alpha (score-only).
    """
    if fact_counts is None:
        fact_counts = FACT_COUNTS
    if configs is None:
        configs = PATTERN_SEP_CONFIGS

    device = get_device(device_name)
    t_start = time.time()

    print("=" * 65)
    print("EXPERIMENT 5: Pattern Separation (Dentate Gyrus)")
    print("=" * 65)

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

    d_k = model.d_model // model.n_heads
    print(f"  d_model={model.d_model}, n_heads={model.n_heads}, d_k={d_k}")

    # ── Step 2: Code overlap diagnostics ──
    print(f"\n{'─' * 65}")
    print("CODE OVERLAP DIAGNOSTICS")
    print(f"{'─' * 65}")

    # Raw Q overlap
    raw_overlap = measure_raw_q_overlap(model)
    print(f"\nRaw Q vectors (d_k={d_k}):")
    print(f"  Mean cosine similarity: {raw_overlap['mean_cosine']:.3f}")
    print(f"  Max cosine similarity:  {raw_overlap['max_cosine']:.3f}")

    # Overlap with each expansion config
    print(f"\n{'Config':<12} {'Dim':>5} {'k':>3} │ "
          f"{'Mean cos':>9} {'Max cos':>9} │ "
          f"{'Mean IoU':>9} {'Max IoU':>9}")
    print("─" * 70)

    for name, cfg in configs.items():
        if cfg['expand_factor'] == 0:
            continue
        expanded_dim = d_k * cfg['expand_factor']
        overlap = measure_code_overlap(
            model, cfg['expand_factor'], cfg['top_k'])
        print(f"{name:<12} {expanded_dim:>5} {cfg['top_k']:>3} │ "
              f"{overlap['mean_cosine']:>9.3f} {overlap['max_cosine']:>9.3f} │ "
              f"{overlap['mean_support_overlap']:>9.3f} "
              f"{overlap['max_support_overlap']:>9.3f}")

    # ── Step 3: Phase 1 — Fixed alpha, compare configs ──
    print(f"\n{'=' * 65}")
    print(f"PHASE 1: Pattern Separation Configs (fixed alpha={ALPHA})")
    print(f"{'=' * 65}")

    all_results = {}
    for name, cfg in configs.items():
        exp_dim = d_k * cfg['expand_factor'] if cfg['expand_factor'] > 0 else d_k
        sparsity = (cfg['top_k'] / exp_dim * 100) if cfg['expand_factor'] > 0 else 100
        print(f"\n{'─' * 65}")
        if cfg['expand_factor'] > 0:
            print(f"{name}: d_k={d_k} -> {exp_dim}, "
                  f"top-{cfg['top_k']} ({sparsity:.1f}% active)")
        else:
            print(f"{name}: standard d_k={d_k} (no expansion)")
        print(f"{'─' * 65}")

        results = []
        for n_facts in fact_counts:
            r = evaluate_config(model, name, cfg, n_facts, n_eval, seed)
            results.append(r)
            print(f"  n={n_facts:>2d}: BL={r['baseline']:.1%}  "
                  f"Hebb={r['hebbian']:.1%}  "
                  f"Cross={r['cross_ctx']:.1%}  "
                  f"CrossBL={r['cross_bl']:.1%}  "
                  f"Gap={r['trace_benefit']:+.1%}")
        all_results[name] = results

    # ── Phase 1 Summary ──
    print(f"\n{'=' * 65}")
    elapsed = time.time() - t_start
    print(f"PHASE 1 SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 65}\n")

    config_names = list(configs.keys())

    print("CROSS-CONTEXT ACCURACY")
    header = f"{'n':>3} │"
    for name in config_names:
        header += f" {name:>10}"
    print(header)
    print("─" * (6 + 11 * len(config_names)))
    for i, n in enumerate(fact_counts):
        row = f"{n:>3d} │"
        for name in config_names:
            acc = all_results[name][i]['cross_ctx']
            row += f" {acc:>9.1%}"
        print(row)

    print(f"\nTRACE BENEFIT (cross_ctx - cross_bl)")
    header = f"{'n':>3} │"
    for name in config_names:
        header += f" {name:>10}"
    print(header)
    print("─" * (6 + 11 * len(config_names)))
    for i, n in enumerate(fact_counts):
        row = f"{n:>3d} │"
        for name in config_names:
            gap = all_results[name][i]['trace_benefit']
            row += f" {gap:>+9.1%}"
        print(row)

    print(f"\nHEBBIAN ACCURACY (in-context + trace)")
    header = f"{'n':>3} │"
    for name in config_names:
        header += f" {name:>10}"
    print(header)
    print("─" * (6 + 11 * len(config_names)))
    for i, n in enumerate(fact_counts):
        row = f"{n:>3d} │"
        for name in config_names:
            acc = all_results[name][i]['hebbian']
            row += f" {acc:>9.1%}"
        print(row)

    # ── Find best config at n=10 ──
    n10_idx = next((i for i, n in enumerate(fact_counts) if n == 10), None)
    if n10_idx is not None:
        best_name = max(
            config_names,
            key=lambda name: all_results[name][n10_idx]['cross_ctx'])
        best_cross = all_results[best_name][n10_idx]['cross_ctx']
        base_cross = all_results['baseline'][n10_idx]['cross_ctx']
        print(f"\nBest at n=10: {best_name} ({best_cross:.1%} "
              f"vs baseline {base_cross:.1%}, "
              f"delta={best_cross - base_cross:+.1%})")
    else:
        best_name = 'baseline'

    # ── Phase 2: Best config + adaptive alpha ──
    print(f"\n{'=' * 65}")
    print(f"PHASE 2: {best_name} + Adaptive Alpha (score-only)")
    print(f"{'=' * 65}")

    norm_1 = calibrate_norm_target(model, seed=99)
    norm_target = norm_1 * 3
    print(f"  1-fact norm: {norm_1:.4f}, norm_target: {norm_target:.4f}")

    best_cfg = configs[best_name]

    # Conditions: baseline, adaptive-only, expansion-only, both
    phase2_conditions = {
        'fixed_baseline': ('baseline', configs['baseline'], False),
        'adaptive_only': ('baseline', configs['baseline'], True),
        f'{best_name}_fixed': (best_name, best_cfg, False),
        f'{best_name}_adaptive': (best_name, best_cfg, True),
    }

    phase2_results = {}
    for cond_name, (cfg_name, cfg, use_adaptive) in phase2_conditions.items():
        print(f"\n{'─' * 65}")
        print(f"{cond_name}")
        print(f"{'─' * 65}")

        results = []
        for n_facts in fact_counts:
            if use_adaptive:
                r = evaluate_with_adaptive(
                    model, cfg_name, cfg, n_facts, n_eval, seed,
                    norm_target=norm_target)
            else:
                r = evaluate_config(model, cfg_name, cfg, n_facts, n_eval, seed)
            results.append(r)
            hb_str = f"Hebb={r['hebbian']:.1%}  " if 'hebbian' in r else ""
            print(f"  n={n_facts:>2d}: {hb_str}"
                  f"Cross={r['cross_ctx']:.1%}  "
                  f"Gap={r['trace_benefit']:+.1%}")
        phase2_results[cond_name] = results

    # Phase 2 summary
    print(f"\n{'=' * 65}")
    elapsed = time.time() - t_start
    print(f"PHASE 2 SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 65}\n")

    p2_names = list(phase2_conditions.keys())

    print("CROSS-CONTEXT ACCURACY")
    header = f"{'n':>3} │"
    for name in p2_names:
        header += f" {name:>18}"
    print(header)
    print("─" * (6 + 19 * len(p2_names)))
    for i, n in enumerate(fact_counts):
        row = f"{n:>3d} │"
        for name in p2_names:
            acc = phase2_results[name][i]['cross_ctx']
            row += f" {acc:>17.1%}"
        print(row)

    print(f"\nHEBBIAN ACCURACY")
    header = f"{'n':>3} │"
    for name in p2_names:
        header += f" {name:>18}"
    print(header)
    print("─" * (6 + 19 * len(p2_names)))
    for i, n in enumerate(fact_counts):
        row = f"{n:>3d} │"
        for name in p2_names:
            acc = phase2_results[name][i].get('hebbian', 0)
            row += f" {acc:>17.1%}"
        print(row)

    # Final verdict
    if n10_idx is not None:
        print(f"\n{'─' * 40}")
        print("n=10 VERDICT:")
        for name in p2_names:
            cc = phase2_results[name][n10_idx]['cross_ctx']
            gap = phase2_results[name][n10_idx]['trace_benefit']
            print(f"  {name:<20s}: cross={cc:.1%}  gap={gap:+.1%}")

        base_cc = phase2_results['fixed_baseline'][n10_idx]['cross_ctx']
        best_p2 = max(p2_names, key=lambda n: phase2_results[n][n10_idx]['cross_ctx'])
        best_cc = phase2_results[best_p2][n10_idx]['cross_ctx']
        delta = best_cc - base_cc
        if delta > 0.02:
            print(f"\n  PATTERN SEPARATION WORKS: {best_p2} "
                  f"+{delta:.1%} at n=10")
        elif delta > -0.02:
            print(f"\n  NEUTRAL: no significant improvement at n=10")
        else:
            print(f"\n  REGRESSION: {best_p2} {delta:+.1%} at n=10")


def run_quick(device_name: str | None = None, load_path: str | None = None):
    """Quick test: fewer episodes, subset of configs."""
    quick_configs = {
        'baseline': dict(expand_factor=0, top_k=0),
        '4x_k16':  dict(expand_factor=4, top_k=16),
        '8x_k16':  dict(expand_factor=8, top_k=16),
    }
    run(
        load_path=load_path,
        n_eval=50,
        fact_counts=[1, 5, 10],
        configs=quick_configs,
        device_name=device_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 5: Pattern Separation (Dentate Gyrus)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (~10 min)")
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
