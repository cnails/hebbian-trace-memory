#!/usr/bin/env python3
"""Experiment 19: Phi-2 (2.7B) cross-architecture transfer test.

Does the Hebbian trace mechanism generalize from GPT-2 (124M, d=768)
to Microsoft Phi-2 (2.7B, d=2560)? This tests architecture-agnostic
transfer: different model family, different tokenizer, different scale.

Key differences from GPT-2:
  - Phi-2 uses parallel attention + MLP (not sequential)
  - Rotary position embeddings (not learned absolute)
  - CodeGen tokenizer (different vocab, different BPE)
  - d_model = 2560 (vs 768 for GPT-2 Small)

Zero-shot transfer: no gate weights loaded (trained gates are
d_model=768, incompatible with d_model=2560). Uses hardcoded
linking-token mask only — the same mechanism that works before
gate training on GPT-2.

Phase 1: Zero-shot proof of concept (alpha=1.0, no PS)
Phase 2: Alpha sweep — logit scale may differ at 2.7B
Phase 3: Pattern separation at best alpha
Phase 4: Head-to-head — GPT-2 Small vs Phi-2 (optional)

Usage:
    python -m hebbian_trace.experiments.exp19_phi2_transfer --quick
    python -m hebbian_trace.experiments.exp19_phi2_transfer --n-eval 50
    python -m hebbian_trace.experiments.exp19_phi2_transfer --n-eval 100
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    evaluate_gpt2_baseline, evaluate_gpt2_cross_context,
    evaluate_gpt2_cross_context_baseline,
    tokenize_fact, GPT2FactType,
)
from .exp22_pattern_completion import extract_auto_pairs, AutoPair, CONCEPT_WORDS
from .exp12_realistic_benchmarks import build_question_variants, QuestionVariant


PHI2_MODEL = "microsoft/phi-2"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_phi2(device: str, alpha: float = 1.0,
              n_trace_heads: int = 8, d_trace: int = 64) -> GPT2WithTrace:
    """Load Phi-2 + trace model (zero-shot, no trained gates)."""
    print(f"\nLoading {PHI2_MODEL} (this may take a minute)...")
    t0 = time.time()

    # Use float16 for memory efficiency (Phi-2 is 2.7B ≈ 5.4GB in fp16)
    model = GPT2WithTrace(
        n_trace_heads=n_trace_heads, d_trace=d_trace,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
        model_name=PHI2_MODEL,
        torch_dtype=torch.float16,
        device=device,
    )

    dt = time.time() - t0
    config = model.base_model.config
    d_model = config.hidden_size
    n_layers = getattr(config, 'n_layer',
                       getattr(config, 'num_hidden_layers', -1))

    print(f"  Loaded in {dt:.1f}s")
    print(f"  Model:       {PHI2_MODEL}")
    print(f"  d_model:     {d_model}")
    print(f"  n_layers:    {n_layers}")
    print(f"  inject_layer:{model.inject_layer} (mid-depth)")
    print(f"  Base params: {sum(p.numel() for p in model.base_model.parameters()):,}")
    print(f"  Trace params:{sum(p.numel() for p in model.trace.parameters()):,}")
    print(f"  Trace d_model: {model.trace.d_model}")
    print(f"  W_proj shape:  {tuple(model.trace.W_proj.weight.shape)}")
    print(f"  W_val shape:   {tuple(model.trace.W_val.weight.shape)}")
    print(f"  W_out shape:   {tuple(model.trace.W_out.weight.shape)}")
    print(f"  Gate mode: hardcoded linking mask (no trained gates)")

    # Ensure no learned gates — use hardcoded linking mask only
    model.set_gate_mode(use_learned_gate=False)
    model.set_dual_gate_mode(enabled=False)

    return model


def load_gpt2_small(device: str, alpha: float = 1.0) -> GPT2WithTrace:
    """Load GPT-2 Small for comparison (linking mask only, fair comparison)."""
    print(f"\nLoading gpt2 (Small, for comparison)...")
    t0 = time.time()
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
        model_name='gpt2', device=device,
    )
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")
    print(f"  Using linking mask (no trained gates, fair comparison with Phi-2)")

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


# ── Phase 1: Zero-Shot Proof of Concept ──────────────────────────────

def run_phase1(model, tokenizer, fact_types, n_eval, n_facts_list):
    """Zero-shot cross-context eval on Phi-2."""
    print(f"\n{'─' * 70}")
    print("PHASE 1: Zero-Shot Transfer (Phi-2, alpha=1.0, no PS, no trained gates)")
    print(f"{'─' * 70}")

    results = run_eval_suite(
        model, tokenizer, fact_types, n_eval, n_facts_list)
    print_results_table(results)
    return results


# ── Phase 2: Alpha Sweep ─────────────────────────────────────────────

def run_phase2(model, tokenizer, fact_types, n_eval):
    """Alpha sweep — Phi-2 logit scale may differ from GPT-2."""
    print(f"\n{'─' * 70}")
    print("PHASE 2: Alpha Sweep (Phi-2)")
    print(f"{'─' * 70}")

    alphas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
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
    """Pattern separation on Phi-2."""
    print(f"\n{'─' * 70}")
    print("PHASE 3: Pattern Separation (8x_k16 on Phi-2)")
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


# ── Phase 4: Head-to-Head ────────────────────────────────────────────

def run_phase4(model_phi2, model_small, tokenizer_phi2, tokenizer_small,
               fact_types_phi2, fact_types_small,
               n_eval, n_facts_list):
    """Compare GPT-2 Small (with trained gates) vs Phi-2 (zero-shot)."""
    print(f"\n{'─' * 70}")
    print("PHASE 4: Head-to-Head — GPT-2 Small (trained) vs Phi-2 (zero-shot)")
    print(f"{'─' * 70}")

    # Both with PS
    model_small.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model_phi2.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    print(f"\n  Alpha: Small={model_small.trace.alpha}, "
          f"Phi-2={model_phi2.trace.alpha}")

    r_small = run_eval_suite(
        model_small, tokenizer_small, fact_types_small,
        n_eval, n_facts_list, label="GPT-2 Small (124M, trained gates)")
    print_results_table(r_small, "GPT-2 Small (124M, trained gates)")

    r_phi2 = run_eval_suite(
        model_phi2, tokenizer_phi2, fact_types_phi2,
        n_eval, n_facts_list, label="Phi-2 (2.7B, zero-shot)")
    print_results_table(r_phi2, "Phi-2 (2.7B, zero-shot)")

    # Comparison table
    print(f"\n  {'n':>3}  {'Small Cross':>13}  {'Phi-2 Cross':>13}  "
          f"{'Delta':>8}  {'Small BL':>10}  {'Phi-2 BL':>10}")
    print(f"  {'─'*3}  {'─'*13}  {'─'*13}  {'─'*8}  {'─'*10}  {'─'*10}")

    for n in sorted(r_small.keys()):
        if n not in r_phi2:
            continue
        sc = r_small[n]['cross_ctx']
        pc = r_phi2[n]['cross_ctx']
        sb = r_small[n]['baseline']
        pb = r_phi2[n]['baseline']
        delta = pc - sc
        print(f"  {n:3d}  {sc:>13.1%}  {pc:>13.1%}  "
              f"{delta:>+7.1%}  {sb:>10.1%}  {pb:>10.1%}")

    return r_small, r_phi2


# ── Phase 5: T_auto Pattern Completion ───────────────────────────────

def _get_all_entity_ids(fact_types: list[GPT2FactType]) -> list[int]:
    """Get all unique entity BPE IDs across all fact types."""
    ids = set()
    for ft in fact_types:
        for _, eid in ft.entities:
            ids.add(eid)
    return sorted(ids)


def run_variant_eval_phi2(
    model: GPT2WithTrace,
    tokenizer,
    fact_types: list[GPT2FactType],
    n_facts_list: list[int],
    n_eval: int,
    auto_pairs: list[AutoPair] | None,
    completion_alpha: float = 0.3,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Evaluate T_auto pattern completion with question variants.

    For each n_facts:
    - Write T_auto pairs once (persistent template knowledge)
    - Per episode: write facts, query with aligned/misaligned/semantic variants
    - Return per-category accuracy at each n

    Returns:
        {n_facts: {"aligned": acc, "misaligned": acc, "semantic": acc, "overall": acc}}
    """
    model.eval()
    device = next(model.parameters()).device
    entity_ids = _get_all_entity_ids(fact_types)

    # Build question variants for this tokenizer
    variants = build_question_variants(tokenizer)

    # Prepare T_auto pairs
    if auto_pairs is not None:
        pair_tuples = [(p.variant_id, p.concept_id) for p in auto_pairs]

    results: dict[int, dict[str, float]] = {}

    for n_facts in n_facts_list:
        rng = random.Random(seed + n_facts * 1000)

        # Write T_auto pairs once for this n_facts block
        model.trace.autoassociative_traces.zero_()
        if auto_pairs is not None:
            model.write_auto_pairs(pair_tuples)
            model.set_auto_mode(True, completion_alpha)
        else:
            model.set_auto_mode(False)

        # Track per-category results: category → [correct, total]
        cat_results: dict[str, list[int]] = {
            "aligned": [0, 0], "misaligned": [0, 0], "semantic": [0, 0],
        }

        for ep_idx in range(n_eval):
            # Zero T_v only (T_auto persists)
            model.trace.value_traces.zero_()

            # Select fact types + entities
            if n_facts <= len(fact_types):
                selected = rng.sample(fact_types, n_facts)
            else:
                selected = [rng.choice(fact_types) for _ in range(n_facts)]

            episode_facts: list[tuple[GPT2FactType, str, int]] = []
            for ft in selected:
                entity_name, entity_id = rng.choice(ft.entities)
                episode_facts.append((ft, entity_name, entity_id))

            # Write phase: encode facts individually
            model.set_trace_mode(use=False, update=True)
            for ft, entity_name, entity_id in episode_facts:
                template = ft.fact_templates[0]
                fact_ids = tokenize_fact(tokenizer, template, entity_name)
                input_tensor = torch.tensor(
                    [fact_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    _ = model(input_tensor)

            # Read phase: query with all variants
            model.set_trace_mode(use=True, update=False)
            for ft, entity_name, entity_id in episode_facts:
                if ft.name not in variants:
                    continue

                for v in variants[ft.name]:
                    input_tensor = torch.tensor(
                        [v.bpe_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = model(input_tensor)

                    pred_logits = logits[0, -1, :]
                    entity_logits = pred_logits[entity_ids]
                    best_idx = entity_logits.argmax().item()
                    pred_id = entity_ids[best_idx]

                    correct = int(pred_id == entity_id)
                    cat = v.category
                    if cat in cat_results:
                        cat_results[cat][0] += correct
                        cat_results[cat][1] += 1

        # Compute accuracies
        r: dict[str, float] = {}
        total_c, total_t = 0, 0
        for cat, (c, t) in cat_results.items():
            r[cat] = c / max(t, 1)
            total_c += c
            total_t += t
        r["overall"] = total_c / max(total_t, 1)
        results[n_facts] = r

    return results


def run_phase5(model, tokenizer, fact_types, n_eval, n_facts_list,
               completion_alpha: float = 0.3):
    """Phase 5: T_auto Pattern Completion on Phi-2."""
    print(f"\n{'─' * 70}")
    print("PHASE 5: T_auto Pattern Completion (Phi-2)")
    print(f"{'─' * 70}")

    # 5a: Extract T_auto pairs
    print("\n  Extracting T_auto pairs...")
    auto_pairs = extract_auto_pairs(tokenizer)
    print(f"  T_auto: {len(auto_pairs)} pairs for Phi-2 tokenizer")

    # Count per category
    cats = {}
    for p in auto_pairs:
        cats[p.category] = cats.get(p.category, 0) + 1
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat}: {cnt} pairs")

    # 5b: Completion alpha sweep at n=5
    print(f"\n  Completion alpha sweep (n=5, {n_eval} episodes):")
    c_alphas = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]

    print(f"  {'c_alpha':>8}  {'aligned':>9}  {'misaligned':>11}  "
          f"{'semantic':>9}  {'overall':>9}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*9}  {'─'*9}")

    best_ca = completion_alpha
    best_overall = -1.0

    for ca in c_alphas:
        r = run_variant_eval_phi2(
            model, tokenizer, fact_types, [5], n_eval,
            auto_pairs=auto_pairs, completion_alpha=ca)
        r5 = r[5]
        marker = ""
        if r5["overall"] > best_overall:
            best_overall = r5["overall"]
            best_ca = ca
            marker = " <-- best"
        print(f"  {ca:8.1f}  {r5['aligned']:>9.1%}  {r5['misaligned']:>11.1%}  "
              f"{r5['semantic']:>9.1%}  {r5['overall']:>9.1%}{marker}")

    # Extend sweep if best is at boundary
    if best_ca >= 5.0:
        print(f"\n  Best at boundary ({best_ca}), extending sweep...")
        ext_alphas = [10.0, 20.0]
        for ca in ext_alphas:
            r = run_variant_eval_phi2(
                model, tokenizer, fact_types, [5], n_eval,
                auto_pairs=auto_pairs, completion_alpha=ca)
            r5 = r[5]
            marker = ""
            if r5["overall"] > best_overall:
                best_overall = r5["overall"]
                best_ca = ca
                marker = " <-- best"
            print(f"  {ca:8.1f}  {r5['aligned']:>9.1%}  {r5['misaligned']:>11.1%}  "
                  f"{r5['semantic']:>9.1%}  {r5['overall']:>9.1%}{marker}")

    print(f"\n  Best completion_alpha: {best_ca} (overall={best_overall:.1%})")

    # 5c: Full eval — without T_auto (baseline)
    print(f"\n  Baseline (no T_auto):")
    r_no_tauto = run_variant_eval_phi2(
        model, tokenizer, fact_types, n_facts_list, n_eval,
        auto_pairs=None)

    # Full eval — with T_auto at best c_alpha
    print(f"  With T_auto (c_alpha={best_ca}):")
    r_tauto = run_variant_eval_phi2(
        model, tokenizer, fact_types, n_facts_list, n_eval,
        auto_pairs=auto_pairs, completion_alpha=best_ca)

    # 5d: Print comparison
    print(f"\n  T_auto Pattern Completion Results (Phi-2, c_alpha={best_ca})")
    print(f"  {'n':>3}  {'Std':>6}  {'aligned':>9}  {'misaligned':>11}  "
          f"{'semantic':>9}  {'overall':>9}")
    print(f"  {'─'*3}  {'─'*6}  {'─'*9}  {'─'*11}  {'─'*9}  {'─'*9}")

    for n in sorted(r_tauto.keys()):
        std = r_no_tauto[n]["aligned"]  # aligned without T_auto = standard recall
        rt = r_tauto[n]
        print(f"  {n:3d}  {std:>6.1%}  {rt['aligned']:>9.1%}  "
              f"{rt['misaligned']:>11.1%}  {rt['semantic']:>9.1%}  "
              f"{rt['overall']:>9.1%}")

    return {"best_ca": best_ca, "no_tauto": r_no_tauto, "tauto": r_tauto}


def run_tauto_comparison(n_eval: int = 30, n_facts_list: list[int] | None = None,
                         completion_alpha: float = 0.3, seed: int = 42):
    """Compare T_auto performance: Phi-2 (2.7B) vs GPT-2 Small (124M)."""
    device = get_device()
    if n_facts_list is None:
        n_facts_list = [1, 3, 5, 7]

    print("=" * 70)
    print("  T_AUTO COMPARISON: Phi-2 (2.7B) vs GPT-2 Small (124M)")
    print("=" * 70)
    print(f"  Episodes:   {n_eval}")
    print(f"  n_facts:    {n_facts_list}")
    print()

    # ── Load Phi-2 ──
    tokenizer_phi2 = AutoTokenizer.from_pretrained(
        PHI2_MODEL, trust_remote_code=True)
    fact_types_phi2 = build_fact_types(tokenizer_phi2)
    linking_ids_phi2 = get_linking_bpe_ids(tokenizer_phi2)

    model_phi2 = load_phi2(device, alpha=50.0)
    model_phi2.set_linking_token_ids(linking_ids_phi2)
    model_phi2.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    # ── Load GPT-2 Small (no trained gates — fair comparison) ──
    model_small = load_gpt2_small(device, alpha=0.5)
    from transformers import GPT2Tokenizer
    tokenizer_small = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types_small = build_fact_types(tokenizer_small)
    linking_ids_small = get_linking_bpe_ids(tokenizer_small)
    model_small.set_linking_token_ids(linking_ids_small)
    model_small.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    # ── Extract T_auto pairs ──
    print("\n  Extracting T_auto pairs...")
    auto_pairs_phi2 = extract_auto_pairs(tokenizer_phi2)
    auto_pairs_small = extract_auto_pairs(tokenizer_small)
    print(f"  Phi-2: {len(auto_pairs_phi2)} pairs")
    print(f"  GPT-2 Small: {len(auto_pairs_small)} pairs")

    # ── Completion alpha sweep on Phi-2 ──
    print(f"\n  Phi-2 completion alpha sweep (n=5, {n_eval} episodes):")
    c_alphas = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]
    best_ca_phi2 = completion_alpha
    best_overall = -1.0

    print(f"  {'c_alpha':>8}  {'aligned':>9}  {'misaligned':>11}  "
          f"{'semantic':>9}  {'overall':>9}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*9}  {'─'*9}")

    for ca in c_alphas:
        r = run_variant_eval_phi2(
            model_phi2, tokenizer_phi2, fact_types_phi2, [5], n_eval,
            auto_pairs=auto_pairs_phi2, completion_alpha=ca)
        r5 = r[5]
        marker = ""
        if r5["overall"] > best_overall:
            best_overall = r5["overall"]
            best_ca_phi2 = ca
            marker = " <-- best"
        print(f"  {ca:8.1f}  {r5['aligned']:>9.1%}  {r5['misaligned']:>11.1%}  "
              f"{r5['semantic']:>9.1%}  {r5['overall']:>9.1%}{marker}")

    if best_ca_phi2 >= 5.0:
        print(f"\n  Extending sweep (best at boundary)...")
        for ca in [10.0, 20.0]:
            r = run_variant_eval_phi2(
                model_phi2, tokenizer_phi2, fact_types_phi2, [5], n_eval,
                auto_pairs=auto_pairs_phi2, completion_alpha=ca)
            r5 = r[5]
            marker = ""
            if r5["overall"] > best_overall:
                best_overall = r5["overall"]
                best_ca_phi2 = ca
                marker = " <-- best"
            print(f"  {ca:8.1f}  {r5['aligned']:>9.1%}  {r5['misaligned']:>11.1%}  "
                  f"{r5['semantic']:>9.1%}  {r5['overall']:>9.1%}{marker}")

    print(f"\n  Phi-2 best completion_alpha: {best_ca_phi2}")

    # ── Full eval: both models ──
    print(f"\n  Running full variant eval...")

    # GPT-2 Small: no T_auto baseline + with T_auto (c_alpha=0.3, known optimal)
    print("  GPT-2 Small (no T_auto)...")
    r_small_no = run_variant_eval_phi2(
        model_small, tokenizer_small, fact_types_small, n_facts_list, n_eval,
        auto_pairs=None)
    print("  GPT-2 Small (with T_auto, c_alpha=0.3)...")
    r_small_ta = run_variant_eval_phi2(
        model_small, tokenizer_small, fact_types_small, n_facts_list, n_eval,
        auto_pairs=auto_pairs_small, completion_alpha=0.3)

    # Phi-2: no T_auto baseline + with T_auto at best c_alpha
    print("  Phi-2 (no T_auto)...")
    r_phi2_no = run_variant_eval_phi2(
        model_phi2, tokenizer_phi2, fact_types_phi2, n_facts_list, n_eval,
        auto_pairs=None)
    print(f"  Phi-2 (with T_auto, c_alpha={best_ca_phi2})...")
    r_phi2_ta = run_variant_eval_phi2(
        model_phi2, tokenizer_phi2, fact_types_phi2, n_facts_list, n_eval,
        auto_pairs=auto_pairs_phi2, completion_alpha=best_ca_phi2)

    # ── Print results ──
    print(f"\n{'=' * 70}")
    print("  T_AUTO COMPARISON RESULTS")
    print(f"{'=' * 70}")

    # Standard recall (aligned without T_auto)
    print(f"\n  Standard Recall (aligned, no T_auto):")
    print(f"  {'n':>3}  {'GPT-2 Small':>13}  {'Phi-2':>8}  {'Delta':>8}")
    print(f"  {'─'*3}  {'─'*13}  {'─'*8}  {'─'*8}")
    for n in sorted(r_small_no.keys()):
        if n not in r_phi2_no:
            continue
        s = r_small_no[n]["aligned"]
        p = r_phi2_no[n]["aligned"]
        print(f"  {n:3d}  {s:>13.1%}  {p:>8.1%}  {p-s:>+7.1%}")

    # Per-category with T_auto
    for cat in ["aligned", "misaligned", "semantic"]:
        print(f"\n  {cat.title()} (with T_auto):")
        print(f"  {'n':>3}  {'GPT-2 Small':>13}  {'Phi-2':>8}  {'Delta':>8}")
        print(f"  {'─'*3}  {'─'*13}  {'─'*8}  {'─'*8}")
        for n in sorted(r_small_ta.keys()):
            if n not in r_phi2_ta:
                continue
            s = r_small_ta[n][cat]
            p = r_phi2_ta[n][cat]
            print(f"  {n:3d}  {s:>13.1%}  {p:>8.1%}  {p-s:>+7.1%}")

    # Overall summary
    print(f"\n  Overall (with T_auto):")
    print(f"  {'n':>3}  {'Small no TA':>13}  {'Small+TA':>10}  "
          f"{'Phi2 no TA':>12}  {'Phi2+TA':>9}  {'Δ Small':>9}  {'Δ Phi2':>8}")
    print(f"  {'─'*3}  {'─'*13}  {'─'*10}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*8}")
    for n in sorted(r_small_ta.keys()):
        if n not in r_phi2_ta:
            continue
        sn = r_small_no[n]["overall"]
        st = r_small_ta[n]["overall"]
        pn = r_phi2_no[n]["overall"]
        pt = r_phi2_ta[n]["overall"]
        print(f"  {n:3d}  {sn:>13.1%}  {st:>10.1%}  {pn:>12.1%}  "
              f"{pt:>9.1%}  {st-sn:>+8.1%}  {pt-pn:>+7.1%}")

    print(f"\n  GPT-2 Small: alpha=0.5, completion_alpha=0.3")
    print(f"  Phi-2: alpha=50.0, completion_alpha={best_ca_phi2}")

    return {
        "small_no_tauto": r_small_no,
        "small_tauto": r_small_ta,
        "phi2_no_tauto": r_phi2_no,
        "phi2_tauto": r_phi2_ta,
        "best_ca_phi2": best_ca_phi2,
    }


# ── Main ─────────────────────────────────────────────────────────────

def run_experiment(n_eval: int = 50, quick: bool = False, seed: int = 42,
                   with_tauto: bool = False, completion_alpha: float = 0.3):
    device = get_device()

    print("=" * 70)
    print("  Exp 19: Phi-2 (2.7B) Cross-Architecture Transfer Test")
    print("=" * 70)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print(f"  Transfer:  Zero-shot (no trained gates)")
    print()

    # -- Load Phi-2 + tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(
        PHI2_MODEL, trust_remote_code=True)
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    print(f"\n  Phi-2 tokenizer: {len(tokenizer)} vocab")
    print(f"  Valid fact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"    {ft.name}: {len(ft.entities)} entities")
    print(f"  Linking token IDs: {linking_ids}")

    n_facts_list = [1, 3, 5, 7] if not quick else [1, 3, 5]

    model = load_phi2(device, alpha=1.0)
    model.set_linking_token_ids(linking_ids)

    # Phase 1: Zero-shot proof of concept
    phase1 = run_phase1(model, tokenizer, fact_types, n_eval, n_facts_list)

    # Phase 2: Alpha sweep
    phase2, best_alpha = run_phase2(model, tokenizer, fact_types, n_eval)

    # Phase 3: Pattern separation at best alpha
    phase3_no_ps, phase3_ps = run_phase3(
        model, tokenizer, fact_types, n_eval, n_facts_list)

    # Phase 4: Head-to-head (optional)
    phase4_small = phase4_phi2 = None
    if not quick:
        from transformers import GPT2Tokenizer
        tokenizer_small = GPT2Tokenizer.from_pretrained('gpt2')
        fact_types_small = build_fact_types(tokenizer_small)
        linking_ids_small = get_linking_bpe_ids(tokenizer_small)

        model_small = load_gpt2_small(device, alpha=1.0)
        model_small.set_linking_token_ids(linking_ids_small)

        phase4_small, phase4_phi2 = run_phase4(
            model, model_small,
            tokenizer, tokenizer_small,
            fact_types, fact_types_small,
            n_eval, n_facts_list)

    # Phase 5: T_auto Pattern Completion (if requested)
    phase5 = None
    if with_tauto:
        phase5 = run_phase5(model, tokenizer, fact_types, n_eval,
                            n_facts_list, completion_alpha)

    # -- Save results --
    config_phi2 = model.base_model.config
    d_model = config_phi2.hidden_size
    n_layers = getattr(config_phi2, 'n_layer',
                       getattr(config_phi2, 'num_hidden_layers', -1))

    output = {
        "config": {
            "model": PHI2_MODEL,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_eval": n_eval,
            "seed": seed,
            "best_alpha": best_alpha,
            "n_trace_heads": 8,
            "d_trace": 64,
            "gate_mode": "hardcoded_linking_mask",
            "dtype": "float16",
            "n_fact_types": len(fact_types),
            "fact_type_names": [ft.name for ft in fact_types],
        },
        "phase1_zero_shot": {
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
            "gpt2_small": {str(k): v for k, v in phase4_small.items()},
            "phi2": {str(k): v for k, v in phase4_phi2.items()},
        }

    os.makedirs("results", exist_ok=True)
    path = "results/exp19_phi2_transfer.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")

    # -- Summary --
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 19 SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model: {PHI2_MODEL} ({d_model}d, {n_layers}L)")
    print(f"  Transfer: Zero-shot (no trained gates)")
    print(f"  Best alpha: {best_alpha}")
    print(f"\n  Cross-context accuracy (PS 8x_k16, alpha={best_alpha}):")
    for n in sorted(phase3_ps.keys()):
        ps = phase3_ps[n]['cross_ctx']
        gap = phase3_ps[n]['gap']
        print(f"    n={n}: {ps:.1%} (gap: {gap:+.1%})")

    if phase4_small is not None and phase4_phi2 is not None:
        print(f"\n  GPT-2 Small vs Phi-2 (PS 8x_k16):")
        for n in sorted(phase4_small.keys()):
            if n not in phase4_phi2:
                continue
            sc = phase4_small[n]['cross_ctx']
            pc = phase4_phi2[n]['cross_ctx']
            print(f"    n={n}: Small {sc:.1%} → Phi-2 {pc:.1%} "
                  f"({pc-sc:+.1%})")

    trace_params = sum(p.numel() for p in model.trace.parameters())
    base_params = sum(p.numel() for p in model.base_model.parameters())
    print(f"\n  Trace params: {trace_params:,} "
          f"({trace_params/base_params:.2%} of base model)")
    print(f"  Base params:  {base_params:,}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 19: Phi-2 cross-architecture transfer test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 episodes, skip phase 4)")
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-tauto", action="store_true",
                        help="Include Phase 5: T_auto pattern completion")
    parser.add_argument("--completion-alpha", type=float, default=0.3,
                        help="Completion alpha for T_auto (default: 0.3)")
    parser.add_argument("--compare-tauto", action="store_true",
                        help="Standalone: Phi-2 vs GPT-2 Small T_auto comparison")
    args = parser.parse_args()

    n_eval = args.n_eval or (20 if args.quick else 50)

    if args.compare_tauto:
        n_facts_list = [1, 3, 5, 7] if not args.quick else [1, 3, 5]
        run_tauto_comparison(
            n_eval=n_eval, n_facts_list=n_facts_list,
            completion_alpha=args.completion_alpha, seed=args.seed)
    else:
        run_experiment(
            n_eval=n_eval, quick=args.quick, seed=args.seed,
            with_tauto=args.with_tauto,
            completion_alpha=args.completion_alpha)
