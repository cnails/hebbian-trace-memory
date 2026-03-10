"""Experiment 8: GPT-2 Trace Integration.

Tests whether Hebbian trace memory generalizes to a pretrained GPT-2 Small
(124M params, frozen). External trace module (~1M params) attaches to GPT-2's
token embeddings for context-free Q/V, stores associations via Hebbian
learning, and adds trace-based logit bias via wte projection.

Phase 1: Setup & Diagnostics
  - Validate single-token entities, map linking tokens to BPE IDs
  - Verify BPE tokenization preserves shift structure

Phase 2: Proof of Concept (random projections, no fine-tuning)
  - Cross-context eval at n=1,3,5,7,10
  - Success threshold: >15% at n=1 (random ≈ 5%)

Phase 3: Alpha Sweep
  - alpha ∈ {0.1, 0.5, 1.0, 2.0, 5.0} at n=1 and n=5
  - Finds optimal logit injection strength

Phase 4: Pattern Separation
  - 8x_k16 sparse expansion on trace Q

Usage:
    python -m hebbian_trace.experiments.exp8_gpt2_trace --quick
    python -m hebbian_trace.experiments.exp8_gpt2_trace
"""

import argparse
import time

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    evaluate_gpt2_baseline, evaluate_gpt2_cross_context,
    evaluate_gpt2_cross_context_baseline,
    tokenize_fact, tokenize_question,
)


def get_device(requested: str | None = None) -> torch.device:
    """Detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Phase 1: Setup & Diagnostics ────────────────────────────────────

def run_phase1_diagnostics(tokenizer, fact_types, linking_ids):
    """Verify BPE tokenization, entity pools, and shift structure."""
    print(f"\n{'─' * 65}")
    print("PHASE 1: Setup & Diagnostics")
    print(f"{'─' * 65}")

    # Report entity pools
    print(f"\nFact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"  {ft.name}: {len(ft.entities)} single-token entities")
        examples = [e[0] for e in ft.entities[:5]]
        print(f"    examples: {', '.join(examples)}")

    # Report linking tokens
    print(f"\nLinking BPE tokens: {len(linking_ids)}")
    for lid in linking_ids:
        tok_str = tokenizer.decode([lid])
        print(f"  ID {lid}: '{tok_str}'")

    # Verify shift structure for each fact type
    print("\n--- BPE Shift Verification ---")
    all_ok = True
    for ft in fact_types:
        entity_name, entity_id = ft.entities[0]
        template = ft.fact_templates[0]
        q_template = ft.question_templates[0]

        # Tokenize fact
        fact_ids = tokenize_fact(tokenizer, template, entity_name)
        fact_tokens = [tokenizer.decode([tid]) for tid in fact_ids]

        # Tokenize question
        q_ids = tokenize_question(tokenizer, q_template)
        q_tokens = [tokenizer.decode([tid]) for tid in q_ids]

        # Find linking token position in fact
        linking_pos = None
        for i, fid in enumerate(fact_ids):
            if fid in linking_ids:
                linking_pos = i
                break

        # Find entity position in fact
        entity_pos = None
        for i, fid in enumerate(fact_ids):
            if fid == entity_id:
                entity_pos = i
                break

        # Check shift: Q at linking_pos-1 (concept word), V at linking_pos+1 (entity)
        q_store_pos = linking_pos - 1 if linking_pos is not None else None
        v_store_pos = linking_pos + 1 if linking_pos is not None else None

        # Check question: concept word should be at position (last - 1)
        q_last = len(q_ids) - 1  # "?" position
        q_concept_pos = q_last - 1  # should be concept word

        # Compare Q at concept position in fact vs question
        fact_q_token = fact_tokens[q_store_pos] if q_store_pos is not None else "?"
        question_q_token = q_tokens[q_concept_pos] if q_concept_pos < len(q_tokens) else "?"

        # Check if they match (same BPE token)
        fact_q_id = fact_ids[q_store_pos] if q_store_pos is not None else -1
        question_q_id = q_ids[q_concept_pos] if q_concept_pos < len(q_ids) else -2

        match = fact_q_id == question_q_id
        status = "✓" if match else "✗ MISMATCH"
        if not match:
            all_ok = False

        print(f"\n  {ft.name}:")
        print(f"    Fact: {fact_tokens}")
        print(f"    Q:    {q_tokens}")
        print(f"    Linking '{fact_tokens[linking_pos]}' at pos {linking_pos}"
              if linking_pos is not None else "    Linking: NOT FOUND")
        print(f"    Store Q: pos {q_store_pos} = '{fact_q_token}', "
              f"Store V: pos {v_store_pos} = "
              f"'{fact_tokens[v_store_pos]}'" if v_store_pos is not None else "")
        print(f"    Query Q_addr: pos {q_concept_pos} = '{question_q_token}'")
        print(f"    Q match: {status} "
              f"(fact_Q='{fact_q_token}' vs query_Q='{question_q_token}')")

    print(f"\n  Overall shift verification: {'ALL OK' if all_ok else 'ISSUES FOUND'}")
    return all_ok


# ── Phase 2: Proof of Concept ───────────────────────────────────────

def run_phase2_proof_of_concept(model, tokenizer, fact_types, n_eval,
                                n_facts_list, verbose=False):
    """Cross-context eval with random projections."""
    print(f"\n{'─' * 65}")
    print("PHASE 2: Proof of Concept (random projections, no fine-tuning)")
    print(f"{'─' * 65}")

    results = {}  # {n_facts: {condition: GPT2EvalResults}}

    for n_facts in n_facts_list:
        print(f"\n--- n_facts = {n_facts} ---")
        episodes = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)

        t0 = time.time()

        # Cross-context baseline (no trace, question-only)
        cc_bl = evaluate_gpt2_cross_context_baseline(
            model, episodes, fact_types, verbose=verbose)

        # Cross-context with trace
        cc = evaluate_gpt2_cross_context(
            model, episodes, fact_types, verbose=verbose)

        # In-context baseline (facts + question)
        bl = evaluate_gpt2_baseline(
            model, episodes, fact_types, tokenizer, verbose=verbose)

        dt = time.time() - t0
        print(f"  Baseline (in-ctx):   {bl.accuracy:.1%} "
              f"({bl.n_correct}/{bl.n_total})")
        print(f"  Cross-ctx + trace:   {cc.accuracy:.1%} "
              f"({cc.n_correct}/{cc.n_total})")
        print(f"  Cross-ctx baseline:  {cc_bl.accuracy:.1%} "
              f"({cc_bl.n_correct}/{cc_bl.n_total})")
        gap = cc.accuracy - cc_bl.accuracy
        print(f"  Gap (trace benefit): {gap:+.1%}")
        print(f"  Time: {dt:.1f}s")

        results[n_facts] = {
            'baseline': bl, 'cross_ctx': cc, 'cross_bl': cc_bl,
        }

    # Summary table
    print(f"\n{'═' * 65}")
    print("PHASE 2 SUMMARY")
    print(f"{'═' * 65}")
    print(f"{'n':>3} │ {'Baseline':>10} {'Cross+Trace':>13} "
          f"{'Cross BL':>10} │ {'Gap':>7}")
    print(f"{'─' * 3}─┼─{'─' * 10}─{'─' * 13}─{'─' * 10}─┼─{'─' * 7}")
    for n_facts in n_facts_list:
        r = results[n_facts]
        gap = r['cross_ctx'].accuracy - r['cross_bl'].accuracy
        print(f"{n_facts:3d} │ {r['baseline'].accuracy:>10.1%} "
              f"{r['cross_ctx'].accuracy:>13.1%} "
              f"{r['cross_bl'].accuracy:>10.1%} │ {gap:>+6.1%}")

    return results


# ── Phase 3: Alpha Sweep ──────────────────────────────────────────

def run_phase3_alpha_sweep(model, tokenizer, fact_types, n_eval,
                           alphas_to_test, verbose=False):
    """Test different alpha (logit injection strength) values.

    With logit injection, alpha directly scales trace_logits added to
    GPT-2's output logits. Too low → trace signal invisible; too high →
    overwhelms GPT-2's own predictions.
    """
    print(f"\n{'─' * 65}")
    print("PHASE 3: Alpha Sweep (logit injection strength)")
    print(f"{'─' * 65}")

    original_alpha = model.trace.alpha
    results = {}  # {alpha: {n_facts: GPT2EvalResults}}

    test_ns = [1, 5]

    for alpha in alphas_to_test:
        model.trace.alpha = alpha
        results[alpha] = {}
        print(f"\n--- alpha = {alpha} ---")

        for n_facts in test_ns:
            episodes = make_gpt2_eval_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                seed=42 + n_facts * 1000)

            cc = evaluate_gpt2_cross_context(
                model, episodes, fact_types, verbose=verbose)

            # Also get in-context baseline at this alpha
            bl = evaluate_gpt2_baseline(
                model, episodes, fact_types, tokenizer, verbose=verbose)

            results[alpha][n_facts] = {'cross_ctx': cc, 'baseline': bl}
            print(f"  n={n_facts}: cross-ctx {cc.accuracy:.1%}, "
                  f"baseline {bl.accuracy:.1%}")

    # Summary
    print(f"\n{'═' * 65}")
    print("ALPHA SWEEP SUMMARY")
    print(f"{'═' * 65}")
    print(f"{'Alpha':>6} │ {'n=1 Cross':>10} {'n=1 BL':>8} │ "
          f"{'n=5 Cross':>10} {'n=5 BL':>8}")
    print(f"{'─' * 6}─┼─{'─' * 10}─{'─' * 8}─┼─{'─' * 10}─{'─' * 8}")

    best_alpha = None
    best_score = -1
    for alpha in alphas_to_test:
        r = results[alpha]
        cc1 = r[1]['cross_ctx'].accuracy if 1 in r else 0
        bl1 = r[1]['baseline'].accuracy if 1 in r else 0
        cc5 = r[5]['cross_ctx'].accuracy if 5 in r else 0
        bl5 = r[5]['baseline'].accuracy if 5 in r else 0
        avg_cross = (cc1 + cc5) / 2
        marker = ""
        if avg_cross > best_score:
            best_score = avg_cross
            best_alpha = alpha
            marker = " ←"
        print(f"{alpha:6.1f} │ {cc1:>10.1%} {bl1:>8.1%} │ "
              f"{cc5:>10.1%} {bl5:>8.1%}{marker}")

    print(f"\nBest alpha: {best_alpha} (avg cross-ctx: {best_score:.1%})")

    # Restore best alpha
    model.trace.alpha = best_alpha
    return results, best_alpha


# ── Phase 4: Pattern Separation ─────────────────────────────────────

def run_phase4_pattern_separation(model, tokenizer, fact_types, n_eval,
                                  n_facts_list, verbose=False):
    """Test pattern separation on trace Q."""
    print(f"\n{'─' * 65}")
    print("PHASE 4: Pattern Separation (8x_k16)")
    print(f"{'─' * 65}")

    results_no_ps = {}
    results_ps = {}

    for n_facts in n_facts_list:
        print(f"\n--- n_facts = {n_facts} ---")
        episodes = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)

        # Without pattern separation
        model.disable_pattern_separation()
        cc_no = evaluate_gpt2_cross_context(
            model, episodes, fact_types, verbose=verbose)
        results_no_ps[n_facts] = cc_no

        # With pattern separation
        model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
        cc_ps = evaluate_gpt2_cross_context(
            model, episodes, fact_types, verbose=verbose)
        results_ps[n_facts] = cc_ps

        diff = cc_ps.accuracy - cc_no.accuracy
        print(f"  No PS:   {cc_no.accuracy:.1%}")
        print(f"  8x_k16: {cc_ps.accuracy:.1%} ({diff:+.1%})")

    # Summary
    print(f"\n{'═' * 65}")
    print("PATTERN SEPARATION SUMMARY")
    print(f"{'═' * 65}")
    print(f"{'n':>3} │ {'No PS':>8} {'8x_k16':>8} │ {'Diff':>7}")
    print(f"{'─' * 3}─┼─{'─' * 8}─{'─' * 8}─┼─{'─' * 7}")
    for n_facts in n_facts_list:
        no = results_no_ps[n_facts].accuracy
        ps = results_ps[n_facts].accuracy
        print(f"{n_facts:3d} │ {no:>8.1%} {ps:>8.1%} │ {ps-no:>+6.1%}")

    return results_no_ps, results_ps


# ── Main experiment runners ─────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick smoke test (~2-5 min)."""
    print("=" * 65)
    print("EXPERIMENT 8: GPT-2 Trace Integration (quick)")
    print("=" * 65)

    dev = get_device(device)
    print(f"Device: {dev}")

    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    print(f"\nLoading GPT-2 Small...")
    t0 = time.time()
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, inject_layer=6,
        alpha=1.0, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(dev)
    model.set_linking_token_ids(linking_ids)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  GPT-2 params: {sum(p.numel() for p in model.gpt2.parameters()):,}")
    print(f"  Trace params: {sum(p.numel() for p in model.trace.parameters()):,}")

    # Phase 1
    run_phase1_diagnostics(tokenizer, fact_types, linking_ids)

    # Phase 2 (quick: 20 episodes)
    run_phase2_proof_of_concept(
        model, tokenizer, fact_types,
        n_eval=20, n_facts_list=[1, 3, 5],
        verbose=True)

    print(f"\n{'═' * 65}")
    print("QUICK TEST COMPLETE")
    print(f"{'═' * 65}")


def run(device=None, seed=42, n_eval=100):
    """Full experiment (~15-30 min)."""
    print("=" * 65)
    print("EXPERIMENT 8: GPT-2 Trace Integration")
    print("=" * 65)

    dev = get_device(device)
    print(f"Device: {dev}")

    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    print(f"\nLoading GPT-2 Small...")
    t0 = time.time()
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, inject_layer=6,
        alpha=1.0, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(dev)
    model.set_linking_token_ids(linking_ids)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  GPT-2 params: {sum(p.numel() for p in model.gpt2.parameters()):,}")
    print(f"  Trace params: {sum(p.numel() for p in model.trace.parameters()):,}")

    # Phase 1: Diagnostics
    shift_ok = run_phase1_diagnostics(tokenizer, fact_types, linking_ids)
    if not shift_ok:
        print("\n⚠ Shift verification issues found. Proceeding anyway.")

    # Phase 2: Proof of concept (full n range)
    phase2_results = run_phase2_proof_of_concept(
        model, tokenizer, fact_types,
        n_eval=n_eval, n_facts_list=[1, 3, 5, 7],
        verbose=True)

    # Phase 3: Alpha sweep
    phase3_results, best_alpha = run_phase3_alpha_sweep(
        model, tokenizer, fact_types,
        n_eval=n_eval, alphas_to_test=[0.1, 0.5, 1.0, 2.0, 5.0],
        verbose=False)

    # Phase 4: Pattern separation (at best alpha)
    phase4_no, phase4_ps = run_phase4_pattern_separation(
        model, tokenizer, fact_types,
        n_eval=n_eval, n_facts_list=[1, 3, 5, 7],
        verbose=False)

    # Final summary
    print(f"\n{'═' * 65}")
    print("EXPERIMENT 8 COMPLETE")
    print(f"{'═' * 65}")
    print(f"Best alpha: {best_alpha}")
    if 1 in phase2_results:
        cc1 = phase2_results[1]['cross_ctx'].accuracy
        bl1 = phase2_results[1]['cross_bl'].accuracy
        threshold = 0.15
        status = "PASS" if cc1 > threshold else "BELOW THRESHOLD"
        print(f"Cross-ctx n=1: {cc1:.1%} (baseline: {bl1:.1%}, "
              f"threshold: {threshold:.0%}) → {status}")
    if 1 in phase4_no and 1 in phase4_ps:
        print(f"Pattern sep n=1: {phase4_no[1].accuracy:.1%} → "
              f"{phase4_ps[1].accuracy:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 8: GPT-2 Trace Integration")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (20 episodes)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=100)

    args = parser.parse_args()

    if args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run(device=args.device, seed=args.seed, n_eval=args.n_eval)
