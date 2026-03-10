"""Experiment 10: Paragraph-Level Storage (Zero Architecture Changes).

Tests whether the current system (learned gate + shift-2 + pattern separation)
generalizes to multi-sentence paragraphs with zero changes.

Central question (Phase 3): In real text, linking tokens always appear in
irrelevant sentences. The gate is token-level, not semantic — it WILL fire
on "is" in "The weather is nice." If noisy filler drops accuracy by >20pp,
that's a fundamental limitation requiring dual gates (exp 11). If within
10pp, current architecture is already viable for free text.

Phase 1: Setup & Gate Training (reuse exp9 pipeline)
Phase 2: Paragraph Format Test (single_pass vs cumulative, no filler)
Phase 3: Filler Noise Resistance (THE CENTRAL TEST)
Phase 4: Gate Activation Visualization on Paragraphs
Phase 5: Scale Stress Test
Phase 6: Failure Mode Analysis

Usage:
    python -m hebbian_trace.experiments.exp10_paragraph_storage --quick
    python -m hebbian_trace.experiments.exp10_paragraph_storage
"""

import argparse
import time

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    make_paragraph_episodes,
    evaluate_gpt2_cross_context, evaluate_gpt2_cross_context_baseline,
    _get_all_entity_ids,
    FILLER_NO_LINK, FILLER_WITH_LINK,
)
from .exp9_learned_gating import train_gate


def get_device(requested: str | None = None) -> torch.device:
    """Detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Phase 1: Setup & Gate Training ──────────────────────────────────

def setup_and_train(device, tokenizer, fact_types, linking_ids,
                    n_steps=1400, seed=42):
    """Load GPT-2 + trace, train W_gate (reuse exp9 pipeline).

    Returns trained model ready for paragraph testing.
    """
    print(f"\n{'=' * 65}")
    print("PHASE 1: Setup & Gate Training")
    print(f"{'=' * 65}")

    print(f"\nLoading GPT-2 Small...")
    t0 = time.time()
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, inject_layer=6,
        alpha=1.0, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(device)
    model.set_linking_token_ids(linking_ids)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  GPT-2 params: {sum(p.numel() for p in model.gpt2.parameters()):,}")
    print(f"  Trace params: {sum(p.numel() for p in model.trace.parameters()):,}")

    print(f"\n  Fact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"    {ft.name}: {len(ft.entities)} entities")
    print(f"  Linking tokens: {len(linking_ids)}")
    for lid in linking_ids:
        print(f"    ID {lid}: '{tokenizer.decode([lid])}'")

    # Train W_gate: 5-stage curriculum (identical to exp9)
    print(f"\n  Training W_gate (exp9 pipeline, {n_steps} total steps)...")
    n_per = n_steps // 5

    # Stage 1: n_facts=1
    train_gate(model, tokenizer, fact_types,
               n_steps=max(n_per, 200), n_facts_train=1,
               lr=3e-3, device=device,
               log_every=100, seed=seed,
               gate_only=True, grad_clip=1.0)

    # Stage 2: n_facts=3
    train_gate(model, tokenizer, fact_types,
               n_steps=max(n_per, 200), n_facts_train=3,
               lr=1e-3, device=device,
               log_every=100, seed=seed + 5000,
               gate_only=True, grad_clip=1.0)

    # Stage 3: n_facts=5
    train_gate(model, tokenizer, fact_types,
               n_steps=max(n_per, 200), n_facts_train=5,
               lr=3e-4, device=device,
               log_every=100, seed=seed + 10000,
               gate_only=True, grad_clip=1.0)

    # Stage 4: n_facts=5, L1=0.5
    train_gate(model, tokenizer, fact_types,
               n_steps=max(n_per, 200), n_facts_train=5,
               lr=3e-4, device=device,
               log_every=100, seed=seed + 15000,
               gate_only=True, grad_clip=1.0,
               l1_lambda=0.5)

    # Stage 5: n_facts=5, L1=1.0
    train_gate(model, tokenizer, fact_types,
               n_steps=max(n_per, 200), n_facts_train=5,
               lr=1e-4, device=device,
               log_every=100, seed=seed + 20000,
               gate_only=True, grad_clip=1.0,
               l1_lambda=1.0)

    # Restore requires_grad for eval
    for p in model.trace.parameters():
        p.requires_grad_(True)

    # Sanity check: reproduce exp9 at n=3
    print(f"\n  Sanity check: exp9-style eval at n=3...")
    episodes = make_gpt2_eval_episodes(
        n_episodes=20, n_facts=3,
        tokenizer=tokenizer, fact_types=fact_types,
        seed=42 + 3000)
    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)
    cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
    print(f"  Exp9-style n=3: {cc.accuracy:.1%} "
          f"({cc.n_correct}/{cc.n_total})")

    return model


# ── Phase 2: Paragraph Format Test ──────────────────────────────────

def phase2_paragraph_format(model, tokenizer, fact_types, linking_ids,
                            n_eval, n_facts_list, device):
    """Compare single-pass vs cumulative write on pure-fact paragraphs."""
    print(f"\n{'=' * 65}")
    print("PHASE 2: Paragraph Format Test (no filler)")
    print(f"  single_pass vs cumulative — quantifies repetition benefit")
    print(f"{'=' * 65}")

    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)

    results = {}  # {condition: {n: GPT2EvalResults}}

    for write_mode in ["cumulative", "single"]:
        cond = f"{write_mode}"
        results[cond] = {}
        print(f"\n--- Write mode: {write_mode} ---")

        for n_facts in n_facts_list:
            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode="none", n_filler=0,
                write_mode=write_mode,
                seed=42 + n_facts * 1000)

            cc = evaluate_gpt2_cross_context(
                model, episodes, fact_types)
            results[cond][n_facts] = cc
            print(f"  n={n_facts}: {cc.accuracy:.1%} "
                  f"({cc.n_correct}/{cc.n_total})")

    # Summary table
    print(f"\n{'=' * 65}")
    print("PHASE 2 SUMMARY: Paragraph Format")
    print(f"{'=' * 65}")

    conds = list(results.keys())
    header = f"{'n':>3} │"
    for c in conds:
        header += f" {c:>14}"
    header += f" {'gap':>10}"
    print(header)
    print(f"{'─' * 3}─┼─" + "─" * (15 * len(conds) + 11))

    for n_facts in n_facts_list:
        row = f"{n_facts:3d} │"
        accs = []
        for c in conds:
            acc = results[c][n_facts].accuracy
            accs.append(acc)
            row += f" {acc:>13.1%}"
        if len(accs) == 2:
            gap = accs[0] - accs[1]
            row += f"   {gap:>+6.1%}"
        print(row)

    print(f"\n  Interpretation:")
    if n_facts_list:
        n5 = 5 if 5 in n_facts_list else n_facts_list[-1]
        cum_acc = results["cumulative"].get(n5)
        sin_acc = results["single"].get(n5)
        if cum_acc and sin_acc:
            gap = cum_acc.accuracy - sin_acc.accuracy
            if gap > 0.15:
                print(f"  >> Large gap ({gap:.1%}): trace needs repetition"
                      f" → replay mechanism argument")
            elif gap > 0.05:
                print(f"  >> Moderate gap ({gap:.1%}): repetition helps"
                      f" but single-pass is viable")
            else:
                print(f"  >> Small gap ({gap:.1%}): single-pass Hebbian"
                      f" sufficient")

    return results


# ── Phase 3: Filler Noise Resistance (THE CENTRAL TEST) ────────────

def phase3_noise_resistance(model, tokenizer, fact_types, linking_ids,
                            n_eval, n_facts, device):
    """Test filler injection on single-pass paragraphs.

    THE MOST IMPORTANT PHASE. Directly answers: can the current
    token-level gate handle realistic text?

    Decision boundary:
      gap < 10pp → current architecture viable for free text
      gap 10-20pp → dual gates worthwhile
      gap > 20pp → dual gates essential (exp 11)
    """
    print(f"\n{'=' * 65}")
    print("PHASE 3: Filler Noise Resistance (THE CENTRAL TEST)")
    print(f"  n_facts={n_facts}, single-pass write")
    print(f"{'=' * 65}")

    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)

    conditions = [
        ("no_filler",      "none",  0),
        ("safe_2",         "safe",  2),
        ("safe_5",         "safe",  5),
        ("noisy_2",        "noisy", 2),
        ("noisy_5",        "noisy", 5),
        ("mixed_3",        "mixed", 3),
    ]

    results = {}

    # SAME seed for all conditions — ensures identical fact selections.
    # Only filler varies (separate RNG in make_paragraph_episodes).
    base_seed = 42 + n_facts * 1000

    # Verify same facts across conditions
    ep_nf = make_paragraph_episodes(
        n_episodes=1, n_facts=n_facts, tokenizer=tokenizer,
        fact_types=fact_types, filler_mode="none", n_filler=0,
        write_mode="single", seed=base_seed)
    ep_s5 = make_paragraph_episodes(
        n_episodes=1, n_facts=n_facts, tokenizer=tokenizer,
        fact_types=fact_types, filler_mode="safe", n_filler=5,
        write_mode="single", seed=base_seed)
    facts_match = all(
        a[:3] == b[:3]
        for a, b in zip(ep_nf[0].facts, ep_s5[0].facts))
    print(f"\n  Fact identity check: {'PASS' if facts_match else 'FAIL'}")
    if not facts_match:
        print(f"  WARNING: different facts across conditions!")

    # Test at multiple tau values to see if sharpening gate fixes leak
    tau_values = [1.0, 0.3, 0.1]

    for tau in tau_values:
        model.trace.set_gate_tau(tau)
        for cond_name, filler_mode, n_filler in conditions:
            key = f"{cond_name}_t{tau}" if tau != 1.0 else cond_name
            print(f"\n--- {key} (filler={filler_mode}, n={n_filler},"
                  f" tau={tau}) ---")

            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode=filler_mode, n_filler=n_filler,
                write_mode="single",
                seed=base_seed)

            # Print example paragraph length
            if episodes:
                seq_len = len(episodes[0].train_sequences[0])
                print(f"  Example paragraph: {seq_len} tokens")

            cc = evaluate_gpt2_cross_context(
                model, episodes, fact_types)
            results[key] = cc
            print(f"  Accuracy: {cc.accuracy:.1%} "
                  f"({cc.n_correct}/{cc.n_total})")

    model.trace.set_gate_tau(1.0)  # restore

    # Hardcoded mask: test both no_filler AND safe_5 (isolate gate vs length)
    model.set_gate_mode(False)
    model.set_linking_token_ids(linking_ids)

    for hm_name, hm_filler, hm_n in [("hm_no_filler", "none", 0),
                                       ("hm_safe_5", "safe", 5)]:
        print(f"\n--- {hm_name} (hardcoded mask) ---")
        episodes = make_paragraph_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode=hm_filler, n_filler=hm_n,
            write_mode="single",
            seed=base_seed)
        cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
        results[hm_name] = cc
        print(f"  Accuracy: {cc.accuracy:.1%}")

    # Restore gate mode
    model.set_gate_mode(True)

    # Summary table — grouped by tau
    print(f"\n{'=' * 65}")
    print(f"PHASE 3 SUMMARY: Noise Resistance (n={n_facts})")
    print(f"{'=' * 65}")

    base_conds = ["no_filler", "safe_2", "safe_5",
                  "noisy_2", "noisy_5", "mixed_3"]

    for tau in tau_values:
        suffix = f"_t{tau}" if tau != 1.0 else ""
        tau_label = f"tau={tau}" if tau != 1.0 else "tau=1.0 (default)"
        nf_key = f"no_filler{suffix}"
        if nf_key not in results:
            continue

        base_acc = results[nf_key].accuracy
        print(f"\n  ── {tau_label} ──")
        print(f"  {'Condition':>18} │ {'Accuracy':>8} │ {'vs no_filler':>12}")
        print(f"  {'─' * 18}─┼─{'─' * 8}─┼─{'─' * 12}")

        for cn_base in base_conds:
            cn = f"{cn_base}{suffix}"
            if cn in results:
                acc = results[cn].accuracy
                gap = acc - base_acc
                gap_str = f"{gap:>+10.1%}" if cn_base != "no_filler" else "  (baseline)"
                print(f"  {cn:>18} │ {acc:>7.1%} │ {gap_str}")

    # Hardcoded mask (tau-independent)
    print(f"\n  ── Hardcoded mask (reference) ──")
    print(f"  {'Condition':>18} │ {'Accuracy':>8} │")
    print(f"  {'─' * 18}─┼─{'─' * 8}─┤")
    for hm_name in ["hm_no_filler", "hm_safe_5"]:
        if hm_name in results:
            acc = results[hm_name].accuracy
            print(f"  {hm_name:>18} │ {acc:>7.1%} │")

    # KEY DIAGNOSTIC: does safe filler hurt hardcoded mask?
    hm_nf = results.get("hm_no_filler")
    hm_s5 = results.get("hm_safe_5")
    if hm_nf and hm_s5:
        hm_gap = hm_nf.accuracy - hm_s5.accuracy
        print(f"\n  DIAGNOSTIC: hardcoded mask + safe_5 gap = {hm_gap:.1%}")
        if abs(hm_gap) < 0.05:
            print(f"  >> Hardcoded mask unaffected by safe filler (as expected)")
            print(f"     → safe filler issue is GATE-SPECIFIC, not length-related")
        else:
            print(f"  >> Hardcoded mask ALSO affected by safe filler!")
            print(f"     → issue is in the write mechanism, not the gate")

    # Tau comparison: how much does sharpening help?
    print(f"\n  ── TAU COMPARISON (safe_5 vs no_filler gap) ──")
    for tau in tau_values:
        suffix = f"_t{tau}" if tau != 1.0 else ""
        nf_key = f"no_filler{suffix}"
        s5_key = f"safe_5{suffix}"
        n5_key = f"noisy_5{suffix}"
        if nf_key in results and s5_key in results:
            nf_acc = results[nf_key].accuracy
            s5_acc = results[s5_key].accuracy
            safe_gap = nf_acc - s5_acc
            noisy_str = ""
            if n5_key in results:
                n5_acc = results[n5_key].accuracy
                noisy_gap = nf_acc - n5_acc
                noisy_str = f"  noisy_5 gap={noisy_gap:.1%}"
            print(f"    tau={tau}: no_filler={nf_acc:.1%}  "
                  f"safe_5 gap={safe_gap:.1%}{noisy_str}")

    # Decision (based on tau=1.0 / default)
    noisy5_acc = results.get("noisy_5")
    base_acc = results["no_filler"].accuracy
    if noisy5_acc:
        noise_gap = base_acc - noisy5_acc.accuracy
        print(f"\n  KEY RESULT (tau=1.0): noisy_5 gap = {noise_gap:.1%}")
        if noise_gap < 0.10:
            print(f"  >> Gap < 10pp: current architecture VIABLE for free text")
            print(f"     Dual gates optional optimization")
        elif noise_gap < 0.20:
            print(f"  >> Gap 10-20pp: current architecture has limits")
            print(f"     Dual gates WORTHWHILE for exp 11")
        else:
            print(f"  >> Gap > 20pp: FUNDAMENTAL limitation")
            print(f"     Dual gates ESSENTIAL for exp 11")

    # Best tau for noisy_5
    best_tau = None
    best_noisy_acc = -1
    for tau in tau_values:
        suffix = f"_t{tau}" if tau != 1.0 else ""
        n5_key = f"noisy_5{suffix}"
        if n5_key in results and results[n5_key].accuracy > best_noisy_acc:
            best_noisy_acc = results[n5_key].accuracy
            best_tau = tau
    if best_tau is not None:
        print(f"\n  Best tau for noisy_5: tau={best_tau} "
              f"({best_noisy_acc:.1%})")

    # Safe vs noisy control (tau=1.0)
    safe5_acc = results.get("safe_5")
    if safe5_acc and noisy5_acc:
        safe_gap = base_acc - safe5_acc.accuracy
        noisy_gap = base_acc - noisy5_acc.accuracy
        print(f"\n  Control (tau=1.0):")
        print(f"    safe_5 gap  = {safe_gap:.1%} (gate leak)")
        print(f"    noisy_5 gap = {noisy_gap:.1%} (leak + noise)")
        print(f"    Pure noise  = {noisy_gap - safe_gap:.1%}")

    return results


# ── Phase 4: Gate Activation Visualization ──────────────────────────

def phase4_gate_visualization(model, tokenizer, fact_types, linking_ids,
                              device):
    """Visualize gate activations on paragraph with facts + filler."""
    print(f"\n{'=' * 65}")
    print("PHASE 4: Gate Activation on Paragraphs")
    print(f"{'=' * 65}")

    trace = model.trace
    wte = model.gpt2.transformer.wte
    linking_set = set(linking_ids)

    # Build a sample paragraph with facts and noisy filler
    ft_name = fact_types[0]  # name
    ft_city = fact_types[1] if len(fact_types) > 1 else fact_types[0]
    ft_comp = fact_types[2] if len(fact_types) > 2 else fact_types[0]

    facts_text = [
        ft_name.fact_templates[0].text.replace("{X}", ft_name.entities[0][0]),
        ft_city.fact_templates[0].text.replace("{X}", ft_city.entities[0][0]),
        ft_comp.fact_templates[0].text.replace("{X}", ft_comp.entities[0][0]),
    ]

    fillers = [FILLER_WITH_LINK[0], FILLER_NO_LINK[0]]

    # Build paragraph: fact0 filler0 fact1 filler1 fact2
    parts = []
    for i, ft in enumerate(facts_text):
        parts.append(ft)
        if i < len(fillers):
            parts.append(fillers[i])
    paragraph = " ".join(parts)

    print(f"\n  Paragraph: \"{paragraph}\"")

    para_ids = tokenizer.encode(paragraph, add_special_tokens=False)
    para_tensor = torch.tensor([para_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        gate = trace.compute_gate(wte, para_tensor)

    gate_vals = gate[0].cpu().tolist()
    tokens = [tokenizer.decode([tid]) for tid in para_ids]

    # Classify each token
    link_fact_vals = []
    link_filler_vals = []
    nonlink_vals = []

    # Rough classification: tokens from fact sentences vs filler
    # Tokenize each part separately to know boundaries
    part_boundaries = []
    pos = 0
    for part_text in (facts_text + fillers):
        part_ids = tokenizer.encode(part_text, add_special_tokens=False)
        part_boundaries.append((pos, pos + len(part_ids), part_text))
        # Account for space separator
        pos += len(part_ids) + 1  # +1 for space token

    # Map fact vs filler
    fact_set = set(range(len(facts_text)))  # first 3 parts are facts
    filler_set = set(range(len(facts_text), len(facts_text) + len(fillers)))

    # Re-tokenize the full paragraph to get exact positions
    print(f"\n  Gate activations per token:")
    print(f"  {'Pos':>4} {'Token':>16} {'Gate':>6} {'Type':>12}")
    print(f"  {'─' * 4} {'─' * 16} {'─' * 6} {'─' * 12}")

    for i, (tok, g) in enumerate(zip(tokens, gate_vals)):
        is_link = para_ids[i] in linking_set

        # Determine if this position is in a fact or filler
        in_fact = False
        in_filler = False
        for p_idx, (start, end, _) in enumerate(part_boundaries):
            if start <= i < end:
                if p_idx < len(facts_text):
                    in_fact = True
                else:
                    in_filler = True
                break

        if is_link and in_fact:
            cat = "LINK-FACT"
            link_fact_vals.append(g)
        elif is_link and in_filler:
            cat = "LINK-FILLER"
            link_filler_vals.append(g)
        elif is_link:
            cat = "LINK-?"
        else:
            cat = ""
            nonlink_vals.append(g)

        bar_len = int(g * 30)
        bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
        print(f"  [{i:3d}] {tok:>14s}  {g:.3f} {bar} {cat}")

    # Category averages
    print(f"\n  Category averages:")
    if link_fact_vals:
        avg = sum(link_fact_vals) / len(link_fact_vals)
        print(f"    LINK-FACT:   {avg:.3f} (n={len(link_fact_vals)})")
    if link_filler_vals:
        avg = sum(link_filler_vals) / len(link_filler_vals)
        print(f"    LINK-FILLER: {avg:.3f} (n={len(link_filler_vals)})")
    if nonlink_vals:
        avg = sum(nonlink_vals) / len(nonlink_vals)
        print(f"    NON-LINK:    {avg:.3f} (n={len(nonlink_vals)})")

    if link_fact_vals and link_filler_vals:
        fact_avg = sum(link_fact_vals) / len(link_fact_vals)
        fill_avg = sum(link_filler_vals) / len(link_filler_vals)
        print(f"\n  LINK-FACT vs LINK-FILLER: {fact_avg:.3f} vs {fill_avg:.3f}")
        if abs(fact_avg - fill_avg) < 0.1:
            print(f"  >> Gate cannot distinguish fact-links from filler-links")
            print(f"     This is the boundary of token-level gating.")
            print(f"     Dual gates (exp 11) will solve this with gate_key.")
        else:
            print(f"  >> Unexpected: gate shows some semantic sensitivity!")


# ── Phase 5: Scale Stress Test ──────────────────────────────────────

def phase5_scale_test(model, tokenizer, fact_types, linking_ids,
                      n_eval, device):
    """Push to high fact counts with filler."""
    print(f"\n{'=' * 65}")
    print("PHASE 5: Scale Stress Test")
    print(f"{'=' * 65}")

    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)

    n_facts_list = [3, 5, 7, 10, 15]
    results = {}  # {condition: {n: accuracy}}

    for cond_name, filler_mode, n_filler_ratio in [
        ("no_filler", "none", 0),
        ("mixed_1:1", "mixed", 1),  # n_filler = n_facts
    ]:
        results[cond_name] = {}
        print(f"\n--- {cond_name} ---")

        for n_facts in n_facts_list:
            n_filler = n_facts * n_filler_ratio

            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode=filler_mode, n_filler=n_filler,
                write_mode="single",
                seed=42 + n_facts * 2000)

            # Show sequence length for first episode
            if episodes:
                seq_len = len(episodes[0].train_sequences[0])
                print(f"  n={n_facts} ({seq_len} tokens): ", end="")

            cc = evaluate_gpt2_cross_context(
                model, episodes, fact_types)
            results[cond_name][n_facts] = cc.accuracy
            print(f"{cc.accuracy:.1%}")

    # Also run hardcoded mask no_filler for reference
    results["hardcoded"] = {}
    print(f"\n--- hardcoded_mask (reference) ---")
    model.set_gate_mode(False)
    model.set_linking_token_ids(linking_ids)
    for n_facts in n_facts_list:
        episodes = make_paragraph_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode="none", n_filler=0,
            write_mode="single",
            seed=42 + n_facts * 2000)
        cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
        results["hardcoded"][n_facts] = cc.accuracy
        print(f"  n={n_facts}: {cc.accuracy:.1%}")

    model.set_gate_mode(True)

    # Summary table
    print(f"\n{'=' * 65}")
    print("PHASE 5 SUMMARY: Scale")
    print(f"{'=' * 65}")

    conds = list(results.keys())
    header = f"  {'n':>3} │"
    for c in conds:
        header += f" {c:>12}"
    print(header)
    print(f"  {'─' * 3}─┼─" + "─" * (13 * len(conds)))

    for n_facts in n_facts_list:
        row = f"  {n_facts:3d} │"
        for c in conds:
            acc = results[c].get(n_facts, 0)
            row += f" {acc:>11.1%}"
        print(row)

    return results


# ── Phase 6: Failure Mode Analysis ──────────────────────────────────

def phase6_diagnostics(model, tokenizer, fact_types, linking_ids,
                       n_facts, device):
    """Diagnose failure modes if accuracy is low."""
    print(f"\n{'=' * 65}")
    print("PHASE 6: Failure Mode Analysis")
    print(f"{'=' * 65}")

    trace = model.trace
    wte = model.gpt2.transformer.wte

    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)

    # 1. Trace norm after write
    print(f"\n  1. Trace norm diagnostic:")
    for filler_mode, n_filler in [("none", 0), ("noisy", 5)]:
        episodes = make_paragraph_episodes(
            n_episodes=5, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode=filler_mode, n_filler=n_filler,
            write_mode="single",
            seed=99)

        norms = []
        for ep in episodes:
            model.reset_traces()
            model.set_trace_mode(use=False, update=True)
            device_t = next(model.parameters()).device
            for train_seq in ep.train_sequences:
                inp = torch.tensor([train_seq], dtype=torch.long,
                                   device=device_t)
                with torch.no_grad():
                    _ = model(inp)
            # Read trace norm
            tv = trace.value_traces
            norm = tv.norm().item()
            norms.append(norm)

        avg_norm = sum(norms) / len(norms)
        fill_str = f"filler={filler_mode}" + (
            f"({n_filler})" if n_filler else "")
        print(f"    {fill_str}: avg ||T_v|| = {avg_norm:.2f}")

    # 2. Normalization diagnostic
    print(f"\n  2. Normalization diagnostic:")
    for write_mode in ["cumulative", "single"]:
        episodes = make_paragraph_episodes(
            n_episodes=3, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode="none", n_filler=0,
            write_mode=write_mode,
            seed=99)
        ep = episodes[0]
        for i, seq in enumerate(ep.train_sequences):
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            gate = trace.compute_gate(wte, seq_tensor)
            gate_sum = gate.sum().item()
            H = trace.n_heads
            denom = max(gate_sum * H, 1)
            if write_mode == "single" or i == len(ep.train_sequences) - 1:
                print(f"    {write_mode} (seq {i}): "
                      f"len={len(seq)}, gate_sum={gate_sum:.1f}, "
                      f"denom={denom:.0f}")

    # 3. Q overlap: fact concepts vs filler concepts
    print(f"\n  3. Q cosine similarity diagnostic:")
    fact_words = set()
    for ft in fact_types:
        tmpl = ft.fact_templates[0]
        # Extract concept word (word before linking word)
        text = tmpl.text.replace("{X}", ft.entities[0][0])
        words = text.split()
        link_word = tmpl.linking_word
        if link_word in words:
            link_pos = words.index(link_word)
            if link_pos > 0:
                fact_words.add(words[link_pos - 1])

    filler_words = set()
    for f in FILLER_WITH_LINK:
        words = f.split()
        for lw in ["is", "in", "at", "from"]:
            if lw in words:
                pos = words.index(lw)
                if pos > 0:
                    filler_words.add(words[pos - 1])

    print(f"    Fact concept words: {fact_words}")
    print(f"    Filler concept words: {filler_words}")

    # Compute Q for each word
    all_words = list(fact_words | filler_words)
    if len(all_words) >= 2:
        q_vecs = []
        for word in all_words:
            ids = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(ids) == 1:
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                Q, _ = trace.compute_qv(wte, inp)
                # Average over heads: (1, H, 1, d) -> (d,)
                q = Q[0, :, 0, :].mean(dim=0)
                q_vecs.append((word, q))

        if len(q_vecs) >= 2:
            print(f"\n    Cross-word Q cosine similarities:")
            for i in range(len(q_vecs)):
                for j in range(i + 1, len(q_vecs)):
                    w1, q1 = q_vecs[i]
                    w2, q2 = q_vecs[j]
                    cos = torch.nn.functional.cosine_similarity(
                        q1.unsqueeze(0), q2.unsqueeze(0)).item()
                    cat1 = "F" if w1 in fact_words else "f"
                    cat2 = "F" if w2 in fact_words else "f"
                    print(f"      {cat1}:{w1:>10s} vs "
                          f"{cat2}:{w2:>10s}  cos={cos:.3f}")


# ── Phase 0: Filler BPE Validation ─────────────────────────────────

def validate_fillers(tokenizer, linking_ids):
    """Verify filler sentences tokenize correctly."""
    print(f"\n  Filler BPE validation:")
    linking_set = set(linking_ids)

    for name, pool, should_have_link in [
        ("FILLER_NO_LINK", FILLER_NO_LINK, False),
        ("FILLER_WITH_LINK", FILLER_WITH_LINK, True),
    ]:
        n_ok = 0
        n_bad = 0
        for sent in pool:
            ids = tokenizer.encode(sent, add_special_tokens=False)
            has_link = any(tid in linking_set for tid in ids)
            if has_link == should_have_link:
                n_ok += 1
            else:
                n_bad += 1
                tokens = [tokenizer.decode([t]) for t in ids]
                print(f"    WARNING: \"{sent}\" -> {tokens}")
                print(f"      has_link={has_link}, expected={should_have_link}")

        status = "OK" if n_bad == 0 else f"{n_bad} WARNINGS"
        print(f"    {name}: {n_ok}/{len(pool)} correct ({status})")


# ── Main experiment runners ─────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick smoke test (~5-8 min)."""
    print("=" * 65)
    print("EXPERIMENT 10: Paragraph-Level Storage (quick)")
    print("=" * 65)

    dev = get_device(device)
    print(f"Device: {dev}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    # Validate fillers
    validate_fillers(tokenizer, linking_ids)

    # Phase 1: Setup & Train gate
    model = setup_and_train(dev, tokenizer, fact_types, linking_ids,
                            n_steps=1400, seed=seed)

    # Phase 2: Paragraph format (quick)
    phase2_paragraph_format(
        model, tokenizer, fact_types, linking_ids,
        n_eval=20, n_facts_list=[3, 5, 7],
        device=dev)

    # Phase 3: Noise resistance (THE CENTRAL TEST)
    phase3_noise_resistance(
        model, tokenizer, fact_types, linking_ids,
        n_eval=20, n_facts=5, device=dev)

    # Phase 4: Gate visualization
    phase4_gate_visualization(
        model, tokenizer, fact_types, linking_ids,
        device=dev)

    print(f"\n{'=' * 65}")
    print("QUICK TEST COMPLETE")
    print(f"{'=' * 65}")


def run(device=None, seed=42, n_eval=100, n_steps=2000):
    """Full experiment (~20-30 min)."""
    print("=" * 65)
    print("EXPERIMENT 10: Paragraph-Level Storage")
    print("=" * 65)

    dev = get_device(device)
    print(f"Device: {dev}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    # Validate fillers
    validate_fillers(tokenizer, linking_ids)

    # Phase 1: Setup & Train gate
    model = setup_and_train(dev, tokenizer, fact_types, linking_ids,
                            n_steps=n_steps, seed=seed)

    # Phase 2: Paragraph format
    phase2_paragraph_format(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, n_facts_list=[3, 5, 7, 10],
        device=dev)

    # Phase 3: Noise resistance (THE CENTRAL TEST)
    phase3_noise_resistance(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, n_facts=5, device=dev)

    # Phase 4: Gate visualization
    phase4_gate_visualization(
        model, tokenizer, fact_types, linking_ids,
        device=dev)

    # Phase 5: Scale test
    phase5_scale_test(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, device=dev)

    # Phase 6: Diagnostics
    phase6_diagnostics(
        model, tokenizer, fact_types, linking_ids,
        n_facts=5, device=dev)

    # Final summary
    print(f"\n{'=' * 65}")
    print("EXPERIMENT 10 COMPLETE")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 10: Paragraph-Level Storage")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (20 eval episodes)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=2000,
                        help="Gate training steps")

    args = parser.parse_args()

    if args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run(device=args.device, seed=args.seed, n_eval=args.n_eval,
            n_steps=args.n_steps)
