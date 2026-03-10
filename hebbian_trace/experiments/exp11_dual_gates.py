"""Experiment 11: Dual Gates — Semantic-Level Storage Gating.

Exp 10 proved the single gate (token-level) has a fundamental 44pp gap with
noisy filler. The gate fires equally on " is" in facts and filler — it cannot
distinguish "My name is Elena" from "The weather is nice".

Solution: dual gates.
  gate_pos:  WHERE — fires on linking tokens (trained in exp9/10)
  gate_key:  IF — evaluates concept word quality (NEW, trained here)

gate_key learns from noisy paragraphs: storing filler concepts causes
interference → higher loss → gradient pushes gate_key DOWN on filler,
UP on fact concepts. This is semantic-level gating.

Phase 1: Setup & Train gate_pos (reuse exp9 pipeline)
Phase 2: Train gate_key (noisy paragraphs, gate_pos frozen)
Phase 3: Test on exp10 conditions (THE COMPARISON)
Phase 4: Gate Visualization on Paragraphs
Phase 5: Ablation (gate_pos only vs dual)
Phase 6: Scale Stress Test

Usage:
    python -m hebbian_trace.experiments.exp11_dual_gates --quick
    python -m hebbian_trace.experiments.exp11_dual_gates
"""

import argparse
import time

import torch
import torch.nn.functional as F
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


# ── Train gate_key (THE CORE INNOVATION) ──────────────────────────

def train_gate_key(model, tokenizer, fact_types, n_steps, n_facts_train,
                   n_filler, filler_mode, lr, device, log_every=50,
                   seed=42, grad_clip=1.0, l1_lambda=0.0):
    """Train W_gate_key on paragraphs with gate_pos frozen.

    Training loop:
    1. Generate paragraph (facts + optional filler)
    2. Compute Q, V, gate_pos (frozen), gate_key (trainable)
    3. write_dual_differentiable → T_v_diff (gradient through gate_key)
    4. For each question: read_from_trace(T_v_diff) → logits → CE loss
    5. Backprop → W_gate_key only

    Args:
        model: GPT2WithTrace with gate_pos already trained
        n_steps: training steps
        n_facts_train: facts per paragraph
        n_filler: filler sentences per paragraph
        filler_mode: "none"|"safe"|"noisy"|"mixed"
        lr: learning rate
        log_every: print stats every N steps
        seed: random seed base
        grad_clip: max gradient norm
        l1_lambda: L1 sparsity penalty on gate_key
    """
    trace = model.trace
    wte = model.gpt2.transformer.wte
    wte_weight = wte.weight.detach()

    # Freeze everything except W_gate_key
    for p in trace.parameters():
        p.requires_grad_(False)
    trace.W_gate_key.weight.requires_grad_(True)
    trace.W_gate_key.bias.requires_grad_(True)
    params = [trace.W_gate_key.weight, trace.W_gate_key.bias]

    optimizer = torch.optim.Adam(params, lr=lr)

    all_entity_ids = _get_all_entity_ids(fact_types)
    entity_tensor = torch.tensor(all_entity_ids, dtype=torch.long,
                                 device=device)

    losses = []
    accuracies = []

    n_trainable = sum(p.numel() for p in params)
    l1_str = f", L1={l1_lambda}" if l1_lambda > 0 else ""
    filler_str = f", filler={filler_mode}({n_filler})" if n_filler > 0 else ""
    print(f"\n  Training W_gate_key: {n_steps} steps, n_facts={n_facts_train}"
          f"{filler_str}, lr={lr}{l1_str}")
    print(f"  Trainable params: {n_trainable:,} "
          f"(W_gate_key: {trace.W_gate_key.weight.numel() + 1})")
    print(f"  Entity pool: {len(all_entity_ids)} tokens")

    t0 = time.time()

    for step in range(n_steps):
        # Generate paragraph episode (different seed each step)
        episodes = make_paragraph_episodes(
            n_episodes=1, n_facts=n_facts_train,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode=filler_mode, n_filler=n_filler,
            write_mode="single",
            seed=step * 7919 + seed)
        episode = episodes[0]

        optimizer.zero_grad()

        # Get the paragraph token IDs
        para_ids = episode.train_sequences[0]
        para_tensor = torch.tensor(
            [para_ids], dtype=torch.long, device=device)

        # Compute Q, V, both gates
        Q, V = trace.compute_qv(wte, para_tensor)
        gate_pos = trace.compute_gate(wte, para_tensor).detach()  # frozen
        gate_key = trace.compute_gate_key(wte, para_tensor)       # trainable

        # Differentiable write with dual gates
        T_v_diff = trace.write_dual_differentiable(Q, V, gate_pos, gate_key)

        # Read from T_v_diff for each question → loss
        total_loss = torch.tensor(0.0, device=device)
        n_correct = 0

        for q_ids, answer_id, _ in episode.test_queries:
            q_tensor = torch.tensor(
                [q_ids], dtype=torch.long, device=device)
            Q_q, _ = trace.compute_qv(wte, q_tensor)

            # Read from differentiable trace
            retrieved = trace.read_from_trace(Q_q, T_v_diff)

            # Project to logit space
            trace_logits = torch.matmul(retrieved, wte_weight.T)

            # Entity-restricted CE loss at last position
            pred = trace_logits[0, -1, entity_tensor]
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]

            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            total_loss = total_loss + loss

            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

        avg_loss = total_loss / max(len(episode.test_queries), 1)

        # L1 sparsity on gate_key
        if l1_lambda > 0:
            avg_loss = avg_loss + l1_lambda * gate_key.mean()

        avg_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        optimizer.step()

        losses.append(avg_loss.item())
        accuracies.append(n_correct / max(len(episode.test_queries), 1))

        if (step + 1) % log_every == 0:
            recent_loss = sum(losses[-log_every:]) / log_every
            recent_acc = sum(accuracies[-log_every:]) / log_every
            elapsed = time.time() - t0

            # Show gate_key stats
            with torch.no_grad():
                gk_vals = gate_key[0].cpu()
                gk_mean = gk_vals.mean().item()
                gk_max = gk_vals.max().item()
                gk_min = gk_vals.min().item()

            print(f"  Step {step+1:4d}/{n_steps}: "
                  f"loss={recent_loss:.4f}, "
                  f"acc={recent_acc:.1%}, "
                  f"gk=[{gk_min:.3f},{gk_mean:.3f},{gk_max:.3f}], "
                  f"time={elapsed:.0f}s")

    total_time = time.time() - t0
    final_loss = sum(losses[-min(50, len(losses)):]) / min(50, len(losses))
    final_acc = sum(accuracies[-min(50, len(accuracies)):]) / min(50, len(accuracies))
    print(f"\n  Training complete: {total_time:.1f}s")
    print(f"  Final loss: {final_loss:.4f}, Final acc: {final_acc:.1%}")

    return losses, accuracies


# ── Phase 1: Setup & Train gate_pos ───────────────────────────────

def setup_and_train(device, tokenizer, fact_types, linking_ids,
                    n_steps_pos=1400, n_steps_key=1200, seed=42):
    """Load GPT-2 + trace, train gate_pos then gate_key.

    Returns trained model with both gates ready.
    """
    print(f"\n{'=' * 65}")
    print("PHASE 1: Setup & Train gate_pos")
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

    # ── Train gate_pos: 5-stage curriculum (identical to exp9/10) ──
    print(f"\n  Training gate_pos (exp9 pipeline, {n_steps_pos} total steps)...")
    n_per = n_steps_pos // 5

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

    # Sanity check: reproduce exp9 at n=3
    print(f"\n  Sanity check: gate_pos at n=3...")
    for p in model.trace.parameters():
        p.requires_grad_(True)
    episodes = make_gpt2_eval_episodes(
        n_episodes=20, n_facts=3,
        tokenizer=tokenizer, fact_types=fact_types,
        seed=42 + 3000)
    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)
    cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
    print(f"  gate_pos n=3: {cc.accuracy:.1%} ({cc.n_correct}/{cc.n_total})")

    return model


# ── Phase 2: Train gate_key ───────────────────────────────────────

def phase2_train_gate_key(model, tokenizer, fact_types, linking_ids,
                          n_steps_key, device, seed=42):
    """Train W_gate_key with facts-first curriculum.

    Stage 1-2: pure facts (gate_key learns to OPEN on concepts)
    Stage 3-4: noisy filler (gate_key learns to CLOSE on filler)
    """
    print(f"\n{'=' * 65}")
    print("PHASE 2: Train gate_key (THE CORE INNOVATION)")
    print(f"  Facts-first, then noise. gate_pos frozen.")
    print(f"{'=' * 65}")

    n_per = n_steps_key // 4

    # Stage 1: pure facts, n=3 — learn to OPEN
    train_gate_key(model, tokenizer, fact_types,
                   n_steps=max(n_per, 200), n_facts_train=3,
                   n_filler=0, filler_mode="none",
                   lr=3e-3, device=device,
                   log_every=100, seed=seed + 30000,
                   grad_clip=1.0)

    # Stage 2: pure facts, n=5 — harder, learn more concepts
    train_gate_key(model, tokenizer, fact_types,
                   n_steps=max(n_per, 200), n_facts_train=5,
                   n_filler=0, filler_mode="none",
                   lr=1e-3, device=device,
                   log_every=100, seed=seed + 35000,
                   grad_clip=1.0)

    # Stage 3: noisy filler — learn to CLOSE on filler concepts
    train_gate_key(model, tokenizer, fact_types,
                   n_steps=max(n_per, 200), n_facts_train=5,
                   n_filler=3, filler_mode="noisy",
                   lr=3e-4, device=device,
                   log_every=100, seed=seed + 40000,
                   grad_clip=1.0)

    # Stage 4: mixed filler + L1 sparsity
    train_gate_key(model, tokenizer, fact_types,
                   n_steps=max(n_per, 200), n_facts_train=5,
                   n_filler=5, filler_mode="mixed",
                   lr=1e-4, device=device,
                   log_every=100, seed=seed + 45000,
                   grad_clip=1.0, l1_lambda=0.5)

    # Restore requires_grad for eval
    for p in model.trace.parameters():
        p.requires_grad_(True)

    # Quick diagnostic: gate_key values on concept words
    print(f"\n  gate_key diagnostic on concept words:")
    wte = model.gpt2.transformer.wte
    concept_words = set()
    for ft in fact_types:
        tmpl = ft.fact_templates[0]
        text = tmpl.text.replace("{X}", ft.entities[0][0])
        words = text.split()
        lw = tmpl.linking_word
        if lw in words:
            idx = words.index(lw)
            if idx > 0:
                concept_words.add(words[idx - 1])

    filler_concept_words = set()
    for f in FILLER_WITH_LINK:
        words = f.split()
        for lw in ["is", "in", "at", "from"]:
            if lw in words:
                idx = words.index(lw)
                if idx > 0:
                    filler_concept_words.add(words[idx - 1])

    with torch.no_grad():
        for label, word_set in [("FACT concepts", concept_words),
                                ("FILLER concepts", filler_concept_words)]:
            vals = []
            for word in sorted(word_set):
                ids = tokenizer.encode(" " + word, add_special_tokens=False)
                if len(ids) == 1:
                    inp = torch.tensor([ids], dtype=torch.long, device=device)
                    gk = model.trace.compute_gate_key(wte, inp)
                    vals.append((word, gk[0, 0].item()))
            if vals:
                avg = sum(v for _, v in vals) / len(vals)
                items = ", ".join(f"{w}={v:.3f}" for w, v in vals)
                print(f"    {label} (avg={avg:.3f}): {items}")


# ── Phase 3: Noise Resistance (THE COMPARISON) ───────────────────

def phase3_noise_resistance(model, tokenizer, fact_types, linking_ids,
                            n_eval, n_facts, device):
    """Compare dual gate vs single gate vs hardcoded on exp10 conditions."""
    print(f"\n{'=' * 65}")
    print("PHASE 3: Noise Resistance — Dual vs Single Gate")
    print(f"  n_facts={n_facts}, single-pass write")
    print(f"{'=' * 65}")

    conditions = [
        ("no_filler",  "none",  0),
        ("safe_2",     "safe",  2),
        ("safe_5",     "safe",  5),
        ("noisy_2",    "noisy", 2),
        ("noisy_5",    "noisy", 5),
        ("mixed_3",    "mixed", 3),
    ]

    base_seed = 42 + n_facts * 1000
    results = {}  # {mode_cond: GPT2EvalResults}

    # ── Dual gate ──
    model.set_dual_gate_mode(True)
    model.set_linking_token_ids(linking_ids)
    print(f"\n--- Dual gate (gate_pos + gate_key) ---")

    for cond_name, filler_mode, n_filler in conditions:
        key = f"dual_{cond_name}"
        episodes = make_paragraph_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode=filler_mode, n_filler=n_filler,
            write_mode="single", seed=base_seed)

        cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
        results[key] = cc
        print(f"  {cond_name}: {cc.accuracy:.1%} ({cc.n_correct}/{cc.n_total})")

    # ── Single gate (gate_pos only, exp10 baseline) ──
    model.set_dual_gate_mode(False)
    model.set_gate_mode(True)
    model.set_linking_token_ids(linking_ids)
    print(f"\n--- Single gate (gate_pos only) ---")

    for cond_name, filler_mode, n_filler in conditions:
        key = f"single_{cond_name}"
        episodes = make_paragraph_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode=filler_mode, n_filler=n_filler,
            write_mode="single", seed=base_seed)

        cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
        results[key] = cc
        print(f"  {cond_name}: {cc.accuracy:.1%} ({cc.n_correct}/{cc.n_total})")

    # ── Dual gate with tau sweep on gate_key ──
    model.set_dual_gate_mode(True)
    model.set_linking_token_ids(linking_ids)

    for tau in [0.5, 0.3]:
        model.trace.set_gate_key_tau(tau)
        print(f"\n--- Dual gate (gate_key tau={tau}) ---")

        for cond_name, filler_mode, n_filler in conditions:
            key = f"dual_tau{tau}_{cond_name}"
            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode=filler_mode, n_filler=n_filler,
                write_mode="single", seed=base_seed)

            cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
            results[key] = cc
            print(f"  {cond_name}: {cc.accuracy:.1%} ({cc.n_correct}/{cc.n_total})")

    # Restore tau=1.0
    model.trace.set_gate_key_tau(1.0)

    # ── Hardcoded mask ──
    model.set_gate_mode(False)
    model.set_linking_token_ids(linking_ids)
    print(f"\n--- Hardcoded mask ---")

    for cond_name, filler_mode, n_filler in [
        ("no_filler", "none", 0), ("noisy_5", "noisy", 5)]:
        key = f"hm_{cond_name}"
        episodes = make_paragraph_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            filler_mode=filler_mode, n_filler=n_filler,
            write_mode="single", seed=base_seed)

        cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
        results[key] = cc
        print(f"  {cond_name}: {cc.accuracy:.1%}")

    # ── Summary table ──
    print(f"\n{'=' * 65}")
    print(f"PHASE 3 SUMMARY: Noise Resistance (n={n_facts})")
    print(f"{'=' * 65}")

    base_conds = ["no_filler", "safe_2", "safe_5",
                  "noisy_2", "noisy_5", "mixed_3"]

    print(f"\n  {'Condition':>12} │ {'Dual τ=1':>9} │ {'Dual τ=.5':>10} │ {'Dual τ=.3':>10} │ {'Single':>8} │ {'Impr(τ=1)':>10}")
    print(f"  {'─' * 12}─┼─{'─' * 9}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 10}")

    for cn in base_conds:
        dual_k = f"dual_{cn}"
        dual_t5 = f"dual_tau0.5_{cn}"
        dual_t3 = f"dual_tau0.3_{cn}"
        single_k = f"single_{cn}"

        parts = [f"  {cn:>12} │"]

        # Dual tau=1
        if dual_k in results:
            parts.append(f" {results[dual_k].accuracy:>8.1%} │")
        else:
            parts.append(f" {'—':>8} │")

        # Dual tau=0.5
        if dual_t5 in results:
            parts.append(f" {results[dual_t5].accuracy:>9.1%} │")
        else:
            parts.append(f" {'—':>9} │")

        # Dual tau=0.3
        if dual_t3 in results:
            parts.append(f" {results[dual_t3].accuracy:>9.1%} │")
        else:
            parts.append(f" {'—':>9} │")

        # Single
        if single_k in results:
            parts.append(f" {results[single_k].accuracy:>7.1%} │")
        else:
            parts.append(f" {'—':>7} │")

        # Improvement (dual tau=1 vs single)
        if dual_k in results and single_k in results:
            imp = results[dual_k].accuracy - results[single_k].accuracy
            parts.append(f" {imp:>+9.1%}")
        else:
            parts.append(f" {'—':>9}")

        print("".join(parts))

    # Hardcoded reference
    print(f"\n  Hardcoded mask reference:")
    for cn in ["no_filler", "noisy_5"]:
        hm_k = f"hm_{cn}"
        if hm_k in results:
            print(f"    hm_{cn}: {results[hm_k].accuracy:.1%}")

    # Decision
    dual_nf = results.get("dual_no_filler")
    dual_n5 = results.get("dual_noisy_5")
    single_n5 = results.get("single_noisy_5")
    if dual_nf and dual_n5:
        gap = dual_nf.accuracy - dual_n5.accuracy
        print(f"\n  KEY RESULT: dual gate noisy_5 gap = {gap:.1%}")
        if gap < 0.10:
            print(f"  >> Gap < 10pp: dual gates SOLVE the problem!")
        elif gap < 0.20:
            print(f"  >> Gap 10-20pp: dual gates HELP significantly")
        else:
            print(f"  >> Gap > 20pp: dual gates not sufficient alone")

    if single_n5 and dual_n5:
        improvement = dual_n5.accuracy - single_n5.accuracy
        print(f"\n  IMPROVEMENT: dual vs single on noisy_5: {improvement:+.1%}")
        print(f"  (exp10 single gate noisy_5 was ~17%)")

    return results


# ── Phase 4: Gate Visualization ───────────────────────────────────

def phase4_gate_visualization(model, tokenizer, fact_types, linking_ids,
                              device):
    """Visualize both gates on a paragraph with facts + noisy filler."""
    print(f"\n{'=' * 65}")
    print("PHASE 4: Dual Gate Visualization on Paragraphs")
    print(f"{'=' * 65}")

    trace = model.trace
    wte = model.gpt2.transformer.wte
    linking_set = set(linking_ids)

    # Build paragraph: fact0, filler0(noisy), fact1, filler1(safe), fact2
    ft_name = fact_types[0]
    ft_city = fact_types[1] if len(fact_types) > 1 else fact_types[0]
    ft_comp = fact_types[2] if len(fact_types) > 2 else fact_types[0]

    facts_text = [
        ft_name.fact_templates[0].text.replace("{X}", ft_name.entities[0][0]),
        ft_city.fact_templates[0].text.replace("{X}", ft_city.entities[0][0]),
        ft_comp.fact_templates[0].text.replace("{X}", ft_comp.entities[0][0]),
    ]
    fillers = [FILLER_WITH_LINK[0], FILLER_NO_LINK[0]]

    # Interleave
    parts = []
    is_fact_part = []
    for i, ft in enumerate(facts_text):
        parts.append(ft)
        is_fact_part.append(True)
        if i < len(fillers):
            parts.append(fillers[i])
            is_fact_part.append(False)
    paragraph = " ".join(parts)

    print(f"\n  Paragraph: \"{paragraph}\"")

    para_ids = tokenizer.encode(paragraph, add_special_tokens=False)
    para_tensor = torch.tensor([para_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        gate_pos = trace.compute_gate(wte, para_tensor)
        gate_key = trace.compute_gate_key(wte, para_tensor)

    gp_vals = gate_pos[0].cpu().tolist()
    gk_vals = gate_key[0].cpu().tolist()
    tokens = [tokenizer.decode([tid]) for tid in para_ids]

    # Build position → part mapping by tokenizing each part separately
    part_token_counts = []
    for part_text in parts:
        part_ids = tokenizer.encode(part_text, add_special_tokens=False)
        part_token_counts.append(len(part_ids))

    # Map position to part index (accounting for space tokens between parts)
    pos_to_part = {}
    cur_pos = 0
    for p_idx, n_toks in enumerate(part_token_counts):
        for j in range(n_toks):
            pos_to_part[cur_pos + j] = p_idx
        cur_pos += n_toks
        # Space token between parts (tokenizer adds 1 space token)
        if p_idx < len(part_token_counts) - 1:
            # The space token is part of the next part's tokenization
            # (GPT-2 BPE encodes " word" as single token)
            pass

    # Print table
    print(f"\n  {'Pos':>4} {'Token':>14} {'gate_pos':>9} {'gate_key':>9} {'combined':>9} {'Type':>14}")
    print(f"  {'─' * 4} {'─' * 14} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 14}")

    link_fact_combined = []
    link_filler_combined = []
    nonlink_combined = []

    # Simpler approach: track position in paragraph
    # Re-tokenize parts to get exact boundaries
    boundaries = []
    full_tokens = []
    for p_idx, part_text in enumerate(parts):
        prefix = " " if full_tokens else ""
        part_ids = tokenizer.encode(prefix + part_text, add_special_tokens=False)
        start = len(full_tokens)
        full_tokens.extend(part_ids)
        boundaries.append((start, start + len(part_ids), is_fact_part[p_idx]))

    for i in range(len(para_ids)):
        tok = tokens[i]
        gp = gp_vals[i]
        gk = gk_vals[i]
        combined = gp * gk
        is_link = para_ids[i] in linking_set

        # Determine if in fact or filler
        in_fact = False
        in_filler = False
        for start, end, is_f in boundaries:
            if start <= i < end:
                in_fact = is_f
                in_filler = not is_f
                break

        if is_link and in_fact:
            cat = "LINK-FACT"
            link_fact_combined.append(combined)
        elif is_link and in_filler:
            cat = "LINK-FILLER"
            link_filler_combined.append(combined)
        elif is_link:
            cat = "LINK-?"
        else:
            cat = ""
            nonlink_combined.append(combined)

        # Visual bar for combined gate
        bar_len = int(combined * 30)
        bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
        print(f"  [{i:3d}] {tok:>12s}  {gp:.3f}    {gk:.3f}    {combined:.4f} {bar} {cat}")

    # Category averages
    print(f"\n  Category averages (combined = gate_pos * gate_key):")
    if link_fact_combined:
        avg = sum(link_fact_combined) / len(link_fact_combined)
        print(f"    LINK-FACT:   {avg:.4f} (n={len(link_fact_combined)})")
    if link_filler_combined:
        avg = sum(link_filler_combined) / len(link_filler_combined)
        print(f"    LINK-FILLER: {avg:.4f} (n={len(link_filler_combined)})")
    if nonlink_combined:
        avg = sum(nonlink_combined) / len(nonlink_combined)
        print(f"    NON-LINK:    {avg:.4f} (n={len(nonlink_combined)})")

    if link_fact_combined and link_filler_combined:
        fact_avg = sum(link_fact_combined) / len(link_fact_combined)
        filler_avg = sum(link_filler_combined) / len(link_filler_combined)
        ratio = fact_avg / max(filler_avg, 1e-8)
        print(f"\n  SELECTIVITY: LINK-FACT / LINK-FILLER = {ratio:.1f}x")
        print(f"  (exp10 single gate: ~1.0x — no selectivity)")
        if ratio > 5:
            print(f"  >> Excellent semantic selectivity!")
        elif ratio > 2:
            print(f"  >> Good semantic selectivity")
        else:
            print(f"  >> Limited selectivity — gate_key may need more training")


# ── Phase 5: Ablation ─────────────────────────────────────────────

def phase5_ablation(model, tokenizer, fact_types, linking_ids,
                    n_eval, n_facts, device):
    """Compare gate configurations on noisy_5."""
    print(f"\n{'=' * 65}")
    print("PHASE 5: Ablation — Gate Configurations")
    print(f"{'=' * 65}")

    base_seed = 42 + n_facts * 1000
    results = {}

    configs = [
        ("dual_gate",    "dual"),     # gate_pos + gate_key
        ("gate_key_only", "key_only"), # gate_key only (no gate_pos)
        ("single_gate",  "single"),   # gate_pos only (exp10)
        ("hardcoded",    "hardcoded"), # hardcoded linking mask
    ]

    trace = model.trace

    for config_name, mode in configs:
        if mode == "dual":
            model.set_dual_gate_mode(True)
        elif mode == "key_only":
            # gate_key only: set gate_pos to ~1.0 everywhere by
            # temporarily overriding W_gate to produce large positive logits
            model.set_dual_gate_mode(True)
            saved_w = trace.W_gate.weight.data.clone()
            saved_b = trace.W_gate.bias.data.clone()
            trace.W_gate.weight.data.zero_()
            trace.W_gate.bias.data.fill_(10.0)  # sigmoid(10)≈0.99995
        elif mode == "single":
            trace._use_dual_gate = False
            model.set_gate_mode(True)
        elif mode == "hardcoded":
            trace._use_dual_gate = False
            model.set_gate_mode(False)
        model.set_linking_token_ids(linking_ids)

        print(f"\n--- {config_name} ---")
        for filler_mode, n_filler, label in [
            ("none", 0, "no_filler"),
            ("noisy", 5, "noisy_5"),
        ]:
            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode=filler_mode, n_filler=n_filler,
                write_mode="single", seed=base_seed)

            cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
            key = f"{config_name}_{label}"
            results[key] = cc.accuracy
            print(f"  {label}: {cc.accuracy:.1%}")

        # Restore W_gate if we overrode it
        if mode == "key_only":
            trace.W_gate.weight.data.copy_(saved_w)
            trace.W_gate.bias.data.copy_(saved_b)

    # Summary
    config_names = [c[0] for c in configs]
    print(f"\n  {'Config':>14} │ {'no_filler':>10} │ {'noisy_5':>10} │ {'gap':>8}")
    print(f"  {'─' * 14}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 8}")

    for config_name in config_names:
        nf = results.get(f"{config_name}_no_filler", 0)
        n5 = results.get(f"{config_name}_noisy_5", 0)
        gap = nf - n5
        print(f"  {config_name:>14} │ {nf:>9.1%} │ {n5:>9.1%} │ {gap:>7.1%}")

    # Analysis
    dual_n5 = results.get("dual_gate_noisy_5", 0)
    key_n5 = results.get("gate_key_only_noisy_5", 0)
    single_n5 = results.get("single_gate_noisy_5", 0)
    if dual_n5 and key_n5:
        if abs(dual_n5 - key_n5) < 0.05:
            print(f"\n  >> gate_key_only ≈ dual_gate on noisy_5 — gate_pos may be redundant!")
        elif key_n5 > single_n5:
            print(f"\n  >> gate_key_only > single_gate: concept gating helps, but gate_pos adds value")
        else:
            print(f"\n  >> gate_key_only ≤ single_gate: gate_key needs gate_pos for position info")

    return results


# ── Phase 6: Scale Test ───────────────────────────────────────────

def phase6_scale_test(model, tokenizer, fact_types, linking_ids,
                      n_eval, device):
    """Scale test: dual gate vs single gate at various n_facts."""
    print(f"\n{'=' * 65}")
    print("PHASE 6: Scale Stress Test")
    print(f"{'=' * 65}")

    n_facts_list = [3, 5, 7, 10]
    results = {}

    for mode_name, setup_fn in [
        ("dual_gate", lambda: model.set_dual_gate_mode(True)),
        ("single_gate", lambda: (
            setattr(model.trace, '_use_dual_gate', False),
            model.set_gate_mode(True))),
        ("hardcoded", lambda: (
            setattr(model.trace, '_use_dual_gate', False),
            model.set_gate_mode(False))),
    ]:
        setup_fn()
        model.set_linking_token_ids(linking_ids)
        results[mode_name] = {}
        print(f"\n--- {mode_name} (mixed filler, 1:1 ratio) ---")

        for n_facts in n_facts_list:
            n_filler = n_facts  # 1:1 ratio
            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode="mixed", n_filler=n_filler,
                write_mode="single",
                seed=42 + n_facts * 2000)

            if episodes:
                seq_len = len(episodes[0].train_sequences[0])
                print(f"  n={n_facts} ({seq_len} tokens): ", end="")

            cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
            results[mode_name][n_facts] = cc.accuracy
            print(f"{cc.accuracy:.1%}")

    # Summary
    mode_names = list(results.keys())
    print(f"\n  {'n':>3} │ {'dual_gate':>10} │ {'single_gate':>12} │ {'hardcoded':>10} │ {'dual-single':>12}")
    print(f"  {'─' * 3}─┼─{'─' * 10}─┼─{'─' * 12}─┼─{'─' * 10}─┼─{'─' * 12}")
    for n in n_facts_list:
        d = results.get("dual_gate", {}).get(n, 0)
        s = results.get("single_gate", {}).get(n, 0)
        h = results.get("hardcoded", {}).get(n, 0)
        imp = d - s
        print(f"  {n:3d} │ {d:>9.1%} │ {s:>11.1%} │ {h:>9.1%} │ {imp:>+10.1%}")

    return results


# ── Phase 7: Pattern Separation + Dual Gates ─────────────────────

def phase7_pattern_separation(model, tokenizer, fact_types, linking_ids,
                              n_eval, device):
    """Test dual gate + pattern separation composition.

    PS reduces Q overlap for stored items (orthogonal to gate filtering).
    Historically +9-12pp on GPT-2 and MiniGPT without dual gates.
    """
    print(f"\n{'=' * 65}")
    print("PHASE 7: Pattern Separation + Dual Gates (composition test)")
    print(f"{'=' * 65}")

    # ── Part A: Noise resistance at n=5 (compare with Phase 3) ──
    print(f"\n--- Part A: Noise conditions (n=5) ---")

    conditions = [
        ("no_filler",  "none",  0),
        ("safe_5",     "safe",  5),
        ("noisy_5",    "noisy", 5),
        ("mixed_3",    "mixed", 3),
    ]
    base_seed = 42 + 5 * 1000
    results_a = {}

    for ps_label, ps_enabled in [("dual+PS", True), ("dual_only", False)]:
        if ps_enabled:
            model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
        else:
            model.disable_pattern_separation()

        model.set_dual_gate_mode(True)
        model.set_linking_token_ids(linking_ids)

        print(f"\n  {ps_label}:")
        for cond_name, filler_mode, n_filler in conditions:
            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=5,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode=filler_mode, n_filler=n_filler,
                write_mode="single", seed=base_seed)

            cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
            key = f"{ps_label}_{cond_name}"
            results_a[key] = cc.accuracy
            print(f"    {cond_name}: {cc.accuracy:.1%} "
                  f"({cc.n_correct}/{cc.n_total})")

    # Part A summary
    print(f"\n  {'Condition':>12} │ {'dual+PS':>9} │ {'dual_only':>10} │ {'Δ PS':>8}")
    print(f"  {'─' * 12}─┼─{'─' * 9}─┼─{'─' * 10}─┼─{'─' * 8}")
    for cn, _, _ in conditions:
        ps_acc = results_a.get(f"dual+PS_{cn}", 0)
        no_acc = results_a.get(f"dual_only_{cn}", 0)
        diff = ps_acc - no_acc
        print(f"  {cn:>12} │ {ps_acc:>8.1%} │ {no_acc:>9.1%} │ {diff:>+7.1%}")

    # ── Part B: Scale test with mixed filler 1:1 ──
    print(f"\n--- Part B: Scale test (mixed filler 1:1) ---")

    n_facts_list = [3, 5, 7, 10]
    results_b = {}

    for ps_label, ps_enabled in [("dual+PS", True), ("dual_only", False),
                                  ("single+PS", True)]:
        if "PS" in ps_label:
            model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
        else:
            model.disable_pattern_separation()

        if "single" in ps_label:
            model.trace._use_dual_gate = False
            model.set_gate_mode(True)
        else:
            model.set_dual_gate_mode(True)
        model.set_linking_token_ids(linking_ids)

        results_b[ps_label] = {}
        print(f"\n  {ps_label}:")
        for n_facts in n_facts_list:
            n_filler = n_facts
            episodes = make_paragraph_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                filler_mode="mixed", n_filler=n_filler,
                write_mode="single",
                seed=42 + n_facts * 2000)

            if episodes:
                seq_len = len(episodes[0].train_sequences[0])
                print(f"    n={n_facts} ({seq_len} tok): ", end="")

            cc = evaluate_gpt2_cross_context(model, episodes, fact_types)
            results_b[ps_label][n_facts] = cc.accuracy
            print(f"{cc.accuracy:.1%}")

    # Part B summary
    print(f"\n  {'n':>3} │ {'dual+PS':>9} │ {'dual_only':>10} │ {'single+PS':>10} │ {'Δ(dual+PS vs dual)':>19}")
    print(f"  {'─' * 3}─┼─{'─' * 9}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 19}")
    for n in n_facts_list:
        dps = results_b.get("dual+PS", {}).get(n, 0)
        d = results_b.get("dual_only", {}).get(n, 0)
        sps = results_b.get("single+PS", {}).get(n, 0)
        diff = dps - d
        print(f"  {n:3d} │ {dps:>8.1%} │ {d:>9.1%} │ {sps:>9.1%} │ {diff:>+18.1%}")

    # Overall assessment
    n10_dps = results_b.get("dual+PS", {}).get(10, 0)
    n10_d = results_b.get("dual_only", {}).get(10, 0)
    n10_sps = results_b.get("single+PS", {}).get(10, 0)
    ps_boost = n10_dps - n10_d

    print(f"\n  KEY: PS boost at n=10: {ps_boost:+.1%} "
          f"(dual+PS {n10_dps:.1%} vs dual {n10_d:.1%})")
    if n10_dps > n10_sps:
        print(f"  >> dual+PS > single+PS at n=10: both mechanisms needed")
    if ps_boost > 0.05:
        print(f"  >> PS composes with dual gates (+{ps_boost:.0%})")
    elif ps_boost > 0:
        print(f"  >> PS helps modestly with dual gates (+{ps_boost:.0%})")
    else:
        print(f"  >> PS does NOT help on top of dual gates")

    # Cleanup: disable PS
    model.disable_pattern_separation()

    return results_a, results_b


# ── Main experiment runners ───────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick smoke test (~8-12 min)."""
    print("=" * 65)
    print("EXPERIMENT 11: Dual Gates — Semantic-Level Gating (quick)")
    print("=" * 65)

    dev = get_device(device)
    print(f"Device: {dev}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    # Phase 1: Setup & train gate_pos
    model = setup_and_train(dev, tokenizer, fact_types, linking_ids,
                            n_steps_pos=1400, n_steps_key=0, seed=seed)

    # Phase 2: Train gate_key
    phase2_train_gate_key(model, tokenizer, fact_types, linking_ids,
                          n_steps_key=1200, device=dev, seed=seed)

    # Phase 3: Noise resistance (THE COMPARISON)
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


def run(device=None, seed=42, n_eval=100, n_steps_pos=2000,
        n_steps_key=1600):
    """Full experiment (~25-35 min)."""
    print("=" * 65)
    print("EXPERIMENT 11: Dual Gates — Semantic-Level Gating")
    print("=" * 65)

    dev = get_device(device)
    print(f"Device: {dev}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    # Phase 1: Setup & train gate_pos
    model = setup_and_train(dev, tokenizer, fact_types, linking_ids,
                            n_steps_pos=n_steps_pos, n_steps_key=0,
                            seed=seed)

    # Phase 2: Train gate_key
    phase2_train_gate_key(model, tokenizer, fact_types, linking_ids,
                          n_steps_key=n_steps_key, device=dev, seed=seed)

    # Phase 3: Noise resistance (THE COMPARISON)
    phase3_noise_resistance(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, n_facts=5, device=dev)

    # Phase 4: Gate visualization
    phase4_gate_visualization(
        model, tokenizer, fact_types, linking_ids,
        device=dev)

    # Phase 5: Ablation
    phase5_ablation(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, n_facts=5, device=dev)

    # Phase 6: Scale test
    phase6_scale_test(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, device=dev)

    # Phase 7: Pattern separation + dual gates
    phase7_pattern_separation(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, device=dev)

    print(f"\n{'=' * 65}")
    print("EXPERIMENT 11 COMPLETE")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 11: Dual Gates — Semantic-Level Gating")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (20 eval episodes)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-steps-pos", type=int, default=2000,
                        help="gate_pos training steps")
    parser.add_argument("--n-steps-key", type=int, default=1600,
                        help="gate_key training steps")

    args = parser.parse_args()

    if args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run(device=args.device, seed=args.seed, n_eval=args.n_eval,
            n_steps_pos=args.n_steps_pos, n_steps_key=args.n_steps_key)
