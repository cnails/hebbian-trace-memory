"""Experiment 9: Learned Storage Gating.

Tests whether a learned gate W_gate can replace the hardcoded linking-token
mask. Uses differentiable shadow trace T_v_diff for surrogate loss training.

Architecture:
    gate = sigmoid(W_gate(wte(token)))     — per-position storage gate
    T_v_diff = gate * Q.T @ V             — differentiable shadow trace
    loss = CE(read_from(T_v_diff) @ wte.T) — surrogate loss -> backprop -> W_gate

Key test: does W_gate learn to activate on positions before linking tokens
without any explicit supervision on gate values?

Phase 1: Setup & Diagnostics (same as exp8)
Phase 2: Train W_gate via surrogate loss
Phase 3: Evaluate (learned gate vs hardcoded mask vs store-all)
Phase 4: Gate activation analysis

Usage:
    python -m hebbian_trace.experiments.exp9_learned_gating --quick
    python -m hebbian_trace.experiments.exp9_learned_gating
"""

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    evaluate_gpt2_cross_context, evaluate_gpt2_cross_context_baseline,
    tokenize_fact, _get_all_entity_ids,
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


# ── Phase 2: Train W_gate ──────────────────────────────────────────

def train_gate(model, tokenizer, fact_types, n_steps, n_facts_train,
               lr, device, log_every=50, seed=42, gate_only=True,
               grad_clip=1.0, l1_lambda=0.0):
    """Train W_gate via surrogate loss on differentiable shadow trace.

    Training loop (trace-only, no GPT-2 forward for speed):
    1. Generate random episode with n_facts
    2. Build concatenated fact sequence
    3. Compute Q, V, gate on facts
    4. write_differentiable → T_v_diff (with gradient through gate)
    5. For each question: read_from_trace(T_v_diff) → logits → CE loss
    6. Backprop → W_gate (and optionally W_proj, W_val, W_out, ln_proj)

    Args:
        model: GPT2WithTrace (GPT-2 frozen, trace module trainable)
        n_steps: number of training episodes
        n_facts_train: facts per training episode (1 = clearest signal)
        lr: learning rate for Adam
        log_every: print stats every N steps
        seed: random seed base
        gate_only: if True, only train W_gate (freeze projections)
        grad_clip: max gradient norm (0 = no clipping)
        l1_lambda: L1 sparsity penalty on gate values (0 = disabled).
            Encourages gate to be sparse — only truly necessary positions
            stay open. Higher values = sparser gate.

    Returns:
        losses: list of per-step losses
        accuracies: list of per-step accuracies
    """
    trace = model.trace
    wte = model.gpt2.transformer.wte
    wte_weight = wte.weight.detach()  # frozen embedding matrix

    if gate_only:
        # Freeze all trace params, then unfreeze only W_gate
        for p in trace.parameters():
            p.requires_grad_(False)
        trace.W_gate.weight.requires_grad_(True)
        trace.W_gate.bias.requires_grad_(True)
        params = [trace.W_gate.weight, trace.W_gate.bias]
        mode_str = "gate_only"
    else:
        # Train all trace params
        for p in trace.parameters():
            p.requires_grad_(True)
        params = list(trace.parameters())
        mode_str = "all_params"

    optimizer = torch.optim.Adam(params, lr=lr)

    all_entity_ids = _get_all_entity_ids(fact_types)
    entity_tensor = torch.tensor(all_entity_ids, dtype=torch.long,
                                 device=device)
    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    losses = []
    accuracies = []

    n_trainable = sum(p.numel() for p in params)
    l1_str = f", L1={l1_lambda}" if l1_lambda > 0 else ""
    tau_str = f", tau={trace._gate_tau}" if trace._gate_tau != 1.0 else ""
    print(f"\n  Training W_gate: {n_steps} steps, n_facts={n_facts_train}, "
          f"lr={lr}, mode={mode_str}{l1_str}{tau_str}")
    print(f"  Trainable params: {n_trainable:,} "
          f"(W_gate: {trace.W_gate.weight.numel() + 1})")
    print(f"  Entity pool: {len(all_entity_ids)} tokens")

    t0 = time.time()

    for step in range(n_steps):
        # Generate random episode (different seed each step)
        episodes = make_gpt2_eval_episodes(
            n_episodes=1, n_facts=n_facts_train,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=step * 7919 + seed)
        episode = episodes[0]

        optimizer.zero_grad()

        # Build concatenated fact sequence
        all_fact_ids = []
        for _, _, _, fact_ids in episode.facts:
            if all_fact_ids:
                all_fact_ids.append(space_id)
            all_fact_ids.extend(fact_ids)

        fact_tensor = torch.tensor(
            [all_fact_ids], dtype=torch.long, device=device)

        # Compute Q, V, gate on facts
        Q, V = trace.compute_qv(wte, fact_tensor)
        gate = trace.compute_gate(wte, fact_tensor)

        # Differentiable write -> T_v_diff
        T_v_diff = trace.write_differentiable(Q, V, gate)

        # Read from T_v_diff for each question -> loss
        total_loss = torch.tensor(0.0, device=device)
        n_correct = 0

        for q_ids, answer_id, _ in episode.test_queries:
            q_tensor = torch.tensor(
                [q_ids], dtype=torch.long, device=device)
            Q_q, _ = trace.compute_qv(wte, q_tensor)

            # Read from differentiable trace
            retrieved = trace.read_from_trace(Q_q, T_v_diff)  # (1, S_q, d_model)

            # Project to logit space via wte
            trace_logits = torch.matmul(
                retrieved, wte_weight.T)  # (1, S_q, vocab)

            # Entity-restricted CE loss at last position
            pred = trace_logits[0, -1, entity_tensor]  # (n_entities,)
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]

            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            total_loss = total_loss + loss

            # Track accuracy
            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

        avg_loss = total_loss / max(len(episode.test_queries), 1)

        # L1 sparsity penalty: penalize gate being open
        if l1_lambda > 0:
            avg_loss = avg_loss + l1_lambda * gate.mean()

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
            print(f"  Step {step+1:4d}/{n_steps}: "
                  f"loss={recent_loss:.4f}, "
                  f"acc={recent_acc:.1%}, "
                  f"time={elapsed:.0f}s")

    total_time = time.time() - t0
    final_loss = sum(losses[-min(50, len(losses)):]) / min(50, len(losses))
    final_acc = sum(accuracies[-min(50, len(accuracies)):]) / min(50, len(accuracies))
    print(f"\n  Training complete: {total_time:.1f}s")
    print(f"  Final loss: {final_loss:.4f}, Final acc: {final_acc:.1%}")

    return losses, accuracies


# ── Phase 3: Evaluation ────────────────────────────────────────────

def evaluate_conditions(model, tokenizer, fact_types, linking_ids,
                        n_eval, n_facts_list, device, verbose=False,
                        tau_values=None):
    """Compare write conditions: learned gate (at multiple tau) vs hardcoded vs store-all.

    For each condition, runs standard cross-context eval
    (ACh modulation: write then read).

    Args:
        tau_values: list of tau values to test learned gate at.
            If None, uses [model.trace._gate_tau] (current setting).
    """
    print(f"\n{'=' * 65}")
    print("PHASE 3: Evaluation")
    print(f"{'=' * 65}")

    if tau_values is None:
        tau_values = [model.trace._gate_tau]

    results = {}  # {condition: {n_facts: GPT2EvalResults}}

    # Test learned gate at each tau
    for tau in tau_values:
        cond_name = f"gate_tau={tau}"
        print(f"\n--- Condition: {cond_name} ---")
        model.set_gate_mode(True)
        model.set_linking_token_ids(linking_ids)
        model.trace.set_gate_tau(tau)
        results[cond_name] = {}

        for n_facts in n_facts_list:
            episodes = make_gpt2_eval_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                seed=42 + n_facts * 1000)

            cc = evaluate_gpt2_cross_context(
                model, episodes, fact_types, verbose=verbose)
            results[cond_name][n_facts] = cc
            print(f"  n={n_facts}: {cc.accuracy:.1%} "
                  f"({cc.n_correct}/{cc.n_total})")

    # Hardcoded mask
    print(f"\n--- Condition: hardcoded_mask ---")
    model.set_gate_mode(False)
    model.set_linking_token_ids(linking_ids)
    results["hardcoded_mask"] = {}
    for n_facts in n_facts_list:
        episodes = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)
        cc = evaluate_gpt2_cross_context(
            model, episodes, fact_types, verbose=verbose)
        results["hardcoded_mask"][n_facts] = cc
        print(f"  n={n_facts}: {cc.accuracy:.1%} "
              f"({cc.n_correct}/{cc.n_total})")

    # Store all (no linking mask)
    print(f"\n--- Condition: store_all ---")
    model.set_gate_mode(False)
    model.set_linking_token_ids(None)
    results["store_all"] = {}
    for n_facts in n_facts_list:
        episodes = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)
        cc = evaluate_gpt2_cross_context(
            model, episodes, fact_types, verbose=verbose)
        results["store_all"][n_facts] = cc
        print(f"  n={n_facts}: {cc.accuracy:.1%} "
              f"({cc.n_correct}/{cc.n_total})")

    # No trace baseline
    print(f"\n--- Condition: no_trace (baseline) ---")
    model.set_gate_mode(False)
    model.set_linking_token_ids(linking_ids)
    results["no_trace"] = {}
    for n_facts in n_facts_list:
        episodes = make_gpt2_eval_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)
        cc_bl = evaluate_gpt2_cross_context_baseline(
            model, episodes, fact_types, verbose=verbose)
        results["no_trace"][n_facts] = cc_bl
        print(f"  n={n_facts}: {cc_bl.accuracy:.1%} "
              f"({cc_bl.n_correct}/{cc_bl.n_total})")

    # Summary table
    print(f"\n{'=' * 65}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 65}")

    cond_names = [f"gate_tau={tau}" for tau in tau_values] + \
                 ["hardcoded_mask", "store_all", "no_trace"]
    header = f"{'n':>3} │"
    for cn in cond_names:
        header += f" {cn:>15}"
    print(header)
    print(f"{'─' * 3}─┼─" + "─" * (16 * len(cond_names)))

    for n_facts in n_facts_list:
        row = f"{n_facts:3d} │"
        for cn in cond_names:
            acc = results[cn][n_facts].accuracy
            row += f" {acc:>14.1%}"
        print(row)

    # Restore tau to 1.0
    model.trace.set_gate_tau(1.0)

    return results


# ── Phase 4: Gate Analysis ──────────────────────────────────────────

def analyze_gate(model, tokenizer, fact_types, linking_ids, device,
                 tau_values=None):
    """Visualize per-token gate activations at multiple temperatures.

    Key test: does low tau sharpen gate to near-binary?
    """
    print(f"\n{'=' * 65}")
    print("PHASE 4: Gate Activation Analysis")
    print(f"{'=' * 65}")

    if tau_values is None:
        tau_values = [1.0]

    trace = model.trace
    wte = model.gpt2.transformer.wte
    linking_set = set(linking_ids)

    all_results = {}  # tau -> (avg_link, avg_other)

    for tau in tau_values:
        trace.set_gate_tau(tau)
        print(f"\n{'─' * 65}")
        print(f"  tau = {tau}")
        print(f"{'─' * 65}")

        linking_gate_vals = []
        nonlinking_gate_vals = []

        for ft in fact_types:
            entity_name, _ = ft.entities[0]
            template = ft.fact_templates[0]
            fact_text = template.text.replace("{X}", entity_name)
            fact_ids = tokenizer.encode(fact_text, add_special_tokens=False)

            fact_tensor = torch.tensor(
                [fact_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                gate = trace.compute_gate(wte, fact_tensor)  # (1, S)

            gate_vals = gate[0].cpu().tolist()
            tokens = [tokenizer.decode([tid]) for tid in fact_ids]

            print(f"\n  {ft.name}: \"{fact_text}\"")
            for i, (tok, g) in enumerate(zip(tokens, gate_vals)):
                is_link = fact_ids[i] in linking_set
                marker = " << LINK" if is_link else ""
                bar_len = int(g * 30)
                bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
                print(f"    [{i}] {tok:>14s}  gate={g:.3f} {bar}{marker}")

                if is_link:
                    linking_gate_vals.append(g)
                else:
                    nonlinking_gate_vals.append(g)

        avg_link = (sum(linking_gate_vals) /
                    max(len(linking_gate_vals), 1))
        avg_other = (sum(nonlinking_gate_vals) /
                     max(len(nonlinking_gate_vals), 1))

        print(f"\n  Avg LINKING: {avg_link:.3f}, Avg OTHER: {avg_other:.3f}")
        if avg_other > 1e-6:
            ratio = avg_link / avg_other
            print(f"  Selectivity ratio: {ratio:.2f}x")
        else:
            print(f"  Selectivity ratio: inf (other ≈ 0)")

        all_results[tau] = (avg_link, avg_other)

    # Cross-tau summary
    if len(tau_values) > 1:
        print(f"\n{'=' * 65}")
        print("  TEMPERATURE SCALING SUMMARY")
        print(f"{'=' * 65}")
        print(f"  {'tau':>6} │ {'link':>8} │ {'other':>8} │ {'ratio':>8}")
        print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
        for tau in tau_values:
            al, ao = all_results[tau]
            ratio = al / max(ao, 1e-6)
            print(f"  {tau:6.2f} │ {al:8.3f} │ {ao:8.3f} │ {ratio:7.1f}x")

    # Restore tau to 1.0
    trace.set_gate_tau(1.0)

    return all_results


# ── Main experiment runners ─────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick smoke test (~3-5 min)."""
    print("=" * 65)
    print("EXPERIMENT 9: Learned Storage Gating (quick)")
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

    # Phase 1: Diagnostics (quick check)
    print(f"\n{'=' * 65}")
    print("PHASE 1: Setup")
    print(f"{'=' * 65}")
    print(f"  Fact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"    {ft.name}: {len(ft.entities)} entities")
    print(f"  Linking tokens: {len(linking_ids)}")

    # Phase 2: Train W_gate (quick: ~1400 steps, gate-only, 5 stages)
    print(f"\n{'=' * 65}")
    print("PHASE 2: Training W_gate (surrogate loss + L1 sparsity)")
    print(f"{'=' * 65}")

    # Stage 1: n_facts=1, no L1 (basic convergence)
    losses_1, accs_1 = train_gate(
        model, tokenizer, fact_types,
        n_steps=200, n_facts_train=1,
        lr=3e-3, device=dev,
        log_every=100, seed=seed,
        gate_only=True, grad_clip=1.0)

    # Stage 2: n_facts=3, no L1 (interference pressure)
    print(f"\n--- Stage 2: n_facts=3 (interference pressure) ---")
    losses_2, accs_2 = train_gate(
        model, tokenizer, fact_types,
        n_steps=300, n_facts_train=3,
        lr=1e-3, device=dev,
        log_every=100, seed=seed + 5000,
        gate_only=True, grad_clip=1.0)

    # Stage 3: n_facts=5, no L1 (strong interference)
    print(f"\n--- Stage 3: n_facts=5 (strong interference) ---")
    losses_3, accs_3 = train_gate(
        model, tokenizer, fact_types,
        n_steps=300, n_facts_train=5,
        lr=3e-4, device=dev,
        log_every=100, seed=seed + 10000,
        gate_only=True, grad_clip=1.0)

    # Stage 4: n_facts=5, L1=0.5 (sparsity pressure — push gate closed)
    print(f"\n--- Stage 4: L1 sparsity (lambda=0.5) ---")
    losses_4, accs_4 = train_gate(
        model, tokenizer, fact_types,
        n_steps=300, n_facts_train=5,
        lr=3e-4, device=dev,
        log_every=100, seed=seed + 15000,
        gate_only=True, grad_clip=1.0,
        l1_lambda=0.5)

    # Stage 5: n_facts=5, L1=1.0 (stronger sparsity)
    print(f"\n--- Stage 5: L1 sparsity (lambda=1.0) ---")
    losses_5, accs_5 = train_gate(
        model, tokenizer, fact_types,
        n_steps=300, n_facts_train=5,
        lr=1e-4, device=dev,
        log_every=100, seed=seed + 20000,
        gate_only=True, grad_clip=1.0,
        l1_lambda=1.0)

    # Restore requires_grad for eval
    for p in model.trace.parameters():
        p.requires_grad_(True)

    # Phase 3: Evaluate with tau sweep
    results = evaluate_conditions(
        model, tokenizer, fact_types, linking_ids,
        n_eval=20, n_facts_list=[1, 3, 5],
        device=dev, verbose=False,
        tau_values=[1.0, 0.3, 0.1])

    # Phase 4: Gate analysis at multiple tau
    analyze_gate(model, tokenizer, fact_types, linking_ids, dev,
                 tau_values=[1.0, 0.3, 0.1])

    print(f"\n{'=' * 65}")
    print("QUICK TEST COMPLETE")
    print(f"{'=' * 65}")


def run(device=None, seed=42, n_eval=100, n_steps=1000, lr=1e-3):
    """Full experiment (~15-30 min)."""
    print("=" * 65)
    print("EXPERIMENT 9: Learned Storage Gating")
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

    # Phase 1: Setup
    print(f"\n{'=' * 65}")
    print("PHASE 1: Setup & Diagnostics")
    print(f"{'=' * 65}")
    print(f"  Fact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"    {ft.name}: {len(ft.entities)} entities")
        examples = [e[0] for e in ft.entities[:5]]
        print(f"      examples: {', '.join(examples)}")
    print(f"  Linking tokens: {len(linking_ids)}")
    for lid in linking_ids:
        print(f"    ID {lid}: '{tokenizer.decode([lid])}'")

    # Phase 2: Train W_gate (5 stages with L1 sparsity annealing)
    print(f"\n{'=' * 65}")
    print("PHASE 2: Training W_gate (surrogate loss + L1 sparsity)")
    print(f"{'=' * 65}")

    # Stage 1: n_facts=1, no L1 (basic convergence)
    n_s1 = n_steps // 5
    print(f"\n--- Stage 1: n_facts=1 ({n_s1} steps) ---")
    losses_1, accs_1 = train_gate(
        model, tokenizer, fact_types,
        n_steps=n_s1, n_facts_train=1,
        lr=lr, device=dev,
        log_every=100, seed=seed,
        gate_only=True, grad_clip=1.0)

    # Stage 2: n_facts=3, no L1 (interference pressure)
    n_s2 = n_steps // 5
    print(f"\n--- Stage 2: n_facts=3 ({n_s2} steps) ---")
    losses_2, accs_2 = train_gate(
        model, tokenizer, fact_types,
        n_steps=n_s2, n_facts_train=3,
        lr=lr / 3, device=dev,
        log_every=100, seed=seed + 5000,
        gate_only=True, grad_clip=1.0)

    # Stage 3: n_facts=5, no L1 (strong interference)
    n_s3 = n_steps // 5
    print(f"\n--- Stage 3: n_facts=5 ({n_s3} steps) ---")
    losses_3, accs_3 = train_gate(
        model, tokenizer, fact_types,
        n_steps=n_s3, n_facts_train=5,
        lr=lr / 10, device=dev,
        log_every=100, seed=seed + 10000,
        gate_only=True, grad_clip=1.0)

    # Stage 4: n_facts=5, L1=0.5 (sparsity pressure)
    n_s4 = n_steps // 5
    print(f"\n--- Stage 4: L1 sparsity, lambda=0.5 ({n_s4} steps) ---")
    losses_4, accs_4 = train_gate(
        model, tokenizer, fact_types,
        n_steps=n_s4, n_facts_train=5,
        lr=lr / 10, device=dev,
        log_every=100, seed=seed + 15000,
        gate_only=True, grad_clip=1.0,
        l1_lambda=0.5)

    # Stage 5: n_facts=5, L1=1.0 (stronger sparsity)
    n_s5 = n_steps // 5
    print(f"\n--- Stage 5: L1 sparsity, lambda=1.0 ({n_s5} steps) ---")
    losses_5, accs_5 = train_gate(
        model, tokenizer, fact_types,
        n_steps=n_s5, n_facts_train=5,
        lr=lr / 30, device=dev,
        log_every=100, seed=seed + 20000,
        gate_only=True, grad_clip=1.0,
        l1_lambda=1.0)

    # Restore all params to their requires_grad state for eval
    for p in model.trace.parameters():
        p.requires_grad_(True)

    # Phase 3: Evaluate at multiple tau values
    results = evaluate_conditions(
        model, tokenizer, fact_types, linking_ids,
        n_eval=n_eval, n_facts_list=[1, 3, 5, 7],
        device=dev, verbose=True,
        tau_values=[1.0, 0.3, 0.1])

    # Phase 4: Gate analysis at multiple tau
    gate_results = analyze_gate(
        model, tokenizer, fact_types, linking_ids, dev,
        tau_values=[1.0, 0.3, 0.1])

    # Final summary
    print(f"\n{'=' * 65}")
    print("EXPERIMENT 9 COMPLETE")
    print(f"{'=' * 65}")

    # Key comparison: best tau vs hardcoded mask
    best_tau = None
    best_acc_5 = -1
    for tau in [1.0, 0.3, 0.1]:
        cn = f"gate_tau={tau}"
        if 5 in results.get(cn, {}):
            acc = results[cn][5].accuracy
            if acc > best_acc_5:
                best_acc_5 = acc
                best_tau = tau

    print(f"\n  Best tau for n=5: {best_tau}")
    print(f"\n  Key comparison (best learned gate vs hardcoded mask):")
    best_cn = f"gate_tau={best_tau}" if best_tau else "gate_tau=1.0"
    for n_facts in [1, 3, 5, 7]:
        if n_facts in results.get(best_cn, {}):
            lg = results[best_cn][n_facts].accuracy
            hm = results["hardcoded_mask"][n_facts].accuracy
            diff = lg - hm
            marker = "+" if diff >= 0 else ""
            print(f"    n={n_facts}: learned={lg:.1%}, "
                  f"hardcoded={hm:.1%} ({marker}{diff:.1%})")

    # Gate selectivity at best tau
    if best_tau and best_tau in gate_results:
        avg_link, avg_other = gate_results[best_tau]
        print(f"\n  Gate selectivity (tau={best_tau}): "
              f"link={avg_link:.3f}, other={avg_other:.3f}")
        if avg_link > 0.9 and avg_other < 0.1:
            print(f"  >> NEAR-BINARY gating achieved!")
        elif avg_link > avg_other * 3:
            print(f"  >> STRONG selectivity with tau scaling!")
        elif avg_link > avg_other * 1.5:
            print(f"  >> MODERATE selectivity")
        else:
            print(f"  >> Selectivity not improved by tau")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 9: Learned Storage Gating")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (200 train steps, 20 eval)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=1000,
                        help="Training steps for W_gate")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam")

    args = parser.parse_args()

    if args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run(device=args.device, seed=args.seed, n_eval=args.n_eval,
            n_steps=args.n_steps, lr=args.lr)
