"""Exp 29: Composite Q for multi-token entity addressing.

Addresses the first-token collision bottleneck in HotpotQA multi-hop:
40.7% of bridge entities share their first BPE token (e.g., "The Capitol"
and "The Republic" both start with "The"), causing identical Q addresses
and hop-2 interference.

Composite Q = Q(first_token) + epsilon * sum(Q(remaining_tokens)).
Additive perturbation preserves cross-entity discrimination (first token
dominates) while breaking ties within collision groups.

Three phases:
  Phase 1: Q discrimination analysis — cosine within collision groups
  Phase 2: HotpotQA batched with composite Q (oracle bridge tokens)
  Phase 3: HotpotQA batched with composite Q (auto-regressive completion)

Usage:
  python -m hebbian_trace.experiments.exp29_composite_q --quick
  python -m hebbian_trace.experiments.exp29_composite_q --phase discrimination
  python -m hebbian_trace.experiments.exp29_composite_q --phase batched --n-batches 50
  python -m hebbian_trace.experiments.exp29_composite_q --phase generated --n-batches 50
  python -m hebbian_trace.experiments.exp29_composite_q --phase epsilon --n-batches 20
"""

import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from .exp24_free_text import setup_model
from .exp27_hotpotqa import load_hotpot_questions, HotpotQuestion


# ── Phase 1: Q Discrimination Analysis ─────────────────────────────

def run_phase1_discrimination(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
) -> dict:
    """Analyze Q cosine similarity within first-token collision groups.

    For entities sharing first BPE token, composite Q should produce
    much lower cosine similarity than first-token Q (which is identical).
    """
    print(f"\n{'='*60}")
    print(f"Phase 1: Q Discrimination Analysis")
    print(f"{'='*60}")

    # Group by first token
    groups: dict[int, list[HotpotQuestion]] = defaultdict(list)
    for q in questions:
        groups[q.bridge_first_token_id].append(q)

    # Filter to collision groups (2+ entities sharing first token)
    collision_groups = {tid: qs for tid, qs in groups.items()
                        if len(qs) >= 2}

    print(f"\n  Total collision groups: {len(collision_groups)}")
    print(f"  Questions in collision groups: "
          f"{sum(len(qs) for qs in collision_groups.values())}")

    first_tok_cosines = []
    composite_cosines = []
    group_details = []

    for tid, qs in sorted(collision_groups.items(),
                           key=lambda x: len(x[1]), reverse=True):
        tok_str = tokenizer.decode([tid]).strip()

        # Deduplicate by bridge entity name
        seen = set()
        unique_qs = []
        for q in qs:
            if q.bridge_entity not in seen:
                seen.add(q.bridge_entity)
                unique_qs.append(q)

        if len(unique_qs) < 2:
            continue

        # Compute first-token Q (identical for all in group)
        Q_first = model.trace.compute_q_for_token(model._wte, tid)
        if model.trace._pattern_sep_enabled:
            Q_first_exp = model.trace._sparse_expand(
                Q_first.unsqueeze(0).unsqueeze(2))
            Q_first_flat = Q_first_exp.squeeze(0).squeeze(1).reshape(-1).float()
        else:
            Q_first_flat = Q_first.reshape(-1).float()

        # Compute composite Q for each unique entity
        composite_Qs = []
        for q in unique_qs:
            Q_comp = model.trace.compute_q_for_tokens(
                model._wte, q.bridge_token_ids)
            if model.trace._pattern_sep_enabled:
                Q_comp_exp = model.trace._sparse_expand(
                    Q_comp.unsqueeze(0).unsqueeze(2))
                Q_flat = Q_comp_exp.squeeze(0).squeeze(1).reshape(-1).float()
            else:
                Q_flat = Q_comp.reshape(-1).float()
            composite_Qs.append(Q_flat)

        # Pairwise cosine: first-token (always 1.0 since identical)
        # Pairwise cosine: composite
        group_comp_cos = []
        for i in range(len(composite_Qs)):
            for j in range(i + 1, len(composite_Qs)):
                cos = F.cosine_similarity(
                    composite_Qs[i].unsqueeze(0),
                    composite_Qs[j].unsqueeze(0)).item()
                group_comp_cos.append(cos)
                composite_cosines.append(cos)
                first_tok_cosines.append(1.0)  # identical token

        mean_comp = sum(group_comp_cos) / len(group_comp_cos)
        group_details.append({
            'token': tok_str,
            'tid': tid,
            'n_entities': len(unique_qs),
            'n_pairs': len(group_comp_cos),
            'first_tok_cos': 1.0,
            'composite_mean_cos': mean_comp,
            'composite_max_cos': max(group_comp_cos),
            'entities': [q.bridge_entity[:25] for q in unique_qs[:5]],
        })

    # Print results
    print(f"\n  Collision groups analyzed: {len(group_details)}")
    print(f"  Total pairs: {len(composite_cosines)}")

    if composite_cosines:
        mean_first = sum(first_tok_cosines) / len(first_tok_cosines)
        mean_comp = sum(composite_cosines) / len(composite_cosines)
        max_comp = max(composite_cosines)

        print(f"\n  First-token Q cosine (within group):")
        print(f"    Mean: {mean_first:.3f}  (always 1.0 — identical token)")
        print(f"\n  Composite Q cosine (within group):")
        print(f"    Mean: {mean_comp:.3f}")
        print(f"    Max:  {max_comp:.3f}")
        print(f"    Reduction: {mean_first - mean_comp:.3f}")

        # Top 5 worst collision groups
        print(f"\n  Top collision groups (by size):")
        for g in group_details[:8]:
            print(f"    '{g['token']}': {g['n_entities']} entities, "
                  f"comp_cos={g['composite_mean_cos']:.3f} "
                  f"(max={g['composite_max_cos']:.3f})")
            for e in g['entities']:
                print(f"      - {e}")

    return {
        'n_groups': len(group_details),
        'n_pairs': len(composite_cosines),
        'first_tok_mean_cos': (sum(first_tok_cosines) / len(first_tok_cosines)
                               if first_tok_cosines else 0),
        'composite_mean_cos': (sum(composite_cosines) / len(composite_cosines)
                               if composite_cosines else 0),
        'composite_max_cos': max(composite_cosines) if composite_cosines else 0,
        'group_details': group_details[:10],
    }


# ── Epsilon sweep ───────────────────────────────────────────────────

def run_epsilon_sweep(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
    epsilons: list[float],
    batch_size: int = 10,
    n_batches: int = 20,
) -> dict:
    """Sweep epsilon values to find optimal perturbation weight.

    For each epsilon, measures:
    - Within-collision-group cosine (want: low)
    - Cross-entity cosine (want: also low)
    - Batched oracle e2e accuracy
    """
    print(f"\n{'='*60}")
    print(f"Epsilon Sweep: {epsilons}")
    print(f"{'='*60}")

    all_answer_ids = sorted(set(q.answer_token_id for q in questions))
    all_bridge_ids = sorted(set(q.bridge_first_token_id for q in questions))
    concept_ids = _get_concept_ids(tokenizer, batch_size)
    if len(concept_ids) < batch_size:
        batch_size = len(concept_ids)

    # Get unique bridge entities for cosine analysis
    seen = set()
    unique_qs = []
    for q in questions:
        if q.bridge_entity not in seen:
            seen.add(q.bridge_entity)
            unique_qs.append(q)

    # First-token cosine baseline (computed once)
    sample = unique_qs[:30]
    first_tok_Qs = []
    for q in sample:
        Q = model.trace.compute_q_for_token(model._wte, q.bridge_first_token_id)
        if model.trace._pattern_sep_enabled:
            Q_exp = model.trace._sparse_expand(Q.unsqueeze(0).unsqueeze(2))
            Q = Q_exp.squeeze(0).squeeze(1)
        first_tok_Qs.append(Q.reshape(-1).float())

    ft_cosines = []
    for i in range(len(first_tok_Qs)):
        for j in range(i + 1, len(first_tok_Qs)):
            c = F.cosine_similarity(
                first_tok_Qs[i].unsqueeze(0),
                first_tok_Qs[j].unsqueeze(0)).item()
            ft_cosines.append(c)
    ft_mean = sum(ft_cosines) / len(ft_cosines)

    results = []

    for eps in epsilons:
        # Compute composite Q cosines
        comp_Qs = []
        for q in sample:
            Q = model.trace.compute_q_for_tokens(
                model._wte, q.bridge_token_ids, epsilon=eps)
            if model.trace._pattern_sep_enabled:
                Q_exp = model.trace._sparse_expand(Q.unsqueeze(0).unsqueeze(2))
                Q = Q_exp.squeeze(0).squeeze(1)
            comp_Qs.append(Q.reshape(-1).float())

        comp_cosines = []
        for i in range(len(comp_Qs)):
            for j in range(i + 1, len(comp_Qs)):
                c = F.cosine_similarity(
                    comp_Qs[i].unsqueeze(0),
                    comp_Qs[j].unsqueeze(0)).item()
                comp_cosines.append(c)
        comp_mean = sum(comp_cosines) / len(comp_cosines)

        # Batched oracle accuracy
        baseline_e2e = 0
        composite_e2e = 0
        total = 0

        for _ in range(n_batches):
            batch = random.sample(questions, min(batch_size, len(questions)))

            # Baseline
            model.trace.reset_traces()
            model.set_trace_mode(use=False, update=False)
            for i, q in enumerate(batch):
                model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
                model.write_fact_direct(
                    q.bridge_first_token_id, q.answer_token_id)
            model.set_trace_mode(use=True, update=False)
            for i, q in enumerate(batch):
                h1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
                h2 = model.retrieve_direct(h1, all_answer_ids)
                if h1 == q.bridge_first_token_id and h2 == q.answer_token_id:
                    baseline_e2e += 1
                total += 1

            # Composite
            model.trace.reset_traces()
            model.set_trace_mode(use=False, update=False)
            for i, q in enumerate(batch):
                model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
                model.write_fact_direct_multi(
                    q.bridge_token_ids, q.answer_token_id)
            model.set_trace_mode(use=True, update=False)
            for i, q in enumerate(batch):
                h1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
                if h1 == q.bridge_first_token_id:
                    h2 = model.retrieve_direct_multi(
                        q.bridge_token_ids, all_answer_ids)
                    if h2 == q.answer_token_id:
                        composite_e2e += 1

        b_pct = baseline_e2e / total * 100
        c_pct = composite_e2e / total * 100

        results.append({
            'epsilon': eps,
            'cross_cos_first': ft_mean,
            'cross_cos_composite': comp_mean,
            'baseline_e2e': b_pct,
            'composite_e2e': c_pct,
            'delta': c_pct - b_pct,
        })

        print(f"  eps={eps:.3f}: cross_cos={comp_mean:.3f} (ft={ft_mean:.3f})  "
              f"e2e: base={b_pct:.1f}% comp={c_pct:.1f}% "
              f"({c_pct - b_pct:+.1f}pp)")

    # Summary
    print(f"\n  {'eps':>6} {'cross_cos':>10} {'base_e2e':>9} "
          f"{'comp_e2e':>9} {'delta':>8}")
    print(f"  {'-'*42}")
    for r in results:
        print(f"  {r['epsilon']:>6.3f} {r['cross_cos_composite']:>10.3f} "
              f"{r['baseline_e2e']:>8.1f}% {r['composite_e2e']:>8.1f}% "
              f"{r['delta']:>+7.1f}pp")

    return {'results': results, 'first_tok_cross_cos': ft_mean}


# ── Phase 2: HotpotQA Batched with Composite Q (Oracle) ────────────

def _get_concept_ids(tokenizer: GPT2Tokenizer,
                     batch_size: int) -> list[int]:
    """Get unique single-token concept IDs for batch slots."""
    concept_words = ["link", "chain", "bridge", "connect", "path",
                     "route", "hop", "step", "jump", "trace",
                     "find", "seek", "query", "fetch", "get"]
    concept_ids = []
    for w in concept_words:
        toks = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(toks) == 1:
            concept_ids.append(toks[0])
    return concept_ids[:batch_size]


def run_phase2_batched_oracle(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
    batch_size: int = 5,
    n_batches: int = 50,
    n_banks: int = 32,
) -> dict:
    """Batched evaluation: baseline vs multi-token bank routing (oracle).

    Multi-token bank routing: Q address uses first-token only
    (PS-compatible), but bank selection uses hash(all_token_ids).
    Entities sharing first token route to different banks.

    Oracle: all bridge token IDs are known at retrieval time.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: Batched Oracle — baseline vs banked routing "
          f"(batch={batch_size}, n={n_batches}, banks={n_banks})")
    print(f"{'='*60}")

    all_answer_ids = sorted(set(q.answer_token_id for q in questions))
    all_bridge_ids = sorted(set(q.bridge_first_token_id for q in questions))
    concept_ids = _get_concept_ids(tokenizer, batch_size)

    if len(concept_ids) < batch_size:
        print(f"  WARNING: only {len(concept_ids)} concept tokens, "
              f"need {batch_size}")
        batch_size = len(concept_ids)

    # Counters for three modes
    no_banks = {'hop1': 0, 'hop2_orc': 0, 'e2e': 0}
    baseline = {'hop1': 0, 'hop2_orc': 0, 'e2e': 0}
    banked = {'hop1': 0, 'hop2_orc': 0, 'e2e': 0}
    total = 0

    for batch_idx in range(n_batches):
        batch = random.sample(questions, min(batch_size, len(questions)))

        # ── No banks: first-token Q, no bank routing ──
        model.set_bank_mode(1)
        model.trace.reset_traces()
        model.set_trace_mode(use=False, update=False)
        for i, q in enumerate(batch):
            model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
            model.write_fact_direct(q.bridge_first_token_id, q.answer_token_id)

        model.set_trace_mode(use=True, update=False)
        for i, q in enumerate(batch):
            hop1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
            hop1_ok = (hop1 == q.bridge_first_token_id)
            hop2_orc = model.retrieve_direct(
                q.bridge_first_token_id, all_answer_ids)
            hop2_orc_ok = (hop2_orc == q.answer_token_id)
            hop2_pred = model.retrieve_direct(hop1, all_answer_ids)
            e2e_ok = hop1_ok and (hop2_pred == q.answer_token_id)
            if hop1_ok: no_banks['hop1'] += 1
            if hop2_orc_ok: no_banks['hop2_orc'] += 1
            if e2e_ok: no_banks['e2e'] += 1

        # ── Baseline: first-token Q, standard Q-based bank routing ──
        model.trace.reset_traces()
        model.set_bank_mode(n_banks)
        model.set_trace_mode(use=False, update=False)
        for i, q in enumerate(batch):
            model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
            model.write_fact_direct(q.bridge_first_token_id, q.answer_token_id)

        model.set_trace_mode(use=True, update=False)
        for i, q in enumerate(batch):
            hop1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
            hop1_ok = (hop1 == q.bridge_first_token_id)
            hop2_orc = model.retrieve_direct(
                q.bridge_first_token_id, all_answer_ids)
            hop2_orc_ok = (hop2_orc == q.answer_token_id)
            hop2_pred = model.retrieve_direct(hop1, all_answer_ids)
            e2e_ok = hop1_ok and (hop2_pred == q.answer_token_id)

            if hop1_ok: baseline['hop1'] += 1
            if hop2_orc_ok: baseline['hop2_orc'] += 1
            if e2e_ok: baseline['e2e'] += 1
            total += 1

        # ── Banked: first-token Q, multi-token bank routing ──
        model.trace.reset_traces()
        model.set_bank_mode(n_banks)
        model.set_trace_mode(use=False, update=False)
        for i, q in enumerate(batch):
            # Hop-1: standard (concept → bridge_first_token)
            model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
            # Hop-2: first-token Q but bank routed by all bridge tokens
            model.write_fact_direct_banked(
                q.bridge_first_token_id, q.answer_token_id,
                q.bridge_token_ids)

        model.set_trace_mode(use=True, update=False)
        for i, q in enumerate(batch):
            # Hop-1: standard
            hop1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
            hop1_ok = (hop1 == q.bridge_first_token_id)

            # Hop-2|oracle: banked retrieval (oracle tokens for bank routing)
            hop2_orc = model.retrieve_direct_banked(
                q.bridge_first_token_id, all_answer_ids,
                q.bridge_token_ids)
            hop2_orc_ok = (hop2_orc == q.answer_token_id)

            # E2E: hop-1 → first token → oracle remaining for bank routing
            if hop1_ok:
                e2e_pred = model.retrieve_direct_banked(
                    q.bridge_first_token_id, all_answer_ids,
                    q.bridge_token_ids)
            else:
                e2e_pred = -1
            e2e_ok = hop1_ok and (e2e_pred == q.answer_token_id)

            if hop1_ok: banked['hop1'] += 1
            if hop2_orc_ok: banked['hop2_orc'] += 1
            if e2e_ok: banked['e2e'] += 1

        model.set_trace_mode(use=False, update=False)

    # Reset bank mode
    model.set_bank_mode(1)

    # Results
    def pct(n): return n / total * 100

    print(f"\n  Results ({total} questions in {n_batches} batches):\n")
    print(f"  {'Metric':<16} {'No banks':>9} {'Q-banks':>9} "
          f"{'MT-banks':>9} {'Δ(MT-no)':>9}")
    print(f"  {'-'*56}")
    for metric in ['hop1', 'hop2_orc', 'e2e']:
        label = {'hop1': 'hop-1', 'hop2_orc': 'hop-2|oracle',
                 'e2e': 'end-to-end'}[metric]
        nb = pct(no_banks[metric])
        bl = pct(baseline[metric])
        bk = pct(banked[metric])
        print(f"  {label:<16} {nb:>8.1f}% {bl:>8.1f}% "
              f"{bk:>8.1f}% {bk - nb:>+8.1f}pp")

    return {
        'batch_size': batch_size,
        'n_batches': n_batches,
        'n_banks': n_banks,
        'total': total,
        'no_banks_e2e': pct(no_banks['e2e']),
        'baseline_e2e': pct(baseline['e2e']),
        'banked_e2e': pct(banked['e2e']),
    }


# ── Phase 3: Composite Q + Auto-Regressive Completion ──────────────

@torch.no_grad()
def complete_entity(model: GPT2WithTrace,
                    tokenizer: GPT2Tokenizer,
                    first_token_id: int,
                    max_tokens: int = 4) -> list[int]:
    """Greedy auto-regressive completion from first BPE token.

    Returns list of token IDs including first_token_id.
    Stops at EOS, punctuation, or max_tokens.
    """
    dev = model.trace.value_traces.device
    token_ids = [first_token_id]
    input_ids = torch.tensor([[first_token_id]], device=dev)

    for _ in range(max_tokens - 1):
        logits = model.gpt2(input_ids).logits[:, -1, :]
        next_id = logits.argmax(dim=-1).item()
        next_str = tokenizer.decode([next_id])
        if (next_id == tokenizer.eos_token_id or
                next_str.strip() in ('.', ',', '!', '?', ';', ':')):
            break
        token_ids.append(next_id)
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_id]], device=dev)
        ], dim=1)

    return token_ids


def run_phase3_batched_generated(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
    batch_size: int = 5,
    n_batches: int = 50,
    n_banks: int = 32,
) -> dict:
    """Batched evaluation: multi-token banked routing with best-bank scan.

    Oracle: route to bank by hash(all_bridge_tokens).
    Best-bank: scan ALL banks with Q(first_token), pick highest confidence.
    No completion needed — pure trace confidence drives bank selection.

    Gap between oracle and best-bank measures information loss from
    not knowing which bank to read from.
    """
    print(f"\n{'='*60}")
    print(f"Phase 3: Best-bank scan — baseline vs oracle vs best-bank "
          f"(batch={batch_size}, n={n_batches}, banks={n_banks})")
    print(f"{'='*60}")

    all_answer_ids = sorted(set(q.answer_token_id for q in questions))
    all_bridge_ids = sorted(set(q.bridge_first_token_id for q in questions))
    concept_ids = _get_concept_ids(tokenizer, batch_size)

    if len(concept_ids) < batch_size:
        batch_size = len(concept_ids)

    baseline = {'hop1': 0, 'e2e': 0}
    oracle = {'hop1': 0, 'e2e': 0}
    generated = {'hop1': 0, 'e2e': 0}
    total = 0

    # Track completion quality
    completion_match = 0
    completion_first_match = 0
    completion_total = 0
    bank_match = 0

    for batch_idx in range(n_batches):
        batch = random.sample(questions, min(batch_size, len(questions)))

        # ── Baseline: first-token Q, standard banks ──
        model.trace.reset_traces()
        model.set_bank_mode(n_banks)
        model.set_trace_mode(use=False, update=False)
        for i, q in enumerate(batch):
            model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
            model.write_fact_direct(q.bridge_first_token_id, q.answer_token_id)

        model.set_trace_mode(use=True, update=False)
        for i, q in enumerate(batch):
            hop1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
            hop1_ok = (hop1 == q.bridge_first_token_id)
            hop2 = model.retrieve_direct(hop1, all_answer_ids)
            e2e_ok = hop1_ok and (hop2 == q.answer_token_id)
            if hop1_ok: baseline['hop1'] += 1
            if e2e_ok: baseline['e2e'] += 1
            total += 1

        # ── Oracle + Generated: banked routing ──
        model.trace.reset_traces()
        model.set_bank_mode(n_banks)
        model.set_trace_mode(use=False, update=False)
        for i, q in enumerate(batch):
            model.write_fact_direct(concept_ids[i], q.bridge_first_token_id)
            model.write_fact_direct_banked(
                q.bridge_first_token_id, q.answer_token_id,
                q.bridge_token_ids)

        model.set_trace_mode(use=True, update=False)
        for i, q in enumerate(batch):
            hop1 = model.retrieve_direct(concept_ids[i], all_bridge_ids)
            hop1_ok = (hop1 == q.bridge_first_token_id)
            if hop1_ok:
                oracle['hop1'] += 1
                generated['hop1'] += 1

            # Oracle: banked retrieval with oracle tokens
            if hop1_ok:
                orc_pred = model.retrieve_direct_banked(
                    q.bridge_first_token_id, all_answer_ids,
                    q.bridge_token_ids)
                orc_ok = (orc_pred == q.answer_token_id)
            else:
                orc_ok = False
            if orc_ok: oracle['e2e'] += 1

            # Best-bank: scan all banks, pick highest confidence
            # No completion needed — reads from every bank, picks max logit
            if hop1_ok:
                gen_pred = model.retrieve_direct_best_bank(
                    q.bridge_first_token_id, all_answer_ids)
                gen_ok = (gen_pred == q.answer_token_id)
            else:
                gen_ok = False
            if gen_ok: generated['e2e'] += 1

        model.set_trace_mode(use=False, update=False)

    # Reset bank mode
    model.set_bank_mode(1)

    # Results
    def pct(n): return n / total * 100

    print(f"\n  Results ({total} questions in {n_batches} batches):\n")
    print(f"  {'Metric':<15} {'Q-banks':>10} {'Oracle':>10} "
          f"{'BestBank':>10} {'Orc-Base':>10} {'BB-Base':>10}")
    print(f"  {'-'*65}")
    b_e2e = pct(baseline['e2e'])
    o_e2e = pct(oracle['e2e'])
    g_e2e = pct(generated['e2e'])
    print(f"  {'hop-1':<15} {pct(baseline['hop1']):>9.1f}% "
          f"{pct(oracle['hop1']):>9.1f}% "
          f"{pct(generated['hop1']):>9.1f}%")
    print(f"  {'end-to-end':<15} {b_e2e:>9.1f}% "
          f"{o_e2e:>9.1f}% {g_e2e:>9.1f}% "
          f"{o_e2e - b_e2e:>+9.1f}pp {g_e2e - b_e2e:>+9.1f}pp")

    # Gap analysis
    gap = o_e2e - g_e2e
    print(f"\n  Oracle - BestBank gap: {gap:.1f}pp")

    return {
        'batch_size': batch_size,
        'total': total,
        'baseline_e2e': b_e2e,
        'oracle_e2e': o_e2e,
        'bestbank_e2e': g_e2e,
        'oracle_delta': o_e2e - b_e2e,
        'bestbank_delta': g_e2e - b_e2e,
        'gap': gap,
    }


# ── Batch sweep ─────────────────────────────────────────────────────

def run_batch_sweep(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
    batch_sizes: list[int],
    n_batches: int = 50,
    include_generated: bool = True,
) -> dict:
    """Sweep batch sizes comparing baseline, oracle, and generated."""
    print(f"\n{'='*60}")
    print(f"Batch Size Sweep: {batch_sizes}")
    print(f"{'='*60}")

    results = {}
    for bs in batch_sizes:
        print(f"\n--- batch_size={bs} ---")
        r2 = run_phase2_batched_oracle(
            model, tokenizer, questions,
            batch_size=bs, n_batches=n_batches)
        results[bs] = {'oracle': r2}

        if include_generated:
            r3 = run_phase3_batched_generated(
                model, tokenizer, questions,
                batch_size=bs, n_batches=n_batches)
            results[bs]['generated'] = r3

    # Summary table
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*60}")
    header = (f"  {'Batch':>5}  {'Base e2e':>9}  {'Orc e2e':>9}  "
              f"{'Orc Δ':>8}")
    if include_generated:
        header += f"  {'Gen e2e':>9}  {'Gen Δ':>8}  {'Gap':>6}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for bs in batch_sizes:
        r = results[bs]
        o = r['oracle']
        line = (f"  {bs:>5}  {o['baseline_e2e']:>8.1f}%  "
                f"{o['composite_e2e']:>8.1f}%  "
                f"{o['composite_e2e'] - o['baseline_e2e']:>+7.1f}pp")
        if include_generated and 'generated' in r:
            g = r['generated']
            line += (f"  {g['generated_e2e']:>8.1f}%  "
                     f"{g['generated_delta']:>+7.1f}pp  "
                     f"{g['sensitivity_gap']:>5.1f}")
        print(line)

    return results


# ── Main ─────────────────────────────────────────────────────────────

def run_experiment(
    n_questions: int = 100,
    phase: str = "all",
    batch_size: int = 10,
    n_batches: int = 50,
    quick: bool = False,
    seed: int = 42,
):
    """Run composite Q experiment."""
    random.seed(seed)
    torch.manual_seed(seed)

    if quick:
        n_questions = min(n_questions, 100)
        n_batches = min(n_batches, 20)

    model, tokenizer = setup_model(alpha=0.5, use_ps=True)

    questions = load_hotpot_questions(
        tokenizer, max_questions=n_questions)

    print(f"\nExp 29: Composite Q for Multi-Token Entity Addressing")
    print(f"  Questions: {len(questions)}")
    n_multi = sum(1 for q in questions if q.bridge_n_tokens >= 2)
    print(f"  Multi-token bridges: {n_multi} "
          f"({n_multi/len(questions)*100:.0f}%)")

    results = {}

    if phase in ("all", "discrimination"):
        results["phase1"] = run_phase1_discrimination(
            model, tokenizer, questions)

    if phase == "epsilon":
        epsilons = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
        if quick:
            epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
        results["epsilon"] = run_epsilon_sweep(
            model, tokenizer, questions,
            epsilons=epsilons,
            batch_size=batch_size, n_batches=n_batches)

    if phase in ("all", "batched", "sweep"):
        if phase == "sweep":
            sizes = [1, 3, 5, 8, 10, 15]
            if quick:
                sizes = [1, 5, 10]
            results["sweep"] = run_batch_sweep(
                model, tokenizer, questions,
                batch_sizes=sizes,
                n_batches=n_batches,
                include_generated=True)
        else:
            results["phase2"] = run_phase2_batched_oracle(
                model, tokenizer, questions,
                batch_size=batch_size, n_batches=n_batches)

    if phase in ("all", "generated"):
        results["phase3"] = run_phase3_batched_generated(
            model, tokenizer, questions,
            batch_size=batch_size, n_batches=n_batches)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    if "phase1" in results:
        p1 = results["phase1"]
        print(f"  Q Discrimination: {p1['n_groups']} collision groups, "
              f"composite cos={p1['composite_mean_cos']:.3f} "
              f"(vs 1.000 first-token)")

    if "phase2" in results:
        p2 = results["phase2"]
        print(f"  Oracle (bs={p2['batch_size']}, banks={p2['n_banks']}): "
              f"no_banks={p2['no_banks_e2e']:.1f}% → "
              f"Q-banks={p2['baseline_e2e']:.1f}% → "
              f"MT-banks={p2['banked_e2e']:.1f}%")

    if "phase3" in results:
        p3 = results["phase3"]
        print(f"  BestBank (bs={p3['batch_size']}): "
              f"Q-banks={p3['baseline_e2e']:.1f}% → "
              f"best-bank={p3['bestbank_e2e']:.1f}% "
              f"({p3['bestbank_delta']:+.1f}pp), "
              f"oracle gap={p3['gap']:.1f}pp")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exp 29: Composite Q for multi-token entity addressing")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-questions", type=int, default=841)
    parser.add_argument("--phase",
                        choices=["discrimination", "batched", "generated",
                                 "sweep", "epsilon", "all"],
                        default="all")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        n_questions=args.n_questions,
        phase=args.phase,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
