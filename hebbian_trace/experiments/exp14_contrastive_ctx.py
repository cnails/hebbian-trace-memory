"""Experiment 14: Contrastive W_ctx Training for GPT-2 Hebbian Trace.

Exp13 showed random W_ctx with β=0.5 at L3 improves paraphrasing +17-25pp
but barely reduces distractor confusion (66.7% → 63.3%). Random projection
doesn't know which aspect of hidden states is discriminative.

Solution: Train W_ctx via surrogate CE loss with distractor episodes so that:
  - Q("My name" in fact) ≈ Q("my name" in question) — same person alignment
  - Q("My name" in fact) ≠ Q("Alice's name" in distractor) — disambiguation
  - Target: confusion rate <30% (from 63.3%), cross-context n=5 ≥80%

Architecture:
    Q = Q_base + beta * W_ctx(LN_ctx(hidden_states.detach()))
    W_ctx: Linear(768, 512, bias=False) = 393K params
    LN_ctx: LayerNorm(768) = 1.5K params
    Total trainable: ~395K params (everything else frozen)

Training:
    1. Build "my" facts + distractor facts → single GPT-2 pass → hidden states
    2. Compute Q, V with train_ctx=True (gradient flows through W_ctx/LN_ctx)
    3. write_differentiable → T_v_diff (gradient through Q at gate=1 positions)
    4. For each "my" question: read_from_trace → logits → CE loss on entity
    5. Backprop → W_ctx, LN_ctx

Gradient flow (both write and read paths):
    loss → W_out(Q_addr @ T_v_diff) @ wte.T
         → T_v_diff = Q_store.T @ V_store    (write: Q_fact → W_ctx)
         → Q_addr @ T_v_diff                  (read: Q_question → W_ctx)
         → Q = Q_base + β * W_ctx(LN_ctx(hidden.detach()))
         → W_ctx, LN_ctx

Usage:
    python -m hebbian_trace.experiments.exp14_contrastive_ctx --quick
    python -m hebbian_trace.experiments.exp14_contrastive_ctx --n-eval 50
"""

import argparse
import copy
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    GPT2EvalEpisode, GPT2FactType,
    _get_all_entity_ids, tokenize_fact, tokenize_question,
)
from .exp12_realistic_benchmarks import (
    build_question_variants,
    DISTRACTOR_PERSONS, THIRD_PERSON_TEMPLATES,
    validate_distractor_persons,
)
from .exp13_contextual_q import (
    evaluate_cross_context_beta,
    evaluate_paraphrasing_beta,
    evaluate_distractors_beta,
    BetaResult,
    get_device,
)


# ── Contrastive episode generation ───────────────────────────────────

@dataclass
class ContrastiveEpisode:
    """One training episode with first-person + distractor facts."""
    my_facts: list[tuple[str, str, int, list[int]]]
    # (type_name, entity_name, entity_bpe_id, fact_bpe_ids)
    all_fact_ids: list[int]
    # Concatenated BPE IDs: all my facts + all distractor facts (shuffled)
    my_queries: list[tuple[list[int], int, str]]
    # (query_bpe_ids, answer_bpe_id, type_name) — only "my" facts
    my_concept_positions: list[int]
    # Absolute position of each my_fact's concept word in all_fact_ids
    # (position before linking token, used for alignment loss)
    n_distractors: int


def make_contrastive_episode(
    fact_types: list[GPT2FactType],
    n_my: int,
    n_dist_per_type: int,
    tokenizer: GPT2Tokenizer,
    distractor_persons: list[tuple[str, int]],
    linking_ids_set: set[int],
    rng: random.Random,
) -> ContrastiveEpisode:
    """Generate one contrastive training episode.

    Contains n_my first-person facts + n_dist_per_type distractor facts
    per selected type. Questions only about "my" facts.
    Tracks concept positions for alignment loss.
    """
    # Select fact types (sample without replacement if possible)
    if n_my <= len(fact_types):
        selected_types = rng.sample(fact_types, n_my)
    else:
        selected_types = [rng.choice(fact_types) for _ in range(n_my)]

    # Build first-person facts
    my_facts = []
    for ft in selected_types:
        entity_name, entity_id = rng.choice(ft.entities)
        template = rng.choice(ft.fact_templates)
        fact_ids = tokenize_fact(tokenizer, template, entity_name)
        my_facts.append((ft.name, entity_name, entity_id, fact_ids))

    # Build distractor facts (third-person)
    dist_sequences: list[list[int]] = []
    n_actual_distractors = 0
    if n_dist_per_type > 0 and distractor_persons:
        for i, ft in enumerate(selected_types):
            my_entity = my_facts[i][1]  # entity_name

            # Other entities for this type (exclude "my" entity)
            other_entities = [
                (e, eid) for e, eid in ft.entities if e != my_entity
            ]
            template_str = THIRD_PERSON_TEMPLATES.get(ft.name, "")
            if not template_str or not other_entities:
                continue

            for _ in range(min(n_dist_per_type, len(other_entities))):
                d_entity, _ = rng.choice(other_entities)
                person, _ = rng.choice(distractor_persons)
                text = template_str.replace("{P}", person).replace(
                    "{X}", d_entity)
                dist_ids = tokenizer.encode(text, add_special_tokens=False)
                dist_sequences.append(dist_ids)
                n_actual_distractors += 1

    # Tag sequences for tracking: (bpe_ids, is_mine, my_fact_index)
    my_sequences = [f[3] for f in my_facts]
    tagged: list[tuple[list[int], bool, int]] = [
        (seq, True, i) for i, seq in enumerate(my_sequences)
    ]
    tagged += [(seq, False, -1) for seq in dist_sequences]
    rng.shuffle(tagged)

    # Concatenate with space separators, tracking concept positions
    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    all_fact_ids: list[int] = []
    my_concept_positions: list[int | None] = [None] * len(my_facts)

    for i, (seq, is_mine, fact_idx) in enumerate(tagged):
        if i > 0:
            all_fact_ids.append(space_id)
        offset = len(all_fact_ids)

        if is_mine:
            # Find concept position: one before first linking token
            for pos, tok_id in enumerate(seq):
                if tok_id in linking_ids_set:
                    concept_pos = offset + max(pos - 1, 0)
                    my_concept_positions[fact_idx] = concept_pos
                    break

        all_fact_ids.extend(seq)

    # Fallback: if no linking token found, use position 0
    my_concept_positions_final = [
        p if p is not None else 0 for p in my_concept_positions
    ]

    # Build questions (only for "my" facts)
    my_queries = []
    for ft, (type_name, entity_name, entity_id, _) in zip(
            selected_types, my_facts):
        q_template = rng.choice(ft.question_templates)
        q_ids = tokenize_question(tokenizer, q_template)
        my_queries.append((q_ids, entity_id, type_name))

    return ContrastiveEpisode(
        my_facts=my_facts,
        all_fact_ids=all_fact_ids,
        my_queries=my_queries,
        my_concept_positions=my_concept_positions_final,
        n_distractors=n_actual_distractors,
    )


# ── Binary linking gate ──────────────────────────────────────────────

def _make_binary_gate(
    input_ids: torch.Tensor,
    linking_token_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Binary linking-token gate: 1.0 at linking positions, 0.0 elsewhere.

    Args:
        input_ids: (B, S) token indices
        linking_token_ids: list of BPE IDs for linking tokens

    Returns:
        gate: (B, S) float tensor, 0.0 or 1.0
    """
    gate = torch.zeros_like(input_ids, dtype=torch.float, device=device)
    for tid in linking_token_ids:
        gate = gate + (input_ids == tid).float()
    return gate.clamp(max=1.0)


# ── Training function ────────────────────────────────────────────────

def train_contrastive_ctx(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    distractor_persons: list[tuple[str, int]],
    n_steps: int,
    n_my: int,
    n_dist_per_type: int,
    beta: float,
    context_layer: int,
    lr: float,
    device: torch.device,
    lambda_align: float = 1.0,
    log_every: int = 50,
    seed: int = 42,
    grad_clip: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Train W_ctx + LN_ctx via contrastive CE + alignment loss.

    Combined loss = CE_loss + lambda_align * alignment_loss

    CE loss (contrastive): discriminate my facts from distractors via trace.
    Alignment loss: cos(Q_fact_concept, Q_question_concept) → push same-person
    Q representations together, preserving cross-context retrieval.

    Gradient flows through both paths → W_ctx learns to:
    1. Distinguish "My name" from "Alice's name" (CE)
    2. Align "name" in fact context with "name" in question context (alignment)

    Returns:
        (losses, accuracies) per step
    """
    trace = model.trace
    gpt2 = model.gpt2
    wte = gpt2.transformer.wte
    wte_weight = wte.weight.detach()  # (50257, 768), frozen

    # Freeze everything except W_ctx and LN_ctx
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trace.W_ctx.parameters():
        p.requires_grad_(True)
    for p in trace.ln_ctx.parameters():
        p.requires_grad_(True)

    trainable = list(trace.W_ctx.parameters()) + list(
        trace.ln_ctx.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.Adam(trainable, lr=lr)

    entity_ids = _get_all_entity_ids(fact_types)
    entity_tensor = torch.tensor(entity_ids, dtype=torch.long, device=device)
    linking_ids = trace.linking_token_ids or []
    linking_ids_set = set(linking_ids)

    align_str = f", λ_align={lambda_align}" if lambda_align > 0 else ""
    print(f"    {n_steps} steps, n_my={n_my}, n_dist={n_dist_per_type}/type, "
          f"beta={beta}, layer={context_layer}, lr={lr:.0e}{align_str}")
    print(f"    Trainable params: {n_trainable:,}, "
          f"Entity pool: {len(entity_ids)}")

    losses: list[float] = []
    accuracies: list[float] = []
    rng = random.Random(seed)
    t0 = time.time()

    for step in range(n_steps):
        # 1. Generate episode (with concept position tracking)
        episode = make_contrastive_episode(
            fact_types, n_my, n_dist_per_type,
            tokenizer, distractor_persons, linking_ids_set, rng)

        optimizer.zero_grad()

        # 2. GPT-2 forward on facts → hidden states
        fact_tensor = torch.tensor(
            [episode.all_fact_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            fact_out = gpt2(
                fact_tensor, output_hidden_states=True, return_dict=True)
            hidden_fact = fact_out.hidden_states[context_layer]

        # 3. Compute Q, V with gradient through W_ctx
        Q, V = trace.compute_qv(
            wte, fact_tensor, hidden_states=hidden_fact,
            beta=beta, train_ctx=True)
        # Q: (1, H, S_f, d_trace)

        # 4. Binary linking gate + write to differentiable trace
        gate = _make_binary_gate(fact_tensor, linking_ids, device)
        T_v_diff = trace.write_differentiable(Q, V, gate)

        # 5. For each question: read → logits → CE loss + alignment loss
        ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
        align_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_correct = 0
        n_align = 0

        for qi, (q_ids, answer_id, _) in enumerate(episode.my_queries):
            q_tensor = torch.tensor(
                [q_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                q_out = gpt2(
                    q_tensor, output_hidden_states=True, return_dict=True)
                hidden_q = q_out.hidden_states[context_layer]

            Q_q, _ = trace.compute_qv(
                wte, q_tensor, hidden_states=hidden_q,
                beta=beta, train_ctx=True)
            # Q_q: (1, H, S_q, d_trace)

            # --- CE loss (contrastive via trace) ---
            retrieved = trace.read_from_trace(Q_q, T_v_diff)
            trace_logits = torch.matmul(retrieved, wte_weight.T)

            pred = trace_logits[0, -1, entity_tensor]
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]

            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            ce_loss = ce_loss + loss

            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

            # --- Alignment loss (fact concept ↔ question concept) ---
            if lambda_align > 0:
                concept_pos = episode.my_concept_positions[qi]
                S_f = Q.shape[2]
                S_q = Q_q.shape[2]

                if concept_pos < S_f and S_q >= 2:
                    # Q at concept position in fact (before linking token)
                    Q_fact_c = Q[0, :, concept_pos, :]  # (H, d_trace)
                    # Q at second-to-last position in question
                    # (Q_addr[last] = Q[last-1] → retrieval key)
                    Q_q_c = Q_q[0, :, -2, :]  # (H, d_trace)

                    # Per-head cosine similarity, averaged
                    cos = F.cosine_similarity(Q_fact_c, Q_q_c, dim=-1)
                    # (H,)
                    align_loss = align_loss + (1.0 - cos).mean()
                    n_align += 1

        n_queries = max(len(episode.my_queries), 1)
        avg_ce = ce_loss / n_queries

        if n_align > 0 and lambda_align > 0:
            avg_align = align_loss / n_align
            total_loss = avg_ce + lambda_align * avg_align
        else:
            total_loss = avg_ce

        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

        optimizer.step()

        losses.append(total_loss.item())
        accuracies.append(n_correct / n_queries)

        if (step + 1) % log_every == 0:
            recent = min(log_every, len(losses))
            r_loss = sum(losses[-recent:]) / recent
            r_acc = sum(accuracies[-recent:]) / recent
            elapsed = time.time() - t0
            print(f"    Step {step + 1:4d}/{n_steps}: "
                  f"loss={r_loss:.4f}, acc={r_acc:.1%} ({elapsed:.0f}s)")

    total_time = time.time() - t0
    tail = min(50, len(losses))
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accuracies[-tail:]) / tail
    print(f"    Done: {total_time:.1f}s, "
          f"final loss={final_loss:.4f}, acc={final_acc:.1%}")

    # Restore requires_grad for eval
    for p in model.parameters():
        p.requires_grad_(False)

    return losses, accuracies


# ── Q similarity diagnostic ──────────────────────────────────────────

@torch.no_grad()
def analyze_q_similarity(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    distractor_persons: list[tuple[str, int]],
    context_layer: int,
    beta: float,
    device: torch.device,
    n_samples: int = 20,
    seed: int = 42,
) -> tuple[float, float]:
    """Measure Q cosine similarity: same-person vs cross-person.

    For each sample:
    - Encode "My name is John." → Q at concept position ("name")
    - Encode "What is my name?" → Q at concept position ("name")
    - Encode "Alice's name is Bob." → Q at concept position ("name")
    Compute cos(Q_fact, Q_question) and cos(Q_fact, Q_distractor).

    Returns (avg_same_person_sim, avg_cross_person_sim).
    """
    trace = model.trace
    gpt2 = model.gpt2
    wte = gpt2.transformer.wte
    linking_ids = set(trace.linking_token_ids or [])

    rng = random.Random(seed)
    same_sims: list[float] = []
    cross_sims: list[float] = []

    for _ in range(n_samples):
        ft = rng.choice(fact_types)
        my_entity, _ = rng.choice(ft.entities)
        other_entities = [(e, eid) for e, eid in ft.entities
                         if e != my_entity]
        if not other_entities or ft.name not in THIRD_PERSON_TEMPLATES:
            continue
        d_entity, _ = rng.choice(other_entities)
        person, _ = rng.choice(distractor_persons)

        # Encode my fact
        template = rng.choice(ft.fact_templates)
        my_fact_ids = tokenize_fact(tokenizer, template, my_entity)

        # Encode question
        q_template = rng.choice(ft.question_templates)
        q_ids = tokenize_question(tokenizer, q_template)

        # Encode distractor fact
        d_text = THIRD_PERSON_TEMPLATES[ft.name].replace(
            "{P}", person).replace("{X}", d_entity)
        d_ids = tokenizer.encode(d_text, add_special_tokens=False)

        # Find linking token positions and extract concept Q (position before)
        def _get_concept_q(token_ids: list[int]) -> torch.Tensor | None:
            t = torch.tensor([token_ids], dtype=torch.long, device=device)
            out = gpt2(t, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[context_layer]
            Q, _ = trace.compute_qv(wte, t, hidden_states=hidden, beta=beta)
            # Q: (1, H, S, d_trace)
            # Find first linking token and take Q at position before it
            for pos in range(1, len(token_ids)):
                if token_ids[pos] in linking_ids:
                    # Concept is at pos-1 (shift-1 addressing)
                    q_vec = Q[0, :, pos - 1, :]  # (H, d_trace)
                    return q_vec.flatten()
            return None

        q_my = _get_concept_q(my_fact_ids)
        q_question = _get_concept_q(q_ids)
        q_dist = _get_concept_q(d_ids)

        if q_my is None or q_question is None or q_dist is None:
            continue

        # Cosine similarities
        sim_same = F.cosine_similarity(
            q_my.unsqueeze(0), q_question.unsqueeze(0)).item()
        sim_cross = F.cosine_similarity(
            q_my.unsqueeze(0), q_dist.unsqueeze(0)).item()

        same_sims.append(sim_same)
        cross_sims.append(sim_cross)

    avg_same = sum(same_sims) / max(len(same_sims), 1)
    avg_cross = sum(cross_sims) / max(len(cross_sims), 1)
    return avg_same, avg_cross


# ── Phase functions ───────────────────────────────────────────────────

def run_phase1_setup(device_str=None):
    """Phase 1: Load model, validate, backward compat check."""
    print("=" * 65)
    print("EXP 14: Contrastive W_ctx Training (Pareto Frontier)")
    print("=" * 65)

    device = get_device(device_str)
    print(f"\nDevice: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    print(f"Fact types: {len(fact_types)}")

    # Validate distractor persons
    all_entity_names = set()
    for ft in fact_types:
        for name, _ in ft.entities:
            all_entity_names.add(name)
    persons = validate_distractor_persons(tokenizer, all_entity_names)
    print(f"Distractor persons: {len(persons)}")

    # Create model
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    ).to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model.eval()

    n_trace = sum(p.numel() for p in model.trace.parameters())
    print(f"Trace params: {n_trace:,}")

    # Backward compat check
    print(f"\n{'─' * 65}")
    print("PHASE 1: Backward Compatibility Check (β=0)")
    print(f"{'─' * 65}")

    episodes = make_gpt2_eval_episodes(
        n_episodes=10, n_facts=3,
        tokenizer=tokenizer, fact_types=fact_types, seed=42)
    acc = evaluate_cross_context_beta(
        model, episodes, fact_types, beta=0.0, context_layer=4)
    print(f"  Cross-context (n=3, β=0): {acc:.1%}")
    if acc < 0.5:
        print("  ⚠ WARNING: baseline below 50%")
    else:
        print("  ✓ Backward compat OK")

    return model, tokenizer, fact_types, persons, device


def run_phase2_random_baseline(
    model, tokenizer, fact_types, n_eval, context_layer, seed=42,
) -> list[BetaResult]:
    """Phase 2: Random W_ctx baseline (before training)."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 2: Random W_ctx Baseline (n_eval={n_eval})")
    print(f"{'─' * 65}")

    variants = build_question_variants(tokenizer)
    results: list[BetaResult] = []

    for beta in [0.0, 0.5, 1.0]:
        print(f"\n  β = {beta:.2f}...")
        t0 = time.time()

        # Cross-context
        eps_3 = make_gpt2_eval_episodes(
            n_eval, 3, tokenizer, fact_types, seed=seed)
        eps_5 = make_gpt2_eval_episodes(
            n_eval, 5, tokenizer, fact_types, seed=seed + 1000)
        cross_3 = evaluate_cross_context_beta(
            model, eps_3, fact_types, beta, context_layer)
        cross_5 = evaluate_cross_context_beta(
            model, eps_5, fact_types, beta, context_layer)

        # Paraphrasing
        para = evaluate_paraphrasing_beta(
            model, tokenizer, fact_types, variants,
            n_eval, n_facts=5, beta=beta,
            context_layer=context_layer, seed=seed)
        aligned = para["aligned"][0] / max(para["aligned"][1], 1)
        misaligned = para["misaligned"][0] / max(para["misaligned"][1], 1)
        semantic = para["semantic"][0] / max(para["semantic"][1], 1)

        # Distractors
        dist_acc, confusion = evaluate_distractors_beta(
            model, tokenizer, fact_types, n_eval,
            n_my_facts=5, n_distractors_per_type=1,
            beta=beta, context_layer=context_layer, seed=seed)

        elapsed = time.time() - t0
        print(f"    cross-ctx: n=3 {cross_3:.1%}, n=5 {cross_5:.1%}")
        print(f"    paraphrase: aligned {aligned:.1%}, "
              f"misaligned {misaligned:.1%}, semantic {semantic:.1%}")
        print(f"    distractors: acc {dist_acc:.1%}, "
              f"confusion {confusion:.1%}")
        print(f"    ({elapsed:.0f}s)")

        results.append(BetaResult(
            beta=beta, context_layer=context_layer,
            cross_ctx_n3=cross_3, cross_ctx_n5=cross_5,
            aligned=aligned, misaligned=misaligned, semantic=semantic,
            dist_accuracy=dist_acc, dist_confusion=confusion,
        ))

    return results


def run_phase3_train(
    model, tokenizer, fact_types, persons,
    context_layer, device, n_steps_total=1000, seed=42,
):
    """Phase 3: Train W_ctx with 4-stage contrastive curriculum."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 3: Train W_ctx ({n_steps_total} steps)")
    print(f"{'─' * 65}")

    # 4-stage curriculum with alignment loss
    # lambda_align=1.0: explicit alignment keeps Q(fact) ≈ Q(question)
    # while CE loss pushes Q(my) ≠ Q(distractor)
    stages = [
        {"n_my": 3, "n_dist": 0, "lr": 3e-4, "lambda_align": 1.0,
         "label": "Stage 1: alignment-only"},
        {"n_my": 3, "n_dist": 1, "lr": 1e-4, "lambda_align": 1.0,
         "label": "Stage 2: CE + alignment"},
        {"n_my": 5, "n_dist": 1, "lr": 3e-5, "lambda_align": 1.0,
         "label": "Stage 3: harder + alignment"},
        {"n_my": 5, "n_dist": 2, "lr": 1e-5, "lambda_align": 1.0,
         "label": "Stage 4: full + alignment"},
    ]

    # Distribute steps: 20%, 30%, 30%, 20%
    step_fracs = [0.20, 0.30, 0.30, 0.20]
    step_counts = [max(int(n_steps_total * f), 50) for f in step_fracs]
    # Adjust last stage to match total
    step_counts[-1] = n_steps_total - sum(step_counts[:-1])

    all_losses: list[float] = []
    all_accs: list[float] = []

    for i, (stage, n_steps) in enumerate(zip(stages, step_counts)):
        print(f"\n  --- {stage['label']} ({n_steps} steps) ---")

        losses, accs = train_contrastive_ctx(
            model, tokenizer, fact_types, persons,
            n_steps=n_steps,
            n_my=stage["n_my"],
            n_dist_per_type=stage["n_dist"],
            beta=1.0,
            context_layer=context_layer,
            lr=stage["lr"],
            device=device,
            lambda_align=stage["lambda_align"],
            log_every=max(n_steps // 4, 25),
            seed=seed + i * 10000,
            grad_clip=1.0,
        )
        all_losses.extend(losses)
        all_accs.extend(accs)

    return all_losses, all_accs


def run_phase4_trained_sweep(
    model, tokenizer, fact_types, n_eval, context_layer, seed=42,
    betas: list[float] | None = None,
) -> list[BetaResult]:
    """Phase 4: β sweep with trained W_ctx."""
    if betas is None:
        betas = [0.0, 0.3, 0.5, 0.7, 1.0]
    print(f"\n{'─' * 65}")
    print(f"PHASE 4: Trained W_ctx β Sweep (n_eval={n_eval})")
    print(f"  Betas: {betas}")
    print(f"{'─' * 65}")

    variants = build_question_variants(tokenizer)
    results: list[BetaResult] = []

    for beta in betas:
        print(f"\n  β = {beta:.2f}...")
        t0 = time.time()

        # Cross-context
        eps_3 = make_gpt2_eval_episodes(
            n_eval, 3, tokenizer, fact_types, seed=seed)
        eps_5 = make_gpt2_eval_episodes(
            n_eval, 5, tokenizer, fact_types, seed=seed + 1000)
        cross_3 = evaluate_cross_context_beta(
            model, eps_3, fact_types, beta, context_layer)
        cross_5 = evaluate_cross_context_beta(
            model, eps_5, fact_types, beta, context_layer)

        # Paraphrasing
        para = evaluate_paraphrasing_beta(
            model, tokenizer, fact_types, variants,
            n_eval, n_facts=5, beta=beta,
            context_layer=context_layer, seed=seed)
        aligned = para["aligned"][0] / max(para["aligned"][1], 1)
        misaligned = para["misaligned"][0] / max(para["misaligned"][1], 1)
        semantic = para["semantic"][0] / max(para["semantic"][1], 1)

        # Distractors
        dist_acc, confusion = evaluate_distractors_beta(
            model, tokenizer, fact_types, n_eval,
            n_my_facts=5, n_distractors_per_type=1,
            beta=beta, context_layer=context_layer, seed=seed)

        elapsed = time.time() - t0
        print(f"    cross-ctx: n=3 {cross_3:.1%}, n=5 {cross_5:.1%}")
        print(f"    paraphrase: aligned {aligned:.1%}, "
              f"misaligned {misaligned:.1%}, semantic {semantic:.1%}")
        print(f"    distractors: acc {dist_acc:.1%}, "
              f"confusion {confusion:.1%}")
        print(f"    ({elapsed:.0f}s)")

        results.append(BetaResult(
            beta=beta, context_layer=context_layer,
            cross_ctx_n3=cross_3, cross_ctx_n5=cross_5,
            aligned=aligned, misaligned=misaligned, semantic=semantic,
            dist_accuracy=dist_acc, dist_confusion=confusion,
        ))

    return results


def _find_best_beta(results: list[BetaResult]) -> float:
    """Find β that minimizes confusion while keeping cross-ctx n=5 ≥ 70%."""
    candidates = [r for r in results if r.cross_ctx_n5 >= 0.70]
    if not candidates:
        candidates = results  # fallback: pick best regardless
    best = min(candidates, key=lambda r: r.dist_confusion)
    return best.beta


def run_phase5_comparison(
    random_results: list[BetaResult],
    trained_results: list[BetaResult],
):
    """Phase 5: Random vs Trained W_ctx comparison."""
    print(f"\n{'═' * 75}")
    print("COMPARISON: Random W_ctx vs Trained W_ctx")
    print(f"{'═' * 75}")

    # Find matching betas
    random_map = {r.beta: r for r in random_results}
    trained_map = {r.beta: r for r in trained_results}

    common_betas = sorted(set(random_map.keys()) & set(trained_map.keys()))
    if not common_betas:
        # Use 0.5 and 1.0 as default comparison points
        common_betas = [0.5, 1.0]

    print(f"\n  {'β':>4s} │ {'':^20s} │ {'':^20s} │ {'':^15s}")
    print(f"  {'':>4s} │ {'Random W_ctx':^20s} │ {'Trained W_ctx':^20s} │ {'Delta':^15s}")
    print(f"  {'':>4s} │ {'Cross5  Confusn':^20s} │ "
          f"{'Cross5  Confusn':^20s} │ "
          f"{'Δconf':^15s}")
    print(f"  {'─' * 4}─┼{'─' * 22}┼{'─' * 22}┼{'─' * 17}")

    for beta in common_betas:
        r = random_map.get(beta)
        t = trained_map.get(beta)
        if r and t:
            d_conf = t.dist_confusion - r.dist_confusion
            print(f"  {beta:4.1f} │ "
                  f"{r.cross_ctx_n5:6.1%}  {r.dist_confusion:6.1%}    │ "
                  f"{t.cross_ctx_n5:6.1%}  {t.dist_confusion:6.1%}    │ "
                  f"{d_conf:+6.1%}")

    # Also show best trained result
    best_trained = min(trained_results, key=lambda r: r.dist_confusion)
    print(f"\n  Best trained: β={best_trained.beta:.1f}")
    print(f"    Cross n=5: {best_trained.cross_ctx_n5:.1%}, "
          f"Confusion: {best_trained.dist_confusion:.1%}")
    print(f"    Aligned: {best_trained.aligned:.1%}, "
          f"Misaligned: {best_trained.misaligned:.1%}")

    # Check targets
    print(f"\n  Target: confusion <30%, cross n=5 ≥80%")
    if best_trained.dist_confusion < 0.30:
        print(f"  ✓ Confusion target MET "
              f"({best_trained.dist_confusion:.1%} < 30%)")
    else:
        print(f"  ✗ Confusion target NOT met "
              f"({best_trained.dist_confusion:.1%} ≥ 30%)")
    if best_trained.cross_ctx_n5 >= 0.80:
        print(f"  ✓ Cross-context target MET "
              f"({best_trained.cross_ctx_n5:.1%} ≥ 80%)")
    else:
        print(f"  ✗ Cross-context target NOT met "
              f"({best_trained.cross_ctx_n5:.1%} < 80%)")


def run_phase6_q_diagnostic(
    model, tokenizer, fact_types, persons,
    context_layer, beta, device,
    label: str = "",
):
    """Q similarity diagnostic: same-person vs cross-person."""
    same_sim, cross_sim = analyze_q_similarity(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, beta=beta,
        device=device, n_samples=30,
    )
    gap = same_sim - cross_sim
    print(f"  Q similarity{label}: same-person={same_sim:.3f}, "
          f"cross-person={cross_sim:.3f}, gap={gap:+.3f}")
    return same_sim, cross_sim


# ── Entry points ──────────────────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick experiment (~8-12 min)."""
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Phase 2: Random W_ctx baseline
    random_results = run_phase2_random_baseline(
        model, tokenizer, fact_types, n_eval=15,
        context_layer=context_layer, seed=seed)

    # Q diagnostic (before training)
    print(f"\n  Q diagnostic (random W_ctx):")
    run_phase6_q_diagnostic(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev, label=" [random]")

    # Phase 3: Train W_ctx (quick: 1000 steps)
    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1000, seed=seed)

    # Q diagnostic (after training)
    print(f"\n  Q diagnostic (trained W_ctx):")
    run_phase6_q_diagnostic(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev, label=" [trained]")

    # Phase 4: Trained W_ctx β sweep
    trained_results = run_phase4_trained_sweep(
        model, tokenizer, fact_types, n_eval=15,
        context_layer=context_layer, seed=seed)

    # Phase 5: Comparison
    run_phase5_comparison(random_results, trained_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_full(device=None, seed=42, n_eval=50):
    """Full experiment (~25-40 min)."""
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Phase 2: Random W_ctx baseline
    random_results = run_phase2_random_baseline(
        model, tokenizer, fact_types, n_eval=n_eval,
        context_layer=context_layer, seed=seed)

    # Q diagnostic (before training)
    print(f"\n  Q diagnostic (random W_ctx):")
    run_phase6_q_diagnostic(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev, label=" [random]")

    # Phase 3: Train W_ctx (full: 1500 steps)
    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1500, seed=seed)

    # Q diagnostic (after training)
    print(f"\n  Q diagnostic (trained W_ctx):")
    run_phase6_q_diagnostic(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev, label=" [trained]")

    # Phase 4: Trained W_ctx β sweep
    trained_results = run_phase4_trained_sweep(
        model, tokenizer, fact_types, n_eval=n_eval,
        context_layer=context_layer, seed=seed)

    # Phase 5: Comparison
    run_phase5_comparison(random_results, trained_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_fine_sweep(device=None, seed=42, n_eval=30):
    """Fine-grained β sweep after training (~10 min).

    Train W_ctx (1000 steps), then sweep β in [0.0, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.60, 0.70, 1.0] with n_eval episodes per beta.
    Finds exact optimal point on the Pareto frontier.
    """
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Phase 3: Train W_ctx (1000 steps)
    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1000, seed=seed)

    # Fine-grained β sweep
    fine_betas = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 1.0]
    results = run_phase4_trained_sweep(
        model, tokenizer, fact_types, n_eval=n_eval,
        context_layer=context_layer, seed=seed,
        betas=fine_betas)

    # Summary table
    print(f"\n{'═' * 75}")
    print("FINE β SWEEP SUMMARY (trained W_ctx, CE+align λ=1.0)")
    print(f"{'═' * 75}")
    print(f"  {'β':>5s} │ {'Cross n=3':>9s} │ {'Cross n=5':>9s} │ "
          f"{'Aligned':>8s} │ {'Confusn':>8s} │ {'DistAcc':>8s}")
    print(f"  {'─' * 5}─┼{'─' * 11}┼{'─' * 11}┼"
          f"{'─' * 10}┼{'─' * 10}┼{'─' * 10}")
    for r in results:
        print(f"  {r.beta:5.2f} │ {r.cross_ctx_n3:9.1%} │ {r.cross_ctx_n5:9.1%} │ "
              f"{r.aligned:8.1%} │ {r.dist_confusion:8.1%} │ {r.dist_accuracy:8.1%}")

    # Find Pareto optimal
    best_confusion = min(results, key=lambda r: r.dist_confusion)
    viable = [r for r in results if r.cross_ctx_n5 >= 0.80]
    if viable:
        best_viable = min(viable, key=lambda r: r.dist_confusion)
        print(f"\n  Best with cross≥80%: β={best_viable.beta:.2f} → "
              f"cross={best_viable.cross_ctx_n5:.1%}, "
              f"confusion={best_viable.dist_confusion:.1%}")
    else:
        print(f"\n  No β achieves cross≥80%")
        # Find best cross
        best_cross = max(results, key=lambda r: r.cross_ctx_n5)
        print(f"  Best cross: β={best_cross.beta:.2f} → "
              f"cross={best_cross.cross_ctx_n5:.1%}, "
              f"confusion={best_cross.dist_confusion:.1%}")

    print(f"  Lowest confusion: β={best_confusion.beta:.2f} → "
          f"cross={best_confusion.cross_ctx_n5:.1%}, "
          f"confusion={best_confusion.dist_confusion:.1%}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 14: Contrastive W_ctx Training")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fine-sweep", action="store_true",
                        help="Fine-grained β sweep (train + 10 betas)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    if args.fine_sweep:
        run_fine_sweep(device=args.device, seed=args.seed,
                       n_eval=args.n_eval)
    elif args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run_full(device=args.device, seed=args.seed, n_eval=args.n_eval)
