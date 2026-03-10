"""Experiment 15: Per-Head β — CLS for Contextual Q.

Exp14 fine sweep confirmed a fundamental Pareto trade-off with uniform β:
  β=0.40 → cross=80.7%, confusion=48.0% (best viable)
  β=0.50 → cross=75.3%, confusion=30.7%

A single β can't simultaneously provide stability AND discrimination.

Solution: Per-head β (CLS specialization):
  - "Slow" heads: β_h ≈ 0 → context-free, stable cross-context
  - "Fast" heads: β_h ≈ 0.7+ → contextual, strong discrimination
  - Target: confusion <30% AND cross-context n=5 ≥80%

Architecture:
    Q = Q_base + β_h * Q_ctx  where β_h = sigmoid(logit_h) per head
    W_ctx frozen (from exp14 training), only β_logits (8 params) trained.

Usage:
    python -m hebbian_trace.experiments.exp15_perhead_beta --quick
    python -m hebbian_trace.experiments.exp15_perhead_beta --n-eval 50
"""

import argparse
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    GPT2FactType, _get_all_entity_ids, tokenize_fact, tokenize_question,
)
from .exp12_realistic_benchmarks import (
    build_question_variants, validate_distractor_persons,
    build_distractor_episodes,
    THIRD_PERSON_TEMPLATES,
)
from .exp13_contextual_q import (
    evaluate_cross_context_beta,
    evaluate_paraphrasing_beta,
    evaluate_distractors_beta,
    _predict_answer_beta,
    BetaResult,
    get_device,
)
from .exp14_contrastive_ctx import (
    make_contrastive_episode,
    _make_binary_gate,
    run_phase1_setup,
    run_phase3_train,
)


# ── Cross-context contrastive alignment ─────────────────────────────

@dataclass
class CrossContextEpisode:
    """Training episode with same facts in two different sequence contexts."""
    # Context 1: my_facts shuffled with distractors (set A)
    ctx1_fact_ids: list[int]
    ctx1_concept_positions: list[int]  # Abs position of concept words (my facts)

    # Context 2: SAME my_facts with DIFFERENT distractors (set B)
    ctx2_fact_ids: list[int]
    ctx2_concept_positions: list[int]

    # Shared
    my_facts: list[tuple[str, str, int, list[int]]]
    # (type_name, entity_name, entity_bpe_id, fact_bpe_ids)
    my_queries: list[tuple[list[int], int, str]]
    # (query_bpe_ids, answer_bpe_id, type_name)

    # Distractor concept positions (for discrimination loss)
    ctx1_distractor_positions: list[int]
    n_distractors_per_ctx: int


def _build_context(
    my_sequences: list[list[int]],
    dist_sequences: list[list[int]],
    linking_ids_set: set[int],
    n_my: int,
    tokenizer: GPT2Tokenizer,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    """Shuffle my_facts + distractor facts, concatenate, track concept positions.

    Returns:
        all_ids: concatenated BPE IDs
        my_concept_positions: absolute positions of concept words (my facts)
        dist_concept_positions: absolute positions of concept words (distractors)
    """
    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    # Tag: (seq, is_mine, index_in_my)
    tagged: list[tuple[list[int], bool, int]] = [
        (seq, True, i) for i, seq in enumerate(my_sequences)
    ]
    tagged += [(seq, False, -1) for seq in dist_sequences]
    rng.shuffle(tagged)

    all_ids: list[int] = []
    my_concept_pos: list[int | None] = [None] * n_my
    dist_concept_pos: list[int] = []

    for i, (seq, is_mine, fact_idx) in enumerate(tagged):
        if i > 0:
            all_ids.append(space_id)
        offset = len(all_ids)

        # Find concept position: one before first linking token
        concept_pos = None
        for pos, tok_id in enumerate(seq):
            if tok_id in linking_ids_set:
                concept_pos = offset + max(pos - 1, 0)
                break

        if is_mine and concept_pos is not None:
            my_concept_pos[fact_idx] = concept_pos
        elif not is_mine and concept_pos is not None:
            dist_concept_pos.append(concept_pos)

        all_ids.extend(seq)

    # Fallback
    my_concept_final = [p if p is not None else 0 for p in my_concept_pos]
    return all_ids, my_concept_final, dist_concept_pos


def make_cross_context_episode(
    fact_types: list[GPT2FactType],
    n_my: int,
    n_dist_per_type: int,
    tokenizer: GPT2Tokenizer,
    distractor_persons: list[tuple[str, int]],
    linking_ids_set: set[int],
    rng: random.Random,
) -> CrossContextEpisode:
    """Generate paired contexts: same my_facts, different distractors.

    Context 1 and Context 2 share the same first-person facts but have
    different third-person distractor facts (different persons/entities).
    This makes GPT-2 hidden states differ at concept positions, testing
    whether W_ctx can produce invariant Q_ctx.
    """
    # Select fact types
    if n_my <= len(fact_types):
        selected_types = rng.sample(fact_types, n_my)
    else:
        selected_types = [rng.choice(fact_types) for _ in range(n_my)]

    # Build first-person facts (SHARED between contexts)
    my_facts = []
    my_sequences: list[list[int]] = []
    for ft in selected_types:
        entity_name, entity_id = rng.choice(ft.entities)
        template = rng.choice(ft.fact_templates)
        fact_ids = tokenize_fact(tokenizer, template, entity_name)
        my_facts.append((ft.name, entity_name, entity_id, fact_ids))
        my_sequences.append(fact_ids)

    # Build distractor set A (for ctx1)
    dist_a: list[list[int]] = []
    if n_dist_per_type > 0 and distractor_persons:
        for i, ft in enumerate(selected_types):
            my_entity = my_facts[i][1]
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
                dist_a.append(
                    tokenizer.encode(text, add_special_tokens=False))

    # Build distractor set B (for ctx2) — DIFFERENT persons/entities
    dist_b: list[list[int]] = []
    if n_dist_per_type > 0 and distractor_persons:
        for i, ft in enumerate(selected_types):
            my_entity = my_facts[i][1]
            other_entities = [
                (e, eid) for e, eid in ft.entities if e != my_entity
            ]
            template_str = THIRD_PERSON_TEMPLATES.get(ft.name, "")
            if not template_str or not other_entities:
                continue
            for _ in range(min(n_dist_per_type, len(other_entities))):
                # Pick different entity/person than set A (best effort)
                d_entity, _ = rng.choice(other_entities)
                person, _ = rng.choice(distractor_persons)
                text = template_str.replace("{P}", person).replace(
                    "{X}", d_entity)
                dist_b.append(
                    tokenizer.encode(text, add_special_tokens=False))

    # Build ctx1 and ctx2 with different shuffles
    ctx1_ids, ctx1_my_pos, ctx1_dist_pos = _build_context(
        my_sequences, dist_a, linking_ids_set, n_my, tokenizer, rng)
    ctx2_ids, ctx2_my_pos, _ = _build_context(
        my_sequences, dist_b, linking_ids_set, n_my, tokenizer, rng)

    # Build queries (only "my" facts)
    my_queries = []
    for ft, (type_name, entity_name, entity_id, _) in zip(
            selected_types, my_facts):
        q_template = rng.choice(ft.question_templates)
        q_ids = tokenize_question(tokenizer, q_template)
        my_queries.append((q_ids, entity_id, type_name))

    return CrossContextEpisode(
        ctx1_fact_ids=ctx1_ids,
        ctx1_concept_positions=ctx1_my_pos,
        ctx2_fact_ids=ctx2_ids,
        ctx2_concept_positions=ctx2_my_pos,
        my_facts=my_facts,
        my_queries=my_queries,
        ctx1_distractor_positions=ctx1_dist_pos,
        n_distractors_per_ctx=len(dist_a),
    )


def train_cross_context_ctx(
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
    lambda_discrim: float = 0.0,
    margin: float = 0.5,
    log_every: int = 50,
    seed: int = 42,
    grad_clip: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Train W_ctx with cross-context alignment + discrimination + CE loss.

    Three loss components:
    1. CE: contrastive retrieval through differentiable trace (ctx1)
    2. Alignment: ||Q_ctx(concept, ctx1) - Q_ctx(concept, ctx2)||² → 0
       Forces W_ctx to be invariant to surrounding facts.
    3. Discrimination: triplet margin between my and distractor Q_ctx.
       Forces W_ctx to distinguish ownership.

    Returns (losses, accuracies) per step.
    """
    trace = model.trace
    gpt2 = model.gpt2
    wte = gpt2.transformer.wte
    wte_weight = wte.weight.detach()

    # Only W_ctx + LN_ctx trainable
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

    disc_str = f", λ_disc={lambda_discrim}, margin={margin}" if (
        lambda_discrim > 0) else ""
    print(f"    {n_steps} steps, n_my={n_my}, n_dist={n_dist_per_type}/type, "
          f"β={beta}, layer={context_layer}, lr={lr:.0e}")
    print(f"    λ_align={lambda_align}{disc_str}")
    print(f"    Trainable: {n_trainable:,} params, "
          f"Entity pool: {len(entity_ids)}")

    losses: list[float] = []
    accuracies: list[float] = []
    align_losses: list[float] = []
    discrim_losses: list[float] = []
    rng = random.Random(seed)
    t0 = time.time()

    for step in range(n_steps):
        episode = make_cross_context_episode(
            fact_types, n_my, n_dist_per_type,
            tokenizer, distractor_persons, linking_ids_set, rng)

        optimizer.zero_grad()

        # ── GPT-2 forward on both contexts ──
        ctx1_tensor = torch.tensor(
            [episode.ctx1_fact_ids], dtype=torch.long, device=device)
        ctx2_tensor = torch.tensor(
            [episode.ctx2_fact_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out1 = gpt2(
                ctx1_tensor, output_hidden_states=True, return_dict=True)
            hidden1 = out1.hidden_states[context_layer]
            out2 = gpt2(
                ctx2_tensor, output_hidden_states=True, return_dict=True)
            hidden2 = out2.hidden_states[context_layer]

        # ── Compute Q_ctx with gradient (for alignment/discrimination) ──
        # Direct W_ctx computation (not through compute_qv) for clean alignment
        Q_ctx1_raw = trace.W_ctx(trace.ln_ctx(hidden1))  # (1, S1, H*d)
        Q_ctx2_raw = trace.W_ctx(trace.ln_ctx(hidden2))  # (1, S2, H*d)

        H, d = trace.n_heads, trace.d_trace
        Q_ctx1 = Q_ctx1_raw.view(1, -1, H, d).permute(0, 2, 1, 3)  # (1,H,S1,d)
        Q_ctx2 = Q_ctx2_raw.view(1, -1, H, d).permute(0, 2, 1, 3)  # (1,H,S2,d)

        # ── Loss 1: Cross-context alignment ──
        align_loss = torch.tensor(0.0, device=device)
        n_align = 0
        for pos1, pos2 in zip(episode.ctx1_concept_positions,
                              episode.ctx2_concept_positions):
            S1 = Q_ctx1.shape[2]
            S2 = Q_ctx2.shape[2]
            if pos1 < S1 and pos2 < S2:
                q1 = Q_ctx1[0, :, pos1, :]  # (H, d)
                q2 = Q_ctx2[0, :, pos2, :]  # (H, d)
                align_loss = align_loss + F.mse_loss(q1, q2)
                n_align += 1
        if n_align > 0:
            align_loss = align_loss / n_align

        # ── Loss 2: Discrimination (my vs distractor in ctx1) ──
        discrim_loss = torch.tensor(0.0, device=device)
        n_discrim = 0
        if lambda_discrim > 0 and episode.ctx1_distractor_positions:
            S1 = Q_ctx1.shape[2]
            # Average my Q_ctx
            my_qs = []
            for pos in episode.ctx1_concept_positions:
                if pos < S1:
                    my_qs.append(Q_ctx1[0, :, pos, :])  # (H, d)
            if my_qs:
                my_avg = torch.stack(my_qs).mean(dim=0)  # (H, d)
                # Average distractor Q_ctx
                dist_qs = []
                for pos in episode.ctx1_distractor_positions:
                    if pos < S1:
                        dist_qs.append(Q_ctx1[0, :, pos, :])
                if dist_qs:
                    dist_avg = torch.stack(dist_qs).mean(dim=0)  # (H, d)
                    # Triplet margin: want dist(my, dist) > margin
                    # Per-head cosine similarity
                    sim_md = F.cosine_similarity(
                        my_avg, dist_avg, dim=-1)  # (H,)
                    # Loss = max(0, sim + margin - 1.0)
                    # Or simpler: max(0, margin - (1 - sim)) = max(0, sim - (1-margin))
                    # We want sim to be low → loss when sim > (1-margin)
                    triplet = torch.clamp(sim_md - (1.0 - margin), min=0.0)
                    discrim_loss = triplet.mean()
                    n_discrim = 1

        # ── Loss 3: CE through differentiable trace (using ctx1) ──
        # Full Q with beta (for trace write)
        Q_full, V = trace.compute_qv(
            wte, ctx1_tensor, hidden_states=hidden1,
            beta=beta, train_ctx=True)

        gate = _make_binary_gate(ctx1_tensor, linking_ids, device)
        T_v_diff = trace.write_differentiable(Q_full, V, gate)

        ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_correct = 0

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

            retrieved = trace.read_from_trace(Q_q, T_v_diff)
            trace_logits = torch.matmul(retrieved, wte_weight.T)

            pred = trace_logits[0, -1, entity_tensor]
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]

            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            ce_loss = ce_loss + loss

            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

        n_queries = max(len(episode.my_queries), 1)
        avg_ce = ce_loss / n_queries

        # ── Combined loss ──
        total_loss = avg_ce
        if lambda_align > 0 and n_align > 0:
            total_loss = total_loss + lambda_align * align_loss
        if lambda_discrim > 0 and n_discrim > 0:
            total_loss = total_loss + lambda_discrim * discrim_loss

        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

        optimizer.step()

        losses.append(avg_ce.item())
        accuracies.append(n_correct / n_queries)
        align_losses.append(align_loss.item() if n_align > 0 else 0.0)
        discrim_losses.append(discrim_loss.item() if n_discrim > 0 else 0.0)

        if (step + 1) % log_every == 0:
            recent = min(log_every, len(losses))
            r_loss = sum(losses[-recent:]) / recent
            r_acc = sum(accuracies[-recent:]) / recent
            r_align = sum(align_losses[-recent:]) / recent
            r_disc = sum(discrim_losses[-recent:]) / recent
            elapsed = time.time() - t0
            disc_str = f", disc={r_disc:.4f}" if lambda_discrim > 0 else ""
            print(f"    Step {step + 1:4d}/{n_steps}: "
                  f"CE={r_loss:.4f}, align={r_align:.4f}{disc_str}, "
                  f"acc={r_acc:.1%} ({elapsed:.0f}s)")

    total_time = time.time() - t0
    tail = min(50, len(losses))
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accuracies[-tail:]) / tail
    final_align = sum(align_losses[-tail:]) / tail
    print(f"    Done: {total_time:.1f}s, CE={final_loss:.4f}, "
          f"align={final_align:.4f}, acc={final_acc:.1%}")

    # Restore
    for p in model.parameters():
        p.requires_grad_(False)

    return losses, accuracies


# ── Per-head β training ─────────────────────────────────────────────

def train_per_head_beta(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    distractor_persons: list[tuple[str, int]],
    n_steps: int,
    n_my: int,
    n_dist_per_type: int,
    context_layer: int,
    device: torch.device,
    lr: float = 0.1,
    lambda_reg: float = 0.5,
    init_logit: float = -1.0,
    log_every: int = 50,
    seed: int = 42,
    grad_clip: float = 1.0,
) -> torch.Tensor:
    """Train per-head β_logits (8 params) via contrastive CE + L2 reg.

    W_ctx is FROZEN. Only β_logits are trainable.
    β_h = sigmoid(logit_h) → bounded in [0, 1].

    Loss = CE (discrimination) + λ_reg * Σβ_h² (pressure toward 0).
    Heads where context genuinely helps will resist L2 and stay high.
    Heads where context is redundant will collapse to low β.

    Gradient flows through:
        loss → Q = Q_base + β_h * Q_ctx → β_h = sigmoid(logit_h) → logit_h
    Even though Q_ctx is computed under torch.no_grad() (W_ctx frozen),
    the product β_h * Q_ctx has gradient through β_h.

    Returns:
        trained_betas: (H,) tensor of learned per-head β values
    """
    trace = model.trace
    gpt2 = model.gpt2
    wte = gpt2.transformer.wte
    wte_weight = wte.weight.detach()  # (50257, 768)
    H = trace.n_heads

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Create trainable β logits
    # sigmoid(init_logit) = initial β per head
    # Default init_logit=-1.0 → sigmoid(-1) ≈ 0.27
    beta_logits = nn.Parameter(
        torch.full((H,), init_logit, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([beta_logits], lr=lr)

    entity_ids = _get_all_entity_ids(fact_types)
    entity_tensor = torch.tensor(entity_ids, dtype=torch.long, device=device)
    linking_ids = trace.linking_token_ids or []
    linking_ids_set = set(linking_ids)

    init_beta = torch.sigmoid(beta_logits).detach().cpu().tolist()[0]
    print(f"    {n_steps} steps, n_my={n_my}, n_dist={n_dist_per_type}/type, "
          f"layer={context_layer}, lr={lr}, λ_reg={lambda_reg}")
    print(f"    Initial β: {init_beta:.3f} per head "
          f"(logit={init_logit:.1f})")

    losses: list[float] = []
    accuracies: list[float] = []
    rng = random.Random(seed)
    t0 = time.time()
    grad_verified = False

    for step in range(n_steps):
        episode = make_contrastive_episode(
            fact_types, n_my, n_dist_per_type,
            tokenizer, distractor_persons, linking_ids_set, rng)

        optimizer.zero_grad()

        # Per-head β from logits
        beta_per_head = torch.sigmoid(beta_logits)  # (H,)

        # GPT-2 forward on facts (frozen)
        fact_tensor = torch.tensor(
            [episode.all_fact_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            fact_out = gpt2(
                fact_tensor, output_hidden_states=True, return_dict=True)
            hidden_fact = fact_out.hidden_states[context_layer]

        # Compute Q, V with per-head β (gradient through beta_logits)
        # train_ctx=False: W_ctx is frozen, but β still gets gradient
        Q, V = trace.compute_qv(
            wte, fact_tensor, hidden_states=hidden_fact,
            beta=beta_per_head, train_ctx=False)

        # Binary linking gate + write to differentiable trace
        gate = _make_binary_gate(fact_tensor, linking_ids, device)
        T_v_diff = trace.write_differentiable(Q, V, gate)

        # For each question: read → logits → CE loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_correct = 0

        for qi, (q_ids, answer_id, _) in enumerate(episode.my_queries):
            q_tensor = torch.tensor(
                [q_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                q_out = gpt2(
                    q_tensor, output_hidden_states=True, return_dict=True)
                hidden_q = q_out.hidden_states[context_layer]

            Q_q, _ = trace.compute_qv(
                wte, q_tensor, hidden_states=hidden_q,
                beta=beta_per_head, train_ctx=False)

            # CE loss through differentiable trace
            retrieved = trace.read_from_trace(Q_q, T_v_diff)
            trace_logits = torch.matmul(retrieved, wte_weight.T)

            pred = trace_logits[0, -1, entity_tensor]
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]

            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            total_loss = total_loss + loss

            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

        n_queries = max(len(episode.my_queries), 1)
        avg_ce = total_loss / n_queries

        # L2 regularization: push β toward 0 (stability pressure)
        if lambda_reg > 0:
            reg = lambda_reg * beta_per_head.pow(2).sum()
            final_loss = avg_ce + reg
        else:
            final_loss = avg_ce
        final_loss.backward()

        # Gradient verification (first backward only)
        if not grad_verified:
            assert beta_logits.grad is not None, \
                "β_logits.grad is None — gradient not flowing through beta!"
            grad_norm = beta_logits.grad.norm().item()
            print(f"    ✓ Gradient verified: β_logits.grad norm = {grad_norm:.4f}")
            grad_verified = True

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([beta_logits], grad_clip)

        optimizer.step()

        losses.append(avg_ce.item())
        accuracies.append(n_correct / n_queries)

        if (step + 1) % log_every == 0:
            recent = min(log_every, len(losses))
            r_loss = sum(losses[-recent:]) / recent
            r_acc = sum(accuracies[-recent:]) / recent
            betas_now = torch.sigmoid(beta_logits).detach().cpu().tolist()
            elapsed = time.time() - t0
            beta_str = " ".join(f"{b:.2f}" for b in betas_now)
            print(f"    Step {step + 1:4d}/{n_steps}: "
                  f"loss={r_loss:.4f}, acc={r_acc:.1%}, "
                  f"β=[{beta_str}] ({elapsed:.0f}s)")

    total_time = time.time() - t0
    trained_betas = torch.sigmoid(beta_logits).detach()
    beta_list = trained_betas.cpu().tolist()
    print(f"    Done: {total_time:.1f}s")
    print(f"    Trained β: [{', '.join(f'{b:.3f}' for b in beta_list)}]")
    print(f"    Range: {min(beta_list):.3f} — {max(beta_list):.3f}")

    # Restore requires_grad
    for p in model.parameters():
        p.requires_grad_(False)

    return trained_betas


# ── Evaluation wrapper ───────────────────────────────────────────────

def evaluate_beta_config(
    model, tokenizer, fact_types, n_eval, context_layer,
    beta, label: str = "", seed=42,
) -> BetaResult:
    """Evaluate a β configuration (scalar or per-head tensor).

    Returns BetaResult with all metrics.
    """
    variants = build_question_variants(tokenizer)

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

    print(f"  {label:>25s}: cross n=3 {cross_3:.1%}, n=5 {cross_5:.1%} │ "
          f"aligned {aligned:.1%} │ "
          f"confusn {confusion:.1%}, dist_acc {dist_acc:.1%}")

    # For BetaResult, store mean β as the beta field
    if isinstance(beta, torch.Tensor):
        beta_val = beta.mean().item()
    else:
        beta_val = beta

    return BetaResult(
        beta=beta_val, context_layer=context_layer,
        cross_ctx_n3=cross_3, cross_ctx_n5=cross_5,
        aligned=aligned, misaligned=misaligned, semantic=semantic,
        dist_accuracy=dist_acc, dist_confusion=confusion,
    )


# ── Phase functions ──────────────────────────────────────────────────

def run_phase3_uniform_ref(
    model, tokenizer, fact_types, n_eval, context_layer, seed=42,
) -> BetaResult:
    """Phase 3: Uniform β=0.40 reference (best from fine sweep)."""
    print(f"\n{'─' * 65}")
    print("PHASE 3: Uniform β=0.40 Reference")
    print(f"{'─' * 65}")
    return evaluate_beta_config(
        model, tokenizer, fact_types, n_eval, context_layer,
        beta=0.40, label="uniform β=0.40", seed=seed)


def run_phase4_fixed_splits(
    model, tokenizer, fact_types, n_eval, context_layer, device, seed=42,
) -> list[tuple[str, list[float], BetaResult]]:
    """Phase 4: Fixed-split per-head β baselines."""
    print(f"\n{'─' * 65}")
    print("PHASE 4: Fixed-Split Per-Head β Baselines")
    print(f"{'─' * 65}")

    splits = [
        ("split_4_4",      [0.0]*4 + [0.7]*4),
        ("split_4_4_mild", [0.2]*4 + [0.6]*4),
        ("split_2_6",      [0.0]*2 + [0.5]*6),
    ]

    results = []
    for name, betas in splits:
        beta_str = " ".join(f"{b:.1f}" for b in betas)
        print(f"\n  {name}: [{beta_str}]")
        beta_tensor = torch.tensor(betas, dtype=torch.float32, device=device)
        r = evaluate_beta_config(
            model, tokenizer, fact_types, n_eval, context_layer,
            beta=beta_tensor, label=name, seed=seed)
        results.append((name, betas, r))

    return results


def run_phase5_train_beta(
    model, tokenizer, fact_types, persons,
    context_layer, device, n_steps=500,
    lambda_reg=0.5, init_logit=-1.0,
    seed=42,
) -> torch.Tensor:
    """Phase 5: Train per-head β_logits."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 5: Train Per-Head β ({n_steps} steps, "
          f"λ_reg={lambda_reg}, init={torch.sigmoid(torch.tensor(init_logit)).item():.2f})")
    print(f"{'─' * 65}")

    trained_betas = train_per_head_beta(
        model, tokenizer, fact_types, persons,
        n_steps=n_steps,
        n_my=5, n_dist_per_type=1,
        context_layer=context_layer,
        device=device,
        lr=0.1,
        lambda_reg=lambda_reg,
        init_logit=init_logit,
        log_every=50,
        seed=seed,
        grad_clip=1.0,
    )
    return trained_betas


def run_phase6_eval_trained(
    model, tokenizer, fact_types, n_eval, context_layer,
    trained_betas, device, seed=42,
) -> BetaResult:
    """Phase 6: Evaluate trained per-head β."""
    print(f"\n{'─' * 65}")
    print("PHASE 6: Trained Per-Head β Evaluation")
    print(f"{'─' * 65}")
    beta_str = " ".join(f"{b:.2f}" for b in trained_betas.cpu().tolist())
    print(f"  Trained β: [{beta_str}]")

    beta_tensor = trained_betas.to(device)
    return evaluate_beta_config(
        model, tokenizer, fact_types, n_eval, context_layer,
        beta=beta_tensor, label="trained per-head β", seed=seed)


def run_phase7_analysis(trained_betas: torch.Tensor):
    """Phase 7: Per-head β analysis."""
    print(f"\n{'─' * 65}")
    print("PHASE 7: Per-Head β Analysis")
    print(f"{'─' * 65}")

    betas = trained_betas.cpu().tolist()
    mean_beta = sum(betas) / len(betas)
    n_low = sum(1 for b in betas if b < 0.3)
    n_mid = sum(1 for b in betas if 0.3 <= b < 0.6)
    n_high = sum(1 for b in betas if b >= 0.6)

    print(f"\n  Per-head β values:")
    for h, b in enumerate(betas):
        role = "CONTEXT-FREE" if b < 0.3 else ("CONTEXTUAL" if b >= 0.6 else "mixed")
        bar = "█" * int(b * 20) + "░" * (20 - int(b * 20))
        print(f"    Head {h}: β={b:.3f} [{bar}] {role}")

    print(f"\n  Mean β: {mean_beta:.3f}")
    print(f"  Low (<0.3): {n_low} heads — stability-focused")
    print(f"  Mid (0.3-0.6): {n_mid} heads — mixed")
    print(f"  High (≥0.6): {n_high} heads — context-focused")

    if n_low >= 2 and n_high >= 2:
        print(f"\n  ✓ CLS specialization confirmed: "
              f"{n_low} stability + {n_high} context heads")
    elif n_low >= 1 and n_high >= 1:
        print(f"\n  ~ Partial CLS: {n_low} stability + {n_high} context heads")
    else:
        print(f"\n  ✗ No clear CLS pattern (all heads similar)")


def run_phase8_comparison(
    uniform_result: BetaResult,
    split_results: list[tuple[str, list[float], BetaResult]],
    trained_result: BetaResult,
):
    """Phase 8: Comparison table."""
    print(f"\n{'═' * 75}")
    print("COMPARISON: Uniform vs Fixed-Split vs Trained Per-Head β")
    print(f"{'═' * 75}")

    print(f"\n  {'Config':>25s} │ {'Cross n=3':>9s} │ {'Cross n=5':>9s} │ "
          f"{'Aligned':>8s} │ {'Confusn':>8s} │ {'DistAcc':>8s}")
    print(f"  {'─' * 25}─┼{'─' * 11}┼{'─' * 11}┼"
          f"{'─' * 10}┼{'─' * 10}┼{'─' * 10}")

    def _row(label, r):
        print(f"  {label:>25s} │ {r.cross_ctx_n3:9.1%} │ {r.cross_ctx_n5:9.1%} │ "
              f"{r.aligned:8.1%} │ {r.dist_confusion:8.1%} │ {r.dist_accuracy:8.1%}")

    _row("uniform β=0.40", uniform_result)
    for name, _, r in split_results:
        _row(name, r)
    _row("trained per-head β", trained_result)

    # Check targets
    print(f"\n  Target: confusion <30% AND cross-context n=5 ≥80%")
    if trained_result.dist_confusion < 0.30 and trained_result.cross_ctx_n5 >= 0.80:
        print(f"  ✓✓ BOTH TARGETS MET! confusion={trained_result.dist_confusion:.1%}, "
              f"cross={trained_result.cross_ctx_n5:.1%}")
        print(f"     Pareto frontier BROKEN — per-head β achieves what uniform cannot.")
    else:
        if trained_result.dist_confusion < 0.30:
            print(f"  ✓ Confusion target MET ({trained_result.dist_confusion:.1%})")
        else:
            print(f"  ✗ Confusion target NOT met ({trained_result.dist_confusion:.1%})")
        if trained_result.cross_ctx_n5 >= 0.80:
            print(f"  ✓ Cross-context target MET ({trained_result.cross_ctx_n5:.1%})")
        else:
            print(f"  ✗ Cross-context target NOT met ({trained_result.cross_ctx_n5:.1%})")

    # Delta vs uniform
    d_conf = trained_result.dist_confusion - uniform_result.dist_confusion
    d_cross = trained_result.cross_ctx_n5 - uniform_result.cross_ctx_n5
    print(f"\n  vs uniform β=0.40: Δconfusion={d_conf:+.1%}, "
          f"Δcross={d_cross:+.1%}")


# ── Split-beta evaluation (β_write ≠ β_read) ────────────────────────

def evaluate_cross_context_split_beta(
    model, episodes, fact_types,
    beta_write: float, beta_read: float,
    context_layer: int = -1,
) -> float:
    """Cross-context eval with different β for write vs read.

    Write facts to trace with beta_write, query with beta_read.
    Hypothesis: high beta_write (discriminative storage) +
    low beta_read (stable retrieval) may break the Pareto frontier.
    """
    model.eval()
    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device
    total_correct = 0
    total_queries = 0

    for episode in episodes:
        model.reset_traces()

        # Write phase: use beta_write
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor, beta=beta_write,
                      context_layer=context_layer)

        # Read phase: use beta_read
        model.set_trace_mode(use=True, update=False)
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer_beta(
                model, query_ids, entity_ids, beta_read, context_layer)
            if pred_id == answer_id:
                total_correct += 1
            total_queries += 1

    return total_correct / max(total_queries, 1)


def evaluate_distractors_split_beta(
    model, tokenizer, fact_types, n_eval, n_my_facts,
    n_distractors_per_type,
    beta_write: float, beta_read: float,
    context_layer: int = -1, seed: int = 42,
) -> tuple[float, float]:
    """Distractor eval with different β for write vs read.

    Returns (my_accuracy, confusion_rate).
    """
    all_entity_names = set()
    for ft in fact_types:
        for name, _ in ft.entities:
            all_entity_names.add(name)
    persons = validate_distractor_persons(tokenizer, all_entity_names)

    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device

    episodes = build_distractor_episodes(
        n_episodes=n_eval,
        n_my_facts=n_my_facts,
        n_distractors_per_type=n_distractors_per_type,
        tokenizer=tokenizer,
        fact_types=fact_types,
        distractor_persons=persons,
        seed=seed,
    )

    total_correct = 0
    total_confused = 0
    total_queries = 0

    for episode in episodes:
        model.reset_traces()

        # Write phase: use beta_write
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor, beta=beta_write,
                      context_layer=context_layer)

        # Read phase: use beta_read
        model.set_trace_mode(use=True, update=False)
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer_beta(
                model, query_ids, entity_ids, beta_read, context_layer)
            total_queries += 1
            if pred_id == answer_id:
                total_correct += 1
            else:
                ft = next(f for f in fact_types if f.name == type_name)
                ft_entity_ids = {eid for _, eid in ft.entities}
                if pred_id in ft_entity_ids:
                    total_confused += 1

    acc = total_correct / max(total_queries, 1)
    confusion = total_confused / max(total_queries, 1)
    return acc, confusion


def run_phase_split_beta(
    model, tokenizer, fact_types, n_eval, context_layer, seed=42,
):
    """β_write ≠ β_read grid sweep.

    Tests whether using different β for write (storage) vs read (retrieval)
    can break the Pareto frontier.

    Three regimes to explore:
    1. β_write=high, β_read=low: discriminative storage, stable retrieval
    2. β_write=low, β_read=high: stable storage, contextual retrieval
    3. β_write=β_read (diagonal): should match uniform β results
    """
    print(f"\n{'═' * 75}")
    print("SPLIT-BETA: β_write ≠ β_read Grid Sweep")
    print(f"{'═' * 75}")
    print("  Hypothesis: different β for write (storage) vs read (retrieval)")
    print("  may break the Pareto frontier.")
    print(f"  n_eval={n_eval}, n_facts=5, n_distractors=1/type\n")

    beta_writes = [0.0, 0.3, 0.5, 0.7, 1.0]
    beta_reads = [0.0, 0.2, 0.4]

    # Pre-generate cross-context episodes (shared across configs)
    eps_5 = make_gpt2_eval_episodes(
        n_eval, 5, tokenizer, fact_types, seed=seed + 1000)

    results = []

    # Header
    print(f"  {'β_wr':>5s} │ {'β_rd':>5s} │ "
          f"{'Cross n=5':>9s} │ {'Confusn':>8s} │ {'DistAcc':>8s}")
    print(f"  {'─' * 5}─┼{'─' * 7}┼"
          f"{'─' * 11}┼{'─' * 10}┼{'─' * 10}")

    for bw in beta_writes:
        for br in beta_reads:
            cross_5 = evaluate_cross_context_split_beta(
                model, eps_5, fact_types,
                beta_write=bw, beta_read=br,
                context_layer=context_layer)

            dist_acc, confusion = evaluate_distractors_split_beta(
                model, tokenizer, fact_types, n_eval,
                n_my_facts=5, n_distractors_per_type=1,
                beta_write=bw, beta_read=br,
                context_layer=context_layer, seed=seed)

            # Mark if both targets met
            mark = " ✓✓" if cross_5 >= 0.80 and confusion < 0.30 else ""
            print(f"  {bw:5.2f} │ {br:5.2f} │ "
                  f"{cross_5:9.1%} │ {confusion:8.1%} │ "
                  f"{dist_acc:8.1%}{mark}")

            results.append((bw, br, cross_5, confusion, dist_acc))

        # Separator between β_write groups
        if bw < beta_writes[-1]:
            print(f"  {'─' * 5}─┼{'─' * 7}┼"
                  f"{'─' * 11}┼{'─' * 10}┼{'─' * 10}")

    # Find best (maximize cross - confusion, prefer both targets met)
    best = None
    for entry in results:
        bw, br, cross, conf, dacc = entry
        score = cross - conf
        both_met = cross >= 0.80 and conf < 0.30
        if best is None:
            best = entry
            best_both = both_met
        else:
            prev_both = best[2] >= 0.80 and best[3] < 0.30
            prev_score = best[2] - best[3]
            # Prefer entries where both targets met
            if both_met and not prev_both:
                best = entry
            elif both_met == prev_both and score > prev_score:
                best = entry

    bw, br, cross, conf, dacc = best
    print(f"\n  Best: β_write={bw:.2f}, β_read={br:.2f} "
          f"→ cross={cross:.1%}, confusion={conf:.1%}")

    if cross >= 0.80 and conf < 0.30:
        print(f"  ✓✓ PARETO BROKEN! Both targets met with split β.")
        print(f"     cross={cross:.1%} (≥80%), confusion={conf:.1%} (<30%)")
    else:
        t_cross = "✓" if cross >= 0.80 else "✗"
        t_conf = "✓" if conf < 0.30 else "✗"
        print(f"  Targets: cross≥80% ({t_cross} {cross:.1%}), "
              f"confusion<30% ({t_conf} {conf:.1%})")

    return results


# ── Asymmetric W_ctx training (write-only context) ──────────────────

def train_write_ctx(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    distractor_persons: list[tuple[str, int]],
    n_steps: int,
    n_my: int,
    n_dist_per_type: int,
    context_layer: int,
    device: torch.device,
    beta_write: float = 1.0,
    beta_read: float = 0.0,
    lr: float = 3e-4,
    log_every: int = 50,
    seed: int = 42,
    grad_clip: float = 1.0,
):
    """Train W_ctx with asymmetric β: high β_write, low β_read.

    Key insight: W_ctx is used only during WRITE (high beta_write).
    Read uses Q_base (beta_read=0) or low beta_read.

    Forces W_ctx to produce Q_ctx that, when added to Q_base for storage,
    creates traces that are:
    1. Still retrievable by Q_base alone (cross-context stable)
    2. Discriminative — "my" facts stored at slightly different addresses
       than distractor facts, because Q_ctx_write shifts the storage Q

    Gradient flow:
        loss → Q_read (Q_base, constant) → T_v_diff → Q_write → W_ctx
        W_ctx gets gradient through the write side of the differentiable trace.
    """
    trace = model.trace
    gpt2 = model.gpt2
    wte = gpt2.transformer.wte
    wte_weight = wte.weight.detach()

    # Only W_ctx + ln_ctx are trainable
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trace.W_ctx.parameters():
        p.requires_grad_(True)
    for p in trace.ln_ctx.parameters():
        p.requires_grad_(True)

    trainable = list(trace.W_ctx.parameters()) + list(trace.ln_ctx.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.Adam(trainable, lr=lr)

    entity_ids = _get_all_entity_ids(fact_types)
    entity_tensor = torch.tensor(entity_ids, dtype=torch.long, device=device)
    linking_ids = trace.linking_token_ids or []
    linking_ids_set = set(linking_ids)

    print(f"    {n_steps} steps, n_my={n_my}, n_dist={n_dist_per_type}/type")
    print(f"    β_write={beta_write}, β_read={beta_read}, layer={context_layer}")
    print(f"    Trainable: {n_trainable} params (W_ctx + ln_ctx), lr={lr}")

    losses: list[float] = []
    accuracies: list[float] = []
    rng = random.Random(seed)
    t0 = time.time()

    for step in range(n_steps):
        episode = make_contrastive_episode(
            fact_types, n_my, n_dist_per_type,
            tokenizer, distractor_persons, linking_ids_set, rng)

        optimizer.zero_grad()

        # ── Write phase: Q_write with high beta, W_ctx gets gradient ──
        fact_tensor = torch.tensor(
            [episode.all_fact_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            fact_out = gpt2(
                fact_tensor, output_hidden_states=True, return_dict=True)
            hidden_fact = fact_out.hidden_states[context_layer]

        Q_write, V = trace.compute_qv(
            wte, fact_tensor, hidden_states=hidden_fact,
            beta=beta_write, train_ctx=True)  # W_ctx gets gradient

        gate = _make_binary_gate(fact_tensor, linking_ids, device)
        T_v_diff = trace.write_differentiable(Q_write, V, gate)

        # ── Read phase: Q_read with low/zero beta ──
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_correct = 0

        for qi, (q_ids, answer_id, _) in enumerate(episode.my_queries):
            q_tensor = torch.tensor(
                [q_ids], dtype=torch.long, device=device)

            if beta_read > 0:
                with torch.no_grad():
                    q_out = gpt2(
                        q_tensor, output_hidden_states=True, return_dict=True)
                    hidden_q = q_out.hidden_states[context_layer]
            else:
                hidden_q = None

            Q_read, _ = trace.compute_qv(
                wte, q_tensor, hidden_states=hidden_q,
                beta=beta_read, train_ctx=False)

            retrieved = trace.read_from_trace(Q_read, T_v_diff)
            trace_logits = torch.matmul(retrieved, wte_weight.T)

            pred = trace_logits[0, -1, entity_tensor]
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]

            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            total_loss = total_loss + loss

            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

        n_queries = max(len(episode.my_queries), 1)
        avg_loss = total_loss / n_queries
        avg_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

        optimizer.step()

        losses.append(avg_loss.item())
        accuracies.append(n_correct / n_queries)

        if (step + 1) % log_every == 0:
            recent = min(log_every, len(losses))
            r_loss = sum(losses[-recent:]) / recent
            r_acc = sum(accuracies[-recent:]) / recent
            elapsed = time.time() - t0
            print(f"    Step {step + 1:4d}/{n_steps}: "
                  f"loss={r_loss:.4f}, acc={r_acc:.1%} ({elapsed:.0f}s)")

    total_time = time.time() - t0
    print(f"    Done: {total_time:.1f}s")

    # Freeze again
    for p in model.parameters():
        p.requires_grad_(False)


def run_phase_write_ctx(
    model, tokenizer, fact_types, persons, n_eval, context_layer, device,
    seed=42,
):
    """Train W_ctx with asymmetric β, evaluate with split-beta grid.

    4-stage curriculum (same structure as exp14 but with β_write≠β_read):
    Stage 1: β_write=1.0, β_read=0.0, n_my=3, no distractors (200 steps)
    Stage 2: β_write=1.0, β_read=0.0, n_my=3, n_dist=1 (300 steps)
    Stage 3: β_write=1.0, β_read=0.0, n_my=5, n_dist=1 (300 steps)
    Stage 4: β_write=1.0, β_read=0.0, n_my=5, n_dist=2 (200 steps)
    """
    print(f"\n{'═' * 75}")
    print("WRITE-CTX: Asymmetric W_ctx Training (β_write=1, β_read=0)")
    print(f"{'═' * 75}")
    print("  W_ctx trained for write-only: must produce stored patterns")
    print("  retrievable by Q_base alone while being discriminative.\n")

    stages = [
        {"n_steps": 200, "n_my": 3, "n_dist": 0, "lr": 3e-4,
         "label": "Stage 1: basic (no distractors)"},
        {"n_steps": 300, "n_my": 3, "n_dist": 1, "lr": 1e-4,
         "label": "Stage 2: + distractors"},
        {"n_steps": 300, "n_my": 5, "n_dist": 1, "lr": 3e-5,
         "label": "Stage 3: harder"},
        {"n_steps": 200, "n_my": 5, "n_dist": 2, "lr": 1e-5,
         "label": "Stage 4: hardest"},
    ]

    for stage in stages:
        print(f"\n  --- {stage['label']} ---")
        train_write_ctx(
            model, tokenizer, fact_types, persons,
            n_steps=stage["n_steps"],
            n_my=stage["n_my"],
            n_dist_per_type=stage["n_dist"],
            context_layer=context_layer,
            device=device,
            beta_write=1.0,
            beta_read=0.0,
            lr=stage["lr"],
            log_every=50,
            seed=seed,
            grad_clip=1.0,
        )

    # Evaluate: sweep β_write × β_read with trained W_ctx
    print(f"\n  Evaluating trained write-ctx W_ctx...")

    beta_writes = [0.0, 0.5, 0.7, 1.0]
    beta_reads = [0.0, 0.2, 0.4]

    eps_5 = make_gpt2_eval_episodes(
        n_eval, 5, tokenizer, fact_types, seed=seed + 1000)

    results = []

    print(f"\n  {'β_wr':>5s} │ {'β_rd':>5s} │ "
          f"{'Cross n=5':>9s} │ {'Confusn':>8s} │ {'DistAcc':>8s}")
    print(f"  {'─' * 5}─┼{'─' * 7}┼"
          f"{'─' * 11}┼{'─' * 10}┼{'─' * 10}")

    for bw in beta_writes:
        for br in beta_reads:
            cross_5 = evaluate_cross_context_split_beta(
                model, eps_5, fact_types,
                beta_write=bw, beta_read=br,
                context_layer=context_layer)

            dist_acc, confusion = evaluate_distractors_split_beta(
                model, tokenizer, fact_types, n_eval,
                n_my_facts=5, n_distractors_per_type=1,
                beta_write=bw, beta_read=br,
                context_layer=context_layer, seed=seed)

            mark = " ✓✓" if cross_5 >= 0.80 and confusion < 0.30 else ""
            print(f"  {bw:5.2f} │ {br:5.2f} │ "
                  f"{cross_5:9.1%} │ {confusion:8.1%} │ "
                  f"{dist_acc:8.1%}{mark}")

            results.append((bw, br, cross_5, confusion, dist_acc))

        if bw < beta_writes[-1]:
            print(f"  {'─' * 5}─┼{'─' * 7}┼"
                  f"{'─' * 11}┼{'─' * 10}┼{'─' * 10}")

    # Find best
    best = None
    for entry in results:
        bw, br, cross, conf, dacc = entry
        score = cross - conf
        both_met = cross >= 0.80 and conf < 0.30
        if best is None:
            best = entry
        else:
            prev_both = best[2] >= 0.80 and best[3] < 0.30
            prev_score = best[2] - best[3]
            if both_met and not prev_both:
                best = entry
            elif both_met == prev_both and score > prev_score:
                best = entry

    bw, br, cross, conf, dacc = best
    print(f"\n  Best: β_write={bw:.2f}, β_read={br:.2f} "
          f"→ cross={cross:.1%}, confusion={conf:.1%}")

    if cross >= 0.80 and conf < 0.30:
        print(f"  ✓✓ PARETO BROKEN! Asymmetric W_ctx achieves both targets.")
    else:
        t_cross = "✓" if cross >= 0.80 else "✗"
        t_conf = "✓" if conf < 0.30 else "✗"
        print(f"  Targets: cross≥80% ({t_cross} {cross:.1%}), "
              f"confusion<30% ({t_conf} {conf:.1%})")

    return results


# ── Cross-context alignment entry point ──────────────────────────────

def run_phase_cross_ctx_alignment(
    model, tokenizer, fact_types, persons, n_eval, context_layer, device,
    n_steps_total=1000, seed=42,
):
    """Train W_ctx with cross-context alignment, then β sweep.

    4-stage curriculum with CE + alignment + discrimination.
    Compares with exp14-style (standard) training.
    """
    print(f"\n{'═' * 75}")
    print("CROSS-CONTEXT ALIGNMENT: Teaching W_ctx context-invariance")
    print(f"{'═' * 75}")
    print("  Same fact in different contexts → alignment loss")
    print("  My vs distractor → discrimination loss")
    print("  CE through trace → prevents W_ctx collapse\n")

    # 4-stage curriculum
    stages = [
        {"n_my": 3, "n_dist": 0, "lr": 3e-4,
         "lambda_align": 2.0, "lambda_discrim": 0.0, "margin": 0.5,
         "label": "Stage 1: CE + alignment (no distractors)"},
        {"n_my": 3, "n_dist": 1, "lr": 1e-4,
         "lambda_align": 1.0, "lambda_discrim": 0.5, "margin": 0.5,
         "label": "Stage 2: + discrimination"},
        {"n_my": 5, "n_dist": 1, "lr": 3e-5,
         "lambda_align": 1.0, "lambda_discrim": 1.0, "margin": 0.5,
         "label": "Stage 3: harder + full loss"},
        {"n_my": 5, "n_dist": 2, "lr": 1e-5,
         "lambda_align": 1.0, "lambda_discrim": 1.0, "margin": 0.5,
         "label": "Stage 4: hardest"},
    ]

    # Distribute steps: 20%, 30%, 30%, 20%
    step_fracs = [0.20, 0.30, 0.30, 0.20]
    step_counts = [max(int(n_steps_total * f), 50) for f in step_fracs]
    step_counts[-1] = n_steps_total - sum(step_counts[:-1])

    for i, (stage, n_steps) in enumerate(zip(stages, step_counts)):
        print(f"\n  --- {stage['label']} ({n_steps} steps) ---")

        train_cross_context_ctx(
            model, tokenizer, fact_types, persons,
            n_steps=n_steps,
            n_my=stage["n_my"],
            n_dist_per_type=stage["n_dist"],
            beta=1.0,
            context_layer=context_layer,
            lr=stage["lr"],
            device=device,
            lambda_align=stage["lambda_align"],
            lambda_discrim=stage["lambda_discrim"],
            margin=stage["margin"],
            log_every=max(n_steps // 4, 25),
            seed=seed + i * 10000,
            grad_clip=1.0,
        )

    # β sweep with trained W_ctx
    print(f"\n{'─' * 65}")
    print(f"CROSS-CTX β SWEEP (n_eval={n_eval})")
    print(f"{'─' * 65}")

    betas = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    results = []

    print(f"\n  {'β':>5s} │ {'Cross n=3':>9s} │ {'Cross n=5':>9s} │ "
          f"{'Confusn':>8s} │ {'DistAcc':>8s}")
    print(f"  {'─' * 5}─┼{'─' * 11}┼{'─' * 11}┼"
          f"{'─' * 10}┼{'─' * 10}")

    for beta in betas:
        eps_3 = make_gpt2_eval_episodes(
            n_eval, 3, tokenizer, fact_types, seed=seed)
        eps_5 = make_gpt2_eval_episodes(
            n_eval, 5, tokenizer, fact_types, seed=seed + 1000)
        cross_3 = evaluate_cross_context_beta(
            model, eps_3, fact_types, beta, context_layer)
        cross_5 = evaluate_cross_context_beta(
            model, eps_5, fact_types, beta, context_layer)

        dist_acc, confusion = evaluate_distractors_beta(
            model, tokenizer, fact_types, n_eval,
            n_my_facts=5, n_distractors_per_type=1,
            beta=beta, context_layer=context_layer, seed=seed)

        mark = " ✓✓" if cross_5 >= 0.80 and confusion < 0.30 else ""
        print(f"  {beta:5.2f} │ {cross_3:9.1%} │ {cross_5:9.1%} │ "
              f"{confusion:8.1%} │ {dist_acc:8.1%}{mark}")

        results.append((beta, cross_3, cross_5, confusion, dist_acc))

    # Summary
    print(f"\n  Target: confusion <30% AND cross n=5 ≥80%")
    viable = [(b, c3, c5, conf, da) for b, c3, c5, conf, da in results
              if c5 >= 0.80 and conf < 0.30]
    if viable:
        best = min(viable, key=lambda x: x[3])
        print(f"  ✓✓ PARETO BROKEN at β={best[0]:.2f}: "
              f"cross={best[2]:.1%}, confusion={best[3]:.1%}")
    else:
        # Best cross with viable confusion
        best_cross = max(results, key=lambda x: x[2])
        best_conf = min(results, key=lambda x: x[3])
        print(f"  Best cross: β={best_cross[0]:.2f} → "
              f"cross={best_cross[2]:.1%}, confusion={best_cross[3]:.1%}")
        print(f"  Best confusion: β={best_conf[0]:.2f} → "
              f"cross={best_conf[2]:.1%}, confusion={best_conf[3]:.1%}")

    return results


def run_cross_ctx_only(device=None, seed=42, n_eval=20):
    """Run cross-context alignment training + β sweep."""
    import copy
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Reference: standard exp14 training
    print(f"\n{'═' * 75}")
    print("REFERENCE: Standard W_ctx Training (exp14)")
    print(f"{'═' * 75}")

    # Save initial state
    wctx_init = copy.deepcopy(model.trace.W_ctx.state_dict())
    lnctx_init = copy.deepcopy(model.trace.ln_ctx.state_dict())

    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1000, seed=seed)

    # Standard β=0.40 reference
    print(f"\n  Standard W_ctx β=0.40 reference:")
    std_r = evaluate_beta_config(
        model, tokenizer, fact_types, n_eval, context_layer,
        beta=0.40, label="std β=0.40", seed=seed)
    print(f"\n  Standard W_ctx β=0.50 reference:")
    std_r5 = evaluate_beta_config(
        model, tokenizer, fact_types, n_eval, context_layer,
        beta=0.50, label="std β=0.50", seed=seed)

    # Reset W_ctx for cross-context training
    model.trace.W_ctx.load_state_dict(wctx_init)
    model.trace.ln_ctx.load_state_dict(lnctx_init)

    # Cross-context alignment training
    cross_ctx_results = run_phase_cross_ctx_alignment(
        model, tokenizer, fact_types, persons, n_eval, context_layer, dev,
        n_steps_total=1000, seed=seed)

    # Comparison
    print(f"\n{'═' * 75}")
    print("COMPARISON: Standard vs Cross-Context Alignment")
    print(f"{'═' * 75}")
    print(f"\n  Standard W_ctx (exp14):")
    print(f"    β=0.40: cross={std_r.cross_ctx_n5:.1%}, "
          f"confusion={std_r.dist_confusion:.1%}")
    print(f"    β=0.50: cross={std_r5.cross_ctx_n5:.1%}, "
          f"confusion={std_r5.dist_confusion:.1%}")
    print(f"\n  Cross-context aligned W_ctx:")
    for b, c3, c5, conf, da in cross_ctx_results:
        if b in [0.4, 0.5, 0.6, 0.7]:
            mark = " ✓✓" if c5 >= 0.80 and conf < 0.30 else ""
            print(f"    β={b:.2f}: cross={c5:.1%}, confusion={conf:.1%}{mark}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    return cross_ctx_results


# ── Entry points ─────────────────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick experiment (~15 min)."""
    import copy
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Phase 2: Train W_ctx (reuse exp14 curriculum)
    print(f"\n{'─' * 65}")
    print("PHASE 2: Train W_ctx (exp14 curriculum, 1000 steps)")
    print(f"{'─' * 65}")
    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1000, seed=seed)

    # Save W_ctx state (β training doesn't modify it, but for clean restarts)
    wctx_state = copy.deepcopy(model.trace.W_ctx.state_dict())
    lnctx_state = copy.deepcopy(model.trace.ln_ctx.state_dict())

    n_eval = 20

    # Phase 3: Uniform β reference
    uniform_r = run_phase3_uniform_ref(
        model, tokenizer, fact_types, n_eval, context_layer, seed=seed)

    # Split-beta: β_write ≠ β_read grid sweep
    split_beta_results = run_phase_split_beta(
        model, tokenizer, fact_types, n_eval, context_layer, seed=seed)

    # Phase 4: Fixed splits
    split_results = run_phase4_fixed_splits(
        model, tokenizer, fact_types, n_eval, context_layer, dev, seed=seed)

    # Phase 5: λ_reg sweep for per-head β training
    print(f"\n{'═' * 65}")
    print("PHASE 5: Per-Head β Training — λ_reg Sweep")
    print(f"{'═' * 65}")

    reg_configs = [
        {"lambda_reg": 0.3, "init_logit": -1.0},
        {"lambda_reg": 0.5, "init_logit": -1.0},
        {"lambda_reg": 1.0, "init_logit": -1.0},
        {"lambda_reg": 2.0, "init_logit": -1.0},
    ]

    best_result = None
    best_betas = None
    best_label = ""
    all_trained = []

    for cfg in reg_configs:
        lreg = cfg["lambda_reg"]
        init = cfg["init_logit"]
        label = f"λ_reg={lreg}"

        # Restore W_ctx (shouldn't change, but safe)
        model.trace.W_ctx.load_state_dict(wctx_state)
        model.trace.ln_ctx.load_state_dict(lnctx_state)

        trained_betas = run_phase5_train_beta(
            model, tokenizer, fact_types, persons,
            context_layer, dev, n_steps=500,
            lambda_reg=lreg, init_logit=init, seed=seed)

        # Evaluate
        beta_tensor = trained_betas.to(dev)
        r = evaluate_beta_config(
            model, tokenizer, fact_types, n_eval, context_layer,
            beta=beta_tensor, label=label, seed=seed)

        beta_list = trained_betas.cpu().tolist()
        n_low = sum(1 for b in beta_list if b < 0.3)
        n_high = sum(1 for b in beta_list if b >= 0.6)
        print(f"    β range: [{min(beta_list):.3f}, {max(beta_list):.3f}], "
              f"low={n_low}, high={n_high}")

        all_trained.append((label, trained_betas, r))

        # Track best (maximize cross while keeping confusion low)
        score = r.cross_ctx_n5 - r.dist_confusion  # higher is better
        if best_result is None or score > (
                best_result.cross_ctx_n5 - best_result.dist_confusion):
            best_result = r
            best_betas = trained_betas
            best_label = label

    # Phase 7: Analysis of best
    print(f"\n  Best config: {best_label}")
    run_phase7_analysis(best_betas)

    # Phase 8: Comparison with best trained
    run_phase8_comparison(uniform_r, split_results, best_result)

    # Full summary table
    print(f"\n{'═' * 75}")
    print("λ_REG SWEEP SUMMARY")
    print(f"{'═' * 75}")
    print(f"  {'Config':>25s} │ {'Cross n=5':>9s} │ {'Confusn':>8s} │ "
          f"{'β_min':>6s} │ {'β_max':>6s} │ {'n_low':>5s} │ {'n_high':>6s}")
    print(f"  {'─' * 25}─┼{'─' * 11}┼{'─' * 10}┼"
          f"{'─' * 8}┼{'─' * 8}┼{'─' * 7}┼{'─' * 8}")
    for label, betas, r in all_trained:
        bl = betas.cpu().tolist()
        n_l = sum(1 for b in bl if b < 0.3)
        n_h = sum(1 for b in bl if b >= 0.6)
        print(f"  {label:>25s} │ {r.cross_ctx_n5:9.1%} │ {r.dist_confusion:8.1%} │ "
              f"{min(bl):6.3f} │ {max(bl):6.3f} │ {n_l:5d} │ {n_h:6d}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_full(device=None, seed=42, n_eval=50):
    """Full experiment (~25-40 min)."""
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Phase 2: Train W_ctx (reuse exp14 curriculum)
    print(f"\n{'─' * 65}")
    print("PHASE 2: Train W_ctx (exp14 curriculum, 1500 steps)")
    print(f"{'─' * 65}")
    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1500, seed=seed)

    # Phase 3: Uniform β reference
    uniform_r = run_phase3_uniform_ref(
        model, tokenizer, fact_types, n_eval, context_layer, seed=seed)

    # Phase 4: Fixed splits
    split_results = run_phase4_fixed_splits(
        model, tokenizer, fact_types, n_eval, context_layer, dev, seed=seed)

    # Phase 5: Train per-head β
    trained_betas = run_phase5_train_beta(
        model, tokenizer, fact_types, persons,
        context_layer, dev, n_steps=800, seed=seed)

    # Phase 6: Evaluate trained
    trained_r = run_phase6_eval_trained(
        model, tokenizer, fact_types, n_eval, context_layer,
        trained_betas, dev, seed=seed)

    # Phase 7: Analysis
    run_phase7_analysis(trained_betas)

    # Phase 8: Comparison
    run_phase8_comparison(uniform_r, split_results, trained_r)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_split_beta_only(device=None, seed=42, n_eval=20):
    """Run only the split-beta test (fastest path to Pareto frontier)."""
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Phase 2: Train W_ctx
    print(f"\n{'─' * 65}")
    print("PHASE 2: Train W_ctx (exp14 curriculum, 1000 steps)")
    print(f"{'─' * 65}")
    run_phase3_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1000, seed=seed)

    # Uniform reference
    uniform_r = run_phase3_uniform_ref(
        model, tokenizer, fact_types, n_eval, context_layer, seed=seed)

    # Split-beta grid
    split_beta_results = run_phase_split_beta(
        model, tokenizer, fact_types, n_eval, context_layer, seed=seed)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_write_ctx_only(device=None, seed=42, n_eval=20):
    """Run only the write-ctx test (asymmetric W_ctx training)."""
    t0 = time.time()
    context_layer = 4  # L3

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    n_eval_use = n_eval

    # Reference: β=0 baseline (no W_ctx involvement)
    print(f"\n{'─' * 65}")
    print("REFERENCE: β=0 (no context)")
    print(f"{'─' * 65}")
    ref_r = evaluate_beta_config(
        model, tokenizer, fact_types, n_eval_use, context_layer,
        beta=0.0, label="β=0 (no ctx)", seed=seed)

    # Train W_ctx with asymmetric β
    write_ctx_results = run_phase_write_ctx(
        model, tokenizer, fact_types, persons, n_eval_use,
        context_layer, dev, seed=seed)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 15: Per-Head β (CLS for Contextual Q)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--split-beta", action="store_true",
                        help="Run only the split-beta test (β_write ≠ β_read)")
    parser.add_argument("--write-ctx", action="store_true",
                        help="Run asymmetric W_ctx training (write-only context)")
    parser.add_argument("--cross-ctx", action="store_true",
                        help="Run cross-context alignment training")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    if args.cross_ctx:
        run_cross_ctx_only(
            device=args.device, seed=args.seed, n_eval=args.n_eval)
    elif args.write_ctx:
        run_write_ctx_only(
            device=args.device, seed=args.seed, n_eval=args.n_eval)
    elif args.split_beta:
        run_split_beta_only(
            device=args.device, seed=args.seed, n_eval=args.n_eval)
    elif args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run_full(device=args.device, seed=args.seed, n_eval=args.n_eval)
