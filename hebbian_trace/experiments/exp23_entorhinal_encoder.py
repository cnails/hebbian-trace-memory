"""Experiment 23: Entorhinal Encoder — MLP Bottleneck for Contextual Q.

Exp 13-15 showed linear W_ctx creates a linear Pareto frontier:
  - β=0: cross=85%, confusion=75% (no context → can't disambiguate)
  - β=0.5: cross=76%, confusion=42% (context → less stable)
  - Cannot achieve both cross≥80% AND confusion<30%

Root cause: linear W_ctx can only rotate/scale hidden states globally.
It can't selectively extract ownership signal while discarding
context-variability — every direction it amplifies, it amplifies equally.

Hypothesis: MLP bottleneck (d_model → d_bottleneck → H*d_trace with ReLU)
can break the Pareto frontier because:
  1. Bottleneck forces compression → keeps only discriminative features
  2. ReLU gates irrelevant dimensions → selective feature extraction
  3. Fewer params (~166K vs ~395K) → less capacity to memorize noise

Architecture:
    Q_enc = W_enc(LN_enc(hidden_states))
    W_enc = Linear(768, 128) + ReLU + Linear(128, 512)   # 166K params
    Q = Q_base + beta * Q_enc                             # same blending

Training: CE + alignment via differentiable trace (same as exp14).
    - CE: discriminate my facts from distractors through stored trace
    - Alignment: cos(Q_fact_concept, Q_question_concept) → 1

Phase 1: Setup + backward compat (β=0)
Phase 2: Train encoder (4-stage curriculum)
Phase 3: β sweep with trained encoder
Phase 4: Comparison with linear W_ctx (optional: --compare-linear)

Usage:
    python -m hebbian_trace.experiments.exp23_entorhinal_encoder --quick
    python -m hebbian_trace.experiments.exp23_entorhinal_encoder --n-eval 50
    python -m hebbian_trace.experiments.exp23_entorhinal_encoder --compare-linear
"""

import argparse
import copy
import random
import time
from dataclasses import dataclass

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
from .exp14_contrastive_ctx import (
    make_contrastive_episode,
    _make_binary_gate,
)


# ── Extended episode with distractor position tracking ────────────────

@dataclass
class DiscEpisode:
    """Contrastive episode with distractor concept position tracking.

    Extends ContrastiveEpisode to also record where distractor
    concept words appear in the concatenated sequence. Needed for
    explicit discrimination loss: push Q_my away from Q_dist.
    """
    my_facts: list[tuple[str, str, int, list[int]]]
    all_fact_ids: list[int]
    my_queries: list[tuple[list[int], int, str]]
    my_concept_positions: list[int]
    dist_positions_by_type: list[list[int]]
    # dist_positions_by_type[i] = list of concept positions for
    # distractors of the same type as my_facts[i]
    n_distractors: int


def make_disc_episode(
    fact_types: list[GPT2FactType],
    n_my: int,
    n_dist_per_type: int,
    tokenizer: GPT2Tokenizer,
    distractor_persons: list[tuple[str, int]],
    linking_ids_set: set[int],
    rng: random.Random,
) -> DiscEpisode:
    """Generate contrastive episode with distractor position tracking.

    Like make_contrastive_episode but also records concept positions
    for distractor facts, grouped by their corresponding my_fact type.
    """
    # Select fact types
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

    # Build distractor facts with type tracking
    # dist_data: (bpe_ids, my_fact_type_idx)
    dist_data: list[tuple[list[int], int]] = []
    n_actual = 0
    if n_dist_per_type > 0 and distractor_persons:
        for i, ft in enumerate(selected_types):
            my_entity = my_facts[i][1]
            other_entities = [
                (e, eid) for e, eid in ft.entities if e != my_entity]
            template_str = THIRD_PERSON_TEMPLATES.get(ft.name, "")
            if not template_str or not other_entities:
                continue
            for _ in range(min(n_dist_per_type, len(other_entities))):
                d_entity, _ = rng.choice(other_entities)
                person, _ = rng.choice(distractor_persons)
                text = template_str.replace("{P}", person).replace(
                    "{X}", d_entity)
                d_ids = tokenizer.encode(text, add_special_tokens=False)
                dist_data.append((d_ids, i))
                n_actual += 1

    # Tag: (bpe_ids, source, type_idx)
    # source: "mine" or "dist"
    tagged: list[tuple[list[int], str, int]] = [
        (f[3], "mine", i) for i, f in enumerate(my_facts)
    ]
    tagged += [(d_ids, "dist", type_idx)
               for d_ids, type_idx in dist_data]
    rng.shuffle(tagged)

    # Concatenate, tracking concept positions for both
    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    all_fact_ids: list[int] = []
    my_concept_positions: list[int | None] = [None] * len(my_facts)
    dist_positions_by_type: list[list[int]] = [[] for _ in my_facts]

    for i, (seq, source, type_idx) in enumerate(tagged):
        if i > 0:
            all_fact_ids.append(space_id)
        offset = len(all_fact_ids)

        # Find concept position: before first linking token
        concept_pos = None
        for pos, tok_id in enumerate(seq):
            if tok_id in linking_ids_set:
                concept_pos = offset + max(pos - 1, 0)
                break

        if source == "mine" and concept_pos is not None:
            my_concept_positions[type_idx] = concept_pos
        elif source == "dist" and concept_pos is not None:
            dist_positions_by_type[type_idx].append(concept_pos)

        all_fact_ids.extend(seq)

    my_concept_positions_final = [
        p if p is not None else 0 for p in my_concept_positions]

    # Build questions (only "my" facts)
    my_queries = []
    for ft, (type_name, entity_name, entity_id, _) in zip(
            selected_types, my_facts):
        q_template = rng.choice(ft.question_templates)
        q_ids = tokenize_question(tokenizer, q_template)
        my_queries.append((q_ids, entity_id, type_name))

    return DiscEpisode(
        my_facts=my_facts,
        all_fact_ids=all_fact_ids,
        my_queries=my_queries,
        my_concept_positions=my_concept_positions_final,
        dist_positions_by_type=dist_positions_by_type,
        n_distractors=n_actual,
    )


# ── Training function ────────────────────────────────────────────────

def train_encoder(
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
    lambda_disc: float = 0.0,
    disc_margin: float = 0.0,
    log_every: int = 50,
    seed: int = 42,
    grad_clip: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Train MLP encoder (W_enc + LN_enc) via CE + alignment + discrimination.

    Three loss components:
    - CE: discriminate my facts from distractors through stored trace
    - Alignment: cos(Q_fact_concept, Q_question_concept) → 1
    - Discrimination: hinge on cos(Q_my_concept, Q_dist_concept) → push below margin

    Returns:
        (losses, accuracies) per step
    """
    trace = model.trace
    gpt2 = model.gpt2
    wte = gpt2.transformer.wte
    wte_weight = wte.weight.detach()

    # Ensure encoder mode is ON
    trace.set_encoder_mode(True)

    # Freeze everything except W_enc and LN_enc
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trace.W_enc.parameters():
        p.requires_grad_(True)
    for p in trace.ln_enc.parameters():
        p.requires_grad_(True)

    trainable = list(trace.W_enc.parameters()) + list(
        trace.ln_enc.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.Adam(trainable, lr=lr)

    entity_ids = _get_all_entity_ids(fact_types)
    entity_tensor = torch.tensor(entity_ids, dtype=torch.long, device=device)
    linking_ids = trace.linking_token_ids or []
    linking_ids_set = set(linking_ids)

    parts = [f"λ_align={lambda_align}"]
    if lambda_disc > 0:
        parts.append(f"λ_disc={lambda_disc}, margin={disc_margin}")
    loss_str = ", ".join(parts)
    print(f"    {n_steps} steps, n_my={n_my}, n_dist={n_dist_per_type}/type, "
          f"beta={beta}, layer={context_layer}, lr={lr:.0e}")
    print(f"    Loss: CE + {loss_str}")
    print(f"    Trainable params: {n_trainable:,} (MLP encoder), "
          f"Entity pool: {len(entity_ids)}")

    losses: list[float] = []
    accuracies: list[float] = []
    rng = random.Random(seed)
    t0 = time.time()

    for step in range(n_steps):
        # 1. Generate episode (with distractor tracking if disc loss)
        if lambda_disc > 0 and n_dist_per_type > 0:
            episode = make_disc_episode(
                fact_types, n_my, n_dist_per_type,
                tokenizer, distractor_persons, linking_ids_set, rng)
        else:
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

        # 3. Compute Q, V with gradient through W_enc
        Q, V = trace.compute_qv(
            wte, fact_tensor, hidden_states=hidden_fact,
            beta=beta, train_ctx=True)
        S_f = Q.shape[2]

        # 4. Binary linking gate + write to differentiable trace
        gate = _make_binary_gate(fact_tensor, linking_ids, device)
        T_v_diff = trace.write_differentiable(Q, V, gate)

        # 5. CE loss + alignment loss (per query)
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

            # CE loss
            retrieved = trace.read_from_trace(Q_q, T_v_diff)
            trace_logits = torch.matmul(retrieved, wte_weight.T)

            pred = trace_logits[0, -1, entity_tensor]
            target_idx = (entity_tensor == answer_id).nonzero(
                as_tuple=True)[0]
            loss = F.cross_entropy(pred.unsqueeze(0), target_idx)
            ce_loss = ce_loss + loss

            if entity_tensor[pred.argmax()].item() == answer_id:
                n_correct += 1

            # Alignment loss
            if lambda_align > 0:
                concept_pos = episode.my_concept_positions[qi]
                S_q = Q_q.shape[2]
                if concept_pos < S_f and S_q >= 2:
                    Q_fact_c = Q[0, :, concept_pos, :]
                    Q_q_c = Q_q[0, :, -2, :]
                    cos = F.cosine_similarity(Q_fact_c, Q_q_c, dim=-1)
                    align_loss = align_loss + (1.0 - cos).mean()
                    n_align += 1

        # 6. Discrimination loss: push Q_my away from Q_dist
        disc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_disc = 0
        if (lambda_disc > 0 and n_dist_per_type > 0
                and hasattr(episode, 'dist_positions_by_type')):
            for qi in range(len(episode.my_facts)):
                my_pos = episode.my_concept_positions[qi]
                d_positions = episode.dist_positions_by_type[qi]
                if not d_positions or my_pos >= S_f:
                    continue
                Q_my = Q[0, :, my_pos, :]  # (H, d_trace)
                for dpos in d_positions:
                    if dpos >= S_f:
                        continue
                    Q_dist = Q[0, :, dpos, :]  # (H, d_trace)
                    # Hinge: penalize cos(Q_my, Q_dist) > margin
                    cos_d = F.cosine_similarity(Q_my, Q_dist, dim=-1)
                    hinge = (cos_d - disc_margin).clamp(min=0).mean()
                    disc_loss = disc_loss + hinge
                    n_disc += 1

        # 7. Total loss
        n_queries = max(len(episode.my_queries), 1)
        avg_ce = ce_loss / n_queries
        total_loss = avg_ce

        if n_align > 0 and lambda_align > 0:
            total_loss = total_loss + lambda_align * (align_loss / n_align)
        if n_disc > 0 and lambda_disc > 0:
            total_loss = total_loss + lambda_disc * (disc_loss / n_disc)

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


# ── Phase functions ──────────────────────────────────────────────────

def run_phase1_setup(device_str=None):
    """Phase 1: Load model, enable encoder, backward compat check."""
    print("=" * 65)
    print("EXP 23: Entorhinal Encoder — MLP Bottleneck (Pareto Breaker)")
    print("=" * 65)

    device = get_device(device_str)
    print(f"\nDevice: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

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

    # Print encoder architecture
    n_enc = sum(p.numel() for p in model.trace.W_enc.parameters())
    n_enc += sum(p.numel() for p in model.trace.ln_enc.parameters())
    n_ctx = sum(p.numel() for p in model.trace.W_ctx.parameters())
    n_ctx += sum(p.numel() for p in model.trace.ln_ctx.parameters())
    print(f"\nEncoder architecture: MLP bottleneck")
    print(f"  768 → {model.trace._d_enc_bottleneck} → 512 (ReLU)")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Linear W_ctx params: {n_ctx:,}")
    print(f"  Ratio: {n_enc/n_ctx:.2f}x (fewer is better)")

    # Backward compat check
    print(f"\n{'─' * 65}")
    print("PHASE 1: Backward Compatibility Check (β=0)")
    print(f"{'─' * 65}")

    episodes = make_gpt2_eval_episodes(
        n_episodes=10, n_facts=3,
        tokenizer=tokenizer, fact_types=fact_types, seed=42)

    # Check both encoder and non-encoder give same result at β=0
    acc_base = evaluate_cross_context_beta(
        model, episodes, fact_types, beta=0.0, context_layer=4)
    model.set_encoder_mode(True)
    acc_enc = evaluate_cross_context_beta(
        model, episodes, fact_types, beta=0.0, context_layer=4)
    print(f"  Cross-context (n=3, β=0, no encoder): {acc_base:.1%}")
    print(f"  Cross-context (n=3, β=0, encoder ON): {acc_enc:.1%}")
    if abs(acc_base - acc_enc) < 0.01:
        print("  ✓ Backward compat OK")
    else:
        print("  ⚠ WARNING: encoder at β=0 should match baseline")

    return model, tokenizer, fact_types, persons, device


def run_phase2_train(
    model, tokenizer, fact_types, persons,
    context_layer, device, n_steps_total=1000, seed=42,
):
    """Phase 2: Train MLP encoder with 4-stage curriculum."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 2: Train MLP Encoder ({n_steps_total} steps)")
    print(f"{'─' * 65}")

    # Enable encoder mode
    model.set_encoder_mode(True)

    # 4-stage curriculum: CE + alignment (disc counterproductive)
    # Disc loss fights CE loss through bottleneck → worse Pareto
    stages = [
        {"n_my": 3, "n_dist": 0, "lr": 3e-4, "lambda_align": 1.0,
         "label": "Stage 1: alignment-only (n_my=3)"},
        {"n_my": 3, "n_dist": 1, "lr": 1e-4, "lambda_align": 1.0,
         "label": "Stage 2: CE + alignment (n_dist=1)"},
        {"n_my": 5, "n_dist": 1, "lr": 3e-5, "lambda_align": 1.0,
         "label": "Stage 3: harder (n_my=5)"},
        {"n_my": 5, "n_dist": 2, "lr": 1e-5, "lambda_align": 1.0,
         "label": "Stage 4: full (n_dist=2)"},
    ]

    # Distribute steps: 20%, 30%, 30%, 20%
    step_fracs = [0.20, 0.30, 0.30, 0.20]
    step_counts = [max(int(n_steps_total * f), 50) for f in step_fracs]
    step_counts[-1] = n_steps_total - sum(step_counts[:-1])

    all_losses: list[float] = []
    all_accs: list[float] = []

    for i, (stage, n_steps) in enumerate(zip(stages, step_counts)):
        print(f"\n  --- {stage['label']} ({n_steps} steps) ---")

        losses, accs = train_encoder(
            model, tokenizer, fact_types, persons,
            n_steps=n_steps,
            n_my=stage["n_my"],
            n_dist_per_type=stage["n_dist"],
            beta=1.0,
            context_layer=context_layer,
            lr=stage["lr"],
            device=device,
            lambda_align=stage["lambda_align"],
            lambda_disc=stage.get("lambda_disc", 0.0),
            disc_margin=0.0,
            log_every=max(n_steps // 4, 25),
            seed=seed + i * 10000,
            grad_clip=1.0,
        )
        all_losses.extend(losses)
        all_accs.extend(accs)

    return all_losses, all_accs


def run_phase3_beta_sweep(
    model, tokenizer, fact_types, n_eval, context_layer, seed=42,
    betas: list[float] | None = None,
) -> list[BetaResult]:
    """Phase 3: β sweep with trained encoder."""
    if betas is None:
        betas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    print(f"\n{'─' * 65}")
    print(f"PHASE 3: Trained Encoder β Sweep (n_eval={n_eval})")
    print(f"  Betas: {betas}")
    print(f"{'─' * 65}")

    # Ensure encoder mode is ON
    model.set_encoder_mode(True)

    variants = build_question_variants(tokenizer)
    results: list[BetaResult] = []

    for beta in betas:
        print(f"\n  β = {beta:.2f}...")
        t0 = time.time()

        # Cross-context n=3, n=5
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

        # Distractors (1 per type)
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


def run_linear_comparison(
    model, tokenizer, fact_types, persons,
    context_layer, device, n_eval, n_steps_total=1000, seed=42,
) -> list[BetaResult]:
    """Train linear W_ctx for head-to-head comparison.

    Creates a fresh model (same init), trains W_ctx (not encoder),
    and runs β sweep.
    """
    print(f"\n{'─' * 65}")
    print("COMPARISON: Training Linear W_ctx")
    print(f"{'─' * 65}")

    # Import exp14 training
    from .exp14_contrastive_ctx import train_contrastive_ctx

    # Fresh model (same config)
    model_lin = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    ).to(device)
    model_lin.set_linking_token_ids(
        get_linking_bpe_ids(tokenizer))
    model_lin.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model_lin.eval()

    # Encoder mode OFF (use linear W_ctx)
    model_lin.set_encoder_mode(False)

    # Same 4-stage curriculum
    stages = [
        {"n_my": 3, "n_dist": 0, "lr": 3e-4, "lambda_align": 1.0,
         "label": "Stage 1: alignment-only"},
        {"n_my": 3, "n_dist": 1, "lr": 1e-4, "lambda_align": 1.0,
         "label": "Stage 2: CE + alignment"},
        {"n_my": 5, "n_dist": 1, "lr": 3e-5, "lambda_align": 1.0,
         "label": "Stage 3: harder"},
        {"n_my": 5, "n_dist": 2, "lr": 1e-5, "lambda_align": 1.0,
         "label": "Stage 4: full"},
    ]
    step_fracs = [0.20, 0.30, 0.30, 0.20]
    step_counts = [max(int(n_steps_total * f), 50) for f in step_fracs]
    step_counts[-1] = n_steps_total - sum(step_counts[:-1])

    for i, (stage, n_steps) in enumerate(zip(stages, step_counts)):
        print(f"\n  --- {stage['label']} ({n_steps} steps) ---")
        train_contrastive_ctx(
            model_lin, tokenizer, fact_types, persons,
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

    # β sweep
    print(f"\n  Linear W_ctx trained. Running β sweep...")
    betas = [0.0, 0.3, 0.4, 0.5, 0.7, 1.0]
    variants = build_question_variants(tokenizer)
    results: list[BetaResult] = []

    for beta in betas:
        eps_5 = make_gpt2_eval_episodes(
            n_eval, 5, tokenizer, fact_types, seed=seed + 1000)
        cross_5 = evaluate_cross_context_beta(
            model_lin, eps_5, fact_types, beta, context_layer)
        dist_acc, confusion = evaluate_distractors_beta(
            model_lin, tokenizer, fact_types, n_eval,
            n_my_facts=5, n_distractors_per_type=1,
            beta=beta, context_layer=context_layer, seed=seed)
        results.append(BetaResult(
            beta=beta, context_layer=context_layer,
            cross_ctx_n3=0, cross_ctx_n5=cross_5,
            aligned=0, misaligned=0, semantic=0,
            dist_accuracy=dist_acc, dist_confusion=confusion,
        ))
        print(f"    β={beta:.1f}: cross={cross_5:.1%}, "
              f"confusion={confusion:.1%}")

    return results


# ── Summary ──────────────────────────────────────────────────────────

def print_summary(enc_results: list[BetaResult],
                  lin_results: list[BetaResult] | None = None):
    """Print Pareto comparison."""
    print(f"\n{'═' * 75}")
    print("SUMMARY: MLP Encoder Pareto Frontier")
    print(f"{'═' * 75}")

    print(f"\n  MLP ENCODER (trained, 166K params):")
    print(f"  {'β':>5s} │ {'Cross n=3':>9s} │ {'Cross n=5':>9s} │ "
          f"{'Aligned':>8s} │ {'Confusn':>8s} │ {'DistAcc':>8s}")
    print(f"  {'─' * 5}─┼{'─' * 11}┼{'─' * 11}┼"
          f"{'─' * 10}┼{'─' * 10}┼{'─' * 10}")
    for r in enc_results:
        print(f"  {r.beta:5.2f} │ {r.cross_ctx_n3:9.1%} │ "
              f"{r.cross_ctx_n5:9.1%} │ {r.aligned:8.1%} │ "
              f"{r.dist_confusion:8.1%} │ {r.dist_accuracy:8.1%}")

    # Find Pareto optimal point
    baseline = enc_results[0]  # β=0
    viable = [r for r in enc_results if r.cross_ctx_n5 >= 0.80]
    if viable:
        best = min(viable, key=lambda r: r.dist_confusion)
        print(f"\n  Best with cross≥80%: β={best.beta:.2f} → "
              f"cross={best.cross_ctx_n5:.1%}, "
              f"confusion={best.dist_confusion:.1%}")
    else:
        print(f"\n  No β achieves cross≥80%")
        best_cross = max(enc_results, key=lambda r: r.cross_ctx_n5)
        best_conf = min(enc_results, key=lambda r: r.dist_confusion)
        print(f"  Best cross: β={best_cross.beta:.2f} → "
              f"cross={best_cross.cross_ctx_n5:.1%}")
        print(f"  Best confusion: β={best_conf.beta:.2f} → "
              f"confusion={best_conf.dist_confusion:.1%}")

    if lin_results:
        print(f"\n  HEAD-TO-HEAD: MLP Encoder vs Linear W_ctx")
        print(f"  {'β':>5s} │ {'MLP cross5':>10s} │ {'MLP conf':>8s} │ "
              f"{'Lin cross5':>10s} │ {'Lin conf':>8s} │ {'Δconf':>8s}")
        print(f"  {'─' * 5}─┼{'─' * 12}┼{'─' * 10}┼"
              f"{'─' * 12}┼{'─' * 10}┼{'─' * 10}")

        enc_map = {r.beta: r for r in enc_results}
        lin_map = {r.beta: r for r in lin_results}
        common = sorted(set(enc_map) & set(lin_map))
        for beta in common:
            e = enc_map[beta]
            l = lin_map[beta]
            d = e.dist_confusion - l.dist_confusion
            print(f"  {beta:5.2f} │ {e.cross_ctx_n5:10.1%} │ "
                  f"{e.dist_confusion:8.1%} │ {l.cross_ctx_n5:10.1%} │ "
                  f"{l.dist_confusion:8.1%} │ {d:+8.1%}")

        # Did we break Pareto?
        enc_viable = [r for r in enc_results
                      if r.cross_ctx_n5 >= 0.80 and r.dist_confusion < 0.30]
        lin_viable = [r for r in lin_results
                      if r.cross_ctx_n5 >= 0.80 and r.dist_confusion < 0.30]

        print(f"\n  TARGET: cross≥80% AND confusion<30%")
        if enc_viable:
            best_enc = min(enc_viable, key=lambda r: r.dist_confusion)
            print(f"  ✓ MLP ENCODER: β={best_enc.beta:.2f} → "
                  f"cross={best_enc.cross_ctx_n5:.1%}, "
                  f"conf={best_enc.dist_confusion:.1%}  ← PARETO BROKEN!")
        else:
            print(f"  ✗ MLP encoder: no β in target zone")
        if lin_viable:
            best_lin = min(lin_viable, key=lambda r: r.dist_confusion)
            print(f"  ✓ Linear W_ctx: β={best_lin.beta:.2f} → "
                  f"cross={best_lin.cross_ctx_n5:.1%}, "
                  f"conf={best_lin.dist_confusion:.1%}")
        else:
            print(f"  ✗ Linear W_ctx: no β in target zone (as expected)")

    print(f"{'═' * 75}")


# ── Q similarity diagnostic ──────────────────────────────────────────

@torch.no_grad()
def diagnose_q_separation(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    distractor_persons: list[tuple[str, int]],
    context_layer: int,
    beta: float,
    device: torch.device,
    n_samples: int = 30,
    seed: int = 42,
) -> tuple[float, float]:
    """Measure Q cosine sim: same-person vs cross-person.

    Higher gap = better discrimination.
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

        def _get_concept_q(token_ids: list[int]) -> torch.Tensor | None:
            t = torch.tensor([token_ids], dtype=torch.long, device=device)
            out = gpt2(t, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[context_layer]
            Q, _ = trace.compute_qv(wte, t, hidden_states=hidden, beta=beta)
            for pos in range(1, len(token_ids)):
                if token_ids[pos] in linking_ids:
                    return Q[0, :, pos - 1, :].flatten()
            return None

        q_my = _get_concept_q(my_fact_ids)
        q_question = _get_concept_q(q_ids)
        q_dist = _get_concept_q(d_ids)

        if q_my is None or q_question is None or q_dist is None:
            continue

        sim_same = F.cosine_similarity(
            q_my.unsqueeze(0), q_question.unsqueeze(0)).item()
        sim_cross = F.cosine_similarity(
            q_my.unsqueeze(0), q_dist.unsqueeze(0)).item()
        same_sims.append(sim_same)
        cross_sims.append(sim_cross)

    avg_same = sum(same_sims) / max(len(same_sims), 1)
    avg_cross = sum(cross_sims) / max(len(cross_sims), 1)
    gap = avg_same - avg_cross
    print(f"  Q similarity: same={avg_same:.3f}, "
          f"cross={avg_cross:.3f}, gap={gap:+.3f}")
    return avg_same, avg_cross


# ── Entry points ──────────────────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick experiment (~8-15 min)."""
    t0 = time.time()
    context_layer = 4  # L3 (best from exp13)

    # Phase 1: Setup
    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Q diagnostic (before training)
    print(f"\n  Q separation (random encoder, β=1.0):")
    diagnose_q_separation(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev)

    # Phase 2: Train encoder (quick: 1000 steps)
    run_phase2_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1000, seed=seed)

    # Q diagnostic (after training)
    print(f"\n  Q separation (trained encoder, β=1.0):")
    diagnose_q_separation(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev)

    # Phase 3: β sweep
    enc_results = run_phase3_beta_sweep(
        model, tokenizer, fact_types, n_eval=15,
        context_layer=context_layer, seed=seed)

    # Summary
    print_summary(enc_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_full(device=None, seed=42, n_eval=30):
    """Full experiment (~20-35 min)."""
    t0 = time.time()
    context_layer = 4  # L3

    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Q diagnostic (before)
    print(f"\n  Q separation (random encoder, β=1.0):")
    diagnose_q_separation(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev)

    # Train (1500 steps)
    run_phase2_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1500, seed=seed)

    # Q diagnostic (after)
    print(f"\n  Q separation (trained encoder, β=1.0):")
    diagnose_q_separation(
        model, tokenizer, fact_types, persons,
        context_layer, beta=1.0, device=dev)

    # β sweep
    enc_results = run_phase3_beta_sweep(
        model, tokenizer, fact_types, n_eval=n_eval,
        context_layer=context_layer, seed=seed,
        betas=[0.0, 0.2, 0.3, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 1.0])

    print_summary(enc_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_with_comparison(device=None, seed=42, n_eval=30):
    """Full experiment + linear W_ctx comparison (~35-50 min)."""
    t0 = time.time()
    context_layer = 4

    model, tokenizer, fact_types, persons, dev = run_phase1_setup(device)

    # Train MLP encoder
    run_phase2_train(
        model, tokenizer, fact_types, persons,
        context_layer=context_layer, device=dev,
        n_steps_total=1500, seed=seed)

    # MLP β sweep
    enc_results = run_phase3_beta_sweep(
        model, tokenizer, fact_types, n_eval=n_eval,
        context_layer=context_layer, seed=seed,
        betas=[0.0, 0.3, 0.4, 0.5, 0.7, 1.0])

    # Linear W_ctx comparison
    lin_results = run_linear_comparison(
        model, tokenizer, fact_types, persons,
        context_layer, dev, n_eval=n_eval,
        n_steps_total=1500, seed=seed)

    print_summary(enc_results, lin_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 23: Entorhinal Encoder — MLP Bottleneck")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (15 episodes, 1000 steps)")
    parser.add_argument("--compare-linear", action="store_true",
                        help="Include linear W_ctx comparison")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=30)
    args = parser.parse_args()

    if args.compare_linear:
        run_with_comparison(
            device=args.device, seed=args.seed, n_eval=args.n_eval)
    elif args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run_full(device=args.device, seed=args.seed, n_eval=args.n_eval)
