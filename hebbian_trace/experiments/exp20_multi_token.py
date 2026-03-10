"""Experiment 20: Multi-Token Entity Support.

Tests whether Hebbian trace memory can recall multi-token entities
(e.g., "San Francisco", "New Zealand") via auto-regressive generation.

The trace stores V of the FIRST entity token (shift-2 mechanism naturally
picks this up). After trace boosts the first token, GPT-2's language model
completes the rest auto-regressively.

Phase 1: Validation & Diagnostics
  - BPE tokenization of multi-token entities
  - GPT-2 completion probability (P(token_2 | question + token_1))
  - Curate valid entity pools

Phase 2: Cross-Context Recall
  - n_facts = 1, 3, 5, 7
  - Metrics: first_token_accuracy, full_entity_accuracy
  - Single-token only vs mixed (single + multi)

Phase 3: Pattern Separation Impact
  - With/without 8x_k16

Phase 4: Per-Entity Analysis
  - Breakdown by entity: first-token and completion accuracy

Phase 5: Alpha Sweep

Usage:
    python -m hebbian_trace.experiments.exp20_multi_token --quick
    python -m hebbian_trace.experiments.exp20_multi_token --n-eval 50
    python -m hebbian_trace.experiments.exp20_multi_token --validate-only
"""

import argparse
import random
import time
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    GPT2FactTemplate, GPT2QuestionTemplate,
    get_linking_bpe_ids, validate_entities, validate_single_token_entities,
    tokenize_fact, tokenize_question,
    MULTI_CITIES, MULTI_COMPANIES, MULTI_COUNTRIES,
    MULTI_CITIES_3, MULTI_COUNTRIES_3,
)
from ..nlp_tasks import (
    NAMES, CITIES, COMPANIES, COLORS, FOODS, PETS, COUNTRIES,
)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Multi-token data structures ─────────────────────────────────────

@dataclass
class MultiTokenFactType:
    """Fact type supporting multi-token entities."""
    name: str
    entities: list[tuple[str, list[int]]]  # (entity_name, bpe_token_ids)
    fact_templates: list[GPT2FactTemplate]
    question_templates: list[GPT2QuestionTemplate]


@dataclass
class MultiTokenEvalEpisode:
    """Evaluation episode with multi-token entity support."""
    facts: list[tuple[str, str, list[int], list[int]]]
    # (type_name, entity_name, entity_bpe_ids, fact_bpe_ids)
    train_sequences: list[list[int]]
    test_queries: list[tuple[list[int], list[int], str]]
    # (query_bpe_ids, answer_bpe_ids, type_name)


@dataclass
class MultiTokenEvalResults:
    """Results with both first-token and full-entity accuracy."""
    first_token_accuracy: float
    full_entity_accuracy: float
    first_correct: int
    full_correct: int
    n_total: int
    per_query: list[dict] = field(default_factory=list)


# ── Fact type construction ──────────────────────────────────────────

def build_multi_token_fact_types(
    tokenizer: GPT2Tokenizer,
    include_single: bool = True,
    include_multi: bool = True,
    max_tokens: int = 3,
    min_entities: int = 3,
) -> list[MultiTokenFactType]:
    """Build fact types with single and/or multi-token entities.

    Args:
        include_single: include standard single-token entities
        include_multi: include multi-token entities
        max_tokens: max BPE tokens per entity
        min_entities: minimum entities needed to include a fact type
    """
    # (type_name, single_pool, multi_2tok_pool, multi_3tok_pool, ftemplates, qtemplates)
    raw_types = [
        ("name", NAMES, [], [], [
            GPT2FactTemplate("My name is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my name?"),
        ]),
        ("city", CITIES, MULTI_CITIES, MULTI_CITIES_3, [
            GPT2FactTemplate("I live in {X}.", "in"),
        ], [
            GPT2QuestionTemplate("Where do I live?"),
        ]),
        ("company", COMPANIES, MULTI_COMPANIES, [], [
            GPT2FactTemplate("I work at {X}.", "at"),
        ], [
            GPT2QuestionTemplate("Where do I work?"),
        ]),
        ("color", COLORS, [], [], [
            GPT2FactTemplate("My favorite color is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my favorite color?"),
        ]),
        ("food", FOODS, [], [], [
            GPT2FactTemplate("My favorite food is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my favorite food?"),
        ]),
        ("pet", PETS, [], [], [
            GPT2FactTemplate("My pet is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my pet?"),
        ]),
        ("country", COUNTRIES, MULTI_COUNTRIES, MULTI_COUNTRIES_3, [
            GPT2FactTemplate("My country is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my country?"),
        ]),
    ]

    fact_types = []
    for name, single_pool, multi_pool, multi3_pool, ftemplates, qtemplates \
            in raw_types:
        entities: list[tuple[str, list[int]]] = []

        if include_single:
            for ename, eid in validate_single_token_entities(
                    tokenizer, single_pool):
                entities.append((ename, [eid]))

        if include_multi and multi_pool:
            entities.extend(
                validate_entities(tokenizer, multi_pool,
                                  max_tokens=max_tokens))

        if include_multi and multi3_pool:
            entities.extend(
                validate_entities(tokenizer, multi3_pool,
                                  max_tokens=max_tokens))

        if len(entities) >= min_entities:
            fact_types.append(MultiTokenFactType(
                name=name,
                entities=entities,
                fact_templates=ftemplates,
                question_templates=qtemplates,
            ))

    return fact_types


# ── Episode generation ──────────────────────────────────────────────

def make_multi_token_episodes(
    n_episodes: int,
    n_facts: int,
    tokenizer: GPT2Tokenizer,
    fact_types: list[MultiTokenFactType],
    seed: int = 42,
) -> list[MultiTokenEvalEpisode]:
    """Generate eval episodes with multi-token entity support."""
    rng = random.Random(seed)
    episodes = []

    for _ in range(n_episodes):
        if n_facts <= len(fact_types):
            selected_types = rng.sample(fact_types, n_facts)
        else:
            selected_types = [rng.choice(fact_types) for _ in range(n_facts)]

        facts = []
        for ft in selected_types:
            entity_name, entity_ids = rng.choice(ft.entities)
            template = rng.choice(ft.fact_templates)
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            facts.append((ft.name, entity_name, entity_ids, fact_ids))

        # Cumulative training sequences
        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        train_sequences = []
        for i in range(len(facts)):
            cumulative = []
            for j in range(i + 1):
                if cumulative:
                    cumulative.append(space_id)
                cumulative.extend(facts[j][3])
            train_sequences.append(cumulative)

        # Test queries
        test_queries = []
        for ft, (type_name, entity_name, entity_ids, _) in zip(
                selected_types, facts):
            q_template = rng.choice(ft.question_templates)
            q_ids = tokenize_question(tokenizer, q_template)
            test_queries.append((q_ids, entity_ids, type_name))

        episodes.append(MultiTokenEvalEpisode(
            facts=facts,
            train_sequences=train_sequences,
            test_queries=test_queries,
        ))

    return episodes


# ── Evaluation helpers ──────────────────────────────────────────────

def _get_all_first_token_ids(
    fact_types: list[MultiTokenFactType],
) -> list[int]:
    """Get all unique first-token entity BPE IDs."""
    ids = set()
    for ft in fact_types:
        for _, entity_ids in ft.entities:
            ids.add(entity_ids[0])
    return sorted(ids)


def _predict_first_token(model, query_ids: list[int],
                         first_token_ids: list[int]) -> int:
    """Predict first entity token (restricted to known first tokens)."""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
    pred_logits = logits[0, -1, :]
    entity_logits = pred_logits[first_token_ids]
    best_pos = entity_logits.argmax().item()
    return first_token_ids[best_pos]


def _generate_answer(model, query_ids: list[int],
                     first_token_ids: list[int],
                     max_new_tokens: int = 4,
                     stop_token_ids: list[int] | None = None,
                     ) -> list[int]:
    """Generate multi-token answer using auto-regressive decoding.

    First token restricted to known entity first-tokens.
    Subsequent tokens unrestricted (GPT-2 LM completes).
    """
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        restrict_first_to=first_token_ids,
        stop_token_ids=stop_token_ids,
    )
    return generated[0].tolist()


def _match_entity(generated: list[int], expected: list[int]) -> tuple[bool, bool]:
    """Check first-token and full-entity match.

    Returns (first_token_match, full_entity_match).
    """
    first_match = len(generated) > 0 and generated[0] == expected[0]
    full_match = generated[:len(expected)] == expected
    return first_match, full_match


# ── Main evaluation functions ───────────────────────────────────────

def evaluate_cross_context(
    model, episodes: list[MultiTokenEvalEpisode],
    fact_types: list[MultiTokenFactType],
    tokenizer: GPT2Tokenizer,
    verbose: bool = False,
) -> MultiTokenEvalResults:
    """Cross-context eval with multi-token generation."""
    model.eval()
    first_token_ids = _get_all_first_token_ids(fact_types)
    # Stop tokens: period, common sentence endings
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]
    stop_tokens = [period_id]

    first_correct = 0
    full_correct = 0
    n_total = 0
    per_query = []

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase
        model.set_trace_mode(use=False, update=True)
        device = next(model.parameters()).device
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)

        # Read phase
        model.set_trace_mode(use=True, update=False)
        for query_ids, answer_ids, type_name in episode.test_queries:
            # Determine max tokens needed
            max_new = max(len(answer_ids) + 1, 4)
            generated = _generate_answer(
                model, query_ids, first_token_ids,
                max_new_tokens=max_new, stop_token_ids=stop_tokens)

            first_match, full_match = _match_entity(generated, answer_ids)
            if first_match:
                first_correct += 1
            if full_match:
                full_correct += 1
            n_total += 1

            per_query.append({
                'type': type_name,
                'expected': answer_ids,
                'generated': generated,
                'first_match': first_match,
                'full_match': full_match,
                'n_tokens': len(answer_ids),
            })

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: "
                  f"first={first_correct}/{n_total} "
                  f"full={full_correct}/{n_total}")

    return MultiTokenEvalResults(
        first_token_accuracy=first_correct / max(n_total, 1),
        full_entity_accuracy=full_correct / max(n_total, 1),
        first_correct=first_correct,
        full_correct=full_correct,
        n_total=n_total,
        per_query=per_query,
    )


def evaluate_cross_context_baseline(
    model, episodes: list[MultiTokenEvalEpisode],
    fact_types: list[MultiTokenFactType],
    tokenizer: GPT2Tokenizer,
    verbose: bool = False,
) -> MultiTokenEvalResults:
    """Cross-context baseline: no trace, question-only."""
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    first_token_ids = _get_all_first_token_ids(fact_types)
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]
    stop_tokens = [period_id]

    first_correct = 0
    full_correct = 0
    n_total = 0
    per_query = []

    for episode in episodes:
        for query_ids, answer_ids, type_name in episode.test_queries:
            max_new = max(len(answer_ids) + 1, 4)
            generated = _generate_answer(
                model, query_ids, first_token_ids,
                max_new_tokens=max_new, stop_token_ids=stop_tokens)

            first_match, full_match = _match_entity(generated, answer_ids)
            if first_match:
                first_correct += 1
            if full_match:
                full_correct += 1
            n_total += 1

            per_query.append({
                'type': type_name,
                'expected': answer_ids,
                'generated': generated,
                'first_match': first_match,
                'full_match': full_match,
                'n_tokens': len(answer_ids),
            })

    return MultiTokenEvalResults(
        first_token_accuracy=first_correct / max(n_total, 1),
        full_entity_accuracy=full_correct / max(n_total, 1),
        first_correct=first_correct,
        full_correct=full_correct,
        n_total=n_total,
        per_query=per_query,
    )


# ── Phase 1: Validation & Diagnostics ──────────────────────────────

def run_phase1_validation(tokenizer, device):
    """Validate multi-token entities and GPT-2 completion rates."""
    print(f"\n{'=' * 65}")
    print("PHASE 1: Multi-Token Entity Validation")
    print(f"{'=' * 65}")

    # Load base GPT-2 for completion check
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.eval().to(device)

    all_pools = [
        ("city", CITIES, MULTI_CITIES + MULTI_CITIES_3,
         "Where do I live?", "I live in {X}."),
        ("company", COMPANIES, MULTI_COMPANIES,
         "Where do I work?", "I work at {X}."),
        ("country", COUNTRIES, MULTI_COUNTRIES + MULTI_COUNTRIES_3,
         "What is my country?", "My country is {X}."),
    ]

    valid_multi: dict[str, list[tuple[str, list[int]]]] = {}

    for type_name, single_pool, multi_pool, question, fact_tmpl in all_pools:
        print(f"\n--- {type_name} ---")

        # Single-token entities
        single = validate_single_token_entities(tokenizer, single_pool)
        print(f"  Single-token: {len(single)} valid")

        # Multi-token entities
        multi = validate_entities(tokenizer, multi_pool, max_tokens=3)
        print(f"  Multi-token candidates: {len(multi)}")

        valid_for_type = []
        for entity_name, entity_ids in multi:
            tokens_str = [tokenizer.decode([tid]) for tid in entity_ids]
            n_tokens = len(entity_ids)

            # Check GPT-2 completion probability
            completion_prob = _check_completion(
                base_model, tokenizer, question, entity_ids, device)

            status = "PASS" if completion_prob >= 0.20 else "FAIL"
            print(f"    {entity_name:20s} → {tokens_str} "
                  f"({n_tokens} tok) "
                  f"P(complete)={completion_prob:.1%} [{status}]")

            if completion_prob >= 0.20:
                valid_for_type.append((entity_name, entity_ids))

        valid_multi[type_name] = valid_for_type
        print(f"  Valid multi-token: {len(valid_for_type)}")

    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return valid_multi


def _check_completion(model, tokenizer, question: str,
                      entity_ids: list[int], device) -> float:
    """Check if GPT-2 completes multi-token entity from question + first token.

    Returns chain probability: P(t2|t1) * P(t3|t1,t2) * ...
    For single-token entities returns 1.0.
    """
    if len(entity_ids) <= 1:
        return 1.0

    q_ids = tokenizer.encode(question, add_special_tokens=False)
    chain_prob = 1.0

    for step in range(1, len(entity_ids)):
        prompt_ids = q_ids + entity_ids[:step]
        input_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor).logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        chain_prob *= probs[entity_ids[step]].item()

    return chain_prob


# ── Phase 2: Cross-Context Recall ──────────────────────────────────

def run_phase2_cross_context(model, tokenizer, n_eval, n_facts_list,
                             verbose=False):
    """Test cross-context recall with mixed entity pools."""
    print(f"\n{'=' * 65}")
    print("PHASE 2: Cross-Context Recall (mixed single + multi)")
    print(f"{'=' * 65}")

    # Build fact types with both single and multi-token entities
    fact_types = build_multi_token_fact_types(
        tokenizer, include_single=True, include_multi=True)

    print(f"\nFact types: {len(fact_types)}")
    for ft in fact_types:
        n_single = sum(1 for _, ids in ft.entities if len(ids) == 1)
        n_multi = sum(1 for _, ids in ft.entities if len(ids) > 1)
        print(f"  {ft.name}: {n_single} single + {n_multi} multi "
              f"= {len(ft.entities)} total")

    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    results = {}
    for n_facts in n_facts_list:
        print(f"\n--- n_facts = {n_facts} ---")
        episodes = make_multi_token_episodes(
            n_episodes=n_eval, n_facts=n_facts,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=42 + n_facts * 1000)

        t0 = time.time()

        # Cross-context with trace
        cc = evaluate_cross_context(
            model, episodes, fact_types, tokenizer, verbose=verbose)

        # Cross-context baseline
        cc_bl = evaluate_cross_context_baseline(
            model, episodes, fact_types, tokenizer, verbose=verbose)

        dt = time.time() - t0

        print(f"  Cross+trace:  first={cc.first_token_accuracy:.1%}  "
              f"full={cc.full_entity_accuracy:.1%}")
        print(f"  Cross BL:     first={cc_bl.first_token_accuracy:.1%}  "
              f"full={cc_bl.full_entity_accuracy:.1%}")
        gap_first = cc.first_token_accuracy - cc_bl.first_token_accuracy
        gap_full = cc.full_entity_accuracy - cc_bl.full_entity_accuracy
        print(f"  Gap:          first={gap_first:+.1%}  "
              f"full={gap_full:+.1%}")
        print(f"  Time: {dt:.1f}s")

        results[n_facts] = {'cross_ctx': cc, 'cross_bl': cc_bl}

    # Summary table
    print(f"\n{'=' * 65}")
    print("PHASE 2 SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'n':>3} │ {'First Tok':>10} {'Full Ent':>10} "
          f"{'BL First':>10} {'BL Full':>10} │ "
          f"{'Gap 1st':>8} {'Gap Full':>8}")
    print(f"{'─' * 3}─┼─{'─' * 10}─{'─' * 10}─"
          f"{'─' * 10}─{'─' * 10}─┼─{'─' * 8}─{'─' * 8}")
    for n_facts in n_facts_list:
        r = results[n_facts]
        cc, bl = r['cross_ctx'], r['cross_bl']
        g1 = cc.first_token_accuracy - bl.first_token_accuracy
        gf = cc.full_entity_accuracy - bl.full_entity_accuracy
        print(f"{n_facts:3d} │ {cc.first_token_accuracy:>10.1%} "
              f"{cc.full_entity_accuracy:>10.1%} "
              f"{bl.first_token_accuracy:>10.1%} "
              f"{bl.full_entity_accuracy:>10.1%} │ "
              f"{g1:>+7.1%} {gf:>+7.1%}")

    return results


# ── Phase 3: Pattern Separation ────────────────────────────────────

def run_phase3_pattern_sep(model, tokenizer, n_eval, verbose=False):
    """Test pattern separation impact on multi-token recall."""
    print(f"\n{'=' * 65}")
    print("PHASE 3: Pattern Separation Impact")
    print(f"{'=' * 65}")

    fact_types = build_multi_token_fact_types(
        tokenizer, include_single=True, include_multi=True)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    test_ns = [1, 3, 5, 7]
    configs = [
        ("No PS", False),
        ("8x_k16", True),
    ]

    all_results = {}
    for config_name, use_ps in configs:
        print(f"\n--- {config_name} ---")
        if use_ps:
            model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
        else:
            model.disable_pattern_separation()

        config_results = {}
        for n_facts in test_ns:
            episodes = make_multi_token_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                seed=42 + n_facts * 1000)

            cc = evaluate_cross_context(
                model, episodes, fact_types, tokenizer, verbose=verbose)
            config_results[n_facts] = cc
            print(f"  n={n_facts}: first={cc.first_token_accuracy:.1%}  "
                  f"full={cc.full_entity_accuracy:.1%}")

        all_results[config_name] = config_results

    # Summary
    print(f"\n{'=' * 65}")
    print("PHASE 3 SUMMARY (full entity accuracy)")
    print(f"{'=' * 65}")
    header = f"{'n':>3} │ " + " ".join(
        f"{name:>12}" for name, _ in configs) + " │ Delta"
    print(header)
    print("─" * len(header))
    for n in test_ns:
        vals = [all_results[name][n].full_entity_accuracy
                for name, _ in configs]
        delta = vals[1] - vals[0] if len(vals) == 2 else 0
        row = f"{n:3d} │ " + " ".join(f"{v:>12.1%}" for v in vals)
        row += f" │ {delta:+.1%}"
        print(row)

    return all_results


# ── Phase 4: Per-Entity Analysis ───────────────────────────────────

def run_phase4_per_entity(model, tokenizer, n_eval, verbose=False):
    """Detailed per-entity breakdown of first-token and completion accuracy."""
    print(f"\n{'=' * 65}")
    print("PHASE 4: Per-Entity Analysis")
    print(f"{'=' * 65}")

    fact_types = build_multi_token_fact_types(
        tokenizer, include_single=True, include_multi=True)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    # Use n=5 for meaningful multi-type interaction
    episodes = make_multi_token_episodes(
        n_episodes=n_eval, n_facts=5,
        tokenizer=tokenizer, fact_types=fact_types,
        seed=42 + 5 * 1000)

    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    result = evaluate_cross_context(
        model, episodes, fact_types, tokenizer, verbose=verbose)

    # Aggregate by entity token count (1, 2, 3+)
    by_ntok: dict[int, dict] = {}
    for q in result.per_query:
        nt = q['n_tokens']
        bucket = nt if nt <= 3 else 3  # group 3+ together
        if bucket not in by_ntok:
            by_ntok[bucket] = {'first': 0, 'full': 0, 'n': 0}
        by_ntok[bucket]['n'] += 1
        by_ntok[bucket]['first'] += int(q['first_match'])
        by_ntok[bucket]['full'] += int(q['full_match'])

    for nt in sorted(by_ntok.keys()):
        s = by_ntok[nt]
        label = f"{nt}-token" if nt < 3 else f"{nt}+-token"
        print(f"\n  {label} entities (n={s['n']}):")
        if s['n']:
            print(f"    First-token: {s['first']/s['n']:.1%}")
            print(f"    Full-entity: {s['full']/s['n']:.1%}")
            if nt >= 2:
                completion = s['full'] / max(s['first'], 1)
                print(f"    Completion rate (full|first correct): "
                      f"{completion:.1%}")

    # Combined multi-token (2+)
    multi_first = sum(s['first'] for nt, s in by_ntok.items() if nt >= 2)
    multi_full = sum(s['full'] for nt, s in by_ntok.items() if nt >= 2)
    multi_n = sum(s['n'] for nt, s in by_ntok.items() if nt >= 2)
    single_n = by_ntok.get(1, {}).get('n', 0)

    print(f"\n  All multi-token combined (n={multi_n}):")
    if multi_n:
        print(f"    First-token: {multi_first/multi_n:.1%}")
        print(f"    Full-entity: {multi_full/multi_n:.1%}")
        completion_rate = multi_full / max(multi_first, 1)
        print(f"    Completion rate (full|first correct): "
              f"{completion_rate:.1%}")

    # Per-type breakdown
    type_stats: dict[str, dict] = {}
    for q in result.per_query:
        t = q['type']
        if t not in type_stats:
            type_stats[t] = {'first': 0, 'full': 0, 'n': 0,
                             'single': 0, 'multi': 0}
        type_stats[t]['n'] += 1
        type_stats[t]['first'] += int(q['first_match'])
        type_stats[t]['full'] += int(q['full_match'])
        if q['n_tokens'] == 1:
            type_stats[t]['single'] += 1
        else:
            type_stats[t]['multi'] += 1

    print(f"\n  Per-type breakdown:")
    print(f"  {'Type':>10} │ {'Single':>6} {'Multi':>6} │ "
          f"{'First':>7} {'Full':>7}")
    print(f"  {'─' * 10}─┼─{'─' * 6}─{'─' * 6}─┼─{'─' * 7}─{'─' * 7}")
    for t, s in sorted(type_stats.items()):
        first_acc = s['first'] / max(s['n'], 1)
        full_acc = s['full'] / max(s['n'], 1)
        print(f"  {t:>10} │ {s['single']:>6} {s['multi']:>6} │ "
              f"{first_acc:>7.1%} {full_acc:>7.1%}")

    # Show example failures
    failures = [q for q in result.per_query
                if q['first_match'] and not q['full_match']]
    if failures:
        print(f"\n  Completion failures (first OK, full wrong): "
              f"{len(failures)}")
        for q in failures[:5]:
            exp_str = tokenizer.decode(q['expected'])
            gen_str = tokenizer.decode(q['generated'][:len(q['expected'])+1])
            print(f"    {q['type']:>10}: expected '{exp_str}', "
                  f"got '{gen_str}'")

    return result


# ── Phase 5: Alpha Sweep ──────────────────────────────────────────

def run_phase5_alpha_sweep(model, tokenizer, n_eval, verbose=False):
    """Test alpha sensitivity for multi-token recall."""
    print(f"\n{'=' * 65}")
    print("PHASE 5: Alpha Sweep")
    print(f"{'=' * 65}")

    fact_types = build_multi_token_fact_types(
        tokenizer, include_single=True, include_multi=True)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    original_alpha = model.trace.alpha
    alphas = [0.1, 0.5, 1.0, 2.0]
    test_ns = [1, 5]

    all_results = {}
    for alpha in alphas:
        model.trace.alpha = alpha
        print(f"\n--- alpha = {alpha} ---")
        all_results[alpha] = {}

        for n_facts in test_ns:
            episodes = make_multi_token_episodes(
                n_episodes=n_eval, n_facts=n_facts,
                tokenizer=tokenizer, fact_types=fact_types,
                seed=42 + n_facts * 1000)

            cc = evaluate_cross_context(
                model, episodes, fact_types, tokenizer, verbose=verbose)
            all_results[alpha][n_facts] = cc
            print(f"  n={n_facts}: first={cc.first_token_accuracy:.1%}  "
                  f"full={cc.full_entity_accuracy:.1%}")

    model.trace.alpha = original_alpha

    # Summary
    print(f"\n{'=' * 65}")
    print("PHASE 5 SUMMARY (full entity accuracy)")
    print(f"{'=' * 65}")
    print(f"{'alpha':>6} │ " + " ".join(
        f"n={n:>2}" for n in test_ns))
    print(f"{'─' * 6}─┼─" + "─".join("─" * 8 for _ in test_ns))
    for alpha in alphas:
        vals = [all_results[alpha][n].full_entity_accuracy for n in test_ns]
        print(f"{alpha:>6.1f} │ " + " ".join(f"{v:>8.1%}" for v in vals))

    return all_results


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp20: Multi-token entity support")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 episodes")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of eval episodes (default: 50)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run Phase 1 validation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    n_eval = 20 if args.quick else args.n_eval
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Episodes: {n_eval}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Phase 1: Validation
    valid_multi = run_phase1_validation(tokenizer, device)

    if args.validate_only:
        return

    # Build model
    model = GPT2WithTrace(
        n_trace_heads=8,
        d_trace=64,
        alpha=args.alpha,
        trace_lr=1.0,
        trace_decay=0.99,
    ).to(device)
    model.eval()

    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    n_facts_list = [1, 3, 5, 7]

    # Phase 2: Cross-context recall
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    phase2 = run_phase2_cross_context(
        model, tokenizer, n_eval, n_facts_list, verbose=False)

    # Phase 3: Pattern separation
    phase3 = run_phase3_pattern_sep(
        model, tokenizer, n_eval, verbose=False)

    # Phase 4: Per-entity analysis
    phase4 = run_phase4_per_entity(
        model, tokenizer, n_eval, verbose=False)

    # Phase 5: Alpha sweep
    phase5 = run_phase5_alpha_sweep(
        model, tokenizer, n_eval, verbose=False)

    print(f"\n{'=' * 65}")
    print("EXPERIMENT 20 COMPLETE")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
