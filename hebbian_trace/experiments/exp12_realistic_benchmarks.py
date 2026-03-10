"""Experiment 12: Realistic Benchmarks — Architectural Ceiling Diagnostic.

All 11 experiments test Hebbian trace on synthetic key-value tasks
("My name is X" → "What is my name?"). This experiment diagnoses
WHERE the architecture breaks on more realistic scenarios.

Phase 1: Setup & Baseline Validation
Phase 2: Question Paraphrasing (Q-addressing brittleness)
Phase 3: Distractor Facts (context-free Q collision)
Phase 4: Scale Stress Test (capacity vs collision isolation)
Phase 5: Ceiling Map Summary

Usage:
    python -m hebbian_trace.experiments.exp12_realistic_benchmarks --quick
    python -m hebbian_trace.experiments.exp12_realistic_benchmarks --n-eval 50
"""

import argparse
import random
import time
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids, make_gpt2_eval_episodes,
    evaluate_gpt2_cross_context,
    GPT2EvalEpisode, GPT2EvalResults, GPT2FactType,
    validate_single_token_entities,
    _predict_answer, _get_all_entity_ids,
    tokenize_fact,
)


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class QuestionVariant:
    """One question variant for testing Q-addressing."""
    text: str              # Full question text
    category: str          # "aligned" | "misaligned" | "semantic"
    note: str              # Human explanation
    # Filled at build time:
    bpe_ids: list[int] = field(default_factory=list)
    q_addr_token: str = ""  # BPE token at position[-2] (the addressing key)
    q_addr_id: int = -1     # BPE ID of that token


@dataclass
class ParaphraseResult:
    """Aggregated results for one question variant."""
    variant: QuestionVariant
    accuracy: float
    n_correct: int
    n_total: int


@dataclass
class DistractorResult:
    """Results for one distractor condition."""
    n_distractors_per_type: int
    my_accuracy: float
    confusion_rate: float   # how often model predicts a distractor entity
    n_total: int


@dataclass
class ScaleResult:
    """Results for one scale condition."""
    n_facts: int
    n_unique_types: int
    n_repeated_types: int
    overall_acc: float
    unique_acc: float       # accuracy on unique-type facts
    repeated_acc: float     # accuracy on repeated-type facts (-1 if none)
    trace_norm: float


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Question Paraphrasing Definitions ─────────────────────────────────

# Per fact type: list of (question_text, category, note)
# Categories: aligned (concept word at pos[-2]), misaligned (wrong position),
#             semantic (related but different concept word)
RAW_QUESTION_VARIANTS: dict[str, list[tuple[str, str, str]]] = {
    "name": [
        # Aligned: Q_addr should = Q(" name")
        ("What is my name?", "aligned", "original"),
        ("My name is", "aligned", "completion"),
        ("Tell me my name.", "aligned", "imperative"),
        # Misaligned: Q_addr ≠ Q(" name")
        ("Who am I?", "misaligned", "Q_addr=Q(I)"),
        ("What am I called?", "misaligned", "Q_addr=Q(called)"),
        # Semantic: different concept word
        ("What is my title?", "semantic", "title≠name"),
    ],
    "city": [
        ("Where do I live?", "aligned", "original"),
        ("I live in", "aligned", "completion"),
        # Misaligned
        ("Where is my home?", "misaligned", "Q_addr=Q(home)"),
        ("What city am I from?", "misaligned", "Q_addr=Q(from)"),
        # Semantic
        ("Where is my residence?", "semantic", "residence≠live"),
    ],
    "company": [
        ("Where do I work?", "aligned", "original"),
        ("I work at", "aligned", "completion"),
        # Misaligned
        ("What is my job?", "misaligned", "Q_addr=Q(job)"),
        ("Where am I employed?", "misaligned", "Q_addr=Q(employed)"),
        # Semantic
        ("What is my employer?", "semantic", "employer≠work"),
    ],
    "color": [
        ("What is my favorite color?", "aligned", "original"),
        ("My favorite color is", "aligned", "completion"),
        # Misaligned
        ("What shade do I like?", "misaligned", "Q_addr=Q(like)"),
        # Semantic
        ("What is my preferred hue?", "semantic", "hue≠color"),
    ],
    "food": [
        ("What is my favorite food?", "aligned", "original"),
        ("My favorite food is", "aligned", "completion"),
        # Misaligned
        ("What do I like to eat?", "misaligned", "Q_addr=Q(eat)"),
        # Semantic
        ("What is my favorite meal?", "semantic", "meal≠food"),
    ],
    "pet": [
        ("What is my pet?", "aligned", "original"),
        ("My pet is", "aligned", "completion"),
        # Misaligned
        ("What animal do I have?", "misaligned", "Q_addr=Q(have)"),
        # Semantic
        ("What is my companion animal?", "semantic", "animal≠pet"),
    ],
    "country": [
        ("What is my country?", "aligned", "original"),
        ("My country is", "aligned", "completion"),
        # Misaligned
        ("Where am I from?", "misaligned", "Q_addr=Q(from)"),
        # Semantic
        ("What is my homeland?", "semantic", "homeland≠country"),
    ],
}


def build_question_variants(
    tokenizer: GPT2Tokenizer,
) -> dict[str, list[QuestionVariant]]:
    """Build and BPE-validate question variants.

    For each variant, identifies which BPE token is at position[-2]
    (the Q_addr key at the last prediction position).
    """
    variants: dict[str, list[QuestionVariant]] = {}

    for fact_type, raw_list in RAW_QUESTION_VARIANTS.items():
        type_variants = []
        for text, category, note in raw_list:
            bpe_ids = tokenizer.encode(text, add_special_tokens=False)
            # Q_addr at last position uses Q[last-1]
            if len(bpe_ids) >= 2:
                q_addr_id = bpe_ids[-2]
                q_addr_token = tokenizer.decode([q_addr_id])
            else:
                q_addr_id = -1
                q_addr_token = "?"

            v = QuestionVariant(
                text=text,
                category=category,
                note=note,
                bpe_ids=bpe_ids,
                q_addr_token=q_addr_token,
                q_addr_id=q_addr_id,
            )
            type_variants.append(v)
        variants[fact_type] = type_variants

    return variants


# ── Third-person templates for distractors ────────────────────────────

# Person names for third-person facts (must be single-token BPE, disjoint
# from entity pools). These are checked at runtime.
DISTRACTOR_PERSONS = [
    "Alice", "Bob", "Charlie", "Diana", "Eve",
    "Frank", "Grace", "Henry", "Iris", "Jack",
]

# Third-person fact templates: {fact_type: template_string}
# Use infinitive forms to match first-person BPE (e.g. "live" not "lives")
THIRD_PERSON_TEMPLATES: dict[str, str] = {
    "name":    "{P}'s name is {X}.",
    "city":    "{P} live in {X}.",       # "live" matches 1st-person "I live in"
    "company": "{P} work at {X}.",       # "work" matches "I work at"
    "color":   "{P}'s favorite color is {X}.",
    "food":    "{P}'s favorite food is {X}.",
    "pet":     "{P}'s pet is {X}.",
    "country": "{P}'s country is {X}.",
}


def validate_distractor_persons(
    tokenizer: GPT2Tokenizer,
    entity_names: set[str],
) -> list[tuple[str, int]]:
    """Validate person names as single-token and not in entity pool."""
    valid = []
    for person in DISTRACTOR_PERSONS:
        # Check BPE: " Alice" should be single token
        ids = tokenizer.encode(" " + person, add_special_tokens=False)
        if len(ids) == 1 and person not in entity_names:
            valid.append((person, ids[0]))
    return valid


# ── Phase 1: Setup ────────────────────────────────────────────────────

def run_phase1_setup(device_str=None, seed=42):
    """Load model, validate entities, confirm baseline."""
    print("=" * 65)
    print("EXP 12: Realistic Benchmarks — Architectural Ceiling Diagnostic")
    print("=" * 65)

    device = get_device(device_str)
    print(f"\nDevice: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    print(f"Fact types: {len(fact_types)}")
    for ft in fact_types:
        print(f"  {ft.name}: {len(ft.entities)} entities")
    print(f"Linking BPE IDs: {len(linking_ids)}")

    # Create model with best config
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    ).to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model.eval()

    # Quick baseline validation
    print(f"\n{'─' * 65}")
    print("PHASE 1: Baseline Validation")
    print(f"{'─' * 65}")

    episodes = make_gpt2_eval_episodes(
        n_episodes=10, n_facts=3,
        tokenizer=tokenizer, fact_types=fact_types, seed=seed)
    result = evaluate_gpt2_cross_context(model, episodes, fact_types)
    print(f"  Baseline (n=3, 10 episodes): {result.accuracy:.1%}")
    if result.accuracy < 0.5:
        print("  ⚠ WARNING: baseline below 50%, results may be unreliable")
    else:
        print("  ✓ Baseline OK")

    return model, tokenizer, fact_types, linking_ids, device


# ── Phase 2: Question Paraphrasing ────────────────────────────────────

def run_phase2_paraphrasing(model, tokenizer, fact_types, n_eval,
                            n_facts, seed=42):
    """Test Q-addressing brittleness via question paraphrasing."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 2: Question Paraphrasing Diagnostic (n_facts={n_facts})")
    print(f"{'─' * 65}")

    variants = build_question_variants(tokenizer)
    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device

    # Print BPE analysis
    print("\n  BPE Analysis (Q_addr token at position[-2]):")
    for ft_name, vlist in variants.items():
        # Find the storage concept word for this fact type
        ft = next(f for f in fact_types if f.name == ft_name)
        fact_ids = tokenize_fact(tokenizer, ft.fact_templates[0], ft.entities[0][0])
        # The concept word is at linking_pos - 1 in the fact
        linking_bpe_ids = set(get_linking_bpe_ids(tokenizer))
        concept_id = -1
        for i, fid in enumerate(fact_ids):
            if fid in linking_bpe_ids and i > 0:
                concept_id = fact_ids[i - 1]
                break
        concept_token = tokenizer.decode([concept_id]) if concept_id >= 0 else "?"

        print(f"\n  {ft_name} (storage key: Q('{concept_token.strip()}'), "
              f"ID={concept_id}):")
        for v in vlist:
            match = "✓" if v.q_addr_id == concept_id else "✗"
            print(f"    {v.category:11s} {match} \"{v.text}\" "
                  f"→ Q_addr=Q('{v.q_addr_token.strip()}') "
                  f"[{v.note}]")

    # Generate episodes and evaluate each variant
    print(f"\n  Evaluating ({n_eval} episodes)...")
    episodes = make_gpt2_eval_episodes(
        n_episodes=n_eval, n_facts=n_facts,
        tokenizer=tokenizer, fact_types=fact_types, seed=seed)

    # Results: {fact_type: {variant_text: (correct, total)}}
    results: dict[str, dict[str, list[int, int]]] = {}
    for ft_name in variants:
        results[ft_name] = {}
        for v in variants[ft_name]:
            results[ft_name][v.text] = [0, 0]

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor)

        # Read phase: test each variant for each stored fact type
        model.set_trace_mode(use=True, update=False)

        for (type_name, entity_name, entity_id, _) in episode.facts:
            if type_name not in variants:
                continue
            for v in variants[type_name]:
                pred_id = _predict_answer(model, v.bpe_ids, entity_ids)
                results[type_name][v.text][1] += 1
                if pred_id == entity_id:
                    results[type_name][v.text][0] += 1

    # Aggregate and print results
    print(f"\n  {'Type':10s} {'Category':11s} {'Accuracy':>8s}  Question")
    print(f"  {'─' * 60}")

    category_totals: dict[str, list[int, int]] = {
        "aligned": [0, 0], "misaligned": [0, 0], "semantic": [0, 0],
    }

    all_results: dict[str, list[ParaphraseResult]] = {}

    for ft_name, vlist in variants.items():
        all_results[ft_name] = []
        for v in vlist:
            c, t = results[ft_name][v.text]
            acc = c / max(t, 1)
            all_results[ft_name].append(ParaphraseResult(
                variant=v, accuracy=acc, n_correct=c, n_total=t))
            category_totals[v.category][0] += c
            category_totals[v.category][1] += t
            mark = "✓" if acc > 0.2 else "✗"
            print(f"  {ft_name:10s} {v.category:11s} {acc:7.1%} {mark} "
                  f"\"{v.text}\"")

    print(f"\n  {'═' * 50}")
    print(f"  SUMMARY BY CATEGORY:")
    for cat in ["aligned", "misaligned", "semantic"]:
        c, t = category_totals[cat]
        acc = c / max(t, 1)
        print(f"    {cat:11s}: {acc:.1%}  ({c}/{t})")

    return all_results, category_totals


# ── Phase 3: Distractor Facts ─────────────────────────────────────────

def build_distractor_episodes(
    n_episodes: int,
    n_my_facts: int,
    n_distractors_per_type: int,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    distractor_persons: list[tuple[str, int]],
    seed: int = 42,
) -> list[GPT2EvalEpisode]:
    """Build episodes with first-person + third-person distractor facts.

    Returns episodes where train_sequences include both "my" facts
    and distractor facts, but test_queries only query "my" facts.
    Also records distractor entity IDs per episode for confusion tracking.
    """
    rng = random.Random(seed)
    episodes = []

    for _ in range(n_episodes):
        # Select fact types
        if n_my_facts <= len(fact_types):
            selected_types = rng.sample(fact_types, n_my_facts)
        else:
            selected_types = [rng.choice(fact_types) for _ in range(n_my_facts)]

        # Build "my" facts
        my_facts = []
        for ft in selected_types:
            entity_name, entity_id = rng.choice(ft.entities)
            template = rng.choice(ft.fact_templates)
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            my_facts.append((ft.name, entity_name, entity_id, fact_ids))

        # Build distractor facts
        distractor_facts_ids = []  # list of BPE ID lists
        distractor_entity_ids = set()

        if n_distractors_per_type > 0:
            for ft in selected_types:
                # Get entities NOT used by "my" fact
                my_entity = next(
                    e for (tn, e, _, _) in my_facts if tn == ft.name)
                other_entities = [
                    (e, eid) for e, eid in ft.entities if e != my_entity]

                for d in range(min(n_distractors_per_type, len(other_entities))):
                    d_entity, d_eid = rng.choice(other_entities)
                    person, _ = rng.choice(distractor_persons)

                    template_str = THIRD_PERSON_TEMPLATES.get(ft.name, "")
                    if not template_str:
                        continue
                    text = template_str.replace("{P}", person).replace("{X}", d_entity)
                    d_ids = tokenizer.encode(text, add_special_tokens=False)
                    distractor_facts_ids.append(d_ids)
                    distractor_entity_ids.add(d_eid)

        # Build cumulative train sequences: my facts first, then distractors
        all_fact_ids = [f[3] for f in my_facts] + distractor_facts_ids
        rng.shuffle(all_fact_ids)  # interleave

        # Single-pass: all facts concatenated
        train_seq = []
        for i, fids in enumerate(all_fact_ids):
            if i > 0:
                sp = tokenizer.encode(" ", add_special_tokens=False)
                train_seq.extend(sp)
            train_seq.extend(fids)
        train_sequences = [train_seq]

        # Test queries: only for "my" facts
        test_queries = []
        for ft, (type_name, entity_name, entity_id, _) in zip(
                selected_types, my_facts):
            q_template = rng.choice(ft.question_templates)
            q_ids = tokenizer.encode(q_template.text, add_special_tokens=False)
            test_queries.append((q_ids, entity_id, type_name))

        episodes.append(GPT2EvalEpisode(
            facts=my_facts,
            train_sequences=train_sequences,
            test_queries=test_queries,
        ))

    return episodes


def run_phase3_distractors(model, tokenizer, fact_types, n_eval,
                           n_my_facts, distractor_counts, seed=42):
    """Test context-free Q collision with third-person distractors."""
    print(f"\n{'─' * 65}")
    print(f"PHASE 3: Distractor Facts — Context-Free Q Collision "
          f"(n_my_facts={n_my_facts})")
    print(f"{'─' * 65}")

    # Validate distractor persons
    all_entity_names = set()
    for ft in fact_types:
        for name, _ in ft.entities:
            all_entity_names.add(name)
    persons = validate_distractor_persons(tokenizer, all_entity_names)
    print(f"\n  Validated distractor persons: {len(persons)}")
    print(f"    {', '.join(p[0] for p in persons[:5])}...")

    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device

    results: list[DistractorResult] = []

    for n_dist in distractor_counts:
        label = f"{n_dist} per type"
        episodes = build_distractor_episodes(
            n_episodes=n_eval,
            n_my_facts=n_my_facts,
            n_distractors_per_type=n_dist,
            tokenizer=tokenizer,
            fact_types=fact_types,
            distractor_persons=persons,
            seed=seed + n_dist * 1000,
        )

        # Evaluate with confusion tracking
        total_correct = 0
        total_queries = 0
        total_confused = 0  # predicted a distractor entity

        for episode in episodes:
            model.reset_traces()

            # Write phase
            model.set_trace_mode(use=False, update=True)
            for train_seq in episode.train_sequences:
                input_tensor = torch.tensor(
                    [train_seq], dtype=torch.long, device=device)
                with torch.no_grad():
                    model(input_tensor)

            # Read phase
            model.set_trace_mode(use=True, update=False)
            for query_ids, answer_id, type_name in episode.test_queries:
                pred_id = _predict_answer(model, query_ids, entity_ids)
                total_queries += 1
                if pred_id == answer_id:
                    total_correct += 1
                elif pred_id != answer_id and n_dist > 0:
                    # Check if predicted entity belongs to this fact type
                    # (indicating confusion with distractor)
                    ft = next(f for f in fact_types if f.name == type_name)
                    ft_entity_ids = {eid for _, eid in ft.entities}
                    if pred_id in ft_entity_ids:
                        total_confused += 1

        acc = total_correct / max(total_queries, 1)
        confusion = total_confused / max(total_queries, 1)
        results.append(DistractorResult(
            n_distractors_per_type=n_dist,
            my_accuracy=acc,
            confusion_rate=confusion,
            n_total=total_queries,
        ))
        print(f"  {label:15s}: accuracy={acc:.1%}, "
              f"confusion={confusion:.1%} ({total_queries} queries)")

    # Summary table
    print(f"\n  {'Distractors':>12s} │ {'My-Acc':>7s} │ {'Confusion':>9s} │ Note")
    print(f"  {'─' * 50}")
    for r in results:
        note = "baseline" if r.n_distractors_per_type == 0 else "Q collision"
        print(f"  {r.n_distractors_per_type:>8d}/type │ {r.my_accuracy:>6.1%} │ "
              f"{r.confusion_rate:>8.1%} │ {note}")

    return results


# ── Phase 4: Scale Stress Test ────────────────────────────────────────

def make_controlled_episodes(
    n_episodes: int,
    n_facts: int,
    n_unique_types: int,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    seed: int = 42,
) -> list[GPT2EvalEpisode]:
    """Generate episodes with controlled type repetition.

    Args:
        n_facts: total number of facts
        n_unique_types: how many unique types to use (rest are duplicates)
    """
    rng = random.Random(seed)
    episodes = []

    for _ in range(n_episodes):
        # Select n_unique_types distinct types
        actual_unique = min(n_unique_types, len(fact_types))
        selected_unique = rng.sample(fact_types, actual_unique)

        # Fill remaining slots with duplicates
        selected_types = list(selected_unique)
        while len(selected_types) < n_facts:
            selected_types.append(rng.choice(selected_unique))
        rng.shuffle(selected_types)

        # Assign entities (different entity for each occurrence, even same type)
        facts = []
        used_entities: dict[str, set[str]] = {}  # type_name -> used entity names
        for ft in selected_types:
            if ft.name not in used_entities:
                used_entities[ft.name] = set()
            # Try to pick unused entity
            available = [(e, eid) for e, eid in ft.entities
                         if e not in used_entities[ft.name]]
            if not available:
                available = ft.entities  # fallback
            entity_name, entity_id = rng.choice(available)
            used_entities[ft.name].add(entity_name)

            template = rng.choice(ft.fact_templates)
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            facts.append((ft.name, entity_name, entity_id, fact_ids))

        # Cumulative train sequences
        train_sequences = []
        for i in range(len(facts)):
            cumulative = []
            for j in range(i + 1):
                if cumulative:
                    sp = tokenizer.encode(" ", add_special_tokens=False)
                    cumulative.extend(sp)
                cumulative.extend(facts[j][3])
            train_sequences.append(cumulative)

        # Test queries for ALL facts
        test_queries = []
        for ft, (type_name, entity_name, entity_id, _) in zip(
                selected_types, facts):
            q_template = rng.choice(ft.question_templates)
            q_ids = tokenizer.encode(q_template.text, add_special_tokens=False)
            test_queries.append((q_ids, entity_id, type_name))

        episodes.append(GPT2EvalEpisode(
            facts=facts,
            train_sequences=train_sequences,
            test_queries=test_queries,
        ))

    return episodes


def evaluate_with_type_tracking(
    model, episodes: list[GPT2EvalEpisode],
    fact_types: list[GPT2FactType],
) -> ScaleResult:
    """Evaluate cross-context with separate tracking for unique vs repeated types."""
    entity_ids = _get_all_entity_ids(fact_types)
    device = next(model.parameters()).device

    unique_correct, unique_total = 0, 0
    repeated_correct, repeated_total = 0, 0

    for episode in episodes:
        model.reset_traces()

        # Write phase
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor)

        # Count type occurrences in this episode
        type_counts: dict[str, int] = {}
        for (type_name, _, _, _) in episode.facts:
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Read phase
        model.set_trace_mode(use=True, update=False)
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer(model, query_ids, entity_ids)
            is_correct = pred_id == answer_id
            if type_counts.get(type_name, 0) > 1:
                repeated_total += 1
                if is_correct:
                    repeated_correct += 1
            else:
                unique_total += 1
                if is_correct:
                    unique_correct += 1

    # Trace norm
    trace_norm = model.trace.value_traces.norm().item()

    total_correct = unique_correct + repeated_correct
    total = unique_total + repeated_total
    n_facts = len(episodes[0].facts) if episodes else 0
    type_names = [f[0] for f in episodes[0].facts] if episodes else []
    n_unique = len(set(type_names))
    n_repeated = n_facts - n_unique

    return ScaleResult(
        n_facts=n_facts,
        n_unique_types=n_unique,
        n_repeated_types=n_repeated,
        overall_acc=total_correct / max(total, 1),
        unique_acc=unique_correct / max(unique_total, 1),
        repeated_acc=(repeated_correct / max(repeated_total, 1)
                      if repeated_total > 0 else -1.0),
        trace_norm=trace_norm,
    )


def run_phase4_scale(model, tokenizer, fact_types, n_eval, seed=42):
    """Scale stress test with capacity vs collision isolation."""
    print(f"\n{'─' * 65}")
    print("PHASE 4: Scale Stress Test (capacity vs collision isolation)")
    print(f"{'─' * 65}")

    n_types = len(fact_types)
    results: list[ScaleResult] = []

    # Sub-phase 4a: Pure capacity (all unique types)
    print(f"\n  4a: Pure capacity (unique types only)")
    capacity_ns = [n for n in [1, 3, 5, 7] if n <= n_types]
    for n in capacity_ns:
        episodes = make_controlled_episodes(
            n_episodes=n_eval, n_facts=n, n_unique_types=n,
            tokenizer=tokenizer, fact_types=fact_types,
            seed=seed + n * 100)
        r = evaluate_with_type_tracking(model, episodes, fact_types)
        results.append(r)
        print(f"    n={n:2d} (all unique): {r.overall_acc:.1%}  "
              f"||T_v||={r.trace_norm:.2f}")

    # Sub-phase 4b: Q-collision control (same n, forced duplicates)
    print(f"\n  4b: Q-collision control (same n, different collision levels)")
    collision_n = min(7, n_types)  # test at n=7

    # All unique
    ep_unique = make_controlled_episodes(
        n_episodes=n_eval, n_facts=collision_n,
        n_unique_types=collision_n,
        tokenizer=tokenizer, fact_types=fact_types,
        seed=seed + 7000)
    r_unique = evaluate_with_type_tracking(model, ep_unique, fact_types)
    results.append(r_unique)

    # With duplicates: 4 unique types + 3 duplicates = 7 facts
    n_unique_subset = max(3, collision_n - 3)
    ep_duped = make_controlled_episodes(
        n_episodes=n_eval, n_facts=collision_n,
        n_unique_types=n_unique_subset,
        tokenizer=tokenizer, fact_types=fact_types,
        seed=seed + 7001)
    r_duped = evaluate_with_type_tracking(model, ep_duped, fact_types)
    results.append(r_duped)

    collision_cost = r_unique.overall_acc - r_duped.overall_acc
    print(f"    n={collision_n}, {collision_n} unique types: "
          f"{r_unique.overall_acc:.1%}")
    print(f"    n={collision_n}, {n_unique_subset} unique types "
          f"(+{collision_n - n_unique_subset} dupes): "
          f"{r_duped.overall_acc:.1%}  "
          f"(unique={r_duped.unique_acc:.1%}, "
          f"repeated={r_duped.repeated_acc:.1%})")
    print(f"    → Q-collision cost: {collision_cost:+.1%}")

    # Sub-phase 4c: Combined scaling (n > n_types)
    print(f"\n  4c: Combined scaling (n > {n_types} types → forced repeats)")
    for n in [10, 15, 20]:
        episodes = make_controlled_episodes(
            n_episodes=n_eval, n_facts=n,
            n_unique_types=n_types,  # use all types, rest are duplicates
            tokenizer=tokenizer, fact_types=fact_types,
            seed=seed + n * 100)
        r = evaluate_with_type_tracking(model, episodes, fact_types)
        results.append(r)
        rep_str = (f"repeated={r.repeated_acc:.1%}"
                   if r.repeated_acc >= 0 else "no repeats")
        print(f"    n={n:2d} ({r.n_unique_types} unique, "
              f"{r.n_repeated_types} repeated): "
              f"overall={r.overall_acc:.1%}  "
              f"unique={r.unique_acc:.1%}, {rep_str}  "
              f"||T_v||={r.trace_norm:.2f}")

    return results


# ── Phase 5: Ceiling Map ──────────────────────────────────────────────

def run_phase5_ceiling_map(
    paraphrase_totals: dict[str, list[int, int]],
    distractor_results: list[DistractorResult],
    scale_results: list[ScaleResult],
):
    """Print comprehensive ceiling map."""
    print(f"\n{'═' * 65}")
    print("CEILING MAP: GPT-2 + Hebbian Trace Architectural Limits")
    print(f"{'═' * 65}")

    # What works
    aligned_acc = (paraphrase_totals["aligned"][0]
                   / max(paraphrase_totals["aligned"][1], 1))
    baseline_dist = next(
        (r for r in distractor_results if r.n_distractors_per_type == 0), None)
    baseline_acc = baseline_dist.my_accuracy if baseline_dist else 0

    # Find unique-only scale results
    unique_results = [r for r in scale_results if r.repeated_acc < 0]

    print("\n  WHAT WORKS:")
    print(f"    Aligned questions (concept at pos[-2]):  {aligned_acc:.1%}")
    print(f"    Clean facts (no distractors):            {baseline_acc:.1%}")
    if unique_results:
        for r in unique_results:
            print(f"    n={r.n_facts} unique types:  "
                  f"                      {r.overall_acc:.1%}")

    # What breaks
    misaligned_acc = (paraphrase_totals["misaligned"][0]
                      / max(paraphrase_totals["misaligned"][1], 1))
    semantic_acc = (paraphrase_totals["semantic"][0]
                    / max(paraphrase_totals["semantic"][1], 1))

    print("\n  WHAT BREAKS:")
    print(f"    Misaligned questions (wrong position):   {misaligned_acc:.1%} "
          f"(≈random)")
    print(f"    Semantic synonyms (title≠name):          {semantic_acc:.1%} "
          f"(≈random)")

    for r in distractor_results:
        if r.n_distractors_per_type > 0:
            print(f"    {r.n_distractors_per_type} distractor(s)/type:    "
                  f"              {r.my_accuracy:.1%} "
                  f"(confusion {r.confusion_rate:.1%})")

    combined = [r for r in scale_results if r.repeated_acc >= 0]
    for r in combined:
        print(f"    n={r.n_facts:2d} ({r.n_repeated_types} repeated types): "
              f"             {r.overall_acc:.1%} "
              f"(repeated={r.repeated_acc:.1%})")

    # Why it breaks
    print("\n  WHY IT BREAKS:")
    print("    1. Context-free Q: Q(\"name\") is identical in \"My name\"")
    print("       and \"Alice's name\" → cannot distinguish entities.")
    print("    2. Shift-1 addressing: concept word MUST be at pos[-2]")
    print("       in the question. Other phrasings give random results.")
    print("    3. Single-token key: two facts sharing a concept word")
    print("       (two names) create unresolvable interference.")

    # Implications
    print("\n  IMPLICATIONS:")
    print("    → Context-dependent Q: would solve entity disambiguation")
    print("       but breaks cross-context property (needs careful design).")
    print("    → Compositional addressing: Q = f(modifier, concept)")
    print("       could distinguish \"my name\" from \"Alice's name\".")
    print("    → End-to-end write gate: would remove shift-1 / linking")
    print("       token dependency, enabling flexible fact formats.")
    print(f"{'═' * 65}")


# ── Entry points ──────────────────────────────────────────────────────

def run_quick(device=None, seed=42):
    """Quick diagnostic (~3-5 min)."""
    t0 = time.time()
    model, tokenizer, fact_types, linking_ids, dev = run_phase1_setup(
        device, seed)

    n_eval = 20

    para_results, para_totals = run_phase2_paraphrasing(
        model, tokenizer, fact_types, n_eval=n_eval, n_facts=3, seed=seed)

    dist_results = run_phase3_distractors(
        model, tokenizer, fact_types, n_eval=n_eval, n_my_facts=3,
        distractor_counts=[0, 1], seed=seed)

    scale_results = run_phase4_scale(
        model, tokenizer, fact_types, n_eval=n_eval, seed=seed)

    run_phase5_ceiling_map(para_totals, dist_results, scale_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


def run_full(device=None, seed=42, n_eval=50):
    """Full diagnostic (~15-25 min)."""
    t0 = time.time()
    model, tokenizer, fact_types, linking_ids, dev = run_phase1_setup(
        device, seed)

    para_results, para_totals = run_phase2_paraphrasing(
        model, tokenizer, fact_types, n_eval=n_eval, n_facts=5, seed=seed)

    dist_results = run_phase3_distractors(
        model, tokenizer, fact_types, n_eval=n_eval, n_my_facts=5,
        distractor_counts=[0, 1, 2], seed=seed)

    scale_results = run_phase4_scale(
        model, tokenizer, fact_types, n_eval=n_eval, seed=seed)

    run_phase5_ceiling_map(para_totals, dist_results, scale_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 12: Realistic Benchmarks — Ceiling Diagnostic")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 episodes, n=3)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    if args.quick:
        run_quick(device=args.device, seed=args.seed)
    else:
        run_full(device=args.device, seed=args.seed, n_eval=args.n_eval)
