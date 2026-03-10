"""Exp 25: Multi-turn dialogue memory.

Level 2: accumulate facts across dialogue turns, handle updates,
maintain retention. Tests the full dialogue pipeline:
  extract → direct write → accumulate → query → measure retention.

Known limitation (v1): mixed turns ("I moved to London. What was my
old city?") are split into separate annotated turns. Fuzzy turn
boundaries deferred to Level 3+.

Three evaluation phases:
  Phase 1: Introduction dialogues (pure accumulation)
  Phase 2: Update dialogues (erase + rewrite)
  Phase 3: Multi-dialogue sessions (trace persists across dialogues)

Usage:
  python -m hebbian_trace.experiments.exp25_dialogue --quick
  python -m hebbian_trace.experiments.exp25_dialogue --phase intro --n-eval 50
  python -m hebbian_trace.experiments.exp25_dialogue --phase update --n-eval 50
  python -m hebbian_trace.experiments.exp25_dialogue --phase multi --n-dialogues 5
  python -m hebbian_trace.experiments.exp25_dialogue --with-tauto
"""

import argparse
import random
import re
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_concept_vocab, ConceptEntry,
    _predict_answer, get_linking_bpe_ids,
)
from .exp24_free_text import (
    ExtractedFact, AnnotatedUtterance,
    OracleExtractor, RegexExtractor, setup_model,
)
from .exp22_pattern_completion import extract_auto_pairs


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    """One turn in a dialogue."""
    text: str                              # natural language utterance
    turn_type: str                         # "info" | "query" | "update"
    facts: list[tuple[str, str]]           # ground truth: [(type_name, entity)]
    queries: list[str] = field(default_factory=list)
    # type_names to query; empty = query all known


@dataclass
class DialogueTemplate:
    """Parameterized dialogue with entity placeholders."""
    turns: list[DialogueTurn]
    description: str
    category: str  # "intro" | "update" | "dense"


@dataclass
class DialogueResult:
    """Results from one dialogue episode."""
    per_query_turn: dict[int, tuple[int, int]] = field(
        default_factory=dict)
    per_age_correct: dict[int, list[bool]] = field(
        default_factory=dict)
    update_correct: int = 0
    update_total: int = 0
    overall_correct: int = 0
    overall_total: int = 0
    n_facts_final: int = 0


# ── Dialogue corpus ──────────────────────────────────────────────────

# Entity placeholders: {NAME}, {CITY}, {CITY2} (for updates), etc.
# Resolved at runtime from concept_vocab pools.

DIALOGUE_TEMPLATES: list[DialogueTemplate] = [
    # ── A. Introduction dialogues (pure accumulation) ──

    DialogueTemplate(
        description="Basic 3-fact introduction",
        category="intro",
        turns=[
            DialogueTurn("Hey, I'm {NAME}. Nice to meet you.",
                         "info", [("name", "{NAME}")]),
            DialogueTurn("I live in {CITY}, moved there last year.",
                         "info", [("city", "{CITY}")]),
            DialogueTurn("I work at {COMPANY}.",
                         "info", [("company", "{COMPANY}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "company"]),
        ],
    ),

    DialogueTemplate(
        description="5-fact gradual introduction",
        category="intro",
        turns=[
            DialogueTurn("My name is {NAME}.", "info",
                         [("name", "{NAME}")]),
            DialogueTurn("I live in {CITY}.", "info",
                         [("city", "{CITY}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city"]),
            DialogueTurn("My favorite food is {FOOD}.", "info",
                         [("food", "{FOOD}")]),
            DialogueTurn("I have a {PET} at home.", "info",
                         [("pet", "{PET}")]),
            DialogueTurn("My favorite color is {COLOR}.", "info",
                         [("color", "{COLOR}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "food", "pet", "color"]),
        ],
    ),

    DialogueTemplate(
        description="6-fact with mid-dialogue query",
        category="intro",
        turns=[
            DialogueTurn("I'm {NAME}, I come from {COUNTRY}.",
                         "info",
                         [("name", "{NAME}"), ("country", "{COUNTRY}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "country"]),
            DialogueTurn("I work at {COMPANY} and my pet is a {PET}.",
                         "info",
                         [("company", "{COMPANY}"), ("pet", "{PET}")]),
            DialogueTurn("I usually drink {DRINK}.", "info",
                         [("drink", "{DRINK}")]),
            DialogueTurn("My sport is {SPORT}.", "info",
                         [("sport", "{SPORT}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "country", "company",
                                  "pet", "drink", "sport"]),
        ],
    ),

    DialogueTemplate(
        description="4-fact with repeated queries",
        category="intro",
        turns=[
            DialogueTurn("My name is {NAME}.", "info",
                         [("name", "{NAME}")]),
            DialogueTurn("", "query", [], queries=["name"]),
            DialogueTurn("I live in {CITY}.", "info",
                         [("city", "{CITY}")]),
            DialogueTurn("", "query", [], queries=["name", "city"]),
            DialogueTurn("My favorite color is {COLOR}.", "info",
                         [("color", "{COLOR}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "color"]),
            DialogueTurn("I work at {COMPANY}.", "info",
                         [("company", "{COMPANY}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "color", "company"]),
        ],
    ),

    # ── B. Update dialogues ──

    DialogueTemplate(
        description="City update",
        category="update",
        turns=[
            DialogueTurn("My name is {NAME}, I live in {CITY}.",
                         "info",
                         [("name", "{NAME}"), ("city", "{CITY}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city"]),
            DialogueTurn("Actually, I moved to {CITY2}.",
                         "update", [("city", "{CITY2}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city"]),
        ],
    ),

    DialogueTemplate(
        description="Company update",
        category="update",
        turns=[
            DialogueTurn("I'm {NAME}, working at {COMPANY}.",
                         "info",
                         [("name", "{NAME}"), ("company", "{COMPANY}")]),
            DialogueTurn("I live in {CITY}.", "info",
                         [("city", "{CITY}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "company", "city"]),
            DialogueTurn("I changed jobs, now I'm at {COMPANY2}.",
                         "update", [("company", "{COMPANY2}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "company", "city"]),
        ],
    ),

    DialogueTemplate(
        description="Double update (city + food)",
        category="update",
        turns=[
            DialogueTurn("My name is {NAME}.", "info",
                         [("name", "{NAME}")]),
            DialogueTurn("I live in {CITY} and love {FOOD}.",
                         "info",
                         [("city", "{CITY}"), ("food", "{FOOD}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "food"]),
            DialogueTurn("I moved to {CITY2}.",
                         "update", [("city", "{CITY2}")]),
            DialogueTurn("Also, I now prefer {FOOD2}.",
                         "update", [("food", "{FOOD2}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "food"]),
        ],
    ),

    # ── C. Dense dialogues (many facts per turn) ──

    DialogueTemplate(
        description="Dense 7-fact dialogue",
        category="dense",
        turns=[
            DialogueTurn(
                "I'm {NAME} from {COUNTRY}, working at {COMPANY}.",
                "info",
                [("name", "{NAME}"), ("country", "{COUNTRY}"),
                 ("company", "{COMPANY}")]),
            DialogueTurn(
                "I have a {PET} and my favorite color is {COLOR}.",
                "info",
                [("pet", "{PET}"), ("color", "{COLOR}")]),
            DialogueTurn(
                "I play {SPORT} and usually drink {DRINK}.",
                "info",
                [("sport", "{SPORT}"), ("drink", "{DRINK}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "country", "company",
                                  "pet", "color", "sport", "drink"]),
        ],
    ),

    DialogueTemplate(
        description="Dense with mid-queries",
        category="dense",
        turns=[
            DialogueTurn(
                "My name is {NAME} and I live in {CITY}.",
                "info",
                [("name", "{NAME}"), ("city", "{CITY}")]),
            DialogueTurn(
                "I work at {COMPANY} and drive a {CAR}.",
                "info",
                [("company", "{COMPANY}"), ("car", "{CAR}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "company", "car"]),
            DialogueTurn(
                "My hobby is {HOBBY} and I like {FOOD}.",
                "info",
                [("hobby", "{HOBBY}"), ("food", "{FOOD}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "company", "car",
                                  "hobby", "food"]),
        ],
    ),

    # ── D. Update + dense ──

    DialogueTemplate(
        description="Dense with update",
        category="update",
        turns=[
            DialogueTurn(
                "I'm {NAME}, I live in {CITY}, work at {COMPANY}.",
                "info",
                [("name", "{NAME}"), ("city", "{CITY}"),
                 ("company", "{COMPANY}")]),
            DialogueTurn("My favorite color is {COLOR}.", "info",
                         [("color", "{COLOR}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "company", "color"]),
            DialogueTurn("I just moved to {CITY2}.", "update",
                         [("city", "{CITY2}")]),
            DialogueTurn("", "query", [],
                         queries=["name", "city", "company", "color"]),
        ],
    ),
]


# ── Dialogue instantiation ───────────────────────────────────────────

def _resolve_entity(
    placeholder: str,
    concept_vocab: dict[str, ConceptEntry],
    assignments: dict[str, list[tuple[str, int]]],
    rng: random.Random,
) -> tuple[str, int] | None:
    """Resolve a placeholder like {NAME} or {CITY2} to a random entity.

    {TYPE} → first assignment (or random).
    {TYPE2} → second distinct entity from same pool (for updates).
    """
    raw = placeholder.strip("{}")
    # Check for update suffix (e.g., CITY2 → type=city, version=2)
    if raw[-1].isdigit():
        type_name = raw[:-1].lower()
        version = int(raw[-1])
    else:
        type_name = raw.lower()
        version = 1

    entry = concept_vocab.get(type_name)
    if entry is None:
        return None

    if type_name not in assignments:
        assignments[type_name] = []

    # Ensure we have enough distinct assignments
    while len(assignments[type_name]) < version:
        used = {name for name, _ in assignments[type_name]}
        available = [(n, i) for n, i in entry.entity_pool if n not in used]
        if not available:
            return None
        choice = rng.choice(available)
        assignments[type_name].append(choice)

    return assignments[type_name][version - 1]


def instantiate_dialogue(
    template: DialogueTemplate,
    concept_vocab: dict[str, ConceptEntry],
    rng: random.Random,
) -> list[DialogueTurn] | None:
    """Instantiate a dialogue template with random entities.

    Returns None if any required type is missing from concept_vocab.
    """
    assignments: dict[str, list[tuple[str, int]]] = {}
    turns = []

    for tmpl_turn in template.turns:
        # Check all required types exist
        for type_name, entity_ph in tmpl_turn.facts:
            raw = entity_ph.strip("{}")
            tn = raw.rstrip("0123456789").lower()
            if tn not in concept_vocab:
                return None

        # Check query types exist
        for qtype in tmpl_turn.queries:
            if qtype not in concept_vocab:
                return None

        # Resolve text placeholders
        resolved_text = tmpl_turn.text
        for ph in re.findall(r'\{[A-Z0-9]+\}', tmpl_turn.text):
            result = _resolve_entity(ph, concept_vocab, assignments, rng)
            if result is None:
                return None
            entity_name, _ = result
            resolved_text = resolved_text.replace(ph, entity_name, 1)

        # Resolve fact annotations
        resolved_facts = []
        for type_name, entity_ph in tmpl_turn.facts:
            if entity_ph.startswith("{"):
                result = _resolve_entity(
                    entity_ph, concept_vocab, assignments, rng)
                if result is None:
                    return None
                entity_name, _ = result
            else:
                entity_name = entity_ph
            resolved_facts.append((type_name, entity_name))

        turns.append(DialogueTurn(
            text=resolved_text,
            turn_type=tmpl_turn.turn_type,
            facts=resolved_facts,
            queries=list(tmpl_turn.queries),
        ))

    return turns


# ── Extractors (adapted for DialogueTurn) ────────────────────────────

class DialogueOracleExtractor:
    """Returns ground-truth annotations from turn."""

    def extract(self, turn: DialogueTurn,
                concept_vocab: dict[str, ConceptEntry]) -> list[ExtractedFact]:
        results = []
        for type_name, entity in turn.facts:
            if type_name in concept_vocab:
                pool_names = {e[0]
                              for e in concept_vocab[type_name].entity_pool}
                if entity in pool_names:
                    results.append(ExtractedFact(type_name, entity))
        return results


class DialogueRegexExtractor:
    """Wraps exp24 RegexExtractor for dialogue turns."""

    def __init__(self):
        self._inner = RegexExtractor()

    def extract(self, turn: DialogueTurn,
                concept_vocab: dict[str, ConceptEntry]) -> list[ExtractedFact]:
        # Create an AnnotatedUtterance for the inner extractor
        utt = AnnotatedUtterance(
            text=turn.text,
            facts=turn.facts,
            tier=3,  # doesn't matter for extraction
        )
        return self._inner.extract(utt, concept_vocab)


# ── Single dialogue evaluation ───────────────────────────────────────

def run_dialogue_episode(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    dialogue: list[DialogueTurn],
    extractor: DialogueOracleExtractor | DialogueRegexExtractor,
    all_entity_ids: list[int],
    type_question_ids: dict[str, list[int]],
    erase_lr: float = 5.0,
) -> DialogueResult:
    """Run one dialogue episode: extract → write → query per turn."""
    result = DialogueResult()
    known_facts: dict[str, tuple[str, int]] = {}  # type → (entity, eid)
    fact_write_turn: dict[str, int] = {}  # type → turn_idx when written

    for turn_idx, turn in enumerate(dialogue):
        if turn.turn_type in ("info", "update"):
            # Extract facts
            extracted = extractor.extract(turn, concept_vocab)

            # Enable erase for updates
            if turn.turn_type == "update":
                model.set_erase_mode(True, erase_lr)

            # Write each extracted fact
            for fact in extracted:
                entry = concept_vocab.get(fact.type_name)
                if entry is None:
                    continue
                eid = None
                for ename, ebpe in entry.entity_pool:
                    if ename == fact.entity:
                        eid = ebpe
                        break
                if eid is None:
                    continue
                model.write_fact_direct(entry.concept_token_id, eid)
                known_facts[fact.type_name] = (fact.entity, eid)
                fact_write_turn[fact.type_name] = turn_idx

            # Disable erase after update
            if turn.turn_type == "update":
                model.set_erase_mode(False)

        elif turn.turn_type == "query":
            # Determine which types to query
            query_types = turn.queries if turn.queries else list(
                known_facts.keys())

            model.set_trace_mode(use=True, update=False)
            turn_correct = 0
            turn_total = 0

            for type_name in query_types:
                if type_name not in known_facts:
                    continue
                expected_entity, expected_eid = known_facts[type_name]

                q_ids = type_question_ids.get(type_name)
                if q_ids is None:
                    continue

                pred_id = _predict_answer(model, q_ids, all_entity_ids)
                correct = (pred_id == expected_eid)

                turn_correct += int(correct)
                turn_total += 1
                result.overall_correct += int(correct)
                result.overall_total += 1

                # Track by age
                age = turn_idx - fact_write_turn.get(type_name, turn_idx)
                if age not in result.per_age_correct:
                    result.per_age_correct[age] = []
                result.per_age_correct[age].append(correct)

                # Track update accuracy (fact was written in an "update" turn)
                # We check if this type was last written in an update turn
                # by checking if current known value differs from any earlier
                # ... simplified: just track queries after update turns
                # Actually check turn type of the write turn
                write_tidx = fact_write_turn.get(type_name, -1)
                if write_tidx >= 0 and write_tidx < len(dialogue):
                    if dialogue[write_tidx].turn_type == "update":
                        result.update_correct += int(correct)
                        result.update_total += 1

            model.set_trace_mode(use=False, update=False)

            result.per_query_turn[turn_idx] = (turn_correct, turn_total)

    result.n_facts_final = len(known_facts)
    return result


# ── Phase runners ────────────────────────────────────────────────────

def _prepare_eval(
    concept_vocab: dict[str, ConceptEntry],
    tokenizer: GPT2Tokenizer,
) -> tuple[list[int], dict[str, list[int]]]:
    """Prepare entity IDs and question IDs for evaluation."""
    all_entity_ids = list({
        eid for entry in concept_vocab.values()
        for _, eid in entry.entity_pool
    })
    type_question_ids: dict[str, list[int]] = {}
    for type_name, entry in concept_vocab.items():
        q_ids = tokenizer.encode(entry.question_template,
                                  add_special_tokens=False)
        type_question_ids[type_name] = q_ids
    return all_entity_ids, type_question_ids


def run_phase1_intro(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_eval: int = 50,
    extractor_name: str = "oracle",
    seed: int = 42,
):
    """Phase 1: Introduction dialogues (pure accumulation)."""
    print("\n=== Phase 1: Introduction Dialogues ===")

    extractor: DialogueOracleExtractor | DialogueRegexExtractor
    if extractor_name == "oracle":
        extractor = DialogueOracleExtractor()
    else:
        extractor = DialogueRegexExtractor()

    all_entity_ids, type_question_ids = _prepare_eval(
        concept_vocab, tokenizer)

    intro_templates = [t for t in DIALOGUE_TEMPLATES
                       if t.category == "intro"]
    if not intro_templates:
        print("  No intro templates found!")
        return

    # Aggregate results across all templates and episodes
    all_age_correct: dict[int, list[bool]] = {}
    total_correct = 0
    total_total = 0
    per_template_acc: dict[str, list[float]] = {}

    for tmpl in intro_templates:
        template_correct = 0
        template_total = 0
        template_accs = []

        for ep in range(n_eval):
            rng = random.Random(seed + ep * 997)
            dialogue = instantiate_dialogue(tmpl, concept_vocab, rng)
            if dialogue is None:
                continue

            model.reset_traces()
            result = run_dialogue_episode(
                model, tokenizer, concept_vocab, dialogue,
                extractor, all_entity_ids, type_question_ids,
            )

            template_correct += result.overall_correct
            template_total += result.overall_total
            total_correct += result.overall_correct
            total_total += result.overall_total

            if result.overall_total > 0:
                template_accs.append(
                    result.overall_correct / result.overall_total)

            for age, corrects in result.per_age_correct.items():
                if age not in all_age_correct:
                    all_age_correct[age] = []
                all_age_correct[age].extend(corrects)

        acc = template_correct / max(template_total, 1)
        per_template_acc[tmpl.description] = template_accs
        print(f"  {tmpl.description}: {acc:.1%} "
              f"({template_correct}/{template_total})")

    overall = total_correct / max(total_total, 1)
    print(f"\n  Overall: {overall:.1%} ({total_correct}/{total_total})")

    # Retention by age
    if all_age_correct:
        print(f"\n  Retention by age (turns since write):")
        for age in sorted(all_age_correct):
            corrects = all_age_correct[age]
            age_acc = sum(corrects) / max(len(corrects), 1)
            print(f"    age={age}: {age_acc:.1%} "
                  f"(n={len(corrects)})")

    return overall


def run_phase2_update(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_eval: int = 50,
    extractor_name: str = "oracle",
    erase_lr: float = 5.0,
    seed: int = 42,
):
    """Phase 2: Update dialogues (erase + rewrite)."""
    print("\n=== Phase 2: Update Dialogues ===")

    extractor: DialogueOracleExtractor | DialogueRegexExtractor
    if extractor_name == "oracle":
        extractor = DialogueOracleExtractor()
    else:
        extractor = DialogueRegexExtractor()

    all_entity_ids, type_question_ids = _prepare_eval(
        concept_vocab, tokenizer)

    update_templates = [t for t in DIALOGUE_TEMPLATES
                        if t.category == "update"]
    if not update_templates:
        print("  No update templates found!")
        return

    total_correct = 0
    total_total = 0
    total_update_correct = 0
    total_update_total = 0
    total_stable_correct = 0
    total_stable_total = 0

    for tmpl in update_templates:
        tmpl_correct = 0
        tmpl_total = 0
        tmpl_upd_c = 0
        tmpl_upd_t = 0

        for ep in range(n_eval):
            rng = random.Random(seed + ep * 1009)
            dialogue = instantiate_dialogue(tmpl, concept_vocab, rng)
            if dialogue is None:
                continue

            model.reset_traces()
            result = run_dialogue_episode(
                model, tokenizer, concept_vocab, dialogue,
                extractor, all_entity_ids, type_question_ids,
                erase_lr=erase_lr,
            )

            tmpl_correct += result.overall_correct
            tmpl_total += result.overall_total
            tmpl_upd_c += result.update_correct
            tmpl_upd_t += result.update_total
            total_correct += result.overall_correct
            total_total += result.overall_total
            total_update_correct += result.update_correct
            total_update_total += result.update_total
            stable_c = result.overall_correct - result.update_correct
            stable_t = result.overall_total - result.update_total
            total_stable_correct += stable_c
            total_stable_total += stable_t

        acc = tmpl_correct / max(tmpl_total, 1)
        upd_acc = tmpl_upd_c / max(tmpl_upd_t, 1)
        print(f"  {tmpl.description}: overall={acc:.1%}, "
              f"update={upd_acc:.1%} ({tmpl_upd_c}/{tmpl_upd_t})")

    overall = total_correct / max(total_total, 1)
    upd_overall = total_update_correct / max(total_update_total, 1)
    stable_overall = total_stable_correct / max(total_stable_total, 1)
    print(f"\n  Overall:  {overall:.1%} ({total_correct}/{total_total})")
    print(f"  Updated:  {upd_overall:.1%} "
          f"({total_update_correct}/{total_update_total})")
    print(f"  Stable:   {stable_overall:.1%} "
          f"({total_stable_correct}/{total_stable_total})")

    return overall, upd_overall, stable_overall


def run_phase3_multi(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_dialogues: int = 5,
    n_eval: int = 30,
    extractor_name: str = "oracle",
    seed: int = 42,
):
    """Phase 3: Multi-dialogue sessions (trace persists across dialogues).

    Run K dialogues in sequence. After each dialogue, query ALL facts
    accumulated across all previous dialogues.
    """
    print(f"\n=== Phase 3: Multi-Dialogue Sessions "
          f"({n_dialogues} dialogues) ===")

    extractor: DialogueOracleExtractor | DialogueRegexExtractor
    if extractor_name == "oracle":
        extractor = DialogueOracleExtractor()
    else:
        extractor = DialogueRegexExtractor()

    all_entity_ids, type_question_ids = _prepare_eval(
        concept_vocab, tokenizer)

    # Use intro templates for multi-session (no updates)
    intro_templates = [t for t in DIALOGUE_TEMPLATES
                       if t.category in ("intro", "dense")]

    per_dialogue_accs: dict[int, list[float]] = {
        d: [] for d in range(1, n_dialogues + 1)}

    for ep in range(n_eval):
        model.reset_traces()
        ep_rng = random.Random(seed + ep * 7919)

        # Track all accumulated facts
        all_known: dict[str, tuple[str, int]] = {}

        for dlg_idx in range(1, n_dialogues + 1):
            # Pick a random intro template
            tmpl = ep_rng.choice(intro_templates)
            dialogue = instantiate_dialogue(tmpl, concept_vocab, ep_rng)
            if dialogue is None:
                continue

            # Run dialogue (don't reset traces!)
            # Only write facts for types NOT already known
            # (each dialogue contributes new knowledge)
            for turn in dialogue:
                if turn.turn_type == "info":
                    extracted = extractor.extract(turn, concept_vocab)
                    for fact in extracted:
                        if fact.type_name in all_known:
                            continue  # skip: already known from earlier
                        entry = concept_vocab.get(fact.type_name)
                        if entry is None:
                            continue
                        eid = None
                        for ename, ebpe in entry.entity_pool:
                            if ename == fact.entity:
                                eid = ebpe
                                break
                        if eid is None:
                            continue
                        model.write_fact_direct(
                            entry.concept_token_id, eid)
                        all_known[fact.type_name] = (fact.entity, eid)

            # After dialogue: query ALL accumulated facts
            model.set_trace_mode(use=True, update=False)
            dlg_correct = 0
            dlg_total = 0
            for type_name, (entity, eid) in all_known.items():
                q_ids = type_question_ids.get(type_name)
                if q_ids is None:
                    continue
                pred_id = _predict_answer(model, q_ids, all_entity_ids)
                if pred_id == eid:
                    dlg_correct += 1
                dlg_total += 1
            model.set_trace_mode(use=False, update=False)

            if dlg_total > 0:
                per_dialogue_accs[dlg_idx].append(
                    (dlg_correct / dlg_total, dlg_total))

    # Report
    print(f"\n  {'Dlg':>4} | {'Known':>5} | {'Accuracy':>8}")
    print(f"  " + "-" * 26)
    for dlg_idx in range(1, n_dialogues + 1):
        entries = per_dialogue_accs[dlg_idx]
        if entries:
            mean_acc = sum(a for a, _ in entries) / len(entries)
            mean_known = sum(n for _, n in entries) / len(entries)
            print(f"  {dlg_idx:>4} | {mean_known:>5.1f} | "
                  f"{mean_acc:.1%}")

    return per_dialogue_accs


# ── Main experiment ──────────────────────────────────────────────────

def run_experiment(
    n_eval: int = 50,
    extractor_name: str = "oracle",
    phase: str = "all",
    with_tauto: bool = False,
    completion_alpha: float = 0.3,
    erase_lr: float = 5.0,
    n_dialogues: int = 5,
    quick: bool = False,
    seed: int = 42,
):
    """Run dialogue memory experiment."""
    if quick:
        n_eval = 10

    device = torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    print(f"Device: {device}")

    model, tokenizer = setup_model(alpha=0.5, use_ps=True, device=device)
    concept_vocab = build_concept_vocab(tokenizer)

    print(f"Concept vocabulary: {len(concept_vocab)} types")
    print(f"Dialogue templates: {len(DIALOGUE_TEMPLATES)} "
          f"(intro={sum(1 for t in DIALOGUE_TEMPLATES if t.category=='intro')}, "
          f"update={sum(1 for t in DIALOGUE_TEMPLATES if t.category=='update')}, "
          f"dense={sum(1 for t in DIALOGUE_TEMPLATES if t.category=='dense')})")

    # Optional T_auto
    if with_tauto:
        auto_pairs = extract_auto_pairs(tokenizer)
        pairs_list = [(p.variant_id, p.concept_id) for p in auto_pairs]
        model.write_auto_pairs(pairs_list)
        model.set_auto_mode(True, completion_alpha)
        print(f"T_auto: {len(pairs_list)} pairs, "
              f"completion_alpha={completion_alpha}")

    extractors_to_test = (
        ["oracle", "regex"] if extractor_name == "all"
        else [extractor_name]
    )

    for ext_name in extractors_to_test:
        print(f"\n{'='*60}")
        print(f"Extractor: {ext_name.upper()}")
        print(f"{'='*60}")

        if phase in ("all", "intro"):
            run_phase1_intro(model, tokenizer, concept_vocab,
                             n_eval=n_eval, extractor_name=ext_name,
                             seed=seed)

        if phase in ("all", "update"):
            run_phase2_update(model, tokenizer, concept_vocab,
                              n_eval=n_eval, extractor_name=ext_name,
                              erase_lr=erase_lr, seed=seed)

        if phase in ("all", "multi"):
            run_phase3_multi(model, tokenizer, concept_vocab,
                             n_dialogues=n_dialogues, n_eval=n_eval,
                             extractor_name=ext_name, seed=seed)

    print(f"\n{'='*60}")
    print("Done.")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 25: Multi-turn dialogue memory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (10 episodes)")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--extractor",
                        choices=["oracle", "regex", "all"],
                        default="all")
    parser.add_argument("--phase",
                        choices=["intro", "update", "multi", "all"],
                        default="all")
    parser.add_argument("--with-tauto", action="store_true")
    parser.add_argument("--completion-alpha", type=float, default=0.3)
    parser.add_argument("--erase-lr", type=float, default=5.0)
    parser.add_argument("--n-dialogues", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        n_eval=args.n_eval,
        extractor_name=args.extractor,
        phase=args.phase,
        with_tauto=args.with_tauto,
        completion_alpha=args.completion_alpha,
        erase_lr=args.erase_lr,
        n_dialogues=args.n_dialogues,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
