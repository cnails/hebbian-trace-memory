"""Exp 24: Free-text fact extraction + direct trace write.

Level 1 of "LLM as extractor + trace as memory": accept free-form text,
extract (concept, entity) pairs, write to trace via direct API.

Four evaluation phases:
  Phase 1: Extraction quality (precision/recall/F1 per extractor)
  Phase 2: Direct write equivalence (T_v match vs template write)
  Phase 3: End-to-end accuracy (extract → direct write → query)
  Phase 4: Multi-session with T_auto (composition test)

Usage:
  python -m hebbian_trace.experiments.exp24_free_text --quick
  python -m hebbian_trace.experiments.exp24_free_text --verify-direct
  python -m hebbian_trace.experiments.exp24_free_text --with-tauto
"""

import argparse
import random
import re
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    GPT2FactType, GPT2FactTemplate, GPT2QuestionTemplate,
    build_fact_types, build_concept_vocab, ConceptEntry,
    validate_single_token_entities, get_linking_bpe_ids,
    tokenize_fact, tokenize_question, _predict_answer, _get_all_entity_ids,
)
from .exp22_pattern_completion import extract_auto_pairs


# ── Device selection ─────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class ExtractedFact:
    """One extracted (concept, entity) pair."""
    type_name: str       # "name", "city", etc.
    entity: str          # "John", "Paris", etc.
    confidence: float = 1.0


@dataclass
class AnnotatedUtterance:
    """Free-text utterance with ground-truth annotations."""
    text: str
    facts: list[tuple[str, str]]  # [(type_name, entity), ...]
    tier: int                     # 1-5 difficulty


# ── Corpus ───────────────────────────────────────────────────────────

# Entity placeholders: {NAME}, {CITY}, etc. will be replaced with
# random entities from the concept vocab at evaluation time.

CORPUS_TEMPLATES: list[tuple[str, list[tuple[str, str]], int]] = [
    # Tier 1: Direct (matches fact template exactly)
    ("My name is {NAME}.", [("name", "{NAME}")], 1),
    ("I live in {CITY}.", [("city", "{CITY}")], 1),
    ("I work at {COMPANY}.", [("company", "{COMPANY}")], 1),
    ("My favorite color is {COLOR}.", [("color", "{COLOR}")], 1),
    ("My favorite food is {FOOD}.", [("food", "{FOOD}")], 1),
    ("My pet is {PET}.", [("pet", "{PET}")], 1),
    ("My country is {COUNTRY}.", [("country", "{COUNTRY}")], 1),
    ("My drink is {DRINK}.", [("drink", "{DRINK}")], 1),
    ("My sport is {SPORT}.", [("sport", "{SPORT}")], 1),

    # Tier 2: Simple rephrase
    ("I'm {NAME}.", [("name", "{NAME}")], 2),
    ("The name's {NAME}.", [("name", "{NAME}")], 2),
    ("My home is in {CITY}.", [("city", "{CITY}")], 2),
    ("I'm based in {CITY}.", [("city", "{CITY}")], 2),
    ("I'm employed at {COMPANY}.", [("company", "{COMPANY}")], 2),
    ("I love {COLOR}.", [("color", "{COLOR}")], 2),
    ("I enjoy eating {FOOD}.", [("food", "{FOOD}")], 2),
    ("I have a {PET}.", [("pet", "{PET}")], 2),
    ("I come from {COUNTRY}.", [("country", "{COUNTRY}")], 2),
    ("I usually drink {DRINK}.", [("drink", "{DRINK}")], 2),
    ("I play {SPORT}.", [("sport", "{SPORT}")], 2),

    # Tier 3: Embedded in chatter
    ("Hey, I'm {NAME}, nice to meet you.", [("name", "{NAME}")], 3),
    ("So yeah, I live in {CITY}, it's great.", [("city", "{CITY}")], 3),
    ("Well, I work at {COMPANY} currently.", [("company", "{COMPANY}")], 3),
    ("By the way, my name is {NAME}.", [("name", "{NAME}")], 3),
    ("Actually, I'm from {COUNTRY}, originally.",
     [("country", "{COUNTRY}")], 3),
    ("You know, my favorite color is {COLOR}, always has been.",
     [("color", "{COLOR}")], 3),
    ("I think my pet is a {PET}, not sure though.",
     [("pet", "{PET}")], 3),

    # Tier 4: Multi-fact
    ("I'm {NAME} from {CITY}.",
     [("name", "{NAME}"), ("city", "{CITY}")], 4),
    ("My name is {NAME} and I live in {CITY}.",
     [("name", "{NAME}"), ("city", "{CITY}")], 4),
    ("I'm {NAME}, I work at {COMPANY}.",
     [("name", "{NAME}"), ("company", "{COMPANY}")], 4),
    ("I'm {NAME} from {COUNTRY}, working at {COMPANY}.",
     [("name", "{NAME}"), ("country", "{COUNTRY}"),
      ("company", "{COMPANY}")], 4),
    ("My name is {NAME}, my favorite color is {COLOR}.",
     [("name", "{NAME}"), ("color", "{COLOR}")], 4),
    ("I live in {CITY} and I love {FOOD}.",
     [("city", "{CITY}"), ("food", "{FOOD}")], 4),
    ("I'm {NAME}, I have a {PET} and I play {SPORT}.",
     [("name", "{NAME}"), ("pet", "{PET}"), ("sport", "{SPORT}")], 4),

    # Tier 5: Indirect / unusual phrasing
    ("Call me {NAME}.", [("name", "{NAME}")], 5),
    ("Everyone calls me {NAME}.", [("name", "{NAME}")], 5),
    ("People know me as {NAME}.", [("name", "{NAME}")], 5),
    ("{CITY} is where I live.", [("city", "{CITY}")], 5),
    ("{COMPANY} is my workplace.", [("company", "{COMPANY}")], 5),
    ("I can't live without {FOOD}.", [("food", "{FOOD}")], 5),
    ("Nothing beats {COLOR} for me.", [("color", "{COLOR}")], 5),
    ("Born and raised in {COUNTRY}.", [("country", "{COUNTRY}")], 5),

    # Tier 6: Free-form (RegexExtractor CANNOT match these)
    ("So {NAME} here, recently relocated to {CITY}.",
     [("name", "{NAME}"), ("city", "{CITY}")], 6),
    ("{NAME} checking in — been spending time in {COUNTRY} lately.",
     [("name", "{NAME}"), ("country", "{COUNTRY}")], 6),
    ("Between you and me, {COLOR} has always been my thing.",
     [("color", "{COLOR}")], 6),
    ("Grew up eating {FOOD}, still my go-to.",
     [("food", "{FOOD}")], 6),
    ("{CITY} has been home for three years now.",
     [("city", "{CITY}")], 6),
    ("{COMPANY} keeps me busy these days.",
     [("company", "{COMPANY}")], 6),
    ("Just moved — {CITY} is the new base.",
     [("city", "{CITY}")], 6),
    ("Funny story, I ended up at {COMPANY} somehow.",
     [("company", "{COMPANY}")], 6),
    ("My go-to drink? Definitely {DRINK}.",
     [("drink", "{DRINK}")], 6),
    ("{FOOD} is basically a food group for me.",
     [("food", "{FOOD}")], 6),
    ("Honestly, {COLOR} is the only color that matters.",
     [("color", "{COLOR}")], 6),
    ("After years abroad, {NAME} settled in {CITY} and found a job at {COMPANY}.",
     [("name", "{NAME}"), ("city", "{CITY}"), ("company", "{COMPANY}")], 6),
    ("{NAME} always had a thing for {COLOR} and {FOOD}.",
     [("name", "{NAME}"), ("color", "{COLOR}"), ("food", "{FOOD}")], 6),
    ("You know {NAME} — the one from {COUNTRY} who adores {PET}.",
     [("name", "{NAME}"), ("country", "{COUNTRY}"), ("pet", "{PET}")], 6),
    ("Ever since {NAME} discovered {SPORT}, {CITY} life changed completely.",
     [("name", "{NAME}"), ("sport", "{SPORT}"), ("city", "{CITY}")], 6),
]


def _resolve_placeholder(placeholder: str, concept_vocab: dict[str, ConceptEntry],
                          assignments: dict[str, tuple[str, int]],
                          rng: random.Random) -> tuple[str, int]:
    """Resolve a placeholder like {NAME} to a random entity."""
    type_name = placeholder.strip("{}").lower()
    if type_name in assignments:
        return assignments[type_name]
    entry = concept_vocab.get(type_name)
    if entry is None:
        raise KeyError(f"Unknown type '{type_name}' in corpus template")
    name, eid = rng.choice(entry.entity_pool)
    assignments[type_name] = (name, eid)
    return name, eid


def instantiate_corpus(
    concept_vocab: dict[str, ConceptEntry],
    n_utterances: int | None = None,
    seed: int = 42,
) -> list[AnnotatedUtterance]:
    """Instantiate corpus templates with random entities.

    Each call produces different random entity assignments.
    If n_utterances is None, uses all templates.
    """
    rng = random.Random(seed)
    templates = list(CORPUS_TEMPLATES)
    if n_utterances is not None:
        templates = rng.sample(templates,
                               min(n_utterances, len(templates)))

    utterances = []
    for text_tmpl, fact_annotations, tier in templates:
        assignments: dict[str, tuple[str, int]] = {}
        # Check all types are in vocab
        skip = False
        for type_name, _ in fact_annotations:
            tn = type_name
            if tn not in concept_vocab:
                skip = True
                break
        if skip:
            continue

        # Resolve all placeholders in the text
        resolved_text = text_tmpl
        for ph in re.findall(r'\{[A-Z]+\}', text_tmpl):
            entity_name, _ = _resolve_placeholder(
                ph, concept_vocab, assignments, rng)
            resolved_text = resolved_text.replace(ph, entity_name, 1)

        # Resolve fact annotations
        resolved_facts = []
        for type_name, entity_ph in fact_annotations:
            if entity_ph.startswith("{"):
                entity_name, _ = _resolve_placeholder(
                    entity_ph, concept_vocab, assignments, rng)
            else:
                entity_name = entity_ph
            resolved_facts.append((type_name, entity_name))

        utterances.append(AnnotatedUtterance(
            text=resolved_text,
            facts=resolved_facts,
            tier=tier,
        ))

    return utterances


# ── Extractors ───────────────────────────────────────────────────────

class OracleExtractor:
    """Returns ground-truth annotations. Upper bound for extraction."""

    def extract(self, utterance: AnnotatedUtterance,
                concept_vocab: dict[str, ConceptEntry]) -> list[ExtractedFact]:
        results = []
        for type_name, entity in utterance.facts:
            if type_name in concept_vocab:
                # Validate entity is in pool
                pool_names = {e[0] for e in concept_vocab[type_name].entity_pool}
                if entity in pool_names:
                    results.append(ExtractedFact(type_name, entity))
        return results


class RegexExtractor:
    """Pattern-based extraction with entity_pool disambiguation."""

    # Patterns per type: list of (regex, capture_group_index)
    # Ordered by specificity — most specific first
    PATTERNS: dict[str, list[str]] = {
        "name": [
            r"My name is (\w+)",
            r"The name's (\w+)",
            r"I'm (\w+)",
            r"Call me (\w+)",
            r"Everyone calls me (\w+)",
            r"People know me as (\w+)",
        ],
        "city": [
            r"I live in (\w+)",
            r"My home is in (\w+)",
            r"I'm based in (\w+)",
            r"(?:I )?moved to (\w+)",
            r"(\w+) is where I live",
        ],
        "company": [
            r"I work at (\w+)",
            r"I'm employed at (\w+)",
            r"working at (\w+)",
            r"I'm at (\w+)",
            r"(\w+) is my workplace",
        ],
        "color": [
            r"My favorite color is (\w+)",
            r"I love (\w+)",
            r"Nothing beats (\w+) for me",
        ],
        "food": [
            r"My favorite food is (\w+)",
            r"I enjoy eating (\w+)",
            r"I can't live without (\w+)",
            r"I (?:now )?prefer (\w+)",
        ],
        "pet": [
            r"My pet is (?:a )?(\w+)",
            r"I have a (\w+)",
        ],
        "country": [
            r"My country is (\w+)",
            r"I come from (\w+)",
            r"I'm from (\w+)",
            r"Born and raised in (\w+)",
        ],
        "drink": [
            r"My drink is (\w+)",
            r"I usually drink (\w+)",
        ],
        "sport": [
            r"My sport is (\w+)",
            r"I play (\w+)",
        ],
    }

    def extract(self, utterance: AnnotatedUtterance,
                concept_vocab: dict[str, ConceptEntry]) -> list[ExtractedFact]:
        text = utterance.text
        results = []
        used_entities: set[str] = set()  # avoid duplicate extractions

        for type_name, patterns in self.PATTERNS.items():
            if type_name not in concept_vocab:
                continue
            pool_names = {e[0] for e in concept_vocab[type_name].entity_pool}

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Entity pool disambiguation: only accept if entity
                    # is in THIS type's pool. This resolves ambiguity
                    # like "I'm from Russia" (country) vs "I'm from Paris" (city)
                    if match in pool_names and match not in used_entities:
                        results.append(ExtractedFact(type_name, match))
                        used_entities.add(match)
                        break  # one match per type per pattern set

        return results


class LLMExtractor:
    """LLM-based fact extraction using Flan-T5-small (80M).

    Strategy: per-type extractive QA with dual validation.
    For each candidate fact type, asks Flan-T5 a targeted question,
    then validates the answer against:
      1. The type's entity pool (must be a known entity)
      2. The source text (extracted entity must appear in input)

    Pre-filter: only queries types whose entity pool has at least one
    entity present in the text, reducing ~24 queries to ~2-3.
    """

    TYPE_PROMPTS: dict[str, str] = {
        "name": 'What is the person\'s name? "{text}"',
        "city": 'What city is mentioned? "{text}"',
        "company": 'What company is mentioned? "{text}"',
        "color": 'What color is mentioned? "{text}"',
        "food": 'What food is mentioned? "{text}"',
        "pet": 'What pet is mentioned? "{text}"',
        "country": 'What country is mentioned? "{text}"',
        "drink": 'What drink is mentioned? "{text}"',
        "sport": 'What sport is mentioned? "{text}"',
        "hobby": 'What hobby is mentioned? "{text}"',
        "animal": 'What animal is mentioned? "{text}"',
        "instrument": 'What instrument is mentioned? "{text}"',
        "fruit": 'What fruit is mentioned? "{text}"',
        "flower": 'What flower is mentioned? "{text}"',
        "tree": 'What tree is mentioned? "{text}"',
        "metal": 'What metal is mentioned? "{text}"',
        "gem": 'What gem is mentioned? "{text}"',
        "car": 'What car brand is mentioned? "{text}"',
        "fabric": 'What fabric is mentioned? "{text}"',
        "tool": 'What tool is mentioned? "{text}"',
        "day": 'What day is mentioned? "{text}"',
        "season": 'What season is mentioned? "{text}"',
        "number": 'What number is mentioned? "{text}"',
        "subject": 'What subject is mentioned? "{text}"',
        "language": 'What language is mentioned? "{text}"',
    }

    def __init__(self, device: torch.device | None = None):
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self):
        """Lazy-load Flan-T5-small on first use."""
        if self._model is not None:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        print("  Loading Flan-T5-small for LLM extraction...")
        self._tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self._model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-small")
        if self._device is None:
            self._device = get_device()
        self._model.to(self._device)
        self._model.eval()

    def _candidate_types(self, text: str,
                         concept_vocab: dict[str, ConceptEntry],
                         ) -> list[str]:
        """Pre-filter: only query types with pool entity present in text."""
        text_lower = text.lower()
        candidates = []
        for type_name, entry in concept_vocab.items():
            for entity_name, _ in entry.entity_pool:
                if entity_name.lower() in text_lower:
                    candidates.append(type_name)
                    break
        return candidates

    def _query_llm(self, text: str, type_name: str) -> str:
        """Run one extractive QA query. Returns raw model output."""
        self._ensure_loaded()
        prompt_template = self.TYPE_PROMPTS.get(
            type_name,
            f'What {type_name} is mentioned? "{{text}}"',
        )
        prompt = prompt_template.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=10)
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @staticmethod
    def _validate_entity(raw_output: str, text: str,
                         pool_names: set[str]) -> str | None:
        """Dual validation: entity must be in pool AND in source text."""
        for word in raw_output.split():
            for candidate in [word, word.capitalize(), word.lower(),
                              word.upper()]:
                if candidate in pool_names:
                    if candidate.lower() in text.lower():
                        return candidate
        return None

    def extract(self, utterance: AnnotatedUtterance,
                concept_vocab: dict[str, ConceptEntry],
                ) -> list[ExtractedFact]:
        results = []
        used_entities: set[str] = set()

        for type_name in self._candidate_types(utterance.text, concept_vocab):
            entry = concept_vocab[type_name]
            pool_names = {e[0] for e in entry.entity_pool}
            raw = self._query_llm(utterance.text, type_name)
            entity = self._validate_entity(raw, utterance.text, pool_names)
            if entity is not None and entity not in used_entities:
                results.append(ExtractedFact(type_name, entity))
                used_entities.add(entity)

        return results


# ── Phase 1: Extraction Quality ──────────────────────────────────────

def _compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def run_phase1_extraction(
    concept_vocab: dict[str, ConceptEntry],
    extractor_names: list[str] | None = None,
    n_eval: int = 50,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Measure extraction precision/recall/F1 per extractor and tier."""
    if extractor_names is None:
        extractor_names = ["oracle", "regex"]

    all_extractors: dict[str, OracleExtractor | RegexExtractor | LLMExtractor]
    all_extractors = {}
    for name in extractor_names:
        if name == "oracle":
            all_extractors[name] = OracleExtractor()
        elif name == "regex":
            all_extractors[name] = RegexExtractor()
        elif name == "llm":
            all_extractors[name] = LLMExtractor()

    results: dict[str, dict[str, float]] = {}
    # Collect per-tier F1 for comparison table
    tier_f1_table: dict[str, dict[int, float]] = {}

    for ext_name, extractor in all_extractors.items():
        corpus = instantiate_corpus(concept_vocab, seed=seed)
        tp = fp = fn = 0
        per_tier: dict[int, dict[str, int]] = {}

        for utt in corpus:
            tier = utt.tier
            if tier not in per_tier:
                per_tier[tier] = {"tp": 0, "fp": 0, "fn": 0}

            extracted = extractor.extract(utt, concept_vocab)
            ext_set = {(f.type_name, f.entity) for f in extracted}
            gt_set = set(utt.facts)

            tier_tp = len(ext_set & gt_set)
            tier_fp = len(ext_set - gt_set)
            tier_fn = len(gt_set - ext_set)

            tp += tier_tp
            fp += tier_fp
            fn += tier_fn
            per_tier[tier]["tp"] += tier_tp
            per_tier[tier]["fp"] += tier_fp
            per_tier[tier]["fn"] += tier_fn

        precision, recall, f1 = _compute_f1(tp, fp, fn)
        results[ext_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        tier_f1_table[ext_name] = {}

        print(f"\n  {ext_name.upper()} extractor:")
        print(f"    Overall: P={precision:.1%} R={recall:.1%} F1={f1:.1%}")
        print(f"    (TP={tp}, FP={fp}, FN={fn})")

        for tier in sorted(per_tier):
            t = per_tier[tier]
            t_p, t_r, t_f = _compute_f1(t["tp"], t["fp"], t["fn"])
            tier_f1_table[ext_name][tier] = t_f
            print(f"    Tier {tier}: P={t_p:.0%} R={t_r:.0%} F1={t_f:.0%} "
                  f"(TP={t['tp']}, FP={t['fp']}, FN={t['fn']})")

    # Per-tier comparison table (Tier × Extractor → F1)
    if len(all_extractors) > 1:
        all_tiers = sorted({t for d in tier_f1_table.values() for t in d})
        ext_names = list(all_extractors.keys())
        print(f"\n  {'Tier':>6} |", end="")
        for en in ext_names:
            print(f" {en:>8}", end="")
        print()
        print("  " + "-" * (9 + 9 * len(ext_names)))
        for tier in all_tiers:
            print(f"  {tier:>6} |", end="")
            for en in ext_names:
                f1_val = tier_f1_table.get(en, {}).get(tier, 0.0)
                print(f" {f1_val:>7.0%}", end="")
            print()
        # Overall row
        print(f"  {'ALL':>6} |", end="")
        for en in ext_names:
            print(f" {results[en]['f1']:>7.1%}", end="")
        print()

    return results


# ── Phase 2: Direct Write Equivalence ────────────────────────────────

def run_phase2_verify_direct(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    fact_types: list[GPT2FactType],
) -> bool:
    """Verify that write_fact_direct produces identical T_v as template write.

    For each base fact type × 10 entities:
      1. Write via template (standard forward pass)
      2. Write via write_fact_direct()
      3. Compare T_v tensors

    Returns True if max abs diff < 1e-5.
    """
    device = next(model.parameters()).device
    max_diff = 0.0
    n_tested = 0

    print("\n=== Phase 2: Direct Write Equivalence ===")

    # Only test base 7 types that exist in both fact_types and concept_vocab
    for ft in fact_types:
        if ft.name not in concept_vocab:
            continue
        entry = concept_vocab[ft.name]
        n_entities = min(10, len(ft.entities))

        type_max_diff = 0.0

        for entity_name, entity_id in ft.entities[:n_entities]:
            # Method A: Template write (standard forward pass)
            model.reset_traces()
            model.set_trace_mode(use=False, update=True)
            fact_ids = tokenize_fact(tokenizer, ft.fact_templates[0],
                                     entity_name)
            input_tensor = torch.tensor(
                [fact_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)
            Tv_template = model.trace.value_traces.clone()

            # Method B: Direct write
            model.reset_traces()
            model.write_fact_direct(entry.concept_token_id, entity_id)
            Tv_direct = model.trace.value_traces.clone()

            # Compare
            diff = (Tv_template - Tv_direct).abs().max().item()
            type_max_diff = max(type_max_diff, diff)
            max_diff = max(max_diff, diff)
            n_tested += 1

        status = "PASS" if type_max_diff < 1e-5 else "FAIL"
        print(f"  {ft.name}: max_diff={type_max_diff:.2e} [{status}]")

    overall_pass = max_diff < 1e-5
    status = "PASS" if overall_pass else "FAIL"
    print(f"\n  Overall: max_diff={max_diff:.2e} [{status}] "
          f"({n_tested} entity writes tested)")

    if not overall_pass:
        print("  WARNING: Direct write does NOT match template write!")
        print("  This means the write_direct() implementation has a bug.")

    return overall_pass


# ── Phase 3: End-to-End Accuracy ─────────────────────────────────────

def run_phase3_e2e(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_facts_list: list[int],
    n_eval: int = 50,
    extractor_name: str = "oracle",
    seed: int = 42,
) -> dict[int, float]:
    """End-to-end: extract → direct write → query → measure accuracy.

    For each n_facts:
      1. Sample n_facts utterances (single-fact only for clean measurement)
      2. Extract facts with chosen extractor
      3. Write to trace via write_fact_direct()
      4. Query with standard questions
      5. Measure accuracy
    """
    extractor: OracleExtractor | RegexExtractor | LLMExtractor
    if extractor_name == "oracle":
        extractor = OracleExtractor()
    elif extractor_name == "llm":
        extractor = LLMExtractor()
    else:
        extractor = RegexExtractor()

    device = next(model.parameters()).device
    rng = random.Random(seed)

    # Pre-compute entity IDs for answer prediction
    all_entity_ids = list({
        eid for entry in concept_vocab.values()
        for _, eid in entry.entity_pool
    })

    # Build per-type question BPE IDs
    type_question_ids: dict[str, list[int]] = {}
    for type_name, entry in concept_vocab.items():
        q_ids = tokenizer.encode(entry.question_template,
                                  add_special_tokens=False)
        type_question_ids[type_name] = q_ids

    results: dict[int, float] = {}

    for n_facts in n_facts_list:
        total_correct = 0
        total_queries = 0

        for ep in range(n_eval):
            # Generate corpus with unique seed per episode
            corpus = instantiate_corpus(concept_vocab,
                                         seed=seed + ep * 1000)
            # Filter to single-fact utterances only
            single_fact = [u for u in corpus if len(u.facts) == 1]
            if len(single_fact) < n_facts:
                continue

            # Sample n_facts utterances
            selected = rng.sample(single_fact, n_facts)

            # Reset trace
            model.reset_traces()

            # Extract and write
            written_facts: list[tuple[str, str, int]] = []
            for utt in selected:
                extracted = extractor.extract(utt, concept_vocab)
                for fact in extracted:
                    entry = concept_vocab.get(fact.type_name)
                    if entry is None:
                        continue
                    # Find entity BPE ID
                    eid = None
                    for ename, ebpe in entry.entity_pool:
                        if ename == fact.entity:
                            eid = ebpe
                            break
                    if eid is None:
                        continue
                    # Direct write
                    model.write_fact_direct(entry.concept_token_id, eid)
                    written_facts.append(
                        (fact.type_name, fact.entity, eid))

            # Query each written fact
            model.set_trace_mode(use=True, update=False)
            for type_name, entity_name, entity_id in written_facts:
                q_ids = type_question_ids.get(type_name)
                if q_ids is None:
                    continue
                pred_id = _predict_answer(model, q_ids, all_entity_ids)
                if pred_id == entity_id:
                    total_correct += 1
                total_queries += 1
            model.set_trace_mode(use=False, update=False)

        acc = total_correct / max(total_queries, 1)
        results[n_facts] = acc

    return results


# ── Phase 4: Multi-session with T_auto ───────────────────────────────

def run_phase4_multi_session(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_sessions: int = 5,
    facts_per_session: int = 3,
    n_eval: int = 30,
    completion_alpha: float = 0.3,
    seed: int = 42,
) -> dict[int, float]:
    """Multi-session: direct write + T_auto + query across sessions."""
    device = next(model.parameters()).device
    rng = random.Random(seed)

    # Setup T_auto
    auto_pair_objs = extract_auto_pairs(tokenizer)
    auto_pairs = [(p.variant_id, p.concept_id) for p in auto_pair_objs]

    all_entity_ids = list({
        eid for entry in concept_vocab.values()
        for _, eid in entry.entity_pool
    })

    type_question_ids: dict[str, list[int]] = {}
    for type_name, entry in concept_vocab.items():
        q_ids = tokenizer.encode(entry.question_template,
                                  add_special_tokens=False)
        type_question_ids[type_name] = q_ids

    # Available types for sampling
    available_types = [tn for tn in concept_vocab
                       if tn in type_question_ids]

    per_session_acc: dict[int, list[float]] = {
        s: [] for s in range(1, n_sessions + 1)}

    for ep in range(n_eval):
        model.reset_traces()

        # Write T_auto pairs
        model.write_auto_pairs(auto_pairs)
        model.set_auto_mode(True, completion_alpha)

        # Track all written facts across sessions
        all_written: list[tuple[str, str, int]] = []
        ep_rng = random.Random(seed + ep * 7919)

        for session in range(1, n_sessions + 1):
            # Sample fact types for this session (no duplicates with previous)
            used_types = {tn for tn, _, _ in all_written}
            remaining = [tn for tn in available_types
                         if tn not in used_types]
            if len(remaining) < facts_per_session:
                remaining = available_types  # allow reuse
            session_types = ep_rng.sample(
                remaining, min(facts_per_session, len(remaining)))

            # Write new facts
            for type_name in session_types:
                entry = concept_vocab[type_name]
                entity_name, entity_id = ep_rng.choice(entry.entity_pool)
                model.write_fact_direct(entry.concept_token_id, entity_id)
                all_written.append((type_name, entity_name, entity_id))

            # Query ALL written facts so far
            model.set_trace_mode(use=True, update=False)
            session_correct = 0
            session_total = 0
            for type_name, entity_name, entity_id in all_written:
                q_ids = type_question_ids.get(type_name)
                if q_ids is None:
                    continue
                pred_id = _predict_answer(model, q_ids, all_entity_ids)
                if pred_id == entity_id:
                    session_correct += 1
                session_total += 1
            model.set_trace_mode(use=False, update=False)

            acc = session_correct / max(session_total, 1)
            per_session_acc[session].append(acc)

    # Average across episodes
    results: dict[int, float] = {}
    for session in range(1, n_sessions + 1):
        accs = per_session_acc[session]
        results[session] = sum(accs) / max(len(accs), 1)

    return results


# ── Model setup ──────────────────────────────────────────────────────

def setup_model(
    alpha: float = 0.5,
    use_ps: bool = True,
    device: torch.device | None = None,
) -> tuple[GPT2WithTrace, GPT2Tokenizer]:
    """Load GPT-2 with trace, pattern separation, linking tokens."""
    if device is None:
        device = get_device()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
    )
    model.to(device)

    # Linking tokens
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    # Pattern separation
    if use_ps:
        model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    return model, tokenizer


# ── Main experiment runner ───────────────────────────────────────────

def run_experiment(
    n_eval: int = 50,
    extractor_name: str = "all",
    verify_direct: bool = False,
    with_tauto: bool = False,
    completion_alpha: float = 0.3,
    quick: bool = False,
    seed: int = 42,
):
    """Run all experiment phases."""
    if quick:
        n_eval = 10

    device = get_device()
    print(f"Device: {device}")

    # Setup
    model, tokenizer = setup_model(alpha=0.5, use_ps=True, device=device)
    fact_types = build_fact_types(tokenizer)
    concept_vocab = build_concept_vocab(tokenizer)

    print(f"\nConcept vocabulary: {len(concept_vocab)} types")
    for tn, entry in sorted(concept_vocab.items()):
        print(f"  {tn}: concept='{entry.concept_word}' "
              f"(id={entry.concept_token_id}), "
              f"{len(entry.entity_pool)} entities")

    # Phase 1: Extraction quality
    print("\n" + "=" * 60)
    print("=== Phase 1: Extraction Quality ===")
    extractors_to_test = (
        ["oracle", "regex", "llm"] if extractor_name == "all"
        else [extractor_name]
    )
    phase1 = run_phase1_extraction(
        concept_vocab, extractor_names=extractors_to_test,
        n_eval=n_eval, seed=seed)

    # Phase 2: Direct write equivalence
    if verify_direct or True:  # always run — it's fast and critical
        print("\n" + "=" * 60)
        phase2_pass = run_phase2_verify_direct(
            model, tokenizer, concept_vocab, fact_types)
        if not phase2_pass:
            print("\n  ABORTING: direct write equivalence failed.")
            return

    # Phase 3: End-to-end accuracy
    print("\n" + "=" * 60)
    print("=== Phase 3: End-to-End Accuracy ===")
    n_facts_list = [1, 3, 5, 7]

    phase3_results: dict[str, dict[int, float]] = {}
    for ext in extractors_to_test:
        print(f"\n  Extractor: {ext.upper()}")
        results = run_phase3_e2e(
            model, tokenizer, concept_vocab,
            n_facts_list=n_facts_list, n_eval=n_eval,
            extractor_name=ext, seed=seed,
        )
        phase3_results[ext] = results

        for n, acc in sorted(results.items()):
            print(f"    n={n}: {acc:.1%}")

    # Comparison table
    if len(extractors_to_test) > 1:
        print(f"\n  {'n':>3} |", end="")
        for ext in extractors_to_test:
            print(f" {ext:>10}", end="")
        print()
        print("  " + "-" * (6 + 11 * len(extractors_to_test)))
        for n in n_facts_list:
            print(f"  {n:>3} |", end="")
            for ext in extractors_to_test:
                acc = phase3_results[ext].get(n, 0)
                print(f" {acc:>9.1%}", end="")
            print()

    # Phase 4: Multi-session with T_auto
    if with_tauto:
        print("\n" + "=" * 60)
        print("=== Phase 4: Multi-Session + T_auto ===")
        phase4 = run_phase4_multi_session(
            model, tokenizer, concept_vocab,
            n_sessions=5, facts_per_session=3,
            n_eval=n_eval, completion_alpha=completion_alpha,
            seed=seed,
        )
        print(f"\n  Session | Known | Accuracy")
        print(f"  " + "-" * 30)
        for session, acc in sorted(phase4.items()):
            known = session * 3
            print(f"  {session:>7} | {known:>5} | {acc:.1%}")

    print("\n" + "=" * 60)
    print("Done.")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 24: Free-text extraction + direct trace write")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (10 episodes)")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--extractor",
                        choices=["oracle", "regex", "llm", "all"],
                        default="all",
                        help="Which extractor to test")
    parser.add_argument("--verify-direct", action="store_true",
                        help="Run Phase 2 only (direct write verification)")
    parser.add_argument("--with-tauto", action="store_true",
                        help="Include Phase 4 (multi-session + T_auto)")
    parser.add_argument("--completion-alpha", type=float, default=0.3,
                        help="T_auto completion alpha")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.verify_direct:
        # Quick Phase 2 only
        device = get_device()
        model, tokenizer = setup_model(alpha=0.5, use_ps=True,
                                        device=device)
        fact_types = build_fact_types(tokenizer)
        concept_vocab = build_concept_vocab(tokenizer)
        run_phase2_verify_direct(model, tokenizer, concept_vocab,
                                  fact_types)
        return

    run_experiment(
        n_eval=args.n_eval,
        extractor_name=args.extractor,
        verify_direct=args.verify_direct,
        with_tauto=args.with_tauto,
        completion_alpha=args.completion_alpha,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
