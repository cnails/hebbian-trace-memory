"""GPT-2 fact memorization tasks and evaluation.

Adapts the NLP fact memorization paradigm from nlp_tasks.py for GPT-2's
BPE tokenizer. Supports both single-token and multi-token entities.

Evaluation protocol (same ACh modulation as nlp_evaluate.py):
  1. Reset traces
  2. Write phase (ACh high): forward fact sequences, trace UPDATE only
  3. Read phase (ACh low): forward question, trace USE only → predict entity

Four modes:
  - baseline: full context (facts + question), no trace
  - cross_context: trace-only, question-only query (THE REAL TEST)
  - cross_context_baseline: no trace, question-only (lower bound, ~random)
"""

import random
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer

from .nlp_tasks import (
    NAMES, CITIES, COMPANIES, COLORS, FOODS,
    PETS, HOBBIES, LANGUAGES, COUNTRIES,
)


# ── Entity pool validation ──────────────────────────────────────────

def validate_single_token_entities(
    tokenizer: GPT2Tokenizer,
    entity_pool: list[str],
    prefix: str = " ",
) -> list[tuple[str, int]]:
    """Filter entities to those that are single BPE tokens.

    GPT-2 uses space-prefixed tokens for words after the first position.
    So "John" in "My name is John" is tokenized as " John" (one token).

    Returns list of (entity_name, bpe_token_id) pairs.
    """
    valid = []
    for entity in entity_pool:
        ids = tokenizer.encode(prefix + entity, add_special_tokens=False)
        if len(ids) == 1:
            valid.append((entity, ids[0]))
    return valid


def validate_entities(
    tokenizer: GPT2Tokenizer,
    entity_pool: list[str],
    prefix: str = " ",
    max_tokens: int = 3,
) -> list[tuple[str, list[int]]]:
    """Validate entities, accepting both single and multi-token BPE.

    Returns list of (entity_name, bpe_token_ids) pairs.
    Filters to entities with 1..max_tokens BPE tokens.
    """
    valid = []
    for entity in entity_pool:
        ids = tokenizer.encode(prefix + entity, add_special_tokens=False)
        if 1 <= len(ids) <= max_tokens:
            valid.append((entity, ids))
    return valid


# ── Multi-token entity pools ────────────────────────────────────────
# BPE-validated for GPT-2. Only entities where GPT-2 reliably completes
# from first token in question context (validated in exp20 Phase 1).

# 2-token entities
MULTI_CITIES = [
    "San Francisco", "New York", "Los Angeles", "Hong Kong",
    "Las Vegas", "Buenos Aires",
]

MULTI_COMPANIES = [
    "Goldman Sachs", "Wells Fargo", "Whole Foods", "Warner Bros",
]

MULTI_COUNTRIES = [
    "New Zealand", "Costa Rica", "Sri Lanka", "Saudi Arabia",
    "South Africa", "North Korea", "Sierra Leone", "Puerto Rico",
]

# 3-token entities (chain completion P(t2)*P(t3) >= 20%)
MULTI_CITIES_3 = [
    "Salt Lake City",   # 65.7% chain
    "Rio de Janeiro",   # 26.1% chain
]

MULTI_COUNTRIES_3 = [
    "Papua New Guinea",  # 98.4% chain
]


# ── Fact type definitions ────────────────────────────────────────────

@dataclass
class GPT2FactTemplate:
    """Fact sentence template for GPT-2."""
    text: str           # e.g. "My name is {X}."
    linking_word: str   # e.g. "is" — which word triggers storage


@dataclass
class GPT2QuestionTemplate:
    """Question template for GPT-2."""
    text: str           # e.g. "What is my name?"


@dataclass
class GPT2FactType:
    """Definition of one fact type."""
    name: str
    entities: list[tuple[str, int]]   # (entity_name, bpe_token_id)
    fact_templates: list[GPT2FactTemplate]
    question_templates: list[GPT2QuestionTemplate]


def build_fact_types(tokenizer: GPT2Tokenizer) -> list[GPT2FactType]:
    """Build fact types with BPE-validated entities.

    Returns only fact types that have ≥5 single-token entities.
    """
    raw_types = [
        ("name", NAMES, [
            GPT2FactTemplate("My name is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my name?"),
        ]),
        ("city", CITIES, [
            GPT2FactTemplate("I live in {X}.", "in"),
        ], [
            GPT2QuestionTemplate("Where do I live?"),
        ]),
        ("company", COMPANIES, [
            GPT2FactTemplate("I work at {X}.", "at"),
        ], [
            GPT2QuestionTemplate("Where do I work?"),
        ]),
        ("color", COLORS, [
            GPT2FactTemplate("My favorite color is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my favorite color?"),
        ]),
        ("food", FOODS, [
            GPT2FactTemplate("My favorite food is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my favorite food?"),
        ]),
        ("pet", PETS, [
            GPT2FactTemplate("My pet is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my pet?"),
        ]),
        ("country", COUNTRIES, [
            GPT2FactTemplate("My country is {X}.", "is"),
        ], [
            GPT2QuestionTemplate("What is my country?"),
        ]),
    ]

    fact_types = []
    for name, pool, ftemplates, qtemplates in raw_types:
        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= 5:
            fact_types.append(GPT2FactType(
                name=name,
                entities=entities,
                fact_templates=ftemplates,
                question_templates=qtemplates,
            ))

    return fact_types


# ── Concept vocabulary (for direct write API) ───────────────────────

# Concept word mapping: type_name → word whose Q addresses trace.
# Base 7 types have non-obvious mappings (shift-1 from linking token).
# Extended types all use concept_word = type_name.
CONCEPT_WORD_MAP: dict[str, str] = {
    "name": "name",
    "city": "live",
    "company": "work",
    "color": "color",
    "food": "food",
    "pet": "pet",
    "country": "country",
}


@dataclass
class ConceptEntry:
    """Maps a concept type to its trace-addressing word and question."""
    type_name: str           # e.g. "name", "city", "drink"
    concept_word: str        # word whose Q addresses trace
    concept_token_id: int    # BPE ID of " {concept_word}"
    fact_template: str       # "My name is {X}." — for display/regex
    question_template: str   # "What is my name?" — for retrieval
    entity_pool: list[tuple[str, int]]  # valid (entity, bpe_id) pairs


def build_concept_vocab(
    tokenizer: GPT2Tokenizer,
    include_extended: bool = True,
    min_entities: int = 4,
) -> dict[str, ConceptEntry]:
    """Build concept vocabulary for direct write API.

    Maps each fact type to its concept word, BPE IDs, and entity pool.
    Covers 7 base types (with CONCEPT_WORD_MAP) + 17 extended types
    (where concept_word = type_name).

    Args:
        tokenizer: GPT-2 tokenizer for BPE validation.
        include_extended: include 17 extended types from EXTRA_POOLS.
        min_entities: minimum single-token entities required.

    Returns:
        dict mapping type_name → ConceptEntry.
    """
    # Avoid circular import — EXTRA_POOLS is only in experiments
    # Define the extended pool data inline (same as exp16)
    extended_pools: dict[str, tuple[list[str], str, str]] = {
        "drink": (["coffee", "tea", "water", "juice", "wine", "beer",
                   "milk", "soda"], "My drink is {X}.", "What is my drink?"),
        "sport": (["tennis", "golf", "soccer", "baseball", "hockey", "boxing",
                   "rugby", "cricket", "skiing"],
                  "My sport is {X}.", "What is my sport?"),
        "hobby": (list(HOBBIES), "My hobby is {X}.", "What is my hobby?"),
        "number": (["seven", "eight", "nine", "three", "four", "five", "six",
                    "ten", "twelve", "fifteen", "twenty"],
                   "My number is {X}.", "What is my number?"),
        "language": (list(LANGUAGES),
                     "My language is {X}.", "What is my language?"),
        "animal": (["lion", "tiger", "bear", "eagle", "wolf", "deer", "fox",
                    "whale", "dolphin", "hawk", "shark"],
                   "My animal is {X}.", "What is my animal?"),
        "instrument": (["piano", "guitar", "violin", "drums", "flute", "harp",
                        "cello", "trumpet", "saxophone"],
                       "My instrument is {X}.", "What is my instrument?"),
        "season": (["spring", "summer", "autumn", "winter"],
                   "My season is {X}.", "What is my season?"),
        "subject": (["math", "science", "history", "art", "music", "physics",
                     "chemistry", "biology", "geography", "philosophy"],
                    "My subject is {X}.", "What is my subject?"),
        "day": (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"], "My day is {X}.", "What is my day?"),
        "car": (["Toyota", "Honda", "Ford", "Tesla", "Volvo", "Audi", "BMW",
                 "Porsche", "Ferrari", "Lexus"],
                "My car is {X}.", "What is my car?"),
        "metal": (["gold", "silver", "iron", "copper", "steel", "bronze",
                   "platinum", "tin", "zinc"],
                  "My metal is {X}.", "What is my metal?"),
        "gem": (["diamond", "ruby", "emerald", "pearl", "jade", "amber",
                 "sapphire", "opal", "garnet"],
                "My gem is {X}.", "What is my gem?"),
        "flower": (["rose", "lily", "tulip", "daisy", "violet", "orchid",
                    "lotus", "iris"],
                   "My flower is {X}.", "What is my flower?"),
        "tree": (["oak", "pine", "maple", "birch", "cedar", "elm", "willow",
                  "palm", "bamboo"],
                 "My tree is {X}.", "What is my tree?"),
        "fruit": (["apple", "banana", "mango", "peach", "cherry", "grape",
                   "lemon", "plum", "melon", "kiwi"],
                  "My fruit is {X}.", "What is my fruit?"),
        "tool": (["hammer", "wrench", "drill", "saw", "pliers", "shovel",
                  "screwdriver", "chisel"],
                 "My tool is {X}.", "What is my tool?"),
        "fabric": (["cotton", "silk", "wool", "leather", "linen", "velvet",
                    "denim", "nylon"],
                   "My fabric is {X}.", "What is my fabric?"),
    }

    vocab: dict[str, ConceptEntry] = {}

    # Base 7 types
    base_data = [
        ("name", NAMES, "My name is {X}.", "What is my name?"),
        ("city", CITIES, "I live in {X}.", "Where do I live?"),
        ("company", COMPANIES, "I work at {X}.", "Where do I work?"),
        ("color", COLORS, "My favorite color is {X}.",
         "What is my favorite color?"),
        ("food", FOODS, "My favorite food is {X}.",
         "What is my favorite food?"),
        ("pet", PETS, "My pet is {X}.", "What is my pet?"),
        ("country", COUNTRIES, "My country is {X}.", "What is my country?"),
    ]

    for type_name, pool, fact_tmpl, q_tmpl in base_data:
        concept_word = CONCEPT_WORD_MAP[type_name]
        cw_ids = tokenizer.encode(" " + concept_word,
                                  add_special_tokens=False)
        if len(cw_ids) != 1:
            continue  # skip multi-token concept words
        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= min_entities:
            vocab[type_name] = ConceptEntry(
                type_name=type_name,
                concept_word=concept_word,
                concept_token_id=cw_ids[0],
                fact_template=fact_tmpl,
                question_template=q_tmpl,
                entity_pool=entities,
            )

    # Extended types (concept_word = type_name for all)
    if include_extended:
        for type_name, (pool, fact_tmpl, q_tmpl) in extended_pools.items():
            if type_name in vocab:
                continue
            concept_word = type_name
            cw_ids = tokenizer.encode(" " + concept_word,
                                      add_special_tokens=False)
            if len(cw_ids) != 1:
                continue
            entities = validate_single_token_entities(tokenizer, pool)
            if len(entities) >= min_entities:
                vocab[type_name] = ConceptEntry(
                    type_name=type_name,
                    concept_word=concept_word,
                    concept_token_id=cw_ids[0],
                    fact_template=fact_tmpl,
                    question_template=q_tmpl,
                    entity_pool=entities,
                )

    return vocab


def get_linking_bpe_ids(tokenizer: GPT2Tokenizer) -> list[int]:
    """Map linking words to BPE token IDs (space-prefixed)."""
    linking_words = ["is", "in", "at", "from", ":", "am", "me"]
    ids = []
    for word in linking_words:
        bpe_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(bpe_ids) == 1:
            ids.append(bpe_ids[0])
    return ids


# ── Tokenization helpers ─────────────────────────────────────────────

def tokenize_fact(tokenizer: GPT2Tokenizer, template: GPT2FactTemplate,
                  entity: str) -> list[int]:
    """Tokenize a fact sentence."""
    text = template.text.replace("{X}", entity)
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_question(tokenizer: GPT2Tokenizer,
                      template: GPT2QuestionTemplate) -> list[int]:
    """Tokenize a question."""
    return tokenizer.encode(template.text, add_special_tokens=False)


# ── Eval episode generation ──────────────────────────────────────────

@dataclass
class GPT2EvalEpisode:
    """One evaluation episode with N facts."""
    facts: list[tuple[str, str, int, list[int]]]
    # (type_name, entity_name, entity_bpe_id, fact_bpe_ids)
    train_sequences: list[list[int]]
    # cumulative fact sequences for write phase
    test_queries: list[tuple[list[int], int, str]]
    # (query_bpe_ids, answer_bpe_id, type_name)


def make_gpt2_eval_episodes(
    n_episodes: int,
    n_facts: int,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    seed: int = 42,
) -> list[GPT2EvalEpisode]:
    """Generate evaluation episodes with BPE tokenization.

    Each episode:
    - Selects n_facts random fact types (no repeat within episode)
    - Selects random entity and template for each
    - Builds cumulative training sequences
    - Generates test queries
    """
    rng = random.Random(seed)
    episodes = []
    all_entity_ids = []
    for ft in fact_types:
        for _, eid in ft.entities:
            if eid not in all_entity_ids:
                all_entity_ids.append(eid)

    for _ in range(n_episodes):
        # Select n_facts fact types (with replacement if needed)
        if n_facts <= len(fact_types):
            selected_types = rng.sample(fact_types, n_facts)
        else:
            selected_types = [rng.choice(fact_types) for _ in range(n_facts)]

        facts = []
        for ft in selected_types:
            entity_name, entity_id = rng.choice(ft.entities)
            template = rng.choice(ft.fact_templates)
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            facts.append((ft.name, entity_name, entity_id, fact_ids))

        # Build cumulative training sequences
        train_sequences = []
        for i in range(len(facts)):
            cumulative = []
            for j in range(i + 1):
                if cumulative:
                    cumulative.append(tokenizer.encode(" ", add_special_tokens=False)[0])
                cumulative.extend(facts[j][3])
            train_sequences.append(cumulative)

        # Generate test queries
        test_queries = []
        for ft, (type_name, entity_name, entity_id, _) in zip(
                selected_types, facts):
            q_template = rng.choice(ft.question_templates)
            q_ids = tokenize_question(tokenizer, q_template)
            test_queries.append((q_ids, entity_id, type_name))

        episodes.append(GPT2EvalEpisode(
            facts=facts,
            train_sequences=train_sequences,
            test_queries=test_queries,
        ))

    return episodes


# ── Evaluation functions ─────────────────────────────────────────────

def _predict_answer(model, query_ids: list[int],
                    entity_ids: list[int]) -> int:
    """Run model on query and return predicted entity token.

    Prediction at last position, restricted to entity tokens.
    """
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)  # (1, S, vocab_size)

    pred_logits = logits[0, -1, :]  # (vocab_size,)
    entity_logits = pred_logits[entity_ids]
    best_pos = entity_logits.argmax().item()
    return entity_ids[best_pos]


def _get_all_entity_ids(fact_types: list[GPT2FactType]) -> list[int]:
    """Get all unique entity BPE IDs across all fact types."""
    ids = set()
    for ft in fact_types:
        for _, eid in ft.entities:
            ids.add(eid)
    return sorted(ids)


@dataclass
class GPT2EvalResults:
    """Results from evaluating one condition."""
    accuracy: float
    n_correct: int
    n_total: int
    per_episode_acc: list[float]


def evaluate_gpt2_baseline(model, episodes: list[GPT2EvalEpisode],
                           fact_types: list[GPT2FactType],
                           tokenizer: GPT2Tokenizer,
                           verbose: bool = False) -> GPT2EvalResults:
    """In-context baseline: full facts + question in one pass, no trace.

    Tests whether GPT-2 can do in-context fact retrieval (should be high).
    """
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    entity_ids = _get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            # Build in-context: all facts + question
            ic_ids = []
            for _, _, _, fact_ids in episode.facts:
                if ic_ids:
                    ic_ids.append(
                        tokenizer.encode(" ", add_special_tokens=False)[0])
                ic_ids.extend(fact_ids)
            # Separator before question
            ic_ids.append(
                tokenizer.encode(" ", add_special_tokens=False)[0])
            ic_ids.extend(query_ids)

            pred_id = _predict_answer(model, ic_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / max(len(episode.test_queries), 1)
        per_episode.append(ep_acc)

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {ep_acc:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    accuracy = total_correct / max(total_queries, 1)
    return GPT2EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_gpt2_cross_context(
    model, episodes: list[GPT2EvalEpisode],
    fact_types: list[GPT2FactType],
    verbose: bool = False,
) -> GPT2EvalResults:
    """Cross-context: trace only, question-only query.

    THE REAL TEST. Write facts to trace, then query with ONLY the question.
    Any accuracy above random comes from trace-stored associations.
    """
    model.eval()
    entity_ids = _get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase (ACh high): encode facts, suppress retrieval
        model.set_trace_mode(use=False, update=True)
        device = next(model.parameters()).device
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)

        # Read phase (ACh low): question-only, retrieve from trace
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer(model, query_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / max(len(episode.test_queries), 1)
        per_episode.append(ep_acc)

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {ep_acc:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    accuracy = total_correct / max(total_queries, 1)
    return GPT2EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


# ── Paragraph-level episode generation (exp10) ─────────────────────

# Filler sentences WITHOUT linking tokens — gate should NOT fire
FILLER_NO_LINK = [
    "Alice loves her job.",
    "The day was sunny and warm.",
    "They often go hiking together.",
    "Everyone enjoyed the party.",
    "She always arrives early.",
    "The team won the championship.",
    "He plays guitar every evening.",
    "The garden looks beautiful today.",
    "We traveled across the country.",
    "The children played outside all day.",
]

# Filler sentences WITH linking tokens — gate WILL fire, stores noise Q→V
FILLER_WITH_LINK = [
    "The weather is nice.",
    "Time is precious.",
    "The answer is unknown.",
    "The solution is simple.",
    "He is tall.",
    "The office is large.",
    "She lives in the suburbs.",       # "in" triggers
    "They work at the library.",       # "at" triggers
    "The package is from overseas.",   # "from" triggers
    "The result is impressive.",
]


def build_session_paragraph(
    fact_bpe_sequences: list[list[int]],
    tokenizer: GPT2Tokenizer,
    filler_mode: str = "mixed",
    n_filler: int = 3,
    rng: random.Random | None = None,
) -> list[int]:
    """Build a paragraph interleaving facts with filler sentences.

    Pattern: fact0 filler0 fact1 filler1 fact2 ... [remaining filler]

    Args:
        fact_bpe_sequences: list of tokenized fact sentences
        tokenizer: GPT-2 tokenizer
        filler_mode: "none"|"safe"|"noisy"|"mixed"
        n_filler: number of filler sentences to insert
        rng: random number generator (default: Random(0))

    Returns:
        paragraph_bpe_ids: single tokenized sequence
    """
    if rng is None:
        rng = random.Random(0)
    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    # Select filler sentences
    fillers_ids: list[list[int]] = []
    if filler_mode != "none" and n_filler > 0:
        if filler_mode == "safe":
            pool = FILLER_NO_LINK
        elif filler_mode == "noisy":
            pool = FILLER_WITH_LINK
        elif filler_mode == "mixed":
            pool = FILLER_NO_LINK + FILLER_WITH_LINK
        else:
            raise ValueError(f"Unknown filler_mode: {filler_mode}")
        fillers_text = [rng.choice(pool) for _ in range(n_filler)]
        fillers_ids = [
            tokenizer.encode(f, add_special_tokens=False) for f in fillers_text
        ]

    # Interleave facts and filler
    paragraph_ids: list[int] = []
    filler_idx = 0
    for fact_ids in fact_bpe_sequences:
        if paragraph_ids:
            paragraph_ids.append(space_id)
        paragraph_ids.extend(fact_ids)
        if filler_idx < len(fillers_ids):
            paragraph_ids.append(space_id)
            paragraph_ids.extend(fillers_ids[filler_idx])
            filler_idx += 1

    # Append any remaining filler at the end
    while filler_idx < len(fillers_ids):
        paragraph_ids.append(space_id)
        paragraph_ids.extend(fillers_ids[filler_idx])
        filler_idx += 1

    return paragraph_ids


def make_paragraph_episodes(
    n_episodes: int,
    n_facts: int,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    filler_mode: str = "none",
    n_filler: int = 0,
    write_mode: str = "single",
    seed: int = 42,
) -> list[GPT2EvalEpisode]:
    """Generate paragraph-format evaluation episodes.

    Facts are interleaved with optional filler sentences and written
    as a single paragraph (or cumulative passes for comparison).

    Args:
        n_episodes: number of episodes to generate
        n_facts: facts per episode
        tokenizer: GPT-2 tokenizer
        fact_types: available fact types with validated entities
        filler_mode:
            "none"  — no filler, just concatenated facts
            "safe"  — filler WITHOUT linking tokens (gate stays quiet)
            "noisy" — filler WITH linking tokens (gate fires on noise)
            "mixed" — 50/50 safe + noisy filler
        n_filler: number of filler sentences to intersperse
        write_mode:
            "single"     — one pass of entire paragraph
            "cumulative" — N passes (same as exp9, for comparison)
        seed: random seed

    Returns:
        list of GPT2EvalEpisode with paragraph-format train_sequences
    """
    # Separate RNGs for facts and filler to ensure same facts
    # regardless of filler_mode/n_filler. Critical for fair comparison:
    # all conditions must test on identical fact combinations.
    fact_rng = random.Random(seed)
    filler_rng = random.Random(seed + 999999)
    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
    episodes = []

    for _ in range(n_episodes):
        # Select fact types (with replacement if needed)
        if n_facts <= len(fact_types):
            selected_types = fact_rng.sample(fact_types, n_facts)
        else:
            selected_types = [fact_rng.choice(fact_types)
                              for _ in range(n_facts)]

        # Generate facts
        facts = []
        for ft in selected_types:
            entity_name, entity_id = fact_rng.choice(ft.entities)
            template = fact_rng.choice(ft.fact_templates)
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            facts.append((ft.name, entity_name, entity_id, fact_ids))

        # Select filler sentences (separate RNG — doesn't affect fact selection)
        fillers_text = []
        if filler_mode != "none" and n_filler > 0:
            if filler_mode == "safe":
                pool = FILLER_NO_LINK
            elif filler_mode == "noisy":
                pool = FILLER_WITH_LINK
            elif filler_mode == "mixed":
                pool = FILLER_NO_LINK + FILLER_WITH_LINK
            else:
                raise ValueError(f"Unknown filler_mode: {filler_mode}")
            fillers_text = [filler_rng.choice(pool) for _ in range(n_filler)]

        # Tokenize fillers
        fillers_ids = [
            tokenizer.encode(f, add_special_tokens=False) for f in fillers_text
        ]

        # Build paragraph: interleave facts and filler
        # Pattern: fact0 [filler0] fact1 [filler1] fact2 ...
        paragraph_ids = []
        filler_idx = 0
        for i, (_, _, _, fact_ids) in enumerate(facts):
            if paragraph_ids:
                paragraph_ids.append(space_id)
            paragraph_ids.extend(fact_ids)
            # Insert filler after this fact (if available)
            if filler_idx < len(fillers_ids):
                paragraph_ids.append(space_id)
                paragraph_ids.extend(fillers_ids[filler_idx])
                filler_idx += 1

        # Build train_sequences based on write_mode
        if write_mode == "single":
            train_sequences = [paragraph_ids]
        elif write_mode == "cumulative":
            # Same as exp9: cumulative fact sequences (no filler in cumulative)
            train_sequences = []
            for i in range(len(facts)):
                cumulative = []
                for j in range(i + 1):
                    if cumulative:
                        cumulative.append(space_id)
                    cumulative.extend(facts[j][3])
                train_sequences.append(cumulative)
        else:
            raise ValueError(f"Unknown write_mode: {write_mode}")

        # Generate test queries (uses fact_rng, not filler_rng)
        test_queries = []
        for ft, (type_name, entity_name, entity_id, _) in zip(
                selected_types, facts):
            q_template = fact_rng.choice(ft.question_templates)
            q_ids = tokenize_question(tokenizer, q_template)
            test_queries.append((q_ids, entity_id, type_name))

        episodes.append(GPT2EvalEpisode(
            facts=facts,
            train_sequences=train_sequences,
            test_queries=test_queries,
        ))

    return episodes


def evaluate_gpt2_cross_context_baseline(
    model, episodes: list[GPT2EvalEpisode],
    fact_types: list[GPT2FactType],
    verbose: bool = False,
) -> GPT2EvalResults:
    """Cross-context baseline: no trace, question-only. Expected ~random."""
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    entity_ids = _get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer(model, query_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / max(len(episode.test_queries), 1)
        per_episode.append(ep_acc)

    accuracy = total_correct / max(total_queries, 1)
    return GPT2EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )
