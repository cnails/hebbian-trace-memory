"""Fact types, entity pools, and evaluation infrastructure.

Provides structured fact templates with BPE-validated single-token entities,
episode generation for evaluation, concept vocabulary for direct write,
multi-hop chain support, and three evaluation modes:
  - baseline: in-context (facts + question), no trace
  - cross_context: trace-only retrieval (THE REAL TEST)
  - cross_context_baseline: no trace, question-only (lower bound)
"""

import random
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer


# -- Entity Pools --

NAMES = [
    "Andrey", "Elena", "John", "Sarah", "Marco", "Yuki", "Priya", "Omar",
    "Lena", "Carlos", "Aisha", "Thomas", "Mika", "Sofia", "David", "Nora",
    "Kenji", "Anna", "Rafael", "Chloe",
]

CITIES = [
    "Moscow", "Paris", "London", "Tokyo", "Berlin", "Sydney", "Cairo",
    "Toronto", "Mumbai", "Seoul", "Rome", "Oslo", "Lima", "Dublin",
    "Prague", "Lisbon", "Vienna", "Athens", "Bangkok", "Havana",
]

COMPANIES = [
    "Google", "Apple", "Tesla", "Amazon", "Netflix", "Spotify", "Adobe",
    "Oracle", "Intel", "Samsung", "Sony", "Nokia", "Uber", "Slack",
    "Stripe", "Zoom", "Figma", "Canva", "Reddit", "Twitter",
]

COLORS = [
    "red", "blue", "green", "yellow", "purple",
    "orange", "pink", "black", "white", "gray",
]

FOODS = [
    "pizza", "sushi", "pasta", "tacos", "curry",
    "ramen", "salad", "steak", "soup", "bread",
]

PETS = [
    "cat", "dog", "rabbit", "parrot", "turtle",
    "hamster", "fish", "snake", "lizard", "horse",
]

HOBBIES = [
    "reading", "swimming", "cooking", "painting", "hiking",
    "cycling", "dancing", "singing", "gaming", "gardening",
]

LANGUAGES = [
    "English", "French", "Russian", "Spanish", "German",
    "Chinese", "Japanese", "Arabic", "Hindi", "Korean",
]

COUNTRIES = [
    "Russia", "France", "USA", "Japan", "Germany", "Brazil", "India",
    "China", "Canada", "Mexico", "Italy", "Spain", "Egypt", "Kenya",
    "Sweden", "Norway", "Chile", "Peru", "Greece", "Poland",
]

# Extended entity pools for 24 fact types
EXTRA_POOLS: dict[str, tuple[list[str], str, str, str]] = {
    "drink": (
        ["coffee", "tea", "water", "juice", "wine", "beer", "milk", "soda"],
        "My drink is {X}.", "is", "What is my drink?",
    ),
    "sport": (
        ["tennis", "golf", "soccer", "baseball", "hockey", "boxing",
         "rugby", "cricket", "skiing"],
        "My sport is {X}.", "is", "What is my sport?",
    ),
    "hobby": (
        list(HOBBIES),
        "My hobby is {X}.", "is", "What is my hobby?",
    ),
    "number": (
        ["seven", "eight", "nine", "three", "four", "five", "six", "ten",
         "twelve", "fifteen", "twenty"],
        "My number is {X}.", "is", "What is my number?",
    ),
    "language": (
        list(LANGUAGES),
        "My language is {X}.", "is", "What is my language?",
    ),
    "animal": (
        ["lion", "tiger", "bear", "eagle", "wolf", "deer", "fox",
         "whale", "dolphin", "hawk", "shark"],
        "My animal is {X}.", "is", "What is my animal?",
    ),
    "instrument": (
        ["piano", "guitar", "violin", "drums", "flute", "harp", "cello",
         "trumpet", "saxophone"],
        "My instrument is {X}.", "is", "What is my instrument?",
    ),
    "season": (
        ["spring", "summer", "autumn", "winter"],
        "My season is {X}.", "is", "What is my season?",
    ),
    "subject": (
        ["math", "science", "history", "art", "music", "physics",
         "chemistry", "biology", "geography", "philosophy"],
        "My subject is {X}.", "is", "What is my subject?",
    ),
    "day": (
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
         "Saturday", "Sunday"],
        "My day is {X}.", "is", "What is my day?",
    ),
    "car": (
        ["Toyota", "Honda", "Ford", "Tesla", "Volvo", "Audi", "BMW",
         "Porsche", "Ferrari", "Lexus"],
        "My car is {X}.", "is", "What is my car?",
    ),
    "metal": (
        ["gold", "silver", "iron", "copper", "steel", "bronze",
         "platinum", "tin", "zinc"],
        "My metal is {X}.", "is", "What is my metal?",
    ),
    "gem": (
        ["diamond", "ruby", "emerald", "pearl", "jade", "amber",
         "sapphire", "opal", "garnet"],
        "My gem is {X}.", "is", "What is my gem?",
    ),
    "flower": (
        ["rose", "lily", "tulip", "daisy", "violet", "orchid",
         "lotus", "iris"],
        "My flower is {X}.", "is", "What is my flower?",
    ),
    "tree": (
        ["oak", "pine", "maple", "birch", "cedar", "elm", "willow",
         "palm", "bamboo"],
        "My tree is {X}.", "is", "What is my tree?",
    ),
    "fruit": (
        ["apple", "banana", "mango", "peach", "cherry", "grape",
         "lemon", "plum", "melon", "kiwi"],
        "My fruit is {X}.", "is", "What is my fruit?",
    ),
    "tool": (
        ["hammer", "wrench", "drill", "saw", "pliers", "shovel",
         "screwdriver", "chisel"],
        "My tool is {X}.", "is", "What is my tool?",
    ),
    "fabric": (
        ["cotton", "silk", "wool", "leather", "linen", "velvet",
         "denim", "nylon"],
        "My fabric is {X}.", "is", "What is my fabric?",
    ),
}

# Filler sentences for paragraph mode
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

FILLER_WITH_LINK = [
    "The weather is nice.",
    "Time is precious.",
    "The answer is unknown.",
    "The solution is simple.",
    "He is tall.",
    "The office is large.",
    "She lives in the suburbs.",
    "They work at the library.",
    "The package is from overseas.",
    "The result is impressive.",
]


# -- Data Classes --

@dataclass
class FactTemplate:
    text: str
    linking_word: str


@dataclass
class QuestionTemplate:
    text: str


@dataclass
class FactType:
    name: str
    entities: list[tuple[str, int]]  # (entity_name, bpe_token_id)
    fact_templates: list[FactTemplate]
    question_templates: list[QuestionTemplate]


@dataclass
class EvalEpisode:
    facts: list[tuple[str, str, int, list[int]]]
    # (type_name, entity_name, entity_bpe_id, fact_bpe_ids)
    train_sequences: list[list[int]]
    test_queries: list[tuple[list[int], int, str]]
    # (query_bpe_ids, answer_bpe_id, type_name)


@dataclass
class EvalResults:
    accuracy: float
    n_correct: int
    n_total: int
    per_episode_acc: list[float]


# -- Entity Validation --

def validate_single_token_entities(
    tokenizer: GPT2Tokenizer,
    entity_pool: list[str],
    prefix: str = " ",
) -> list[tuple[str, int]]:
    """Filter entities to single BPE tokens (space-prefixed)."""
    valid = []
    for entity in entity_pool:
        ids = tokenizer.encode(prefix + entity, add_special_tokens=False)
        if len(ids) == 1:
            valid.append((entity, ids[0]))
    return valid


# -- Fact Type Construction --

def build_fact_types(tokenizer: GPT2Tokenizer) -> list[FactType]:
    """Build 7 base fact types with BPE-validated entities."""
    raw_types = [
        ("name", NAMES, [
            FactTemplate("My name is {X}.", "is"),
        ], [QuestionTemplate("What is my name?")]),
        ("city", CITIES, [
            FactTemplate("I live in {X}.", "in"),
        ], [QuestionTemplate("Where do I live?")]),
        ("company", COMPANIES, [
            FactTemplate("I work at {X}.", "at"),
        ], [QuestionTemplate("Where do I work?")]),
        ("color", COLORS, [
            FactTemplate("My favorite color is {X}.", "is"),
        ], [QuestionTemplate("What is my favorite color?")]),
        ("food", FOODS, [
            FactTemplate("My favorite food is {X}.", "is"),
        ], [QuestionTemplate("What is my favorite food?")]),
        ("pet", PETS, [
            FactTemplate("My pet is {X}.", "is"),
        ], [QuestionTemplate("What is my pet?")]),
        ("country", COUNTRIES, [
            FactTemplate("My country is {X}.", "is"),
        ], [QuestionTemplate("What is my country?")]),
    ]

    fact_types = []
    for name, pool, ftemplates, qtemplates in raw_types:
        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= 5:
            fact_types.append(FactType(
                name=name, entities=entities,
                fact_templates=ftemplates, question_templates=qtemplates,
            ))
    return fact_types


def build_extended_fact_types(
    tokenizer: GPT2Tokenizer,
    min_entities: int = 4,
) -> list[FactType]:
    """Build 24 fact types (7 base + 17 extra) with BPE validation."""
    base_types = build_fact_types(tokenizer)
    base_names = {ft.name for ft in base_types}

    extra_types = []
    for type_name, (pool, fact_text, link_word, q_text) in EXTRA_POOLS.items():
        if type_name in base_names:
            continue
        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= min_entities:
            extra_types.append(FactType(
                name=type_name,
                entities=entities,
                fact_templates=[FactTemplate(fact_text, link_word)],
                question_templates=[QuestionTemplate(q_text)],
            ))

    return base_types + extra_types


def get_linking_bpe_ids(tokenizer: GPT2Tokenizer) -> list[int]:
    """Map linking words to BPE token IDs (space-prefixed)."""
    linking_words = ["is", "in", "at", "from", ":", "am", "me"]
    ids = []
    for word in linking_words:
        bpe_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(bpe_ids) == 1:
            ids.append(bpe_ids[0])
    return ids


# -- Tokenization --

def tokenize_fact(tokenizer: GPT2Tokenizer, template: FactTemplate,
                  entity: str) -> list[int]:
    text = template.text.replace("{X}", entity)
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_question(tokenizer: GPT2Tokenizer,
                      template: QuestionTemplate) -> list[int]:
    return tokenizer.encode(template.text, add_special_tokens=False)


# -- Episode Generation --

def make_eval_episodes(
    n_episodes: int,
    n_facts: int,
    tokenizer: GPT2Tokenizer,
    fact_types: list[FactType],
    seed: int = 42,
) -> list[EvalEpisode]:
    """Generate evaluation episodes with cumulative replay.

    Each episode selects n_facts random types, generates cumulative
    training sequences (replay) and test queries.
    """
    rng = random.Random(seed)
    episodes = []

    for _ in range(n_episodes):
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

        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        train_sequences = []
        for i in range(len(facts)):
            cumulative = []
            for j in range(i + 1):
                if cumulative:
                    cumulative.append(space_id)
                cumulative.extend(facts[j][3])
            train_sequences.append(cumulative)

        test_queries = []
        for ft, (type_name, entity_name, entity_id, _) in zip(
                selected_types, facts):
            q_template = rng.choice(ft.question_templates)
            q_ids = tokenize_question(tokenizer, q_template)
            test_queries.append((q_ids, entity_id, type_name))

        episodes.append(EvalEpisode(
            facts=facts,
            train_sequences=train_sequences,
            test_queries=test_queries,
        ))

    return episodes


# -- Evaluation --

def get_all_entity_ids(fact_types: list[FactType]) -> list[int]:
    """Get all unique entity BPE IDs across all fact types."""
    ids = set()
    for ft in fact_types:
        for _, eid in ft.entities:
            ids.add(eid)
    return sorted(ids)


def _predict_answer(model, query_ids: list[int],
                    entity_ids: list[int]) -> int:
    """Run model on query, return predicted entity token."""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
    pred_logits = logits[0, -1, :]
    entity_logits = pred_logits[entity_ids]
    best_pos = entity_logits.argmax().item()
    return entity_ids[best_pos]


def evaluate_baseline(model, episodes: list[EvalEpisode],
                      fact_types: list[FactType],
                      tokenizer: GPT2Tokenizer,
                      verbose: bool = False) -> EvalResults:
    """In-context baseline: facts + question in one pass, no trace."""
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    entity_ids = get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            ic_ids = []
            for _, _, _, fact_ids in episode.facts:
                if ic_ids:
                    ic_ids.append(
                        tokenizer.encode(" ", add_special_tokens=False)[0])
                ic_ids.extend(fact_ids)
            ic_ids.append(
                tokenizer.encode(" ", add_special_tokens=False)[0])
            ic_ids.extend(query_ids)

            pred_id = _predict_answer(model, ic_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        per_episode.append(ep_correct / max(len(episode.test_queries), 1))

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {per_episode[-1]:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    return EvalResults(
        accuracy=total_correct / max(total_queries, 1),
        n_correct=total_correct, n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_cross_context(
    model, episodes: list[EvalEpisode],
    fact_types: list[FactType],
    verbose: bool = False,
) -> EvalResults:
    """Cross-context: write facts to trace, query with question only.

    THE REAL TEST. Any accuracy above random (~4%) proves that the
    trace module successfully stores and retrieves facts across
    sequence boundaries.
    """
    model.eval()
    entity_ids = get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase (ACh high)
        model.set_trace_mode(use=False, update=True)
        device = next(model.parameters()).device
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor)

        # Read phase (ACh low)
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer(model, query_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        per_episode.append(ep_correct / max(len(episode.test_queries), 1))

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {per_episode[-1]:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    return EvalResults(
        accuracy=total_correct / max(total_queries, 1),
        n_correct=total_correct, n_total=total_queries,
        per_episode_acc=per_episode,
    )


# -- Concept Vocabulary (for direct write) --

# Maps fact type name -> concept word used for Q addressing
# These are the words at position (link_pos - 1) in shift-1 mechanism
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
    """A concept type with its addressing word and entity pool."""
    type_name: str
    concept_word: str
    concept_token_id: int
    fact_template: str
    question_template: str
    entity_pool: list[tuple[str, int]]  # (name, bpe_id)


def build_concept_vocab(
    tokenizer: GPT2Tokenizer,
) -> dict[str, ConceptEntry]:
    """Build concept vocabulary: type_name -> ConceptEntry.

    Each entry maps a fact type to its concept word (the Q-addressing
    token used in direct write) and validated entity pool.
    Includes 7 base + 17 extended = 24 types.
    """
    entries: dict[str, ConceptEntry] = {}

    # Base 7 types
    base_defs = [
        ("name", "name", "My name is {X}.", "What is my name?", NAMES),
        ("city", "live", "I live in {X}.", "Where do I live?", CITIES),
        ("company", "work", "I work at {X}.", "Where do I work?", COMPANIES),
        ("color", "color", "My favorite color is {X}.",
         "What is my favorite color?", COLORS),
        ("food", "food", "My favorite food is {X}.",
         "What is my favorite food?", FOODS),
        ("pet", "pet", "My pet is {X}.", "What is my pet?", PETS),
        ("country", "country", "My country is {X}.",
         "What is my country?", COUNTRIES),
    ]

    for type_name, cword, fact_t, q_t, pool in base_defs:
        cw_ids = tokenizer.encode(" " + cword, add_special_tokens=False)
        if len(cw_ids) != 1:
            continue
        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= 4:
            entries[type_name] = ConceptEntry(
                type_name=type_name,
                concept_word=cword,
                concept_token_id=cw_ids[0],
                fact_template=fact_t,
                question_template=q_t,
                entity_pool=entities,
            )

    # Extended types (same concept word = type name)
    for type_name, (pool, fact_t, link_w, q_t) in EXTRA_POOLS.items():
        cw_ids = tokenizer.encode(" " + type_name, add_special_tokens=False)
        if len(cw_ids) != 1:
            continue
        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= 4:
            entries[type_name] = ConceptEntry(
                type_name=type_name,
                concept_word=type_name,
                concept_token_id=cw_ids[0],
                fact_template=fact_t,
                question_template=q_t,
                entity_pool=entities,
            )

    return entries


# -- Multi-Hop Chain Data --

# Validated city->country pairs (both single-token BPE)
CITY_COUNTRY_PAIRS = [
    ("Moscow", "Russia"),
    ("Paris", "France"),
    ("Tokyo", "Japan"),
    ("Berlin", "Germany"),
    ("Cairo", "Egypt"),
    ("Toronto", "Canada"),
    ("Mumbai", "India"),
    ("Rome", "Italy"),
    ("Oslo", "Norway"),
    ("Lima", "Peru"),
    ("Athens", "Greece"),
]


@dataclass
class ChainEntry:
    """A city->country chain pair with validated token IDs."""
    city: str
    country: str
    city_token_id: int
    country_token_id: int


def build_chain_entries(
    tokenizer: GPT2Tokenizer,
) -> list[ChainEntry]:
    """Build validated city->country chain pairs for multi-hop."""
    entries = []
    for city, country in CITY_COUNTRY_PAIRS:
        city_ids = tokenizer.encode(" " + city, add_special_tokens=False)
        country_ids = tokenizer.encode(" " + country, add_special_tokens=False)
        if len(city_ids) == 1 and len(country_ids) == 1:
            entries.append(ChainEntry(
                city=city, country=country,
                city_token_id=city_ids[0],
                country_token_id=country_ids[0],
            ))
    return entries


# -- T_auto Paraphrase Pairs --

# Maps variant query word -> canonical concept word
# Used to populate T_auto for pattern completion
TAUTO_PAIRS = [
    # Misaligned: shifted template position
    ("I", "name"),
    ("called", "name"),
    ("Call", "name"),
    ("home", "live"),
    ("reside", "live"),
    ("employed", "work"),
    ("hired", "work"),
    # Semantic: entirely different phrasing
    ("title", "name"),
    ("identity", "name"),
    ("residence", "live"),
    ("employer", "work"),
    ("occupation", "work"),
    ("hue", "color"),
    ("shade", "color"),
    ("cuisine", "food"),
    ("dish", "food"),
    ("companion", "pet"),
]


def evaluate_cross_context_baseline(
    model, episodes: list[EvalEpisode],
    fact_types: list[FactType],
) -> EvalResults:
    """Cross-context baseline: no trace, question-only. Expected ~random."""
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    entity_ids = get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode = []

    for episode in episodes:
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            pred_id = _predict_answer(model, query_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        per_episode.append(ep_correct / max(len(episode.test_queries), 1))

    return EvalResults(
        accuracy=total_correct / max(total_queries, 1),
        n_correct=total_correct, n_total=total_queries,
        per_episode_acc=per_episode,
    )
