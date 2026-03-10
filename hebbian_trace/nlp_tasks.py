"""Natural language fact memorization tasks for Hebbian Attention experiments.

Fact-based cross-session memory: the model sees facts like
"My name is Andrey . I live in Moscow ." in session 1,
and must answer "What is my name ?" → "Andrey" in session 2
using only trace-stored associations.

Template tiers:
  Tier 1: "concept LINK entity" pattern (shift=2) — mechanism works directly.
  Tier 2: alternative phrasings without concept word — hard mode, expect ~random.
"""

import random
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset


# ── Entity value pools ──────────────────────────────────────────────

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

AGES = [
    "twenty", "twentyfive", "thirty", "thirtyfive", "forty",
    "fortyfive", "fifty", "fiftyfive", "sixty", "sixtyfive",
]

COUNTRIES = [
    "Russia", "France", "USA", "Japan", "Germany", "Brazil", "India",
    "China", "Canada", "Mexico", "Italy", "Spain", "Egypt", "Kenya",
    "Sweden", "Norway", "Chile", "Peru", "Greece", "Poland",
]


# ── Fact type definitions ───────────────────────────────────────────

@dataclass
class FactTemplate:
    """A single sentence template for expressing a fact."""
    words: list[str]       # e.g. ["My", "name", "is", "{X}", "."]
    tier: int              # 1 = concept-link-entity (mechanism works), 2 = hard
    concept_word: str | None = None  # Q override: inject this word's Q at storage pos


@dataclass
class QuestionTemplate:
    """A question template for querying a fact."""
    words: list[str]       # e.g. ["What", "is", "my", "name", "?"]


@dataclass
class FactType:
    """Definition of one fact type with templates and value pool."""
    name: str
    values: list[str]
    fact_templates: list[FactTemplate]
    question_templates: list[QuestionTemplate]


FACT_TYPES = [
    FactType(
        name="name",
        values=NAMES,
        fact_templates=[
            FactTemplate(["My", "name", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "name", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "name", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "name", "is", "{X}", "."], tier=1),
            FactTemplate(["A", "name", ":", "{X}", "."], tier=1),
            FactTemplate(["I", "am", "{X}", "."], tier=2, concept_word="name"),
            FactTemplate(["Call", "me", "{X}", "."], tier=2, concept_word="name"),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "name", "?"]),
        ],
    ),
    FactType(
        name="city",
        values=CITIES,
        fact_templates=[
            FactTemplate(["I", "live", "in", "{X}", "."], tier=1),
            FactTemplate(["We", "live", "in", "{X}", "."], tier=1),
            FactTemplate(["They", "live", "in", "{X}", "."], tier=1),
            FactTemplate(["He", "live", "in", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["Where", "do", "I", "live", "?"]),
        ],
    ),
    FactType(
        name="company",
        values=COMPANIES,
        fact_templates=[
            FactTemplate(["I", "work", "at", "{X}", "."], tier=1),
            FactTemplate(["We", "work", "at", "{X}", "."], tier=1),
            FactTemplate(["They", "work", "at", "{X}", "."], tier=1),
            FactTemplate(["He", "work", "at", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["Where", "do", "I", "work", "?"]),
        ],
    ),
    FactType(
        name="color",
        values=COLORS,
        fact_templates=[
            FactTemplate(["My", "color", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "color", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "color", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "color", "is", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "color", "?"]),
        ],
    ),
    FactType(
        name="food",
        values=FOODS,
        fact_templates=[
            FactTemplate(["My", "food", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "food", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "food", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "food", "is", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "food", "?"]),
        ],
    ),
    FactType(
        name="pet",
        values=PETS,
        fact_templates=[
            FactTemplate(["My", "pet", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "pet", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "pet", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "pet", "is", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "pet", "?"]),
        ],
    ),
    FactType(
        name="hobby",
        values=HOBBIES,
        fact_templates=[
            FactTemplate(["My", "hobby", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "hobby", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "hobby", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "hobby", "is", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "hobby", "?"]),
        ],
    ),
    FactType(
        name="language",
        values=LANGUAGES,
        fact_templates=[
            FactTemplate(["My", "language", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "language", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "language", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "language", "is", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "language", "?"]),
        ],
    ),
    FactType(
        name="age",
        values=AGES,
        fact_templates=[
            FactTemplate(["My", "age", "is", "{X}", "."], tier=1),
            FactTemplate(["The", "age", "is", "{X}", "."], tier=1),
            FactTemplate(["His", "age", "is", "{X}", "."], tier=1),
            FactTemplate(["Her", "age", "is", "{X}", "."], tier=1),
        ],
        question_templates=[
            QuestionTemplate(["What", "is", "my", "age", "?"]),
        ],
    ),
    FactType(
        name="country",
        values=COUNTRIES,
        fact_templates=[
            # Original templates store Q("come"), but question retrieves
            # Q("from") → mismatch. With concept injection, override
            # storage Q to Q("from") for consistency.
            FactTemplate(["I", "come", "from", "{X}", "."], tier=1,
                         concept_word="from"),
            FactTemplate(["We", "come", "from", "{X}", "."], tier=1,
                         concept_word="from"),
            FactTemplate(["They", "come", "from", "{X}", "."], tier=1,
                         concept_word="from"),
            FactTemplate(["He", "come", "from", "{X}", "."], tier=1,
                         concept_word="from"),
        ],
        question_templates=[
            QuestionTemplate(["Where", "do", "I", "come", "from", "?"]),
        ],
    ),
]


# ── Vocabulary ──────────────────────────────────────────────────────

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"


def _collect_all_words() -> list[str]:
    """Gather all unique words from templates and entity pools."""
    words = set()
    for ft in FACT_TYPES:
        for tmpl in ft.fact_templates:
            for w in tmpl.words:
                if w != "{X}":
                    words.add(w)
        for qt in ft.question_templates:
            words.update(qt.words)
        words.update(ft.values)
    return sorted(words)


@dataclass
class NLPVocab:
    """Word-level vocabulary for NLP fact tasks."""

    tokens: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tokens:
            all_words = _collect_all_words()
            self.tokens = [PAD, BOS, EOS] + all_words
        self.tok2idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx2tok = {i: t for i, t in enumerate(self.tokens)}
        self.pad_idx = self.tok2idx[PAD]
        self.bos_idx = self.tok2idx[BOS]
        self.eos_idx = self.tok2idx[EOS]

        # Linking token indices (used for trace filtering)
        self.linking_tokens = []
        for tok in ["is", "in", "at", "from", ":", "am", "me"]:
            if tok in self.tok2idx:
                self.linking_tokens.append(self.tok2idx[tok])

        # All entity token indices (for restricting predictions)
        self._entity_indices = None

    @property
    def entity_indices(self) -> list[int]:
        """Indices of all entity tokens (names, cities, etc.)."""
        if self._entity_indices is None:
            entities = set()
            for ft in FACT_TYPES:
                entities.update(ft.values)
            self._entity_indices = sorted(
                self.tok2idx[e] for e in entities if e in self.tok2idx
            )
        return self._entity_indices

    def __len__(self):
        return len(self.tokens)

    def encode(self, words: list[str]) -> list[int]:
        """Convert word list to index list."""
        return [self.tok2idx[w] for w in words]

    def decode(self, indices: list[int]) -> list[str]:
        """Convert index list to word list."""
        return [self.idx2tok[i] for i in indices]

    def decode_str(self, indices: list[int]) -> str:
        """Convert index list to readable string."""
        return " ".join(self.decode(indices))


# Global vocab instance
NLP_VOCAB = NLPVocab()


# ── Dataset ─────────────────────────────────────────────────────────

class NLPFactDataset(Dataset):
    """Training dataset: fact statements + question-answer pairs.

    Each sample is a sequence like:
        <bos> My name is Andrey . I live in Moscow . What is my name ? Andrey <eos>

    The model learns autoregressive prediction with 10x weight on the
    answer token position.
    """

    def __init__(self, n_sequences: int = 10000, max_facts: int = 5,
                 max_seq_len: int = 128, seed: int = 42,
                 tier: int | None = None):
        """
        Args:
            n_sequences: number of training sequences to generate.
            max_facts: max number of facts per sequence (sampled 1..max_facts).
            max_seq_len: maximum sequence length (pad/truncate).
            seed: random seed.
            tier: if set, only use templates of this tier (1 or 2).
        """
        self.max_seq_len = max_seq_len
        self.vocab = NLP_VOCAB
        rng = random.Random(seed)
        self.samples = []

        for _ in range(n_sequences):
            n_facts = rng.randint(1, max_facts)
            # Sample fact types without replacement
            fact_types = rng.sample(FACT_TYPES, min(n_facts, len(FACT_TYPES)))
            facts = []
            for ft in fact_types:
                value = rng.choice(ft.values)
                # Pick a random template
                templates = ft.fact_templates
                if tier is not None:
                    templates = [t for t in templates if t.tier == tier]
                    if not templates:
                        templates = ft.fact_templates
                tmpl = rng.choice(templates)
                words = [value if w == "{X}" else w for w in tmpl.words]
                facts.append((ft, value, words))

            # Pick a random question from the selected fact types
            q_ft, q_value, _ = rng.choice(facts)
            q_tmpl = rng.choice(q_ft.question_templates)

            # Build sequence: <bos> facts... question answer <eos>
            seq_words = [BOS]
            for _, _, words in facts:
                seq_words.extend(words)
            seq_words.extend(q_tmpl.words)
            seq_words.append(q_value)  # answer
            seq_words.append(EOS)

            indices = self.vocab.encode(seq_words)
            self.samples.append((indices, len(seq_words) - 2))  # answer pos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, answer_pos = self.samples[idx]

        # Pad or truncate
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        padded = indices + [self.vocab.pad_idx] * (self.max_seq_len - len(indices))

        input_ids = torch.tensor(padded[:-1], dtype=torch.long)
        target_ids = torch.tensor(padded[1:], dtype=torch.long)

        # Loss mask: 1.0 everywhere, 10.0 on answer token, 0.0 on padding
        loss_mask = torch.zeros(self.max_seq_len - 1, dtype=torch.float)
        seq_len = min(len(indices), self.max_seq_len) - 1
        loss_mask[:seq_len] = 1.0
        # Answer position in target is answer_pos - 1 (shifted by 1)
        ans_target_pos = answer_pos - 1
        if 0 <= ans_target_pos < self.max_seq_len - 1:
            loss_mask[ans_target_pos] = 10.0

        return input_ids, target_ids, loss_mask


# ── Evaluation episodes ────────────────────────────────────────────

@dataclass
class NLPEvalEpisode:
    """One evaluation episode: a set of facts to memorize and queries to test."""
    facts: list[tuple[str, str, list[str]]]  # (fact_type_name, value, word_list)
    fact_type_objs: list[FactType]           # corresponding FactType objects
    fact_templates_used: list[FactTemplate]  # which template was used per fact
    train_sequences: list[list[int]]          # cumulative fact sequences for trace
    test_queries: list[tuple[list[int], int, str, int]]
    # (query_indices, answer_idx, fact_type_name, template_tier)


def _build_fact_sequence(facts_words: list[list[str]], vocab: NLPVocab) -> list[int]:
    """Build a token sequence from fact word lists."""
    seq = [BOS]
    for words in facts_words:
        seq.extend(words)
    seq.append(EOS)
    return vocab.encode(seq)


def make_nlp_eval_episodes(
    n_episodes: int = 200,
    n_facts: int = 3,
    seed: int = 42,
    tier: int | None = None,
) -> list[NLPEvalEpisode]:
    """Generate evaluation episodes.

    Each episode:
    1. Sample n_facts fact types + values
    2. Build cumulative training sequences (for trace accumulation)
    3. Build test queries (one per fact, both Tier 1 and Tier 2)

    Args:
        n_episodes: number of episodes.
        n_facts: number of facts per episode.
        seed: random seed.
        tier: if set, only use templates of this tier for storage.
    """
    rng = random.Random(seed)
    vocab = NLP_VOCAB
    episodes = []

    for _ in range(n_episodes):
        # Sample fact types and values
        fact_types = rng.sample(FACT_TYPES, min(n_facts, len(FACT_TYPES)))
        facts = []
        templates_used = []
        for ft in fact_types:
            value = rng.choice(ft.values)
            # For storage, pick a template of the requested tier
            templates = ft.fact_templates
            if tier is not None:
                tier_templates = [t for t in templates if t.tier == tier]
                if tier_templates:
                    templates = tier_templates
            tmpl = rng.choice(templates)
            words = [value if w == "{X}" else w for w in tmpl.words]
            facts.append((ft.name, value, words))
            templates_used.append(tmpl)

        fact_type_objs = fact_types

        # Cumulative training sequences
        train_seqs = []
        for i in range(1, len(facts) + 1):
            words_list = [f[2] for f in facts[:i]]
            train_seqs.append(_build_fact_sequence(words_list, vocab))

        # Test queries: one per fact
        test_queries = []
        for ft, (ft_name, value, _) in zip(fact_types, facts):
            q_tmpl = rng.choice(ft.question_templates)
            # Query: <bos> question_words (without answer)
            query_words = [BOS] + q_tmpl.words
            query_indices = vocab.encode(query_words)
            answer_idx = vocab.tok2idx[value]

            # Determine which tier was used for storage
            storage_tier = 1  # default
            for t in ft.fact_templates:
                stored_words = [value if w == "{X}" else w
                                for w in t.words]
                if stored_words == facts[fact_types.index(ft)][2]:
                    storage_tier = t.tier
                    break

            test_queries.append(
                (query_indices, answer_idx, ft_name, storage_tier))

        episodes.append(NLPEvalEpisode(
            facts=facts,
            fact_type_objs=fact_type_objs,
            fact_templates_used=templates_used,
            train_sequences=train_seqs,
            test_queries=test_queries,
        ))

    return episodes


# ── Fact Update Episodes ──────────────────────────────────────────

@dataclass
class NLPUpdateEvalEpisode:
    """Evaluation episode with fact updates.

    Phase 1: store N original facts via cumulative sequences.
    Phase 2: update K facts with new values (different entity, same type).
    Test: query all N facts.
        - Updated facts: correct answer = NEW value.
        - Stable facts: correct answer = ORIGINAL value.
    """
    original_facts: list[tuple[str, str, list[str]]]  # (type_name, value, words)
    updated_facts: list[tuple[str, str, list[str]]]   # (type_name, new_value, words)
    updated_indices: list[int]                          # which originals were updated
    fact_type_objs: list[FactType]

    phase1_sequences: list[list[int]]  # cumulative original facts
    phase2_sequences: list[list[int]]  # cumulative updated facts only

    # (query_indices, correct_idx, old_idx_or_None, type_name, was_updated)
    test_queries: list[tuple[list[int], int, int | None, str, bool]]


def make_nlp_update_eval_episodes(
    n_episodes: int = 200,
    n_facts: int = 5,
    n_updates: int = 2,
    seed: int = 42,
    tier: int | None = 1,
) -> list[NLPUpdateEvalEpisode]:
    """Generate episodes where some facts are updated after initial storage.

    Phase 1 stores all n_facts via cumulative sequences (same as standard eval).
    Phase 2 presents only the updated facts (new values, same concept types).
    Test queries all n_facts — updated ones should return new value.

    Args:
        n_episodes: number of episodes.
        n_facts: total number of facts per episode.
        n_updates: number of facts to update (randomly selected).
        seed: random seed.
        tier: template tier for storage (1 = concept-link-entity).
    """
    rng = random.Random(seed)
    vocab = NLP_VOCAB
    episodes = []

    for _ in range(n_episodes):
        fact_types = rng.sample(FACT_TYPES, min(n_facts, len(FACT_TYPES)))

        # Generate original facts
        original_facts = []
        for ft in fact_types:
            value = rng.choice(ft.values)
            templates = ft.fact_templates
            if tier is not None:
                tier_templates = [t for t in templates if t.tier == tier]
                if tier_templates:
                    templates = tier_templates
            tmpl = rng.choice(templates)
            words = [value if w == "{X}" else w for w in tmpl.words]
            original_facts.append((ft.name, value, words))

        # Select which facts to update
        n_upd = min(n_updates, len(original_facts))
        update_indices = sorted(rng.sample(range(len(original_facts)), n_upd))

        # Generate updated facts (same type, DIFFERENT value)
        updated_facts = []
        for idx in update_indices:
            ft = fact_types[idx]
            old_value = original_facts[idx][1]
            new_value = rng.choice([v for v in ft.values if v != old_value])
            templates = ft.fact_templates
            if tier is not None:
                tier_templates = [t for t in templates if t.tier == tier]
                if tier_templates:
                    templates = tier_templates
            tmpl = rng.choice(templates)
            words = [new_value if w == "{X}" else w for w in tmpl.words]
            updated_facts.append((ft.name, new_value, words))

        # Phase 1: cumulative original facts
        phase1_seqs = []
        for i in range(1, len(original_facts) + 1):
            words_list = [f[2] for f in original_facts[:i]]
            phase1_seqs.append(_build_fact_sequence(words_list, vocab))

        # Phase 2: cumulative updated facts only
        phase2_seqs = []
        for i in range(1, len(updated_facts) + 1):
            words_list = [f[2] for f in updated_facts[:i]]
            phase2_seqs.append(_build_fact_sequence(words_list, vocab))

        # Test queries
        test_queries = []
        for i, (ft, (ft_name, orig_value, _)) in enumerate(
                zip(fact_types, original_facts)):
            q_tmpl = rng.choice(ft.question_templates)
            query_words = [BOS] + q_tmpl.words
            query_indices = vocab.encode(query_words)

            was_updated = i in update_indices
            if was_updated:
                update_pos = update_indices.index(i)
                new_value = updated_facts[update_pos][1]
                correct_idx = vocab.tok2idx[new_value]
                old_idx = vocab.tok2idx[orig_value]
            else:
                correct_idx = vocab.tok2idx[orig_value]
                old_idx = None

            test_queries.append(
                (query_indices, correct_idx, old_idx, ft_name, was_updated))

        episodes.append(NLPUpdateEvalEpisode(
            original_facts=original_facts,
            updated_facts=updated_facts,
            updated_indices=update_indices,
            fact_type_objs=fact_types,
            phase1_sequences=phase1_seqs,
            phase2_sequences=phase2_seqs,
            test_queries=test_queries,
        ))

    return episodes


# ── Concept Injection ────────────────────────────────────────────

def compute_concept_injection(
    facts_words: list[list[str]],
    templates_used: list[FactTemplate],
    vocab: NLPVocab,
) -> dict[int, int] | None:
    """Compute Q injection map for a fact sequence.

    For templates with concept_word set, returns a dict mapping
    sequence positions to concept token IDs. These positions need
    their Q overridden with the concept word's Q during trace storage.

    The injection position accounts for the shift-1 offset in
    _hebbian_update: Q_store[i] is activated when the linking mask
    fires at token_ids[i+1]. So for "I am Andrey ." in a sequence
    starting at position `pos` after BOS:
      - "am" is at sequence position pos+1
      - mask checks token_ids[:, 1:-1], so mask_index = pos+1-1 = pos
      - Q_store[pos] = Q at position pos = Q("I")
      - We inject Q("name") at position pos

    Args:
        facts_words: list of word lists, one per fact.
        templates_used: corresponding FactTemplate per fact.
        vocab: vocabulary for token ID lookup.

    Returns:
        dict {position: concept_token_id} or None if no injection needed.
    """
    injection = {}
    pos = 1  # start after BOS
    for fact_words, tmpl in zip(facts_words, templates_used):
        if tmpl.concept_word is not None:
            concept_tok_id = vocab.tok2idx[tmpl.concept_word]
            # Find {X} in template to locate linking token
            for j, w in enumerate(tmpl.words):
                if w == "{X}" and j > 0:
                    # Word before {X} is the linking token (e.g., "am")
                    # Linking token is at sequence position: pos + (j-1)
                    # mask checks token_ids[:, 1:-1]:
                    #   mask_index = (pos + j - 1) - 1 = pos + j - 2
                    # Q_store[mask_index] = Q at position (pos + j - 2)
                    # We inject concept Q at this position
                    inject_pos = pos + j - 2
                    if inject_pos >= 0:
                        injection[inject_pos] = concept_tok_id
                    break
        pos += len(fact_words)

    return injection if injection else None
