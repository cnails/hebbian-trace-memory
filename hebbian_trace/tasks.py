"""Synthetic tasks for Hebbian Attention experiments.

Key-value association: sequences like "A→7,B→3,C→9,A→?" → "7"
The model learns in-context key-value retrieval during pretraining (backprop).
During evaluation, trace accumulates associations across separate forward passes.
"""

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


# ── Vocabulary ──────────────────────────────────────────────────────

# Keys: uppercase letters A-Z
KEYS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Values: digits 0-9
VALUES = list("0123456789")
# Special tokens
SPECIAL = ["→", "?", ","]
# Control tokens
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

ALL_TOKENS = [PAD, BOS, EOS] + SPECIAL + KEYS + VALUES


@dataclass
class Vocab:
    """Token ↔ index mapping."""

    tokens: list[str]

    def __post_init__(self):
        self.tok2idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx2tok = {i: t for i, t in enumerate(self.tokens)}
        self.pad_idx = self.tok2idx[PAD]
        self.bos_idx = self.tok2idx[BOS]
        self.eos_idx = self.tok2idx[EOS]

    def __len__(self):
        return len(self.tokens)

    def encode(self, text: str | list[str]) -> list[int]:
        """Convert token list (or string of single-char tokens) to indices."""
        if isinstance(text, str):
            tokens = list(text)
        else:
            tokens = text
        return [self.tok2idx[t] for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """Convert indices to token list."""
        return [self.idx2tok[i] for i in indices]

    def decode_str(self, indices: list[int]) -> str:
        """Convert indices to concatenated string."""
        return "".join(self.decode(indices))


VOCAB = Vocab(ALL_TOKENS)


# ── Sequence Builders ───────────────────────────────────────────────

def make_kv_pairs(n_pairs: int, rng: random.Random | None = None) -> list[tuple[str, str]]:
    """Generate n random key-value pairs with unique keys.

    Returns:
        List of (key, value) tuples, e.g. [('A', '7'), ('B', '3'), ('C', '9')]
    """
    rng = rng or random.Random()
    keys = rng.sample(KEYS, n_pairs)
    vals = [rng.choice(VALUES) for _ in range(n_pairs)]
    return list(zip(keys, vals))


def build_sequence_tokens(pairs: list[tuple[str, str]],
                          query_key: str | None = None) -> list[str]:
    """Build token sequence from key-value pairs + optional query.

    Examples:
        pairs=[('A','7'),('B','3')], query_key=None
            → ['<bos>', 'A', '→', '7', ',', 'B', '→', '3', '<eos>']
        pairs=[('A','7'),('B','3')], query_key='A'
            → ['<bos>', 'A', '→', '7', ',', 'B', '→', '3', ',', 'A', '→', '?', '<eos>']
    """
    tokens = [BOS]
    for i, (k, v) in enumerate(pairs):
        if i > 0:
            tokens.append(",")
        tokens.extend([k, "→", v])
    if query_key is not None:
        tokens.append(",")
        tokens.extend([query_key, "→", "?"])
    tokens.append(EOS)
    return tokens


def build_cumulative_tokens(pairs: list[tuple[str, str]],
                            up_to: int) -> list[str]:
    """Build cumulative context from pairs[0:up_to].

    Used in trace evaluation: pass 1 has 1 pair, pass 2 has 2, etc.
    This ensures each forward pass has the same format as pretraining
    (a well-formed sequence of key-value pairs).

    Example with pairs=[('A','7'),('B','3'),('C','9')], up_to=2:
        → ['<bos>', 'A', '→', '7', ',', 'B', '→', '3', '<eos>']
    """
    return build_sequence_tokens(pairs[:up_to])


# ── Pretraining Dataset ────────────────────────────────────────────

class KeyValueDataset(Dataset):
    """Dataset for pretraining: key-value retrieval with single queries.

    For each set of n_pairs key-value associations, generates n_pairs
    separate training sequences — one query per sequence:

        <bos> K1→V1, K2→V2, ..., Kn→Vn, Ki→Vi <eos>

    where Ki is queried and Vi is the answer. Loss is only on the
    answer digit position.

    This prevents the elimination shortcut (where the model deduces
    the answer from previously seen retrieval answers) while providing
    strong retrieval gradient signal (n_pairs queries per set of pairs).

    The eval format matches exactly: context pairs + single query.
    """

    def __init__(self, n_sequences: int, n_pairs: int = 5,
                 max_seq_len: int = 64, seed: int = 42):
        self.n_sequences = n_sequences
        self.n_pairs = n_pairs  # max pairs
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)
        self.vocab = VOCAB

        digit_indices = set(self.vocab.tok2idx[d] for d in "0123456789")

        # Pre-generate: variable number of pairs per sequence (1 to n_pairs)
        # This prevents the model from learning position-based retrieval
        self.samples = []

        while len(self.samples) < n_sequences:
            # Random number of pairs between 1 and n_pairs
            cur_n = self.rng.randint(1, n_pairs)
            pairs = make_kv_pairs(cur_n, self.rng)

            # One query per pair
            for query_idx in range(cur_n):
                if len(self.samples) >= n_sequences:
                    break

                query_key = pairs[query_idx][0]
                answer_val = pairs[query_idx][1]

                # Sequence: <bos> K1→V1,...,Kn→Vn, Ki→Vi <eos>
                tokens = [BOS]
                for i, (k, v) in enumerate(pairs):
                    if i > 0:
                        tokens.append(",")
                    tokens.extend([k, "→", v])
                tokens.append(",")
                tokens.extend([query_key, "→", answer_val])
                tokens.append(EOS)

                full_indices = self.vocab.encode(tokens)

                # Autoregressive: input = seq[:-1], target = seq[1:]
                input_ids = full_indices[:-1]
                target_ids = full_indices[1:]

                # Loss mask: all positions get weight 1.0 (learn structure),
                # answer digit gets weight 10.0 (focus on retrieval)
                loss_mask = [1.0] * len(target_ids)
                answer_pos = len(target_ids) - 2  # digit before <eos>
                if target_ids[answer_pos] in digit_indices:
                    loss_mask[answer_pos] = 10.0

                self.samples.append({
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                    'loss_mask': loss_mask,
                })

        self.samples = self.samples[:n_sequences]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['input_ids'], s['target_ids'], s['loss_mask']


def collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences to same length within batch."""
    input_ids, target_ids, loss_masks = zip(*batch)
    max_len = max(len(x) for x in input_ids)

    pad_idx = VOCAB.pad_idx
    padded_inputs = []
    padded_targets = []
    padded_masks = []

    for inp, tgt, mask in zip(input_ids, target_ids, loss_masks):
        pad_len = max_len - len(inp)
        padded_inputs.append(inp + [pad_idx] * pad_len)
        padded_targets.append(tgt + [pad_idx] * pad_len)
        padded_masks.append(mask + [0] * pad_len)

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_targets, dtype=torch.long),
        torch.tensor(padded_masks, dtype=torch.float),
    )


def make_pretrain_loader(n_sequences: int = 5000, n_pairs: int = 5,
                         batch_size: int = 32, max_seq_len: int = 64,
                         seed: int = 42) -> DataLoader:
    """Create DataLoader for pretraining."""
    ds = KeyValueDataset(n_sequences, n_pairs, max_seq_len, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_fn)


# ── Evaluation Episodes ────────────────────────────────────────────

@dataclass
class EvalEpisode:
    """One evaluation episode for cross-sequence trace testing.

    Attributes:
        pairs: list of (key, value) tuples — the associations to learn
        train_sequences: list of token-index sequences for trace accumulation
            Each is a cumulative context: pass i contains pairs[0:i+1]
        test_queries: list of (input_indices, answer_index) for testing
            Input: full context + query (e.g., "A→7,B→3,C→9,A→?")
            Answer: the correct digit index
    """
    pairs: list[tuple[str, str]]
    train_sequences: list[list[int]]
    test_queries: list[tuple[list[int], int]]


def make_eval_episodes(n_episodes: int = 100, n_pairs: int = 5,
                       seed: int = 123) -> list[EvalEpisode]:
    """Generate evaluation episodes for cross-sequence trace testing.

    Each episode:
    1. Generate n_pairs random key-value associations
    2. Training sequences (cumulative context, for trace accumulation):
       - Pass 1: "A→7"
       - Pass 2: "A→7,B→3"
       - Pass 3: "A→7,B→3,C→9"
       - ... up to all n_pairs
    3. Test queries (for accuracy measurement):
       - For each pair, query: "A→7,B→3,...,Ki→?" → answer Vi
       - Full context provided so Q/K representations are meaningful
    """
    rng = random.Random(seed)
    episodes = []

    for _ in range(n_episodes):
        pairs = make_kv_pairs(n_pairs, rng)

        # Training sequences: cumulative context
        train_seqs = []
        for i in range(1, n_pairs + 1):
            tokens = build_cumulative_tokens(pairs, up_to=i)
            indices = VOCAB.encode(tokens)
            train_seqs.append(indices)

        # Test queries: full context + query for each pair
        test_queries = []
        for k, v in pairs:
            # Full context with query
            query_tokens = build_sequence_tokens(pairs, query_key=k)
            # Replace ? with answer for autoregressive input
            # Actually for evaluation, we feed up to "?" and check prediction
            query_indices = VOCAB.encode(query_tokens)
            answer_idx = VOCAB.tok2idx[v]
            test_queries.append((query_indices, answer_idx))

        episodes.append(EvalEpisode(
            pairs=pairs,
            train_sequences=train_seqs,
            test_queries=test_queries,
        ))

    return episodes


def make_in_context_baseline_queries(episode: EvalEpisode) -> list[tuple[list[int], int]]:
    """Create in-context (single-pass) queries for ceiling measurement.

    Same as test_queries but all pairs + query in one sequence.
    This is what standard attention can handle without trace.
    """
    return episode.test_queries  # Already has full context!


# ── Utilities ──────────────────────────────────────────────────────

def decode_sequence(indices: list[int]) -> str:
    """Pretty-print a token index sequence."""
    return VOCAB.decode_str(indices)


def get_vocab_size() -> int:
    """Return vocabulary size."""
    return len(VOCAB)


if __name__ == "__main__":
    # Quick sanity check
    print(f"Vocab size: {len(VOCAB)}")
    print(f"Tokens: {VOCAB.tokens}")
    print()

    # Test KV pairs
    rng = random.Random(42)
    pairs = make_kv_pairs(5, rng)
    print(f"Pairs: {pairs}")

    # Test sequence building
    tokens = build_sequence_tokens(pairs, query_key=pairs[0][0])
    print(f"Sequence: {''.join(tokens)}")
    indices = VOCAB.encode(tokens)
    print(f"Indices: {indices}")
    decoded = VOCAB.decode_str(indices)
    print(f"Decoded: {decoded}")
    print()

    # Test cumulative
    for i in range(1, len(pairs) + 1):
        cum = build_cumulative_tokens(pairs, up_to=i)
        print(f"Cumulative {i}: {''.join(cum)}")
    print()

    # Test dataset
    ds = KeyValueDataset(10, n_pairs=3, seed=42)
    inp, tgt, mask = ds[0]
    print(f"Input:  {VOCAB.decode_str(inp)}")
    print(f"Target: {VOCAB.decode_str(tgt)}")
    print(f"Mask:   {mask}")
    print(f"Answer: {ds.samples[0]['answer']}")
    print()

    # Test eval episodes
    episodes = make_eval_episodes(3, n_pairs=3, seed=42)
    ep = episodes[0]
    print(f"Episode pairs: {ep.pairs}")
    for i, seq in enumerate(ep.train_sequences):
        print(f"  Train {i}: {decode_sequence(seq)}")
    for i, (q, a) in enumerate(ep.test_queries):
        print(f"  Test {i}: {decode_sequence(q)} → answer={VOCAB.idx2tok[a]}")
