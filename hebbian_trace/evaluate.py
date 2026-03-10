"""Evaluation: trace-based cross-sequence memory testing.

Protocol:
1. Reset traces
2. Training phase: forward each cumulative context with trace update ON
   - Pass 1: "A→7"           → trace update
   - Pass 2: "A→7,B→3"       → trace update
   - Pass 3: "A→7,B→3,C→9"   → trace update
3. Test phase: forward each query with trace ON but update OFF
   - "A→7,B→3,C→9,A→?" → predict answer → should be "7"

Baseline: same model, trace OFF in all phases.
Ceiling: same model, in-context (all info in one pass), trace OFF.
"""

import torch
from dataclasses import dataclass

from .model import MiniGPT
from .tasks import VOCAB, EvalEpisode, make_eval_episodes


@dataclass
class EvalResults:
    """Results from evaluating one condition."""
    accuracy: float
    n_correct: int
    n_total: int
    per_episode_acc: list[float]


def _predict_answer(model: MiniGPT, query_indices: list[int]) -> int:
    """Run model on query and return predicted token index at answer position.

    The query ends with: ..., Ki, →, ?, <eos>
    During training, "?" was replaced with the answer digit.
    So the model has never seen "?". We feed up to and including "→"
    and let the model predict the next token (the digit).

    Input fed:  <bos> A→7, B→3, A→
    Prediction: at last position (the "→"), predict the digit.
    """
    # Find "?" or "→" before it
    q_mark_idx = VOCAB.tok2idx["?"]
    arrow_idx = VOCAB.tok2idx["→"]

    # Find position of ? in the sequence
    try:
        q_pos = query_indices.index(q_mark_idx)
    except ValueError:
        # No ? — might be a minimal query like <bos> K →
        # Feed as-is, predict at last token
        input_indices = query_indices
        q_pos = len(query_indices)

    # Feed up to the "→" before "?" (i.e., up to position q_pos - 1)
    # query: [..., K, →, ?, <eos>]
    # We want: [..., K, →]
    input_indices = query_indices[:q_pos]  # everything before "?"
    input_tensor = torch.tensor([input_indices], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)  # (1, seq_len, vocab_size)

    # Prediction at last position (the "→" before the answer)
    pred_logits = logits[0, -1, :]  # (vocab_size,)

    # Only consider digit tokens as valid answers
    digit_indices = [VOCAB.tok2idx[d] for d in "0123456789"]
    digit_logits = pred_logits[digit_indices]
    best_digit_pos = digit_logits.argmax().item()
    predicted_idx = digit_indices[best_digit_pos]

    return predicted_idx


def evaluate_hebbian(model: MiniGPT, episodes: list[EvalEpisode],
                     verbose: bool = False) -> EvalResults:
    """Evaluate with Hebbian trace: accumulate during training, use during test.

    Training phase: trace accumulates (use=True, update=True)
    Test phase: trace biases attention (use=True, update=False)
    """
    model.eval()
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        # Reset traces for this episode
        model.reset_traces()

        # --- Training phase: accumulate trace ---
        model.set_trace_mode(use=True, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor([train_seq], dtype=torch.long)
            with torch.no_grad():
                _ = model(input_tensor)
            # Trace is updated inside forward pass

        # --- Test phase: use trace, no update ---
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for query_indices, answer_idx in episode.test_queries:
            pred_idx = _predict_answer(model, query_indices)
            if pred_idx == answer_idx:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / len(episode.test_queries)
        per_episode.append(ep_acc)

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {ep_acc:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    accuracy = total_correct / max(total_queries, 1)
    return EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_baseline(model: MiniGPT, episodes: list[EvalEpisode],
                      verbose: bool = False) -> EvalResults:
    """Evaluate WITHOUT trace: standard transformer, no cross-sequence memory.

    All phases have trace OFF. The model sees the test query with full
    in-context information but has no trace from training phase.

    Since test queries contain all pairs + query (e.g., "A→7,B→3,C→9,A→?"),
    this is actually a strong baseline — standard in-context retrieval.
    """
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for query_indices, answer_idx in episode.test_queries:
            pred_idx = _predict_answer(model, query_indices)
            if pred_idx == answer_idx:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / len(episode.test_queries)
        per_episode.append(ep_acc)

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {ep_acc:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    accuracy = total_correct / max(total_queries, 1)
    return EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_cross_context(model: MiniGPT, episodes: list[EvalEpisode],
                           verbose: bool = False) -> EvalResults:
    """Evaluate cross-context: training pairs in separate passes, test in new pass.

    This is the TRUE test of trace memory. The test query does NOT contain
    the key-value pairs — only the query itself ("Ki→?"). Without trace,
    the model has zero information to answer.

    Training phase: trace accumulates from cumulative contexts (use=True, update=True)
    Test phase: query contains ONLY the query key, not the pairs (use=True, update=False)
    """
    model.eval()
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        # Reset traces for this episode
        model.reset_traces()

        # --- Training phase: accumulate trace ---
        model.set_trace_mode(use=True, update=True)
        for train_seq in episode.train_sequences:
            input_tensor = torch.tensor([train_seq], dtype=torch.long)
            with torch.no_grad():
                _ = model(input_tensor)

        # --- Test phase: query WITHOUT context ---
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for (k, v) in episode.pairs:
            # Minimal query: just "<bos> K → ?"
            query_tokens = ["<bos>", k, "→", "?"]  # ? will be stripped by _predict_answer
            query_indices = VOCAB.encode(query_tokens)
            answer_idx = VOCAB.tok2idx[v]

            pred_idx = _predict_answer(model, query_indices)
            if pred_idx == answer_idx:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / len(episode.pairs)
        per_episode.append(ep_acc)

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {ep_acc:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    accuracy = total_correct / max(total_queries, 1)
    return EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_cross_context_baseline(model: MiniGPT, episodes: list[EvalEpisode],
                                    verbose: bool = False) -> EvalResults:
    """Baseline for cross-context: no trace, minimal query.

    Without trace and without context, the model has nothing to go on.
    Expected accuracy: ~10% (random among 10 digits).
    """
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for (k, v) in episode.pairs:
            query_tokens = ["<bos>", k, "→", "?"]  # ? will be stripped by _predict_answer
            query_indices = VOCAB.encode(query_tokens)
            answer_idx = VOCAB.tok2idx[v]

            pred_idx = _predict_answer(model, query_indices)
            if pred_idx == answer_idx:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / len(episode.pairs)
        per_episode.append(ep_acc)

    accuracy = total_correct / max(total_queries, 1)
    return EvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


if __name__ == "__main__":
    from .train import pretrain

    print("=== Quick eval test ===\n")

    # Quick pretrain
    model, _ = pretrain(
        n_sequences=500, n_pairs=3, batch_size=32, epochs=5,
        verbose=True,
    )

    # Generate eval episodes
    episodes = make_eval_episodes(n_episodes=20, n_pairs=3, seed=99)

    print("\n--- In-context baseline (full context, no trace) ---")
    baseline = evaluate_baseline(model, episodes, verbose=True)
    print(f"Baseline accuracy: {baseline.accuracy:.1%}")

    print("\n--- Hebbian (trace accumulation + in-context) ---")
    hebbian = evaluate_hebbian(model, episodes, verbose=True)
    print(f"Hebbian accuracy: {hebbian.accuracy:.1%}")

    print("\n--- Cross-context (trace only, minimal query) ---")
    cross = evaluate_cross_context(model, episodes, verbose=True)
    print(f"Cross-context accuracy: {cross.accuracy:.1%}")

    print("\n--- Cross-context baseline (no trace, minimal query) ---")
    cross_bl = evaluate_cross_context_baseline(model, episodes, verbose=True)
    print(f"Cross-context baseline accuracy: {cross_bl.accuracy:.1%}")
