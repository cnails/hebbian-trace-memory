"""Evaluation: trace-based cross-session fact memory testing (NLP).

Protocol (ACh-modulated write/read separation):
1. Reset traces
2. Write phase (high ACh): forward fact sequences with trace UPDATE only,
   retrieval suppressed to prevent interference from partial traces.
   - Pass 1: "<bos> My name is Andrey . <eos>"
   - Pass 2: "<bos> My name is Andrey . I live in Moscow . <eos>"
3. Read phase (low ACh): forward question with trace USE only, no update.
   - "<bos> What is my name ?" → predict "Andrey"

Baseline: same model, trace OFF, full context (facts + question) in one pass.
Cross-context baseline: no trace, question only — expected ~random.
"""

import torch
from dataclasses import dataclass

from .model import MiniGPT
from .nlp_tasks import (
    NLP_VOCAB, NLPEvalEpisode, make_nlp_eval_episodes,
    NLPUpdateEvalEpisode, make_nlp_update_eval_episodes,
    compute_concept_injection,
)


@dataclass
class NLPEvalResults:
    """Results from evaluating one condition."""
    accuracy: float
    n_correct: int
    n_total: int
    per_episode_acc: list[float]
    # Breakdown by template tier (if available)
    tier1_accuracy: float | None = None
    tier2_accuracy: float | None = None


def _get_device(model: MiniGPT) -> torch.device:
    """Get device from model parameters."""
    return next(model.parameters()).device


def _predict_answer(model: MiniGPT, query_indices: list[int],
                    entity_indices: list[int] | None = None,
                    concept_injection: dict[int, int] | None = None,
                    ) -> int:
    """Run model on query and return predicted entity token index.

    Query format: [<bos>, question_words..., ?]
    We feed the full query and predict at the last position ("?").
    The prediction is restricted to entity tokens only.
    """
    device = _get_device(model)
    input_tensor = torch.tensor([query_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor,
                       concept_injection=concept_injection)  # (1, seq_len, vocab_size)

    # Prediction at last position
    pred_logits = logits[0, -1, :]  # (vocab_size,)

    if entity_indices is not None:
        # Restrict to entity tokens
        entity_logits = pred_logits[entity_indices]
        best_pos = entity_logits.argmax().item()
        predicted_idx = entity_indices[best_pos]
    else:
        predicted_idx = pred_logits.argmax().item()

    return predicted_idx


def evaluate_baseline(model: MiniGPT, episodes: list[NLPEvalEpisode],
                      verbose: bool = False) -> NLPEvalResults:
    """In-context baseline: full facts + question in one pass, no trace.

    The model sees all facts AND the question in a single sequence.
    This tests standard in-context retrieval (no cross-session memory needed).
    """
    device = _get_device(model)
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for query_indices, answer_idx, ft_name, tier in episode.test_queries:
            # Build in-context query: <bos> facts... question
            ic_words = ["<bos>"]
            for _, _, words in episode.facts:
                ic_words.extend(words)
            # Add question words (skip <bos> from query)
            ic_words.extend(vocab.decode(query_indices[1:]))  # skip <bos>
            ic_indices = vocab.encode(ic_words)

            pred_idx = _predict_answer(model, ic_indices, entity_indices)
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
    return NLPEvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_hebbian(model: MiniGPT, episodes: list[NLPEvalEpisode],
                     verbose: bool = False,
                     use_injection: bool = False) -> NLPEvalResults:
    """In-context + trace: accumulate traces during training, use during test.

    Write phase: cumulative fact sequences, trace update ON, retrieval OFF
    Read phase: full context (facts + question) with trace ON but update OFF
    """
    device = _get_device(model)
    model.eval()
    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase (ACh high): encode facts, suppress retrieval
        model.set_trace_mode(use=False, update=True)
        for seq_idx, train_seq in enumerate(episode.train_sequences):
            # Compute concept injection map for this cumulative sequence
            inj = None
            if use_injection and hasattr(episode, 'fact_templates_used'):
                n_facts = seq_idx + 1  # cumulative: first n facts
                facts_words = [f[2] for f in episode.facts[:n_facts]]
                inj = compute_concept_injection(
                    facts_words, episode.fact_templates_used[:n_facts], vocab)
            input_tensor = torch.tensor([train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor, concept_injection=inj)

        # Read phase (ACh low): retrieve from trace, no update
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for query_indices, answer_idx, ft_name, tier in episode.test_queries:
            # In-context query with trace
            ic_words = ["<bos>"]
            for _, _, words in episode.facts:
                ic_words.extend(words)
            ic_words.extend(vocab.decode(query_indices[1:]))
            ic_indices = vocab.encode(ic_words)

            pred_idx = _predict_answer(model, ic_indices, entity_indices)
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
    return NLPEvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


def evaluate_cross_context(model: MiniGPT, episodes: list[NLPEvalEpisode],
                            verbose: bool = False,
                            use_injection: bool = False) -> NLPEvalResults:
    """Cross-context: trace only, question-only query (no facts in context).

    THE REAL TEST. The test query contains ONLY the question
    (e.g., "<bos> What is my name ?"). Without trace, the model has
    zero information to answer. Any accuracy above random comes from
    trace-stored associations.

    Write phase (ACh high): trace accumulates, retrieval suppressed
    Read phase (ACh low): question-only query, trace ON, no update
    """
    device = _get_device(model)
    model.eval()
    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices
    total_correct = 0
    total_queries = 0
    per_episode = []
    tier1_correct = 0
    tier1_total = 0
    tier2_correct = 0
    tier2_total = 0

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase (ACh high): accumulate traces, suppress retrieval
        model.set_trace_mode(use=False, update=True)
        for seq_idx, train_seq in enumerate(episode.train_sequences):
            # Compute concept injection map for this cumulative sequence
            inj = None
            if use_injection and hasattr(episode, 'fact_templates_used'):
                n_facts = seq_idx + 1
                facts_words = [f[2] for f in episode.facts[:n_facts]]
                inj = compute_concept_injection(
                    facts_words, episode.fact_templates_used[:n_facts], vocab)
            input_tensor = torch.tensor([train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor, concept_injection=inj)

        # Read phase (ACh low): question-only query, trace ON, no update
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for query_indices, answer_idx, ft_name, tier in episode.test_queries:
            # Minimal query: just the question, no facts
            pred_idx = _predict_answer(model, query_indices, entity_indices)
            correct = pred_idx == answer_idx
            if correct:
                ep_correct += 1
            total_queries += 1

            # Track by tier
            if tier == 1:
                tier1_total += 1
                if correct:
                    tier1_correct += 1
            elif tier == 2:
                tier2_total += 1
                if correct:
                    tier2_correct += 1

        total_correct += ep_correct
        ep_acc = ep_correct / len(episode.test_queries)
        per_episode.append(ep_acc)

        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"  Episode {ep_idx+1}: {ep_acc:.0%} "
                  f"(running: {total_correct/total_queries:.1%})")

    accuracy = total_correct / max(total_queries, 1)
    tier1_acc = tier1_correct / max(tier1_total, 1) if tier1_total > 0 else None
    tier2_acc = tier2_correct / max(tier2_total, 1) if tier2_total > 0 else None

    return NLPEvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
        tier1_accuracy=tier1_acc,
        tier2_accuracy=tier2_acc,
    )


def evaluate_cross_context_baseline(model: MiniGPT,
                                     episodes: list[NLPEvalEpisode],
                                     verbose: bool = False) -> NLPEvalResults:
    """Cross-context baseline: no trace, question-only query.

    Without trace and without context, the model has nothing to go on.
    Expected accuracy: ~1/N_entities per category (~5% for 20 values).
    """
    device = _get_device(model)
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices
    total_correct = 0
    total_queries = 0
    per_episode = []

    for ep_idx, episode in enumerate(episodes):
        ep_correct = 0
        for query_indices, answer_idx, ft_name, tier in episode.test_queries:
            pred_idx = _predict_answer(model, query_indices, entity_indices)
            if pred_idx == answer_idx:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        ep_acc = ep_correct / len(episode.test_queries)
        per_episode.append(ep_acc)

    accuracy = total_correct / max(total_queries, 1)
    return NLPEvalResults(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_queries,
        per_episode_acc=per_episode,
    )


# ── Fact Update Evaluation ────────────────────────────────────────

@dataclass
class NLPUpdateEvalResults:
    """Results from fact update evaluation."""
    update_accuracy: float      # updated facts: predicted NEW value
    old_value_rate: float       # updated facts: predicted OLD value
    stable_accuracy: float      # non-updated facts: predicted original value
    overall_accuracy: float
    n_updated: int
    n_stable: int


def evaluate_fact_update(model: MiniGPT, episodes: list[NLPUpdateEvalEpisode],
                         verbose: bool = False,
                         erase_lr: float | None = None,
                         ) -> NLPUpdateEvalResults:
    """Evaluate fact update via trace.

    Protocol (ACh-modulated):
    1. Reset traces
    2. Write phase 1 (ACh high): encode original facts (cumulative), erase OFF
    3. Write phase 2 (ACh high): encode updated facts (cumulative),
       erase ON if erase_lr is set (reconsolidation erasure)
    4. Read phase (ACh low): query ALL facts
       - Updated facts: correct = new value
       - Stable facts: correct = original value

    Args:
        erase_lr: if set, enable reconsolidation erasure during phase 2.
            Erases old Q→V association before writing new one.
    """
    device = _get_device(model)
    model.eval()
    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices

    update_correct = 0
    update_old = 0
    update_total = 0
    stable_correct = 0
    stable_total = 0

    for ep_idx, episode in enumerate(episodes):
        model.reset_traces()

        # Write phase 1: original facts (erase OFF)
        model.set_erase_mode(False)
        model.set_trace_mode(use=False, update=True)
        for train_seq in episode.phase1_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)

        # Write phase 2: updated facts (erase ON if erase_lr set)
        if erase_lr is not None:
            model.set_erase_mode(True, erase_lr)
        for train_seq in episode.phase2_sequences:
            input_tensor = torch.tensor(
                [train_seq], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)

        # Read phase: query all facts (erase OFF)
        model.set_erase_mode(False)
        model.set_trace_mode(use=True, update=False)
        for query_indices, correct_idx, old_idx, ft_name, was_updated \
                in episode.test_queries:
            pred_idx = _predict_answer(model, query_indices, entity_indices)

            if was_updated:
                update_total += 1
                if pred_idx == correct_idx:
                    update_correct += 1
                elif pred_idx == old_idx:
                    update_old += 1
            else:
                stable_total += 1
                if pred_idx == correct_idx:
                    stable_correct += 1

        if verbose and (ep_idx + 1) % 20 == 0:
            ua = update_correct / max(update_total, 1)
            or_ = update_old / max(update_total, 1)
            sa = stable_correct / max(stable_total, 1)
            print(f"  Episode {ep_idx+1}: update={ua:.1%} "
                  f"old_val={or_:.1%} stable={sa:.1%}")

    total_correct = update_correct + stable_correct
    total = update_total + stable_total

    return NLPUpdateEvalResults(
        update_accuracy=update_correct / max(update_total, 1),
        old_value_rate=update_old / max(update_total, 1),
        stable_accuracy=stable_correct / max(stable_total, 1),
        overall_accuracy=total_correct / max(total, 1),
        n_updated=update_total,
        n_stable=stable_total,
    )


def evaluate_fact_update_baseline(model: MiniGPT,
                                  episodes: list[NLPUpdateEvalEpisode],
                                  verbose: bool = False) -> NLPUpdateEvalResults:
    """Baseline: no trace, question-only query. Expected ~random."""
    device = _get_device(model)
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    vocab = NLP_VOCAB
    entity_indices = vocab.entity_indices

    update_correct = 0
    update_total = 0
    stable_correct = 0
    stable_total = 0

    for episode in episodes:
        for query_indices, correct_idx, old_idx, ft_name, was_updated \
                in episode.test_queries:
            pred_idx = _predict_answer(model, query_indices, entity_indices)

            if was_updated:
                update_total += 1
                if pred_idx == correct_idx:
                    update_correct += 1
            else:
                stable_total += 1
                if pred_idx == correct_idx:
                    stable_correct += 1

    total_correct = update_correct + stable_correct
    total = update_total + stable_total

    return NLPUpdateEvalResults(
        update_accuracy=update_correct / max(update_total, 1),
        old_value_rate=0.0,
        stable_accuracy=stable_correct / max(stable_total, 1),
        overall_accuracy=total_correct / max(total, 1),
        n_updated=update_total,
        n_stable=stable_total,
    )
