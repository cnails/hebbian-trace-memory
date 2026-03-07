"""RAG baselines for comparison with Hebbian trace memory.

Implements three retrieval-augmented generation variants using the same
evaluation protocol as the trace module:
  - OracleRAG: perfect retrieval (upper bound)
  - EmbeddingRAG: cosine similarity over GPT-2 wte mean-pool
  - TFIDFRAG: keyword-based TF-IDF retrieval

All baselines use frozen GPT-2 as the reader model with retrieved facts
prepended to the query in-context.
"""

import math
import random
from collections import Counter
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.tasks import (
    EvalEpisode,
    EvalResults,
    FactType,
    get_all_entity_ids,
    _predict_answer,
    tokenize_fact,
    tokenize_question,
)


# -- Stored Fact --

@dataclass
class StoredFact:
    """A fact stored in the RAG document store."""
    type_name: str
    entity_name: str
    entity_bpe_id: int
    fact_bpe_ids: list[int]
    fact_text: str


# -- RAG Store Base --

class RAGStore:
    """Base class for RAG fact stores."""

    def __init__(self, tokenizer: GPT2Tokenizer):
        self.tokenizer = tokenizer
        self.facts: list[StoredFact] = []
        self._space_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    def reset(self):
        self.facts = []

    def store(self, type_name: str, entity_name: str,
              entity_bpe_id: int, fact_bpe_ids: list[int]):
        fact_text = self.tokenizer.decode(fact_bpe_ids)
        self.facts.append(StoredFact(
            type_name=type_name,
            entity_name=entity_name,
            entity_bpe_id=entity_bpe_id,
            fact_bpe_ids=fact_bpe_ids,
            fact_text=fact_text,
        ))

    def store_update(self, type_name: str, entity_name: str,
                     entity_bpe_id: int, fact_bpe_ids: list[int]):
        """Replace old fact of same type with new value."""
        self.facts = [f for f in self.facts if f.type_name != type_name]
        self.store(type_name, entity_name, entity_bpe_id, fact_bpe_ids)

    def retrieve(self, query_bpe_ids: list[int], top_k: int = 1,
                 **kwargs) -> list[StoredFact]:
        raise NotImplementedError

    def build_context(self, query_bpe_ids: list[int],
                      retrieved: list[StoredFact]) -> list[int]:
        """Build input: retrieved facts + query."""
        context: list[int] = []
        for fact in retrieved:
            if context:
                context.append(self._space_id)
            context.extend(fact.fact_bpe_ids)
        context.append(self._space_id)
        context.extend(query_bpe_ids)
        return context


# -- Oracle RAG --

class OracleRAGStore(RAGStore):
    """Perfect retrieval — always finds the correct fact by type_name.

    Upper bound for RAG performance. With top_k=1, provides only the
    relevant fact in context, equivalent to in-context baseline at n=1.
    """

    def retrieve(self, query_bpe_ids: list[int], top_k: int = 1,
                 target_type: str | None = None,
                 **kwargs) -> list[StoredFact]:
        if target_type is not None:
            matches = [f for f in self.facts if f.type_name == target_type]
            return matches[:top_k]
        return self.facts[:top_k]


# -- TF-IDF RAG --

class TFIDFRAGStore(RAGStore):
    """TF-IDF retrieval using only Python stdlib.

    Tokenizes text by whitespace, computes TF-IDF scores between
    query and stored facts, retrieves by score.
    """

    def _tokenize_text(self, text: str) -> list[str]:
        words = text.lower().split()
        return [w.strip('.,?!;:') for w in words if w.strip('.,?!;:')]

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self.facts)
        if n_docs == 0:
            return {}
        doc_freq: Counter = Counter()
        for fact in self.facts:
            terms = set(self._tokenize_text(fact.fact_text))
            for t in terms:
                doc_freq[t] += 1
        return {t: math.log((n_docs + 1) / (df + 1)) + 1
                for t, df in doc_freq.items()}

    def retrieve(self, query_bpe_ids: list[int], top_k: int = 1,
                 **kwargs) -> list[StoredFact]:
        if not self.facts:
            return []

        query_text = self.tokenizer.decode(query_bpe_ids)
        query_terms = self._tokenize_text(query_text)
        idf = self._compute_idf()

        scores = []
        for fact in self.facts:
            fact_terms = self._tokenize_text(fact.fact_text)
            fact_tf = Counter(fact_terms)
            n_terms = max(len(fact_terms), 1)
            score = 0.0
            for qt in query_terms:
                if qt in fact_tf:
                    tf = fact_tf[qt] / n_terms
                    score += tf * idf.get(qt, 0)
            scores.append(score)

        ranked = sorted(zip(scores, range(len(self.facts))),
                        key=lambda x: -x[0])
        return [self.facts[i] for _, i in ranked[:top_k]]


# -- Embedding RAG --

class EmbeddingRAGStore(RAGStore):
    """Embedding-based retrieval using GPT-2's own token embeddings.

    Mean-pools wte embeddings over tokens. Retrieves by cosine similarity.
    No external dependencies.
    """

    def __init__(self, tokenizer: GPT2Tokenizer,
                 wte_weight: torch.Tensor):
        super().__init__(tokenizer)
        self.wte_weight = wte_weight  # (vocab_size, d_model)
        self._fact_embeddings: list[torch.Tensor] = []

    def reset(self):
        super().reset()
        self._fact_embeddings = []

    def _embed(self, token_ids: list[int]) -> torch.Tensor:
        ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        with torch.no_grad():
            embeds = self.wte_weight[ids_tensor]
        return embeds.mean(dim=0)

    def store(self, type_name: str, entity_name: str,
              entity_bpe_id: int, fact_bpe_ids: list[int]):
        super().store(type_name, entity_name, entity_bpe_id, fact_bpe_ids)
        self._fact_embeddings.append(self._embed(fact_bpe_ids))

    def store_update(self, type_name: str, entity_name: str,
                     entity_bpe_id: int, fact_bpe_ids: list[int]):
        old_indices = [i for i, f in enumerate(self.facts)
                       if f.type_name == type_name]
        for i in reversed(old_indices):
            self.facts.pop(i)
            self._fact_embeddings.pop(i)
        self.store(type_name, entity_name, entity_bpe_id, fact_bpe_ids)

    def retrieve(self, query_bpe_ids: list[int], top_k: int = 1,
                 **kwargs) -> list[StoredFact]:
        if not self.facts:
            return []

        q_emb = self._embed(query_bpe_ids)
        q_norm = q_emb / q_emb.norm().clamp(min=1e-8)

        similarities = []
        for f_emb in self._fact_embeddings:
            f_norm = f_emb / f_emb.norm().clamp(min=1e-8)
            sim = torch.dot(q_norm, f_norm).item()
            similarities.append(sim)

        ranked = sorted(zip(similarities, range(len(self.facts))),
                        key=lambda x: -x[0])
        return [self.facts[i] for _, i in ranked[:top_k]]


# -- Evaluation Functions --

def evaluate_rag(
    model,
    episodes: list[EvalEpisode],
    fact_types: list[FactType],
    rag_store: RAGStore,
    top_k: int = 1,
    verbose: bool = False,
) -> EvalResults:
    """Evaluate RAG baseline using same protocol as trace evaluation.

    Write phase: store facts in RAG store.
    Read phase: retrieve top_k facts, prepend to query, predict via GPT-2.
    """
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    entity_ids = get_all_entity_ids(fact_types)
    total_correct = 0
    total_queries = 0
    per_episode: list[float] = []

    for ep_idx, episode in enumerate(episodes):
        rag_store.reset()

        # Write phase
        for type_name, entity_name, entity_bpe_id, fact_bpe_ids in episode.facts:
            rag_store.store(type_name, entity_name,
                            entity_bpe_id, fact_bpe_ids)

        # Read phase
        ep_correct = 0
        for query_ids, answer_id, type_name in episode.test_queries:
            if isinstance(rag_store, OracleRAGStore):
                retrieved = rag_store.retrieve(
                    query_ids, top_k=top_k, target_type=type_name)
            else:
                retrieved = rag_store.retrieve(query_ids, top_k=top_k)

            context_ids = rag_store.build_context(query_ids, retrieved)
            pred_id = _predict_answer(model, context_ids, entity_ids)

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


def evaluate_retrieval_accuracy(
    episodes: list[EvalEpisode],
    rag_store: RAGStore,
    top_k: int = 1,
) -> float:
    """Measure retrieval accuracy independently of the reader model."""
    total = 0
    correct = 0
    for episode in episodes:
        rag_store.reset()
        for type_name, entity_name, entity_bpe_id, fact_bpe_ids in episode.facts:
            rag_store.store(type_name, entity_name,
                            entity_bpe_id, fact_bpe_ids)
        for query_ids, answer_id, type_name in episode.test_queries:
            retrieved = rag_store.retrieve(query_ids, top_k=top_k)
            if any(f.type_name == type_name for f in retrieved):
                correct += 1
            total += 1
    return correct / max(total, 1)


# -- Multi-Session RAG Evaluation --

def run_rag_multisession(
    model,
    rag_store: RAGStore,
    fact_types: list[FactType],
    tokenizer: GPT2Tokenizer,
    entity_ids: list[int],
    n_sessions: int = 5,
    facts_per_session: int = 5,
    n_episodes: int = 50,
    seed: int = 42,
) -> tuple[dict[int, dict], dict[int, list[float]]]:
    """Run multi-session evaluation with RAG store.

    Mirrors demo.py session logic: introduction phase + update phase.
    Returns (session_results, retention_data) in the same format as demo.py.
    """
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    n_types = min(len(fact_types), 24)

    from demo import make_session_schedule
    new_schedule, update_schedule = make_session_schedule(
        n_types, facts_per_session, n_sessions)

    session_results = {s: {"correct": 0, "total": 0,
                           "new_c": 0, "new_t": 0,
                           "old_c": 0, "old_t": 0,
                           "upd_c": 0, "upd_t": 0}
                       for s in range(n_sessions)}
    retention_data: dict[int, list[float]] = {}

    device = next(model.parameters()).device

    for ep in range(n_episodes):
        rng = random.Random(seed + ep)
        rag_store.reset()

        known_facts: dict[str, tuple[str, int, list[int]]] = {}
        entity_history: dict[str, set[str]] = {}
        last_write_session: dict[str, int] = {}

        # Track which facts are new/updated this session
        for s in range(n_sessions):
            session_new_types: set[str] = set()
            session_upd_types: set[str] = set()

            # New facts
            for tidx in new_schedule[s]:
                ft = fact_types[tidx]
                ent_name, ent_id = rng.choice(ft.entities)
                template = rng.choice(ft.fact_templates)
                fact_ids = tokenize_fact(tokenizer, template, ent_name)
                rag_store.store(ft.name, ent_name, ent_id, fact_ids)
                q_ids = tokenize_question(tokenizer,
                                          rng.choice(ft.question_templates))
                known_facts[ft.name] = (ent_name, ent_id, q_ids)
                entity_history.setdefault(ft.name, set()).add(ent_name)
                last_write_session[ft.name] = s
                session_new_types.add(ft.name)

            # Updates
            for tidx in update_schedule[s]:
                ft = fact_types[tidx]
                used = entity_history.get(ft.name, set())
                available = [e for e in ft.entities if e[0] not in used]
                if not available:
                    available = ft.entities
                ent_name, ent_id = rng.choice(available)
                template = rng.choice(ft.fact_templates)
                fact_ids = tokenize_fact(tokenizer, template, ent_name)
                rag_store.store_update(ft.name, ent_name, ent_id, fact_ids)
                q_ids = tokenize_question(tokenizer,
                                          rng.choice(ft.question_templates))
                known_facts[ft.name] = (ent_name, ent_id, q_ids)
                entity_history.setdefault(ft.name, set()).add(ent_name)
                last_write_session[ft.name] = s
                session_upd_types.add(ft.name)

            # Read phase: query all known facts
            sr = session_results[s]
            for type_name, (ent_name, ent_id, q_ids) in known_facts.items():
                if isinstance(rag_store, OracleRAGStore):
                    retrieved = rag_store.retrieve(
                        q_ids, top_k=1, target_type=type_name)
                else:
                    retrieved = rag_store.retrieve(q_ids, top_k=1)

                context_ids = rag_store.build_context(q_ids, retrieved)
                input_tensor = torch.tensor(
                    [context_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(input_tensor)
                pred_logits = logits[0, -1, :]
                el = pred_logits[entity_ids]
                pred_id = entity_ids[el.argmax().item()]
                correct = (pred_id == ent_id)

                sr["total"] += 1
                if correct:
                    sr["correct"] += 1

                if type_name in session_new_types:
                    sr["new_t"] += 1
                    if correct:
                        sr["new_c"] += 1
                elif type_name in session_upd_types:
                    sr["upd_t"] += 1
                    if correct:
                        sr["upd_c"] += 1
                else:
                    sr["old_t"] += 1
                    if correct:
                        sr["old_c"] += 1

                age = s - last_write_session[type_name]
                retention_data.setdefault(age, []).append(float(correct))

    return session_results, retention_data
