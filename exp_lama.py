"""LAMA benchmark evaluation for Hebbian Trace Memory.

Evaluates the trace module on the LAMA T-REx subset — a standard
knowledge probing benchmark. Tests whether the Hebbian trace can
store and retrieve real-world Wikidata knowledge.

Approach: Adapt LAMA facts to system-compatible templates.
  LAMA "Paris is the capital of France"
    → storage: "capital is Paris."
    → query:   "capital is"
  This respects the shift-1 addressing constraint while testing
  real-world knowledge.

Usage:
    python exp_lama.py --quick          # ~2 min, 20 episodes
    python exp_lama.py --n-eval 100     # full run, ~10 min
"""

import argparse
import io
import json
import os
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_linking_bpe_ids(tokenizer: GPT2Tokenizer) -> list[int]:
    """Map linking words to BPE token IDs (space-prefixed)."""
    linking_words = ["is", "in", "at", "from", ":", "am", "me"]
    ids = []
    for word in linking_words:
        bpe_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(bpe_ids) == 1:
            ids.append(bpe_ids[0])
    return ids


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


# ═══════════════════════════════════════════════════════════════════
#  LAMA DATA LOADING
# ═══════════════════════════════════════════════════════════════════

LAMA_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
DATA_DIR = Path(__file__).resolve().parent / "data" / "lama"


def download_lama(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract LAMA dataset if not present."""
    # Check both possible paths (zip may add extra data/ level)
    for candidate in [data_dir / "TREx",
                      data_dir / "data" / "TREx"]:
        if candidate.exists() and any(candidate.glob("*.jsonl")):
            return candidate

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading LAMA from {LAMA_URL} ...")

    resp = urllib.request.urlopen(LAMA_URL)
    buf = io.BytesIO(resp.read())

    with zipfile.ZipFile(buf) as zf:
        for member in zf.namelist():
            if "TREx/" in member:
                zf.extract(member, data_dir)

    # Find the extracted TREx dir
    for candidate in [data_dir / "TREx",
                      data_dir / "data" / "TREx"]:
        if candidate.exists() and any(candidate.glob("*.jsonl")):
            print(f"  Extracted to {candidate}")
            return candidate

    raise FileNotFoundError("TREx data not found after extraction")


def load_trex(data_dir: Path = DATA_DIR) -> dict[str, list[dict]]:
    """Load T-REx data grouped by predicate ID.

    Returns dict: predicate_id -> list of fact dicts.
    """
    trex_dir = download_lama(data_dir)

    data: dict[str, list[dict]] = {}
    for path in sorted(trex_dir.glob("*.jsonl")):
        pid = path.stem  # e.g. "P19"
        facts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    facts.append(json.loads(line))
        if facts:
            data[pid] = facts

    return data


# ═══════════════════════════════════════════════════════════════════
#  RELATION MAPPING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RelationMapping:
    predicate_id: str
    concept_word: str       # single BPE token for Q-address
    linking_token: str      # "is" | "in" | "at" | "from"
    description: str
    concept_bpe_id: int = -1
    valid: bool = False


# Each LAMA T-REx predicate → (concept_word, linking_token, description)
# concept_word must tokenize to 1 BPE token (with space prefix).
# We use unique concept words to avoid Q-address collisions.
RELATION_MAP: dict[str, tuple[str, str, str]] = {
    # ── Place relations → "in" ──
    "P19":   ("birthplace",    "in", "place of birth"),
    "P20":   ("deathplace",    "in", "place of death"),
    "P30":   ("continent",     "in", "continent"),
    "P131":  ("located",       "in", "located in territory"),
    "P159":  ("headquarters",  "in", "headquarters location"),
    "P276":  ("location",      "in", "location"),
    "P495":  ("origin",        "in", "country of origin"),
    "P740":  ("founded",       "in", "formation location"),
    "P937":  ("workplace",     "in", "work location"),
    "P413":  ("position",      "in", "position played"),
    "P1376": ("capitalof",     "in", "capital of"),
    # ── Identity / attribute relations → "is" ──
    "P27":   ("citizen",       "is", "country of citizenship"),
    "P36":   ("capital",       "is", "capital"),
    "P37":   ("language",      "is", "official language"),
    "P101":  ("occupation",    "is", "occupation"),
    "P103":  ("native",        "is", "native language"),
    "P106":  ("profession",    "is", "profession"),
    "P127":  ("owner",         "is", "owned by"),
    "P136":  ("genre",         "is", "genre"),
    "P138":  ("namesake",      "is", "named after"),
    "P140":  ("religion",      "is", "religion"),
    "P176":  ("manufacturer",  "is", "manufacturer"),
    "P178":  ("developer",     "is", "developer"),
    "P264":  ("label",         "is", "record label"),
    "P279":  ("subclass",      "is", "subclass of"),
    "P361":  ("part",          "is", "part of"),
    "P364":  ("tongue",        "is", "language of work"),
    "P407":  ("written",       "in", "language of work/name"),
    "P449":  ("network",       "is", "original network"),
    "P463":  ("member",        "is", "member of"),
    "P527":  ("component",     "is", "has part"),
    "P530":  ("ally",          "is", "diplomatic relation"),
    "P1001": ("jurisdiction",  "is", "applies to jurisdiction"),
    "P1303": ("instrument",    "is", "instrument played"),
}


def validate_mappings(tokenizer: GPT2Tokenizer) -> list[RelationMapping]:
    """Validate that concept words are single BPE tokens."""
    mappings = []
    for pid, (concept, link, desc) in RELATION_MAP.items():
        ids = tokenizer.encode(" " + concept, add_special_tokens=False)
        m = RelationMapping(
            predicate_id=pid, concept_word=concept,
            linking_token=link, description=desc,
        )
        if len(ids) == 1:
            m.concept_bpe_id = ids[0]
            m.valid = True
        mappings.append(m)
    return mappings


# ── Wikidata property labels for all 41 T-REx predicates ──
# Source: Wikidata property pages (public metadata).
WIKIDATA_LABELS: dict[str, str] = {
    "P19":   "place of birth",
    "P20":   "place of death",
    "P27":   "country of citizenship",
    "P17":   "country",
    "P30":   "continent",
    "P31":   "instance of",
    "P36":   "capital",
    "P37":   "official language",
    "P39":   "position held",
    "P47":   "shares border with",
    "P101":  "field of work",
    "P103":  "native language",
    "P106":  "occupation",
    "P108":  "employer",
    "P127":  "owned by",
    "P131":  "located in territory",
    "P136":  "genre",
    "P138":  "named after",
    "P140":  "religion",
    "P159":  "headquarters location",
    "P176":  "manufacturer",
    "P178":  "developer",
    "P190":  "twinned administrative body",
    "P264":  "record label",
    "P276":  "location",
    "P279":  "subclass of",
    "P361":  "part of",
    "P364":  "original language of film or TV show",
    "P407":  "language of work or name",
    "P413":  "position played on team",
    "P449":  "original broadcast network",
    "P463":  "member of",
    "P495":  "country of origin",
    "P527":  "has part",
    "P530":  "diplomatic relation",
    "P740":  "location of formation",
    "P937":  "work location",
    "P1001": "applies to jurisdiction",
    "P1303": "instrument",
    "P1376": "capital of",
    "P1412": "languages spoken written or signed",
}

_STOPWORDS = {"of", "in", "the", "to", "by", "and", "or", "a", "on", "for",
              "with", "as", "at", "from"}

_PLACE_KEYWORDS = {"place", "location", "capital", "continent", "country",
                   "border", "territory", "formation", "headquarters",
                   "work", "birth", "death", "twinned", "administrative"}


def auto_assign_concepts(
    tokenizer: GPT2Tokenizer,
    verbose: bool = True,
) -> list[RelationMapping]:
    """Automatically assign concept words from Wikidata labels.

    Algorithm: for each predicate, extract candidate words from its
    Wikidata label, try compounds then individual words, pick first
    single-token BPE match. Resolve collisions by falling back to
    next candidate.
    """
    def _candidates(label: str) -> list[str]:
        """Generate candidate concept words from a label."""
        words = [w.lower() for w in label.replace("/", " ").split()
                 if w.lower() not in _STOPWORDS]
        cands = []
        # Compounds first (pairs, left-to-right)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                cands.append(words[i] + words[j])
        # Reversed compounds
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                cands.append(words[j] + words[i])
        # Individual words
        cands.extend(words)
        # Deduplicate preserving order
        seen = set()
        result = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result

    def _is_single_token(word: str) -> int | None:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _infer_linking(label: str) -> str:
        label_lower = label.lower()
        return "in" if any(kw in label_lower for kw in _PLACE_KEYWORDS) else "is"

    # First pass: collect all candidates per predicate
    pred_candidates: dict[str, list[tuple[str, int]]] = {}
    for pid, label in WIKIDATA_LABELS.items():
        valid_cands = []
        for word in _candidates(label):
            bpe_id = _is_single_token(word)
            if bpe_id is not None:
                valid_cands.append((word, bpe_id))
        pred_candidates[pid] = valid_cands

    # Second pass: greedy assignment with collision resolution
    assigned: dict[str, str] = {}   # pid → concept_word
    used_words: dict[str, str] = {} # concept_word → pid (first claimer)
    collisions: list[tuple[str, str, str, str]] = []  # (pid, tried, collision_with, got)

    for pid in sorted(WIKIDATA_LABELS.keys()):
        for word, _ in pred_candidates.get(pid, []):
            if word not in used_words:
                assigned[pid] = word
                used_words[word] = pid
                break
            else:
                # Collision — try next candidate
                collisions.append((pid, word, used_words[word], ""))
        # If nothing assigned, mark the final fallback in collisions
        if pid not in assigned and pred_candidates.get(pid):
            collisions[-1] = (*collisions[-1][:3], "FAILED")

    # Update collision records with what was actually assigned
    for i, (pid, tried, coll_with, got) in enumerate(collisions):
        if pid in assigned and got != "FAILED":
            collisions[i] = (pid, tried, coll_with, assigned[pid])

    # Build mappings
    mappings = []
    for pid, label in WIKIDATA_LABELS.items():
        if pid not in assigned:
            if verbose:
                cands = [w for w, _ in pred_candidates.get(pid, [])]
                print(f"    SKIP {pid} ({label}): "
                      f"no valid single-token concept "
                      f"[tried: {cands[:5]}]")
            continue

        concept = assigned[pid]
        bpe_id = _is_single_token(concept)
        link = _infer_linking(label)

        m = RelationMapping(
            predicate_id=pid, concept_word=concept,
            linking_token=link, description=label,
            concept_bpe_id=bpe_id, valid=True,
        )
        mappings.append(m)

    if verbose:
        print(f"  Auto-assigned: {len(mappings)} / "
              f"{len(WIKIDATA_LABELS)} predicates")
        # Report collisions
        resolved = [(p, t, c, g) for p, t, c, g in collisions
                     if g and g != "FAILED"]
        failed = [(p, t, c, g) for p, t, c, g in collisions
                   if g == "FAILED"]
        if resolved:
            print(f"  Collisions resolved: {len(resolved)}")
            for pid, tried, coll_with, got in resolved:
                print(f"    {pid} wanted '{tried}' "
                      f"(taken by {coll_with}) → '{got}'")
        if failed:
            print(f"  Collisions unresolved: {len(failed)}")
            for pid, tried, coll_with, _ in failed:
                print(f"    {pid} wanted '{tried}' "
                      f"(taken by {coll_with}) → DROPPED")
        # Show all assignments
        print(f"\n  {'Pred':>6} {'Label':>35} → {'Concept':>15} "
              f"{'Link':>4}")
        print(f"  {'─' * 68}")
        for m in sorted(mappings, key=lambda x: x.predicate_id):
            print(f"  {m.predicate_id:>6} {m.description:>35} → "
                  f"{m.concept_word:>15} {m.linking_token:>4}")

    return mappings


# ═══════════════════════════════════════════════════════════════════
#  FACT FILTERING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LAMAFact:
    predicate_id: str
    sub_label: str
    obj_label: str
    obj_bpe_id: int
    storage_text: str       # "capital is Paris."
    storage_ids: list[int]  # BPE token IDs
    query_text: str         # "capital is"
    query_ids: list[int]


def filter_facts(
    trex_data: dict[str, list[dict]],
    mappings: list[RelationMapping],
    tokenizer: GPT2Tokenizer,
    max_per_relation: int = 100,
    verbose: bool = True,
) -> dict[str, list[LAMAFact]]:
    """Filter LAMA facts to single-BPE-token objects."""
    valid_map = {m.predicate_id: m for m in mappings if m.valid}
    filtered: dict[str, list[LAMAFact]] = {}
    stats = {"total": 0, "no_mapping": 0, "multi_token": 0, "valid": 0}

    # Track seen objects per relation to deduplicate
    for pid, facts in trex_data.items():
        if pid not in valid_map:
            stats["no_mapping"] += len(facts)
            continue

        mapping = valid_map[pid]
        relation_facts = []
        seen_objs: set[str] = set()

        for fact in facts:
            stats["total"] += 1
            obj = fact.get("obj_label", "")
            if not obj or obj in seen_objs:
                continue

            obj_ids = tokenizer.encode(" " + obj, add_special_tokens=False)
            if len(obj_ids) != 1:
                stats["multi_token"] += 1
                continue

            seen_objs.add(obj)
            storage = f"{mapping.concept_word} {mapping.linking_token} {obj}."
            query = f"{mapping.concept_word} {mapping.linking_token}"

            relation_facts.append(LAMAFact(
                predicate_id=pid,
                sub_label=fact.get("sub_label", ""),
                obj_label=obj,
                obj_bpe_id=obj_ids[0],
                storage_text=storage,
                storage_ids=tokenizer.encode(storage, add_special_tokens=False),
                query_text=query,
                query_ids=tokenizer.encode(query, add_special_tokens=False),
            ))
            stats["valid"] += 1

        if relation_facts:
            rng = random.Random(42)
            rng.shuffle(relation_facts)
            filtered[pid] = relation_facts[:max_per_relation]

    if verbose:
        print(f"  Filtering: {stats['valid']} valid / "
              f"{stats['total']} total")
        print(f"    No mapping: {stats['no_mapping']}, "
              f"multi-token obj: {stats['multi_token']}")
        print(f"  Relations with valid facts: {len(filtered)}")
        for pid in sorted(filtered):
            m = valid_map[pid]
            n = len(filtered[pid])
            print(f"    {pid:>6} {m.description:>28}: "
                  f"{n:>4} facts  [{m.concept_word} {m.linking_token} ...]")

    return filtered


# ═══════════════════════════════════════════════════════════════════
#  EPISODE GENERATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    facts: list[LAMAFact]


def make_episodes(
    filtered: dict[str, list[LAMAFact]],
    n_episodes: int,
    n_facts: int,
    seed: int = 42,
) -> list[Episode]:
    """Generate evaluation episodes sampling from all relations."""
    rng = random.Random(seed)
    pool = []
    for facts in filtered.values():
        pool.extend(facts)

    episodes = []
    for _ in range(n_episodes):
        selected = rng.sample(pool, min(n_facts, len(pool)))
        episodes.append(Episode(facts=selected))
    return episodes


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Results:
    accuracy: float
    n_correct: int
    n_total: int
    per_episode: list[float]
    per_relation_acc: dict[str, float]
    per_relation_n: dict[str, int]
    ci_lo: float = 0.0
    ci_hi: float = 0.0


def bootstrap_ci(values: list[float], n_boot: int = 10_000,
                 seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n <= 1:
        m = sum(values) / max(n, 1)
        return m, m
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _all_entity_ids(filtered: dict[str, list[LAMAFact]]) -> list[int]:
    ids = set()
    for facts in filtered.values():
        for f in facts:
            ids.add(f.obj_bpe_id)
    return sorted(ids)


def eval_cross_context(
    model, episodes: list[Episode], entity_ids: list[int],
    verbose: bool = False,
) -> Results:
    """THE REAL TEST: write facts to trace, query with trace only."""
    device = next(model.parameters()).device
    model.eval()

    total_c, total_n = 0, 0
    per_ep: list[float] = []
    rel_c: dict[str, int] = {}
    rel_n: dict[str, int] = {}

    for i, ep in enumerate(episodes):
        model.reset_traces()

        # Write
        model.set_trace_mode(use=False, update=True)
        for fact in ep.facts:
            ids = torch.tensor([fact.storage_ids], dtype=torch.long,
                               device=device)
            with torch.no_grad():
                model(ids)

        # Read
        model.set_trace_mode(use=True, update=False)
        correct = 0
        for fact in ep.facts:
            pred = _predict_answer(model, fact.query_ids, entity_ids)
            ok = pred == fact.obj_bpe_id
            if ok:
                correct += 1
                rel_c[fact.predicate_id] = rel_c.get(
                    fact.predicate_id, 0) + 1
            rel_n[fact.predicate_id] = rel_n.get(
                fact.predicate_id, 0) + 1

        total_c += correct
        total_n += len(ep.facts)
        per_ep.append(correct / max(len(ep.facts), 1))

        if verbose and (i + 1) % 20 == 0:
            print(f"    ep {i+1}: running "
                  f"{total_c/max(total_n, 1):.1%}")

    acc = total_c / max(total_n, 1)
    lo, hi = bootstrap_ci(per_ep)
    rel_acc = {p: rel_c.get(p, 0) / max(rel_n.get(p, 0), 1)
               for p in rel_n}

    return Results(accuracy=acc, n_correct=total_c, n_total=total_n,
                   per_episode=per_ep, per_relation_acc=rel_acc,
                   per_relation_n=dict(rel_n), ci_lo=lo, ci_hi=hi)


def eval_baseline(
    model, episodes: list[Episode], entity_ids: list[int],
    tokenizer: GPT2Tokenizer,
) -> Results:
    """In-context baseline: all facts + query in one forward pass."""
    device = next(model.parameters()).device
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    space_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    total_c, total_n = 0, 0
    per_ep: list[float] = []
    rel_c: dict[str, int] = {}
    rel_n: dict[str, int] = {}

    for ep in episodes:
        correct = 0
        for fact in ep.facts:
            # Build context: all storage texts + this query
            ic_ids: list[int] = []
            for f in ep.facts:
                if ic_ids:
                    ic_ids.append(space_id)
                ic_ids.extend(f.storage_ids)
            ic_ids.append(space_id)
            ic_ids.extend(fact.query_ids)

            pred = _predict_answer(model, ic_ids, entity_ids)
            ok = pred == fact.obj_bpe_id
            if ok:
                correct += 1
                rel_c[fact.predicate_id] = rel_c.get(
                    fact.predicate_id, 0) + 1
            rel_n[fact.predicate_id] = rel_n.get(
                fact.predicate_id, 0) + 1

        total_c += correct
        total_n += len(ep.facts)
        per_ep.append(correct / max(len(ep.facts), 1))

    acc = total_c / max(total_n, 1)
    lo, hi = bootstrap_ci(per_ep)
    rel_acc = {p: rel_c.get(p, 0) / max(rel_n.get(p, 0), 1)
               for p in rel_n}

    return Results(accuracy=acc, n_correct=total_c, n_total=total_n,
                   per_episode=per_ep, per_relation_acc=rel_acc,
                   per_relation_n=dict(rel_n), ci_lo=lo, ci_hi=hi)


def eval_knn_lm(
    model, episodes: list[Episode], entity_ids: list[int],
    k: int = 8, temperature: float = 10.0, lam: float = 0.25,
    verbose: bool = False,
) -> Results:
    """kNN-LM baseline (Khandelwal et al., 2020).

    Stores (hidden_state, next_token) from fact passages, retrieves by
    nearest neighbor, interpolates with LM distribution.

    On LAMA, queries are exact prefixes of storage text, so causal
    attention produces identical hidden states — kNN-LM should achieve
    near-perfect accuracy (this is the easy case for kNN-LM).
    """
    from hebbian_trace.rag_baselines import KNNMemoryStore

    device = next(model.parameters()).device
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    knn = KNNMemoryStore(model, k=k, temperature=temperature, lam=lam)

    total_c, total_n = 0, 0
    per_ep: list[float] = []
    rel_c: dict[str, int] = {}
    rel_n: dict[str, int] = {}

    for i, ep in enumerate(episodes):
        knn.reset()

        # Store each fact's hidden states
        for fact in ep.facts:
            knn.store_passage(fact.storage_ids)

        # Query via kNN-LM
        correct = 0
        for fact in ep.facts:
            pred = knn.predict(fact.query_ids, entity_ids)
            ok = pred == fact.obj_bpe_id
            if ok:
                correct += 1
                rel_c[fact.predicate_id] = rel_c.get(
                    fact.predicate_id, 0) + 1
            rel_n[fact.predicate_id] = rel_n.get(
                fact.predicate_id, 0) + 1

        total_c += correct
        total_n += len(ep.facts)
        per_ep.append(correct / max(len(ep.facts), 1))

        if verbose and (i + 1) % 20 == 0:
            print(f"    ep {i+1}: running "
                  f"{total_c/max(total_n, 1):.1%}")

    acc = total_c / max(total_n, 1)
    lo, hi = bootstrap_ci(per_ep)
    rel_acc = {p: rel_c.get(p, 0) / max(rel_n.get(p, 0), 1)
               for p in rel_n}

    return Results(accuracy=acc, n_correct=total_c, n_total=total_n,
                   per_episode=per_ep, per_relation_acc=rel_acc,
                   per_relation_n=dict(rel_n), ci_lo=lo, ci_hi=hi)


def eval_no_memory(
    model, episodes: list[Episode], entity_ids: list[int],
) -> Results:
    """No-memory baseline: query-only, no trace. Expected ~random."""
    model.eval()
    model.set_trace_mode(use=False, update=False)
    model.reset_traces()

    total_c, total_n = 0, 0
    per_ep: list[float] = []
    rel_c: dict[str, int] = {}
    rel_n: dict[str, int] = {}

    for ep in episodes:
        correct = 0
        for fact in ep.facts:
            pred = _predict_answer(model, fact.query_ids, entity_ids)
            ok = pred == fact.obj_bpe_id
            if ok:
                correct += 1
                rel_c[fact.predicate_id] = rel_c.get(
                    fact.predicate_id, 0) + 1
            rel_n[fact.predicate_id] = rel_n.get(
                fact.predicate_id, 0) + 1

        total_c += correct
        total_n += len(ep.facts)
        per_ep.append(correct / max(len(ep.facts), 1))

    acc = total_c / max(total_n, 1)
    lo, hi = bootstrap_ci(per_ep)
    rel_acc = {p: rel_c.get(p, 0) / max(rel_n.get(p, 0), 1)
               for p in rel_n}

    return Results(accuracy=acc, n_correct=total_c, n_total=total_n,
                   per_episode=per_ep, per_relation_acc=rel_acc,
                   per_relation_n=dict(rel_n), ci_lo=lo, ci_hi=hi)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_pipeline(
    mode_label: str,
    model,
    mappings: list[RelationMapping],
    trex: dict[str, list[dict]],
    tokenizer: GPT2Tokenizer,
    n_eval: int,
    n_facts_list: list[int],
    max_per_relation: int,
    seed: int,
    verbose: bool,
) -> dict[int, dict[str, Results]]:
    """Run full eval pipeline for a given set of mappings."""
    print(f"\n{'═' * 70}")
    print(f"  PIPELINE: {mode_label}")
    print(f"{'═' * 70}")

    valid = [m for m in mappings if m.valid]
    print(f"  Valid relations: {len(valid)}")

    filtered = filter_facts(trex, mappings, tokenizer,
                            max_per_relation=max_per_relation,
                            verbose=verbose)
    entity_ids = _all_entity_ids(filtered)
    n_entities = len(entity_ids)
    n_valid_facts = sum(len(v) for v in filtered.values())
    print(f"\n  Entity vocab: {n_entities} unique tokens")
    print(f"  Random chance: {1.0 / max(n_entities, 1):.2%}")
    print(f"  Total valid facts: {n_valid_facts}")

    all_results: dict[int, dict[str, Results]] = {}

    for nf in n_facts_list:
        print(f"\n{'─' * 70}")
        print(f"  n_facts = {nf}")
        print(f"{'─' * 70}")

        episodes = make_episodes(filtered, n_eval, nf,
                                 seed=seed + nf * 1000)

        print("  [1/4] Cross-context (trace)...")
        cross = eval_cross_context(model, episodes, entity_ids,
                                   verbose=verbose)

        print("  [2/4] kNN-LM...")
        knn = eval_knn_lm(model, episodes, entity_ids,
                          k=32, temperature=10.0, lam=0.25,
                          verbose=verbose)

        print("  [3/4] In-context baseline...")
        base = eval_baseline(model, episodes, entity_ids, tokenizer)

        print("  [4/4] No memory...")
        nomem = eval_no_memory(model, episodes, entity_ids)

        all_results[nf] = {
            "cross_context": cross,
            "knn_lm": knn,
            "baseline": base,
            "no_memory": nomem,
        }

        print(f"\n  {'Condition':>20} {'Accuracy':>10} {'95% CI':>16}")
        print(f"  {'─' * 50}")
        for name, res in [("Cross-context", cross),
                          ("kNN-LM", knn),
                          ("In-context", base),
                          ("No memory", nomem)]:
            print(f"  {name:>20} {res.accuracy:>9.1%} "
                  f"[{res.ci_lo:.1%}, {res.ci_hi:.1%}]")

    # Per-relation breakdown
    max_nf = max(n_facts_list)
    ctx = all_results[max_nf]["cross_context"]
    valid_map = {m.predicate_id: m for m in mappings if m.valid}

    print(f"\n  PER-RELATION (n={max_nf}, cross-context)")
    print(f"  {'Pred':>6} {'Concept':>15} {'Description':>28} "
          f"{'Acc':>7} {'N':>5}")
    print(f"  {'─' * 65}")
    for pid, acc in sorted(ctx.per_relation_acc.items(),
                           key=lambda x: -x[1]):
        n = ctx.per_relation_n.get(pid, 0)
        m = valid_map.get(pid)
        desc = m.description if m else "?"
        cw = m.concept_word if m else "?"
        print(f"  {pid:>6} {cw:>15} {desc:>28} {acc:>6.0%} {n:>5}")

    return all_results


def run(
    n_eval: int = 100,
    n_facts_list: list[int] | None = None,
    max_per_relation: int = 100,
    seed: int = 42,
    quick: bool = False,
    verbose: bool = True,
    concept_mode: str = "oracle",
):
    device = get_device()
    t0 = time.time()

    if n_facts_list is None:
        n_facts_list = [1, 3, 5] if quick else [1, 3, 5, 7, 10]

    print("=" * 70)
    print("  LAMA BENCHMARK — Hebbian Trace Memory")
    print("=" * 70)
    print(f"  Device:       {device}")
    print(f"  Episodes:     {n_eval}")
    print(f"  n_facts:      {n_facts_list}")
    print(f"  Concept mode: {concept_mode}")
    print(f"  Seed:         {seed}")
    print()

    # ── Setup ──
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    linking_ids = get_linking_bpe_ids(tokenizer)

    print("\n--- Loading LAMA T-REx ---")
    trex = load_trex()
    total_facts = sum(len(v) for v in trex.values())
    print(f"  Loaded {total_facts} facts across {len(trex)} relations")

    # ── Model ──
    print("\n--- Model ---")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=0.5,
        trace_lr=1.0, trace_decay=0.99, device=device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    model.eval()
    print("  GPT-2 Small + Trace (α=0.5, PS 8×/k=16)")

    # ── Build mappings ──
    oracle_mappings = None
    auto_mappings = None

    if concept_mode in ("oracle", "both"):
        print("\n--- Oracle Concept Assignment ---")
        oracle_mappings = validate_mappings(tokenizer)
        valid = [m for m in oracle_mappings if m.valid]
        print(f"  Valid: {len(valid)} / {len(oracle_mappings)} relations")

    if concept_mode in ("auto", "both"):
        print("\n--- Auto Concept Assignment ---")
        auto_mappings = auto_assign_concepts(tokenizer, verbose=verbose)

    # ── Run pipelines ──
    oracle_results = None
    auto_results = None

    if oracle_mappings is not None:
        oracle_results = _run_pipeline(
            "Oracle concept words", model, oracle_mappings,
            trex, tokenizer, n_eval, n_facts_list,
            max_per_relation, seed, verbose)

    if auto_mappings is not None:
        auto_results = _run_pipeline(
            "Auto concept words (from Wikidata labels)", model,
            auto_mappings, trex, tokenizer, n_eval, n_facts_list,
            max_per_relation, seed, verbose)

    # ── Comparison table (both mode) ──
    if oracle_results and auto_results:
        n_oracle_rel = len([m for m in oracle_mappings if m.valid])
        n_auto_rel = len([m for m in auto_mappings if m.valid])

        print(f"\n{'═' * 70}")
        print("  ORACLE vs AUTO COMPARISON (cross-context trace)")
        print(f"{'═' * 70}")
        print(f"  Oracle: {n_oracle_rel} relations, "
              f"Auto: {n_auto_rel} relations")
        print(f"\n  {'n':>4} {'Oracle':>16} {'Auto':>16} {'Δ':>8}")
        print(f"  {'─' * 48}")
        for nf in n_facts_list:
            o = oracle_results[nf]["cross_context"]
            a = auto_results[nf]["cross_context"]
            delta = a.accuracy - o.accuracy
            print(f"  {nf:>4} "
                  f"{o.accuracy:>6.1%} [{o.ci_lo:.1%},{o.ci_hi:.1%}] "
                  f"{a.accuracy:>6.1%} [{a.ci_lo:.1%},{a.ci_hi:.1%}] "
                  f"{delta:>+7.1%}")

    # ── Summary ──
    # Use last pipeline's results for summary
    main_results = auto_results or oracle_results
    main_mappings = auto_mappings or oracle_mappings
    filtered = filter_facts(trex, main_mappings, tokenizer,
                            max_per_relation=max_per_relation,
                            verbose=False)
    entity_ids = _all_entity_ids(filtered)
    n_entities = len(entity_ids)
    rnd_chance = 1.0 / max(n_entities, 1)

    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print("  LAMA BENCHMARK SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'n':>4} {'Cross-ctx':>10} {'kNN-LM':>10} {'In-ctx':>10} "
          f"{'No-mem':>10} {'Trace-kNN':>10}")
    print(f"  {'─' * 58}")
    for nf in n_facts_list:
        c = main_results[nf]["cross_context"].accuracy
        k = main_results[nf]["knn_lm"].accuracy
        b = main_results[nf]["baseline"].accuracy
        nm = main_results[nf]["no_memory"].accuracy
        gap = c - k
        print(f"  {nf:>4} {c:>9.1%} {k:>9.1%} {b:>9.1%} "
              f"{nm:>9.1%} {gap:>+9.1%}")

    print(f"\n  Entities: {n_entities}, "
          f"random: {rnd_chance:.2%}, "
          f"relations: {len(filtered)}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'═' * 70}")

    return main_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LAMA benchmark for Hebbian Trace Memory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 episodes, n=[1,3,5])")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Episodes (default: 20 quick, 100 full)")
    parser.add_argument("--max-per-rel", type=int, default=100,
                        help="Max facts per relation (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept-mode",
                        choices=["oracle", "auto", "both"],
                        default="oracle",
                        help="Concept assignment: oracle (hand-picked), "
                             "auto (from Wikidata labels), both (compare)")
    args = parser.parse_args()

    n_eval = args.n_eval or (20 if args.quick else 100)
    run(n_eval=n_eval, max_per_relation=args.max_per_rel,
        seed=args.seed, quick=args.quick,
        concept_mode=args.concept_mode)
