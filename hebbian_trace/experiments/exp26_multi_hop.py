"""Exp 26: Multi-hop reasoning via trace chain.

Variant B: decode intermediate via wte argmax, re-encode as Q, query T_v.
Zero new parameters — uses existing write_fact_direct + retrieve_direct.

Pipeline:
  Hop 1: Q(concept) → T_v → V → W_out → wte.T → argmax → intermediate_token
  Hop 2: Q(intermediate) → T_v → V → W_out → wte.T → argmax → final_answer

Both hops use pure trace retrieval (no GPT-2 forward pass, no alpha).
This is a DIFFERENT regime than standard retrieval — no LM prior bias.

Four evaluation phases:
  Phase 1: Direct retrieval validation (read_direct ≈ standard forward?)
  Phase 2: Chain link storage + Q similarity diagnostic
  Phase 3: Multi-hop chains (hop-1, hop-2|oracle, hop-2|pred, end-to-end)
  Phase 4: Multi-chain capacity sweep

Usage:
  python -m hebbian_trace.experiments.exp26_multi_hop --quick
  python -m hebbian_trace.experiments.exp26_multi_hop --phase validate
  python -m hebbian_trace.experiments.exp26_multi_hop --phase chains
  python -m hebbian_trace.experiments.exp26_multi_hop --phase multihop --n-eval 50
  python -m hebbian_trace.experiments.exp26_multi_hop --phase capacity --n-eval 50
"""

import argparse
import random
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_concept_vocab, ConceptEntry,
    _predict_answer, get_linking_bpe_ids,
    validate_single_token_entities,
)
from .exp24_free_text import setup_model


# ── Chain pair data ──────────────────────────────────────────────────

# Geographically correct city→country pairs.
# Both cities and countries are in existing NLP entity pools,
# so single-token BPE is guaranteed by pool validation.
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
    """One validated city→country chain link with BPE IDs."""
    city_name: str
    city_token_id: int
    country_name: str
    country_token_id: int


def build_chain_entries(tokenizer: GPT2Tokenizer) -> list[ChainEntry]:
    """Validate city→country pairs as single-token BPE."""
    entries = []
    for city, country in CITY_COUNTRY_PAIRS:
        city_ids = tokenizer.encode(" " + city, add_special_tokens=False)
        country_ids = tokenizer.encode(" " + country, add_special_tokens=False)
        if len(city_ids) == 1 and len(country_ids) == 1:
            entries.append(ChainEntry(
                city_name=city,
                city_token_id=city_ids[0],
                country_name=country,
                country_token_id=country_ids[0],
            ))
    return entries


# ── Result data structures ───────────────────────────────────────────

@dataclass
class MultiHopResult:
    """Results for one multi-hop evaluation configuration."""
    hop1_direct_acc: float = 0.0       # via retrieve_direct
    hop1_question_acc: float = 0.0     # via _predict_answer (reference)
    hop2_oracle_acc: float = 0.0       # hop-2 given correct intermediate
    hop2_predicted_acc: float = 0.0    # hop-2 given predicted intermediate
    end_to_end_acc: float = 0.0        # both hops correct
    n_chains: int = 0                  # how many chain links stored
    n_extra_facts: int = 0             # interference from standard facts
    n_episodes: int = 0


# ── Phase 1: Direct retrieval validation ─────────────────────────────

def run_phase1_validate(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_eval: int = 50,
    n_facts: int = 5,
) -> dict:
    """Validate read_direct matches standard forward pass retrieval.

    Writes N facts, queries each via:
    1. retrieve_direct (pure trace, no alpha, no GPT-2)
    2. _predict_answer (full forward: GPT-2 + alpha * trace)

    Reports accuracy for both paths.
    """
    print(f"\n{'='*60}")
    print(f"Phase 1: Direct retrieval validation")
    print(f"  n_eval={n_eval}, n_facts={n_facts}")
    print(f"{'='*60}")

    # Select fact types for testing
    type_names = list(concept_vocab.keys())[:n_facts]
    all_entity_ids = sorted({
        eid for tn in type_names
        for _, eid in concept_vocab[tn].entity_pool
    })

    direct_correct = 0
    question_correct = 0
    total = 0

    for ep in range(n_eval):
        # Reset trace
        model.trace.reset_traces()

        # Pick random entities per type
        episode_facts = {}
        for tn in type_names:
            entry = concept_vocab[tn]
            entity_name, entity_id = random.choice(entry.entity_pool)
            episode_facts[tn] = (entity_name, entity_id)

        # Write all facts
        model.set_trace_mode(use=False, update=False)
        for tn in type_names:
            entry = concept_vocab[tn]
            _, entity_id = episode_facts[tn]
            model.write_fact_direct(entry.concept_token_id, entity_id)

        # Query via both paths
        model.set_trace_mode(use=True, update=False)
        for tn in type_names:
            entry = concept_vocab[tn]
            _, expected_id = episode_facts[tn]

            # Path 1: retrieve_direct (pure trace)
            pred_direct = model.retrieve_direct(
                entry.concept_token_id, all_entity_ids)
            if pred_direct == expected_id:
                direct_correct += 1

            # Path 2: _predict_answer (full forward)
            q_ids = tokenizer.encode(
                entry.question_template, add_special_tokens=False)
            pred_question = _predict_answer(model, q_ids, all_entity_ids)
            if pred_question == expected_id:
                question_correct += 1

            total += 1

        model.set_trace_mode(use=False, update=False)

    direct_acc = direct_correct / total * 100
    question_acc = question_correct / total * 100

    print(f"\n  Results ({total} queries across {n_eval} episodes):")
    print(f"    retrieve_direct:  {direct_acc:5.1f}% ({direct_correct}/{total})")
    print(f"    _predict_answer:  {question_acc:5.1f}% ({question_correct}/{total})")
    print(f"    Delta:            {direct_acc - question_acc:+.1f}pp")

    return {
        "direct_acc": direct_acc,
        "question_acc": question_acc,
        "n_eval": n_eval,
        "n_facts": n_facts,
        "total_queries": total,
    }


# ── Phase 2: Chain link storage + Q diagnostic ──────────────────────

def compute_q_similarity(
    model: GPT2WithTrace,
    chain_entries: list[ChainEntry],
) -> dict:
    """Compute cosine similarity between Q vectors of city tokens.

    Diagnostic: if mean cosine > 0.5, expect early capacity degradation.
    If < 0.3, capacity should be good.
    """
    Qs = []
    names = []
    for entry in chain_entries:
        Q = model.trace.compute_q_for_token(
            model._wte, entry.city_token_id)  # (H, d_trace)
        # Apply pattern separation to match retrieval path
        if model.trace._pattern_sep_enabled:
            Q_exp = model.trace._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2))
            Q_flat = Q_exp.squeeze(0).squeeze(1)  # (H, expanded)
        else:
            Q_flat = Q
        Qs.append(Q_flat.view(-1).float())  # flatten to single vector
        names.append(entry.city_name)

    n = len(Qs)
    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = torch.nn.functional.cosine_similarity(
                Qs[i].unsqueeze(0), Qs[j].unsqueeze(0)).item()
            cos_sims.append(cos)

    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0
    max_cos = max(cos_sims) if cos_sims else 0
    min_cos = min(cos_sims) if cos_sims else 0

    return {
        "mean_cosine": mean_cos,
        "max_cosine": max_cos,
        "min_cosine": min_cos,
        "n_cities": n,
    }


def run_phase2_chains(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    chain_entries: list[ChainEntry],
    n_eval: int = 50,
) -> dict:
    """Test chain link storage: city→country in isolation.

    Writes M chain links, queries each via retrieve_direct.
    Sweep M = 1, 3, 5, 8, 11 (or all available).
    Also computes Q similarity diagnostic.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: Chain link storage + Q similarity diagnostic")
    print(f"  n_eval={n_eval}, available chains={len(chain_entries)}")
    print(f"{'='*60}")

    # Q similarity diagnostic
    q_sim = compute_q_similarity(model, chain_entries)
    print(f"\n  Q similarity diagnostic (city tokens, pattern-sep expanded):")
    print(f"    Mean cosine: {q_sim['mean_cosine']:.3f}")
    print(f"    Max cosine:  {q_sim['max_cosine']:.3f}")
    print(f"    Min cosine:  {q_sim['min_cosine']:.3f}")
    if q_sim['mean_cosine'] > 0.5:
        print(f"    WARNING: high mean cosine — expect early degradation")
    elif q_sim['mean_cosine'] < 0.3:
        print(f"    GOOD: low mean cosine — capacity should be good")
    else:
        print(f"    MODERATE: capacity may degrade at higher N")

    # All country IDs as candidates
    all_country_ids = sorted({e.country_token_id for e in chain_entries})

    # Sweep N chains
    max_n = len(chain_entries)
    sweep_ns = [n for n in [1, 3, 5, 8, 11] if n <= max_n]
    if max_n not in sweep_ns:
        sweep_ns.append(max_n)

    results_by_n = {}

    for n_chains in sweep_ns:
        correct = 0
        total = 0

        for ep in range(n_eval):
            model.trace.reset_traces()
            model.set_trace_mode(use=False, update=False)

            # Pick N random chains
            selected = random.sample(chain_entries, n_chains)

            # Write chain links
            for ce in selected:
                model.write_fact_direct(ce.city_token_id, ce.country_token_id)

            # Query each chain
            model.set_trace_mode(use=True, update=False)
            for ce in selected:
                pred = model.retrieve_direct(
                    ce.city_token_id, all_country_ids)
                if pred == ce.country_token_id:
                    correct += 1
                total += 1

            model.set_trace_mode(use=False, update=False)

        acc = correct / total * 100
        results_by_n[n_chains] = acc
        print(f"    N={n_chains:2d} chains: {acc:5.1f}% ({correct}/{total})")

    return {
        "q_similarity": q_sim,
        "accuracy_by_n": results_by_n,
        "n_eval": n_eval,
    }


# ── Phase 3: Multi-hop chains ───────────────────────────────────────

def run_phase3_multihop(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    chain_entries: list[ChainEntry],
    n_eval: int = 50,
    n_extra_facts: int = 3,
) -> dict:
    """Multi-hop: write person's city + city→country, query end-to-end.

    For each episode:
    1. Pick a random city→country chain
    2. Write: concept("live") → city_token  (person's city)
    3. Write: city_token → country_token    (chain link)
    4. Optionally write N extra standard facts  (interference)
    5. Measure: hop-1, hop-2|oracle, hop-2|pred, end-to-end

    Also tests hop-1 via standard question as reference.
    """
    print(f"\n{'='*60}")
    print(f"Phase 3: Multi-hop chains")
    print(f"  n_eval={n_eval}, n_extra_facts={n_extra_facts}")
    print(f"{'='*60}")

    city_entry = concept_vocab.get("city")
    if city_entry is None:
        print("  ERROR: 'city' type not in concept_vocab!")
        return {}

    # Candidates
    all_city_ids = sorted({eid for _, eid in city_entry.entity_pool})
    all_country_ids = sorted({e.country_token_id for e in chain_entries})

    # Extra fact types (exclude city and country to avoid addressing collision)
    extra_types = [
        tn for tn in concept_vocab
        if tn not in ("city", "country")
    ][:n_extra_facts]

    all_extra_entity_ids = sorted({
        eid for tn in extra_types
        for _, eid in concept_vocab[tn].entity_pool
    })

    # Sweep n_extra_facts
    sweep_extras = sorted(set([0, n_extra_facts]))

    for n_extra in sweep_extras:
        hop1_direct_correct = 0
        hop1_question_correct = 0
        hop2_oracle_correct = 0
        hop2_pred_correct = 0
        end_to_end_correct = 0
        total = 0

        et = extra_types[:n_extra]

        for ep in range(n_eval):
            model.trace.reset_traces()
            model.set_trace_mode(use=False, update=False)

            # Pick random chain
            chain = random.choice(chain_entries)

            # Write person's city: concept("live") → city
            model.write_fact_direct(
                city_entry.concept_token_id, chain.city_token_id)

            # Write chain link: city → country
            model.write_fact_direct(
                chain.city_token_id, chain.country_token_id)

            # Write extra facts for interference
            for tn in et:
                entry = concept_vocab[tn]
                _, eid = random.choice(entry.entity_pool)
                model.write_fact_direct(entry.concept_token_id, eid)

            # Query: hop-1 (get city)
            model.set_trace_mode(use=True, update=False)

            # Hop-1 via retrieve_direct
            hop1_pred = model.retrieve_direct(
                city_entry.concept_token_id, all_city_ids)
            hop1_ok = (hop1_pred == chain.city_token_id)
            if hop1_ok:
                hop1_direct_correct += 1

            # Hop-1 via standard question (reference)
            q_ids = tokenizer.encode(
                city_entry.question_template, add_special_tokens=False)
            hop1_q_pred = _predict_answer(model, q_ids, all_city_ids)
            if hop1_q_pred == chain.city_token_id:
                hop1_question_correct += 1

            # Hop-2 given oracle intermediate
            hop2_oracle_pred = model.retrieve_direct(
                chain.city_token_id, all_country_ids)
            if hop2_oracle_pred == chain.country_token_id:
                hop2_oracle_correct += 1

            # Hop-2 given predicted intermediate (chained)
            hop2_pred = model.retrieve_direct(
                hop1_pred, all_country_ids)
            if hop2_pred == chain.country_token_id:
                hop2_pred_correct += 1

            # End-to-end: hop1 correct AND hop2|pred correct
            if hop1_ok and hop2_pred == chain.country_token_id:
                end_to_end_correct += 1

            total += 1

            model.set_trace_mode(use=False, update=False)

        h1d = hop1_direct_correct / total * 100
        h1q = hop1_question_correct / total * 100
        h2o = hop2_oracle_correct / total * 100
        h2p = hop2_pred_correct / total * 100
        e2e = end_to_end_correct / total * 100

        print(f"\n  n_extra_facts={n_extra}, {total} episodes:")
        print(f"    hop-1 (direct):     {h1d:5.1f}%")
        print(f"    hop-1 (question):   {h1q:5.1f}%")
        print(f"    hop-2|oracle:       {h2o:5.1f}%")
        print(f"    hop-2|predicted:    {h2p:5.1f}%")
        print(f"    end-to-end:         {e2e:5.1f}%")

    return {
        "n_eval": n_eval,
        "n_extra_facts": n_extra_facts,
        "hop1_direct": h1d,
        "hop1_question": h1q,
        "hop2_oracle": h2o,
        "hop2_predicted": h2p,
        "end_to_end": e2e,
    }


# ── Phase 4: Multi-chain capacity ───────────────────────────────────

def run_phase4_capacity(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    chain_entries: list[ChainEntry],
    n_eval: int = 50,
    n_extra_facts: int = 3,
) -> dict:
    """Multi-chain capacity: N city→country links + 1 person→city.

    Person's city is one of the N stored chains. Multi-hop must:
    1. Get person's city (hop-1)
    2. Find THAT city's country among N stored chains (hop-2)

    Sweep N = 1, 3, 5, 8, 11.
    """
    print(f"\n{'='*60}")
    print(f"Phase 4: Multi-chain capacity sweep")
    print(f"  n_eval={n_eval}, n_extra_facts={n_extra_facts}")
    print(f"{'='*60}")

    city_entry = concept_vocab.get("city")
    if city_entry is None:
        print("  ERROR: 'city' type not in concept_vocab!")
        return {}

    all_city_ids = sorted({eid for _, eid in city_entry.entity_pool})
    all_country_ids = sorted({e.country_token_id for e in chain_entries})

    extra_types = [
        tn for tn in concept_vocab
        if tn not in ("city", "country")
    ][:n_extra_facts]

    max_n = len(chain_entries)
    sweep_ns = [n for n in [1, 3, 5, 8, 11] if n <= max_n]
    if max_n not in sweep_ns:
        sweep_ns.append(max_n)

    results = {}

    for n_chains in sweep_ns:
        hop1_correct = 0
        hop2_oracle_correct = 0
        hop2_pred_correct = 0
        end_to_end_correct = 0
        total = 0

        for ep in range(n_eval):
            model.trace.reset_traces()
            model.set_trace_mode(use=False, update=False)

            # Pick N chains, one is the person's city
            selected = random.sample(chain_entries, n_chains)
            target_chain = random.choice(selected)

            # Write person's city
            model.write_fact_direct(
                city_entry.concept_token_id, target_chain.city_token_id)

            # Write ALL N chain links
            for ce in selected:
                model.write_fact_direct(ce.city_token_id, ce.country_token_id)

            # Write extra facts
            for tn in extra_types:
                entry = concept_vocab[tn]
                _, eid = random.choice(entry.entity_pool)
                model.write_fact_direct(entry.concept_token_id, eid)

            # Query
            model.set_trace_mode(use=True, update=False)

            # Hop-1: get person's city
            hop1_pred = model.retrieve_direct(
                city_entry.concept_token_id, all_city_ids)
            hop1_ok = (hop1_pred == target_chain.city_token_id)
            if hop1_ok:
                hop1_correct += 1

            # Hop-2|oracle: given correct city, get country
            hop2_oracle = model.retrieve_direct(
                target_chain.city_token_id, all_country_ids)
            if hop2_oracle == target_chain.country_token_id:
                hop2_oracle_correct += 1

            # Hop-2|predicted: given hop-1 prediction
            hop2_pred = model.retrieve_direct(
                hop1_pred, all_country_ids)
            if hop2_pred == target_chain.country_token_id:
                hop2_pred_correct += 1

            # End-to-end
            if hop1_ok and hop2_pred == target_chain.country_token_id:
                end_to_end_correct += 1

            total += 1
            model.set_trace_mode(use=False, update=False)

        h1 = hop1_correct / total * 100
        h2o = hop2_oracle_correct / total * 100
        h2p = hop2_pred_correct / total * 100
        e2e = end_to_end_correct / total * 100

        results[n_chains] = {
            "hop1": h1, "hop2_oracle": h2o,
            "hop2_pred": h2p, "end_to_end": e2e,
        }
        print(f"    N={n_chains:2d}: hop1={h1:5.1f}%  "
              f"hop2|orc={h2o:5.1f}%  hop2|pred={h2p:5.1f}%  "
              f"e2e={e2e:5.1f}%")

    return {
        "results_by_n": results,
        "n_eval": n_eval,
        "n_extra_facts": n_extra_facts,
    }


# ── Main experiment runner ───────────────────────────────────────────

def run_experiment(
    n_eval: int = 50,
    phase: str = "all",
    n_extra_facts: int = 3,
    quick: bool = False,
):
    """Run multi-hop reasoning experiment."""
    if quick:
        n_eval = min(n_eval, 10)

    # Setup
    model, tokenizer = setup_model(alpha=0.5, use_ps=True)
    concept_vocab = build_concept_vocab(tokenizer, include_extended=True)
    chain_entries = build_chain_entries(tokenizer)

    print(f"Multi-hop reasoning experiment (Variant B: wte argmax)")
    print(f"  Model: GPT-2 Small + Hebbian trace (PS 8x_k16)")
    print(f"  Chain pairs: {len(chain_entries)} validated city→country")
    print(f"  Concept types: {len(concept_vocab)}")

    for ce in chain_entries:
        print(f"    {ce.city_name} → {ce.country_name} "
              f"(city_id={ce.city_token_id}, "
              f"country_id={ce.country_token_id})")

    results = {}

    if phase in ("all", "validate"):
        results["phase1"] = run_phase1_validate(
            model, tokenizer, concept_vocab, n_eval=n_eval)

    if phase in ("all", "chains"):
        results["phase2"] = run_phase2_chains(
            model, tokenizer, chain_entries, n_eval=n_eval)

    if phase in ("all", "multihop"):
        results["phase3"] = run_phase3_multihop(
            model, tokenizer, concept_vocab, chain_entries,
            n_eval=n_eval, n_extra_facts=n_extra_facts)

    if phase in ("all", "capacity"):
        results["phase4"] = run_phase4_capacity(
            model, tokenizer, concept_vocab, chain_entries,
            n_eval=n_eval, n_extra_facts=n_extra_facts)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    if "phase1" in results:
        p1 = results["phase1"]
        print(f"  Phase 1 (validation): direct={p1['direct_acc']:.1f}%  "
              f"question={p1['question_acc']:.1f}%")

    if "phase2" in results:
        p2 = results["phase2"]
        qs = p2["q_similarity"]
        print(f"  Phase 2 (chains): Q cos_sim mean={qs['mean_cosine']:.3f}")
        for n, acc in p2["accuracy_by_n"].items():
            print(f"    N={n}: {acc:.1f}%")

    if "phase3" in results:
        p3 = results["phase3"]
        print(f"  Phase 3 (multi-hop): e2e={p3['end_to_end']:.1f}%  "
              f"hop1={p3['hop1_direct']:.1f}%  "
              f"hop2|orc={p3['hop2_oracle']:.1f}%")

    if "phase4" in results:
        p4 = results["phase4"]
        print(f"  Phase 4 (capacity):")
        for n, r in p4["results_by_n"].items():
            print(f"    N={n}: e2e={r['end_to_end']:.1f}%  "
                  f"hop1={r['hop1']:.1f}%  hop2|orc={r['hop2_oracle']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exp 26: Multi-hop reasoning via trace chain")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (10 episodes)")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Episodes per evaluation")
    parser.add_argument("--phase",
                        choices=["validate", "chains", "multihop",
                                 "capacity", "all"],
                        default="all",
                        help="Which phase to run")
    parser.add_argument("--n-extra-facts", type=int, default=3,
                        help="Extra standard facts for interference")
    args = parser.parse_args()

    run_experiment(
        n_eval=args.n_eval,
        phase=args.phase,
        n_extra_facts=args.n_extra_facts,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
