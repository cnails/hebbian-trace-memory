#!/usr/bin/env python3
"""Multi-hop reasoning via Hebbian trace chains.

Demonstrates two-hop reasoning with ZERO new parameters:
  Hop 1: Q(concept) -> T_v -> V -> W_out -> wte.T -> argmax -> intermediate
  Hop 2: Q(intermediate) -> T_v -> V -> W_out -> wte.T -> argmax -> answer

Example: "Where does the person live?" -> "Paris" -> "What country?" -> "France"

Both hops use pure trace retrieval (retrieve_direct), no GPT-2 forward pass.
Entity-as-concept addressing: city name becomes a concept word whose Q
addresses the country value in chain links.

Four evaluation phases:
  Phase 1: Direct retrieval validation (read_direct matches standard forward?)
  Phase 2: Chain link storage (city->country in isolation, capacity sweep)
  Phase 3: Multi-hop chains (hop-1, hop-2|oracle, hop-2|pred, end-to-end)
  Phase 4: Multi-chain capacity (N chains coexist, navigate to correct one)

Usage:
    python exp_multihop.py --quick           # 10 episodes, all phases
    python exp_multihop.py --phase validate  # validate direct retrieval
    python exp_multihop.py --phase chains    # chain link storage
    python exp_multihop.py --phase multihop  # multi-hop chains
    python exp_multihop.py --phase capacity  # capacity sweep
    python exp_multihop.py --n-eval 50       # full run, all phases
"""

import argparse
import random

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_concept_vocab,
    build_chain_entries,
    get_linking_bpe_ids,
    _predict_answer,
    ConceptEntry,
    ChainEntry,
)


# -- Setup --

def setup_model(
    weights_path: str = "weights/trace_module.pt",
) -> tuple[GPT2WithTrace, GPT2Tokenizer]:
    """Load GPT-2 + trace module with standard configuration."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    linking_ids = get_linking_bpe_ids(tokenizer)

    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.trace.load_state_dict(state, strict=False)
        print(f"Loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found, using random projections")

    return model, tokenizer


# -- Phase 1: Direct Retrieval Validation --

def run_phase1_validate(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    n_eval: int = 50,
    n_facts: int = 5,
) -> dict:
    """Validate read_direct matches standard forward pass retrieval.

    Writes N facts via write_fact_direct, queries each via:
    1. retrieve_direct (pure trace, no GPT-2 forward)
    2. _predict_answer (full forward: GPT-2 + alpha * trace)

    Expected: both paths should give ~same accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Phase 1: Direct retrieval validation")
    print(f"  n_eval={n_eval}, n_facts={n_facts}")
    print(f"{'='*60}")

    type_names = list(concept_vocab.keys())[:n_facts]
    all_entity_ids = sorted({
        eid for tn in type_names
        for _, eid in concept_vocab[tn].entity_pool
    })

    direct_correct = 0
    question_correct = 0
    total = 0

    for ep in range(n_eval):
        model.reset_traces()

        # Pick random entities per type
        episode_facts = {}
        for tn in type_names:
            entry = concept_vocab[tn]
            entity_name, entity_id = random.choice(entry.entity_pool)
            episode_facts[tn] = (entity_name, entity_id)

        # Write all facts via direct write
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

            # Path 2: _predict_answer (full GPT-2 forward + trace)
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
    print(f"    retrieve_direct:  {direct_acc:5.1f}%")
    print(f"    _predict_answer:  {question_acc:5.1f}%")
    print(f"    Delta:            {direct_acc - question_acc:+.1f}pp")

    return {
        "direct_acc": direct_acc,
        "question_acc": question_acc,
        "n_eval": n_eval,
        "n_facts": n_facts,
    }


# -- Phase 2: Chain Link Storage --

def run_phase2_chains(
    model: GPT2WithTrace,
    chain_entries: list[ChainEntry],
    n_eval: int = 50,
) -> dict:
    """Test chain link storage: city->country in isolation.

    Writes M chain links via write_fact_direct(city_token, country_token),
    queries each via retrieve_direct(city_token, country_candidates).
    Sweeps M = 1, 3, 5, 8, 11.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: Chain link storage")
    print(f"  n_eval={n_eval}, available chains={len(chain_entries)}")
    print(f"{'='*60}")

    all_country_ids = sorted({e.country_token_id for e in chain_entries})

    max_n = len(chain_entries)
    sweep_ns = [n for n in [1, 3, 5, 8, 11] if n <= max_n]
    if max_n not in sweep_ns:
        sweep_ns.append(max_n)

    results_by_n = {}

    for n_chains in sweep_ns:
        correct = 0
        total = 0

        for ep in range(n_eval):
            model.reset_traces()
            model.set_trace_mode(use=False, update=False)

            selected = random.sample(chain_entries, n_chains)

            # Write chain links: city -> country
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
        print(f"    N={n_chains:2d} chains: {acc:5.1f}%")

    return {"accuracy_by_n": results_by_n, "n_eval": n_eval}


# -- Phase 3: Multi-Hop Chains --

def run_phase3_multihop(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    chain_entries: list[ChainEntry],
    n_eval: int = 50,
    n_extra_facts: int = 3,
) -> dict:
    """Multi-hop: write person's city + city->country, query end-to-end.

    For each episode:
    1. Pick a random city->country chain
    2. Write: concept("live") -> city_token  (person's city)
    3. Write: city_token -> country_token    (chain link)
    4. Optionally write N extra standard facts (interference)
    5. Measure: hop-1, hop-2|oracle, hop-2|predicted, end-to-end
    """
    print(f"\n{'='*60}")
    print(f"Phase 3: Multi-hop chains")
    print(f"  n_eval={n_eval}, n_extra_facts={n_extra_facts}")
    print(f"{'='*60}")

    city_entry = concept_vocab.get("city")
    if city_entry is None:
        print("  ERROR: 'city' type not in concept_vocab!")
        return {}

    all_city_ids = sorted({eid for _, eid in city_entry.entity_pool})
    all_country_ids = sorted({e.country_token_id for e in chain_entries})

    # Extra fact types for interference (exclude city/country)
    extra_types = [
        tn for tn in concept_vocab
        if tn not in ("city", "country")
    ][:n_extra_facts]

    for n_extra in sorted(set([0, n_extra_facts])):
        hop1_direct_correct = 0
        hop1_question_correct = 0
        hop2_oracle_correct = 0
        hop2_pred_correct = 0
        end_to_end_correct = 0
        total = 0

        et = extra_types[:n_extra]

        for ep in range(n_eval):
            model.reset_traces()
            model.set_trace_mode(use=False, update=False)

            chain = random.choice(chain_entries)

            # Write person's city: concept("live") -> city
            model.write_fact_direct(
                city_entry.concept_token_id, chain.city_token_id)

            # Write chain link: city -> country
            model.write_fact_direct(
                chain.city_token_id, chain.country_token_id)

            # Write extra facts for interference
            for tn in et:
                entry = concept_vocab[tn]
                _, eid = random.choice(entry.entity_pool)
                model.write_fact_direct(entry.concept_token_id, eid)

            model.set_trace_mode(use=True, update=False)

            # Hop-1: get person's city via retrieve_direct
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

            # Hop-2 given oracle intermediate (correct city)
            hop2_oracle = model.retrieve_direct(
                chain.city_token_id, all_country_ids)
            if hop2_oracle == chain.country_token_id:
                hop2_oracle_correct += 1

            # Hop-2 given predicted intermediate (chained)
            hop2_pred = model.retrieve_direct(
                hop1_pred, all_country_ids)
            if hop2_pred == chain.country_token_id:
                hop2_pred_correct += 1

            # End-to-end: both hops correct
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
        "hop1_direct": h1d, "hop1_question": h1q,
        "hop2_oracle": h2o, "hop2_predicted": h2p,
        "end_to_end": e2e, "n_eval": n_eval,
        "n_extra_facts": n_extra_facts,
    }


# -- Phase 4: Multi-Chain Capacity --

def run_phase4_capacity(
    model: GPT2WithTrace,
    concept_vocab: dict[str, ConceptEntry],
    chain_entries: list[ChainEntry],
    n_eval: int = 50,
    n_extra_facts: int = 3,
) -> dict:
    """Multi-chain capacity: N city->country links + 1 person->city.

    Person's city is one of the N stored chains. Multi-hop must:
    1. Get person's city (hop-1)
    2. Find THAT city's country among N stored chains (hop-2)

    Sweeps N = 1, 3, 5, 8, 11 chains.
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

    print(f"\n  {'N':>3}  {'hop-1':>7}  {'h2|orc':>7}  {'h2|pred':>7}  {'e2e':>7}")
    print(f"  {'-'*38}")

    for n_chains in sweep_ns:
        hop1_correct = 0
        hop2_oracle_correct = 0
        hop2_pred_correct = 0
        end_to_end_correct = 0
        total = 0

        for ep in range(n_eval):
            model.reset_traces()
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

            # Write extra facts for interference
            for tn in extra_types:
                entry = concept_vocab[tn]
                _, eid = random.choice(entry.entity_pool)
                model.write_fact_direct(entry.concept_token_id, eid)

            model.set_trace_mode(use=True, update=False)

            # Hop-1: get person's city
            hop1_pred = model.retrieve_direct(
                city_entry.concept_token_id, all_city_ids)
            hop1_ok = (hop1_pred == target_chain.city_token_id)
            if hop1_ok:
                hop1_correct += 1

            # Hop-2|oracle
            hop2_oracle = model.retrieve_direct(
                target_chain.city_token_id, all_country_ids)
            if hop2_oracle == target_chain.country_token_id:
                hop2_oracle_correct += 1

            # Hop-2|predicted (chained)
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
        print(f"  {n_chains:>3}  {h1:6.1f}%  {h2o:6.1f}%  {h2p:6.1f}%  {e2e:6.1f}%")

    return {"results_by_n": results, "n_eval": n_eval}


# -- Main --

def run_experiment(
    n_eval: int = 50,
    phase: str = "all",
    n_extra_facts: int = 3,
    quick: bool = False,
    weights_path: str = "weights/trace_module.pt",
):
    if quick:
        n_eval = min(n_eval, 10)

    model, tokenizer = setup_model(weights_path)
    concept_vocab = build_concept_vocab(tokenizer)
    chain_entries = build_chain_entries(tokenizer)

    print()
    print("=" * 60)
    print("  Multi-Hop Reasoning via Trace Chains")
    print("=" * 60)
    print(f"  Concept types:   {len(concept_vocab)}")
    print(f"  Chain pairs:     {len(chain_entries)}")
    print(f"  Episodes:        {n_eval}")
    print(f"  Phase:           {phase}")
    print()

    random.seed(42)

    if phase in ("all", "validate"):
        run_phase1_validate(model, tokenizer, concept_vocab, n_eval)

    if phase in ("all", "chains"):
        run_phase2_chains(model, chain_entries, n_eval)

    if phase in ("all", "multihop"):
        run_phase3_multihop(
            model, tokenizer, concept_vocab, chain_entries,
            n_eval, n_extra_facts)

    if phase in ("all", "capacity"):
        run_phase4_capacity(
            model, concept_vocab, chain_entries,
            n_eval, n_extra_facts)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-hop reasoning via Hebbian trace chains")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (10 episodes)")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--phase",
                        choices=["validate", "chains", "multihop",
                                 "capacity", "all"],
                        default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--n-extra-facts", type=int, default=3,
                        help="Extra facts for interference (default: 3)")
    parser.add_argument("--weights", type=str,
                        default="weights/trace_module.pt",
                        help="Path to trace module weights")
    args = parser.parse_args()

    run_experiment(
        n_eval=args.n_eval,
        phase=args.phase,
        n_extra_facts=args.n_extra_facts,
        quick=args.quick,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
