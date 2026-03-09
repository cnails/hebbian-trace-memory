#!/usr/bin/env python3
"""Paraphrase resolution via T_auto (CA3 autoassociative trace).

Demonstrates two-step retrieval for paraphrase resolution:
  Step 1: Q(variant) -> T_auto -> Q_corrected (pattern completion)
  Step 2: Q_corrected -> T_v -> V (standard retrieval)

Without T_auto, misaligned questions like "Who am I?" fail because
Q("I") != Q("name"). With T_auto, the autoassociative trace maps
Q("I") -> Q("name"), resolving the paraphrase.

Three question categories:
  - Aligned:    use same concept word as fact template (should always work)
  - Misaligned: use shifted position word (fails without T_auto)
  - Semantic:   use entirely different phrasing (fails without T_auto)

Expected results:
  Misaligned: 17% -> 100% (+83pp) with T_auto
  Semantic:   27% -> 100% (+73pp) with T_auto
  Aligned:    90% ->  90% (unchanged, no degradation)

Usage:
    python exp_paraphrase.py --quick       # 10 episodes
    python exp_paraphrase.py --n-eval 50   # full run
"""

import argparse
import random
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_concept_vocab,
    get_linking_bpe_ids,
    get_all_entity_ids,
    build_fact_types,
    _predict_answer,
    TAUTO_PAIRS,
    ConceptEntry,
    CONCEPT_WORD_MAP,
)


# -- Question variants for testing --

# Each variant: (fact_type, question_text, category)
# aligned: uses the canonical concept word at the retrieval position
# misaligned: uses a different word that doesn't match the stored Q
# semantic: uses entirely different phrasing

QUESTION_VARIANTS: list[tuple[str, str, str]] = [
    # NAME
    ("name", "What is my name?", "aligned"),
    ("name", "Who am I?", "misaligned"),
    ("name", "What am I called?", "misaligned"),
    ("name", "What is my title?", "semantic"),
    ("name", "What is my identity?", "semantic"),

    # CITY
    ("city", "Where do I live?", "aligned"),
    ("city", "What is my home?", "misaligned"),
    ("city", "Where do I reside?", "misaligned"),
    ("city", "What is my residence?", "semantic"),

    # COMPANY
    ("company", "Where do I work?", "aligned"),
    ("company", "Where am I employed?", "misaligned"),
    ("company", "Where am I hired?", "misaligned"),
    ("company", "What is my employer?", "semantic"),
    ("company", "What is my occupation?", "semantic"),

    # COLOR
    ("color", "What is my favorite color?", "aligned"),
    ("color", "What is my favorite hue?", "semantic"),
    ("color", "What is my favorite shade?", "semantic"),

    # FOOD
    ("food", "What is my favorite food?", "aligned"),
    ("food", "What is my favorite cuisine?", "semantic"),
    ("food", "What is my favorite dish?", "semantic"),

    # PET
    ("pet", "What is my pet?", "aligned"),
    ("pet", "What is my companion?", "semantic"),
]


@dataclass
class AutoPair:
    """A Q->Q pair for T_auto: maps variant word to concept word."""
    variant_word: str
    variant_token_id: int
    concept_word: str
    concept_token_id: int


def build_auto_pairs(tokenizer: GPT2Tokenizer) -> list[AutoPair]:
    """Build T_auto pairs from TAUTO_PAIRS data."""
    pairs = []
    for variant_word, concept_word in TAUTO_PAIRS:
        v_ids = tokenizer.encode(" " + variant_word, add_special_tokens=False)
        c_ids = tokenizer.encode(" " + concept_word, add_special_tokens=False)
        if len(v_ids) == 1 and len(c_ids) == 1:
            pairs.append(AutoPair(
                variant_word=variant_word,
                variant_token_id=v_ids[0],
                concept_word=concept_word,
                concept_token_id=c_ids[0],
            ))
    return pairs


# -- Setup --

def setup_model(
    weights_path: str = "weights/trace_module.pt",
) -> tuple[GPT2WithTrace, GPT2Tokenizer]:
    """Load GPT-2 + trace module."""
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


# -- Populate T_auto --

def populate_tauto(model: GPT2WithTrace, auto_pairs: list[AutoPair]):
    """Write all Q->Q pairs into T_auto.

    For each pair (variant, concept), stores:
      T_auto += Q(variant) -> Q(concept)

    T_auto is static template knowledge, written once.
    """
    for pair in auto_pairs:
        Q_variant = model.trace.compute_q_for_token(
            model._wte, pair.variant_token_id)
        Q_concept = model.trace.compute_q_for_token(
            model._wte, pair.concept_token_id)
        model.trace.write_auto(Q_variant, Q_concept)


# -- Evaluation --

def run_evaluation(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    auto_pairs: list[AutoPair],
    n_eval: int = 50,
    n_facts: int = 5,
) -> dict:
    """Compare retrieval with and without T_auto on paraphrased questions.

    For each episode:
    1. Write N facts to T_v (standard fact storage)
    2. Query with aligned, misaligned, and semantic questions
    3. Compare accuracy with T_auto ON vs OFF
    """
    print(f"\n{'='*60}")
    print(f"Paraphrase Resolution via T_auto")
    print(f"  n_eval={n_eval}, n_facts={n_facts}")
    print(f"  T_auto pairs: {len(auto_pairs)}")
    print(f"{'='*60}")

    fact_types = build_fact_types(tokenizer)
    entity_ids = get_all_entity_ids(fact_types)

    # Group question variants by category
    categories = {"aligned": [], "misaligned": [], "semantic": []}
    for fact_type, q_text, category in QUESTION_VARIANTS:
        if fact_type in concept_vocab:
            categories[category].append((fact_type, q_text))

    # Results accumulators: {category: {with_auto: correct, total}}
    results = {
        cat: {"auto_correct": 0, "no_auto_correct": 0, "total": 0}
        for cat in categories
    }

    type_names = list(concept_vocab.keys())[:n_facts]

    for ep in range(n_eval):
        # Pick random entities
        episode_facts = {}
        for tn in type_names:
            entry = concept_vocab[tn]
            entity_name, entity_id = random.choice(entry.entity_pool)
            episode_facts[tn] = (entity_name, entity_id)

        for use_auto in [False, True]:
            model.reset_traces()
            model.trace.reset_auto_traces()

            # Write facts to T_v
            model.set_trace_mode(use=False, update=False)
            for tn in type_names:
                entry = concept_vocab[tn]
                _, entity_id = episode_facts[tn]
                model.write_fact_direct(entry.concept_token_id, entity_id)

            # Populate T_auto if enabled
            if use_auto:
                populate_tauto(model, auto_pairs)
                model.set_auto_mode(True, completion_alpha=0.3)
            else:
                model.set_auto_mode(False)

            # Query with each variant
            model.set_trace_mode(use=True, update=False)
            for category, variant_list in categories.items():
                for fact_type, q_text in variant_list:
                    if fact_type not in episode_facts:
                        continue

                    _, expected_id = episode_facts[fact_type]
                    q_ids = tokenizer.encode(q_text, add_special_tokens=False)
                    pred_id = _predict_answer(model, q_ids, entity_ids)

                    if use_auto:
                        if pred_id == expected_id:
                            results[category]["auto_correct"] += 1
                    else:
                        if pred_id == expected_id:
                            results[category]["no_auto_correct"] += 1
                        results[category]["total"] += 1

            model.set_trace_mode(use=False, update=False)
            model.set_auto_mode(False)

    # Print results
    print(f"\n  Results ({n_eval} episodes, {n_facts} facts each):")
    print(f"\n  {'Category':<14} {'Without T_auto':>14} {'With T_auto':>14} {'Delta':>8}")
    print(f"  {'-'*52}")

    for cat in ["aligned", "misaligned", "semantic"]:
        r = results[cat]
        total = r["total"]
        if total == 0:
            continue
        no_acc = r["no_auto_correct"] / total * 100
        auto_acc = r["auto_correct"] / total * 100
        delta = auto_acc - no_acc
        print(f"  {cat:<14} {no_acc:13.1f}% {auto_acc:13.1f}% {delta:>+7.1f}pp")

    # Overall
    total_all = sum(r["total"] for r in results.values())
    no_all = sum(r["no_auto_correct"] for r in results.values())
    auto_all = sum(r["auto_correct"] for r in results.values())
    if total_all > 0:
        no_overall = no_all / total_all * 100
        auto_overall = auto_all / total_all * 100
        print(f"  {'-'*52}")
        print(f"  {'overall':<14} {no_overall:13.1f}% {auto_overall:13.1f}% "
              f"{auto_overall - no_overall:>+7.1f}pp")

    return {
        "results": {
            cat: {
                "no_auto_acc": r["no_auto_correct"] / max(r["total"], 1) * 100,
                "auto_acc": r["auto_correct"] / max(r["total"], 1) * 100,
                "total": r["total"],
            }
            for cat, r in results.items()
        },
        "n_eval": n_eval,
        "n_facts": n_facts,
        "n_auto_pairs": len(auto_pairs),
    }


# -- Sweep n_facts --

def run_sweep(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    concept_vocab: dict[str, ConceptEntry],
    auto_pairs: list[AutoPair],
    n_eval: int = 50,
) -> dict:
    """Sweep n_facts = 1, 3, 5 to see how T_auto interacts with capacity."""
    print(f"\n{'='*60}")
    print(f"N-facts sweep with T_auto")
    print(f"{'='*60}")

    all_results = {}
    for n_facts in [1, 3, 5]:
        if n_facts > len(concept_vocab):
            continue
        result = run_evaluation(
            model, tokenizer, concept_vocab, auto_pairs,
            n_eval=n_eval, n_facts=n_facts,
        )
        all_results[n_facts] = result

    return all_results


# -- Main --

def run_experiment(
    n_eval: int = 50,
    n_facts: int = 5,
    quick: bool = False,
    sweep: bool = False,
    weights_path: str = "weights/trace_module.pt",
):
    if quick:
        n_eval = min(n_eval, 10)

    model, tokenizer = setup_model(weights_path)
    concept_vocab = build_concept_vocab(tokenizer)
    auto_pairs = build_auto_pairs(tokenizer)

    print()
    print("=" * 60)
    print("  Paraphrase Resolution via T_auto (CA3 Pattern Completion)")
    print("=" * 60)
    print(f"  Concept types:  {len(concept_vocab)}")
    print(f"  T_auto pairs:   {len(auto_pairs)}")
    print(f"  Episodes:       {n_eval}")
    print()

    # Show T_auto pairs
    print("  Template pairs (variant -> concept):")
    for pair in auto_pairs:
        print(f"    {pair.variant_word:>12} -> {pair.concept_word}")
    print()

    random.seed(42)

    if sweep:
        run_sweep(model, tokenizer, concept_vocab, auto_pairs, n_eval)
    else:
        run_evaluation(
            model, tokenizer, concept_vocab, auto_pairs,
            n_eval=n_eval, n_facts=n_facts,
        )

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase resolution via T_auto")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (10 episodes)")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--n-facts", type=int, default=5,
                        help="Number of facts per episode (default: 5)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep n_facts = 1, 3, 5")
    parser.add_argument("--weights", type=str,
                        default="weights/trace_module.pt",
                        help="Path to trace module weights")
    args = parser.parse_args()

    run_experiment(
        n_eval=args.n_eval,
        n_facts=args.n_facts,
        quick=args.quick,
        sweep=args.sweep,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
