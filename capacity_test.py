#!/usr/bin/env python3
"""Capacity stress test: how many unique facts can the trace hold?

Measures cross-context retrieval accuracy as the number of stored
facts increases from 1 to 100. Uses auto-discovered concept words
beyond the 24 standard fact types to test trace matrix capacity.

Key question: where does pattern separation hit its ceiling?

Usage:
    python capacity_test.py                # quick (20 episodes, ~10 min)
    python capacity_test.py --n-eval 50    # full (~30 min)
    python capacity_test.py --no-ps        # without pattern separation
"""

import argparse
import json
import math
import os
import random
import time

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_extended_fact_types,
    get_linking_bpe_ids,
    get_all_entity_ids,
    validate_single_token_entities,
    FactType,
    FactTemplate,
    QuestionTemplate,
    tokenize_fact,
    tokenize_question,
    _predict_answer,
)


# -- Additional concept words for stress testing --
# Each must be a single BPE token and work in "My {X} is {entity}."

STRESS_CONCEPT_CANDIDATES = [
    # Nature
    "bird", "fish", "insect", "planet", "star",
    "river", "lake", "ocean", "cliff", "cave",
    "desert", "island", "valley", "reef", "marsh",
    # Objects & possessions
    "ring", "crown", "mask", "coin", "flag",
    "hat", "belt", "scarf", "sword", "shield",
    "arrow", "blade", "torch", "horn", "drum",
    # Abstract
    "talent", "fear", "dream", "goal", "rule",
    "method", "path", "code", "mode", "style",
    "skill", "craft", "quest", "task", "trick",
    # Entertainment
    "movie", "book", "song", "game", "show",
    "dance", "joke", "puzzle", "toy", "card",
    # Food/drink variants
    "cheese", "spice", "herb", "sauce", "snack",
    "grain", "nut", "bean", "seed", "root",
    # Materials
    "wood", "glass", "clay", "wax", "brick",
    "rope", "wire", "tape", "foam", "dust",
    # Places
    "port", "camp", "den", "nest", "tower",
    "fort", "bridge", "gate", "wall", "roof",
    # Groups
    "tribe", "clan", "crew", "team", "band",
    # More objects
    "lamp", "mirror", "bell", "wheel", "lock",
    "clock", "chain", "bolt", "nail", "pin",
    # More abstract
    "rank", "grade", "score", "prize", "charm",
    "curse", "spell", "omen", "fate", "myth",
]


def _get_entity_names(fact_types: list[FactType]) -> set[str]:
    """Collect all entity names (lowercase) from existing types."""
    names = set()
    for ft in fact_types:
        for ent_name, _ in ft.entities:
            names.add(ent_name.lower())
    return names


def _get_shared_entity_pool(fact_types: list[FactType]) -> list[tuple[str, int]]:
    """Merge all entity pools into one shared pool (deduplicated by BPE ID)."""
    seen_ids: set[int] = set()
    pool: list[tuple[str, int]] = []
    for ft in fact_types:
        for ent_name, ent_id in ft.entities:
            if ent_id not in seen_ids:
                pool.append((ent_name, ent_id))
                seen_ids.add(ent_id)
    return pool


def build_stress_fact_types(
    tokenizer: GPT2Tokenizer,
    n_target: int,
) -> list[FactType]:
    """Build up to n_target unique fact types.

    Starts with 24 standard types, then auto-discovers additional
    concept words from STRESS_CONCEPT_CANDIDATES.
    """
    base_types = build_extended_fact_types(tokenizer)

    if n_target <= len(base_types):
        return base_types

    # Collect existing names to avoid overlap
    existing_concepts = {ft.name for ft in base_types}
    existing_entities = _get_entity_names(base_types)
    shared_pool = _get_shared_entity_pool(base_types)

    # Discover valid concept words
    extra_types: list[FactType] = []
    for concept in STRESS_CONCEPT_CANDIDATES:
        if len(base_types) + len(extra_types) >= n_target:
            break
        if concept.lower() in existing_concepts:
            continue
        if concept.lower() in existing_entities:
            continue

        # Must be single BPE token
        ids = tokenizer.encode(" " + concept, add_special_tokens=False)
        if len(ids) != 1:
            continue

        extra_types.append(FactType(
            name=concept,
            entities=shared_pool,
            fact_templates=[FactTemplate(f"My {concept} is {{X}}.", "is")],
            question_templates=[QuestionTemplate(f"What is my {concept}?")],
        ))
        existing_concepts.add(concept.lower())

    all_types = base_types + extra_types
    return all_types


# -- Bootstrap CI --

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    if n <= 1:
        mean = sum(values) / max(n, 1)
        return mean, mean

    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap)
    return means[lo_idx], means[hi_idx]


# -- Core evaluation --

def evaluate_capacity(
    model,
    n_facts: int,
    fact_types: list[FactType],
    entity_ids: list[int],
    tokenizer: GPT2Tokenizer,
    n_episodes: int = 50,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Evaluate cross-context accuracy at exactly n_facts unique facts.

    Protocol:
        1. Reset traces
        2. Write phase: store n_facts individually (ACh-modulated)
        3. Read phase: query all n_facts, measure accuracy

    Returns:
        (mean_accuracy, per_episode_accuracies)
    """
    device = next(model.parameters()).device
    model.eval()

    total_correct = 0
    total_queries = 0
    per_episode: list[float] = []

    for ep in range(n_episodes):
        ep_rng = random.Random(seed + ep)
        model.reset_traces()

        # Sample n_facts types (each gets one random entity)
        selected_types = ep_rng.sample(fact_types, n_facts)

        facts = []
        for ft in selected_types:
            ent_name, ent_id = ep_rng.choice(ft.entities)
            template = ft.fact_templates[0]
            q_template = ft.question_templates[0]
            fact_ids = tokenize_fact(tokenizer, template, ent_name)
            q_ids = tokenize_question(tokenizer, q_template)
            facts.append((fact_ids, q_ids, ent_id))

        # Write phase: individual writes (strongest mode)
        model.set_trace_mode(use=False, update=True)
        for fact_ids, _, _ in facts:
            input_tensor = torch.tensor(
                [fact_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor)

        # Read phase: query all facts
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for _, q_ids, answer_id in facts:
            pred_id = _predict_answer(model, q_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        per_episode.append(ep_correct / max(len(facts), 1))

    accuracy = total_correct / max(total_queries, 1)
    return accuracy, per_episode


# -- Main --

def run_capacity_test(
    n_eval: int = 20,
    weights_path: str = "weights/trace_module.pt",
    test_no_ps: bool = False,
    n_banks: int = 1,
    seed: int = 42,
):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 80)
    print("  Capacity Stress Test")
    print("=" * 80)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print()

    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    linking_ids = get_linking_bpe_ids(tokenizer)

    # Discover how many types we can build
    max_target = 110
    all_types = build_stress_fact_types(tokenizer, max_target)
    print(f"  Discovered {len(all_types)} valid fact types "
          f"(24 standard + {len(all_types) - 24} synthetic)")
    print()

    # Build shared entity pool for predictions
    entity_ids = get_all_entity_ids(all_types)
    print(f"  Entity vocabulary: {len(entity_ids)} unique BPE tokens")
    print()

    # n_facts to test
    max_n = len(all_types)
    n_facts_list = [n for n in [1, 3, 5, 7, 10, 15, 20, 24, 30, 40, 50, 60, 75, 100]
                    if n <= max_n]

    # -- Run with pattern separation --
    print("Loading model (with pattern separation 8x_k16)...")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=0.5,
        trace_lr=1.0, trace_decay=0.99, device=device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.trace.load_state_dict(state, strict=False)
        print(f"  Loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"  Warning: {weights_path} not found, using random projections")

    if n_banks > 1:
        model.set_bank_mode(n_banks)
        print(f"  Hashed trace banks: {n_banks}")

    print()
    results_ps: dict[int, dict] = {}

    print("-" * 80)
    print(f"  {'n':>5}  {'Accuracy':>10}  {'95% CI':>18}  {'Std':>8}  {'Time':>6}")
    print("-" * 80)

    for n_facts in n_facts_list:
        t0 = time.time()
        acc, per_ep = evaluate_capacity(
            model, n_facts, all_types[:max(n_facts, 24)],
            entity_ids, tokenizer, n_eval, seed)
        dt = time.time() - t0

        ci_lo, ci_hi = bootstrap_ci(per_ep)
        std = (sum((x - acc) ** 2 for x in per_ep) / max(len(per_ep) - 1, 1)) ** 0.5

        results_ps[n_facts] = {
            "accuracy": acc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "std": std,
            "per_episode": per_ep,
        }

        print(f"  {n_facts:>5}  {acc:>9.1%}  [{ci_lo:>6.1%}, {ci_hi:>6.1%}]  "
              f"{std:>7.1%}  {dt:>5.0f}s")

    print("-" * 80)
    print()

    # -- Run without pattern separation (comparison) --
    results_no_ps: dict[int, dict] = {}

    if test_no_ps:
        print("Running WITHOUT pattern separation...")
        model.disable_pattern_separation()
        model.enable_pattern_separation(expand_factor=1, top_k=0, seed=0)
        # Actually just disable and use raw trace
        model.trace.disable_pattern_separation()

        print()
        print("-" * 80)
        print(f"  {'n':>5}  {'With PS':>10}  {'No PS':>10}  {'Delta':>8}")
        print("-" * 80)

        for n_facts in n_facts_list:
            t0 = time.time()
            acc, per_ep = evaluate_capacity(
                model, n_facts, all_types[:max(n_facts, 24)],
                entity_ids, tokenizer, n_eval, seed)

            ci_lo, ci_hi = bootstrap_ci(per_ep)
            std = (sum((x - acc) ** 2 for x in per_ep) / max(len(per_ep) - 1, 1)) ** 0.5

            results_no_ps[n_facts] = {
                "accuracy": acc,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "std": std,
                "per_episode": per_ep,
            }

            ps_acc = results_ps[n_facts]["accuracy"]
            delta = ps_acc - acc
            print(f"  {n_facts:>5}  {ps_acc:>9.1%}  {acc:>9.1%}  "
                  f"{'+' if delta > 0 else ''}{delta:>6.1%}")

        print("-" * 80)
        print()

    # -- Save results --
    output = {
        "config": {
            "n_eval": n_eval,
            "seed": seed,
            "n_types_available": len(all_types),
            "n_entity_ids": len(entity_ids),
            "alpha": 0.5,
            "trace_lr": 1.0,
            "trace_decay": 0.99,
            "d_trace": 64,
            "expand_factor": 8,
            "top_k": 16,
        },
        "with_ps": {
            str(k): {kk: vv for kk, vv in v.items() if kk != "per_episode"}
            for k, v in results_ps.items()
        },
    }
    if results_no_ps:
        output["without_ps"] = {
            str(k): {kk: vv for kk, vv in v.items() if kk != "per_episode"}
            for k, v in results_no_ps.items()
        }

    results_path = "capacity_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {results_path}")

    # -- Generate figure --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ns = sorted(results_ps.keys())
        accs = [results_ps[n]["accuracy"] for n in ns]
        ci_los = [results_ps[n]["ci_lo"] for n in ns]
        ci_his = [results_ps[n]["ci_hi"] for n in ns]

        ax.plot(ns, accs, 'o-', color='#2563eb', linewidth=2.5,
                markersize=8, label='With pattern separation (8x, k=16)', zorder=3)
        ax.fill_between(ns, ci_los, ci_his, alpha=0.15, color='#2563eb', zorder=2)

        if results_no_ps:
            ns2 = sorted(results_no_ps.keys())
            accs2 = [results_no_ps[n]["accuracy"] for n in ns2]
            ci_los2 = [results_no_ps[n]["ci_lo"] for n in ns2]
            ci_his2 = [results_no_ps[n]["ci_hi"] for n in ns2]

            ax.plot(ns2, accs2, 's--', color='#f59e0b', linewidth=2,
                    markersize=7, label='Without pattern separation', zorder=2)
            ax.fill_between(ns2, ci_los2, ci_his2, alpha=0.12,
                            color='#f59e0b', zorder=1)

        # Random baseline
        ax.axhline(y=1/len(entity_ids), color='#9ca3af', linestyle=':',
                   linewidth=1, label=f'Random ({1/len(entity_ids):.1%})')

        ax.set_xlabel('Number of Facts Stored', fontsize=13)
        ax.set_ylabel('Cross-Context Retrieval Accuracy', fontsize=13)
        ax.set_title('Capacity Stress Test: Accuracy vs Number of Stored Facts',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(-0.02, 1.08)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate key points
        if len(ns) > 0:
            last_n = ns[-1]
            last_acc = results_ps[last_n]["accuracy"]
            ax.annotate(f'{last_acc:.0%} at n={last_n}',
                        xy=(last_n, last_acc),
                        xytext=(last_n - 15, last_acc - 0.12),
                        fontsize=10, fontweight='bold', color='#2563eb',
                        arrowprops=dict(arrowstyle='->', color='#2563eb',
                                        lw=1.5))

        plt.tight_layout()
        fig_path = os.path.join("figures", "capacity_curve.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved to {fig_path}")

    except ImportError:
        print("  matplotlib not found, skipping figure generation")

    print()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Capacity stress test for Hebbian trace memory")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Episodes per n_facts point (default: 20)")
    parser.add_argument("--weights", type=str,
                        default="weights/trace_module.pt")
    parser.add_argument("--no-ps", action="store_true",
                        help="Also test without pattern separation")
    parser.add_argument("--banks", type=int, default=1,
                        help="Number of hashed trace banks (default: 1 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_capacity_test(
        n_eval=args.n_eval,
        weights_path=args.weights,
        test_no_ps=args.no_ps,
        n_banks=args.banks,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
