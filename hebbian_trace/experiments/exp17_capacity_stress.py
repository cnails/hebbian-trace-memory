#!/usr/bin/env python3
"""Experiment 17: Capacity stress test.

How many unique facts can the Hebbian trace hold before accuracy degrades?
Auto-discovers concept words beyond the 24 standard types to test up to n=100.

Compares: with vs without pattern separation.

Usage:
    python -m hebbian_trace.experiments.exp17_capacity_stress --quick
    python -m hebbian_trace.experiments.exp17_capacity_stress --n-eval 50
    python -m hebbian_trace.experiments.exp17_capacity_stress --n-eval 50 --no-ps
"""

import argparse
import json
import math
import os
import random
import time

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.gpt2_trace import GPT2WithTrace
from hebbian_trace.gpt2_tasks import (
    build_fact_types,
    get_linking_bpe_ids,
    _get_all_entity_ids,
    validate_single_token_entities,
    GPT2FactType,
    GPT2FactTemplate,
    GPT2QuestionTemplate,
    tokenize_fact,
    tokenize_question,
    _predict_answer,
)
from hebbian_trace.experiments.exp16_multi_session import (
    build_extended_fact_types,
    EXTRA_POOLS,
)


# -- Additional concept words for stress testing beyond 24 types --

STRESS_CONCEPT_CANDIDATES = [
    # Nature
    "bird", "fish", "insect", "planet", "star",
    "river", "lake", "ocean", "cliff", "cave",
    "desert", "island", "valley", "reef", "marsh",
    # Objects
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
    # Food/drink
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
    # More
    "lamp", "mirror", "bell", "wheel", "lock",
    "clock", "chain", "bolt", "nail", "pin",
    "rank", "grade", "score", "prize", "charm",
    "curse", "spell", "omen", "fate", "myth",
]


def _get_entity_names(fact_types: list[GPT2FactType]) -> set[str]:
    """Collect all entity names from existing types."""
    names = set()
    for ft in fact_types:
        for ent_name, _ in ft.entities:
            names.add(ent_name.lower())
    return names


def _get_shared_entity_pool(
    fact_types: list[GPT2FactType],
) -> list[tuple[str, int]]:
    """Merge all entity pools, deduplicated by BPE ID."""
    seen: set[int] = set()
    pool: list[tuple[str, int]] = []
    for ft in fact_types:
        for ent_name, ent_id in ft.entities:
            if ent_id not in seen:
                pool.append((ent_name, ent_id))
                seen.add(ent_id)
    return pool


def build_stress_fact_types(
    tokenizer: GPT2Tokenizer,
    n_target: int,
    verbose: bool = True,
) -> list[GPT2FactType]:
    """Build up to n_target unique fact types.

    Uses 24 standard types + auto-discovered synthetic types.
    """
    base_types = build_extended_fact_types(tokenizer, verbose=verbose)

    if n_target <= len(base_types):
        return base_types

    existing_concepts = {ft.name for ft in base_types}
    existing_entities = _get_entity_names(base_types)
    shared_pool = _get_shared_entity_pool(base_types)

    extra_types: list[GPT2FactType] = []
    for concept in STRESS_CONCEPT_CANDIDATES:
        if len(base_types) + len(extra_types) >= n_target:
            break
        if concept.lower() in existing_concepts:
            continue
        if concept.lower() in existing_entities:
            continue

        ids = tokenizer.encode(" " + concept, add_special_tokens=False)
        if len(ids) != 1:
            continue

        extra_types.append(GPT2FactType(
            name=concept,
            entities=shared_pool,
            fact_templates=[GPT2FactTemplate(f"My {concept} is {{X}}.", "is")],
            question_templates=[
                GPT2QuestionTemplate(f"What is my {concept}?")],
        ))
        existing_concepts.add(concept.lower())

    all_types = base_types + extra_types
    if verbose:
        print(f"Stress types: {len(base_types)} standard + "
              f"{len(extra_types)} synthetic = {len(all_types)} total")
    return all_types


# -- Bootstrap CI --

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    if n <= 1:
        m = sum(values) / max(n, 1)
        return m, m

    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


# -- Core evaluation --

def evaluate_capacity(
    model,
    n_facts: int,
    fact_types: list[GPT2FactType],
    entity_ids: list[int],
    tokenizer: GPT2Tokenizer,
    n_episodes: int = 50,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Cross-context accuracy at n_facts unique facts.

    Individual write per fact (ACh-modulated), question-only retrieval.
    """
    device = next(model.parameters()).device
    model.eval()

    total_correct = 0
    total_queries = 0
    per_episode: list[float] = []

    for ep in range(n_episodes):
        ep_rng = random.Random(seed + ep)
        model.reset_traces()

        selected = ep_rng.sample(fact_types, n_facts)
        facts = []
        for ft in selected:
            ent_name, ent_id = ep_rng.choice(ft.entities)
            fact_ids = tokenize_fact(tokenizer, ft.fact_templates[0], ent_name)
            q_ids = tokenize_question(tokenizer, ft.question_templates[0])
            facts.append((fact_ids, q_ids, ent_id))

        # Write phase
        model.set_trace_mode(use=False, update=True)
        for fact_ids, _, _ in facts:
            inp = torch.tensor([fact_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(inp)

        # Read phase
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for _, q_ids, answer_id in facts:
            pred_id = _predict_answer(model, q_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        per_episode.append(ep_correct / max(len(facts), 1))

    return total_correct / max(total_queries, 1), per_episode


# -- Main --

def run_stress_test(
    n_eval: int = 20,
    test_no_ps: bool = False,
    seed: int = 42,
):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 80)
    print("  Exp 17: Capacity Stress Test")
    print("=" * 80)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    linking_ids = get_linking_bpe_ids(tokenizer)

    all_types = build_stress_fact_types(tokenizer, 110)
    entity_ids = _get_all_entity_ids(all_types)
    print(f"  Entity vocabulary: {len(entity_ids)} unique BPE tokens")
    print()

    max_n = len(all_types)
    n_facts_list = [n for n in
                    [1, 3, 5, 7, 10, 15, 20, 24, 30, 40, 50, 60, 75, 100]
                    if n <= max_n]

    # -- With pattern separation --
    print("Loading GPT-2 + trace (PS 8x_k16, alpha=0.5)...")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=0.5,
        trace_lr=1.0, trace_decay=0.99, device=device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
    print()

    results_ps: dict[int, dict] = {}

    print("-" * 80)
    print(f"  {'n':>5}  {'Accuracy':>10}  {'95% CI':>18}  "
          f"{'Std':>8}  {'Time':>6}")
    print("-" * 80)

    for n_facts in n_facts_list:
        t0 = time.time()
        acc, per_ep = evaluate_capacity(
            model, n_facts, all_types[:max(n_facts, 24)],
            entity_ids, tokenizer, n_eval, seed)
        dt = time.time() - t0

        ci_lo, ci_hi = bootstrap_ci(per_ep)
        std = (sum((x - acc)**2 for x in per_ep) / max(len(per_ep)-1, 1))**0.5

        results_ps[n_facts] = dict(
            accuracy=acc, ci_lo=ci_lo, ci_hi=ci_hi,
            std=std, per_episode=per_ep)

        print(f"  {n_facts:>5}  {acc:>9.1%}  "
              f"[{ci_lo:>6.1%}, {ci_hi:>6.1%}]  "
              f"{std:>7.1%}  {dt:>5.0f}s")

    print("-" * 80)
    print()

    # -- Without pattern separation --
    results_no_ps: dict[int, dict] = {}

    if test_no_ps:
        print("Running WITHOUT pattern separation...")
        model.trace.disable_pattern_separation()
        print()

        print("-" * 80)
        print(f"  {'n':>5}  {'With PS':>10}  {'No PS':>10}  {'Delta':>8}")
        print("-" * 80)

        for n_facts in n_facts_list:
            acc, per_ep = evaluate_capacity(
                model, n_facts, all_types[:max(n_facts, 24)],
                entity_ids, tokenizer, n_eval, seed)

            ci_lo, ci_hi = bootstrap_ci(per_ep)
            std = (sum((x-acc)**2 for x in per_ep)/max(len(per_ep)-1,1))**0.5
            results_no_ps[n_facts] = dict(
                accuracy=acc, ci_lo=ci_lo, ci_hi=ci_hi,
                std=std, per_episode=per_ep)

            ps_acc = results_ps[n_facts]["accuracy"]
            delta = ps_acc - acc
            print(f"  {n_facts:>5}  {ps_acc:>9.1%}  {acc:>9.1%}  "
                  f"{'+' if delta > 0 else ''}{delta:>6.1%}")

        print("-" * 80)
        print()

    # -- Save --
    output = {
        "config": dict(n_eval=n_eval, seed=seed,
                        n_types=len(all_types), n_entities=len(entity_ids)),
        "with_ps": {str(k): {kk: vv for kk, vv in v.items()
                              if kk != "per_episode"}
                    for k, v in results_ps.items()},
    }
    if results_no_ps:
        output["without_ps"] = {
            str(k): {kk: vv for kk, vv in v.items() if kk != "per_episode"}
            for k, v in results_no_ps.items()
        }

    os.makedirs("results", exist_ok=True)
    path = "results/exp17_capacity.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")

    # -- Figure --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ns = sorted(results_ps.keys())
        accs = [results_ps[n]["accuracy"] for n in ns]
        ci_lo = [results_ps[n]["ci_lo"] for n in ns]
        ci_hi = [results_ps[n]["ci_hi"] for n in ns]

        ax.plot(ns, accs, 'o-', color='#2563eb', linewidth=2.5,
                markersize=8, label='With PS (8x, k=16)', zorder=3)
        ax.fill_between(ns, ci_lo, ci_hi, alpha=0.15, color='#2563eb')

        if results_no_ps:
            ns2 = sorted(results_no_ps.keys())
            accs2 = [results_no_ps[n]["accuracy"] for n in ns2]
            ci2_lo = [results_no_ps[n]["ci_lo"] for n in ns2]
            ci2_hi = [results_no_ps[n]["ci_hi"] for n in ns2]
            ax.plot(ns2, accs2, 's--', color='#f59e0b', linewidth=2,
                    markersize=7, label='Without PS', zorder=2)
            ax.fill_between(ns2, ci2_lo, ci2_hi, alpha=0.12, color='#f59e0b')

        ax.axhline(y=1/len(entity_ids), color='#9ca3af', linestyle=':',
                   linewidth=1, label=f'Random ({1/len(entity_ids):.1%})')

        # Mark capacity milestones
        for threshold, label in [(0.95, '95%'), (0.80, '80%'), (0.50, '50%')]:
            for i in range(len(ns) - 1):
                if accs[i] >= threshold and accs[i+1] < threshold:
                    # Interpolate
                    frac = (threshold - accs[i+1]) / (accs[i] - accs[i+1])
                    n_thresh = ns[i+1] - frac * (ns[i+1] - ns[i])
                    ax.axvline(x=n_thresh, color='#e5e7eb', linewidth=1,
                               linestyle='--', zorder=0)
                    ax.text(n_thresh + 1, threshold + 0.03,
                            f'{label} @ n≈{n_thresh:.0f}',
                            fontsize=9, color='#6b7280')
                    break

        ax.set_xlabel('Number of Facts Stored', fontsize=13)
        ax.set_ylabel('Cross-Context Retrieval Accuracy', fontsize=13)
        ax.set_title('Capacity Stress Test: Trace Degradation Curve',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(-0.02, 1.08)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = "results/exp17_capacity_curve.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not found, skipping figure")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 17: capacity stress test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 episodes)")
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--no-ps", action="store_true",
                        help="Also test without pattern separation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_eval = args.n_eval or (20 if args.quick else 50)

    run_stress_test(n_eval=n_eval, test_no_ps=args.no_ps, seed=args.seed)
