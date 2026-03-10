"""Exp 28: Hashed trace banks for capacity scaling.

Routes facts to separate trace matrices based on sparse Q activation pattern.
Each bank accumulates only ~N/n_banks facts → interference reduced by n_banks.
Decay is isolated per bank → signal preserved across writes.

Compares scaling curves: baseline (1 bank) vs hashed (16, 32 banks).
Uses write_fact_direct for clean per-fact storage (same as exp24-27).

Usage:
    python -m hebbian_trace.experiments.exp28_hashed_traces --quick
    python -m hebbian_trace.experiments.exp28_hashed_traces --n-eval 50
    python -m hebbian_trace.experiments.exp28_hashed_traces --n-eval 30 --banks 1 16 32 64
"""

import argparse
import json
import os
import random
import time

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_concept_vocab, ConceptEntry,
    _predict_answer, _get_all_entity_ids,
    GPT2FactType, tokenize_fact, tokenize_question,
)
from .exp24_free_text import setup_model
from .exp17_capacity_stress import (
    build_stress_fact_types, bootstrap_ci,
)


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_scaling(
    model: GPT2WithTrace,
    n_facts: int,
    fact_types: list[GPT2FactType],
    entity_ids: list[int],
    tokenizer: GPT2Tokenizer,
    n_episodes: int = 30,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Evaluate accuracy at n_facts using forward-pass write.

    Same protocol as exp17: individual write per fact via forward pass
    (ACh-modulated), question-only retrieval.

    Returns:
        (accuracy, per_episode_accuracies)
    """
    device = next(model.parameters()).device
    model.eval()

    total_correct = 0
    total_queries = 0
    per_episode: list[float] = []

    for ep in range(n_episodes):
        ep_rng = random.Random(seed + ep)
        model.reset_traces()

        selected = ep_rng.sample(fact_types, min(n_facts, len(fact_types)))
        # If n_facts > len(fact_types), sample with replacement for remainder
        if n_facts > len(fact_types):
            selected += [ep_rng.choice(fact_types)
                         for _ in range(n_facts - len(fact_types))]

        facts = []
        for ft in selected:
            ent_name, ent_id = ep_rng.choice(ft.entities)
            fact_ids = tokenize_fact(tokenizer, ft.fact_templates[0], ent_name)
            q_ids = tokenize_question(tokenizer, ft.question_templates[0])
            facts.append((fact_ids, q_ids, ent_id))

        # Write phase (individual forward pass per fact)
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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Bank distribution diagnostic ────────────────────────────────────

def diagnose_bank_distribution(
    model: GPT2WithTrace,
    concept_vocab: dict[str, ConceptEntry],
    n_banks: int,
):
    """Show how concept tokens distribute across banks."""
    from collections import Counter

    model.trace.set_bank_mode(n_banks)
    model.reset_traces()

    bank_counts = Counter()
    for name, entry in concept_vocab.items():
        if not entry.entity_pool:
            continue
        Q = model.trace.compute_q_for_token(model._wte, entry.concept_token_id)
        if model.trace._pattern_sep_enabled:
            Q_exp = model.trace._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1)
        else:
            Q_exp = Q
        bid = model.trace._compute_bank_id(Q_exp)
        bank_counts[bid] += 1

    n_concepts = sum(bank_counts.values())
    print(f"\n  Bank distribution ({n_concepts} concepts, {n_banks} banks):")
    max_per_bank = max(bank_counts.values()) if bank_counts else 0
    min_per_bank = min(bank_counts.values()) if bank_counts else 0
    print(f"    Min/Max per bank: {min_per_bank}/{max_per_bank}")
    print(f"    Ideal: {n_concepts/n_banks:.1f} per bank")

    # Check for empty banks
    empty = n_banks - len(bank_counts)
    if empty > 0:
        print(f"    WARNING: {empty} empty banks (uneven distribution)")

    return bank_counts


# ── Main experiment ─────────────────────────────────────────────────

def run_experiment(
    n_eval: int = 20,
    bank_configs: list[int] | None = None,
    n_facts_list: list[int] | None = None,
    seed: int = 42,
):
    if bank_configs is None:
        bank_configs = [1, 16, 32]
    if n_facts_list is None:
        n_facts_list = [1, 3, 5, 7, 10, 15, 20, 24, 30, 40, 50, 60, 75, 100]

    device = get_device()

    print("=" * 80)
    print("  Exp 28: Hashed Trace Banks — Capacity Scaling")
    print("=" * 80)
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print(f"  Banks:     {bank_configs}")
    print()

    # Setup model
    print("Loading GPT-2 + trace (PS 8x_k16, alpha=0.5)...")
    model, tokenizer = setup_model(alpha=0.5, use_ps=True, device=device)
    print()

    # Build stress fact types (up to 110 unique concept types)
    all_types = build_stress_fact_types(tokenizer, 110)
    entity_ids = _get_all_entity_ids(all_types)
    n_concepts = len(all_types)
    print(f"  Fact types: {n_concepts}")
    print(f"  Entity pool: {len(entity_ids)} unique tokens")

    # Cap n_facts at available types
    n_facts_list = [n for n in n_facts_list if n <= n_concepts]
    print(f"  Sweep: {n_facts_list}")
    print()

    # Bank distribution diagnostic (using concept vocab)
    if max(bank_configs) > 1:
        concept_vocab = build_concept_vocab(tokenizer)
        diagnose_bank_distribution(model, concept_vocab, max(bank_configs))
        print()

    # ── Run sweep for each bank config ──
    all_results: dict[int, dict[int, dict]] = {}

    for n_banks in bank_configs:
        label = f"{n_banks} bank{'s' if n_banks != 1 else ''}"
        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(f"{'─' * 80}")

        model.set_bank_mode(n_banks)

        results: dict[int, dict] = {}

        print(f"  {'n':>5}  {'Accuracy':>10}  {'95% CI':>18}  "
              f"{'Std':>8}  {'Time':>6}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*18}  {'─'*8}  {'─'*6}")

        for n_facts in n_facts_list:
            t0 = time.time()
            acc, per_ep = evaluate_scaling(
                model, n_facts, all_types, entity_ids,
                tokenizer, n_eval, seed)
            dt = time.time() - t0

            ci_lo, ci_hi = bootstrap_ci(per_ep)
            std = (sum((x - acc)**2 for x in per_ep)
                   / max(len(per_ep) - 1, 1)) ** 0.5

            results[n_facts] = dict(
                accuracy=acc, ci_lo=ci_lo, ci_hi=ci_hi,
                std=std, per_episode=per_ep)

            print(f"  {n_facts:>5}  {acc:>9.1%}  "
                  f"[{ci_lo:>6.1%}, {ci_hi:>6.1%}]  "
                  f"{std:>7.1%}  {dt:>5.0f}s")

        all_results[n_banks] = results

    # ── Comparison table ──
    print(f"\n{'=' * 80}")
    print("  Comparison: Accuracy by n_facts")
    print(f"{'=' * 80}")

    header = f"  {'n':>5}"
    for nb in bank_configs:
        header += f"  {'B=' + str(nb):>8}"
    if len(bank_configs) > 1:
        header += f"  {'Delta':>8}"
    print(header)
    print(f"  {'─' * (5 + 10 * len(bank_configs) + (10 if len(bank_configs) > 1 else 0))}")

    for n_facts in n_facts_list:
        row = f"  {n_facts:>5}"
        accs = []
        for nb in bank_configs:
            acc = all_results[nb].get(n_facts, {}).get("accuracy", 0)
            accs.append(acc)
            row += f"  {acc:>7.1%}"
        if len(bank_configs) > 1:
            delta = accs[-1] - accs[0]
            row += f"  {'+' if delta >= 0 else ''}{delta:>6.1%}"
        print(row)

    # ── Find 100% thresholds ──
    print(f"\n  100% accuracy threshold:")
    for nb in bank_configs:
        results = all_results[nb]
        threshold = 0
        for n in sorted(results.keys()):
            if results[n]["accuracy"] >= 0.999:
                threshold = n
            else:
                break
        print(f"    B={nb:>3}: n={threshold}")

    # ── Save results ──
    output = {
        "config": dict(
            n_eval=n_eval, seed=seed,
            bank_configs=bank_configs,
            n_concepts=n_concepts,
            n_entities=len(entity_ids)),
        "results": {
            str(nb): {
                str(n): {k: v for k, v in r.items() if k != "per_episode"}
                for n, r in results.items()
            }
            for nb, results in all_results.items()
        },
    }

    os.makedirs("results", exist_ok=True)
    path = "results/exp28_hashed_traces.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.5))

        colors = ['#9ca3af', '#2563eb', '#059669', '#d97706']
        markers = ['s', 'o', '^', 'D']

        for i, nb in enumerate(bank_configs):
            results = all_results[nb]
            ns = sorted(results.keys())
            accs = [results[n]["accuracy"] for n in ns]
            ci_lo = [results[n]["ci_lo"] for n in ns]
            ci_hi = [results[n]["ci_hi"] for n in ns]

            label = f"Baseline (no banks)" if nb == 1 else f"{nb} banks"
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            lw = 2.5 if nb > 1 else 1.5
            ls = '--' if nb == 1 else '-'

            ax.plot(ns, accs, f'{m}{ls}', color=c, linewidth=lw,
                    markersize=7, label=label, zorder=3 + i)
            ax.fill_between(ns, ci_lo, ci_hi, alpha=0.12, color=c)

        # Random baseline
        ax.axhline(y=1 / len(entity_ids), color='#e5e7eb',
                    linestyle=':', linewidth=1,
                    label=f'Random ({1/len(entity_ids):.1%})')

        ax.set_xlabel("Number of facts stored", fontsize=12)
        ax.set_ylabel("Cross-context accuracy", fontsize=12)
        ax.set_title("Hashed Trace Banks: Capacity Scaling", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

        fig_path = "results/exp28_hashed_traces.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Figure: {fig_path}")
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not available, skipping figure)")

    return all_results


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 28: Hashed trace banks scaling")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (10 episodes, fewer points)")
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--banks", type=int, nargs="+",
                        default=[1, 16, 32],
                        help="Bank configurations to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        run_experiment(
            n_eval=10,
            bank_configs=[1, 16],
            n_facts_list=[1, 5, 10, 20, 30, 50],
            seed=args.seed,
        )
    else:
        run_experiment(
            n_eval=args.n_eval,
            bank_configs=args.banks,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
