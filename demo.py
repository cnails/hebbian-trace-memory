#!/usr/bin/env python3
"""Interactive multi-session demo of Hebbian trace memory.

Demonstrates persistent cross-session fact storage and retrieval
using a frozen GPT-2 with an external Hebbian trace module.

Usage:
    python demo.py                    # 5 sessions, 24 fact types
    python demo.py --sessions 10      # 10 sessions with updates
    python demo.py --sessions 15      # full 15-session demo
"""

import argparse
import random
import sys
from dataclasses import dataclass

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_extended_fact_types,
    get_linking_bpe_ids,
    get_all_entity_ids,
    tokenize_fact,
    tokenize_question,
    FactType,
)
from hebbian_trace.rag_baselines import (
    RAGStore,
    OracleRAGStore,
    TFIDFRAGStore,
    EmbeddingRAGStore,
    run_rag_multisession,
)


# -- Terminal Colors --

class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def colored(text: str, color: str) -> str:
    return f"{color}{text}{C.END}"


# -- Session Schedule --

@dataclass
class SessionFact:
    type_name: str
    entity_name: str
    entity_bpe_id: int
    fact_bpe_ids: list[int]
    is_update: bool = False


def make_session_schedule(
    n_types: int, facts_per_session: int, n_sessions: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Create introduction + update schedule.

    Phase 1: introduce all types across sessions.
    Phase 2: update existing types with new values.

    Returns:
        new_types: list of type indices to introduce per session
        update_types: list of type indices to update per session
    """
    new_types: list[list[int]] = []
    update_types: list[list[int]] = []

    introduced: list[int] = []
    type_idx = 0

    for s in range(n_sessions):
        session_new = []
        session_update = []

        if type_idx < n_types:
            n_new = min(facts_per_session, n_types - type_idx)
            session_new = list(range(type_idx, type_idx + n_new))
            type_idx += n_new
            introduced.extend(session_new)
        else:
            n_upd = min(facts_per_session, len(introduced))
            start = ((s - len(new_types)) * facts_per_session) % len(introduced)
            session_update = []
            for i in range(n_upd):
                idx = introduced[(start + i) % len(introduced)]
                session_update.append(idx)

        new_types.append(session_new)
        update_types.append(session_update)

    return new_types, update_types


# -- Core Demo --

def run_demo(
    n_sessions: int = 5,
    facts_per_session: int = 5,
    n_episodes: int = 50,
    erase_lr: float = 5.0,
    seed: int = 42,
    weights_path: str = "weights/trace_module.pt",
    rag_comparison: bool = False,
):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(colored("=" * 65, C.BOLD))
    print(colored("  Hebbian Trace Memory — Multi-Session Demo", C.BOLD + C.CYAN))
    print(colored("=" * 65, C.BOLD))
    print()
    print(f"  Device:            {device}")
    print(f"  Sessions:          {n_sessions}")
    print(f"  Facts/session:     {facts_per_session}")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Erase LR:          {erase_lr}")
    print()

    # Setup
    print(colored("Loading GPT-2 + trace module...", C.DIM))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_extended_fact_types(tokenizer, min_entities=4)
    linking_ids = get_linking_bpe_ids(tokenizer)
    entity_ids = get_all_entity_ids(fact_types)

    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    # Load trained weights
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.trace.load_state_dict(state, strict=False)
        print(colored(f"Loaded weights from {weights_path}", C.GREEN))
    except FileNotFoundError:
        print(colored(f"Warning: {weights_path} not found, using random projections", C.YELLOW))
    print()

    n_types = min(len(fact_types), 24)
    new_schedule, update_schedule = make_session_schedule(
        n_types, facts_per_session, n_sessions)

    # Accumulate results
    session_results = {s: {"correct": 0, "total": 0, "new_c": 0, "new_t": 0,
                           "old_c": 0, "old_t": 0, "upd_c": 0, "upd_t": 0}
                       for s in range(n_sessions)}
    retention_data: dict[int, list[float]] = {}

    for ep in range(n_episodes):
        rng = random.Random(seed + ep)
        model.reset_traces()

        known_facts: dict[str, tuple[str, int, list[int]]] = {}
        entity_history: dict[str, set[str]] = {}
        last_write_session: dict[str, int] = {}

        for s in range(n_sessions):
            session_facts: list[SessionFact] = []

            # New facts
            for tidx in new_schedule[s]:
                ft = fact_types[tidx]
                ent_name, ent_id = rng.choice(ft.entities)
                template = rng.choice(ft.fact_templates)
                fact_ids = tokenize_fact(tokenizer, template, ent_name)
                session_facts.append(SessionFact(
                    ft.name, ent_name, ent_id, fact_ids, is_update=False))
                q_ids = tokenize_question(tokenizer,
                                          rng.choice(ft.question_templates))
                known_facts[ft.name] = (ent_name, ent_id, q_ids)
                entity_history.setdefault(ft.name, set()).add(ent_name)
                last_write_session[ft.name] = s

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
                session_facts.append(SessionFact(
                    ft.name, ent_name, ent_id, fact_ids, is_update=True))
                q_ids = tokenize_question(tokenizer,
                                          rng.choice(ft.question_templates))
                known_facts[ft.name] = (ent_name, ent_id, q_ids)
                entity_history.setdefault(ft.name, set()).add(ent_name)
                last_write_session[ft.name] = s

            # Write phase
            model.set_trace_mode(use=False, update=True)
            for fact in session_facts:
                if fact.is_update:
                    model.set_erase_mode(True, erase_lr)
                else:
                    model.set_erase_mode(False)
                t = torch.tensor([fact.fact_bpe_ids],
                                 dtype=torch.long, device=device)
                with torch.no_grad():
                    model(t)
            model.set_erase_mode(False)

            # Read phase
            model.set_trace_mode(use=True, update=False)
            sr = session_results[s]

            for type_name, (ent_name, ent_id, q_ids) in known_facts.items():
                t = torch.tensor([q_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(t)
                pred_logits = logits[0, -1, :]
                el = pred_logits[entity_ids]
                pred_id = entity_ids[el.argmax().item()]
                correct = (pred_id == ent_id)

                sr["total"] += 1
                if correct:
                    sr["correct"] += 1

                is_new_this_session = any(
                    f.type_name == type_name and not f.is_update
                    for f in session_facts)
                is_upd_this_session = any(
                    f.type_name == type_name and f.is_update
                    for f in session_facts)

                if is_new_this_session:
                    sr["new_t"] += 1
                    if correct:
                        sr["new_c"] += 1
                elif is_upd_this_session:
                    sr["upd_t"] += 1
                    if correct:
                        sr["upd_c"] += 1
                else:
                    sr["old_t"] += 1
                    if correct:
                        sr["old_c"] += 1

                age = s - last_write_session[type_name]
                retention_data.setdefault(age, []).append(float(correct))

    # Print results
    print(colored("-" * 65, C.DIM))
    print(colored(f"  {'Sess':>4}  {'Known':>5}  {'Overall':>8}  {'New':>8}  "
                  f"{'Old':>8}  {'Update':>8}", C.BOLD))
    print(colored("-" * 65, C.DIM))

    for s in range(n_sessions):
        sr = session_results[s]
        n_known = sum(len(new_schedule[i]) for i in range(s + 1))
        overall = sr["correct"] / max(sr["total"], 1)
        new_acc = sr["new_c"] / max(sr["new_t"], 1) if sr["new_t"] else float('nan')
        old_acc = sr["old_c"] / max(sr["old_t"], 1) if sr["old_t"] else float('nan')
        upd_acc = sr["upd_c"] / max(sr["upd_t"], 1) if sr["upd_t"] else float('nan')

        overall_s = colored(f"{overall:>7.0%}", C.GREEN if overall >= 0.95 else
                            (C.YELLOW if overall >= 0.80 else C.RED))

        def fmt(v):
            if v != v:  # nan
                return colored(f"{'--':>7}", C.DIM)
            return f"{v:>7.0%}"

        print(f"  {s+1:>4}  {n_known:>5}  {overall_s}  {fmt(new_acc)}  "
              f"{fmt(old_acc)}  {fmt(upd_acc)}")

    print(colored("-" * 65, C.DIM))

    # Retention curve
    print()
    print(colored("  Retention by age (sessions since last write):", C.BOLD))
    for age in sorted(retention_data.keys()):
        vals = retention_data[age]
        acc = sum(vals) / len(vals)
        bar_len = int(acc * 30)
        bar = colored("█" * bar_len, C.GREEN) + colored("░" * (30 - bar_len), C.DIM)
        print(f"    Age {age:>2}: {bar} {acc:>5.0%}  (n={len(vals)})")

    # Summary
    last_sr = session_results[n_sessions - 1]
    final_acc = last_sr["correct"] / max(last_sr["total"], 1)
    print()
    print(colored("=" * 65, C.BOLD))
    print(colored(f"  Final session recall: {final_acc:.0%}  "
                  f"({n_types} fact types, {n_sessions} sessions, "
                  f"{n_episodes} episodes)", C.BOLD + C.GREEN))
    print(colored("=" * 65, C.BOLD))

    # RAG comparison
    if rag_comparison:
        print()
        print(colored("=" * 65, C.BOLD))
        print(colored("  RAG Baseline Comparison", C.BOLD + C.CYAN))
        print(colored("=" * 65, C.BOLD))
        print()

        wte_weight = model.gpt2.transformer.wte.weight.detach().cpu()
        rag_variants: list[tuple[str, RAGStore]] = [
            ("Oracle", OracleRAGStore(tokenizer)),
            ("Embed", EmbeddingRAGStore(tokenizer, wte_weight)),
            ("TF-IDF", TFIDFRAGStore(tokenizer)),
        ]

        rag_cached: dict[str, tuple[dict, dict]] = {}
        for rag_name, rag_store in rag_variants:
            print(colored(f"  Running RAG-{rag_name}...", C.DIM))
            rag_sr, rag_ret = run_rag_multisession(
                model, rag_store, fact_types, tokenizer, entity_ids,
                n_sessions=n_sessions,
                facts_per_session=facts_per_session,
                n_episodes=n_episodes,
                seed=seed,
            )

            print(colored(f"\n  RAG-{rag_name} Results:", C.BOLD))
            print(colored("-" * 65, C.DIM))
            print(colored(f"  {'Sess':>4}  {'Known':>5}  {'Overall':>8}  "
                          f"{'New':>8}  {'Old':>8}  {'Update':>8}", C.BOLD))
            print(colored("-" * 65, C.DIM))

            for s in range(n_sessions):
                sr = rag_sr[s]
                n_known = sum(len(new_schedule[i]) for i in range(s + 1))
                overall = sr["correct"] / max(sr["total"], 1)
                new_acc = sr["new_c"] / max(sr["new_t"], 1) if sr["new_t"] else float('nan')
                old_acc = sr["old_c"] / max(sr["old_t"], 1) if sr["old_t"] else float('nan')
                upd_acc = sr["upd_c"] / max(sr["upd_t"], 1) if sr["upd_t"] else float('nan')

                overall_s = colored(f"{overall:>7.0%}",
                                    C.GREEN if overall >= 0.95 else
                                    (C.YELLOW if overall >= 0.80 else C.RED))

                def fmt(v):
                    if v != v:
                        return colored(f"{'--':>7}", C.DIM)
                    return f"{v:>7.0%}"

                print(f"  {s+1:>4}  {n_known:>5}  {overall_s}  "
                      f"{fmt(new_acc)}  {fmt(old_acc)}  {fmt(upd_acc)}")

            print(colored("-" * 65, C.DIM))

            rag_final = rag_sr[n_sessions - 1]
            rag_final_acc = rag_final["correct"] / max(rag_final["total"], 1)
            print(f"  Final: {rag_final_acc:.0%}")
            print()

            rag_cached[rag_name] = (rag_ret, rag_sr)

        # Comparative summary
        print(colored("=" * 65, C.BOLD))
        print(colored("  Comparison Summary (Final Session)", C.BOLD))
        print(colored("-" * 65, C.DIM))
        print(f"  {'Method':<20} {'Final Recall':>12}")
        print(colored("-" * 65, C.DIM))
        print(colored(f"  {'Hebbian Trace':<20} {final_acc:>11.0%}", C.GREEN))
        for rag_name, (_, rag_session_results) in rag_cached.items():
            rag_final = rag_session_results[n_sessions - 1]
            rag_acc = rag_final["correct"] / max(rag_final["total"], 1)
            print(f"  {'RAG-' + rag_name:<20} {rag_acc:>11.0%}")
        print(colored("-" * 65, C.DIM))


def main():
    parser = argparse.ArgumentParser(
        description="Hebbian Trace Memory — Multi-Session Demo")
    parser.add_argument("--sessions", type=int, default=5,
                        help="Number of sessions (default: 5)")
    parser.add_argument("--facts-per-session", type=int, default=5,
                        help="Facts per session (default: 5)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--erase-lr", type=float, default=5.0,
                        help="Erasure learning rate for updates (default: 5.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="weights/trace_module.pt",
                        help="Path to trace module weights")
    parser.add_argument("--rag-comparison", action="store_true",
                        help="Run RAG baselines for comparison")
    args = parser.parse_args()

    run_demo(
        n_sessions=args.sessions,
        facts_per_session=args.facts_per_session,
        n_episodes=args.episodes,
        erase_lr=args.erase_lr,
        seed=args.seed,
        weights_path=args.weights,
        rag_comparison=args.rag_comparison,
    )


if __name__ == "__main__":
    main()
