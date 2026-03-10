"""Exp 21: Hippocampal Replay — sleep re-activation from trace.

Tests whether replaying stored Q→V associations from the trace itself
(without re-presenting original data) strengthens memory retention.

Biological analogy: sharp-wave ripples during sleep re-activate
hippocampal patterns, consolidating recent memories.

Mechanism:
    1. During write: Q keys are saved to a replay buffer
    2. During "sleep" (after each session): iterate over buffer,
       read V from trace via Q, re-write Q→V to strengthen association
    3. Buffer cleared after replay (like waking clears STM)

Key questions:
    - Does replay improve retention at high fact counts (n=20+)?
    - How many replay passes are optimal?
    - Does replay_lr differ from trace_lr?
    - Does replay interact with decay (slowing effective forgetting)?

Usage:
    python -m hebbian_trace.experiments.exp21_replay --quick
    python -m hebbian_trace.experiments.exp21_replay --n-episodes 50
    python -m hebbian_trace.experiments.exp21_replay --sweep-replays
"""

import argparse
import random
import time
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    get_linking_bpe_ids,
    GPT2FactType,
    tokenize_fact, tokenize_question,
    _get_all_entity_ids,
)
from .exp16_multi_session import (
    build_extended_fact_types, SessionSchedule, make_session_schedule,
    SessionFact,
)
from .exp13_contextual_q import get_device


# ── Replay-aware multi-session episode ──────────────────────────────

@dataclass
class ReplaySessionResult:
    session_idx: int
    n_known: int
    recall: dict[str, bool]
    trace_norm: float
    replay_buffer_size: int


@dataclass
class ReplayEpisodeResult:
    sessions: list[ReplaySessionResult]
    n_sessions: int


def run_replay_episode(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    schedule: SessionSchedule,
    n_sessions: int,
    device: torch.device,
    rng: random.Random,
    n_replays: int = 0,           # 0 = no replay (baseline)
    replay_lr: float | None = None,  # None = same as trace_lr
    replay_per_session: bool = True,  # replay after each session vs only at end
) -> ReplayEpisodeResult:
    """Run one multi-session episode with optional replay.

    Same protocol as exp16 individual write, but with replay between sessions.
    """
    model.eval()
    model.reset_traces()
    model.trace.set_replay_mode(n_replays > 0)
    all_entity_ids = _get_all_entity_ids(fact_types)

    known_facts: dict[str, tuple[str, int, list[int]]] = {}
    sessions = []

    for s in range(n_sessions):
        session_facts: list[SessionFact] = []

        # Gather facts for this session
        new_type_idxs = schedule.new_types[s] if s < len(schedule.new_types) else []

        for tidx in new_type_idxs:
            ft = fact_types[tidx]
            entity_name, entity_id = rng.choice(ft.entities)
            template = ft.fact_templates[0]
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            q_ids = tokenize_question(tokenizer, ft.question_templates[0])

            session_facts.append(SessionFact(
                type_name=ft.name,
                entity_name=entity_name,
                entity_bpe_id=entity_id,
                fact_bpe_ids=fact_ids,
                is_update=False,
                old_entity=None,
            ))
            known_facts[ft.name] = (entity_name, entity_id, q_ids)

        # ── Write phase ──
        model.set_trace_mode(use=False, update=True)
        for fact in session_facts:
            model.set_erase_mode(False)
            input_ids = torch.tensor(
                [fact.fact_bpe_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids, beta=0.0)

        buf_size = model.trace.replay_buffer_size

        # ── Replay phase (sleep) ──
        if n_replays > 0 and replay_per_session:
            model.set_trace_mode(use=False, update=False)  # no regular write
            model.replay(n_replays=n_replays, replay_lr=replay_lr)
            model.clear_replay_buffer()

        # ── Read phase: query ALL known types ──
        model.set_trace_mode(use=True, update=False)
        recall: dict[str, bool] = {}
        for type_name, (entity_name, entity_id, q_ids) in known_facts.items():
            input_ids = torch.tensor(
                [q_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids, beta=0.0)
            pred_logits = logits[0, -1, :]
            entity_logits = pred_logits[all_entity_ids]
            pred_id = all_entity_ids[entity_logits.argmax().item()]
            recall[type_name] = (pred_id == entity_id)

        trace_norm = model.trace.value_traces.norm().item()
        sessions.append(ReplaySessionResult(
            session_idx=s,
            n_known=len(known_facts),
            recall=recall,
            trace_norm=trace_norm,
            replay_buffer_size=buf_size,
        ))

    # End-of-run replay (alternative: replay only at the very end)
    if n_replays > 0 and not replay_per_session:
        model.set_trace_mode(use=False, update=False)
        model.replay(n_replays=n_replays, replay_lr=replay_lr)
        model.clear_replay_buffer()

        # Re-query after replay
        model.set_trace_mode(use=True, update=False)
        recall = {}
        for type_name, (entity_name, entity_id, q_ids) in known_facts.items():
            input_ids = torch.tensor(
                [q_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids, beta=0.0)
            pred_logits = logits[0, -1, :]
            entity_logits = pred_logits[all_entity_ids]
            pred_id = all_entity_ids[entity_logits.argmax().item()]
            recall[type_name] = (pred_id == entity_id)

        trace_norm = model.trace.value_traces.norm().item()
        sessions.append(ReplaySessionResult(
            session_idx=n_sessions,
            n_known=len(known_facts),
            recall=recall,
            trace_norm=trace_norm,
            replay_buffer_size=0,
        ))

    model.trace.set_replay_mode(False)
    return ReplayEpisodeResult(sessions=sessions, n_sessions=n_sessions)


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate_episodes(
    results: list[ReplayEpisodeResult],
) -> dict[int, dict]:
    """Aggregate per-session accuracy across episodes.

    Returns: {session_idx: {"accuracy": float, "n_known": int, "trace_norm": float}}
    """
    from collections import defaultdict
    session_data = defaultdict(lambda: {"correct": 0, "total": 0,
                                         "norms": [], "n_known": 0})

    for ep in results:
        for sr in ep.sessions:
            d = session_data[sr.session_idx]
            n_correct = sum(sr.recall.values())
            n_total = len(sr.recall)
            d["correct"] += n_correct
            d["total"] += n_total
            d["norms"].append(sr.trace_norm)
            d["n_known"] = sr.n_known

    out = {}
    for sidx, d in sorted(session_data.items()):
        import numpy as np
        out[sidx] = {
            "accuracy": d["correct"] / max(d["total"], 1),
            "n_known": d["n_known"],
            "trace_norm": float(np.mean(d["norms"])),
        }
    return out


def print_comparison(
    label_a: str, agg_a: dict,
    label_b: str, agg_b: dict,
):
    """Print side-by-side comparison of two conditions."""
    print(f"\n{'Session':>8} | {'Known':>5} | {label_a:>12} | {label_b:>12} | {'Delta':>7} | "
          f"{'Norm A':>7} | {'Norm B':>7}")
    print("-" * 75)

    for sidx in sorted(set(agg_a.keys()) | set(agg_b.keys())):
        da = agg_a.get(sidx, {"accuracy": 0, "n_known": 0, "trace_norm": 0})
        db = agg_b.get(sidx, {"accuracy": 0, "n_known": 0, "trace_norm": 0})
        delta = db["accuracy"] - da["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"  {sidx+1:>5}   | {da['n_known']:>5} | {da['accuracy']:>11.1%} | "
              f"{db['accuracy']:>11.1%} | {sign}{delta:>5.1%}pp | "
              f"{da['trace_norm']:>7.1f} | {db['trace_norm']:>7.1f}")


# ── Main experiment ──────────────────────────────────────────────────

def run_experiment(
    n_episodes: int = 30,
    n_sessions: int = 8,
    facts_per_session: int = 3,
    replay_values: list[int] | None = None,
    replay_lr: float | None = None,
    alpha: float = 0.5,
    device_str: str | None = None,
):
    """Compare no-replay vs replay across multiple settings."""
    if replay_values is None:
        replay_values = [0, 1, 3, 5, 10]

    device = get_device(device_str)
    print(f"Device: {device}")

    print("Loading GPT-2 + trace module...")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=alpha,
        trace_lr=1.0, trace_decay=0.99,
    )
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    # Build all fact types (base 7 + extended)
    all_types = build_extended_fact_types(tokenizer)

    n_types_needed = n_sessions * facts_per_session
    if n_types_needed > len(all_types):
        print(f"WARNING: need {n_types_needed} types, have {len(all_types)}. "
              f"Will reuse types.")
        n_types_needed = len(all_types)

    schedule = make_session_schedule(
        n_sessions=n_sessions,
        n_types=min(n_types_needed, len(all_types)),
        facts_per_session=facts_per_session,
    )

    print(f"\nSetup: {n_episodes} episodes, {n_sessions} sessions, "
          f"{facts_per_session} facts/session")
    print(f"Total types: {min(n_types_needed, len(all_types))}")
    print(f"Replay values to test: {replay_values}")
    if replay_lr is not None:
        print(f"Replay LR: {replay_lr}")
    print()

    # Run each replay condition
    all_results: dict[int, dict] = {}
    for n_rep in replay_values:
        label = f"replay={n_rep}"
        print(f"Running {label}...")
        t0 = time.time()

        episodes = []
        for ep in range(n_episodes):
            rng = random.Random(42 + ep)
            result = run_replay_episode(
                model=model,
                tokenizer=tokenizer,
                fact_types=all_types,
                schedule=schedule,
                n_sessions=n_sessions,
                device=device,
                rng=rng,
                n_replays=n_rep,
                replay_lr=replay_lr,
            )
            episodes.append(result)

        agg = aggregate_episodes(episodes)
        all_results[n_rep] = agg
        dt = time.time() - t0

        # Print per-session accuracy
        final_session = max(agg.keys())
        final_acc = agg[final_session]["accuracy"]
        final_norm = agg[final_session]["trace_norm"]
        print(f"  {label}: final_acc={final_acc:.1%}, "
              f"norm={final_norm:.1f}, time={dt:.1f}s")

    # Print comparison tables
    baseline = all_results[0]
    for n_rep in replay_values:
        if n_rep == 0:
            continue
        print_comparison("no_replay", baseline, f"replay={n_rep}", all_results[n_rep])

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (final session accuracy)")
    print("=" * 60)
    final_s = max(baseline.keys())
    for n_rep in replay_values:
        agg = all_results[n_rep]
        acc = agg[final_s]["accuracy"]
        norm = agg[final_s]["trace_norm"]
        delta = acc - baseline[final_s]["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"  replay={n_rep:>2}: {acc:.1%}  (norm={norm:.1f})  "
              f"{sign}{delta:.1%}pp vs no-replay")


def run_replay_lr_sweep(
    n_episodes: int = 30,
    n_sessions: int = 8,
    facts_per_session: int = 3,
    n_replays: int = 3,
    alpha: float = 0.5,
    device_str: str | None = None,
):
    """Sweep replay_lr at fixed n_replays."""
    lr_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    device = get_device(device_str)
    print(f"Device: {device}")

    print("Loading GPT-2 + trace module...")
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=alpha,
        trace_lr=1.0, trace_decay=0.99,
    )
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    all_types = build_extended_fact_types(tokenizer)

    n_types_needed = n_sessions * facts_per_session
    schedule = make_session_schedule(
        n_sessions=n_sessions,
        n_types=min(n_types_needed, len(all_types)),
        facts_per_session=facts_per_session,
    )

    print(f"\nSetup: {n_episodes} episodes, {n_sessions} sessions, "
          f"n_replays={n_replays}")
    print(f"LR values: {lr_values}\n")

    # Baseline (no replay)
    print("Running baseline (no replay)...")
    baseline_eps = []
    for ep in range(n_episodes):
        rng = random.Random(42 + ep)
        result = run_replay_episode(
            model=model, tokenizer=tokenizer, fact_types=all_types,
            schedule=schedule, n_sessions=n_sessions, device=device,
            rng=rng, n_replays=0,
        )
        baseline_eps.append(result)
    baseline_agg = aggregate_episodes(baseline_eps)

    # Sweep
    results = {}
    for lr in lr_values:
        print(f"Running replay_lr={lr}...")
        eps = []
        for ep in range(n_episodes):
            rng = random.Random(42 + ep)
            result = run_replay_episode(
                model=model, tokenizer=tokenizer, fact_types=all_types,
                schedule=schedule, n_sessions=n_sessions, device=device,
                rng=rng, n_replays=n_replays, replay_lr=lr,
            )
            eps.append(result)
        results[lr] = aggregate_episodes(eps)

    # Summary
    final_s = max(baseline_agg.keys())
    bl_acc = baseline_agg[final_s]["accuracy"]
    print(f"\n{'replay_lr':>10} | {'Final Acc':>10} | {'Delta':>7} | {'Norm':>7}")
    print("-" * 45)
    print(f"{'baseline':>10} | {bl_acc:>10.1%} | {'—':>7} | "
          f"{baseline_agg[final_s]['trace_norm']:>7.1f}")
    for lr in lr_values:
        agg = results[lr]
        acc = agg[final_s]["accuracy"]
        delta = acc - bl_acc
        sign = "+" if delta >= 0 else ""
        print(f"{lr:>10.1f} | {acc:>10.1%} | {sign}{delta:>5.1%}pp | "
              f"{agg[final_s]['trace_norm']:>7.1f}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 21: Hippocampal Replay")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 10 episodes, 5 sessions")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--n-sessions", type=int, default=8)
    parser.add_argument("--facts-per-session", type=int, default=3)
    parser.add_argument("--sweep-replays", action="store_true",
                        help="Sweep n_replays values")
    parser.add_argument("--sweep-lr", action="store_true",
                        help="Sweep replay_lr at fixed n_replays")
    parser.add_argument("--n-replays", type=int, default=3,
                        help="Fixed n_replays for lr sweep")
    parser.add_argument("--replay-lr", type=float, default=None,
                        help="Replay learning rate (default: same as trace_lr)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_episodes = 10
        args.n_sessions = 5

    if args.sweep_lr:
        run_replay_lr_sweep(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            n_replays=args.n_replays,
            alpha=args.alpha,
            device_str=args.device,
        )
    else:
        replay_values = [0, 1, 3, 5, 10] if args.sweep_replays else [0, 3]
        run_experiment(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            replay_values=replay_values,
            replay_lr=args.replay_lr,
            alpha=args.alpha,
            device_str=args.device,
        )


if __name__ == "__main__":
    main()
