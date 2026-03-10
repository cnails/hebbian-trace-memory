"""Exp 16: Multi-Session Demo — persistent trace memory across sessions.

Tests the unique value proposition: memory accumulates between sessions
without retraining or RAG. Each session adds new facts; trace persists.

Setup:
  - beta=0 (context-free Q — no distractors = no confusion penalty)
  - alpha=0.5, pattern separation 8x_k16
  - trace_decay=0.99, trace_lr=1.0
  - Extended fact types (15+) for capacity test

Protocol:
  - N sessions (default 10), K facts per session (default 5)
  - First sessions introduce new fact types
  - Later sessions update existing facts (new entity values)
  - After each session: query ALL known types
  - Track: retention curve, update success, trace norm

Key insight: beta=0 is optimal here. All facts are "my" facts, no third-person
distractors. Confusion (the metric that beta>0 helps) is irrelevant.
Cross-context recall at beta=0 is ~87% at n=5 with PS — maximum possible.

Usage:
    python -m hebbian_trace.experiments.exp16_multi_session --quick
    python -m hebbian_trace.experiments.exp16_multi_session --n-episodes 50
    python -m hebbian_trace.experiments.exp16_multi_session --n-sessions 15
"""

import argparse
import random
import time
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from ..gpt2_tasks import (
    build_fact_types, get_linking_bpe_ids,
    GPT2FactType, GPT2FactTemplate, GPT2QuestionTemplate,
    validate_single_token_entities, tokenize_fact, tokenize_question,
    _get_all_entity_ids, build_session_paragraph,
)
from ..nlp_tasks import HOBBIES, LANGUAGES
from .exp9_learned_gating import train_gate
from .exp11_dual_gates import train_gate_key
from .exp12_realistic_benchmarks import build_question_variants, QuestionVariant
from .exp13_contextual_q import get_device
from .exp22_pattern_completion import extract_auto_pairs, AutoPair


# ── Extended fact types ──────────────────────────────────────────────

# Extra entity pools beyond the 7 in gpt2_tasks.py
EXTRA_POOLS: dict[str, tuple[list[str], str, str, str]] = {
    # type_name: (entities, fact_template, linking_word, question_template)
    "drink": (
        ["coffee", "tea", "water", "juice", "wine", "beer", "milk", "soda"],
        "My drink is {X}.", "is", "What is my drink?",
    ),
    "sport": (
        ["tennis", "golf", "soccer", "baseball", "hockey", "boxing",
         "rugby", "cricket", "skiing"],
        "My sport is {X}.", "is", "What is my sport?",
    ),
    "hobby": (
        list(HOBBIES),
        "My hobby is {X}.", "is", "What is my hobby?",
    ),
    "number": (
        ["seven", "eight", "nine", "three", "four", "five", "six", "ten",
         "twelve", "fifteen", "twenty"],
        "My number is {X}.", "is", "What is my number?",
    ),
    "language": (
        list(LANGUAGES),
        "My language is {X}.", "is", "What is my language?",
    ),
    "animal": (
        ["lion", "tiger", "bear", "eagle", "wolf", "deer", "fox",
         "whale", "dolphin", "hawk", "shark"],
        "My animal is {X}.", "is", "What is my animal?",
    ),
    "instrument": (
        ["piano", "guitar", "violin", "drums", "flute", "harp", "cello",
         "trumpet", "saxophone"],
        "My instrument is {X}.", "is", "What is my instrument?",
    ),
    "season": (
        ["spring", "summer", "autumn", "winter"],
        "My season is {X}.", "is", "What is my season?",
    ),
    "subject": (
        ["math", "science", "history", "art", "music", "physics",
         "chemistry", "biology", "geography", "philosophy"],
        "My subject is {X}.", "is", "What is my subject?",
    ),
    "day": (
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
         "Saturday", "Sunday"],
        "My day is {X}.", "is", "What is my day?",
    ),
    "car": (
        ["Toyota", "Honda", "Ford", "Tesla", "Volvo", "Audi", "BMW",
         "Porsche", "Ferrari", "Lexus"],
        "My car is {X}.", "is", "What is my car?",
    ),
    "metal": (
        ["gold", "silver", "iron", "copper", "steel", "bronze",
         "platinum", "tin", "zinc"],
        "My metal is {X}.", "is", "What is my metal?",
    ),
    "gem": (
        ["diamond", "ruby", "emerald", "pearl", "jade", "amber",
         "sapphire", "opal", "garnet"],
        "My gem is {X}.", "is", "What is my gem?",
    ),
    "flower": (
        ["rose", "lily", "tulip", "daisy", "violet", "orchid",
         "lotus", "iris"],
        "My flower is {X}.", "is", "What is my flower?",
    ),
    "tree": (
        ["oak", "pine", "maple", "birch", "cedar", "elm", "willow",
         "palm", "bamboo"],
        "My tree is {X}.", "is", "What is my tree?",
    ),
    "fruit": (
        ["apple", "banana", "mango", "peach", "cherry", "grape",
         "lemon", "plum", "melon", "kiwi"],
        "My fruit is {X}.", "is", "What is my fruit?",
    ),
    "tool": (
        ["hammer", "wrench", "drill", "saw", "pliers", "shovel",
         "screwdriver", "chisel"],
        "My tool is {X}.", "is", "What is my tool?",
    ),
    "fabric": (
        ["cotton", "silk", "wool", "leather", "linen", "velvet",
         "denim", "nylon"],
        "My fabric is {X}.", "is", "What is my fabric?",
    ),
}


def build_extended_fact_types(
    tokenizer: GPT2Tokenizer,
    min_entities: int = 4,
    verbose: bool = True,
) -> list[GPT2FactType]:
    """Build extended fact types (7 base + extras) with BPE validation.

    Returns all types with >= min_entities single-token BPE entities.
    """
    # Start with the base 7 types
    base_types = build_fact_types(tokenizer)
    base_names = {ft.name for ft in base_types}

    if verbose:
        print(f"Base types: {len(base_types)} "
              f"({', '.join(ft.name for ft in base_types)})")

    # Add extra types
    extra_types = []
    for type_name, (pool, fact_tmpl, link_word, q_tmpl) in EXTRA_POOLS.items():
        if type_name in base_names:
            continue  # Skip if already in base

        entities = validate_single_token_entities(tokenizer, pool)
        if len(entities) >= min_entities:
            ft = GPT2FactType(
                name=type_name,
                entities=entities,
                fact_templates=[GPT2FactTemplate(fact_tmpl, link_word)],
                question_templates=[GPT2QuestionTemplate(q_tmpl)],
            )
            extra_types.append(ft)
            if verbose:
                print(f"  + {type_name}: {len(entities)} entities "
                      f"({', '.join(e for e, _ in entities[:5])}...)")
        elif verbose:
            print(f"  - {type_name}: only {len(entities)} entities (skipped)")

    all_types = base_types + extra_types
    if verbose:
        print(f"Total: {len(all_types)} fact types")
    return all_types


# ── Session planning ─────────────────────────────────────────────────

@dataclass
class SessionFact:
    """One fact to write in a session."""
    type_name: str
    entity_name: str
    entity_bpe_id: int
    fact_bpe_ids: list[int]
    is_update: bool          # True if this updates a previously known type
    old_entity: str | None   # Previous entity if update


@dataclass
class SessionSchedule:
    """Plan for which types to introduce/update at each session."""
    # session_idx -> list of type indices to introduce (new) or update
    new_types: list[list[int]]      # session_idx -> type indices to introduce
    update_types: list[list[int]]   # session_idx -> type indices to update


def make_session_schedule(
    n_types: int,
    facts_per_session: int,
    n_sessions: int,
) -> SessionSchedule:
    """Create a fixed schedule for type introduction and updates.

    Phase 1: introduce types (ceil(n_types / facts_per_session) sessions)
    Phase 2: rotate through updates
    """
    new_types: list[list[int]] = []
    update_types: list[list[int]] = []

    remaining = list(range(n_types))
    introduced: list[int] = []

    for s in range(n_sessions):
        if remaining:
            # Phase 1: introduce new types
            batch = remaining[:facts_per_session]
            remaining = remaining[facts_per_session:]
            new_types.append(batch)
            update_types.append([])
            introduced.extend(batch)
        else:
            # Phase 2: update existing types (round-robin)
            new_types.append([])
            # Cycle through introduced types
            n_to_update = min(facts_per_session, len(introduced))
            # Rotate: pick types based on session index
            offset = (s - len(new_types) + len(update_types)) * n_to_update
            indices = []
            for i in range(n_to_update):
                idx = introduced[(offset + i) % len(introduced)]
                if idx not in indices:
                    indices.append(idx)
                else:
                    # Find next non-duplicate
                    for j in range(len(introduced)):
                        cand = introduced[(offset + i + j) % len(introduced)]
                        if cand not in indices:
                            indices.append(cand)
                            break
            update_types.append(indices[:n_to_update])

    return SessionSchedule(new_types=new_types, update_types=update_types)


# ── Dual gate training ───────────────────────────────────────────────

def train_dual_gates(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    """Train both W_gate (position) and W_gate_key (concept) for paragraph storage.

    Uses base 7 fact types for training (sufficient for generalization —
    gate operates on wte embeddings, not specific fact content).

    Returns:
        gate_key_values: dict mapping concept words to gate_key activation
    """
    base_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    if verbose:
        print("\n" + "=" * 50)
        print("DUAL GATE TRAINING")
        print("=" * 50)

    # ── Phase A: Train W_gate (position gate) ──
    if verbose:
        print("\n--- Phase A: W_gate (position gate) ---")

    gate_stages = [
        # (n_steps, n_facts, lr, l1)
        (280, 1, 3e-3, 0.0),
        (280, 3, 1e-3, 0.0),
        (280, 5, 3e-4, 0.0),
        (280, 5, 3e-4, 0.5),
        (280, 5, 1e-4, 1.0),
    ]
    for i, (steps, nf, lr, l1) in enumerate(gate_stages):
        train_gate(
            model=model, tokenizer=tokenizer,
            fact_types=base_types, n_steps=steps,
            n_facts_train=nf, lr=lr, device=device,
            log_every=steps, seed=seed + i * 30000,
            gate_only=True, grad_clip=1.0, l1_lambda=l1,
        )

    # ── Phase B: Train W_gate_key (concept gate) ──
    if verbose:
        print("\n--- Phase B: W_gate_key (concept gate) ---")

    key_stages = [
        # (n_steps, n_facts, n_filler, filler_mode, lr, l1)
        (300, 3, 0, "none", 3e-3, 0.0),
        (300, 5, 0, "none", 1e-3, 0.0),
        (300, 5, 3, "noisy", 3e-4, 0.0),
        (300, 5, 5, "mixed", 1e-4, 0.5),
    ]
    for i, (steps, nf, nfill, fmode, lr, l1) in enumerate(key_stages):
        train_gate_key(
            model=model, tokenizer=tokenizer,
            fact_types=base_types, n_steps=steps,
            n_facts_train=nf, n_filler=nfill,
            filler_mode=fmode, lr=lr, device=device,
            log_every=steps, seed=seed + i * 35000,
            grad_clip=1.0, l1_lambda=l1,
        )

    # Restore requires_grad for all trace params
    for p in model.trace.parameters():
        p.requires_grad_(True)

    # ── Validate gate_key on concept words ──
    if verbose:
        print("\n--- Gate_key validation ---")

    wte = model.gpt2.transformer.wte
    gate_key_values: dict[str, float] = {}

    # Collect all concept words from extended types
    extended_types = build_extended_fact_types(tokenizer, verbose=False)
    concept_words = [ft.name for ft in extended_types]
    # Add known filler concept words for comparison
    filler_words = ["weather", "time", "answer", "solution", "office",
                    "result", "day", "package"]

    for word in concept_words + filler_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            t = torch.tensor([[ids[0]]], dtype=torch.long, device=device)
            with torch.no_grad():
                gk = model.trace.compute_gate_key(wte, t)
            gate_key_values[word] = gk[0, 0].item()

    if verbose:
        # Print sorted by value
        sorted_vals = sorted(gate_key_values.items(), key=lambda x: -x[1])
        fact_vals = [v for w, v in sorted_vals if w in concept_words]
        filler_vals = [v for w, v in sorted_vals if w in filler_words]

        print(f"  Fact concepts ({len(fact_vals)}): "
              f"avg={sum(fact_vals)/max(len(fact_vals),1):.3f}")
        for w, v in sorted_vals:
            if w in concept_words:
                print(f"    {w:>12}: {v:.3f}")
        print(f"  Filler words ({len(filler_vals)}): "
              f"avg={sum(filler_vals)/max(len(filler_vals),1):.3f}")
        for w, v in sorted_vals:
            if w in filler_words:
                print(f"    {w:>12}: {v:.3f}")

    return gate_key_values


# ── Multi-session execution ──────────────────────────────────────────

@dataclass
class SessionResult:
    """Results from one session within an episode."""
    session_idx: int
    facts_written: list[SessionFact]
    recall: dict[str, bool]         # type_name -> correct?
    trace_norm: float
    n_known_types: int              # how many types known at this point
    # Paraphrase recall (T_auto pattern completion)
    paraphrase_recall: dict[tuple[str, str, str], bool] = field(
        default_factory=dict)   # (fact_type, category, question) -> correct?
    paraphrase_updated_types: set[str] = field(
        default_factory=set)    # types that have EVER been updated by this session


@dataclass
class EpisodeResult:
    """Results from one complete multi-session episode."""
    sessions: list[SessionResult]
    n_sessions: int
    n_types_used: int


def run_single_episode(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    fact_types: list[GPT2FactType],
    schedule: SessionSchedule,
    n_sessions: int,
    device: torch.device,
    rng: random.Random,
    write_mode: str = "individual",  # "individual", "batch", or "paragraph"
    erase_lr: float | None = None,   # if set, enable erase for updates
    filler_mode: str = "mixed",      # for paragraph mode
    n_filler_per_session: int = 3,   # for paragraph mode
    filler_rng: random.Random | None = None,  # separate RNG for filler
    # T_auto pattern completion
    auto_pairs: list[tuple[int, int]] | None = None,
    completion_alpha: float = 0.3,
    question_variants: dict[str, list[QuestionVariant]] | None = None,
) -> EpisodeResult:
    """Run one multi-session episode.

    Args:
        model: GPT2WithTrace (will reset traces internally)
        tokenizer: GPT-2 tokenizer
        fact_types: list of all available fact types
        schedule: SessionSchedule defining type intro/update order
        n_sessions: number of sessions to run
        device: torch device
        rng: random number generator (for entity selection)
        write_mode: "individual" (one forward pass per fact, stronger storage)
                    "batch" (one forward pass per session, weaker per-fact)
                    "paragraph" (facts + filler in one pass, dual gates)
        erase_lr: if set, enable reconsolidation erasure during update writes.
                  Erases old Q→V before writing new. None = no erase.
        filler_mode: filler type for paragraph mode ("none"|"safe"|"noisy"|"mixed")
        n_filler_per_session: number of filler sentences per paragraph
        filler_rng: separate RNG for filler selection (default: Random(0))
        auto_pairs: T_auto Q→Q pairs [(variant_id, concept_id), ...].
                    If set, enables pattern completion channel.
        completion_alpha: weight for completion logits (default: 0.3)
        question_variants: paraphrase questions per fact type (7 base types).
                           If set, runs paraphrase eval after standard eval.

    Returns:
        EpisodeResult with per-session recall data
    """
    model.eval()
    model.reset_traces()

    # Write T_auto pairs (static template knowledge) if enabled
    if auto_pairs is not None:
        model.write_auto_pairs(auto_pairs)
        model.set_auto_mode(True, completion_alpha)
    else:
        model.set_auto_mode(False)
    all_entity_ids = _get_all_entity_ids(fact_types)

    # Track current state
    known_facts: dict[str, tuple[str, int, list[int]]] = {}
    # type_name -> (entity_name, entity_bpe_id, question_bpe_ids)

    # Track entity history to avoid re-selecting same entity on update
    entity_history: dict[str, set[str]] = {}  # type_name -> used entities

    sessions = []
    ever_updated_types: set[str] = set()  # cumulative: types updated in any session

    for s in range(n_sessions):
        session_facts: list[SessionFact] = []

        # Gather facts for this session
        new_type_idxs = schedule.new_types[s] if s < len(schedule.new_types) else []
        upd_type_idxs = schedule.update_types[s] if s < len(schedule.update_types) else []

        # New types: select random entity
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
            entity_history.setdefault(ft.name, set()).add(entity_name)

        # Updated types: select NEW entity (different from current)
        for tidx in upd_type_idxs:
            ft = fact_types[tidx]
            old_entity = known_facts[ft.name][0]
            # Pick entity different from current
            candidates = [(e, eid) for e, eid in ft.entities
                          if e != old_entity]
            if not candidates:
                continue
            entity_name, entity_id = rng.choice(candidates)
            template = ft.fact_templates[0]
            fact_ids = tokenize_fact(tokenizer, template, entity_name)
            q_ids = tokenize_question(tokenizer, ft.question_templates[0])

            session_facts.append(SessionFact(
                type_name=ft.name,
                entity_name=entity_name,
                entity_bpe_id=entity_id,
                fact_bpe_ids=fact_ids,
                is_update=True,
                old_entity=old_entity,
            ))
            known_facts[ft.name] = (entity_name, entity_id, q_ids)
            entity_history[ft.name].add(entity_name)

        # ── Write phase ──
        model.set_trace_mode(use=False, update=True)
        if write_mode == "individual":
            # One forward pass per fact (stronger per-fact storage)
            for fact in session_facts:
                # Toggle erase: ON for updates, OFF for new facts
                if fact.is_update and erase_lr is not None:
                    model.set_erase_mode(True, erase_lr)
                else:
                    model.set_erase_mode(False)
                input_ids = torch.tensor(
                    [fact.fact_bpe_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    model(input_ids, beta=0.0)
            model.set_erase_mode(False)  # always disable after write phase
        elif write_mode == "batch":
            # One forward pass for all session facts (weaker per-fact)
            if session_facts:
                space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
                all_ids: list[int] = []
                for fact in session_facts:
                    if all_ids:
                        all_ids.append(space_id)
                    all_ids.extend(fact.fact_bpe_ids)
                input_ids = torch.tensor(
                    [all_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    model(input_ids, beta=0.0)
        elif write_mode == "paragraph":
            # Paragraph mode: system receives paragraph (facts + filler),
            # dual gate identifies which sentences are facts, writes them
            # individually with hardcoded mask (full strength).
            # Updates use individual write with erase.
            new_facts = [f for f in session_facts if not f.is_update]
            update_facts = [f for f in session_facts if f.is_update]

            if new_facts:
                model.set_erase_mode(False)
                model.set_dual_gate_mode(False)
                model.set_gate_mode(False)  # hardcoded mask for storage
                f_rng = filler_rng if filler_rng is not None \
                    else random.Random(0)
                wte = model.gpt2.transformer.wte

                # Generate filler sequences
                filler_seqs: list[list[int]] = []
                if filler_mode != "none" and n_filler_per_session > 0:
                    from ..gpt2_tasks import FILLER_NO_LINK, FILLER_WITH_LINK
                    if filler_mode == "safe":
                        pool = FILLER_NO_LINK
                    elif filler_mode == "noisy":
                        pool = FILLER_WITH_LINK
                    elif filler_mode == "mixed":
                        pool = FILLER_NO_LINK + FILLER_WITH_LINK
                    else:
                        pool = []
                    for _ in range(n_filler_per_session):
                        text = f_rng.choice(pool)
                        ids = tokenizer.encode(text,
                                               add_special_tokens=False)
                        filler_seqs.append(ids)

                # Interleave facts and filler into sentence list
                all_sentences: list[list[int]] = []
                fi = 0
                for fact in new_facts:
                    all_sentences.append(fact.fact_bpe_ids)
                    if fi < len(filler_seqs):
                        all_sentences.append(filler_seqs[fi])
                        fi += 1
                while fi < len(filler_seqs):
                    all_sentences.append(filler_seqs[fi])
                    fi += 1

                # Process each sentence: dual gate decides store or skip
                # Threshold: combined = gate_pos * gate_key
                # Fact concepts: gate_key ~0.09-0.40 → combined ~0.02-0.10
                # Filler concepts: gate_key ~0.03-0.10 → combined ~0.007-0.025
                gate_threshold = 0.010
                n_written = 0
                n_skipped = 0
                for sent_ids in all_sentences:
                    t = torch.tensor(
                        [sent_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        # Compute dual gate signal
                        gate_pos = model.trace.compute_gate(wte, t)
                        gate_key = model.trace.compute_gate_key(wte, t)
                        S = t.shape[1]
                        if S > 2:
                            combined = (gate_pos[:, 1:-1]
                                        * gate_key[:, :-2])
                            max_gate = combined.max().item()
                        else:
                            max_gate = 0.0

                        # Write only if gate detects a storable fact
                        if max_gate > gate_threshold:
                            model(t, beta=0.0)
                            n_written += 1
                        else:
                            n_skipped += 1

            # Updates: individual write with hardcoded mask + erase
            if update_facts:
                model.set_dual_gate_mode(False)
                model.set_gate_mode(False)
                for fact in update_facts:
                    if erase_lr is not None:
                        model.set_erase_mode(True, erase_lr)
                    else:
                        model.set_erase_mode(False)
                    input_ids = torch.tensor(
                        [fact.fact_bpe_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        model(input_ids, beta=0.0)
                model.set_erase_mode(False)
        else:
            raise ValueError(f"Unknown write_mode: {write_mode}")

        # Track which types were updated this session (cumulative)
        for fact in session_facts:
            if fact.is_update:
                ever_updated_types.add(fact.type_name)

        # ── Read phase: query ALL known types ──
        model.set_trace_mode(use=True, update=False)
        # Disable completion channel for standard queries — T_auto pairs cover
        # only 7 base types; for the other 17 types, the completion channel
        # produces spurious Q_corrected → noise in logits.
        if auto_pairs is not None:
            model.set_auto_mode(False)
        recall: dict[str, bool] = {}
        for type_name, (entity_name, entity_id, q_ids) in known_facts.items():
            input_ids = torch.tensor(
                [q_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids, beta=0.0)
            # Predict: argmax over entity vocabulary at last position
            pred_logits = logits[0, -1, :]
            entity_logits = pred_logits[all_entity_ids]
            best_idx = entity_logits.argmax().item()
            pred_id = all_entity_ids[best_idx]
            recall[type_name] = (pred_id == entity_id)

        # ── Paraphrase queries (T_auto pattern completion test) ──
        # Re-enable completion channel for paraphrase queries only
        if auto_pairs is not None:
            model.set_auto_mode(True, completion_alpha)
        paraphrase_recall: dict[tuple[str, str, str], bool] = {}
        if question_variants is not None:
            for type_name, (entity_name, entity_id, _) in known_facts.items():
                if type_name not in question_variants:
                    continue  # only 7 base types have variants
                for v in question_variants[type_name]:
                    input_ids = torch.tensor(
                        [v.bpe_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = model(input_ids, beta=0.0)
                    pred_logits = logits[0, -1, :]
                    entity_logits = pred_logits[all_entity_ids]
                    best_idx = entity_logits.argmax().item()
                    pred_id = all_entity_ids[best_idx]
                    key = (type_name, v.category, v.text)
                    paraphrase_recall[key] = (pred_id == entity_id)

        # Record
        trace_norm = model.trace.value_traces.norm().item()
        sessions.append(SessionResult(
            session_idx=s,
            facts_written=session_facts,
            recall=recall,
            trace_norm=trace_norm,
            n_known_types=len(known_facts),
            paraphrase_recall=paraphrase_recall,
            paraphrase_updated_types=set(ever_updated_types),
        ))

    return EpisodeResult(
        sessions=sessions,
        n_sessions=n_sessions,
        n_types_used=len(known_facts),
    )


# ── Aggregation and display ──────────────────────────────────────────

@dataclass
class AggregatedResults:
    """Aggregated results across multiple episodes."""
    n_episodes: int
    n_sessions: int
    n_types: int
    type_names: list[str]

    # Per-session metrics (averaged over episodes)
    session_recall: list[float]       # overall recall per session
    session_new_recall: list[float]   # recall of facts introduced this session
    session_old_recall: list[float]   # recall of facts from previous sessions
    session_update_recall: list[float]  # recall of updated facts
    session_trace_norm: list[float]

    # Retention curve: recall by "age" (sessions since introduction)
    retention_by_age: dict[int, float]  # age -> avg recall

    # Per-type recall at final session
    final_type_recall: dict[str, float]

    # Introduction session for each type
    type_intro_session: dict[str, int]

    # Paraphrase metrics (7 base types with T_auto)
    paraphrase_by_category: dict[str, float] = field(default_factory=dict)
    # "aligned"/"misaligned"/"semantic" -> accuracy at final session
    paraphrase_by_session: list[dict[str, float]] = field(default_factory=list)
    # per-session: category -> accuracy
    paraphrase_overall: float = 0.0
    # Updated vs stable paraphrase split (T_auto + erasure composition)
    paraphrase_updated_by_session: list[float] = field(default_factory=list)
    paraphrase_stable_by_session: list[float] = field(default_factory=list)


def aggregate_results(
    episodes: list[EpisodeResult],
    schedule: SessionSchedule,
    fact_types: list[GPT2FactType],
) -> AggregatedResults:
    """Aggregate results across episodes."""
    n_episodes = len(episodes)
    n_sessions = episodes[0].n_sessions

    # Map type_idx -> type_name and intro session
    type_names = [ft.name for ft in fact_types]
    type_intro_session: dict[str, int] = {}
    for s, new_idxs in enumerate(schedule.new_types):
        for tidx in new_idxs:
            type_intro_session[fact_types[tidx].name] = s

    # Per-session accumulators
    session_recall_sum = [0.0] * n_sessions
    session_recall_count = [0] * n_sessions
    session_new_correct = [0] * n_sessions
    session_new_total = [0] * n_sessions
    session_old_correct = [0] * n_sessions
    session_old_total = [0] * n_sessions
    session_update_correct = [0] * n_sessions
    session_update_total = [0] * n_sessions
    session_trace_norm_sum = [0.0] * n_sessions

    # Retention by age
    retention_correct: dict[int, int] = {}
    retention_total: dict[int, int] = {}

    # Final session per-type
    final_type_correct: dict[str, int] = {name: 0 for name in type_names}
    final_type_total: dict[str, int] = {name: 0 for name in type_names}

    for ep in episodes:
        # Track which types were updated at which session in this episode
        updated_at: dict[str, int] = {}  # type_name -> last update session

        for sr in ep.sessions:
            s = sr.session_idx

            # Track updates
            for fact in sr.facts_written:
                if fact.is_update:
                    updated_at[fact.type_name] = s

            # New facts this session (just introduced)
            new_type_names = set()
            for fact in sr.facts_written:
                if not fact.is_update:
                    new_type_names.add(fact.type_name)

            # Updated facts this session
            update_type_names = set()
            for fact in sr.facts_written:
                if fact.is_update:
                    update_type_names.add(fact.type_name)

            # Recall accounting
            for type_name, correct in sr.recall.items():
                session_recall_sum[s] += int(correct)
                session_recall_count[s] += 1

                # Categorize
                if type_name in new_type_names:
                    session_new_correct[s] += int(correct)
                    session_new_total[s] += 1
                elif type_name in update_type_names:
                    session_update_correct[s] += int(correct)
                    session_update_total[s] += 1
                else:
                    session_old_correct[s] += int(correct)
                    session_old_total[s] += 1

                # Retention by age (sessions since last write)
                intro_s = type_intro_session.get(type_name, 0)
                last_write = updated_at.get(type_name, intro_s)
                age = s - last_write
                retention_correct[age] = retention_correct.get(age, 0) + int(correct)
                retention_total[age] = retention_total.get(age, 0) + 1

            session_trace_norm_sum[s] += sr.trace_norm

            # Final session per-type recall
            if s == ep.n_sessions - 1:
                for type_name, correct in sr.recall.items():
                    final_type_correct[type_name] += int(correct)
                    final_type_total[type_name] += 1

    # Compute averages
    session_recall = [
        session_recall_sum[s] / max(session_recall_count[s], 1)
        for s in range(n_sessions)
    ]
    session_new_recall = [
        session_new_correct[s] / max(session_new_total[s], 1)
        for s in range(n_sessions)
    ]
    session_old_recall = [
        session_old_correct[s] / max(session_old_total[s], 1)
        for s in range(n_sessions)
    ]
    session_update_recall = [
        session_update_correct[s] / max(session_update_total[s], 1)
        if session_update_total[s] > 0 else -1.0
        for s in range(n_sessions)
    ]
    session_trace_norm = [
        session_trace_norm_sum[s] / n_episodes
        for s in range(n_sessions)
    ]

    retention_by_age = {
        age: retention_correct[age] / max(retention_total[age], 1)
        for age in sorted(retention_correct.keys())
    }

    final_type_recall = {
        name: final_type_correct[name] / max(final_type_total[name], 1)
        for name in type_names
        if final_type_total.get(name, 0) > 0
    }

    # ── Paraphrase aggregation ──
    has_paraphrase = any(
        sr.paraphrase_recall
        for ep in episodes for sr in ep.sessions
    )

    paraphrase_by_category: dict[str, float] = {}
    paraphrase_by_session: list[dict[str, float]] = []
    paraphrase_overall = 0.0
    paraphrase_updated_by_session: list[float] = []
    paraphrase_stable_by_session: list[float] = []

    if has_paraphrase:
        cats = ["aligned", "misaligned", "semantic"]

        # Per-session, per-category accumulators
        sess_cat_correct = [[0] * len(cats) for _ in range(n_sessions)]
        sess_cat_total = [[0] * len(cats) for _ in range(n_sessions)]
        # Per-session updated/stable accumulators
        sess_upd_correct = [0] * n_sessions
        sess_upd_total = [0] * n_sessions
        sess_stb_correct = [0] * n_sessions
        sess_stb_total = [0] * n_sessions

        for ep in episodes:
            for sr in ep.sessions:
                s = sr.session_idx
                for (ft_name, category, q_text), correct in sr.paraphrase_recall.items():
                    ci = cats.index(category) if category in cats else -1
                    if ci >= 0:
                        sess_cat_correct[s][ci] += int(correct)
                        sess_cat_total[s][ci] += 1

                    # Updated vs stable split
                    if ft_name in sr.paraphrase_updated_types:
                        sess_upd_correct[s] += int(correct)
                        sess_upd_total[s] += 1
                    else:
                        sess_stb_correct[s] += int(correct)
                        sess_stb_total[s] += 1

        # Compute per-session category averages
        for s in range(n_sessions):
            cat_acc: dict[str, float] = {}
            for ci, cat in enumerate(cats):
                t = sess_cat_total[s][ci]
                cat_acc[cat] = sess_cat_correct[s][ci] / max(t, 1) if t > 0 else -1.0
            paraphrase_by_session.append(cat_acc)

        # Final session category breakdown
        final_s = n_sessions - 1
        total_c, total_t = 0, 0
        for ci, cat in enumerate(cats):
            t = sess_cat_total[final_s][ci]
            if t > 0:
                paraphrase_by_category[cat] = sess_cat_correct[final_s][ci] / t
                total_c += sess_cat_correct[final_s][ci]
                total_t += t
        paraphrase_overall = total_c / max(total_t, 1)

        # Per-session updated/stable averages
        for s in range(n_sessions):
            ut = sess_upd_total[s]
            st = sess_stb_total[s]
            paraphrase_updated_by_session.append(
                sess_upd_correct[s] / max(ut, 1) if ut > 0 else -1.0)
            paraphrase_stable_by_session.append(
                sess_stb_correct[s] / max(st, 1) if st > 0 else -1.0)

    return AggregatedResults(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        n_types=len(type_names),
        type_names=type_names,
        session_recall=session_recall,
        session_new_recall=session_new_recall,
        session_old_recall=session_old_recall,
        session_update_recall=session_update_recall,
        session_trace_norm=session_trace_norm,
        retention_by_age=retention_by_age,
        final_type_recall=final_type_recall,
        type_intro_session=type_intro_session,
        paraphrase_by_category=paraphrase_by_category,
        paraphrase_by_session=paraphrase_by_session,
        paraphrase_overall=paraphrase_overall,
        paraphrase_updated_by_session=paraphrase_updated_by_session,
        paraphrase_stable_by_session=paraphrase_stable_by_session,
    )


def print_results(agg: AggregatedResults):
    """Pretty-print aggregated multi-session results."""
    print("\n" + "=" * 70)
    print("MULTI-SESSION DEMO RESULTS")
    print(f"  {agg.n_episodes} episodes, {agg.n_sessions} sessions, "
          f"{agg.n_types} fact types")
    print("=" * 70)

    # Session-by-session recall
    print("\n--- Per-Session Recall ---")
    print(f"{'Session':>8} {'Known':>6} {'Overall':>8} {'New':>8} "
          f"{'Old':>8} {'Update':>8} {'Trace':>10}")
    print("-" * 62)

    # Count known types at each session from the schedule
    known_at = 0
    for s in range(agg.n_sessions):
        intro_count = sum(
            1 for name, intro_s in agg.type_intro_session.items()
            if intro_s == s
        )
        known_at += intro_count

        upd_str = (f"{agg.session_update_recall[s]:>7.0%}"
                   if agg.session_update_recall[s] >= 0 else "    ---")
        print(f"{s+1:>8} {known_at:>6} {agg.session_recall[s]:>7.0%} "
              f"{agg.session_new_recall[s]:>7.0%} "
              f"{agg.session_old_recall[s]:>7.0%} "
              f"{upd_str} "
              f"{agg.session_trace_norm[s]:>10.2f}")

    # Retention curve
    print("\n--- Retention Curve (recall by fact age) ---")
    print(f"{'Age':>5} {'Recall':>8}")
    print("-" * 15)
    for age in sorted(agg.retention_by_age.keys()):
        if age <= agg.n_sessions:
            print(f"{age:>5} {agg.retention_by_age[age]:>7.0%}")

    # Final session per-type recall
    print(f"\n--- Per-Type Recall at Session {agg.n_sessions} ---")
    # Sort by introduction session, then name
    sorted_types = sorted(
        agg.final_type_recall.items(),
        key=lambda x: (agg.type_intro_session.get(x[0], 99), x[0]),
    )
    for type_name, recall in sorted_types:
        intro_s = agg.type_intro_session.get(type_name, -1)
        bar = "#" * int(recall * 20)
        print(f"  {type_name:>12} (s{intro_s+1}): {recall:>5.0%} |{bar:<20}|")

    # Summary statistics
    print("\n--- Summary ---")
    first_session_recall = agg.session_recall[0]
    last_session_recall = agg.session_recall[-1]
    retention_drop = first_session_recall - last_session_recall

    print(f"  Session 1 recall:  {first_session_recall:.0%}")
    print(f"  Session {agg.n_sessions} recall: {last_session_recall:.0%}")
    print(f"  Retention drop:    {retention_drop:+.0%}")

    if 0 in agg.retention_by_age:
        print(f"  Age 0 recall:      {agg.retention_by_age[0]:.0%} "
              f"(just written)")
    max_age = max(agg.retention_by_age.keys())
    if max_age > 0:
        print(f"  Age {max_age} recall:     "
              f"{agg.retention_by_age[max_age]:.0%} "
              f"(oldest facts)")

    # Update success
    update_recalls = [r for r in agg.session_update_recall if r >= 0]
    if update_recalls:
        avg_upd = sum(update_recalls) / len(update_recalls)
        print(f"  Avg update recall: {avg_upd:.0%}")

    # ── Paraphrase results (T_auto pattern completion) ──
    if agg.paraphrase_by_session:
        cats = ["aligned", "misaligned", "semantic"]
        print("\n--- Paraphrase Recall (7 base types, T_auto) ---")
        print(f"{'Session':>8}", end="")
        for cat in cats:
            print(f"  {cat:>12}", end="")
        print(f"  {'updated':>10} {'stable':>10}")
        print("-" * 72)

        for s in range(agg.n_sessions):
            cat_acc = agg.paraphrase_by_session[s]
            # Check if any paraphrase data exists for this session
            has_data = any(cat_acc.get(c, -1.0) >= 0 for c in cats)
            if not has_data:
                continue
            print(f"{s+1:>8}", end="")
            for cat in cats:
                v = cat_acc.get(cat, -1.0)
                if v >= 0:
                    print(f"  {v:>11.0%}", end="")
                else:
                    print(f"  {'---':>11}", end="")

            # Updated vs stable
            upd_v = agg.paraphrase_updated_by_session[s] if s < len(agg.paraphrase_updated_by_session) else -1.0
            stb_v = agg.paraphrase_stable_by_session[s] if s < len(agg.paraphrase_stable_by_session) else -1.0
            if upd_v >= 0:
                print(f"  {upd_v:>9.0%}", end="")
            else:
                print(f"  {'---':>9}", end="")
            if stb_v >= 0:
                print(f"  {stb_v:>9.0%}", end="")
            else:
                print(f"  {'---':>9}", end="")
            print()

        # Final session summary
        print(f"\n  Final session paraphrase:")
        for cat, acc in agg.paraphrase_by_category.items():
            print(f"    {cat:>12}: {acc:.0%}")
        print(f"    {'overall':>12}: {agg.paraphrase_overall:.0%}")

        # Updated vs stable at final session
        final_upd = agg.paraphrase_updated_by_session[-1] if agg.paraphrase_updated_by_session and agg.paraphrase_updated_by_session[-1] >= 0 else None
        final_stb = agg.paraphrase_stable_by_session[-1] if agg.paraphrase_stable_by_session and agg.paraphrase_stable_by_session[-1] >= 0 else None
        if final_upd is not None or final_stb is not None:
            print(f"\n  T_auto + erasure composition:")
            if final_stb is not None:
                print(f"    Para(stable):  {final_stb:.0%}")
            if final_upd is not None:
                print(f"    Para(updated): {final_upd:.0%}")


# ── Main entry point ─────────────────────────────────────────────────

def run_demo(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    write_mode: str = "individual",
    alpha: float = 0.5,
    trace_lr: float = 1.0,
    trace_decay: float = 0.99,
    use_ps: bool = True,
    erase_lr: float | None = None,
    seed: int = 42,
    verbose: bool = True,
    with_tauto: bool = False,
    completion_alpha: float = 0.3,
) -> AggregatedResults:
    """Run the full multi-session demo.

    Args:
        n_episodes: number of independent episodes (different entities)
        n_sessions: number of sessions per episode
        facts_per_session: facts written per session
        write_mode: "individual" (1 forward pass per fact) or "batch"
        alpha: trace alpha (0.5 optimal for GPT-2)
        trace_lr: trace learning rate
        trace_decay: trace decay rate per write
        use_ps: enable pattern separation (8x_k16)
        erase_lr: reconsolidation erasure strength for updates (None=disabled)
        seed: random seed
        verbose: print progress
        with_tauto: enable T_auto pattern completion for paraphrase support
        completion_alpha: weight for completion channel (default: 0.3)

    Returns:
        AggregatedResults with full metrics
    """
    device = get_device()
    t0 = time.time()

    if verbose:
        print(f"Device: {device}")
        print(f"Config: {n_episodes} episodes, {n_sessions} sessions, "
              f"{facts_per_session} facts/session")
        print(f"  alpha={alpha}, trace_lr={trace_lr}, decay={trace_decay}")
        print(f"  write_mode={write_mode}, pattern_sep={use_ps}"
              f"{f', erase_lr={erase_lr}' if erase_lr is not None else ''}")
        print()

    # ── Setup ──
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_extended_fact_types(tokenizer, verbose=verbose)

    n_types = len(fact_types)
    total_facts_needed = n_sessions * facts_per_session
    if verbose:
        print(f"\nAvailable: {n_types} types, need {total_facts_needed} "
              f"fact slots across {n_sessions} sessions")
        intro_sessions = -(-n_types // facts_per_session)  # ceil division
        update_sessions = max(0, n_sessions - intro_sessions)
        print(f"  Introduction phase: {intro_sessions} sessions "
              f"({n_types} types)")
        print(f"  Update phase: {update_sessions} sessions")

    # Build session schedule
    schedule = make_session_schedule(n_types, facts_per_session, n_sessions)

    # Initialize model
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=alpha,
        trace_lr=trace_lr, trace_decay=trace_decay,
    )
    model = model.to(device)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    if use_ps:
        model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
        if verbose:
            print("  Pattern separation: 8x_k16 enabled")

    # Prepare T_auto data if enabled
    auto_pairs_list: list[tuple[int, int]] | None = None
    q_variants: dict[str, list[QuestionVariant]] | None = None
    if with_tauto:
        auto_pair_objs = extract_auto_pairs(tokenizer)
        auto_pairs_list = [(p.variant_id, p.concept_id) for p in auto_pair_objs]
        q_variants = build_question_variants(tokenizer)
        if verbose:
            print(f"  T_auto: {len(auto_pairs_list)} pairs, "
                  f"completion_alpha={completion_alpha}")

    if verbose:
        print(f"\nRunning {n_episodes} episodes...")

    # ── Run episodes ──
    rng = random.Random(seed)
    episodes: list[EpisodeResult] = []

    for ep in range(n_episodes):
        ep_result = run_single_episode(
            model=model,
            tokenizer=tokenizer,
            fact_types=fact_types,
            schedule=schedule,
            n_sessions=n_sessions,
            device=device,
            rng=rng,
            write_mode=write_mode,
            erase_lr=erase_lr,
            auto_pairs=auto_pairs_list,
            completion_alpha=completion_alpha,
            question_variants=q_variants,
        )
        episodes.append(ep_result)

        if verbose and (ep + 1) % 10 == 0:
            # Quick running average
            last_recalls = [
                ep_result.sessions[-1].recall
            ]
            avg = sum(
                sum(r.values()) / len(r)
                for r in last_recalls
            ) / len(last_recalls)
            print(f"  Episode {ep+1}/{n_episodes} "
                  f"(final session recall: {avg:.0%})")

    # ── Aggregate ──
    agg = aggregate_results(episodes, schedule, fact_types)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nCompleted in {elapsed:.1f}s")
        print_results(agg)

    return agg


def compare_write_modes(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    seed: int = 42,
):
    """Compare individual vs batch write modes."""
    print("\n" + "=" * 70)
    print("WRITE MODE COMPARISON")
    print("=" * 70)

    for mode in ["individual", "batch"]:
        print(f"\n{'='*30} {mode.upper()} {'='*30}")
        run_demo(
            n_episodes=n_episodes,
            n_sessions=n_sessions,
            facts_per_session=facts_per_session,
            write_mode=mode,
            seed=seed,
        )


def compare_with_without_ps(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    seed: int = 42,
):
    """Compare with and without pattern separation."""
    print("\n" + "=" * 70)
    print("PATTERN SEPARATION COMPARISON")
    print("=" * 70)

    for use_ps in [True, False]:
        label = "WITH PS (8x_k16)" if use_ps else "NO PS"
        print(f"\n{'='*30} {label} {'='*30}")
        run_demo(
            n_episodes=n_episodes,
            n_sessions=n_sessions,
            facts_per_session=facts_per_session,
            use_ps=use_ps,
            seed=seed,
        )


def sweep_erase_lr(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    seed: int = 42,
):
    """Sweep erase_lr to find optimal value for updates."""
    print("\n" + "=" * 70)
    print("RECONSOLIDATION ERASURE SWEEP")
    print("=" * 70)

    erase_values = [None, 0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    for elr in erase_values:
        label = f"erase_lr={elr}" if elr is not None else "NO ERASE"
        print(f"\n{'='*30} {label} {'='*30}")
        agg = run_demo(
            n_episodes=n_episodes,
            n_sessions=n_sessions,
            facts_per_session=facts_per_session,
            erase_lr=elr,
            seed=seed,
            verbose=False,
        )
        final = agg.session_recall[-1]
        upd_recalls = [r for r in agg.session_update_recall if r >= 0]
        avg_upd = sum(upd_recalls) / len(upd_recalls) if upd_recalls else 0
        old_recalls = [agg.session_old_recall[s] for s in range(n_sessions)
                       if agg.session_update_recall[s] >= 0]
        avg_old = sum(old_recalls) / len(old_recalls) if old_recalls else 0

        results.append((elr, final, avg_upd, avg_old,
                         agg.session_trace_norm[-1]))
        print(f"  Final: {final:.0%}, Update: {avg_upd:.0%}, "
              f"Old: {avg_old:.0%}, Trace: {agg.session_trace_norm[-1]:.1f}")

    # Summary table
    print("\n" + "=" * 70)
    print("ERASE SWEEP SUMMARY")
    print(f"{'erase_lr':>10} {'Final':>8} {'Update':>8} "
          f"{'Old':>8} {'Trace':>8}")
    print("-" * 46)
    for elr, final, upd, old, tnorm in results:
        label = f"{elr}" if elr is not None else "None"
        print(f"{label:>10} {final:>7.0%} {upd:>7.0%} "
              f"{old:>7.0%} {tnorm:>7.1f}")


# ── T_auto comparison ────────────────────────────────────────────────

def compare_with_without_tauto(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    erase_lr: float | None = 5.0,
    completion_alpha: float = 0.3,
    seed: int = 42,
):
    """Compare standard eval WITH vs WITHOUT T_auto pattern completion.

    Tests:
    1. Standard recall: should be ~0pp change (T_auto only adds completion channel)
    2. Paraphrase recall: misaligned/semantic variants should improve dramatically
    3. T_auto + erasure: paraphrased queries on updated facts return NEW value
    """
    print("\n" + "=" * 70)
    print("T_AUTO COMPARISON (with vs without pattern completion)")
    print(f"  {n_episodes} episodes, {n_sessions} sessions, "
          f"{facts_per_session} facts/session")
    print(f"  erase_lr={erase_lr}, completion_alpha={completion_alpha}")
    print("=" * 70)

    # WITHOUT T_auto
    print(f"\n{'─'*30} WITHOUT T_AUTO {'─'*30}")
    agg_no = run_demo(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        erase_lr=erase_lr,
        with_tauto=False,
        seed=seed,
    )

    # WITH T_auto
    print(f"\n{'─'*30} WITH T_AUTO {'─'*30}")
    agg_yes = run_demo(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        erase_lr=erase_lr,
        with_tauto=True,
        completion_alpha=completion_alpha,
        seed=seed,
    )

    # ── Comparison summary ──
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Standard recall side-by-side
    print("\n--- Standard Recall (24 types) ---")
    print(f"{'Session':>8} {'No T_auto':>12} {'With T_auto':>12} {'Delta':>8}")
    print("-" * 44)
    for s in range(n_sessions):
        r_no = agg_no.session_recall[s]
        r_yes = agg_yes.session_recall[s]
        delta = r_yes - r_no
        sign = "+" if delta >= 0 else ""
        print(f"{s+1:>8} {r_no:>11.0%} {r_yes:>11.0%} {sign}{delta:>6.0%}")

    # Paraphrase summary (only WITH T_auto has these)
    if agg_yes.paraphrase_by_category:
        print("\n--- Paraphrase Recall at Final Session (7 base types) ---")
        for cat, acc in agg_yes.paraphrase_by_category.items():
            print(f"  {cat:>12}: {acc:.0%}")
        print(f"  {'overall':>12}: {agg_yes.paraphrase_overall:.0%}")

    # T_auto + erasure composition
    if agg_yes.paraphrase_updated_by_session:
        final_upd = agg_yes.paraphrase_updated_by_session[-1]
        final_stb = agg_yes.paraphrase_stable_by_session[-1]
        if final_upd >= 0 or final_stb >= 0:
            print("\n--- T_auto + Erasure Composition ---")
            if final_stb >= 0:
                print(f"  Para(stable facts):  {final_stb:.0%}")
            if final_upd >= 0:
                print(f"  Para(updated facts): {final_upd:.0%}")
                if final_stb >= 0:
                    delta = final_upd - final_stb
                    sign = "+" if delta >= 0 else ""
                    print(f"  Delta:               {sign}{delta:.0%}")

    return agg_no, agg_yes


# ── Paragraph mode functions ─────────────────────────────────────────

def run_paragraph_demo(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    alpha: float = 0.5,
    trace_lr: float = 1.0,
    trace_decay: float = 0.99,
    use_ps: bool = True,
    erase_lr: float | None = 5.0,
    filler_mode: str = "mixed",
    n_filler: int = 3,
    seed: int = 42,
    verbose: bool = True,
    with_tauto: bool = False,
    completion_alpha: float = 0.3,
) -> AggregatedResults:
    """Run multi-session demo with paragraph-level writes and dual gates.

    New facts are written as paragraphs (facts + filler), using dual gates
    to filter noise. Updates use individual writes with hardcoded mask + erase.

    If with_tauto=True, also evaluates paraphrase recall using T_auto pattern
    completion (autoassociative trace for misaligned/semantic query resolution).
    """
    device = get_device()
    t0 = time.time()

    if verbose:
        print(f"Device: {device}")
        print(f"Config: {n_episodes} episodes, {n_sessions} sessions, "
              f"{facts_per_session} facts/session")
        print(f"  alpha={alpha}, trace_lr={trace_lr}, decay={trace_decay}")
        print(f"  write_mode=paragraph, filler={filler_mode}({n_filler}), "
              f"pattern_sep={use_ps}"
              f"{f', erase_lr={erase_lr}' if erase_lr is not None else ''}")
        print()

    # ── Setup ──
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_extended_fact_types(tokenizer, verbose=verbose)

    n_types = len(fact_types)
    if verbose:
        intro_sessions = -(-n_types // facts_per_session)
        update_sessions = max(0, n_sessions - intro_sessions)
        print(f"\nIntroduction: {intro_sessions} sessions ({n_types} types)")
        print(f"Update: {update_sessions} sessions")

    schedule = make_session_schedule(n_types, facts_per_session, n_sessions)

    # Initialize model
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64, alpha=alpha,
        trace_lr=trace_lr, trace_decay=trace_decay,
    )
    model = model.to(device)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    if use_ps:
        model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)
        if verbose:
            print("  Pattern separation: 8x_k16 enabled")

    # ── Train dual gates ──
    gate_key_values = train_dual_gates(
        model, tokenizer, device, seed=seed, verbose=verbose)

    # Enable dual gate mode
    model.set_dual_gate_mode(True)

    # ── Diagnostic: test gate on all fact sentences and filler ──
    if verbose:
        print("\n--- Gate diagnostic: fact sentences ---")
        wte_diag = model.gpt2.transformer.wte
        from ..gpt2_tasks import FILLER_NO_LINK, FILLER_WITH_LINK
        for ft in fact_types:
            ename, eid = ft.entities[0]
            text = ft.fact_templates[0].text.replace("{X}", ename)
            ids = tokenizer.encode(text, add_special_tokens=False)
            t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                gp = model.trace.compute_gate(wte_diag, t)
                gk = model.trace.compute_gate_key(wte_diag, t)
                S = len(ids)
                if S > 2:
                    comb = gp[0, 1:-1] * gk[0, :-2]
                    mc = comb.max().item()
                else:
                    mc = 0
            tokens = [tokenizer.decode([i]) for i in ids]
            status = "WRITE" if mc > 0.010 else "SKIP!"
            print(f"  {ft.name:>12}: max_comb={mc:.4f} [{status}] "
                  f"{text[:40]}")
        print("--- Filler sentences ---")
        for text in FILLER_WITH_LINK[:5]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                gp = model.trace.compute_gate(wte_diag, t)
                gk = model.trace.compute_gate_key(wte_diag, t)
                S = len(ids)
                if S > 2:
                    comb = gp[0, 1:-1] * gk[0, :-2]
                    mc = comb.max().item()
                else:
                    mc = 0
            status = "WRITE" if mc > 0.010 else "SKIP"
            print(f"  filler: max_comb={mc:.4f} [{status}] {text[:40]}")

    # ── Prepare T_auto data if enabled ──
    auto_pairs_list: list[tuple[int, int]] | None = None
    q_variants: dict[str, list[QuestionVariant]] | None = None
    if with_tauto:
        auto_pair_objs = extract_auto_pairs(tokenizer)
        auto_pairs_list = [(p.variant_id, p.concept_id) for p in auto_pair_objs]
        q_variants = build_question_variants(tokenizer)
        if verbose:
            print(f"  T_auto: {len(auto_pairs_list)} pairs, "
                  f"completion_alpha={completion_alpha}")

    if verbose:
        tauto_str = f", T_auto={with_tauto}" if with_tauto else ""
        print(f"\nRunning {n_episodes} episodes (paragraph mode{tauto_str})...")

    # ── Run episodes ──
    rng = random.Random(seed)
    filler_rng = random.Random(seed + 999999)
    episodes: list[EpisodeResult] = []

    for ep in range(n_episodes):
        ep_result = run_single_episode(
            model=model,
            tokenizer=tokenizer,
            fact_types=fact_types,
            schedule=schedule,
            n_sessions=n_sessions,
            device=device,
            rng=rng,
            write_mode="paragraph",
            erase_lr=erase_lr,
            filler_mode=filler_mode,
            n_filler_per_session=n_filler,
            filler_rng=filler_rng,
            auto_pairs=auto_pairs_list,
            completion_alpha=completion_alpha,
            question_variants=q_variants,
        )
        episodes.append(ep_result)

        if verbose and (ep + 1) % 10 == 0:
            last_recall = ep_result.sessions[-1].recall
            avg = sum(last_recall.values()) / len(last_recall)
            print(f"  Episode {ep+1}/{n_episodes} "
                  f"(final session recall: {avg:.0%})")

    # ── Aggregate ──
    agg = aggregate_results(episodes, schedule, fact_types)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nCompleted in {elapsed:.1f}s")
        print_results(agg)

    return agg


def compare_paragraph_vs_individual(
    n_episodes: int = 30,
    n_sessions: int = 5,
    facts_per_session: int = 5,
    erase_lr: float | None = 5.0,
    seed: int = 42,
):
    """Compare paragraph write (dual gate) vs individual write."""
    print("\n" + "=" * 70)
    print("PARAGRAPH vs INDIVIDUAL COMPARISON")
    print("=" * 70)

    # Individual mode (baseline)
    print(f"\n{'='*30} INDIVIDUAL {'='*30}")
    agg_ind = run_demo(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        write_mode="individual",
        erase_lr=erase_lr,
        seed=seed,
    )

    # Paragraph mode with mixed filler
    print(f"\n{'='*30} PARAGRAPH (mixed_3) {'='*30}")
    agg_para = run_paragraph_demo(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        erase_lr=erase_lr,
        filler_mode="mixed",
        n_filler=3,
        seed=seed,
    )

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print(f"{'Session':>8} {'Individual':>12} {'Paragraph':>12} {'Delta':>8}")
    print("-" * 44)
    for s in range(n_sessions):
        ind_r = agg_ind.session_recall[s]
        par_r = agg_para.session_recall[s]
        delta = par_r - ind_r
        print(f"{s+1:>8} {ind_r:>11.0%} {par_r:>11.0%} {delta:>+7.0%}")

    # Final session
    print("-" * 44)
    final_ind = agg_ind.session_recall[-1]
    final_par = agg_para.session_recall[-1]
    print(f"{'Final':>8} {final_ind:>11.0%} {final_par:>11.0%} "
          f"{final_par - final_ind:>+7.0%}")


def compare_filler_modes(
    n_episodes: int = 30,
    n_sessions: int = 5,
    facts_per_session: int = 5,
    seed: int = 42,
):
    """Compare different filler modes in paragraph write."""
    print("\n" + "=" * 70)
    print("FILLER MODE COMPARISON (paragraph write)")
    print("=" * 70)

    conditions = [
        ("none_0", "none", 0),
        ("safe_3", "safe", 3),
        ("noisy_3", "noisy", 3),
        ("mixed_3", "mixed", 3),
        ("noisy_5", "noisy", 5),
    ]
    results = []

    for cond_name, fmode, nfill in conditions:
        print(f"\n{'='*25} {cond_name} {'='*25}")
        agg = run_paragraph_demo(
            n_episodes=n_episodes,
            n_sessions=n_sessions,
            facts_per_session=facts_per_session,
            erase_lr=5.0,
            filler_mode=fmode,
            n_filler=nfill,
            seed=seed,
            verbose=False,
        )
        final = agg.session_recall[-1]
        results.append((cond_name, final, agg))
        print(f"  Final recall: {final:.0%}")

    # Summary table
    print("\n" + "=" * 70)
    print("FILLER MODE SUMMARY")
    print(f"{'Condition':>12} ", end="")
    for s in range(n_sessions):
        print(f"{'S'+str(s+1):>6}", end="")
    print(f"  {'Final':>6}")
    print("-" * (14 + 6 * n_sessions + 8))
    for cond_name, final, agg in results:
        print(f"{cond_name:>12} ", end="")
        for s in range(n_sessions):
            print(f"{agg.session_recall[s]:>5.0%}", end=" ")
        print(f"  {final:>5.0%}")


def compare_paragraph_tauto(
    n_episodes: int = 30,
    n_sessions: int = 10,
    facts_per_session: int = 5,
    erase_lr: float | None = 5.0,
    completion_alpha: float = 0.3,
    filler_mode: str = "mixed",
    n_filler: int = 3,
    seed: int = 42,
):
    """Compare paragraph mode WITH vs WITHOUT T_auto pattern completion.

    Three-way composition test: dual gates (filter filler) + T_auto (resolve
    paraphrases) + erasure (clean updates). The full system.

    Tests:
    1. Standard recall: should be ~0pp change (paragraph + T_auto vs paragraph alone)
    2. Paraphrase recall: misaligned/semantic variants on paragraph-written facts
    3. T_auto + erasure + paragraph: updated facts via paraphrased queries
    4. Filler noise interaction: does noisy filler degrade paraphrase resolution?
    """
    print("\n" + "=" * 70)
    print("PARAGRAPH + T_AUTO COMPARISON")
    print(f"  {n_episodes} episodes, {n_sessions} sessions, "
          f"{facts_per_session} facts/session")
    print(f"  erase_lr={erase_lr}, completion_alpha={completion_alpha}")
    print(f"  filler={filler_mode}({n_filler})")
    print("=" * 70)

    # WITHOUT T_auto (paragraph baseline)
    print(f"\n{'─'*25} PARAGRAPH WITHOUT T_AUTO {'─'*25}")
    agg_no = run_paragraph_demo(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        erase_lr=erase_lr,
        filler_mode=filler_mode,
        n_filler=n_filler,
        with_tauto=False,
        seed=seed,
    )

    # WITH T_auto (three-way composition)
    print(f"\n{'─'*25} PARAGRAPH WITH T_AUTO {'─'*25}")
    agg_yes = run_paragraph_demo(
        n_episodes=n_episodes,
        n_sessions=n_sessions,
        facts_per_session=facts_per_session,
        erase_lr=erase_lr,
        filler_mode=filler_mode,
        n_filler=n_filler,
        with_tauto=True,
        completion_alpha=completion_alpha,
        seed=seed,
    )

    # ── Comparison summary ──
    print("\n" + "=" * 70)
    print("PARAGRAPH + T_AUTO COMPARISON SUMMARY")
    print("=" * 70)

    # Standard recall side-by-side
    print("\n--- Standard Recall (24 types, paragraph write) ---")
    print(f"{'Session':>8} {'No T_auto':>12} {'With T_auto':>12} {'Delta':>8}")
    print("-" * 44)
    for s in range(n_sessions):
        r_no = agg_no.session_recall[s]
        r_yes = agg_yes.session_recall[s]
        delta = r_yes - r_no
        sign = "+" if delta >= 0 else ""
        print(f"{s+1:>8} {r_no:>11.0%} {r_yes:>11.0%} {sign}{delta:>6.0%}")

    # Paraphrase summary
    if agg_yes.paraphrase_by_category:
        print("\n--- Paraphrase Recall at Final Session (7 base types) ---")
        for cat, acc in agg_yes.paraphrase_by_category.items():
            print(f"  {cat:>12}: {acc:.0%}")
        print(f"  {'overall':>12}: {agg_yes.paraphrase_overall:.0%}")

    # Paraphrase by session (aligned / misaligned / semantic)
    if agg_yes.paraphrase_by_category:
        print("\n--- Paraphrase by Session ---")
        cats = list(agg_yes.paraphrase_by_category.keys())
        hdr = f"{'Session':>8}"
        for cat in cats:
            hdr += f" {cat:>12}"
        hdr += f" {'overall':>10}"
        print(hdr)
        print("-" * len(hdr))

        # Walk through per-session paraphrase data from episodes
        # Re-aggregate from episodes for session-level view
        # (agg only stores final session paraphrase, so compute per-session here)

    # T_auto + erasure + paragraph composition
    if agg_yes.paraphrase_updated_by_session:
        final_upd = agg_yes.paraphrase_updated_by_session[-1]
        final_stb = agg_yes.paraphrase_stable_by_session[-1]
        if final_upd >= 0 or final_stb >= 0:
            print("\n--- Three-Way Composition: Dual Gates + T_auto + Erasure ---")
            if final_stb >= 0:
                print(f"  Para(stable facts):  {final_stb:.0%}")
            if final_upd >= 0:
                print(f"  Para(updated facts): {final_upd:.0%}")
                if final_stb >= 0:
                    delta = final_upd - final_stb
                    sign = "+" if delta >= 0 else ""
                    print(f"  Delta:               {sign}{delta:.0%}")

    # Cross-reference with standard (non-paragraph) T_auto
    print("\n--- Reference: Standard (individual) + T_auto ---")
    print("  (Run --compare-tauto for full individual-mode comparison)")

    return agg_no, agg_yes


def main():
    parser = argparse.ArgumentParser(
        description="Exp 16: Multi-session trace memory demo")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (10 episodes, 5 sessions)")
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="Number of episodes (default: 30)")
    parser.add_argument("--n-sessions", type=int, default=10,
                        help="Number of sessions per episode (default: 10)")
    parser.add_argument("--facts-per-session", type=int, default=5,
                        help="Facts per session (default: 5)")
    parser.add_argument("--write-mode", default="individual",
                        choices=["individual", "batch"],
                        help="Write mode (default: individual)")
    parser.add_argument("--compare-write", action="store_true",
                        help="Compare individual vs batch write modes")
    parser.add_argument("--compare-ps", action="store_true",
                        help="Compare with vs without pattern separation")
    parser.add_argument("--sweep-erase", action="store_true",
                        help="Sweep erase_lr values for update optimization")
    parser.add_argument("--erase-lr", type=float, default=None,
                        help="Reconsolidation erasure strength (default: None)")
    parser.add_argument("--no-ps", action="store_true",
                        help="Disable pattern separation")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Trace alpha (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    # Paragraph mode flags
    parser.add_argument("--paragraph", action="store_true",
                        help="Use paragraph write with dual gates")
    parser.add_argument("--filler-mode", default="mixed",
                        choices=["none", "safe", "noisy", "mixed"],
                        help="Filler type for paragraph mode (default: mixed)")
    parser.add_argument("--n-filler", type=int, default=3,
                        help="Filler sentences per session paragraph (default: 3)")
    parser.add_argument("--compare-paragraph", action="store_true",
                        help="Compare paragraph vs individual write modes")
    parser.add_argument("--compare-filler", action="store_true",
                        help="Compare filler modes in paragraph write")
    # T_auto pattern completion flags
    parser.add_argument("--with-tauto", action="store_true",
                        help="Enable T_auto pattern completion")
    parser.add_argument("--completion-alpha", type=float, default=0.3,
                        help="Completion channel weight (default: 0.3)")
    parser.add_argument("--compare-tauto", action="store_true",
                        help="Compare with vs without T_auto pattern completion")
    parser.add_argument("--compare-paragraph-tauto", action="store_true",
                        help="Compare paragraph mode with vs without T_auto")
    args = parser.parse_args()

    if args.quick:
        args.n_episodes = 10
        args.n_sessions = 5

    if args.compare_write:
        compare_write_modes(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            seed=args.seed,
        )
    elif args.compare_ps:
        compare_with_without_ps(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            seed=args.seed,
        )
    elif args.sweep_erase:
        sweep_erase_lr(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            seed=args.seed,
        )
    elif args.compare_paragraph:
        compare_paragraph_vs_individual(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            erase_lr=args.erase_lr if args.erase_lr is not None else 5.0,
            seed=args.seed,
        )
    elif args.compare_filler:
        compare_filler_modes(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            seed=args.seed,
        )
    elif args.compare_tauto:
        compare_with_without_tauto(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            erase_lr=args.erase_lr if args.erase_lr is not None else 5.0,
            completion_alpha=args.completion_alpha,
            seed=args.seed,
        )
    elif args.compare_paragraph_tauto:
        compare_paragraph_tauto(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            erase_lr=args.erase_lr if args.erase_lr is not None else 5.0,
            completion_alpha=args.completion_alpha,
            filler_mode=args.filler_mode,
            n_filler=args.n_filler,
            seed=args.seed,
        )
    elif args.paragraph:
        run_paragraph_demo(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            alpha=args.alpha,
            trace_lr=1.0,
            trace_decay=0.99,
            use_ps=not args.no_ps,
            erase_lr=args.erase_lr if args.erase_lr is not None else 5.0,
            filler_mode=args.filler_mode,
            n_filler=args.n_filler,
            seed=args.seed,
            with_tauto=args.with_tauto,
            completion_alpha=args.completion_alpha,
        )
    else:
        run_demo(
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
            facts_per_session=args.facts_per_session,
            write_mode=args.write_mode,
            alpha=args.alpha,
            trace_lr=1.0,
            trace_decay=0.99,
            use_ps=not args.no_ps,
            erase_lr=args.erase_lr,
            seed=args.seed,
            with_tauto=args.with_tauto,
            completion_alpha=args.completion_alpha,
        )


if __name__ == "__main__":
    main()
