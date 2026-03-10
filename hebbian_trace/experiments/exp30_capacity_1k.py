"""Exp 30: Capacity scaling to 1000 facts with hashed trace banks.

Auto-discovers concept tokens from model vocabulary (no hardcoded list).
Tests scaling curves: baseline vs 16/64/128 banks at 100-1000 facts.
Supports GPT-2 (local) and LLaMA-2 7B (A100).

Extends exp28 (hashed banks) with 10x larger concept pool.

Usage:
    # GPT-2 Small (local, quick sanity check)
    python -m hebbian_trace.experiments.exp30_capacity_1k --quick

    # GPT-2 with full sweep
    python -m hebbian_trace.experiments.exp30_capacity_1k --n-eval 30

    # LLaMA-2 7B on A100 (32 heads for d_trace bottleneck fix)
    python -m hebbian_trace.experiments.exp30_capacity_1k \\
        --model meta-llama/Llama-2-7b-hf --alpha 20 --n-trace-heads 32 \\
        --banks 1 16 64 128 --n-eval 50

    # Custom max facts and banks
    python -m hebbian_trace.experiments.exp30_capacity_1k \\
        --max-facts 2000 --banks 1 128 256
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Conditional import: use package import when run as module,
# direct import when run as standalone script on remote server.
try:
    from ..gpt2_trace import GPT2WithTrace
except ImportError:
    from hebbian_trace.gpt2_trace import GPT2WithTrace


# ── Inlined from gpt2_tasks.py (avoids nlp_tasks.py dependency) ───

@dataclass
class GPT2FactTemplate:
    text: str
    linking_word: str

@dataclass
class GPT2QuestionTemplate:
    text: str

@dataclass
class GPT2FactType:
    name: str
    entities: list[tuple[str, int]]
    fact_templates: list[GPT2FactTemplate]
    question_templates: list[GPT2QuestionTemplate]


def get_linking_bpe_ids(tokenizer) -> list[int]:
    linking_words = ["is", "in", "at", "from", ":", "am", "me"]
    ids = []
    for word in linking_words:
        bpe_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(bpe_ids) == 1:
            ids.append(bpe_ids[0])
    return ids


def _predict_answer(model, query_ids: list[int],
                    entity_ids: list[int]) -> int:
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
    pred_logits = logits[0, -1, :]
    entity_logits = pred_logits[entity_ids]
    best_pos = entity_logits.argmax().item()
    return entity_ids[best_pos]


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
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


# ── Model configs (known alpha values from exp_scaling.py) ─────────

MODEL_CONFIGS = {
    "gpt2": {"alpha": 0.5, "dtype": None, "n_trace_heads": 8},
    "gpt2-medium": {"alpha": 0.5, "dtype": None, "n_trace_heads": 8},
    "microsoft/phi-2": {"alpha": 50.0, "dtype": "float16", "n_trace_heads": 8},
    "meta-llama/Llama-2-7b-hf": {
        "alpha": 20.0, "dtype": "float16", "n_trace_heads": 32,
    },
}


# ── Load HF token from .env ────────────────────────────────────────

def _load_hf_token():
    """Load HF_TOKEN from .env if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    for env_path in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = (
                        line.split("=", 1)[1].strip().strip('"').strip("'"))
                    return


# ── Model loading ───────────────────────────────────────────────────

def load_model(
    model_name: str = "gpt2",
    alpha: float | None = None,
    n_trace_heads: int | None = None,
    d_trace: int = 64,
    device: torch.device | None = None,
) -> tuple[GPT2WithTrace, AutoTokenizer]:
    """Load model + trace with appropriate settings.

    Uses MODEL_CONFIGS for defaults; CLI args override.
    """
    _load_hf_token()

    cfg = MODEL_CONFIGS.get(model_name, {})
    if alpha is None:
        alpha = cfg.get("alpha", 0.5)
    if n_trace_heads is None:
        n_trace_heads = cfg.get("n_trace_heads", 8)
    dtype_str = cfg.get("dtype")
    torch_dtype = getattr(torch, dtype_str) if dtype_str else None

    if device is None:
        device = get_device()

    print(f"Loading {model_name}...")
    t0 = time.time()

    model = GPT2WithTrace(
        n_trace_heads=n_trace_heads,
        d_trace=d_trace,
        alpha=alpha,
        trace_lr=1.0,
        trace_decay=0.99,
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=str(device),
    )

    dt = time.time() - t0
    config = model.base_model.config
    d_model = config.hidden_size
    n_layers = getattr(config, 'n_layer',
                       getattr(config, 'num_hidden_layers', -1))

    print(f"  Loaded in {dt:.1f}s")
    print(f"  d_model={d_model}, n_layers={n_layers}")
    print(f"  alpha={alpha}, heads={n_trace_heads}×d{d_trace}")
    print(f"  Base params: {sum(p.numel() for p in model.base_model.parameters()):,}")
    print(f"  Trace params: {sum(p.numel() for p in model.trace.parameters()):,}")

    # Zero-shot: no learned gates, hardcoded linking mask only
    model.set_gate_mode(use_learned_gate=False)
    model.set_dual_gate_mode(enabled=False)

    # Pattern separation 8x_k16
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    # Set linking tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    return model, tokenizer


# ── Auto-discover concept tokens from vocabulary ───────────────────

def discover_concept_tokens(
    tokenizer,
    n_target: int = 1200,
    exclude_names: set[str] | None = None,
    min_len: int = 3,
    max_len: int = 8,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Find single-token words suitable as concept addresses.

    Works with any tokenizer (BPE, SentencePiece, etc.).
    """
    if exclude_names is None:
        exclude_names = set()
    exclude_lower = {n.lower() for n in exclude_names}

    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "got", "let", "say", "she", "too", "use", "him", "his",
        "own", "than", "them", "then", "they", "this", "that", "what",
        "when", "will", "with", "been", "from", "have", "here", "just",
        "like", "long", "make", "many", "more", "most", "much", "must",
        "only", "over", "some", "such", "take", "very", "well", "were",
        "your", "into", "also", "back", "been", "come", "each", "even",
        "give", "good", "great", "keep", "last", "look", "made", "need",
        "still", "want", "where", "which", "while", "would", "about",
        "after", "being", "could", "every", "first", "found", "going",
        "might", "never", "other", "right", "shall", "since", "their",
        "there", "these", "thing", "think", "those", "under", "using",
        "would", "does", "done", "down", "else", "ever",
    }

    candidates: list[tuple[str, int]] = []

    for token_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([token_id]).strip()
        word = decoded.lower()

        if not word.isalpha():
            continue
        if len(word) < min_len or len(word) > max_len:
            continue
        if word in stop_words:
            continue
        if word in exclude_lower:
            continue

        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            continue

        candidates.append((word, ids[0]))

    # Deduplicate by word
    seen_words: set[str] = set()
    unique: list[tuple[str, int]] = []
    for word, tid in candidates:
        if word not in seen_words:
            seen_words.add(word)
            unique.append((word, tid))

    rng = random.Random(seed)
    rng.shuffle(unique)

    if len(unique) < n_target:
        print(f"  WARNING: only {len(unique)} concept tokens found "
              f"(target: {n_target})")

    return unique[:n_target]


# ── Build entity pool for any tokenizer ─────────────────────────────

# Canonical entity names (superset of all exp16 pools)
ENTITY_POOL_CANDIDATES = [
    # Names
    "Elena", "John", "Sarah", "Marco", "Omar", "Alice", "Bob", "Charlie",
    "David", "Anna", "Carlos", "Grace", "Henry", "Julia", "Kevin", "Laura",
    "Ming", "Nora", "Pedro", "Quinn",
    # Cities
    "Paris", "London", "Berlin", "Tokyo", "Cairo", "Rome", "Seoul", "Lima",
    "Dublin", "Oslo", "Lagos", "Hanoi", "Osaka", "Milan",
    # Companies
    "Google", "Apple", "Amazon", "Oracle", "Adobe",
    # Colors
    "red", "blue", "green", "black", "white", "gray", "pink", "brown",
    "purple", "orange",
    # Food
    "pizza", "pasta", "sushi", "bread", "cheese", "rice", "soup",
    # Animals
    "lion", "tiger", "bear", "eagle", "wolf", "hawk", "fox", "deer",
    "elk", "owl", "seal",
    # Drinks
    "coffee", "tea", "water", "juice", "wine", "beer", "milk", "soda",
    # Sports
    "tennis", "golf", "soccer", "baseball", "hockey", "boxing",
    "rugby", "cricket", "skiing",
    # Hobbies
    "reading", "swimming", "cooking", "painting", "hiking",
    "dancing", "singing", "running", "writing", "fishing",
    # Numbers
    "seven", "eight", "nine", "three", "four", "five", "six", "ten",
    "eleven", "twelve", "zero",
    # Languages
    "English", "French", "Russian", "Spanish", "German",
    "Chinese", "Arabic", "Korean", "Italian", "Greek",
    # Instruments
    "piano", "guitar", "violin", "drums", "trumpet",
    # Seasons
    "spring", "summer", "autumn", "winter",
    # Subjects
    "math", "science", "history", "art", "music", "biology",
    "physics", "chemistry", "english",
    # Days
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    # Cars
    "Toyota", "Honda", "Ford", "Tesla", "Volvo", "Audi", "BMW",
    "Lexus", "Mazda",
    # Metals
    "gold", "silver", "iron", "copper", "steel", "bronze", "tin",
    "zinc", "lead",
    # Gems
    "diamond", "ruby", "pearl", "amber",
    # Trees
    "oak", "pine", "maple", "palm", "bamboo",
    # Fruits
    "apple", "banana", "mango", "peach", "cherry", "grape", "lemon", "lime",
    # Tools
    "hammer", "wrench", "drill", "saw", "shovel",
    # Fabrics
    "cotton", "silk", "wool", "leather", "linen", "nylon", "denim", "satin",
    # Countries
    "France", "Japan", "Brazil", "Canada", "India", "Egypt",
    "Mexico", "Spain",
    # Pets
    "cat", "dog", "hamster", "parrot", "rabbit", "turtle", "fish",
]


def build_entity_pool(tokenizer) -> list[tuple[str, int]]:
    """Validate entity pool for this specific tokenizer.

    Returns (entity_name, bpe_token_id) for single-token entities.
    """
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    pool: list[tuple[str, int]] = []

    for name in ENTITY_POOL_CANDIDATES:
        if name.lower() in seen_names:
            continue
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in seen_ids:
            pool.append((name, ids[0]))
            seen_ids.add(ids[0])
            seen_names.add(name.lower())

    return pool


def build_scaled_fact_types(
    tokenizer,
    n_target: int = 1200,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[GPT2FactType], list[int]]:
    """Build up to n_target fact types using auto-discovered concept tokens.

    Returns (fact_types, entity_ids).
    """
    # Build entity pool validated for this tokenizer
    entity_pool = build_entity_pool(tokenizer)
    entity_ids = sorted({eid for _, eid in entity_pool})

    if verbose:
        print(f"  Entity pool: {len(entity_pool)} single-token entities")

    # Collect entity names to exclude from concept discovery
    entity_names = {name.lower() for name, _ in entity_pool}
    linking_words = {"is", "in", "at", "from", "am", "me"}
    exclude = entity_names | linking_words

    # Discover concept tokens
    concepts = discover_concept_tokens(
        tokenizer, n_target + 100,
        exclude_names=exclude,
        seed=seed,
    )

    # Build fact types
    fact_types: list[GPT2FactType] = []
    for word, tid in concepts:
        if len(fact_types) >= n_target:
            break
        fact_types.append(GPT2FactType(
            name=word,
            entities=entity_pool,
            fact_templates=[GPT2FactTemplate(f"My {word} is {{X}}.", "is")],
            question_templates=[
                GPT2QuestionTemplate(f"What is my {word}?")],
        ))

    if verbose:
        print(f"  Fact types: {len(fact_types)} (auto-discovered)")

    return fact_types, entity_ids


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_scaling(
    model: GPT2WithTrace,
    n_facts: int,
    fact_types: list[GPT2FactType],
    entity_ids: list[int],
    tokenizer,
    n_episodes: int = 30,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Evaluate cross-context accuracy at n_facts using direct write."""
    device = next(model.parameters()).device
    model.eval()

    total_correct = 0
    total_queries = 0
    per_episode: list[float] = []

    for ep in range(n_episodes):
        ep_rng = random.Random(seed + ep)
        model.reset_traces()

        selected = ep_rng.sample(fact_types, min(n_facts, len(fact_types)))
        if n_facts > len(fact_types):
            selected += [ep_rng.choice(fact_types)
                         for _ in range(n_facts - len(fact_types))]

        facts = []
        for ft in selected:
            ent_name, ent_id = ep_rng.choice(ft.entities)
            concept_ids = tokenizer.encode(
                " " + ft.name, add_special_tokens=False)
            concept_id = concept_ids[0] if len(concept_ids) == 1 else None
            q_ids = tokenizer.encode(
                ft.question_templates[0].text, add_special_tokens=False)
            facts.append((concept_id, ent_id, q_ids))

        # Write phase — direct write
        model.set_trace_mode(use=False, update=False)
        for concept_id, ent_id, _ in facts:
            if concept_id is not None:
                model.write_fact_direct(concept_id, ent_id)

        # Read phase
        model.set_trace_mode(use=True, update=False)
        ep_correct = 0
        for _, answer_id, q_ids in facts:
            pred_id = _predict_answer(model, q_ids, entity_ids)
            if pred_id == answer_id:
                ep_correct += 1
            total_queries += 1

        total_correct += ep_correct
        per_episode.append(ep_correct / max(len(facts), 1))

    return total_correct / max(total_queries, 1), per_episode


# ── Bank distribution diagnostic ────────────────────────────────────

def diagnose_bank_distribution(
    model: GPT2WithTrace,
    fact_types: list[GPT2FactType],
    tokenizer,
    n_banks: int,
    n_concepts: int = 200,
):
    """Show how concept tokens distribute across banks."""
    model.trace.set_bank_mode(n_banks)
    model.reset_traces()

    bank_counts: Counter = Counter()
    n_checked = 0
    for ft in fact_types[:n_concepts]:
        ids = tokenizer.encode(" " + ft.name, add_special_tokens=False)
        if len(ids) != 1:
            continue
        Q = model.trace.compute_q_for_token(model._wte, ids[0])
        if model.trace._pattern_sep_enabled:
            Q_exp = model.trace._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2)).squeeze(0).squeeze(1)
        else:
            Q_exp = Q
        bid = model.trace._compute_bank_id(Q_exp)
        bank_counts[bid] += 1
        n_checked += 1

    print(f"\n  Bank distribution ({n_checked} concepts, {n_banks} banks):")
    if bank_counts:
        min_b = min(bank_counts.values())
        max_b = max(bank_counts.values())
        empty = n_banks - len(bank_counts)
        print(f"    Min/Max per bank: {min_b}/{max_b}")
        print(f"    Ideal: {n_checked / n_banks:.1f} per bank")
        if empty > 0:
            print(f"    Empty banks: {empty}")
    return bank_counts


# ── Helpers ─────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Main experiment ─────────────────────────────────────────────────

def run_experiment(
    model_name: str = "gpt2",
    alpha: float | None = None,
    n_trace_heads: int | None = None,
    d_trace: int = 64,
    n_eval: int = 30,
    bank_configs: list[int] | None = None,
    n_facts_list: list[int] | None = None,
    max_facts: int = 1000,
    seed: int = 42,
):
    if bank_configs is None:
        bank_configs = [1, 16, 64, 128]
    if n_facts_list is None:
        n_facts_list = [10, 20, 50, 100, 200, 300, 500, 750, 1000]

    n_facts_list = [n for n in n_facts_list if n <= max_facts]

    device = get_device()

    print("=" * 80)
    print("  Exp 30: Capacity Scaling to 1000 Facts")
    print("=" * 80)
    print(f"  Model:     {model_name}")
    print(f"  Device:    {device}")
    print(f"  Episodes:  {n_eval}")
    print(f"  Banks:     {bank_configs}")
    print(f"  Facts:     {n_facts_list}")
    print()

    # Load model
    model, tokenizer = load_model(
        model_name, alpha=alpha,
        n_trace_heads=n_trace_heads,
        d_trace=d_trace, device=device)

    actual_alpha = model.trace.alpha
    actual_heads = model.trace.n_heads
    d_model = model.base_model.config.hidden_size
    print()

    # Build scaled fact types
    n_types_needed = max(n_facts_list) + 50
    all_types, entity_ids = build_scaled_fact_types(
        tokenizer, n_types_needed, seed=seed)
    print()

    # Verify we have enough types
    if len(all_types) < max(n_facts_list):
        print(f"  WARNING: only {len(all_types)} types, "
              f"capping facts list")
        n_facts_list = [n for n in n_facts_list if n <= len(all_types)]

    # Bank distribution diagnostic
    if max(bank_configs) > 1:
        diagnose_bank_distribution(
            model, all_types, tokenizer,
            max(bank_configs), n_concepts=min(300, len(all_types)))
        print()

    # ── Run sweep ──
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
        print(f"  {'─' * 5}  {'─' * 10}  {'─' * 18}  {'─' * 8}  {'─' * 6}")

        for n_facts in n_facts_list:
            t0 = time.time()
            acc, per_ep = evaluate_scaling(
                model, n_facts, all_types, entity_ids,
                tokenizer, n_eval, seed)
            dt = time.time() - t0

            ci_lo, ci_hi = bootstrap_ci(per_ep)
            std = (sum((x - acc) ** 2 for x in per_ep)
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
    print(f"  Comparison: {model_name} — Accuracy by n_facts × n_banks")
    print(f"{'=' * 80}")

    header = f"  {'n':>5}"
    for nb in bank_configs:
        header += f"  {'B=' + str(nb):>8}"
    if len(bank_configs) > 1:
        header += f"  {'Δ(best)':>8}"
    print(header)
    sep_len = 7 + 10 * len(bank_configs) + (10 if len(bank_configs) > 1 else 0)
    print(f"  {'─' * sep_len}")

    for n_facts in n_facts_list:
        row = f"  {n_facts:>5}"
        accs = []
        for nb in bank_configs:
            acc = all_results[nb].get(n_facts, {}).get("accuracy", 0)
            accs.append(acc)
            row += f"  {acc:>7.1%}"
        if len(bank_configs) > 1:
            delta = max(accs[1:]) - accs[0]
            row += f"  {'+' if delta >= 0 else ''}{delta:>6.1%}"
        print(row)

    # ── Key metrics ──
    print(f"\n  Key metrics:")
    for nb in bank_configs:
        results = all_results[nb]
        best_95 = 0
        for n in sorted(results.keys()):
            if results[n]["accuracy"] >= 0.95:
                best_95 = n
        best_99 = 0
        for n in sorted(results.keys()):
            if results[n]["accuracy"] >= 0.99:
                best_99 = n
        label = "baseline" if nb == 1 else f"{nb} banks"
        print(f"    {label:>12}: ≥99% through n={best_99}, "
              f"≥95% through n={best_95}")

    # ── Save results ──
    safe_name = model_name.replace("/", "_")
    output = {
        "config": dict(
            model=model_name,
            d_model=d_model,
            alpha=actual_alpha,
            n_trace_heads=actual_heads,
            d_trace=d_trace,
            n_eval=n_eval, seed=seed,
            bank_configs=bank_configs,
            n_types=len(all_types),
            n_entities=len(entity_ids),
            max_facts=max_facts,
        ),
        "results": {
            str(nb): {
                str(n): {k: v for k, v in r.items() if k != "per_episode"}
                for n, r in results.items()
            }
            for nb, results in all_results.items()
        },
    }

    os.makedirs("results", exist_ok=True)
    path = f"results/exp30_{safe_name}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#9ca3af', '#2563eb', '#059669', '#d97706', '#dc2626']
        markers = ['s', 'o', '^', 'D', 'v']

        for i, nb in enumerate(bank_configs):
            results = all_results[nb]
            ns = sorted(results.keys())
            accs = [results[n]["accuracy"] for n in ns]
            ci_lo = [results[n]["ci_lo"] for n in ns]
            ci_hi = [results[n]["ci_hi"] for n in ns]

            label = "Baseline (no banks)" if nb == 1 else f"{nb} banks"
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            lw = 2.5 if nb > 1 else 1.5
            ls = '--' if nb == 1 else '-'

            ax.plot(ns, accs, f'{m}{ls}', color=c, linewidth=lw,
                    markersize=7, label=label, zorder=3 + i)
            ax.fill_between(ns, ci_lo, ci_hi, alpha=0.1, color=c)

        ax.axhline(y=0.95, color='#d1d5db', linestyle=':', linewidth=1)
        ax.text(max(n_facts_list) * 0.02, 0.96, '95%',
                fontsize=9, color='#9ca3af')

        ax.axhline(y=1 / len(entity_ids), color='#e5e7eb',
                    linestyle=':', linewidth=1,
                    label=f'Random ({1 / len(entity_ids):.1%})')

        ax.set_xlabel("Number of facts stored", fontsize=13)
        ax.set_ylabel("Cross-context accuracy", fontsize=13)
        short_name = model_name.split("/")[-1]
        ax.set_title(f"Capacity Scaling: {short_name} + "
                     f"Hashed Trace Banks ({actual_heads}h×d{d_trace})",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='center right')
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, max(n_facts_list) * 1.05)
        ax.grid(True, alpha=0.3)

        fig_path = f"results/exp30_{safe_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Figure: {fig_path}")
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not available, skipping figure)")

    return all_results


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 30: Capacity scaling to 1000 facts")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (10 episodes, fewer points, GPT-2)")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Trace injection alpha (auto from MODEL_CONFIGS)")
    parser.add_argument("--n-trace-heads", type=int, default=None,
                        help="Number of trace heads (auto from MODEL_CONFIGS)")
    parser.add_argument("--d-trace", type=int, default=64,
                        help="Trace dimension per head")
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--banks", type=int, nargs="+",
                        default=[1, 16, 64, 128],
                        help="Bank configurations to test")
    parser.add_argument("--max-facts", type=int, default=1000,
                        help="Maximum number of facts to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        run_experiment(
            model_name=args.model,
            alpha=args.alpha,
            n_trace_heads=args.n_trace_heads,
            d_trace=args.d_trace,
            n_eval=10,
            bank_configs=[1, 16, 64],
            n_facts_list=[10, 50, 100, 200, 500],
            max_facts=args.max_facts,
            seed=args.seed,
        )
    else:
        run_experiment(
            model_name=args.model,
            alpha=args.alpha,
            n_trace_heads=args.n_trace_heads,
            d_trace=args.d_trace,
            n_eval=args.n_eval,
            bank_configs=args.banks,
            max_facts=args.max_facts,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
