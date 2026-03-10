#!/usr/bin/env python3
"""Model scaling experiment: trace transfer across architectures and scales.

Tests Hebbian trace on multiple frozen LLMs to demonstrate architecture-agnostic
transfer. Each model runs zero-shot (no gate training) with hardcoded linking-token
mask and pattern separation (8x expansion, top-k=16).

Phases:
  1. Alpha sweep: find optimal trace injection strength for each model
  2. Cross-context eval: n=1,3,5,7 at best alpha (THE REAL TEST)
  3. Baseline: in-context eval (no trace) for comparison

Supported models:
  - gpt2 (124M, d=768)           alpha=0.5
  - gpt2-medium (355M, d=1024)   alpha=0.5
  - microsoft/phi-2 (2.7B, d=2560)  alpha=50.0
  - meta-llama/Llama-2-7b-hf (7B)   alpha=TBD
  - mistralai/Mistral-7B-v0.1 (7B)  alpha=TBD
  - meta-llama/Meta-Llama-3-8B (8B) alpha=TBD

Usage:
    # Local sanity check (GPT-2, no GPU needed)
    python exp_scaling.py --model gpt2 --n-eval 5

    # Alpha sweep for new model
    python exp_scaling.py --model meta-llama/Llama-2-7b-hf --phase sweep

    # Full eval at known alpha
    python exp_scaling.py --model meta-llama/Llama-2-7b-hf --alpha 2.0 --n-eval 100

    # All models with known alphas
    python exp_scaling.py --all --n-eval 100
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# Load HF_TOKEN from .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists() and not os.environ.get("HF_TOKEN"):
    for line in _env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

# This script uses the public repo's API (hebbian_trace package).
# Run from /Users/cnails/hebbian-trace-memory/ with model.py generalized
# (AutoModelForCausalLM etc. — see memory/experiments.md for code changes).
# The code changes are reverted in the public repo but can be re-applied.
from hebbian_trace.model import GPT2WithTrace
from hebbian_trace.tasks import (
    build_fact_types,
    get_linking_bpe_ids,
    make_eval_episodes,
    evaluate_baseline,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)

# ── Model Configs ──

MODEL_CONFIGS = {
    "gpt2": {
        "alpha": 0.5,
        "dtype": None,
        "params": "124M",
    },
    "gpt2-medium": {
        "alpha": 0.5,
        "dtype": None,
        "params": "355M",
    },
    "microsoft/phi-2": {
        "alpha": 50.0,
        "dtype": "float16",
        "params": "2.7B",
    },
    "meta-llama/Llama-2-7b-hf": {
        "alpha": None,  # needs sweep
        "dtype": "float16",
        "params": "7B",
    },
    "mistralai/Mistral-7B-v0.1": {
        "alpha": None,
        "dtype": "float16",
        "params": "7B",
    },
    "meta-llama/Meta-Llama-3-8B": {
        "alpha": None,
        "dtype": "float16",
        "params": "8B",
    },
}

ALPHA_SWEEP_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
                      500.0, 1000.0, 2000.0, 5000.0, 10000.0]

N_FACTS_LIST = [1, 3, 5, 7]

# ── Counterfactual Facts ──
# Well-known facts with WRONG answers to test trace vs pretrained prior.
# Format: (concept_word, linking_token, wrong_entity, fact_template, question)
# The model "knows" the real answer; trace must override with the wrong one.

COUNTERFACTUAL_FACTS = [
    # Capitals (swapped)
    ("capital", "is", "Berlin", "The capital of France is {X}.",
     "What is the capital of France?"),
    ("capital", "is", "Paris", "The capital of Germany is {X}.",
     "What is the capital of Germany?"),
    ("capital", "is", "Madrid", "The capital of Italy is {X}.",
     "What is the capital of Italy?"),
    ("capital", "is", "Rome", "The capital of Spain is {X}.",
     "What is the capital of Spain?"),
    ("capital", "is", "Tokyo", "The capital of China is {X}.",
     "What is the capital of China?"),
    ("capital", "is", "Beijing", "The capital of Japan is {X}.",
     "What is the capital of Japan?"),
    ("capital", "is", "London", "The capital of Russia is {X}.",
     "What is the capital of Russia?"),
    ("capital", "is", "Moscow", "The capital of England is {X}.",
     "What is the capital of England?"),
    ("capital", "is", "Ottawa", "The capital of Mexico is {X}.",
     "What is the capital of Mexico?"),
    ("capital", "is", "Lima", "The capital of Canada is {X}.",
     "What is the capital of Canada?"),
    # Colors (wrong)
    ("color", "is", "blue", "The color of grass is {X}.",
     "What is the color of grass?"),
    ("color", "is", "green", "The color of the sky is {X}.",
     "What is the color of the sky?"),
    ("color", "is", "yellow", "The color of blood is {X}.",
     "What is the color of blood?"),
    ("color", "is", "red", "The color of snow is {X}.",
     "What is the color of snow?"),
    # Languages (swapped)
    ("language", "is", "French", "The language of Brazil is {X}.",
     "What is the language of Brazil?"),
    ("language", "is", "Spanish", "The language of France is {X}.",
     "What is the language of France?"),
    ("language", "is", "German", "The language of Italy is {X}.",
     "What is the language of Italy?"),
    ("language", "is", "Italian", "The language of Germany is {X}.",
     "What is the language of Germany?"),
    # Continents (wrong)
    ("continent", "in", "Europe", "Japan is in {X}.",
     "What continent is Japan in?"),
    ("continent", "in", "Asia", "France is in {X}.",
     "What continent is France in?"),
]


# ── Utilities ──

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bootstrap_ci(values: list[float], n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for mean."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def validate_tokenizer(tokenizer, fact_types) -> dict:
    """Check which linking tokens and entities are single-token for this model."""
    linking_words = ["is", "in", "at", "from", ":", "am"]
    linking_report = {}
    for word in linking_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        linking_report[word] = {
            "ids": ids,
            "single_token": len(ids) == 1,
        }

    valid_linking = [w for w, r in linking_report.items() if r["single_token"]]

    entity_report = {}
    for ft in fact_types:
        total = len(ft.entities)
        valid = []
        for name, tid in ft.entities:
            # Verify token ID is correct for this tokenizer
            check_ids = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(check_ids) == 1:
                valid.append((name, check_ids[0]))
        entity_report[ft.name] = {
            "total": total,
            "valid": len(valid),
            "coverage": len(valid) / total if total > 0 else 0,
        }

    return {
        "linking": linking_report,
        "valid_linking": valid_linking,
        "entities": entity_report,
    }


def load_model(model_name: str, alpha: float, device: str,
               n_trace_heads: int = 8, d_trace: int = 64) -> GPT2WithTrace:
    """Load model + trace with appropriate settings."""
    cfg = MODEL_CONFIGS.get(model_name, {})
    dtype_str = cfg.get("dtype")
    torch_dtype = getattr(torch, dtype_str) if dtype_str else None

    print(f"\n  Loading {model_name}...")
    t0 = time.time()

    model = GPT2WithTrace(
        n_trace_heads=n_trace_heads, d_trace=d_trace,
        alpha=alpha, trace_lr=1.0, trace_decay=0.99,
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
    )

    dt = time.time() - t0
    config = model.base_model.config
    d_model = config.hidden_size
    n_layers = getattr(config, 'n_layer',
                       getattr(config, 'num_hidden_layers', -1))

    print(f"  Loaded in {dt:.1f}s")
    print(f"  d_model={d_model}, n_layers={n_layers}, alpha={alpha}")
    print(f"  Base params: {sum(p.numel() for p in model.base_model.parameters()):,}")
    print(f"  Trace params: {sum(p.numel() for p in model.trace.parameters()):,}")

    # Zero-shot: no gates, hardcoded linking mask only
    model.set_gate_mode(use_learned_gate=False)
    model.set_dual_gate_mode(enabled=False)

    # Pattern separation 8x_k16
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    return model


def rebuild_fact_types_for_tokenizer(tokenizer):
    """Build fact types validated against this specific tokenizer.

    The default build_fact_types uses GPT-2 token IDs. For other tokenizers,
    entity pools need re-validation since token IDs differ.
    """
    return build_fact_types(tokenizer)


# ── Phase 1: Alpha Sweep ──

def run_alpha_sweep(model_name: str, device: str,
                    n_eval: int = 20, seed: int = 42,
                    n_trace_heads: int = 8, d_trace: int = 64) -> dict:
    """Find optimal alpha for a model by sweeping at n=1."""
    print(f"\n{'='*70}")
    print(f"  Alpha Sweep: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Validate tokenizer compatibility
    base_types = rebuild_fact_types_for_tokenizer(tokenizer)
    compat = validate_tokenizer(tokenizer, base_types)

    print(f"\n  Linking tokens:")
    for word, info in compat["linking"].items():
        status = "OK" if info["single_token"] else "MULTI-TOKEN"
        print(f"    '{word}': {status} (ids={info['ids']})")

    print(f"\n  Entity coverage:")
    for cword, info in compat["entities"].items():
        print(f"    {cword}: {info['valid']}/{info['total']} "
              f"({info['coverage']:.0%})")

    valid_linking = compat["valid_linking"]
    if not valid_linking:
        print("\n  ERROR: No valid linking tokens for this tokenizer!")
        return {"error": "no valid linking tokens"}

    linking_ids = get_linking_bpe_ids(tokenizer)

    # Load model once, change alpha in-place (saves ~30s/alpha for 7B)
    model = load_model(model_name, ALPHA_SWEEP_VALUES[0], device,
                       n_trace_heads=n_trace_heads, d_trace=d_trace)
    model.set_linking_token_ids(linking_ids)

    results = {}
    best_alpha = ALPHA_SWEEP_VALUES[0]
    best_acc = 0.0

    for alpha in ALPHA_SWEEP_VALUES:
        model.trace.alpha = alpha
        model.reset_traces()
        model.set_trace_mode(use=True, update=True)

        random.seed(seed)
        episodes = make_eval_episodes(
            n_eval, 1, tokenizer, base_types, seed=seed)

        res = evaluate_cross_context(model, episodes, base_types)
        mean_acc = res.accuracy

        results[alpha] = {
            "mean": float(mean_acc),
            "per_episode": [float(a) for a in res.per_episode_acc],
        }

        marker = " <-- BEST" if mean_acc > best_acc else ""
        print(f"  alpha={alpha:>6.1f}: n=1 acc = {mean_acc:.1%}{marker}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Best alpha: {best_alpha} ({best_acc:.1%})")

    return {
        "model": model_name,
        "phase": "sweep",
        "n_eval": n_eval,
        "results": results,
        "best_alpha": best_alpha,
        "best_acc": float(best_acc),
    }


# ── Phase 2: Cross-Context Eval ──

def run_cross_context_eval(model_name: str, alpha: float, device: str,
                           n_eval: int = 100, seed: int = 42,
                           n_trace_heads: int = 8, d_trace: int = 64) -> dict:
    """Full cross-context evaluation at multiple fact counts."""
    print(f"\n{'='*70}")
    print(f"  Cross-Context Eval: {model_name} (alpha={alpha})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_types = rebuild_fact_types_for_tokenizer(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    model = load_model(model_name, alpha, device,
                       n_trace_heads=n_trace_heads, d_trace=d_trace)
    model.set_linking_token_ids(linking_ids)

    # Try loading trained weights (only works for GPT-2 d_model=768)
    weights_path = Path(__file__).parent / "weights" / "trace_module.pt"
    d_model = model.base_model.config.hidden_size
    if weights_path.exists() and d_model == 768:
        try:
            state = torch.load(weights_path, map_location=device,
                               weights_only=True)
            model.trace.load_state_dict(state, strict=False)
            print(f"  Loaded trained weights (d_model=768 match)")
        except Exception as e:
            print(f"  Weight loading failed: {e}")
    else:
        print(f"  Zero-shot mode (d_model={d_model}, no trained gates)")

    results = {}
    for n_facts in N_FACTS_LIST:
        if n_facts > len(base_types):
            print(f"  Skipping n={n_facts} (only {len(base_types)} types)")
            continue

        random.seed(seed)
        episodes = make_eval_episodes(
            n_eval, n_facts, tokenizer, base_types, seed=seed)

        # Cross-context (trace only)
        res = evaluate_cross_context(model, episodes, base_types)
        mean_acc = res.accuracy
        lo, hi = bootstrap_ci(res.per_episode_acc)

        results[n_facts] = {
            "cross_context": {
                "mean": float(mean_acc),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "per_episode": [float(a) for a in res.per_episode_acc],
            },
        }

        print(f"  n={n_facts}: cross-ctx {mean_acc:.1%} [{lo:.1%}-{hi:.1%}]")

    # Baseline (in-context, no trace)
    print(f"\n  Baseline (in-context):")
    for n_facts in N_FACTS_LIST:
        if n_facts > len(base_types):
            continue

        random.seed(seed)
        episodes = make_eval_episodes(
            n_eval, n_facts, tokenizer, base_types, seed=seed)

        res = evaluate_baseline(model, episodes, base_types, tokenizer)
        mean_acc = res.accuracy
        lo, hi = bootstrap_ci(res.per_episode_acc)

        results[n_facts]["baseline"] = {
            "mean": float(mean_acc),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
        print(f"  n={n_facts}: baseline {mean_acc:.1%} [{lo:.1%}-{hi:.1%}]")

    # No-trace baseline (question only)
    print(f"\n  No-trace (question only):")
    for n_facts in N_FACTS_LIST:
        if n_facts > len(base_types):
            continue

        random.seed(seed)
        episodes = make_eval_episodes(
            n_eval, n_facts, tokenizer, base_types, seed=seed)

        res = evaluate_cross_context_baseline(model, episodes, base_types)
        mean_acc = res.accuracy

        results[n_facts]["no_trace"] = {"mean": float(mean_acc)}
        print(f"  n={n_facts}: no-trace {mean_acc:.1%}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "params": MODEL_CONFIGS.get(model_name, {}).get("params", "?"),
        "alpha": alpha,
        "d_model": d_model,
        "d_trace": d_trace,
        "n_trace_heads": n_trace_heads,
        "n_eval": n_eval,
        "n_fact_types": len(base_types),
        "phase": "eval",
        "results": {str(k): v for k, v in results.items()},
    }


# ── Phase 3: Counterfactual Eval ──

def _build_counterfactual_episodes(
    tokenizer, n_facts_list: list[int], n_episodes: int, seed: int = 42,
) -> tuple[list, list]:
    """Build counterfactual episodes from COUNTERFACTUAL_FACTS.

    Returns (valid_facts, episodes_by_n) where valid_facts are facts
    whose entity is a single BPE token for this tokenizer.
    """
    rng = random.Random(seed)

    # Validate which facts work with this tokenizer
    valid_facts = []
    for concept, link, entity, fact_tpl, question in COUNTERFACTUAL_FACTS:
        entity_ids = tokenizer.encode(" " + entity, add_special_tokens=False)
        concept_ids = tokenizer.encode(" " + concept, add_special_tokens=False)
        link_ids = tokenizer.encode(" " + link, add_special_tokens=False)
        if len(entity_ids) == 1 and len(concept_ids) == 1 and len(link_ids) == 1:
            fact_text = fact_tpl.replace("{X}", entity)
            fact_bpe = tokenizer.encode(fact_text, add_special_tokens=False)
            q_bpe = tokenizer.encode(question, add_special_tokens=False)
            valid_facts.append({
                "concept": concept,
                "entity": entity,
                "entity_id": entity_ids[0],
                "concept_id": concept_ids[0],
                "fact_bpe": fact_bpe,
                "question_bpe": q_bpe,
                "fact_text": fact_text,
                "question": question,
            })

    return valid_facts


def run_counterfactual_eval(
    model_name: str, alpha: float, device: str,
    n_eval: int = 50, seed: int = 42,
    n_trace_heads: int = 8, d_trace: int = 64,
) -> dict:
    """Test trace override of pretrained knowledge with counterfactual facts.

    For each episode:
    1. Measure model's pretrained answer (no trace, no in-context facts)
    2. Write counterfactual facts to trace
    3. Query — does trace override the pretrained prior?

    Reports: pretrained_acc (should be ~0% since answers are WRONG),
    trace_acc (should be high if trace overrides prior).
    """
    print(f"\n{'='*70}")
    print(f"  Counterfactual Eval: {model_name} (alpha={alpha})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    valid_facts = _build_counterfactual_episodes(tokenizer, [], 0, seed)

    print(f"  Valid counterfactual facts: {len(valid_facts)}/{len(COUNTERFACTUAL_FACTS)}")
    if len(valid_facts) < 3:
        print("  ERROR: Too few valid facts for this tokenizer")
        return {"error": "too few valid facts"}

    model = load_model(model_name, alpha, device,
                       n_trace_heads=n_trace_heads, d_trace=d_trace)
    linking_ids = get_linking_bpe_ids(tokenizer)
    model.set_linking_token_ids(linking_ids)

    # Collect all candidate entity IDs from valid facts
    all_entity_ids = list({f["entity_id"] for f in valid_facts})

    rng = random.Random(seed)
    n_facts_list = [1, 3, 5]
    max_n = min(max(n_facts_list), len(valid_facts))
    n_facts_list = [n for n in n_facts_list if n <= max_n]

    results = {}
    for n_facts in n_facts_list:
        trace_correct = 0
        prior_correct = 0  # model's own answer matches counterfactual
        prior_real = 0     # model gives real-world answer (opposing trace)
        total = 0

        for ep in range(n_eval):
            # Sample n_facts counterfactual facts
            episode_facts = rng.sample(valid_facts, n_facts)

            # 1. Measure pretrained prior (no trace, question only)
            model.reset_traces()
            model.set_trace_mode(use=False, update=False)

            for fact in episode_facts:
                q_tensor = torch.tensor(
                    [fact["question_bpe"]], dtype=torch.long,
                    device=device)
                with torch.no_grad():
                    logits = model(q_tensor)
                pred_logits = logits[0, -1, :]
                cand_logits = pred_logits[all_entity_ids]
                pred_id = all_entity_ids[cand_logits.argmax().item()]
                if pred_id == fact["entity_id"]:
                    prior_correct += 1  # model already says the wrong answer
                total += 1

            # 2. Write facts to trace
            model.reset_traces()
            model.set_trace_mode(use=False, update=True)
            for fact in episode_facts:
                fact_tensor = torch.tensor(
                    [fact["fact_bpe"]], dtype=torch.long, device=device)
                with torch.no_grad():
                    model(fact_tensor)

            # 3. Query with trace
            model.set_trace_mode(use=True, update=False)
            for fact in episode_facts:
                q_tensor = torch.tensor(
                    [fact["question_bpe"]], dtype=torch.long,
                    device=device)
                with torch.no_grad():
                    logits = model(q_tensor)
                pred_logits = logits[0, -1, :]
                cand_logits = pred_logits[all_entity_ids]
                pred_id = all_entity_ids[cand_logits.argmax().item()]
                if pred_id == fact["entity_id"]:
                    trace_correct += 1

        total_queries = total  # same for both phases
        prior_rate = prior_correct / total_queries if total_queries > 0 else 0
        trace_rate = trace_correct / total_queries if total_queries > 0 else 0

        results[n_facts] = {
            "prior_acc": float(prior_rate),
            "trace_acc": float(trace_rate),
            "override_delta": float(trace_rate - prior_rate),
            "n_queries": total_queries,
        }

        print(f"  n={n_facts}: prior={prior_rate:.1%}  "
              f"trace={trace_rate:.1%}  "
              f"override=+{trace_rate - prior_rate:.1%}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "params": MODEL_CONFIGS.get(model_name, {}).get("params", "?"),
        "alpha": alpha,
        "phase": "counterfactual",
        "n_valid_facts": len(valid_facts),
        "n_eval": n_eval,
        "results": {str(k): v for k, v in results.items()},
    }


# ── Print Summary ──

def print_summary_table(all_results: list[dict]):
    """Print formatted comparison table across models."""
    print(f"\n{'='*90}")
    print(f"  Model Scaling Summary (7 fact types, PS 8x_k16, zero-shot)")
    print(f"{'='*90}")

    header = f"  {'Model':<35} {'Params':>6} {'alpha':>6}"
    for n in N_FACTS_LIST:
        header += f"  {'n='+str(n):>12}"
    print(header)
    print("  " + "-" * 86)

    for res in all_results:
        name = res["model"]
        params = res.get("params", "?")
        alpha = res.get("alpha", "?")
        row = f"  {name:<35} {params:>6} {alpha:>6}"

        for n in N_FACTS_LIST:
            key = str(n)
            if key in res.get("results", {}):
                cc = res["results"][key].get("cross_context", {})
                mean = cc.get("mean", 0)
                lo = cc.get("ci_lo", 0)
                hi = cc.get("ci_hi", 0)
                row += f"  {mean:>5.1%} [{lo:.0%}-{hi:.0%}]"
            else:
                row += f"  {'--':>12}"
        print(row)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Model scaling: trace transfer across architectures")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name (e.g., gpt2, meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--all", action="store_true",
                        help="Run all models with known alphas")
    parser.add_argument("--phase", choices=["sweep", "eval", "both", "counterfactual"],
                        default="eval",
                        help="Phase: sweep, eval, both, counterfactual")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha (skip sweep)")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Episodes per condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-trace", type=int, default=64,
                        help="Trace dimension per head (default: 64)")
    parser.add_argument("--n-trace-heads", type=int, default=8,
                        help="Number of trace heads (default: 8)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for JSON results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Check HF token for gated models
    if args.model and ("llama" in args.model.lower() or
                        "mistral" in args.model.lower()):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print("\nWARNING: No HF_TOKEN found. LLaMA/Mistral models require")
            print("  accepted license + token. Set HF_TOKEN env var or run:")
            print("  huggingface-cli login")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        # Run all models with known alphas
        all_results = []
        for model_name, cfg in MODEL_CONFIGS.items():
            alpha = cfg["alpha"]
            if alpha is None:
                print(f"\n  Skipping {model_name} (alpha unknown, run --phase sweep first)")
                continue
            res = run_cross_context_eval(
                model_name, alpha, device,
                n_eval=args.n_eval, seed=args.seed,
                n_trace_heads=args.n_trace_heads, d_trace=args.d_trace)
            all_results.append(res)

            # Save per-model results
            safe_name = model_name.replace("/", "_")
            out_path = Path(args.output_dir) / f"scaling_{safe_name}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved: {out_path}")

        print_summary_table(all_results)
        return

    if not args.model:
        parser.error("Specify --model or --all")

    model_name = args.model

    if args.phase in ("sweep", "both"):
        sweep_res = run_alpha_sweep(
            model_name, device,
            n_eval=min(args.n_eval, 20), seed=args.seed,
            n_trace_heads=args.n_trace_heads, d_trace=args.d_trace)

        safe_name = model_name.replace("/", "_")
        out_path = Path(args.output_dir) / f"sweep_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(sweep_res, f, indent=2)
        print(f"  Saved: {out_path}")

        if args.phase == "sweep":
            return
        # For "both", use best alpha for eval
        args.alpha = sweep_res["best_alpha"]

    if args.phase in ("eval", "both"):
        alpha = args.alpha
        if alpha is None:
            cfg = MODEL_CONFIGS.get(model_name, {})
            alpha = cfg.get("alpha")
        if alpha is None:
            parser.error(f"No known alpha for {model_name}. "
                         f"Run --phase sweep first or specify --alpha")

        res = run_cross_context_eval(
            model_name, alpha, device,
            n_eval=args.n_eval, seed=args.seed,
            n_trace_heads=args.n_trace_heads, d_trace=args.d_trace)

        safe_name = model_name.replace("/", "_")
        out_path = Path(args.output_dir) / f"scaling_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved: {out_path}")

        print_summary_table([res])

    if args.phase == "counterfactual":
        alpha = args.alpha
        if alpha is None:
            cfg = MODEL_CONFIGS.get(model_name, {})
            alpha = cfg.get("alpha")
        if alpha is None:
            parser.error(f"No known alpha for {model_name}. "
                         f"Run --phase sweep first or specify --alpha")

        res = run_counterfactual_eval(
            model_name, alpha, device,
            n_eval=args.n_eval, seed=args.seed,
            n_trace_heads=args.n_trace_heads, d_trace=args.d_trace)

        safe_name = model_name.replace("/", "_")
        out_path = Path(args.output_dir) / f"counterfactual_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
