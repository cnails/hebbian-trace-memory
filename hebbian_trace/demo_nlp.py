"""Interactive demo: test Hebbian trace memory with natural language facts.

Usage:
    # Train a new model (quick, ~5 min) and play:
    python -m hebbian_trace.demo_nlp

    # Train and save:
    python -m hebbian_trace.demo_nlp --save models/nlp_model.pt

    # Load a saved model:
    python -m hebbian_trace.demo_nlp --load models/nlp_model.pt

    # Full training (better quality, ~80 min):
    python -m hebbian_trace.demo_nlp --full

Interactive commands:
    fact> My name is Andrey .      (store a fact — traces accumulate)
    ask>  What is my name ?        (ask a question — predict from trace)
    reset                          (clear all traces — new session)
    traces                         (show trace norms)
    quit / exit
"""

import argparse
import os
import torch

from .nlp_tasks import NLP_VOCAB, FACT_TYPES
from .experiments.exp2_nlp_facts import (
    pretrain_nlp, save_nlp_model, load_nlp_model, get_device,
)


def _predict_top_k(model, query_indices: list[int], vocab, k: int = 5):
    """Predict top-k entity tokens at the last position."""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([query_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)

    pred_logits = logits[0, -1, :]
    entity_indices = vocab.entity_indices
    entity_logits = pred_logits[entity_indices]

    probs = torch.softmax(entity_logits, dim=0)
    top_vals, top_ids = probs.topk(min(k, len(entity_indices)))

    results = []
    for val, idx in zip(top_vals, top_ids):
        token_idx = entity_indices[idx.item()]
        word = vocab.idx2tok[token_idx]
        results.append((word, val.item()))
    return results


def interactive(model, vocab):
    """Interactive fact-and-question loop."""
    device = next(model.parameters()).device
    model.eval()

    # Set trace params
    for attn in model.get_attention_layers():
        attn.trace_decay = 0.99
        attn.trace_lr = 0.1
        attn.alpha = 0.1

    model.reset_traces()
    model.set_trace_mode(use=True, update=True)

    print("\n" + "=" * 50)
    print("INTERACTIVE HEBBIAN TRACE DEMO")
    print("=" * 50)
    print("\nStore facts, then ask questions from trace memory.")
    print("Traces persist across inputs (like cross-session memory).\n")
    print("Commands:")
    print("  fact> <sentence>   Store a fact (e.g., My name is Andrey .)")
    print("  ask>  <question>   Ask a question (e.g., What is my name ?)")
    print("  reset              Clear all traces")
    print("  traces             Show trace norms")
    print("  quit               Exit\n")

    # Show available fact templates
    print("Supported fact patterns (Tier 1 — traces can store these):")
    for ft in FACT_TYPES:
        t1 = [t for t in ft.fact_templates if t.tier == 1]
        example = t1[0].words if t1 else ft.fact_templates[0].words
        example_str = " ".join(w.replace("{X}", ft.values[0]) for w in example)
        q_str = " ".join(ft.question_templates[0].words)
        print(f"  {example_str:35s}  →  {q_str}")
    print()

    n_stored = 0

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "q"):
            break

        if line.lower() == "reset":
            model.reset_traces()
            n_stored = 0
            print("Traces cleared.\n")
            continue

        if line.lower() == "traces":
            for i, attn in enumerate(model.get_attention_layers()):
                s_norm = attn.traces.norm().item()
                v_norm = attn.value_traces.norm().item()
                print(f"  Layer {i}: score_trace={s_norm:.4f}  "
                      f"value_trace={v_norm:.4f}")
            print()
            continue

        # Parse: "fact> ..." or "ask> ..."
        if line.startswith("fact>"):
            text = line[5:].strip()
            mode = "fact"
        elif line.startswith("ask>"):
            text = line[4:].strip()
            mode = "ask"
        else:
            # Auto-detect: if ends with "?" it's a question
            if line.endswith("?"):
                text = line
                mode = "ask"
            else:
                text = line
                mode = "fact"

        # Tokenize
        words = text.split()
        try:
            indices = vocab.encode(["<bos>"] + words)
        except KeyError as e:
            print(f"Unknown word: {e}")
            print(f"Vocabulary has {len(vocab)} tokens. "
                  f"Use words from the templates above.\n")
            continue

        if mode == "fact":
            # Store: forward with trace update ON
            model.set_trace_mode(use=True, update=True)
            input_tensor = torch.tensor(
                [indices + [vocab.eos_idx]], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_tensor)
            n_stored += 1
            print(f"  Stored. ({n_stored} facts in trace)\n")

        else:
            # Ask: forward with trace ON but NO update
            model.set_trace_mode(use=True, update=False)
            top_k = _predict_top_k(model, indices, vocab, k=5)
            print(f"  Top predictions:")
            for word, prob in top_k:
                bar = "█" * int(prob * 40)
                print(f"    {word:15s} {prob:5.1%} {bar}")

            # Also show what it predicts WITHOUT trace
            model.set_trace_mode(use=False, update=False)
            top_k_no_trace = _predict_top_k(model, indices, vocab, k=3)
            print(f"  Without trace:")
            for word, prob in top_k_no_trace:
                print(f"    {word:15s} {prob:5.1%}")
            print()

            # Restore trace mode for next fact
            model.set_trace_mode(use=True, update=True)


def main():
    parser = argparse.ArgumentParser(description="Interactive Hebbian trace demo")
    parser.add_argument("--load", type=str, default=None,
                        help="Load model from file")
    parser.add_argument("--save", type=str, default=None,
                        help="Save model after training")
    parser.add_argument("--full", action="store_true",
                        help="Full training (20k seq, 60 epochs, ~80 min)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    vocab = NLP_VOCAB
    device = get_device(args.device)

    if args.load:
        print(f"Loading model from {args.load}...")
        model = load_nlp_model(args.load, args.device)
        print(f"Loaded. Device: {device}")
    else:
        if args.full:
            n_seq, epochs = 20000, 60
            print("Full training (~80 min)...")
        else:
            n_seq, epochs = 5000, 30
            print("Quick training (~10 min)...")

        model, stats = pretrain_nlp(
            n_sequences=n_seq,
            max_facts=5 if args.full else 3,
            batch_size=64,
            epochs=epochs,
            lr=1e-3,
            d_model=256, n_heads=8, n_layers=8,
            max_seq_len=128, dropout=0.1,
            alpha=0.1, trace_lr=0.1, trace_decay=0.99,
            use_raw_embed=True, use_key_q=True,
            seed=42, device=device, verbose=True,
        )
        model.set_linking_token_ids(vocab.linking_tokens)

        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            save_nlp_model(model, args.save)
            print(f"Model saved to {args.save}")

    interactive(model, vocab)


if __name__ == "__main__":
    main()
