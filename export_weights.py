#!/usr/bin/env python3
"""Export trained trace module weights for the public repository.

Trains W_gate and W_gate_key via the standard curriculum,
then saves only the trace module state dict.
"""

import torch
from transformers import GPT2Tokenizer

from hebbian_trace.gpt2_trace import GPT2WithTrace
from hebbian_trace.gpt2_tasks import build_fact_types, get_linking_bpe_ids

# Import training functions from experiments
from hebbian_trace.experiments.exp9_learned_gating import train_gate
from hebbian_trace.experiments.exp11_dual_gates import train_gate_key


def export():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    fact_types = build_fact_types(tokenizer)
    linking_ids = get_linking_bpe_ids(tokenizer)

    # Create model
    model = GPT2WithTrace(
        n_trace_heads=8, d_trace=64,
        alpha=0.5, trace_lr=1.0, trace_decay=0.99,
    )
    model = model.to(device)
    model.set_linking_token_ids(linking_ids)
    model.enable_pattern_separation(expand_factor=8, top_k=16, seed=0)

    wte = model.gpt2.transformer.wte

    # Phase A: Train W_gate (5-stage curriculum)
    print("\n=== Phase A: Training W_gate ===")
    stages = [
        {"n_steps": 280, "n_facts": 1, "lr": 5e-3, "l1": 0.0},
        {"n_steps": 280, "n_facts": 3, "lr": 3e-3, "l1": 0.0},
        {"n_steps": 280, "n_facts": 5, "lr": 3e-3, "l1": 0.0},
        {"n_steps": 280, "n_facts": 5, "lr": 1e-3, "l1": 0.5},
        {"n_steps": 280, "n_facts": 5, "lr": 1e-3, "l1": 1.0},
    ]
    for i, cfg in enumerate(stages):
        print(f"  Stage {i+1}: n_facts={cfg['n_facts']}, lr={cfg['lr']}, l1={cfg['l1']}")
        train_gate(
            model, tokenizer, fact_types,
            n_steps=cfg["n_steps"], n_facts_train=cfg["n_facts"],
            lr=cfg["lr"], l1_lambda=cfg["l1"],
            device=device, log_every=9999,
        )

    # Phase B: Train W_gate_key (4-stage curriculum)
    print("\n=== Phase B: Training W_gate_key ===")
    gate_key_stages = [
        {"n_steps": 300, "n_facts": 3, "lr": 5e-3, "l1": 0.0,
         "filler_mode": "none", "n_filler": 0},
        {"n_steps": 300, "n_facts": 5, "lr": 3e-3, "l1": 0.0,
         "filler_mode": "none", "n_filler": 0},
        {"n_steps": 300, "n_facts": 5, "lr": 3e-3, "l1": 0.0,
         "filler_mode": "noisy", "n_filler": 3},
        {"n_steps": 300, "n_facts": 5, "lr": 1e-3, "l1": 0.5,
         "filler_mode": "mixed", "n_filler": 3},
    ]
    for i, cfg in enumerate(gate_key_stages):
        print(f"  Stage {i+1}: n_facts={cfg['n_facts']}, filler={cfg['filler_mode']}")
        train_gate_key(
            model, tokenizer, fact_types,
            n_steps=cfg["n_steps"], n_facts_train=cfg["n_facts"],
            n_filler=cfg["n_filler"], filler_mode=cfg["filler_mode"],
            lr=cfg["lr"], l1_lambda=cfg["l1"],
            device=device, log_every=9999,
        )

    # Save trace module state dict (excludes GPT-2 and value_traces buffer)
    save_path = "/Users/cnails/hebbian-trace-memory/weights/trace_module.pt"

    # Get state dict but exclude value_traces (runtime buffer, not weights)
    state = {}
    for k, v in model.trace.state_dict().items():
        if k == 'value_traces' or k == 'W_expand':
            continue  # Runtime buffers, not trained weights
        state[k] = v.cpu()

    torch.save(state, save_path)

    # Print summary
    total_params = sum(v.numel() for v in state.values())
    size_kb = sum(v.numel() * 4 for v in state.values()) / 1024
    print(f"\nSaved {len(state)} tensors ({total_params:,} params, {size_kb:.0f} KB)")
    print(f"Path: {save_path}")

    # Verify gate quality
    print("\n=== Gate Verification ===")
    concept_words = ["name", "city", "company", "color", "food", "pet", "country"]
    filler_words = ["weather", "answer", "solution", "time", "office"]

    for label, words in [("FACT concepts", concept_words), ("FILLER words", filler_words)]:
        vals = []
        for w in words:
            ids = tokenizer.encode(" " + w, add_special_tokens=False)
            if len(ids) == 1:
                t = torch.tensor([[ids[0]]], dtype=torch.long, device=device)
                gk = model.trace.compute_gate_key(wte, t).item()
                vals.append(gk)
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {label}: avg gate_key = {avg:.4f}")


if __name__ == "__main__":
    export()
