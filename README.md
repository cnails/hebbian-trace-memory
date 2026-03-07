# Hebbian Trace Memory

**Persistent cross-session memory for frozen LLMs via bio-inspired Hebbian trace module.**

An external memory module (~1.1M parameters) that attaches to a frozen GPT-2 (124M parameters) and provides persistent fact storage across sessions — without fine-tuning, without RAG, without retraining.

---

## Flagship Result

**99% recall across 15 sessions with 24 distinct fact types.**

The trace module accumulates knowledge session by session. Facts stored in session 1 remain perfectly retrievable at session 15, even as new facts are added and existing ones are updated.

<p align="center">
  <img src="figures/retention_curve.png" width="700" alt="Retention curve: 99% recall across 15 sessions">
</p>

| Property | Value |
|----------|-------|
| Recall at session 15 | **99%** |
| Fact types | 24 (name, city, company, color, food, pet, ...) |
| Sessions | 15 (introduction + updates) |
| Base model | GPT-2 Small (frozen, unmodified) |
| Trainable parameters | ~1.1M (trace module only) |
| Fine-tuning required | None |

---

## Architecture

Six bio-inspired components, each addressing a specific memory challenge:

<p align="center">
  <img src="figures/architecture.png" width="700" alt="Architecture diagram">
</p>

| Component | Biological Analogy | Role |
|-----------|-------------------|------|
| Context-free Q/V | Hippocampal indexing | Same word = same address, regardless of context |
| Pattern separation | Dentate gyrus | Sparse expansion reduces Q overlap (0.477 → 0.308 cosine) |
| Hebbian trace | CA3 associative memory | Outer-product accumulation: `T += lr * Q^T @ V` |
| Dual gates | ACh modulation | Learned fact/filler filtering (103x selectivity) |
| Logit injection | Hippocampal output | Bypasses residual stream scale mismatch |
| Reconsolidation erasure | Memory reconsolidation | L2-normalized erase before overwrite |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed formulas and design rationale.

---

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

First run downloads GPT-2 Small (~500MB). Subsequent runs use the cached model.

### Reproduce Evaluation Results

```bash
python evaluate.py                # 50 episodes (quick, ~5 min)
python evaluate.py --n-eval 100   # 100 episodes (paper results)
```

### Multi-Session Demo

```bash
python demo.py --sessions 5       # 5 sessions, introduction only
python demo.py --sessions 10      # 10 sessions with updates
python demo.py --sessions 15      # full 15-session demo
```

---

## Results

### Cross-Context Retrieval (Pattern Separation, alpha=0.5)

Facts are stored in separate forward passes, then queried with question-only input (no in-context facts). Any accuracy above ~4% comes entirely from the trace module.

| n_facts | Cross-context | In-context (GPT-2) | No trace | Gap |
|---------|:------------:|:------------------:|:--------:|:---:|
| 1 | **100.0%** | 99.0% | 6.0% | +94.0pp |
| 3 | **89.7%** | 73.0% | 4.3% | +85.3pp |
| 5 | **85.4%** | 62.8% | 3.2% | +82.2pp |
| 7 | **82.0%** | 61.6% | 4.3% | +77.7pp |

Cross-context retrieval **exceeds GPT-2's own in-context learning** at all fact counts.

*100 episodes, seed=42. Reproducible via `python evaluate.py --n-eval 100`.*

<p align="center">
  <img src="figures/cross_context.png" width="600" alt="Cross-context retrieval results">
</p>

### Component Ablation (15-session demo)

Each mechanism contributes independently:

<p align="center">
  <img src="figures/ablation_chart.png" width="600" alt="Component ablation">
</p>

### Multi-Session Capacity (24 fact types, 15 sessions)

| Session | Known Facts | Overall | New | Old | Update |
|:-------:|:-----------:|:-------:|:---:|:---:|:------:|
| 1 | 5 | 100% | 100% | -- | -- |
| 5 | 24 | 99% | 100% | 98% | -- |
| 10 | 24 | 98% | -- | 99% | 97% |
| 15 | 24 | **98%** | -- | **98%** | **100%** |

---

## Limitations

- **Structured templates**: facts must follow "{concept} {linking_token} {entity}" pattern (e.g., "My name is John"). Free-form text is not supported.
- **Single-token entities**: entity values must be single BPE tokens. Multi-token entities (e.g., "New York") require tokenizer-level solutions.
- **Linking-token dependency**: storage is triggered by specific linking tokens ("is", "in", "at", "from"). Facts without these tokens are not stored.
- **Template-locked retrieval**: questions must use the same concept word as the fact template. Paraphrased questions (e.g., "What do people call you?" instead of "What is my name?") fail.
- **No ownership discrimination**: the trace cannot distinguish "my name" from "Alice's name" — both produce the same context-free Q.

---

## Repository Structure

```
hebbian-trace-memory/
├── hebbian_trace/
│   ├── __init__.py
│   ├── model.py           # HebbianTraceModule + GPT2WithTrace
│   └── tasks.py           # Fact types, evaluation infrastructure
├── demo.py                # Multi-session demo
├── evaluate.py            # Reproduce paper results
├── weights/
│   └── trace_module.pt   # Trained gate weights (~6KB)
├── figures/
│   ├── generate.py        # Reproducible figure generation
│   ├── architecture.png
│   ├── retention_curve.png
│   ├── ablation_chart.png
│   └── cross_context.png
├── ARCHITECTURE.md        # Detailed component descriptions
├── requirements.txt       # torch, transformers
└── LICENSE                # Apache 2.0
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- ~500MB disk for GPT-2 model (downloaded automatically)

---

## Citation

```bibtex
@article{hebbian-trace-memory,
  title={Persistent Cross-Session Memory for Frozen Language Models via Bio-Inspired Hebbian Trace Module},
  author={Andrey Pustovit},
  year={2026},
  note={Preprint}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
