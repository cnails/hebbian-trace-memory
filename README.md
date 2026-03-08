# Hebbian Trace Memory

**Persistent cross-session memory for frozen LLMs via bio-inspired Hebbian trace module.**

An external memory module (~1.1M parameters) that attaches to frozen LLMs and provides persistent fact storage across sessions — without fine-tuning, without RAG, without retraining. Validated on GPT-2 Small (124M), GPT-2 Medium (355M), and Phi-2 (2.7B).

---

## Flagship Result

**98% recall across 15 sessions with 24 distinct fact types.**

The trace module accumulates knowledge session by session. Facts stored in session 1 remain retrievable at session 15, even as new facts are added and existing ones are updated.

<p align="center">
  <img src="figures/retention_curve.png" width="700" alt="Retention curve: 98% recall across 15 sessions">
</p>

| Property | Value |
|----------|-------|
| Mean recall across sessions | **98.6%** |
| Session 15 recall (all 24 facts) | **98%** |
| Single-pass recall at n=24 | 94.4% (CI: 92.3–96.5%) |
| Fact types | 24 (name, city, company, color, food, pet, ...) |
| Sessions | 15 (introduction + updates) |
| Base model | GPT-2 Small (frozen, unmodified) |
| Cross-architecture | Phi-2 (2.7B): 98.4% at n=5 (zero-shot) |
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
python exp_lama.py --quick        # LAMA benchmark (~2 min)
python exp_lama.py --n-eval 100   # LAMA full run (~10 min)
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

### Capacity Stress Test

How many facts can the trace store before accuracy degrades?

<p align="center">
  <img src="figures/capacity_curve.png" width="600" alt="Capacity stress test">
</p>

Pattern separation extends the capacity frontier by ~2x: 95% accuracy at ~31 facts, 80% at ~48 facts.

### RAG Comparison

<p align="center">
  <img src="figures/rag_comparison.png" width="700" alt="Hebbian Trace vs RAG baselines">
</p>

In the realistic regime (24 types, 229 entity candidates), the trace outperforms RAG (k=1) by up to +42pp at n=1.

### Model Scaling

The trace mechanism generalizes from GPT-2 Small (124M) to GPT-2 Medium (355M):

<p align="center">
  <img src="figures/model_scaling.png" width="700" alt="Model scaling: Small vs Medium">
</p>

### Cross-Architecture Transfer: Phi-2 (2.7B)

Zero-shot transfer to Microsoft Phi-2 — a completely different architecture (parallel attention, rotary embeddings, CodeGen tokenizer). No trained weights loaded, only random projections + linking-token mask.

| n_facts | GPT-2 Small (124M) | Phi-2 (2.7B) | Delta | Phi-2 in-context |
|---------|:------------------:|:------------:|:-----:|:----------------:|
| 1 | 100.0% | **100.0%** | 0.0 | 98.0% |
| 3 | 100.0% | **99.3%** | −0.7 | 83.3% |
| 5 | 90.4% | **98.4%** | **+8.0** | 94.4% |
| 7 | 85.7% | **92.9%** | **+7.1** | 97.4% |

Phi-2 **exceeds GPT-2 Small** at n≥5. Trace params = 5.3M (0.19% of Phi-2).

*50 episodes, pattern separation 8x_k16, seed=42.*

### LAMA Knowledge Probes

Evaluation on the standard LAMA T-REx benchmark (Petroni et al., 2019) with real-world Wikidata facts:

| n_facts | Cross-context (trace) | In-context (GPT-2) | No memory |
|---------|:--------------------:|:-----------------:|:---------:|
| 1 | **100.0%** | 95.0% | 1.0% |
| 3 | **94.0%** | 43.3% | 0.3% |
| 5 | **93.6%** | 41.6% | 0.2% |
| 7 | **88.7%** | 29.7% | 0.3% |
| 10 | **81.2%** | 24.7% | 0.1% |

The trace exceeds GPT-2's in-context baseline by +50–60pp at n>=3. Coverage is limited to ~6% of LAMA T-REx due to the single-token entity constraint.

*100 episodes, seed=42. Reproducible via `python exp_lama.py --n-eval 100`.*

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
├── exp_lama.py            # LAMA T-REx benchmark evaluation
├── capacity_test.py       # Capacity stress test (1–100 facts)
├── weights/
│   └── trace_module.pt    # Trained gate weights (~6KB)
├── figures/
│   ├── generate.py        # Reproducible figure generation
│   ├── architecture.png
│   ├── retention_curve.png
│   ├── ablation_chart.png
│   ├── cross_context.png
│   ├── capacity_curve.png
│   ├── rag_comparison.png
│   └── model_scaling.png
├── paper.tex              # LaTeX paper (arXiv-ready)
├── references.bib         # BibTeX references
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
@article{pustovit2026hebbian,
  title={Persistent Memory for Frozen Language Models via Bio-Inspired Hebbian Trace},
  author={Pustovit, Andrey},
  year={2026},
  note={arXiv preprint}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
