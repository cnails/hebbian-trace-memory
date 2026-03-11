# Hebbian Trace: Persistent Memory for Frozen Language Models

**A lightweight external memory module that gives frozen LLMs the ability to remember across sessions — no fine-tuning, no vector database, no retrieval infrastructure.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## Why Hebbian Trace?

**Forget vector DBs.** Store facts directly in a Hebbian trace matrix with a single 0.4ms write. No indexing, no embeddings pipeline, no infrastructure.

**True zero-shot transfer.** Attach to GPT-2, Phi-2, LLaMA-2, or Mistral without training on them. Same module, same code, five model scales, three tokenizer families.

**Neurally pluggable.** ~1.1M parameters — smaller than a single transformer layer. Injects via logit space, so it never touches the frozen model's internal representations.

**1,000 facts at 99.4%.** Hashed trace banks scale capacity with zero latency overhead. The LM forward pass takes ~17ms; bank routing adds <0.1ms.

---

## What It Does

LLMs are stateless — every conversation starts from scratch. Hebbian Trace fixes this by attaching a bio-inspired external memory that:

- **Persists across sessions.** Store a fact today, retrieve it next week. The trace serializes to disk.
- **Updates organically.** "I moved to London" automatically overwrites "I live in Paris" via reconsolidation erasure.
- **Resolves paraphrases.** Ask "What do people call you?" and get back the answer stored under "My name is John" (+83pp improvement).
- **Chains multi-hop reasoning.** "Where does John live?" -> Paris -> "What country is Paris in?" -> France. Two trace lookups, zero new parameters.
- **Filters noise.** Feed it a paragraph with 80% filler; dual semantic gates store only the meaningful facts (167x selectivity).

<p align="center">
  <img src="figures/retention_curve.png" width="700" alt="98% recall across 15 sessions with 24 fact types">
</p>

---

## How It Works (in 30 seconds)

1. **Address by token identity.** The word "name" always maps to the same Q-vector, regardless of context. This is why cross-session retrieval works — and why kNN-LM (which uses contextual hidden states) fails (-63pp).

2. **Store via outer product.** `T += Q_concept^T * V_entity`. One matrix multiply. 0.4ms.

3. **Retrieve via matrix-vector product.** `V = Q_query * T`. Project through the LM's own embedding matrix. Inject as logit bias. The frozen model does the rest.

4. **Scale via partitioning.** Hash the sparse Q pattern to route facts into separate banks. Each bank sees ~N/B facts. Interference drops. Capacity scales linearly. Cost: one argmax.

---

## Architecture

Seven bio-inspired, independently ablated components. Each targets a specific bottleneck:

<p align="center">
  <img src="figures/architecture.png" width="700" alt="Architecture: trace module attached to frozen GPT-2">
</p>

| Component | Biological Analog | What It Solves | Impact |
|-----------|------------------|----------------|:------:|
| Pattern Separation | Dentate gyrus | Storage interference | +9-26pp |
| Dual Gating | Cholinergic modulation | Noise filtering | +46pp |
| Reconsolidation Erasure | Memory reconsolidation | Fact updates | +38pp |
| CLS Decay | Fast/slow learning | Temporal dynamics | +7.4pp |
| Autoassociative Trace (T_auto) | CA3 autoassociative | Paraphrase resolution | +83pp |
| Hashed Banks | Sparse distributed memory | Capacity scaling | 9.8% -> 99.4% at n=1000 |
| Logit Injection | Hippocampal output | Residual stream scale mismatch | Model-agnostic |

All seven mechanisms compose orthogonally — their effects are additive. See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed formulas.

<p align="center">
  <img src="figures/ablation_chart.png" width="600" alt="Cumulative contribution of each mechanism">
</p>

---

## Quick Start

```bash
git clone https://github.com/apustovit/hebbian-trace.git
cd hebbian-trace
pip install -r requirements.txt
python demo.py
```

First run downloads GPT-2 Small (~500MB). Subsequent runs use cached model.

```python
from hebbian_trace.model import GPT2WithTrace

# Create trace-augmented model
model = GPT2WithTrace(
    n_trace_heads=8, d_trace=64,
    alpha=0.5, trace_lr=1.0, trace_decay=0.99,
)
model.enable_pattern_separation(expand_factor=8, top_k=16)

# Write phase: store facts
model.set_trace_mode(use=False, update=True)
model(tokenized_fact)  # "My name is Alice"

# Read phase: retrieve across sessions
model.set_trace_mode(use=True, update=False)
logits = model(tokenized_query)  # "What is my name?" -> "Alice"

# Direct write API (no template needed)
model.write_fact_direct(concept_token_id, entity_token_id)

# Multi-hop chain
model.write_fact_direct(city_id, country_id)  # Paris -> France
answer = model.retrieve_direct(concept_id, candidate_ids)  # hop 1
answer = model.retrieve_direct(answer_id, candidate_ids)   # hop 2
```

### Reproduce Paper Results

```bash
python evaluate.py --n-eval 100        # Multi-session flagship (Table 4)
python exp_lama.py --n-eval 100        # LAMA T-REx benchmark (Table 6)
python -m hebbian_trace.experiments.exp27_hotpotqa --no-oracle --batch-sweep --bank-configs 0 32  # Multi-hop HotpotQA (Table 8)
python exp_paraphrase.py --n-eval 50   # Paraphrase resolution (Table 9)
python capacity_test.py --banks 16     # Capacity with hashed banks
```

---

## Key Results

### Capacity Scaling (LLaMA-2 7B)

<p align="center">
  <img src="figures/capacity_curve.png" width="600" alt="Capacity scaling with hashed trace banks">
</p>

| n_facts | Baseline | 16 banks | 64 banks | 128 banks |
|:-------:|:--------:|:--------:|:--------:|:---------:|
| 100 | 76.1% | 99.7% | 99.8% | 99.9% |
| 500 | 18.9% | 96.5% | 99.6% | 99.7% |
| 1000 | 9.8% | 86.6% | 98.5% | **99.4%** |

*50 episodes, LLaMA-2 7B fp16, A100. Degradation at 128 banks: -0.3pp per 500 facts — capacity knee beyond ~2,000.*

Latency overhead: **zero measurable** (17.60ms baseline -> 17.65ms with 128 banks).

### Cross-Architecture Transfer (Zero-Shot)

<p align="center">
  <img src="figures/model_scaling.png" width="700" alt="Transfer across five model scales">
</p>

| Model | Params | d_model | alpha | Trace H x d | n=1 | n=3 | n=5 |
|-------|:------:|:-------:|:-----:|:-----------:|:---:|:---:|:---:|
| GPT-2 Small | 124M | 768 | 0.5 | 8 x 64 | **100%** | 89.7% | 85.4% |
| GPT-2 Medium | 355M | 1024 | 0.5 | 8 x 64 | **100%** | 72.7% | 70.4% |
| Phi-2 | 2.7B | 2560 | 50 | 8 x 64 | **100%** | 99.3% | **98.4%** |
| LLaMA-2 7B | 7B | 4096 | 20 | 32 x 64 | **100%** | 94.0% | 85.6% |
| Mistral 7B | 7B | 4096 | 1000 | 32 x 64 | 96% | 73.3% | 67.6% |

*Five models, three tokenizer families (BPE, CodeGen, SentencePiece), four architectures.*

**Counterfactual override** — the trace can suppress even 7B-scale pretrained priors:

| Model | Prior | Trace n=1 | n=3 | n=5 |
|-------|:-----:|:---------:|:---:|:---:|
| GPT-2 Small (124M) | 0% | **96%** | 90% | 80.4% |
| LLaMA-2 7B | 4% | 64% | 55.3% | 51.6% |
| Mistral 7B | 0% | 70% | 52.7% | 48.4% |

### Cross-Context Retrieval

Facts stored in one phrasing, queried with another. This is the core test of concept-addressed vs contextual memory.

<p align="center">
  <img src="figures/cross_context.png" width="600" alt="Cross-context retrieval: trace vs kNN-LM vs in-context">
</p>

| n_facts | Cross-context | kNN-LM | In-context (GPT-2) | No trace |
|:-------:|:------------:|:------:|:------------------:|:--------:|
| 1 | **100.0%** | 100.0% | 99.0% | 6.0% |
| 3 | **89.7%** | 37.0% | 73.0% | 4.3% |
| 5 | **85.4%** | 22.4% | 62.8% | 3.2% |
| 7 | **82.0%** | 14.9% | 61.6% | 4.3% |

*kNN-LM degrades to near-baseline because contextual hidden states diverge when phrasing changes. The trace addresses by token identity — phrasing-invariant by design.*

### HotpotQA Multi-Hop (4,159 Bridge Questions, Non-Oracle)

Real-world 2-hop retrieval **without oracle support**: bridge entities identified automatically from context paragraphs (no supporting fact annotations). First BPE token answer match. Not comparable to end-to-end QA leaderboard entries.

<p align="center">
  <img src="figures/multihop_capacity.png" width="600" alt="Multi-hop retrieval scaling">
</p>

| Mode | N | E2E (no banks) | E2E (32 banks + best-bank) |
|------|:-:|:--------------:|:--------------------------:|
| Per-question | 1 | **100%** (4,159/4,159) | **100%** |
| Batched | 5 | 98.0% | **100%** |
| Batched | 10 | 88.2% | **98.8%** |
| Batched | 15 | 62.7% | **98.7%** |

*Auto bridge detection agrees with oracle on 94.6% of shared questions. 32 banks + best-bank scan (no oracle at read time) resolves first-token collisions: +36pp at batch 15.*

### LAMA T-REx (2,034 Wikidata Facts)

| n_facts | Oracle (32 rel) | Auto (40 rel) | In-context | No memory |
|:-------:|:--------------:|:-------------:|:----------:|:---------:|
| 1 | **100.0%** | 99.0% | 95.0% | 1.0% |
| 5 | 93.8% | **94.2%** | 41.6% | 0.2% |
| 10 | 81.1% | **83.9%** | 24.7% | 0.1% |

*Auto concept-word assignment (zero manual curation) matches oracle within 95% CI. +50-60pp vs GPT-2 in-context baseline.*

### Paraphrase Resolution via T_auto

<p align="center">
  <img src="figures/paraphrase_tauto.png" width="600" alt="T_auto paraphrase resolution">
</p>

| | Aligned | | Misaligned | | Semantic | |
|:-:|:---:|:---:|:---:|:---:|:---:|:---:|
| **n** | **Std** | **+T_auto** | **Std** | **+T_auto** | **Std** | **+T_auto** |
| 1 | 100% | 100% | 17% | **100%** | 27% | **100%** |
| 3 | 100% | 100% | 22% | **95%** | 30% | **100%** |
| 5 | 100% | 100% | 19% | **86%** | 33% | **100%** |
| 7 | 100% | 100% | 24% | **82%** | 31% | **100%** |

*Semantic queries (entirely different phrasing) achieve 100% across all fact counts.*

### RAG Comparison

<p align="center">
  <img src="figures/rag_comparison.png" width="700" alt="Hebbian Trace vs RAG baselines">
</p>

In the realistic regime (24 types, 229 entity candidates), the trace outperforms Oracle RAG (perfect retrieval upper bound) by up to +42pp. Concept-addressed retrieval is phrasing-invariant; RAG degrades with vocabulary size.

---

## Comparison with Alternatives

| | Hebbian Trace | RAG | Fine-tuning | kNN-LM |
|---|:---:|:---:|:---:|:---:|
| Modifies weights | No | No | Yes | No |
| Infrastructure | None | Vector DB | Training loop | Datastore |
| Incremental updates | 0.4ms write | Re-index | Retrain | Append |
| Cross-phrasing retrieval | **Yes** (+63pp vs kNN) | Partial | Yes | No |
| Capacity | ~1,000 facts @99% | 10^6+ docs | Unbounded | Unbounded |
| Latency overhead | ~0.1ms | ~50-200ms | 0 | ~10-50ms |
| Memory footprint | O(d^2), fixed | O(N*d), linear | O(1) | O(N*d), linear |

The trace handles fast-changing, user-specific "episodic" memory that RAG and fine-tuning address poorly. A practical deployment combines all three: trace for per-user facts, RAG for organizational knowledge, frozen model for world knowledge.

---

## Limitations

- **Single-token entities** (partially addressed): entity values must be single BPE tokens. Multi-token entities use first-token addressing; hashed banks resolve 95%+ of collisions.
- **Context-free addressing trade-off**: cannot distinguish "my name" from "Alice's name". T_auto resolves the paraphrase problem but ownership discrimination remains open.
- **Structured input** (largely addressed): LLM-based extraction (Flan-T5-small) achieves 94% F1 on unconstrained text where regex fails entirely.
- **Model-specific geometry**: Mistral 7B requires alpha=1000 (vs 10 for LLaMA-2 at the same d_model) and doesn't benefit from head scaling.

---

## Repository Structure

```
hebbian-trace-memory/
  hebbian_trace/
    __init__.py
    model.py           # HebbianTraceModule + GPT2WithTrace
    tasks.py           # Fact types, evaluation, concept vocab, chains
  demo.py              # Interactive multi-session demo
  evaluate.py          # Flagship evaluation with RAG baselines
  exp_lama.py          # LAMA T-REx benchmark
  exp_multihop.py      # Multi-hop HotpotQA benchmark
  exp_paraphrase.py    # Paraphrase resolution via T_auto
  capacity_test.py     # Capacity stress test (1-100 facts)
  medium_test.py       # GPT-2 Medium (355M) transfer test
  weights/
    trace_module.pt    # Trained gate weights (~6KB)
  figures/
    generate.py        # Reproducible figure generation
  paper.tex            # arXiv paper
  ARCHITECTURE.md      # Detailed component descriptions
  requirements.txt     # torch, transformers
  LICENSE              # Apache 2.0
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- ~500MB disk for GPT-2 model (downloaded automatically)

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
