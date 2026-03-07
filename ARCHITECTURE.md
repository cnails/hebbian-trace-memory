# Architecture

Detailed description of each component in the Hebbian Trace Memory module.

The trace module (~1.1M parameters) attaches to a frozen GPT-2 Small (124M parameters) as an external memory system. GPT-2 is never modified — all memory functionality comes from the trace module alone.

---

## Overview

```
Input tokens ──> [Dual Gates] ──> [Pattern Separation] ──> [Hebbian Trace] ──> [Logit Injection] ──> Output logits
                      │                    │                      │                    │
                 ACh modulation      Dentate gyrus          CA3 memory         Hippocampal output
                                                                                     │
                                        Frozen GPT-2 ─────────────────────────── base logits
```

The trace module operates in two phases, inspired by acetylcholine (ACh) modulation in the hippocampus:

- **Write phase** (`use=False, update=True`): encode facts into trace, suppress retrieval
- **Read phase** (`use=True, update=False`): retrieve from trace, suppress encoding

This separation prevents partial-trace interference — retrieval from an incomplete trace during encoding corrupts the association.

---

## 1. Context-Free Q/V (Hippocampal Indexing)

**Problem**: Standard transformer Q/K/V representations are context-dependent — the same word produces different representations in different sequences. This prevents cross-session retrieval: a fact stored in one context cannot be found from another.

**Solution**: Compute Q and V as deterministic functions of token identity only, with no positional encoding and no contextual information:

```
Q = W_proj(LayerNorm(wte(token)))    [storage/retrieval keys]
V = W_val(wte(token))                [storage values]
```

where `wte` is GPT-2's frozen token embedding layer.

**Why LayerNorm matters**: GPT-2's W_q was trained on LayerNorm'd inputs. Applying W_proj (random projection in the same space) to raw embeddings without LayerNorm produces out-of-distribution inputs with zero effect. LayerNorm normalizes the embedding distribution to match what linear projections expect.

**Why context-free**: The same word always maps to the same Q vector regardless of surrounding tokens. "My **name** is John" and "What is my **name**?" both produce Q("name") = W_proj(LN(wte("name"))). This enables cross-session retrieval — facts stored in session 1 are retrievable in session 15.

**Shift-1 offset**: Storage uses Q at the concept position (one before the linking token) paired with V at the entity position (one after the linking token):

```
    "My  name  is   John"
         Q[1]  ←──  V[3]     (concept Q, entity V)
              link
```

When "is" appears at position j, the mask activates Q at position j-1 (= Q("name")) and V at position j+1 (= V("John")). The retrieval query Q("name") then recovers V("John").

---

## 2. Pattern Separation (Dentate Gyrus)

**Problem**: Different concept words ("name", "city", "color") have Q vectors with high cosine similarity (mean 0.477, max 0.660). Storing multiple facts causes interference — retrieving Q("name") partially activates the V stored under Q("city").

**Solution**: Sparse random expansion inspired by the dentate gyrus, which expands cortical inputs into a much larger, sparser representation in the hippocampus:

```
Q_expanded = ReLU(Q @ W_expand)          [d_trace → d_trace * factor]
Q_sparse   = top_k(Q_expanded, k)        [keep only k largest activations]
```

`W_expand` is a frozen random matrix (Johnson-Lindenstrauss projection). It is never trained — the random structure is sufficient for dimensionality expansion. Deterministic: same token always produces the same sparse code.

**Parameters**: `expand_factor=8, top_k=16` (d_trace=64 → 512 dimensions, 16 active).

**Effect on Q overlap**:

| Metric | Raw Q | After pattern separation |
|--------|-------|------------------------|
| Mean cosine | 0.477 | 0.308 |
| Max cosine | 0.660 | 0.497 |
| Mean IoU | — | 0.173 |

The reduced overlap translates directly to less interference between stored facts:

| n_facts | Without PS | With PS (8x, k=16) | Improvement |
|---------|-----------|-------------------|-------------|
| 3 | 87.3% | 89.7% | +2.4pp |
| 5 | 67.8% | 85.4% | +17.6pp |
| 7 | 65.6% | 82.0% | +16.4pp |

---

## 3. Hebbian Trace (CA3 Associative Memory)

**Problem**: Standard transformers have no persistent memory — all information is lost between forward passes.

**Solution**: An external association matrix `T_v` that accumulates Q→V associations via Hebbian outer-product learning:

```
T_v ← decay * T_v + lr * (Q_store^T @ V_store) / denom
```

where:
- `T_v`: (H, d_addr, d_trace) trace matrix — one per head
- `Q_store`: concept-word Q at linking positions
- `V_store`: entity-word V at entity positions
- `decay=0.99`: exponential decay prevents unbounded growth
- `lr=1.0`: learning rate for new associations
- `denom`: normalization by number of active positions

**Retrieval**: Query the trace with a concept-word Q to recover the associated V:

```
V_retrieved = Q_addr @ T_v
output = W_out(V_retrieved)
```

The retrieval Q uses shift-1 addressing: at position i, the retrieval key is Q from position i-1. This matches the storage convention where Q("name") was stored from the concept position.

**No gradient required**: The trace update is a simple outer product — no backpropagation, no optimization. This makes it compatible with any frozen pretrained model.

---

## 4. Dual Gates (ACh Modulation)

**Problem**: Not all tokens should trigger trace storage. "My name is John" should store Q("name")→V("John"), but "The weather is nice" should store nothing. A single position-level gate cannot distinguish these — both contain the linking token "is".

**Solution**: Two complementary gates, inspired by acetylcholine modulation:

```
gate_pos = sigmoid(W_gate(wte(token)) / tau)       [WHERE — linking token detector]
gate_key = sigmoid(W_gate_key(wte(token)) / tau)    [IF — concept relevance]
```

**Position gate** (`gate_pos`): fires on linking tokens ("is", "in", "at", "from"). Determines WHERE in the sequence a fact boundary exists.

**Concept gate** (`gate_key`): evaluates the concept word one position before the linking token. Determines IF the fact is worth storing. High for fact concepts ("name" → 0.35), low for filler concepts ("weather" → 0.06).

**Combined gate**:
```
gate_pos_mid    = gate_pos[:, 1:-1]        [linking positions]
gate_key_concept = gate_key[:, :-2]        [concept positions, shift-1]
combined = gate_pos_mid * gate_key_concept
Q_store *= combined
```

**Selectivity**: 103x ratio between fact and filler gate activations. Trained via cross-context retrieval loss — no manual token labels.

**Training**: Each gate is a single linear layer (769 parameters). Trained sequentially:
1. `W_gate` (5 stages): learns to detect linking tokens from retrieval loss alone
2. `W_gate_key` (4 stages): learns concept relevance on paragraph inputs with interleaved filler

---

## 5. Logit Injection (Hippocampal Output)

**Problem**: GPT-2's hidden representations have norms ~3000. The trace module's retrieved values have norms ~0.06. Injecting trace output into GPT-2's residual stream would be 50,000x too small to have any effect.

**Solution**: Bypass the residual stream entirely. Project retrieved values to vocabulary space via weight-tying with GPT-2's embedding matrix:

```
trace_logits = W_out(Q_addr @ T_v) @ wte^T
logits = GPT2_logits + alpha * trace_logits
```

where:
- `W_out`: (d_model, H * d_trace) projects from trace space to model space
- `wte^T`: (vocab_size, d_model) GPT-2's transposed embedding matrix
- `alpha=0.5`: injection strength

The trace module contributes a logit bias that directly shifts the probability of each vocabulary token. No modification to GPT-2's internal representations.

**Orthogonal initialization**: W_val and W_out are initialized as approximate inverses via QR decomposition, ensuring the round-trip `W_out(W_val(x)) ≈ x`. This means the trace stores and retrieves token embeddings with minimal distortion.

---

## 6. Reconsolidation Erasure (Memory Reconsolidation)

**Problem**: When a fact is updated ("My city is Moscow" → "My city is London"), both old and new associations exist in the trace. Q("city") retrieves a superposition of V("Moscow") and V("London"), degrading accuracy.

**Solution**: Before writing a new association, erase the old one by subtracting the current Q→V mapping:

```
Q_norm = Q / ||Q||                                    [L2 normalize]
V_old  = Q_norm @ T_v                                 [retrieve old value]
T_v   -= erase_lr * (Q_norm^T @ V_old) / denom        [subtract old association]
T_v   += lr * (Q^T @ V_new) / denom                   [write new association]
```

**Why L2 normalization is critical**: Pattern-separated Q vectors have ||Q||^2 ~ 175 per head (top-k of expanded dimensions with high activation magnitudes). Without normalization, the erase step removes 20x+ the actual stored value, causing catastrophic trace destruction. L2 normalization makes the erase scale-independent.

**Selective activation**: Erasure is enabled only during update phases, not initial encoding. This preserves the accumulative property of the trace during introduction sessions.

**Effect** (15-session demo, 24 fact types):

| Configuration | Session 15 recall |
|--------------|------------------|
| No erasure | 64% |
| With erasure (lr=5.0) | **98%** |

---

## Parameter Summary

| Component | Parameters | Notes |
|-----------|-----------|-------|
| W_proj | 768 x 512 | Q projection |
| ln_proj | 768 + 768 | LayerNorm (weight + bias) |
| W_val | 512 x 768 | V projection |
| W_out | 768 x 512 | Output projection |
| W_gate | 768 x 1 + 1 | Position gate |
| W_gate_key | 768 x 1 + 1 | Concept gate |
| W_expand | 64 x 512 | Frozen random (not trained) |
| T_v | 8 x 512 x 64 | Runtime buffer (not saved) |
| **Total trainable** | **~1.1M** | |

GPT-2 Small: 124M parameters (frozen, unmodified).

---

## Biological Correspondence

| Component | Brain Region | Shared Principle |
|-----------|-------------|-----------------|
| Context-free Q/V | Hippocampal place cells | Location-independent identity coding |
| Pattern separation | Dentate gyrus | Sparse expansion to reduce overlap |
| Hebbian trace | CA3 recurrent network | Outer-product associative storage |
| Dual gates | Cholinergic modulation | Encoding/retrieval mode switching |
| Logit injection | Hippocampal-cortical output | Memory influences decisions without modifying representations |
| Reconsolidation erasure | Memory reconsolidation | Reactivated memories become labile and can be updated |

These correspondences are functional analogies, not claims of biological fidelity. The module uses simplified versions of biological mechanisms that address the same computational problems.
