# Persistent Memory for Frozen Language Models via Bio-Inspired Hebbian Trace

## Abstract

Large language models lack persistent memory — knowledge acquired during one session is lost when the context resets. We introduce the Hebbian Trace Module, a lightweight external memory that attaches to frozen pretrained LLMs and demonstrates the feasibility of cross-session knowledge retention without fine-tuning. Inspired by hippocampal memory systems, our architecture combines five composable mechanisms: sparse pattern separation (dentate gyrus analog), Hebbian associative storage (CA3), dual semantic gating (cholinergic modulation), reconsolidation erasure, and complementary learning systems via per-head specialization. The trace module stores associations at inference time through Hebbian outer-product updates and retrieves them via logit-space injection. On structured fact association tasks, the module attached to frozen GPT-2 Small (124M parameters) with ~1.1M trainable parameters achieves 98–99% mean cross-session recall when 24 fact types are introduced incrementally (3–5 per session) across 15 sessions with fact updates, processing natural paragraph input with noise filtering. In single-pass capacity tests, accuracy is 94.4% (95% CI: 92.3–96.5%) at 24 simultaneous facts. We validate each component through systematic ablation across 16 experiments, demonstrating that all five mechanisms compose orthogonally. Our results suggest that persistent, updatable memory can be added to pretrained language models as a modular plug-in, without modifying model weights, though significant challenges remain for unconstrained natural language.

---

## 1. Introduction

Current large language models operate as stateless systems — each conversation begins with a blank slate. While in-context learning allows temporary use of information within a session, this knowledge is lost when the context window resets. Existing solutions each have fundamental limitations: retrieval-augmented generation (RAG) requires explicit indexing infrastructure and does not learn or update organically; fine-tuning is computationally expensive, risks catastrophic forgetting, and cannot operate incrementally at inference time; and extended context windows, while accommodating more tokens, still reset between sessions and scale quadratically in compute.

The human brain solves this problem through the hippocampal memory system — a fast-learning complementary structure that rapidly encodes episodic memories, updates them through reconsolidation, and operates alongside the slow-learning neocortex. The hippocampus achieves this through a coordinated set of mechanisms: sparse coding in the dentate gyrus minimizes interference between memories, autoassociative networks in CA3 store and retrieve patterns, cholinergic modulation gates encoding versus retrieval modes, and reconsolidation enables selective updating of existing memories.

We propose the Hebbian Trace Module, an external memory system that translates these hippocampal principles into a computational module attachable to any frozen pretrained language model. Our key contributions are:

1. **A modular persistent memory architecture** comprising five bio-inspired, independently validated components that compose orthogonally, enabling cross-session knowledge retention for frozen LLMs.

2. **Architecture-agnostic transfer from custom to pretrained models.** Mechanisms developed on a small custom transformer (6.4M parameters) transfer directly to GPT-2 Small (124M parameters) with random projections and no fine-tuning, achieving 100% single-fact recall.

3. **Systematic ablation across 16 experiments**, isolating the contribution of each component and mapping architectural limitations. We demonstrate +26pp from pattern separation, +46pp from semantic gating, and +38pp from reconsolidation erasure on targeted benchmarks.

4. **A flagship demonstration** of 98–99% mean recall across 15 sessions with 24 incrementally introduced fact types, including fact updates and natural paragraph input with noise filtering, using only ~1.1M trainable parameters attached to a frozen 124M-parameter model.

---

## 2. Related Work

**Fast weights and associative memory.** The idea of rapidly updated associative weights has a long history. Ba et al. (2016) proposed fast weights as outer-product accumulations for short-term memory in RNNs. Schlag et al. (2021) connected linear attention to fast weight memories. Our work extends this line by introducing sparse addressing, gated writing, and selective erasure — mechanisms absent from prior fast weight systems.

**Modern Hopfield Networks.** Ramsauer et al. (2020) showed that transformer attention can be viewed as a modern Hopfield network with exponential storage capacity. While our Hebbian trace uses similar outer-product storage, we operate as an *external* module with explicit write/read separation, gated storage, and cross-session persistence — properties not present in the attention-as-memory framework.

**Memory-augmented transformers.** Several works have added external memory to transformers: Memorizing Transformers (Wu et al., 2022) cache key-value pairs from previous contexts; MemoryLLM (Wang et al., 2024) integrates updatable memory within model parameters. Our approach differs in three ways: (1) the base model remains completely frozen, (2) memory updates use Hebbian learning without backpropagation, and (3) we provide explicit mechanisms for selective erasure and fact updates.

**Sparse distributed memory.** Kanerva's (1988) SDM introduced sparse high-dimensional addressing for associative memory. Our pattern separation mechanism is a direct computational analog, using random sparse projections to minimize address overlap. We demonstrate that this classical technique provides consistent +12-26pp improvements when integrated with modern transformer architectures.

**Recurrent and linear-attention architectures.** Recent architectures maintain persistent recurrent state across sequences: RWKV (Peng et al., 2023) uses linear attention with time-mixing; Mamba (Gu & Dao, 2023) employs selective state spaces; RetNet (Sun et al., 2023) combines recurrence with parallel training. These approaches bake memory into the architecture itself, requiring training from scratch. In contrast, our module attaches to existing frozen models post-hoc, preserving the base model's capabilities while adding persistent memory as a separable component.

**Compressive and infinite-context memory.** Infini-attention (Munkhdalai et al., 2024) extends transformers with a compressive memory that accumulates key-value states across segments, enabling unbounded context within a single session. Our approach differs in targeting *cross-session* persistence — memory survives context resets — and in using Hebbian learning rather than gradient-based memory updates, enabling operation on fully frozen models.

**Complementary Learning Systems.** McClelland et al. (1995) proposed that memory relies on complementary fast (hippocampal) and slow (neocortical) learning systems. Our per-head decay specialization (Section 3.5) implements this principle within the trace module, with fast-decay heads tracking recent updates and slow-decay heads maintaining stable long-term associations.

---

## 3. Architecture

The Hebbian Trace Module attaches to a frozen pretrained language model as an external memory. It intercepts token embeddings for addressing, stores associations via Hebbian learning, and injects retrieved information into the model's output logits. Figure 1 shows the complete architecture.

### 3.1 Core Mechanism: Hebbian Associative Trace

The trace stores key-value associations in a matrix T_v ∈ R^{d×d} (per attention head) via rank-1 Hebbian updates:

    T_v ← γ · T_v + η · Q_store^T · V_store

where γ is a decay factor, η is a learning rate, Q_store is the key vector derived from the concept word's embedding, and V_store is the value vector derived from the entity embedding. Retrieval computes:

    V_retrieved = Q_addr · T_v

The retrieved values are projected back to model dimension and injected as a logit bias:

    logits ← logits + α · W_out(V_retrieved) · W_embed^T

where W_embed is the model's token embedding matrix (weight-tied with the language model head), and α controls injection strength. This logit-space injection directly biases predictions toward stored entities without requiring injection into intermediate residual streams.

**Addressing via token embeddings.** Both Q and V are derived from the model's token embeddings (wte) through learned projections: Q = W_proj(LN(wte(token))), V = W_val(wte(token)). Using raw embeddings rather than contextual hidden states provides context-free addressing — the same concept word produces the same address regardless of surrounding context, enabling cross-session retrieval.

### 3.2 Pattern Separation (Dentate Gyrus Analog)

In the hippocampus, the dentate gyrus creates sparse, decorrelated representations from cortical input, minimizing interference between similar memories. We implement this as a frozen random sparse expansion:

    Q_sparse = TopK(ReLU(W_expand · Q), k)

where W_expand ∈ R^{(f·d)×d} is a frozen random matrix (expansion factor f, typically 8×), and TopK retains only the k largest activations (typically 16). This transforms dense Q vectors into sparse codes where different concepts activate nearly non-overlapping dimensions.

The expansion matrix W_expand receives no gradient and remains random — directly analogous to the largely random mossy fiber projections from dentate gyrus to CA3. This is theoretically grounded in the Johnson-Lindenstrauss lemma: random projections preserve pairwise distances, while sparsification further decorrelates representations.

**Empirical impact.** Pattern separation consistently provides +9-26pp accuracy improvement across all experimental settings — from custom transformers to pretrained GPT-2, from single facts to multi-session scenarios. At expansion 8× with k=16 (6.25% sparsity), mean cosine similarity between concept Q-vectors drops from 0.477 to 0.308, with intersection-over-union of 0.173.

### 3.3 Dual Semantic Gating (Cholinergic Modulation Analog)

In the hippocampus, acetylcholine modulates the transition between encoding (high ACh) and retrieval (low ACh) modes, controlling what information enters long-term storage. We implement this through two complementary gates:

**Position gate (gate_pos):** A learned linear classifier on token embeddings that identifies *where* linking tokens occur — positions that signal key-value relationships (e.g., "is", "in", "at"):

    gate_pos = σ(W_gate · wte(token))

**Semantic gate (gate_key):** A learned linear classifier that evaluates *whether* the concept word at the linking position is worth storing:

    gate_key = σ(W_gate_key · wte(token))

The combined gate is their product: combined = gate_pos[link_pos] · gate_key[concept_pos]. This dual gating enables paragraph-level storage: the model processes natural text containing both relevant facts and irrelevant filler, storing only meaningful associations.

**Discovery without supervision.** Both gates are trained purely through cross-context retrieval loss — no explicit labels for linking tokens or concept words. The position gate achieves 167× selectivity between linking and non-linking tokens. The semantic gate achieves 6.4× selectivity between fact concepts (avg=0.401) and filler concepts (avg=0.063). Together, they reduce the accuracy gap between clean and noisy input from 44pp to 2.6pp.

### 3.4 Reconsolidation Erasure

Memory reconsolidation in neuroscience refers to the process where retrieved memories become labile and can be modified or erased. We implement selective erasure for fact updates:

    Q_norm = Q / ||Q||₂
    V_old = Q_norm · T_v                    (retrieve old association)
    T_v ← T_v - η_erase · Q_norm^T · V_old  (subtract old)
    T_v ← T_v + η · Q^T · V_new             (write new)

L2 normalization of Q before erasure is critical — without it, the erasure magnitude scales with ||Q||², causing catastrophic over-erasure. With normalized Q, erasure precisely targets the stored association at that address.

**Impact on fact updates.** In a 10-session scenario with 5 fact updates, erasure improves update accuracy from 61% to 99% (+38pp). Trace norm decreases during updates (11.0 → 8.6), confirming that erasure actively cleans the trace rather than merely overwriting.

### 3.5 Complementary Learning Systems: Per-Head Specialization

Inspired by CLS theory, we assign different temporal dynamics to different attention heads within the trace. "Slow" heads (high decay, γ=0.99) maintain stable long-term associations, while "fast" heads (low decay, γ=0.90) quickly adapt to updated information:

    T_v^h ← γ_h · T_v^h + η · Q^T · V    (per-head decay γ_h)

This specialization is irrelevant when all facts are equally important (confirmed in ablation), but becomes critical for fact update tasks: split-decay heads improve overall accuracy by +7.4pp compared to uniform decay, specifically by reducing old-value interference from 28.5% to 16.2%.

---

## 4. Experimental Validation

We validate the architecture through 16 experiments across two model scales, following a systematic progression from individual components to the complete system.

### 4.1 Development Platform: Custom Transformer (MiniGPT)

Initial development used a custom 6.4M-parameter transformer (d_model=128, 4-8 heads, 176-token vocabulary) trained on synthetic key-value association tasks. This controlled setting allowed rapid iteration on architectural choices.

**Key findings from ablation studies:**

| Component | Experiment | Effect (n=10) | Mechanism |
|-----------|-----------|---------------|-----------|
| Pattern separation | Exp 5 | +9.3pp cross-context | Sparse Q reduces interference |
| Adaptive alpha | Exp 3 | +16pp Hebbian, +2pp cross | Score trace normalization |
| CLS decay | Exp 6 | +7.4pp overall (updates) | Fast/slow head specialization |
| Selective erasure | Exp 6 | +11.7pp overall (updates) | Anti-Hebbian old-value removal |
| Concept injection | Exp 7 | +90pp Tier 2 templates | Semantic Q override for diverse templates |
| Dual gates | Exp 11 | +46pp noisy paragraphs | Semantic filtering of storage |

All components compose orthogonally: combining pattern separation with adaptive alpha yields the sum of their individual improvements, confirming they address independent bottlenecks.

### 4.2 Transfer to Pretrained GPT-2

We attached the trace module to frozen GPT-2 Small (124M parameters) as an external memory with ~1.1M trainable parameters (W_proj, W_val, W_out, W_gate, W_gate_key).

**Zero-shot proof of concept (Experiment 8).** With random projections and zero fine-tuning, the trace achieves 100% cross-context recall at n=1 (vs. 4% baseline) and 84.1% at n=7 with pattern separation. This demonstrates that GPT-2's embedding space is sufficiently structured for Hebbian association without learned projections.

**Pattern separation transfers identically.** The same +12pp improvement magnitude observed on MiniGPT appears on GPT-2, confirming the mechanism's model-independence.

**Alpha calibration.** Optimal injection strength shifts from α=0.1 (MiniGPT) to α=0.5 (GPT-2), reflecting the larger model's higher confidence in its own predictions requiring stronger trace signal.

### 4.3 Learned Storage Gating (Experiments 9-11)

Replacing the handcrafted linking-token mask with a learned gate (769 parameters) achieves equivalent accuracy while discovering the same token-level structure without supervision. Adding the semantic gate (769 additional parameters) reduces the noisy-paragraph accuracy gap from 44pp to 7.8pp, and to 2.6pp when combined with pattern separation.

### 4.4 Architectural Limitations (Experiment 12)

Systematic diagnostic reveals three ceiling effects of context-free addressing:

| Limitation | Impact | Root cause |
|-----------|--------|-----------|
| Shift-1 addressing | -60pp on paraphrased questions | Retrieval depends on exact token position |
| Context-free Q | -48pp with distractors (70% confusion) | Cannot distinguish "My name" from "Alice's name" |
| Q-collision | -29pp with repeated fact types | Same concept word → same address |

These limitations motivate context-enriched addressing (Section 4.5) and define clear targets for future work.

### 4.5 Context-Enriched Addressing (Experiments 13-14)

Blending context-free and contextual Q via Q = Q_base + β · W_ctx(hidden_states) reveals a Pareto trade-off: increasing β improves distractor discrimination at the cost of cross-session stability. Contrastive training of W_ctx reduces confusion from 74.7% to 28.0% at β=0.5, while cross-context alignment loss further improves the frontier.

### 4.6 Flagship Result: Multi-Session Persistent Memory (Experiment 16)

The complete system demonstrates persistent memory across sessions:

**Setup.** Frozen GPT-2 Small + trace module (α=0.5, 8 heads, d_trace=64, pattern separation 8×/k=16). 24 fact types, 15 sessions, 3-5 facts per session introduced in natural paragraphs with filler sentences.

**Results.**

| Metric | Value |
|--------|-------|
| Total facts stored | 24 |
| Sessions | 15 (3–5 facts introduced per session) |
| Mean recall across sessions | 98.6% |
| Session 15 recall (all 24 facts active) | 98% |
| Single-pass recall at n=24 (no incremental introduction) | 94.4% (CI: 92.3–96.5%) |
| Retention drop over 15 sessions | ≤2pp |
| Update accuracy (with erasure) | 99% |
| Noisy paragraph filtering | 98–99% (all filler modes) |
| Trainable parameters | ~1.1M (0.9% of GPT-2) |

**Incremental vs. single-pass.** The 4.2pp gap between incremental multi-session recall (98%) and single-pass capacity at n=24 (94.4%) reflects two effects: (1) incremental introduction allows the trace to consolidate associations with less mutual interference, and (2) the decay factor progressively attenuates older entries, which is less harmful when entries are refreshed across sessions via cumulative replay.

The dual gate achieves 103× selectivity between fact and filler sentences, enabling paragraph-level input where facts are interleaved with irrelevant content. Reconsolidation erasure maintains 99% accuracy through fact updates, with trace norm decreasing during updates (confirming active cleanup rather than mere overwriting).

### 4.7 Standard Benchmark: LAMA Knowledge Probes

To validate beyond custom synthetic tasks, we evaluate on the LAMA T-REx knowledge probing benchmark (Petroni et al., 2019). LAMA probes factual knowledge through cloze-style templates over Wikidata relations.

**Adaptation.** LAMA facts are reformatted into the system's `{concept} {linking_token} {entity}` templates, with each T-REx relation assigned a unique concept word. For example, the LAMA relation P36 (capital) becomes: storage `"capital is Paris."`, query `"capital is"`. We filter to facts where the object label is a single BPE token, yielding 2,034 valid facts across 32 relations (1,014 unique entity tokens; random chance = 0.1%).

**Results.** Cross-context retrieval accuracy across fact counts (100 episodes, bootstrap 95% CIs):

| n_facts | Cross-context (trace) | In-context (GPT-2) | No memory |
|---------|----------------------|-------------------|-----------|
| 1 | **100.0%** | 95.0% | 1.0% |
| 3 | **94.0%** (91.3–96.3%) | 43.3% | 0.3% |
| 5 | **93.6%** (91.4–95.6%) | 41.6% | 0.2% |
| 7 | **88.7%** (86.7–90.7%) | 29.7% | 0.3% |
| 10 | **81.2%** (79.0–83.3%) | 24.7% | 0.1% |

The trace consistently exceeds the in-context baseline by +50–60pp at n≥3, demonstrating that Hebbian storage outperforms GPT-2's native in-context learning for structured fact retrieval even on real-world knowledge. Per-relation accuracy ranges from 100% (rare concept words: genre, instrument) to 66% (frequent concept words: birthplace, location), consistent with the Q-collision limitation identified in Section 4.4.

**Caveats.** The single-token constraint limits coverage to ~6% of LAMA T-REx. Template reformatting means we test the trace mechanism's storage capacity on real knowledge, not GPT-2's ability to parse natural LAMA templates. These results complement, rather than replace, the controlled ablation on synthetic tasks.

---

## 5. Discussion

### Composability as a design principle

A central finding is that bio-inspired mechanisms compose orthogonally when they target independent bottlenecks. Pattern separation reduces storage interference; gating reduces input noise; erasure handles updates; CLS decay manages temporal dynamics. No component interferes with another, and their benefits are approximately additive. This suggests that the hippocampal memory system's multi-mechanism architecture is not merely an evolutionary accident but a functionally optimal decomposition.

### Limitations

We identify five concrete limitations that define the gap between our proof-of-concept and practical deployment:

1. **Structured input dependency.** Facts must follow `{concept} {linking_token} {entity}` templates with specific linking tokens ("is", "in", "at"). Fully unconstrained text — where facts are expressed through arbitrary syntax ("The red one has always been my favorite") — is not supported by the current shift-based addressing mechanism.

2. **Single-token entities.** Entity values must tokenize to exactly one BPE token. Multi-token entities ("San Francisco", "machine learning") require chain-write mechanisms not yet implemented, significantly limiting the vocabulary of storable facts.

3. **Context-free addressing trade-off.** Using token identity (not context) for addressing enables cross-session stability but fundamentally cannot distinguish entities sharing a concept word — "my name" and "Alice's name" produce identical Q vectors. The Pareto trade-off between discrimination and stability (Section 4.5) remains partially unresolved.

4. **Scale.** Current validation covers 24 fact types (~50 facts) on GPT-2 Small (124M parameters). Scaling behavior to hundreds of facts and larger models (7B+) is unknown and cannot be extrapolated from current results.

5. **Limited standard benchmark coverage.** Our LAMA evaluation (§4.7) covers only ~6% of T-REx relations due to the single-token constraint, and remaining experiments use custom synthetic tasks designed to isolate cross-session persistence. Broader evaluation on multi-token benchmarks (TriviaQA, Natural Questions) and larger relation sets would strengthen generalizability claims.

### Connection to biological memory

Our architecture implements a simplified hippocampal loop: dentate gyrus (pattern separation) → CA3 (Hebbian trace) → cholinergic modulation (dual gates) → reconsolidation (erasure). We do not claim biological fidelity — rather, we use neuroscience as a source of architectural inductive biases that prove empirically effective. The consistent success of these biases across two model scales suggests that the computational principles underlying hippocampal memory are relevant beyond biological neural networks.

---

## 6. Future Work

Several directions extend from the current foundation:

**Episodic memory for agents.** The current system stores declarative facts. Extending trace to store experiential episodes — situation-action-outcome tuples — would enable inference-time learning for LLM-based agents, where performance improves across episodes without weight updates.

**Neocortical consolidation.** In biological memory, hippocampal traces are gradually consolidated into neocortical representations. An analog process — periodically distilling trace contents into model weights through targeted fine-tuning — could combine the fast learning of Hebbian traces with the capacity of parametric memory.

**Scaling to larger models.** Current validation is on GPT-2 Small (124M). Testing on larger models (GPT-2 Medium/Large, open-source LLMs) would establish scaling properties of the trace module.

---

## 7. Conclusion

We have presented the Hebbian Trace Module, a bio-inspired external memory system that demonstrates persistent cross-session memory for frozen language models on structured fact association tasks. Through systematic development across 16 experiments, we showed that five composable mechanisms — pattern separation, Hebbian storage, dual semantic gating, reconsolidation erasure, and complementary learning systems — can be combined to achieve 98–99% mean recall across 15 sessions with only 1M additional parameters. Each mechanism is independently motivated by hippocampal neuroscience, independently validated through ablation, and composes orthogonally with the others. Significant limitations remain — notably structured input requirements, single-token entities, and unvalidated scaling behavior — but our results suggest that the computational principles underlying biological memory systems offer viable architectural inductive biases for adding persistent, updatable memory to pretrained language models without modifying their weights.

---

## References

Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016). Using fast weights to attend to the recent past. NeurIPS.

Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. NeurIPS.

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. arXiv preprint arXiv:1410.5401.

Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. Current Opinion in Neurobiology, 16(6), 710-715.

Hebb, D. O. (1949). The Organization of Behavior: A Neuropsychological Theory. Wiley.

Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. Contemporary Mathematics, 26, 189-206.

Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419-457.

Munkhdalai, T., Faruqui, M., & Gopal, S. (2024). Leave no context behind: Efficient infinite context transformers with Infini-attention. arXiv preprint arXiv:2404.07143.

Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. Nature, 406(6797), 722-726.

Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the transformer era. EMNLP.

Petroni, F., Rocktäschel, T., Lewis, P., Bakhtin, A., Wu, Y., Miller, A. H., & Riedel, S. (2019). Language models as knowledge bases? EMNLP.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI Blog.

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L., ... & Hochreiter, S. (2020). Hopfield networks is all you need. ICLR.

Rolls, E. T. (2013). The mechanisms for pattern completion and pattern separation in the hippocampus. Frontiers in Systems Neuroscience, 7, 74.

Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear transformers are secretly fast weight programmers. ICML.

Sukhbaatar, S., Szlam, A., Weston, J., & Fergus, R. (2015). End-to-end memory networks. NeurIPS.

Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. NeurIPS.

Wang, Q., Ding, L., Cao, Y., et al. (2024). MemoryLLM: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624.

Weston, J., Chopra, S., & Bordes, A. (2015). Memory networks. ICLR.

Wu, Y., Rabe, M. N., Hutchins, D., & Szegedy, C. (2022). Memorizing transformers. ICLR.
