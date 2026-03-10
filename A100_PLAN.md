# 7B Model Scaling — A100 Execution Plan

## Goal

Validate Hebbian Trace on 7-8B models (LLaMA-2-7B, Mistral-7B, LLaMA-3-8B).
Zero-shot transfer: frozen LLM + external trace (~1M params), no gate training.
Demonstrates architecture-agnostic scaling from 124M to 8B parameters.

## What's Already Done (locally)

- [x] `model.py` generalized: `AutoModelForCausalLM` + `get_input_embeddings()`
- [x] Tokenizer references: `GPT2Tokenizer` -> `AutoTokenizer` / `PreTrainedTokenizerBase`
- [x] `exp_scaling.py` — complete experiment script (3 phases)
- [x] GPT-2 sanity checks passed (sweep, eval, counterfactual, evaluate.py, exp_lama.py)
- [x] Alpha sweep optimization: loads model once, changes alpha in-place

### Modified files (not yet committed)

```
evaluate.py              # AutoTokenizer, model._wte, base_model refs
exp_lama.py              # AutoTokenizer, PreTrainedTokenizerBase
exp_multihop.py          # tokenizer generalization
exp_paraphrase.py        # tokenizer generalization
hebbian_trace/model.py   # AutoModelForCausalLM, torch_dtype, hidden_size
hebbian_trace/rag_baselines.py  # tokenizer generalization
hebbian_trace/tasks.py   # PreTrainedTokenizerBase type hints
```

### New file

```
exp_scaling.py           # 750 lines, complete scaling experiment
```

---

## Hardware

- **GPU**: A100 PCIe 80GB ($1.19/hr on HOSTKEY)
- **VRAM requirement**: 7B fp16 = ~14GB, + trace + activations ~ 20-25GB -> fits easily
- **Estimated time**: ~2 hours
- **Estimated cost**: ~$2.50

## Models

| Model | Params | dtype | Known alpha | Status |
|-------|--------|-------|-------------|--------|
| `gpt2` | 124M | fp32 | 0.5 | baseline (done) |
| `gpt2-medium` | 355M | fp32 | 0.5 | baseline (done) |
| `microsoft/phi-2` | 2.7B | fp16 | 50.0 | baseline (done) |
| `meta-llama/Llama-2-7b-hf` | 7B | fp16 | TBD | **NEW** |
| `mistralai/Mistral-7B-v0.1` | 7B | fp16 | TBD | **NEW** |
| `meta-llama/Meta-Llama-3-8B` | 8B | fp16 | TBD | **NEW** |

All 7B+ models require accepted HuggingFace license + HF_TOKEN.

---

## Pre-Flight (before renting GPU)

### 1. Upload code to GPU server

```bash
# Option A: git clone + apply changes
git clone https://github.com/<repo>/hebbian-trace-memory.git
cd hebbian-trace-memory
# copy modified files + exp_scaling.py

# Option B: rsync from local
rsync -avz /Users/cnails/hebbian-trace-memory/ user@gpu-server:~/hebbian-trace-memory/
```

### 2. Environment setup on GPU (~5 min)

```bash
pip install torch transformers accelerate numpy

# HF token (from /Users/cnails/soma/.env)
export HF_TOKEN="<token>"
# or
huggingface-cli login
```

### 3. Verify setup

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python exp_scaling.py --model gpt2 --n-eval 5  # ~2 min, proves code works
```

---

## Execution (on A100)

### Phase 1: Alpha Sweep (~30 min total, ~10 min per model)

Find optimal trace injection strength for each new model.
Tests 10 alpha values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
20 episodes at n=1, pattern separation 8x_k16, zero-shot gates.

```bash
python exp_scaling.py --model meta-llama/Llama-2-7b-hf --phase sweep
# -> results/sweep_meta-llama_Llama-2-7b-hf.json

python exp_scaling.py --model mistralai/Mistral-7B-v0.1 --phase sweep
# -> results/sweep_mistralai_Mistral-7B-v0.1.json

python exp_scaling.py --model meta-llama/Meta-Llama-3-8B --phase sweep
# -> results/sweep_meta-llama_Meta-Llama-3-8B.json
```

**Output**: best alpha for each model. Record them:

```
LLaMA-2-7B:  alpha = ???
Mistral-7B:  alpha = ???
LLaMA-3-8B:  alpha = ???
```

### Phase 2: Cross-Context Eval (~60 min total, ~20 min per model)

The real test. n=1,3,5,7 facts, 100 episodes each, best alpha.
Includes: cross-context (trace), in-context baseline, no-trace baseline.

```bash
python exp_scaling.py --model meta-llama/Llama-2-7b-hf --alpha {BEST} --n-eval 100
python exp_scaling.py --model mistralai/Mistral-7B-v0.1 --alpha {BEST} --n-eval 100
python exp_scaling.py --model meta-llama/Meta-Llama-3-8B --alpha {BEST} --n-eval 100
```

### Phase 3: Counterfactual (~15 min total, ~5 min per model)

Test trace override of pretrained knowledge. 20 wrong facts (swapped capitals, wrong colors, etc.).
Measures whether trace can suppress strong LM priors at 7B scale.

```bash
python exp_scaling.py --model meta-llama/Llama-2-7b-hf --phase counterfactual --alpha {BEST}
python exp_scaling.py --model mistralai/Mistral-7B-v0.1 --phase counterfactual --alpha {BEST}
python exp_scaling.py --model meta-llama/Meta-Llama-3-8B --phase counterfactual --alpha {BEST}
```

### Phase 4 (optional): Re-run existing models on A100

If time permits, run Phi-2 and GPT-2 Medium to confirm consistency.

```bash
python exp_scaling.py --model microsoft/phi-2 --phase eval --n-eval 100
python exp_scaling.py --model gpt2-medium --phase eval --n-eval 100
```

---

## Expected Results Format

### Alpha Sweep (`results/sweep_*.json`)

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "phase": "sweep",
  "best_alpha": 2.0,
  "best_acc": 1.0,
  "results": { "0.1": {"mean": 0.6}, "0.5": {"mean": 0.95}, ... }
}
```

### Cross-Context Eval (`results/scaling_*.json`)

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "params": "7B",
  "alpha": 2.0,
  "results": {
    "1": { "cross_context": {"mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0},
           "baseline": {"mean": 1.0}, "no_trace": {"mean": 0.15} },
    "3": { ... },
    "5": { ... },
    "7": { ... }
  }
}
```

### Counterfactual (`results/counterfactual_*.json`)

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "results": {
    "1": {"prior_acc": 0.0, "trace_acc": 0.95, "override_delta": 0.95},
    "3": { ... },
    "5": { ... }
  }
}
```

---

## After A100 Session

### 1. Download results

```bash
scp user@gpu-server:~/hebbian-trace-memory/results/*.json /Users/cnails/hebbian-trace-memory/results/
```

### 2. Update paper (paper.tex)

**Abstract** — add sentence:
> "We demonstrate transfer to 7-8B parameter models (LLaMA-2, Mistral, LLaMA-3) with zero-shot trace injection, achieving XX% cross-context accuracy at n=7."

**Contributions** — add bullet:
> "Architecture-agnostic scaling from 124M to 8B parameters (6 models, 3 architecture families)"

**New table (Section 4.X)** — Model Scaling:

```latex
\begin{table}[t]
\centering
\caption{Cross-context accuracy across model scales.
Zero-shot trace (no gate training), PS 8$\times$/k=16, 7 fact types, 100 episodes.}
\begin{tabular}{@{}lrrcccc@{}}
\toprule
Model & Params & $\alpha$ & $n{=}1$ & $n{=}3$ & $n{=}5$ & $n{=}7$ \\
\midrule
GPT-2 Small & 124M & 0.5 & X\% & X\% & X\% & X\% \\
GPT-2 Medium & 355M & 0.5 & X\% & X\% & X\% & X\% \\
Phi-2 & 2.7B & 50.0 & X\% & X\% & X\% & X\% \\
LLaMA-2 & 7B & ? & X\% & X\% & X\% & X\% \\
Mistral & 7B & ? & X\% & X\% & X\% & X\% \\
LLaMA-3 & 8B & ? & X\% & X\% & X\% & X\% \\
\bottomrule
\end{tabular}
\end{table}
```

**Counterfactual table** — if results are strong:

```latex
\begin{table}[t]
\centering
\caption{Counterfactual fact override. Trace accuracy on facts contradicting
pretrained knowledge (e.g., ``capital of France is Berlin'').}
\begin{tabular}{@{}lrccc@{}}
\toprule
Model & Params & Prior & Trace & $\Delta$ \\
\midrule
GPT-2 Small & 124M & 0\% & X\% & +X\% \\
...
\bottomrule
\end{tabular}
\end{table}
```

### 3. Update figure

```bash
cd /Users/cnails/hebbian-trace-memory
python figures/generate.py  # regenerate model_scaling figure
```

X-axis: log scale params (124M -> 8B), Y-axis: accuracy at n=5 or n=7.

### 4. Commit

```bash
git add exp_scaling.py hebbian_trace/model.py hebbian_trace/tasks.py \
        evaluate.py exp_lama.py exp_multihop.py exp_paraphrase.py \
        hebbian_trace/rag_baselines.py results/
git commit -m "Add 7B model scaling: LLaMA-2, Mistral, LLaMA-3 transfer"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model download slow on GPU server | Pre-download to HF cache before timing starts |
| Linking tokens multi-token on SentencePiece | `validate_tokenizer()` auto-checks; all 6 linking words verified single-token on GPT-2, need runtime check on LLaMA/Mistral |
| Entity pool too small for some tokenizers | Script reports coverage; 7 fact types with 8-20 entities each |
| Alpha sweep finds no good alpha | Extended range [0.1-200.0]; 10 values should bracket optimum |
| OOM on 7B | fp16 = 14GB, A100 has 80GB; not a concern |
| Counterfactual prior too strong at 7B | This IS the interesting result — trace must work harder to override stronger priors |

## Experiment Details

- **7 base fact types**: name, city, company, color, food, pet, country
- **Linking tokens**: is, in, at, from, :, am
- **Pattern separation**: 8x expansion, top-k=16, seed=0
- **Trace config**: n_heads=8, d_trace=64, trace_lr=1.0, trace_decay=0.99
- **Zero-shot**: no learned gates, hardcoded linking-token mask
- **20 counterfactual facts**: 10 swapped capitals, 4 wrong colors, 4 swapped languages, 2 wrong continents
