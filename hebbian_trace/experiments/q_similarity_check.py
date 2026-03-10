"""
Phase 0: Q-similarity check in GPT-2 wte space.

Measures cosine similarity between concept words and paraphrase cues
to determine if soft key lookup is viable for pattern completion.

Key question: is Q("I") closer to Q("name") than to Q("city")?
If yes → soft attention over stored keys can work.
If no → need T_auto or entorhinal encoder.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_q_vectors(model, tokenizer, words: list[str], n_heads=8, d_trace=64):
    """
    Compute trace Q vectors for given words.
    Uses same pipeline as GPT2WithTrace: wte → LN → W_proj → reshape to heads.
    But since we don't have trained W_proj here, we measure raw wte similarity
    (which is what trace Q is based on — W_proj is random orthogonal, preserves distances).
    """
    wte = model.transformer.wte
    ln = torch.nn.LayerNorm(model.config.n_embd)
    ln.eval()

    vectors = {}
    for word in words:
        # GPT-2 BPE: leading space matters
        for prefix in [f" {word}", word]:
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            if len(ids) == 1:
                emb = wte(torch.tensor([ids[0]]))  # (1, d_model)
                vectors[word] = emb.squeeze(0).detach()
                break
        else:
            print(f"  WARNING: '{word}' is multi-token, skipping")

    return vectors


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    # === Concept words (stored as trace keys) ===
    concepts = ["name", "city", "company", "color", "food", "pet", "country", "language", "hobby", "age"]

    # === Paraphrase cues (what appears in alternative questions) ===
    paraphrase_cues = {
        "name": ["I", "me", "am", "called", "who"],
        "city": ["live", "where", "reside", "located", "from"],
        "company": ["work", "employed", "job", "employer"],
        "color": ["favorite", "prefer", "like", "love"],
    }

    # === Get embeddings ===
    all_words = set(concepts)
    for cues in paraphrase_cues.values():
        all_words.update(cues)

    vectors = get_q_vectors(model, tokenizer, list(all_words))

    # === 1. Concept-concept similarity matrix ===
    print("\n" + "=" * 70)
    print("1. CONCEPT-CONCEPT SIMILARITY (stored keys)")
    print("=" * 70)
    available_concepts = [c for c in concepts if c in vectors]
    n = len(available_concepts)
    sim_matrix = np.zeros((n, n))
    for i, c1 in enumerate(available_concepts):
        for j, c2 in enumerate(available_concepts):
            sim_matrix[i, j] = cosine_sim(vectors[c1], vectors[c2])

    # Print matrix
    header = "          " + "".join(f"{c[:7]:>8}" for c in available_concepts)
    print(header)
    for i, c in enumerate(available_concepts):
        row = f"{c[:9]:>9} " + "".join(f"{sim_matrix[i, j]:8.3f}" for j in range(n))
        print(row)

    off_diag = sim_matrix[np.triu_indices(n, k=1)]
    print(f"\nOff-diagonal: mean={off_diag.mean():.3f}, std={off_diag.std():.3f}, "
          f"min={off_diag.min():.3f}, max={off_diag.max():.3f}")

    # === 2. Paraphrase cue → concept similarity ===
    print("\n" + "=" * 70)
    print("2. PARAPHRASE CUE → CONCEPT SIMILARITY")
    print("   Question: is Q('I') closer to Q('name') than to Q('city')?")
    print("=" * 70)

    for target_concept, cues in paraphrase_cues.items():
        print(f"\n  Target concept: '{target_concept}'")
        print(f"  {'cue':>10} | {'→ target':>8} | {'→ others (mean)':>15} | {'rank':>4} | verdict")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*15}-+-{'-'*4}-+--------")

        for cue in cues:
            if cue not in vectors:
                continue
            # Similarity to target concept
            sim_target = cosine_sim(vectors[cue], vectors[target_concept])

            # Similarity to all other concepts
            sims_other = []
            for c in available_concepts:
                if c != target_concept:
                    sims_other.append((c, cosine_sim(vectors[cue], vectors[c])))

            mean_other = np.mean([s for _, s in sims_other])

            # Rank of target among all concepts
            all_sims = [(target_concept, sim_target)] + sims_other
            all_sims.sort(key=lambda x: x[1], reverse=True)
            rank = [c for c, _ in all_sims].index(target_concept) + 1

            # Verdict
            if rank == 1 and sim_target > mean_other + 0.05:
                verdict = "GOOD"
            elif rank <= 3:
                verdict = "weak"
            else:
                verdict = "FAIL"

            print(f"  {cue:>10} | {sim_target:8.3f} | {mean_other:15.3f} | {rank:4d} | {verdict}")

    # === 3. Softmax attention simulation ===
    print("\n" + "=" * 70)
    print("3. SOFTMAX ATTENTION SIMULATION")
    print("   If we do attention over stored concept keys, what distribution do we get?")
    print("=" * 70)

    concept_vecs = torch.stack([vectors[c] for c in available_concepts])  # (n_concepts, d_model)

    for target_concept, cues in paraphrase_cues.items():
        print(f"\n  Query context: '{target_concept}'")
        for cue in cues:
            if cue not in vectors:
                continue
            q = vectors[cue].unsqueeze(0)  # (1, d_model)

            # Raw dot product attention
            logits = (q @ concept_vecs.T).squeeze(0)  # (n_concepts,)

            for temp in [1.0, 0.1, 0.01]:
                attn = F.softmax(logits / temp, dim=0)
                target_idx = available_concepts.index(target_concept)
                target_weight = attn[target_idx].item()
                max_idx = attn.argmax().item()
                max_concept = available_concepts[max_idx]
                max_weight = attn[max_idx].item()

                mark = " ✓" if max_concept == target_concept else f" ✗ (max={max_concept})"
                print(f"    Q('{cue}') τ={temp}: target '{target_concept}'={target_weight:.3f}, "
                      f"max={max_weight:.3f}{mark}")

    # === 4. Summary statistics ===
    print("\n" + "=" * 70)
    print("4. SUMMARY")
    print("=" * 70)

    n_good, n_weak, n_fail = 0, 0, 0
    for target_concept, cues in paraphrase_cues.items():
        for cue in cues:
            if cue not in vectors:
                continue
            sim_target = cosine_sim(vectors[cue], vectors[target_concept])
            sims = [(c, cosine_sim(vectors[cue], vectors[c])) for c in available_concepts]
            sims.sort(key=lambda x: x[1], reverse=True)
            rank = [c for c, _ in sims].index(target_concept) + 1
            mean_other = np.mean([s for c, s in sims if c != target_concept])

            if rank == 1 and sim_target > mean_other + 0.05:
                n_good += 1
            elif rank <= 3:
                n_weak += 1
            else:
                n_fail += 1

    total = n_good + n_weak + n_fail
    print(f"  GOOD (rank=1, margin>0.05): {n_good}/{total}")
    print(f"  weak (rank<=3):             {n_weak}/{total}")
    print(f"  FAIL (rank>3):              {n_fail}/{total}")

    if n_good > total * 0.5:
        print("\n  → VERDICT: Soft key lookup has potential. Worth prototyping.")
    elif n_good + n_weak > total * 0.5:
        print("\n  → VERDICT: Marginal structure. Soft lookup might work with temperature tuning.")
    else:
        print("\n  → VERDICT: No structure. Need T_auto or entorhinal encoder.")


if __name__ == "__main__":
    main()
