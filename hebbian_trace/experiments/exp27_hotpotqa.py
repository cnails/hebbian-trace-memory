"""Exp 27: HotpotQA 2-hop evaluation.

Real-world multi-hop QA from HotpotQA bridge questions.
Uses oracle supporting facts + single-token BPE answer filter.

Pipeline per question:
  1. Identify bridge entity from supporting fact annotations
  2. Store: Q(hop1_concept) → V(bridge_first_token)
  3. Store: Q(bridge_first_token) → V(answer_token)
  4. Multi-hop: concept → bridge → answer via retrieve_direct

Three evaluation modes:
  Phase 1: Per-question (reset trace, 2 facts only) — validates pipeline
  Phase 2: Batched (N questions share trace) — tests capacity
  Phase 3: Diagnostic — where does it break?

Usage:
  python -m hebbian_trace.experiments.exp27_hotpotqa --quick
  python -m hebbian_trace.experiments.exp27_hotpotqa --n-questions 100
  python -m hebbian_trace.experiments.exp27_hotpotqa --phase batch --batch-size 5
"""

import argparse
import random
from dataclasses import dataclass, field

import torch
from transformers import GPT2Tokenizer

from ..gpt2_trace import GPT2WithTrace
from .exp24_free_text import setup_model


# ── Dataset loading and filtering ────────────────────────────────────

@dataclass
class HotpotQuestion:
    """One filtered HotpotQA bridge question."""
    question: str
    answer: str
    answer_token_id: int
    bridge_entity: str
    bridge_first_token_id: int
    bridge_token_ids: list[int]      # ALL BPE tokens of bridge entity
    bridge_n_tokens: int
    subject: str
    sf_bridge_text: str     # supporting fact sentences from bridge paragraph
    sf_subject_text: str    # supporting fact sentences from subject paragraph


def load_hotpot_questions(
    tokenizer: GPT2Tokenizer,
    max_questions: int = 0,
) -> list[HotpotQuestion]:
    """Load and filter HotpotQA bridge questions.

    Filters:
    1. Bridge type only (not comparison)
    2. Single-token BPE answer
    3. Bridge entity identifiable from annotations
    4. Non-degenerate (bridge != answer)
    """
    from datasets import load_dataset

    print("Loading HotpotQA validation set...")
    ds = load_dataset('hotpot_qa', 'distractor', split='validation')

    questions = []

    for ex in ds:
        if ex['type'] != 'bridge':
            continue

        answer = ex['answer']
        if answer.lower() in ('yes', 'no'):
            continue

        # Answer tokenization — accept multi-token, use first token
        ans_toks = tokenizer.encode(' ' + answer, add_special_tokens=False)
        if len(ans_toks) == 0:
            continue

        # Identify bridge entity
        bridge_entity, subject = _find_bridge_entity(ex)
        if bridge_entity is None:
            continue

        # Filter degenerate
        if bridge_entity.lower().strip() == answer.lower().strip():
            continue

        # First token of bridge entity
        bridge_toks = tokenizer.encode(
            ' ' + bridge_entity, add_special_tokens=False)

        # Get supporting fact text
        sf_bridge = _get_sf_sentences(ex, bridge_entity)
        sf_subject = _get_sf_sentences(ex, subject)

        questions.append(HotpotQuestion(
            question=ex['question'],
            answer=answer,
            answer_token_id=ans_toks[0],
            bridge_entity=bridge_entity,
            bridge_first_token_id=bridge_toks[0],
            bridge_token_ids=bridge_toks,
            bridge_n_tokens=len(bridge_toks),
            subject=subject,
            sf_bridge_text=sf_bridge,
            sf_subject_text=sf_subject,
        ))

        if max_questions > 0 and len(questions) >= max_questions:
            break

    return questions


def _find_bridge_entity(ex) -> tuple:
    """Identify bridge entity from supporting facts.

    Bridge entity: SF title whose paragraph contains the answer,
    and is mentioned in the other SF paragraph.
    """
    sf_titles = list(set(ex['supporting_facts']['title']))
    if len(sf_titles) != 2:
        return None, None

    t1, t2 = sf_titles[0], sf_titles[1]
    text1 = ' '.join(_get_para_text(ex, t1))
    text2 = ' '.join(_get_para_text(ex, t2))
    answer = ex['answer']

    t1_in_t2 = t1 in text2 or t1.lower() in text2.lower()
    t2_in_t1 = t2 in text1 or t2.lower() in text1.lower()
    a_in_t1 = answer in text1 or answer.lower() in text1.lower()
    a_in_t2 = answer in text2 or answer.lower() in text2.lower()

    # Bridge = title mentioned in other para AND its para has the answer
    if t2_in_t1 and a_in_t2:
        return t2, t1
    elif t1_in_t2 and a_in_t1:
        return t1, t2

    # Fallback: answer in which paragraph?
    if a_in_t2 and not a_in_t1:
        return t2, t1
    elif a_in_t1 and not a_in_t2:
        return t1, t2

    return None, None


def _get_para_text(ex, title) -> list[str]:
    """Get all sentences from a context paragraph by title."""
    for i, t in enumerate(ex['context']['title']):
        if t == title:
            return ex['context']['sentences'][i]
    return []


def _get_sf_sentences(ex, title) -> str:
    """Get supporting fact sentences for a title."""
    ctx = ex['context']
    sf_titles = ex['supporting_facts']['title']
    sf_sent_ids = ex['supporting_facts']['sent_id']

    para_sents = None
    for i, t in enumerate(ctx['title']):
        if t == title:
            para_sents = ctx['sentences'][i]
            break
    if para_sents is None:
        return ''

    result = []
    for sf_t, sf_id in zip(sf_titles, sf_sent_ids):
        if sf_t == title and sf_id < len(para_sents):
            result.append(para_sents[sf_id])
    return ' '.join(result)


# ── Evaluation phases ────────────────────────────────────────────────

def run_phase1_per_question(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
    use_banks: bool = False,
) -> dict:
    """Per-question evaluation: reset trace between questions.

    Tests pure pipeline correctness with minimal trace load (2 facts).
    Uses a fixed concept token for hop-1 ("link").
    """
    bank_label = f", banks={model.trace.n_trace_banks}" if use_banks else ""
    print(f"\n{'='*60}")
    print(f"Phase 1: Per-question evaluation "
          f"({len(questions)} questions{bank_label})")
    print(f"{'='*60}")

    # Fixed hop-1 concept: use " link" token
    hop1_concept_id = tokenizer.encode(" link", add_special_tokens=False)[0]

    # All unique answer token IDs as candidates
    all_answer_ids = sorted(set(q.answer_token_id for q in questions))
    # All unique bridge first token IDs
    all_bridge_ids = sorted(set(q.bridge_first_token_id for q in questions))
    # Combined candidate pool
    all_candidates = sorted(set(all_answer_ids + all_bridge_ids))

    hop1_correct = 0
    hop2_oracle_correct = 0
    hop2_pred_correct = 0
    e2e_correct = 0
    total = 0

    failures = []

    for q in questions:
        model.trace.reset_traces()
        model.set_trace_mode(use=False, update=False)

        # Store hop-1: concept → bridge_first_token
        model.write_fact_direct(hop1_concept_id, q.bridge_first_token_id)

        # Store hop-2: bridge_first_token → answer
        # Banks route by full bridge tokens to avoid first-token collision
        if use_banks:
            model.write_fact_direct_banked(
                q.bridge_first_token_id, q.answer_token_id,
                q.bridge_token_ids)
        else:
            model.write_fact_direct(
                q.bridge_first_token_id, q.answer_token_id)

        # Query
        model.set_trace_mode(use=True, update=False)

        # Hop-1: concept → bridge
        hop1_pred = model.retrieve_direct(hop1_concept_id, all_bridge_ids)
        hop1_ok = (hop1_pred == q.bridge_first_token_id)

        # Hop-2|oracle: correct bridge → answer (best-bank scan)
        if use_banks:
            hop2_oracle = model.retrieve_direct_best_bank(
                q.bridge_first_token_id, all_answer_ids)
        else:
            hop2_oracle = model.retrieve_direct(
                q.bridge_first_token_id, all_answer_ids)
        hop2_oracle_ok = (hop2_oracle == q.answer_token_id)

        # Hop-2|predicted: predicted bridge → answer
        if use_banks:
            hop2_pred = model.retrieve_direct_best_bank(
                hop1_pred, all_answer_ids)
        else:
            hop2_pred = model.retrieve_direct(hop1_pred, all_answer_ids)
        hop2_pred_ok = (hop2_pred == q.answer_token_id)

        # End-to-end
        e2e_ok = hop1_ok and hop2_pred_ok

        if hop1_ok:
            hop1_correct += 1
        if hop2_oracle_ok:
            hop2_oracle_correct += 1
        if hop2_pred_ok:
            hop2_pred_correct += 1
        if e2e_ok:
            e2e_correct += 1
        total += 1

        if not e2e_ok:
            pred_bridge = tokenizer.decode([hop1_pred]).strip()
            pred_answer = tokenizer.decode([hop2_pred]).strip() if hop2_pred else '?'
            failures.append({
                'question': q.question[:60],
                'answer': q.answer,
                'bridge': q.bridge_entity[:20],
                'hop1_ok': hop1_ok,
                'hop2_oracle_ok': hop2_oracle_ok,
                'pred_bridge': pred_bridge,
                'pred_answer': pred_answer,
            })

        model.set_trace_mode(use=False, update=False)

    h1 = hop1_correct / total * 100
    h2o = hop2_oracle_correct / total * 100
    h2p = hop2_pred_correct / total * 100
    e2e = e2e_correct / total * 100

    print(f"\n  Results ({total} questions):")
    print(f"    hop-1 (concept→bridge):  {h1:5.1f}% ({hop1_correct}/{total})")
    print(f"    hop-2|oracle:            {h2o:5.1f}% ({hop2_oracle_correct}/{total})")
    print(f"    hop-2|predicted:         {h2p:5.1f}% ({hop2_pred_correct}/{total})")
    print(f"    end-to-end:              {e2e:5.1f}% ({e2e_correct}/{total})")
    print(f"    Candidate pool: {len(all_answer_ids)} answers, "
          f"{len(all_bridge_ids)} bridges")

    # Show first failures
    if failures:
        print(f"\n  First 5 failures:")
        for f in failures[:5]:
            print(f"    Q: {f['question']}")
            print(f"    Expected: {f['bridge']}→{f['answer']}  "
                  f"Got: {f['pred_bridge']}→{f['pred_answer']}  "
                  f"h1={'OK' if f['hop1_ok'] else 'FAIL'}  "
                  f"h2o={'OK' if f['hop2_oracle_ok'] else 'FAIL'}")

    return {
        "hop1": h1, "hop2_oracle": h2o,
        "hop2_pred": h2p, "end_to_end": e2e,
        "total": total,
        "n_answer_candidates": len(all_answer_ids),
        "n_bridge_candidates": len(all_bridge_ids),
        "n_failures": len(failures),
    }


def run_phase2_batched(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
    batch_size: int = 5,
    n_batches: int = 50,
    use_banks: bool = False,
) -> dict:
    """Batched evaluation: N questions share the same trace.

    Tests capacity with real entities. Each batch stores 2*N facts.
    """
    bank_label = f", banks={model.trace.n_trace_banks}" if use_banks else ""
    print(f"\n{'='*60}")
    print(f"Phase 2: Batched evaluation (batch_size={batch_size}, "
          f"n_batches={n_batches}{bank_label})")
    print(f"{'='*60}")

    all_answer_ids = sorted(set(q.answer_token_id for q in questions))
    all_bridge_ids = sorted(set(q.bridge_first_token_id for q in questions))

    # We need unique hop-1 concepts per question in batch
    # Use different concept words for each slot in batch
    concept_words = ["link", "chain", "bridge", "connect", "path",
                     "route", "hop", "step", "jump", "trace",
                     "find", "seek", "query", "fetch", "get"]
    concept_ids = []
    for w in concept_words:
        toks = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(toks) == 1:
            concept_ids.append(toks[0])
    concept_ids = concept_ids[:batch_size]

    if len(concept_ids) < batch_size:
        print(f"  WARNING: only {len(concept_ids)} concept tokens, "
              f"need {batch_size}")
        batch_size = len(concept_ids)

    hop1_correct = 0
    hop2_oracle_correct = 0
    e2e_correct = 0
    total = 0

    for batch_idx in range(n_batches):
        model.trace.reset_traces()
        model.set_trace_mode(use=False, update=False)

        # Pick batch_size random questions
        batch = random.sample(questions, min(batch_size, len(questions)))

        # Store all facts
        for i, q in enumerate(batch):
            cid = concept_ids[i]
            # Hop-1: unique concept → bridge
            model.write_fact_direct(cid, q.bridge_first_token_id)
            # Hop-2: bridge → answer
            # Banks route by full bridge tokens to avoid first-token collision
            if use_banks:
                model.write_fact_direct_banked(
                    q.bridge_first_token_id, q.answer_token_id,
                    q.bridge_token_ids)
            else:
                model.write_fact_direct(
                    q.bridge_first_token_id, q.answer_token_id)

        # Query each
        model.set_trace_mode(use=True, update=False)
        for i, q in enumerate(batch):
            cid = concept_ids[i]

            # Hop-1
            hop1_pred = model.retrieve_direct(cid, all_bridge_ids)
            hop1_ok = (hop1_pred == q.bridge_first_token_id)

            # Hop-2|oracle
            if use_banks:
                hop2_oracle = model.retrieve_direct_best_bank(
                    q.bridge_first_token_id, all_answer_ids)
            else:
                hop2_oracle = model.retrieve_direct(
                    q.bridge_first_token_id, all_answer_ids)
            hop2_oracle_ok = (hop2_oracle == q.answer_token_id)

            # Hop-2|predicted
            if use_banks:
                hop2_pred = model.retrieve_direct_best_bank(
                    hop1_pred, all_answer_ids)
            else:
                hop2_pred = model.retrieve_direct(hop1_pred, all_answer_ids)
            e2e_ok = hop1_ok and (hop2_pred == q.answer_token_id)

            if hop1_ok:
                hop1_correct += 1
            if hop2_oracle_ok:
                hop2_oracle_correct += 1
            if e2e_ok:
                e2e_correct += 1
            total += 1

        model.set_trace_mode(use=False, update=False)

    h1 = hop1_correct / total * 100
    h2o = hop2_oracle_correct / total * 100
    e2e = e2e_correct / total * 100

    print(f"\n  Results ({total} questions in {n_batches} batches):")
    print(f"    hop-1:       {h1:5.1f}%")
    print(f"    hop-2|orc:   {h2o:5.1f}%")
    print(f"    end-to-end:  {e2e:5.1f}%")

    return {"hop1": h1, "hop2_oracle": h2o, "end_to_end": e2e,
            "batch_size": batch_size, "total": total}


def run_phase3_diagnostic(
    model: GPT2WithTrace,
    tokenizer: GPT2Tokenizer,
    questions: list[HotpotQuestion],
) -> dict:
    """Diagnostic: analyze WHY failures happen.

    - Bridge entity first-token collision rate
    - Answer token overlap between questions
    - Q cosine similarity for bridge tokens
    """
    print(f"\n{'='*60}")
    print(f"Phase 3: Diagnostic analysis")
    print(f"{'='*60}")

    # Bridge first-token collision
    bridge_first_tokens = {}
    for q in questions:
        tid = q.bridge_first_token_id
        if tid not in bridge_first_tokens:
            bridge_first_tokens[tid] = []
        bridge_first_tokens[tid].append(q.bridge_entity)

    collisions = {tid: ents for tid, ents in bridge_first_tokens.items()
                  if len(ents) > 1}
    n_collision_questions = sum(len(e) for e in collisions.values())

    print(f"\n  Bridge first-token collisions:")
    print(f"    Unique first tokens: {len(bridge_first_tokens)}")
    print(f"    Tokens with collision: {len(collisions)}")
    print(f"    Questions affected: {n_collision_questions}/{len(questions)}")

    # Show worst collisions
    if collisions:
        worst = sorted(collisions.items(),
                       key=lambda x: len(x[1]), reverse=True)[:5]
        for tid, ents in worst:
            tok_str = tokenizer.decode([tid]).strip()
            unique_ents = list(set(ents))[:5]
            print(f"    Token '{tok_str}' ({tid}): "
                  f"{len(ents)} questions, entities: {unique_ents}")

    # Answer token distribution
    answer_counts = {}
    for q in questions:
        a = q.answer
        answer_counts[a] = answer_counts.get(a, 0) + 1

    print(f"\n  Answer distribution:")
    print(f"    Unique answers: {len(answer_counts)}")
    top_answers = sorted(answer_counts.items(),
                         key=lambda x: x[1], reverse=True)[:5]
    for a, c in top_answers:
        print(f"    '{a}': {c} questions")

    # Bridge token n_tokens distribution
    tok_dist = {}
    for q in questions:
        n = q.bridge_n_tokens
        tok_dist[n] = tok_dist.get(n, 0) + 1
    print(f"\n  Bridge entity BPE token count:")
    for n in sorted(tok_dist):
        print(f"    {n}-token: {tok_dist[n]} ({tok_dist[n]/len(questions)*100:.1f}%)")

    # Q cosine similarity for a sample of bridge first tokens
    sample_tids = list(bridge_first_tokens.keys())[:20]
    if len(sample_tids) >= 2:
        Qs = []
        for tid in sample_tids:
            Q = model.trace.compute_q_for_token(model._wte, tid)
            if model.trace._pattern_sep_enabled:
                Q_exp = model.trace._sparse_expand(
                    Q.unsqueeze(0).unsqueeze(2))
                Q_flat = Q_exp.squeeze(0).squeeze(1).view(-1).float()
            else:
                Q_flat = Q.view(-1).float()
            Qs.append(Q_flat)

        cos_sims = []
        for i in range(len(Qs)):
            for j in range(i + 1, len(Qs)):
                cos = torch.nn.functional.cosine_similarity(
                    Qs[i].unsqueeze(0), Qs[j].unsqueeze(0)).item()
                cos_sims.append(cos)

        print(f"\n  Q cosine similarity (bridge first tokens, sample={len(sample_tids)}):")
        print(f"    Mean: {sum(cos_sims)/len(cos_sims):.3f}")
        print(f"    Max:  {max(cos_sims):.3f}")
        print(f"    Min:  {min(cos_sims):.3f}")

    return {
        "n_unique_bridge_tokens": len(bridge_first_tokens),
        "n_collisions": len(collisions),
        "n_collision_questions": n_collision_questions,
        "n_unique_answers": len(answer_counts),
    }


# ── Main ─────────────────────────────────────────────────────────────

def run_experiment(
    n_questions: int = 100,
    phase: str = "all",
    batch_size: int = 5,
    n_banks: int = 0,
    quick: bool = False,
):
    """Run HotpotQA multi-hop experiment."""
    if quick:
        n_questions = min(n_questions, 50)

    model, tokenizer = setup_model(alpha=0.5, use_ps=True)

    use_banks = n_banks > 0
    if use_banks:
        model.set_bank_mode(n_banks)
        print(f"Bank mode enabled: {n_banks} banks")

    questions = load_hotpot_questions(
        tokenizer, max_questions=n_questions)

    print(f"\nHotpotQA 2-hop evaluation")
    print(f"  Questions loaded: {len(questions)}")
    print(f"  Bridge 1-tok: "
          f"{sum(1 for q in questions if q.bridge_n_tokens == 1)}")
    print(f"  Bridge 2-tok: "
          f"{sum(1 for q in questions if q.bridge_n_tokens == 2)}")
    print(f"  Bridge 3+tok: "
          f"{sum(1 for q in questions if q.bridge_n_tokens >= 3)}")

    results = {}

    if phase in ("all", "per_question"):
        results["phase1"] = run_phase1_per_question(
            model, tokenizer, questions, use_banks=use_banks)

    if phase in ("all", "batch"):
        results["phase2"] = run_phase2_batched(
            model, tokenizer, questions,
            batch_size=batch_size, n_batches=50,
            use_banks=use_banks)

    if phase in ("all", "diagnostic"):
        results["phase3"] = run_phase3_diagnostic(
            model, tokenizer, questions)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    if "phase1" in results:
        p1 = results["phase1"]
        print(f"  Per-question: e2e={p1['end_to_end']:.1f}%  "
              f"hop1={p1['hop1']:.1f}%  hop2|orc={p1['hop2_oracle']:.1f}%  "
              f"({p1['total']} questions)")

    if "phase2" in results:
        p2 = results["phase2"]
        print(f"  Batched (bs={p2['batch_size']}): e2e={p2['end_to_end']:.1f}%  "
              f"hop1={p2['hop1']:.1f}%  hop2|orc={p2['hop2_oracle']:.1f}%")

    if "phase3" in results:
        p3 = results["phase3"]
        print(f"  Diagnostic: {p3['n_unique_bridge_tokens']} unique bridge tokens, "
              f"{p3['n_collisions']} collisions, "
              f"{p3['n_unique_answers']} unique answers")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exp 27: HotpotQA 2-hop evaluation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--phase",
                        choices=["per_question", "batch", "diagnostic", "all"],
                        default="all")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--banks", type=int, default=0,
                        help="Enable hashed trace banks (0=disabled)")
    args = parser.parse_args()

    run_experiment(
        n_questions=args.n_questions,
        phase=args.phase,
        batch_size=args.batch_size,
        n_banks=args.banks,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
