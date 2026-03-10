"""Experiment 2: Natural Language Fact Memorization.

Tests whether Hebbian traces can memorize facts expressed in natural language
(e.g., "My name is Andrey") across sessions and answer questions
("What is my name?") using only trace-stored associations.

Protocol:
    1. Pretrain MiniGPT on fact-question-answer sequences (backprop, trace OFF)
    2. Evaluate trace-based memory across sessions:
       - Session 1: process fact statements → traces accumulate
       - Session 2: answer questions from trace alone (no facts in context)

Template tiers:
    - Tier 1: "concept LINK entity" (shift=2) — mechanism works directly
    - Tier 2: alternative phrasings — hard mode, expect ~random

Supports MPS (Apple Silicon), CUDA, and CPU.
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model import MiniGPT
from ..nlp_tasks import (
    NLP_VOCAB, NLPFactDataset, make_nlp_eval_episodes,
)
from ..nlp_evaluate import (
    NLPEvalResults,
    evaluate_baseline,
    evaluate_hebbian,
    evaluate_cross_context,
    evaluate_cross_context_baseline,
)


def get_device(requested: str | None = None) -> torch.device:
    """Detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ExperimentResult:
    """Results for one n_facts condition."""
    n_facts: int
    baseline: NLPEvalResults
    hebbian: NLPEvalResults
    cross_context: NLPEvalResults
    cross_baseline: NLPEvalResults


def pretrain_nlp(
    n_sequences: int = 10000,
    max_facts: int = 5,
    batch_size: int = 64,
    epochs: int = 60,
    lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 8,
    max_seq_len: int = 128,
    dropout: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
    **attention_kwargs,
) -> tuple[MiniGPT, dict]:
    """Pretrain MiniGPT on NLP fact-question-answer sequences.

    Returns (model, stats).
    """
    if device is None:
        device = get_device()

    torch.manual_seed(seed)

    vocab = NLP_VOCAB
    vocab_size = len(vocab)

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        **attention_kwargs,
    ).to(device)

    # Trace OFF during pretraining
    model.set_trace_mode(use=False, update=False)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"MiniGPT: {n_params:,} parameters")
        print(f"  vocab_size={vocab_size}, d_model={d_model}, "
              f"n_heads={n_heads}, n_layers={n_layers}")
        print(f"  device={device}")
        print(f"  Training: {n_sequences} sequences, max {max_facts} facts, "
              f"{epochs} epochs")

    dataset = NLPFactDataset(
        n_sequences=n_sequences,
        max_facts=max_facts,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader))

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    stats = {
        'epoch_loss': [],
        'epoch_acc': [],
        'epoch_time': [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_positions = 0
        t0 = time.time()

        for input_ids, target_ids, loss_mask in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(input_ids)  # (B, S, V)
            B, S, V = logits.shape

            loss_flat = loss_fn(
                logits.reshape(B * S, V),
                target_ids.reshape(B * S),
            )

            mask_flat = loss_mask.reshape(B * S)
            masked_loss = (loss_flat * mask_flat).sum()
            n_positions = mask_flat.sum()

            if n_positions > 0:
                loss = masked_loss / n_positions
            else:
                loss = masked_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += masked_loss.item()
            preds = logits.argmax(dim=-1)
            correct = ((preds == target_ids) * loss_mask).sum().item()
            total_correct += correct
            total_positions += n_positions.item()

        epoch_loss = total_loss / max(total_positions, 1)
        epoch_acc = total_correct / max(total_positions, 1)
        epoch_time = time.time() - t0

        stats['epoch_loss'].append(epoch_loss)
        stats['epoch_acc'].append(epoch_acc)
        stats['epoch_time'].append(epoch_time)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={epoch_loss:.4f}  acc={epoch_acc:.1%}  "
                  f"({epoch_time:.1f}s)")

    if verbose:
        print(f"  Final: loss={stats['epoch_loss'][-1]:.4f}  "
              f"acc={stats['epoch_acc'][-1]:.1%}")

    return model, stats


def save_nlp_model(model: MiniGPT, path: str):
    """Save model weights + config for later loading."""
    torch.save({
        'state_dict': model.state_dict(),
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': len(model.blocks),
        'max_seq_len': model.max_seq_len,
        'use_raw_embed': model.use_raw_embed,
        'use_first_layer_q': model.use_first_layer_q,
    }, path)


def load_nlp_model(path: str, device: str | None = None) -> MiniGPT:
    """Load a saved NLP model.

    Handles vocab size mismatch: if checkpoint was saved with a smaller
    vocab (e.g., 176 tokens) and current vocab is larger (e.g., 177),
    loads with the old size and expands embeddings for new tokens.
    """
    dev = get_device(device)
    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    vocab = NLP_VOCAB

    # Check saved vocab size vs current
    saved_embed = checkpoint['state_dict']['token_embed.weight']
    saved_vocab_size = saved_embed.shape[0]
    current_vocab_size = len(vocab)

    if saved_vocab_size != current_vocab_size:
        # Load with saved vocab size, then expand
        model = MiniGPT(
            vocab_size=saved_vocab_size,
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_layers=checkpoint['n_layers'],
            max_seq_len=checkpoint['max_seq_len'],
            use_raw_embed=checkpoint.get('use_raw_embed', True),
            use_first_layer_q=checkpoint.get('use_first_layer_q', False),
            use_key_q=True,
        ).to(dev)
        model.load_state_dict(checkpoint['state_dict'])

        # Expand embeddings for new tokens
        if current_vocab_size > saved_vocab_size:
            n_new = current_vocab_size - saved_vocab_size
            d_model = checkpoint['d_model']
            old_embed = model.token_embed.weight.data
            new_embed = nn.Embedding(current_vocab_size, d_model).to(dev)
            new_embed.weight.data[:saved_vocab_size] = old_embed
            # New tokens get random init (same as _init_weights)
            nn.init.xavier_uniform_(
                new_embed.weight.data[saved_vocab_size:].unsqueeze(0))
            model.token_embed = new_embed
            # Re-tie head weights
            model.head = nn.Linear(d_model, current_vocab_size, bias=False).to(dev)
            model.head.weight = model.token_embed.weight
    else:
        model = MiniGPT(
            vocab_size=current_vocab_size,
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_layers=checkpoint['n_layers'],
            max_seq_len=checkpoint['max_seq_len'],
            use_raw_embed=checkpoint.get('use_raw_embed', True),
            use_first_layer_q=checkpoint.get('use_first_layer_q', False),
            use_key_q=True,
        ).to(dev)
        model.load_state_dict(checkpoint['state_dict'])

    model.set_linking_token_ids(vocab.linking_tokens)
    return model


def run(
    n_pretrain: int = 20000,
    max_facts: int = 5,
    pretrain_epochs: int = 60,
    n_eval_episodes: int = 200,
    fact_counts: list[int] | None = None,
    alpha: float = 0.1,
    trace_lr: float = 0.1,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 8,
    seed: int = 42,
    device_name: str | None = None,
    save_path_arg: str | None = None,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """Run Experiment 2: NLP Fact Memorization.

    Returns list of ExperimentResult, one per n_facts.
    """
    if fact_counts is None:
        fact_counts = [1, 2, 3, 5, 10]

    device = get_device(device_name)
    t_start = time.time()

    # ── Step 1: Pretrain ──
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 2: Natural Language Fact Memorization")
        print("=" * 60)
        print(f"\nStep 1: Pretraining ({n_pretrain} sequences, "
              f"{pretrain_epochs} epochs)")

    model, train_stats = pretrain_nlp(
        n_sequences=n_pretrain,
        max_facts=max_facts,
        batch_size=64,
        epochs=pretrain_epochs,
        lr=1e-3,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=128,
        dropout=0.1,
        alpha=alpha,
        trace_lr=trace_lr,
        trace_decay=0.99,
        use_raw_embed=True,
        use_key_q=True,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print(f"\nPretrain done. Final accuracy: "
              f"{train_stats['epoch_acc'][-1]:.1%}")

    # Save model if requested
    save_path = save_path_arg
    if save_path:
        save_nlp_model(model, save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    # Set linking tokens for trace filtering
    vocab = NLP_VOCAB
    model.set_linking_token_ids(vocab.linking_tokens)

    # ── Step 2: Evaluate ──
    results = []

    for n_facts in fact_counts:
        if verbose:
            print(f"\n{'─' * 55}")
            print(f"n_facts={n_facts}")
            print(f"{'─' * 55}")

        # Set trace parameters
        for attn in model.get_attention_layers():
            attn.trace_decay = 0.99
            attn.trace_lr = trace_lr
            attn.alpha = alpha

        # Generate episodes (Tier 1 templates for storage)
        episodes = make_nlp_eval_episodes(
            n_episodes=n_eval_episodes,
            n_facts=n_facts,
            seed=seed + n_facts * 1000,
            tier=1,  # Use Tier 1 templates for storage
        )

        # Evaluate all conditions
        if verbose:
            print("  Baseline (in-context, no trace)...")
        bl = evaluate_baseline(model, episodes)
        if verbose:
            print(f"    → {bl.accuracy:.1%}")

        if verbose:
            print("  Hebbian (in-context + trace)...")
        hb = evaluate_hebbian(model, episodes)
        if verbose:
            print(f"    → {hb.accuracy:.1%}")

        if verbose:
            print("  Cross-context (trace only, question only)...")
        cc = evaluate_cross_context(model, episodes)
        if verbose:
            print(f"    → {cc.accuracy:.1%}")
            if cc.tier1_accuracy is not None:
                print(f"      Tier 1: {cc.tier1_accuracy:.1%}")
            if cc.tier2_accuracy is not None:
                print(f"      Tier 2: {cc.tier2_accuracy:.1%}")

        if verbose:
            print("  Cross-context baseline (no trace, question only)...")
        cc_bl = evaluate_cross_context_baseline(model, episodes)
        if verbose:
            print(f"    → {cc_bl.accuracy:.1%}")

        results.append(ExperimentResult(
            n_facts=n_facts,
            baseline=bl,
            hebbian=hb,
            cross_context=cc,
            cross_baseline=cc_bl,
        ))

    # ── Summary ──
    if verbose:
        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"SUMMARY (elapsed: {elapsed:.0f}s, device: {device})")
        print(f"{'=' * 60}\n")

        print(f"{'n_facts':>7} │ {'Baseline':>8} {'Hebbian':>8} "
              f"{'Cross':>8} {'Cross-BL':>8} │ {'Gap':>6}")
        print("─" * 62)

        for r in results:
            gap = r.cross_context.accuracy - r.cross_baseline.accuracy
            print(f"{r.n_facts:>7d} │ "
                  f"{r.baseline.accuracy:>7.1%} {r.hebbian.accuracy:>7.1%} "
                  f"{r.cross_context.accuracy:>7.1%} "
                  f"{r.cross_baseline.accuracy:>7.1%} │ "
                  f"{gap:>+5.1%}")

        # Tier breakdown if available
        has_tiers = any(
            r.cross_context.tier1_accuracy is not None for r in results)
        if has_tiers:
            print(f"\nTier breakdown (cross-context):")
            for r in results:
                t1 = r.cross_context.tier1_accuracy
                t2 = r.cross_context.tier2_accuracy
                t1_str = f"{t1:.1%}" if t1 is not None else "n/a"
                t2_str = f"{t2:.1%}" if t2 is not None else "n/a"
                print(f"  n_facts={r.n_facts}: Tier1={t1_str}  Tier2={t2_str}")

        # Success check
        print(f"\n{'─' * 40}")
        cc_1 = [r for r in results if r.n_facts == 1]
        if cc_1:
            r1 = cc_1[0]
            gap1 = r1.cross_context.accuracy - r1.cross_baseline.accuracy
            print(f"Cross-context @ n=1: {r1.cross_context.accuracy:.1%} "
                  f"vs baseline {r1.cross_baseline.accuracy:.1%} "
                  f"(gap={gap1:+.1%})")

        cc_3 = [r for r in results if r.n_facts == 3]
        if cc_3:
            r3 = cc_3[0]
            gap3 = r3.cross_context.accuracy - r3.cross_baseline.accuracy
            print(f"Cross-context @ n=3: {r3.cross_context.accuracy:.1%} "
                  f"vs baseline {r3.cross_baseline.accuracy:.1%} "
                  f"(gap={gap3:+.1%})")

        success = True
        if cc_1:
            gap1 = cc_1[0].cross_context.accuracy - cc_1[0].cross_baseline.accuracy
            if gap1 < 0.25:
                success = False
        if cc_3:
            gap3 = cc_3[0].cross_context.accuracy - cc_3[0].cross_baseline.accuracy
            if gap3 < 0.15:
                success = False
        if success:
            print("SUCCESS: cross-session NLP fact memory demonstrated")
        else:
            print("BELOW TARGET: insufficient cross-session memory "
                  "(targets: n=1 gap>25pp, n=3 gap>15pp)")

    return results


def run_quick(device_name: str | None = None,
              verbose: bool = True) -> list[ExperimentResult]:
    """Quick smoke test: smaller pretrain, fewer episodes."""
    return run(
        n_pretrain=5000,
        max_facts=3,
        pretrain_epochs=30,
        n_eval_episodes=50,
        fact_counts=[1, 3],
        d_model=256,
        n_heads=8,
        n_layers=8,
        device_name=device_name,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2: NLP Fact Memorization")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (~5 min)")
    parser.add_argument("--n-pretrain", type=int, default=20000)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--trace-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: mps, cuda, cpu (auto-detect if omitted)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save model to this path after pretraining")
    args = parser.parse_args()

    if args.quick:
        run_quick(device_name=args.device)
    else:
        run(
            n_pretrain=args.n_pretrain,
            pretrain_epochs=args.pretrain_epochs,
            n_eval_episodes=args.n_eval,
            alpha=args.alpha,
            trace_lr=args.trace_lr,
            seed=args.seed,
            device_name=args.device,
            save_path_arg=args.save,
        )
