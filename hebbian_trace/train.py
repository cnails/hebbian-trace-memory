"""Pretraining: standard autoregressive training with trace OFF.

The model learns in-context key-value retrieval using backprop.
Trace matrices are NOT used during pretraining — just standard attention.
After pretraining, the model has learned meaningful Q/K representations
that the Hebbian trace can later exploit.
"""

import time
import torch
import torch.nn as nn

from .model import MiniGPT
from .tasks import VOCAB, make_pretrain_loader, get_vocab_size


def pretrain(n_sequences: int = 5000,
             n_pairs: int = 5,
             batch_size: int = 32,
             epochs: int = 20,
             lr: float = 3e-4,
             d_model: int = 64,
             n_heads: int = 2,
             n_layers: int = 2,
             max_seq_len: int = 64,
             dropout: float = 0.1,
             seed: int = 42,
             verbose: bool = True,
             **attention_kwargs) -> tuple[MiniGPT, dict]:
    """Pretrain MiniGPT on key-value retrieval task.

    Args:
        n_sequences: number of training sequences
        n_pairs: key-value pairs per sequence
        batch_size: training batch size
        epochs: number of training epochs
        lr: learning rate
        d_model: model dimension
        n_heads: number of attention heads
        n_layers: number of transformer blocks
        max_seq_len: maximum sequence length
        dropout: dropout rate
        seed: random seed
        verbose: print progress
        **attention_kwargs: passed to HebbianAttention (alpha, trace_lr, etc.)

    Returns:
        (model, stats) where stats contains training history
    """
    torch.manual_seed(seed)

    vocab_size = get_vocab_size()
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        **attention_kwargs,
    )

    # Trace OFF during pretraining
    model.set_trace_mode(use=False, update=False)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"MiniGPT: {n_params:,} parameters")
        print(f"  vocab_size={vocab_size}, d_model={d_model}, "
              f"n_heads={n_heads}, n_layers={n_layers}")
        print(f"  Training: {n_sequences} sequences, {n_pairs} pairs each, "
              f"{epochs} epochs")

    loader = make_pretrain_loader(
        n_sequences=n_sequences,
        n_pairs=n_pairs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        seed=seed,
    )

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
            # input_ids: (batch, seq_len)
            # target_ids: (batch, seq_len)
            # loss_mask: (batch, seq_len) — 1.0 on all non-pad positions

            logits = model(input_ids)  # (batch, seq_len, vocab_size)

            # Flatten for loss
            B, S, V = logits.shape
            loss_flat = loss_fn(
                logits.reshape(B * S, V),
                target_ids.reshape(B * S),
            )  # (B * S,)

            # Apply loss mask (all non-pad positions)
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
            # Overall accuracy (all positions)
            preds = logits.argmax(dim=-1)  # (B, S)
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


def save_model(model: MiniGPT, path: str):
    """Save model weights."""
    torch.save(model.state_dict(), path)


def load_model(path: str, vocab_size: int | None = None,
               **model_kwargs) -> MiniGPT:
    """Load model weights."""
    if vocab_size is None:
        vocab_size = get_vocab_size()
    model = MiniGPT(vocab_size=vocab_size, **model_kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


if __name__ == "__main__":
    # Quick test: small pretraining run
    model, stats = pretrain(
        n_sequences=500,
        n_pairs=3,
        batch_size=32,
        epochs=5,
        verbose=True,
    )
    print(f"\nFinal accuracy: {stats['epoch_acc'][-1]:.1%}")
