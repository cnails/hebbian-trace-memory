"""Hebbian Attention layer and MiniGPT model.

Core idea: standard attention computes A = softmax(Q @ K.T / sqrt(d)).
We add a trace matrix T (d_k x d_k) that biases attention based on
past experience:

    scores = Q @ (I + alpha * T) @ K.T
    A = softmax(scores / sqrt(d_k))
    T = decay * T + lr * hebbian_update(Q, K, A)

T is NOT a parameter — it's not updated by backprop. It accumulates
Hebbian updates during inference, giving the model persistent memory
across forward passes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianAttention(nn.Module):
    """Multi-head attention with Hebbian trace matrices.

    Each head maintains a trace matrix T of shape (d_k, d_k) that
    modifies attention scores: scores += alpha * (Q @ T @ K.T).

    Three update variants:
        'outer':     Q.T @ A @ K        (classic Hebbian outer product)
        'covariance': K.T @ A.T @ A @ K (attention-weighted covariance)
        'error':     Q.T @ error @ K    (delta rule, surprise-driven)
    """

    def __init__(self, d_model: int, n_heads: int,
                 alpha: float = 0.05,
                 trace_lr: float = 0.05,
                 trace_decay: float = 0.95,
                 update_variant: str = 'outer',
                 dropout: float = 0.0,
                 use_key_q: bool = True,
                 adaptive_alpha: bool = False,
                 trace_norm_target: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.alpha = alpha
        self.trace_lr = trace_lr
        self.trace_decay = trace_decay
        self.adaptive_alpha = adaptive_alpha
        self.trace_norm_target = trace_norm_target
        self.update_variant = update_variant
        self.use_key_q = use_key_q  # True=Q[key_pos], False=Q[arrow_pos]

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Score trace: (n_heads, d_k, d_k) — biases attention routing
        self.register_buffer(
            'traces', torch.zeros(n_heads, self.d_k, self.d_k))
        # Value trace: (n_heads, d_k, d_k) — stores Q→V associations
        # for direct value retrieval across sequences
        self.register_buffer(
            'value_traces', torch.zeros(n_heads, self.d_k, self.d_k))

        # Control flags
        self._use_trace = False
        self._update_trace = False
        self._trace_Q = None  # external Q for trace ops (e.g., first-layer Q)

        # Pattern separation (dentate gyrus — sparse expansion for value trace)
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = self.d_k

        # Reconsolidation erasure (selective forgetting for value trace)
        self._erase_before_write = False
        self._erase_lr = 1.0

    def _get_effective_alpha(self, trace_matrix: torch.Tensor) -> torch.Tensor | float:
        """Compute per-head adaptive alpha based on trace Frobenius norms.

        When adaptive_alpha=False, returns scalar self.alpha.
        When True, returns (1, H, 1, 1) tensor:
            alpha_eff = alpha / (1 + ||T_h|| / trace_norm_target)
        So alpha stays near full when trace is small, and decreases
        as trace norm grows (preventing trace from overwhelming attention).
        """
        if not self.adaptive_alpha:
            return self.alpha
        # Per-head Frobenius norms: (H,)
        head_norms = trace_matrix.norm(dim=(1, 2))
        alpha_eff = self.alpha / (1.0 + head_norms / self.trace_norm_target)
        return alpha_eff.view(1, -1, 1, 1)

    def _make_linking_mask(self, token_slice: torch.Tensor) -> torch.Tensor:
        """Create boolean mask for linking token positions.

        Supports both single arrow_token_id (backward compat) and
        multiple linking_token_ids for NLP experiments.
        """
        mask = torch.zeros_like(token_slice, dtype=torch.bool)
        if hasattr(self, 'linking_token_ids') and self.linking_token_ids is not None:
            for tid in self.linking_token_ids:
                mask |= (token_slice == tid)
        return mask

    def set_per_head_decay(self, decay_rates: list[float]):
        """Set per-head trace decay rates.

        Args:
            decay_rates: list of H floats, one decay rate per head.
        """
        assert len(decay_rates) == self.n_heads
        self.trace_decay = torch.tensor(
            decay_rates, dtype=self.traces.dtype, device=self.traces.device
        ).view(self.n_heads, 1, 1)

    def enable_pattern_separation(self, expand_factor: int, top_k: int,
                                  seed: int = 0):
        """Enable sparse expansion for value trace (dentate gyrus).

        Frozen random projection d_k -> d_k*expand_factor + ReLU + top-k.
        Creates near-unique addressing codes per concept word, reducing
        interference in the value trace at high fact counts.

        Only affects value trace (not score trace).

        Args:
            expand_factor: expansion ratio (e.g., 4 or 8).
            top_k: number of dimensions to keep after sparsification.
            seed: RNG seed for reproducible random projection.
        """
        self._pattern_sep_enabled = True
        self._expand_factor = expand_factor
        self._top_k = top_k
        self._expanded_dim = self.d_k * expand_factor

        # Frozen random projection (Johnson-Lindenstrauss)
        gen = torch.Generator()
        gen.manual_seed(seed)
        W = torch.randn(self.d_k, self._expanded_dim, generator=gen)
        W = W / math.sqrt(self.d_k)
        self.register_buffer('W_expand', W.to(self.traces.device))

        # Resize value traces: (H, d_k, d_k) -> (H, expanded_dim, d_k)
        self.value_traces = torch.zeros(
            self.n_heads, self._expanded_dim, self.d_k,
            device=self.traces.device, dtype=self.traces.dtype)

    def disable_pattern_separation(self):
        """Disable sparse expansion, restore standard value trace."""
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = self.d_k

        # Restore standard value trace shape
        self.value_traces = torch.zeros(
            self.n_heads, self.d_k, self.d_k,
            device=self.traces.device, dtype=self.traces.dtype)

    def _sparse_expand(self, Q: torch.Tensor) -> torch.Tensor:
        """Project Q through frozen random expansion + ReLU + top-k.

        Args:
            Q: (..., d_k) tensor
        Returns:
            (..., expanded_dim) sparse tensor with only top_k non-zero entries
        """
        Q_exp = torch.matmul(Q, self.W_expand)  # (..., expanded_dim)
        Q_exp = F.relu(Q_exp)
        if self._top_k > 0 and self._top_k < self._expanded_dim:
            topk_vals, topk_idx = Q_exp.topk(self._top_k, dim=-1)
            Q_sparse = torch.zeros_like(Q_exp)
            Q_sparse.scatter_(-1, topk_idx, topk_vals)
            return Q_sparse
        return Q_exp

    def reset_traces(self):
        """Zero all trace matrices."""
        self.traces.zero_()
        self.value_traces.zero_()

    def set_trace_mode(self, use: bool = False, update: bool = False):
        """Control trace behavior.

        Args:
            use: if True, trace biases attention scores.
            update: if True, trace accumulates Hebbian updates.
        """
        self._use_trace = use
        self._update_trace = update

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            mask: (seq_len, seq_len) causal mask, True = masked positions.
            token_ids: (batch, seq_len) optional input token indices.
                       If provided and arrow_token_id is set, trace updates
                       are restricted to arrow positions (reduces noise).

        Returns:
            (batch, seq_len, d_model)
        """
        B, S, _ = x.shape
        d_k = self.d_k

        # Project to Q, K, V: (batch, seq_len, d_model)
        Q = self.W_q(x).view(B, S, self.n_heads, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, seq_len, d_k)

        # Cache Q for first-layer sharing
        self._cached_Q = Q.detach()

        # Q for trace operations: use external (first-layer) Q if provided
        Q_trace = self._trace_Q if self._trace_Q is not None else Q

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, S, S)

        # Add trace bias: Q @ T @ K.T
        if self._use_trace:
            # traces: (H, d_k, d_k), expand for batch
            T = self.traces.unsqueeze(0)  # (1, H, d_k, d_k)
            QT = torch.matmul(Q, T)      # (B, H, S, d_k)
            trace_scores = torch.matmul(QT, K.transpose(-2, -1))  # (B, H, S, S)
            score_alpha = self._get_effective_alpha(self.traces)
            scores = scores + score_alpha * trace_scores

        # Scale
        scores = scores / math.sqrt(d_k)

        # Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        A = F.softmax(scores, dim=-1)
        A = self.dropout(A)

        # Output
        out = torch.matmul(A, V)  # (B, H, S, d_k)

        # Add value trace: direct retrieval of stored values
        if self._use_trace:
            Tv = self.value_traces.unsqueeze(0)  # (1, H, *, d_k)
            if self.use_key_q and S > 1:
                # Use Q from PREVIOUS position (the key letter) to address
                # trace, matching storage which uses Q[key_pos].
                Q_addr = torch.cat([
                    torch.zeros_like(Q_trace[:, :, :1, :]),  # pos 0: no previous
                    Q_trace[:, :, :-1, :],                    # pos 1..S-1: Q[i-1]
                ], dim=2)  # (B, H, S, d_k)
            else:
                # Original: use Q at current position (arrow's Q)
                Q_addr = Q_trace
            # Pattern separation: sparse expand Q before addressing value trace
            if self._pattern_sep_enabled:
                Q_addr = self._sparse_expand(Q_addr)  # (B, H, S, expanded_dim)
            stored_values = torch.matmul(Q_addr, Tv)  # (B, H, S, d_k)
            if getattr(self, 'adaptive_score_only', False):
                value_alpha = self.alpha  # fixed: preserve cross-context retrieval
            else:
                value_alpha = self._get_effective_alpha(self.value_traces)
            out = out + value_alpha * stored_values

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(out)

        # Hebbian trace update (no grad)
        if self._update_trace:
            self._hebbian_update(Q_trace.detach(), K.detach(),
                                 A.detach(), V.detach(),
                                 token_ids=token_ids)

        return out

    @torch.no_grad()
    def _hebbian_update(self, Q: torch.Tensor, K: torch.Tensor,
                        A: torch.Tensor, V: torch.Tensor,
                        token_ids: torch.Tensor | None = None):
        """Update trace matrices with Hebbian rule.

        Q, K, V: (batch, n_heads, seq_len, d_k)
        A: (batch, n_heads, seq_len, seq_len) — attention weights
        token_ids: (batch, seq_len) optional — if provided and
                   arrow_token_id is set, only arrow positions update
                   the value trace (drastically reduces noise).
        """
        B, H, S, d_k = Q.shape

        # ── Score trace update ──
        if self.update_variant == 'outer':
            update = torch.einsum('bhip,bhij,bhjq->hpq', Q, A, K)
            update = update / (B * S)

        elif self.update_variant == 'covariance':
            AK = torch.matmul(A, K)
            update = torch.einsum('bhip,bhjq->hpq', AK, AK)
            update = update / (B * S)

        elif self.update_variant == 'error':
            T = self.traces.unsqueeze(0)
            predicted_scores = torch.matmul(
                torch.matmul(Q, T), K.transpose(-2, -1)) / math.sqrt(d_k)
            predicted_A = F.softmax(predicted_scores, dim=-1)
            error = A - predicted_A
            update = torch.einsum('bhip,bhij,bhjq->hpq', Q, error, K)
            update = update / (B * S)

        else:
            raise ValueError(f"Unknown variant: {self.update_variant}")

        self.traces = self.trace_decay * self.traces + self.trace_lr * update

        # ── Value trace update ──
        if self.use_key_q:
            # KEY→DIGIT: Q[key_pos] paired with V[digit_pos]
            # key is at arrow_pos-1, digit is at arrow_pos+1
            # So Q[i] paired with V[i+2] where token[i+1] is "→"
            if S > 2:
                Q_store = Q[:, :, :-2, :]   # positions 0..S-3
                V_store = V[:, :, 2:, :]    # positions 2..S-1

                if (token_ids is not None and
                        hasattr(self, 'linking_token_ids') and
                        self.linking_token_ids is not None):
                    arrow_mask = self._make_linking_mask(
                        token_ids[:, 1:-1])
                    mask_expanded = arrow_mask.unsqueeze(1).unsqueeze(-1).float()
                    Q_store = Q_store * mask_expanded
                    n_arrows = arrow_mask.sum().item()
                    denom = max(n_arrows * H, 1)
                else:
                    denom = B * (S - 2)

                # Pattern separation: sparse expand Q for value trace
                if self._pattern_sep_enabled:
                    Q_store = self._sparse_expand(Q_store)

                # Reconsolidation erasure: read old value, erase before write
                # Normalize Q per-position to make erase scale-independent.
                # Without normalization, ||Q_expanded||^2 ~ 175 per head
                # causes erase to remove 20x+ the stored value (catastrophic).
                if self._erase_before_write:
                    Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    Q_erase = Q_store / Q_norms
                    V_old = torch.einsum(
                        'bhip,hpq->bhiq', Q_erase, self.value_traces)
                    erase = torch.einsum(
                        'bhip,bhiq->hpq', Q_erase, V_old) / denom
                    self.value_traces = (
                        self.value_traces - self._erase_lr * erase)

                v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
                v_update = v_update / denom
            else:
                v_update = torch.zeros_like(self.value_traces)
        else:
            # ARROW→DIGIT: Q[arrow_pos] paired with V[digit_pos] (original)
            if S > 1:
                Q_store = Q[:, :, :-1, :]   # (B, H, S-1, d_k)
                V_store = V[:, :, 1:, :]    # (B, H, S-1, d_k)

                if (token_ids is not None and
                        hasattr(self, 'linking_token_ids') and
                        self.linking_token_ids is not None):
                    arrow_mask = self._make_linking_mask(
                        token_ids[:, :-1])
                    mask_expanded = arrow_mask.unsqueeze(1).unsqueeze(-1).float()
                    Q_store = Q_store * mask_expanded
                    n_arrows = arrow_mask.sum().item()
                    denom = max(n_arrows * H, 1)
                else:
                    denom = B * (S - 1)

                # Pattern separation: sparse expand Q for value trace
                if self._pattern_sep_enabled:
                    Q_store = self._sparse_expand(Q_store)

                # Reconsolidation erasure (normalized Q, see above)
                if self._erase_before_write:
                    Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    Q_erase = Q_store / Q_norms
                    V_old = torch.einsum(
                        'bhip,hpq->bhiq', Q_erase, self.value_traces)
                    erase = torch.einsum(
                        'bhip,bhiq->hpq', Q_erase, V_old) / denom
                    self.value_traces = (
                        self.value_traces - self._erase_lr * erase)

                v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
                v_update = v_update / denom
            else:
                v_update = torch.zeros_like(self.value_traces)

        self.value_traces = (self.trace_decay * self.value_traces
                             + self.trace_lr * v_update)


class TransformerBlock(nn.Module):
    """Standard transformer block: attention + MLP with residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int | None = None, dropout: float = 0.0,
                 **attention_kwargs):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = HebbianAttention(d_model, n_heads, dropout=dropout,
                                     **attention_kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                token_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask, token_ids=token_ids)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Tiny GPT-2-style autoregressive model with Hebbian attention.

    Architecture:
        token_embed + pos_embed → N × TransformerBlock → LayerNorm → logits
    """

    def __init__(self, vocab_size: int, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2,
                 max_seq_len: int = 64, dropout: float = 0.1,
                 use_first_layer_q: bool = False,
                 use_raw_embed: bool = False,
                 **attention_kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.use_first_layer_q = use_first_layer_q
        self.use_raw_embed = use_raw_embed

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # When use_raw_embed=True, uses first layer's W_q on raw token
        # embeddings (no position, no context) for trace storage keys.
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout,
                             **attention_kwargs)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.token_embed.weight

        # Causal mask (registered as buffer)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
                          diagonal=1)
        self.register_buffer('causal_mask', mask)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: torch.Tensor,
                concept_injection: dict[int, int] | None = None,
                ) -> torch.Tensor:
        """Forward pass.

        Args:
            idx: (batch, seq_len) token indices.
            concept_injection: optional dict {position: concept_token_id}
                for overriding trace Q at specific positions. Used for
                Tier 2 templates where the concept word doesn't naturally
                appear at the correct storage position. B=1 assumed.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = idx.shape
        assert S <= self.max_seq_len

        tok = self.token_embed(idx)
        pos = self.pos_embed(torch.arange(S, device=idx.device))
        x = self.drop(tok + pos)

        mask = self.causal_mask[:S, :S]
        d_k = self.d_model // self.n_heads

        if self.use_raw_embed:
            # Compute trace Q from raw token embeddings: W_q(LN(tok))
            # No position → same key letter always maps to same Q
            # LN is critical (normalizes distribution for W_q)
            # W_q is pretrained (projects into head-wise trace space)
            with torch.no_grad():
                store_Q = self.blocks[0].attn.W_q(
                    self.blocks[0].ln1(tok))  # (B, S, d_model)
                store_Q = store_Q.view(B, S, self.n_heads, d_k).transpose(1, 2)
                # (B, H, S, d_k) — context-free storage keys

                # Concept injection: override Q at specific positions
                # with the Q of a concept word (e.g., inject Q("name")
                # at the position before "am" for "I am {X}" templates)
                if concept_injection is not None:
                    ln1 = self.blocks[0].ln1
                    W_q = self.blocks[0].attn.W_q
                    for pos_idx, concept_tok_id in concept_injection.items():
                        if pos_idx < S:
                            c_embed = self.token_embed.weight[concept_tok_id]
                            c_q = W_q(ln1(c_embed.unsqueeze(0)))  # (1, d_model)
                            c_q = c_q.view(1, self.n_heads, d_k)  # (1, H, d_k)
                            store_Q[:, :, pos_idx, :] = c_q

            for block in self.blocks:
                block.attn._trace_Q = store_Q
            for block in self.blocks:
                x = block(x, mask=mask, token_ids=idx)
        elif self.use_first_layer_q and len(self.blocks) > 1:
            # Clear stale Q from previous forward pass
            for block in self.blocks:
                block.attn._trace_Q = None
            # Run first block (uses its own Q naturally = first-layer Q)
            x = self.blocks[0](x, mask=mask, token_ids=idx)
            first_Q = self.blocks[0].attn._cached_Q
            # Share first-layer Q with remaining layers only
            for block in self.blocks[1:]:
                block.attn._trace_Q = first_Q
            # Run remaining blocks
            for block in self.blocks[1:]:
                x = block(x, mask=mask, token_ids=idx)
        else:
            # Clear external Q — each layer uses its own
            for block in self.blocks:
                block.attn._trace_Q = None
            for block in self.blocks:
                x = block(x, mask=mask, token_ids=idx)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def reset_traces(self):
        """Reset all trace matrices to zero."""
        for block in self.blocks:
            block.attn.reset_traces()

    def set_trace_mode(self, use: bool = False, update: bool = False):
        """Set trace mode for all attention layers."""
        for block in self.blocks:
            block.attn.set_trace_mode(use=use, update=update)

    def get_attention_layers(self) -> list[HebbianAttention]:
        """Return all HebbianAttention layers."""
        return [block.attn for block in self.blocks]

    def set_arrow_token_id(self, token_id: int):
        """Enable arrow-only trace updates (backward-compatible wrapper).

        Calls set_linking_token_ids with a single token.
        """
        self.set_linking_token_ids([token_id])

    def set_adaptive_alpha(self, enabled: bool, norm_target: float = 1.0,
                           score_only: bool = False):
        """Toggle adaptive alpha for all attention layers.

        Args:
            enabled: enable adaptive alpha scaling.
            norm_target: trace norm at which alpha halves.
            score_only: if True, only score trace uses adaptive alpha;
                        value trace keeps fixed alpha (preserves cross-context).
        """
        for attn in self.get_attention_layers():
            attn.adaptive_alpha = enabled
            attn.trace_norm_target = norm_target
            attn.adaptive_score_only = score_only

    def set_per_head_decay(self, decay_rates: list[float]):
        """Set per-head trace decay for all attention layers.

        Args:
            decay_rates: list of H floats, one decay rate per head.
        """
        for attn in self.get_attention_layers():
            attn.set_per_head_decay(decay_rates)

    def enable_pattern_separation(self, expand_factor: int, top_k: int,
                                  seed: int = 0):
        """Enable sparse expansion for value trace in all attention layers.

        See HebbianAttention.enable_pattern_separation for details.
        """
        for attn in self.get_attention_layers():
            attn.enable_pattern_separation(expand_factor, top_k, seed=seed)

    def disable_pattern_separation(self):
        """Disable sparse expansion in all attention layers."""
        for attn in self.get_attention_layers():
            attn.disable_pattern_separation()

    def set_erase_mode(self, enabled: bool, erase_lr: float = 1.0):
        """Toggle reconsolidation erasure for value trace.

        When enabled, before writing new Q→V associations, the model
        first reads the current stored value and subtracts it from the
        trace (reconsolidation). This allows clean overwriting of
        updated facts.

        Should be OFF during initial encoding and ON during updates.

        Args:
            enabled: enable erase-before-write.
            erase_lr: erasure strength (1.0 = fully erase retrieved value).
        """
        for attn in self.get_attention_layers():
            attn._erase_before_write = enabled
            attn._erase_lr = erase_lr

    def set_linking_token_ids(self, token_ids: list[int]):
        """Enable linking-token-filtered trace updates.

        Only update value trace at positions where the input token is
        one of the linking tokens (e.g., '→' for synthetic, or
        'is'/'in'/'at'/'from' for NLP facts).
        """
        for attn in self.get_attention_layers():
            attn.linking_token_ids = token_ids

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int = 1) -> torch.Tensor:
        """Autoregressive generation (greedy)."""
        for _ in range(max_new):
            logits = self.forward(idx[:, -self.max_seq_len:])
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
