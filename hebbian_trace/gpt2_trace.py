"""Hebbian trace module for pretrained GPT-2.

External memory that attaches to a frozen GPT-2 model.
Stores Q→V associations via Hebbian learning using GPT-2's
token embeddings (wte) for context-free addressing.

Architecture:
    Q_base = W_proj(LN_proj(wte(token)))  — context-free storage keys
    Q_ctx  = W_ctx(LN_ctx(hidden_states)) — contextual component (optional)
    Q = Q_base + beta * Q_ctx             — blended addressing
    V = W_val(wte(token))                 — context-free values (always)
    Trace: (H, d_trace, d_trace) symmetric matrix
    Retrieved: W_out(Q_addr @ T_v)        — projected back to d_model

Injection: logit-space bias via wte weight-tying.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class HebbianTraceModule(nn.Module):
    """External Hebbian trace memory for pretrained transformers.

    Stores Q→V associations via Hebbian learning, retrieves via Q addressing.
    Uses linking-token mask with shift-1 offset (same as MiniGPT).
    """

    def __init__(self, d_model: int = 768, n_heads: int = 8,
                 d_trace: int = 64, alpha: float = 0.1,
                 trace_lr: float = 0.1, trace_decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_trace = d_trace
        self.alpha = alpha
        self.trace_lr = trace_lr
        self.trace_decay = trace_decay

        # Key projection: d_model → n_heads * d_trace (context-free)
        self.W_proj = nn.Linear(d_model, n_heads * d_trace, bias=False)
        self.ln_proj = nn.LayerNorm(d_model)

        # Contextual key projection: hidden_states → n_heads * d_trace
        self.W_ctx = nn.Linear(d_model, n_heads * d_trace, bias=False)
        self.ln_ctx = nn.LayerNorm(d_model)

        # Value projection: d_model → n_heads * d_trace
        self.W_val = nn.Linear(d_model, n_heads * d_trace, bias=False)

        # Output projection: n_heads * d_trace → d_model
        self.W_out = nn.Linear(n_heads * d_trace, d_model, bias=False)

        # Value trace: (n_heads, d_trace, d_trace) — stores Q→V associations
        self.register_buffer(
            'value_traces', torch.zeros(n_heads, d_trace, d_trace))

        # Control flags (ACh modulation)
        self._use_trace = False
        self._update_trace = False

        # Linking tokens (BPE token IDs for "is", "in", etc.)
        self.linking_token_ids: list[int] | None = None

        # Learned storage gate (position-level: fires on linking tokens)
        self.W_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)  # sigmoid(0)=0.5, half-open
        self._use_learned_gate = False
        self._gate_tau = 1.0  # temperature: lower → sharper (more binary)

        # Semantic gate (concept-level: evaluates concept word quality)
        self.W_gate_key = nn.Linear(d_model, 1)
        nn.init.zeros_(self.W_gate_key.weight)
        nn.init.constant_(self.W_gate_key.bias, -2.0)  # sigmoid(-2)≈0.12, starts mostly closed
        self._use_dual_gate = False
        self._gate_key_tau = 1.0

        # MLP encoder (entorhinal bottleneck) for contextual Q
        # Replaces linear W_ctx: d_model → bottleneck → H*d_trace with ReLU
        # Bottleneck forces information compression; ReLU enables selective
        # feature gating (keep ownership signal, discard context variability)
        self._d_enc_bottleneck = max(d_model // 6, 64)
        self.ln_enc = nn.LayerNorm(d_model)
        self.W_enc = nn.Sequential(
            nn.Linear(d_model, self._d_enc_bottleneck),
            nn.ReLU(),
            nn.Linear(self._d_enc_bottleneck, n_heads * d_trace),
        )
        # Zero-init last layer: encoder output = 0 at start → pure Q_base
        nn.init.zeros_(self.W_enc[-1].weight)
        nn.init.zeros_(self.W_enc[-1].bias)
        self._use_encoder = False

        # Pattern separation (dentate gyrus)
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = d_trace

        # Reconsolidation erasure (fact updates)
        self._erase_before_write = False
        self._erase_lr = 1.0

        # Replay buffer: stores Q keys for sleep re-activation
        self._replay_buffer: list[torch.Tensor] = []  # list of (H, addr_dim)
        self._replay_enabled = False

        # Autoassociative trace (pattern completion, CA3)
        # Maps Q_variant → Q_concept for paraphrase resolution
        self.register_buffer(
            'autoassociative_traces',
            torch.zeros(n_heads, d_trace, d_trace))
        self._auto_enabled = False
        self._completion_alpha = 1.0

        # Per-head beta for contextual Q blending (CLS specialization)
        self.beta_per_head: torch.Tensor | None = None  # (H,) or None

        # Trace banks (hash-routed memory for capacity scaling)
        # When n_trace_banks > 1, facts are routed to separate trace matrices
        # based on their sparse Q pattern. Reduces interference from O(N) to
        # O(N/n_banks) and eliminates cross-bank decay.
        self.n_trace_banks = 1
        self._bank_traces: torch.Tensor | None = None

        # Initialize with orthogonal projections
        self._init_orthogonal()

    def _init_orthogonal(self):
        """Initialize W_val and W_out as approximate inverses.

        W_val projects 768 → 512 (orthogonal rows).
        W_out = W_val.T so round-trip W_out(W_val(x)) ≈ x (projection).
        W_proj uses standard random init (JL theorem).
        """
        total_dim = self.n_heads * self.d_trace
        # Orthogonal init for value path
        Q, _ = torch.linalg.qr(torch.randn(self.d_model, self.d_model))
        self.W_val.weight.data = Q[:total_dim, :]  # (total_dim, d_model)
        self.W_out.weight.data = Q[:total_dim, :].T  # (d_model, total_dim)

    def set_per_head_beta(self, values: list[float]):
        """Set per-head β values for contextual Q blending (CLS).

        Different heads use different amounts of contextual information:
        low β → context-free (stable cross-context), high β → contextual
        (strong discrimination).

        Args:
            values: list of H floats, one β per head.
        """
        assert len(values) == self.n_heads
        self.beta_per_head = torch.tensor(values, dtype=torch.float32)

    def compute_qv(self, wte: nn.Embedding,
                   input_ids: torch.Tensor,
                   hidden_states: torch.Tensor | None = None,
                   beta: float | torch.Tensor = 0.0,
                   train_ctx: bool = False,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute trace Q and V, optionally with contextual enrichment.

        Q = Q_base + beta * Q_ctx, where:
          Q_base = W_proj(LN_proj(wte(token)))    — context-free (stable)
          Q_ctx  = W_ctx(LN_ctx(hidden_states))   — contextual (discriminative)

        V always stays context-free: V = W_val(wte(token)).

        Args:
            wte: GPT-2 token embedding layer (frozen).
            input_ids: (B, S) token indices.
            hidden_states: (B, S, d_model) GPT-2 hidden states at some layer.
                          If None or beta=0, Q is purely context-free.
            beta: blending weight for contextual component. Can be:
                  - float: uniform β for all heads (0 = context-free)
                  - Tensor (H,): per-head β (CLS specialization)
            train_ctx: if True, allow gradient through W_ctx/LN_ctx
                      (for contrastive training of W_ctx). Default False.

        Returns:
            Q: (B, H, S, d_trace) storage/retrieval keys
            V: (B, H, S, d_trace) context-free values
        """
        B, S = input_ids.shape
        with torch.no_grad():
            tok_embed = wte(input_ids).float()  # (B, S, d_model), cast to fp32

        # Context-free base Q
        Q = self.W_proj(self.ln_proj(tok_embed))  # (B, S, H*d_trace)
        Q = Q.view(B, S, self.n_heads, self.d_trace).permute(0, 2, 1, 3)

        # Add contextual component if provided
        if isinstance(beta, torch.Tensor):
            needs_ctx = beta.abs().sum() > 0
        else:
            needs_ctx = beta > 0
        if hidden_states is not None and needs_ctx:
            # Select context encoder: MLP bottleneck or linear
            if self._use_encoder:
                _ctx = lambda h: self.W_enc(self.ln_enc(h))
            else:
                _ctx = lambda h: self.W_ctx(self.ln_ctx(h))
            if train_ctx:
                Q_ctx = _ctx(hidden_states)
            else:
                with torch.no_grad():
                    Q_ctx = _ctx(hidden_states)
            Q_ctx = Q_ctx.view(B, S, self.n_heads, self.d_trace).permute(
                0, 2, 1, 3)
            # Per-head beta: (H,) → (1, H, 1, 1) for broadcast
            if isinstance(beta, torch.Tensor):
                beta_exp = beta.view(1, self.n_heads, 1, 1)
            else:
                beta_exp = beta
            Q = Q + beta_exp * Q_ctx

        # Context-free V (always)
        V = self.W_val(tok_embed)  # (B, S, H*d_trace)
        V = V.view(B, S, self.n_heads, self.d_trace).permute(0, 2, 1, 3)

        return Q, V  # both (B, H, S, d_trace)

    def _make_linking_mask(self, token_slice: torch.Tensor) -> torch.Tensor:
        """Create boolean mask for linking token positions."""
        mask = torch.zeros_like(token_slice, dtype=torch.bool)
        if self.linking_token_ids is not None:
            for tid in self.linking_token_ids:
                mask |= (token_slice == tid)
        return mask

    @torch.no_grad()
    def write(self, Q: torch.Tensor, V: torch.Tensor,
              token_ids: torch.Tensor):
        """Hebbian trace update: T_v += lr * Q_store.T @ V_store.

        Uses same shift logic as MiniGPT:
        - Q_store at linking_pos - 1 (concept word position)
        - V_store at linking_pos + 1 (entity position)

        Args:
            Q: (B, H, S, d_trace) storage keys
            V: (B, H, S, d_trace) storage values
            token_ids: (B, S) BPE token indices for linking mask
        """
        B, H, S, d_trace = Q.shape

        if S <= 2:
            return

        # Shift: Q[0..S-3] paired with V[2..S-1], linking mask on tokens[1..S-2]
        Q_store = Q[:, :, :-2, :]   # (B, H, S-2, d_trace)
        V_store = V[:, :, 2:, :]    # (B, H, S-2, d_trace)

        # Linking-token mask
        if self.linking_token_ids is not None:
            arrow_mask = self._make_linking_mask(token_ids[:, 1:-1])
            mask_expanded = arrow_mask.unsqueeze(1).unsqueeze(-1).float()
            Q_store = Q_store * mask_expanded
            n_arrows = arrow_mask.sum().item()
            denom = max(n_arrows * H, 1)
        else:
            denom = B * (S - 2)

        # Pattern separation: sparse expand Q
        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        # Record Q keys for replay before erasure/write
        if self.linking_token_ids is not None:
            self._record_for_replay(Q_store, arrow_mask)
        else:
            self._record_for_replay(Q_store)

        # Bank routing
        trace, bank_id = self._resolve_trace(Q_store)

        # Reconsolidation erasure: remove old Q→V before writing new
        # Q must be L2-normalized to prevent over-erasure
        # (sparse-expanded Q has ||Q||² ~ 175 per head)
        if self._erase_before_write:
            Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Q_erase = Q_store / Q_norms
            V_old = torch.einsum(
                'bhip,hpq->bhiq', Q_erase, trace)
            erase = torch.einsum(
                'bhip,bhiq->hpq', Q_erase, V_old) / denom
            trace = trace - self._erase_lr * erase

        # Hebbian outer product update
        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self._commit_trace(self.trace_decay * trace + self.trace_lr * v_update,
                           bank_id)

    @torch.no_grad()
    def write_direct(self, Q: torch.Tensor, V: torch.Tensor):
        """Write a single (concept, entity) association directly.

        Bypasses linking mask and shift logic. Produces identical trace
        state to template-based write for single facts.

        Args:
            Q: (H, d_trace) concept Q vector from compute_q_for_token
            V: (H, d_trace) entity V vector from compute_v_for_token
        """
        H = self.n_heads
        denom = 1 * H  # matches individual write (n_arrows=1)

        # Pattern separation
        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2))  # (1,H,1,expanded)
            Q_store = Q_exp.squeeze(0).squeeze(1)  # (H, expanded)
        else:
            Q_store = Q  # (H, d_trace)

        # Bank routing
        trace, bank_id = self._resolve_trace(Q_store)

        # Erasure (if enabled)
        if self._erase_before_write:
            Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Q_erase = Q_store / Q_norms  # (H, expanded)
            V_old = torch.einsum('hp,hpq->hq', Q_erase, trace)
            erase = torch.einsum('hp,hq->hpq', Q_erase, V_old) / denom
            trace = trace - self._erase_lr * erase

        # Hebbian outer product: T_v += lr * Q.T @ V / denom
        update = torch.einsum('hp,hq->hpq', Q_store, V) / denom
        self._commit_trace(self.trace_decay * trace + self.trace_lr * update,
                           bank_id)

    def read(self, Q: torch.Tensor) -> torch.Tensor:
        """Retrieve from trace: V_ret = Q_addr @ T_v.

        Q_addr at position i uses Q[i-1] (shift-1), matching storage
        which uses Q at concept_word position.

        Args:
            Q: (B, H, S, d_trace) query keys

        Returns:
            output: (B, S, d_model) retrieved values projected to model space
        """
        B, H, S, d_trace = Q.shape

        # Shift-1 for retrieval: Q_addr[i] = Q[i-1]
        Q_addr = torch.cat([
            torch.zeros_like(Q[:, :, :1, :]),  # pos 0: no previous
            Q[:, :, :-1, :],                    # pos 1..S-1: Q[i-1]
        ], dim=2)  # (B, H, S, d_trace)

        # Pattern separation: sparse expand Q for addressing
        if self._pattern_sep_enabled:
            Q_addr = self._sparse_expand(Q_addr)  # (B, H, S, expanded_dim)

        # Bank routing: use last position's Q (prediction target)
        # Last Q_addr = Q[S-2] (shift-1), which is the concept word
        # for questions like "What is my name?"
        trace, _ = self._resolve_trace(Q_addr[:, :, -1:, :])

        # Retrieve: Q_addr @ T_v → (B, H, S, d_trace)
        Tv = trace.unsqueeze(0)  # (1, H, *, d_trace)
        V_ret = torch.matmul(Q_addr, Tv)  # (B, H, S, d_trace)

        # Reshape and project to d_model
        V_ret = V_ret.permute(0, 2, 1, 3).reshape(B, S, H * d_trace)
        return self.W_out(V_ret)  # (B, S, d_model)

    @torch.no_grad()
    def read_direct(self, Q: torch.Tensor) -> torch.Tensor:
        """Read from T_v using a single pre-computed Q, no shift.

        Unlike read() which applies shift-1 for sequence-level retrieval,
        this does direct Q @ T_v for single-token lookups.
        Used for multi-hop chain retrieval where intermediate tokens
        are decoded and re-queried.

        Args:
            Q: (H, d_trace) query vector from compute_q_for_token

        Returns:
            retrieved: (d_model,) value projected to model space
        """
        # Pattern separation (same W_expand as write_direct)
        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2))  # (1,H,1,expanded)
            Q_addr = Q_exp.squeeze(0).squeeze(1)  # (H, expanded_dim)
        else:
            Q_addr = Q  # (H, d_trace)

        # Bank routing
        trace, _ = self._resolve_trace(Q_addr)

        # Retrieve: Q @ T_v → (H, d_trace)
        V_ret = torch.einsum('hp,hpq->hq', Q_addr, trace)

        # Project to d_model via W_out
        V_flat = V_ret.view(1, self.n_heads * self.d_trace)
        return self.W_out(V_flat).squeeze(0)  # (d_model,)

    def reset_traces(self):
        """Zero trace matrices (including all banks)."""
        self.value_traces.zero_()
        self.autoassociative_traces.zero_()
        if self._bank_traces is not None:
            self._bank_traces.zero_()

    def set_erase_mode(self, enabled: bool, erase_lr: float = 1.0):
        """Toggle reconsolidation erasure for fact updates.

        When enabled, before writing new Q→V associations, the old stored
        value is read and subtracted from the trace. This allows clean
        overwriting when a fact value changes.

        Should be OFF during initial encoding, ON during updates only.

        Args:
            enabled: enable erase-before-write.
            erase_lr: erasure strength (1.0 = fully erase retrieved value).
        """
        self._erase_before_write = enabled
        self._erase_lr = erase_lr

    def set_trace_mode(self, use: bool = False, update: bool = False):
        """Control trace behavior (ACh modulation)."""
        self._use_trace = use
        self._update_trace = update

    def set_linking_token_ids(self, token_ids: list[int]):
        """Set BPE token IDs for linking-token filtered updates."""
        self.linking_token_ids = token_ids

    # ── Replay (hippocampal sharp-wave ripples) ─────────────────────

    def set_replay_mode(self, enabled: bool):
        """Enable/disable replay buffer recording during writes."""
        self._replay_enabled = enabled
        if not enabled:
            self._replay_buffer.clear()

    def clear_replay_buffer(self):
        """Clear replay buffer (e.g. after sleep phase)."""
        self._replay_buffer.clear()

    def _record_for_replay(self, Q_store: torch.Tensor,
                           mask: torch.Tensor | None = None):
        """Record Q keys from a write operation into replay buffer.

        Stores per-fact Q keys (mean over batch, masked positions only).
        Each entry is (H, addr_dim) — one Q key per head.

        Args:
            Q_store: (B, H, S-2, addr_dim) storage keys (possibly sparse-expanded)
            mask: (B, S-2) boolean mask of active positions, or None for all
        """
        if not self._replay_enabled:
            return

        with torch.no_grad():
            if mask is not None:
                # Average Q over active (masked) positions per batch
                mask_f = mask.float().unsqueeze(1).unsqueeze(-1)  # (B,1,S-2,1)
                Q_masked = Q_store * mask_f
                n_active = mask_f.sum(dim=2).clamp(min=1)  # (B,1,1,1)
                Q_mean = Q_masked.sum(dim=2) / n_active.squeeze(-1)  # (B,H,addr_dim)
            else:
                Q_mean = Q_store.mean(dim=2)  # (B, H, addr_dim)

            # Store one key per batch element
            for b in range(Q_mean.shape[0]):
                self._replay_buffer.append(Q_mean[b].clone())  # (H, addr_dim)

    @torch.no_grad()
    def replay(self, n_replays: int = 1, replay_lr: float | None = None):
        """Sleep replay: re-activate stored Q→V associations from buffer.

        For each stored Q key:
        1. Read current V from trace: V_ret = Q @ T_v
        2. L2-normalize both Q and V_ret to prevent amplification
        3. Re-write to reinforce: T_v = decay * T_v + lr * Q_norm.T @ V_norm

        Normalization is critical: without it, replay computes
        Q.T @ (Q @ T_v) = (Q.T @ Q) @ T_v, which amplifies T_v
        exponentially (trace norm explodes to 10^12 in 5 sessions).

        With L2-normalized Q and V, the update has unit scale regardless
        of trace magnitude — pure directional reinforcement.

        Args:
            n_replays: number of replay passes over the buffer.
            replay_lr: learning rate for replay writes. If None, uses trace_lr.
        """
        if not self._replay_buffer:
            return

        lr = replay_lr if replay_lr is not None else self.trace_lr
        H = self.n_heads

        for _ in range(n_replays):
            for Q_key in self._replay_buffer:
                # Q_key: (H, addr_dim)
                # Normalize Q to prevent amplification
                Q_norm = F.normalize(Q_key, dim=-1)  # (H, addr_dim), unit norm

                # Bank routing
                trace, bank_id = self._resolve_trace(Q_norm)

                # Read: V_ret = Q_norm @ T_v → (H, d_trace)
                V_ret = torch.einsum('hp,hpq->hq', Q_norm, trace)

                # Normalize V_ret to get pure direction
                V_norm = F.normalize(V_ret, dim=-1)  # (H, d_trace), unit norm

                # Pure additive reinforcement (NO decay).
                # Decay during replay would cost 0.99^(n_facts*n_replays) on
                # all stored associations — devastating at high replay counts.
                # Replay is consolidation, not new encoding.
                update = torch.einsum('hp,hq->hpq', Q_norm, V_norm)
                update = update / H

                self._commit_trace(trace + lr * update, bank_id)

    @property
    def replay_buffer_size(self) -> int:
        return len(self._replay_buffer)

    # ── Pattern Completion (CA3 autoassociative) ────────────────────

    def set_auto_mode(self, enabled: bool, completion_alpha: float = 1.0):
        """Enable/disable pattern completion via T_auto.

        When enabled, read adds a completion channel:
        Q → T_auto → Q_corrected → T_v → V (additive to standard read).

        Args:
            enabled: enable completion channel.
            completion_alpha: weight for completion logits relative to
                standard trace (1.0 = equal weight).
        """
        self._auto_enabled = enabled
        self._completion_alpha = completion_alpha

    def compute_q_for_token(self, wte: nn.Embedding,
                            token_id: int) -> torch.Tensor:
        """Compute base Q vector for a single token.

        Used to generate Q→Q pairs for T_auto from template-driven
        paraphrase mappings.

        Args:
            wte: GPT-2 token embedding layer (frozen).
            token_id: single BPE token ID.

        Returns:
            Q: (H, d_trace) one Q vector per head.
        """
        with torch.no_grad():
            tok_embed = wte(torch.tensor(
                [[token_id]], device=self.value_traces.device)).float()
        Q = self.W_proj(self.ln_proj(tok_embed))  # (1, 1, H*d_trace)
        Q = Q.view(self.n_heads, self.d_trace)
        return Q  # (H, d_trace)

    def compute_q_for_tokens(self, wte: nn.Embedding,
                             token_ids: list[int],
                             epsilon: float = 0.1,
                             ) -> torch.Tensor:
        """Composite Q via additive perturbation from remaining tokens.

        Q = Q(first_token) + epsilon * sum(Q(remaining_tokens)).
        First token dominates (preserving cross-entity discrimination),
        while remaining tokens break ties within collision groups
        (entities sharing first BPE token).

        NOTE: Interacts poorly with pattern separation's top-k step.
        Prefer multi-token bank routing instead (write_direct_banked).

        For single-token input, equivalent to compute_q_for_token.

        Args:
            wte: GPT-2 token embedding layer (frozen).
            token_ids: list of BPE token IDs.
            epsilon: perturbation weight for remaining tokens.

        Returns:
            Q: (H, d_trace) one Q vector per head.
        """
        if len(token_ids) == 1:
            return self.compute_q_for_token(wte, token_ids[0])

        # Base Q from first token
        Q = self.compute_q_for_token(wte, token_ids[0])  # (H, d_trace)

        # Additive perturbation from remaining tokens
        for tid in token_ids[1:]:
            Q_i = self.compute_q_for_token(wte, tid)  # (H, d_trace)
            Q = Q + epsilon * Q_i

        return Q

    def _compute_bank_id_from_tokens(self, token_ids: list[int]) -> int:
        """Compute bank ID from multi-token hash.

        Uses all token IDs to determine bank, so entities sharing
        first token but differing in remaining tokens route to
        different banks. Q address is unchanged (first-token only),
        preserving pattern separation compatibility.

        Args:
            token_ids: list of BPE token IDs.

        Returns:
            bank_id: int in [0, n_trace_banks).
        """
        h = hash(tuple(token_ids))
        return h % self.n_trace_banks

    @torch.no_grad()
    def write_direct_banked(self, Q: torch.Tensor, V: torch.Tensor,
                            bank_id: int):
        """Write to a specific bank, bypassing Q-based bank routing.

        Q address is first-token only (PS-compatible). Bank selection
        is determined externally (e.g., from multi-token hash).

        Args:
            Q: (H, d_trace) concept Q vector.
            V: (H, d_trace) entity V vector.
            bank_id: explicit bank to write to.
        """
        H = self.n_heads
        denom = 1 * H

        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2))
            Q_store = Q_exp.squeeze(0).squeeze(1)
        else:
            Q_store = Q

        # Use explicit bank
        if self._bank_traces is not None and bank_id is not None:
            trace = self._bank_traces[bank_id]
        else:
            trace = self.value_traces

        # Erasure
        if self._erase_before_write:
            Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Q_erase = Q_store / Q_norms
            V_old = torch.einsum('hp,hpq->hq', Q_erase, trace)
            erase = torch.einsum('hp,hq->hpq', Q_erase, V_old) / denom
            trace = trace - self._erase_lr * erase

        update = torch.einsum('hp,hq->hpq', Q_store, V) / denom
        new_trace = self.trace_decay * trace + self.trace_lr * update

        if self._bank_traces is not None and bank_id is not None:
            self._bank_traces[bank_id] = new_trace
        else:
            self.value_traces = new_trace

    @torch.no_grad()
    def read_direct_banked(self, Q: torch.Tensor,
                           bank_id: int) -> torch.Tensor:
        """Read from a specific bank, bypassing Q-based bank routing.

        Args:
            Q: (H, d_trace) query vector.
            bank_id: explicit bank to read from.

        Returns:
            retrieved: (d_model,) value projected to model space.
        """
        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(
                Q.unsqueeze(0).unsqueeze(2))
            Q_addr = Q_exp.squeeze(0).squeeze(1)
        else:
            Q_addr = Q

        if self._bank_traces is not None and bank_id is not None:
            trace = self._bank_traces[bank_id]
        else:
            trace = self.value_traces

        V_ret = torch.einsum('hp,hpq->hq', Q_addr, trace)
        V_flat = V_ret.view(1, self.n_heads * self.d_trace)
        return self.W_out(V_flat).squeeze(0)

    def compute_v_for_token(self, wte: nn.Embedding,
                            token_id: int) -> torch.Tensor:
        """Compute V vector for a single token (no LN, matches compute_qv).

        V = W_val(wte(token)) — context-free, no LayerNorm.
        Used with write_direct() for template-free fact storage.

        Args:
            wte: GPT-2 token embedding layer (frozen).
            token_id: single BPE token ID.

        Returns:
            V: (H, d_trace) one V vector per head.
        """
        with torch.no_grad():
            tok_embed = wte(torch.tensor(
                [[token_id]], device=self.value_traces.device)).float()
        V = self.W_val(tok_embed)  # (1, 1, H*d_trace)
        V = V.view(self.n_heads, self.d_trace)
        return V  # (H, d_trace)

    @torch.no_grad()
    def write_auto(self, Q_variant: torch.Tensor,
                   Q_concept: torch.Tensor):
        """Write Q_variant → Q_concept in autoassociative trace.

        Pure additive (no decay) — T_auto stores static template
        knowledge that doesn't change across episodes.

        Args:
            Q_variant: (H, d_trace) paraphrase query key (e.g., Q("I"))
            Q_concept: (H, d_trace) target concept key (e.g., Q("name"))
        """
        # Expand Q_variant for addressing (same space as T_v keys)
        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(
                Q_variant.unsqueeze(0).unsqueeze(2))  # (1,H,1,d) → expanded
            Q_exp = Q_exp.squeeze(0).squeeze(1)  # (H, expanded_dim)
        else:
            Q_exp = Q_variant  # (H, d_trace)

        # Outer product: (H, addr_dim) x (H, d_trace) → (H, addr_dim, d_trace)
        update = torch.einsum('hp,hq->hpq', Q_exp, Q_concept)
        update = update / self.n_heads

        # Pure additive — no decay for static knowledge
        self.autoassociative_traces = (
            self.autoassociative_traces + self.trace_lr * update)

    def read_completion(self, Q: torch.Tensor) -> torch.Tensor:
        """Completion channel: Q → T_auto → Q_corrected → T_v → V.

        Independent from standard read(). Two-step retrieval:
        1. Q_addr → T_auto → Q_corrected (base d_trace space)
        2. Q_corrected → expand → T_v → V

        For aligned queries (Q already correct), completion adds
        redundant signal. For misaligned queries (Q("I") instead of
        Q("name")), completion provides the correct addressing.

        Args:
            Q: (B, H, S, d_trace) query keys (base space)

        Returns:
            output: (B, S, d_model) completion-retrieved values
        """
        B, H, S, d_trace = Q.shape

        # Shift-1 for retrieval: Q_addr[i] = Q[i-1]
        Q_addr = torch.cat([
            torch.zeros_like(Q[:, :, :1, :]),
            Q[:, :, :-1, :],
        ], dim=2)  # (B, H, S, d_trace) — base space

        # Step 1: Q → T_auto → Q_corrected (base d_trace space)
        if self._pattern_sep_enabled:
            Q_addr_exp = self._sparse_expand(Q_addr)  # (B, H, S, expanded)
        else:
            Q_addr_exp = Q_addr

        T_auto = self.autoassociative_traces.unsqueeze(0)  # (1, H, *, d_trace)
        Q_corrected = torch.matmul(
            Q_addr_exp, T_auto)  # (B, H, S, d_trace) — base space

        # Step 2: Q_corrected → expand → T_v → V
        if self._pattern_sep_enabled:
            Q_corrected_exp = self._sparse_expand(Q_corrected)
        else:
            Q_corrected_exp = Q_corrected

        # Bank routing: use last position's corrected Q
        trace_c, _ = self._resolve_trace(Q_corrected_exp[:, :, -1:, :])
        Tv = trace_c.unsqueeze(0)  # (1, H, expanded, d_trace)
        V_ret = torch.matmul(Q_corrected_exp, Tv)  # (B, H, S, d_trace)

        # Reshape and project to d_model
        V_ret = V_ret.permute(0, 2, 1, 3).reshape(B, S, H * d_trace)
        return self.W_out(V_ret)  # (B, S, d_model)

    # ── Learned Storage Gate ────────────────────────────────────────

    def compute_gate(self, wte: nn.Embedding,
                     input_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-position storage gate from token embeddings.

        Args:
            wte: GPT-2 token embedding layer (frozen).
            input_ids: (B, S) token indices.

        Returns:
            gate: (B, S) sigmoid-activated gate values in [0, 1].
        """
        with torch.no_grad():
            tok_embed = wte(input_ids)  # (B, S, d_model)
        logit = self.W_gate(tok_embed).squeeze(-1)  # (B, S)
        gate = torch.sigmoid(logit / self._gate_tau)
        return gate

    def compute_gate_key(self, wte: nn.Embedding,
                         input_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-position concept gate from token embeddings.

        Evaluates whether a token is a "memorable concept" (name, city, etc.)
        vs noise (weather, time, etc.). Used at concept-word positions
        with :-2 slicing in write_dual_gated.

        Args:
            wte: GPT-2 token embedding layer (frozen).
            input_ids: (B, S) token indices.

        Returns:
            gate_key: (B, S) sigmoid-activated gate values in [0, 1].
        """
        with torch.no_grad():
            tok_embed = wte(input_ids)  # (B, S, d_model)
        logit = self.W_gate_key(tok_embed).squeeze(-1)  # (B, S)
        gate_key = torch.sigmoid(logit / self._gate_key_tau)
        return gate_key

    def write_differentiable(self, Q: torch.Tensor, V: torch.Tensor,
                             gate: torch.Tensor) -> torch.Tensor:
        """Differentiable trace write with learned gate (surrogate loss).

        Mirrors write() but:
        - Uses soft gate instead of linking-token mask (WITH gradient)
        - Returns T_v_diff tensor (not stored in self.value_traces)
        - Used during training only for W_gate optimization

        Same shift logic: Q[0..S-3] paired with V[2..S-1],
        gate on middle positions [1..S-2].

        Normalization: gate-adaptive denom = gate_mid.sum() * H.
        Matches hardcoded write() where denom = n_arrows * H.
        When gate is sparse (~1 active position), this gives the same
        scaling as the hardcoded mask.

        Args:
            Q: (B, H, S, d_trace) storage keys (grad through W_proj, ln_proj)
            V: (B, H, S, d_trace) storage values (grad through W_val)
            gate: (B, S) storage gate (grad through W_gate)

        Returns:
            T_v_diff: (H, addr_dim, d_trace) differentiable trace matrix
        """
        B, H, S, d_trace = Q.shape

        addr_dim = self._expanded_dim if self._pattern_sep_enabled \
            else self.d_trace

        if S <= 2:
            return torch.zeros(H, addr_dim, d_trace, device=Q.device)

        # Shift: Q[0..S-3], V[2..S-1], gate on [1..S-2]
        Q_store = Q[:, :, :-2, :]   # (B, H, S-2, d_trace)
        V_store = V[:, :, 2:, :]    # (B, H, S-2, d_trace)

        # Soft gate (replaces linking mask, WITH gradient)
        gate_mid = gate[:, 1:-1]    # (B, S-2)
        gate_exp = gate_mid.unsqueeze(1).unsqueeze(-1)  # (B, 1, S-2, 1)
        Q_store = Q_store * gate_exp  # grad flows through gate

        # Pattern separation (if enabled)
        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        # Gate-adaptive normalization (differentiable):
        # denom = gate_mid.sum() * H, matching write() where denom = n_arrows * H
        denom = gate_mid.sum() * H + 1e-8  # differentiable, no divide-by-zero
        T_v_diff = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        T_v_diff = self.trace_lr * T_v_diff / denom

        return T_v_diff  # (H, addr_dim, d_trace)

    def read_from_trace(self, Q: torch.Tensor,
                        trace: torch.Tensor) -> torch.Tensor:
        """Read from explicit trace tensor (for differentiable training).

        Same as read() but uses provided trace instead of self.value_traces.
        Gradient flows through trace → T_v_diff → gate → W_gate.

        Args:
            Q: (B, H, S, d_trace) query keys
            trace: (H, addr_dim, d_trace) trace matrix (with gradient)

        Returns:
            output: (B, S, d_model) retrieved values projected to model space
        """
        B, H, S, d_trace = Q.shape

        # Shift-1 for retrieval: Q_addr[i] = Q[i-1]
        Q_addr = torch.cat([
            torch.zeros_like(Q[:, :, :1, :]),
            Q[:, :, :-1, :],
        ], dim=2)  # (B, H, S, d_trace)

        # Pattern separation
        if self._pattern_sep_enabled:
            Q_addr = self._sparse_expand(Q_addr)

        # Retrieve: Q_addr @ trace → (B, H, S, d_trace)
        Tv = trace.unsqueeze(0)  # (1, H, addr_dim, d_trace)
        V_ret = torch.matmul(Q_addr, Tv)  # (B, H, S, d_trace)

        # Reshape and project to d_model
        V_ret = V_ret.permute(0, 2, 1, 3).reshape(B, S, H * d_trace)
        return self.W_out(V_ret)  # (B, S, d_model)

    @torch.no_grad()
    def write_gated(self, Q: torch.Tensor, V: torch.Tensor,
                    gate: torch.Tensor):
        """Hebbian trace update using learned gate (inference).

        Like write() but uses soft gate instead of linking-token mask.
        Used during inference after W_gate has been trained.

        Normalization: gate-adaptive denom = gate_mid.sum() * H.
        Matches hardcoded write() where denom = n_arrows * H.

        Args:
            Q: (B, H, S, d_trace) storage keys
            V: (B, H, S, d_trace) storage values
            gate: (B, S) learned gate values
        """
        B, H, S, d_trace = Q.shape

        if S <= 2:
            return

        Q_store = Q[:, :, :-2, :]   # (B, H, S-2, d_trace)
        V_store = V[:, :, 2:, :]    # (B, H, S-2, d_trace)

        # Soft gate (replaces linking mask)
        gate_mid = gate[:, 1:-1]    # (B, S-2)
        gate_exp = gate_mid.unsqueeze(1).unsqueeze(-1)
        Q_store = Q_store * gate_exp

        # Gate-adaptive normalization (matching write_differentiable and write)
        denom = max(gate_mid.sum().item() * H, 1)

        # Pattern separation
        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        # Record Q keys for replay (gate_mid > threshold as mask)
        gate_mask = (gate_mid > 0.01)  # (B, S-2)
        self._record_for_replay(Q_store, gate_mask)

        # Bank routing
        trace, bank_id = self._resolve_trace(Q_store)

        # Hebbian outer product update
        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self._commit_trace(self.trace_decay * trace + self.trace_lr * v_update,
                           bank_id)

    @torch.no_grad()
    def write_dual_gated(self, Q: torch.Tensor, V: torch.Tensor,
                         gate_pos: torch.Tensor, gate_key: torch.Tensor):
        """Hebbian trace update with dual gates (inference).

        Combines position-level gate (fires on linking tokens) with
        concept-level gate (evaluates concept word quality).

        Position alignment:
            gate_pos fires on LINK at position i+1 → gate_pos[:, 1:-1]
            gate_key evaluates concept at position i → gate_key[:, :-2]
            Q_store[i] = Q[i] (concept word)
            V_store[i] = V[i+2] (entity word)

        Args:
            Q: (B, H, S, d_trace) storage keys
            V: (B, H, S, d_trace) storage values
            gate_pos: (B, S) position gate (linking tokens)
            gate_key: (B, S) concept gate (semantic relevance)
        """
        B, H, S, d_trace = Q.shape

        if S <= 2:
            return

        Q_store = Q[:, :, :-2, :]   # (B, H, S-2, d_trace)
        V_store = V[:, :, 2:, :]    # (B, H, S-2, d_trace)

        # Dual gate with position alignment
        gate_pos_mid = gate_pos[:, 1:-1]       # (B, S-2) — LINK positions
        gate_key_concept = gate_key[:, :-2]    # (B, S-2) — concept positions
        combined = gate_pos_mid * gate_key_concept  # (B, S-2)

        combined_exp = combined.unsqueeze(1).unsqueeze(-1)  # (B, 1, S-2, 1)
        Q_store = Q_store * combined_exp

        # Gate-adaptive normalization
        denom = max(combined.sum().item() * H, 1)

        # Pattern separation
        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        # Record Q keys for replay (combined > threshold as mask)
        combined_mask = (combined > 0.001)  # (B, S-2)
        self._record_for_replay(Q_store, combined_mask)

        # Bank routing
        trace, bank_id = self._resolve_trace(Q_store)

        # Hebbian outer product update
        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self._commit_trace(self.trace_decay * trace + self.trace_lr * v_update,
                           bank_id)

    def write_dual_differentiable(self, Q: torch.Tensor, V: torch.Tensor,
                                  gate_pos: torch.Tensor,
                                  gate_key: torch.Tensor) -> torch.Tensor:
        """Differentiable dual-gate write for training gate_key.

        gate_pos is detached (frozen, no gradient).
        gate_key has gradient (trainable).

        Args:
            Q: (B, H, S, d_trace) storage keys
            V: (B, H, S, d_trace) storage values
            gate_pos: (B, S) position gate (DETACHED, no gradient)
            gate_key: (B, S) concept gate (WITH gradient)

        Returns:
            T_v_diff: (H, addr_dim, d_trace) differentiable trace matrix
        """
        B, H, S, d_trace = Q.shape

        addr_dim = self._expanded_dim if self._pattern_sep_enabled \
            else self.d_trace

        if S <= 2:
            return torch.zeros(H, addr_dim, d_trace, device=Q.device)

        Q_store = Q[:, :, :-2, :]   # (B, H, S-2, d_trace)
        V_store = V[:, :, 2:, :]    # (B, H, S-2, d_trace)

        # Dual gate: gate_pos frozen, gate_key trainable
        gate_pos_mid = gate_pos[:, 1:-1].detach()  # (B, S-2) — no gradient
        gate_key_concept = gate_key[:, :-2]        # (B, S-2) — gradient flows
        combined = gate_pos_mid * gate_key_concept  # (B, S-2)

        combined_exp = combined.unsqueeze(1).unsqueeze(-1)
        Q_store = Q_store * combined_exp

        # Pattern separation
        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        # Gate-adaptive normalization (differentiable)
        denom = combined.sum() * H + 1e-8
        T_v_diff = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        T_v_diff = self.trace_lr * T_v_diff / denom

        return T_v_diff  # (H, addr_dim, d_trace)

    def set_gate_mode(self, use_learned_gate: bool):
        """Toggle between learned gate and hardcoded linking mask."""
        self._use_learned_gate = use_learned_gate

    def set_gate_tau(self, tau: float):
        """Set gate temperature. Lower → sharper (more binary)."""
        self._gate_tau = tau

    def set_dual_gate_mode(self, enabled: bool):
        """Enable/disable dual gate (gate_pos + gate_key)."""
        self._use_dual_gate = enabled
        if enabled:
            self._use_learned_gate = True  # dual implies learned

    def set_gate_key_tau(self, tau: float):
        """Set concept gate temperature."""
        self._gate_key_tau = tau

    def set_encoder_mode(self, enabled: bool):
        """Toggle MLP bottleneck encoder for contextual Q.

        When enabled, compute_qv uses W_enc(LN_enc(hidden)) instead
        of W_ctx(LN_ctx(hidden)) for the contextual component.
        MLP bottleneck enables nonlinear feature selection, potentially
        breaking the linear Pareto frontier of W_ctx.
        """
        self._use_encoder = enabled

    # ── Pattern Separation (Dentate Gyrus) ──────────────────────────

    def enable_pattern_separation(self, expand_factor: int, top_k: int,
                                  seed: int = 0):
        """Enable sparse expansion for value trace.

        Frozen random projection d_trace -> d_trace*factor + ReLU + top-k.
        """
        self._pattern_sep_enabled = True
        self._expand_factor = expand_factor
        self._top_k = top_k
        self._expanded_dim = self.d_trace * expand_factor

        gen = torch.Generator()
        gen.manual_seed(seed)
        W = torch.randn(self.d_trace, self._expanded_dim, generator=gen)
        W = W / math.sqrt(self.d_trace)
        self.register_buffer('W_expand', W.to(self.value_traces.device))

        # Resize value traces and autoassociative traces
        self.value_traces = torch.zeros(
            self.n_heads, self._expanded_dim, self.d_trace,
            device=self.value_traces.device, dtype=self.value_traces.dtype)
        self.autoassociative_traces = torch.zeros(
            self.n_heads, self._expanded_dim, self.d_trace,
            device=self.autoassociative_traces.device,
            dtype=self.autoassociative_traces.dtype)

        # Resize bank traces if active
        if self._bank_traces is not None:
            self._bank_traces = torch.zeros(
                self.n_trace_banks, self.n_heads,
                self._expanded_dim, self.d_trace,
                device=self._bank_traces.device,
                dtype=self._bank_traces.dtype)

    def disable_pattern_separation(self):
        """Disable sparse expansion."""
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = self.d_trace

        self.value_traces = torch.zeros(
            self.n_heads, self.d_trace, self.d_trace,
            device=self.value_traces.device, dtype=self.value_traces.dtype)
        self.autoassociative_traces = torch.zeros(
            self.n_heads, self.d_trace, self.d_trace,
            device=self.autoassociative_traces.device,
            dtype=self.autoassociative_traces.dtype)

    # ── Trace Banks (hash-routed memory) ───────────────────────────

    def set_bank_mode(self, n_banks: int):
        """Enable hash-routed trace banks for capacity scaling.

        Routes each fact to one of n_banks separate trace matrices based
        on the sparse Q activation pattern (argmax of expanded dims).
        Each bank accumulates only ~N/n_banks facts, reducing interference.
        Decay is applied only to the written bank, preserving signal in others.

        Requires pattern separation to be enabled first (for meaningful routing).

        Args:
            n_banks: number of trace banks. 1 = disabled (standard single trace).
        """
        self.n_trace_banks = n_banks
        if n_banks > 1:
            exp_dim = self._expanded_dim if self._pattern_sep_enabled \
                else self.d_trace
            self._bank_traces = torch.zeros(
                n_banks, self.n_heads, exp_dim, self.d_trace,
                device=self.value_traces.device,
                dtype=self.value_traces.dtype)
        else:
            self._bank_traces = None

    def _compute_bank_id(self, Q_sparse: torch.Tensor) -> int:
        """Route sparse Q to a trace bank via argmax hash.

        Uses the index of the dominant expanded dimension (summed across
        all batch/head/position dims) as a hash key. Deterministic for
        a given concept token since Q is context-free.

        Args:
            Q_sparse: (..., expanded_dim) sparse Q after expansion.

        Returns:
            bank_id: int in [0, n_trace_banks).
        """
        flat = Q_sparse.reshape(-1, Q_sparse.shape[-1])
        activity = flat.abs().sum(dim=0)  # (expanded_dim,)
        return activity.argmax().item() % self.n_trace_banks

    def _resolve_trace(self, Q_sparse: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, int | None]:
        """Get trace matrix with optional bank routing.

        Returns:
            (trace, bank_id): trace is (H, expanded_dim, d_trace),
            bank_id is int if banking active, else None.
        """
        if self._bank_traces is not None and Q_sparse is not None:
            bank_id = self._compute_bank_id(Q_sparse)
            return self._bank_traces[bank_id], bank_id
        return self.value_traces, None

    def _commit_trace(self, trace: torch.Tensor,
                      bank_id: int | None):
        """Write back trace matrix after update.

        Args:
            trace: updated trace matrix (H, expanded_dim, d_trace).
            bank_id: bank index (int) or None for default trace.
        """
        if bank_id is not None and self._bank_traces is not None:
            self._bank_traces[bank_id] = trace
        else:
            self.value_traces = trace

    def _sparse_expand(self, Q: torch.Tensor) -> torch.Tensor:
        """Project Q through frozen expansion + ReLU + top-k."""
        Q_exp = torch.matmul(Q, self.W_expand)
        Q_exp = F.relu(Q_exp)
        if self._top_k > 0 and self._top_k < self._expanded_dim:
            topk_vals, topk_idx = Q_exp.topk(self._top_k, dim=-1)
            Q_sparse = torch.zeros_like(Q_exp)
            Q_sparse.scatter_(-1, topk_idx, topk_vals)
            return Q_sparse
        return Q_exp


class GPT2WithTrace(nn.Module):
    """Frozen GPT-2 Small with external Hebbian trace memory.

    GPT-2 stays completely frozen. The trace module:
    - Reads GPT-2's token embeddings (wte) for context-free Q/V
    - Stores Hebbian Q→V associations during write phase
    - Retrieves and adds trace-based logit bias during read phase

    Architecture:
        logits = GPT-2(input_ids) + alpha * (W_out(Q_addr @ T_v) @ wte.T)

    Logit injection bypasses the residual stream scale mismatch
    (GPT-2 hidden norms ~3000, trace output ~0.06).
    """

    def __init__(self, n_trace_heads: int = 8, d_trace: int = 64,
                 inject_layer: int = -1, alpha: float = 0.1,
                 trace_lr: float = 0.1, trace_decay: float = 0.99,
                 model_name: str = 'gpt2',
                 torch_dtype=None,
                 device: str | None = None):
        super().__init__()

        # Load and freeze base model (GPT-2, Phi-2, any CausalLM)
        load_kwargs = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs)
        self.base_model.requires_grad_(False)
        self.base_model.eval()

        # Backward compat alias
        self.gpt2 = self.base_model

        # Universal: works for GPT-2 (n_embd), Phi-2 (hidden_size), etc.
        d_model = self.base_model.config.hidden_size

        # Cache embedding layer reference
        self._wte = self.base_model.get_input_embeddings()

        # External trace module (trainable)
        self.trace = HebbianTraceModule(
            d_model=d_model,
            n_heads=n_trace_heads,
            d_trace=d_trace,
            alpha=alpha,
            trace_lr=trace_lr,
            trace_decay=trace_decay,
        )

        # inject_layer: -1 = auto (mid-depth), else explicit
        n_layers = getattr(self.base_model.config, 'n_layer',
                           getattr(self.base_model.config, 'num_hidden_layers', 12))
        if inject_layer < 0:
            self.inject_layer = n_layers // 2
        else:
            self.inject_layer = inject_layer

        if device:
            self.to(device)

    def forward(self, input_ids: torch.Tensor,
                beta: float | torch.Tensor = 0.0,
                context_layer: int = -1) -> torch.Tensor:
        """Forward pass with trace read/write.

        Trace injection directly into logits: retrieved embedding dotted
        with wte gives trace-based logit bias. Bypasses the residual
        stream scale mismatch (GPT-2 hidden norms ~3000, trace ~0.06).

        When beta > 0, Q gets contextual enrichment from GPT-2 hidden states:
            Q = Q_base + beta * Q_ctx
        This helps distinguish "my name" from "Alice's name" at the cost
        of some cross-context stability.

        Args:
            input_ids: (B, S) BPE token indices.
            beta: contextual Q blending weight. Can be:
                  - float: uniform β for all heads (0 = context-free)
                  - Tensor (H,): per-head β (CLS specialization)
            context_layer: which GPT-2 hidden state layer to use (-1 = last).

        Returns:
            logits: (B, S, vocab_size)
        """
        # Check if contextual Q needed (handle both float and tensor beta)
        if isinstance(beta, torch.Tensor):
            _needs_hidden = (beta.abs().sum() > 0).item()
        else:
            _needs_hidden = beta > 0

        # 1. Run GPT-2 forward (need hidden states before computing Q)
        outputs = self.base_model(
            input_ids,
            output_hidden_states=_needs_hidden,
            return_dict=True,
        )
        logits = outputs.logits.float()  # (B, S, vocab_size), ensure fp32

        # 2. Extract hidden states if contextual Q is requested
        hidden = None
        if _needs_hidden and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[context_layer].detach().float()

        # 3. Compute trace Q/V (with optional contextual enrichment)
        trace_Q, trace_V = self.trace.compute_qv(
            self._wte, input_ids,
            hidden_states=hidden, beta=beta)

        # 4. Write phase: update trace if in write mode
        if self.trace._update_trace:
            if self.trace._use_dual_gate:
                gate_pos = self.trace.compute_gate(
                    self._wte, input_ids)
                gate_key = self.trace.compute_gate_key(
                    self._wte, input_ids)
                self.trace.write_dual_gated(
                    trace_Q, trace_V, gate_pos, gate_key)
            elif self.trace._use_learned_gate:
                gate = self.trace.compute_gate(
                    self._wte, input_ids)
                self.trace.write_gated(trace_Q, trace_V, gate)
            else:
                self.trace.write(trace_Q, trace_V, input_ids)

        # 5. Read phase: add trace-based logit bias
        if self.trace._use_trace:
            retrieved = self.trace.read(trace_Q)  # (B, S, d_model)
            # Project to logit space via wte: retrieved @ wte.T → (B, S, vocab)
            wte_weight = self._wte.weight.float()  # ensure fp32 match
            trace_logits = torch.matmul(retrieved, wte_weight.T)
            logits = logits + self.trace.alpha * trace_logits

            # Completion channel: Q → T_auto → Q_corrected → T_v → V
            if self.trace._auto_enabled:
                completion = self.trace.read_completion(trace_Q)
                completion_logits = torch.matmul(completion, wte_weight.T)
                logits = logits + (self.trace.alpha
                                   * self.trace._completion_alpha
                                   * completion_logits)

        return logits

    # ── Auto-regressive generation ─────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 4,
        restrict_first_to: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
        beta: float = 0.0,
        context_layer: int = -1,
    ) -> torch.Tensor:
        """Auto-regressive generation with trace-augmented logits.

        For multi-token entity prediction:
        1. First token: trace-augmented logits, optionally restricted
           to known entity first-tokens.
        2. Subsequent tokens: GPT-2 LM + trace (trace near-zero for
           non-concept tokens due to pattern separation).

        Args:
            input_ids: (B, S) prompt token IDs
            max_new_tokens: maximum tokens to generate
            restrict_first_to: if set, restrict first token to these IDs
            stop_token_ids: stop generation on any of these tokens
            beta: contextual Q blending weight
            context_layer: hidden state layer for contextual Q

        Returns:
            generated_ids: (B, n_generated) generated token IDs
        """
        generated = []

        for step in range(max_new_tokens):
            logits = self.forward(
                input_ids, beta=beta, context_layer=context_layer)
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            # Restrict first token to known entity set
            if step == 0 and restrict_first_to is not None:
                mask = torch.full_like(next_logits, float('-inf'))
                mask[:, restrict_first_to] = 0
                next_logits = next_logits + mask

            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(next_token)

            # Check stop condition
            if stop_token_ids is not None:
                if next_token.squeeze().item() in stop_token_ids:
                    break

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return torch.cat(generated, dim=1)  # (B, n_generated)

    # ── Delegation to trace module ─────────────────────────────────

    def reset_traces(self):
        self.trace.reset_traces()

    def set_trace_mode(self, use: bool = False, update: bool = False):
        self.trace.set_trace_mode(use=use, update=update)

    def set_linking_token_ids(self, token_ids: list[int]):
        self.trace.set_linking_token_ids(token_ids)

    def set_erase_mode(self, enabled: bool, erase_lr: float = 1.0):
        self.trace.set_erase_mode(enabled, erase_lr)

    def enable_pattern_separation(self, expand_factor: int, top_k: int,
                                  seed: int = 0):
        self.trace.enable_pattern_separation(expand_factor, top_k, seed=seed)

    def disable_pattern_separation(self):
        self.trace.disable_pattern_separation()

    def set_bank_mode(self, n_banks: int):
        self.trace.set_bank_mode(n_banks)

    def set_gate_mode(self, use_learned_gate: bool):
        self.trace.set_gate_mode(use_learned_gate)

    def set_dual_gate_mode(self, enabled: bool):
        self.trace.set_dual_gate_mode(enabled)

    def set_replay_mode(self, enabled: bool):
        self.trace.set_replay_mode(enabled)

    def replay(self, n_replays: int = 1, replay_lr: float | None = None):
        self.trace.replay(n_replays=n_replays, replay_lr=replay_lr)

    def clear_replay_buffer(self):
        self.trace.clear_replay_buffer()

    def set_encoder_mode(self, enabled: bool):
        self.trace.set_encoder_mode(enabled)

    def set_auto_mode(self, enabled: bool, completion_alpha: float = 1.0):
        self.trace.set_auto_mode(enabled, completion_alpha)

    def write_auto_pairs(self, pairs: list[tuple[int, int]]):
        """Write template-driven Q→Q pairs to T_auto.

        Args:
            pairs: list of (variant_token_id, concept_token_id) pairs.
                   e.g., [(id("I"), id("name")), (id("called"), id("name"))]
        """
        for var_id, concept_id in pairs:
            Q_var = self.trace.compute_q_for_token(self._wte, var_id)
            Q_con = self.trace.compute_q_for_token(self._wte, concept_id)
            self.trace.write_auto(Q_var, Q_con)

    def write_fact_direct(self, concept_token_id: int,
                          entity_token_id: int):
        """Write a single concept→entity fact to trace.

        Equivalent to writing "My {concept} is {entity}." via template,
        but without requiring full forward pass or template structure.

        Args:
            concept_token_id: BPE ID of the concept word (e.g., " name").
            entity_token_id: BPE ID of the entity (e.g., " John").
        """
        Q = self.trace.compute_q_for_token(self._wte, concept_token_id)
        V = self.trace.compute_v_for_token(self._wte, entity_token_id)
        self.trace.write_direct(Q, V)

    def write_fact_direct_multi(self, concept_token_ids: list[int],
                                entity_token_id: int):
        """Write fact with multi-token composite Q addressing.

        Uses position-weighted combination of all concept tokens
        (weights [1.0, 0.5, 0.25, ...]) so entities sharing a prefix
        get distinct trace addresses.

        Args:
            concept_token_ids: list of BPE IDs for multi-token concept.
            entity_token_id: BPE ID of the entity value.
        """
        Q = self.trace.compute_q_for_tokens(self._wte, concept_token_ids)
        V = self.trace.compute_v_for_token(self._wte, entity_token_id)
        self.trace.write_direct(Q, V)

    @torch.no_grad()
    def retrieve_direct(self, token_id: int,
                        candidate_ids: list[int]) -> int:
        """Direct trace retrieval: Q(token) → T_v → logits → argmax.

        Pure trace lookup — no GPT-2 forward pass, no alpha scaling.
        This is a different regime than standard retrieval which blends
        GPT-2 logits + alpha * trace_logits. Here trace IS the signal.

        For multi-hop chains this can be preferable: no LM prior
        biasing toward frequent tokens.

        Args:
            token_id: BPE token whose Q addresses the trace.
            candidate_ids: restrict prediction to these token IDs.

        Returns:
            predicted token ID (from candidate_ids).
        """
        Q = self.trace.compute_q_for_token(self._wte, token_id)
        retrieved = self.trace.read_direct(Q)  # (d_model,)
        wte_weight = self._wte.weight.float()
        logits = torch.matmul(retrieved, wte_weight.T)  # (vocab_size,)
        cand_logits = logits[candidate_ids]
        return candidate_ids[cand_logits.argmax().item()]

    @torch.no_grad()
    def retrieve_direct_multi(self, token_ids: list[int],
                              candidate_ids: list[int]) -> int:
        """Direct trace retrieval with composite Q from multiple tokens.

        Same as retrieve_direct but uses position-weighted composite Q
        for multi-token entity addressing.

        Args:
            token_ids: list of BPE token IDs for composite Q.
            candidate_ids: restrict prediction to these token IDs.

        Returns:
            predicted token ID (from candidate_ids).
        """
        Q = self.trace.compute_q_for_tokens(self._wte, token_ids)
        retrieved = self.trace.read_direct(Q)  # (d_model,)
        wte_weight = self._wte.weight.float()
        logits = torch.matmul(retrieved, wte_weight.T)  # (vocab_size,)
        cand_logits = logits[candidate_ids]
        return candidate_ids[cand_logits.argmax().item()]

    @torch.no_grad()
    def retrieve_direct_best_bank(self, token_id: int,
                                   candidate_ids: list[int]) -> int:
        """Retrieve from the bank with highest confidence (all-bank scan).

        Reads Q(token_id) from every bank and returns the answer with
        the highest logit. No external bank routing needed — confidence
        drives the selection. Cost: n_banks reads instead of 1.

        Args:
            token_id: BPE token for Q address.
            candidate_ids: restrict prediction to these token IDs.

        Returns:
            predicted token ID (from candidate_ids).
        """
        Q = self.trace.compute_q_for_token(self._wte, token_id)
        wte_weight = self._wte.weight.float()
        cand_t = torch.tensor(candidate_ids,
                               device=self.trace.value_traces.device)

        best_logit = float('-inf')
        best_pred = candidate_ids[0]

        n_banks = self.trace.n_trace_banks
        if n_banks <= 1 or self.trace._bank_traces is None:
            # No banks: standard retrieval
            return self.retrieve_direct(token_id, candidate_ids)

        for bank_id in range(n_banks):
            retrieved = self.trace.read_direct_banked(Q, bank_id)
            logits = torch.matmul(retrieved, wte_weight.T)
            cand_logits = logits[cand_t]
            max_logit = cand_logits.max().item()
            if max_logit > best_logit:
                best_logit = max_logit
                best_pred = candidate_ids[cand_logits.argmax().item()]

        return best_pred

    def write_fact_direct_banked(self, concept_token_id: int,
                                 entity_token_id: int,
                                 bank_token_ids: list[int]):
        """Write fact using first-token Q but multi-token bank routing.

        Q address uses concept_token_id only (PS-compatible).
        Bank selection uses hash(bank_token_ids) — entities sharing
        first token but differing in remaining tokens route to
        different banks, eliminating collision interference.

        Args:
            concept_token_id: BPE ID for Q address (first token).
            entity_token_id: BPE ID for V (answer).
            bank_token_ids: all entity token IDs for bank routing.
        """
        Q = self.trace.compute_q_for_token(self._wte, concept_token_id)
        V = self.trace.compute_v_for_token(self._wte, entity_token_id)
        bank_id = self.trace._compute_bank_id_from_tokens(bank_token_ids)
        self.trace.write_direct_banked(Q, V, bank_id)

    @torch.no_grad()
    def retrieve_direct_banked(self, token_id: int,
                               candidate_ids: list[int],
                               bank_token_ids: list[int]) -> int:
        """Direct trace retrieval from multi-token-routed bank.

        Q address uses token_id only. Bank selection uses
        hash(bank_token_ids).

        Args:
            token_id: BPE token for Q address (first token).
            candidate_ids: restrict prediction to these token IDs.
            bank_token_ids: all entity token IDs for bank routing.

        Returns:
            predicted token ID (from candidate_ids).
        """
        Q = self.trace.compute_q_for_token(self._wte, token_id)
        bank_id = self.trace._compute_bank_id_from_tokens(bank_token_ids)
        retrieved = self.trace.read_direct_banked(Q, bank_id)
        wte_weight = self._wte.weight.float()
        logits = torch.matmul(retrieved, wte_weight.T)
        cand_logits = logits[candidate_ids]
        return candidate_ids[cand_logits.argmax().item()]
