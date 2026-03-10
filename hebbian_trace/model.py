"""Hebbian Trace Memory for frozen pretrained transformers.

External memory module (~1.1M parameters) that attaches to a frozen LLM
and provides persistent cross-session fact storage, paraphrase resolution,
and multi-hop reasoning via bio-inspired Hebbian trace learning.

Architecture:
    Q = W_proj(LN(wte(token)))           -- context-free storage keys
    V = W_val(wte(token))                -- context-free values
    T_v (H, d_addr, d_trace)             -- heteroassociative trace (Q->V)
    T_auto (H, d_addr, d_trace)          -- autoassociative trace (Q->Q)
    Retrieved: W_out(Q_addr @ T_v)       -- projected back to d_model
    Injection: logits += alpha * (retrieved @ wte.T)

Biological analogies:
    Pattern separation (dentate gyrus):  sparse random expansion of Q
    Dual gates (ACh modulation):         learned fact/filler filtering
    Reconsolidation erasure:             selective overwrite for updates
    Autoassociative trace (CA3):         paraphrase -> concept mapping
    Multi-hop chains:                    entity-as-concept addressing
    Replay (sharp-wave ripples):         sleep-phase trace strengthening
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


class HebbianTraceModule(nn.Module):
    """External Hebbian trace memory for pretrained transformers.

    Stores Q->V associations via Hebbian outer-product learning.
    Retrieves via Q-addressed lookup with shift-1 offset.
    Supports pattern separation, dual gating, and reconsolidation erasure.
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

        # Context-free key projection: wte -> Q
        self.W_proj = nn.Linear(d_model, n_heads * d_trace, bias=False)
        self.ln_proj = nn.LayerNorm(d_model)

        # Value projection: wte -> V
        self.W_val = nn.Linear(d_model, n_heads * d_trace, bias=False)

        # Output projection: retrieved -> d_model
        self.W_out = nn.Linear(n_heads * d_trace, d_model, bias=False)

        # Hebbian trace matrix: stores Q->V associations
        self.register_buffer(
            'value_traces', torch.zeros(n_heads, d_trace, d_trace))

        # ACh modulation flags
        self._use_trace = False
        self._update_trace = False

        # Linking tokens (BPE IDs for "is", "in", "at", etc.)
        self.linking_token_ids: list[int] | None = None

        # Position-level gate: fires on linking tokens
        self.W_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        self._use_learned_gate = False
        self._gate_tau = 1.0

        # Concept-level gate: evaluates semantic relevance
        self.W_gate_key = nn.Linear(d_model, 1)
        nn.init.zeros_(self.W_gate_key.weight)
        nn.init.constant_(self.W_gate_key.bias, -2.0)
        self._use_dual_gate = False
        self._gate_key_tau = 1.0

        # Pattern separation (dentate gyrus)
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = d_trace

        # Reconsolidation erasure
        self._erase_before_write = False
        self._erase_lr = 1.0

        # Trace banks (hash-routed memory for capacity scaling)
        self.n_trace_banks = 1
        self._bank_traces: torch.Tensor | None = None

        # Autoassociative trace (T_auto) for pattern completion
        self.register_buffer(
            'autoassociative_traces', torch.zeros(n_heads, d_trace, d_trace))
        self._auto_enabled = False
        self._completion_alpha = 0.3

        # Replay buffer
        self._replay_enabled = False
        self._replay_buffer: list[torch.Tensor] = []

        self._init_orthogonal()

    def _init_orthogonal(self):
        """Initialize W_val and W_out as approximate inverses via QR."""
        total_dim = self.n_heads * self.d_trace
        Q, _ = torch.linalg.qr(torch.randn(self.d_model, self.d_model))
        self.W_val.weight.data = Q[:total_dim, :]
        self.W_out.weight.data = Q[:total_dim, :].T

    def compute_qv(self, wte: nn.Embedding,
                   input_ids: torch.Tensor,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context-free Q and V from token embeddings.

        Q = W_proj(LN(wte(token))), V = W_val(wte(token)).
        Both are deterministic functions of token identity only --
        no positional encoding, no contextual information.

        Args:
            wte: token embedding layer (frozen).
            input_ids: (B, S) token indices.

        Returns:
            Q: (B, H, S, d_trace) storage/retrieval keys
            V: (B, H, S, d_trace) values
        """
        B, S = input_ids.shape
        with torch.no_grad():
            tok_embed = wte(input_ids)

        Q = self.W_proj(self.ln_proj(tok_embed))
        Q = Q.view(B, S, self.n_heads, self.d_trace).permute(0, 2, 1, 3)

        V = self.W_val(tok_embed)
        V = V.view(B, S, self.n_heads, self.d_trace).permute(0, 2, 1, 3)

        return Q, V

    def compute_q_for_token(self, wte: nn.Embedding,
                             token_id: int) -> torch.Tensor:
        """Compute base Q vector for a single token.

        Context-free: same token always produces the same Q.
        Used for direct write/read operations.

        Args:
            wte: token embedding layer.
            token_id: BPE token ID.

        Returns:
            Q: (H, d_trace)
        """
        with torch.no_grad():
            tok_embed = wte(torch.tensor(
                [[token_id]], device=self.value_traces.device)).float()
        Q = self.W_proj(self.ln_proj(tok_embed))
        Q = Q.view(self.n_heads, self.d_trace)
        return Q

    def compute_v_for_token(self, wte: nn.Embedding,
                             token_id: int) -> torch.Tensor:
        """Compute V vector for a single token.

        Args:
            wte: token embedding layer.
            token_id: BPE token ID.

        Returns:
            V: (H, d_trace)
        """
        with torch.no_grad():
            tok_embed = wte(torch.tensor(
                [[token_id]], device=self.value_traces.device)).float()
        V = self.W_val(tok_embed)
        V = V.view(self.n_heads, self.d_trace)
        return V

    @torch.no_grad()
    def write_direct(self, Q: torch.Tensor, V: torch.Tensor):
        """Write a single Q->V association directly to trace.

        Bypasses template/shift machinery. Used by write_fact_direct
        and for multi-hop chain link storage.

        Args:
            Q: (H, d_trace) storage key
            V: (H, d_trace) storage value
        """
        H = self.n_heads
        denom = 1 * H

        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(Q.unsqueeze(0).unsqueeze(2))
            Q_store = Q_exp.squeeze(0).squeeze(1)
        else:
            Q_store = Q

        trace, bank_id = self._resolve_trace(Q_store)

        if self._erase_before_write:
            Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Q_erase = Q_store / Q_norms
            V_old = torch.einsum('hp,hpq->hq', Q_erase, trace)
            erase = torch.einsum('hp,hq->hpq', Q_erase, V_old) / denom
            trace = trace - self._erase_lr * erase

        update = torch.einsum('hp,hq->hpq', Q_store, V) / denom
        self._commit_trace(self.trace_decay * trace + self.trace_lr * update,
                           bank_id)

    @torch.no_grad()
    def read_direct(self, Q: torch.Tensor) -> torch.Tensor:
        """Read from T_v using a single pre-computed Q, no shift.

        Unlike read() which applies shift-1 for sequence-level retrieval,
        this does direct Q @ T_v for single-token lookups (multi-hop chains).

        Args:
            Q: (H, d_trace) query vector from compute_q_for_token

        Returns:
            retrieved: (d_model,) value projected to model space
        """
        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(Q.unsqueeze(0).unsqueeze(2))
            Q_addr = Q_exp.squeeze(0).squeeze(1)
        else:
            Q_addr = Q

        trace, _ = self._resolve_trace(Q_addr)
        V_ret = torch.einsum('hp,hpq->hq', Q_addr, trace)
        V_flat = V_ret.view(1, self.n_heads * self.d_trace)
        return self.W_out(V_flat).squeeze(0)

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
        """Hebbian trace update with linking-token mask.

        Shift-1 offset: Q at concept position (link-1), V at entity
        position (link+1). Only positions adjacent to linking tokens
        participate in the outer product.

        Args:
            Q: (B, H, S, d_trace) storage keys
            V: (B, H, S, d_trace) storage values
            token_ids: (B, S) token indices for linking mask
        """
        B, H, S, d_trace = Q.shape
        if S <= 2:
            return

        Q_store = Q[:, :, :-2, :]
        V_store = V[:, :, 2:, :]

        if self.linking_token_ids is not None:
            arrow_mask = self._make_linking_mask(token_ids[:, 1:-1])
            mask_expanded = arrow_mask.unsqueeze(1).unsqueeze(-1).float()
            Q_store = Q_store * mask_expanded
            n_arrows = arrow_mask.sum().item()
            denom = max(n_arrows * H, 1)
        else:
            denom = B * (S - 2)

        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        trace, bank_id = self._resolve_trace(Q_store)

        if self._erase_before_write:
            Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Q_erase = Q_store / Q_norms
            V_old = torch.einsum(
                'bhip,hpq->bhiq', Q_erase, trace)
            erase = torch.einsum(
                'bhip,bhiq->hpq', Q_erase, V_old) / denom
            trace = trace - self._erase_lr * erase

        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self._commit_trace(self.trace_decay * trace + self.trace_lr * v_update,
                           bank_id)

    def read(self, Q: torch.Tensor) -> torch.Tensor:
        """Retrieve from trace: V_ret = Q_addr @ T_v.

        Shift-1 retrieval: Q_addr at position i uses Q from position i-1,
        matching the concept-word Q used during storage.

        Args:
            Q: (B, H, S, d_trace) query keys

        Returns:
            output: (B, S, d_model) retrieved values in model space
        """
        B, H, S, d_trace = Q.shape

        Q_addr = torch.cat([
            torch.zeros_like(Q[:, :, :1, :]),
            Q[:, :, :-1, :],
        ], dim=2)

        if self._pattern_sep_enabled:
            Q_addr = self._sparse_expand(Q_addr)

        # Bank routing: use last position's Q (prediction target)
        trace, _ = self._resolve_trace(Q_addr[:, :, -1:, :])
        Tv = trace.unsqueeze(0)
        V_ret = torch.matmul(Q_addr, Tv)

        V_ret = V_ret.permute(0, 2, 1, 3).reshape(B, S, H * d_trace)
        return self.W_out(V_ret)

    def set_erase_mode(self, enabled: bool, erase_lr: float = 1.0):
        """Toggle reconsolidation erasure for fact updates.

        When enabled, old Q->V associations are erased before writing new
        ones. Should be ON during updates only, OFF during initial encoding.

        Args:
            enabled: enable erase-before-write.
            erase_lr: erasure strength.
        """
        self._erase_before_write = enabled
        self._erase_lr = erase_lr

    def set_trace_mode(self, use: bool = False, update: bool = False):
        """Control trace read/write (ACh modulation).

        Write phase: use=False, update=True  (encode without retrieval)
        Read phase:  use=True,  update=False (retrieve without encoding)
        """
        self._use_trace = use
        self._update_trace = update

    def set_linking_token_ids(self, token_ids: list[int]):
        """Set BPE token IDs that trigger trace storage."""
        self.linking_token_ids = token_ids

    # -- Learned Storage Gates --

    def compute_gate(self, wte: nn.Embedding,
                     input_ids: torch.Tensor) -> torch.Tensor:
        """Compute position-level gate (linking token detector).

        Returns:
            gate: (B, S) values in [0, 1].
        """
        with torch.no_grad():
            tok_embed = wte(input_ids)
        logit = self.W_gate(tok_embed).squeeze(-1)
        return torch.sigmoid(logit / self._gate_tau)

    def compute_gate_key(self, wte: nn.Embedding,
                         input_ids: torch.Tensor) -> torch.Tensor:
        """Compute concept-level gate (semantic relevance).

        Evaluates whether a token is a storable concept (name, city)
        vs noise (weather, time).

        Returns:
            gate_key: (B, S) values in [0, 1].
        """
        with torch.no_grad():
            tok_embed = wte(input_ids)
        logit = self.W_gate_key(tok_embed).squeeze(-1)
        return torch.sigmoid(logit / self._gate_key_tau)

    @torch.no_grad()
    def write_gated(self, Q: torch.Tensor, V: torch.Tensor,
                    gate: torch.Tensor):
        """Hebbian write with learned position gate."""
        B, H, S, d_trace = Q.shape
        if S <= 2:
            return

        Q_store = Q[:, :, :-2, :]
        V_store = V[:, :, 2:, :]

        gate_mid = gate[:, 1:-1]
        gate_exp = gate_mid.unsqueeze(1).unsqueeze(-1)
        Q_store = Q_store * gate_exp

        denom = max(gate_mid.sum().item() * H, 1)

        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self.value_traces = (self.trace_decay * self.value_traces
                             + self.trace_lr * v_update)

    @torch.no_grad()
    def write_dual_gated(self, Q: torch.Tensor, V: torch.Tensor,
                         gate_pos: torch.Tensor, gate_key: torch.Tensor):
        """Hebbian write with dual gates (position + concept).

        Position gate fires on linking tokens (WHERE to store).
        Concept gate evaluates the concept word (WHETHER to store).
        Combined: only facts with relevant concepts get stored.

        Args:
            Q: (B, H, S, d_trace) storage keys
            V: (B, H, S, d_trace) storage values
            gate_pos: (B, S) position gate
            gate_key: (B, S) concept gate
        """
        B, H, S, d_trace = Q.shape
        if S <= 2:
            return

        Q_store = Q[:, :, :-2, :]
        V_store = V[:, :, 2:, :]

        gate_pos_mid = gate_pos[:, 1:-1]
        gate_key_concept = gate_key[:, :-2]
        combined = gate_pos_mid * gate_key_concept

        combined_exp = combined.unsqueeze(1).unsqueeze(-1)
        Q_store = Q_store * combined_exp

        denom = max(combined.sum().item() * H, 1)

        if self._pattern_sep_enabled:
            Q_store = self._sparse_expand(Q_store)

        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self.value_traces = (self.trace_decay * self.value_traces
                             + self.trace_lr * v_update)

    def set_gate_mode(self, use_learned_gate: bool):
        """Toggle between learned gate and hardcoded linking mask."""
        self._use_learned_gate = use_learned_gate

    def set_gate_tau(self, tau: float):
        """Set position gate temperature (lower = sharper)."""
        self._gate_tau = tau

    def set_dual_gate_mode(self, enabled: bool):
        """Enable dual gate mode (position + concept gating)."""
        self._use_dual_gate = enabled
        if enabled:
            self._use_learned_gate = True

    def set_gate_key_tau(self, tau: float):
        """Set concept gate temperature."""
        self._gate_key_tau = tau

    # -- Pattern Separation (Dentate Gyrus) --

    def enable_pattern_separation(self, expand_factor: int, top_k: int,
                                  seed: int = 0):
        """Enable sparse expansion for value trace addressing.

        Frozen random projection Q -> Q_expanded (d_trace * factor dims),
        followed by ReLU and top-k sparsification. Creates near-unique
        addressing codes that reduce interference between stored facts.

        Args:
            expand_factor: expansion ratio (e.g. 8 -> 8x dimensions)
            top_k: number of active dimensions after sparsification
            seed: random seed for reproducible expansion matrix
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

        self.value_traces = torch.zeros(
            self.n_heads, self._expanded_dim, self.d_trace,
            device=self.value_traces.device, dtype=self.value_traces.dtype)

        # T_auto also uses expanded Q addressing
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
        """Disable sparse expansion, restore standard trace shape."""
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = self.d_trace

        self.value_traces = torch.zeros(
            self.n_heads, self.d_trace, self.d_trace,
            device=self.value_traces.device, dtype=self.value_traces.dtype)

        # Reset T_auto to match
        self.autoassociative_traces = torch.zeros(
            self.n_heads, self.d_trace, self.d_trace,
            device=self.autoassociative_traces.device,
            dtype=self.autoassociative_traces.dtype)

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

    # -- Trace Banks (Hash-Routed Memory) --

    def set_bank_mode(self, n_banks: int):
        """Enable hash-routed trace banks for capacity scaling.

        Routes each fact to one of n_banks separate trace matrices based
        on the sparse Q activation pattern (argmax of expanded dims).
        Each bank accumulates only ~N/n_banks facts, reducing interference.
        Decay is applied only to the written bank, preserving signal in others.

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
        all batch/head/position dims) as a hash key.

        Args:
            Q_sparse: (..., expanded_dim) sparse Q after expansion.

        Returns:
            bank_id: int in [0, n_trace_banks).
        """
        flat = Q_sparse.reshape(-1, Q_sparse.shape[-1])
        activity = flat.abs().sum(dim=0)
        return activity.argmax().item() % self.n_trace_banks

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
        """Write back trace matrix after update."""
        if bank_id is not None and self._bank_traces is not None:
            self._bank_traces[bank_id] = trace
        else:
            self.value_traces = trace

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
            Q_exp = self._sparse_expand(Q.unsqueeze(0).unsqueeze(2))
            Q_store = Q_exp.squeeze(0).squeeze(1)
        else:
            Q_store = Q

        if self._bank_traces is not None and bank_id is not None:
            trace = self._bank_traces[bank_id]
        else:
            trace = self.value_traces

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
            Q_exp = self._sparse_expand(Q.unsqueeze(0).unsqueeze(2))
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

    # -- Autoassociative Trace (CA3 Pattern Completion) --

    def set_auto_mode(self, enabled: bool, completion_alpha: float = 0.3):
        """Enable/disable pattern completion via T_auto.

        Args:
            enabled: whether completion channel is active during read.
            completion_alpha: strength of completion channel (optimal: 0.3).
        """
        self._auto_enabled = enabled
        self._completion_alpha = completion_alpha

    @torch.no_grad()
    def write_auto(self, Q_variant: torch.Tensor, Q_concept: torch.Tensor):
        """Write Q_variant -> Q_concept mapping in T_auto.

        Stores a template pair: variant query word maps to canonical
        concept word. For example, Q("I") -> Q("name"), Q("home") -> Q("live").

        Args:
            Q_variant: (H, d_trace) source Q
            Q_concept: (H, d_trace) target Q
        """
        if self._pattern_sep_enabled:
            Q_exp = self._sparse_expand(Q_variant.unsqueeze(0).unsqueeze(2))
            Q_exp = Q_exp.squeeze(0).squeeze(1)
        else:
            Q_exp = Q_variant

        update = torch.einsum('hp,hq->hpq', Q_exp, Q_concept)
        update = update / self.n_heads
        self.autoassociative_traces = (
            self.autoassociative_traces + self.trace_lr * update)

    def read_completion(self, Q: torch.Tensor) -> torch.Tensor:
        """Completion channel: Q -> T_auto -> Q_corrected -> T_v -> V.

        Two-step retrieval for paraphrase resolution.
        Operates in parallel with standard read(), results summed at logit level.

        Args:
            Q: (B, H, S, d_trace) query keys

        Returns:
            output: (B, S, d_model) completion-retrieved values
        """
        B, H, S, d_trace = Q.shape

        Q_addr = torch.cat([
            torch.zeros_like(Q[:, :, :1, :]),
            Q[:, :, :-1, :],
        ], dim=2)

        if self._pattern_sep_enabled:
            Q_addr_exp = self._sparse_expand(Q_addr)
        else:
            Q_addr_exp = Q_addr

        T_auto = self.autoassociative_traces.unsqueeze(0)
        Q_corrected = torch.matmul(Q_addr_exp, T_auto)

        if self._pattern_sep_enabled:
            Q_corrected_exp = self._sparse_expand(Q_corrected)
        else:
            Q_corrected_exp = Q_corrected

        Tv = self.value_traces.unsqueeze(0)
        V_ret = torch.matmul(Q_corrected_exp, Tv)

        V_ret = V_ret.permute(0, 2, 1, 3).reshape(B, S, H * d_trace)
        return self.W_out(V_ret)

    def reset_auto_traces(self):
        """Zero autoassociative traces only."""
        self.autoassociative_traces.zero_()

    # -- Replay (Hippocampal Sharp-Wave Ripples) --

    def set_replay_mode(self, enabled: bool):
        """Enable/disable replay buffer recording during writes."""
        self._replay_enabled = enabled
        if not enabled:
            self._replay_buffer.clear()

    def clear_replay_buffer(self):
        """Clear replay buffer."""
        self._replay_buffer.clear()

    @torch.no_grad()
    def replay(self, n_replays: int = 1):
        """Sleep-phase replay: re-activate stored Q->V associations.

        Normalized replay prevents amplification: both Q and V are
        L2-normalized before re-write. No decay during replay.

        Args:
            n_replays: number of replay iterations over the buffer.
        """
        if not self._replay_buffer:
            return

        H = self.n_heads
        for _ in range(n_replays):
            for Q_key in self._replay_buffer:
                Q_norm = F.normalize(Q_key, dim=-1)
                V_ret = torch.einsum('hp,hpq->hq', Q_norm, self.value_traces)
                V_norm = F.normalize(V_ret, dim=-1)
                update = torch.einsum('hp,hq->hpq', Q_norm, V_norm)
                update = update / H
                self.value_traces = self.value_traces + self.trace_lr * update

    @property
    def replay_buffer_size(self) -> int:
        return len(self._replay_buffer)

    def reset_traces(self):
        """Zero all trace matrices (including all banks)."""
        self.value_traces.zero_()
        self.autoassociative_traces.zero_()
        if self._bank_traces is not None:
            self._bank_traces.zero_()
        self._replay_buffer.clear()


class GPT2WithTrace(nn.Module):
    """Frozen GPT-2 with external Hebbian trace memory.

    GPT-2 remains completely frozen (124M params). The trace module
    (~1.1M params) provides persistent cross-session memory via
    logit-space injection:

        logits = GPT2(input) + alpha * (W_out(Q @ T_v) @ wte.T)

    Logit injection bypasses the residual stream scale mismatch
    (GPT-2 hidden norms ~3000 vs trace output ~0.06).
    """

    def __init__(self, n_trace_heads: int = 8, d_trace: int = 64,
                 alpha: float = 0.1, trace_lr: float = 0.1,
                 trace_decay: float = 0.99,
                 model_name: str = 'gpt2',
                 device: str | None = None):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2.requires_grad_(False)
        self.gpt2.eval()

        d_model = self.gpt2.config.n_embd

        self.trace = HebbianTraceModule(
            d_model=d_model,
            n_heads=n_trace_heads,
            d_trace=d_trace,
            alpha=alpha,
            trace_lr=trace_lr,
            trace_decay=trace_decay,
        )

        if device:
            self.to(device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: GPT-2 logits + trace-based logit bias.

        During write phase (update=True): stores Q->V in trace.
        During read phase (use=True): adds trace retrieval to logits.

        Args:
            input_ids: (B, S) BPE token indices.

        Returns:
            logits: (B, S, vocab_size)
        """
        outputs = self.gpt2(input_ids, return_dict=True)
        logits = outputs.logits

        wte = self.gpt2.transformer.wte
        trace_Q, trace_V = self.trace.compute_qv(wte, input_ids)

        if self.trace._update_trace:
            if self.trace._use_dual_gate:
                gate_pos = self.trace.compute_gate(wte, input_ids)
                gate_key = self.trace.compute_gate_key(wte, input_ids)
                self.trace.write_dual_gated(
                    trace_Q, trace_V, gate_pos, gate_key)
            elif self.trace._use_learned_gate:
                gate = self.trace.compute_gate(wte, input_ids)
                self.trace.write_gated(trace_Q, trace_V, gate)
            else:
                self.trace.write(trace_Q, trace_V, input_ids)

        if self.trace._use_trace:
            retrieved = self.trace.read(trace_Q)
            trace_logits = torch.matmul(retrieved, wte.weight.T)
            logits = logits + self.trace.alpha * trace_logits

            # Pattern completion channel (T_auto)
            if self.trace._auto_enabled:
                completed = self.trace.read_completion(trace_Q)
                comp_logits = torch.matmul(completed, wte.weight.T)
                logits = logits + (self.trace.alpha
                                   * self.trace._completion_alpha
                                   * comp_logits)

        return logits

    # -- Delegation --

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

    def set_gate_mode(self, use_learned_gate: bool):
        self.trace.set_gate_mode(use_learned_gate)

    def set_dual_gate_mode(self, enabled: bool):
        self.trace.set_dual_gate_mode(enabled)

    def set_auto_mode(self, enabled: bool, completion_alpha: float = 0.3):
        self.trace.set_auto_mode(enabled, completion_alpha)

    def set_bank_mode(self, n_banks: int):
        self.trace.set_bank_mode(n_banks)

    # -- Direct Write/Read (bypasses template machinery) --

    @property
    def _wte(self) -> nn.Embedding:
        """Access frozen GPT-2 token embeddings."""
        return self.gpt2.transformer.wte

    @torch.no_grad()
    def write_fact_direct(self, concept_token_id: int,
                          entity_token_id: int):
        """Write a single concept->entity fact to trace.

        Bypasses template/linking-mask machinery. Used for free-text
        extraction pipeline and multi-hop chain storage.

        Args:
            concept_token_id: BPE token ID of the concept word (Q address)
            entity_token_id: BPE token ID of the entity (V value)
        """
        Q = self.trace.compute_q_for_token(self._wte, concept_token_id)
        V = self.trace.compute_v_for_token(self._wte, entity_token_id)
        self.trace.write_direct(Q, V)

    @torch.no_grad()
    def retrieve_direct(self, token_id: int,
                        candidate_ids: list[int]) -> int:
        """Direct trace retrieval: Q(token) -> T_v -> logits -> argmax.

        Pure trace lookup -- no GPT-2 forward pass, no alpha scaling.
        Used for multi-hop chains where each hop reads from trace directly.

        Args:
            token_id: BPE token whose Q addresses the trace
            candidate_ids: restrict prediction to these token IDs

        Returns:
            predicted token ID (from candidate_ids)
        """
        Q = self.trace.compute_q_for_token(self._wte, token_id)
        retrieved = self.trace.read_direct(Q)
        wte_weight = self._wte.weight.float()
        logits = torch.matmul(retrieved, wte_weight.T)
        cand_logits = logits[candidate_ids]
        return candidate_ids[cand_logits.argmax().item()]

    @torch.no_grad()
    def retrieve_direct_best_bank(self, token_id: int,
                                   candidate_ids: list[int]) -> int:
        """Retrieve from the bank with highest confidence (all-bank scan).

        Reads Q(token_id) from every bank and returns the answer with
        the highest logit. No external bank routing needed — confidence
        drives the selection. Cost: n_banks reads instead of 1.

        Useful when the correct bank is unknown at retrieval time
        (e.g., multi-hop chains where bridge entity tokens are predicted,
        not given). Falls back to standard retrieve_direct when banks
        are disabled.

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

    @torch.no_grad()
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

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 4,
                 restrict_first_to: list[int] | None = None,
                 stop_token_ids: list[int] | None = None,
                 ) -> torch.Tensor:
        """Auto-regressive generation with trace-augmented logits.

        Used for multi-token entity completion: trace retrieves first
        token, GPT-2's LM head completes subsequent tokens.

        Args:
            input_ids: (B, S) prompt tokens.
            max_new_tokens: maximum tokens to generate.
            restrict_first_to: restrict first token to these IDs.
            stop_token_ids: stop generation at these tokens.

        Returns:
            generated: (B, n_generated) new token IDs.
        """
        generated = []

        for step in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :]

            if step == 0 and restrict_first_to is not None:
                mask = torch.full_like(next_logits, float('-inf'))
                mask[:, restrict_first_to] = 0
                next_logits = next_logits + mask

            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)

            if stop_token_ids is not None:
                if next_token.squeeze().item() in stop_token_ids:
                    break

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return torch.cat(generated, dim=1)
