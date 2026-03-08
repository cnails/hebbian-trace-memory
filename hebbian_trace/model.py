"""Hebbian Trace Memory for frozen pretrained transformers.

External memory module (~1.1M parameters) that attaches to a frozen GPT-2
and provides persistent cross-session fact storage via bio-inspired
Hebbian trace learning.

Architecture:
    Q = W_proj(LN(wte(token)))           -- context-free storage keys
    V = W_val(wte(token))                -- context-free values
    Trace: T_v (H, d_addr, d_trace)      -- Hebbian association matrix
    Retrieved: W_out(Q_addr @ T_v)       -- projected back to d_model
    Injection: logits += alpha * (retrieved @ wte.T)

Biological analogies:
    Pattern separation (dentate gyrus):  sparse random expansion of Q
    Dual gates (ACh modulation):         learned fact/filler filtering
    Reconsolidation erasure:             selective overwrite for updates
    Linking-token mask:                  hippocampal indexing
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

        if self._erase_before_write:
            Q_norms = Q_store.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Q_erase = Q_store / Q_norms
            V_old = torch.einsum(
                'bhip,hpq->bhiq', Q_erase, self.value_traces)
            erase = torch.einsum(
                'bhip,bhiq->hpq', Q_erase, V_old) / denom
            self.value_traces = self.value_traces - self._erase_lr * erase

        v_update = torch.einsum('bhip,bhiq->hpq', Q_store, V_store)
        v_update = v_update / denom

        self.value_traces = (self.trace_decay * self.value_traces
                             + self.trace_lr * v_update)

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

        Tv = self.value_traces.unsqueeze(0)
        V_ret = torch.matmul(Q_addr, Tv)

        V_ret = V_ret.permute(0, 2, 1, 3).reshape(B, S, H * d_trace)
        return self.W_out(V_ret)

    def reset_traces(self):
        """Zero all trace matrices."""
        self.value_traces.zero_()

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

    def disable_pattern_separation(self):
        """Disable sparse expansion, restore standard trace shape."""
        self._pattern_sep_enabled = False
        self._expand_factor = 1
        self._top_k = 0
        self._expanded_dim = self.d_trace

        self.value_traces = torch.zeros(
            self.n_heads, self.d_trace, self.d_trace,
            device=self.value_traces.device, dtype=self.value_traces.dtype)

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
