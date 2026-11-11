"""Equivariant Relational Graph Transformer.

One set of weights processes *any* compiled puzzle graph:

* no parameter is indexed by facet, colour or action identity -- colours are embedded through the
  positional encoding of their home class, relations through pooled structure of their edges and
  actions are scored by pooling node states along their own transition edges.  The network is
  therefore equivariant to facet relabelling (graph isomorphism), colour renaming and action
  reordering, and its action space grows for free when macros inject new relations.
* LapPE enters through a sign-invariant encoder (``phi(v) + phi(-v)``); all pairwise structure
  (relations, diffusion kernels) enters as an additive attention bias.
* Message passing uses only ``index_add_`` / gather / dense attention, which run natively on MPS.

Heads: a categorical distribution over integer cost-to-go (whose mean is the scalar heuristic
``h(s)``) and a policy prior over the puzzle's actions.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_compiler import KERNEL_DIM, NUM_REL_TYPES, REL_FEAT_DIM, CompiledGraph


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    d_msg: int = 64
    lap_k: int = 16
    rw_k: int = 12
    max_dist: int = 47          # distance bins 0..max_dist
    context_dim: int = 0
    dropout: float = 0.0

    @property
    def num_bins(self) -> int:
        return self.max_dist + 1

    def to_dict(self) -> dict:
        return asdict(self)


class ModelOutput(NamedTuple):
    value: torch.Tensor          # [B] expected cost-to-go
    dist_logits: torch.Tensor    # [B, D]
    policy_logits: torch.Tensor  # [B, A]
    pooled: torch.Tensor         # [B, 2*d]


class StaticEncoding(NamedTuple):
    x_pos: torch.Tensor          # [N, d]
    class_emb: torch.Tensor      # [C, d]
    rel_emb: torch.Tensor        # [A+1, d]
    edge_emb: torch.Tensor       # [E, d]
    attn_bias: list              # per layer [1, H, N, N]
    edge_msg: list               # per layer [E, d_msg]
    home_feat: torch.Tensor      # [N, C, d]


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class SignInvariantLapPE(nn.Module):
    """``rho(sum_k [phi(v_k, lambda_k) + phi(-v_k, lambda_k)])`` -- invariant to eigenvector sign flips."""

    def __init__(self, lap_k: int, d_out: int, d_phi: int = 16):
        super().__init__()
        self.phi = mlp(2, 32, d_phi)
        self.rho = nn.Linear(lap_k * d_phi, d_out)
        self.lap_k = lap_k

    def forward(self, pe: torch.Tensor, eigval: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, k = pe.shape
        if k < self.lap_k:
            pad = self.lap_k - k
            pe = F.pad(pe, (0, pad)); eigval = F.pad(eigval, (0, pad)); mask = F.pad(mask, (0, pad))
        elif k > self.lap_k:
            pe, eigval, mask = pe[:, : self.lap_k], eigval[: self.lap_k], mask[: self.lap_k]
        lam = eigval.unsqueeze(0).expand(n, -1)
        pos = self.phi(torch.stack([pe, lam], dim=-1))
        neg = self.phi(torch.stack([-pe, lam], dim=-1))
        feat = (pos + neg) * mask.view(1, -1, 1)
        return self.rho(feat.reshape(n, -1))


class RGTLayer(nn.Module):
    """Relational message passing + biased global attention + feed-forward (pre-norm)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, dm = cfg.d_model, cfg.d_msg
        self.n_heads = cfg.n_heads
        self.norm_m = nn.LayerNorm(d)
        self.msg_in = nn.Linear(d, dm)
        self.msg_out = nn.Linear(dm, d)
        self.norm_a = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.norm_f = nn.LayerNorm(d)
        self.ff = mlp(d, cfg.d_ff, d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, g: CompiledGraph, edge_msg: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, N, d = h.shape
        # (1) relational message passing along generator / macro / binding edges
        x = self.msg_in(self.norm_m(h))                                   # [B, N, dm]
        m = F.gelu(x[:, g.edge_src] + edge_msg.unsqueeze(0))              # [B, E, dm]
        agg = torch.zeros(B, N, m.shape[-1], dtype=h.dtype, device=h.device).index_add_(1, g.edge_dst, m)
        h = h + self.drop(self.msg_out(agg * g.in_degree_inv.view(1, N, 1)))
        # (2) global attention with relational + spectral-kernel bias
        q, k, v = self.qkv(self.norm_a(h)).view(B, N, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        h = h + self.drop(self.attn_out(att.transpose(1, 2).reshape(B, N, d)))
        # (3) feed-forward
        return h + self.drop(self.ff(self.norm_f(h)))


class RelationalGraphTransformer(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d, dm = cfg.d_model, cfg.d_msg
        assert d % cfg.n_heads == 0
        # static (per-topology) encoders
        self.lap_enc = SignInvariantLapPE(cfg.lap_k, d)
        self.rw_enc = nn.Linear(cfg.rw_k, d)
        self.piece_emb = nn.Embedding(16, d)
        self.pos_mlp = mlp(d, d, d)
        self.class_proj = nn.Linear(d, d)
        self.rel_type_emb = nn.Embedding(NUM_REL_TYPES, d)
        self.rel_feat = nn.Linear(REL_FEAT_DIM, d)
        self.rel_src = nn.Linear(d, d)
        self.rel_dst = nn.Linear(d, d)
        self.rel_mlp = mlp(d, d, d)
        self.kernel_mlp = mlp(KERNEL_DIM, 32, 32)
        self.kernel_heads = nn.ModuleList([nn.Linear(32, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.edge_heads = nn.ModuleList([nn.Linear(d, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.edge_msgs = nn.ModuleList([nn.Linear(d, dm) for _ in range(cfg.n_layers)])
        self.home_proj = nn.Linear(KERNEL_DIM, d)
        # dynamic input
        self.match_emb = nn.Embedding(2, d)
        self.in_norm = nn.LayerNorm(d)
        self.context_proj = nn.Linear(cfg.context_dim, d) if cfg.context_dim > 0 else None
        self.layers = nn.ModuleList([RGTLayer(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(d)
        # heads
        self.value_head = mlp(2 * d, d, cfg.num_bins)
        self.pol_src = nn.Linear(d, dm)
        self.pol_dst = nn.Linear(d, dm)
        self.pol_rel = nn.Linear(d, dm)
        self.pol_glob = nn.Linear(2 * d, dm)
        self.pol_out = mlp(dm, dm, 1)
        self.register_buffer("bins", torch.arange(cfg.num_bins, dtype=torch.float32), persistent=False)
        self._static_cache: dict[int, tuple[CompiledGraph, StaticEncoding]] = {}

    # -- static encoding (depends on topology + weights, not on the state) -------------------
    def encode_static(self, g: CompiledGraph) -> StaticEncoding:
        cfg = self.cfg
        N = g.n
        rw = g.rw_pe
        if rw.shape[1] < cfg.rw_k:
            rw = F.pad(rw, (0, cfg.rw_k - rw.shape[1]))
        x = self.lap_enc(g.lap_pe, g.lap_eigval, g.lap_mask) + self.rw_enc(rw[:, : cfg.rw_k]) + self.piece_emb(g.piece_size)
        x_pos = self.pos_mlp(x)                                                    # [N, d]
        class_emb = self.class_proj(g.class_matrix @ x_pos)                        # [C, d]
        R = g.rel_type.shape[0]
        zeros = torch.zeros(R, x_pos.shape[1], dtype=x_pos.dtype, device=x_pos.device)
        src_mean = zeros.index_add(0, g.edge_rel, x_pos[g.edge_src]) * g.rel_edge_count_inv.unsqueeze(1)
        dst_mean = zeros.index_add(0, g.edge_rel, x_pos[g.edge_dst]) * g.rel_edge_count_inv.unsqueeze(1)
        rel = self.rel_mlp(self.rel_type_emb(g.rel_type) + self.rel_feat(g.rel_feat)
                           + self.rel_src(src_mean) + self.rel_dst(dst_mean))       # [A+1, d]
        edge_emb = rel[g.edge_rel]                                                 # [E, d]
        kern = self.kernel_mlp(g.kernel)                                           # [N, N, 32]
        biases, msgs = [], []
        for layer in range(cfg.n_layers):
            kb = self.kernel_heads[layer](kern).reshape(N * N, cfg.n_heads)
            kb = kb.index_add(0, g.edge_pair_index, self.edge_heads[layer](edge_emb))
            biases.append(kb.view(N, N, cfg.n_heads).permute(2, 0, 1).unsqueeze(0))
            msgs.append(self.edge_msgs[layer](edge_emb))
        home_feat = self.home_proj(g.home_kernel)                                  # [N, C, d]
        return StaticEncoding(x_pos, class_emb, rel, edge_emb, biases, msgs, home_feat)

    def train(self, mode: bool = True):
        self._static_cache.clear()
        return super().train(mode)

    def _static(self, g: CompiledGraph) -> StaticEncoding:
        if self.training or torch.is_grad_enabled():
            return self.encode_static(g)
        hit = self._static_cache.get(id(g))
        if hit is None or hit[0] is not g:
            if len(self._static_cache) > 32:
                self._static_cache.clear()
            hit = (g, self.encode_static(g))
            self._static_cache[id(g)] = hit
        return hit[1]

    def invalidate_cache(self) -> None:
        self._static_cache.clear()

    # -- forward --------------------------------------------------------------------------------
    def forward(self, g: CompiledGraph, colors: torch.Tensor, legal: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None) -> ModelOutput:
        """``colors``: ``[B, N]`` long colour state; ``legal``: optional ``[B, A]`` action mask."""
        st = self._static(g)
        B, N = colors.shape
        match = (colors == g.solved_colors.unsqueeze(0)).long()
        C = st.home_feat.shape[1]
        flat = torch.arange(N, device=colors.device).unsqueeze(0) * C + colors
        home = st.home_feat.reshape(N * C, -1)[flat]                               # [B, N, d]
        h = st.x_pos.unsqueeze(0) + st.class_emb[colors] + self.match_emb(match) + home
        if self.context_proj is not None:
            if context is None:
                context = torch.zeros(B, self.cfg.context_dim, dtype=h.dtype, device=h.device)
            h = h + self.context_proj(context).unsqueeze(1)
        h = self.in_norm(h)
        for layer, bias, msg in zip(self.layers, st.attn_bias, st.edge_msg):
            h = layer(h, g, msg, bias)
        h = self.out_norm(h)

        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)                 # [B, 2d]
        dist_logits = self.value_head(pooled)
        value = (F.softmax(dist_logits, dim=-1) * self.bins).sum(dim=-1)

        E = g.num_action_edges
        src, dst, rel = g.edge_src[:E], g.edge_dst[:E], g.edge_rel[:E]
        e = F.gelu(self.pol_src(h)[:, src] + self.pol_dst(h)[:, dst] + self.pol_rel(st.rel_emb)[rel].unsqueeze(0))
        A = g.num_actions
        act = torch.zeros(B, A, e.shape[-1], dtype=h.dtype, device=h.device).index_add_(1, rel, e)
        act = act * g.rel_edge_count_inv[:A].view(1, A, 1)
        act = act + self.pol_rel(st.rel_emb[:A]).unsqueeze(0) + self.pol_glob(pooled).unsqueeze(1)
        policy_logits = self.pol_out(act).squeeze(-1)
        if legal is not None:
            policy_logits = policy_logits.masked_fill(~legal, -1e9)
        return ModelOutput(value, dist_logits, policy_logits, pooled)

    @torch.no_grad()
    def evaluate(self, g: CompiledGraph, colors: torch.Tensor, legal: Optional[torch.Tensor] = None,
                 chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunked inference -> ``(value [B], dist_probs [B, D], policy_logits [B, A])``."""
        values, dists, pols = [], [], []
        for s in range(0, colors.shape[0], chunk):
            out = self.forward(g, colors[s:s + chunk], None if legal is None else legal[s:s + chunk])
            values.append(out.value); dists.append(F.softmax(out.dist_logits, dim=-1)); pols.append(out.policy_logits)
        if not values:
            dev = colors.device
            return (torch.zeros(0, device=dev), torch.zeros(0, self.cfg.num_bins, device=dev),
                    torch.zeros(0, g.num_actions, device=dev))
        return torch.cat(values), torch.cat(dists), torch.cat(pols)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def auto_chunk(graph: CompiledGraph, cfg: ModelConfig, budget_mb: float = 768.0) -> int:
    """Largest inference chunk whose message tensor ``[chunk, E, d_msg]`` fits in ``budget_mb``."""
    per_state = max(graph.num_edges, graph.n * graph.n // 4, 1) * cfg.d_msg * 4 * 3
    return int(max(16, min(8192, budget_mb * 2 ** 20 // per_state)))


def save_checkpoint(path: str, model: RelationalGraphTransformer, extra: Optional[dict] = None) -> None:
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[RelationalGraphTransformer, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = RelationalGraphTransformer(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    return model.to(device), blob.get("extra", {})
