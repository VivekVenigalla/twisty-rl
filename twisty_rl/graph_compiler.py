"""Compile a puzzle into a directed multi-relational graph ``G = (V, E, R)``.

* **Nodes** are facets.  Static node features are Laplacian eigenvector positional encodings
  (LapPE) and random-walk positional encodings (RWPE) of the puzzle's transition topology.
* **Relations**: one relation per action (generator, inverse generator or macro) whose edges are
  the directed permutation cycles it induces (``src -> dst`` means "the facet at ``src`` travels to
  ``dst``"), plus one structural *binding* relation joining facets of the same rigid piece.
* Pairwise structural kernels (diffusion kernels, shortest-path proximity, same-piece indicator)
  are provided as a basis-invariant attention bias.

Nothing here depends on the puzzle family, so every puzzle -- and every macro-extended action
space -- compiles to the same tensor layout.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch

from .schema import ACTION_MACRO, Puzzle

REL_GENERATOR, REL_INVERSE, REL_MACRO, REL_BINDING = 0, 1, 2, 3
NUM_REL_TYPES = 4
KERNEL_DIM = 8
REL_FEAT_DIM = 3
DIFFUSION_TIMES = (0.5, 1.0, 2.0, 4.0)


@dataclass
class CompiledGraph:
    puzzle: Puzzle
    n: int
    num_actions: int
    num_colors: int
    solved_colors: torch.Tensor      # [N] long
    piece_size: torch.Tensor         # [N] long
    edge_src: torch.Tensor           # [E] long (action edges first, then binding edges)
    edge_dst: torch.Tensor           # [E] long
    edge_rel: torch.Tensor           # [E] long in [0, A]  (A == binding relation)
    num_action_edges: int
    rel_type: torch.Tensor           # [A+1] long
    rel_feat: torch.Tensor           # [A+1, 3] float
    in_degree_inv: torch.Tensor      # [N] float
    rel_edge_count_inv: torch.Tensor  # [A+1] float
    lap_pe: torch.Tensor             # [N, K] float
    lap_eigval: torch.Tensor         # [K] float
    lap_mask: torch.Tensor           # [K] float (1 for real eigenvectors)
    rw_pe: torch.Tensor              # [N, R] float
    kernel: torch.Tensor             # [N, N, KERNEL_DIM] float
    class_matrix: torch.Tensor       # [C, N] float, rows sum to one
    home_kernel: torch.Tensor        # [N, C, KERNEL_DIM] float
    edge_pair_index: torch.Tensor    # [E] long: dst * N + src

    def to(self, device: torch.device | str) -> "CompiledGraph":
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kwargs[f.name] = v.to(device) if isinstance(v, torch.Tensor) else v
        return CompiledGraph(**kwargs)

    @property
    def device(self) -> torch.device:
        return self.solved_colors.device

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])


def _action_edges(puzzle: Puzzle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = puzzle.n
    src_all, dst_all, rel_all = [], [], []
    ar = np.arange(n)
    for a in range(puzzle.num_actions):
        tab = puzzle.table[a]                                   # [K, N]
        mask = puzzle.legal[a][:, None] & (tab != ar[None, :])
        src = tab[mask]
        dst = np.broadcast_to(ar[None, :], tab.shape)[mask]
        keys = np.unique(src * n + dst)
        src_all.append(keys // n)
        dst_all.append(keys % n)
        rel_all.append(np.full(len(keys), a))
    return np.concatenate(src_all), np.concatenate(dst_all), np.concatenate(rel_all)


def _binding_edges(puzzle: Puzzle) -> tuple[np.ndarray, np.ndarray]:
    src, dst = [], []
    for piece in range(puzzle.num_pieces):
        facets = np.nonzero(puzzle.piece_of == piece)[0]
        if len(facets) < 2:
            continue
        for i in facets:
            for j in facets:
                if i != j:
                    src.append(i)
                    dst.append(j)
    return np.asarray(src, dtype=np.intp), np.asarray(dst, dtype=np.intp)


def topology_adjacency(puzzle: Puzzle) -> np.ndarray:
    """Symmetric static topology: primitive transition edges plus piece bindings."""
    n = puzzle.n
    W = np.zeros((n, n))
    src, dst, rel = _action_edges(puzzle)
    prim = rel < puzzle.num_primitive
    W[src[prim], dst[prim]] = 1.0
    bs, bd = _binding_edges(puzzle)
    if len(bs):
        W[bs, bd] = 1.0
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def spectral_encodings(W: np.ndarray, lap_k: int, rw_k: int):
    """LapPE (eigenvectors of the normalised Laplacian with non-zero eigenvalue), RWPE and diffusion kernels."""
    n = W.shape[0]
    deg = W.sum(axis=1)
    dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    L = np.eye(n) - dinv_sqrt[:, None] * W * dinv_sqrt[None, :]
    L[deg == 0, deg == 0] = 0.0
    evals, evecs = np.linalg.eigh(L)
    keep = np.nonzero(evals > 1e-8)[0][:lap_k]
    pe = np.zeros((n, lap_k))
    lam = np.zeros(lap_k)
    mask = np.zeros(lap_k)
    pe[:, : len(keep)] = evecs[:, keep] * np.sqrt(n)
    lam[: len(keep)] = evals[keep]
    mask[: len(keep)] = 1.0

    P = np.where(deg[:, None] > 0, W / np.maximum(deg[:, None], 1e-12), np.eye(n))
    rw = np.zeros((n, rw_k))
    Pk = np.eye(n)
    for k in range(rw_k):
        Pk = Pk @ P
        rw[:, k] = np.diag(Pk)

    diff = []
    for t in DIFFUSION_TIMES:
        diff.append((evecs * np.exp(-t * evals)[None, :]) @ evecs.T)
    return pe, lam, mask, rw, np.stack(diff, axis=-1)


def shortest_paths(W: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    reach = np.eye(n, dtype=bool)
    adj = W > 0
    frontier = np.eye(n, dtype=bool)
    d = 0
    while frontier.any():
        d += 1
        nxt = (frontier.astype(np.float32) @ adj.astype(np.float32) > 0) & ~reach
        dist[nxt] = d
        reach |= nxt
        frontier = nxt
    return dist


def compile_graph(puzzle: Puzzle, lap_k: int = 16, rw_k: int = 12,
                  device: Optional[torch.device | str] = None) -> CompiledGraph:
    n, A = puzzle.n, puzzle.num_actions
    src, dst, rel = _action_edges(puzzle)
    bs, bd = _binding_edges(puzzle)
    num_action_edges = len(src)
    e_src = np.concatenate([src, bs]).astype(np.int64)
    e_dst = np.concatenate([dst, bd]).astype(np.int64)
    e_rel = np.concatenate([rel, np.full(len(bs), A)]).astype(np.int64)

    rel_type = np.concatenate([puzzle.action_kind, [REL_BINDING]]).astype(np.int64)
    rel_feat = np.zeros((A + 1, REL_FEAT_DIM), dtype=np.float32)
    k0 = puzzle.anchor if puzzle.anchor is not None else 0
    for a in range(A):
        legal_k = np.nonzero(puzzle.legal[a])[0]
        perm = puzzle.table[a, k0 if puzzle.legal[a, k0] else int(legal_k[0])]
        # cycle order of the action (2 for conditional slides), support fraction, cost
        seen = np.zeros(n, dtype=bool)
        order = 1
        for i in range(n):
            if seen[i] or perm[i] == i:
                continue
            length, j = 0, i
            while not seen[j]:
                seen[j] = True
                j = int(perm[j])
                length += 1
            order = int(np.lcm(order, length))
        rel_feat[a] = (np.log(order) / np.log(12.0), (rel == a).sum() / max(n, 1), puzzle.cost[a] / (1.0 + puzzle.cost[a]))
    rel_feat[A] = (0.0, len(bs) / max(n * n, 1), 0.0)

    W = topology_adjacency(puzzle)
    pe, lam, mask, rw, diff = spectral_encodings(W, lap_k, rw_k)
    sp = shortest_paths(W)
    same_piece = (puzzle.piece_of[:, None] == puzzle.piece_of[None, :]).astype(np.float64)
    kernel = np.concatenate([
        diff * np.sqrt(n),                                   # 4 diffusion scales
        (1.0 / (1.0 + sp))[..., None],                       # proximity (0 when disconnected)
        (sp == 1)[..., None].astype(np.float64),             # adjacency
        same_piece[..., None],
        np.eye(n)[..., None],
    ], axis=-1)
    assert kernel.shape[-1] == KERNEL_DIM

    C = puzzle.num_colors
    cls = np.zeros((C, n))
    cls[puzzle.solved_colors, np.arange(n)] = 1.0
    cls /= np.maximum(cls.sum(axis=1, keepdims=True), 1.0)
    home_kernel = np.einsum("ijk,cj->ick", kernel, cls)       # [N, C, KERNEL_DIM]

    in_deg = np.bincount(e_dst, minlength=n).astype(np.float32)
    rel_cnt = np.bincount(e_rel, minlength=A + 1).astype(np.float32)
    piece_sizes = np.bincount(puzzle.piece_of, minlength=puzzle.num_pieces)[puzzle.piece_of]

    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    i64 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.long)
    graph = CompiledGraph(
        puzzle=puzzle, n=n, num_actions=A, num_colors=C,
        solved_colors=i64(puzzle.solved_colors), piece_size=i64(np.minimum(piece_sizes, 15)),
        edge_src=i64(e_src), edge_dst=i64(e_dst), edge_rel=i64(e_rel), num_action_edges=num_action_edges,
        rel_type=i64(rel_type), rel_feat=f32(rel_feat),
        in_degree_inv=f32(1.0 / np.maximum(in_deg, 1.0)), rel_edge_count_inv=f32(1.0 / np.maximum(rel_cnt, 1.0)),
        lap_pe=f32(pe), lap_eigval=f32(lam), lap_mask=f32(mask), rw_pe=f32(rw),
        kernel=f32(kernel), class_matrix=f32(cls), home_kernel=f32(home_kernel),
        edge_pair_index=i64(e_dst * n + e_src),
    )
    return graph.to(device) if device is not None else graph


def macro_edge_count(graph: CompiledGraph) -> int:
    """Number of shortcut transition edges injected by macro-actions."""
    return int((graph.rel_type[graph.edge_rel] == ACTION_MACRO).sum().item())
