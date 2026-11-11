"""High-throughput batched graph search.

* :class:`ZobristHasher` -- additive Zobrist hashing of colour states on the device, optionally
  canonicalised over whole-puzzle symmetries (the minimum hash over the symmetry orbit).
* :class:`TranspositionTable` -- lock-free (no per-entry locking, deterministic single-writer per
  slot, always-replace) open-addressed table living in device tensors.
* :class:`PatternDatabase` -- exact coset-space distance tables derived from the stabiliser chain,
  used as admissible heuristics and as lower bounds when bootstrapping the value network.
* :class:`BatchedWeightedAStar` and :class:`BatchedBeamSearch` -- pop / expand ``B`` nodes per
  iteration with one vectorised heuristic evaluation; macro-actions are ordinary actions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .env import TensorPuzzle
from .group_theory import PuzzleGroup, SearchBudgetExceeded
from .schema import Puzzle, invert_perm
from .utils import Telemetry, sync_device

HASH_BITS = 62
HASH_MASK = (1 << HASH_BITS) - 1


# ----------------------------------------------------------------------------------
# Hashing and transposition table
# ----------------------------------------------------------------------------------
class ZobristHasher:
    def __init__(self, tp: TensorPuzzle, seed: int = 0x5EED, use_symmetry: bool = False, max_symmetries: int = 48):
        self.tp = tp
        n, C = tp.n, tp.puzzle.num_colors
        rng = np.random.default_rng(seed)
        Z = rng.integers(1, 1 << HASH_BITS, size=(n, C), dtype=np.int64)
        tables = [Z]
        self.num_symmetries = 1
        if use_symmetry:
            for rho, pi in tp.puzzle.valid_symmetries()[1:max_symmetries]:
                # hash_sym(C) = sum_i Z[i, pi[C[rho[i]]]] = sum_j Z[rho^-1[j], pi[C[j]]]
                tables.append(Z[invert_perm(rho)][:, pi])
            self.num_symmetries = len(tables)
        self.table = torch.as_tensor(np.stack(tables).reshape(len(tables), n * C), device=tp.device)   # [S, N*C]
        self.offsets = torch.arange(n, device=tp.device) * C

    def hash(self, P: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """62-bit non-zero hash per state (canonical over symmetries when enabled)."""
        out = []
        for s in range(0, P.shape[0], chunk):
            idx = self.offsets.unsqueeze(0) + self.tp.colors(P[s:s + chunk])          # [b, N]
            if self.num_symmetries == 1:
                h = self.table[0][idx].sum(dim=-1)
            else:
                h = (self.table[:, idx].sum(dim=-1) & HASH_MASK).amin(dim=0)
            h = h & HASH_MASK
            out.append(torch.where(h == 0, torch.ones_like(h), h))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=P.device)


class TranspositionTable:
    """Open-addressed table ``slot = top bits of hash`` holding the best known ``g`` per state."""

    def __init__(self, log2_size: int = 20, device: torch.device | str = "cpu"):
        self.log2 = log2_size
        self.shift = HASH_BITS - log2_size
        self.keys = torch.zeros(1 << log2_size, dtype=torch.long, device=device)
        self.g = torch.full((1 << log2_size,), float("inf"), dtype=torch.float32, device=device)
        self.stored = 0
        self.hits = 0

    def filter_insert(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Deduplicate a batch and merge it into the table.

        Returns a mask of entries that are new or strictly better than everything known.  Within the
        batch the minimum-``g`` copy of every hash survives; one deterministic writer is chosen
        per slot, so no entry can ever be torn between two writers.
        """
        n = h.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=h.device)
        o1 = torch.argsort(g, stable=True)
        o2 = torch.argsort(h[o1], stable=True)
        perm = o1[o2]                                    # lexicographic by (hash, g)
        hs, gs = h[perm], g[perm]
        first = torch.ones(n, dtype=torch.bool, device=h.device)
        first[1:] = hs[1:] != hs[:-1]
        slot = hs >> self.shift
        writer = torch.ones(n, dtype=torch.bool, device=h.device)
        writer[1:] = slot[1:] != slot[:-1]
        known = (self.keys[slot] == hs) & (self.g[slot] <= gs)
        keep_sorted = first & ~known
        w = torch.nonzero(writer & keep_sorted).flatten()
        self.keys[slot[w]] = hs[w]
        self.g[slot[w]] = gs[w]
        self.stored += int(w.shape[0])
        self.hits += int(known.sum().item())
        keep = torch.zeros(n, dtype=torch.bool, device=h.device)
        keep[perm] = keep_sorted
        return keep

    def clear(self) -> None:
        self.keys.zero_()
        self.g.fill_(float("inf"))
        self.stored = self.hits = 0


# ----------------------------------------------------------------------------------
# Pattern databases
# ----------------------------------------------------------------------------------
class PatternDatabase:
    """Exact distances in the abstraction that tracks only the locations of ``tracked`` facets.

    The abstract goal set is the orbit of the home locations under the colour-stabiliser, so the
    table is an admissible heuristic even with indistinguishable facets.  For conditional puzzles
    the anchor must be the first tracked facet.
    """

    def __init__(self, puzzle: Puzzle, tracked: Sequence[int], max_entries: int = 2_000_000,
                 goal_generators: Sequence[np.ndarray] = (), device: torch.device | str = "cpu"):
        self.puzzle = puzzle
        self.tracked = [int(t) for t in tracked]
        m, n = len(self.tracked), puzzle.n
        if not puzzle.is_group and (m == 0 or self.tracked[0] != puzzle.anchor):
            raise ValueError("conditional puzzles must track the anchor first")
        if n ** max(m, 1) >= 2 ** 62:
            raise ValueError("too many tracked facets for a 62-bit key")
        radix = n ** np.arange(m, dtype=np.int64)
        prim = puzzle.num_primitive
        inv_tab = np.argsort(puzzle.table[:prim], axis=-1)                    # [a, K, N]
        unit = float(puzzle.cost[:prim].min())

        start = np.asarray(self.tracked, dtype=np.int64)[None, :]
        goals = {tuple(start[0].tolist())}
        stack = [start[0]]
        while stack and goal_generators:
            t = stack.pop()
            for gperm in goal_generators:
                u = gperm[t]
                key = tuple(u.tolist())
                if key not in goals:
                    goals.add(key)
                    stack.append(u)
                    if len(goals) > max_entries:
                        raise ValueError("pattern database goal set exceeds max_entries")
        frontier = np.asarray(sorted(goals), dtype=np.int64).reshape(-1, m)
        keys_all = [np.unique(frontier @ radix)]
        dist_all = [np.zeros(len(keys_all[0]), dtype=np.float32)]
        visited = keys_all[0]
        depth = 0
        while len(frontier):
            depth += 1
            nxt_keys = []
            for a in range(prim):
                if puzzle.is_group:
                    nxt = inv_tab[a, 0][frontier]
                else:
                    k = frontier[:, 0]
                    ok = puzzle.legal[a, k]
                    f = frontier[ok]
                    nxt = inv_tab[a][f[:, :1], f]
                nxt_keys.append(nxt @ radix)
            cand = np.unique(np.concatenate(nxt_keys)) if nxt_keys else np.zeros(0, dtype=np.int64)
            new = cand[~np.isin(cand, visited, assume_unique=True)]
            if len(new) == 0:
                break
            visited = np.union1d(visited, new)
            if len(visited) > max_entries:
                raise ValueError(f"pattern database exceeds {max_entries} entries")
            keys_all.append(new)
            dist_all.append(np.full(len(new), depth * unit, dtype=np.float32))
            frontier = (new[:, None] // radix[None, :]) % n
        keys = np.concatenate(keys_all)
        dist = np.concatenate(dist_all)
        order = np.argsort(keys)
        self.size = len(keys)
        self.max_depth = float(dist.max()) if len(dist) else 0.0
        self.mean = float(dist.mean()) if len(dist) else 0.0
        self.device = torch.device(device)
        self.keys = torch.as_tensor(keys[order], dtype=torch.long, device=self.device)
        self.dist = torch.as_tensor(dist[order], dtype=torch.float32, device=self.device)
        self.tracked_t = torch.as_tensor(self.tracked, dtype=torch.int16, device=self.device)
        self.radix = torch.as_tensor(radix, dtype=torch.long, device=self.device)

    def lookup(self, P: torch.Tensor) -> torch.Tensor:
        """Admissible lower bound on the cost-to-go of each permutation state ``[B, N]``."""
        if len(self.tracked) == 0 or self.size == 0:
            return torch.zeros(P.shape[0], dtype=torch.float32, device=P.device)
        locs = (P.unsqueeze(1) == self.tracked_t.view(1, -1, 1)).to(torch.int32).argmax(dim=-1).long()
        key = (locs * self.radix.unsqueeze(0)).sum(dim=-1)
        idx = torch.searchsorted(self.keys, key).clamp(max=self.size - 1)
        return torch.where(self.keys[idx] == key, self.dist[idx], torch.zeros_like(self.dist[idx]))

    def lookup_np(self, P: np.ndarray) -> np.ndarray:
        return self.lookup(torch.as_tensor(np.atleast_2d(P), dtype=torch.int16, device=self.device)).cpu().numpy()


def build_pdbs(puzzle: Puzzle, max_entries: int = 400_000, count: int = 2,
               device: torch.device | str = "cpu", group: Optional[PuzzleGroup] = None,
               stabilizer_budget: int = 50_000) -> list[PatternDatabase]:
    """Stabiliser-chain pattern databases: PDB ``j`` tracks the longest base prefix that fits."""
    pdbs: list[PatternDatabase] = []
    used: set[int] = set()
    for j in range(count):
        if j == 0 and group is not None:
            grp = group
        else:
            order = [f for f in range(puzzle.n - 1, -1, -1) if f not in used] if j else None
            grp = PuzzleGroup(puzzle, seed=j, base_order=order)
        try:
            goal_gens, _ = grp.color_stabilizer(stabilizer_budget)
        except SearchBudgetExceeded:
            break
        base, sizes = grp.bsgs.base, grp.bsgs.orbit_sizes
        prod, m = len(grp.anchor_paths), 0
        while m < len(base) and prod * sizes[m] <= max_entries:
            prod *= sizes[m]
            m += 1
        tracked = ([puzzle.anchor] if puzzle.anchor is not None else []) + [int(b) for b in base[:m]]
        if not tracked or (j > 0 and set(tracked) <= used):
            break
        try:
            pdbs.append(PatternDatabase(puzzle, tracked, max_entries * 4, goal_gens, device))
        except ValueError:
            break
        used |= set(tracked)
    return pdbs


# ----------------------------------------------------------------------------------
# Heuristics
# ----------------------------------------------------------------------------------
class ZeroHeuristic:
    """``h = 0`` (uniform-cost search)."""

    admissible = True

    def __init__(self, tp: TensorPuzzle):
        self.tp = tp

    def __call__(self, P: torch.Tensor):
        return torch.zeros(P.shape[0], dtype=torch.float32, device=P.device), None


class PDBHeuristic:
    """Maximum over pattern databases (admissible and consistent)."""

    admissible = True

    def __init__(self, tp: TensorPuzzle, pdbs: Sequence[PatternDatabase]):
        self.tp, self.pdbs = tp, list(pdbs)

    def __call__(self, P: torch.Tensor):
        h = torch.zeros(P.shape[0], dtype=torch.float32, device=P.device)
        for pdb in self.pdbs:
            h = torch.maximum(h, pdb.lookup(P))
        return h, None


class NeuralHeuristic:
    """``h_theta(s)`` and ``pi_theta(.|s)`` from one batched forward pass, optionally max-ed with PDBs."""

    admissible = False

    def __init__(self, model, graph, tp: TensorPuzzle, pdbs: Sequence[PatternDatabase] = (), chunk: Optional[int] = None):
        from .model import auto_chunk
        self.model, self.graph, self.tp, self.pdbs = model, graph, tp, list(pdbs)
        self.chunk = chunk or auto_chunk(graph, model.cfg)

    def __call__(self, P: torch.Tensor):
        was_training = self.model.training
        self.model.eval()
        legal = self.tp.legal_mask(P) if self.tp.conditional else None
        value, _, logits = self.model.evaluate(self.graph, self.tp.colors(P), legal, self.chunk)
        if was_training:
            self.model.train()
        value = torch.where(self.tp.is_solved(P), torch.zeros_like(value), value.clamp(min=0.0))
        for pdb in self.pdbs:
            value = torch.maximum(value, pdb.lookup(P))
        return value, logits


# ----------------------------------------------------------------------------------
# Search results
# ----------------------------------------------------------------------------------
@dataclass
class SearchResult:
    solved: bool
    path: list[int] = field(default_factory=list)            # actions (may contain macros)
    primitive_path: list[int] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    cost: float = float("inf")
    nodes_expanded: int = 0
    nodes_generated: int = 0
    iterations: int = 0
    elapsed: float = 0.0
    max_depth: float = 0.0
    optimal_proven: bool = False
    table_hits: int = 0
    reason: str = ""

    @property
    def nodes_per_sec(self) -> float:
        return self.nodes_generated / max(self.elapsed, 1e-9)

    def to_dict(self) -> dict:
        return dict(solved=self.solved, moves=self.moves, cost=None if not self.solved else self.cost,
                    length=len(self.primitive_path), nodes_expanded=self.nodes_expanded,
                    nodes_generated=self.nodes_generated, iterations=self.iterations,
                    elapsed=self.elapsed, nodes_per_sec=self.nodes_per_sec, max_depth=self.max_depth,
                    optimal_proven=self.optimal_proven, table_hits=self.table_hits, reason=self.reason)


def _finish(result: SearchResult, puzzle: Puzzle, path: list[int], t0: float, device: torch.device) -> SearchResult:
    sync_device(device)
    result.path = path
    result.primitive_path = puzzle.expand_to_primitive(path)
    result.moves = puzzle.names_of(result.primitive_path)
    result.elapsed = time.perf_counter() - t0
    return result


def _as_state(tp: TensorPuzzle, start) -> torch.Tensor:
    if isinstance(start, torch.Tensor):
        return start.to(tp.device, tp.state_dtype).reshape(1, tp.n)
    return tp.from_numpy(np.asarray(start))


# ----------------------------------------------------------------------------------
# Batch weighted A*
# ----------------------------------------------------------------------------------
class BatchedWeightedAStar:
    """Batch Weighted A* (BWA*) with the whole OPEN/CLOSED arena stored in device tensors.

    Every iteration pops the ``batch_size`` lowest ``f = g + w*h`` nodes with one ``topk``, expands
    all of their children with one gather, deduplicates through the Zobrist transposition table
    and scores the survivors with one heuristic call.

    ``eval_mode="children"`` evaluates the heuristic on every generated child (DeepCubeA style);
    ``eval_mode="deferred"`` evaluates only the popped parents and ranks children by
    ``g + w*max(h(parent) - c, 0) - policy_weight*log pi(a|parent)`` -- one forward pass per batch.

    With ``optimal=True``, ``weight == 1`` and a consistent heuristic the search continues until the
    best OPEN ``f`` reaches the incumbent cost, proving optimality.
    """

    def __init__(self, tp: TensorPuzzle, heuristic: Callable, weight: float = 1.0, batch_size: int = 512,
                 max_nodes: int = 2_000_000, table_log2: int = 21, use_symmetry: bool = False,
                 eval_mode: str = "children", policy_weight: float = 0.0, optimal: bool = False,
                 telemetry: Optional[Telemetry] = None):
        if eval_mode not in ("children", "deferred"):
            raise ValueError("eval_mode must be 'children' or 'deferred'")
        self.tp, self.heuristic = tp, heuristic
        self.weight, self.batch_size, self.max_nodes = weight, batch_size, max_nodes
        self.eval_mode, self.policy_weight, self.optimal = eval_mode, policy_weight, optimal
        self.hasher = ZobristHasher(tp, use_symmetry=use_symmetry)
        self.table_log2 = table_log2
        self.telemetry = telemetry

    @torch.no_grad()
    def solve(self, start, max_iters: int = 10_000, time_limit: Optional[float] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> SearchResult:
        tp, dev = self.tp, self.tp.device
        puzzle = tp.puzzle
        t0 = time.perf_counter()
        res = SearchResult(solved=False)
        s0 = _as_state(tp, start)
        if bool(tp.is_solved(s0)[0]):
            res.solved, res.cost, res.optimal_proven = True, 0.0, True
            return _finish(res, puzzle, [], t0, dev)

        table = TranspositionTable(self.table_log2, dev)
        cap = min(max(4 * self.batch_size * tp.num_actions, 1 << 14), self.max_nodes)
        states = torch.zeros(cap, tp.n, dtype=tp.state_dtype, device=dev)
        g = torch.zeros(cap, dtype=torch.float32, device=dev)
        f = torch.full((cap,), float("inf"), dtype=torch.float32, device=dev)
        parent = torch.full((cap,), -1, dtype=torch.long, device=dev)
        act = torch.full((cap,), -1, dtype=torch.long, device=dev)
        states[0] = s0[0]
        h0, _ = self.heuristic(s0)
        f[0] = self.weight * h0[0]
        table.filter_insert(self.hasher.hash(s0), g[:1])
        size = 1
        best_cost, best_node = float("inf"), -1
        A = tp.num_actions
        arange_a = torch.arange(A, device=dev)
        full = False

        while res.iterations < max_iters:
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                res.reason = "time limit"
                break
            if should_stop is not None and should_stop():
                res.reason = "cancelled"
                break
            k = min(self.batch_size, size)
            vals, idx = torch.topk(f[:size], k, largest=False)
            live = vals < float("inf")
            if not bool(live.any()):
                res.reason = "open list exhausted"
                res.optimal_proven = best_node >= 0 and self.optimal
                break
            fmin = float(vals[0])
            if best_node >= 0 and fmin >= best_cost - 1e-6:
                res.optimal_proven = self.optimal and self.weight <= 1.0 and getattr(self.heuristic, "admissible", False)
                break
            idx = idx[live]
            f[idx] = float("inf")
            res.iterations += 1
            res.nodes_expanded += int(idx.shape[0])
            S, gp = states[idx], g[idx]
            children, legal = tp.expand(S)                                     # [b, A, N]
            gc = gp.unsqueeze(1) + tp.cost.unsqueeze(0)
            ok = legal & (gc < best_cost - 1e-6)
            if self.eval_mode == "deferred":
                hp, logits = self.heuristic(S)
                h_est = (hp.unsqueeze(1) - tp.cost.unsqueeze(0)).clamp(min=0.0)
                prior = torch.zeros_like(h_est)
                if logits is not None and self.policy_weight > 0:
                    prior = -self.policy_weight * torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)
                pri_all = gc + self.weight * h_est + prior
            sel = torch.nonzero(ok.reshape(-1)).flatten()
            res.nodes_generated += int(sel.shape[0])
            C = children.reshape(-1, tp.n)[sel]
            gcs = gc.reshape(-1)[sel]
            par = idx.repeat_interleave(A)[sel]
            acs = arange_a.repeat(idx.shape[0])[sel]
            keep = table.filter_insert(self.hasher.hash(C), gcs)
            C, gcs, par, acs = C[keep], gcs[keep], par[keep], acs[keep]
            if C.shape[0] == 0:
                continue
            solved = tp.is_solved(C)
            if self.eval_mode == "children":
                hc, _ = self.heuristic(C)
                fc = gcs + self.weight * hc
            else:
                fc = pri_all.reshape(-1)[sel][keep]
            fc = torch.where(solved, gcs, fc)
            m = C.shape[0]
            if size + m > cap:
                new_cap = min(max(cap * 2, size + m), self.max_nodes)
                if new_cap > cap:
                    grow = new_cap - cap
                    states = torch.cat([states, torch.zeros(grow, tp.n, dtype=tp.state_dtype, device=dev)])
                    g = torch.cat([g, torch.zeros(grow, dtype=torch.float32, device=dev)])
                    f = torch.cat([f, torch.full((grow,), float("inf"), dtype=torch.float32, device=dev)])
                    parent = torch.cat([parent, torch.full((grow,), -1, dtype=torch.long, device=dev)])
                    act = torch.cat([act, torch.full((grow,), -1, dtype=torch.long, device=dev)])
                    cap = new_cap
                if size + m > cap:                         # arena exhausted: keep the most promising
                    room = cap - size
                    full = True
                    if room <= 0:
                        res.reason = "node budget exhausted"
                        break
                    top = torch.topk(fc, room, largest=False).indices
                    C, gcs, par, acs, fc, solved = C[top], gcs[top], par[top], acs[top], fc[top], solved[top]
                    m = room
            sl = slice(size, size + m)
            states[sl], g[sl], f[sl], parent[sl], act[sl] = C, gcs, fc, par, acs
            if bool(solved.any()):
                cand = torch.where(solved, gcs, torch.full_like(gcs, float("inf")))
                j = int(cand.argmin())
                if float(cand[j]) < best_cost:
                    best_cost, best_node = float(cand[j]), size + j
                f[sl] = torch.where(solved, torch.full_like(fc, float("inf")), fc)
                if not self.optimal:
                    size += m
                    break
            size += m
            res.max_depth = max(res.max_depth, float(gcs.max()))
            if self.telemetry is not None and res.iterations % 4 == 0:
                el = time.perf_counter() - t0
                self.telemetry.log(search_depth=res.max_depth, search_nodes_per_sec=res.nodes_generated / max(el, 1e-9),
                                   search_open=float(size), search_fmin=fmin)
        else:
            res.reason = "iteration limit"

        res.table_hits = table.hits
        path: list[int] = []
        if best_node >= 0:
            res.solved, res.cost = True, best_cost
            par_np, act_np = parent[:size].cpu().numpy(), act[:size].cpu().numpy()
            node = best_node
            while par_np[node] >= 0:
                path.append(int(act_np[node]))
                node = int(par_np[node])
            path.reverse()
        elif full and not res.reason:
            res.reason = "node budget exhausted"
        if self.telemetry is not None:
            el = time.perf_counter() - t0
            self.telemetry.log(search_depth=res.max_depth, search_nodes_per_sec=res.nodes_generated / max(el, 1e-9))
        return _finish(res, puzzle, path, t0, dev)


# ----------------------------------------------------------------------------------
# Batched beam search
# ----------------------------------------------------------------------------------
class BatchedBeamSearch:
    """Breadth-synchronised beam search: expand the whole beam, keep the ``beam_width`` best children.

    Children are ranked by ``g_weight*g + h - policy_weight*log pi(a|parent)``; states already seen
    with an equal or smaller ``g`` are discarded through the transposition table.
    """

    def __init__(self, tp: TensorPuzzle, heuristic: Callable, beam_width: int = 1024, max_depth: int = 64,
                 g_weight: float = 0.05, policy_weight: float = 0.0, table_log2: int = 21,
                 use_symmetry: bool = False, telemetry: Optional[Telemetry] = None):
        self.tp, self.heuristic = tp, heuristic
        self.beam_width, self.max_depth = beam_width, max_depth
        self.g_weight, self.policy_weight = g_weight, policy_weight
        self.hasher = ZobristHasher(tp, use_symmetry=use_symmetry)
        self.table_log2 = table_log2
        self.telemetry = telemetry

    @torch.no_grad()
    def solve(self, start, time_limit: Optional[float] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> SearchResult:
        tp, dev = self.tp, self.tp.device
        puzzle = tp.puzzle
        t0 = time.perf_counter()
        res = SearchResult(solved=False)
        beam = _as_state(tp, start)
        if bool(tp.is_solved(beam)[0]):
            res.solved, res.cost, res.optimal_proven = True, 0.0, True
            return _finish(res, puzzle, [], t0, dev)
        table = TranspositionTable(self.table_log2, dev)
        gb = torch.zeros(1, dtype=torch.float32, device=dev)
        table.filter_insert(self.hasher.hash(beam), gb)
        A = tp.num_actions
        arange_a = torch.arange(A, device=dev)
        history: list[tuple[torch.Tensor, torch.Tensor]] = []
        final = -1
        for depth in range(self.max_depth):
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                res.reason = "time limit"
                break
            if should_stop is not None and should_stop():
                res.reason = "cancelled"
                break
            res.iterations += 1
            res.nodes_expanded += int(beam.shape[0])
            children, legal = tp.expand(beam)
            gc = gb.unsqueeze(1) + tp.cost.unsqueeze(0)
            prior = None
            if self.policy_weight > 0:
                _, logits = self.heuristic(beam)
                if logits is not None:
                    prior = -self.policy_weight * torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)
            sel = torch.nonzero(legal.reshape(-1)).flatten()
            res.nodes_generated += int(sel.shape[0])
            C = children.reshape(-1, tp.n)[sel]
            gcs = gc.reshape(-1)[sel]
            par = torch.arange(beam.shape[0], device=dev).repeat_interleave(A)[sel]
            acs = arange_a.repeat(beam.shape[0])[sel]
            keep = table.filter_insert(self.hasher.hash(C), gcs)
            C, gcs, par, acs = C[keep], gcs[keep], par[keep], acs[keep]
            if C.shape[0] == 0:
                res.reason = "beam exhausted"
                break
            solved = tp.is_solved(C)
            if bool(solved.any()):
                cand = torch.where(solved, gcs, torch.full_like(gcs, float("inf")))
                final = int(cand.argmin())
                history.append((par, acs))
                res.solved, res.cost = True, float(cand[final])
                res.max_depth = float(gcs.max())
                break
            hc, _ = self.heuristic(C)
            score = self.g_weight * gcs + hc
            if prior is not None:
                score = score + prior.reshape(-1)[sel][keep]
            if C.shape[0] > self.beam_width:
                top = torch.topk(score, self.beam_width, largest=False).indices
                C, gcs, par, acs = C[top], gcs[top], par[top], acs[top]
            history.append((par, acs))
            beam, gb = C, gcs
            res.max_depth = float(gcs.max())
            if self.telemetry is not None:
                el = time.perf_counter() - t0
                self.telemetry.log(search_depth=res.max_depth, search_nodes_per_sec=res.nodes_generated / max(el, 1e-9),
                                   search_open=float(beam.shape[0]), search_fmin=float(score.min()))
        else:
            res.reason = "depth limit"
        res.table_hits = table.hits
        path: list[int] = []
        if res.solved:
            node = final
            for par, acs in reversed(history):
                path.append(int(acs[node]))
                node = int(par[node])
            path.reverse()
        return _finish(res, puzzle, path, t0, dev)


# ----------------------------------------------------------------------------------
# Reference solver (exact BFS) used by tests and small puzzles
# ----------------------------------------------------------------------------------
def bfs_distances(puzzle: Puzzle, max_states: int = 2_000_000) -> dict[bytes, int]:
    """Exact distance-to-solved of every colour state by breadth-first search (small puzzles only)."""
    start = puzzle.identity
    dist = {puzzle.colors(start).astype(np.uint8).tobytes(): 0}
    frontier = start[None, :]
    depth = 0
    prim = puzzle.num_primitive
    while len(frontier):
        depth += 1
        nxt = []
        for a in range(prim):
            if puzzle.is_group:
                nxt.append(frontier[:, puzzle.table[a, 0]])
            else:
                k = (frontier == puzzle.anchor).argmax(axis=1)
                ok = puzzle.legal[a, k]
                nxt.append(np.take_along_axis(frontier[ok], puzzle.table[a][k[ok]], axis=1))
        cand = np.concatenate(nxt)
        cols = puzzle.solved_colors[cand].astype(np.uint8)
        fresh = []
        for i in range(len(cand)):
            key = cols[i].tobytes()
            if key not in dist:
                dist[key] = depth
                fresh.append(i)
        if len(dist) > max_states:
            raise ValueError("state space too large for exhaustive BFS")
        frontier = cand[fresh]
    return dist


def make_solver(puzzle: Puzzle, device: torch.device | str | None = None, model=None, graph=None,
                method: str = "astar", pdbs: Optional[Sequence[PatternDatabase]] = None, weight: float = 1.0,
                batch_size: int = 512, beam_width: int = 2048, telemetry: Optional[Telemetry] = None,
                tp: Optional[TensorPuzzle] = None, **kwargs):
    """Assemble a search object: neural heuristic when a model is given, pattern databases otherwise."""
    tp = tp or TensorPuzzle(puzzle, device)
    pdbs = list(pdbs) if pdbs is not None else []
    if model is not None:
        if graph is None:
            from .graph_compiler import compile_graph
            graph = compile_graph(puzzle, model.cfg.lap_k, model.cfg.rw_k, tp.device)
        heuristic: Callable = NeuralHeuristic(model, graph, tp, pdbs)
    elif pdbs:
        heuristic = PDBHeuristic(tp, pdbs)
    else:
        heuristic = ZeroHeuristic(tp)
    if method == "beam":
        return BatchedBeamSearch(tp, heuristic, beam_width=beam_width, telemetry=telemetry, **kwargs)
    return BatchedWeightedAStar(tp, heuristic, weight=weight, batch_size=batch_size, telemetry=telemetry, **kwargs)
