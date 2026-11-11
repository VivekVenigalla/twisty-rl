"""Computational group theory engine.

* :class:`BSGS` -- a from-scratch Schreier-Sims implementation (randomised pre-sifting followed by
  the deterministic Schreier-generator closure) producing a base and strong generating set,
  exact group order, membership testing (sifting) and uniform random sampling.
* :class:`PuzzleGroup` -- puzzle-level wrapper: group order / reachable state count, solvability
  of permutation *and* colour states (parity traps are detected instantly), the colour-stabiliser
  subgroup, groupoid (sliding puzzle) support via vertex groups, and the stabiliser curriculum
  ``G = G_0 > G_1 > ... > {e}``.
* :class:`MacroMiner` -- symbolic mining of commutators, conjugates and powers with minimal
  support, promoted to macro-actions.

Permutations are integer image arrays.  Two composition orders appear:

* *gather* composition (puzzle moves): applying ``a`` then ``b`` to a state is ``a[b]``;
* inside :class:`BSGS` elements act on points on the right, ``x^(gh) = (x^g)^h``, so the product
  ``g*h`` is the array ``h[g]``.

Both generate the same set of permutations, so set-level questions (order, membership, cosets)
are independent of the convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .schema import Macro, Puzzle, invert_perm


# ----------------------------------------------------------------------------------
# Elementary permutation utilities
# ----------------------------------------------------------------------------------
def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gather composition: the single move equivalent to applying ``a`` and then ``b``."""
    return a[b]


def perm_cycles(p: np.ndarray, include_fixed: bool = False) -> list[list[int]]:
    n = len(p)
    seen = np.zeros(n, dtype=bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        cyc = [i]
        seen[i] = True
        j = int(p[i])
        while j != i:
            cyc.append(j)
            seen[j] = True
            j = int(p[j])
        if len(cyc) > 1 or include_fixed:
            out.append(cyc)
    return out


def perm_order(p: np.ndarray) -> int:
    return math.lcm(*[len(c) for c in perm_cycles(p)] or [1])


def perm_parity(p: np.ndarray) -> int:
    """0 for even permutations, 1 for odd ones."""
    return sum(len(c) - 1 for c in perm_cycles(p)) % 2


def support(p: np.ndarray) -> int:
    """``|supp(sigma)| = |{x : sigma(x) != x}|``."""
    return int((p != np.arange(len(p))).sum())


def perm_power(p: np.ndarray, k: int) -> np.ndarray:
    out = np.arange(len(p), dtype=p.dtype)
    base = p.copy()
    while k > 0:
        if k & 1:
            out = out[base]
        base = base[base]
        k >>= 1
    return out


class SearchBudgetExceeded(RuntimeError):
    """Raised when a backtrack search runs out of its node budget."""


# ----------------------------------------------------------------------------------
# Schreier-Sims
# ----------------------------------------------------------------------------------
class BSGS:
    """Base and strong generating set of ``<generators> <= S_n`` via Schreier-Sims.

    The stabiliser chain is ``G^(0) = G >= G^(1) >= ... >= G^(k) = 1`` where ``G^(i)`` fixes the first
    ``i`` base points.  For each level the basic orbit ``Delta_i`` of ``base[i]`` under ``G^(i)`` and a
    transversal ``u_i(gamma)`` with ``base[i]^u = gamma`` are stored explicitly, so that every group
    element factors uniquely as ``g = u_0 o u_1 o ... o u_{k-1}`` (function composition).
    """

    def __init__(self, n: int, generators: Iterable[np.ndarray], base_order: Optional[Sequence[int]] = None,
                 seed: int = 0, presift: int = 96, deterministic: bool = True):
        self.n = n
        self._id = np.arange(n, dtype=np.intp)
        self.generators = [np.asarray(g, dtype=np.intp) for g in generators]
        self.generators = [g for g in self.generators if not np.array_equal(g, self._id)]
        pref = list(base_order) if base_order is not None else []
        seen = set(pref)
        self._pref = pref + [i for i in range(n) if i not in seen]
        self.base: list[int] = []
        self.S: list[list[np.ndarray]] = []
        self.orbit: list[list[int]] = []
        self.in_orbit: list[np.ndarray] = []
        self.trans: list[np.ndarray] = []
        self.trans_inv: list[np.ndarray] = []
        self._checked: list[set[tuple[int, int]]] = []
        self._orbit_ids: dict[int, np.ndarray] = {}
        self.schreier_generators_tested = 0
        self.verified = False
        self._rng = np.random.default_rng(seed)
        self._build(presift, deterministic)

    # -- construction -----------------------------------------------------------------
    def _moved_point(self, g: np.ndarray) -> int:
        for p in self._pref:
            if g[p] != p:
                return p
        raise ValueError("identity has no moved point")

    def _new_level(self, point: int) -> None:
        n = self.n
        self.base.append(point)
        self.S.append([])
        self.orbit.append([point])
        ino = np.zeros(n, dtype=bool)
        ino[point] = True
        self.in_orbit.append(ino)
        T = np.zeros((n, n), dtype=np.intp)
        Ti = np.zeros((n, n), dtype=np.intp)
        T[point] = self._id
        Ti[point] = self._id
        self.trans.append(T)
        self.trans_inv.append(Ti)
        self._checked.append(set())

    def _add_generator(self, level: int, h: np.ndarray) -> None:
        self.S[level].append(h)
        self._extend_orbit(level, h)
        self._orbit_ids.clear()

    def _extend_orbit(self, level: int, new_gen: Optional[np.ndarray]) -> None:
        orb, ino = self.orbit[level], self.in_orbit[level]
        T, Ti = self.trans[level], self.trans_inv[level]
        gens = self.S[level]

        def visit(p: int, x: np.ndarray, out: list[int]) -> None:
            q = int(x[p])
            if not ino[q]:
                ino[q] = True
                orb.append(q)
                T[q] = x[T[p]]
                Ti[q] = invert_perm(T[q])
                out.append(q)

        frontier: list[int] = []
        if new_gen is None:
            frontier = list(orb)
        else:
            for p in list(orb):
                visit(p, new_gen, frontier)
        while frontier:
            nxt: list[int] = []
            for p in frontier:
                for x in gens:
                    visit(p, x, nxt)
            frontier = nxt

    def _install(self, h: np.ndarray, lo: int, hi: int) -> None:
        """Add the non-trivial residue ``h`` (which fixes ``base[:hi]``) to levels ``lo..hi``."""
        if hi == len(self.base):
            self._new_level(self._moved_point(h))
        for lvl in range(lo, hi + 1):
            self._add_generator(lvl, h)

    def _random_stream(self):
        """Product-replacement ("rattle") generator of pseudo-random group elements."""
        rng = self._rng
        pool = [g.copy() for g in self.generators]
        while len(pool) < 10:
            pool.append(self.generators[len(pool) % len(self.generators)].copy())
        acc = self._id.copy()
        m = len(pool)
        step = 0
        while True:
            i, j = rng.integers(m), rng.integers(m - 1)
            j = j + 1 if j >= i else j
            other = pool[j] if rng.random() < 0.5 else invert_perm(pool[j])
            pool[i] = other[pool[i]] if rng.random() < 0.5 else pool[i][other]
            acc = pool[i][acc]
            step += 1
            if step > 60:
                yield acc

    def _build(self, presift: int, deterministic: bool) -> None:
        if not self.generators:
            self.verified = True
            return
        for g in self.generators:
            if all(g[b] == b for b in self.base):
                self._new_level(self._moved_point(g))
        for lvl in range(len(self.base)):
            for g in self.generators:
                if all(g[b] == b for b in self.base[:lvl]):
                    self.S[lvl].append(g)
            self._extend_orbit(lvl, None)

        # Phase 1: randomised Schreier-Sims -- cheaply builds an almost complete chain.
        if presift > 0:
            stream = self._random_stream()
            quiet = 0
            for _ in range(presift * 4):
                h, j = self.strip(next(stream), 0)
                if j < len(self.base) or not np.array_equal(h, self._id):
                    self._install(h, 1, j)
                    quiet = 0
                else:
                    quiet += 1
                    if quiet >= presift // 4:
                        break

        # Phase 2: deterministic closure -- every Schreier generator must sift to the identity.
        if deterministic:
            self._close()
            self.verified = True

    def _close(self) -> None:
        i = len(self.base) - 1
        while i >= 0:
            restart = False
            orb, T, Ti = self.orbit[i], self.trans[i], self.trans_inv[i]
            checked = self._checked[i]
            bi = 0
            while bi < len(orb) and not restart:
                beta = orb[bi]
                u = T[beta]
                gi = 0
                gens = self.S[i]
                while gi < len(gens):
                    if (beta, gi) not in checked:
                        x = gens[gi]
                        ux = x[u]
                        tb = T[int(x[beta])]
                        if not np.array_equal(ux, tb):
                            self.schreier_generators_tested += 1
                            h, j = self.strip(Ti[int(x[beta])][ux], i + 1)
                            if j < len(self.base) or not np.array_equal(h, self._id):
                                self._install(h, i + 1, j)
                                i = j
                                restart = True
                                break
                        checked.add((beta, gi))
                    gi += 1
                bi += 1
            if not restart:
                i -= 1

    # -- queries ------------------------------------------------------------------------
    def strip(self, g: np.ndarray, start: int = 0) -> tuple[np.ndarray, int]:
        """Sift ``g`` through the chain from ``start``; returns ``(residue, level reached)``."""
        base, ino, Ti = self.base, self.in_orbit, self.trans_inv
        for lvl in range(start, len(base)):
            b = int(g[base[lvl]])
            if not ino[lvl][b]:
                return g, lvl
            if b != base[lvl]:
                g = Ti[lvl][b][g]
        return g, len(base)

    def contains(self, g: np.ndarray) -> bool:
        g = np.asarray(g, dtype=np.intp)
        h, lvl = self.strip(g, 0)
        return lvl == len(self.base) and bool(np.array_equal(h, self._id))

    @property
    def orbit_sizes(self) -> list[int]:
        return [len(o) for o in self.orbit]

    @property
    def order(self) -> int:
        return math.prod(self.orbit_sizes)

    def stabilizer_order(self, level: int) -> int:
        return math.prod(self.orbit_sizes[level:])

    @property
    def strong_generators(self) -> list[np.ndarray]:
        seen: dict[bytes, np.ndarray] = {}
        for lvl in self.S:
            for g in lvl:
                seen.setdefault(g.tobytes(), g)
        return list(seen.values())

    def random_element(self, rng: np.random.Generator, level: int = 0) -> np.ndarray:
        """Uniformly random element of the stabiliser ``G^(level)``."""
        g = self._id
        for lvl in range(len(self.base) - 1, level - 1, -1):
            gamma = self.orbit[lvl][int(rng.integers(len(self.orbit[lvl])))]
            g = self.trans[lvl][gamma][g]
        return g.copy()

    def random_elements(self, rng: np.random.Generator, batch: int, level: int = 0) -> np.ndarray:
        """``[batch, n]`` uniformly random elements of ``G^(level)`` (vectorised over the batch)."""
        g = np.tile(self._id, (batch, 1))
        for lvl in range(len(self.base) - 1, level - 1, -1):
            orb = np.asarray(self.orbit[lvl], dtype=np.intp)
            gamma = orb[rng.integers(len(orb), size=batch)]
            g = np.take_along_axis(self.trans[lvl][gamma], g, axis=1)
        return g

    def level_orbits(self, level: int) -> np.ndarray:
        """Orbit id of every point under ``G^(level)`` (all singletons for ``level == len(base)``)."""
        if level in self._orbit_ids:
            return self._orbit_ids[level]
        n = self.n
        ids = -np.ones(n, dtype=np.intp)
        gens = self.S[level] if level < len(self.base) else []
        cur = 0
        for s in range(n):
            if ids[s] >= 0:
                continue
            ids[s] = cur
            stack = [s]
            while stack:
                p = stack.pop()
                for x in gens:
                    q = int(x[p])
                    if ids[q] < 0:
                        ids[q] = cur
                        stack.append(q)
            cur += 1
        self._orbit_ids[level] = ids
        return ids

    # -- colour-constrained backtrack search -------------------------------------------------
    def find_coloring_element(self, color: np.ndarray, target: np.ndarray, start: int = 0,
                              forced: Optional[int] = None, budget: int = 200_000) -> Optional[np.ndarray]:
        """Find ``g`` in ``G^(start)`` with ``color[g[x]] == target[x]`` for all points ``x``.

        Depth-first search over base images, pruned by the necessary condition that every orbit
        ``O`` of the remaining stabiliser satisfies ``multiset(color[p[O]]) == multiset(target[O])``.
        """
        k = len(self.base)
        ncol = int(max(color.max(), target.max())) + 1
        nkeys = self.n * ncol

        def consistent(p: np.ndarray, level: int) -> bool:
            oid = self.level_orbits(level) * ncol
            return bool(np.array_equal(np.bincount(oid + color[p], minlength=nkeys),
                                       np.bincount(oid + target, minlength=nkeys)))

        def candidates(p: np.ndarray, level: int) -> list[int]:
            orb = np.asarray(self.orbit[level], dtype=np.intp)
            if forced is not None and level == start:
                orb = orb[orb == forced]
            return orb[color[p[orb]] == target[self.base[level]]].tolist()

        if not consistent(self._id, start):
            return None
        if start >= k:
            return self._id.copy()
        stack: list[tuple[int, np.ndarray, list[int]]] = [(start, self._id, candidates(self._id, start))]
        nodes = 0
        while stack:
            level, p, cands = stack[-1]
            if not cands:
                stack.pop()
                continue
            gamma = cands.pop()
            nodes += 1
            if nodes > budget:
                raise SearchBudgetExceeded(f"colour search exceeded {budget} nodes")
            p2 = p[self.trans[level][gamma]]
            if not consistent(p2, level + 1):
                continue
            if level + 1 == k:
                return p2
            stack.append((level + 1, p2, candidates(p2, level + 1)))
        return None

    def color_stabilizer(self, color: np.ndarray, budget: int = 200_000) -> tuple[list[np.ndarray], int]:
        """Generators and order of ``H = {g in G : color o g = color}`` (level-by-level base-image search)."""
        gens: list[np.ndarray] = []
        order = 1

        def orbit_of(point: int) -> set[int]:
            seen = {point}
            stack = [point]
            while stack:
                p = stack.pop()
                for g in gens:
                    q = int(g[p])
                    if q not in seen:
                        seen.add(q)
                        stack.append(q)
            return seen

        for s in range(len(self.base) - 1, -1, -1):
            beta = self.base[s]
            known = orbit_of(beta)
            failed: set[int] = set()
            for gamma in self.orbit[s]:
                if gamma in known or gamma in failed or color[gamma] != color[beta]:
                    continue
                g = self.find_coloring_element(color, color, start=s, forced=gamma, budget=budget)
                if g is None:
                    failed |= orbit_of(gamma)
                else:
                    gens.append(g)
                    known = orbit_of(beta)
            order *= len(known)
        return gens, order


# ----------------------------------------------------------------------------------
# Puzzle-level group
# ----------------------------------------------------------------------------------
@dataclass
class Stage:
    index: int
    level: int            # number of leading base points fixed by this subgroup
    order: int            # |G_j|
    index_in_previous: int


class StabilizerCurriculum:
    """Multi-stage subgroup chain ``G = G_0 > G_1 > ... > {e}`` built from the BSGS.

    Stage ``j`` is the pointwise stabiliser of the first ``level_j`` base points -- the automatic
    analogue of Thistlethwaite/Kociemba phases for an arbitrary permutation puzzle.
    """

    def __init__(self, group: "PuzzleGroup", num_stages: int = 4):
        self.group = group
        bsgs = group.bsgs
        sizes = bsgs.orbit_sizes
        k = len(sizes)
        logs = np.log(np.asarray(sizes, dtype=np.float64)) if k else np.zeros(0)
        total = float(logs.sum())
        cuts = [0]
        num_stages = max(1, min(num_stages, k))
        acc = 0.0
        for lvl in range(k):
            acc += float(logs[lvl])
            if len(cuts) < num_stages and acc >= total * len(cuts) / num_stages - 1e-12 and lvl + 1 < k:
                cuts.append(lvl + 1)
        cuts.append(k)
        self.levels = sorted(set(cuts))
        self.base = np.asarray(bsgs.base, dtype=np.intp)
        self.stages: list[Stage] = []
        prev = None
        for j, lvl in enumerate(self.levels):
            order = bsgs.stabilizer_order(lvl)
            self.stages.append(Stage(j, lvl, order, 1 if prev is None else prev // order))
            prev = order

    def __len__(self) -> int:
        return len(self.stages)

    def sample(self, stage: int, rng: np.random.Generator, batch: int) -> np.ndarray:
        """``[batch, n]`` uniformly random *states* of the subgroup ``G_stage``."""
        return self.group.random_states(rng, batch, level=self.levels[stage])

    def stage_of(self, P: np.ndarray) -> np.ndarray:
        """Deepest stage whose subgroup contains each state (number of completed sub-goals)."""
        P = np.atleast_2d(P)
        fixed = P[:, self.base] == self.base[None, :] if len(self.base) else np.ones((len(P), 0), bool)
        if self.group.puzzle.anchor is not None:
            home = (P[:, self.group.puzzle.anchor] == self.group.puzzle.anchor)[:, None]
            fixed = fixed & home
        lead = np.cumprod(fixed, axis=1).sum(axis=1) if fixed.shape[1] else np.zeros(len(P), dtype=np.intp)
        return np.searchsorted(np.asarray(self.levels), lead, side="right") - 1

    def tracked_facets(self, stage: int) -> list[int]:
        """Facets whose home-coming defines membership of ``G_stage`` (pattern-database key)."""
        out = self.base[: self.levels[stage]].tolist()
        if self.group.puzzle.anchor is not None:
            out = [self.group.puzzle.anchor] + out
        return out

    def describe(self) -> list[dict]:
        return [dict(stage=s.index, fixed_base_points=s.level, order=s.order, index=s.index_in_previous)
                for s in self.stages]


class PuzzleGroup:
    """Group-theoretic view of a compiled :class:`Puzzle`.

    For conditional (sliding) puzzles the state space is a groupoid; its vertex group at the home
    anchor position is generated by the Schreier loops ``w_k . a . w_{k'}^{-1}`` over a spanning tree
    of anchor positions, and every reachable state is ``v[T_k]`` with ``v`` in the vertex group.
    """

    def __init__(self, puzzle: Puzzle, seed: int = 0, deterministic: bool = True,
                 base_order: Optional[Sequence[int]] = None):
        self.puzzle = puzzle
        self.n = puzzle.n
        self.color = puzzle.solved_colors
        pref = list(base_order or []) + self._base_preference()
        pref = list(dict.fromkeys(pref))
        if puzzle.is_group:
            self.anchor_paths: dict[int, np.ndarray] = {0: puzzle.identity}
            self.anchor_words: dict[int, tuple[int, ...]] = {0: ()}
            gens = puzzle.primitive_generators
        else:
            gens = self._vertex_group_generators()
        self.bsgs = BSGS(self.n, gens, base_order=pref, seed=seed, deterministic=deterministic)
        self._anchor_list = sorted(self.anchor_paths)
        self._anchor_inv = {k: invert_perm(t) for k, t in self.anchor_paths.items()}
        self._color_stab: Optional[tuple[list[np.ndarray], int]] = None

    def _base_preference(self) -> list[int]:
        """Prefer facets on pieces with a unique colour signature (keeps colour searches shallow)."""
        pz = self.puzzle
        sigs: dict[int, tuple] = {}
        for piece in range(pz.num_pieces):
            facets = np.nonzero(pz.piece_of == piece)[0]
            sigs[piece] = tuple(sorted(pz.solved_colors[facets].tolist()))
        counts: dict[tuple, int] = {}
        for s in sigs.values():
            counts[s] = counts.get(s, 0) + 1
        return sorted(range(pz.n), key=lambda f: (counts[sigs[int(pz.piece_of[f])]], f))

    def _vertex_group_generators(self) -> list[np.ndarray]:
        pz = self.puzzle
        home = pz.anchor
        paths = {home: pz.identity}
        words: dict[int, tuple[int, ...]] = {home: ()}
        queue = [home]
        while queue:
            k = queue.pop(0)
            for a in range(pz.num_primitive):
                if not pz.legal[a, k]:
                    continue
                m = pz.table[a, k]
                k2 = int(np.nonzero(m == k)[0][0])
                if k2 not in paths:
                    paths[k2] = paths[k][m]
                    words[k2] = words[k] + (a,)
                    queue.append(k2)
        self.anchor_paths, self.anchor_words = paths, words
        loops: dict[bytes, np.ndarray] = {}
        for k, Tk in paths.items():
            for a in range(pz.num_primitive):
                if not pz.legal[a, k]:
                    continue
                m = pz.table[a, k]
                k2 = int(np.nonzero(m == k)[0][0])
                loop = Tk[m][invert_perm(paths[k2])]
                if not np.array_equal(loop, pz.identity):
                    loops.setdefault(loop.tobytes(), loop)
        return list(loops.values())

    # -- sizes --------------------------------------------------------------------------
    @property
    def order(self) -> int:
        """Order of the generated group (vertex group for conditional puzzles)."""
        return self.bsgs.order

    @property
    def num_states(self) -> int:
        """Number of reachable permutation states."""
        return self.bsgs.order * len(self.anchor_paths)

    def color_stabilizer(self, budget: int = 200_000) -> tuple[list[np.ndarray], int]:
        if self._color_stab is None:
            self._color_stab = self.bsgs.color_stabilizer(self.color, budget)
        return self._color_stab

    @property
    def num_color_states(self) -> int:
        """Number of visually distinct reachable states (``|G| / |Stab(colouring)|``)."""
        return self.num_states // self.color_stabilizer()[1]

    # -- membership / solvability -----------------------------------------------------------
    def contains_state(self, P: np.ndarray) -> bool:
        P = np.asarray(P, dtype=np.intp)
        if sorted(P.tolist()) != list(range(self.n)):
            return False
        pz = self.puzzle
        if pz.is_group:
            return self.bsgs.contains(P)
        k = pz.anchor_pos(P)
        if k not in self._anchor_inv:
            return False
        return self.bsgs.contains(P[self._anchor_inv[k]])

    def state_from_colors(self, colors: np.ndarray, budget: int = 200_000) -> Optional[np.ndarray]:
        """A reachable permutation state showing ``colors`` (``None`` when the colouring is unsolvable)."""
        colors = np.asarray(colors, dtype=np.intp)
        if colors.shape != (self.n,) or not np.array_equal(np.sort(colors), np.sort(self.color)):
            return None
        pz = self.puzzle
        if not pz.is_group:
            if len(set(self.color.tolist())) != self.n:
                raise ValueError("colour-state solvability of conditional puzzles requires distinct facets")
            lookup = invert_perm(self.color)
            P = lookup[colors]
            return P if self.contains_state(P) else None
        return self.bsgs.find_coloring_element(self.color, colors, budget=budget)

    def is_solvable(self, state: np.ndarray, colors: bool = False) -> bool:
        """Exact reachability test; ``colors=True`` interprets ``state`` as a colour state."""
        if colors:
            return self.state_from_colors(state) is not None
        return self.contains_state(state)

    # -- sampling ---------------------------------------------------------------------------
    def random_states(self, rng: np.random.Generator, batch: int, level: int = 0) -> np.ndarray:
        g = self.bsgs.random_elements(rng, batch, level)
        if self.puzzle.is_group or level > 0:
            return g
        ks = rng.choice(self._anchor_list, size=batch)
        paths = np.stack([self.anchor_paths[int(k)] for k in ks])
        return np.take_along_axis(g, paths, axis=1)

    def random_state(self, rng: np.random.Generator, level: int = 0) -> np.ndarray:
        return self.random_states(rng, 1, level)[0]

    def stabilizer_curriculum(self, num_stages: int = 4) -> StabilizerCurriculum:
        return StabilizerCurriculum(self, num_stages)

    def summary(self) -> dict:
        b = self.bsgs
        return dict(
            puzzle=self.puzzle.name, n=self.n, order=self.order, num_states=self.num_states,
            base=list(map(int, b.base)), orbit_sizes=b.orbit_sizes,
            strong_generators=len(b.strong_generators), anchors=len(self.anchor_paths),
            schreier_generators_tested=b.schreier_generators_tested, verified=b.verified,
        )


# ----------------------------------------------------------------------------------
# Commutator / conjugate macro mining
# ----------------------------------------------------------------------------------
@dataclass
class MinedMacro:
    word: tuple[int, ...]
    perm: np.ndarray
    support: int
    pieces_moved: int
    pieces_twisted: int
    kind: str
    origin: str
    order: int
    text: str = ""

    @property
    def pieces_affected(self) -> int:
        return self.pieces_moved + self.pieces_twisted

    def to_dict(self) -> dict:
        return dict(word=list(self.word), text=self.text, support=self.support, kind=self.kind,
                    origin=self.origin, order=self.order, pieces_moved=self.pieces_moved,
                    pieces_twisted=self.pieces_twisted)


class MacroMiner:
    """Mine low-support commutators ``[A,B] = A B A^-1 B^-1``, conjugates ``X C X^-1`` and powers."""

    def __init__(self, puzzle: Puzzle, max_len_a: int = 3, max_len_b: int = 2, max_conj_len: int = 1,
                 max_support: Optional[int] = None, max_power_len: int = 24):
        self.puzzle = puzzle
        self.max_len_a, self.max_len_b = max_len_a, max_len_b
        self.max_conj_len = max_conj_len
        self.max_support = max_support
        self.max_power_len = max_power_len
        self.home = puzzle.anchor if puzzle.anchor is not None else 0
        self.candidates_examined = 0

    # -- word machinery -----------------------------------------------------------------
    def _reduced_words(self, max_len: int) -> list[tuple[int, ...]]:
        pz = self.puzzle
        prim = range(pz.num_primitive)
        words: list[tuple[int, ...]] = [(a,) for a in prim]
        frontier = list(words)
        for _ in range(max_len - 1):
            nxt = []
            for w in frontier:
                for a in prim:
                    if a == pz.inverse[w[-1]]:
                        continue
                    if len(w) >= 2 and w[-1] == a and w[-2] == a and perm_order_of_action(pz, a) <= 3:
                        continue
                    nxt.append(w + (a,))
            words.extend(nxt)
            frontier = nxt
        return words

    def _free_reduce(self, word: Sequence[int]) -> tuple[int, ...]:
        out: list[int] = []
        inv = self.puzzle.inverse
        for a in word:
            if out and inv[out[-1]] == a:
                out.pop()
            else:
                out.append(int(a))
        return tuple(out)

    def _inverse_word(self, word: Sequence[int]) -> tuple[int, ...]:
        inv = self.puzzle.inverse
        return tuple(int(inv[a]) for a in reversed(word))

    def _eval(self, word: Sequence[int]) -> Optional[np.ndarray]:
        """Permutation of a word executed from the home anchor (``None`` if illegal / not a loop)."""
        pz = self.puzzle
        if pz.is_group:
            p = pz.identity
            for a in word:
                p = p[pz.table[a, 0]]
            return p
        pos = self.home
        p = pz.identity
        for a in word:
            if not pz.legal[a, pos]:
                return None
            m = pz.table[a, pos]
            p = p[m]
            pos = int(np.nonzero(m == pos)[0][0])
        return p if pos == self.home else None

    def classify(self, perm: np.ndarray) -> tuple[int, int, str]:
        pz = self.puzzle
        po = pz.piece_of
        src_piece = po[perm]                       # piece that supplied the facet now at each position
        moved_mask = src_piece != po
        moved = set(po[moved_mask].tolist())
        twisted = set(po[(perm != pz.identity) & ~moved_mask].tolist()) - moved
        pp = np.arange(pz.num_pieces)
        pp[po[moved_mask]] = src_piece[moved_mask]
        lens = sorted(len(c) for c in perm_cycles(pp))
        if not moved and twisted:
            kind = f"{len(twisted)}-twist"
        elif lens == [3] and not twisted:
            kind = "pure 3-cycle"
        elif lens == [2] and not twisted:
            kind = "transposition"
        elif lens == [2, 2] and not twisted:
            kind = "double swap"
        elif lens:
            kind = "cycles " + "+".join(map(str, lens)) + (f" & {len(twisted)} twists" if twisted else "")
        else:
            kind = "identity"
        return len(moved), len(twisted), kind

    def _record(self, pool: dict[bytes, MinedMacro], word: Sequence[int], perm: np.ndarray, origin: str) -> None:
        pz = self.puzzle
        word = self._free_reduce(word)
        if not word:
            return
        sup = support(perm)
        if sup == 0:
            return
        key = perm.tobytes()
        old = pool.get(key)
        if old is not None and len(old.word) <= len(word):
            return
        moved, twisted, kind = self.classify(perm)
        pool[key] = MinedMacro(word=word, perm=perm.copy(), support=sup, pieces_moved=moved,
                               pieces_twisted=twisted, kind=kind, origin=origin, order=perm_order(perm),
                               text=" ".join(pz.names_of(word)))

    # -- mining -------------------------------------------------------------------------
    def mine(self, top_k: int = 16) -> list[MinedMacro]:
        pz = self.puzzle
        ident = pz.identity
        words_a = self._reduced_words(self.max_len_a)
        words_b = self._reduced_words(self.max_len_b)
        pool: dict[bytes, MinedMacro] = {}
        prim_keys = {pz.table[a, self.home if not pz.is_group else 0].tobytes() for a in range(pz.num_primitive)}

        if pz.is_group:
            pa = np.stack([self._eval(w) for w in words_a])
            pb = np.stack([self._eval(w) for w in words_b])
            pa_inv, pb_inv = np.argsort(pa, axis=1), np.argsort(pb, axis=1)
            rows = []
            for i in range(len(words_a)):
                t = pa[i][pb]                                  # A then B
                t = t[:, pa_inv[i]]                            # then A^-1
                t = np.take_along_axis(t, pb_inv, axis=1)      # then B^-1
                sup = (t != ident[None, :]).sum(axis=1)
                rows.append(sup)
                self.candidates_examined += len(words_b)
            sup_all = np.stack(rows)
            positive = np.unique(sup_all[sup_all > 0])
            limit = self.max_support if self.max_support is not None else (
                int(positive[min(2, len(positive) - 1)]) if len(positive) else 0)
            ii, jj = np.nonzero((sup_all > 0) & (sup_all <= limit))
            order_idx = np.argsort(sup_all[ii, jj], kind="stable")[: 4000]
            for idx in order_idx:
                i, j = int(ii[idx]), int(jj[idx])
                A, B = words_a[i], words_b[j]
                word = A + B + self._inverse_word(A) + self._inverse_word(B)
                perm = pa[i][pb[j]][pa_inv[i]][pb_inv[j]]
                self._record(pool, word, perm, "commutator")
        else:
            for A in words_a:
                for B in words_b:
                    word = A + B + self._inverse_word(A) + self._inverse_word(B)
                    perm = self._eval(word)
                    self.candidates_examined += 1
                    if perm is not None:
                        self._record(pool, word, perm, "commutator")
            limit = self.max_support if self.max_support is not None else pz.n

        # powers of short words: w^k kills every cycle whose length divides k
        for w in words_a:
            if len(w) > 4:
                continue
            perm = self._eval(w)
            if perm is None:
                continue
            lens = sorted({len(c) for c in perm_cycles(perm)})
            order = math.lcm(*lens) if lens else 1
            for k in sorted({l for l in lens} | {math.lcm(a, b) for a in lens for b in lens}):
                if k <= 1 or k >= order or k * len(w) > self.max_power_len:
                    continue
                pk = perm_power(perm, k)
                self.candidates_examined += 1
                if 0 < support(pk) <= limit:
                    self._record(pool, tuple(w) * k, pk, "power")

        # conjugates X C X^-1 relocate each low-support element without changing its cycle type
        ranked = sorted(pool.values(), key=lambda m: (m.pieces_affected, m.support, len(m.word)))
        if self.max_conj_len > 0:
            setups = self._reduced_words(self.max_conj_len)
            for base_macro in ranked[: max(4, top_k)]:
                for X in setups:
                    word = tuple(X) + base_macro.word + self._inverse_word(X)
                    perm = self._eval(word)
                    self.candidates_examined += 1
                    if perm is not None:
                        self._record(pool, word, perm, "conjugate")

        ranked = sorted((m for m in pool.values() if m.perm.tobytes() not in prim_keys),
                        key=lambda m: (m.pieces_affected, m.support, len(m.word), m.word))
        return ranked[:top_k]

    def promote(self, mined: Sequence[MinedMacro], top_k: int = 4, prefix: str = "m") -> list[Macro]:
        """Turn the best mined sequences into macro-actions (diverse cycle types first)."""
        chosen: list[MinedMacro] = []
        kinds: set[str] = set()
        for m in mined:
            if m.kind not in kinds:
                chosen.append(m)
                kinds.add(m.kind)
        for m in mined:
            if m not in chosen:
                chosen.append(m)
        offset = len(self.puzzle.macros)
        return [Macro(name=f"{prefix}{offset + i + 1}", word=m.word, support=m.support, kind=m.kind)
                for i, m in enumerate(chosen[:top_k])]


def perm_order_of_action(puzzle: Puzzle, action: int) -> int:
    k = puzzle.anchor if puzzle.anchor is not None else 0
    return perm_order(puzzle.table[action, k]) if puzzle.is_group else 2


def mine_and_promote(puzzle: Puzzle, top_k: int = 4, **kwargs) -> tuple[Puzzle, list[MinedMacro]]:
    """Convenience: mine macros and return the puzzle with its action space extended."""
    miner = MacroMiner(puzzle, **kwargs)
    mined = miner.mine(top_k=max(16, top_k * 4))
    return puzzle.with_macros(miner.promote(mined, top_k)), mined
