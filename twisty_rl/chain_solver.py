"""A solver that always finishes: short-word factorisation through the stabiliser chain.

Schreier-Sims tells us *that* a state is solvable; this module turns the chain into moves.  For
every level ``i`` and every point ``gamma`` of the basic orbit it stores a *word* in the puzzle's own
moves whose permutation fixes the first ``i`` base points and carries ``base[i]`` to ``gamma``.  A
state is then solved level by level: look up where facet ``base[i]`` sits, play that word.

The table is built Minkwitz-style: many candidate words are sifted through the chain and each
residue replaces a table entry whenever it is shorter.  Candidates come from (1) all short
products of the generators, (2) products of existing entries and (3) random walks, which guarantee
that every entry gets filled.  The sifting is vectorised over thousands of candidates at a time.

Solutions are far from optimal (think 100+ quarter turns for a 3x3) but exist for *every*
reachable state of *every* puzzle, including the sliding-puzzle groupoids.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .group_theory import PuzzleGroup
from .schema import invert_perm

BIG = np.iinfo(np.int32).max // 4
CACHE_VERSION = 1
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache"


class ChainSolver:
    def __init__(self, group: PuzzleGroup, seed: int = 0):
        self.group = group
        self.puzzle = pz = group.puzzle
        self.n = n = pz.n
        bsgs = group.bsgs
        self.base = np.asarray(bsgs.base, dtype=np.intp)
        self.k = k = len(self.base)
        self.orbit_sizes = np.asarray(bsgs.orbit_sizes, dtype=np.intp)
        self.rng = np.random.default_rng(seed)
        self._id = np.arange(n, dtype=np.intp)
        # generators closed under inversion, each with its word in primitive actions
        gens: dict[bytes, tuple[np.ndarray, tuple[int, ...]]] = {}
        for perm, word in zip(bsgs.generators, group.generator_words):
            for p, w in ((perm, tuple(word)), (invert_perm(perm), self._inverse_word(word))):
                key = p.tobytes()
                if key not in gens or len(w) < len(gens[key][1]):
                    gens[key] = (p, w)
        self.gen_perms = np.stack([g[0] for g in gens.values()]) if gens else np.zeros((0, n), dtype=np.intp)
        self.gen_words = [g[1] for g in gens.values()]
        self.gen_lens = np.asarray([len(w) for w in self.gen_words], dtype=np.int64)
        keys = {p.tobytes(): i for i, p in enumerate(self.gen_perms)}
        self.gen_inverse = np.asarray([keys[invert_perm(p).tobytes()] for p in self.gen_perms], dtype=np.intp)
        # the table
        self.has = np.zeros((k, n), dtype=bool)
        self.tlen = np.full((k, n), BIG, dtype=np.int64)
        self.tperm = np.zeros((k, n, n), dtype=np.int16)
        self.tinv = np.zeros((k, n, n), dtype=np.int16)
        self.words: list[dict[int, tuple[int, ...]]] = [dict() for _ in range(k)]
        for i, b in enumerate(self.base):
            self._store(i, int(b), (), self._id)
        self.candidates = 0
        self.patience = 12
        self.build_seconds = 0.0

    # -- words --------------------------------------------------------------------------------
    def _inverse_word(self, word: Sequence[int]) -> tuple[int, ...]:
        inv = self.puzzle.inverse
        return tuple(int(inv[a]) for a in reversed(word))

    def _word_perm(self, word: Sequence[int]) -> np.ndarray:
        """Gather permutation of a word of primitive actions executed from the home anchor."""
        pz = self.puzzle
        p = self._id
        if pz.is_group:
            tab = pz.table[:, 0]
            for a in word:
                p = p[tab[a]]
            return p
        pos = pz.anchor
        for a in word:
            m = pz.table[a, pos]
            p = p[m]
            pos = int(np.nonzero(m == pos)[0][0])
        return p

    def _store(self, level: int, gamma: int, word: tuple[int, ...], perm: np.ndarray) -> None:
        self.has[level, gamma] = True
        self.tlen[level, gamma] = len(word)
        self.tperm[level, gamma] = perm
        self.tinv[level, gamma] = invert_perm(perm)
        self.words[level][gamma] = word

    def _offer(self, level: int, word: Sequence[int]) -> bool:
        """Reduce ``word`` (an element of ``G^(level)``) and store it, and its inverse, where shorter."""
        word = tuple(self.puzzle.reduce_word(word))
        improved = False
        for w in (word, self._inverse_word(word)):
            perm = self._word_perm(w)
            gamma = int(perm[self.base[level]])
            if len(w) < self.tlen[level, gamma]:
                self._store(level, gamma, w, perm)
                improved = True
        return improved

    # -- sifting ------------------------------------------------------------------------------
    def _sift(self, perms: np.ndarray, lens: np.ndarray, word_of: Callable[[int], tuple[int, ...]],
              max_len: Optional[int] = None) -> int:
        """Sift a batch of group elements; returns the number of table entries improved."""
        B = perms.shape[0]
        self.candidates += B
        T = perms.astype(np.intp, copy=True)
        L = lens.astype(np.int64, copy=True)
        alive = np.ones(B, dtype=bool)
        trace = np.full((B, self.k), -1, dtype=np.int32)
        wins: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(self.k):
            idx = np.nonzero(alive)[0]
            if len(idx) == 0:
                break
            g = T[idx, self.base[i]]
            trace[idx, i] = g
            have = self.has[i, g]
            better = np.nonzero(L[idx] < self.tlen[i, g])[0]
            if len(better):
                order = better[np.argsort(L[idx][better], kind="stable")]
                for j in order:
                    key = (i, int(g[j]))
                    if key not in wins:
                        wins[key] = (int(L[idx[j]]), int(idx[j]))
            alive[idx[~have]] = False
            ci, gi = idx[have], g[have]
            T[ci] = np.take_along_axis(self.tinv[i, gi].astype(np.intp), T[ci], axis=1)
            L[ci] += self.tlen[i, gi]
            done = (T[ci] == self._id).all(axis=1)
            if max_len is not None:
                done |= L[ci] > max_len
            alive[ci[done]] = False
        # reconstruct the winning residues from the (still unmodified) table, then install them
        pending = []
        for (level, _gamma), (_length, c) in wins.items():
            parts: list[int] = []
            for j in range(level - 1, -1, -1):
                parts.extend(self._inverse_word(self.words[j][int(trace[c, j])]))
            parts.extend(word_of(c))
            pending.append((level, parts))
        return sum(self._offer(level, word) for level, word in pending)

    # -- candidate sources --------------------------------------------------------------------
    def _short_words(self, deadline: float, cap: int = 60_000, max_depth: int = 8) -> None:
        """All reduced products of generators by increasing length (sub-sampled beyond ``cap``)."""
        G = len(self.gen_perms)
        perms = self.gen_perms.copy()
        seqs = np.arange(G, dtype=np.int16)[:, None]
        lens = self.gen_lens.copy()
        for _depth in range(max_depth):
            cur = seqs
            self._sift(perms, lens, lambda c: tuple(a for gi in cur[c] for a in self.gen_words[gi]))
            if time.perf_counter() > deadline or self.complete and _depth >= 4:
                break
            last = seqs[:, -1]
            keep = (self.gen_inverse[last][:, None] != np.arange(G)[None, :])           # no back-tracking
            src, gen = np.nonzero(keep)
            if len(src) > cap:
                pick = self.rng.choice(len(src), size=cap, replace=False)
                src, gen = src[pick], gen[pick]
            perms = np.take_along_axis(perms[src], self.gen_perms[gen], axis=1)         # word then generator
            seqs = np.concatenate([seqs[src], gen[:, None].astype(np.int16)], axis=1)
            lens = lens[src] + self.gen_lens[gen]

    def _random_walks(self, batch: int = 2048, steps: int = 24) -> None:
        G = len(self.gen_perms)
        seqs = self.rng.integers(0, G, size=(batch, steps))
        perms = np.tile(self._id, (batch, 1))
        for t in range(steps):
            perms = np.take_along_axis(perms, self.gen_perms[seqs[:, t]], axis=1)
        self._sift(perms, self.gen_lens[seqs].sum(axis=1),
                   lambda c: tuple(a for gi in seqs[c] for a in self.gen_words[gi]))

    def _improve(self, batch: int = 4096, max_len: Optional[int] = None) -> int:
        """Sift products of two table entries (the Minkwitz improvement step)."""
        levels, gammas = np.nonzero(self.has & (self.tlen > 0))
        if len(levels) < 2:
            return 0
        a = self.rng.integers(0, len(levels), size=batch)
        b = self.rng.integers(0, len(levels), size=batch)
        pa = self.tperm[levels[a], gammas[a]].astype(np.intp)
        pb = self.tperm[levels[b], gammas[b]].astype(np.intp)
        perms = np.take_along_axis(pa, pb, axis=1)
        lens = self.tlen[levels[a], gammas[a]] + self.tlen[levels[b], gammas[b]]
        if max_len is not None:
            ok = lens <= max_len
            a, b, perms, lens = a[ok], b[ok], perms[ok], lens[ok]
            if len(a) == 0:
                return 0
        snapshot = [dict(w) for w in self.words]
        return self._sift(perms, lens,
                          lambda c: snapshot[levels[a[c]]][int(gammas[a[c]])] + snapshot[levels[b[c]]][int(gammas[b[c]])],
                          max_len)

    # -- building -----------------------------------------------------------------------------
    @property
    def complete(self) -> bool:
        return bool((self.has.sum(axis=1) == self.orbit_sizes).all())

    def expected_length(self) -> float:
        """Mean solution length over uniformly random states (before final word reduction)."""
        total = 0.0
        for i in range(self.k):
            row = self.tlen[i][self.has[i]]
            total += float(row.mean()) if len(row) else 0.0
        return total

    def max_length(self) -> int:
        return int(sum(int(self.tlen[i][self.has[i]].max()) for i in range(self.k)))

    def build(self, time_budget: float = 6.0, verbose: bool = False) -> "ChainSolver":
        """Fill the table completely, then spend the remaining budget shortening its words."""
        t0 = time.perf_counter()
        deadline = t0 + time_budget
        if self.k == 0:
            return self
        self._short_words(t0 + 0.45 * time_budget)
        guard = 0
        while not self.complete:                       # random elements hit every coset eventually
            self._random_walks()
            guard += 1
            if guard > 2000:
                raise RuntimeError("stabiliser-chain table could not be completed")
        if verbose:
            print(f"  complete after {time.perf_counter() - t0:.2f}s: expected {self.expected_length():.0f}, "
                  f"max {self.max_length()}")
        stale = 0
        while time.perf_counter() < deadline and stale < self.patience:
            cap = int(np.percentile(self.tlen[self.has], 75)) + 2
            stale = 0 if self._improve(max_len=cap) + self._improve() else stale + 1
        self.build_seconds = time.perf_counter() - t0
        if verbose:
            print(f"  built in {self.build_seconds:.2f}s from {self.candidates:,} candidates: "
                  f"expected {self.expected_length():.0f}, max {self.max_length()}")
        return self

    # -- persistence --------------------------------------------------------------------------
    def _cache_key(self) -> str:
        h = hashlib.sha1()
        h.update(f"v{CACHE_VERSION}|{self.puzzle.name}|{self.n}|".encode())
        h.update(self.base.tobytes())
        h.update(self.gen_perms.tobytes())
        return h.hexdigest()[:16]

    def save(self, path: os.PathLike) -> None:
        levels, gammas = np.nonzero(self.has)
        words = [self.words[i][int(g)] for i, g in zip(levels, gammas)]
        offsets = np.cumsum([0] + [len(w) for w in words])
        flat = np.asarray([a for w in words for a in w], dtype=np.int16)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, levels=levels, gammas=gammas, offsets=offsets, flat=flat,
                            build_seconds=self.build_seconds, candidates=self.candidates)
        os.replace(tmp, path)

    def load(self, path: os.PathLike) -> bool:
        try:
            with np.load(path) as z:
                levels, gammas, offsets, flat = z["levels"], z["gammas"], z["offsets"], z["flat"]
                for j, (i, g) in enumerate(zip(levels, gammas)):
                    word = tuple(int(a) for a in flat[offsets[j]:offsets[j + 1]])
                    self._store(int(i), int(g), word, self._word_perm(word))
                self.build_seconds = float(z["build_seconds"])
                self.candidates = int(z["candidates"])
        except (OSError, KeyError, ValueError, IndexError):
            return False
        return self.complete

    @classmethod
    def cached(cls, group: PuzzleGroup, time_budget: float = 6.0, cache_dir: Optional[os.PathLike] = DEFAULT_CACHE,
               seed: int = 0) -> "ChainSolver":
        """Load the table from disk when this exact chain was built before, else build and store it."""
        solver = cls(group, seed)
        path = None if cache_dir is None else Path(cache_dir) / f"chain-{group.puzzle.name}-{solver._cache_key()}.npz"
        if path is not None and path.exists() and solver.load(path):
            return solver
        solver = cls(group, seed).build(time_budget)
        if path is not None:
            try:
                solver.save(path)
            except OSError:
                pass                                   # a read-only checkout only costs the rebuild next time
        return solver

    # -- solving ------------------------------------------------------------------------------
    def solve(self, P: np.ndarray, tries: int = 48) -> list[int]:
        """Primitive actions that solve permutation state ``P`` (which must be reachable).

        Solution lengths vary a lot from state to state, so besides ``P`` itself ``tries`` neighbours
        (``P`` after a few random moves) are solved as well and the shortest total is returned.
        """
        pz = self.puzzle
        P = np.asarray(P, dtype=np.intp)
        best = self._solve_once(P)
        for _ in range(max(0, tries)):
            if len(best) <= 1:
                break
            Q, prefix = P, []
            for _step in range(int(self.rng.integers(1, 4))):
                legal = [a for a in pz.legal_actions(Q) if a < pz.num_primitive]
                a = int(self.rng.choice(legal))
                Q = pz.apply(Q, a)
                prefix.append(a)
            cand = pz.reduce_word(prefix + self._solve_once(Q))
            if len(cand) < len(best):
                best = cand
        return best

    def _solve_once(self, P: np.ndarray) -> list[int]:
        pz = self.puzzle
        word: list[int] = []
        if not pz.is_group:
            k = pz.anchor_pos(P)
            if k not in self.group.anchor_words:
                raise ValueError("state is not reachable")
            back = self._inverse_word(self.group.anchor_words[k])
            for a in back:
                P = pz.apply(P, a)
            word.extend(back)
        for i, b in enumerate(self.base):
            gamma = int(np.nonzero(P == b)[0][0])
            if not self.has[i, gamma]:
                raise ValueError("state is not reachable")
            word.extend(self.words[i][gamma])
            P = P[self.tperm[i, gamma].astype(np.intp)]
        if not np.array_equal(P, self._id):
            raise ValueError("state is not reachable")
        return pz.reduce_word(word)
