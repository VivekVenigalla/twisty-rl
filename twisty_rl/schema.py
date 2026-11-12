"""Declarative puzzle schema and its compiled runtime form.

Conventions used everywhere in TwistyRL
---------------------------------------
* A puzzle has ``n`` *facets* (stickers, tiles, balls, register slots ...).  Facet ``i``
  also names the *position* it occupies in the solved state.
* A **permutation state** ``P`` is an integer array of length ``n`` where ``P[i]`` is the
  facet currently sitting at position ``i``.  The solved state is ``arange(n)``.
* A move is stored in **gather form**: applying move ``m`` yields ``P' = P[m]``, i.e. the
  content of position ``m[i]`` travels to position ``i``.  Applying ``a`` and then ``b`` is
  the single gather permutation ``a[b]``.
* The **colour state** is ``C = solved_state[P]``.  Facets with equal colour are
  indistinguishable ("target equivalence classes"); a state is solved iff ``C`` equals
  ``solved_state``.
* *Conditional* generators (sliding puzzles) depend on where an **anchor** facet (the
  blank) currently is; they are given as one gather permutation per anchor position.
"""
from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


_REPEAT = re.compile(r"^(.+?)(\d+)('?)$")


# ----------------------------------------------------------------------------------
# Small permutation helpers shared by the schema and the puzzle builders
# ----------------------------------------------------------------------------------
def is_permutation(p: Sequence[int], n: int) -> bool:
    arr = np.asarray(p)
    if arr.shape != (n,):
        return False
    seen = np.zeros(n, dtype=bool)
    ok = (arr >= 0).all() and (arr < n).all()
    if not ok:
        return False
    seen[arr] = True
    return bool(seen.all())


def perm_from_cycles(n: int, cycles: Sequence[Sequence[int]]) -> np.ndarray:
    """Cycle ``(a b c)`` means: content of ``a`` moves to ``b``, ``b`` to ``c``, ``c`` to ``a``."""
    p = np.arange(n, dtype=np.intp)
    seen: set[int] = set()
    for cyc in cycles:
        for k, a in enumerate(cyc):
            if not 0 <= a < n:
                raise ValueError(f"cycle element {a} out of range [0, {n})")
            if a in seen:
                raise ValueError(f"element {a} appears twice in cycle notation")
            seen.add(a)
            p[cyc[(k + 1) % len(cyc)]] = a
    return p


def invert_perm(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(p.shape[0], dtype=p.dtype)
    return inv


def default_palette(k: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(k):
        h = (i * 0.61803398875) % 1.0
        s = 0.55 + 0.35 * ((i * 7) % 3) / 2.0
        v = 0.95 - 0.25 * ((i * 5) % 4) / 3.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


# ----------------------------------------------------------------------------------
# Declarative pydantic schema
# ----------------------------------------------------------------------------------
class GeneratorSpec(BaseModel):
    """One generator ``g_i``.  Exactly one of ``perm`` / ``cycles`` / ``variants`` is given."""

    model_config = ConfigDict(extra="forbid")
    name: str
    perm: Optional[list[int]] = None
    cycles: Optional[list[list[int]]] = None
    variants: Optional[dict[int, list[int]]] = None
    cost: float = 1.0

    @model_validator(mode="after")
    def _one_form(self) -> "GeneratorSpec":
        given = sum(x is not None for x in (self.perm, self.cycles, self.variants))
        if given != 1:
            raise ValueError(f"generator {self.name!r}: give exactly one of perm/cycles/variants")
        if self.cost <= 0:
            raise ValueError("generator cost must be positive")
        return self

    @property
    def conditional(self) -> bool:
        return self.variants is not None


class PieceSpec(BaseModel):
    """A rigid physical piece: the facets listed always travel together."""

    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str = "piece"
    facets: list[int]


class SymmetrySpec(BaseModel):
    """Whole-puzzle symmetry: ``sym(C)[i] = color_map[C[perm[i]]]`` maps solved to solved."""

    model_config = ConfigDict(extra="forbid")
    name: str
    perm: list[int]
    color_map: list[int]


class MoveAnimation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    axis: tuple[float, float, float]
    angle: float
    facets: list[int]


class Geometry(BaseModel):
    """Drawable geometry: one 3D polygon and one 2D (net) polygon per facet."""

    model_config = ConfigDict(extra="forbid")
    kind: str = "planar"
    polygons3d: list[list[tuple[float, float, float]]]
    polygons2d: list[list[tuple[float, float]]]
    body3d: list[list[tuple[float, float, float]]] = Field(default_factory=list)
    double_sided: bool = False
    animations: dict[str, MoveAnimation] = Field(default_factory=dict)


class PuzzleSpec(BaseModel):
    """Declarative definition of a permutation puzzle ``(N, colouring, G = <g_1..g_k>)``."""

    model_config = ConfigDict(extra="forbid")
    name: str
    family: str = "permutation"
    description: str = ""
    n: int = Field(gt=0)
    solved_state: list[int]
    generators: list[GeneratorSpec]
    pieces: list[PieceSpec] = Field(default_factory=list)
    symmetries: list[SymmetrySpec] = Field(default_factory=list)
    anchor: Optional[int] = None
    palette: list[tuple[int, int, int]] = Field(default_factory=list)
    color_names: list[str] = Field(default_factory=list)
    labels: Optional[list[str]] = None
    geometry: Optional[Geometry] = None
    known_order: Optional[int] = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "PuzzleSpec":
        n = self.n
        if len(self.solved_state) != n:
            raise ValueError("solved_state must have length n")
        if min(self.solved_state) < 0:
            raise ValueError("colour classes must be non-negative")
        ncol = max(self.solved_state) + 1
        if not self.generators:
            raise ValueError("at least one generator is required")
        names = [g.name for g in self.generators]
        if len(set(names)) != len(names):
            raise ValueError("generator names must be unique")
        any_cond = False
        for g in self.generators:
            if g.perm is not None and not is_permutation(g.perm, n):
                raise ValueError(f"generator {g.name!r} is not a permutation of size {n}")
            if g.cycles is not None:
                perm_from_cycles(n, g.cycles)
            if g.variants is not None:
                any_cond = True
                for pos, perm in g.variants.items():
                    if not 0 <= pos < n:
                        raise ValueError(f"generator {g.name!r}: anchor position {pos} out of range")
                    if not is_permutation(perm, n):
                        raise ValueError(f"generator {g.name!r}: variant {pos} is not a permutation")
        if any_cond:
            if self.anchor is None:
                raise ValueError("conditional generators require an anchor facet")
            if self.solved_state.count(self.solved_state[self.anchor]) != 1:
                raise ValueError("the anchor facet must have a unique colour")
        if self.anchor is not None and not 0 <= self.anchor < n:
            raise ValueError("anchor out of range")
        used: set[int] = set()
        for piece in self.pieces:
            for f in piece.facets:
                if not 0 <= f < n:
                    raise ValueError(f"piece {piece.name!r}: facet {f} out of range")
                if f in used:
                    raise ValueError(f"facet {f} belongs to two pieces")
                used.add(f)
        for sym in self.symmetries:
            if not is_permutation(sym.perm, n):
                raise ValueError(f"symmetry {sym.name!r} is not a permutation")
            if not is_permutation(sym.color_map, ncol):
                raise ValueError(f"symmetry {sym.name!r}: colour map is not a bijection")
        if len(self.palette) < ncol:
            self.palette = list(self.palette) + default_palette(ncol)[len(self.palette):]
        if len(self.color_names) < ncol:
            self.color_names = list(self.color_names) + [f"c{i}" for i in range(len(self.color_names), ncol)]
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("labels must have one entry per facet")
        if self.geometry is not None:
            if len(self.geometry.polygons3d) != n or len(self.geometry.polygons2d) != n:
                raise ValueError("geometry must provide one polygon per facet")
        return self

    @property
    def num_colors(self) -> int:
        return max(self.solved_state) + 1

    def generator_perm(self, g: GeneratorSpec) -> np.ndarray:
        if g.perm is not None:
            return np.asarray(g.perm, dtype=np.intp)
        if g.cycles is not None:
            return perm_from_cycles(self.n, g.cycles)
        raise ValueError(f"generator {g.name!r} is conditional")


# ----------------------------------------------------------------------------------
# Z_n extension helper: registers mod m compiled to their regular permutation representation
# ----------------------------------------------------------------------------------
class ModularMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    increments: dict[int, int]
    cost: float = 1.0


class ModularSpec(BaseModel):
    """A puzzle on ``Z_{m_1} x ... x Z_{m_r}`` whose moves add fixed vectors.

    Register ``r`` with modulus ``m`` becomes ``m`` facets (slots); a single "hand" token
    marks the current value and every move acts as a product of disjoint ``m``-cycles.
    This is the regular representation of the abelian extension inside ``S_N``.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    moduli: list[int]
    moves: list[ModularMove]
    register_names: list[str] = Field(default_factory=list)

    def offsets(self) -> list[int]:
        out, acc = [], 0
        for m in self.moduli:
            out.append(acc)
            acc += m
        return out

    def to_puzzle_spec(self, **extra) -> PuzzleSpec:
        offs = self.offsets()
        n = sum(self.moduli)
        solved = [0] * n
        for o in offs:
            solved[o] = 1
        gens = []
        for mv in self.moves:
            perm = np.arange(n, dtype=np.intp)
            for reg, inc in mv.increments.items():
                m, o = self.moduli[reg], offs[reg]
                for k in range(m):
                    perm[o + (k + inc) % m] = o + k
            gens.append(GeneratorSpec(name=mv.name, perm=perm.tolist(), cost=mv.cost))
        names = self.register_names or [f"r{i}" for i in range(len(self.moduli))]
        pieces = [
            PieceSpec(name=names[i], kind="register", facets=list(range(o, o + m)))
            for i, (o, m) in enumerate(zip(offs, self.moduli))
        ]
        kwargs = dict(
            name=self.name, family="modular", description=self.description, n=n,
            solved_state=solved, generators=gens, pieces=pieces,
            palette=[(46, 48, 58), (255, 214, 64)], color_names=["blank", "hand"],
        )
        kwargs.update(extra)
        return PuzzleSpec(**kwargs)

    def decode(self, perm_state: np.ndarray) -> np.ndarray:
        """Register values of a permutation state (position of each hand token)."""
        vals = []
        for o, m in zip(self.offsets(), self.moduli):
            seg = perm_state[o:o + m]
            vals.append(int(np.nonzero(seg == o)[0][0]))
        return np.asarray(vals)


# ----------------------------------------------------------------------------------
# Compiled runtime puzzle
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Macro:
    """A promoted macro-action: a word over primitive actions."""

    name: str
    word: tuple[int, ...]
    support: int = 0
    kind: str = "macro"


ACTION_GENERATOR, ACTION_INVERSE, ACTION_MACRO = 0, 1, 2


class Puzzle:
    """Compiled, numpy-backed form of a :class:`PuzzleSpec` (plus optional macro-actions).

    Attributes
    ----------
    table : int array ``[A, K, N]``
        Gather permutation of action ``a`` when the anchor sits at position ``k``
        (``K == 1`` for ordinary group puzzles).
    legal : bool array ``[A, K]``
    inverse : int array ``[A]`` -- index of the inverse action.
    cost : float array ``[A]``
    """

    def __init__(self, spec: PuzzleSpec, macros: Sequence[Macro] = ()):
        self.spec = spec
        self.name = spec.name
        self.n = n = spec.n
        self.solved_colors = np.asarray(spec.solved_state, dtype=np.intp)
        self.num_colors = spec.num_colors
        self.anchor = spec.anchor if any(g.conditional for g in spec.generators) else None
        self.K = n if self.anchor is not None else 1
        self.identity = np.arange(n, dtype=np.intp)

        tables: list[np.ndarray] = []
        legal: list[np.ndarray] = []
        names: list[str] = []
        kinds: list[int] = []
        costs: list[float] = []
        K = self.K
        for g in spec.generators:
            if g.conditional:
                tab = np.tile(self.identity, (K, 1))
                leg = np.zeros(K, dtype=bool)
                for pos, perm in g.variants.items():
                    tab[pos] = np.asarray(perm, dtype=np.intp)
                    leg[pos] = True
            else:
                tab = np.tile(spec.generator_perm(g), (K, 1))
                leg = np.ones(K, dtype=bool)
            tables.append(tab); legal.append(leg); names.append(g.name)
            kinds.append(ACTION_GENERATOR); costs.append(float(g.cost))
        # add inverses of unconditional generators when they are not already present
        base_count = len(tables)
        for gi in range(base_count):
            g = spec.generators[gi]
            if g.conditional:
                continue
            inv = invert_perm(tables[gi][0])
            if any((not spec.generators[j].conditional) and np.array_equal(tables[j][0], inv)
                   for j in range(base_count)):
                continue
            tables.append(np.tile(inv, (K, 1))); legal.append(np.ones(K, dtype=bool))
            names.append(g.name + "'"); kinds.append(ACTION_INVERSE); costs.append(float(g.cost))
        self.num_primitive = len(tables)
        self._prim_table = np.stack(tables)
        self._prim_legal = np.stack(legal)
        self._prim_inv_table = np.argsort(self._prim_table, axis=-1)
        words: list[tuple[int, ...]] = [(a,) for a in range(self.num_primitive)]
        prim_inverse = self._find_inverses(self._prim_table, self._prim_legal)

        self.macros: tuple[Macro, ...] = tuple(macros)
        for mac in self.macros:
            forms = [(mac.name, tuple(mac.word))]
            inv_word = tuple(int(prim_inverse[a]) for a in reversed(mac.word))
            tab, leg, _ = self._compose(mac.word)
            itab, ileg, _ = self._compose(inv_word)
            if not (np.array_equal(tab, itab) and np.array_equal(leg, ileg)):
                forms.append((mac.name + "'", inv_word))
            for nm, word in forms:
                t, l, _ = self._compose(word)
                tables.append(t); legal.append(l); names.append(nm); kinds.append(ACTION_MACRO)
                costs.append(float(sum(costs[a] for a in word)))
                words.append(tuple(word))

        self.table = np.stack(tables).astype(np.intp)
        self.legal = np.stack(legal)
        self.action_names = names
        self.action_kind = np.asarray(kinds, dtype=np.intp)
        self.cost = np.asarray(costs, dtype=np.float64)
        self.action_words = words
        self.num_actions = len(names)
        self.inverse = self._find_inverses(self.table, self.legal)
        self._name_to_action = {nm: i for i, nm in enumerate(names)}

        piece_of = -np.ones(n, dtype=np.intp)
        piece_kind: list[str] = []
        for pi, piece in enumerate(spec.pieces):
            piece_of[piece.facets] = pi
            piece_kind.append(piece.kind)
        nxt = len(spec.pieces)
        for f in range(n):
            if piece_of[f] < 0:
                piece_of[f] = nxt
                piece_kind.append("facet")
                nxt += 1
        self.piece_of = piece_of
        self.piece_kind = piece_kind
        self.num_pieces = nxt
        self._symmetries: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
        self._reduction = None

    # -- construction helpers ---------------------------------------------------------
    def _anchor_dest(self, prim: int, k: int) -> int:
        """Position of the anchor after primitive action ``prim`` applied with anchor at ``k``."""
        return int(self._prim_inv_table[prim, k if self.K > 1 else 0, k]) if self.K > 1 else 0

    def _compose(self, word: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K, n = self.K, self.n
        tab = np.tile(self.identity, (K, 1))
        leg = np.ones(K, dtype=bool)
        end = np.arange(K, dtype=np.intp)
        for k in range(K):
            pos = k
            perm = self.identity
            for a in word:
                if not self._prim_legal[a, pos]:
                    leg[k] = False
                    break
                perm = perm[self._prim_table[a, pos]]
                pos = self._anchor_dest(a, pos)
            if leg[k]:
                tab[k] = perm
                end[k] = pos
            else:
                tab[k] = self.identity
        return tab, leg, end

    def _find_inverses(self, table: np.ndarray, legal: np.ndarray) -> np.ndarray:
        A, K, n = table.shape
        inv_idx = -np.ones(A, dtype=np.intp)
        if K == 1:
            keys = {table[a, 0].tobytes(): a for a in range(A - 1, -1, -1)}
            for a in range(A):
                inv_idx[a] = keys.get(invert_perm(table[a, 0]).tobytes(), -1)
        else:
            inv_tab = np.argsort(table, axis=-1)
            for a in range(A):
                for b in range(A):
                    ok = True
                    for k in range(K):
                        if not legal[a, k]:
                            continue
                        k2 = int(inv_tab[a, k, k])
                        if not legal[b, k2] or not np.array_equal(table[a, k][table[b, k2]], self.identity):
                            ok = False
                            break
                    if ok:
                        inv_idx[a] = b
                        break
        if (inv_idx < 0).any():
            bad = [i for i in range(A) if inv_idx[i] < 0]
            raise ValueError(f"puzzle {self.name!r}: actions {bad} have no inverse in the action set")
        return inv_idx

    # -- basic queries ----------------------------------------------------------------
    @property
    def is_group(self) -> bool:
        return self.K == 1

    @property
    def perms(self) -> np.ndarray:
        """``[A, N]`` gather permutations (group puzzles only)."""
        if not self.is_group:
            raise ValueError("perms is only defined for unconditional (group) puzzles")
        return self.table[:, 0]

    @property
    def primitive_generators(self) -> list[np.ndarray]:
        return [self._prim_table[a, 0] for a in range(self.num_primitive)]

    def action_index(self, name: str) -> int:
        return self._name_to_action[name]

    def parse_moves(self, text: str | Sequence[str]) -> list[int]:
        toks = text.split() if isinstance(text, str) else list(text)
        out: list[int] = []
        for tok in toks:
            if tok in self._name_to_action:
                out.append(self._name_to_action[tok])
            elif tok.endswith("'") and tok[:-1] in self._name_to_action:
                out.append(int(self.inverse[self._name_to_action[tok[:-1]]]))
            else:
                m = _REPEAT.match(tok)                      # U2, U2', R3 ...
                base = m and (m.group(1) + m.group(3))
                if base and base in self._name_to_action:
                    out.extend([self._name_to_action[base]] * int(m.group(2)))
                elif m and m.group(3) and m.group(1) in self._name_to_action:
                    out.extend([int(self.inverse[self._name_to_action[m.group(1)]])] * int(m.group(2)))
                else:
                    raise KeyError(f"unknown move {tok!r} for puzzle {self.name!r}")
        return out

    def format_moves(self, actions: Sequence[int | str]) -> list[str]:
        """Human notation: runs of one move are written once with a count (``U U`` -> ``U2``)."""
        names = [a if isinstance(a, str) else self.action_names[a] for a in actions]
        out: list[str] = []
        i = 0
        while i < len(names):
            j = i
            while j + 1 < len(names) and names[j + 1] == names[i]:
                j += 1
            count, name = j - i + 1, names[i]
            half = False                               # X2' is the same turn as X2 when X has order 4
            if count > 1 and self.is_group and name in self._name_to_action:
                perm = self.table[self._name_to_action[name], 0]
                q = perm
                for _ in range(2 * count - 1):
                    q = q[perm]
                half = bool(np.array_equal(q, self.identity))
            if count == 1:
                out.append(name)
            elif name.endswith("'") and not half:
                out.append(f"{name[:-1]}{count}'")
            elif name.endswith("'"):
                out.append(f"{name[:-1]}{count}")
            else:
                out.append(f"{name}{count}")
            i = j + 1
        return out

    def reduce_word(self, word: Sequence[int]) -> list[int]:
        """Shorten a word without changing its effect.

        Group puzzles: powers of one generator are merged (also across moves that commute with it)
        and written with the fewest turns.  Conditional puzzles: adjacent inverse pairs cancel.
        """
        word = self.expand_to_primitive(word)
        inv = self.inverse
        if not self.is_group:
            out: list[int] = []
            for a in word:
                if out and inv[out[-1]] == a:
                    out.pop()
                else:
                    out.append(int(a))
            return out
        if self._reduction is None:
            P = self.num_primitive
            perms = self._prim_table[:, 0]
            canon = np.minimum(np.arange(P), inv[:P])
            order = np.ones(P, dtype=np.intp)
            for a in range(P):
                q, k = perms[a], 1
                while not np.array_equal(q, self.identity):
                    q, k = q[perms[a]], k + 1
                order[a] = k
            commute = np.array([[np.array_equal(perms[a][perms[b]], perms[b][perms[a]]) for b in range(P)]
                                for a in range(P)])
            self._reduction = (canon, order, commute)
        canon, order, commute = self._reduction
        stack: list[list[int]] = []
        for a in word:
            g = int(canon[a])
            e = 1 if a == g else -1
            j = len(stack) - 1
            while j >= 0 and stack[j][0] != g and commute[stack[j][0], g]:
                j -= 1
            if j >= 0 and stack[j][0] == g:
                stack[j][1] = (stack[j][1] + e) % int(order[g])
                if stack[j][1] == 0:
                    del stack[j]
            else:
                stack.append([g, e % int(order[g])])
        out = []
        for g, e in stack:
            n = int(order[g])
            out.extend([g] * e if e <= n // 2 else [int(inv[g])] * (n - e))
        return out

    def anchor_pos(self, P: np.ndarray) -> int:
        return int(np.nonzero(P == self.anchor)[0][0]) if self.anchor is not None else 0

    def apply(self, P: np.ndarray, action: int) -> np.ndarray:
        k = self.anchor_pos(P)
        if not self.legal[action, k]:
            raise ValueError(f"action {self.action_names[action]!r} is illegal in this state")
        return P[self.table[action, k]]

    def apply_word(self, P: np.ndarray, word: Sequence[int]) -> np.ndarray:
        for a in word:
            P = self.apply(P, a)
        return P

    def legal_actions(self, P: np.ndarray) -> np.ndarray:
        return np.nonzero(self.legal[:, self.anchor_pos(P)])[0]

    def colors(self, P: np.ndarray) -> np.ndarray:
        return self.solved_colors[P]

    def is_solved(self, P: np.ndarray) -> bool:
        return bool((self.solved_colors[P] == self.solved_colors).all())

    def scramble(self, rng: np.random.Generator, depth: int, primitive_only: bool = True):
        P = self.identity.copy()
        actions: list[int] = []
        prev = -1
        hi = self.num_primitive if primitive_only else self.num_actions
        for _ in range(depth):
            cand = [a for a in self.legal_actions(P) if a < hi and (prev < 0 or a != self.inverse[prev])]
            if not cand:
                cand = [a for a in self.legal_actions(P) if a < hi]
            a = int(rng.choice(cand))
            P = self.apply(P, a)
            actions.append(a)
            prev = a
        return P, actions

    def expand_to_primitive(self, actions: Sequence[int]) -> list[int]:
        out: list[int] = []
        for a in actions:
            out.extend(self.action_words[a])
        return out

    def names_of(self, actions: Sequence[int]) -> list[str]:
        return [self.action_names[a] for a in actions]

    def compose_word(self, word: Sequence[int]):
        """Compose a word of (possibly macro) actions -> ``(table[K,N], legal[K], end_anchor[K])``."""
        return self._compose(self.expand_to_primitive(word))

    def with_macros(self, macros: Sequence[Macro]) -> "Puzzle":
        """Return a new puzzle whose action space is extended by ``macros`` (words are primitive)."""
        existing = {m.word for m in self.macros}
        merged = list(self.macros) + [m for m in macros if tuple(m.word) not in existing]
        return Puzzle(self.spec, merged)

    # -- symmetries -------------------------------------------------------------------
    def valid_symmetries(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Symmetries ``(pos_perm, color_map)`` that preserve the solved state and the action set."""
        if self._symmetries is not None:
            return self._symmetries
        out: list[tuple[np.ndarray, np.ndarray]] = []
        ncol = self.num_colors
        out.append((self.identity.copy(), np.arange(ncol, dtype=np.intp)))
        if self.is_group:
            prim = self._prim_table[:, 0]
            keys = {prim[a].tobytes() for a in range(self.num_primitive)}
            for sym in self.spec.symmetries:
                rho = np.asarray(sym.perm, dtype=np.intp)
                pi = np.asarray(sym.color_map, dtype=np.intp)
                if np.array_equal(rho, self.identity) and np.array_equal(pi, np.arange(ncol)):
                    continue
                if not np.array_equal(pi[self.solved_colors[rho]], self.solved_colors):
                    continue
                rho_inv = invert_perm(rho)
                # need m_a[rho] == rho[m_a'] for some action a'  <=>  rho_inv[m_a[rho]] is an action
                if all(rho_inv[prim[a][rho]].tobytes() in keys for a in range(self.num_primitive)):
                    out.append((rho, pi))
        self._symmetries = out
        return out

    def __repr__(self) -> str:
        return (f"Puzzle({self.name!r}, n={self.n}, actions={self.num_actions}, "
                f"colors={self.num_colors}, conditional={not self.is_group})")
