"""Out-of-the-box puzzle definitions.

Polyhedral: NxN cubes (2x2, 3x3, 4x4 ...), Pyraminx, Skewb, Megaminx.
Combinatorial: (n^2-1)-sliding puzzles (15-puzzle), pancake graph, transposition star graph,
Hungarian Rings, Lights Out and Rubik's Clock (both as Z_n extensions).

Every factory returns a declarative :class:`PuzzleSpec`; :func:`get_puzzle` compiles it.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numpy as np

from .geometry import MoveDef, PolyhedralBuilder, planar_geometry, rect, regular_polygon
from .schema import GeneratorSpec, ModularMove, ModularSpec, PieceSpec, Puzzle, PuzzleSpec, SymmetrySpec

WHITE, YELLOW, GREEN, BLUE = (245, 245, 245), (255, 213, 0), (0, 158, 96), (0, 81, 186)
RED, ORANGE = (196, 30, 58), (255, 120, 20)


# ----------------------------------------------------------------------------------
# Polyhedral puzzles
# ----------------------------------------------------------------------------------
_CUBE_FACES = [("U", (0, 1, 0), WHITE), ("D", (0, -1, 0), YELLOW), ("F", (0, 0, 1), GREEN),
               ("B", (0, 0, -1), BLUE), ("R", (1, 0, 0), RED), ("L", (-1, 0, 0), ORANGE)]


def cube(n: int = 3) -> PuzzleSpec:
    """NxNxN cube.  The 2x2 uses <U, R, F> (DBL corner fixed); larger cubes use all outer/wide turns."""
    if n < 2:
        raise ValueError("cube size must be >= 2")
    cuts = [(axis, -1 + 2 * k / n) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)) for k in range(1, n)]
    moves: list[MoveDef] = []
    if n == 2:
        for name, axis, _ in _CUBE_FACES:
            if name in ("U", "R", "F"):
                moves.append(MoveDef(name, axis, 0.0, 4))
    else:
        for j in range(1, n // 2 + 1):
            for name, axis, _ in _CUBE_FACES:
                label = name if j == 1 else (f"{name}w" if j == 2 else f"{j}{name}w")
                moves.append(MoveDef(label, axis, 1 - 2 * j / n, 4))
    known = {2: 3_674_160, 3: 43_252_003_274_489_856_000,
             4: math.factorial(8) * 3 ** 7 * math.factorial(24) ** 2 // 2}.get(n)
    return PolyhedralBuilder(
        name=f"cube{n}", kind="cube",
        normals=[f[1] for f in _CUBE_FACES], offsets=[1.0] * 6,
        face_names=[f[0] for f in _CUBE_FACES], face_colors=[f[2] for f in _CUBE_FACES],
        cuts=cuts, moves=moves, net_root=2, known_order=known,
        description=f"{n}x{n}x{n} Rubik's cube (quarter-turn generators)",
    ).build()


def pyraminx() -> PuzzleSpec:
    s2, s6 = math.sqrt(2.0), math.sqrt(6.0)
    verts = {"U": (0.0, 3.0, 0.0), "L": (-s6, -1.0, s2), "R": (s6, -1.0, s2), "B": (0.0, -1.0, -2 * s2)}
    # face opposite a vertex: opposite U -> D, opposite B -> F, opposite L -> right face, opposite R -> left
    faces = [("F", "B", GREEN), ("R", "L", BLUE), ("L", "R", RED), ("D", "U", YELLOW)]
    normals = [tuple(-np.asarray(verts[opp]) / 3.0) for _, opp, _ in faces]
    cuts = [(verts[v], d) for v in verts for d in (5.0 / 3.0, 1.0 / 3.0)]
    moves = [MoveDef(v, verts[v], 1.0 / 3.0, 3) for v in verts]
    moves += [MoveDef(v.lower(), verts[v], 5.0 / 3.0, 3) for v in verts]
    return PolyhedralBuilder(
        name="pyraminx", kind="tetrahedron", normals=normals, offsets=[1.0] * 4,
        face_names=[f[0] for f in faces], face_colors=[f[2] for f in faces],
        cuts=cuts, moves=moves, net_root=0, known_order=75_582_720,
        description="Pyraminx with layer turns U L R B and trivial tips u l r b",
    ).build()


def skewb() -> PuzzleSpec:
    cuts = [((1, 1, 1), 0.0), ((1, 1, -1), 0.0), ((1, -1, 1), 0.0), ((-1, 1, 1), 0.0)]
    # WCA axes: every move turns the half around a corner that excludes UFR, so UFR never moves.
    moves = [MoveDef("R", (1, -1, -1), 0.0, 3), MoveDef("U", (-1, 1, -1), 0.0, 3),
             MoveDef("L", (-1, -1, 1), 0.0, 3), MoveDef("B", (-1, -1, -1), 0.0, 3)]
    return PolyhedralBuilder(
        name="skewb", kind="cube",
        normals=[f[1] for f in _CUBE_FACES], offsets=[1.0] * 6,
        face_names=[f[0] for f in _CUBE_FACES], face_colors=[f[2] for f in _CUBE_FACES],
        cuts=cuts, moves=moves, net_root=2, known_order=3_149_280,
        description="Skewb (deep-cut corner-turning cube, WCA move axes)",
    ).build()


def megaminx() -> PuzzleSpec:
    theta = math.acos(1.0 / math.sqrt(5.0))
    upper = ["F", "R", "BR", "BL", "L"]
    lower = ["B", "DBL", "DL", "DR", "DBR"]          # antipodes of the upper ring, in order
    normals = [(0.0, 1.0, 0.0)]
    names = ["U"]
    for k, nm in enumerate(upper):
        phi = 2 * math.pi * k / 5
        normals.append((math.sin(theta) * math.sin(phi), math.cos(theta), math.sin(theta) * math.cos(phi)))
        names.append(nm)
    normals.append((0.0, -1.0, 0.0))
    names.append("D")
    for k, nm in enumerate(lower):
        x, y, z = normals[1 + k]
        normals.append((-x, -y, -z))
        names.append(nm)
    colors = [WHITE, GREEN, RED, BLUE, YELLOW, (128, 0, 160), (150, 150, 150), (120, 220, 120),
              (255, 150, 190), (250, 240, 180), (100, 190, 255), ORANGE]
    depth = 0.775                                     # megaminx regime: shallow face cuts
    return PolyhedralBuilder(
        name="megaminx", kind="dodecahedron", normals=normals, offsets=[1.0] * 12,
        face_names=names, face_colors=colors,
        cuts=[(nrm, depth) for nrm in normals],
        moves=[MoveDef(nm, nrm, depth, 5) for nm, nrm in zip(names, normals)], net_root=1,
        known_order=math.factorial(30) * math.factorial(20) * 2 ** 27 * 3 ** 19,
        description="Megaminx (face-turning dodecahedron)",
    ).build()


# ----------------------------------------------------------------------------------
# Combinatorial / Cayley-graph puzzles
# ----------------------------------------------------------------------------------
def sliding(rows: int = 4, cols: int = 4) -> PuzzleSpec:
    """(rows*cols - 1)-puzzle.  Moves are named after the direction the *blank* travels."""
    n = rows * cols
    blank = n - 1
    gens = []
    for name, (dr, dc) in (("U", (-1, 0)), ("D", (1, 0)), ("L", (0, -1)), ("R", (0, 1))):
        variants: dict[int, list[int]] = {}
        for pos in range(n):
            r, c = divmod(pos, cols)
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < rows and 0 <= c2 < cols:
                perm = list(range(n))
                tgt = r2 * cols + c2
                perm[pos], perm[tgt] = tgt, pos
                variants[pos] = perm
        gens.append(GeneratorSpec(name=name, variants=variants))
    palette = []
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        palette.append((int(60 + 180 * t), int(200 - 110 * t), int(235 - 150 * abs(t - 0.5))))
    palette.append((18, 18, 22))
    polys = [rect(c + 0.04, -r - 0.96, c + 0.96, -r - 0.04) for r in range(rows) for c in range(cols)]
    return PuzzleSpec(
        name=f"sliding{rows}x{cols}" if (rows, cols) != (4, 4) else "fifteen",
        family="sliding", description=f"{n - 1}-puzzle on a {rows}x{cols} board (groupoid; blank is the anchor)",
        n=n, solved_state=list(range(n)), generators=gens, anchor=blank,
        pieces=[PieceSpec(name=f"tile{i + 1}" if i != blank else "blank", kind="tile", facets=[i]) for i in range(n)],
        palette=palette, color_names=[str(i + 1) for i in range(n - 1)] + ["blank"],
        labels=[str(i + 1) for i in range(n - 1)] + [""],
        geometry=planar_geometry(polys, "grid"),
        known_order=math.factorial(n) // 2 if min(rows, cols) >= 2 else n,
        metadata={"rows": rows, "cols": cols},
    )


def fifteen() -> PuzzleSpec:
    return sliding(4, 4)


def pancake(n: int = 8) -> PuzzleSpec:
    """Pancake graph P_n: Cayley graph of S_n under prefix reversals."""
    gens = []
    for k in range(2, n + 1):
        perm = list(range(n))
        perm[:k] = perm[:k][::-1]
        gens.append(GeneratorSpec(name=f"F{k}", perm=perm))
    palette = [(int(250 - 150 * i / max(n - 1, 1)), int(190 - 110 * i / max(n - 1, 1)), int(90 + 60 * i / max(n - 1, 1)))
               for i in range(n)]
    polys = [rect(0.0, -i - 0.9, 6.0, -i - 0.1) for i in range(n)]
    return PuzzleSpec(
        name=f"pancake{n}", family="cayley", description=f"Pancake sorting graph on {n} pancakes (prefix reversals)",
        n=n, solved_state=list(range(n)), generators=gens,
        pieces=[PieceSpec(name=f"p{i + 1}", kind="token", facets=[i]) for i in range(n)],
        palette=palette, color_names=[str(i + 1) for i in range(n)], labels=[str(i + 1) for i in range(n)],
        geometry=planar_geometry(polys, "stack"), known_order=math.factorial(n),
    )


def star(n: int = 7) -> PuzzleSpec:
    """Star graph ST_n: Cayley graph of S_n generated by the transpositions (0 i)."""
    gens = [GeneratorSpec(name=f"T{i}", cycles=[[0, i]]) for i in range(1, n)]
    polys = [regular_polygon(0.0, 0.0, 0.42, 12)]
    polys += [regular_polygon(1.6 * math.cos(2 * math.pi * i / (n - 1)), 1.6 * math.sin(2 * math.pi * i / (n - 1)), 0.42, 12)
              for i in range(n - 1)]
    return PuzzleSpec(
        name=f"star{n}", family="cayley", description=f"Star graph on {n} tokens (transpositions with the hub)",
        n=n, solved_state=list(range(n)), generators=gens,
        pieces=[PieceSpec(name=f"t{i}", kind="token", facets=[i]) for i in range(n)],
        labels=[str(i) for i in range(n)], geometry=planar_geometry(polys, "star"),
        known_order=math.factorial(n),
    )


def hungarian_rings() -> PuzzleSpec:
    """Two 20-ball rings sharing two balls (38 balls, colours 10/9/10/9)."""
    ring = 20
    a_idx = list(range(ring))                                   # ring A: facets 0..19
    b_idx = [0 if k == 0 else 5 if k == 5 else 0 for k in range(ring)]
    nxt = ring
    for k in range(ring):
        if k not in (0, 5):
            b_idx[k] = nxt
            nxt += 1
    n = nxt                                                     # 38
    solved = [0] * n
    for k in range(6, 16):
        solved[a_idx[k]] = 0
    for k in (16, 17, 18, 19, 0, 1, 2, 3, 4):
        solved[a_idx[k]] = 1
    for k in range(6, 16):
        solved[b_idx[k]] = 2
    for k in (16, 17, 18, 19, 1, 2, 3, 4):
        solved[b_idx[k]] = 3
    solved[5] = 3
    gens = [GeneratorSpec(name="A", cycles=[a_idx]), GeneratorSpec(name="B", cycles=[b_idx])]
    r = 3.0
    off = r * math.cos(math.pi / 4)
    polys: list = [None] * n
    for k in range(ring):
        ang = -math.pi / 4 + k * 2 * math.pi / ring
        polys[a_idx[k]] = regular_polygon(-off + r * math.cos(ang), r * math.sin(ang), 0.40, 10)
    for k in range(ring):
        ang = 5 * math.pi / 4 - k * 2 * math.pi / ring
        if k not in (0, 5):
            polys[b_idx[k]] = regular_polygon(off + r * math.cos(ang), r * math.sin(ang), 0.40, 10)
    return PuzzleSpec(
        name="hungarian_rings", family="cayley",
        description="Hungarian Rings: two intersecting rings of 20 balls sharing 2 positions",
        n=n, solved_state=solved, generators=gens,
        pieces=[PieceSpec(name=f"ball{i}", kind="ball", facets=[i]) for i in range(n)],
        palette=[RED, YELLOW, BLUE, (30, 30, 30)], color_names=["red", "yellow", "blue", "black"],
        geometry=planar_geometry(polys, "rings"), known_order=math.factorial(38),
    )


def lights_out(size: int = 5) -> PuzzleSpec:
    """Lights Out on a size x size board: a (Z_2)^{size^2} extension, one involution per button."""
    cells = size * size
    moves = []
    for r in range(size):
        for c in range(size):
            inc = {r * size + c: 1}
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= r + dr < size and 0 <= c + dc < size:
                    inc[(r + dr) * size + (c + dc)] = 1
            moves.append(ModularMove(name=f"T{r}{c}", increments=inc))
    mod = ModularSpec(name=f"lights_out{size}", moduli=[2] * cells, moves=moves,
                      register_names=[f"L{r}{c}" for r in range(size) for c in range(size)],
                      description=f"Lights Out {size}x{size} over Z_2")
    polys = []
    for r in range(size):
        for c in range(size):
            polys.append(rect(c + 0.06, -r - 0.24, c + 0.24, -r - 0.06))     # slot 0: "off" marker
            polys.append(rect(c + 0.30, -r - 0.94, c + 0.94, -r - 0.30))     # slot 1: the lamp
    syms = []
    for si, f in enumerate((lambda r, c: (c, size - 1 - r), lambda r, c: (size - 1 - r, size - 1 - c),
                            lambda r, c: (size - 1 - c, r), lambda r, c: (r, size - 1 - c),
                            lambda r, c: (size - 1 - r, c), lambda r, c: (c, r),
                            lambda r, c: (size - 1 - c, size - 1 - r))):
        perm = [0] * (2 * cells)
        for r in range(size):
            for c in range(size):
                r2, c2 = f(r, c)
                for s in range(2):
                    perm[2 * (r2 * size + c2) + s] = 2 * (r * size + c) + s
        syms.append(SymmetrySpec(name=f"d4_{si}", perm=perm, color_map=[0, 1]))
    rank = _gf2_rank(np.array([[m.increments.get(i, 0) for i in range(cells)] for m in moves]))
    return mod.to_puzzle_spec(geometry=planar_geometry(polys, "grid"), symmetries=syms,
                              known_order=2 ** rank, metadata={"size": size, "modular": mod.model_dump()})


def _gf2_rank(mat: np.ndarray) -> int:
    m = mat.copy() % 2
    rank = 0
    rows, cols = m.shape
    for c in range(cols):
        piv = next((r for r in range(rank, rows) if m[r, c]), None)
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(rows):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
    return rank


def rubiks_clock() -> PuzzleSpec:
    """Rubik's Clock as a (Z_12)^14 extension.

    Registers 0..8 are the front dials (row-major, seen from the front); 9..13 are the back
    edge/centre dials (seen from the back: top, left, centre, right, bottom).  Back corner dials
    are the front corner dials seen from behind and therefore run backwards.
    """
    back_reg = {(0, 1): 9, (1, 0): 10, (1, 1): 11, (1, 2): 12, (2, 1): 13}
    blocks = {"UL": [(0, 0), (0, 1), (1, 0), (1, 1)], "UR": [(0, 1), (0, 2), (1, 1), (1, 2)],
              "DL": [(1, 0), (1, 1), (2, 0), (2, 1)], "DR": [(1, 1), (1, 2), (2, 1), (2, 2)]}
    configs = {"UR": ["UR"], "DR": ["DR"], "DL": ["DL"], "UL": ["UL"], "U": ["UL", "UR"], "R": ["UR", "DR"],
               "D": ["DL", "DR"], "L": ["UL", "DL"], "ALL": ["UL", "UR", "DL", "DR"]}
    moves = []
    for side in ("f", "b"):
        for cname, pins in configs.items():
            cells = sorted({cell for p in pins for cell in blocks[p]})
            inc: dict[int, int] = {}
            for r, c in cells:
                if side == "f":
                    inc[r * 3 + c] = 1
                elif (r, c) in back_reg:
                    inc[back_reg[(r, c)]] = 1
                else:
                    inc[r * 3 + (2 - c)] = 11          # back corner == front corner turning backwards
            moves.append(ModularMove(name=("" if side == "f" else "b") + cname, increments=inc))
    names = [f"F{r}{c}" for r in range(3) for c in range(3)] + ["Bt", "Bl", "Bc", "Br", "Bb"]
    mod = ModularSpec(name="rubiks_clock", moduli=[12] * 14, moves=moves, register_names=names,
                      description="Rubik's Clock: 14 independent dials over Z_12, 18 pin/wheel moves")

    def dial(cx: float, cy: float, clockwise: bool) -> list:
        out = []
        for k in range(12):
            a0 = math.pi / 2 - (k - 0.5) * math.pi / 6 * (1 if clockwise else -1)
            a1 = math.pi / 2 - (k + 0.5) * math.pi / 6 * (1 if clockwise else -1)
            lo, hi = sorted((a0, a1))
            out.append([(cx + 0.12 * math.cos(lo), cy + 0.12 * math.sin(lo)),
                        (cx + 0.44 * math.cos(lo), cy + 0.44 * math.sin(lo)),
                        (cx + 0.44 * math.cos(hi), cy + 0.44 * math.sin(hi)),
                        (cx + 0.12 * math.cos(hi), cy + 0.12 * math.sin(hi))])
        return out

    polys: list = []
    for r in range(3):
        for c in range(3):
            polys.extend(dial(c, -r, True))
    for (r, c) in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
        polys.extend(dial(4.0 + c, -r, True))
    return mod.to_puzzle_spec(geometry=planar_geometry(polys, "clock"), known_order=12 ** 14,
                              metadata={"modular": mod.model_dump()})


# ----------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------
REGISTRY: dict[str, Callable[[], PuzzleSpec]] = {
    "cube2": lambda: cube(2),
    "cube3": lambda: cube(3),
    "cube4": lambda: cube(4),
    "pyraminx": pyraminx,
    "skewb": skewb,
    "megaminx": megaminx,
    "fifteen": fifteen,
    "eight": lambda: sliding(3, 3),
    "pancake": lambda: pancake(8),
    "star": lambda: star(7),
    "hungarian_rings": hungarian_rings,
    "lights_out": lambda: lights_out(5),
    "lights_out3": lambda: lights_out(3),
    "rubiks_clock": rubiks_clock,
}

ALIASES = {"2x2": "cube2", "3x3": "cube3", "4x4": "cube4", "15": "fifteen", "15-puzzle": "fifteen",
           "8": "eight", "rings": "hungarian_rings", "clock": "rubiks_clock", "lights": "lights_out"}


def list_puzzles() -> list[str]:
    return list(REGISTRY)


@lru_cache(maxsize=None)
def get_spec(name: str) -> PuzzleSpec:
    key = ALIASES.get(name, name)
    if key in REGISTRY:
        spec = REGISTRY[key]()
        return spec if spec.name == key else spec.model_copy(update={"name": key})
    for prefix, factory in (("cube", cube), ("pancake", pancake), ("star", star), ("lights_out", lights_out)):
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return factory(int(key[len(prefix):]))
    if key.startswith("sliding") and "x" in key:
        r, c = key[len("sliding"):].split("x")
        return sliding(int(r), int(c))
    raise KeyError(f"unknown puzzle {name!r}; available: {', '.join(REGISTRY)}")


@lru_cache(maxsize=None)
def get_puzzle(name: str) -> Puzzle:
    return Puzzle(get_spec(name))
