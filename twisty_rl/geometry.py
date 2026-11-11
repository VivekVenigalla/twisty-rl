"""Computational geometry for polyhedral twisty puzzles.

Every polyhedral puzzle is *derived* instead of hand-tabulated:

1. the solid is the intersection of half-spaces ``n_i . x <= rho_i``;
2. each face polygon is sliced by the puzzle's cutting planes, producing stickers;
3. a move is a rigid rotation of every sticker on one side of a cutting plane and its
   permutation is recovered by matching rotated sticker centroids;
4. pieces are the cells of the plane arrangement, symmetries are the isometries of the
   solid, and the 2D net is obtained by unfolding faces around shared edges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .schema import Geometry, GeneratorSpec, MoveAnimation, PieceSpec, PuzzleSpec, SymmetrySpec

EPS = 1e-7


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def rotation_matrix(axis, angle: float) -> np.ndarray:
    """Right-handed rotation about ``axis`` by ``angle`` (Rodrigues)."""
    a = normalize(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def euler_matrix(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    return (rotation_matrix((0, 0, 1), roll) @ rotation_matrix((1, 0, 0), pitch)
            @ rotation_matrix((0, 1, 0), yaw))


def split_polygon(poly: np.ndarray, normal: np.ndarray, d: float, eps: float = EPS):
    """Split a convex polygon by the plane ``normal . x = d`` -> ``(front, back)`` (or ``None``)."""
    dist = poly @ normal - d
    front: list[np.ndarray] = []
    back: list[np.ndarray] = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        da, db = dist[i], dist[(i + 1) % m]
        if da >= -eps:
            front.append(a)
        if da <= eps:
            back.append(a)
        if (da > eps and db < -eps) or (da < -eps and db > eps):
            p = a + (da / (da - db)) * (b - a)
            front.append(p)
            back.append(p)
    return _clean(front), _clean(back)


def _clean(pts: list[np.ndarray]) -> Optional[np.ndarray]:
    out: list[np.ndarray] = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) > 1e-6:
            out.append(p)
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) <= 1e-6:
        out.pop()
    if len(out) < 3:
        return None
    arr = np.asarray(out)
    if polygon_area(arr) < 1e-8:
        return None
    return arr


def polygon_area(poly: np.ndarray) -> float:
    if poly.shape[1] == 2:
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    acc = np.zeros(3)
    for i in range(1, len(poly) - 1):
        acc += np.cross(poly[i] - poly[0], poly[i + 1] - poly[0])
    return 0.5 * float(np.linalg.norm(acc))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area-weighted centroid of a planar convex polygon (2D or 3D)."""
    total = 0.0
    acc = np.zeros(poly.shape[1])
    for i in range(1, len(poly) - 1):
        a, b, c = poly[0], poly[i], poly[i + 1]
        if poly.shape[1] == 2:
            e1, e2 = b - a, c - a
            area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        else:
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        acc += area * (a + b + c) / 3.0
        total += area
    return acc / total if total > 0 else poly.mean(axis=0)


def face_basis(normal: np.ndarray, up_hint=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal ``(u, v)`` on a face with ``u x v = normal`` and ``v`` as close to ``up`` as possible."""
    n = normalize(normal)
    up = np.asarray(up_hint, dtype=np.float64)
    v = up - (up @ n) * n
    if np.linalg.norm(v) < 1e-6:
        alt = np.array([0.0, 0.0, -1.0]) * (1.0 if n[1] >= 0 else -1.0)
        v = alt - (alt @ n) * n
    v = normalize(v)
    u = np.cross(v, n)
    return u, v


def halfspace_faces(normals: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Face polygons (CCW seen from outside) of ``{x : normals @ x <= offsets}``."""
    faces = []
    big = 50.0 * float(np.max(np.abs(offsets)))
    for i, (n, rho) in enumerate(zip(normals, offsets)):
        u, v = face_basis(n)
        c = n * rho
        poly = np.array([c + big * (u + v), c + big * (-u + v), c + big * (-u - v), c + big * (u - v)])
        for j, (m, sig) in enumerate(zip(normals, offsets)):
            if j == i:
                continue
            _, back = split_polygon(poly, m, sig)
            if back is None:
                raise ValueError("degenerate polyhedron: a half-space removes a whole face")
            poly = back
        faces.append(poly)
    return faces


def isometry_group(normals: np.ndarray, offsets: np.ndarray, reflections: bool = True) -> list[np.ndarray]:
    """All orthogonal maps sending the set of face planes of the solid onto itself."""
    F = len(normals)
    dots = normals @ normals.T
    best = -2.0
    pair = (0, 1)
    for j in range(1, F):
        if dots[0, j] < 1 - 1e-6 and dots[0, j] > best and abs(dots[0, j]) < 1 - 1e-6:
            best, pair = dots[0, j], (0, j)
    a, b = pair

    def frame(i: int, j: int, sign: float) -> np.ndarray:
        e1 = normals[i]
        e2 = normalize(normals[j] - (normals[j] @ e1) * e1)
        e3 = sign * np.cross(e1, e2)
        return np.stack([e1, e2, e3], axis=1)

    E = frame(a, b, 1.0)
    mats: list[np.ndarray] = []
    for i in range(F):
        if abs(offsets[i] - offsets[a]) > 1e-6:
            continue
        for j in range(F):
            if j == i or abs(dots[i, j] - dots[a, b]) > 1e-6 or abs(offsets[j] - offsets[b]) > 1e-6:
                continue
            for sign in ((1.0, -1.0) if reflections else (1.0,)):
                M = frame(i, j, sign) @ E.T
                img = normals @ M.T
                match = img @ normals.T
                idx = match.argmax(axis=1)
                if (match.max(axis=1) > 1 - 1e-6).all() and len(set(idx.tolist())) == F \
                        and np.allclose(offsets[idx], offsets, atol=1e-6):
                    mats.append(M)
    return mats


# ----------------------------------------------------------------------------------
# Net unfolding
# ----------------------------------------------------------------------------------
def _convex_overlap(p: np.ndarray, q: np.ndarray, tol: float = 1e-6) -> bool:
    """Separating-axis test for two convex 2D polygons (touching is not overlapping)."""
    for poly in (p, q):
        m = len(poly)
        for i in range(m):
            e = poly[(i + 1) % m] - poly[i]
            axis = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(axis)
            if nrm < 1e-12:
                continue
            axis /= nrm
            a, b = p @ axis, q @ axis
            if a.max() <= b.min() + tol or b.max() <= a.min() + tol:
                return False
    return True


def unfold_faces(faces: list[np.ndarray], normals: np.ndarray, root: int = 0,
                 up_hint=(0.0, 1.0, 0.0)) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Unfold a convex polyhedron into the plane.

    Returns for each face ``(origin, u, v, R, t)`` such that a 3D point ``x`` on the face maps to
    ``R @ [u.(x-origin), v.(x-origin)] + t``.
    """
    F = len(faces)
    bases = []
    for f in range(F):
        u, v = face_basis(normals[f], up_hint)
        bases.append((faces[f].mean(axis=0), u, v))

    def local(f: int, pts: np.ndarray) -> np.ndarray:
        o, u, v = bases[f]
        return np.stack([(pts - o) @ u, (pts - o) @ v], axis=-1)

    shared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for f in range(F):
        for g in range(F):
            if f == g:
                continue
            common = [p for p in faces[f] if np.min(np.linalg.norm(faces[g] - p, axis=1)) < 1e-6]
            if len(common) >= 2:
                shared[(f, g)] = (common[0], common[1])

    placed: dict[int, tuple[np.ndarray, np.ndarray]] = {root: (np.eye(2), np.zeros(2))}
    polys2d: dict[int, np.ndarray] = {root: local(root, faces[root])}
    order = [root]

    def candidate(g: int, f: int):
        p, q = shared[(f, g)]
        Rf, tf = placed[f]
        P = local(f, np.stack([p, q])) @ Rf.T + tf
        L = local(g, np.stack([p, q]))
        ang = math.atan2(*(P[1] - P[0])[::-1]) - math.atan2(*(L[1] - L[0])[::-1])
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        t = P[0] - R @ L[0]
        return R, t, local(g, faces[g]) @ R.T + t

    while len(placed) < F:
        progress = False
        for allow_overlap in (False, True):
            for f in list(order):
                for g in range(F):
                    if g in placed or (f, g) not in shared:
                        continue
                    R, t, poly = candidate(g, f)
                    if not allow_overlap and any(_convex_overlap(poly, polys2d[h]) for h in placed):
                        continue
                    placed[g] = (R, t)
                    polys2d[g] = poly
                    order.append(g)
                    progress = True
                if progress and allow_overlap:
                    break
            if progress:
                break
        if not progress:
            raise ValueError("polyhedron faces are not edge-connected")
    return [(bases[f][0], bases[f][1], bases[f][2], placed[f][0], placed[f][1]) for f in range(F)]


# ----------------------------------------------------------------------------------
# Polyhedral puzzle builder
# ----------------------------------------------------------------------------------
@dataclass
class MoveDef:
    name: str
    axis: tuple[float, float, float]
    depth: float
    order: int


@dataclass
class PolyhedralBuilder:
    name: str
    kind: str
    normals: Sequence[Sequence[float]]
    offsets: Sequence[float]
    face_names: Sequence[str]
    face_colors: Sequence[tuple[int, int, int]]
    cuts: Sequence[tuple[Sequence[float], float]]
    moves: Sequence[MoveDef]
    description: str = ""
    net_root: int = 0
    known_order: Optional[int] = None
    reflections: bool = True
    metadata: dict = field(default_factory=dict)

    def build(self) -> PuzzleSpec:
        normals = np.array([normalize(n) for n in self.normals])
        offsets = np.asarray(self.offsets, dtype=np.float64)
        faces = halfspace_faces(normals, offsets)

        # unique cutting planes (n, d) ~ (-n, -d)
        planes: list[tuple[np.ndarray, float]] = []
        for axis, depth in self.cuts:
            a = normalize(axis)
            if not any((np.allclose(a, p) and abs(depth - d) < 1e-9) or
                       (np.allclose(a, -p) and abs(depth + d) < 1e-9) for p, d in planes):
                planes.append((a, float(depth)))

        stickers: list[np.ndarray] = []
        sticker_face: list[int] = []
        for f, poly in enumerate(faces):
            parts = [poly]
            for a, d in planes:
                if abs(abs(a @ normals[f]) - 1.0) < 1e-9:
                    continue
                nxt = []
                for part in parts:
                    fr, bk = split_polygon(part, a, d)
                    nxt.extend(x for x in (fr, bk) if x is not None)
                parts = nxt
            u, v = face_basis(normals[f])
            cents = [polygon_centroid(p) for p in parts]
            keyed = sorted(range(len(parts)), key=lambda i: (-round(float(cents[i] @ v), 5),
                                                             round(float(cents[i] @ u), 5)))
            for i in keyed:
                stickers.append(parts[i])
                sticker_face.append(f)
        n = len(stickers)
        cents = np.array([polygon_centroid(p) for p in stickers])
        snorm = normals[sticker_face]

        def match(M: np.ndarray, which: np.ndarray) -> Optional[np.ndarray]:
            """Gather permutation of the rigid map ``M`` applied to the stickers in ``which``."""
            perm = np.arange(n, dtype=np.intp)
            tc = cents[which] @ M.T
            tn = snorm[which] @ M.T
            d2 = ((tc[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
            j = d2.argmin(axis=1)
            ok = (d2[np.arange(len(which)), j] < 1e-8) & ((tn * snorm[j]).sum(-1) > 1 - 1e-6)
            if not ok.all() or len(set(j.tolist())) != len(which):
                return None
            perm[j] = which
            return perm

        gens: list[GeneratorSpec] = []
        anims: dict[str, MoveAnimation] = {}
        for mv in self.moves:
            a = normalize(mv.axis)
            which = np.nonzero(cents @ a > mv.depth + 1e-7)[0]
            ang = -2.0 * math.pi / mv.order
            perm = match(rotation_matrix(a, ang), which)
            if perm is None:
                raise ValueError(f"move {mv.name!r}: cut structure is not invariant under the rotation")
            gens.append(GeneratorSpec(name=mv.name, perm=perm.tolist()))
            anims[mv.name] = MoveAnimation(axis=tuple(a.tolist()), angle=ang, facets=which.tolist())
            anims[mv.name + "'"] = MoveAnimation(axis=tuple(a.tolist()), angle=-ang, facets=which.tolist())

        # pieces = cells of the plane arrangement
        sig: dict[tuple, list[int]] = {}
        for i in range(n):
            key = tuple(int(np.sign(round(float(cents[i] @ a - d), 6))) for a, d in planes)
            sig.setdefault(key, []).append(i)
        kind_names = {1: "center", 2: "edge", 3: "corner"}
        pieces = []
        counters: dict[str, int] = {}
        for key, facets in sorted(sig.items(), key=lambda kv: min(kv[1])):
            kind = kind_names.get(len(facets), f"piece{len(facets)}")
            counters[kind] = counters.get(kind, 0) + 1
            pieces.append(PieceSpec(name=f"{kind}{counters[kind]}", kind=kind, facets=sorted(facets)))

        syms: list[SymmetrySpec] = []
        for si, M in enumerate(isometry_group(normals, offsets, self.reflections)):
            perm_content = match(M, np.arange(n))
            if perm_content is None:
                continue
            # perm_content[j] = i  <=>  sticker i is carried to j: that *is* rho (gather form)
            img = normals @ M.T
            cmap = (img @ normals.T).argmax(axis=1)
            syms.append(SymmetrySpec(name=f"iso{si}", perm=perm_content.tolist(), color_map=cmap.tolist()))

        frames = unfold_faces(faces, normals, self.net_root)
        polys2d = []
        for i, poly in enumerate(stickers):
            o, u, v, R, t = frames[sticker_face[i]]
            loc = np.stack([(poly - o) @ u, (poly - o) @ v], axis=-1)
            polys2d.append([tuple(x) for x in (loc @ R.T + t).tolist()])
        scale = 1.0 / float(np.max(np.linalg.norm(np.concatenate(faces), axis=1)))
        geom = Geometry(
            kind=self.kind,
            polygons3d=[[tuple(x) for x in (p * scale).tolist()] for p in stickers],
            polygons2d=polys2d,
            body3d=[[tuple(x) for x in (p * scale).tolist()] for p in faces],
            double_sided=False,
            animations=anims,
        )
        return PuzzleSpec(
            name=self.name, family="polyhedral", description=self.description, n=n,
            solved_state=list(sticker_face), generators=gens, pieces=pieces, symmetries=syms,
            palette=list(self.face_colors), color_names=list(self.face_names), geometry=geom,
            known_order=self.known_order, metadata=dict(self.metadata),
        )


# ----------------------------------------------------------------------------------
# Planar geometry helpers for combinatorial puzzles
# ----------------------------------------------------------------------------------
def regular_polygon(cx: float, cy: float, r: float, sides: int = 10, phase: float = 0.0):
    return [(cx + r * math.cos(phase + 2 * math.pi * k / sides),
             cy + r * math.sin(phase + 2 * math.pi * k / sides)) for k in range(sides)]


def rect(x0: float, y0: float, x1: float, y1: float):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def planar_geometry(polygons2d: list[list[tuple[float, float]]], kind: str = "planar") -> Geometry:
    """Lift a 2D layout into a thin double-sided 3D plate so every puzzle can be shown in 3D."""
    pts = np.array([p for poly in polygons2d for p in poly], dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    scale = 1.0 / max(float(np.linalg.norm(hi - lo)) / 2, 1e-9)
    polys3d = []
    for poly in polygons2d:
        arr = (np.asarray(poly, dtype=np.float64) - centre) * scale
        polys3d.append([(float(x), float(y), 0.0) for x, y in arr])
    w, h = (hi - lo) * scale / 2
    m = 0.04
    body = [[(-w - m, -h - m, -0.02), (w + m, -h - m, -0.02), (w + m, h + m, -0.02), (-w - m, h + m, -0.02)]]
    return Geometry(kind=kind, polygons3d=polys3d, polygons2d=polygons2d, body3d=body, double_sided=True)
