"""Software rasteriser for terminals.

A frame is a ``W x 2H`` pixel buffer shown with Unicode half blocks (``▀`` / ``▄``): the upper pixel
of a cell is the foreground colour, the lower pixel the background colour.  The 3D pipeline is a
classic one -- model rotation, perspective projection, back-face culling, flat Lambert shading and
a z-buffer -- implemented with numpy; the 2D pipeline rasterises the universal net unfolding.

Frames convert to ``rich.text.Text`` (Textual), ANSI true-colour strings (plain terminals) and
run-length encoded ``[text, fg, bg]`` rows (Emacs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
from rich.style import Style
from rich.text import Text

from .geometry import euler_matrix, halfspace_faces, normalize, polygon_centroid, rotation_matrix
from .schema import Puzzle

UPPER, LOWER = "▀", "▄"
BODY_COLOR = np.array([14, 14, 17], dtype=np.float64)
LIGHT_DIR = normalize((0.30, 0.78, 0.55))
AMBIENT = 0.64                      # stickers stay recognisably their own colour on every face


# ----------------------------------------------------------------------------------
# Frame
# ----------------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _style(fg: Optional[tuple[int, int, int]], bg: Optional[tuple[int, int, int]]) -> Style:
    return Style(color=None if fg is None else f"rgb({fg[0]},{fg[1]},{fg[2]})",
                 bgcolor=None if bg is None else f"rgb({bg[0]},{bg[1]},{bg[2]})")


@dataclass
class Frame:
    """``color [2H, W, 3]`` uint8 and ``mask [2H, W]`` (True where something was drawn)."""

    color: np.ndarray
    mask: np.ndarray
    overlay: dict[tuple[int, int], tuple[str, tuple[int, int, int], tuple[int, int, int]]] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.color.shape[0] // 2

    @property
    def cols(self) -> int:
        return self.color.shape[1]

    def cells(self) -> list[list[tuple[str, Optional[tuple[int, int, int]], Optional[tuple[int, int, int]]]]]:
        """Run-length encoded rows of ``(text, fg, bg)``; ``None`` means the terminal default."""
        out = []
        col, mask = self.color, self.mask
        for r in range(self.rows):
            runs: list[list] = []
            top, bot = col[2 * r], col[2 * r + 1]
            mt, mb = mask[2 * r], mask[2 * r + 1]
            for c in range(self.cols):
                ov = self.overlay.get((r, c))
                if ov is not None:
                    cell = ov
                elif mt[c] and mb[c]:
                    t, b = tuple(int(v) for v in top[c]), tuple(int(v) for v in bot[c])
                    cell = (" ", None, t) if t == b else (UPPER, t, b)
                elif mt[c]:
                    cell = (UPPER, tuple(int(v) for v in top[c]), None)
                elif mb[c]:
                    cell = (LOWER, tuple(int(v) for v in bot[c]), None)
                else:
                    cell = (" ", None, None)
                if runs and runs[-1][1] == cell[1] and runs[-1][2] == cell[2]:
                    runs[-1][0] += cell[0]
                else:
                    runs.append([cell[0], cell[1], cell[2]])
            out.append([(t, f, b) for t, f, b in runs])
        return out

    def to_text(self) -> Text:
        text = Text(no_wrap=True, end="")
        rows = self.cells()
        for i, runs in enumerate(rows):
            for t, fg, bg in runs:
                text.append(t, _style(fg, bg))
            if i + 1 < len(rows):
                text.append("\n")
        return text

    def to_lines(self) -> list[Text]:
        lines = []
        for runs in self.cells():
            line = Text(no_wrap=True, end="")
            for t, fg, bg in runs:
                line.append(t, _style(fg, bg))
            lines.append(line)
        return lines

    def to_ansi(self) -> str:
        out = []
        for runs in self.cells():
            line = []
            for t, fg, bg in runs:
                codes = []
                if fg is not None:
                    codes.append(f"38;2;{fg[0]};{fg[1]};{fg[2]}")
                if bg is not None:
                    codes.append(f"48;2;{bg[0]};{bg[1]};{bg[2]}")
                line.append(f"\x1b[{';'.join(codes)}m{t}\x1b[0m" if codes else t)
            out.append("".join(line))
        return "\n".join(out)

    def to_runs(self) -> list[list[list]]:
        """JSON-friendly rows of ``[text, "#rrggbb" | None, "#rrggbb" | None]``."""
        hexc = lambda c: None if c is None else "#%02x%02x%02x" % c
        return [[[t, hexc(fg), hexc(bg)] for t, fg, bg in runs] for runs in self.cells()]

    def coverage(self) -> float:
        return float(self.mask.mean())


# ----------------------------------------------------------------------------------
# Rasteriser
# ----------------------------------------------------------------------------------
class Rasterizer:
    """Z-buffered triangle rasteriser on a ``width x 2*height`` pixel grid.

    ``supersample = s`` rasterises ``s x s`` samples per pixel (``pw x ph`` samples in total) and
    :meth:`frame` box-filters them down, which anti-aliases every edge.
    """

    def __init__(self, width: int, height: int, supersample: int = 1):
        self.width, self.height = max(int(width), 1), max(int(height), 1)
        self.h2 = 2 * self.height
        self.ss = max(1, min(int(supersample), 8))
        self.pw, self.ph = self.width * self.ss, self.h2 * self.ss
        self.color = np.zeros((self.ph, self.pw, 3), dtype=np.uint8)
        self.depth = np.full((self.ph, self.pw), np.inf)

    def clear(self) -> None:
        self.color.fill(0)
        self.depth.fill(np.inf)

    def triangle(self, p: np.ndarray, z: np.ndarray, rgb: np.ndarray) -> None:
        """``p [3, 2]`` pixel coordinates, ``z [3]`` depths (smaller is nearer)."""
        x0 = max(int(math.floor(p[:, 0].min())), 0)
        x1 = min(int(math.ceil(p[:, 0].max())), self.pw - 1)
        y0 = max(int(math.floor(p[:, 1].min())), 0)
        y1 = min(int(math.ceil(p[:, 1].max())), self.ph - 1)
        if x0 > x1 or y0 > y1:
            return
        area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])
        if abs(area) < 1e-9:
            return
        px = np.arange(x0, x1 + 1) + 0.5
        py = (np.arange(y0, y1 + 1) + 0.5)[:, None]
        w0 = ((p[1, 0] - px) * (p[2, 1] - py) - (p[1, 1] - py) * (p[2, 0] - px)) / area
        w1 = ((p[2, 0] - px) * (p[0, 1] - py) - (p[2, 1] - py) * (p[0, 0] - px)) / area
        w2 = 1.0 - w0 - w1
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub_d = self.depth[y0:y1 + 1, x0:x1 + 1]
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (zz < sub_d)
        if inside.any():
            sub_d[inside] = zz[inside]
            self.color[y0:y1 + 1, x0:x1 + 1][inside] = rgb

    def polygon(self, p: np.ndarray, z: np.ndarray, rgb: np.ndarray) -> None:
        for i in range(1, len(p) - 1):
            self.triangle(p[[0, i, i + 1]], z[[0, i, i + 1]], rgb)

    def frame(self, bg: Optional[tuple[int, int, int]] = None) -> Frame:
        """Resolve the sample buffer into a frame.

        With ``bg`` the image is composited over that colour, so silhouettes are anti-aliased too
        and every cell is fully specified (only ``▀`` and blanks are needed -- nothing depends on how
        a font draws ``▄``).  Without it, pixels at least half covered are kept.
        """
        covered = np.isfinite(self.depth)
        s = self.ss
        if s == 1 and bg is None:
            return Frame(self.color.copy(), covered)
        acc = (self.color.astype(np.float64) * covered[..., None]).reshape(self.h2, s, self.width, s, 3).sum((1, 3))
        cnt = covered.reshape(self.h2, s, self.width, s).sum((1, 3)).astype(np.float64)
        if bg is not None:
            out = (acc + (s * s - cnt)[..., None] * np.asarray(bg, dtype=np.float64)) / (s * s)
            mask = np.ones((self.h2, self.width), dtype=bool)
        else:
            out = acc / np.maximum(cnt, 1.0)[..., None]
            mask = cnt >= (s * s) / 2.0
        # a small colour lattice keeps runs long and the number of distinct terminal faces bounded
        out = np.clip(np.round(out / 6.0) * 6.0, 0, 255).astype(np.uint8)
        return Frame(out, mask)


# ----------------------------------------------------------------------------------
# Meshes
# ----------------------------------------------------------------------------------
def platonic_solid(kind: str) -> list[np.ndarray]:
    """Face polygons of a unit-circumradius cube, tetrahedron or dodecahedron."""
    if kind == "cube":
        normals = [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]
    elif kind == "tetrahedron":
        normals = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif kind == "dodecahedron":
        phi = (1 + math.sqrt(5)) / 2
        normals = [(0, s1, s2 * phi) for s1 in (1, -1) for s2 in (1, -1)]
        normals += [(s1, s2 * phi, 0) for s1 in (1, -1) for s2 in (1, -1)]
        normals += [(s1 * phi, 0, s2) for s1 in (1, -1) for s2 in (1, -1)]
    else:
        raise KeyError(f"unknown solid {kind!r}")
    nrm = np.array([normalize(n) for n in normals])
    faces = halfspace_faces(nrm, np.ones(len(nrm)))
    scale = 1.0 / max(float(np.linalg.norm(v)) for f in faces for v in f)
    return [f * scale for f in faces]


SOLID_PALETTES = {
    "cube": [(245, 245, 245), (255, 213, 0), (0, 158, 96), (0, 81, 186), (196, 30, 58), (255, 120, 20)],
    "tetrahedron": [(0, 158, 96), (0, 81, 186), (196, 30, 58), (255, 213, 0)],
    "dodecahedron": [(245, 245, 245), (0, 158, 96), (196, 30, 58), (0, 81, 186), (255, 213, 0), (128, 0, 160),
                     (150, 150, 150), (120, 220, 120), (255, 150, 190), (250, 240, 180), (100, 190, 255), (255, 120, 20)],
}


def _project(pts: np.ndarray, width: int, h2: int, zoom: float, distance: float,
             cell_aspect: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Perspective projection onto a ``width x h2`` sample grid.

    ``cell_aspect`` is the height / width of a terminal cell; a half-block pixel is therefore
    ``cell_aspect / 2`` times taller than wide and the vertical scale compensates for it.
    """
    depth = distance - pts[:, 2]
    f = distance * 0.92
    tall = max(cell_aspect, 0.5) / 2.0
    scale = 0.5 * min(width, h2 * tall) * zoom
    sx = width / 2 + pts[:, 0] * f / depth * scale
    sy = h2 / 2 - pts[:, 1] * f / depth * scale / tall
    return np.stack([sx, sy], axis=1), depth


def _shade(rgb: np.ndarray, normal_cam: np.ndarray, double_sided: bool) -> np.ndarray:
    lam = float(normal_cam @ LIGHT_DIR)
    lam = abs(lam) if double_sided else max(lam, 0.0)
    return np.clip(rgb * (AMBIENT + (1.0 - AMBIENT) * lam), 0, 255).astype(np.uint8)


def render_solid(kind: str, width: int = 40, height: int = 20, yaw: float = 0.6, pitch: float = 0.45,
                 roll: float = 0.0, zoom: float = 1.0, distance: float = 4.0) -> Frame:
    """Flat-shaded, z-buffered rendering of a bare platonic solid."""
    faces = platonic_solid(kind)
    pal = SOLID_PALETTES[kind]
    R = euler_matrix(yaw, pitch, roll)
    ras = Rasterizer(width, height)
    cam = np.array([0.0, 0.0, distance])
    for i, face in enumerate(faces):
        pts = face @ R.T
        n = normalize(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if n @ (cam - pts[0]) <= 0:
            continue
        scr, depth = _project(pts, ras.width, ras.h2, zoom, distance)
        ras.polygon(scr, depth, _shade(np.asarray(pal[i % len(pal)], dtype=np.float64), n, False))
    return ras.frame()


class PuzzleRenderer:
    """Cached mesh of one puzzle; renders arbitrary states in 3D and as a 2D net."""

    def __init__(self, puzzle: Puzzle, inset: float = 0.88):
        self.puzzle = puzzle
        geom = puzzle.spec.geometry
        if geom is None:
            raise ValueError(f"puzzle {puzzle.name!r} has no geometry")
        self.geom = geom
        self.double_sided = geom.double_sided
        self.palette = np.asarray(puzzle.spec.palette, dtype=np.float64)
        self.polys: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        for poly in geom.polygons3d:
            arr = np.asarray(poly, dtype=np.float64)
            c = polygon_centroid(arr)
            nrm = np.zeros(3)
            for i in range(1, len(arr) - 1):
                nrm += np.cross(arr[i] - arr[0], arr[i + 1] - arr[0])
            nrm = normalize(nrm) if np.linalg.norm(nrm) > 1e-12 else np.array([0.0, 0.0, 1.0])
            self.polys.append(c + (arr - c) * inset + nrm * 0.004)
            self.normals.append(nrm)
        self.body = [np.asarray(p, dtype=np.float64) for p in geom.body3d]
        pts2 = [np.asarray(p, dtype=np.float64) for p in geom.polygons2d]
        allp = np.concatenate(pts2)
        self.net_lo, self.net_hi = allp.min(axis=0), allp.max(axis=0)
        self.net_polys = []
        for p in pts2:
            c = polygon_centroid(p)
            self.net_polys.append(c + (p - c) * 0.9)
        self.net_centroids = np.array([polygon_centroid(p) for p in pts2])
        radius = max(float(np.linalg.norm(v)) for p in self.polys for v in p)
        self.base_zoom = 0.82 / max(radius, 1e-9)
        self.labels = puzzle.spec.labels
        self.animations = geom.animations

    # -- 3D ---------------------------------------------------------------------------------
    def render_3d(self, P: np.ndarray, width: int = 48, height: int = 24, yaw: float = 0.6, pitch: float = 0.45,
                  roll: float = 0.0, zoom: float = 1.0, distance: float = 4.2,
                  anim: Optional[tuple[str, float]] = None, supersample: int = 1,
                  bg: Optional[tuple[int, int, int]] = None, cell_aspect: float = 2.0) -> Frame:
        """Render permutation state ``P``.  ``anim=(move_name, t)`` shows that move ``t`` of the way through.

        ``supersample`` anti-aliases, ``bg`` composites over a known background colour and
        ``cell_aspect`` corrects for terminal cells that are not exactly twice as tall as wide.
        """
        colors = self.palette[self.puzzle.solved_colors[np.asarray(P)]]
        R = euler_matrix(yaw, pitch, roll)
        ras = Rasterizer(width, height, supersample)
        cam = np.array([0.0, 0.0, distance])
        zoom = zoom * self.base_zoom
        moving: set[int] = set()
        Rm = None
        if anim is not None and anim[0] in self.animations:
            a = self.animations[anim[0]]
            moving = set(a.facets)
            Rm = R @ rotation_matrix(a.axis, a.angle * float(np.clip(anim[1], 0.0, 1.0)))
        for body in self.body:
            pts = body * 0.985 @ R.T
            n = normalize(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            if not self.double_sided and n @ (cam - pts[0]) <= 0:
                continue
            scr, depth = _project(pts, ras.pw, ras.ph, zoom, distance, cell_aspect)
            ras.polygon(scr, depth + (0.06 if self.double_sided else 0.0), _shade(BODY_COLOR, n, self.double_sided))
        for i, poly in enumerate(self.polys):
            M = Rm if i in moving else R
            pts = poly @ M.T
            n = M @ self.normals[i]
            if not self.double_sided and n @ (cam - pts[0]) <= 0:
                continue
            scr, depth = _project(pts, ras.pw, ras.ph, zoom, distance, cell_aspect)
            ras.polygon(scr, depth, _shade(colors[i], n, self.double_sided))
        return ras.frame(bg)

    # -- 2D net ---------------------------------------------------------------------------------
    def render_net(self, P: np.ndarray, width: int = 60, height: int = 20, labels: bool = True) -> Frame:
        P = np.asarray(P)
        col_idx = self.puzzle.solved_colors[P]
        ras = Rasterizer(width, height)
        span = np.maximum(self.net_hi - self.net_lo, 1e-9)
        scale = min((ras.width - 2) / span[0], (ras.h2 - 2) / span[1])
        ox = (ras.width - span[0] * scale) / 2
        oy = (ras.h2 - span[1] * scale) / 2

        def to_px(pts: np.ndarray) -> np.ndarray:
            return np.stack([ox + (pts[:, 0] - self.net_lo[0]) * scale,
                             oy + (self.net_hi[1] - pts[:, 1]) * scale], axis=1)

        zero = np.zeros(16)
        for i, poly in enumerate(self.net_polys):
            scr = to_px(poly)
            ras.polygon(scr, zero[: len(scr)], self.palette[col_idx[i]].astype(np.uint8))
        frame = ras.frame()
        if labels and self.labels is not None:
            cent = to_px(self.net_centroids)
            for i in range(len(P)):
                text = self.labels[int(P[i])]
                if not text:
                    continue
                extent = to_px(self.net_polys[i])
                if extent[:, 0].max() - extent[:, 0].min() < len(text) + 0.5:
                    continue
                r = int(cent[i, 1] // 2)
                c0 = int(round(cent[i, 0] - len(text) / 2))
                bg = tuple(int(v) for v in self.palette[col_idx[i]])
                fg = (0, 0, 0) if sum(bg) > 380 else (255, 255, 255)
                for k, ch in enumerate(text):
                    if 0 <= r < frame.rows and 0 <= c0 + k < frame.cols:
                        frame.overlay[(r, c0 + k)] = (ch, fg, bg)
        return frame


_RENDERERS: dict[int, PuzzleRenderer] = {}


def get_renderer(puzzle: Puzzle) -> PuzzleRenderer:
    key = id(puzzle.spec)
    if key not in _RENDERERS:
        _RENDERERS[key] = PuzzleRenderer(puzzle)
    return _RENDERERS[key]


def render_net_ansi(puzzle: Puzzle, P: np.ndarray, width: int = 60, height: int = 20) -> str:
    return get_renderer(puzzle).render_net(P, width, height).to_ansi()


def render_3d_ansi(puzzle: Puzzle, P: np.ndarray, width: int = 48, height: int = 24, **kwargs) -> str:
    return get_renderer(puzzle).render_3d(P, width, height, **kwargs).to_ansi()
