"""Interactive session shared by the TUI, the Emacs daemon and the CLI.

A session owns one puzzle state plus the lazily built heavy objects around it (BSGS, pattern
databases, model / trainer) and exposes thread-safe verbs: apply moves, scramble, solve, mine
macros, train in the background, render.
"""
from __future__ import annotations

import datetime
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import torch

from .chain_solver import ChainSolver
from .geometry import polygon_centroid
from .group_theory import MacroMiner, PuzzleGroup
from .model import ModelConfig, RelationalGraphTransformer, load_checkpoint
from .puzzles import get_puzzle, list_puzzles
from .renderer_3d import Frame, get_renderer
from .schema import Puzzle
from .search import SearchResult, build_pdbs, make_solver
from .utils import Telemetry, human_int, pick_device

SHOW_MODES = ("spin", "attract", "daily")


class Session:
    def __init__(self, puzzle: str = "cube3", device: Optional[str] = None, telemetry: Optional[Telemetry] = None,
                 seed: int = 0, model_cfg: Optional[ModelConfig] = None):
        self.device = pick_device(device)
        self.telemetry = telemetry or Telemetry()
        self.rng = np.random.default_rng(seed)
        self.model_cfg = model_cfg or ModelConfig(d_model=96, n_layers=3, d_ff=192, d_msg=48)
        self.model: Optional[RelationalGraphTransformer] = None
        self.model_trained_on: set[str] = set()
        self._lock = threading.RLock()
        self._train_thread: Optional[threading.Thread] = None
        self._train_stop = threading.Event()
        self._cancel_search = threading.Event()
        self.training_mode = "idle"
        self.last_result: Optional[SearchResult] = None
        self.mined: list = []
        # camera
        self.yaw, self.pitch, self.zoom = 0.6, 0.55, 1.0
        self.spin = True
        self.spin_speed = 0.6
        self._spin_t0 = time.monotonic()
        # look of the 3D view: anti-aliasing level, known background colour, cell height / width
        self.quality = 4
        self.camera_distance = 7.0          # far camera: nearly parallel edges read cleaner at low resolution
        self.background: Optional[tuple[int, int, int]] = None
        self.cell_aspect = 2.0
        # timed move playback (smooth layer turns for clients that only poll frames)
        self._play_queue: deque[tuple[str, float]] = deque()
        self._play_current: Optional[tuple[str, float]] = None
        self._play_t0 = 0.0
        self._play_free_at = 0.0
        self.move_time = 0.22
        # idle show: "spin" (nothing), "attract" (scramble + solve forever), "daily" (scramble of the day)
        self.show_mode = "spin"
        self.show_paused = False
        self.show_depth = 9
        self.show_daily_depth = 20
        self.show_pause = 2.5
        self.show_time_limit = 6.0
        self.show_caption = ""
        self._show_phase = "pause"
        self._show_until = 0.0
        self._show_thread: Optional[threading.Thread] = None
        self._show_result: Optional[SearchResult] = None
        self._show_scramble: list[str] = []
        self.show_carousel: list[str] = []
        self._show_rounds = 0
        self.show_max_play = 30.0           # seconds a long (chain-solver) solution may take to play back
        self.astar_patience = 3.0           # seconds "auto" searches for an optimal answer before falling back
        # heavy per-puzzle objects survive puzzle switches (carousel, coming back to a puzzle)
        self._per_puzzle: dict[str, dict] = {}
        self._build_lock = threading.Lock()
        self.set_puzzle(puzzle)

    # -- puzzle management ------------------------------------------------------------------
    def set_puzzle(self, name: str) -> None:
        with self._lock:
            self.stop_training()
            self.base_puzzle = get_puzzle(name)
            self.puzzle: Puzzle = self.base_puzzle
            self.state = self.puzzle.identity.copy()
            self.history: list[str] = []
            self.scramble_moves: list[str] = []
            self._solver_cache: dict = {}
            self.trainer = None
            self.mined = []
            self.last_result = None
            self._clear_play()
            self._show_phase, self._show_until, self.show_caption = "pause", time.monotonic() + 0.8, ""
            known = self._per_puzzle.get(self.base_puzzle.name, {}).get("group")
            self.telemetry.set(puzzle=self.puzzle.name, facets=self.puzzle.n, actions=self.puzzle.num_actions,
                               group_order="…" if known is None else human_int(known.num_states), macros=[])
            self.telemetry.event(f"loaded {self.puzzle.name}: {self.puzzle.spec.description}")

    def _built(self, puzzle: Puzzle, key: str, build: Callable[[], object]):
        """Per-puzzle lazily built object.  Building never holds the state lock, so frames keep flowing."""
        entry = self._per_puzzle.setdefault(puzzle.name, {})
        if key not in entry:
            with self._build_lock:
                if key not in entry:
                    entry[key] = build()
        return entry[key]

    def _group_for(self, puzzle: Puzzle) -> PuzzleGroup:
        def build() -> PuzzleGroup:
            t0 = time.perf_counter()
            g = PuzzleGroup(puzzle)
            if puzzle is self.base_puzzle:
                self.telemetry.set(group_order=human_int(g.num_states), bsgs_base=len(g.bsgs.base),
                                   bsgs_time=time.perf_counter() - t0)
            return g
        return self._built(puzzle, "group", build)

    @property
    def group(self) -> PuzzleGroup:
        return self._group_for(self.base_puzzle)

    def _chain_for(self, puzzle: Puzzle) -> ChainSolver:
        group = self._group_for(puzzle)

        def build() -> ChainSolver:
            t0 = time.perf_counter()
            chain = ChainSolver.cached(group)
            self.telemetry.event(f"stabiliser-chain solver for {puzzle.name} ready in {time.perf_counter() - t0:.1f}s "
                                 f"(expected solution length {chain.expected_length():.0f})")
            return chain
        return self._built(puzzle, "chain", build)

    def chain(self) -> ChainSolver:
        """The always-terminating solver of the current puzzle (built or loaded from disk on first use)."""
        return self._chain_for(self.base_puzzle)

    def group_info(self) -> dict:
        g = self.group
        info = g.summary()
        info["order_text"] = human_int(g.num_states)
        info["order"] = str(info["order"])
        info["num_states"] = str(info["num_states"])
        info["curriculum"] = [{**d, "order": str(d["order"]), "index": str(d["index"])}
                              for d in g.stabilizer_curriculum().describe()]
        info["solvable"] = self.is_solvable()
        return info

    # -- state manipulation ---------------------------------------------------------------------
    def is_solved(self) -> bool:
        return self.puzzle.is_solved(self.state)

    def is_solvable(self) -> bool:
        return self.group.contains_state(self.state)

    def legal_moves(self) -> list[str]:
        return [self.puzzle.action_names[a] for a in self.puzzle.legal_actions(self.state)]

    def apply(self, moves: str | list[str]) -> list[str]:
        """Apply a move string such as ``"R U R' U'"``; returns the applied move names."""
        with self._lock:
            actions = self.puzzle.parse_moves(moves)
            self.flush_play()
            state = self.state
            for a in actions:
                state = self.puzzle.apply(state, a)
            self.state = state
            names = self.puzzle.names_of(actions)
            self.history.extend(names)
            return names

    def undo(self) -> Optional[str]:
        with self._lock:
            self.flush_play()
            if not self.history:
                return None
            name = self.history.pop()
            a = int(self.puzzle.inverse[self.puzzle.action_index(name)])
            self.state = self.puzzle.apply(self.state, a)
            return self.puzzle.action_names[a]

    def reset(self) -> None:
        with self._lock:
            self._clear_play()
            self.state = self.puzzle.identity.copy()
            self.history.clear()
            self.scramble_moves = []

    def scramble(self, depth: int = 20, seed: Optional[int] = None) -> list[str]:
        """Scramble from the solved state; a ``seed`` makes the scramble reproducible."""
        with self._lock:
            self._clear_play()
            rng = self.rng if seed is None else np.random.default_rng(seed)
            self.state, actions = self.puzzle.scramble(rng, depth)
            self.history.clear()
            self.scramble_moves = self.puzzle.names_of(actions)
            return self.scramble_moves

    def set_state(self, P) -> None:
        with self._lock:
            self._clear_play()
            self.state = np.asarray(P, dtype=np.intp).copy()
            self.history.clear()

    def set_colors(self, colors) -> dict:
        """Load a colour state -> ``{"accepted", "reason"}``; an unsolvable one leaves the state alone."""
        verdict = self.group.diagnose_colors(colors)
        if verdict["solvable"]:
            self.set_state(verdict["state"])
        return {"accepted": bool(verdict["solvable"]), "reason": verdict["reason"]}

    def paint_layout(self) -> dict:
        """Everything a client needs to let the user paint a physical puzzle onto a character grid."""
        pz = self.base_puzzle
        geom = pz.spec.geometry
        cent = np.array([polygon_centroid(np.asarray(p, dtype=np.float64)) for p in geom.polygons2d])
        if pz.spec.family == "polyhedral":                # pull the faces one cell apart
            centre = cent.mean(axis=0)
            for c in range(pz.num_colors):
                sel = pz.solved_colors == c
                cent[sel] += (cent[sel].mean(axis=0) - centre) / 3.0
        x, y = cent[:, 0] - cent[:, 0].min(), cent[:, 1].max() - cent[:, 1]
        scale, cells = 0.4, None
        while scale < 40:
            cand = np.stack([np.round(y * scale), np.round(x * scale)], axis=1).astype(int)
            if len({(int(r), int(c)) for r, c in cand}) == pz.n or cand[:, 0].max() > 34 or cand[:, 1].max() > 44:
                cells = cand
                break
            scale *= 1.07
        taken: set[tuple[int, int]] = set()
        out = []
        for r, c in cells.tolist():                       # dense nets: nudge collisions to the nearest free cell
            radius = 0
            while (r, c) in taken:
                radius += 1
                ring = [(r + dr, c + dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
                        if max(abs(dr), abs(dc)) == radius and r + dr >= 0 and c + dc >= 0]
                free = [q for q in ring if q not in taken]
                if free:
                    r, c = min(free, key=lambda q: (q[0] - r) ** 2 + (q[1] - c) ** 2)
            taken.add((r, c))
            out.append([r, c])
        prim = pz.table[: pz.num_primitive]
        fixed = ~(prim != pz.identity[None, None, :]).any(axis=(0, 1))
        named = {"w": (245, 245, 245), "y": (255, 213, 0), "g": (0, 158, 96), "b": (0, 81, 186), "r": (196, 30, 58),
                 "o": (255, 120, 20), "p": (128, 0, 160), "k": (255, 150, 190), "c": (250, 240, 180),
                 "s": (100, 190, 255), "l": (120, 220, 120), "a": (150, 150, 150), "d": (40, 40, 48)}
        palette = [tuple(c) for c in pz.spec.palette[: pz.num_colors]]
        letters = [min(named, key=lambda k: sum((a - b) ** 2 for a, b in zip(named[k], col))) for col in palette]
        if len(set(letters)) != len(letters):
            letters = list("123456789abcdefghijklmnoprstuvwxyz"[: len(palette)])
        return dict(puzzle=pz.name, rows=int(max(r for r, _ in out)) + 1, cols=int(max(c for _, c in out)) + 1,
                    cells=out, colors=self.colors(), fixed=fixed.tolist(), keys=letters,
                    palette=["#%02x%02x%02x" % c for c in palette], names=list(pz.spec.color_names[: pz.num_colors]),
                    need=np.bincount(pz.solved_colors, minlength=pz.num_colors).tolist())

    def colors(self) -> list[int]:
        return self.puzzle.colors(self.state).tolist()

    # -- search -----------------------------------------------------------------------------------
    def _pdbs_for(self, puzzle: Puzzle):
        group = self._group_for(puzzle)

        def build():
            budget = 300_000 if puzzle.n <= 60 else 60_000
            t0 = time.perf_counter()
            pdbs = build_pdbs(puzzle, max_entries=budget, count=2, device=self.device, group=group)
            self.telemetry.event(f"built {len(pdbs)} pattern databases for {puzzle.name} "
                                 f"({sum(p.size for p in pdbs):,} entries, {time.perf_counter() - t0:.1f}s)")
            return pdbs
        return self._built(puzzle, "pdbs", build)

    def pdbs(self):
        return self._pdbs_for(self.base_puzzle)

    def solve(self, method: str = "auto", time_limit: float = 20.0, weight: Optional[float] = None,
              use_model: Optional[bool] = None, quiet: bool = False) -> SearchResult:
        """Find a solution of the current state (does not modify it).

        ``method``: ``astar`` / ``beam`` search, ``chain`` uses the stabiliser-chain solver (always
        succeeds, not optimal) and ``auto`` searches for ``astar_patience`` seconds, then falls back
        to the chain solver.
        """
        with self._lock:
            state = self.state.copy()
            puzzle = self.puzzle
            base = self.base_puzzle
        if not self._group_for(base).contains_state(state):
            res = SearchResult(solved=False, reason="unsolvable: state is not in the puzzle group")
            self.last_result = res
            if not quiet:
                self.telemetry.event(res.reason)
            return res
        self._cancel_search.clear()
        if method == "chain":
            return self._finish_solve(self._chain_solve(base, state), quiet)
        pdbs = self._pdbs_for(base)
        have_model = self.model is not None and base.name in self.model_trained_on
        use_model = have_model if use_model is None else (use_model and self.model is not None)
        auto = method == "auto"
        if auto:
            method = "astar"
            time_limit = min(time_limit, self.astar_patience)
        key = (puzzle.name, puzzle.num_actions, method, use_model, weight)
        solver = self._solver_cache.get(key)
        if solver is None:
            w = weight if weight is not None else (1.4 if use_model else 1.0)
            kwargs = dict(optimal=not use_model and w <= 1.0) if method == "astar" else {}
            solver = make_solver(puzzle, self.device, model=self.model if use_model else None, method=method,
                                 pdbs=pdbs, weight=w, telemetry=self.telemetry,
                                 batch_size=256 if use_model else 1024, **kwargs)
            self._solver_cache = {key: solver}
        res = solver.solve(state, time_limit=time_limit, should_stop=self._cancel_search.is_set)
        if auto and not res.solved and not self._cancel_search.is_set():
            searched = res
            res = self._chain_solve(base, state)
            res.nodes_expanded, res.nodes_generated = searched.nodes_expanded, searched.nodes_generated
            res.elapsed += searched.elapsed
        return self._finish_solve(res, quiet)

    def _chain_solve(self, base: Puzzle, state: np.ndarray) -> SearchResult:
        t0 = time.perf_counter()
        word = self._chain_for(base).solve(state)
        return SearchResult(solved=True, path=list(word), primitive_path=list(word), moves=base.names_of(word),
                            cost=float(len(word)), elapsed=time.perf_counter() - t0,
                            reason="stabiliser-chain solver (always finishes, not optimal)")

    def _finish_solve(self, res: SearchResult, quiet: bool) -> SearchResult:
        self.last_result = res
        self.telemetry.set(last_solution=" ".join(res.moves) if res.solved else "", last_solution_len=len(res.moves),
                           search_nodes=res.nodes_generated)
        if not quiet:
            self.telemetry.event(
                (f"solved in {len(res.moves)} moves" + (" (optimal)" if res.optimal_proven else "")
                 if res.solved else f"no solution: {res.reason or 'budget'}")
                + f" | {res.nodes_generated:,} nodes, {res.nodes_per_sec:,.0f}/s, {res.elapsed:.2f}s")
        return res

    def cancel_search(self) -> None:
        self._cancel_search.set()

    # -- macros -----------------------------------------------------------------------------------
    def mine_macros(self, top_k: int = 8, promote: int = 0, **kwargs) -> list[dict]:
        miner = MacroMiner(self.base_puzzle, **kwargs)
        mined = miner.mine(top_k=top_k)
        with self._lock:
            self.mined = mined
            if promote > 0:
                macros = miner.promote(mined, promote)
                self.puzzle = self.base_puzzle.with_macros(macros)
                self._solver_cache = {}
                self.telemetry.event(f"promoted {len(macros)} macro-actions; action space is now {self.puzzle.num_actions}")
        self.telemetry.set(macros=[f"{m.text}  [{m.kind}, supp {m.support}]" for m in mined],
                           macro_candidates=miner.candidates_examined)
        self.telemetry.event(f"mined {len(mined)} macros from {miner.candidates_examined:,} candidates")
        return [m.to_dict() for m in mined]

    # -- learning ---------------------------------------------------------------------------------
    def ensure_model(self) -> RelationalGraphTransformer:
        if self.model is None:
            torch.manual_seed(0)
            self.model = RelationalGraphTransformer(self.model_cfg).to(self.device)
        return self.model

    def load_model(self, path: str) -> dict:
        self.model, extra = load_checkpoint(path, self.device)
        self.model_cfg = self.model.cfg
        self.model_trained_on = set(extra.get("tasks", []))
        self._solver_cache = {}
        return extra

    def start_training(self, steps: int = 300, adversarial: bool = False, batch_size: int = 96,
                       on_done: Optional[Callable[[], None]] = None) -> bool:
        """Launch ADI (or PAIRED) training of the current puzzle in a background thread."""
        if self._train_thread is not None and self._train_thread.is_alive():
            return False
        from .adversarial import PAIREDConfig, PAIREDTrainer
        from .train import ADITrainer, PuzzleTask, TrainConfig
        model = self.ensure_model()
        puzzle = self.base_puzzle
        self._train_stop.clear()
        self.training_mode = "paired" if adversarial else "adi"

        def run() -> None:
            try:
                if self.trainer is None:
                    task = PuzzleTask(puzzle, self.device, model.cfg, self.pdbs())
                    cfg = TrainConfig(batch_size=batch_size, max_depth=min(24, model.cfg.max_dist), start_depth=2,
                                      depth_grow_every=30, target_sync=20, lr=2e-3)
                    self.trainer = ADITrainer(model, task, cfg, self.telemetry)
                trainer = self.trainer
                self.telemetry.event(f"{self.training_mode.upper()} training started on {puzzle.name} ({self.device})")
                if adversarial:
                    paired = PAIREDTrainer(trainer, cfg=PAIREDConfig(batch_size=48))
                    paired.fit(steps, should_stop=self._train_stop.is_set)
                    if paired.hard_examples:
                        self.telemetry.event("hardest adversarial scramble: " + paired.hard_examples[0]["scramble"])
                else:
                    trainer.fit(steps, should_stop=self._train_stop.is_set)
                self.model_trained_on.add(puzzle.name)
                self._solver_cache = {}
                self.telemetry.event(f"training finished at step {trainer.step}")
            except Exception as exc:                                    # surfaced in the UI log
                self.telemetry.event(f"training failed: {type(exc).__name__}: {exc}")
            finally:
                self.training_mode = "idle"
                if on_done is not None:
                    on_done()

        self._train_thread = threading.Thread(target=run, name="twisty-train", daemon=True)
        self._train_thread.start()
        return True

    def stop_training(self) -> None:
        self._train_stop.set()
        t = self._train_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=30)
        self._train_thread = None

    @property
    def training(self) -> bool:
        return self._train_thread is not None and self._train_thread.is_alive()

    # -- timed playback ---------------------------------------------------------------------------
    def _clear_play(self) -> None:
        self._play_queue.clear()
        self._play_current = None

    @property
    def playing(self) -> bool:
        return self._play_current is not None or bool(self._play_queue)

    def play(self, moves: str | list[str], move_time: Optional[float] = None) -> list[str]:
        """Queue moves for animated playback; they are applied one by one as frames are rendered."""
        with self._lock:
            names = self.puzzle.names_of(self.puzzle.parse_moves(moves))
            dur = float(move_time) if move_time else self.move_time
            self._play_queue.extend((n, max(dur, 0.01)) for n in names)
            return names

    def flush_play(self) -> None:
        """Apply every queued move immediately."""
        with self._lock:
            pending = ([self._play_current] if self._play_current else []) + list(self._play_queue)
            self._clear_play()
            for name, _ in pending:
                try:
                    self._apply_named(name)
                except (KeyError, ValueError):
                    break

    def _apply_named(self, name: str) -> None:
        self.state = self.puzzle.apply(self.state, self.puzzle.action_index(name))
        self.history.append(name)

    def _advance_play(self, now: float) -> Optional[tuple[str, float]]:
        """Advance the playback clock; returns ``(move, fraction)`` of the move in progress."""
        with self._lock:
            while True:
                if self._play_current is None:
                    if not self._play_queue:
                        return None
                    self._play_current = self._play_queue.popleft()
                    self._play_t0 = self._play_free_at if 0 <= now - self._play_free_at < 0.25 else now
                name, dur = self._play_current
                t = (now - self._play_t0) / dur
                if t < 1.0:
                    return name, t
                self._play_current = None
                self._play_free_at = self._play_t0 + dur
                try:
                    self._apply_named(name)
                except (KeyError, ValueError):
                    self._clear_play()
                    return None

    # -- idle show --------------------------------------------------------------------------------
    def set_show(self, mode: Optional[str] = None, paused: Optional[bool] = None, depth: Optional[int] = None,
                 pause: Optional[float] = None, move_time: Optional[float] = None,
                 daily_depth: Optional[int] = None, carousel: Optional[list] = None) -> dict:
        """Configure what the puzzle does while idle (see :data:`SHOW_MODES`).

        ``carousel`` is a list of puzzle names the attract show rotates through, one per round
        (an empty list stays on the current puzzle).
        """
        if mode is not None and mode not in SHOW_MODES:
            raise ValueError(f"unknown show mode {mode!r}; choose from {', '.join(SHOW_MODES)}")
        if carousel is not None:
            try:
                for name in carousel:
                    get_puzzle(name)
            except KeyError as exc:
                raise ValueError(str(exc)) from None
        with self._lock:
            if carousel is not None:
                self.show_carousel = [str(n) for n in carousel]
                self._show_rounds = 0
            if depth is not None:
                self.show_depth = max(1, int(depth))
            if daily_depth is not None:
                self.show_daily_depth = max(1, int(daily_depth))
            if pause is not None:
                self.show_pause = max(0.0, float(pause))
            if move_time is not None:
                self.move_time = max(0.02, float(move_time))
            restart = mode is not None and (mode != self.show_mode or self.show_paused)
            if mode is not None:
                self.show_mode = mode
                self.show_paused = False
            if paused is not None:
                restart = restart or (self.show_paused and not paused)
                self.show_paused = bool(paused)
            if restart and not self.show_paused:
                self._show_phase, self._show_until = "pause", time.monotonic() + 0.6
                self.show_caption = ""
                if self.show_mode == "daily":
                    today = datetime.date.today()
                    moves = self.scramble(self.show_daily_depth, seed=today.toordinal())
                    self.show_caption = f"scramble of the day ({today:%d %b}): " + " ".join(moves)
            return self.show_status()

    def pause_show(self) -> None:
        """Called when a human takes over the puzzle."""
        self.show_paused = True

    def show_status(self) -> dict:
        return dict(mode=self.show_mode, paused=self.show_paused, phase=self._show_phase,
                    caption=self.show_caption, queued=len(self._play_queue) + (self._play_current is not None),
                    carousel=list(self.show_carousel))

    def _show_tick(self, now: float) -> None:
        """Attract-mode state machine, advanced whenever a frame is rendered."""
        if self.show_mode != "attract" or self.show_paused:
            return
        phase = self._show_phase
        if phase == "pause":
            if now < self._show_until:
                return
            if self.show_carousel and self._show_rounds > 0 and not self.training:
                names = self.show_carousel
                cur = self.base_puzzle.name
                nxt = names[(names.index(cur) + 1) % len(names)] if cur in names else names[0]
                self._show_rounds = 0
                if nxt != cur:
                    self.set_puzzle(nxt)                  # restarts the pause; the next tick scrambles it
                    self._show_until = now + 0.3
                    return
            with self._lock:
                self._show_rounds += 1
                self._clear_play()
                self.state = self.puzzle.identity.copy()
                _, actions = self.puzzle.scramble(self.rng, self.show_depth)
                names = self.puzzle.names_of(actions)
                self.history.clear()
                self.scramble_moves = self._show_scramble = names
                self._play_queue.extend((n, max(self.move_time * 0.55, 0.01)) for n in names)
                self.show_caption = "scramble: " + " ".join(names)
                self._show_phase = "scrambling"
        elif phase == "scrambling":
            if self.playing:
                return
            self._show_result = None

            def work() -> None:
                try:
                    self._show_result = self.solve(method="auto", time_limit=self.show_time_limit, quiet=True)
                except Exception as exc:                                  # shown in the caption below
                    self._show_result = SearchResult(solved=False, reason=f"{type(exc).__name__}: {exc}")

            self._show_thread = threading.Thread(target=work, name="twisty-show", daemon=True)
            self._show_thread.start()
            self.show_caption = "searching …"
            self._show_phase = "solving"
        elif phase == "solving":
            if self._show_thread is not None and self._show_thread.is_alive():
                return
            res = self._show_result
            with self._lock:
                if self._show_phase != "solving" or self.show_paused:
                    return
                if res is not None and res.solved:
                    moves = res.moves
                    text = " ".join(self.base_puzzle.format_moves(moves))
                    self.show_caption = (
                        f"solved in {len(moves)} moves (optimal): {text}  ·  {res.nodes_generated:,} nodes in {res.elapsed:.2f}s"
                        if res.optimal_proven else f"solved in {len(moves)} moves (chain solver): {text}")
                else:
                    inv = self.puzzle.inverse
                    moves = self.puzzle.names_of(
                        [int(inv[self.puzzle.action_index(n)]) for n in reversed(self._show_scramble)])
                    self.show_caption = "search budget exceeded, undoing the scramble: " + " ".join(moves)
                dur = max(0.04, min(self.move_time, self.show_max_play / max(len(moves), 1)))
                self._play_queue.extend((n, dur) for n in moves)
                self._show_phase = "playing"
        elif phase == "playing":
            if self.playing:
                return
            self._show_phase, self._show_until = "pause", now + self.show_pause

    # -- rendering --------------------------------------------------------------------------------
    def camera_yaw(self) -> float:
        if self.spin:
            return self.yaw + self.spin_speed * (time.monotonic() - self._spin_t0)
        return self.yaw

    def rotate(self, dyaw: float = 0.0, dpitch: float = 0.0) -> None:
        self.yaw = self.camera_yaw() + dyaw
        self._spin_t0 = time.monotonic()
        self.pitch = float(np.clip(self.pitch + dpitch, -1.5, 1.5))

    def set_style(self, bg: Optional[str] = None, aspect: Optional[float] = None, quality: Optional[int] = None,
                  pitch: Optional[float] = None, zoom: Optional[float] = None) -> dict:
        """Tune the 3D view.  ``bg`` is ``"#rrggbb"`` (``""`` or ``"none"`` forgets it)."""
        if bg is not None:
            text = bg.strip().lstrip("#").lower()
            if text in ("", "none"):
                self.background = None
            elif len(text) == 6 and all(ch in "0123456789abcdef" for ch in text):
                self.background = (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
            else:
                raise ValueError(f"background must look like #rrggbb, got {bg!r}")
        if aspect is not None:
            self.cell_aspect = float(np.clip(float(aspect), 1.0, 4.0))
        if quality is not None:
            self.quality = int(np.clip(int(quality), 1, 6))
        if pitch is not None:
            self.pitch = float(np.clip(float(pitch), -1.5, 1.5))
        if zoom is not None:
            self.zoom = float(np.clip(float(zoom), 0.3, 3.0))
        return dict(bg=None if self.background is None else "#%02x%02x%02x" % self.background,
                    aspect=self.cell_aspect, quality=self.quality, pitch=self.pitch, zoom=self.zoom)

    def set_spin(self, on: bool) -> None:
        self.yaw = self.camera_yaw()
        self._spin_t0 = time.monotonic()
        self.spin = on

    def render(self, width: int = 48, height: int = 24, view: str = "3d",
               anim: Optional[tuple[str, float]] = None, state: Optional[np.ndarray] = None) -> Frame:
        if state is None and anim is None:
            now = time.monotonic()
            self._show_tick(now)                          # may switch puzzle (carousel)
            anim = self._advance_play(now)
        with self._lock:
            r = get_renderer(self.puzzle)
            st = self.state if state is None else state
        if view == "net":
            return r.render_net(st, width, height)
        return r.render_3d(st, width, height, yaw=self.camera_yaw(), pitch=self.pitch, zoom=self.zoom, anim=anim,
                           supersample=self.quality, bg=self.background, cell_aspect=self.cell_aspect,
                           fit=True, distance=self.camera_distance)

    def status(self) -> dict:
        res = self.last_result
        return dict(
            puzzle=self.puzzle.name, description=self.puzzle.spec.description, facets=self.puzzle.n,
            actions=self.puzzle.action_names, solved=self.is_solved(), moves=len(self.history),
            history=self.history[-40:], scramble=self.scramble_moves,
            scramble_text=" ".join(self.puzzle.format_moves(self.scramble_moves)), device=str(self.device),
            training=self.training, training_mode=self.training_mode,
            model_ready=self.model is not None and self.base_puzzle.name in self.model_trained_on,
            group_order=self.telemetry.get("group_order", "…"),
            last_solution=res.moves if res is not None and res.solved else [],
            puzzles=list_puzzles(), legal=self.legal_moves(), show=self.show_status(),
        )
