"""Interactive session shared by the TUI, the Emacs daemon and the CLI.

A session owns one puzzle state plus the lazily built heavy objects around it (BSGS, pattern
databases, model / trainer) and exposes thread-safe verbs: apply moves, scramble, solve, mine
macros, train in the background, render.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

from .group_theory import MacroMiner, PuzzleGroup
from .model import ModelConfig, RelationalGraphTransformer, load_checkpoint
from .puzzles import get_puzzle, list_puzzles
from .renderer_3d import Frame, get_renderer
from .schema import Puzzle
from .search import SearchResult, build_pdbs, make_solver
from .utils import Telemetry, human_int, pick_device


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
        self.yaw, self.pitch, self.zoom = 0.6, 0.45, 1.0
        self.spin = True
        self.spin_speed = 0.6
        self._spin_t0 = time.monotonic()
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
            self._group: Optional[PuzzleGroup] = None
            self._pdbs = None
            self._solver_cache: dict = {}
            self.trainer = None
            self.mined = []
            self.last_result = None
            self.telemetry.set(puzzle=self.puzzle.name, facets=self.puzzle.n, actions=self.puzzle.num_actions,
                               group_order="…", macros=[])
            self.telemetry.event(f"loaded {self.puzzle.name}: {self.puzzle.spec.description}")

    @property
    def group(self) -> PuzzleGroup:
        with self._lock:
            if self._group is None:
                t0 = time.perf_counter()
                self._group = PuzzleGroup(self.base_puzzle)
                self.telemetry.set(group_order=human_int(self._group.num_states), bsgs_base=len(self._group.bsgs.base),
                                   bsgs_time=time.perf_counter() - t0)
            return self._group

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
            state = self.state
            for a in actions:
                state = self.puzzle.apply(state, a)
            self.state = state
            names = self.puzzle.names_of(actions)
            self.history.extend(names)
            return names

    def undo(self) -> Optional[str]:
        with self._lock:
            if not self.history:
                return None
            name = self.history.pop()
            a = int(self.puzzle.inverse[self.puzzle.action_index(name)])
            self.state = self.puzzle.apply(self.state, a)
            return self.puzzle.action_names[a]

    def reset(self) -> None:
        with self._lock:
            self.state = self.puzzle.identity.copy()
            self.history.clear()
            self.scramble_moves = []

    def scramble(self, depth: int = 20) -> list[str]:
        with self._lock:
            self.state, actions = self.puzzle.scramble(self.rng, depth)
            self.history.clear()
            self.scramble_moves = self.puzzle.names_of(actions)
            return self.scramble_moves

    def set_state(self, P) -> None:
        with self._lock:
            self.state = np.asarray(P, dtype=np.intp).copy()
            self.history.clear()

    def set_colors(self, colors) -> bool:
        """Load a colour state; returns ``False`` (state unchanged) when it is unsolvable."""
        P = self.group.state_from_colors(np.asarray(colors, dtype=np.intp))
        if P is None:
            return False
        self.set_state(P)
        return True

    def colors(self) -> list[int]:
        return self.puzzle.colors(self.state).tolist()

    # -- search -----------------------------------------------------------------------------------
    def pdbs(self):
        with self._lock:
            if self._pdbs is None:
                budget = 300_000 if self.base_puzzle.n <= 60 else 60_000
                t0 = time.perf_counter()
                self._pdbs = build_pdbs(self.base_puzzle, max_entries=budget, count=2, device=self.device,
                                        group=self.group)
                self.telemetry.event(f"built {len(self._pdbs)} pattern databases "
                                     f"({sum(p.size for p in self._pdbs):,} entries, {time.perf_counter() - t0:.1f}s)")
            return self._pdbs

    def solve(self, method: str = "auto", time_limit: float = 20.0, weight: Optional[float] = None,
              use_model: Optional[bool] = None) -> SearchResult:
        """Search for a solution of the current state (does not modify it)."""
        with self._lock:
            state = self.state.copy()
            puzzle = self.puzzle
        if not self.group.contains_state(state):
            res = SearchResult(solved=False, reason="unsolvable: state is not in the puzzle group")
            self.last_result = res
            self.telemetry.event(res.reason)
            return res
        self._cancel_search.clear()
        pdbs = self.pdbs()
        have_model = self.model is not None and self.base_puzzle.name in self.model_trained_on
        use_model = have_model if use_model is None else (use_model and self.model is not None)
        if method == "auto":
            method = "astar"
        key = (puzzle.num_actions, method, use_model, weight)
        solver = self._solver_cache.get(key)
        if solver is None:
            w = weight if weight is not None else (1.4 if use_model else 1.0)
            kwargs = dict(optimal=not use_model and w <= 1.0) if method == "astar" else {}
            solver = make_solver(puzzle, self.device, model=self.model if use_model else None, method=method,
                                 pdbs=pdbs, weight=w, telemetry=self.telemetry,
                                 batch_size=256 if use_model else 1024, **kwargs)
            self._solver_cache = {key: solver}
        res = solver.solve(state, time_limit=time_limit, should_stop=self._cancel_search.is_set)
        self.last_result = res
        self.telemetry.set(last_solution=" ".join(res.moves) if res.solved else "", last_solution_len=len(res.moves),
                           search_nodes=res.nodes_generated)
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

    # -- rendering --------------------------------------------------------------------------------
    def camera_yaw(self) -> float:
        if self.spin:
            return self.yaw + self.spin_speed * (time.monotonic() - self._spin_t0)
        return self.yaw

    def rotate(self, dyaw: float = 0.0, dpitch: float = 0.0) -> None:
        self.yaw = self.camera_yaw() + dyaw
        self._spin_t0 = time.monotonic()
        self.pitch = float(np.clip(self.pitch + dpitch, -1.5, 1.5))

    def set_spin(self, on: bool) -> None:
        self.yaw = self.camera_yaw()
        self._spin_t0 = time.monotonic()
        self.spin = on

    def render(self, width: int = 48, height: int = 24, view: str = "3d",
               anim: Optional[tuple[str, float]] = None, state: Optional[np.ndarray] = None) -> Frame:
        r = get_renderer(self.puzzle)
        st = self.state if state is None else state
        if view == "net":
            return r.render_net(st, width, height)
        return r.render_3d(st, width, height, yaw=self.camera_yaw(), pitch=self.pitch, zoom=self.zoom, anim=anim)

    def status(self) -> dict:
        res = self.last_result
        return dict(
            puzzle=self.puzzle.name, description=self.puzzle.spec.description, facets=self.puzzle.n,
            actions=self.puzzle.action_names, solved=self.is_solved(), moves=len(self.history),
            history=self.history[-40:], scramble=self.scramble_moves, device=str(self.device),
            training=self.training, training_mode=self.training_mode,
            model_ready=self.model is not None and self.base_puzzle.name in self.model_trained_on,
            group_order=self.telemetry.get("group_order", "…"),
            last_solution=res.moves if res is not None and res.solved else [],
            puzzles=list_puzzles(), legal=self.legal_moves(),
        )
