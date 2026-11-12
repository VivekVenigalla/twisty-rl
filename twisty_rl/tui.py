"""Textual dashboard: live 3D/2D puzzle views, telemetry plots and interactive solving.

Keys (when the command line is not focused)::

    s scramble   x solve   r reset   u undo      n / p  next / previous puzzle
    t ADI train  a PAIRED  m mine macros  g BSGS   space spin   arrows rotate   + - zoom
    i or :  focus the command line (type moves such as  R U R' U'  or commands such as
            puzzle megaminx | scramble 25 | solve beam | train 500 | promote 2 | help)
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, RichLog, Static

from .puzzles import list_puzzles
from .session import Session
from .utils import Telemetry

BRAILLE_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def braille_plot(values: list[float], width: int, height: int) -> list[str]:
    """Line plot of ``values`` on a ``width x height`` character canvas using braille dots."""
    width, height = max(width, 1), max(height, 1)
    grid = [[0] * width for _ in range(height)]
    if values:
        px_w, px_h = width * 2, height * 4
        vals = np.asarray(values[-px_w:], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = float(vals.min()), float(vals.max())
            span = hi - lo if hi > lo else 1.0
            ys = np.clip(((hi - vals) / span * (px_h - 1)).round().astype(int), 0, px_h - 1)
            prev = None
            for x, y in enumerate(ys):
                x = x if len(vals) >= px_w else int(x * (px_w - 1) / max(len(vals) - 1, 1))
                for yy in ([y] if prev is None else range(min(prev, y), max(prev, y) + 1)):
                    grid[yy // 4][x // 2] |= BRAILLE_BITS[yy % 4][x % 2]
                prev = y
    return ["".join(chr(0x2800 + c) for c in row) for row in grid]


class PuzzleView(Widget):
    """Half-block rendering of the session's puzzle (``view`` is ``"3d"`` or ``"net"``)."""

    DEFAULT_CSS = "PuzzleView { height: 1fr; }"

    def __init__(self, session: Session, view: str = "3d", **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.view = view
        self.anim: Optional[tuple[str, float]] = None
        self.frames = 0
        self._drag: Optional[tuple[int, int]] = None

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        if w < 4 or h < 2:
            return Text("")
        self.frames += 1
        return self.session.render(w, h, self.view, anim=self.anim if self.view == "3d" else None).to_text()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if self.view == "3d":
            self._drag = (event.screen_x, event.screen_y)
            self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag is not None:
            dx, dy = event.screen_x - self._drag[0], event.screen_y - self._drag[1]
            self._drag = (event.screen_x, event.screen_y)
            self.session.rotate(dx * 0.08, dy * 0.12)
            self.refresh()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag is not None:
            self._drag = None
            self.release_mouse()


class Plot(Widget):
    """Braille sparkline of one telemetry series."""

    DEFAULT_CSS = "Plot { height: 7; }"

    def __init__(self, telemetry: Telemetry, key: str, title: str, color: str = "cyan", **kwargs):
        super().__init__(**kwargs)
        self.telemetry, self.key, self.title, self.color = telemetry, key, title, color

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        vals = self.telemetry.series(self.key)
        text = Text(no_wrap=True)
        if vals:
            head = f"{self.title}  last {vals[-1]:.4g}  min {min(vals):.4g}  max {max(vals):.4g}  n={len(vals)}"
        else:
            head = f"{self.title}  (no data yet)"
        text.append(head[:w] + "\n", style="bold")
        for line in braille_plot(vals, w, max(h - 1, 1)):
            text.append(line + "\n", style=self.color)
        return text


class TwistyApp(App):
    TITLE = "TwistyRL"
    CSS = """
    #main { height: 1fr; }
    #left { width: 3fr; }
    #right { width: 2fr; min-width: 46; }
    #view3d { height: 3fr; border: round $accent; }
    #net { height: 2fr; border: round $primary; }
    #stats { height: auto; border: round $secondary; padding: 0 1; }
    #loss, #search { border: round $primary; }
    #macros { height: auto; max-height: 9; border: round $secondary; padding: 0 1; }
    #log { height: 1fr; min-height: 5; border: round $primary; }
    #cmd { dock: bottom; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "scramble", "Scramble"),
        Binding("x", "solve", "Solve"),
        Binding("r", "reset", "Reset"),
        Binding("u", "undo", "Undo", show=False),
        Binding("t", "train", "Train"),
        Binding("a", "paired", "PAIRED"),
        Binding("m", "mine", "Macros"),
        Binding("g", "group", "BSGS"),
        Binding("n", "next_puzzle", "Next"),
        Binding("p", "prev_puzzle", "Prev", show=False),
        Binding("space", "spin", "Spin", show=False),
        Binding("left", "rotate(-0.25, 0)", show=False),
        Binding("right", "rotate(0.25, 0)", show=False),
        Binding("up", "rotate(0, -0.15)", show=False),
        Binding("down", "rotate(0, 0.15)", show=False),
        Binding("plus,equals_sign", "zoom(1.1)", show=False),
        Binding("minus", "zoom(0.9)", show=False),
        Binding("i,colon", "focus_cmd", "Command"),
        Binding("escape", "blur_cmd", show=False),
    ]

    def __init__(self, puzzle: str = "cube3", device: Optional[str] = None, session: Optional[Session] = None,
                 fps: int = 30, checkpoint: Optional[str] = None):
        super().__init__()
        self.session = session or Session(puzzle, device)
        if checkpoint:
            self.session.load_model(checkpoint)
        self.telemetry = self.session.telemetry
        self.fps = fps
        self._queue: deque[str] = deque()
        self._anim_name: Optional[str] = None
        self._anim_t0 = 0.0
        self.anim_duration = 0.16
        self._events_seen = 0
        self._last_frames = 0
        self._last_fps_t = time.monotonic()
        self.measured_fps = 0.0

    # -- layout ---------------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield PuzzleView(self.session, "3d", id="view3d")
                yield PuzzleView(self.session, "net", id="net")
            with Vertical(id="right"):
                yield Static(id="stats")
                yield Plot(self.telemetry, "loss", "Bellman / ADI loss", "green", id="loss")
                yield Plot(self.telemetry, "search_depth", "search depth", "magenta", id="search")
                yield Static(id="macros")
                yield RichLog(id="log", wrap=True, markup=False, max_lines=400)
        yield Input(placeholder="moves (R U R' U') or command (help)", id="cmd")
        yield Footer()

    def on_mount(self) -> None:
        self.view3d = self.query_one("#view3d", PuzzleView)
        self.net = self.query_one("#net", PuzzleView)
        self.view3d.border_title = "3D"
        self.net.border_title = "net"
        self.query_one("#stats").border_title = "telemetry"
        self.query_one("#macros").border_title = "commutator macros"
        self.query_one("#log").border_title = "events"
        self.set_interval(1.0 / self.fps, self._tick)
        self.set_interval(0.25, self._slow_tick)
        self.set_focus(None)
        self._compute_group()
        self._slow_tick()

    # -- timers ---------------------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        if self._anim_name is None and self._queue:
            self._anim_name = self._queue.popleft()
            self._anim_t0 = now
            if self._anim_name not in self.view3d.session.puzzle.spec.geometry.animations:
                self._finish_move()
        if self._anim_name is not None:
            t = (now - self._anim_t0) / self.anim_duration
            if t >= 1.0:
                self._finish_move()
            else:
                self.view3d.anim = (self._anim_name, t)
        self.view3d.refresh()

    def _finish_move(self) -> None:
        name, self._anim_name = self._anim_name, None
        self.view3d.anim = None
        if name is not None:
            try:
                self.session.apply([name])
            except (KeyError, ValueError) as exc:
                self._log(f"move {name!r} rejected: {exc}")
                self._queue.clear()
        self.net.refresh()

    def _slow_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_fps_t
        if dt > 0:
            self.measured_fps = (self.view3d.frames - self._last_frames) / dt
        self._last_frames, self._last_fps_t = self.view3d.frames, now
        s, tel = self.session, self.telemetry
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(overflow="ellipsis", no_wrap=True)
        state = "[green]solved[/]" if s.is_solved() else "[yellow]scrambled[/]"
        table.add_row("puzzle", f"{s.puzzle.name}  ({s.puzzle.n} facets, {s.puzzle.num_actions} actions)")
        table.add_row("|G| (BSGS)", f"{tel.get('group_order', '…')}   base {tel.get('bsgs_base', '…')}")
        table.add_row("state", f"{state}  moves {len(s.history)}  queue {len(self._queue)}")
        table.add_row("device", f"{s.device}   render {self.measured_fps:4.1f} fps")
        mode = s.training_mode if s.training else "idle"
        loss = tel.get("loss")
        table.add_row("training", f"{mode}  step {tel.get('train_step', 0)}  loss "
                                  f"{'—' if loss is None else f'{loss:.4f}'}  depth {int(tel.get('scramble_depth', 0))}")
        nps, depth = tel.get("search_nodes_per_sec"), tel.get("search_depth")
        table.add_row("search", f"{'—' if nps is None else f'{nps:,.0f}'} nodes/s   depth "
                                f"{'—' if depth is None else f'{depth:.0f}'}   nodes {tel.get('search_nodes', 0):,}")
        if tel.get("adv_horizon") is not None:
            table.add_row("adversary", f"H={int(tel.get('adv_horizon'))}  regret {tel.get('adv_regret', 0):.3f}  "
                                       f"solve {tel.get('adv_solve_rate', 0):.2f}")
        sol = tel.get("last_solution") or ""
        table.add_row("solution", sol if sol else "—")
        self.query_one("#stats", Static).update(table)
        macros = tel.get("macros") or []
        self.query_one("#macros", Static).update("\n".join(macros[:7]) if macros else "press m to mine commutators")
        events_ = tel.events(200)
        log = self.query_one("#log", RichLog)
        total = len(events_)
        if total < self._events_seen:
            self._events_seen = 0
        for line in events_[self._events_seen:]:
            log.write(line)
        self._events_seen = total
        self.query_one("#loss", Plot).refresh()
        self.query_one("#search", Plot).refresh()
        self.net.refresh()

    def _log(self, text: str) -> None:
        self.telemetry.event(text)

    # -- background work --------------------------------------------------------------------------
    def _compute_group(self) -> None:
        session = self.session

        def work() -> None:
            g = session.group
            session.telemetry.event(f"BSGS: |G| = {g.num_states:,}  (base length {len(g.bsgs.base)}, "
                                    f"{len(g.bsgs.strong_generators)} strong generators)")

        self.run_worker(work, thread=True, group="group", exclusive=True)

    def _solve(self, method: str = "auto") -> None:
        session = self.session
        if self._queue or self._anim_name:
            return
        self._log(f"searching ({method}) …")

        def work() -> None:
            res = session.solve(method=method)
            if res.solved:
                self.call_from_thread(self._queue.extend, res.moves)

        self.run_worker(work, thread=True, group="solve", exclusive=True)

    def _mine(self, promote: int = 0) -> None:
        session = self.session
        self._log("mining commutators / conjugates …")
        self.run_worker(lambda: session.mine_macros(top_k=8, promote=promote), thread=True, group="mine", exclusive=True)

    def _switch(self, name: str) -> None:
        self._queue.clear()
        self._anim_name = None
        self.view3d.anim = None
        try:
            self.session.set_puzzle(name)
        except KeyError as exc:
            self._log(str(exc))
            return
        self.sub_title = self.session.puzzle.spec.description
        self._compute_group()

    # -- actions --------------------------------------------------------------------------------
    def action_scramble(self, depth: int = 0) -> None:
        self._queue.clear()
        moves = self.session.scramble(depth or min(25, 3 * self.session.puzzle.num_primitive))
        self._log("scramble: " + " ".join(moves))

    def action_solve(self) -> None:
        self._solve()

    def action_reset(self) -> None:
        self._queue.clear()
        self.session.reset()

    def action_undo(self) -> None:
        self.session.undo()

    def action_train(self) -> None:
        if self.session.training:
            self.session.stop_training()
            self._log("training stopped")
        else:
            self.session.start_training(steps=400)

    def action_paired(self) -> None:
        if self.session.training:
            self.session.stop_training()
        else:
            self.session.start_training(steps=150, adversarial=True)

    def action_mine(self) -> None:
        self._mine()

    def action_group(self) -> None:
        info = self.session.group_info()
        self._log(f"|G| = {info['num_states']}  orbits {info['orbit_sizes']}")
        for st in info["curriculum"]:
            self._log(f"  stage {st['stage']}: fix {st['fixed_base_points']} base points, |G_j| = {st['order']}")

    def action_next_puzzle(self) -> None:
        names = list_puzzles()
        self._switch(names[(names.index(self.session.base_puzzle.name) + 1) % len(names)]
                     if self.session.base_puzzle.name in names else names[0])

    def action_prev_puzzle(self) -> None:
        names = list_puzzles()
        self._switch(names[(names.index(self.session.base_puzzle.name) - 1) % len(names)]
                     if self.session.base_puzzle.name in names else names[0])

    def action_spin(self) -> None:
        self.session.set_spin(not self.session.spin)

    def action_rotate(self, dyaw: float, dpitch: float) -> None:
        self.session.rotate(dyaw, dpitch)

    def action_zoom(self, factor: float) -> None:
        self.session.zoom = float(np.clip(self.session.zoom * factor, 0.3, 3.0))

    def action_focus_cmd(self) -> None:
        self.query_one("#cmd", Input).focus()

    def action_blur_cmd(self) -> None:
        self.set_focus(None)

    # -- command line -----------------------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self.run_command(text)

    def run_command(self, text: str) -> None:
        parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]
        num = lambda default: int(args[0]) if args and args[0].isdigit() else default
        if cmd in ("quit", "exit"):
            self.exit()
        elif cmd == "help":
            for line in (__doc__ or "").strip().splitlines():
                self._log(line)
            self._log("puzzles: " + " ".join(list_puzzles()))
        elif cmd == "puzzle" and args:
            self._switch(args[0])
        elif cmd == "scramble":
            self.action_scramble(num(0))
        elif cmd == "solve":
            self._solve(args[0] if args else "auto")
        elif cmd == "reset":
            self.action_reset()
        elif cmd == "undo":
            self.action_undo()
        elif cmd == "train":
            self.session.start_training(steps=num(400))
        elif cmd == "paired":
            self.session.start_training(steps=num(150), adversarial=True)
        elif cmd == "stop":
            self.session.stop_training()
            self.session.cancel_search()
        elif cmd == "mine":
            self._mine()
        elif cmd == "promote":
            self._mine(promote=num(2))
        elif cmd == "group":
            self.action_group()
        elif cmd == "spin":
            self.action_spin()
        elif cmd == "load" and args:
            try:
                self.session.load_model(args[0])
                self._log(f"loaded checkpoint {args[0]}")
            except (OSError, RuntimeError, KeyError) as exc:
                self._log(f"could not load checkpoint: {exc}")
        else:
            try:
                names = self.session.puzzle.names_of(self.session.puzzle.parse_moves(text))
            except KeyError as exc:
                self._log(str(exc))
                return
            self._queue.extend(names)

    def on_unmount(self) -> None:
        self.session.cancel_search()
        self.session.stop_training()


def run_tui(puzzle: str = "cube3", device: Optional[str] = None, checkpoint: Optional[str] = None) -> None:
    TwistyApp(puzzle, device, checkpoint=checkpoint).run()
