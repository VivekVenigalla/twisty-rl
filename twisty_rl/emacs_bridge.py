"""Asynchronous JSON-RPC 2.0 daemon over a Unix domain socket (newline-delimited JSON).

The daemon wraps one :class:`~twisty_rl.session.Session`.  Cheap calls (state, rendering) are
answered inline on the event loop; heavy calls (solve, BSGS, macro mining) run in a worker thread so
frames keep flowing while a search is in progress.  ``emacs/twisty-dashboard.el`` is the reference
client; :class:`BridgeClient` is a small synchronous Python client used by the tests and the CLI.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from typing import Any, Callable, Optional

from .puzzles import list_puzzles
from .session import Session

PROTOCOL_VERSION = "1.0"
HEAVY = {"solve", "group_info", "mine_macros", "set_puzzle", "is_solvable", "set_colors", "show"}


def default_socket_path() -> str:
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(base, f"twisty-rl-{os.getuid()}.sock")


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class TwistyDaemon:
    def __init__(self, session: Optional[Session] = None, socket_path: Optional[str] = None,
                 puzzle: str = "cube3", device: Optional[str] = None, exit_with_parent: bool = False):
        self.session = session or Session(puzzle, device)
        self.socket_path = socket_path or default_socket_path()
        self.exit_with_parent = exit_with_parent
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop: Optional[asyncio.Event] = None
        self.requests_served = 0
        self.methods: dict[str, Callable[..., Any]] = {
            name[4:]: getattr(self, name) for name in dir(self) if name.startswith("rpc_")
        }

    # -- RPC methods ------------------------------------------------------------------------------
    def rpc_ping(self) -> dict:
        return {"pong": True, "version": PROTOCOL_VERSION, "pid": os.getpid()}

    def rpc_list_puzzles(self) -> list[str]:
        return list_puzzles()

    def rpc_status(self) -> dict:
        return self.session.status()

    def rpc_set_puzzle(self, name: str) -> dict:
        try:
            self.session.set_puzzle(name)
        except KeyError as exc:
            raise RpcError(-32602, str(exc)) from None
        return self.session.status()

    def rpc_apply(self, moves: str | list[str]) -> dict:
        self.session.pause_show()
        try:
            applied = self.session.apply(moves)
        except (KeyError, ValueError) as exc:
            raise RpcError(-32602, str(exc)) from None
        return {"applied": applied, "solved": self.session.is_solved()}

    def rpc_play(self, moves: str | list[str], move_time: Optional[float] = None) -> dict:
        """Queue moves for animated playback (applied as frames are rendered)."""
        self.session.pause_show()
        try:
            queued = self.session.play(moves, move_time)
        except (KeyError, ValueError) as exc:
            raise RpcError(-32602, str(exc)) from None
        return {"queued": queued}

    def rpc_show(self, mode: Optional[str] = None, paused: Optional[bool] = None, depth: Optional[int] = None,
                 pause: Optional[float] = None, move_time: Optional[float] = None,
                 daily_depth: Optional[int] = None) -> dict:
        """Idle behaviour: ``spin``, ``attract`` (scramble and solve forever) or ``daily``."""
        try:
            return self.session.set_show(mode, paused, depth, pause, move_time, daily_depth)
        except ValueError as exc:
            raise RpcError(-32602, str(exc)) from None

    def rpc_undo(self) -> dict:
        self.session.pause_show()
        return {"undone": self.session.undo(), "solved": self.session.is_solved()}

    def rpc_reset(self) -> dict:
        self.session.pause_show()
        self.session.reset()
        return {"solved": True}

    def rpc_scramble(self, depth: int = 20, seed: Optional[int] = None) -> dict:
        self.session.pause_show()
        return {"scramble": self.session.scramble(int(depth), None if seed is None else int(seed))}

    def rpc_state(self) -> dict:
        s = self.session
        return {"permutation": s.state.tolist(), "colors": s.colors(), "solved": s.is_solved(),
                "palette": [list(c) for c in s.puzzle.spec.palette]}

    def rpc_set_colors(self, colors: list[int]) -> dict:
        self.session.pause_show()
        ok = self.session.set_colors(colors)
        return {"accepted": ok, "reason": "" if ok else "colouring is not reachable (parity / orientation violation)"}

    def rpc_is_solvable(self) -> dict:
        return {"solvable": self.session.is_solvable()}

    def rpc_rotate(self, dyaw: float = 0.0, dpitch: float = 0.0, zoom: float = 1.0) -> dict:
        self.session.rotate(float(dyaw), float(dpitch))
        self.session.zoom = min(3.0, max(0.3, self.session.zoom * float(zoom)))
        return {"yaw": self.session.camera_yaw(), "pitch": self.session.pitch, "zoom": self.session.zoom}

    def rpc_style(self, bg: Optional[str] = None, aspect: Optional[float] = None, quality: Optional[int] = None,
                  pitch: Optional[float] = None, zoom: Optional[float] = None) -> dict:
        """Look of the 3D view: background colour to blend into, cell aspect, anti-aliasing, camera."""
        try:
            return self.session.set_style(bg, aspect, quality, pitch, zoom)
        except ValueError as exc:
            raise RpcError(-32602, str(exc)) from None

    def rpc_spin(self, on: Optional[bool] = None, speed: Optional[float] = None) -> dict:
        if speed is not None:
            self.session.set_spin(self.session.spin)
            self.session.spin_speed = float(speed)
        self.session.set_spin((not self.session.spin) if on is None else bool(on))
        return {"spin": self.session.spin}

    def rpc_render(self, width: int = 40, height: int = 20, view: str = "3d", format: str = "runs",
                   anim: Optional[list] = None) -> dict:
        width, height = max(4, min(int(width), 400)), max(2, min(int(height), 200))
        frame = self.session.render(width, height, view, anim=tuple(anim) if anim else None)
        if format == "ansi":
            return {"format": "ansi", "ansi": frame.to_ansi(), "width": width, "height": height}
        return {"format": "runs", "rows": frame.to_runs(), "width": width, "height": height}

    def rpc_dashboard(self, width: int = 40, height: int = 20, view: str = "3d", net_width: int = 0,
                      net_height: int = 0, tail: int = 60) -> dict:
        """Everything a client needs for one refresh, in a single round trip."""
        out = {"frame": self.rpc_render(width, height, view), "status": self.session.status(),
               "telemetry": self.session.telemetry.snapshot(int(tail))}
        if net_width and net_height:
            out["net"] = self.rpc_render(net_width, net_height, "net")
        return out

    def rpc_solve(self, method: str = "auto", time_limit: float = 20.0, apply: bool = False) -> dict:
        res = self.session.solve(method=method, time_limit=float(time_limit))
        if res.solved and apply:
            self.session.apply(res.moves)
        return res.to_dict()

    def rpc_cancel(self) -> dict:
        self.session.cancel_search()
        self.session.stop_training()
        return {"cancelled": True}

    def rpc_group_info(self) -> dict:
        return self.session.group_info()

    def rpc_mine_macros(self, top_k: int = 8, promote: int = 0) -> list[dict]:
        return self.session.mine_macros(top_k=int(top_k), promote=int(promote))

    def rpc_train(self, steps: int = 300, adversarial: bool = False) -> dict:
        started = self.session.start_training(steps=int(steps), adversarial=bool(adversarial))
        return {"started": started, "mode": self.session.training_mode}

    def rpc_train_stop(self) -> dict:
        self.session.stop_training()
        return {"training": False}

    def rpc_telemetry(self, tail: int = 120) -> dict:
        return self.session.telemetry.snapshot(int(tail))

    def rpc_shutdown(self) -> dict:
        if self._stop is not None:
            asyncio.get_running_loop().call_soon(self._stop.set)
        return {"bye": True}

    # -- protocol -----------------------------------------------------------------------------------
    async def dispatch(self, message: dict) -> Optional[dict]:
        mid = message.get("id")
        try:
            if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
                raise RpcError(-32600, "invalid request")
            fn = self.methods.get(message["method"])
            if fn is None:
                raise RpcError(-32601, f"method not found: {message['method']}")
            params = message.get("params") or {}
            args, kwargs = (params, {}) if isinstance(params, list) else ((), params)
            try:
                if message["method"] in HEAVY:
                    result = await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))
                else:
                    result = fn(*args, **kwargs)
            except TypeError as exc:
                raise RpcError(-32602, f"invalid params: {exc}") from None
            self.requests_served += 1
            return None if mid is None else {"jsonrpc": "2.0", "id": mid, "result": result}
        except RpcError as exc:
            return None if mid is None else {"jsonrpc": "2.0", "id": mid, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:                        # never let one request kill the daemon
            return None if mid is None else {"jsonrpc": "2.0", "id": mid,
                                             "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}}

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        pending: set[asyncio.Task] = set()
        lock = asyncio.Lock()

        async def serve(line: bytes) -> None:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response: Optional[dict] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                response = await self.dispatch(message) if isinstance(message, dict) else {
                    "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
            if response is not None:
                async with lock:
                    writer.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                    await writer.drain()

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                if line.strip():
                    task = asyncio.create_task(serve(line))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError):
                pass

    async def serve(self, ready: Optional[Callable[[], None]] = None) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._stop = asyncio.Event()
        self._server = await asyncio.start_unix_server(self._handle, path=self.socket_path, limit=1 << 22)
        os.chmod(self.socket_path, 0o600)
        watchdog = None
        if self.exit_with_parent:
            parent = os.getppid()

            async def watch() -> None:                    # an editor that crashed must not leave us behind
                while os.getppid() == parent:
                    await asyncio.sleep(2.0)
                self._stop.set()

            watchdog = asyncio.create_task(watch())
        if ready is not None:
            ready()
        try:
            await self._stop.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()
            self.session.cancel_search()
            self._server.close()
            await self._server.wait_closed()
            self.session.stop_training()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def run(self) -> None:
        print(f"twisty-rl daemon listening on {self.socket_path}", flush=True)
        try:
            asyncio.run(self.serve())
        except KeyboardInterrupt:
            pass


class DaemonThread:
    """Run a daemon on a background thread (tests, notebooks, embedding)."""

    def __init__(self, daemon: TwistyDaemon):
        import threading
        self.daemon = daemon
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._main, name="twisty-daemon", daemon=True)

    def _main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.daemon.serve(self._ready.set))
        self._loop.close()

    def start(self, timeout: float = 10.0) -> "DaemonThread":
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("daemon did not start")
        return self

    def stop(self) -> None:
        if self._loop is not None and self.daemon._stop is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self.daemon._stop.set)
        self._thread.join(timeout=10)

    def __enter__(self) -> "DaemonThread":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class BridgeClient:
    """Minimal synchronous JSON-RPC client."""

    def __init__(self, socket_path: Optional[str] = None, timeout: float = 60.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(socket_path or default_socket_path())
        self.file = self.sock.makefile("rwb")
        self._id = 0

    def call(self, method: str, **params) -> Any:
        self._id += 1
        self.file.write(json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}).encode() + b"\n")
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                raise ConnectionError("daemon closed the connection")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RpcError(msg["error"]["code"], msg["error"]["message"])
                return msg["result"]

    def raw(self, payload: bytes) -> dict:
        self.file.write(payload.rstrip(b"\n") + b"\n")
        self.file.flush()
        return json.loads(self.file.readline())

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
