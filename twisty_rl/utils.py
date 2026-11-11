"""Device selection, seeding and a tiny thread-safe telemetry bus."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import torch


def pick_device(name: Optional[str] = None) -> torch.device:
    """``auto`` prefers Apple-Silicon MPS, then CUDA, then CPU."""
    if name in (None, "auto"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync_device(device: torch.device) -> None:
    """Block until queued kernels finished (for honest timings)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def human_int(x: int) -> str:
    """Compact rendering of huge integers, e.g. ``4.33e19``."""
    if abs(x) < 10 ** 7:
        return f"{x:,}"
    s = str(abs(x))
    return f"{'-' if x < 0 else ''}{s[0]}.{s[1:3]}e{len(s) - 1}"


class Telemetry:
    """Thread-safe metric store shared by trainers, searches, the TUI and the Emacs daemon."""

    def __init__(self, history: int = 2048):
        self._lock = threading.Lock()
        self._series: dict[str, deque] = {}
        self._scalars: dict[str, Any] = {}
        self._events: deque = deque(maxlen=256)
        self._history = history

    def log(self, **metrics: float) -> None:
        with self._lock:
            for k, v in metrics.items():
                self._series.setdefault(k, deque(maxlen=self._history)).append(float(v))
                self._scalars[k] = float(v)

    def set(self, **values: Any) -> None:
        with self._lock:
            self._scalars.update(values)

    def event(self, text: str) -> None:
        with self._lock:
            self._events.append((time.time(), text))

    def series(self, key: str) -> list[float]:
        with self._lock:
            return list(self._series.get(key, ()))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._scalars.get(key, default)

    def events(self, last: int = 20) -> list[str]:
        with self._lock:
            return [t for _, t in list(self._events)[-last:]]

    def snapshot(self, tail: int = 120) -> dict:
        with self._lock:
            return dict(
                scalars={k: v for k, v in self._scalars.items() if isinstance(v, (int, float, str, bool, list))},
                series={k: list(v)[-tail:] for k, v in self._series.items()},
                events=[t for _, t in list(self._events)[-20:]],
            )
