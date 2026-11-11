"""Vectorised permutation environment (GPU / MPS / CPU) and Gymnasium wrappers.

All state transitions are pure tensor gathers so thousands of puzzle instances advance in a
single kernel launch.  States are permutation states ``P`` (``int16``); the observation fed to
networks is the colour state ``solved_colors[P]``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
import torch

from .schema import Puzzle
from .utils import pick_device


class TensorPuzzle:
    """Device-resident move tables of a compiled :class:`Puzzle`."""

    state_dtype = torch.int16

    def __init__(self, puzzle: Puzzle, device: torch.device | str | None = None):
        self.puzzle = puzzle
        self.device = pick_device(device) if not isinstance(device, torch.device) else device
        dev = self.device
        self.n = puzzle.n
        self.num_actions = puzzle.num_actions
        self.num_primitive = puzzle.num_primitive
        self.K = puzzle.K
        self.conditional = not puzzle.is_group
        self.anchor = puzzle.anchor
        self.table = torch.as_tensor(puzzle.table, dtype=torch.long, device=dev)            # [A, K, N]
        self.table_k = self.table.permute(1, 0, 2).contiguous()                             # [K, A, N]
        self.legal_k = torch.as_tensor(puzzle.legal.T.copy(), dtype=torch.bool, device=dev)  # [K, A]
        self.inverse = torch.as_tensor(puzzle.inverse, dtype=torch.long, device=dev)
        self.cost = torch.as_tensor(puzzle.cost, dtype=torch.float32, device=dev)
        self.solved_colors = torch.as_tensor(puzzle.solved_colors, dtype=torch.long, device=dev)
        self.identity = torch.arange(self.n, dtype=self.state_dtype, device=dev)
        self.primitive_mask = (torch.arange(self.num_actions, device=dev) < self.num_primitive)
        self._arange_a = torch.arange(self.num_actions, device=dev)

    # -- basic state algebra ---------------------------------------------------------------
    def solved(self, batch: int) -> torch.Tensor:
        return self.identity.unsqueeze(0).repeat(batch, 1)

    def colors(self, P: torch.Tensor) -> torch.Tensor:
        return self.solved_colors[P.long()]

    def is_solved(self, P: torch.Tensor) -> torch.Tensor:
        return (self.colors(P) == self.solved_colors).all(dim=-1)

    def anchor_pos(self, P: torch.Tensor) -> torch.Tensor:
        if not self.conditional:
            return torch.zeros(P.shape[0], dtype=torch.long, device=P.device)
        return (P == self.anchor).to(torch.int32).argmax(dim=-1)

    def legal_mask(self, P: torch.Tensor) -> torch.Tensor:
        """``[B, A]`` boolean mask of legal actions."""
        return self.legal_k[self.anchor_pos(P)]

    def apply(self, P: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply one action per state (illegal actions are no-ops)."""
        idx = self.table[actions, self.anchor_pos(P)]
        return torch.gather(P, 1, idx)

    def expand(self, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """All successors: ``children [B, A, N]`` and ``legal [B, A]``."""
        if not self.conditional:
            children = P[:, self.table[:, 0]]
            legal = self.legal_k[0].unsqueeze(0).expand(P.shape[0], -1)
            return children, legal
        pos = self.anchor_pos(P)
        idx = self.table_k[pos]                                               # [B, A, N]
        children = torch.gather(P.unsqueeze(1).expand(-1, self.num_actions, -1), 2, idx)
        return children, self.legal_k[pos]

    def apply_sequence(self, P: torch.Tensor, actions: Sequence[int]) -> torch.Tensor:
        for a in actions:
            P = self.apply(P, torch.full((P.shape[0],), int(a), dtype=torch.long, device=P.device))
        return P

    # -- vectorised backward walks ------------------------------------------------------------
    def random_walk(self, batch: int, depths: torch.Tensor, generator: Optional[torch.Generator] = None,
                    primitive_only: bool = True) -> torch.Tensor:
        """Scramble ``batch`` solved puzzles; row ``b`` receives ``depths[b]`` random non-backtracking moves."""
        P = self.solved(batch)
        prev = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        max_depth = int(depths.max().item()) if batch > 0 else 0
        for t in range(max_depth):
            mask = self.legal_mask(P)
            if primitive_only:
                mask = mask & self.primitive_mask
            back = self.inverse[prev.clamp(min=0)]
            no_back = (self._arange_a.unsqueeze(0) != back.unsqueeze(1)) | (prev < 0).unsqueeze(1)
            cand = mask & no_back
            cand = torch.where(cand.any(dim=1, keepdim=True), cand, mask)
            a = torch.multinomial(cand.float(), 1, generator=generator).squeeze(1)
            active = depths > t
            P = torch.where(active.unsqueeze(1), self.apply(P, a), P)
            prev = torch.where(active, a, prev)
        return P

    def from_numpy(self, P: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(P), dtype=self.state_dtype, device=self.device).reshape(-1, self.n)


class VecPuzzleEnv:
    """Batched RL environment: ``num_envs`` independent puzzles living on one device.

    Reward is ``-cost(action)`` per step; an episode terminates when the puzzle is solved and is
    truncated after ``max_steps``.  Finished sub-environments are re-scrambled automatically.
    """

    def __init__(self, puzzle: Puzzle, num_envs: int = 1024, device: torch.device | str | None = None,
                 max_steps: int = 64, scramble_depth: tuple[int, int] = (1, 20), seed: int = 0,
                 auto_reset: bool = True):
        self.tp = TensorPuzzle(puzzle, device)
        self.puzzle = puzzle
        self.num_envs = num_envs
        self.device = self.tp.device
        self.max_steps = max_steps
        self.scramble_depth = scramble_depth
        self.auto_reset = auto_reset
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.state = self.tp.solved(num_envs)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.depth = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def num_actions(self) -> int:
        return self.tp.num_actions

    def _sample_depths(self, batch: int) -> torch.Tensor:
        lo, hi = self.scramble_depth
        return torch.randint(lo, hi + 1, (batch,), device=self.device, generator=self.generator)

    def observe(self) -> torch.Tensor:
        return self.tp.colors(self.state)

    def reset(self, depths: Optional[torch.Tensor] = None) -> torch.Tensor:
        depths = self._sample_depths(self.num_envs) if depths is None else depths.to(self.device)
        self.state = self.tp.random_walk(self.num_envs, depths, self.generator)
        self.depth = depths
        self.steps.zero_()
        return self.observe()

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device).long()
        legal = self.tp.legal_mask(self.state).gather(1, actions.unsqueeze(1)).squeeze(1)
        self.state = self.tp.apply(self.state, actions)
        self.steps += 1
        reward = -self.tp.cost[actions]
        terminated = self.tp.is_solved(self.state)
        truncated = (self.steps >= self.max_steps) & ~terminated
        info = {"legal": legal, "solved_state": self.state.clone(), "episode_steps": self.steps.clone()}
        done = terminated | truncated
        if self.auto_reset and bool(done.any()):
            depths = self._sample_depths(self.num_envs)
            fresh = self.tp.random_walk(self.num_envs, depths, self.generator)
            self.state = torch.where(done.unsqueeze(1), fresh, self.state)
            self.depth = torch.where(done, depths, self.depth)
            self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
        return self.observe(), reward, terminated, truncated, info


class TwistyGymEnv(gym.Env):
    """Single-instance Gymnasium environment (numpy backend)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, puzzle: Puzzle | str, scramble_depth: tuple[int, int] = (1, 20), max_steps: int = 64,
                 render_mode: Optional[str] = None):
        super().__init__()
        if isinstance(puzzle, str):
            from .puzzles import get_puzzle
            puzzle = get_puzzle(puzzle)
        self.puzzle = puzzle
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.MultiDiscrete(np.full(puzzle.n, puzzle.num_colors, dtype=np.int64))
        self.action_space = gym.spaces.Discrete(puzzle.num_actions)
        self.state = puzzle.identity.copy()
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return self.puzzle.colors(self.state).astype(np.int64)

    def action_masks(self) -> np.ndarray:
        return self.puzzle.legal[:, self.puzzle.anchor_pos(self.state)].copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        lo, hi = self.scramble_depth
        depth = int((options or {}).get("depth", self.np_random.integers(lo, hi + 1)))
        self.state, scramble = self.puzzle.scramble(self.np_random, depth)
        self._steps = 0
        return self._obs(), {"scramble": self.puzzle.names_of(scramble), "action_mask": self.action_masks()}

    def step(self, action: int):
        action = int(action)
        mask = self.action_masks()
        legal = bool(mask[action])
        if legal:
            self.state = self.puzzle.apply(self.state, action)
        self._steps += 1
        terminated = self.puzzle.is_solved(self.state)
        truncated = self._steps >= self.max_steps and not terminated
        reward = -float(self.puzzle.cost[action])
        return self._obs(), reward, terminated, truncated, {"legal": legal, "action_mask": self.action_masks()}

    def render(self):
        if self.render_mode == "ansi":
            from .renderer_3d import render_net_ansi
            return render_net_ansi(self.puzzle, self.state)
        return None


class TwistyVectorEnv(gym.vector.VectorEnv):
    """Gymnasium ``VectorEnv`` facade over :class:`VecPuzzleEnv` (numpy in / numpy out)."""

    def __init__(self, puzzle: Puzzle | str, num_envs: int = 64, device: torch.device | str | None = None,
                 max_steps: int = 64, scramble_depth: tuple[int, int] = (1, 20), seed: int = 0):
        if isinstance(puzzle, str):
            from .puzzles import get_puzzle
            puzzle = get_puzzle(puzzle)
        self.env = VecPuzzleEnv(puzzle, num_envs, device, max_steps, scramble_depth, seed)
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.MultiDiscrete(np.full(puzzle.n, puzzle.num_colors, dtype=np.int64))
        self.single_action_space = gym.spaces.Discrete(puzzle.num_actions)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.env.generator.manual_seed(int(seed))
        obs = self.env.reset()
        return obs.cpu().numpy().astype(np.int64), {}

    def step(self, actions):
        obs, reward, term, trunc, info = self.env.step(torch.as_tensor(np.asarray(actions)))
        return (obs.cpu().numpy().astype(np.int64), reward.cpu().numpy().astype(np.float64),
                term.cpu().numpy(), trunc.cpu().numpy(), {"legal": info["legal"].cpu().numpy()})

    def close_extras(self, **kwargs):
        return None
