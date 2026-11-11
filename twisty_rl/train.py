"""Autodidactic Iteration (ADI) with distributional targets, plus the cross-topology multi-task loader.

Each step:

1. scramble a batch of solved puzzles by vectorised non-backtracking backward walks;
2. expand *all* children on the device and evaluate them with a frozen target network;
3. the distributional Bellman target is the cost-shifted distance distribution of the best child,
   projected into ``[lower, upper]`` where ``upper`` is the scramble length (a certified upper
   bound) and ``lower`` comes from pattern databases (PDB bootstrapping);
4. train the dual-head network: KL divergence on the distance distribution, Huber loss on its
   mean and cross-entropy of the policy prior against the greedy action.

A :class:`MultiTaskLoader` interleaves any number of puzzle topologies through one model.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .env import TensorPuzzle
from .graph_compiler import CompiledGraph, compile_graph
from .group_theory import MacroMiner, MinedMacro, PuzzleGroup, StabilizerCurriculum
from .model import ModelConfig, RelationalGraphTransformer, auto_chunk
from .schema import Puzzle
from .search import PatternDatabase
from .utils import Telemetry, pick_device


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_depth: int = 20
    start_depth: int = 3
    depth_grow_every: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    target_sync: int = 25
    dist_coef: float = 1.0
    expect_coef: float = 0.5
    policy_coef: float = 0.5
    depth_weight_power: float = 0.0
    grad_clip: float = 1.0
    use_upper_bound: bool = True
    stabilizer_mix: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class PuzzleTask:
    """Everything needed to train / evaluate one topology: tables, compiled graph, optional PDBs."""

    def __init__(self, puzzle: Puzzle, device: torch.device, model_cfg: ModelConfig,
                 pdbs: Sequence[PatternDatabase] = (), weight: float = 1.0, max_depth: Optional[int] = None,
                 curriculum: Optional[StabilizerCurriculum] = None):
        self.device = device
        self.model_cfg = model_cfg
        self.pdbs = list(pdbs)
        self.weight = weight
        self.max_depth = max_depth
        self.curriculum = curriculum
        self.mined: list[MinedMacro] = []
        self.refresh(puzzle)

    def refresh(self, puzzle: Puzzle) -> None:
        """(Re)compile after the action space changed (macro promotion)."""
        self.puzzle = puzzle
        self.name = puzzle.name
        self.tp = TensorPuzzle(puzzle, self.device)
        self.graph: CompiledGraph = compile_graph(puzzle, self.model_cfg.lap_k, self.model_cfg.rw_k, self.device)
        self.chunk = auto_chunk(self.graph, self.model_cfg)


class MultiTaskLoader:
    """Weighted interleaving of puzzle tasks for joint cross-topology training."""

    def __init__(self, tasks: Sequence[PuzzleTask], seed: int = 0):
        if not tasks:
            raise ValueError("at least one task is required")
        self.tasks = list(tasks)
        self.rng = np.random.default_rng(seed)
        self._cursor = 0
        w = np.asarray([t.weight for t in self.tasks], dtype=np.float64)
        self.probs = w / w.sum()

    def __len__(self) -> int:
        return len(self.tasks)

    def by_name(self, name: str) -> PuzzleTask:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(name)

    def sample(self) -> PuzzleTask:
        """Round-robin for the first sweep (every topology is seen early), weighted sampling afterwards."""
        if self._cursor < len(self.tasks):
            task = self.tasks[self._cursor]
            self._cursor += 1
            return task
        return self.tasks[int(self.rng.choice(len(self.tasks), p=self.probs))]


class ADITrainer:
    def __init__(self, model: RelationalGraphTransformer, tasks: Sequence[PuzzleTask] | PuzzleTask,
                 cfg: Optional[TrainConfig] = None, telemetry: Optional[Telemetry] = None):
        self.cfg = cfg or TrainConfig()
        self.model = model
        self.device = next(model.parameters()).device
        self.loader = MultiTaskLoader([tasks] if isinstance(tasks, PuzzleTask) else tasks, self.cfg.seed)
        self.target = copy.deepcopy(model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.telemetry = telemetry or Telemetry()
        self.step = 0
        self.depth = {t.name: min(self.cfg.start_depth, t.max_depth or self.cfg.max_depth) for t in self.loader.tasks}
        self.task_steps = {t.name: 0 for t in self.loader.tasks}
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)
        self.bins = torch.arange(model.cfg.num_bins, dtype=torch.float32, device=self.device)
        self.history: list[dict] = []

    # -- data ---------------------------------------------------------------------------------
    def generate_states(self, task: PuzzleTask, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward walks from the solved state: ``(P [B, N], depth [B])``."""
        cur = self.depth[task.name]
        depths = torch.randint(1, cur + 1, (batch,), device=self.device, generator=self.generator)
        P = task.tp.random_walk(batch, depths, self.generator)
        ub = depths.clone()
        mix = self.cfg.stabilizer_mix
        if mix > 0 and task.curriculum is not None and len(task.curriculum) > 1:
            k = max(1, int(batch * mix))
            cur_max = task.max_depth or self.cfg.max_depth
            progress = min(1.0, self.depth[task.name] / max(cur_max, 1))
            last = len(task.curriculum) - 2                       # deepest non-trivial subgroup
            stage = int(round(last * (1.0 - progress)))
            sub = task.curriculum.sample(stage, self.np_rng, k)
            P[:k] = task.tp.from_numpy(sub)
            ub[:k] = self.model.cfg.max_dist
        return P, ub

    # -- targets ------------------------------------------------------------------------------
    @torch.no_grad()
    def compute_targets(self, task: PuzzleTask, P: torch.Tensor, upper: Optional[torch.Tensor] = None):
        """Distributional Bellman backup -> ``(value [B], dist [B, D], best_action [B], legal [B, A], q [B, A])``."""
        tp = task.tp
        B, A, D = P.shape[0], tp.num_actions, self.bins.shape[0]
        children, legal = tp.expand(P)
        flat = children.reshape(B * A, tp.n)
        solved_c = tp.is_solved(flat)
        _, dist, _ = self.target.evaluate(task.graph, tp.colors(flat), None, task.chunk)
        delta0 = torch.zeros(D, device=self.device)
        delta0[0] = 1.0
        dist = torch.where(solved_c.unsqueeze(1), delta0.unsqueeze(0), dist)
        cost_int = tp.cost.round().long().clamp(min=1)
        idx = (torch.arange(D, device=self.device).unsqueeze(0) + cost_int.unsqueeze(1)).clamp(max=D - 1)   # [A, D]
        shifted = torch.zeros(B, A, D, device=self.device).scatter_add_(
            2, idx.unsqueeze(0).expand(B, A, D), dist.view(B, A, D))
        q = (shifted * self.bins).sum(dim=-1).masked_fill(~legal, float("inf"))
        best = q.argmin(dim=1)
        tdist = shifted[torch.arange(B, device=self.device), best]
        solved = tp.is_solved(P)
        lb = torch.ones(B, device=self.device)
        for pdb in task.pdbs:
            lb = torch.maximum(lb, pdb.lookup(P))
        lb = lb.ceil().long().clamp(max=D - 1)
        if upper is not None and self.cfg.use_upper_bound:
            ub = torch.maximum(upper.long().clamp(max=D - 1), lb)
        else:
            ub = torch.full_like(lb, D - 1)
        pidx = torch.maximum(torch.minimum(torch.arange(D, device=self.device).unsqueeze(0), ub.unsqueeze(1)),
                             lb.unsqueeze(1))
        tdist = torch.zeros(B, D, device=self.device).scatter_add_(1, pidx, tdist)
        tdist = torch.where(solved.unsqueeze(1), delta0.unsqueeze(0), tdist)
        tval = (tdist * self.bins).sum(dim=-1)
        return tval, tdist, best, legal, q

    # -- optimisation -----------------------------------------------------------------------
    def train_step(self, task: Optional[PuzzleTask] = None, states: Optional[torch.Tensor] = None,
                   upper: Optional[torch.Tensor] = None) -> dict:
        cfg = self.cfg
        task = task or self.loader.sample()
        if states is None:
            states, upper = self.generate_states(task, cfg.batch_size)
        tval, tdist, best, legal, _ = self.compute_targets(task, states, upper)
        self.model.train()
        out = self.model(task.graph, task.tp.colors(states), legal)
        logp = F.log_softmax(out.dist_logits, dim=-1)
        # KL(target || prediction): cross-entropy minus the (constant) target entropy, so 0 == converged
        ce = (tdist * (torch.log(tdist.clamp(min=1e-12)) - logp)).sum(dim=-1)
        hub = F.smooth_l1_loss(out.value, tval, reduction="none")
        pol = F.cross_entropy(out.policy_logits, best, reduction="none")
        w = torch.ones_like(ce)
        if cfg.depth_weight_power > 0 and upper is not None:
            w = upper.float().clamp(min=1.0).pow(-cfg.depth_weight_power)
            w = w / w.mean()
        loss = (w * (cfg.dist_coef * ce + cfg.expect_coef * hub + cfg.policy_coef * pol)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.step += 1
        self.task_steps[task.name] += 1
        if self.step % cfg.target_sync == 0:
            self.sync_target()
        cap = task.max_depth or cfg.max_depth
        if self.task_steps[task.name] % cfg.depth_grow_every == 0 and self.depth[task.name] < cap:
            self.depth[task.name] += 1
        with torch.no_grad():
            bellman = (out.value - tval).abs().mean()
            acc = (out.policy_logits.argmax(dim=1) == best).float().mean()
        metrics = dict(step=self.step, task=task.name, loss=loss.item(), dist_kl=ce.detach().mean().item(),
                       bellman=float(bellman), policy_ce=pol.detach().mean().item(), policy_acc=float(acc),
                       target_mean=float(tval.mean()), depth=self.depth[task.name], grad_norm=float(gnorm))
        self.telemetry.log(loss=metrics["loss"], bellman=metrics["bellman"], dist_kl=metrics["dist_kl"],
                           policy_acc=metrics["policy_acc"], scramble_depth=float(metrics["depth"]))
        self.telemetry.set(train_step=self.step, train_task=task.name)
        self.history.append(metrics)
        return metrics

    def sync_target(self) -> None:
        self.target.load_state_dict(self.model.state_dict())
        self.target.invalidate_cache()

    def fit(self, steps: int, log_every: int = 0, callback: Optional[Callable[[dict], Optional[bool]]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> list[dict]:
        t0 = time.perf_counter()
        out = []
        for _ in range(steps):
            if should_stop is not None and should_stop():
                break
            m = self.train_step()
            out.append(m)
            if log_every and m["step"] % log_every == 0:
                recent = [x["loss"] for x in out[-log_every:]]
                print(f"step {m['step']:5d} [{m['task']:>12s}] loss {np.mean(recent):7.4f} bellman {m['bellman']:6.3f} "
                      f"pol_acc {m['policy_acc']:.2f} depth {m['depth']:2d} ({time.perf_counter() - t0:5.1f}s)", flush=True)
            if callback is not None and callback(m) is False:
                break
        return out

    # -- evaluation ---------------------------------------------------------------------------
    @torch.no_grad()
    def greedy_solve(self, task: PuzzleTask, P: torch.Tensor, max_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step-lookahead greedy rollouts with the online network -> ``(solved [B], cost [B])``."""
        tp = task.tp
        self.model.eval()
        B, A = P.shape[0], tp.num_actions
        solved = tp.is_solved(P)
        cost = torch.zeros(B, device=self.device)
        prev = torch.full((B,), -1, dtype=torch.long, device=self.device)
        ar = torch.arange(A, device=self.device).unsqueeze(0)
        for _ in range(max_steps):
            if bool(solved.all()):
                break
            children, legal = tp.expand(P)
            flat = children.reshape(B * A, tp.n)
            v, _, _ = self.model.evaluate(task.graph, tp.colors(flat), None, task.chunk)
            v = torch.where(tp.is_solved(flat), torch.zeros_like(v), v).view(B, A)
            q = (v + tp.cost.unsqueeze(0)).masked_fill(~legal, float("inf"))
            back = (ar == tp.inverse[prev.clamp(min=0)].unsqueeze(1)) & (prev >= 0).unsqueeze(1)
            q = torch.where(back & (legal.sum(dim=1, keepdim=True) > 1), torch.full_like(q, float("inf")), q)
            a = q.argmin(dim=1)
            P = torch.where(solved.unsqueeze(1), P, tp.apply(P, a))
            cost = cost + torch.where(solved, torch.zeros_like(cost), tp.cost[a])
            prev = torch.where(solved, prev, a)
            solved = solved | tp.is_solved(P)
        return solved, cost

    @torch.no_grad()
    def evaluate(self, task: Optional[PuzzleTask] = None, depth: int = 6, episodes: int = 128,
                 max_steps: Optional[int] = None) -> dict:
        task = task or self.loader.tasks[0]
        depths = torch.full((episodes,), depth, dtype=torch.long, device=self.device)
        P = task.tp.random_walk(episodes, depths, self.generator)
        solved, cost = self.greedy_solve(task, P, max_steps or depth * 2 + 4)
        rate = float(solved.float().mean())
        mean_cost = float(cost[solved].mean()) if bool(solved.any()) else float("nan")
        self.telemetry.log(solve_rate=rate)
        return dict(task=task.name, depth=depth, solve_rate=rate, mean_cost=mean_cost)

    @torch.no_grad()
    def depth_correlation(self, task: PuzzleTask, max_depth: int = 12, samples: int = 512) -> float:
        """Pearson correlation between ``h(s)`` and the scramble depth -- a zero-shot transfer probe."""
        depths = torch.randint(1, max_depth + 1, (samples,), device=self.device, generator=self.generator)
        P = task.tp.random_walk(samples, depths, self.generator)
        self.model.eval()
        v, _, _ = self.model.evaluate(task.graph, task.tp.colors(P), None, task.chunk)
        x, y = v - v.mean(), depths.float() - depths.float().mean()
        return float((x * y).sum() / (x.norm() * y.norm() + 1e-9))

    # -- macro promotion ----------------------------------------------------------------------
    def promote_macros(self, task: PuzzleTask, top_k: int = 2, **miner_kwargs) -> list[MinedMacro]:
        """Mine commutators/conjugates and append the best as macro-actions (new relations + edges)."""
        miner = MacroMiner(task.puzzle, **miner_kwargs)
        mined = miner.mine(top_k=max(12, 4 * top_k))
        macros = miner.promote(mined, top_k)
        if macros:
            before = task.graph.num_edges
            task.refresh(task.puzzle.with_macros(macros))
            task.mined = mined
            self.model.invalidate_cache()
            self.target.invalidate_cache()
            self.telemetry.set(macros=[f"{m.name}: {mm.text}  [{mm.kind}, supp {mm.support}]"
                                       for m, mm in zip(macros, mined)])
            self.telemetry.event(f"promoted {len(macros)} macros on {task.name}: "
                                 f"{task.graph.num_edges - before} shortcut edges injected")
        return mined

    # -- persistence --------------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"config": self.model.cfg.to_dict(), "state_dict": self.model.state_dict(),
                    "extra": {"step": self.step, "train_config": self.cfg.to_dict(), "depth": self.depth,
                              "tasks": [t.name for t in self.loader.tasks]}}, path)


def build_trainer(puzzles: Sequence[Puzzle], device: Optional[torch.device | str] = None,
                  model_cfg: Optional[ModelConfig] = None, train_cfg: Optional[TrainConfig] = None,
                  telemetry: Optional[Telemetry] = None, model: Optional[RelationalGraphTransformer] = None,
                  pdbs: Optional[dict[str, Sequence[PatternDatabase]]] = None,
                  max_depths: Optional[dict[str, int]] = None, with_curriculum: bool = False) -> ADITrainer:
    """Convenience constructor: one shared model, one task per puzzle."""
    dev = pick_device(device) if not isinstance(device, torch.device) else device
    model_cfg = model_cfg or (model.cfg if model is not None else ModelConfig())
    train_cfg = train_cfg or TrainConfig()
    torch.manual_seed(train_cfg.seed)
    model = (model or RelationalGraphTransformer(model_cfg)).to(dev)
    tasks = []
    for pz in puzzles:
        cur = PuzzleGroup(pz).stabilizer_curriculum() if with_curriculum else None
        tasks.append(PuzzleTask(pz, dev, model_cfg, (pdbs or {}).get(pz.name, ()),
                                max_depth=(max_depths or {}).get(pz.name), curriculum=cur))
    return ADITrainer(model, tasks, train_cfg, telemetry)
