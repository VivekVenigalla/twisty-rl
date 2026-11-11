"""PAIRED-style asymmetric self-play: an adversarial Scrambler versus the Solver.

The Scrambler is a policy over the puzzle's own generators.  Starting from the solved state it
emits a move sequence (with a learned *stop* action) inside a horizon ``H``.  Because the
Scrambler knows a solution of length ``T`` (the inverse of its scramble) it plays the PAIRED
antagonist for free, and the Solver's regret is measured against it:

    regret = (cost of the Solver's greedy solution  -  T)^+      (full budget when unsolved)
    reward = bellman_coef * |h(s) - (T h)(s)|  +  regret_coef * regret

Maximising this reward steers the Scrambler towards states that are certifiably easy yet badly
modelled -- parity traps, orientation dead-ends and other topological bottlenecks -- and the
Solver is trained on exactly those states (mixed with uniform backward walks).  The horizon
expands whenever the Solver masters the current one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ModelConfig, RelationalGraphTransformer
from .train import ADITrainer, PuzzleTask
from .utils import Telemetry


@dataclass
class PAIREDConfig:
    batch_size: int = 64
    horizon_start: int = 3
    horizon_max: int = 20
    expand_threshold: float = 0.85
    expand_patience: int = 3
    scrambler_lr: float = 3e-4
    entropy_coef: float = 0.02
    bellman_coef: float = 1.0
    regret_coef: float = 1.0
    bellman_clip: float = 6.0
    adversarial_mix: float = 0.5
    slack: int = 2
    min_length: int = 1
    solver_steps: int = 1
    d_model: int = 64
    n_layers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class Scrambler(nn.Module):
    """Graph-transformer policy over generators plus a stop action; conditioned on (t/H, H/H_max)."""

    def __init__(self, base_cfg: ModelConfig, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        cfg = ModelConfig(d_model=d_model, n_heads=4, n_layers=n_layers, d_ff=2 * d_model, d_msg=d_model // 2,
                          lap_k=base_cfg.lap_k, rw_k=base_cfg.rw_k, max_dist=base_cfg.max_dist, context_dim=2)
        self.backbone = RelationalGraphTransformer(cfg)
        self.stop_head = nn.Linear(2 * d_model, 1)

    def forward(self, graph, colors: torch.Tensor, legal: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out = self.backbone(graph, colors, legal, context)
        return torch.cat([out.policy_logits, self.stop_head(out.pooled)], dim=-1)       # [B, A + 1]


class PAIREDTrainer:
    def __init__(self, solver: ADITrainer, task: Optional[PuzzleTask] = None, cfg: Optional[PAIREDConfig] = None,
                 telemetry: Optional[Telemetry] = None):
        self.solver = solver
        self.task = task or solver.loader.tasks[0]
        self.cfg = cfg or PAIREDConfig()
        self.device = solver.device
        self.telemetry = telemetry or solver.telemetry
        self.scrambler = Scrambler(solver.model.cfg, self.cfg.d_model, self.cfg.n_layers).to(self.device)
        self.opt = torch.optim.Adam(self.scrambler.parameters(), lr=self.cfg.scrambler_lr)
        self.horizon = self.cfg.horizon_start
        self.iteration = 0
        self._streak = 0
        self.reward_mean = 0.0
        self.hard_examples: list[dict] = []
        self.history: list[dict] = []

    # -- adversary rollout ---------------------------------------------------------------------
    def rollout(self, batch: int, greedy: bool = False):
        """Generate scrambles -> ``(P, lengths, log_prob, entropy, actions [B, H])``."""
        tp, graph, cfg = self.task.tp, self.task.graph, self.cfg
        A = tp.num_actions
        H = self.horizon
        P = tp.solved(batch)
        alive = torch.ones(batch, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(batch, dtype=torch.long, device=self.device)
        logp = torch.zeros(batch, device=self.device)
        ent = torch.zeros(batch, device=self.device)
        prev = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        actions = torch.full((batch, H), -1, dtype=torch.long, device=self.device)
        ar = torch.arange(A, device=self.device).unsqueeze(0)
        for t in range(H):
            ctx = torch.tensor([t / max(H, 1), H / cfg.horizon_max], device=self.device).expand(batch, 2)
            legal = tp.legal_mask(P) & tp.primitive_mask.unsqueeze(0)
            back = (ar == tp.inverse[prev.clamp(min=0)].unsqueeze(1)) & (prev >= 0).unsqueeze(1)
            legal = torch.where((legal & ~back).any(dim=1, keepdim=True), legal & ~back, legal)
            logits = self.scrambler(graph, tp.colors(P), legal, ctx)
            if t < cfg.min_length:
                logits = torch.cat([logits[:, :A], torch.full_like(logits[:, A:], -1e9)], dim=1)
            logprobs = F.log_softmax(logits, dim=-1)
            probs = logprobs.exp()
            a = probs.argmax(dim=1) if greedy else torch.multinomial(probs, 1).squeeze(1)
            step_lp = logprobs.gather(1, a.unsqueeze(1)).squeeze(1)
            step_ent = -(probs * logprobs.clamp(min=-30.0)).sum(dim=1)
            logp = logp + torch.where(alive, step_lp, torch.zeros_like(step_lp))
            ent = ent + torch.where(alive, step_ent, torch.zeros_like(step_ent))
            move = alive & (a < A)
            P = torch.where(move.unsqueeze(1), tp.apply(P, a.clamp(max=A - 1)), P)
            actions[:, t] = torch.where(move, a, actions[:, t])
            lengths = lengths + move.long()
            prev = torch.where(move, a, prev)
            alive = move
        return P, lengths, logp, ent, actions

    # -- protagonist evaluation ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, P: torch.Tensor, lengths: torch.Tensor) -> dict:
        task, cfg = self.task, self.cfg
        tval, _, _, _, _ = self.solver.compute_targets(task, P, lengths.clamp(min=1))
        self.solver.model.eval()
        pred, _, _ = self.solver.model.evaluate(task.graph, task.tp.colors(P), None, task.chunk)
        bellman = (pred - tval).abs()
        budget = self.horizon + cfg.slack
        solved, cost = self.solver.greedy_solve(task, P, budget)
        over = (cost - lengths.float()).clamp(min=0.0)
        regret = torch.where(solved, over, torch.full_like(over, float(budget))) / float(budget)
        reward = cfg.bellman_coef * bellman.clamp(max=cfg.bellman_clip) / cfg.bellman_clip + cfg.regret_coef * regret
        return dict(reward=reward, bellman=bellman, regret=regret, solved=solved, cost=cost)

    # -- one PAIRED iteration ----------------------------------------------------------------------
    def train_iteration(self) -> dict:
        cfg, task = self.cfg, self.task
        self.scrambler.train()
        P, lengths, logp, ent, actions = self.rollout(cfg.batch_size)
        sc = self.score(P, lengths)
        reward = sc["reward"]
        self.reward_mean = 0.9 * self.reward_mean + 0.1 * float(reward.mean()) if self.iteration else float(reward.mean())
        adv = (reward - self.reward_mean) / (reward.std() + 1e-6)
        loss = -(adv.detach() * logp).mean() - cfg.entropy_coef * ent.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.scrambler.parameters(), 1.0)
        self.opt.step()

        # Solver update on the adversarial states mixed with uniform backward walks.
        solver_metrics = {}
        for _ in range(cfg.solver_steps):
            n_adv = max(1, int(self.solver.cfg.batch_size * cfg.adversarial_mix))
            pick = torch.randint(0, P.shape[0], (n_adv,), device=self.device)
            walk_P, walk_ub = self.solver.generate_states(task, max(self.solver.cfg.batch_size - n_adv, 1))
            states = torch.cat([P[pick], walk_P])
            upper = torch.cat([lengths[pick].clamp(min=1), walk_ub])
            solver_metrics = self.solver.train_step(task, states, upper)

        solve_rate = float(sc["solved"].float().mean())
        self._streak = self._streak + 1 if solve_rate >= cfg.expand_threshold else 0
        expanded = False
        if self._streak >= cfg.expand_patience and self.horizon < cfg.horizon_max:
            self.horizon += 1
            self._streak = 0
            expanded = True
            self.telemetry.event(f"adversary horizon expanded to {self.horizon}")
        self.iteration += 1

        top = torch.topk(reward, min(4, reward.shape[0])).indices.tolist()
        self.hard_examples = [dict(
            scramble=" ".join(task.puzzle.names_of([a for a in actions[i].tolist() if a >= 0])),
            reward=float(reward[i]), regret=float(sc["regret"][i]), bellman=float(sc["bellman"][i]),
            solved=bool(sc["solved"][i])) for i in top]
        metrics = dict(iteration=self.iteration, horizon=self.horizon, adv_reward=float(reward.mean()),
                       adv_regret=float(sc["regret"].mean()), adv_bellman=float(sc["bellman"].mean()),
                       adv_solve_rate=solve_rate, adv_length=float(lengths.float().mean()),
                       scrambler_loss=loss.item(), scrambler_entropy=ent.detach().mean().item(), expanded=expanded,
                       solver_loss=solver_metrics.get("loss", float("nan")))
        self.telemetry.log(adv_reward=metrics["adv_reward"], adv_regret=metrics["adv_regret"],
                           adv_bellman=metrics["adv_bellman"], adv_solve_rate=solve_rate,
                           adv_horizon=float(self.horizon), adv_length=metrics["adv_length"])
        self.history.append(metrics)
        return metrics

    def fit(self, iterations: int, log_every: int = 0, should_stop=None) -> list[dict]:
        out = []
        for _ in range(iterations):
            if should_stop is not None and should_stop():
                break
            m = self.train_iteration()
            out.append(m)
            if log_every and m["iteration"] % log_every == 0:
                print(f"paired {m['iteration']:4d} H={m['horizon']:2d} reward {m['adv_reward']:.3f} "
                      f"regret {m['adv_regret']:.3f} bellman {m['adv_bellman']:.3f} "
                      f"solve {m['adv_solve_rate']:.2f} len {m['adv_length']:.1f} "
                      f"solver_loss {m['solver_loss']:.3f}", flush=True)
        return out
