#!/usr/bin/env python3
"""TwistyRL command line.

    python main.py list
    python main.py info megaminx
    python main.py render cube3 --scramble 12
    python main.py solve pyraminx --scramble 9
    python main.py mine cube3
    python main.py train --puzzles cube2,pyraminx --steps 600 --out checkpoints/multi.pt
    python main.py bench                       # MPS verification + micro-training convergence proof
    python main.py tui megaminx
    python main.py daemon                      # JSON-RPC server for emacs/twisty-dashboard.el
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np


def _device(args):
    from twisty_rl.utils import pick_device
    return pick_device(getattr(args, "device", None))


def cmd_list(args) -> int:
    from twisty_rl.puzzles import get_puzzle, list_puzzles
    print(f"{'puzzle':18s} {'family':11s} {'facets':>6s} {'actions':>7s} {'pieces':>6s}  description")
    for name in list_puzzles():
        pz = get_puzzle(name)
        print(f"{name:18s} {pz.spec.family:11s} {pz.n:6d} {pz.num_actions:7d} {pz.num_pieces:6d}  {pz.spec.description}")
    return 0


def cmd_info(args) -> int:
    from twisty_rl.group_theory import PuzzleGroup
    from twisty_rl.puzzles import get_puzzle
    from twisty_rl.utils import human_int
    pz = get_puzzle(args.puzzle)
    t0 = time.perf_counter()
    grp = PuzzleGroup(pz)
    dt = time.perf_counter() - t0
    b = grp.bsgs
    print(f"{pz.name}: {pz.spec.description}")
    print(f"  facets N = {pz.n}, colours = {pz.num_colors}, actions = {pz.num_actions} "
          f"({', '.join(pz.action_names)})")
    print(f"  Schreier-Sims: {dt:.3f}s, base length {len(b.base)}, {len(b.strong_generators)} strong generators, "
          f"{b.schreier_generators_tested:,} Schreier generators sifted, verified={b.verified}")
    print(f"  base         = {list(map(int, b.base))}")
    print(f"  basic orbits = {b.orbit_sizes}")
    print(f"  |G|          = {grp.order:,}")
    if not pz.is_group:
        print(f"  groupoid: {len(grp.anchor_paths)} anchor positions, reachable states = {grp.num_states:,}")
    if pz.spec.known_order is not None:
        print(f"  literature   = {pz.spec.known_order:,}  ->  {'MATCH' if pz.spec.known_order == grp.num_states else 'MISMATCH'}")
    try:
        _, h = grp.color_stabilizer()
        print(f"  colour stabiliser order = {h:,};  visually distinct states = {human_int(grp.num_color_states)}")
    except Exception as exc:                     # budgeted backtrack search
        print(f"  colour stabiliser: not computed ({exc})")
    print(f"  whole-puzzle symmetries preserving the generators: {len(pz.valid_symmetries())}")
    print("  stabiliser curriculum  G = G_0 > G_1 > ... > {e}:")
    for st in grp.stabilizer_curriculum(args.stages).describe():
        print(f"    G_{st['stage']}: fixes {st['fixed_base_points']:2d} base points, |G_j| = {human_int(st['order'])}, "
              f"index in previous = {human_int(st['index'])}")
    return 0


def cmd_render(args) -> int:
    from twisty_rl.puzzles import get_puzzle
    from twisty_rl.renderer_3d import get_renderer, render_solid
    if args.solid:
        frames = max(args.spin, 1)
        for i in range(frames):
            frame = render_solid(args.solid, args.width, args.height, yaw=0.6 + 0.12 * i, pitch=0.45)
            sys.stdout.write(("\x1b[H\x1b[2J" if frames > 1 else "") + frame.to_ansi() + "\n")
            sys.stdout.flush()
            if frames > 1:
                time.sleep(1 / 30)
        return 0
    pz = get_puzzle(args.puzzle)
    rng = np.random.default_rng(args.seed)
    P = pz.identity
    if args.moves:
        P = pz.apply_word(P, pz.parse_moves(args.moves))
    elif args.scramble:
        P, scr = pz.scramble(rng, args.scramble)
        print("scramble:", " ".join(pz.names_of(scr)))
    r = get_renderer(pz)
    if args.spin:
        for i in range(args.spin):
            frame = r.render_3d(P, args.width, args.height, yaw=0.6 + 0.08 * i, pitch=0.45)
            sys.stdout.write("\x1b[H\x1b[2J" + frame.to_ansi() + "\n")
            sys.stdout.flush()
            time.sleep(1 / 30)
        return 0
    if args.view in ("3d", "both"):
        print(r.render_3d(P, args.width, args.height, yaw=args.yaw, pitch=args.pitch).to_ansi())
    if args.view in ("net", "both"):
        print(r.render_net(P, max(args.width, 60), args.height).to_ansi())
    return 0


def cmd_solve(args) -> int:
    from twisty_rl.group_theory import MacroMiner, PuzzleGroup
    from twisty_rl.model import load_checkpoint
    from twisty_rl.puzzles import get_puzzle
    from twisty_rl.search import build_pdbs, make_solver
    dev = _device(args)
    pz = get_puzzle(args.puzzle)
    rng = np.random.default_rng(args.seed)
    if args.moves:
        P = pz.apply_word(pz.identity, pz.parse_moves(args.moves))
        print("state :", args.moves)
    else:
        P, scr = pz.scramble(rng, args.scramble)
        print("scramble:", " ".join(pz.names_of(scr)))
    grp = PuzzleGroup(pz)
    print(f"|G| = {grp.num_states:,}; state in group: {grp.contains_state(P)}")
    pdbs = build_pdbs(pz, max_entries=args.pdb_entries, count=2, device=dev, group=grp)
    print("pattern databases:", [(len(p.tracked), p.size, p.max_depth) for p in pdbs])
    model = None
    if args.checkpoint:
        model, extra = load_checkpoint(args.checkpoint, dev)
        print(f"loaded {args.checkpoint} (trained on {extra.get('tasks')}, step {extra.get('step')})")
    search_pz = pz
    if args.macros:
        miner = MacroMiner(pz)
        macros = miner.promote(miner.mine(top_k=4 * args.macros), args.macros)
        search_pz = pz.with_macros(macros)
        print(f"macro-actions: {[f'{m.name}={chr(32).join(pz.names_of(m.word))}' for m in macros]}")
    weight = args.weight if args.weight is not None else (1.4 if model is not None else 1.0)
    kwargs = {}
    if args.method == "astar":
        kwargs = dict(optimal=model is None and weight <= 1.0, use_symmetry=args.symmetry)
    solver = make_solver(search_pz, dev, model=model, method=args.method, pdbs=pdbs, weight=weight,
                         batch_size=args.batch_size, beam_width=args.beam_width, **kwargs)
    res = solver.solve(P, time_limit=args.time_limit)
    if res.solved:
        ok = pz.is_solved(pz.apply_word(P, res.primitive_path))
        print(f"solution ({len(res.moves)} moves{', optimal' if res.optimal_proven else ''}, verified={ok}): "
              + " ".join(res.moves))
    else:
        print("no solution:", res.reason or "budget exhausted")
    print(f"expanded {res.nodes_expanded:,}, generated {res.nodes_generated:,} nodes in {res.elapsed:.2f}s "
          f"({res.nodes_per_sec:,.0f} nodes/s), transposition hits {res.table_hits:,}, device {dev}")
    return 0 if res.solved else 1


def cmd_mine(args) -> int:
    from twisty_rl.group_theory import MacroMiner
    from twisty_rl.puzzles import get_puzzle
    pz = get_puzzle(args.puzzle)
    miner = MacroMiner(pz, max_len_a=args.len_a, max_len_b=args.len_b)
    t0 = time.perf_counter()
    mined = miner.mine(top_k=args.top_k)
    print(f"{pz.name}: examined {miner.candidates_examined:,} sequences in {time.perf_counter() - t0:.2f}s")
    print(f"{'supp':>4s} {'pieces':>6s} {'order':>5s} {'origin':10s} {'type':22s} sequence")
    for m in mined:
        print(f"{m.support:4d} {m.pieces_affected:6d} {m.order:5d} {m.origin:10s} {m.kind:22s} {m.text}")
    return 0


def cmd_train(args) -> int:
    import torch
    from twisty_rl.adversarial import PAIREDConfig, PAIREDTrainer
    from twisty_rl.model import ModelConfig
    from twisty_rl.puzzles import get_puzzle
    from twisty_rl.train import TrainConfig, build_trainer
    dev = _device(args)
    puzzles = [get_puzzle(n) for n in args.puzzles.split(",")]
    mcfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, d_ff=2 * args.d_model, d_msg=args.d_model // 2)
    tcfg = TrainConfig(batch_size=args.batch_size, max_depth=args.max_depth, lr=args.lr, seed=args.seed,
                       stabilizer_mix=args.stabilizer_mix)
    trainer = build_trainer(puzzles, dev, mcfg, tcfg, with_curriculum=args.stabilizer_mix > 0)
    print(f"device {dev}; model {trainer.model.num_parameters():,} parameters; tasks {[p.name for p in puzzles]}")
    trainer.fit(args.steps, log_every=args.log_every)
    if args.macros:
        for task in trainer.loader.tasks:
            mined = trainer.promote_macros(task, args.macros)
            print(f"[{task.name}] promoted macros:", [m.text for m in mined[: args.macros]])
        trainer.fit(max(args.steps // 4, 1), log_every=args.log_every)
    if args.adversarial:
        for task in trainer.loader.tasks:
            paired = PAIREDTrainer(trainer, task, PAIREDConfig())
            paired.fit(args.adversarial, log_every=max(args.adversarial // 10, 1))
            print(f"[{task.name}] hardest adversarial scrambles:", json.dumps(paired.hard_examples[:2]))
    for task in trainer.loader.tasks:
        print(trainer.evaluate(task, depth=min(6, args.max_depth)))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        trainer.save(args.out)
        print("saved", args.out)
    return 0


def micro_train(name: str, device, steps: int = 400, seed: int = 0, verbose: bool = True) -> dict:
    """Train on one puzzle and measure the heuristic against *exact* optimal distances."""
    import torch
    from twisty_rl.model import ModelConfig
    from twisty_rl.puzzles import get_puzzle
    from twisty_rl.search import BatchedWeightedAStar, PDBHeuristic, build_pdbs, make_solver
    from twisty_rl.env import TensorPuzzle
    from twisty_rl.train import TrainConfig, build_trainer
    pz = get_puzzle(name)
    pdbs = build_pdbs(pz, max_entries=300_000, count=2, device=device)
    trainer = build_trainer([pz], device, ModelConfig(d_model=96, n_layers=3, d_ff=192, d_msg=48),
                            TrainConfig(batch_size=128, max_depth=12, start_depth=2, depth_grow_every=25,
                                        target_sync=20, lr=2e-3, seed=seed), pdbs={pz.name: pdbs})
    task = trainer.loader.tasks[0]
    tp = task.tp
    # fixed validation set with ground-truth optimal distances (admissible PDB A*, optimality proven)
    rng = np.random.default_rng(seed + 1)
    val = np.stack([pz.scramble(rng, int(d))[0] for d in rng.integers(1, 11, size=64)])
    exact_solver = BatchedWeightedAStar(tp, PDBHeuristic(tp, pdbs), weight=1.0, batch_size=1024, optimal=True)
    truth = []
    for P in val:
        r = exact_solver.solve(P, max_iters=20_000)
        assert r.solved and r.optimal_proven
        truth.append(r.cost)
    truth_t = torch.tensor(truth, device=device)
    val_t = tp.from_numpy(val)

    def probe() -> tuple[float, float]:
        trainer.model.eval()
        with torch.no_grad():
            v, _, _ = trainer.model.evaluate(task.graph, tp.colors(val_t), None, task.chunk)
        x, y = v - v.mean(), truth_t - truth_t.mean()
        return float((v - truth_t).abs().mean()), float((x * y).sum() / (x.norm() * y.norm() + 1e-9))

    mae0, corr0 = probe()
    t0 = time.perf_counter()
    hist = trainer.fit(steps, log_every=max(steps // 8, 1) if verbose else 0)
    train_time = time.perf_counter() - t0
    mae1, corr1 = probe()
    k = max(steps // 10, 1)
    first, last = float(np.mean([m["loss"] for m in hist[:k]])), float(np.mean([m["loss"] for m in hist[-k:]]))
    greedy = {d: trainer.evaluate(task, depth=d, episodes=128)["solve_rate"] for d in (3, 5, 7)}
    # neural-guided weighted A* on fresh scrambles
    solver = make_solver(pz, device, model=trainer.model, graph=task.graph, method="astar", weight=1.2,
                         batch_size=256, tp=tp)
    solved, lengths, nodes = 0, [], 0
    for i in range(16):
        P, _ = pz.scramble(rng, 10)
        r = solver.solve(P, max_iters=300, time_limit=10)
        solved += int(r.solved and pz.is_solved(pz.apply_word(P, r.primitive_path)))
        nodes += r.nodes_generated
        if r.solved:
            lengths.append(len(r.primitive_path))
    out = dict(puzzle=name, device=str(device), steps=steps, train_seconds=train_time,
               loss_first=first, loss_last=last, value_mae_before=mae0, value_mae_after=mae1,
               corr_before=corr0, corr_after=corr1, greedy_solve_rate=greedy,
               astar_solved=f"{solved}/16", astar_mean_len=float(np.mean(lengths)) if lengths else float("nan"),
               astar_nodes=nodes)
    if verbose:
        print(f"[{name}] loss {first:.3f} -> {last:.3f} | value MAE vs optimal {mae0:.2f} -> {mae1:.2f} | "
              f"corr {corr0:+.2f} -> {corr1:+.2f} | greedy solve {greedy} | neural A* {solved}/16 "
              f"(mean len {out['astar_mean_len']:.1f}) | {train_time:.0f}s on {device}")
    return out


def verify_mps(verbose: bool = True) -> dict:
    """Run every tensor path on MPS with warnings promoted to errors (no CPU fallback allowed)."""
    import torch
    from twisty_rl.adversarial import PAIREDConfig, PAIREDTrainer
    from twisty_rl.env import TensorPuzzle
    from twisty_rl.graph_compiler import compile_graph
    from twisty_rl.model import ModelConfig, RelationalGraphTransformer
    from twisty_rl.puzzles import get_puzzle, list_puzzles
    from twisty_rl.search import BatchedBeamSearch, BatchedWeightedAStar, PDBHeuristic, build_pdbs
    from twisty_rl.train import TrainConfig, build_trainer
    from twisty_rl.utils import sync_device
    if not torch.backends.mps.is_available():
        return {"available": False}
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK is set; unset it to verify native MPS execution")
    dev = torch.device("mps")
    report = {"available": True, "fallback_env": False, "puzzles": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = RelationalGraphTransformer(ModelConfig(d_model=64, n_layers=2, d_ff=128, d_msg=32)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3)
        for name in list_puzzles():
            pz = get_puzzle(name)
            tp = TensorPuzzle(pz, dev)
            g = compile_graph(pz, device=dev)
            P = tp.random_walk(256, torch.randint(1, 15, (256,), device=dev))
            children, legal = tp.expand(P)
            out = model(g, tp.colors(P), legal)
            loss = out.value.mean() + out.dist_logits.logsumexp(-1).mean() + out.policy_logits.logsumexp(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sync_device(dev)
            t0 = time.perf_counter()
            for _ in range(20):
                P = tp.apply(P, torch.randint(0, tp.num_actions, (P.shape[0],), device=dev))
                children, legal = tp.expand(P)
            sync_device(dev)
            rate = 20 * 256 * tp.num_actions / (time.perf_counter() - t0)
            report["puzzles"][name] = dict(device=str(out.value.device), transitions_per_sec=rate)
            assert out.value.device.type == "mps" and children.device.type == "mps"
        pz = get_puzzle("pyraminx")
        tp = TensorPuzzle(pz, dev)
        pdbs = build_pdbs(pz, max_entries=100_000, count=1, device=dev)
        P, _ = pz.scramble(np.random.default_rng(0), 8)
        assert BatchedWeightedAStar(tp, PDBHeuristic(tp, pdbs), optimal=True, use_symmetry=True).solve(P).solved
        assert BatchedBeamSearch(tp, PDBHeuristic(tp, pdbs), beam_width=256).solve(P).solved
        trainer = build_trainer([get_puzzle("cube2")], dev, ModelConfig(d_model=64, n_layers=2, d_ff=128, d_msg=32),
                                TrainConfig(batch_size=32, max_depth=4))
        trainer.fit(3)
        PAIREDTrainer(trainer, cfg=PAIREDConfig(batch_size=16)).fit(2)
        report["train_and_search"] = "ok"
    if verbose:
        slow = min(report["puzzles"].items(), key=lambda kv: kv[1]["transitions_per_sec"])
        fast = max(report["puzzles"].items(), key=lambda kv: kv[1]["transitions_per_sec"])
        print(f"MPS verification passed for {len(report['puzzles'])} puzzles with warnings-as-errors and no fallback: "
              f"env+model+search+ADI+PAIRED all native. Throughput {slow[0]} {slow[1]['transitions_per_sec']:,.0f}/s "
              f".. {fast[0]} {fast[1]['transitions_per_sec']:,.0f}/s child states")
    return report


def cmd_bench(args) -> int:
    import torch
    dev = _device(args)
    print(f"torch {torch.__version__}, device {dev}")
    if dev.type == "mps":
        verify_mps()
    results = [micro_train(name, dev, steps=args.steps, seed=args.seed) for name in args.puzzles.split(",")]
    ok = all(r["loss_last"] < r["loss_first"] and r["value_mae_after"] < r["value_mae_before"] for r in results)
    print("convergence:", "PASS" if ok else "FAIL")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0 if ok else 1


def cmd_tui(args) -> int:
    from twisty_rl.tui import run_tui
    run_tui(args.puzzle, args.device, args.checkpoint)
    return 0


def cmd_daemon(args) -> int:
    from twisty_rl.emacs_bridge import TwistyDaemon
    TwistyDaemon(socket_path=args.socket, puzzle=args.puzzle, device=args.device).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="twisty", description="TwistyRL: RL + group theory + search for permutation puzzles")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list built-in puzzles").set_defaults(fn=cmd_list)

    s = sub.add_parser("info", help="Schreier-Sims analysis of a puzzle")
    s.add_argument("puzzle"); s.add_argument("--stages", type=int, default=4); s.set_defaults(fn=cmd_info)

    s = sub.add_parser("render", help="print a 3D / net rendering")
    s.add_argument("puzzle", nargs="?", default="cube3")
    s.add_argument("--view", choices=["3d", "net", "both"], default="both")
    s.add_argument("--scramble", type=int, default=0); s.add_argument("--moves", default="")
    s.add_argument("--width", type=int, default=56); s.add_argument("--height", type=int, default=26)
    s.add_argument("--yaw", type=float, default=0.6); s.add_argument("--pitch", type=float, default=0.45)
    s.add_argument("--spin", type=int, default=0, help="animate this many frames")
    s.add_argument("--solid", choices=["cube", "tetrahedron", "dodecahedron"], default=None)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_render)

    s = sub.add_parser("solve", help="solve a scramble with batched search")
    s.add_argument("puzzle"); s.add_argument("--scramble", type=int, default=8); s.add_argument("--moves", default="")
    s.add_argument("--method", choices=["astar", "beam"], default="astar")
    s.add_argument("--checkpoint", default=None); s.add_argument("--weight", type=float, default=None)
    s.add_argument("--batch-size", type=int, default=1024); s.add_argument("--beam-width", type=int, default=4096)
    s.add_argument("--pdb-entries", type=int, default=400_000); s.add_argument("--time-limit", type=float, default=60.0)
    s.add_argument("--macros", type=int, default=0); s.add_argument("--symmetry", action="store_true")
    s.add_argument("--device", default=None); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_solve)

    s = sub.add_parser("mine", help="mine commutator / conjugate macros")
    s.add_argument("puzzle"); s.add_argument("--top-k", type=int, default=12)
    s.add_argument("--len-a", type=int, default=3); s.add_argument("--len-b", type=int, default=2); s.set_defaults(fn=cmd_mine)

    s = sub.add_parser("train", help="ADI (+ optional PAIRED) training, single or multi-topology")
    s.add_argument("--puzzles", default="cube2"); s.add_argument("--steps", type=int, default=500)
    s.add_argument("--batch-size", type=int, default=128); s.add_argument("--max-depth", type=int, default=20)
    s.add_argument("--lr", type=float, default=1e-3); s.add_argument("--d-model", type=int, default=128)
    s.add_argument("--layers", type=int, default=4); s.add_argument("--adversarial", type=int, default=0)
    s.add_argument("--macros", type=int, default=0); s.add_argument("--stabilizer-mix", type=float, default=0.0)
    s.add_argument("--log-every", type=int, default=50); s.add_argument("--out", default=None)
    s.add_argument("--device", default=None); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_train)

    s = sub.add_parser("bench", help="MPS verification + micro-training convergence check")
    s.add_argument("--puzzles", default="cube2,pyraminx"); s.add_argument("--steps", type=int, default=400)
    s.add_argument("--device", default=None); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", default=None); s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("tui", help="interactive Textual dashboard")
    s.add_argument("puzzle", nargs="?", default="cube3"); s.add_argument("--device", default=None)
    s.add_argument("--checkpoint", default=None); s.set_defaults(fn=cmd_tui)

    s = sub.add_parser("daemon", help="JSON-RPC daemon on a Unix socket (Emacs bridge)")
    s.add_argument("--socket", default=None); s.add_argument("--puzzle", default="cube3")
    s.add_argument("--device", default=None); s.set_defaults(fn=cmd_daemon)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
