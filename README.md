# TwistyRL

Topology-agnostic reinforcement learning, computational group theory and batched heuristic search
for arbitrary permutation puzzles, with a real-time terminal dashboard and an Emacs front end.

One declarative schema, one neural architecture and one search engine cover the 2x2/3x3/4x4 cubes,
Pyraminx, Skewb, Megaminx, the 15-puzzle, the pancake and star Cayley graphs, Hungarian Rings,
Lights Out and Rubik's Clock.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python main.py bench
```

```bash
.venv/bin/python main.py tui megaminx
```

Other entry points: `list`, `info <puzzle>`, `solve <puzzle> --scramble N`, `mine <puzzle>`,
`render <puzzle>`, `train --puzzles cube2,pyraminx --steps 600 --out checkpoints/m.pt`, `daemon`.

## Architecture

| Module | Role |
| --- | --- |
| `schema.py` | pydantic `PuzzleSpec` (facets, colour classes, generators, pieces, symmetries, conditional generators, `ModularSpec` for Z_n extensions) and the compiled `Puzzle` |
| `geometry.py`, `puzzles.py` | polyhedral puzzles derived from half-space geometry (stickers, moves, pieces, symmetries and nets are computed, not tabulated) plus the combinatorial puzzles |
| `group_theory.py` | Schreier-Sims BSGS, exact order, membership / solvability, colour stabiliser, groupoid vertex groups, stabiliser curriculum, commutator / conjugate macro miner |
| `env.py` | `TensorPuzzle` / `VecPuzzleEnv` (batched gather transitions on mps / cuda / cpu) and Gymnasium wrappers |
| `graph_compiler.py`, `model.py` | multi-relational graph with LapPE / RWPE / diffusion kernels; relational graph transformer with distributional value head and edge-pooled policy head |
| `train.py`, `adversarial.py` | distributional ADI with bound-projected Bellman targets, multi-task loader, macro promotion; PAIRED-style scrambler / solver self-play |
| `search.py` | Zobrist hashing with symmetry canonicalisation, tensor transposition table, stabiliser-chain pattern databases, batch weighted A*, batched beam search |
| `renderer_3d.py`, `tui.py` | z-buffered half-block rasteriser, universal net view, Textual dashboard |
| `session.py`, `emacs_bridge.py`, `emacs/twisty-dashboard.el` | shared session, JSON-RPC daemon on a Unix socket, Emacs corner widget and interactive mode |

### Conventions

A permutation state `P` has `P[i]` = the facet at position `i`; a move `m` acts as `P' = P[m]`
(gather form). The colour state is `solved_state[P]`; a puzzle is solved when its colour state
equals `solved_state`, so indistinguishable facets are handled throughout (search hashing, pattern
database goals, solvability).

### Things worth knowing

- **Sliding puzzles are groupoids.** Moves depend on the blank, so generators carry one
  permutation per anchor position. The group engine works on the vertex group obtained from
  Schreier loops over a spanning tree of blank positions (A_15 for the 15-puzzle).
- **Z_n puzzles** (Lights Out, Rubik's Clock) use the regular representation: a register mod `m`
  is `m` facets with one "hand" token, and each move is a product of `m`-cycles.
- **Equivariance.** No parameter is indexed by facet, colour or action. With LapPE masked the
  network is exactly invariant to facet relabelling and equivariant to action reordering (tested
  to 1e-6). LapPE itself is sign-invariant but not basis-invariant in degenerate eigenspaces.
- **Optimality.** `BatchedWeightedAStar(optimal=True)` with weight 1 and a pattern-database
  heuristic proves optimality; the tests compare it against exhaustive BFS.
- **Macros** cost the length of their word, so they shorten search depth without changing
  solution cost. They appear as new relations and edges in the compiled graph and need no new
  model parameters.

## Emacs

```elisp
(add-to-list 'load-path "/path/to/twisty-rl/emacs")
(require 'twisty-dashboard)
(twisty-dashboard-mode 1)   ; rotating puzzle in the corner of the startup buffer
```

`M-x twisty-dashboard` opens the interactive buffer: `U D L R F B` turn faces (lowercase for the
inverse), `m` enters any move string, `s` scrambles, `SPC` solves and plays the solution back,
`g` runs Schreier-Sims, `c` mines commutators, `t` / `a` start ADI / PAIRED training, arrows rotate.
The daemon is started on demand from the project's `.venv`.

## Limits

- Without a trained checkpoint, solving uses pattern databases only. That is optimal and fast for
  the small puzzles and shallow scrambles of the large ones, but it will not solve a fully
  scrambled 3x3, 4x4 or Megaminx; those need a trained model (`train`, then `solve --checkpoint`).
- `bench` and the tests demonstrate convergence at micro scale (minutes), not a fully trained
  solver for the large puzzles.
- Macro mining is exhaustive over short words; the Megaminx takes a few seconds.
