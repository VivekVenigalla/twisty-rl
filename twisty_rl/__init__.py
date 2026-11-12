"""TwistyRL: topology-agnostic RL, computational group theory and search for permutation puzzles."""
from .puzzles import get_puzzle, get_spec, list_puzzles
from .schema import GeneratorSpec, Macro, ModularSpec, PieceSpec, Puzzle, PuzzleSpec

__version__ = "0.1.0"
__all__ = ["get_puzzle", "get_spec", "list_puzzles", "GeneratorSpec", "Macro", "ModularSpec", "PieceSpec",
           "Puzzle", "PuzzleSpec", "__version__"]
