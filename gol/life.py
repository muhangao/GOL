"""Conway B3/S23 trajectories with an explicit five-symbol interface."""
import numpy as np
from .config import LifeConfig

DEAD, ALIVE, FRAME, BOS, EOS = range(5)
VOCAB_SIZE = 5


def step(board: np.ndarray, boundary: str = "toroidal") -> np.ndarray:
    board = np.asarray(board, dtype=bool)
    if board.ndim != 2 or min(board.shape) < 3:
        raise ValueError("Expected a two-dimensional board with both dimensions >= 3")
    neighbors = np.zeros(board.shape, dtype=np.uint8)
    if boundary == "toroidal":
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    neighbors += np.roll(board, (dy, dx), axis=(0, 1))
    elif boundary == "dead":
        padded = np.pad(board, 1)
        h, w = board.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    neighbors += padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    else:
        raise ValueError(f"Unknown boundary: {boundary}")
    return (neighbors == 3) | (board & (neighbors == 2))


def trajectory(rng: np.random.Generator, config: LifeConfig) -> np.ndarray:
    density = rng.uniform(config.density_min, config.density_max)
    board = rng.random((config.board_size, config.board_size)) < density
    result = [BOS]
    for _ in range(config.frames):
        result.extend(board.ravel(order="C").astype(np.int64).tolist())
        result.append(FRAME)
        board = step(board, config.boundary)
    result.append(EOS)
    return np.asarray(result, dtype=np.int64)


def sample(index: int, seed: int, length: int, config: LifeConfig) -> np.ndarray:
    """Stateless sample: pack whole trajectories, then truncate to length.

    Initial frames and separator tokens are included in the next-token loss.
    Extinct boards are kept; no rejection sampling or novelty claim is hidden
    in the generator. Each training sequence gets a fresh deterministic RNG.
    """
    if min(index, seed) < 0 or length <= 0:
        raise ValueError("index/seed must be nonnegative and length positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    chunks, count = [], 0
    while count < length:
        chunk = trajectory(rng, config)
        chunks.append(chunk)
        count += len(chunk)
    return np.concatenate(chunks)[:length]
