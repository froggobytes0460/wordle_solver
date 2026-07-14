import time

import numpy as np
from numpy.typing import NDArray


def build_pattern_matrix(guesses: list[str]) -> tuple[NDArray[np.uint8], float]:
    """Build an NxN pattern matrix for a Wordle bot.

    Each cell [i, j] encodes the Wordle result of guessing word i when the
    secret is word j, compressed to a uint8 via base-3: tile values 0=gray,
    1=yellow, 2=green mapped to digits of a 5-digit base-3 number (ones
    digit = position 0).

    Args:
        guesses: All valid 5-letter words (used as both guesses and answers).

    Returns:
        Array of shape (N, N), values in [0, 242], and elapsed time as float.
    """

    n = len(guesses)
    g: NDArray[np.uint8] = np.array(
        [[ord(c) for c in w] for w in guesses], dtype=np.uint8
    )
    powers: NDArray[np.uint8] = np.array([3**i for i in range(5)], dtype=np.uint8)
    matrix: NDArray[np.uint8] = np.zeros((n, n), dtype=np.uint8)

    t0 = time.time()

    for i in range(n):
        guess: NDArray[np.uint8] = g[i]
        tiles: NDArray[np.uint8] = np.zeros((n, 5), dtype=np.uint8)
        remaining: NDArray[np.uint8] = g.copy()

        green: NDArray[np.bool_] = remaining == guess
        tiles[green] = 2
        remaining[green] = 0

        for pos in range(5):
            not_green: NDArray[np.bool_] = tiles[:, pos] != 2
            for match_pos in range(5):
                yellow: NDArray[np.bool_] = not_green & (
                    remaining[:, match_pos] == guess[pos]
                )
                tiles[yellow, pos] = 1
                remaining[yellow, match_pos] = 0
                not_green = not_green & ~yellow

        np.dot(tiles, powers, out=matrix[i])

    return matrix, time.time() - t0
