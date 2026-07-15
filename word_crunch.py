import time
from collections.abc import Iterator
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class LetterState(IntEnum):
    YELLOW = 1
    GREEN = 2


def build_state_lut(
    guesses: list[str],
) -> Iterator[tuple[int, NDArray[np.uint8], float]]:
    """Build an NxN pattern matrix for a Wordle bot, yielding progress per row.

    Each cell [i, j] encodes the Wordle result of guessing word i when the
    secret is word j, compressed to a uint8 via base-3: state values 0=gray,
    1=yellow, 2=green mapped to digits of a 5-digit base-3 number (ones
    digit = position 0).

    :param guesses: All valid 5-letter words (used as both guesses and answers).
    :type guesses: list[str]
    :yields: Tuple of (rows completed, matrix-so-far, elapsed time). The
        matrix is the same array on every yield, filled in-place; only the
        final yield (rows completed == n) is fully populated.
    :rtype: Iterator[tuple[int, NDArray[np.uint8], float]]
    """

    # Compress word-string to 2D matrix.
    n = len(guesses)
    g: NDArray[np.uint8] = np.array(
        [[ord(c) for c in w] for w in guesses], dtype=np.uint8
    )

    # Initialize 2D zero-matrix.
    matrix: NDArray[np.uint8] = np.zeros((n, n), dtype=np.uint8)

    t0 = time.perf_counter()

    for i in range(n):
        guess: NDArray[np.uint8] = g[i]

        # Tile scores for all N secrets against this guess: shape (N, 5).
        state: NDArray[np.uint8] = np.zeros((n, 5), dtype=np.uint8)

        # Copy of secret matrix; used to avoid double-counting.
        remaining: NDArray[np.uint8] = g.copy()

        # Mark exact position matches (green) across all N secrets simultaneously.
        green: NDArray[np.bool_] = remaining == guess
        state[green] = LetterState.GREEN

        # Zero out matched letters so they can't also score yellow.
        remaining[green] = 0

        for pos in range(5):
            # Only non-green positions at this column are candidates for yellow.
            not_green: NDArray[np.bool_] = state[:, pos] != LetterState.GREEN
            for match_pos in range(5):
                # Yellow: guess letter exists somewhere in secret but not at this pos.
                yellow: NDArray[np.bool_] = not_green & (
                    remaining[:, match_pos] == guess[pos]
                )
                state[yellow, pos] = LetterState.YELLOW

                # Consume matched letter to prevent it scoring yellow again.
                remaining[yellow, match_pos] = 0
                not_green = not_green & ~yellow

        # Encode 5 tile values as single base-3 uint8 via dot product.
        np.dot(state, np.array([3**i for i in range(5)], dtype=np.uint8), out=matrix[i])

        yield i + 1, matrix, time.perf_counter() - t0
