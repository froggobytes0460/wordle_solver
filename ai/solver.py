"""
Main solver for wordle. Uses max-entropy algorithm.
"""

import numpy as np
from numpy.typing import NDArray


class WordleSolver:
    """Wordle Solver AI. Uses max entropy"""

    words: list[str]
    state_lut: NDArray[np.uint8]
    candidates: NDArray[np.intp]

    def __init__(self, words: list[str], state_lut: NDArray[np.uint8]) -> None:
        self.words = words
        self.state_lut = state_lut
        self.candidates = np.arange(len(words), dtype=np.intp)

    def best_guess(self) -> int:
        """Picks the guess (row index into words/state_lut) with highest
        expected information (Shannon entropy) against current candidates.

        :return: Row index of the best guess.
        :rtype: int
        """

        n = self.state_lut.shape[0]
        n_candidates = len(self.candidates)

        if n_candidates <= 1:
            return int(self.candidates[0])

        sub = self.state_lut[:, self.candidates]

        offset = sub.astype(np.int64) + (243 * np.arange(n, dtype=np.int64))[:, None]
        counts = np.bincount(offset.ravel(), minlength=243 * n).reshape(n, 243)

        probs = counts / n_candidates
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
        entropy = -terms.sum(axis=1)

        return int(np.argmax(entropy))

    def update_state_lut(self, guess_idx: int, pattern: int) -> None:
        """Updates state LUT for the pattern after a guess has been made.

        :param guess_idx: The ID of guess word, present in candidates
        :type guess_idx: int
        :param pattern: The state pattern ["green", "yellow", "grey", "grey", "grey"]
        compressed as a base-3 integer
        :type pattern: int
        """

        mask = self.state_lut[guess_idx, self.candidates] == pattern
        self.candidates = self.candidates[mask]
