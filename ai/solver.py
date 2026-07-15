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
    weights: NDArray[np.float32]

    # nudges best_guess() toward common guess words among near-equal-entropy
    # options, without letting commonness override a real entropy gap
    _GUESS_WEIGHT_LAMBDA = 0.1

    def __init__(
        self,
        words: list[str],
        state_lut: NDArray[np.uint8],
        weights: NDArray[np.float32] | None = None,
    ) -> None:
        self.words = words
        self.state_lut = state_lut
        self.candidates = np.arange(len(words), dtype=np.intp)
        self.weights = (
            weights if weights is not None else np.ones(len(words), dtype=np.float32)
        )

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

        # prior weight of each remaining candidate
        cand_weights = self.weights[self.candidates]
        total_weight = cand_weights.sum()

        # state_lut[g, c] = pattern (0-242) guess g would get if answer were candidate c
        sub = self.state_lut[:, self.candidates]

        # flatten (guess, pattern) into a single bincount bucket index per guess row
        offset = sub.astype(np.int64) + (243 * np.arange(n, dtype=np.int64))[:, None]

        # per guess: weight falling into each of the 243 possible patterns
        counts = np.bincount(
            offset.ravel(), weights=np.tile(cand_weights, n), minlength=243 * n
        ).reshape(n, 243)

        # P(pattern | guess), weighted by candidate prior instead of raw candidate count
        probs = counts / total_weight
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probs > 0, probs * np.log2(probs), 0.0)

        # Shannon entropy per guess: expected bits of info this guess yields
        entropy = -terms.sum(axis=1)

        # gentle bonus toward common guess words, breaks ties among
        # near-equal-entropy guesses without overriding real entropy gaps
        score = entropy + self._GUESS_WEIGHT_LAMBDA * np.log2(self.weights)

        return int(np.argmax(score))

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
