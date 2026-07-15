"""Loading of word lists and precomputed pattern matrices from disk."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from wordfreq import zipf_frequency

# zipf_frequency returns 0.0 for words absent from the corpus; floor keeps
# them guessable instead of zeroing out total_weight in best_guess().
_MIN_ZIPF_WEIGHT = 1.0


def load_words_and_lut_and_weights(
    word_list: Path, matrix: Path
) -> tuple[list[str], NDArray, NDArray[np.float32]]:
    """Loads the word list, precomputed pattern matrix, and answer-likelihood
    weights (Zipf word frequency) from disk.

    :param word_list: Path to newline-separated word list file.
    :type word_list: Path
    :param matrix: Path to the .npy pattern matrix file.
    :type matrix: Path
    :return: Tuple of (words, state_lut, weights).
    :rtype: tuple[list[str], NDArray, NDArray[np.float32]]
    """

    words = [
        valid_w for w in word_list.read_text().splitlines() if (valid_w := w.strip())
    ]
    state_lut = np.load(file=matrix)
    weights = np.array(
        [max(zipf_frequency(w, "en"), _MIN_ZIPF_WEIGHT) for w in words],
        dtype=np.float32,
    )
    return words, state_lut, weights
