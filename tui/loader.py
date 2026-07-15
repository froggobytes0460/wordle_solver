"""Loading of word lists and precomputed pattern matrices from disk."""

from pathlib import Path

import numpy as np


def load_words_and_lut(word_list: Path, matrix: Path) -> tuple[list[str], np.ndarray]:
    """Loads the word list and precomputed pattern matrix from disk.

    :param word_list: Path to newline-separated word list file.
    :type word_list: Path
    :param matrix: Path to the .npy pattern matrix file.
    :type matrix: Path
    :return: Tuple of (words, state_lut).
    :rtype: tuple[list[str], np.ndarray]
    """

    words = [
        valid_w for w in word_list.read_text().splitlines() if (valid_w := w.strip())
    ]
    state_lut = np.load(file=matrix)
    return words, state_lut
