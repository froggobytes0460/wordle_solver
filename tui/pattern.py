"""Base-3 pattern encoding for Wordle tile states."""

WORD_LEN = 5

# Tile states cycle grey(0) -> yellow(1) -> green(2) via [space].
STATE_BG = {0: "#3a3a3c", 1: "#b59f3b", 2: "#538d4e"}


def encode_pattern(states: list[int]) -> int:
    """Encodes tile states as base-3 int (ones digit = position 0)."""

    return sum(s * (3**i) for i, s in enumerate(states))


def decode_pattern(pattern: int) -> list[int]:
    """Decodes a base-3 pattern int into 5 tile states (ones digit = position 0)."""

    states: list[int] = []
    for _ in range(WORD_LEN):
        states.append(pattern % 3)
        pattern //= 3
    return states
