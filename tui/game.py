"""Game state and transition logic for the Wordle solver TUI."""

from ai.solver import WordleSolver
from tui.pattern import WORD_LEN, decode_pattern, encode_pattern

MAX_ATTEMPTS = 6


class WordleGame:
    """Drives a full-screen prompt_toolkit Wordle grid backed by WordleSolver."""

    words: list[str]
    solver: WordleSolver
    target_idx: int | None = None
    rows: list[tuple[str, list[int]]]
    cur_guess: str
    cur_states: list[int]
    cursor: int
    status: str
    finished: bool
    won: bool

    def __init__(
        self,
        words: list[str],
        solver: WordleSolver,
        target_idx: int | None = None,
    ) -> None:
        self.words = words
        self.solver = solver
        self.target_idx = target_idx
        self.rows = []
        self.cur_guess = ""
        self.cur_guess_idx = -1
        self.cur_states = [0] * WORD_LEN
        self.cursor = 0
        self.status = "Computing first guess…"
        self.finished = False
        self.won = False
        self.thinking = True

    @property
    def auto(self) -> bool:
        return self.target_idx is not None

    def apply_guess(self, guess_idx: int) -> None:
        """Applies a guess computed off the UI thread by `solver.best_guess()`."""
        self.thinking = False
        self.cur_guess = self.words[guess_idx]
        self.cur_guess_idx = guess_idx
        self.cursor = 0
        n = len(self.solver.candidates)

        if self.target_idx is not None:
            pattern = int(self.solver.state_lut[guess_idx, self.target_idx])
            self.cur_states = decode_pattern(pattern)
            self.status = f"{n:,} candidate(s) remain. Press enter to step."
        else:
            self.cur_states = [0] * WORD_LEN
            self.status = f"{n:,} candidate(s) remain."
            if n == 1:
                self.status += " This must be the answer!"

    def move_cursor(self, delta: int) -> None:
        self.cursor = max(0, min(WORD_LEN - 1, self.cursor + delta))

    def cycle_state(self) -> None:
        self.cur_states[self.cursor] = (self.cur_states[self.cursor] + 1) % 3

    def confirm_row(self) -> bool:
        """Records the current row. Returns True if the caller must now
        compute the next guess (via `solver.best_guess()` + `apply_guess`)."""
        if self.finished:
            return False
        self.rows.append((self.cur_guess, list(self.cur_states)))

        if self.cur_states == [2] * WORD_LEN:
            self.won = True
            self.finished = True
            self.status = f"Solved in {len(self.rows)} attempt(s)!"
            return False

        self.solver.update_state_lut(
            guess_idx=self.cur_guess_idx, pattern=encode_pattern(self.cur_states)
        )

        if len(self.rows) >= MAX_ATTEMPTS:
            self.finished = True
            n = len(self.solver.candidates)
            self.status = f"Out of attempts. {n:,} candidate(s) still possible."
            return False

        if len(self.solver.candidates) == 0:
            self.status = "No candidates remain — check entered patterns."
            self.finished = True
            return False

        self.thinking = True
        self.status = "Computing next guess…"
        return True
