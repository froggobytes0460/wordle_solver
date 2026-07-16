# Wordle Solver

A max-entropy Wordle solver with an interactive full-screen TUI.

## How it works

For each guess, the solver picks the word maximizing expected Shannon
information (entropy) over the remaining candidate set, using a precomputed
NxN pattern matrix (`state_lut`) where cell `[i, j]` encodes the tile result
(gray/yellow/green, base-3 packed) of guessing word `i` against secret `j`.
After each guess the candidate set is filtered by the observed pattern.

Entropy is computed with each remaining candidate weighted by its real-world
word frequency (via [`wordfreq`](https://pypi.org/project/wordfreq/) Zipf
score), and a small log-frequency bonus (`_GUESS_WEIGHT_LAMBDA = 0.1`) nudges
ties toward common words — without overriding a genuine entropy gap.

**Complexity** ($N$ = word list size, $M$ = current candidate count):

- Matrix build (`word_crunch.build_state_lut`): $O(N^2)$ time and space — every
  word compared against every other word, once, cached to `.npy`.
- Best-guess selection (`WordleSolver.best_guess`): $O(N \cdot M)$ time, $O(N \cdot M)$
  space per guess — scores all $N$ words against the $M$ remaining candidates
  via vectorized numpy ops (`sub`, `offset`, `counts` all sized $N \times M$ or
  $N \times 243$).
- Candidate filtering (`WordleSolver.update_state_lut`): $O(M)$ time, $O(M)$
  space per guess.

## Setup

Requires Python >= 3.13. Install with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

## Usage

First, build the pattern matrix for a word list (one-time, cached to disk):

```sh
uv run wordle-solver-tools build-matrix word_list.txt state_lut.npy
```

Then play an interactive session:

```sh
uv run wordle-solver word_list.txt state_lut.npy
```

Controls: `←`/`→` move cursor, `space` cycle tile grey→yellow→green,
`enter` confirm row, `q` quit.

To auto-play against a known target word instead of entering results manually:

```sh
uv run wordle-solver word_list.txt state_lut.npy --target crane
```

## Project layout

- `main.py` — `wordle-solver` CLI entrypoint (interactive play session)
- `tools.py` — `wordle-solver-tools` CLI entrypoint (matrix building)
- `word_crunch.py` — builds the NxN pattern matrix
- `ai/solver.py` — `WordleSolver`, frequency-weighted max-entropy guess selection
- `tui/` — prompt_toolkit full-screen game loop, rendering, key bindings
- `word_list.txt` — newline-separated word list
- `state_lut.npy` — precomputed pattern matrix for `word_list.txt`
