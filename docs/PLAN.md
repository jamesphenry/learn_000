# The plan

The plan of record for the work ahead, written on 2026-10-21 after the
review of the project and of [notes.md](notes.md). What has been measured so
far is in [RESULTS.md](RESULTS.md); what it means is in
[FINDINGS.md](FINDINGS.md) and [LEAGUE.md](LEAGUE.md).

## The decisions

- **Diagnosis first.** The Nim model is investigated before anything about it
  is changed.
- **Small Connect Four second.** A 5x5 board with four in a row, exactly
  solvable, so the perfect player stays perfect and every measurement keeps
  its meaning.
- **No architectural upgrade** until the measurements say which limitation is
  actually being faced.
- **`src/main.c` stays the running experiment book.** Every experiment is
  driven from the demonstration, so that what is measured is what anyone can
  run.

## Phase 1 - the Nim diagnosis

Nothing about the learning method or the architecture changes in this phase.
What changes is the instrument, because the current one cannot answer the
questions.

### The instrument

Nim's reachable positions from (3, 4, 5) are 120 of them: heaps 0-3, 0-4 and
0-5. The heaps belong to no player, so those 120 positions are the whole
space: no mirroring and no doubling.

The game's worth is measured over every one of them, against the truth. That
error is exact, so it is the same on every machine and comparable between
seeds. The measure it replaces - the positions the model happens to visit in
its games against a random opponent - drifts as the model drifts, and is not
the fixed evaluation set the fair fight asks for.

A game may offer such a set of positions, and the framework asks for it when
it measures a model. A game that offers none is measured the way it is now.

### Stage 1 - map the current failure

A game with a trained model, move by move: the heaps, the nim-sum before, who
moved, the move, the nim-sum after, whether the move left the opponent in a
losing position, the oracle's worth before and after, and the final outcome.

Built on the public API: the demonstration already replays traces, so the
framework is not touched.

### Stage 2 - the fixed evaluation set

- The error of the model's worth over all 120 positions.
- The share of positions where the policy's move is as good as the best
  available.

### Stage 3 - the three questions

- **Does it understand the position?** Does the worth track the truth across
  the whole space? Does the move change with the configuration, or is it
  degenerate?
- **Does it learn from outcomes?** The curve: the fixed-set error against the
  budget (0, 1k, 2k, 5k, 10k, 20k), the truth-trained model beside the
  self-trained one, three seeds, the spread reported.
- **Does it discover the winning strategy?** The share of winning positions
  where it plays a move that leaves the opponent in a losing position.

### Stage 4 - the representation

A fair fight with one variable changed: the same architecture, budget, seeds
and evaluation, only the encoder differs.

| | features | width |
|---|---|---|
| A | the three heap sizes, as today | 3 |
| D | the three bit counts: the exclusive or is zero exactly when every bit count is even | 3 |
| C | the nine bits | 9 |
| E | the three sizes, normalised by the real maximum heap instead of 7 | 3 |

A against D is the like-for-like fight: the same width and the same number of
parameters. C has more information and more parameters, and the write-up says
so rather than pretending otherwise. E is a pure range fix.

Each variant is measured on the fixed set, on the curve, and on the
winning-move share.

### Stage 5 - the learning dynamics

The three initialisations inside Nim, on the fixed set and the curve; and
whether the hidden weights actually move during training.

### What comes out of it

A dated entry in [RESULTS.md](RESULTS.md), a Nim section in
[FINDINGS.md](FINDINGS.md), and the changelog. The write-up says which
limitation the evidence points at - representation, budget, signal, or
capacity - and what it does not show. If the evidence does not distinguish
between them, that is said plainly.

### Out of scope, deliberately

- No new features as the default.
- No change to the architecture.
- No change to the training method.
- No change to the existing measurements.

The variant code stays in the repository so the experiment can be repeated.
The default stays as it is until a later phase is opened on the evidence.

### The tests

Hand-verifiable ones: the nim-sum of a position, the count of the positions
(120), the start among them, every one legal, the heaps matching the sum. The
analysis functions are tested on a model with known weights. Every test is
mutation-checked: mutate, and confirm that the test fails.

## Phase 2 - Connect Four, small board

5x5, four in a row, exactly solvable.

A game module in `games/connect-four/`, following the shape of Nim and
tic-tac-toe: the rules, the features, a memoised minimax with memory of its
own, the samples, and a run of the judge. The oracle is cross-checked against
an independent brute-force search, as Nim's is. The framework is not touched;
the demonstration grows a Connect Four section and a league.

The 7x6 question stays open until the small board is done and measured.

## Parked, and where it stands

From [notes.md](notes.md), still to do:

- The CPU benchmark and profile: training time for 1, 10 and 100 models, the
  time for all 4,950 pairings with the sides reversed, and where the time
  goes. This is the prerequisite for the CUDA roadmap, and it also says
  whether the hundred leagues of the tournament arc are affordable here.
- The hundred leagues: 100 leagues of 100 models, the top ten of each into a
  knockout of paired games, the lineage kept, and the report.
- The opponent diversity groups: one fixed opponent, a rotating pool, the
  league, and opponents chosen to expose weaknesses.
- The three small experiments: exploration and temperature, whether
  specialists emerge, and whether search amplifies learning.
- The dynamic storage of networks, and the visualiser.
