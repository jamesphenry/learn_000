# learn_000

A small C project that learns to play tic-tac-toe, and measures what it has
learned. It exists to answer one question carefully: **which way of starting
a network off learns the game best?**

The answer only means something if the fight is fair, so all the models get
the same architecture, the same training algorithm, the same number of
games and the same evaluation set, and they are measured before and after
training rather than on the final win rate alone.

## The model

The model answers with the worth of a position to X: one if X wins it, a
half if it is drawn and nought if O wins it. The policy looks one cell
ahead, asks what each cell would leave behind, and takes the best of them —
the highest for X and the lowest for O. That is the same shape as the
perfect player, so what the model says about a position can be compared
straight against what the perfect player knows about it.

The model is taught by the perfect player, which says what each position it
visits is worth, or by its own games, where the result of the game is the
target. The second has a moving target and learns more slowly; both are in
the arena.

## The pieces

| module | what it is for |
|---|---|
| `board` | the board, its cells and whether somebody has won. No global state, so any number of games can be in flight at once. |
| `neuron` | a weighted sum, and the activations: a clamp, a rectified unit and a sigmoid. |
| `layer` | a layer of neurons with a chosen activation. |
| `network` | layers joined up, and the two ways of filling them in: at random, or all at the same value. |
| `rng` | a seeded generator, so every run can be repeated. |
| `policy` | a board and a network, and the move they make together. |
| `game` | a game being played, with the moves that were made and the moves that were offered. |
| `trainer` | teaches a network the worth of a position. |
| `evaluate` | measures a model: its win rate, how honest its probabilities are, how much worth it gives away, and how it does against the perfect player. |
| `minimax` | the perfect player, and what a position is really worth. It keeps what it has worked out. |
| `judge` | a referee, which checks that every move offered was allowed and that the result agrees with the perfect player's reckoning. |
| `arena` | builds the models, teaches them, measures them, and has them play each other. |

## Building and testing

```sh
make            # build
make test       # run every test
make test-asan  # run every test under the address and behaviour sanitisers
make test-quick # run every test without the memory checker
make lint       # check the sources
make run        # build and run the demonstration
```

To run one test:

```sh
TST=test/board.test.c make test
```

## What we found

The short version: **a network started at random learns the game; one
started at zero never does; one started with every weight at the same large
value cannot learn from a win at all.** The last of those is not a hunch —
the output layer's sum starts at 33, the sigmoid is pinned at one, and the
gradient towards a winning target is exactly zero.

Getting to that answer took some fixing first. The measurement that looked
like it was measuring skill was measuring the model on the positions it
already knew, and the model could not learn at all until the shape of the
network, the activation and the target were changed. All of it is written
up in [FINDINGS.md](FINDINGS.md).

The results, over five runs with a seed of their own:

```
  run  seed       zero   random  maximum   the models against each other
  1    1         0.296    0.089    0.206   zero 43  random 367  maximum 150
  2    2         0.279    0.070    0.201   zero 45  random 264  maximum 141
  3    3         0.281    0.094    0.230   zero 57  random 345  maximum 127
  4    4         0.277    0.089    0.219   zero 42  random 247  maximum 141
  5    5         0.291    0.081    0.242   zero 53  random 256  maximum 141

  over all 5 runs
  zero     won the round robin 0 times out of 5;  it won 240 and lost 1479
  random   won the round robin 5 times out of 5;  it won 1479 and lost 22
  maximum  won the round robin 0 times out of 5;  it won 700 and lost 918
```

## A league of a hundred models

A hundred models, each with a seed of its own, every one playing every
other one, half of them taught by the perfect player and half learning from
their own games. It produces **three rankings that do not agree**:

| ranking | what it tells you |
|---|---|
| training | how well did the model learn from its training experience? |
| league | how well does it perform against unfamiliar opponents? |
| general | does its performance hold up against a different kind of model? |

The champion was eleventh by the training measure and first by the league;
the model that topped the training measure came thirty-second. The whole
of it is written up in [LEAGUE.md](LEAGUE.md).

The general ranking is a controlled test: four models are held back from
the league and every model is measured against them, so the comparison is
the same for everybody. It does not agree with the league either — the
league's champion is ninth by it, and the model it likes best is only
twelfth in the league.

## Two models and nobody to tell them the answers

The arena above is taught by the perfect player. The duel is not: two
models, each with a seed of its own, each learning from the result of the
games it plays. They meet three times — before either has learned
anything, after each has learned from its own games, and after they have
learned against each other.

```
  before either has learned anything   seed 42 won 100, seed 1978 won 100, none drawn
  after each has learned on its own     28 won, 109 drawn, 63 lost
  after they have learned together      37 won, 105 drawn, 58 lost
```

A model does learn from its own games — the truth error falls from 0.34 to
0.21 over fifty thousand of them — and two models that learned separately
agree with each other. Learning against each other afterwards adds nothing,
and the measurement says so: against the perfect player the first of them
gets worse, having spent fifty thousand games learning one opponent
instead of the game.

## Project structure

```
learn_000
├── include
│   ├── arena.h
│   ├── board.h
│   ├── evaluate.h
│   ├── game.h
│   ├── judge.h
│   ├── layer.h
│   ├── main.h
│   ├── minimax.h
│   ├── network.h
│   ├── neuron.h
│   ├── policy.h
│   ├── rng.h
│   └── trainer.h
├── src
│   ├── arena.c
│   ├── board.c
│   ├── evaluate.c
│   ├── game.c
│   ├── judge.c
│   ├── layer.c
│   ├── main.c
│   ├── minimax.c
│   ├── network.c
│   ├── neuron.c
│   ├── policy.c
│   ├── rng.c
│   └── trainer.c
├── test
│   ├── test-runner
│   └── *.test.c
├── CHANGELOG.md
├── FINDINGS.md
├── LEAGUE.md
├── LICENSE.md
├── Makefile
├── README.md
└── notes.md
```

`notes.md` holds the thinking behind the project, including the questions
this was built to answer.

## Licence

See [LICENSE.md](LICENSE.md).
