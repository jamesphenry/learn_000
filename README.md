# learn_000

A small C project that learns to play games, and measures what it has
learned. It exists to answer one question carefully: **which way of starting
a network off learns the game best?**

The answer only means something if the fight is fair, so all the models get
the same architecture, the same training algorithm, the same number of
games and the same evaluation set, and they are measured before and after
training rather than on the final win rate alone.

There are two games now: tic-tac-toe, and Nim. The framework does not know
which game it is playing — the game says what a position is worth to the
player to move, and the framework measures it.

## The model

The model answers with the worth of a position to the player to move: one if
they win it, a half if it is drawn and nought if they lose it. The policy
looks one move ahead, asks what each move would leave behind, and takes the
best of them. That is the same shape as the perfect player, so what the model
says about a position can be compared straight against what the perfect
player knows about it.

The model is taught by the perfect player, which says what each position it
visits is worth, or by its own games, where the result of the game is the
target. The second has a moving target and learns more slowly; both are in
the arena.

<video src="assets/Perfect_Model_Paradox.mp4" controls width="800"></video>

## The pieces

The framework is in `include/` and `src/`, and knows nothing about any
particular game. Each game is in its own directory, with its own tests.

| module | what it is for |
|---|---|
| `game` | the framework's view of a game: what a position is, how a game is played, and what the perfect player knows about it. |
| `neuron` | a weighted sum, and the activations: a clamp, a rectified unit and a sigmoid. |
| `layer` | a layer of neurons with a chosen activation. |
| `network` | layers joined up, and the two ways of filling them in: at random, or all at the same value. |
| `rng` | a seeded generator, so every run can be repeated. |
| `policy` | a game and a network, and the move they make together. |
| `trainer` | teaches a network the worth of a position. |
| `evaluate` | measures a model: its win rate, how honest its probabilities are, how much worth it gives away, and how it does against the perfect player. |
| `judge` | a referee, which checks that every move offered was allowed and that the result agrees with the perfect player's reckoning. |
| `arena` | builds the models, teaches them, measures them, and has them play each other. |
| `games/tic-tac-toe` | the board, its cells, whether somebody has won, and the perfect player that works it all out. |
| `games/nim` | the heaps of stones, and the perfect player that knows the rule of the exclusive or. |

## Building and testing

```sh
make              # build
make test         # run every test, under valgrind when it is installed
make test-valgrind # run every test under valgrind, and say so if it is not there
make test-asan    # run every test under the address and behaviour sanitisers
make test-quick   # run every test without the memory checker
make lint         # check the sources
make run          # build and run the demonstration
```

To run one test:

```sh
TST=test/game.test.c make test
```

The demonstration takes the number of models for the league, and prints the
whole table of the league when the word `table` is given as well:

```sh
./src/learn_000 100        # a league of a hundred models
./src/learn_000 100 table  # the same, with the whole table of the league
```

## What we found

The short version: **a network started at random learns the game; one
started at zero never does; and every network started with all the weights
the same fails, whatever the value they are given.** The sameness, not the
size, is what does it — a weight of 1.0 and a weight of 100.0 are very
different experiments and they fail in the same way. The one that is
pinned is the large one: its output's sum starts at 17, the sigmoid sits
at one, and the pull of a win towards it is exactly zero.

Getting to that answer took some fixing first. The measurement that looked
like it was measuring skill was measuring the model on the positions it
already knew, and the model could not learn at all until the shape of the
network, the activation and the target were changed. All of it is written
up in [FINDINGS.md](docs/FINDINGS.md), and every measurement is logged with
the date it was taken in [RESULTS.md](docs/RESULTS.md).

The results, over five runs with a seed of their own, and then again with
the game of Nim:

```
  tic-tac-toe
  run  seed       zero   random  maximum   the models against each other
  1    1         0.278    0.090    0.216   zero 57  random 275  maximum 121
  2    2         0.276    0.086    0.208   zero 71  random 241  maximum 108
  3    3         0.279    0.098    0.244   zero 80  random 284  maximum 93
  4    4         0.281    0.090    0.226   zero 62  random 264  maximum 126
  5    5         0.278    0.104    0.214   zero 70  random 249  maximum 140

  Nim
  run  seed       zero   random  maximum   the models against each other
  1    1         0.195    0.105    0.197   zero 113  random 381  maximum 106
  2    2         0.219    0.060    0.237   zero 99  random 392  maximum 109
  3    3         0.210    0.051    0.241   zero 99  random 393  maximum 108
  4    4         0.183    0.075    0.235   zero 149  random 310  maximum 141
  5    5         0.195    0.105    0.229   zero 102  random 386  maximum 112
```

The numbers are how far the model was from the truth, after training: the
smaller the better. The random initialisation wins in both games, by a
wide margin, and the other two learn nothing worth the name in either.

## The second game

Nim is nothing like tic-tac-toe: the heaps do not belong to a player, there
are no draws, and the perfect player knows a position is worth one or
nought and never a half. It is in the framework because the notes ask for a
second game, to show that the abstractions are reusable rather than
tic-tac-toe with a different coat on.

Two things came out of it that tic-tac-toe could not have told us.

**The perfect player has to keep playing.** The perfect player for Nim had
no move to offer in a position it could not win, so the game stopped there
and was counted as a draw. The arena found it: half of the games against the
perfect player were being counted as draws, in a game that cannot be drawn.

**The rules of the game are not the rules of the framework.** A game of Nim
has no perfect play from every position: the player to move in a position
whose heaps all have the same bits in them is lost, and the model has to
learn that too. The model does: after training it wins every game it starts
and loses every game it does not, against a perfect player that cannot lose.

## A league of a hundred models

A hundred models, each with a seed of its own, every one playing every
other one, half of them taught by the perfect player and half learning from
their own games. It produces **three rankings that do not agree**:

| ranking | what it tells you |
|---|---|
| training | how well did the model learn from its training experience? |
| league | how well does it perform against unfamiliar opponents? |
| general | does its performance hold up against a different kind of model? |

**The average model moves twenty-seven places** between the training
measure and the league. The champion was sixty-fifth by the training
measure, the model that topped the training measure came thirty-fourth in
the league, and the champion turned out to be a model that taught itself.
The whole of it is written up in [LEAGUE.md](docs/LEAGUE.md).

The general ranking is a controlled test: four models are held back from
the league and every model is measured against them, so the comparison is
the same for everybody. It agrees with the training measure and disagrees
with the league — the league's champion is seventy-eighth on it.

The same league is played in the game of Nim, and there the league *does*
tell the two kinds of model apart: in tic-tac-toe the models draw with one
another so often that a hundred of them cannot be told apart, and in Nim
no game is ever drawn.

## Two models and nobody to tell them the answers

The arena above is taught by the perfect player. The duel is not: two
models, each with a seed of its own, each learning from the result of the
games it plays. They meet three times — before either has learned
anything, after each has learned from its own games, and after they have
learned against each other.

```
  before either has learned anything   seed 42 won 100, seed 1978 won 100, none drawn
  after each has learned on its own     0 won, 100 drawn, 100 lost
  after they have learned together      19 won, 173 drawn, 8 lost
```

A model does learn from its own games — the truth error falls from 0.34 to
0.20 over fifty thousand of them — and two models that learned separately
agree with each other. Learning against each other afterwards adds nothing,
and the measurement says so: against the perfect player the first of them
gets worse, having spent fifty thousand games learning one opponent
instead of the game.

## Project structure

```
learn_000
├── include
│   ├── arena.h
│   ├── evaluate.h
│   ├── game.h
│   ├── judge.h
│   ├── layer.h
│   ├── main.h
│   ├── network.h
│   ├── neuron.h
│   ├── policy.h
│   ├── rng.h
│   └── trainer.h
├── src
│   ├── arena.c
│   ├── evaluate.c
│   ├── game.c
│   ├── judge.c
│   ├── layer.c
│   ├── main.c
│   ├── network.c
│   ├── neuron.c
│   ├── policy.c
│   ├── rng.c
│   └── trainer.c
├── games
│   ├── nim
│   │   ├── include/nim.h
│   │   ├── src/nim.c
│   │   └── test/nim.test.c
│   └── tic-tac-toe
│       ├── include/board.h
│       ├── include/minimax.h
│       ├── include/tictactoe.h
│       ├── src/board.c
│       ├── src/minimax.c
│       ├── src/tictactoe.c
│       └── test
│           ├── board.test.c
│           ├── minimax.test.c
│           └── tictactoe.test.c
├── test
│   ├── test-runner
│   └── *.test.c
├── docs
│   ├── FINDINGS.md
│   ├── LEAGUE.md
│   ├── PLAN.md
│   ├── RESULTS.md
│   └── notes.md
├── CHANGELOG.md
├── LICENSE.md
├── Makefile
└── README.md
```

`docs/` holds the documentation: what was found and what it means
([FINDINGS.md](docs/FINDINGS.md)), the league of a hundred models
([LEAGUE.md](docs/LEAGUE.md)), a log of the measurements with the date each
was taken ([RESULTS.md](docs/RESULTS.md)), the plan of record for the work
ahead ([PLAN.md](docs/PLAN.md)), and the thinking behind the project,
including the questions this was built to answer ([notes.md](docs/notes.md)).

## Licence

See [LICENSE.md](LICENSE.md).
