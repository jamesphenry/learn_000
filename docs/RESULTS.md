# Results

A log of the measurements, newest first. Every number in here was produced
by the code in this repository on the date the entry carries, and the way to
reproduce it is given with it.

The write-ups are elsewhere: [FINDINGS.md](FINDINGS.md) says what the
measurements mean, and [LEAGUE.md](LEAGUE.md) is about the league of a
hundred models.

---

## 2026-10-21

The first measurements with the game of Nim in the framework, and the first
of the league of a hundred models in both games.

The machine: an Intel Celeron N4020, two cores, on Linux 6.12. The times
below are for this machine and will be different on another.

### The grid of initialisations

Every way of starting a network off, measured the same way for every one
of them: three runs, twenty thousand games of training each, and the worth
given away measured against a player that plays at random.

The game of tic-tac-toe:

| started | out by the truth | gave away a move | against the perfect player |
|---|---|---|---|
| zero | 0.280 | 0.281 | 0 won, 61 drawn, 539 lost |
| varied, small (0.05) | 0.109 | 0.130 | 0 won, 540 drawn, 60 lost |
| varied, medium (0.5) | 0.090 | 0.096 | 0 won, 600 drawn, 0 lost |
| varied, big (1.0) | 0.100 | 0.112 | 0 won, 600 drawn, 0 lost |
| all the same, small (0.05) | 0.229 | 0.260 | 0 won, 99 drawn, 501 lost |
| all the same, medium (0.5) | 0.233 | 0.207 | 0 won, 12 drawn, 588 lost |
| all the same, big (1.0) | 0.228 | 0.223 | 0 won, 62 drawn, 538 lost |

The game of Nim:

| started | out by the truth | gave away a move | against the perfect player |
|---|---|---|---|
| zero | 0.208 | 0.613 | 0 won, 0 drawn, 600 lost |
| varied, small (0.05) | 0.168 | 0.227 | 251 won, 0 drawn, 349 lost |
| varied, medium (0.5) | 0.072 | 0.147 | 300 won, 0 drawn, 300 lost |
| varied, big (1.0) | 0.107 | 0.168 | 300 won, 0 drawn, 300 lost |
| all the same, small (0.05) | 0.233 | 0.540 | 0 won, 0 drawn, 600 lost |
| all the same, medium (0.5) | 0.234 | 0.541 | 0 won, 0 drawn, 600 lost |
| all the same, big (1.0) | 0.217 | 0.580 | 0 won, 0 drawn, 600 lost |

### The training arena

Five runs, each with a seed of its own, twenty thousand games of training
each. The first three numbers are how far the model was from the truth
after training: the smaller the better.

```
  the game of tic-tac-toe
  run  seed       zero   random  maximum   the models against each other
  1    1         0.278    0.090    0.216   zero 57  random 275  maximum 121
  2    2         0.276    0.086    0.208   zero 71  random 241  maximum 108
  3    3         0.279    0.098    0.244   zero 80  random 284  maximum 93
  4    4         0.281    0.090    0.226   zero 62  random 264  maximum 126
  5    5         0.278    0.104    0.214   zero 70  random 249  maximum 140

  the game of Nim
  run  seed       zero   random  maximum   the models against each other
  1    1         0.195    0.105    0.197   zero 113  random 381  maximum 106
  2    2         0.219    0.060    0.237   zero 99  random 392  maximum 109
  3    3         0.210    0.051    0.241   zero 99  random 393  maximum 108
  4    4         0.183    0.075    0.235   zero 149  random 310  maximum 141
  5    5         0.195    0.105    0.229   zero 102  random 386  maximum 112
```

Over all five runs:

| | tic-tac-toe | Nim |
|---|---|---|
| the random model won the round robin | 5 times out of 5 | 5 times out of 5 |
| the zero model won the round robin | 0 times out of 5 | 0 times out of 5 |
| the maximum model won the round robin | 0 times out of 5 | 0 times out of 5 |
| the random model against the perfect player | 0 won, 1,000 drawn, 0 lost | 500 won, 0 drawn, 500 lost |
| the random model gave away a move | 0.102 | 0.153 |
| the zero model gave away a move | 0.293 | 0.615 |
| the maximum model gave away a move | 0.226 | 0.555 |

The model that starts at random wins every game it starts in Nim, and
loses every game it does not, against a perfect player that cannot lose.

### Two models, and nobody to tell them the answers

Tic-tac-toe, seeds 42 and 1978, fifty thousand games of training each,
meeting three times.

| | seed 42 | seed 1978 |
|---|---|---|
| **before either has learned anything** | | |
| gave away a move | 0.269 | 0.175 |
| out by the truth | 0.340 | 0.220 |
| drew with the perfect player | 0 | 100 |
| lost to the perfect player | 200 | 100 |
| the games each of them won | 100 | 100 |
| the games were drawn | 0 | 0 |
| **after each has learned from its own games** | | |
| gave away a move | 0.108 | 0.104 |
| out by the truth | 0.203 | 0.141 |
| drew with the perfect player | 200 | 200 |
| lost to the perfect player | 0 | 0 |
| the games each of them won | 0 | 100 |
| the games were drawn | 100 | 100 |
| **after they have learned against each other** | | |
| gave away a move | 0.162 | 0.123 |
| out by the truth | 0.231 | 0.189 |
| drew with the perfect player | 100 | 200 |
| lost to the perfect player | 100 | 0 |
| the games each of them won | 19 | 8 |
| the games were drawn | 173 | 173 |

### The league of a hundred models

A hundred models, seeds 1 to 100, half of them taught by the perfect player
and half learning from their own games, twenty thousand games of training
each, and two hundred games a match: 4,950 matches and 990,000 games.

```
  the game of tic-tac-toe
  the champion is model 97, with the seed 98: 2006 won, 17609 drawn, 185 lost
    place by training: 65, by league: 1, by generalisation: 78, of 100
  the average model moves 27.0 places between the training measure and the league
  the truth-taught models took places 2 to 69, the self-taught ones 1 to 100
  47 of the 50 truth-taught models give away nothing at all while learning
  19 of the 50 self-taught models give away nothing at all while learning

  the game of Nim
  the champion is model 40, with the seed 41: 15143 won, 0 drawn, 4657 lost
    place by training: 21, by league: 1, by generalisation: 10, of 100
  the average model moves 16.4 places between the training measure and the league
  the truth-taught models took places 1 to 50, the self-taught ones 51 to 100
  43 of the 50 truth-taught models give away nothing at all while learning
  none of the 50 self-taught models gives away nothing at all
```

The league in Nim tells the two kinds of model apart and the league in
tic-tac-toe does not, because a game of Nim is never drawn and a game of
tic-tac-toe very often is.

### How long it took

| | time |
|---|---|
| the league of a hundred models, tic-tac-toe | 2 minutes 40 seconds |
| the league of a hundred models, Nim | 1 minute 16 seconds |
| the whole demonstration with a hundred models, both games | about 4 minutes 30 seconds |

### How to reproduce it

The settings are the ones in `arena_settings_init` in `src/arena.c`: sixteen
hidden neurons, twenty thousand games of training, two hundred games of
evaluation, two hundred games a match, a learning rate of 0.05, a
temperature of 0.3, and the models started off with their weights varied by
0.5 around a value of 1.0.

```sh
make                 # build
./src/learn_000 100  # the whole demonstration, with a league of a hundred models
```

The league of a hundred models is `arena_league` in `src/arena.c`, the
arena of five runs is `arena_survey`, and the grid of initialisations is
`arena_run` called once for each way of starting a network off.

The demonstration prints the leader of each of the three rankings and the
record of the champion. The whole table of the league comes out of it when
the word `table` is given as well:

```sh
./src/learn_000 100 table
```

Every model is printed in the order it finished, with the place it took in
the other two rankings beside it. The figures above about the places the
models moved and about the models that give away nothing at all while
learning are counted from that table.
