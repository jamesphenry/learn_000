# What we found

The notes ask which way of starting a network off learns tic-tac-toe best,
and warn that the answer is only worth having if the fight is fair. This is
what came out of the experiment, and what had to be fixed before the
question could be answered at all.

Everything here is measured. Where a number appears it was produced by the
code in this repository, and the way to reproduce it is given.

---

## 1. The model could not learn, and there were three reasons

The arena measures a model before and after training. The first version of
the model passed those measurements and learned nothing: five hundred games
of training moved the win rate from 66.7% to 67.0%, and sixty four times as
many games made no difference at all.

The measure that showed it was the worth of the positions the model played:
the share of the cells it chose that left the position worth as much as the
best cell would have. Over every one of the 4,520 positions there is, an
untrained model kept the value 58% of the time and a trained one 71%.

Three separate faults, each measured on its own.

### The target taught the wrong thing

The trainer regressed only the cell that was played, towards the result of
the whole game: one if X won and nought if it did not. Every cell of a
position was therefore pushed towards the same number, so what the network
learned was *how good a board is*, not *which cell to play*. The nine
outputs' spread collapsed from 0.42 to 0.15 while it trained.

### Nine numbers for one position do not fit

The model answered with a number for every cell, all of which had to come
out of one shared hidden layer. That layer then had to hold enough for nine
different questions about the same board at once, and it does not: trained
on the perfect answer for every cell, the shape still only reached 84%.

### The clamped activation was linear in practice

`neuron_clamp` is `max(x, -1)`, which passes everything above its floor on
unchanged. The hidden layer was therefore linear over almost all of the
range it worked in, and the only thing that made it bend was a sum falling
below the floor, where the unit stops learning altogether.

Replacing it with a rectified unit took the same task from 78% to 95%.

### What the model became

One number: the worth of the position to X. The policy looks one cell
ahead and takes the best of them. The truth of a position can now be
measured straight against what the perfect player knows about it, and the
same task reaches 95% — and a model trained that way draws every game
against the perfect player in both colours.

---

## 2. The measurement that flattered the model

The measurement of how much worth a model gives away played the model
against *itself*, so every position it was measured in was one it already
knew. The same model gave away:

| measured against | worth given away, a move |
|---|---|
| itself | 0.048 |
| a player that plays at random | 0.199 |

Nearly four times as much once the opponent was varied. The arena now
reports both, because the gap between them is the difference between a
model that knows the positions it has seen and one that knows the game.

The round robin is what exposed it. A model that never lost a game to the
perfect player was losing a third of its games to a random one — a
contradiction that no measurement of the model's own games could have shown.

---

## 3. The three ways of starting off

The arena gives all three models the same architecture, the same training
algorithm, the same number of games and the same evaluation set, and it
runs the whole thing several times with a seed of its own so the ranking can
be seen to survive.

### Zero: never learns

Every weight and bias at zero. The gradient that would reach the hidden
layer is multiplied by the hidden values, and those are zero, so nothing
below the output layer ever moves. The model can learn a constant and
nothing else. It gives away the same worth whether it has trained or not,
and it draws twenty one games out of two hundred against the perfect
player — the same as a player that had never heard of the game.

### Random: learns the game

Small random values, so the neurons start out different from one another
and the hidden sums gather around zero. The output's sum starts between 0.2
and 0.7, the sigmoid is nowhere near pinned, and the pull towards a winning
target is 0.38 — nearly as much as the pull towards a losing one.

It drew 974 games out of a thousand against the perfect player, lost 26,
and beat the model that started with every mark the same value in every
single game of the round robin.

### Maximum: cannot learn from a win

Every weight and bias at the same value of 1.0. Two things go wrong at once,
and they can be told apart.

**The size.** The hidden sums all come to 1.00, and the output's sum comes
to **33**. The sigmoid is pinned at one, so the gradient towards a winning
target is **exactly zero**. The model cannot learn from a single win. It
can learn from losses, though — the pull towards a losing target is 1.00 —
so it settles into believing that every position is a loss, which is
exactly what it does: it drew none of a thousand games against the perfect
player and lost every one.

**The sameness.** Every hidden unit sees the same thing, so the hidden layer
cannot tell one position from another.

The notes ask for this experiment and warn that "a weight of 1.0 and a
weight of 100.0 are very different experiments". Here is the grid, over
three runs each, with the worth given away measured against a player that
plays at random:

| started | out by the truth | gave away a move | against the perfect player |
|---|---|---|---|
| zero | 0.285 | 0.586 | 63 drawn, 537 lost |
| varied, small (0.05) | 0.104 | 0.218 | 496 drawn, 104 lost |
| **varied, medium (0.5)** | **0.085** | **0.192** | **600 drawn, 0 lost** |
| varied, big (1.0) | 0.079 | 0.196 | 565 drawn, 35 lost |
| all the same, small (0.05) | 0.211 | 0.315 | 0 drawn, 600 lost |
| all the same, medium (0.5) | 0.233 | 0.442 | 0 drawn, 600 lost |
| all the same, big (1.0) | 0.213 | 0.366 | 0 drawn, 600 lost |

Three runs each, twenty thousand games each, the same conditions for every
one of them.

**The size is not what does it.** Every initialisation with all the weights
the same fails completely, whether the value is 0.05, 0.5 or 1.0. The size
only decides *how* it fails: at 1.0 the output's sum is 17 and the sigmoid
is pinned at one, so the model cannot learn from a win at all; at 0.05 the
sum is small and the sigmoid is free, and the model still fails, because the
hidden units are all the same function of the board and the gradient that
reaches them is the same for each, so they stay the same. It learns one
number and cannot tell one position from another.

**The varied initialisations all learn**, and the size matters to how well:
0.05 is clearly worse than 0.5, and 1.0 is a little worse than 0.5 as well.
The middle of the range is the one to pick.

### The same thing again, in one table

What each model looks like before it has learned anything, which is what
the arena now reports:

```
  started   the hidden sums the output's sum   it says   to a win  to a loss
  zero       0.00 to 0.00     0.00 to 0.00       0.500      0.500      0.500
  random    -1.19 to 1.21    -0.06 to 1.02       0.618      0.382      0.618
  maximum    1.00 to 1.00    17.00 to 17.00      1.000      0.000      1.000
```

Read it like this. The zero model has no sums in the hidden layer at all, so
the error that would reach it is multiplied by nothing and it never moves.
The random model has sums spread either side of zero, so its units are
different from one another and it can tell one position from another. The
maximum model has one hidden sum and one output sum and nothing else, its
sigmoid is pinned at one, and the pull of the loss towards a winning target
is **exactly zero** — it can only learn from losses, so it settles into
believing that every position is a loss, which is exactly what it does.

---

## 4. The answers to the questions in the notes

**Which initialisation learns the game best?** Random. It is the only one
that learns, and it wins the round robin in five runs out of five.

**Does the zeroed model reveal something fundamental about the training
algorithm?** Yes, and the model with every weight the same reveals it more
clearly still. The training algorithm cannot break the symmetry of
identical neurons through its own gradient: the error that reaches a hidden
unit is the same for every unit, so units that start the same stay the
same, and the model learns one number where it needed to learn nine. The
zeroed model is the same failure with the error cut off altogether — the
error that would reach the hidden layer is multiplied by the hidden values,
and if those are zero there is nothing to multiply — so it can learn a
constant and nothing else. Neither is a fault in the optimiser but a
property of how the gradient flows.

**Is "maximum value" a fair name for it?** Only with the number spelled
out, which the notes asked for, and the number turns out not to be the
point. Every value tried fails; the failure is the sameness, not the size.
At 1.0 there is saturation on top of the sameness, at 0.05 the saturation
goes away and the sameness alone is enough.

**Can the evaluation be trusted?** Only after the round robin caught it
out. A measurement of a model's own games cannot see the faults in the
model, because the faults are in the positions it never reaches.

---

## 5. What is still open

**The self-play teacher.** Its target is the result of the game under the
model's own current play, so it chases a moving target: 93 draws out of a
hundred against the perfect player after fifty thousand games, where the
perfect player's own teaching reaches 200 out of 200 after ten thousand. A
bootstrapped target — the worth of the position after the move, taken from
the model's own estimates — was tried with the old shape of the model and
stalled at 60%; it has not been tried with the current one.

**The models are narrow.** A model learns the worth of the positions it
visits. With enough games it covers the game well enough to play it
perfectly against a perfect opponent, but it is still worse than a random
player in the positions it has not seen.

**The framework.** The notes want tic-tac-toe moved into `games/` and a
second game to show the abstractions are genuinely reusable. The
game-to-feature conversion is already separate from the neural network,
which was the first step on that list.
