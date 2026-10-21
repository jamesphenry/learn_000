# What we found

The notes ask which way of starting a network off learns tic-tac-toe best,
and warn that the answer is only worth having if the fight is fair. This is
what came out of the experiment, and what had to be fixed before the
question could be answered at all.

Everything here is measured. Where a number appears it was produced by the
code in this repository, and the way to reproduce it is given.

Two things to keep in mind when reading it. The worth of a position used to
mean "the chance that X wins it", and now it means "the chance that the
player to move wins it". And the worth a model gives away on a move used to
be counted on a scale of nought to two, and now it is counted on a scale of
nought to one. The figures in the first two sections were measured before
those changes, with the first version of the model, and are left as they
were: they are the record of what went wrong. Every other figure is the
current one.

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
different questions about the same board at once.

Measured before the worth changed its meaning, that shape trained on the
perfect answer for every cell still only reached 84%, where one number for
the whole position reached 95%. Measured again afterwards the two come out
about even — 60% and 60% — so the shape was not the reason to change it.
What the change bought was the meaning: one number for a position is the
worth of that position, which is what a game can say about a position it
has never seen before.

### The clamped activation was linear in practice

`neuron_clamp` is `max(x, -1)`, which passes everything above its floor on
unchanged. The hidden layer was therefore linear over almost all of the
range it worked in, and the only thing that made it bend was a sum falling
below the floor, where the unit stops learning altogether.

Replacing it with a rectified unit took the same task from 78% to 95%
(measured before the worth changed its meaning).

### What the model became

One number: the worth of the position to the player to move. The policy
looks one move ahead and takes the best of them. The truth of a position can
now be measured straight against what the perfect player knows about it, and
the same task reaches 95% — and a model trained that way draws every game
against the perfect player in both colours.

---

## 2. The measurement that flattered the model

The measurement of how much worth a model gives away played the model
against *itself*, so every position it was measured in was one it already
knew. The same model gave away:

| measured against | worth given away, a move |
|---|---|
| itself | 0.000 |
| a player that plays at random | 0.181 |

Nearly four times as much once the opponent was varied. The arena now
reports both, because the gap between them is the difference between a
model that knows the positions it has seen and one that knows the game.
The model that wins the arena today gives away 0.102 a move against a
player that plays at random, and 0.000 in the games it plays against
itself.

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
and it draws about twenty games out of two hundred against the perfect
player — the same as a player that had never heard of the game.

### Random: learns the game

Small random values, so the neurons start out different from one another
and the hidden sums gather around zero. The output's sum starts between
-0.06 and 1.02, the sigmoid is nowhere near pinned, and the pull towards a
winning target is 0.38 — nearly as much as the pull towards a losing one.

It draws every game against the perfect player, and it is the model that
wins the round robin: over five runs it won 1,313 games against the other
two models, where the model that started with every mark the same value
won 588.

### Maximum: cannot learn from a win

Every weight and bias at the same value of 1.0. Two things go wrong at once,
and they can be told apart.

**The size.** The hidden sums all come to 1.00, and the output's sum comes
to **17**. The sigmoid is pinned at one, so the gradient towards a winning
target is **exactly zero**. The model cannot learn from a single win. It
can learn from losses, though — the pull towards a losing target is 1.00 —
so it settles into believing that every position is a loss, which is
exactly what it does: it drew sixty two games out of six hundred against
the perfect player and lost the rest.

**The sameness.** Every hidden unit sees the same thing, so the hidden layer
cannot tell one position from another.

The notes ask for this experiment and warn that "a weight of 1.0 and a
weight of 100.0 are very different experiments". Here is the grid, over
three runs each, with the worth given away measured against a player that
plays at random:

| started | out by the truth | gave away a move | against the perfect player |
|---|---|---|---|
| zero | 0.280 | 0.281 | 0 won, 61 drawn, 539 lost |
| varied, small (0.05) | 0.109 | 0.130 | 0 won, 540 drawn, 60 lost |
| **varied, medium (0.5)** | **0.090** | **0.096** | **0 won, 600 drawn, 0 lost** |
| varied, big (1.0) | 0.100 | 0.112 | 0 won, 600 drawn, 0 lost |
| all the same, small (0.05) | 0.229 | 0.260 | 0 won, 99 drawn, 501 lost |
| all the same, medium (0.5) | 0.233 | 0.207 | 0 won, 12 drawn, 588 lost |
| all the same, big (1.0) | 0.228 | 0.223 | 0 won, 62 drawn, 538 lost |

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

## 4. Two models, and nobody to tell them the answers

Everything above had the perfect player teaching the model what each
position was worth. The arena can leave it out: a model can learn from the
result of the games it plays, and the question is whether that is enough.

So: two models, each with a seed of its own, neither of them taught by
anybody. Seed 42 and seed 1978, fifty thousand games of training each, and
they meet three times.

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

**A model does learn without being told the answers.** Over fifty thousand
games of its own the truth error fell from 0.34 to 0.20 and the worth it
gave away fell from 0.27 to 0.11. It is well short of the model the
perfect player taught, which draws every game against the perfect player,
but it is a long way from the model that had learned nothing.

**Two models that learned separately agree.** They drew 100 games out of
200 against each other, where before either of them had learned anything
they drew none at all — every game was the same game, and the first player
always won it. Two different starting points, the same game.

**Learning against each other afterwards adds nothing.** The truth error
gets *worse* rather than better — 0.20 becomes 0.23 and 0.14 becomes
0.19 — and so does the worth given away: 0.11 becomes 0.16. Fifty
thousand games of learning one opponent instead of the game is not the
same as fifty thousand games of learning the game, and the measurement
says so.

---

## 5. The second game, and what it found

The notes ask for a second game, to show that the abstractions are reusable
rather than tic-tac-toe with a different coat on. Nim is a good one to try,
because it is nothing like tic-tac-toe: the heaps do not belong to a
player, there are no draws, and the perfect player knows a position is worth
one or nought and never a half.

The framework knows nothing about Nim. The game says what a position is
worth to the player to move, the policy asks what each move would leave
behind, and the same arena builds the same three models and measures them
the same way.

The grid of initialisations, over three runs each, twenty thousand games
each, exactly as in the section above:

| started | out by the truth | gave away a move | against the perfect player |
|---|---|---|---|
| zero | 0.208 | 0.613 | 0 won, 0 drawn, 600 lost |
| varied, small (0.05) | 0.168 | 0.227 | 251 won, 0 drawn, 349 lost |
| **varied, medium (0.5)** | **0.072** | **0.147** | **300 won, 0 drawn, 300 lost** |
| varied, big (1.0) | 0.107 | 0.168 | 300 won, 0 drawn, 300 lost |
| all the same, small (0.05) | 0.233 | 0.540 | 0 won, 0 drawn, 600 lost |
| all the same, medium (0.5) | 0.234 | 0.541 | 0 won, 0 drawn, 600 lost |
| all the same, big (1.0) | 0.217 | 0.580 | 0 won, 0 drawn, 600 lost |

**The answer is the same in the second game.** The varied initialisation
wins, the zeroed one never learns, and every initialisation that gives the
units the same value fails completely — with one interesting difference: in
Nim the failure is worse. The zeroed model gives away 0.613 a move against
a random player, where in tic-tac-toe it gives away 0.281, because a game
of Nim has more ways to be lost than a game of tic-tac-toe has.

**The models that learn Nim learn it properly.** The best of them wins every
game it starts and loses every game it does not, against a perfect player
that cannot lose: three hundred won, three hundred lost, and no draws,
because a game of Nim cannot be drawn. That is the perfect game, played
perfectly.

**The second game found a fault in the first game's perfect player.** The
perfect player for Nim had no move to offer in a position it could not win,
so the game stopped there and was counted as a draw — half of the games
against the perfect player were being counted as draws, in a game that
cannot be drawn. The perfect player now takes a stone when there is no move
that wins, which is as good as any other move in a position that is lost
anyway. The tic-tac-toe perfect player was never asked that question,
because in tic-tac-toe every position that is not won is drawn, and there
is always a move to make.

---

## 6. The answers to the questions in the notes

**Which initialisation learns the game best?** Random. It is the only one
that learns, and it wins the round robin in five runs out of five — in
tic-tac-toe and in Nim alike.

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

## 7. What is still open

**The self-play teacher.** Its target is the result of the game under the
model's own current play, so it chases a moving target. Fifty thousand
games of it take the truth error from 0.34 to 0.20 and draw every game
against the perfect player, where the perfect player's own teaching reaches
the same place after ten thousand. A bootstrapped target — the worth of the
position after the move, taken from the model's own estimates — was tried
with the old shape of the model and stalled at 60%; it has not been tried
with the current one.

**Learning from games against a fixed opponent.** The duel shows what
happens when two models spend their games on each other: they get good at
each other and no better. Whether a model could be taught by an opponent
that is itself improving — the way a league works — is untried.

**The models are narrow.** A model learns the worth of the positions it
visits. With enough games it covers the game well enough to play it
perfectly against a perfect opponent, but it is still worse than a random
player in the positions it has not seen.

**A league for the second game.** Done, and it is written up in
[LEAGUE.md](LEAGUE.md): the same hundred models in the game of Nim move
sixteen places on average between the training measure and the league, and
the training measure still cannot separate the models that were taught from
the ones that were not. What is different in Nim is the league, which does
tell the two kinds apart — because a game of Nim is never drawn, so a
hundred models cannot hide behind a hundred thousand draws.
