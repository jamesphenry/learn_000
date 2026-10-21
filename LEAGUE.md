# A league of a hundred models

A hundred models, each with a seed of its own, every one of them playing
every other one. The points are added up and the champion is crowned.

Half of the models are taught by the perfect player, which says what each
position is worth. Half learn from the result of the games they play
against themselves, with nobody to tell them the answers. So the league
holds two kinds of opponent, and a model has to be able to handle an
opponent it has never met.

The point of the league is that it produces **three rankings that do not
agree**, and that each of them answers a different question.

| ranking | what it tells you |
|---|---|
| **training** | how well did the model learn from its training experience? |
| **league** | how well does it perform against unfamiliar opponents? |
| **general** | does its performance hold up against a different kind of model? |

The training ranking is the worth a model gave away, on each move it
played, **in the positions it visited while learning**. It is the narrowest
measurement there is: it says how well the model learned the positions it
already knew.

The league ranking is the points: three for a win, one for a draw, earned
against the other ninety-nine models.

The general ranking is the points against the models of the *other* kind —
the only one of the three that says whether a model can handle an opponent
whose games are not like its own.

## How it was run

A hundred models, seeds 1 to 100. Each learned from twenty thousand games
and was then measured. Then every pair played two hundred games: 4,950
matches and 990,000 games in all.

Reproduce it with:

```sh
./src/learn_000 100
```

which builds the league and prints the three rankings. The league itself is
`arena_league` in `src/arena.c`.

## The results

```
the average model moves 12.1 places when the league decides instead of the training
and 13.9 places when the models of the other kind decide

the leader of each ranking
  ranking        seed   taught   training     league
  training          9    truth      0.000      29378
  league           59    truth      0.000      32764
  general          59    truth      0.000      32764

the champion is model 58, with the seed 59, taught by the perfect player
  6652 won, 12808 drawn, 340 lost, 32764 points
  place by training: 11, by league: 1, by generalisation: 1, of 100
```

The champion was **eleventh** by the training measure and **first** by the
league. The model that topped the training measure, seed 9, came **32nd**
in the league.

## What it says

### The training measure cannot rank the models it ought to

Twenty of the fifty models taught by the perfect player give away
**nothing at all** in their own games — they play their own lines
perfectly. Their training measure is exactly zero, all twenty of them, and
the measure cannot separate them by so much as a hair.

The league separates all fifty, from the champion on 32,764 points to the
last on 23,040. It took ninety-nine opponents and a million games to say
what twenty thousand games of training could not.

### Where the training measure does say something, the league disagrees

The fifty models that learned from their own games are separated by the
training measure, which runs from 0.061 to 0.444 of worth given away a
move. And the league reorders them:

| seed | by training | by the league | moved |
|---|---|---|---|
| 38 | 46 | 95 | down 49 |
| 84 | 56 | 88 | down 32 |
| 2 | 61 | 91 | down 30 |
| 40 | 82 | 58 | up 24 |
| 12 | 96 | 67 | up 29 |
| 46 | 99 | 59 | up 40 |

The average of the fifty moves **12.6 places**. The model that gave away
the least worth while learning finished forty-second of the fifty in the
league; the model that finished forty-sixth on that measure finished
ninety-fifth.

Here the training measure really is telling the truth about something —
these fifty models are properly separated by it — and the league still
disagrees with it about most of them.

(The fifty truth-taught models move an average of 11.7 places too, but
that number means much less: twenty of them are tied at zero, so their
places in the training ranking are arbitrary and the league has nothing to
disagree with.)

### The two kinds are not interchangeable

A model taught by the perfect player and a model that learned from its own
games are not opponents of the same kind, and the league says so plainly.
The truth-taught models took the first fifty-one places and the
self-taught ones the fiftieth to the hundredth, and the points say why —
the truth-taught models scored between 23,040 and 32,764, the self-taught
ones between 13,238 and 24,668.

One place is given away at the boundary: a self-taught model finished
fiftieth and a truth-taught model fifty-first, so even here the two kinds
are not cleanly separated.

Among the self-taught models the best by the training measure was seed 50
and the best in the league was also seed 50. Among the truth-taught
models the best by training was seed 9 and the best in the league was
seed 59, and the best of them against models of the other kind was
seed 61.

## The thing worth taking away

A model that dominates its training opponents is not thereby good, and the
gap between the two is not a small one. Seed 9 gave away nothing at all in
the games it trained on — it could not lose to itself — and it finished
32nd of a hundred against opponents it had never seen.

This is the same finding as the narrow measurement in
[FINDINGS.md](FINDINGS.md), arrived at from the other direction: a model
that only practises against one thing learns that one thing. Measuring a
model on its own games tells you how well it learned the positions it
already knew, and nothing else.

The three rankings are kept apart on purpose. The training measure says
what happened while learning; the league says what the model can do; the
third says whether it can do it against something unfamiliar. Only one of
the three is a measurement of the model, and it is not the one that is
easiest to take.
