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

The **training** ranking is the worth a model gave away, on each move it
played, **in the positions it visited while learning**. It is the narrowest
measurement there is: it says how well the model learned the positions it
already knew.

The **league** ranking is the points: three for a win, one for a draw, earned
against the other ninety-nine models.

The **general** ranking is a controlled test. Four models are held back from
the league and never play in it — one of each kind, each trained to play in
two different ways — and every model is measured against them. What it
earns against the held-back models of the *other* kind is its score. The
opponents are the same for everybody, so this is a fair comparison, which
the league's own table is not: the points a model earns against the other
kind in the league depend on which models happened to be in the league and
how strong they were.

## How it was run

A hundred models, seeds 1 to 100. Each learned from twenty thousand games
and was then measured. Then every pair played two hundred games: 4,950
matches and 990,000 games in all. Then every model played the four
held-back models, 200 games each.

Reproduce it with:

```sh
./src/learn_000 100
```

which builds the league and prints the three rankings. The league itself is
`arena_league` in `src/arena.c`.

## The results

```
the average model moves 27.0 places when the league decides instead of the training
and 20.6 places when the models of the other kind decide

the leader of each ranking
  ranking        seed   taught   training     league
  training          1    truth      0.000      21262
  league           98     self      0.000      23627
  general           1    truth      0.000      21262

the champion is model 97, with the seed 98, taught by its own games
  2006 won, 17609 drawn, 185 lost, 23627 points
  place by training: 65, by league: 1, by generalisation: 78, of 100
```

**The average model moves twenty-seven places** between the training
ranking and the league ranking. The champion was **65th** by the training
measure. The model that topped the training measure came **34th** in the
league.

And the champion is a model that taught *itself*.

## What it says

### The training measure cannot rank the models it ought to

Forty-seven of the fifty models taught by the perfect player give away
**nothing at all** in their own games — they play their own lines
perfectly. Their training measure is exactly zero, all forty-seven of
them, and the measure cannot separate them by so much as a hair. Nineteen
of the fifty self-taught models are tied at zero too.

The league separates all hundred: the truth-taught models scored between
20,056 and 23,256 points, the self-taught ones between 16,124 and 23,627.
The champion is a self-taught model on 23,627.

### The two kinds are not told apart by the league

The truth-taught models took places 2 to 69 and the self-taught ones 1 to
100. The champion came from the kind that was taught by nobody.

That is the surprise. It is tempting to think the models taught by the
perfect player are the better ones, and on the training measure they are:
every one of the top three is truth-taught. But the league does not see it
that way, and the reason is in the next section.

### Where the training measure does say something, the league disagrees

The fifty models that learned from their own games are separated by the
training measure, and the league reorders them by **31.7 places** on
average:

| seed | by training | by the league | moved |
|---|---|---|---|
| 6 | 5 | 99 | down 94 |
| 8 | 7 | 93 | down 86 |
| 20 | 17 | 95 | down 78 |
| 68 | 76 | 4 | up 72 |
| 72 | 93 | 22 | up 71 |
| 98 | 65 | 1 | up 64 |

Seed 6 gave away almost nothing while learning and finished ninety-ninth
of a hundred against opponents it had never met. Seed 68 was seventy-sixth
by the training measure and finished fourth.

Here the training measure really is telling the truth about something —
these models are properly separated by it — and the league still disagrees
with it about most of them.

### The controlled test and the league do not agree either

| | by the league | by the controlled test |
|---|---|---|
| seed 98 | **1st** | 78th |
| seed 1 | 34th | **1st** |
| seed 6 | 99th | 89th |
| seed 81 | 63rd | **5th** |
| seed 12 | 57th | 100th |

The controlled test and the training measure agree with each other — both
favour seed 1 — and the league disagrees with both. The champion is
seventy-eighth on the controlled test.

This is worth being careful about, because each measure is right about
something. The controlled test is a fair comparison, but it is narrow: with
the models this alike, only **five** of the truth-taught models took a
different score against the held-back models of the other kind, and eight
of the self-taught ones. A hundred models that all draw with each other
cannot be told apart by one opponent.

The league asks a different question — not *can this model handle that
one opponent* but *can it handle ninety-nine unfamiliar ones* — and it
gives a different answer, because the two questions are different.

### The thing worth taking away

A model that dominates its training opponents is not thereby good, and the
gap between the two is not a small one: twenty-seven places on average, and
the champion was sixty-fifth on the measure that says how well a model
learned.

Seed 1 gave away nothing at all in the games it trained on — it could not
lose to itself — and it finished thirty-fourth of a hundred against
opponents it had never met. Seed 6, fifth on that measure, finished
ninety-ninth.

This is the same finding as the narrow measurement in
[FINDINGS.md](FINDINGS.md), arrived at from the other direction: a model
that only practises against one thing learns that one thing. Measuring a
model on its own games tells you how well it learned the positions it
already knew, and nothing else.

The three rankings are kept apart on purpose. The training measure says
what happened while learning; the league says what the model can do
against a variety of strangers; the controlled test says whether it can do
it against one unfamiliar opponent of a kind it has never met. Only one of
the three is a measurement of the model, and it is not the one that is
easiest to take.
