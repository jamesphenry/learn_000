three controlled experiments:

Model A — Zero initialization

Every weight and bias starts at 0.0.

The blank slate. It also gives us a useful test of whether our training method can break symmetry; identical neurons initialized identically may learn identical things.




Model B — Random initialization

Small random values for weights and biases.

A conventional starting point. Different neurons begin with different behaviors, giving them a chance to learn different features.




Model C — Maximum-value initialization

Every weight and bias starts at a chosen upper bound.

The deliberately extreme case. With ReLU, large positive sums can produce large activations; depending on the training method, that can make learning unstable or unhelpful.



The important part: make it a fair fight
When the time comes, we'll give all three models:
- The same network architecture.
- The same training data or self-play conditions.
- The same training algorithm and number of training steps.
- The same evaluation set, with separate games they haven't trained on.
Then we'll measure performance before and after training, not just the final win rate.

One small detail to decide when we get there: “maximum value” needs a defined number. A weight of 1.0 and a weight of 100.0 are very different experiments.

And I especially like that the zeroed model is included. It may reveal something fundamental about how our training algorithm works, rather than just which initialization happens to win.

Let's keep this experiment on the roadmap. No need to build the experiment harness yet; we'll get the network working first, then give the three models their own little training arena.


# End Goal
- seperate tic-tac-toe, make this a framework, with games/tic-tac-toe.
- keep expanding with other games.

#Future layout???
```
game-learning/
├── include/             # Public framework interfaces
├── src/                 # Shared implementation
├── tests/               # Framework tests
├── games/
│   └── tic-tac-toe/
│       ├── include/
│       ├── src/
│       └── tests/
├── experiments/
│   └── initialization/  # Zero vs random vs maximum
├── artifacts/           # Models and training results
└── README.md
```