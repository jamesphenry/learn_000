#ifndef ARENA_H
#define ARENA_H

#include <stdbool.h>
#include "evaluate.h"
#include "network.h"
#include "rng.h"

/* The three ways a network can be started off. */
typedef enum {
	INIT_ZERO,   /* every weight and bias at zero */
	INIT_RANDOM, /* small random values */
	INIT_MAXIMUM /* every weight and bias at the same value */
} init_scheme_t;

/* The two ways a model can be taught. */
typedef enum {
	ARENA_LEARN_TRUTH, /* the perfect player says what each position is worth */
	ARENA_LEARN_SELF   /* the model learns the result of its own games */
} arena_learning_t;

/* The conditions every model in a run is given, so that the only thing that
 * differs between them is how they were started. */
typedef struct {
	int hidden_neurons;      /* how many neurons the hidden layer has */
	int training_games;      /* how many games to learn from */
	int evaluation_games;    /* how many games to measure with */
	unsigned long long training_seed;
	unsigned long long evaluation_seed;
	double learning_rate;
	double temperature;      /* how much the policy explores while playing */
	double init_scale;       /* how big the random values are */
	double init_value;       /* the value every weight and bias starts at */
	arena_learning_t learning; /* who teaches the model */
} arena_settings_t;

/* What a model did. */
typedef struct {
	double win_rate_before;
	double win_rate_after;
	double brier_before;
	double brier_after;
	double truth_before;
	double truth_after;
	/* the worth the model gave away, on each move it played */
	double value_lost_before;
	double value_lost_after;
	/* and how it did against the perfect player */
	perfect_result_t perfect_before;
	perfect_result_t perfect_after;
	int trained_games;
} arena_result_t;

/* The conditions a run is given, with sensible values for tic-tac-toe. */
void arena_settings_init(arena_settings_t *settings);

/*
 * Builds a network, measures it, trains it, and measures it again. The
 * games it learns from and the games it is measured with are apart from one
 * another, so the measurement is on games the model has not seen.
 */
bool arena_run(const arena_settings_t *settings, init_scheme_t scheme,
               arena_result_t *result);

#endif // ARENA_H
