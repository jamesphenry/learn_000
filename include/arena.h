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

/* How many ways there are to start a network off. */
#define ARENA_SCHEMES 3

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
	int match_games;         /* how many games the models play each other */
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

/*
 * What all the models did, and how they fare against each other.
 */
typedef struct {
	/* the seed the models were trained with */
	unsigned long long training_seed;

	/* what each model did on its own, in the order of the ways of
	 * starting off */
	arena_result_t results[ARENA_SCHEMES];

	/* how the models fare against each other. The match at [i][j] is the
	 * one between the models started off in the ways i and j, and only the
	 * upper half is filled in: a model does not play itself. */
	match_result_t matches[ARENA_SCHEMES][ARENA_SCHEMES];
} arena_report_t;

/* How many runs a survey holds. */
#define ARENA_MAX_RUNS 10

/*
 * What the models did over a number of runs, each with its own seed.
 */
typedef struct {
	int runs;

	/* what each run found */
	arena_report_t reports[ARENA_MAX_RUNS];

	/* how many of the runs each model won the round robin in */
	int round_robin_wins[ARENA_SCHEMES];

	/* and the games it won and lost against the other models, over all the
	 * runs */
	int games_won[ARENA_SCHEMES];
	int games_lost[ARENA_SCHEMES];

	/* and the games each model drew and lost against the perfect player,
	 * over all the runs */
	int perfect_draws[ARENA_SCHEMES];
	int perfect_losses[ARENA_SCHEMES];
} arena_survey_t;

/* The conditions a run is given, with sensible values for tic-tac-toe. */
void arena_settings_init(arena_settings_t *settings);

/*
 * Builds a network, measures it, trains it, and measures it again. The
 * games it learns from and the games it is measured with are apart from one
 * another, so the measurement is on games the model has not seen.
 */
bool arena_run(const arena_settings_t *settings, init_scheme_t scheme,
               arena_result_t *result);

/*
 * Builds all three models, gives them the same conditions, measures each of
 * them, and then has them play each other.
 */
bool arena_run_all(const arena_settings_t *settings, arena_report_t *report);

/*
 * Runs the arena several times, each time with a seed of its own, and says
 * what the models did over all of them. A ranking that holds from one seed
 * to the next is worth more than one that was measured once.
 */
bool arena_survey(const arena_settings_t *settings, int runs, arena_survey_t *survey);

#endif // ARENA_H
