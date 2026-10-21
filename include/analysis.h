#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <stdbool.h>
#include "arena.h"
#include "game.h"
#include "network.h"

/* How many positions a fixed set of them may hold. */
#define ANALYSIS_MAX_POSITIONS 512

/*
 * What a model knows about a fixed set of positions: the same positions for
 * every model, so that two models are measured on the same ground.
 *
 * The error is exact, so it does not wander from seed to seed: it is the
 * same number for the same weights on every machine.
 */
typedef struct {
	/* how many positions were measured */
	int positions;

	/* how far the worths are from the truth, over every position */
	double truth_error;

	/* the mean of the worths the model gives, and the truth's mean */
	double mean_worth;
	double mean_truth;

	/* the mean worth it gives for the positions that are a win, and for the
	 * ones that are a loss */
	double win_worth;
	double loss_worth;

	/* the positions it calls a win, and the positions that are wins */
	int calls_win;
	int truth_wins;

	/* the moves it played, how many of them were as good as the best one
	 * there was, and how many different moves they were */
	int moves;
	int best_moves;
	int distinct_moves;

	/* the worth it gave away, over all the moves */
	double value_lost;

	/* the positions the player to move wins, and the moves that left the
	 * opponent in a losing position */
	int winning_positions;
	int winning_moves;
} analysis_t;

/*
 * Measures a model over the fixed set of positions the game offers.
 *
 * A game that cannot say which positions there are has no fixed set, and
 * this says so.
 *
 * The seed is the one the policy breaks its ties with, so that the same model
 * is measured the same way every time.
 */
bool analysis_fixed_set(const game_t *game, const oracle_t *oracle,
                         const network_t *network, unsigned long long seed,
                         analysis_t *result);

/* How many budgets a curve may hold, and how many seeds each of them is
 * measured with. */
#define ANALYSIS_MAX_BUDGETS 8
#define ANALYSIS_MAX_SEEDS 8

/*
 * How much a model moved while it was learning: the mean change in its
 * weights and its biases.
 */
typedef struct {
	/* the mean change in the hidden layer */
	double hidden;

	/* and in the output layer */
	double output;

	/* and in all of them together */
	double all;
} movement_t;

/*
 * How much a model moved between two of its states.
 */
bool analysis_movement(const network_t *before, const network_t *after,
                       movement_t *result);

/*
 * The curve of learning: how a model does on the fixed set as it learns more.
 *
 * Every budget is given to a model of its own, and every model is measured
 * on the same positions, so that the curve is the same ground all the way
 * along it.
 */
typedef struct {
	/* how many budgets there are, and how many games each of them is */
	int budgets;
	int budget[ANALYSIS_MAX_BUDGETS];

	/* how many seeds each budget was measured with */
	int seeds;

	/* the measurement for each budget and each seed */
	analysis_t measure[ANALYSIS_MAX_BUDGETS][ANALYSIS_MAX_SEEDS];

	/* how much the weights moved while each model was learning */
	movement_t movement[ANALYSIS_MAX_BUDGETS][ANALYSIS_MAX_SEEDS];
} analysis_curve_t;

/*
 * The curve of learning, on the fixed set of positions the game offers.
 *
 * Every model starts off in the way that is asked for, and the seed in the
 * settings is the first of the seeds the curve is measured with: each seed is
 * the one before it with one added to it.
 */
bool analysis_curve(const arena_settings_t *settings, const game_t *game,
                    const oracle_t *oracle, init_scheme_t scheme,
                    const int *budgets, int budgets_count, int seeds,
                    analysis_curve_t *curve);

#endif // ANALYSIS_H
