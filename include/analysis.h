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

#endif // ANALYSIS_H
