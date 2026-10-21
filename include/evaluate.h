#ifndef EVALUATE_H
#define EVALUATE_H

#include <stdbool.h>
#include "board.h"
#include "network.h"

// The probabilities are grouped into this many bands.
#define EVALUATION_BUCKETS 5

/*
 * How a network did against an opponent that plays at random, and how
 * honest its probabilities were.
 */
typedef struct {
	int games;
	int wins;
	int draws;
	int losses;
	/* how many of the games the network took each colour in, and how many
	 * it won in each */
	int as_x;
	int as_o;
	int wins_as_x;
	int wins_as_o;
	/* how many of the games X won, whoever was playing. The network
	 * answers with the chance that X wins, so this is what its
	 * probabilities are measured against. */
	int x_wins;
	/* the moves that were played, which is how many the probabilities
	 * were measured over */
	int moves;
	/* the moves that were played in the games X won, which is what the
	 * probabilities are measured against */
	int moves_in_x_wins;
	/* How far the probabilities were from what happened, over every move
	 * that was played: the mean of (probability - result) squared. */
	double brier_score;
	/* How far the probabilities are from what the perfect player knows the
	 * position to be worth. This is not the same as the brier score: the
	 * brier score asks whether the probabilities are honest about the games
	 * that were played, and this asks whether they are right about the
	 * position. It is exact, so it does not wander from seed to seed. */
	double truth_error;
	int bucket_counts[EVALUATION_BUCKETS];
	double bucket_predicted[EVALUATION_BUCKETS];
	double bucket_observed[EVALUATION_BUCKETS];
} evaluation_t;

/*
 * Plays games against an opponent that plays at random, with the network
 * taking one colour and then the other, and measures how the network did.
 *
 * The buckets hold the same thing in groups, so a network that says 0.7
 * can be held against how often that turned out to be right.
 */
bool evaluate_network(const network_t *network, unsigned long long seed, int games,
                      evaluation_t *result);

#endif // EVALUATE_H
