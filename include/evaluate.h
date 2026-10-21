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
 * How a match between two networks went.
 */
typedef struct {
	int games;
	int first_wins;
	int draws;
	int second_wins;
	int flagged;  /* the games the referee did not accept */
} match_result_t;

/*
 * How much value a network gave away.
 *
 * Every position is worth something to the player to move: a win, a draw or
 * a loss. The best move there is keeps that worth; any other move leaves
 * the position worth less. This counts the moves that kept the worth and
 * the moves that gave it away.
 */
typedef struct {
	int moves;        /* the moves that were played */
	int kept;         /* the moves that left the position worth as much as the best */
	int gave_away;    /* the moves that left it worth less */
	int threw_wins;   /* the moves that gave away a win that was there */
	double value_lost; /* the worth given away, over all the moves */
} skill_t;

/*
 * How a network did against the perfect player, which never makes a
 * mistake. The network takes one colour and then the other.
 */
typedef struct {
	int games;
	int wins;
	int draws;
	int losses;
} perfect_result_t;

/*
 * Plays two networks against each other, with each of them taking one
 * colour, and says how the first one did. Every game is put past the
 * referee, so a model that played a move that was not allowed is counted
 * and the game is left out of the tally.
 */
bool evaluate_against(const network_t *first, const network_t *second,
                      unsigned long long seed, int games, double temperature,
                      match_result_t *result);

/*
 * Plays games against itself and says how much value it gave away: for
 * every move it played, how much the position was worth less than the
 * best move there was. This is exact, so unlike a win rate it does not
 * wander from seed to seed.
 */
bool evaluate_skill(const network_t *network, unsigned long long seed, int games,
                    skill_t *result);

/*
 * Plays against the perfect player and says how it did.
 */
bool evaluate_against_perfect(const network_t *network, unsigned long long seed,
                              int games, perfect_result_t *result);

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
