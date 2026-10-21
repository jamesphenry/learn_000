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
	/* the moves that were played in the games X won and in the games that
	 * were drawn */
	int moves_in_x_wins;
	int moves_in_draws;
	/* and the moves the player to move went on to win, which is what the
	 * worth of a position is measured against */
	int moves_won;
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
 * What a network looks like before it has learned anything.
 *
 * The sums the hidden layer sees, the sum the output layer sees, and what
 * the network therefore says. The last two numbers are the pull of the loss
 * on the network: how hard it would be to learn that a position is a win,
 * and how hard it would be to learn that it is a loss. A network whose
 * output is pinned at one cannot learn from a win at all.
 */
typedef struct {
	double hidden_lowest;
	double hidden_highest;
	double output_lowest;
	double output_highest;
	double worth;           /* what it says about an empty board */
	double gradient_to_one;  /* how hard it would be to learn that X wins */
	double gradient_to_zero; /* and how hard it would be to learn that O wins */
} initialisation_t;

/*
 * How a match between two networks went.
 */
typedef struct {
	int games;
	int first_wins;
	int draws;
	int second_wins;
	int flagged;  /* the games the referee did not accept */
	/* how many of the games the first network played as X, and how many it
	 * won in them */
	int first_as_x;
	int first_wins_as_x;
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
	/* how many of the games the network took each colour in, and how many
	 * it won in each */
	int as_x;
	int as_o;
	int wins_as_x;
	int wins_as_o;
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
 * Plays games and says how much value the network gave away: for every
 * move it played, how much the position was worth less than the best move
 * there was. This is exact, so unlike a win rate it does not wander from
 * seed to seed.
 *
 * The opponent plays at the given temperature. At zero it is the network
 * playing its own game, which measures the network on the positions it
 * knows best. At an infinite temperature it plays at random, which reaches
 * the positions the network would not have reached on its own.
 */
bool evaluate_skill(const network_t *network, unsigned long long seed, int games,
                    double opponent_temperature, skill_t *result);

/*
 * Plays against the perfect player and says how it did.
 */
bool evaluate_against_perfect(const network_t *network, unsigned long long seed,
                              int games, perfect_result_t *result);

/*
 * Looks at a network before it has learned anything, and says what it looks
 * like.
 */
bool evaluate_initialisation(const network_t *network, initialisation_t *result);

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
