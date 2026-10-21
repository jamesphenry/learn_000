#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"
#include "../src/evaluate.c"
#include "../src/game.c"
#include "../src/judge.c"
#include "../src/layer.c"
#include "../src/minimax.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"

static void build_a_network(network_t *network, double scale)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 4, ACTIVATION_CLAMP);
	network_add_layer(network, 4, CELL_COUNT, ACTIVATION_SIGMOID);

	for (int neuron = 0; neuron < network->layers[0].n_neurons; neuron++) {
		for (int input = 0; input < network->layers[0].n_inputs; input++)
			network->layers[0].weights[neuron][input] = scale * (input + 1);
	}
}

/* The same network on both sides: the games are played out and the tally
 * adds up, whatever the games turn out to be. */
static char *test_a_network_against_itself_is_played_out(void)
{
	network_t network;
	match_result_t result;

	build_a_network(&network, 0.0);

	_it_should("play the match", evaluate_against(&network, &network, 1, 40, 0.0, &result));
	_it_should("count the games", result.games == 40);
	_it_should("tally every game once",
		result.first_wins + result.draws + result.second_wins + result.flagged == 40);
	_it_should("have no games flagged", result.flagged == 0);

	return NULL;
}

static char *test_the_tally_adds_up(void)
{
	network_t first;
	network_t second;
	match_result_t result;

	build_a_network(&first, 0.2);
	build_a_network(&second, 0.3);

	evaluate_against(&first, &second, 1, 30, 0.0, &result);

	_it_should("count every game once",
		result.first_wins + result.draws + result.second_wins + result.flagged == 30);
	_it_should("have no games flagged", result.flagged == 0);

	return NULL;
}

static char *test_the_two_sides_are_told_apart(void)
{
	network_t network;
	match_result_t result;

	build_a_network(&network, 0.0);

	evaluate_against(&network, &network, 1, 40, 0.0, &result);

	/* the games are played out, and however they come out, the two sides
	 * are counted separately and no game is counted twice */
	_it_should("count every game once",
		result.first_wins + result.draws + result.second_wins + result.flagged == 40);
	_it_should("leave nothing unaccounted for",
		result.games == result.first_wins + result.draws + result.second_wins +
		result.flagged);

	return NULL;
}

static char *test_it_refuses_bad_arguments(void)
{
	network_t network;
	match_result_t result;

	build_a_network(&network, 0.0);

	_it_should("refuse a null first network", !evaluate_against(NULL, &network, 1, 10, 0.0, &result));
	_it_should("refuse a null second network", !evaluate_against(&network, NULL, 1, 10, 0.0, &result));
	_it_should("refuse nowhere to put the result", !evaluate_against(&network, &network, 1, 10, 0.0, NULL));
	_it_should("refuse no games", !evaluate_against(&network, &network, 1, 0, 0.0, &result));

	return NULL;
}

static char *test_the_same_match_gives_the_same_tally(void)
{
	network_t first;
	network_t second;
	match_result_t one;
	match_result_t two;

	build_a_network(&first, 0.2);
	build_a_network(&second, 0.3);

	evaluate_against(&first, &second, 5, 30, 0.0, &one);
	evaluate_against(&first, &second, 5, 30, 0.0, &two);

	_it_should("give the same wins", one.first_wins == two.first_wins);
	_it_should("give the same draws", one.draws == two.draws);
	_it_should("give the same losses", one.second_wins == two.second_wins);

	return NULL;
}

/*
 * A network that plays the cells in a fixed order, whichever order it is
 * given. The hidden value is the sum of the marks, each counted by the
 * weight of its cell, so the cell with the highest weight leaves the best
 * position. X wants the best position and O wants the worst, and taking a
 * cell of weight w adds w for X and takes w away for O, so both of them
 * take the cell with the highest weight.
 *
 * The weights are small, so that the sum of a whole board stays above the
 * floor of the clamp.
 */
static void build_the_ordered_network(network_t *network, const int order[CELL_COUNT])
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 1, ACTIVATION_CLAMP);

	for (int place = 0; place < CELL_COUNT; place++)
		network->layers[0].weights[0][order[place]] =
			(double)(CELL_COUNT - place) / 45.0;

	network_add_layer(network, 1, 1, ACTIVATION_SIGMOID);

	/* and the worth of a position is that sum turned into a probability */
	network->layers[1].weights[0][0] = 1.0;
}

static char *test_the_match_counts_the_games_correctly(void)
{
	network_t network;
	match_result_t result;

	/* X takes the cells 0, 2, 4, 6 and 8 and O takes the cells 1, 3, 5
	 * and 7, which leaves X with the anti diagonal and the win */
	const int order[CELL_COUNT] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

	build_the_ordered_network(&network, order);

	evaluate_against(&network, &network, 1, 9, 0.0, &result);

	/* X wins every game, so the first network wins the games it plays as X
	 * and loses the ones it plays as O */
	_it_should("win the games it plays as X", result.first_wins == 5);
	_it_should("lose the games it plays as O", result.second_wins == 4);
	_it_should("have no draws", result.draws == 0);
	_it_should("have no games flagged", result.flagged == 0);
	_it_should("count every game once",
		result.first_wins + result.draws + result.second_wins + result.flagged == 9);

	/* the games it won are exactly the games it played as X */
	_it_should("play as X in half the games", result.first_as_x == 5);
	_it_should("win every game it played as X", result.first_wins_as_x == 5);

	return NULL;
}

static char *test_the_match_counts_the_draws_correctly(void)
{
	network_t network;
	match_result_t result;

	/* the cells are taken in this order, with X taking the first five and
	 * O the last four. This leaves the board
	 *
	 *   X O X
	 *   X O O
	 *   O O X
	 *
	 * which neither player has won, so every game is a draw */
	const int order[CELL_COUNT] = {0, 1, 2, 4, 3, 5, 7, 6, 8};

	build_the_ordered_network(&network, order);

	evaluate_against(&network, &network, 1, 9, 0.0, &result);

	_it_should("draw every game", result.draws == 9);
	_it_should("give neither side a win", result.first_wins == 0 && result.second_wins == 0);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_a_network_against_itself_is_played_out);
	_run_test(test_the_tally_adds_up);
	_run_test(test_the_two_sides_are_told_apart);
	_run_test(test_the_match_counts_the_games_correctly);
	_run_test(test_the_match_counts_the_draws_correctly);
	_run_test(test_it_refuses_bad_arguments);
	_run_test(test_the_same_match_gives_the_same_tally);

	return NULL;
}
