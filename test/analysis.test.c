#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/analysis.c"
#include "../src/game.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../games/nim/src/nim.c"

static game_t game;
static oracle_t oracle;

/*
 * A network that says the same thing about every position: a half. Its
 * hidden layer sees nothing and its output is given a bias of nought, so the
 * sigmoid gives a half.
 */
static void build_a_flat_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, NIM_HEAPS, 4, ACTIVATION_RELU);
	network_add_layer(network, 4, 1, ACTIVATION_SIGMOID);
}

static bool nearly_within(double actual, double expected, double tolerance)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < tolerance;
}

static bool nearly_equal(double actual, double expected)
{
	return nearly_within(actual, expected, 1e-9);
}

static char *test_it_refuses_bad_arguments(void)
{
	network_t network;
	analysis_t result;

	build_a_flat_network(&network);

	_it_should("refuse no game",
		!analysis_fixed_set(NULL, &oracle, &network, 1, &result));
	_it_should("refuse no perfect player",
		!analysis_fixed_set(&game, NULL, &network, 1, &result));
	_it_should("refuse no model",
		!analysis_fixed_set(&game, &oracle, NULL, 1, &result));
	_it_should("refuse nowhere to put the result",
		!analysis_fixed_set(&game, &oracle, &network, 1, NULL));

	return NULL;
}

/*
 * A game that cannot say which positions there is has no fixed set to be
 * measured on.
 */
static char *test_a_game_without_a_fixed_set_has_none(void)
{
	network_t network;
	analysis_t result;
	game_t without;

	build_a_flat_network(&network);

	nim_game(&without);
	without.positions = NULL;

	_it_should("say so", !analysis_fixed_set(&without, &oracle, &network, 1, &result));

	return NULL;
}

/*
 * A model that says a half about everything is half way between the truth
 * whatever the truth is, so its error is a half over every position there is.
 */
static char *test_a_model_that_says_nothing_is_half_way_off(void)
{
	network_t network;
	analysis_t result;

	build_a_flat_network(&network);

	_it_should("measure it", analysis_fixed_set(&game, &oracle, &network, 1, &result));

	/* every position the game can reach */
	_it_should("measure every position there is", result.positions == NIM_MAX_POSITIONS);
	_it_should("be half way off", nearly_equal(result.truth_error, 0.5));
	_it_should("say a half everywhere", nearly_equal(result.mean_worth, 0.5));

	/* the truth is a win for the player to move in 102 of the 120
	 * positions, and a loss in the other 18 */
	_it_should("find the wins the truth has", result.truth_wins == 102);
	_it_should("call none of them a win", result.calls_win == 0);

	/* and it gives a half for the wins as well as for the losses */
	_it_should("say a half about the wins", nearly_equal(result.win_worth, 0.5));
	_it_should("say a half about the losses", nearly_equal(result.loss_worth, 0.5));

	return NULL;
}

/*
 * The moves: every position that is not over has a move to play, and the
 * position with no stones in it has none.
 */
static char *test_it_measures_the_moves_too(void)
{
	network_t network;
	analysis_t result;

	build_a_flat_network(&network);

	analysis_fixed_set(&game, &oracle, &network, 1, &result);

	_it_should("play a move in every position that is not over",
		result.moves == NIM_MAX_POSITIONS - 1);
	_it_should("never play a better move than the best there is",
		result.best_moves <= result.moves);
	_it_should("play more than one kind of move", result.distinct_moves > 1);

	/* the player to move wins in 102 of the positions, and a move that
	 * leaves the opponent in a losing position is one of them */
	_it_should("count the winning positions", result.winning_positions == 102);
	_it_should("find no more winning moves than winning positions",
		result.winning_moves <= result.winning_positions);

	/* every position that is a loss is lost whatever the model plays, so
	 * the only worth it can give away is in the positions that are a win */
	_it_should("give away only what was there to win",
		nearly_equal(result.value_lost,
		             (double)(result.winning_positions - result.winning_moves)));

	return NULL;
}

/*
 * A model that says nought about every position is wrong by the whole of the
 * winning positions, which are 102 of the 120.
 */
static char *test_a_model_that_calls_everything_a_loss_is_wrong_by_the_wins(void)
{
	network_t network;
	analysis_t result;

	network_init(&network);
	network_add_layer(&network, NIM_HEAPS, 1, ACTIVATION_RELU);
	network_add_layer(&network, 1, 1, ACTIVATION_SIGMOID);

	/* the output is pushed far down, so the sigmoid says nought */
	network.layers[1].biases[0] = -20.0;

	analysis_fixed_set(&game, &oracle, &network, 1, &result);

	_it_should("be wrong by the winning positions",
		nearly_within(result.truth_error, 102.0 / 120.0, 1e-6));
	_it_should("call nothing a win", result.calls_win == 0);
	_it_should("find the wins the truth has", result.truth_wins == 102);

	return NULL;
}

static char *run_tests(void)
{
	nim_game(&game);
	nim_oracle(&oracle);

	_run_test(test_it_refuses_bad_arguments);
	_run_test(test_a_game_without_a_fixed_set_has_none);
	_run_test(test_a_model_that_says_nothing_is_half_way_off);
	_run_test(test_a_model_that_calls_everything_a_loss_is_wrong_by_the_wins);
	_run_test(test_it_measures_the_moves_too);

	return NULL;
}
