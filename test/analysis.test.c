#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/analysis.c"
#include "../src/arena.c"
#include "../src/evaluate.c"
#include "../src/game.c"
#include "../src/judge.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../src/trainer.c"
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

/*
 * The curve of learning: every budget is measured on the same positions, and
 * the first point of the curve is the model the arena would teach with the
 * same seed.
 */
static char *test_the_curve_measures_the_same_ground_all_the_way_along(void)
{
	arena_settings_t settings;
	analysis_curve_t curve;
	analysis_t direct;
	network_t network;
	rng_t rng;
	int budgets[2] = { 0, 10 };

	arena_settings_init(&settings);
	settings.hidden_neurons = 4;
	settings.training_games = 0;

	_it_should("measure the curve",
		analysis_curve(&settings, &game, &oracle, INIT_RANDOM, budgets, 2, 1, &curve));

	_it_should("hold both budgets", curve.budgets == 2 && curve.budget[0] == 0 &&
	                                      curve.budget[1] == 10);
	_it_should("measure every position there is",
		curve.measure[0][0].positions == NIM_MAX_POSITIONS);

	/* and the model at the start of the curve is the model the arena
	 * teaches with the same seed */
	rng_seed(&rng, settings.training_seed);
	arena_train(&settings, &game, &oracle, INIT_RANDOM, &rng, &network, NULL);
	analysis_fixed_set(&game, &oracle, &network, settings.evaluation_seed, &direct);

	_it_should("give the same error",
		nearly_equal(curve.measure[0][0].truth_error, direct.truth_error));
	_it_should("give the same mean worth",
		nearly_equal(curve.measure[0][0].mean_worth, direct.mean_worth));
	_it_should("play the same number of moves",
		curve.measure[0][0].moves == direct.moves);

	return NULL;
}

/*
 * How far a model moved: a model that has one weight changed by one, and
 * nothing else, has four numbers in it, so the movement is a quarter.
 */
static char *test_it_measures_how_far_a_model_moved(void)
{
	network_t before;
	network_t after;
	movement_t moved;

	network_init(&before);
	network_add_layer(&before, 1, 1, ACTIVATION_RELU);
	network_add_layer(&before, 1, 1, ACTIVATION_SIGMOID);

	after = before;

	_it_should("say that a model that has not moved has not moved",
		analysis_movement(&before, &after, &moved) && moved.all == 0.0 &&
		moved.hidden == 0.0 && moved.output == 0.0);

	after.layers[0].weights[0][0] += 1.0;

	_it_should("measure the change in the hidden layer",
		analysis_movement(&before, &after, &moved) && nearly_equal(moved.hidden, 0.5));
	_it_should("leave the output layer alone",
		nearly_equal(moved.output, 0.0));
	_it_should("measure the change over all of it",
		nearly_equal(moved.all, 0.25));

	return NULL;
}

/*
 * Two models that are not the same shape cannot be compared.
 */
static char *test_it_refuses_to_compare_models_of_different_shapes(void)
{
	network_t before;
	network_t after;
	movement_t moved;

	network_init(&before);
	network_add_layer(&before, 1, 1, ACTIVATION_RELU);
	network_add_layer(&before, 1, 1, ACTIVATION_SIGMOID);

	network_init(&after);
	network_add_layer(&after, 2, 1, ACTIVATION_RELU);
	network_add_layer(&after, 1, 1, ACTIVATION_SIGMOID);

	_it_should("refuse no model before",
		!analysis_movement(NULL, &after, &moved));
	_it_should("refuse no model after",
		!analysis_movement(&before, NULL, &moved));
	_it_should("refuse nowhere to put the result",
		!analysis_movement(&before, &after, NULL));
	_it_should("refuse two models of different shapes",
		!analysis_movement(&before, &after, &moved));

	return NULL;
}

/*
 * The curve keeps the movement of the weights of every model it measured: a
 * model that has not played a game has not moved.
 */
static char *test_the_curve_keeps_how_far_the_models_moved(void)
{
	arena_settings_t settings;
	analysis_curve_t curve;
	int budgets[2] = { 0, 10 };

	arena_settings_init(&settings);
	settings.hidden_neurons = 4;

	_it_should("measure the curve",
		analysis_curve(&settings, &game, &oracle, INIT_RANDOM, budgets, 2, 1, &curve));

	_it_should("say that a model that has not played has not moved",
		curve.movement[0][0].all == 0.0);
	_it_should("say that a model that has played has moved",
		curve.movement[1][0].all > 0.0);

	return NULL;
}

static char *test_the_curve_refuses_bad_arguments(void)
{
	arena_settings_t settings;
	analysis_curve_t curve;
	int budgets[2] = { 0, 10 };

	arena_settings_init(&settings);

	_it_should("refuse no settings",
		!analysis_curve(NULL, &game, &oracle, INIT_RANDOM, budgets, 2, 1, &curve));
	_it_should("refuse no game",
		!analysis_curve(&settings, NULL, &oracle, INIT_RANDOM, budgets, 2, 1, &curve));
	_it_should("refuse no perfect player",
		!analysis_curve(&settings, &game, NULL, INIT_RANDOM, budgets, 2, 1, &curve));
	_it_should("refuse no budgets",
		!analysis_curve(&settings, &game, &oracle, INIT_RANDOM, NULL, 2, 1, &curve));
	_it_should("refuse nowhere to put the curve",
		!analysis_curve(&settings, &game, &oracle, INIT_RANDOM, budgets, 2, 1, NULL));
	_it_should("refuse no seeds",
		!analysis_curve(&settings, &game, &oracle, INIT_RANDOM, budgets, 2, 0, &curve));
	_it_should("refuse more budgets than it can hold",
		!analysis_curve(&settings, &game, &oracle, INIT_RANDOM, budgets,
		ANALYSIS_MAX_BUDGETS + 1, 1, &curve));

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
	_run_test(test_it_measures_how_far_a_model_moved);
	_run_test(test_it_refuses_to_compare_models_of_different_shapes);
	_run_test(test_the_curve_keeps_how_far_the_models_moved);
	_run_test(test_the_curve_measures_the_same_ground_all_the_way_along);
	_run_test(test_the_curve_refuses_bad_arguments);

	return NULL;
}
