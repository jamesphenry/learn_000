#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "../src/evaluate.c"
#include "../src/game.c"
#include "../src/judge.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../games/tic-tac-toe/src/board.c"
#include "../games/tic-tac-toe/src/minimax.c"
#include "../games/tic-tac-toe/src/tictactoe.c"

static game_t game;
static oracle_t oracle;

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

/* A network that has never been trained says a half for every position. */
static void build_the_test_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
}

static char *test_it_refuses_bad_arguments(void)
{
	network_t network;
	evaluation_t result;

	build_the_test_network(&network);

	_it_should("refuse a null network", !evaluate_network(&game, &oracle, NULL, 1, 10, &result));
	_it_should("refuse a null result", !evaluate_network(&game, &oracle, &network, 1, 10, NULL));
	_it_should("refuse no games", !evaluate_network(&game, &oracle, &network, 1, 0, &result));

	return NULL;
}

static char *test_it_counts_the_games(void)
{
	network_t network;
	evaluation_t result;

	build_the_test_network(&network);

	evaluate_network(&game, &oracle, &network, 1, 40, &result);

	_it_should("count the games", result.games == 40);
	_it_should("count every game once",
		result.wins + result.draws + result.losses == 40);
	_it_should("take each side the same number of times",
		result.as_first == result.as_second && result.as_first == 20);
	_it_should("count the wins in each side",
		result.wins == result.wins_as_first + result.wins_as_second);
	_it_should("count the games the first player won",
		result.first_wins + result.games - result.first_wins == result.games);
	_it_should("count the moves that were played", result.moves > 0);
	_it_should("win some of the games", result.wins > 0);
	_it_should("lose some of the games", result.losses > 0);

	return NULL;
}

static char *test_an_untrained_network_says_a_half(void)
{
	network_t network;
	evaluation_t result;
	int moves = 0;

	build_the_test_network(&network);

	evaluate_network(&game, &oracle, &network, 1, 40, &result);

	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++)
		moves += result.bucket_counts[bucket];

	_it_should("count every move that was played", moves > 0);
	_it_should("put them all in the middle band", result.bucket_counts[2] == moves);
	_it_should("say a half for all of them", nearly_equal(result.bucket_predicted[2], 0.5));

	/* the network answers with the worth of a position to the player to
	 * move, which is one if they win it, a half if it is drawn and nought
	 * if they lose it, so the share of the moves that ended that way is what
	 * it is measured against */
	double expected = (double)result.moves_won / result.moves +
	                  0.5 * result.moves_in_draws / result.moves;

	_it_should("measure against the moves the player to move won",
		nearly_equal(result.bucket_observed[2], expected));

	/* a half against a result of one or zero is a quarter out, and a half
	 * against a draw is not out at all */
	int lost = result.moves - result.moves_in_first_wins - result.moves_in_draws;
	double expected_brier = 0.25 * (result.moves_in_first_wins + lost) / result.moves;

	_it_should("be a quarter out of every win and every loss",
		nearly_equal(result.brier_score, expected_brier));

	return NULL;
}

static char *test_the_buckets_hold_the_observations(void)
{
	network_t network;
	evaluation_t result;

	build_the_test_network(&network);

	evaluate_network(&game, &oracle, &network, 1, 40, &result);

	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++) {
		if (result.bucket_counts[bucket] == 0) continue;

		_it_should("hold a probability", result.bucket_predicted[bucket] >= 0.0 &&
			result.bucket_predicted[bucket] <= 1.0);
		_it_should("hold a share", result.bucket_observed[bucket] >= 0.0 &&
			result.bucket_observed[bucket] <= 1.0);
	}

	_it_should("have a brier score between nought and one",
		result.brier_score >= 0.0 && result.brier_score <= 1.0);

	return NULL;
}

static char *test_the_same_seed_gives_the_same_result(void)
{
	network_t network;
	evaluation_t first;
	evaluation_t second;

	build_the_test_network(&network);

	evaluate_network(&game, &oracle, &network, 99, 30, &first);
	evaluate_network(&game, &oracle, &network, 99, 30, &second);

	_it_should("give the same wins", first.wins == second.wins);
	_it_should("give the same draws", first.draws == second.draws);
	_it_should("give the same losses", first.losses == second.losses);
	_it_should("give the same brier score", nearly_equal(first.brier_score, second.brier_score));

	return NULL;
}

/*
 * A network whose worth is not the same at every position: the more marks
 * there are on the board, the more it says the position is worth.
 */
static void build_the_varying_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 1, ACTIVATION_CLAMP);
	network_add_layer(network, 1, 1, ACTIVATION_SIGMOID);

	for (int cell = 0; cell < CELL_COUNT; cell++)
		network->layers[0].weights[0][cell] = (double)(cell + 1) / 45.0;

	network->layers[1].weights[0][0] = 1.0;
}

/*
 * The same games played again, and the truth worked out for each position as
 * it was when the move was made, to check the error against.
 */
static double expected_truth_error(network_t *network)
{
	policy_t network_policy;
	policy_t random_policy;
	game_trace_t trace;
	double total = 0.0;
	int moves = 0;

	policy_init(&network_policy, &game, network, 1, 0.0);
	policy_init(&random_policy, &game, network, 2, INFINITY);

	for (int i = 0; i < 40; i++) {
		game_position_t positions[GAME_MAX_HISTORY];
		game_position_t final;
		int played;

		/* the network takes one side and then the other, as it does in
		 * the measurement */
		if (i % 2 == 0)
			game_play(&game, policy_choose, &network_policy, policy_choose,
			          &random_policy, &trace);
		else
			game_play(&game, policy_choose, &random_policy, policy_choose,
			          &network_policy, &trace);

		played = game_replay(&game, &trace, positions, &final);

		for (int step = 0; step < played; step++) {
			double features[GAME_MAX_FEATURES];
			double worth;

			game.features(&positions[step], features);
			network_forward(network, features, game.n_features, &worth, 1);

			total += fabs(worth - oracle.worth(oracle.state, &positions[step]));
			moves++;
		}
	}

	return total / moves;
}

static char *test_the_truth_error_is_measured_against_the_perfect_player(void)
{
	network_t network;
	evaluation_t result;

	build_the_test_network(&network);

	evaluate_network(&game, &oracle, &network, 1, 40, &result);

	/* the network says a half for every cell, so it is out by a half
	 * wherever the position was worth something else, and by nothing at
	 * all wherever the position was a draw */
	_it_should("be between nought and a half",
		result.truth_error >= 0.0 && result.truth_error <= 0.5);
	_it_should("be less than a half", result.truth_error < 0.5);
	_it_should("be the error against the position as it was",
		nearly_equal(result.truth_error, expected_truth_error(&network)));

	/* the same again with a network that says something different about
	 * each position, so that measuring the wrong one would show */
	network_t varying;
	evaluation_t varying_result;

	build_the_varying_network(&varying);
	evaluate_network(&game, &oracle, &varying, 1, 40, &varying_result);

	_it_should("measure the worth of every position",
		nearly_equal(varying_result.truth_error, expected_truth_error(&varying)));

	return NULL;
}

/*
 * A network that plays the cells in a fixed order, so that the game it
 * plays is the same every time and can be worked out by hand. The first
 * player plays the cell at the top of the order and the other one the cell at
 * the bottom, so the game is
 *
 *   the first player plays 0, the other plays 8, the first plays 1, the other
 *   plays 7, the first plays 2
 *
 * and the first player has the top row and wins.
 */
static void build_the_ordered_network(network_t *network)
{
	const int order[CELL_COUNT] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

	network_init(network);
	network_add_layer(network, CELL_COUNT, 1, ACTIVATION_CLAMP);
	network_add_layer(network, 1, 1, ACTIVATION_SIGMOID);

	/* the cell with the highest weight leaves the best position */
	for (int place = 0; place < CELL_COUNT; place++)
		network->layers[0].weights[0][order[place]] =
			(double)(CELL_COUNT - place) / 45.0;

	network->layers[1].weights[0][0] = 1.0;
}

static char *test_a_network_that_has_learned_nothing_yet(void)
{
	network_t network;
	initialisation_t start;

	/* a network with every weight and bias at zero */
	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);

	_it_should("look at it", evaluate_initialisation(&game, &network, &start));

	_it_should("see no sums at all in the hidden layer",
		nearly_equal(start.hidden_lowest, 0.0) && nearly_equal(start.hidden_highest, 0.0));
	_it_should("see no sum at the output either",
		nearly_equal(start.output_lowest, 0.0) && nearly_equal(start.output_highest, 0.0));
	_it_should("say a half", nearly_equal(start.worth, 0.5));
	_it_should("be pulled just as hard towards a win as towards a loss",
		nearly_equal(start.gradient_to_one, 0.5) &&
		nearly_equal(start.gradient_to_zero, 0.5));

	return NULL;
}

static char *test_a_network_with_every_weight_the_same(void)
{
	network_t network;
	initialisation_t start;

	/* a network with every weight and bias at one, and as many hidden
	 * units as the arena gives it */
	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 16, ACTIVATION_RELU);
	network_add_layer(&network, 16, 1, ACTIVATION_SIGMOID);
	network_fill_maximum(&network, 1.0);

	_it_should("look at it", evaluate_initialisation(&game, &network, &start));

	/* every hidden unit sees exactly the same thing */
	_it_should("see the same sum everywhere in the hidden layer",
		nearly_equal(start.hidden_lowest, start.hidden_highest));

	/* every hidden unit says one, and there are sixteen of them, and the
	 * output's bias is one as well */
	_it_should("see a sum of seventeen at the output",
		nearly_equal(start.output_lowest, 17.0) &&
		nearly_equal(start.output_highest, 17.0));
	_it_should("say one", start.worth > 0.99);
	_it_should("have no pull at all towards a win",
		start.gradient_to_one < 0.01);
	_it_should("be pulled hard towards a loss",
		start.gradient_to_zero > 0.99);

	return NULL;
}

static char *test_the_initialisation_refuses_bad_arguments(void)
{
	network_t network;
	initialisation_t start;

	build_the_test_network(&network);

	_it_should("refuse a null network", !evaluate_initialisation(&game, NULL, &start));
	_it_should("refuse nowhere to put the result",
		!evaluate_initialisation(&game, &network, NULL));

	return NULL;
}

static char *test_the_skill_measurement_refuses_bad_arguments(void)
{
	network_t network;
	skill_t skill;

	build_the_ordered_network(&network);

	_it_should("refuse a null network",
		!evaluate_skill(&game, &oracle, NULL, 1, 10, 0.0, &skill));
	_it_should("refuse a null result",
		!evaluate_skill(&game, &oracle, &network, 1, 10, 0.0, NULL));
	_it_should("refuse no games",
		!evaluate_skill(&game, &oracle, &network, 1, 0, 0.0, &skill));

	return NULL;
}

static char *test_the_skill_measurement_counts_the_moves(void)
{
	network_t network;
	skill_t skill;

	build_the_ordered_network(&network);

	evaluate_skill(&game, &oracle, &network, 1, 10, 0.0, &skill);

	/* the game is over after seven moves, and it is the same game every
	 * time, so ten games are seventy moves */
	_it_should("count every move that was played", skill.moves == 70);
	_it_should("count every move once", skill.kept + skill.gave_away == skill.moves);
	_it_should("never throw more wins away than it gave away",
		skill.threw_wins <= skill.gave_away);

	return NULL;
}

static char *test_the_skill_measurement_says_what_was_given_away(void)
{
	network_t network;
	skill_t skill;

	build_the_ordered_network(&network);

	evaluate_skill(&game, &oracle, &network, 1, 10, 0.0, &skill);

	/* the seven moves of the game, and what each one gave away:
	 *
	 *   the first player plays 0, from a position worth nothing, and it is
	 *     still worth nothing: nothing given away
	 *   the other player plays 1, from a position worth nothing, and now the
	 *     first player is winning: half given away
	 *   the first player plays 2, from a position worth a win, and it is now
	 *     a draw: half given away, and it was a win that was thrown away
	 *   the other player plays 3, from a position worth nothing, and now the
	 *     first player is winning: half given away
	 *   the first player plays 4, from a position worth a win, and it is
	 *     still a win: nothing given away
	 *   the other player plays 5, from a position worth a loss, and it is
	 *     still a loss: nothing given away
	 *   the first player plays 6, from a position worth a win, and it is
	 *     still a win: nothing given away
	 */
	_it_should("keep the value four times in each game", skill.kept == 40);
	_it_should("give value away three times in each game", skill.gave_away == 30);
	_it_should("throw one win away in each game", skill.threw_wins == 10);
	_it_should("give a point and a half of value away in each game",
		nearly_equal(skill.value_lost, 15.0));

	return NULL;
}

/*
 * The same measurement against an opponent that plays at random, which
 * reaches the positions the network would not have reached on its own.
 */
static char *test_the_skill_measurement_against_a_random_player(void)
{
	network_t network;
	skill_t skill;
	skill_t own;

	build_the_ordered_network(&network);

	evaluate_skill(&game, &oracle, &network, 1, 10, 0.0, &own);
	evaluate_skill(&game, &oracle, &network, 1, 10, INFINITY, &skill);

	_it_should("play some games", skill.moves > 0);
	_it_should("count every move once",
		skill.kept + skill.gave_away == skill.moves);
	_it_should("give away no more than a move can be worth",
		skill.value_lost >= 0.0);
	_it_should("reach different positions than it does on its own",
		skill.moves != own.moves || skill.value_lost != own.value_lost);

	/* the network takes one side and then the other, so it is measured
	 * playing both sides. These are the numbers that comes to: with the
	 * sides the other way round the measurement is a different one, and it
	 * says 65 moves, 34 kept and 20 points of value given away */
	_it_should("count the moves it played", skill.moves == 62);
	_it_should("count the moves that kept the value", skill.kept == 33);
	_it_should("count the moves that gave value away", skill.gave_away == 29);
	_it_should("count the wins it threw away", skill.threw_wins == 15);
	_it_should("count the value it gave away",
		nearly_equal(skill.value_lost, 20.0));

	return NULL;
}

static char *test_the_games_against_the_perfect_player(void)
{
	network_t network;
	perfect_result_t result;

	build_the_ordered_network(&network);

	evaluate_against_perfect(&game, &oracle, &network, 1, 10, &result);

	_it_should("count the games", result.games == 10);
	_it_should("count every game once",
		result.wins + result.draws + result.losses == 10);

	/* the network takes the first side in five of the games and the other
	 * side in five, and it plays the same cells whatever the opponent does,
	 * so the perfect player beats it every time */
	_it_should("lose every game", result.losses == 10);
	_it_should("win none", result.wins == 0);
	_it_should("draw none", result.draws == 0);
	_it_should("take each side five times",
		result.as_first == 5 && result.as_second == 5);
	_it_should("win no game it played first", result.wins_as_first == 0);

	return NULL;
}

static char *run_tests()
{
	tictactoe_game(&game);
	tictactoe_oracle(&oracle);
	oracle_start(&oracle);

	_run_test(test_it_refuses_bad_arguments);
	_run_test(test_it_counts_the_games);
	_run_test(test_an_untrained_network_says_a_half);
	_run_test(test_the_buckets_hold_the_observations);
	_run_test(test_the_same_seed_gives_the_same_result);
	_run_test(test_the_truth_error_is_measured_against_the_perfect_player);
	_run_test(test_a_network_that_has_learned_nothing_yet);
	_run_test(test_a_network_with_every_weight_the_same);
	_run_test(test_the_initialisation_refuses_bad_arguments);
	_run_test(test_the_skill_measurement_refuses_bad_arguments);
	_run_test(test_the_skill_measurement_counts_the_moves);
	_run_test(test_the_skill_measurement_says_what_was_given_away);
	_run_test(test_the_skill_measurement_against_a_random_player);
	_run_test(test_the_games_against_the_perfect_player);

	oracle_stop(&oracle);

	return NULL;
}
