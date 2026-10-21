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
#include "../src/minimax.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"

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

	_it_should("refuse a null network", !evaluate_network(NULL, 1, 10, &result));
	_it_should("refuse a null result", !evaluate_network(&network, 1, 10, NULL));
	_it_should("refuse no games", !evaluate_network(&network, 1, 0, &result));

	return NULL;
}

static char *test_it_counts_the_games(void)
{
	network_t network;
	evaluation_t result;

	build_the_test_network(&network);

	evaluate_network(&network, 1, 40, &result);

	_it_should("count the games", result.games == 40);
	_it_should("count every game once",
		result.wins + result.draws + result.losses == 40);
	_it_should("take each colour the same number of times",
		result.as_x == result.as_o && result.as_x == 20);
	_it_should("count the wins in each colour",
		result.wins == result.wins_as_x + result.wins_as_o);
	_it_should("count the games X won",
		result.x_wins + result.games - result.x_wins == result.games);
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

	evaluate_network(&network, 1, 40, &result);

	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++)
		moves += result.bucket_counts[bucket];

	_it_should("count every move that was played", moves > 0);
	_it_should("put them all in the middle band", result.bucket_counts[2] == moves);
	_it_should("say a half for all of them", nearly_equal(result.bucket_predicted[2], 0.5));

	/* the network answers with the worth of a position, which is one if X
	 * wins it, a half if it is drawn and nought if O wins it, so the share
	 * of the moves that ended that way is what it is measured against */
	double expected = (double)result.moves_in_x_wins / result.moves +
	                  0.5 * result.moves_in_draws / result.moves;

	_it_should("measure against the moves in the games X won",
		nearly_equal(result.bucket_observed[2], expected));

	/* a half against a result of one or zero is a quarter out, and a half
	 * against a draw is not out at all */
	int lost = result.moves - result.moves_in_x_wins - result.moves_in_draws;
	double expected_brier = 0.25 * (result.moves_in_x_wins + lost) / result.moves;

	_it_should("be a quarter out of every win and every loss",
		nearly_equal(result.brier_score, expected_brier));

	return NULL;
}

static char *test_the_buckets_hold_the_observations(void)
{
	network_t network;
	evaluation_t result;

	build_the_test_network(&network);

	evaluate_network(&network, 1, 40, &result);

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

	evaluate_network(&network, 99, 30, &first);
	evaluate_network(&network, 99, 30, &second);

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
	double total = 0.0;
	int moves = 0;
	policy_init(&network_policy, network, 1, 0.0);
	policy_init(&random_policy, network, 2, INFINITY);

	for (int i = 0; i < 40; i++) {
		game_t played;
		board_t positions[CELL_COUNT];
		double inputs[CELL_COUNT];
		double worth;

		/* the network takes one colour and then the other, as it does in
		 * the measurement */
		if (i % 2 == 0)
			game_play(&played, &network_policy, &random_policy);
		else
			game_play(&played, &random_policy, &network_policy);

		game_positions(&played, positions);

		for (int step = 0; step < played.moves; step++) {
			board_to_inputs(&positions[step], inputs);
			network_forward(network, inputs, CELL_COUNT, &worth, 1);

			total += fabs(worth - minimax_x_win_chance(&positions[step]));
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

	evaluate_network(&network, 1, 40, &result);

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
	evaluate_network(&varying, 1, 40, &varying_result);

	_it_should("measure the worth of every position",
		nearly_equal(varying_result.truth_error, expected_truth_error(&varying)));

	return NULL;
}

/*
 * A network that plays the cells in a fixed order, so that the game it
 * plays is the same every time and can be worked out by hand. X plays the
 * cell at the top of the order and O the one at the bottom, so the game is
 *
 *   X plays 0, O plays 8, X plays 1, O plays 7, X plays 2
 *
 * and X has the top row and wins.
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

static char *test_the_skill_measurement_refuses_bad_arguments(void)
{
	network_t network;
	skill_t skill;

	build_the_ordered_network(&network);

	_it_should("refuse a null network", !evaluate_skill(NULL, 1, 10, &skill));
	_it_should("refuse a null result", !evaluate_skill(&network, 1, 10, NULL));
	_it_should("refuse no games", !evaluate_skill(&network, 1, 0, &skill));

	return NULL;
}

static char *test_the_skill_measurement_counts_the_moves(void)
{
	network_t network;
	skill_t skill;

	build_the_ordered_network(&network);

	evaluate_skill(&network, 1, 10, &skill);

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

	evaluate_skill(&network, 1, 10, &skill);

	/* the seven moves of the game, and what each one gave away:
	 *
	 *   X plays 0, from a position worth nothing, and it is still worth
	 *     nothing: nothing given away
	 *   O plays 1, from a position worth nothing, and now X is winning:
	 *     one given away
	 *   X plays 2, from a position worth a win, and it is now a draw:
	 *     one given away, and it was a win that was thrown away
	 *   O plays 3, from a position worth nothing, and now X is winning:
	 *     one given away
	 *   X plays 4, from a position worth a win, and it is still a win:
	 *     nothing given away
	 *   O plays 5, from a position worth a loss, and it is still a loss:
	 *     nothing given away
	 *   X plays 6, from a position worth a win, and it is still a win:
	 *     nothing given away
	 */
	_it_should("keep the value four times in each game", skill.kept == 40);
	_it_should("give value away three times in each game", skill.gave_away == 30);
	_it_should("throw one win away in each game", skill.threw_wins == 10);
	_it_should("give three points of value away in each game",
		nearly_equal(skill.value_lost, 30.0));

	return NULL;
}

static char *test_the_games_against_the_perfect_player(void)
{
	network_t network;
	perfect_result_t result;

	build_the_ordered_network(&network);

	evaluate_against_perfect(&network, 1, 10, &result);

	_it_should("count the games", result.games == 10);
	_it_should("count every game once",
		result.wins + result.draws + result.losses == 10);

	/* the network takes X in five of the games and O in five, and it
	 * plays the same cells whatever the opponent does, so the perfect
	 * player beats it every time */
	_it_should("lose every game", result.losses == 10);
	_it_should("win none", result.wins == 0);
	_it_should("draw none", result.draws == 0);
	_it_should("take each colour five times",
		result.as_x == 5 && result.as_o == 5);
	_it_should("win no game it played as X", result.wins_as_x == 0);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_it_refuses_bad_arguments);
	_run_test(test_it_counts_the_games);
	_run_test(test_an_untrained_network_says_a_half);
	_run_test(test_the_buckets_hold_the_observations);
	_run_test(test_the_same_seed_gives_the_same_result);
	_run_test(test_the_truth_error_is_measured_against_the_perfect_player);
	_run_test(test_the_skill_measurement_refuses_bad_arguments);
	_run_test(test_the_skill_measurement_counts_the_moves);
	_run_test(test_the_skill_measurement_says_what_was_given_away);
	_run_test(test_the_games_against_the_perfect_player);

	return NULL;
}
