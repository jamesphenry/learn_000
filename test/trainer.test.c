#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"
#include "../src/game.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../src/trainer.c"

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

/* A network whose output is a probability, so every cell answers with the
 * chance that X wins if that move is played. */
static void build_the_test_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(network, 2, CELL_COUNT, ACTIVATION_SIGMOID);
}

/* X takes the top row and wins. */
static void build_a_won_game(game_t *game)
{
	game_init(game);
	game_play_move(game, CELL_TOP_LEFT);
	game_play_move(game, CELL_CENTER);
	game_play_move(game, CELL_TOP);
	game_play_move(game, CELL_MIDDLE_LEFT);
	game_play_move(game, CELL_TOP_RIGHT);
}

/* O takes the middle row and wins. */
static void build_a_lost_game(game_t *game)
{
	game_init(game);
	game_play_move(game, CELL_TOP_LEFT);
	game_play_move(game, CELL_CENTER);
	game_play_move(game, CELL_TOP);
	game_play_move(game, CELL_MIDDLE_LEFT);
	game_play_move(game, CELL_BOTTOM_RIGHT);
	game_play_move(game, CELL_MIDDLE_RIGHT);
}

/* The score of the top left cell, with X to move and nothing on the board. */
static double score_of_the_first_cell(const network_t *network)
{
	board_t board;
	double inputs[CELL_COUNT];
	double scores[CELL_COUNT];

	board_init(&board);
	board_to_inputs(&board, inputs);
	network_forward(network, inputs, CELL_COUNT, scores, CELL_COUNT);

	return scores[CELL_TOP_LEFT];
}

static char *test_it_refuses_a_null_network(void)
{
	trainer_t trainer;

	_it_should("refuse a null network", !trainer_init(&trainer, NULL, 0.1));

	return NULL;
}

static char *test_it_refuses_a_network_that_does_not_answer_with_a_probability(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_CLAMP);

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	_it_should("refuse a network whose output is not a probability",
		!trainer_train_game(&trainer, &game));

	return NULL;
}

static char *test_a_win_raises_the_score_of_the_move_that_won(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;
	double before;
	double after;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	before = score_of_the_first_cell(&network);
	_it_should("start at a half", nearly_equal(before, 0.5));

	trainer_train_game(&trainer, &game);

	after = score_of_the_first_cell(&network);
	_it_should("raise the score of the move that won", after > before);
	_it_should("raise it towards one", after < 1.0);

	return NULL;
}

static char *test_a_loss_lowers_the_score_of_the_move_that_lost(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;
	double before;
	double after;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_lost_game(&game);

	before = score_of_the_first_cell(&network);

	trainer_train_game(&trainer, &game);

	after = score_of_the_first_cell(&network);
	_it_should("lower the score of the move that lost", after < before);
	_it_should("lower it towards zero", after > 0.0);

	return NULL;
}

static char *test_it_leaves_the_other_moves_alone(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	/* only the neurons for the moves that were played are wrong, so the
	 * others are untouched. The bottom right was never played. */
	_it_should("leave the moves that were not played alone",
		network.layers[1].biases[CELL_BOTTOM_RIGHT] == 0.0);
	_it_should("move the bias that was played",
		network.layers[1].biases[CELL_TOP_LEFT] != 0.0);

	return NULL;
}

/*
 * A network that starts with nothing but zeros can learn the bias of its
 * output and nothing else. The error has to travel through the hidden
 * values to reach the hidden layer, and those are all zero, so it never
 * gets there. A network started this way never breaks its symmetry.
 */
static char *test_a_network_with_no_weights_cannot_learn_anything_else(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	_it_should("leave the hidden layer dead", network.layers[0].weights[0][0] == 0.0);
	_it_should("still move the output bias", network.layers[1].biases[CELL_TOP_LEFT] != 0.0);

	return NULL;
}

static char *test_the_error_reaches_the_hidden_layer(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;

	build_the_test_network(&network);

	/* weights of their own, so that the error has something to travel
	 * through: from the output neuron, through its weight, to the hidden
	 * value, and on to the hidden weights. The hidden neuron reads the
	 * cell X took first, so it is awake by the time the top middle is
	 * played. */
	network.layers[0].weights[0][0] = 1.0;
	network.layers[1].weights[CELL_TOP][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	_it_should("change the hidden layer", network.layers[0].weights[0][0] != 1.0);
	_it_should("move the weight that was played",
		network.layers[1].weights[CELL_TOP][0] != 1.0);

	return NULL;
}

static char *test_it_plays_and_learns(void)
{
	network_t network;
	policy_t policy;
	trainer_t trainer;
	int played;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.5);
	trainer_init(&trainer, &network, 0.1);

	played = trainer_run(&trainer, &policy, 10);

	_it_should("play every game", played == 10);
	_it_should("have changed the network",
		network.layers[1].biases[CELL_TOP_LEFT] != 0.0);

	return NULL;
}

/*
 * The trainer has to see each position as it was when the move was made, so
 * the game is played again from the start.
 */
static char *test_it_learns_from_the_position_the_move_was_made_in(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;

	build_the_test_network(&network);

	/* the hidden neuron reads the top left cell, and the output for the top
	 * left cell reads the hidden value */
	network.layers[0].weights[0][0] = 1.0;
	network.layers[1].weights[CELL_TOP_LEFT][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	/* the top left was played on an empty board, where the hidden value is
	 * zero, so the weight of that output cannot move */
	_it_should("not learn from a later position",
		network.layers[1].weights[CELL_TOP_LEFT][0] == 1.0);

	return NULL;
}

/*
 * A hidden neuron whose sum is negative is switched off, and the error must
 * not travel through it.
 */
static char *test_the_activation_stops_the_error(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;

	build_the_test_network(&network);

	/* the hidden neuron reads the top left cell with a weight of its own, so
	 * that its sum is negative and it is switched off, and the output for
	 * the centre reads the hidden value */
	network.layers[0].weights[0][0] = -1.0;
	network.layers[1].weights[CELL_CENTER][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	_it_should("not change a hidden neuron that is switched off",
		network.layers[0].weights[0][0] == -1.0);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_it_refuses_a_null_network);
	_run_test(test_it_refuses_a_network_that_does_not_answer_with_a_probability);
	_run_test(test_a_win_raises_the_score_of_the_move_that_won);
	_run_test(test_a_loss_lowers_the_score_of_the_move_that_lost);
	_run_test(test_it_leaves_the_other_moves_alone);
	_run_test(test_a_network_with_no_weights_cannot_learn_anything_else);
	_run_test(test_the_error_reaches_the_hidden_layer);
	_run_test(test_the_activation_stops_the_error);
	_run_test(test_it_learns_from_the_position_the_move_was_made_in);
	_run_test(test_it_plays_and_learns);

	return NULL;
}
