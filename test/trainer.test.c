#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <math.h>

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

/* A network whose output is a probability, so it answers with the worth of
 * a position. */
static void build_the_test_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
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

/* A game that is drawn: neither player gets a line. */
static void build_a_drawn_game(game_t *game)
{
	game_init(game);
	game_play_move(game, CELL_TOP_LEFT);
	game_play_move(game, CELL_TOP);
	game_play_move(game, CELL_TOP_RIGHT);
	game_play_move(game, CELL_CENTER);
	game_play_move(game, CELL_MIDDLE_LEFT);
	game_play_move(game, CELL_MIDDLE_RIGHT);
	game_play_move(game, CELL_BOTTOM);
	game_play_move(game, CELL_BOTTOM_LEFT);
	game_play_move(game, CELL_BOTTOM_RIGHT);
}

/* The worth of the empty board, which is the position the first move was
 * made in. */
static double worth_of_the_empty_board(const network_t *network)
{
	board_t board;
	double inputs[CELL_COUNT];
	double worth;

	board_init(&board);
	board_to_inputs(&board, inputs);
	network_forward(network, inputs, CELL_COUNT, &worth, 1);

	return worth;
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

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, 1, ACTIVATION_CLAMP);

	_it_should("refuse a network whose output is not a probability",
		!trainer_init(&trainer, &network, 0.1));

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_SIGMOID);

	_it_should("refuse a network that answers with nine numbers",
		!trainer_init(&trainer, &network, 0.1));

	return NULL;
}

static char *test_a_win_raises_the_worth_of_the_position(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;
	double before;
	double after;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	before = worth_of_the_empty_board(&network);
	_it_should("start at a half", nearly_equal(before, 0.5));

	trainer_train_game(&trainer, &game);

	after = worth_of_the_empty_board(&network);
	_it_should("raise the worth of the position", after > before);
	_it_should("raise it towards one", after < 1.0);

	return NULL;
}

static char *test_a_loss_lowers_the_worth_of_the_position(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;
	double before;
	double after;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_lost_game(&game);

	before = worth_of_the_empty_board(&network);

	trainer_train_game(&trainer, &game);

	after = worth_of_the_empty_board(&network);
	_it_should("lower the worth of the position", after < before);
	_it_should("lower it towards zero", after > 0.0);

	return NULL;
}

static char *test_a_draw_leaves_the_worth_where_it_was(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;
	double before;
	double after;

	build_the_test_network(&network);

	/* the network starts off saying the position is worth nothing, so the
	 * target of a drawn game has something to pull it towards */
	network.layers[1].biases[0] = -5.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_drawn_game(&game);

	before = worth_of_the_empty_board(&network);

	trainer_train_game(&trainer, &game);

	after = worth_of_the_empty_board(&network);
	_it_should("pull the worth towards a half", after > before);
	_it_should("not go past a half", after < 0.5);

	return NULL;
}

static char *test_it_learns_a_position_towards_a_target(void)
{
	network_t network;
	trainer_t trainer;
	board_t board;
	double before;
	double after;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	board_init(&board);

	before = worth_of_the_empty_board(&network);

	trainer_train_position(&trainer, &board, 1.0);

	after = worth_of_the_empty_board(&network);
	_it_should("raise the worth towards the target", after > before);
	_it_should("not go past it", after < 1.0);

	trainer_train_position(&trainer, &board, 0.0);

	_it_should("lower the worth towards the other target",
		worth_of_the_empty_board(&network) < after);

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
	_it_should("still move the output bias", network.layers[1].biases[0] != 0.0);

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
	network.layers[1].weights[0][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	_it_should("change the hidden layer", network.layers[0].weights[0][0] != 1.0);
	_it_should("move the weight of the output", network.layers[1].weights[0][0] != 1.0);

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
	_it_should("have changed the network", network.layers[1].biases[0] != 0.0);

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

	/* the hidden neuron reads the top right cell, which is the last cell
	 * to be played, and the output reads the hidden value */
	network.layers[0].weights[0][CELL_TOP_RIGHT] = 1.0;
	network.layers[1].weights[0][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	/* the top right was empty in every position the trainer saw, because
	 * the position after the last move is not one it learns from */
	_it_should("not learn from a position that was never played",
		network.layers[0].weights[0][CELL_TOP_RIGHT] == 1.0);
	_it_should("still learn something", network.layers[1].biases[0] != 0.0);

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
	 * that its sum is well below the floor and it is switched off, and the
	 * output reads the hidden value */
	network.layers[0].weights[0][0] = -5.0;
	network.layers[1].weights[0][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&game);

	trainer_train_game(&trainer, &game);

	_it_should("not change a hidden neuron that is switched off",
		network.layers[0].weights[0][0] == -5.0);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_it_refuses_a_null_network);
	_run_test(test_it_refuses_a_network_that_does_not_answer_with_a_probability);
	_run_test(test_a_win_raises_the_worth_of_the_position);
	_run_test(test_a_loss_lowers_the_worth_of_the_position);
	_run_test(test_a_draw_leaves_the_worth_where_it_was);
	_run_test(test_it_learns_a_position_towards_a_target);
	_run_test(test_a_network_with_no_weights_cannot_learn_anything_else);
	_run_test(test_the_error_reaches_the_hidden_layer);
	_run_test(test_the_activation_stops_the_error);
	_run_test(test_it_learns_from_the_position_the_move_was_made_in);
	_run_test(test_it_plays_and_learns);

	return NULL;
}
