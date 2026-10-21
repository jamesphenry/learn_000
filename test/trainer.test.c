#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <math.h>

#include "../src/game.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../src/trainer.c"
#include "../games/tic-tac-toe/src/board.c"
#include "../games/tic-tac-toe/src/minimax.c"
#include "../games/tic-tac-toe/src/tictactoe.c"

static game_t game;

/* A network whose output is a probability, so it answers with the worth of
 * a position. */
static void build_the_test_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
}

/*
 * A network whose hidden layer is awake, so that the worth of one position
 * can differ from the worth of another.
 */
static void build_the_awake_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);

	/* the hidden unit reads the middle cell, and the output reads it */
	network->layers[0].weights[0][CELL_CENTER] = 1.0;
	network->layers[1].weights[0][0] = 1.0;
}

/* A trace of the moves that were played, and where the game ended up. */
static void build_a_trace(game_trace_t *trace, const board_cell_t *moves, int count)
{
	game_position_t position;

	game.start(&position);

	trace->n_offered = 0;
	trace->n_played = 0;

	for (int i = 0; i < count; i++) {
		trace->offered[trace->n_offered++] = moves[i];
		game.play(&position, moves[i]);
		trace->n_played++;
	}

	trace->position = position;
}

/* The first player takes the top row and wins. */
static void build_a_won_game(game_trace_t *trace)
{
	const board_cell_t moves[] = {
		CELL_TOP_LEFT, CELL_CENTER, CELL_TOP, CELL_MIDDLE_LEFT, CELL_TOP_RIGHT
	};

	build_a_trace(trace, moves, 5);
}

/* The second player takes the middle row and wins. */
static void build_a_lost_game(game_trace_t *trace)
{
	const board_cell_t moves[] = {
		CELL_TOP_LEFT, CELL_CENTER, CELL_TOP, CELL_MIDDLE_LEFT,
		CELL_BOTTOM_RIGHT, CELL_MIDDLE_RIGHT
	};

	build_a_trace(trace, moves, 6);
}

/* A game that is drawn: neither player gets a line. */
static void build_a_drawn_game(game_trace_t *trace)
{
	const board_cell_t moves[] = {
		CELL_TOP_LEFT, CELL_TOP, CELL_TOP_RIGHT, CELL_CENTER, CELL_MIDDLE_LEFT,
		CELL_MIDDLE_RIGHT, CELL_BOTTOM, CELL_BOTTOM_LEFT, CELL_BOTTOM_RIGHT
	};

	build_a_trace(trace, moves, 9);
}

/* What the network says about a position. */
static double worth_of(const network_t *network, const game_position_t *position)
{
	double features[GAME_MAX_FEATURES];
	double worth;

	game.features(position, features);
	network_forward(network, features, game.n_features, &worth, 1);

	return worth;
}

/* The worth of the empty board, which is the position the first move was
 * made in. */
static double worth_of_the_empty_board(const network_t *network)
{
	game_position_t position;

	game.start(&position);

	return worth_of(network, &position);
}

/*
 * The worth of the position after a given number of moves, which is the
 * position the trainer saw at that point in the game.
 */
static double worth_after_moves(const network_t *network, const game_trace_t *trace,
                                int moves)
{
	game_position_t positions[GAME_MAX_HISTORY];
	game_position_t final;

	game_replay(&game, trace, positions, &final);

	return worth_of(network, &positions[moves]);
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
	game_trace_t trace;
	double before;
	double after;

	build_the_awake_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&trace);

	before = worth_after_moves(&network, &trace, 2);

	trainer_learn_from_outcome(&trainer, &game, &trace);

	/* the position after two moves was the first player's to move in and
	 * the first player won it, so it is worth more afterwards */
	after = worth_after_moves(&network, &trace, 2);
	_it_should("raise the worth of the position", after > before);
	_it_should("raise it towards one", after < 1.0);

	return NULL;
}

static char *test_a_loss_lowers_the_worth_of_the_position(void)
{
	network_t network;
	trainer_t trainer;
	game_trace_t trace;
	double before;
	double after;

	build_the_awake_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_lost_game(&trace);

	before = worth_of_the_empty_board(&network);

	trainer_learn_from_outcome(&trainer, &game, &trace);

	/* the position before the first move was lost by the player to move in
	 * it, so it is worth less afterwards */
	after = worth_after_moves(&network, &trace, 2);
	_it_should("lower the worth of the position", after < before);
	_it_should("lower it towards zero", after > 0.0);

	return NULL;
}

static char *test_a_draw_leaves_the_worth_where_it_was(void)
{
	network_t network;
	trainer_t trainer;
	game_trace_t trace;
	double before;
	double after;

	build_the_test_network(&network);

	/* the network starts off saying the position is worth nothing, so the
	 * target of a drawn game has something to pull it towards */
	network.layers[1].biases[0] = -5.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_drawn_game(&trace);

	before = worth_of_the_empty_board(&network);

	trainer_learn_from_outcome(&trainer, &game, &trace);

	after = worth_of_the_empty_board(&network);
	_it_should("pull the worth towards a half", after > before);
	_it_should("not go past a half", after < 0.5);

	return NULL;
}

static char *test_it_learns_a_position_towards_a_target(void)
{
	network_t network;
	trainer_t trainer;
	game_position_t position;
	double before;
	double after;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	game.start(&position);

	before = worth_of_the_empty_board(&network);

	trainer_train_position(&trainer, &game, &position, 1.0);

	after = worth_of_the_empty_board(&network);
	_it_should("raise the worth towards the target", after > before);
	_it_should("not go past it", after < 1.0);

	trainer_train_position(&trainer, &game, &position, 0.0);

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
	game_trace_t trace;

	build_the_test_network(&network);
	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&trace);

	trainer_learn_from_outcome(&trainer, &game, &trace);

	_it_should("leave the hidden layer dead", network.layers[0].weights[0][0] == 0.0);
	_it_should("still move the output bias", network.layers[1].biases[0] != 0.0);

	return NULL;
}

static char *test_the_error_reaches_the_hidden_layer(void)
{
	network_t network;
	trainer_t trainer;
	game_trace_t trace;

	build_the_test_network(&network);

	/* weights of their own, so that the error has something to travel
	 * through: from the output neuron, through its weight, to the hidden
	 * value, and on to the hidden weights. The hidden neuron reads the
	 * cell the first player took first, so it is awake by the time the top
	 * middle is played. */
	network.layers[0].weights[0][0] = 1.0;
	network.layers[1].weights[0][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&trace);

	trainer_learn_from_outcome(&trainer, &game, &trace);

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
	policy_init(&policy, &game, &network, 1, 0.5);
	trainer_init(&trainer, &network, 0.1);

	played = trainer_run(&trainer, &game, &policy, 10);

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
	game_trace_t trace;

	build_the_test_network(&network);

	/* the hidden neuron reads the top right cell, which is the last cell
	 * to be played, and the output reads the hidden value */
	network.layers[0].weights[0][CELL_TOP_RIGHT] = 1.0;
	network.layers[1].weights[0][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&trace);

	trainer_learn_from_outcome(&trainer, &game, &trace);

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
	game_trace_t trace;

	build_the_test_network(&network);

	/* the hidden neuron reads the top left cell with a weight of its own and
	 * a bias of its own, so that its sum is well below the floor whichever
	 * player is to move, and it is switched off, and the output reads the
	 * hidden value */
	network.layers[0].weights[0][0] = -1.0;
	network.layers[0].biases[0] = -5.0;
	network.layers[1].weights[0][0] = 1.0;

	trainer_init(&trainer, &network, 0.1);
	build_a_won_game(&trace);

	trainer_learn_from_outcome(&trainer, &game, &trace);

	_it_should("not change a hidden neuron that is switched off",
		network.layers[0].weights[0][0] == -1.0);

	return NULL;
}

/*
 * The perfect player says what a position is really worth, and the trainer
 * learns that instead of the result of the game.
 */
static char *test_it_learns_from_the_perfect_player(void)
{
	network_t network;
	trainer_t trainer;
	oracle_t oracle;
	game_trace_t trace;
	double before;
	double after;

	build_the_awake_network(&network);
	trainer_init(&trainer, &network, 0.1);
	tictactoe_oracle(&oracle);
	oracle_start(&oracle);
	build_a_won_game(&trace);

	before = worth_after_moves(&network, &trace, 2);

	trainer_learn_from_oracle(&trainer, &game, &oracle, &trace);

	/* the position after two moves was won by the player to move in it, so
	 * the perfect player says it is worth one */
	after = worth_after_moves(&network, &trace, 2);
	_it_should("raise the worth of the position", after > before);
	_it_should("raise it towards one", after < 1.0);

	oracle_stop(&oracle);

	return NULL;
}

static char *run_tests()
{
	tictactoe_game(&game);

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
	_run_test(test_it_learns_from_the_perfect_player);
	_run_test(test_it_plays_and_learns);

	return NULL;
}
