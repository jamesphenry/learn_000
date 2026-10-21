#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <math.h>

#include "../src/board.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"

/*
 * A network that says a position is worth more when X has the top row.
 *
 * The hidden layer has two units: one that fires when X has a mark in the
 * top row, and one that fires when O has a mark in the top row. The worth is
 * those two turned into a probability, and the first counts for more than
 * the second.
 *
 * So X wants the top row and O wants to keep away from it, which means the
 * two players want different cells and the policy can be caught taking the
 * wrong one.
 */
static void build_the_test_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_RELU);

	/* the first unit fires for X in the top row, and the more of it there
	 * is the more it fires */
	network->layers[0].weights[0][CELL_TOP_LEFT] = 3.0;
	network->layers[0].weights[0][CELL_TOP] = 2.0;
	network->layers[0].weights[0][CELL_TOP_RIGHT] = 1.0;

	/* and the second for O in the top row */
	network->layers[0].weights[1][CELL_TOP_LEFT] = -3.0;
	network->layers[0].weights[1][CELL_TOP] = -2.0;
	network->layers[0].weights[1][CELL_TOP_RIGHT] = -1.0;

	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
	network->layers[1].weights[0][0] = 4.0;
	network->layers[1].weights[0][1] = 1.0;
	network->layers[1].biases[0] = -6.0;
}

/* A network that says every position is worth the same. */
static void build_the_flat_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
}

static char *test_it_picks_an_empty_cell(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	_it_should("pick a move", policy_select_move(&policy, &board, &move));
	_it_should("pick a cell that is empty", board.cells[move] == VALUE_EMPTY);

	return NULL;
}

static char *test_it_never_picks_a_cell_that_is_taken(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'O');

	_it_should("still pick a move", policy_select_move(&policy, &board, &move));
	_it_should("not pick the cell that is taken", move != CELL_BOTTOM_RIGHT);
	_it_should("not pick the cell with the X", move != CELL_TOP_LEFT);

	return NULL;
}

static char *test_it_looks_ahead(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);

	/* the top row is worth the most to X, and the middle left is the
	 * first cell outside it */
	board_init(&board);

	policy_select_move(&policy, &board, &move);

	_it_should("take the cell that leaves the best position",
		move == CELL_TOP_LEFT);

	return NULL;
}

static char *test_o_wants_the_other_end_of_the_worth(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	board.current_player = 'O';

	policy_select_move(&policy, &board, &move);

	/* the worth says how well X would do, so O wants the other end of it,
	 * and the top row is worth the least to O: every cell outside it leaves
	 * the same worth, so any of them will do */
	_it_should("keep away from the row X wants", move != CELL_TOP_LEFT);
	_it_should("keep away from the whole of it",
		move != CELL_TOP && move != CELL_TOP_RIGHT);
	_it_should("take a cell outside it", move >= CELL_MIDDLE_LEFT);

	return NULL;
}

static char *test_it_picks_the_only_cell_left(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_CENTER, 'X');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');

	_it_should("pick the last cell", policy_select_move(&policy, &board, &move));
	_it_should("pick the one that is left", move == CELL_BOTTOM);

	return NULL;
}

static char *test_it_gives_up_on_a_full_board(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_CENTER, 'X');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM, 'O');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');

	_it_should("refuse to move", !policy_select_move(&policy, &board, &move));

	return NULL;
}

static char *test_the_same_seed_gives_the_same_move(void)
{
	network_t network;
	policy_t first;
	policy_t second;
	board_t board;
	board_cell_t first_move;
	board_cell_t second_move;

	build_the_flat_network(&network);
	policy_init(&first, &network, 99, 1.0);
	policy_init(&second, &network, 99, 1.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	policy_select_move(&first, &board, &first_move);
	policy_select_move(&second, &board, &second_move);

	_it_should("give the same move", first_move == second_move);

	return NULL;
}

static char *test_a_hot_temperature_explores(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;
	int other_cells = 0;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 1000.0);

	board_init(&board);

	int cells_seen[CELL_COUNT] = {0};

	for (int i = 0; i < 200; i++) {
		policy_select_move(&policy, &board, &move);
		cells_seen[move]++;

		if (move != CELL_TOP_LEFT)
			other_cells++;
	}

	int different_cells = 0;
	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (cells_seen[cell] > 0)
			different_cells++;
	}

	_it_should("pick cells other than the best one", other_cells > 0);
	_it_should("give a different cell each time", different_cells > 1);

	return NULL;
}

static char *test_a_cold_temperature_stays_with_the_best_cell(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;
	int other_cells = 0;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.01);

	board_init(&board);

	for (int i = 0; i < 200; i++) {
		policy_select_move(&policy, &board, &move);

		if (move != CELL_TOP_LEFT)
			other_cells++;
	}

	_it_should("stay with the best cell", other_cells == 0);

	return NULL;
}

static char *test_null_arguments(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, 0.0);
	board_init(&board);

	_it_should("refuse a null policy", !policy_select_move(NULL, &board, &move));
	_it_should("refuse a null board", !policy_select_move(&policy, NULL, &move));
	_it_should("refuse a null move", !policy_select_move(&policy, &board, NULL));

	return NULL;
}

static char *test_it_copes_with_very_large_sums(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network);

	/* a network whose sums are far too large to exponentiate */
	network.layers[1].biases[0] = 1000.0;

	policy_init(&policy, &network, 1, 1.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	_it_should("still pick a move", policy_select_move(&policy, &board, &move));
	_it_should("still leave a cell that is empty", board.cells[move] == VALUE_EMPTY);

	return NULL;
}

static char *test_a_tie_is_broken_at_random(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;
	int cells_seen[CELL_COUNT] = {0};

	/* a network whose worth is the same for every position, so every cell
	 * is a tie */
	build_the_flat_network(&network);

	board_init(&board);

	for (int seed = 0; seed < 200; seed++) {
		policy_init(&policy, &network, (unsigned long long)seed, 0.0);
		policy_select_move(&policy, &board, &move);
		cells_seen[move]++;
	}

	int different_cells = 0;
	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (cells_seen[cell] > 0)
			different_cells++;
	}

	_it_should("not always take the same cell", different_cells > 1);

	return NULL;
}

/* At an infinite temperature every legal cell counts the same, so the
 * policy plays at random. The evaluation uses it as its opponent. */
static char *test_an_infinite_temperature_plays_at_random(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;
	int seen[CELL_COUNT] = {0};

	build_the_test_network(&network);
	policy_init(&policy, &network, 1, INFINITY);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	for (int i = 0; i < 400; i++) {
		policy_select_move(&policy, &board, &move);
		seen[move]++;
	}

	int distinct = 0;
	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (seen[cell] > 0)
			distinct++;
	}

	_it_should("pick more than one cell", distinct > 1);
	_it_should("not always pick the same cell", seen[CELL_TOP_LEFT] < 400);

	return NULL;
}

static char *test_it_wants_a_network_that_answers_with_one_number(void)
{
	network_t network;
	network_t value_network;
	policy_t policy;

	/* a network that answers with a number for every cell */
	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_SIGMOID);

	_it_should("refuse a network that answers with nine numbers",
		!policy_init(&policy, &network, 1, 0.0));

	network_init(&value_network);
	network_add_layer(&value_network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(&value_network, 2, 1, ACTIVATION_SIGMOID);

	_it_should("take a network that answers with one number",
		policy_init(&policy, &value_network, 1, 0.0));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_it_picks_an_empty_cell);
	_run_test(test_it_never_picks_a_cell_that_is_taken);
	_run_test(test_it_looks_ahead);
	_run_test(test_o_wants_the_other_end_of_the_worth);
	_run_test(test_it_picks_the_only_cell_left);
	_run_test(test_it_gives_up_on_a_full_board);
	_run_test(test_a_tie_is_broken_at_random);
	_run_test(test_the_same_seed_gives_the_same_move);
	_run_test(test_a_hot_temperature_explores);
	_run_test(test_a_cold_temperature_stays_with_the_best_cell);
	_run_test(test_it_copes_with_very_large_sums);
	_run_test(test_an_infinite_temperature_plays_at_random);
	_run_test(test_null_arguments);
	_run_test(test_it_wants_a_network_that_answers_with_one_number);

	return NULL;
}
