#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>

#include "../src/board.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"

/*
 * A network that gives one cell a score and leaves the rest at zero. The
 * hidden value is one when X is at the top left, so the favourite cell only
 * scores when X is there.
 */
static void build_the_test_network(network_t *network, board_cell_t favourite)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 1, ACTIVATION_RELU);
	network->layers[0].weights[0][0] = 1.0;

	network_add_layer(network, 1, CELL_COUNT, ACTIVATION_RELU);
	network->layers[1].weights[favourite][0] = 1.0;
}

static char *test_it_picks_an_empty_cell(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
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

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'O');

	_it_should("still pick a move", policy_select_move(&policy, &board, &move));
	_it_should("not pick the cell that is taken", move != CELL_BOTTOM_RIGHT);
	_it_should("not pick the cell with the X", move != CELL_TOP_LEFT);

	return NULL;
}

static char *test_it_picks_the_best_cell(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	/* a cell in the middle, so that falling back on the last one is caught */
	build_the_test_network(&network, CELL_CENTER);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	policy_select_move(&policy, &board, &move);

	_it_should("take the best scoring cell", move == CELL_CENTER);

	return NULL;
}

static char *test_it_picks_the_only_cell_left(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
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

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
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

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
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

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
	policy_init(&policy, &network, 1, 1000.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	int cells_seen[CELL_COUNT] = {0};

	for (int i = 0; i < 200; i++) {
		policy_select_move(&policy, &board, &move);
		cells_seen[move]++;

		if (move != CELL_BOTTOM_RIGHT)
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

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
	policy_init(&policy, &network, 1, 0.01);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	for (int i = 0; i < 200; i++) {
		policy_select_move(&policy, &board, &move);

		if (move != CELL_BOTTOM_RIGHT)
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

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
	policy_init(&policy, &network, 1, 0.0);
	board_init(&board);

	_it_should("refuse a null policy", !policy_select_move(NULL, &board, &move));
	_it_should("refuse a null board", !policy_select_move(&policy, NULL, &move));
	_it_should("refuse a null move", !policy_select_move(&policy, &board, NULL));

	return NULL;
}

static char *test_it_copes_with_very_large_scores(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	/* a cell in the middle, so that falling back on the last one is caught */
	build_the_test_network(&network, CELL_CENTER);

	/* a network whose scores are far too large to exponentiate */
	network.layers[1].biases[CELL_CENTER] = 1000.0;

	policy_init(&policy, &network, 1, 1.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	_it_should("still pick a move", policy_select_move(&policy, &board, &move));
	_it_should("still take the best cell", move == CELL_CENTER);

	return NULL;
}

static char *test_a_tie_is_broken_at_random(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;
	int cells_seen[CELL_COUNT] = {0};

	/* a network whose scores are all the same, so every cell is a tie */
	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 1, ACTIVATION_RELU);
	network_add_layer(&network, 1, CELL_COUNT, ACTIVATION_RELU);

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

static char *test_o_wants_the_other_end_of_the_scores(void)
{
	network_t network;
	policy_t policy;
	board_t board;
	board_cell_t move;

	build_the_test_network(&network, CELL_CENTER);
	policy_init(&policy, &network, 1, 0.0);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	/* the scores say how well X would do, so O wants the other end */
	board.current_player = 'O';

	policy_select_move(&policy, &board, &move);

	_it_should("not take the cell X wants", move != CELL_CENTER);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_it_picks_an_empty_cell);
	_run_test(test_it_never_picks_a_cell_that_is_taken);
	_run_test(test_it_picks_the_best_cell);
	_run_test(test_o_wants_the_other_end_of_the_scores);
	_run_test(test_it_picks_the_only_cell_left);
	_run_test(test_it_gives_up_on_a_full_board);
	_run_test(test_a_tie_is_broken_at_random);
	_run_test(test_the_same_seed_gives_the_same_move);
	_run_test(test_a_hot_temperature_explores);
	_run_test(test_a_cold_temperature_stays_with_the_best_cell);
	_run_test(test_it_copes_with_very_large_scores);
	_run_test(test_null_arguments);

	return NULL;
}
