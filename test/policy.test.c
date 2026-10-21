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
#include "../games/tic-tac-toe/src/board.c"
#include "../games/tic-tac-toe/src/minimax.c"
#include "../games/tic-tac-toe/src/tictactoe.c"

static game_t game;

/*
 * A network that says a position is worth more when the player to move has
 * the top left cell.
 *
 * The hidden layer has two units: one that fires when the player to move has
 * a mark in the top left, and one that fires when the other one has it. The
 * board is always given to the model from the point of view of the player to
 * move, so the first unit means the cell is the player's own.
 */
static void build_the_test_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_RELU);

	network->layers[0].weights[0][CELL_TOP_LEFT] = 1.0;
	network->layers[0].biases[0] = 0.5;
	network->layers[0].weights[1][CELL_TOP_LEFT] = -1.0;
	network->layers[0].biases[1] = 0.5;

	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
	network->layers[1].weights[0][0] = 40.0;
	network->layers[1].biases[0] = -20.0;
}

/* A network that says every position is worth the same. */
static void build_the_flat_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
}

/* Whether the move is one of the moves the position allows. */
static bool is_allowed(const game_position_t *position, game_move_t move)
{
	for (int i = 0; i < position->n_moves; i++) {
		if (position->moves[i] == move) return true;
	}

	return false;
}

static char *test_it_picks_an_empty_cell(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);

	_it_should("pick a move", policy_select_move(&policy, &position, &move));
	_it_should("pick a cell that is empty", is_allowed(&position, move));

	return NULL;
}

static char *test_it_never_picks_a_cell_that_is_taken(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);
	game.play(&position, CELL_BOTTOM_RIGHT);

	_it_should("still pick a move", policy_select_move(&policy, &position, &move));
	_it_should("not pick the cell that is taken", move != CELL_BOTTOM_RIGHT);
	_it_should("not pick the cell with the X", move != CELL_TOP_LEFT);

	return NULL;
}

static char *test_it_looks_ahead(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);

	/* the top row is worth the most to X, and the middle left is the
	 * first cell outside it */
	game.start(&position);

	policy_select_move(&policy, &position, &move);

	_it_should("take the cell that leaves the best position",
		move == CELL_TOP_LEFT);

	return NULL;
}

static char *test_the_policy_takes_the_best_position_whichever_player_it_is(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);

	/* one move in, so that it is the other player to move */
	game.start(&position);
	game.play(&position, CELL_CENTER);

	_it_should("be the other player's turn", game.to_move(&position) == 1);

	policy_select_move(&policy, &position, &move);

	/* the worth belongs to the player to move, so the policy takes the best
	 * position whoever it is playing for */
	_it_should("take the top left for the other player as well",
		move == CELL_TOP_LEFT);

	return NULL;
}

static char *test_it_picks_the_only_cell_left(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;
	const board_cell_t taken[] = {
		CELL_TOP_LEFT, CELL_TOP_RIGHT, CELL_TOP, CELL_CENTER,
		CELL_MIDDLE_RIGHT, CELL_MIDDLE_LEFT, CELL_BOTTOM_LEFT, CELL_BOTTOM_RIGHT
	};

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);

	game.start(&position);

	for (int i = 0; i < 8; i++)
		game.play(&position, taken[i]);

	_it_should("have one cell left", position.n_moves == 1);

	_it_should("pick the last cell", policy_select_move(&policy, &position, &move));
	_it_should("pick the one that is left", move == CELL_BOTTOM);

	return NULL;
}

static char *test_it_gives_up_on_a_full_board(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;
	const board_cell_t taken[] = {
		CELL_TOP_LEFT, CELL_TOP, CELL_TOP_RIGHT, CELL_CENTER, CELL_MIDDLE_LEFT,
		CELL_MIDDLE_RIGHT, CELL_BOTTOM, CELL_BOTTOM_LEFT, CELL_BOTTOM_RIGHT
	};

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);

	game.start(&position);

	for (int i = 0; i < 9; i++)
		game.play(&position, taken[i]);

	_it_should("refuse to move", !policy_select_move(&policy, &position, &move));

	return NULL;
}

static char *test_the_same_seed_gives_the_same_move(void)
{
	network_t network;
	policy_t first;
	policy_t second;
	game_position_t position;
	game_move_t first_move;
	game_move_t second_move;

	build_the_flat_network(&network);
	policy_init(&first, &game, &network, 99, 1.0);
	policy_init(&second, &game, &network, 99, 1.0);

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);

	policy_select_move(&first, &position, &first_move);
	policy_select_move(&second, &position, &second_move);

	_it_should("give the same move", first_move == second_move);

	return NULL;
}

static char *test_a_hot_temperature_explores(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;
	int other_cells = 0;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 1000.0);

	game.start(&position);

	int cells_seen[CELL_COUNT] = {0};

	for (int i = 0; i < 200; i++) {
		policy_select_move(&policy, &position, &move);
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
	game_position_t position;
	game_move_t move;
	int other_cells = 0;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.01);

	game.start(&position);

	for (int i = 0; i < 200; i++) {
		policy_select_move(&policy, &position, &move);

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
	game_position_t position;
	game_move_t move;

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, 0.0);
	game.start(&position);

	_it_should("refuse a null policy", !policy_select_move(NULL, &position, &move));
	_it_should("refuse a null position", !policy_select_move(&policy, NULL, &move));
	_it_should("refuse a null move", !policy_select_move(&policy, &position, NULL));

	return NULL;
}

static char *test_it_copes_with_very_large_sums(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;

	build_the_test_network(&network);

	/* a network whose sums are far too large to exponentiate */
	network.layers[1].biases[0] = 1000.0;

	policy_init(&policy, &game, &network, 1, 1.0);

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);

	_it_should("still pick a move", policy_select_move(&policy, &position, &move));
	_it_should("still leave a cell that is empty", is_allowed(&position, move));

	return NULL;
}

static char *test_a_tie_is_broken_at_random(void)
{
	network_t network;
	policy_t policy;
	game_position_t position;
	game_move_t move;
	int cells_seen[CELL_COUNT] = {0};

	/* a network whose worth is the same for every position, so every cell
	 * is a tie */
	build_the_flat_network(&network);

	game.start(&position);

	for (int seed = 0; seed < 200; seed++) {
		policy_init(&policy, &game, &network, (unsigned long long)seed, 0.0);
		policy_select_move(&policy, &position, &move);
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
	game_position_t position;
	game_move_t move;
	int seen[CELL_COUNT] = {0};

	build_the_test_network(&network);
	policy_init(&policy, &game, &network, 1, INFINITY);

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);

	for (int i = 0; i < 400; i++) {
		policy_select_move(&policy, &position, &move);
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
		!policy_init(&policy, &game, &network, 1, 0.0));

	network_init(&value_network);
	network_add_layer(&value_network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(&value_network, 2, 1, ACTIVATION_SIGMOID);

	_it_should("take a network that answers with one number",
		policy_init(&policy, &game, &value_network, 1, 0.0));

	return NULL;
}

static char *run_tests()
{
	tictactoe_game(&game);

	_run_test(test_it_picks_an_empty_cell);
	_run_test(test_it_never_picks_a_cell_that_is_taken);
	_run_test(test_it_looks_ahead);
	_run_test(test_the_policy_takes_the_best_position_whichever_player_it_is);
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
