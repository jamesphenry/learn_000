#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>

#include "../src/board.c"
#include "../src/game.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"

/* A network that gives one cell a score and leaves the rest at zero. */
static void build_the_test_network(network_t *network, board_cell_t favourite)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 1, ACTIVATION_RELU);
	network->layers[0].weights[0][0] = 1.0;

	network_add_layer(network, 1, CELL_COUNT, ACTIVATION_RELU);
	network->layers[1].weights[favourite][0] = 1.0;
}

static char *test_a_game_starts_empty(void)
{
	game_t game;

	game_init(&game);

	_it_should("start with no moves", game.moves == 0);
	_it_should("start with X to move", game_turn(&game) == 'X');
	_it_should("start with an empty board", board_is_full(&game.board) == false);

	return NULL;
}

static char *test_a_move_places_the_mark(void)
{
	game_t game;

	game_init(&game);

	_it_should("take the move", game_play_move(&game, CELL_TOP_LEFT));
	_it_should("put an X down", game.board.cells[CELL_TOP_LEFT] == VALUE_X);
	_it_should("count the move", game.moves == 1);
	_it_should("pass the turn to O", game_turn(&game) == 'O');

	game_play_move(&game, CELL_CENTER);
	_it_should("put an O down next", game.board.cells[CELL_CENTER] == VALUE_O);
	_it_should("pass the turn back to X", game_turn(&game) == 'X');

	return NULL;
}

static char *test_it_refuses_an_illegal_move(void)
{
	game_t game;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);

	_it_should("refuse a cell that is taken", !game_play_move(&game, CELL_TOP_LEFT));
	_it_should("not count the move", game.moves == 1);
	_it_should("refuse a cell that is outside the board",
		!game_play_move(&game, (board_cell_t)CELL_COUNT));
	_it_should("not count that move either", game.moves == 1);

	return NULL;
}

static char *test_it_stops_when_someone_wins(void)
{
	game_t game;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_TOP_RIGHT);

	_it_should("see that the game is over", game_is_over(&game));
	_it_should("say that X won", game_result(&game) == GAME_X_WINS);
	_it_should("refuse another move", !game_play_move(&game, CELL_BOTTOM));

	return NULL;
}

static char *test_a_full_board_with_no_line_is_a_draw(void)
{
	game_t game;

	/* X O X / X O O / O X X, which is a draw and not a win for either */
	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_TOP_RIGHT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_MIDDLE_RIGHT);
	game_play_move(&game, CELL_BOTTOM);
	game_play_move(&game, CELL_BOTTOM_LEFT);
	game_play_move(&game, CELL_BOTTOM_RIGHT);

	_it_should("see that the game is over", game_is_over(&game));
	_it_should("say that it is a draw", game_result(&game) == GAME_DRAW);

	return NULL;
}

static char *test_two_policies_can_play_a_whole_game(void)
{
	network_t network;
	policy_t x_policy;
	policy_t o_policy;
	game_t game;
	game_result_t result;

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
	policy_init(&x_policy, &network, 1, 0.0);
	policy_init(&o_policy, &network, 2, 0.0);

	result = game_play(&game, &x_policy, &o_policy);

	_it_should("finish the game", result == GAME_X_WINS || result == GAME_O_WINS ||
		result == GAME_DRAW);
	_it_should("have played a whole game", game.moves >= 5 && game.moves <= 9);
	_it_should("have recorded every move", game.history[game.moves - 1] != CELL_COUNT);

	return NULL;
}

static char *test_a_game_never_goes_on_for_ever(void)
{
	network_t network;
	policy_t policy;
	game_t game;
	int moves = 0;

	build_the_test_network(&network, CELL_BOTTOM_RIGHT);
	policy_init(&policy, &network, 3, 1.0);

	game_init(&game);
	while (!game_is_over(&game) && moves < 100) {
		board_cell_t move;

		if (!policy_select_move(&policy, &game.board, &move)) break;
		game_play_move(&game, move);
		moves++;
	}

	_it_should("finish within nine moves", moves >= 5 && moves <= 9);
	_it_should("have come to an end", game_is_over(&game));

	return NULL;
}

static char *test_o_can_win(void)
{
	game_t game;

	game_init(&game);
	/* X takes the top left and the top middle, O takes the middle row */
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_BOTTOM_RIGHT);
	game_play_move(&game, CELL_MIDDLE_RIGHT);

	_it_should("see that the game is over", game_is_over(&game));
	_it_should("say that O won", game_result(&game) == GAME_O_WINS);

	return NULL;
}

static char *test_it_records_the_moves(void)
{
	game_t game;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);

	_it_should("record the first move", game.history[0] == CELL_TOP_LEFT);
	_it_should("record the second move", game.history[1] == CELL_CENTER);
	_it_should("record the third move", game.history[2] == CELL_TOP);
	_it_should("record as many moves as were played", game.moves == 3);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_a_game_starts_empty);
	_run_test(test_a_move_places_the_mark);
	_run_test(test_it_refuses_an_illegal_move);
	_run_test(test_it_stops_when_someone_wins);
	_run_test(test_a_full_board_with_no_line_is_a_draw);
	_run_test(test_o_can_win);
	_run_test(test_it_records_the_moves);
	_run_test(test_two_policies_can_play_a_whole_game);
	_run_test(test_a_game_never_goes_on_for_ever);

	return NULL;
}
