#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../../../src/game.c"
#include "../src/board.c"
#include "../src/minimax.c"
#include "../src/tictactoe.c"

static game_t game;
static oracle_t oracle;

/* The board a position holds. */
static const board_t *board_in_position(const game_position_t *position)
{
	return (const board_t *)position->state.bytes;
}

static char *test_a_game_starts_empty(void)
{
	game_position_t position;

	game.start(&position);

	_it_should("start with an empty board", board_is_full(board_in_position(&position)) == false);
	_it_should("start with the first player to move", game.to_move(&position) == 0);
	_it_should("offer every cell", position.n_moves == CELL_COUNT);

	return NULL;
}

static char *test_a_move_places_the_mark(void)
{
	game_position_t position;

	game.start(&position);

	_it_should("take the move", game.play(&position, CELL_TOP_LEFT));
	_it_should("put an X down",
		board_in_position(&position)->cells[CELL_TOP_LEFT] == VALUE_X);
	_it_should("pass the turn to the other player", game.to_move(&position) == 1);
	_it_should("offer one cell fewer", position.n_moves == CELL_COUNT - 1);

	game.play(&position, CELL_CENTER);

	_it_should("put an O down next",
		board_in_position(&position)->cells[CELL_CENTER] == VALUE_O);
	_it_should("pass the turn back", game.to_move(&position) == 0);

	return NULL;
}

static char *test_it_refuses_an_illegal_move(void)
{
	game_position_t position;

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);

	_it_should("refuse a cell that is taken", !game.play(&position, CELL_TOP_LEFT));
	_it_should("refuse a cell that is outside the board",
		!game.play(&position, (board_cell_t)CELL_COUNT));
	_it_should("leave the board alone",
		board_in_position(&position)->cells[CELL_TOP_LEFT] == VALUE_X);

	return NULL;
}

static char *test_it_stops_when_someone_wins(void)
{
	game_position_t position;

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);
	game.play(&position, CELL_CENTER);
	game.play(&position, CELL_TOP);
	game.play(&position, CELL_MIDDLE_LEFT);
	game.play(&position, CELL_TOP_RIGHT);

	_it_should("see that the game is over", game.over(&position));
	_it_should("say that the first player won", game.outcome(&position) == GAME_FIRST_WINS);
	_it_should("refuse another move", !game.play(&position, CELL_BOTTOM));

	return NULL;
}

static char *test_a_full_board_with_no_line_is_a_draw(void)
{
	game_position_t position;
	const board_cell_t moves[] = {
		CELL_TOP_LEFT, CELL_TOP, CELL_TOP_RIGHT, CELL_CENTER, CELL_MIDDLE_LEFT,
		CELL_MIDDLE_RIGHT, CELL_BOTTOM, CELL_BOTTOM_LEFT, CELL_BOTTOM_RIGHT
	};

	game.start(&position);

	for (int i = 0; i < 9; i++)
		game.play(&position, moves[i]);

	_it_should("see that the game is over", game.over(&position));
	_it_should("say that it was a draw", game.outcome(&position) == GAME_DRAW);

	return NULL;
}

static char *test_the_other_player_can_win(void)
{
	game_position_t position;

	game.start(&position);
	/* the first player takes the top left and the top middle, the other
	 * one takes the middle row */
	game.play(&position, CELL_TOP_LEFT);
	game.play(&position, CELL_CENTER);
	game.play(&position, CELL_TOP);
	game.play(&position, CELL_MIDDLE_LEFT);
	game.play(&position, CELL_BOTTOM_RIGHT);
	game.play(&position, CELL_MIDDLE_RIGHT);

	_it_should("see that the game is over", game.over(&position));
	_it_should("say that the second player won", game.outcome(&position) == GAME_SECOND_WINS);

	return NULL;
}

static char *test_the_features_are_the_board_as_the_player_sees_it(void)
{
	game_position_t position;
	double features[CELL_COUNT];

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);
	game.play(&position, CELL_CENTER);
	game.play(&position, CELL_TOP);

	game.features(&position, features);

	/* the player to move is the second one, so the O is its own mark and
	 * the X belongs to the other one */
	_it_should("give the player to move its own mark as one",
		features[CELL_CENTER] == 1.0);
	_it_should("give the other one as minus one", features[CELL_TOP_LEFT] == -1.0);
	_it_should("give an empty cell as nought", features[CELL_MIDDLE_RIGHT] == 0.0);

	return NULL;
}

static char *test_the_perfect_player_knows_what_a_position_is_worth(void)
{
	game_position_t position;

	oracle_start(&oracle);

	game.start(&position);

	/* neither player can win a tic-tac-toe board that is empty if the
	 * other one plays well, so it is worth a half to the player to move */
	_it_should("say the empty board is worth a half",
		oracle.worth(oracle.state, &position) == 0.5);

	/* the first player takes two cells of the top row, and the other one
	 * takes two cells of the middle row */
	game.play(&position, CELL_TOP_LEFT);
	game.play(&position, CELL_MIDDLE_LEFT);
	game.play(&position, CELL_TOP);
	game.play(&position, CELL_CENTER);

	/* the first player can take the top right and win */
	_it_should("say the position is worth one to the player to move",
		oracle.worth(oracle.state, &position) == 1.0);

	game.play(&position, CELL_TOP_RIGHT);

	/* and now the first player has a line of three, so the other one has
	 * lost the game */
	_it_should("say the position is worth nought to the player to move",
		oracle.worth(oracle.state, &position) == 0.0);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_the_perfect_player_agrees_with_the_game(void)
{
	game_position_t position;

	oracle_start(&oracle);

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);
	game.play(&position, CELL_CENTER);
	game.play(&position, CELL_TOP);
	game.play(&position, CELL_MIDDLE_LEFT);
	game.play(&position, CELL_TOP_RIGHT);

	_it_should("say the same as the game",
		oracle.outcome(oracle.state, &position) == game.outcome(&position));

	oracle_stop(&oracle);

	return NULL;
}

static char *test_it_offers_positions_to_look_at(void)
{
	game_position_t samples[GAME_MAX_SAMPLES];
	int count;

	count = game.samples(samples, GAME_MAX_SAMPLES);

	_it_should("offer the empty board", count > 1);
	_it_should("not offer more than were asked for", count <= GAME_MAX_SAMPLES);

	return NULL;
}

static char *run_tests()
{
	tictactoe_game(&game);
	tictactoe_oracle(&oracle);

	_run_test(test_a_game_starts_empty);
	_run_test(test_a_move_places_the_mark);
	_run_test(test_it_refuses_an_illegal_move);
	_run_test(test_it_stops_when_someone_wins);
	_run_test(test_a_full_board_with_no_line_is_a_draw);
	_run_test(test_the_other_player_can_win);
	_run_test(test_the_features_are_the_board_as_the_player_sees_it);
	_run_test(test_the_perfect_player_knows_what_a_position_is_worth);
	_run_test(test_the_perfect_player_agrees_with_the_game);
	_run_test(test_it_offers_positions_to_look_at);

	return NULL;
}
