#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"
#include "../src/minimax.c"

static minimax_t perfect;

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

/* X at the top left, O in the middle, X at the top middle. It is X's turn
 * and X can take the top right to win. */
static void build_a_position_with_a_win(board_t *board)
{
	board_init(board);
	place_mark(board, CELL_TOP_LEFT, 'X');
	place_mark(board, CELL_CENTER, 'O');
	place_mark(board, CELL_TOP, 'X');
	place_mark(board, CELL_MIDDLE_LEFT, 'O');
}

static char *test_the_empty_board_is_a_draw(void)
{
	board_t board;
	minimax_result_t result;

	board_init(&board);
	result = minimax_at(&perfect, &board);

	_it_should("be worth a draw", result.value == 0);
	_it_should("still have a move to make", result.move >= 0 && result.move < CELL_COUNT);
	_it_should("say the player to move has a half of a chance", nearly_equal(minimax_worth(&perfect, &board), 0.5));

	return NULL;
}

static char *test_a_win_that_is_there_to_be_taken(void)
{
	board_t board;
	minimax_result_t result;

	build_a_position_with_a_win(&board);
	result = minimax_at(&perfect, &board);

	_it_should("be worth a win", result.value == 1);
	_it_should("take the cell that wins", result.move == CELL_TOP_RIGHT);
	_it_should("say the player to move is certain to win", nearly_equal(minimax_worth(&perfect, &board), 1.0));

	return NULL;
}

static char *test_a_threat_that_has_to_be_blocked(void)
{
	board_t board;
	minimax_result_t result;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_TOP, 'X');

	/* X has two on the top row, so it is O's turn and O has to block */
	result = minimax_at(&perfect, &board);

	_it_should("be worth a draw", result.value == 0);
	_it_should("block the top right", result.move == CELL_TOP_RIGHT);
	_it_should("say the player to move has a half of a chance", nearly_equal(minimax_worth(&perfect, &board), 0.5));

	return NULL;
}

static char *test_a_game_that_is_already_over(void)
{
	board_t board;
	minimax_result_t result;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'X');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');

	/* O has the middle row */
	result = minimax_at(&perfect, &board);

	_it_should("have no move to make", result.move == -1);
	_it_should("be worth a loss for whoever is to move", result.value == -1);
	_it_should("say the player to move will not win", nearly_equal(minimax_worth(&perfect, &board), 0.0));

	return NULL;
}

static char *test_the_minimax_never_loses(void)
{
	board_t board;
	int moves = 0;

	board_init(&board);

	while (moves < CELL_COUNT) {
		minimax_result_t result = minimax_at(&perfect, &board);

		if (result.move < 0) break;

		/* an illegal move would be refused and the board would not
		 * change, so the game would never finish */
		if (!place_mark(&board, (board_cell_t)result.move, board.current_player))
			break;

		board.current_player = (board.current_player == 'X') ? 'O' : 'X';
		moves++;
	}

	_it_should("fill the board", board_is_full(&board));
	_it_should("take nine moves", moves == CELL_COUNT);
	_it_should("draw the game", !detect_win(&board, 'X') && !detect_win(&board, 'O'));

	return NULL;
}

static char *test_the_move_is_always_legal(void)
{
	board_t board;

	/* every position reachable in a game */
	for (int first = 0; first < CELL_COUNT; first++) {
		for (int second = 0; second < CELL_COUNT; second++) {
			if (second == first) continue;

			board_init(&board);
			place_mark(&board, (board_cell_t)first, 'X');
			place_mark(&board, (board_cell_t)second, 'O');

			minimax_result_t result = minimax_at(&perfect, &board);

			if (result.move < 0) continue;

			_it_should("pick a cell that is empty",
				result.move < CELL_COUNT && board.cells[result.move] == VALUE_EMPTY);
		}
	}

	return NULL;
}

static char *test_a_loss_for_the_player_to_move(void)
{
	board_t board;
	minimax_result_t result;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_CENTER, 'X');

	/* X threatens the bottom right on the diagonal and the bottom left on
	 * the other one, and it is O's turn */
	result = minimax_at(&perfect, &board);

	_it_should("be worth a loss for whoever is to move", result.value == -1);
	_it_should("say the player to move will not win", nearly_equal(minimax_worth(&perfect, &board), 0.0));

	return NULL;
}

/*
 * The perfect player keeps what it has worked out, so it has to give the
 * same answers whether it has worked a position out before or not.
 */
static char *test_the_answers_do_not_change_once_it_has_worked_things_out(void)
{
	minimax_t another;
	board_t board;
	minimax_result_t first;
	minimax_result_t second;

	/* a second perfect player, with nothing worked out */
	minimax_init(&another);

	/* and the first one, which has already worked out the empty board */
	board_init(&board);
	first = minimax_at(&perfect, &board);
	second = minimax_at(&another, &board);

	_it_should("give the same answer to the same position",
		first.value == second.value && first.move == second.move);

	/* and the same again once it has worked out something else */
	build_a_position_with_a_win(&board);
	first = minimax_at(&perfect, &board);
	second = minimax_at(&another, &board);

	_it_should("still give the same answer",
		first.value == second.value && first.move == second.move);

	return NULL;
}

static char *run_tests()
{
	minimax_init(&perfect);

	_run_test(test_the_empty_board_is_a_draw);
	_run_test(test_a_win_that_is_there_to_be_taken);
	_run_test(test_a_threat_that_has_to_be_blocked);
	_run_test(test_a_loss_for_the_player_to_move);
	_run_test(test_a_game_that_is_already_over);
	_run_test(test_the_minimax_never_loses);
	_run_test(test_the_move_is_always_legal);
	_run_test(test_the_answers_do_not_change_once_it_has_worked_things_out);

	return NULL;
}
