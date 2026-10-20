#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

int _retval = 0;
const char *_greet = NULL;
#define puts(s) (_greet = (s), _retval)

#include "../src/main.c"

static char *test_greet(void)
{
	char *s = "hello";

	_retval = rand();
	_it_should("return zero", _retval == puts(s));
	_it_should("output greeting", s == _greet);

	return NULL;
}

static void reset_board(void)
{
	char initial[3][3] = {
		{'1', '2', '3'},
		{'4', '5', '6'},
		{'7', '8', '9'}
	};

	memcpy(board, initial, sizeof(board));
}

static char *test_board(void)
{
	reset_board();

	/* Place X at (1, 1) -> Success */
	_it_should("place X at (1, 1)", place_mark(1, 1, 'X'));
	_it_should("store X at (1, 1)", board[1][1] == 'X');

	/* Place O at (1, 1) again -> Failure */
	_it_should("reject O at (1, 1) when X is there", !place_mark(1, 1, 'O'));
	_it_should("keep X at (1, 1)", board[1][1] == 'X');

	/* Place X at (-1, 0) -> Failure */
	_it_should("reject X at (-1, 0)", !place_mark(-1, 0, 'X'));

	/* Place O at (0, 3) -> Failure */
	_it_should("reject O at (0, 3)", !place_mark(0, 3, 'O'));

	/* Place X at (3, 0) -> Failure */
	_it_should("reject X at (3, 0)", !place_mark(3, 0, 'X'));

	/* Place O at (0, 0) -> Success */
	_it_should("place O at (0, 0)", place_mark(0, 0, 'O'));
	_it_should("store O at (0, 0)", board[0][0] == 'O');

	return NULL;
}

static char *test_place_mark_invalid_player(void)
{
	reset_board();

	_it_should("reject a mark that is neither X nor O", !place_mark(0, 0, 'Z'));
	_it_should("leave the board untouched", board[0][0] == '1');

	return NULL;
}

static char *test_detect_win_row(void)
{
	reset_board();
	place_mark(0, 0, 'X');
	place_mark(0, 1, 'X');
	place_mark(0, 2, 'X');

	_it_should("detect a win across the top row", detect_win('X'));
	_it_should("not report a win for the other player", !detect_win('O'));

	return NULL;
}

static char *test_detect_win_column(void)
{
	reset_board();
	place_mark(0, 0, 'O');
	place_mark(1, 0, 'O');
	place_mark(2, 0, 'O');

	_it_should("detect a win down the left column", detect_win('O'));
	_it_should("not report a win for the other player", !detect_win('X'));

	return NULL;
}

static char *test_detect_win_diagonal(void)
{
	reset_board();
	place_mark(0, 0, 'X');
	place_mark(1, 1, 'X');
	place_mark(2, 2, 'X');

	_it_should("detect a win on the main diagonal", detect_win('X'));
	_it_should("not report a win for the other player", !detect_win('O'));

	return NULL;
}

static char *test_detect_win_anti_diagonal(void)
{
	reset_board();
	place_mark(0, 2, 'O');
	place_mark(1, 1, 'O');
	place_mark(2, 0, 'O');

	_it_should("detect a win on the anti diagonal", detect_win('O'));
	_it_should("not report a win for the other player", !detect_win('X'));

	return NULL;
}

static char *test_detect_win_no_win(void)
{
	reset_board();
	_it_should("report no win on an empty board", !detect_win('X'));
	_it_should("report no win on an empty board", !detect_win('O'));

	/* Two in a row, but the third cell of the line is still empty. */
	reset_board();
	board[0][0] = 'X';
	board[0][1] = 'X';
	board[1][1] = 'O';
	board[1][2] = 'X';
	board[2][0] = 'O';
	_it_should("report no win for a broken line", !detect_win('X'));
	_it_should("report no win for a broken line", !detect_win('O'));

	_it_should("report no win for an unknown mark", !detect_win('Z'));

	return NULL;
}

/* Every winning line on a 3x3 board, as flat cell indices (row * 3 + col). */
static const int win_lines[8][3] = {
	{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
	{0, 3, 6}, {1, 4, 7}, {2, 5, 8},
	{0, 4, 8}, {2, 4, 6}
};

static char *test_detect_win_every_line(void)
{
	static char message[64];

	for (int line = 0; line < 8; line++) {
		reset_board();
		for (int i = 0; i < 3; i++) {
			int cell = win_lines[line][i];
			place_mark(cell / 3, cell % 3, 'X');
		}

		snprintf(message, sizeof(message), "detect a win on line %d", line);
		_it_should(message, detect_win('X'));
	}

	return NULL;
}

static char *test_detect_current_win(void)
{
	char saved = current_player;

	reset_board();
	place_mark(0, 0, 'X');
	place_mark(0, 1, 'X');
	place_mark(0, 2, 'X');

	current_player = 'X';
	_it_should("detect a win for the current player", detect_current_win());

	current_player = 'O';
	_it_should("not detect a win for the other player", !detect_current_win());

	current_player = saved;
	return NULL;
}

static char *test_board_is_full(void)
{
	reset_board();
	_it_should("report an empty board as not full", !board_is_full());

	place_mark(0, 0, 'X');
	place_mark(0,  1, 'O');
	place_mark(0, 2, 'X');
	place_mark(1, 0, 'O');
	place_mark(1, 1, 'X');
	place_mark(1, 2, 'O');
	place_mark(2, 0, 'X');
	place_mark(2, 1, 'O');
	_it_should("report a board with one empty cell as not full", !board_is_full());

	place_mark(2, 2, 'X');
	_it_should("report a board with no empty cells as full", board_is_full());

	return NULL;
}

static char *test_detect_draw(void)
{
	reset_board();
	_it_should("report no draw on an empty board", !detect_draw());

	/* A full board where both players have a line is not a draw. */
	reset_board();
	place_mark(0, 0, 'X');
	place_mark(0, 1, 'X');
	place_mark(0, 2, 'X');
	place_mark(1, 0, 'O');
	place_mark(1, 1, 'O');
	place_mark(1, 2, 'O');
	place_mark(2, 0, 'X');
	place_mark(2, 1, 'O');
	place_mark(2, 2, 'X');
	_it_should("report no draw when a player has won", !detect_draw());

	/* A full board where only X has won is not a draw. */
	reset_board();
	place_mark(0, 0, 'X');
	place_mark(0, 1, 'X');
	place_mark(0, 2, 'X');
	place_mark(1, 0, 'O');
	place_mark(1, 1, 'X');
	place_mark(1, 2, 'O');
	place_mark(2, 0, 'O');
	place_mark(2, 1, 'O');
	place_mark(2, 2, 'X');
	_it_should("report no draw when only X has won", !detect_draw());

	/* A full board where only O has won is not a draw. */
	reset_board();
	place_mark(0, 0, 'O');
	place_mark(0, 1, 'X');
	place_mark(0, 2, 'X');
	place_mark(1, 0, 'X');
	place_mark(1, 1, 'O');
	place_mark(1, 2, 'O');
	place_mark(2, 0, 'X');
	place_mark(2, 1, 'X');
	place_mark(2, 2, 'O');
	_it_should("report no draw when the other player has won", !detect_draw());

	/* A full board with no winning line is a draw. */
	reset_board();
	place_mark(0, 0, 'X');
	place_mark(0, 1, 'O');
	place_mark(0, 2, 'X');
	place_mark(1, 1, 'O');
	place_mark(1, 0, 'X');
	place_mark(1, 2, 'O');
	place_mark(2, 1, 'X');
	place_mark(2, 0, 'O');
	place_mark(2, 2, 'X');
	_it_should("report a draw on a full board with no winner", detect_draw());

	/* The same board, one cell short of full, is not a draw yet. */
	reset_board();
	place_mark(0, 0, 'X');
	place_mark(0, 1, 'O');
	place_mark(0, 2, 'X');
	place_mark(1, 1, 'O');
	place_mark(1, 0, 'X');
	place_mark(1, 2, 'O');
	place_mark(2, 1, 'X');
	place_mark(2, 0, 'O');
	_it_should("report no draw while a cell is still empty", !detect_draw());

	return NULL;
}

static char *run_tests()
{
	_run_test(test_greet);
	_run_test(test_board);
	_run_test(test_place_mark_invalid_player);
	_run_test(test_detect_win_row);
	_run_test(test_detect_win_column);
	_run_test(test_detect_win_diagonal);
	_run_test(test_detect_win_anti_diagonal);
	_run_test(test_detect_win_no_win);
	_run_test(test_detect_win_every_line);
	_run_test(test_detect_current_win);
	_run_test(test_board_is_full);
	_run_test(test_detect_draw);

	return NULL;
}
