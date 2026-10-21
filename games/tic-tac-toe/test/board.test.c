#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"

static char *test_cell_encodings(void)
{
	board_t board;

	board_init(&board);

	_it_should("encode X as +1", encode_mark('X') == VALUE_X);
	_it_should("encode O as -1", encode_mark('O') == VALUE_O);
	_it_should("encode an unknown mark as empty", encode_mark('Z') == VALUE_EMPTY);

	_it_should("decode +1 as X", decode_value(VALUE_X) == 'X');
	_it_should("decode -1 as O", decode_value(VALUE_O) == 'O');
	_it_should("decode 0 as an empty cell", decode_value(VALUE_EMPTY) == '.');

	_it_should("place the top left cell at row 0", cell_row(CELL_TOP_LEFT) == 0);
	_it_should("place the top right cell at row 0", cell_row(CELL_TOP_RIGHT) == 0);
	_it_should("place the middle left cell at row 1", cell_row(CELL_MIDDLE_LEFT) == 1);
	_it_should("place the bottom right cell at row 2", cell_row(CELL_BOTTOM_RIGHT) == 2);
	_it_should("place the top left cell at column 0", cell_col(CELL_TOP_LEFT) == 0);
	_it_should("place the middle left cell at column 0", cell_col(CELL_MIDDLE_LEFT) == 0);
	_it_should("place the top right cell at column 2", cell_col(CELL_TOP_RIGHT) == 2);
	_it_should("place the bottom right cell at column 2", cell_col(CELL_BOTTOM_RIGHT) == 2);

	_it_should("start every cell empty", board.cells[CELL_TOP_LEFT] == VALUE_EMPTY);
	_it_should("start the current player at X", board.current_player == 'X');

	place_mark(&board, CELL_CENTER, 'X');
	_it_should("store an X as +1", board.cells[CELL_CENTER] == VALUE_X);

	place_mark(&board, CELL_TOP, 'O');
	_it_should("store an O as -1", board.cells[CELL_TOP] == VALUE_O);

	return NULL;
}

static char *test_place_mark(void)
{
	board_t board;

	board_init(&board);

	/* Place X at the centre -> Success */
	_it_should("place X at the centre", place_mark(&board, CELL_CENTER, 'X'));
	_it_should("store X at the centre", board.cells[CELL_CENTER] == VALUE_X);

	/* Place O at the centre again -> Failure */
	_it_should("reject O at the centre when X is there", !place_mark(&board, CELL_CENTER, 'O'));
	_it_should("keep X at the centre", board.cells[CELL_CENTER] == VALUE_X);

	/* Place X one past the last cell -> Failure */
	_it_should("reject X past the last cell", !place_mark(&board, (board_cell_t)CELL_COUNT, 'X'));

	/* Place X before the first cell -> Failure */
	_it_should("reject X before the first cell", !place_mark(&board, (board_cell_t)-1, 'X'));

	/* Place O at the top left -> Success */
	_it_should("place O at the top left", place_mark(&board, CELL_TOP_LEFT, 'O'));
	_it_should("store O at the top left", board.cells[CELL_TOP_LEFT] == VALUE_O);

	return NULL;
}

static char *test_place_mark_invalid_player(void)
{
	board_t board;

	board_init(&board);

	_it_should("reject a mark that is neither X nor O", !place_mark(&board, CELL_TOP_LEFT, 'Z'));
	_it_should("leave the board untouched", board.cells[CELL_TOP_LEFT] == VALUE_EMPTY);

	return NULL;
}

static char *test_detect_win_row(void)
{
	board_t board;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_TOP_RIGHT, 'X');

	_it_should("detect a win across the top row", detect_win(&board, 'X'));
	_it_should("not report a win for the other player", !detect_win(&board, 'O'));

	return NULL;
}

static char *test_detect_win_column(void)
{
	board_t board;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'O');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'O');

	_it_should("detect a win down the left column", detect_win(&board, 'O'));
	_it_should("not report a win for the other player", !detect_win(&board, 'X'));

	return NULL;
}

static char *test_detect_win_diagonal(void)
{
	board_t board;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'X');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');

	_it_should("detect a win on the main diagonal", detect_win(&board, 'X'));
	_it_should("not report a win for the other player", !detect_win(&board, 'O'));

	return NULL;
}

static char *test_detect_win_anti_diagonal(void)
{
	board_t board;

	board_init(&board);
	place_mark(&board, CELL_TOP_RIGHT, 'O');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'O');

	_it_should("detect a win on the anti diagonal", detect_win(&board, 'O'));
	_it_should("not report a win for the other player", !detect_win(&board, 'X'));

	return NULL;
}

static char *test_detect_win_no_win(void)
{
	board_t board;

	board_init(&board);
	_it_should("report no win on an empty board", !detect_win(&board, 'X'));
	_it_should("report no win on an empty board", !detect_win(&board, 'O'));
	_it_should("report no win for an unknown mark", !detect_win(&board, 'Z'));

	/* Two in a row, but the third cell of the line is still empty. */
	board_init(&board);
	board.cells[CELL_TOP_LEFT] = VALUE_X;
	board.cells[CELL_TOP] = VALUE_X;
	board.cells[CELL_CENTER] = VALUE_O;
	board.cells[CELL_MIDDLE_RIGHT] = VALUE_X;
	board.cells[CELL_BOTTOM_LEFT] = VALUE_O;
	_it_should("report no win for a broken line", !detect_win(&board, 'X'));
	_it_should("report no win for a broken line", !detect_win(&board, 'O'));

	_it_should("report no win for an unknown mark", !detect_win(&board, 'Z'));

	return NULL;
}

static char *test_detect_win_every_line(void)
{
	/* the eight lines of the board, written out here rather than taken
	 * from the board module, so that the test does not simply believe
	 * whatever the board module happens to think a line is */
	static const board_cell_t lines[8][3] = {
		{CELL_TOP_LEFT, CELL_TOP, CELL_TOP_RIGHT},
		{CELL_MIDDLE_LEFT, CELL_CENTER, CELL_MIDDLE_RIGHT},
		{CELL_BOTTOM_LEFT, CELL_BOTTOM, CELL_BOTTOM_RIGHT},
		{CELL_TOP_LEFT, CELL_MIDDLE_LEFT, CELL_BOTTOM_LEFT},
		{CELL_TOP, CELL_CENTER, CELL_BOTTOM},
		{CELL_TOP_RIGHT, CELL_MIDDLE_RIGHT, CELL_BOTTOM_RIGHT},
		{CELL_TOP_LEFT, CELL_CENTER, CELL_BOTTOM_RIGHT},
		{CELL_TOP_RIGHT, CELL_CENTER, CELL_BOTTOM_LEFT}
	};
	static char message[64];
	board_t board;

	for (int line = 0; line < 8; line++) {
		board_init(&board);
		for (int i = 0; i < 3; i++)
			place_mark(&board, lines[line][i], 'X');

		snprintf(message, sizeof(message), "detect a win on line %d", line);
		_it_should(message, detect_win(&board, 'X'));
	}

	return NULL;
}

static char *test_detect_current_win(void)
{
	board_t board;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_TOP_RIGHT, 'X');

	board.current_player = 'X';
	_it_should("detect a win for the current player", detect_current_win(&board));

	board.current_player = 'O';
	_it_should("not detect a win for the other player", !detect_current_win(&board));

	return NULL;
}

static char *test_board_is_full(void)
{
	board_t board;

	board_init(&board);
	_it_should("report an empty board as not full", !board_is_full(&board));

	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_CENTER, 'X');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM, 'O');
	_it_should("report a board with one empty cell as not full", !board_is_full(&board));

	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');
	_it_should("report a board with no empty cells as full", board_is_full(&board));

	return NULL;
}

static char *test_detect_draw(void)
{
	board_t board;

	board_init(&board);
	_it_should("report no draw on an empty board", !detect_draw(&board));

	/* A full board where both players have a line is not a draw. */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM, 'O');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');
	_it_should("report no draw when a player has won", !detect_draw(&board));

	/* A full board where only X has won is not a draw. */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');
	place_mark(&board, CELL_CENTER, 'X');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'O');
	place_mark(&board, CELL_BOTTOM, 'O');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');
	_it_should("report no draw when only X has won", !detect_draw(&board));

	/* A full board where only O has won is not a draw. */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'O');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'X');
	place_mark(&board, CELL_BOTTOM, 'X');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'O');
	_it_should("report no draw when the other player has won", !detect_draw(&board));

	/* A full board with no winning line is a draw. */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'O');
	place_mark(&board, CELL_BOTTOM, 'X');
	place_mark(&board, CELL_BOTTOM_RIGHT, 'X');
	_it_should("report a draw on a full board with no winner", detect_draw(&board));

	/* The same board, one cell short of full, is not a draw yet. */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	place_mark(&board, CELL_TOP_RIGHT, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_MIDDLE_RIGHT, 'O');
	place_mark(&board, CELL_BOTTOM_LEFT, 'O');
	place_mark(&board, CELL_BOTTOM, 'X');
	_it_should("report no draw while a cell is still empty", !detect_draw(&board));

	return NULL;
}

/* Many games in flight at once must not share any state. */
static char *test_boards_are_independent(void)
{
	board_t games[100];
	int independent = 1;

	for (int i = 0; i < 100; i++) {
		board_init(&games[i]);
		place_mark(&games[i], (board_cell_t)(i % CELL_COUNT), 'X');
	}

	for (int i = 0; i < 100; i++) {
		int marks = 0;

		for (int cell = 0; cell < CELL_COUNT; cell++) {
			if (games[i].cells[cell] != VALUE_EMPTY)
				marks++;
		}

		if (marks != 1 || games[i].cells[i % CELL_COUNT] != VALUE_X)
			independent = 0;
	}

	_it_should("keep 100 boards independent of each other", independent);
	_it_should("report no win for a board with a single mark", !detect_win(&games[0], 'X'));

	return NULL;
}

static char *test_board_to_inputs(void)
{
	board_t board;
	double inputs[CELL_COUNT];

	board_init(&board);
	board_to_inputs(&board, inputs);

	_it_should("give an empty cell as zero", inputs[CELL_CENTER] == 0.0);

	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	board_to_inputs(&board, inputs);

	_it_should("give an X as 1", inputs[CELL_TOP_LEFT] == 1.0);
	_it_should("give an O as -1", inputs[CELL_CENTER] == -1.0);
	_it_should("give an empty cell as zero", inputs[CELL_BOTTOM] == 0.0);
	_it_should("leave the other cells alone", inputs[CELL_BOTTOM_RIGHT] == 0.0);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_cell_encodings);
	_run_test(test_place_mark);
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
	_run_test(test_boards_are_independent);
	_run_test(test_board_to_inputs);

	return NULL;
}
