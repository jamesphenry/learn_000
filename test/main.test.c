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

static char *run_tests()
{
	_run_test(test_greet);
	_run_test(test_board);
	_run_test(test_place_mark_invalid_player);

	return NULL;
}
