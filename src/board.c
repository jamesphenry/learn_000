#include <stdio.h>
#include "board.h"

void board_init(board_t *board)
{
	board->current_player = 'X';

	for (int cell = 0; cell < CELL_COUNT; cell++)
		board->cells[cell] = VALUE_EMPTY;
}

int cell_row(board_cell_t cell)
{
	return (int)cell / BOARD_SIZE;
}

int cell_col(board_cell_t cell)
{
	return (int)cell % BOARD_SIZE;
}

cell_value_t encode_mark(char player_mark)
{
	if (player_mark == 'X') return VALUE_X;
	if (player_mark == 'O') return VALUE_O;

	return VALUE_EMPTY;
}

char decode_value(cell_value_t value)
{
	if (value == VALUE_X) return 'X';
	if (value == VALUE_O) return 'O';

	return '.';
}

void print_board(const board_t *board)
{
	for (int row = 0; row < BOARD_SIZE; row++) {
		printf(" %c | %c | %c \n",
			decode_value(board->cells[row * BOARD_SIZE + 0]),
			decode_value(board->cells[row * BOARD_SIZE + 1]),
			decode_value(board->cells[row * BOARD_SIZE + 2]));

		// Only print the divider after the first and second rows
		if (row < 2) {
			printf("---+---+---\n");
		}
	}
}

void board_to_inputs(const board_t *board, double out[CELL_COUNT])
{
	for (int cell = 0; cell < CELL_COUNT; cell++)
		out[cell] = (double)board->cells[cell];
}

bool place_mark(board_t *board, board_cell_t cell, char player_mark)
{
	if (cell < 0 || cell >= CELL_COUNT) return false;
	if (player_mark != 'X' && player_mark != 'O') return false;
	if (board->cells[cell] != VALUE_EMPTY) return false;

	board->cells[cell] = encode_mark(player_mark);
	return true;
}

static const board_cell_t win_lines[8][3] = {
	{CELL_TOP_LEFT, CELL_TOP, CELL_TOP_RIGHT},
	{CELL_MIDDLE_LEFT, CELL_CENTER, CELL_MIDDLE_RIGHT},
	{CELL_BOTTOM_LEFT, CELL_BOTTOM, CELL_BOTTOM_RIGHT},
	{CELL_TOP_LEFT, CELL_MIDDLE_LEFT, CELL_BOTTOM_LEFT},
	{CELL_TOP, CELL_CENTER, CELL_BOTTOM},
	{CELL_TOP_RIGHT, CELL_MIDDLE_RIGHT, CELL_BOTTOM_RIGHT},
	{CELL_TOP_LEFT, CELL_CENTER, CELL_BOTTOM_RIGHT},
	{CELL_TOP_RIGHT, CELL_CENTER, CELL_BOTTOM_LEFT}
};

bool detect_win(const board_t *board, char player_mark)
{
	if (player_mark != 'X' && player_mark != 'O') return false;

	cell_value_t value = encode_mark(player_mark);

	for (int line = 0; line < 8; line++) {
		if (board->cells[win_lines[line][0]] == value &&
		    board->cells[win_lines[line][1]] == value &&
		    board->cells[win_lines[line][2]] == value)
			return true;
	}

	return false;
}

bool detect_current_win(const board_t *board)
{
	return detect_win(board, board->current_player);
}

bool board_is_full(const board_t *board)
{
	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (board->cells[cell] == VALUE_EMPTY)
			return false;
	}

	return true;
}

bool detect_draw(const board_t *board)
{
	if (!board_is_full(board)) return false;

	return !detect_win(board, 'X') && !detect_win(board, 'O');
}
