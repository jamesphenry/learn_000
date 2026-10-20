#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>

#define BOARD_SIZE 3

/*
 * The position of a cell on the board, in row major order. The numeric value
 * is the index into board_t.cells, and the order a model would predict them.
 */
typedef enum {
	CELL_TOP_LEFT = 0,
	CELL_TOP,
	CELL_TOP_RIGHT,
	CELL_MIDDLE_LEFT,
	CELL_CENTER,
	CELL_MIDDLE_RIGHT,
	CELL_BOTTOM_LEFT,
	CELL_BOTTOM,
	CELL_BOTTOM_RIGHT,
	CELL_COUNT = 9
} board_cell_t;

/*
 * The value a cell holds, encoded so a board is a plain vector of numbers:
 * empty is 0, X is +1 and O is -1.
 */
typedef enum {
	VALUE_EMPTY = 0,
	VALUE_X = 1,
	VALUE_O = -1
} cell_value_t;

/*
 * A single game state. Everything the game needs lives here, so any number
 * of games can be in flight at once without sharing state.
 */
typedef struct {
	cell_value_t cells[CELL_COUNT];
	char current_player;
} board_t;

void board_init(board_t *board);
int cell_row(board_cell_t cell);
int cell_col(board_cell_t cell);
cell_value_t encode_mark(char player_mark);
char decode_value(cell_value_t value);
void print_board(const board_t *board);
bool place_mark(board_t *board, board_cell_t cell, char player_mark);
bool detect_win(const board_t *board, char player_mark);
bool detect_current_win(const board_t *board);
bool board_is_full(const board_t *board);
bool detect_draw(const board_t *board);

#endif // BOARD_H
