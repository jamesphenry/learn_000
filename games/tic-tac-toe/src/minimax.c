#include <stddef.h>
#include "minimax.h"

/* The board in the minimax's own terms, so that the rules it plays by are
 * its own. */
#define MINIMAX_CELLS 9
#define MINIMAX_EMPTY 0
#define MINIMAX_X 1
#define MINIMAX_O 2

/* The eight lines of three. */
static const int minimax_lines[8][3] = {
	{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
	{0, 3, 6}, {1, 4, 7}, {2, 5, 8},
	{0, 4, 8}, {2, 4, 6}
};

static void minimax_read(const board_t *board, int cells[MINIMAX_CELLS])
{
	for (int cell = 0; cell < MINIMAX_CELLS; cell++) {
		if (board->cells[cell] == VALUE_X) cells[cell] = MINIMAX_X;
		else if (board->cells[cell] == VALUE_O) cells[cell] = MINIMAX_O;
		else cells[cell] = MINIMAX_EMPTY;
	}
}

/* X goes first, so X is to move when the two have played the same number. */
static int minimax_turn(const int cells[MINIMAX_CELLS])
{
	int x = 0;
	int o = 0;

	for (int cell = 0; cell < MINIMAX_CELLS; cell++) {
		if (cells[cell] == MINIMAX_X) x++;
		else if (cells[cell] == MINIMAX_O) o++;
	}

	return (x == o) ? MINIMAX_X : MINIMAX_O;
}

/* The player who has won, or MINIMAX_EMPTY if nobody has. */
static int minimax_winner_of(const int cells[MINIMAX_CELLS])
{
	for (int line = 0; line < 8; line++) {
		int first = cells[minimax_lines[line][0]];

		if (first != MINIMAX_EMPTY &&
		    first == cells[minimax_lines[line][1]] &&
		    first == cells[minimax_lines[line][2]])
			return first;
	}

	return MINIMAX_EMPTY;
}

/* Where a position sits in the table of every position there is. */
static int minimax_index(const int cells[MINIMAX_CELLS])
{
	int index = 0;

	for (int cell = 0; cell < MINIMAX_CELLS; cell++)
		index = index * 3 + cells[cell];

	return index;
}

/*
 * The value of the position for the player to move. What has been worked
 * out is kept in the table, with two added to it so that a zero means it has
 * not been worked out yet.
 */
static int minimax_score(int cells[MINIMAX_CELLS], int turn, int *values)
{
	int index = minimax_index(cells);

	if (values[index] != 0) return values[index] - 2;

	int winner = minimax_winner_of(cells);
	int best = -2;
	int other = (turn == MINIMAX_X) ? MINIMAX_O : MINIMAX_X;

	if (winner != MINIMAX_EMPTY) {
		best = (winner == turn) ? 1 : -1;
	} else {
		for (int cell = 0; cell < MINIMAX_CELLS; cell++) {
			if (cells[cell] != MINIMAX_EMPTY) continue;

			cells[cell] = turn;
			int score = -minimax_score(cells, other, values);
			cells[cell] = MINIMAX_EMPTY;

			if (score > best) best = score;
			if (best == 1) break;
		}

		/* nothing left to play: a draw */
		if (best == -2) best = 0;
	}

	values[index] = best + 2;

	return best;
}

char minimax_winner(const board_t *board)
{
	int cells[MINIMAX_CELLS];
	int winner;

	minimax_read(board, cells);
	winner = minimax_winner_of(cells);

	if (winner == MINIMAX_X) return 'X';
	if (winner == MINIMAX_O) return 'O';

	return 0;
}

void minimax_init(minimax_t *minimax)
{
	if (minimax == NULL) return;

	for (int i = 0; i < MINIMAX_STATES; i++)
		minimax->values[i] = 0;
}

minimax_result_t minimax_at(minimax_t *minimax, const board_t *board)
{
	int cells[MINIMAX_CELLS];
	minimax_result_t result = { -1, 0 };
	int turn;
	int winner;

	if (minimax == NULL || board == NULL) return result;

	minimax_read(board, cells);
	turn = minimax_turn(cells);
	winner = minimax_winner_of(cells);

	if (winner != MINIMAX_EMPTY) {
		result.value = (winner == turn) ? 1 : -1;

		return result;
	}

	int best = -2;
	int other = (turn == MINIMAX_X) ? MINIMAX_O : MINIMAX_X;

	for (int cell = 0; cell < MINIMAX_CELLS; cell++) {
		if (cells[cell] != MINIMAX_EMPTY) continue;

		cells[cell] = turn;
		int score = -minimax_score(cells, other, minimax->values);
		cells[cell] = MINIMAX_EMPTY;

		if (score > best) {
			best = score;
			result.move = cell;
		}
	}

	if (best != -2) result.value = best;

	return result;
}

double minimax_worth(minimax_t *minimax, const board_t *board)
{
	minimax_result_t result = minimax_at(minimax, board);

	/* the value is already from the point of view of whoever is to move:
	 * one for a win, nought for a draw and minus one for a loss */
	return ((double)result.value + 1.0) / 2.0;
}
