#ifndef MINIMAX_H
#define MINIMAX_H

#include <stdbool.h>
#include "board.h"

/*
 * The best play for whoever is to move, and what the position is worth:
 * +1 a win for the player to move, 0 a draw, -1 a loss.
 */
typedef struct {
	int move;   /* the cell to play, or -1 when the game is over */
	int value;  /* +1, 0 or -1 */
} minimax_result_t;

/* Every position there is, three to the ninth: 19683 of them. */
#define MINIMAX_STATES 19683

/*
 * The perfect player, and what it has worked out so far.
 *
 * The worth of a position does not depend on how it was reached, so working
 * one out once is enough and it can be kept. The table is large, so it
 * belongs to whoever is asking the questions rather than to the perfect
 * player itself, and that keeps the perfect player free of any state of
 * its own.
 */
typedef struct {
	int values[MINIMAX_STATES];
} minimax_t;

/* Starts the perfect player off with nothing worked out. */
void minimax_init(minimax_t *minimax);

/*
 * The best move for whoever is to move, and the value of the position.
 *
 * The rules are worked out here rather than taken from the board module, so
 * that a referee can disagree with the game it is refereeing.
 */
minimax_result_t minimax_at(minimax_t *minimax, const board_t *board);

/*
 * Who has won: 'X', 'O', or 0 if nobody has. Worked out from the lines
 * alone, so it needs nothing the perfect player has worked out.
 */
char minimax_winner(const board_t *board);

/*
 * The chance that X wins from this position if both play perfectly: one, a
 * half or nought.
 */
double minimax_x_win_chance(minimax_t *minimax, const board_t *board);

#endif // MINIMAX_H
