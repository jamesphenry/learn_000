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

/*
 * The best move for whoever is to move, and the value of the position.
 *
 * The rules are worked out here rather than taken from the board module, so
 * that a referee can disagree with the game it is refereeing.
 */
minimax_result_t minimax_at(const board_t *board);

/*
 * Who has won: 'X', 'O', or 0 if nobody has. Worked out by the minimax's
 * own reckoning, so that it can be compared with the board module's.
 */
char minimax_winner(const board_t *board);

/*
 * The chance that X wins from this position if both play perfectly: one, a
 * half or nought.
 */
double minimax_x_win_chance(const board_t *board);

#endif // MINIMAX_H
