#ifndef JUDGE_H
#define JUDGE_H

#include <stdbool.h>
#include "board.h"
#include "game.h"

/*
 * What a referee made of a game.
 */
typedef struct {
	int moves;        /* the moves that were played */
	int attempts;     /* the moves that were offered */
	int unexpected;   /* the moves that should not have been allowed */
	int result_wrong; /* the two reckonings of the result disagreed */
} judgement_t;

/*
 * Whether a move would be allowed right now: the cell has to be on the
 * board, it has to be empty, and the game has to be still running.
 */
bool judge_move_is_allowed(const game_t *game, board_cell_t move);

/*
 * Checks a game over. Every move that was offered is played again on a
 * fresh board, and each one is checked as it was offered, so that a move
 * that was not allowed is found even though the game never applied it.
 *
 * The result is worked out twice, once by the board module and once by the
 * perfect player, and the two are compared.
 */
bool judge_game(const game_t *game, judgement_t *verdict);

#endif // JUDGE_H
