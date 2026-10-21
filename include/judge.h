#ifndef JUDGE_H
#define JUDGE_H

#include <stdbool.h>
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
 * Checks a game over. Every move that was offered is played again from the
 * start, and each one is checked as it was offered, so that a move that was
 * not allowed is found even though the game never applied it.
 *
 * The result is worked out twice, once by the game and once by the perfect
 * player, and the two are compared.
 */
bool judge_game(const game_t *game, const oracle_t *oracle, const game_trace_t *trace,
                judgement_t *verdict);

#endif // JUDGE_H
