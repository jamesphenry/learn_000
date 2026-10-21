#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include "board.h"
#include "policy.h"

typedef enum {
	GAME_X_WINS,
	GAME_O_WINS,
	GAME_DRAW
} game_result_t;

/* How many moves may be offered for a game. A game takes nine, and the
 * rest are room for the ones that were not allowed. */
#define GAME_MAX_ATTEMPTS 16

typedef struct {
	board_t board;
	int moves;
	board_cell_t history[CELL_COUNT];
	/* every move that was offered, whether it was allowed or not */
	board_cell_t attempts[GAME_MAX_ATTEMPTS];
	int attempt_count;
} game_t;

void game_init(game_t *game);
bool game_play_move(game_t *game, board_cell_t cell);
bool game_is_over(const game_t *game);
game_result_t game_result(const game_t *game);

/*
 * The worth of a result for the player who was to move: one if they won, a
 * half if the game was drawn and nought if they lost.
 *
 * The worth of a position is always in the terms of the player to move, so
 * that a model can learn one thing about the game rather than two.
 */
double game_worth_of_result(char player_to_move, game_result_t outcome);
char game_turn(const game_t *game);
game_result_t game_play(game_t *game, policy_t *x_policy, policy_t *o_policy);

/* The board as it was before each of the moves that were played. */
void game_positions(const game_t *game, board_t positions[CELL_COUNT]);

#endif // GAME_H
