#ifndef TICTACTOE_H
#define TICTACTOE_H

#include <stdbool.h>
#include "board.h"
#include "game.h"
#include "minimax.h"

/*
 * The game of tic-tac-toe, as the framework sees it.
 *
 * The board lives in the state of a position, and the perfect player is the
 * one that plays the game perfectly and knows what every position is worth.
 */
void tictactoe_game(game_t *game);
void tictactoe_oracle(oracle_t *oracle);

#endif // TICTACTOE_H
