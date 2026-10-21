#ifndef NIM_H
#define NIM_H

#include <stdbool.h>
#include "game.h"

/* How many heaps there are, and how many stones a heap may hold. */
#define NIM_HEAPS 3
#define NIM_MAX_HEAP 7

/* The heaps a game starts with. */
#define NIM_START_HEAP_0 3
#define NIM_START_HEAP_1 4
#define NIM_START_HEAP_2 5

/*
 * A move takes stones from one heap: it says which heap, and how many
 * stones to take from it.
 */
#define NIM_MOVE(heap, count) ((heap) * (NIM_MAX_HEAP + 1) + (count))

/*
 * The state of a Nim game: the heaps, and whose turn it is. The heaps do
 * not belong to a player, so they are the same whoever is to move.
 */
typedef struct {
	int heaps[NIM_HEAPS];
	int turn;
} nim_t;

/* The state of a position. */
const nim_t *nim_state_of(const game_position_t *position);

/* The heap a move is from, and how many stones it takes from it. */
int nim_heap_of(game_move_t move);
int nim_count_of(game_move_t move);

/*
 * The game of Nim, as the framework sees it.
 *
 * The heaps live in the state of a position, and the perfect player is the
 * one that plays the game perfectly and knows what every position is worth.
 */
void nim_game(game_t *game);
void nim_oracle(oracle_t *oracle);

#endif // NIM_H
