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

/* How many positions the game can reach: the heaps only ever get smaller. */
#define NIM_MAX_POSITIONS ((NIM_START_HEAP_0 + 1) * (NIM_START_HEAP_1 + 1) * \
			    (NIM_START_HEAP_2 + 1))

/* How many bits a heap is written in: the biggest heap there is is seven. */
#define NIM_BITS 3

/* The biggest heap there really is: the game starts with heaps of three, four
 * and five, and a heap only ever gets smaller. */
#define NIM_REAL_MAX_HEAP NIM_START_HEAP_2

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
 * The nim-sum of a position: the exclusive or of the heaps. It is zero when
 * the position is a loss for the player to move, and not zero when it is a
 * win.
 */
int nim_sum(const nim_t *nim);

/*
 * The ways the heaps may be given to a model. They are the same game and the
 * same perfect player; only the numbers the model is given are different.
 */
typedef enum {
	/* the three sizes, as a fraction of the biggest heap the game allows */
	NIM_FEATURES_SIZES,
	/* the three sizes, as a fraction of the biggest heap there really is */
	NIM_FEATURES_REAL_RANGE,
	/* the nine bits of the three sizes */
	NIM_FEATURES_BITS,
	/* the three bit counts: how many of the heaps have each bit in them */
	NIM_FEATURES_COUNTS
} nim_features_t;

/*
 * The game of Nim, as the framework sees it.
 *
 * The heaps live in the state of a position, and the perfect player is the
 * one that plays the game perfectly and knows what every position is worth.
 */
void nim_game(game_t *game);

/*
 * The same game, with the heaps given to the model in another way.
 */
void nim_game_with_features(game_t *game, nim_features_t features);

void nim_oracle(oracle_t *oracle);

#endif // NIM_H
