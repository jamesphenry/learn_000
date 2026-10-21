#ifndef POLICY_H
#define POLICY_H

#include <stdbool.h>
#include "board.h"
#include "network.h"
#include "rng.h"

/*
 * Turns a board and a network into a move.
 *
 * The network answers with the worth of a position to X: one if X wins it,
 * a half if it is drawn and nought if O wins it. The policy looks one cell
 * ahead, asks what each cell would leave behind, and takes the one that
 * leaves the best position: the highest for X and the lowest for O.
 *
 * The temperature decides how much the scores are obeyed. Above zero the
 * move is drawn from a distribution over the empty cells, weighted by their
 * scores. At zero the best scoring empty cell is taken, and a tie is broken
 * at random, so a network that says the same for every cell still plays
 * every cell in turn.
 */
typedef struct {
	network_t *network;
	rng_t rng;
	double temperature;
} policy_t;

bool policy_init(policy_t *policy, network_t *network, unsigned long long seed,
                 double temperature);

/*
 * The worth of a position to X, as the network sees it: one if X wins it, a
 * half if it is drawn and nought if O wins it.
 */
bool policy_position_worth(const network_t *network, const board_t *board,
                           double *worth);
bool policy_select_move(policy_t *policy, const board_t *board, board_cell_t *move);

#endif // POLICY_H
