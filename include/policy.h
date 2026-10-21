#ifndef POLICY_H
#define POLICY_H

#include <stdbool.h>
#include "board.h"
#include "network.h"
#include "rng.h"

/*
 * Turns a board and a network into a move.
 *
 * The temperature decides how much the scores are obeyed. Above zero the
 * move is drawn from a distribution over the empty cells, weighted by their
 * scores. At zero the best scoring empty cell is taken, and a tie is broken
 * at random, so a network whose scores are all the same still plays every
 * cell in turn.
 */
typedef struct {
	network_t *network;
	rng_t rng;
	double temperature;
} policy_t;

bool policy_init(policy_t *policy, network_t *network, unsigned long long seed,
                 double temperature);
bool policy_select_move(policy_t *policy, const board_t *board, board_cell_t *move);

#endif // POLICY_H
