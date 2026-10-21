#ifndef TRAINER_H
#define TRAINER_H

#include <stdbool.h>
#include "game.h"
#include "network.h"

/*
 * Teaches a network whose output layer is a sigmoid, so that its answer is
 * the worth of a position to X: one if X wins it, a half if it is drawn and
 * nought if O wins it.
 *
 * The loss is a cross entropy against the target, and the derivative of the
 * sigmoid is folded into it, so the error of the neuron is just its output
 * less the target.
 */
typedef struct {
	network_t *network;
	double learning_rate;
} trainer_t;

bool trainer_init(trainer_t *trainer, network_t *network, double learning_rate);

/* Learns the worth of one position. */
bool trainer_train_position(trainer_t *trainer, const board_t *board, double target);

/* Learns the worth of every position in a game that has been played, from
 * the result of that game. */
bool trainer_train_game(trainer_t *trainer, const game_t *game);

/* Plays games against itself and learns from each of them. */
int trainer_run(trainer_t *trainer, policy_t *policy, int games);

#endif // TRAINER_H
