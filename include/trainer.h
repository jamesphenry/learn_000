#ifndef TRAINER_H
#define TRAINER_H

#include <stdbool.h>
#include "game.h"
#include "network.h"

/*
 * Teaches a network whose output layer is a sigmoid, so that each of its
 * outputs is the chance that X wins if that move is played.
 *
 * The loss is a cross entropy against the result of the game, and the
 * derivative of the sigmoid is folded into it, so the error of the neuron
 * that was played is just its output less the target.
 */
typedef struct {
	network_t *network;
	double learning_rate;
} trainer_t;

bool trainer_init(trainer_t *trainer, network_t *network, double learning_rate);

/* Learns from a game that has been played. */
bool trainer_train_game(trainer_t *trainer, const game_t *game);

/* Plays games against itself and learns from each of them. */
int trainer_run(trainer_t *trainer, policy_t *policy, int games);

#endif // TRAINER_H
