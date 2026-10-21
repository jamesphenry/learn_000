#ifndef TRAINER_H
#define TRAINER_H

#include <stdbool.h>
#include "game.h"
#include "network.h"
#include "policy.h"

/*
 * Teaches a network whose output layer is a sigmoid, so that its answer is
 * the worth of a position to the player to move: one if they win it, a half
 * if it is drawn and nought if they lose it.
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
bool trainer_train_position(trainer_t *trainer, const game_t *game,
                            const game_position_t *position, double target);

/*
 * Learns the worth of every position in a game that has been played, from
 * the perfect player, which knows what each position is really worth.
 */
bool trainer_learn_from_oracle(trainer_t *trainer, const game_t *game,
                               const oracle_t *oracle, const game_trace_t *trace);

/*
 * Learns the worth of every position in a game that has been played, from
 * the result of that game: each position takes the worth the game gave the
 * player who was to move in it.
 */
bool trainer_learn_from_outcome(trainer_t *trainer, const game_t *game,
                                const game_trace_t *trace);

/* Plays games against itself and learns from each of them. */
int trainer_run(trainer_t *trainer, const game_t *game, policy_t *policy, int games);

#endif // TRAINER_H
