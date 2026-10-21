#ifndef POLICY_H
#define POLICY_H

#include <math.h>
#include <stdbool.h>
#include "game.h"
#include "network.h"
#include "rng.h"

/*
 * Turns a position and a network into a move.
 *
 * The network answers with the worth of a position to the player to move:
 * one if they win it, a half if it is drawn and nought if they lose it. The
 * policy looks one move ahead, asks what each move would leave behind, and
 * takes the one that leaves the best position: the highest for the player to
 * move and the lowest for the opponent.
 *
 * The temperature decides how much the scores are obeyed. Above zero the
 * move is drawn from a distribution over the legal moves, weighted by their
 * scores. At zero the best scoring move is taken, and a tie is broken at
 * random, so a network that says the same for every move still plays every
 * move in turn.
 */
typedef struct {
	const game_t *game;
	network_t *network;
	rng_t rng;
	double temperature;
} policy_t;

/* At this temperature every legal move ends up with the same weight, so the
 * policy plays at random. */
#define POLICY_RANDOM_TEMPERATURE INFINITY

bool policy_init(policy_t *policy, const game_t *game, network_t *network,
                 unsigned long long seed, double temperature);

/*
 * The worth of a position to the player to move, as the network sees it:
 * one if they win it, a half if it is drawn and nought if they lose it.
 */
bool policy_worth_of(const network_t *network, const game_t *game,
                     const game_position_t *position, double *worth);

/* The move the policy would play in a position. */
bool policy_select_move(policy_t *policy, const game_position_t *position,
                        game_move_t *move);

/* The callback the framework plays a game with. */
game_move_t policy_choose(void *context, const game_position_t *position);

#endif // POLICY_H
