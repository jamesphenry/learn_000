#include <math.h>
#include "policy.h"

/*
 * The worth of a position to the player to move: one if they win it, a half
 * if it is drawn and nought if they lose it.
 */
bool policy_worth_of(const network_t *network, const game_t *game,
                     const game_position_t *position, double *worth)
{
	double inputs[GAME_MAX_FEATURES];

	if (network == NULL || game == NULL || position == NULL || worth == NULL) return false;
	if (game->n_features < 1 || game->n_features > GAME_MAX_FEATURES) return false;

	game->features(position, inputs);

	return network_forward(network, inputs, (size_t)game->n_features, worth, 1);
}

bool policy_init(policy_t *policy, const game_t *game, network_t *network,
                 unsigned long long seed, double temperature)
{
	if (policy == NULL) return false;

	/* a policy that cannot be used is left empty, so that whoever holds
	 * it cannot make a move with it by mistake */
	policy->game = NULL;
	policy->network = NULL;
	policy->temperature = 0.0;

	if (game == NULL || network == NULL) return false;

	/* the policy looks ahead, so it needs a network that answers with the
	 * worth of a position */
	if (network_output_size(network) != 1) return false;

	policy->game = game;
	policy->network = network;
	policy->temperature = temperature;
	rng_seed(&policy->rng, seed);

	return true;
}

bool policy_select_move(policy_t *policy, const game_position_t *position,
                        game_move_t *move)
{
	double scores[GAME_MAX_MOVES] = { 0.0 };
	double preference[GAME_MAX_MOVES] = { 0.0 };
	double weights[GAME_MAX_MOVES];
	int count;
	double biggest;
	double total = 0.0;
	double pick;

	if (policy == NULL || position == NULL || move == NULL) return false;
	if (policy->network == NULL || policy->game == NULL) return false;

	count = position->n_moves;
	if (count < 1 || count > GAME_MAX_MOVES) return false;

	/* the worth of the position each move would leave behind */
	for (int i = 0; i < count; i++) {
		game_position_t next = *position;

		if (!policy->game->play(&next, position->moves[i])) return false;

		if (!policy_worth_of(policy->network, policy->game, &next, &scores[i])) return false;
	}

	/* each successor is worth what the opponent would get out of it, so the
	 * player to move wants the smallest of them */
	for (int i = 0; i < count; i++)
		preference[i] = -scores[i];

	biggest = preference[0];
	for (int i = 1; i < count; i++) {
		if (preference[i] > biggest)
			biggest = preference[i];
	}

	if (policy->temperature <= 0.0) {
		/* take the best move, and break a tie at random */
		int ties = 0;

		for (int i = 0; i < count; i++) {
			if (preference[i] == biggest)
				ties++;
		}

		int choice = rng_next_int(&policy->rng, ties);

		for (int i = 0; i < count; i++) {
			if (preference[i] != biggest) continue;

			if (choice == 0) {
				*move = position->moves[i];
				return true;
			}

			choice--;
		}

		return false;
	}

	/* a softmax over the legal moves, shifted so the largest one is zero */
	for (int i = 0; i < count; i++) {
		weights[i] = exp((preference[i] - biggest) / policy->temperature);
		total += weights[i];
	}

	pick = rng_next_double(&policy->rng) * total;

	for (int i = 0; i < count; i++) {
		pick -= weights[i];

		if (pick <= 0.0) {
			*move = position->moves[i];
			return true;
		}
	}

	*move = position->moves[count - 1];
	return true;
}

game_move_t policy_choose(void *context, const game_position_t *position)
{
	policy_t *policy = (policy_t *)context;
	game_move_t move;

	if (policy == NULL || position == NULL) return -1;
	if (!policy_select_move(policy, position, &move)) return -1;

	return move;
}
