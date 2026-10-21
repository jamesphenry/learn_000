#include <string.h>
#include "arena.h"
#include "evaluate.h"
#include "game.h"
#include "minimax.h"
#include "policy.h"
#include "trainer.h"

void arena_settings_init(arena_settings_t *settings)
{
	if (settings == NULL) return;

	settings->hidden_neurons = 16;
	settings->training_games = 500;
	settings->evaluation_games = 200;
	settings->training_seed = 1;
	settings->evaluation_seed = 2;
	settings->learning_rate = 0.05;
	settings->temperature = 0.3;
	settings->init_scale = 0.5;
	settings->init_value = 1.0;
	settings->learning = ARENA_LEARN_TRUTH;
}

/* Builds the network, and starts it off in the way that was asked for. */
static bool arena_build(network_t *network, const arena_settings_t *settings,
                        init_scheme_t scheme, rng_t *rng)
{
	network_init(network);

	if (!network_add_layer(network, CELL_COUNT, settings->hidden_neurons, ACTIVATION_RELU))
		return false;

	if (!network_add_layer(network, settings->hidden_neurons, 1, ACTIVATION_SIGMOID))
		return false;

	switch (scheme) {
	case INIT_RANDOM:
		return network_fill_random(network, rng, settings->init_scale);

	case INIT_MAXIMUM:
		return network_fill_maximum(network, settings->init_value);

	case INIT_ZERO:
		/* a network starts with everything at zero on its own */
		return true;
	}

	return false;
}

/*
 * Learns what the perfect player knows about every position it plays. The
 * games are its own, so the positions are the ones it visits, and the
 * perfect player says what each of them is worth.
 */
static int arena_learn_from_the_truth(trainer_t *trainer, policy_t *policy, int games)
{
	game_t game;
	board_t positions[CELL_COUNT];
	int learned = 0;

	for (int i = 0; i < games; i++) {
		game_play(&game, policy, policy);
		game_positions(&game, positions);

		for (int step = 0; step < game.moves; step++) {
			if (!trainer_train_position(trainer, &positions[step],
			                            minimax_x_win_chance(&positions[step])))
				return learned;
		}

		learned++;
	}

	return learned;
}

bool arena_run(const arena_settings_t *settings, init_scheme_t scheme,
               arena_result_t *result)
{
	network_t network;
	policy_t policy;
	trainer_t trainer;
	rng_t rng;
	evaluation_t before;
	evaluation_t after;
	skill_t skill_before;
	skill_t skill_after;
	perfect_result_t perfect_before;
	perfect_result_t perfect_after;
	int trained;

	if (settings == NULL || result == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(result, 0, sizeof(*result));

	rng_seed(&rng, settings->training_seed);

	if (!arena_build(&network, settings, scheme, &rng)) return false;

	/* what the model is like before it has learned anything */
	if (!evaluate_network(&network, settings->evaluation_seed, settings->evaluation_games, &before))
		return false;

	evaluate_skill(&network, settings->evaluation_seed, settings->evaluation_games, &skill_before);
	evaluate_against_perfect(&network, settings->evaluation_seed,
	                         settings->evaluation_games, &perfect_before);

	/* and then it learns */
	trainer_init(&trainer, &network, settings->learning_rate);
	policy_init(&policy, &network, settings->training_seed, settings->temperature);

	if (settings->learning == ARENA_LEARN_TRUTH)
		trained = arena_learn_from_the_truth(&trainer, &policy, settings->training_games);
	else
		trained = trainer_run(&trainer, &policy, settings->training_games);

	/* and what it is like afterwards, on games it has not seen */
	if (!evaluate_network(&network, settings->evaluation_seed, settings->evaluation_games, &after))
		return false;

	evaluate_skill(&network, settings->evaluation_seed, settings->evaluation_games, &skill_after);
	evaluate_against_perfect(&network, settings->evaluation_seed,
	                         settings->evaluation_games, &perfect_after);

	result->win_rate_before = 100.0 * before.wins / before.games;
	result->win_rate_after = 100.0 * after.wins / after.games;
	result->brier_before = before.brier_score;
	result->brier_after = after.brier_score;
	result->truth_before = before.truth_error;
	result->truth_after = after.truth_error;
	result->value_lost_before = skill_before.moves > 0 ?
		skill_before.value_lost / skill_before.moves : 0.0;
	result->value_lost_after = skill_after.moves > 0 ?
		skill_after.value_lost / skill_after.moves : 0.0;
	result->perfect_before = perfect_before;
	result->perfect_after = perfect_after;
	result->trained_games = trained;

	return true;
}
