#include <string.h>
#include "arena.h"
#include "evaluate.h"
#include "game.h"
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
}

/* Builds the network, and starts it off in the way that was asked for. */
static bool arena_build(network_t *network, const arena_settings_t *settings,
                        init_scheme_t scheme, rng_t *rng)
{
	network_init(network);

	if (!network_add_layer(network, CELL_COUNT, settings->hidden_neurons, ACTIVATION_CLAMP))
		return false;

	if (!network_add_layer(network, settings->hidden_neurons, CELL_COUNT, ACTIVATION_SIGMOID))
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

bool arena_run(const arena_settings_t *settings, init_scheme_t scheme,
               arena_result_t *result)
{
	network_t network;
	policy_t policy;
	trainer_t trainer;
	rng_t rng;
	evaluation_t before;
	evaluation_t after;
	int trained;

	if (settings == NULL || result == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(result, 0, sizeof(*result));

	rng_seed(&rng, settings->training_seed);

	if (!arena_build(&network, settings, scheme, &rng)) return false;

	/* what the model is like before it has learned anything */
	policy_init(&policy, &network, settings->evaluation_seed, settings->temperature);
	if (!evaluate_network(&network, settings->evaluation_seed, settings->evaluation_games, &before))
		return false;

	/* and then it learns from games of its own */
	trainer_init(&trainer, &network, settings->learning_rate);
	policy_init(&policy, &network, settings->training_seed, settings->temperature);
	trained = trainer_run(&trainer, &policy, settings->training_games);

	/* and what it is like afterwards, on games it has not seen */
	policy_init(&policy, &network, settings->evaluation_seed, settings->temperature);
	if (!evaluate_network(&network, settings->evaluation_seed, settings->evaluation_games, &after))
		return false;

	result->win_rate_before = 100.0 * before.wins / before.games;
	result->win_rate_after = 100.0 * after.wins / after.games;
	result->brier_before = before.brier_score;
	result->brier_after = after.brier_score;
	result->truth_before = before.truth_error;
	result->truth_after = after.truth_error;
	result->trained_games = trained;

	return true;
}
