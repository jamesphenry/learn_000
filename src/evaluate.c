#include <math.h>
#include <string.h>
#include "evaluate.h"
#include "judge.h"
#include "policy.h"

/* The move the perfect player would play. */
static game_move_t oracle_choose(void *context, const game_position_t *position)
{
	const oracle_t *oracle = (const oracle_t *)context;

	if (oracle == NULL || oracle->choose == NULL || oracle->state == NULL) return -1;

	return oracle->choose(oracle->state, position);
}

bool evaluate_against(const game_t *game, const oracle_t *oracle,
                      const network_t *first, const network_t *second,
                      unsigned long long seed, int games, double temperature,
                      match_result_t *result)
{
	policy_t first_policy;
	policy_t second_policy;
	game_trace_t trace;
	judgement_t verdict;
	int first_wins = 0;
	int draws = 0;
	int second_wins = 0;
	int flagged = 0;

	if (game == NULL || oracle == NULL) return false;
	if (first == NULL || second == NULL || result == NULL || games < 1) return false;

	memset(result, 0, sizeof(*result));

	policy_init(&first_policy, game, (network_t *)first, seed, temperature);
	policy_init(&second_policy, game, (network_t *)second, seed + 1, temperature);

	for (int i = 0; i < games; i++) {
		int first_is_first = (i % 2) == 0;
		game_outcome_t outcome;

		/* each network takes one side and then the other, so that
		 * neither of them gets the first move every time */
		if (first_is_first)
			game_play(game, policy_choose, &first_policy, policy_choose,
			          &second_policy, &trace);
		else
			game_play(game, policy_choose, &second_policy, policy_choose,
			          &first_policy, &trace);

		if (first_is_first) result->first_as_first++;

		/* and the referee looks over it */
		if (!judge_game(game, oracle, &trace, &verdict)) {
			flagged++;
			continue;
		}

		outcome = game->outcome(&trace.position);

		if (outcome == GAME_DRAW) draws++;
		else if ((outcome == GAME_FIRST_WINS) == (first_is_first != 0)) {
			first_wins++;

			if (first_is_first) result->first_wins_as_first++;
		}
		else second_wins++;
	}

	result->games = games;
	result->first_wins = first_wins;
	result->draws = draws;
	result->second_wins = second_wins;
	result->flagged = flagged;

	return true;
}

bool evaluate_skill(const game_t *game, const oracle_t *oracle,
                    const network_t *network, unsigned long long seed, int games,
                    double opponent_temperature, skill_t *result)
{
	policy_t policy;
	policy_t opponent;
	game_trace_t trace;
	game_position_t positions[GAME_MAX_HISTORY];
	game_position_t final;
	int moves = 0;
	int kept = 0;
	int gave_away = 0;
	int threw_wins = 0;
	double value_lost = 0.0;

	if (game == NULL || oracle == NULL || network == NULL) return false;
	if (result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&policy, game, (network_t *)network, seed, 0.0);
	policy_init(&opponent, game, (network_t *)network, seed + 1, opponent_temperature);

	for (int i = 0; i < games; i++) {
		int played;

		/* the network takes one side and then the other */
		if (i % 2 == 0)
			game_play(game, policy_choose, &policy, policy_choose, &opponent, &trace);
		else
			game_play(game, policy_choose, &opponent, policy_choose, &policy, &trace);

		played = game_replay(game, &trace, positions, &final);

		for (int step = 0; step < played; step++) {
			const game_position_t *after =
				(step + 1 < played) ? &positions[step + 1] : &final;
			/* what the position was worth to the player to move, and
			 * what the position the move left behind is worth to the
			 * opponent */
			double perfect = oracle->worth(oracle->state, &positions[step]);
			double then = oracle->worth(oracle->state, after);
			/* so the worth the move gave away */
			double lost = perfect + then - 1.0;

			moves++;
			value_lost += lost;

			if (lost <= 0.0) {
				kept++;
			} else {
				gave_away++;

				/* a win that was there and was not taken */
				if (perfect >= 1.0) threw_wins++;
			}
		}
	}

	result->moves = moves;
	result->kept = kept;
	result->gave_away = gave_away;
	result->threw_wins = threw_wins;
	result->value_lost = value_lost;

	return true;
}

bool evaluate_against_perfect(const game_t *game, const oracle_t *oracle,
                              const network_t *network, unsigned long long seed,
                              int games, perfect_result_t *result)
{
	policy_t policy;
	int wins = 0;
	int draws = 0;
	int losses = 0;

	if (game == NULL || oracle == NULL || network == NULL) return false;
	if (result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&policy, game, (network_t *)network, seed, 0.0);

	for (int i = 0; i < games; i++) {
		int model_is_first = (i % 2) == 0;
		game_trace_t trace;
		game_outcome_t outcome;

		/* the model takes one side and then the other */
		if (model_is_first)
			game_play(game, policy_choose, &policy, oracle_choose,
			          (void *)oracle, &trace);
		else
			game_play(game, oracle_choose, (void *)oracle, policy_choose,
			          &policy, &trace);

		if (model_is_first) result->as_first++;
		else result->as_second++;

		outcome = game->outcome(&trace.position);

		if (outcome == GAME_DRAW) {
			draws++;
		} else if ((outcome == GAME_FIRST_WINS) == (model_is_first != 0)) {
			wins++;

			if (model_is_first) result->wins_as_first++;
			else result->wins_as_second++;
		} else {
			losses++;
		}
	}

	result->games = games;
	result->wins = wins;
	result->draws = draws;
	result->losses = losses;

	return true;
}

bool evaluate_initialisation(const game_t *game, const network_t *network,
                             initialisation_t *result)
{
	network_t *mutable_network = (network_t *)network;
	game_position_t samples[GAME_MAX_SAMPLES];
	game_position_t start;
	double inputs[GAME_MAX_FEATURES];
	int count;

	if (game == NULL || network == NULL || result == NULL) return false;
	if (network->n_layers < 1) return false;
	if (game->n_features < 1 || game->n_features > GAME_MAX_FEATURES) return false;

	memset(result, 0, sizeof(*result));

	result->hidden_lowest = 1e9;
	result->hidden_highest = -1e9;
	result->output_lowest = 1e9;
	result->output_highest = -1e9;

	/* over the positions the game offers to look at */
	count = game->samples(samples, GAME_MAX_SAMPLES);

	for (int s = 0; s < count; s++) {
		double sums[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];
		double outputs[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];
		const double *in = inputs;
		double output_sum;

		game->features(&samples[s], inputs);

		for (int layer = 0; layer < network->n_layers; layer++) {
			const layer_t *l = &network->layers[layer];

			for (int neuron = 0; neuron < l->n_neurons; neuron++) {
				double sum = neuron_weighted_sum(in, l->weights[neuron],
				                                l->n_inputs, l->biases[neuron]);

				sums[layer][neuron] = sum;
				outputs[layer][neuron] = neuron_activate(sum, l->activation);

				if (layer == 0) {
					if (sum < result->hidden_lowest) result->hidden_lowest = sum;
					if (sum > result->hidden_highest) result->hidden_highest = sum;
				}
			}

			in = outputs[layer];
		}

		/* and the sum the output layer sees */
		output_sum = sums[network->n_layers - 1][0];

		if (output_sum < result->output_lowest) result->output_lowest = output_sum;
		if (output_sum > result->output_highest) result->output_highest = output_sum;
	}

	/* and what it says about the position the game starts from */
	game->start(&start);
	game->features(&start, inputs);
	network_forward(mutable_network, inputs, (size_t)game->n_features, &result->worth, 1);

	/* the pull of the loss on it: the error of a sigmoid is its output
	 * less the target */
	result->gradient_to_one = fabs(result->worth - 1.0);
	result->gradient_to_zero = fabs(result->worth - 0.0);

	return true;
}

bool evaluate_network(const game_t *game, const oracle_t *oracle,
                      const network_t *network, unsigned long long seed, int games,
                      evaluation_t *result)
{
	policy_t network_policy;
	policy_t random_policy;
	game_trace_t trace;
	game_position_t positions[GAME_MAX_HISTORY];
	game_position_t final;
	double total = 0.0;
	double truth_total = 0.0;
	int moves = 0;

	if (game == NULL || oracle == NULL || network == NULL) return false;
	if (result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&network_policy, game, (network_t *)network, seed, 0.0);
	policy_init(&random_policy, game, (network_t *)network, seed + 1,
	            POLICY_RANDOM_TEMPERATURE);

	for (int i = 0; i < games; i++) {
		int network_is_first = (i % 2) == 0;
		game_outcome_t outcome;
		int won;
		int drew;
		int played;

		/* the network takes one side and then the other, so that it
		 * does not get the first move every time */
		if (network_is_first)
			game_play(game, policy_choose, &network_policy, policy_choose,
			          &random_policy, &trace);
		else
			game_play(game, policy_choose, &random_policy, policy_choose,
			          &network_policy, &trace);

		if (network_is_first) result->as_first++;
		else result->as_second++;

		outcome = game->outcome(&trace.position);

		won = (network_is_first && outcome == GAME_FIRST_WINS) ||
		      (!network_is_first && outcome == GAME_SECOND_WINS);
		drew = (outcome == GAME_DRAW);

		if (won) {
			if (network_is_first) result->wins_as_first++;
			else result->wins_as_second++;
		} else if (drew) {
			result->draws++;
		} else {
			result->losses++;
		}

		if (outcome == GAME_FIRST_WINS) result->first_wins++;

		/* and the worth it gave for the positions it was in */
		played = game_replay(game, &trace, positions, &final);

		for (int step = 0; step < played; step++) {
			double worth;
			double actual;
			int bucket;

			if (!policy_worth_of(network, game, &positions[step], &worth)) continue;

			actual = game_worth_of_outcome(outcome, game->to_move(&positions[step]));

			bucket = (int)(worth * EVALUATION_BUCKETS);
			if (bucket < 0) bucket = 0;
			if (bucket >= EVALUATION_BUCKETS) bucket = EVALUATION_BUCKETS - 1;

			result->bucket_counts[bucket]++;
			result->bucket_predicted[bucket] += worth;
			result->bucket_observed[bucket] += actual;

			total += (worth - actual) * (worth - actual);
			moves++;
			result->moves++;

			/* and how far it is from what the perfect player knows the
			 * position to be worth */
			truth_total += fabs(worth - oracle->worth(oracle->state, &positions[step]));

			if (outcome == GAME_FIRST_WINS) result->moves_in_first_wins++;
			if (outcome == GAME_DRAW) result->moves_in_draws++;
			if (actual == 1.0) result->moves_won++;
		}
	}

	result->games = games;
	result->wins = result->wins_as_first + result->wins_as_second;

	if (moves > 0) {
		result->brier_score = total / moves;
		result->truth_error = truth_total / moves;
	}

	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++) {
		if (result->bucket_counts[bucket] == 0) continue;

		result->bucket_predicted[bucket] /= result->bucket_counts[bucket];
		result->bucket_observed[bucket] /= result->bucket_counts[bucket];
	}

	return true;
}
