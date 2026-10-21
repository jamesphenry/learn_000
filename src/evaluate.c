#include <math.h>
#include <string.h>
#include "evaluate.h"
#include "game.h"
#include "policy.h"

/* At this temperature every legal cell ends up with the same weight, so
 * the policy plays at random. */
#define RANDOM_TEMPERATURE INFINITY

/* The probability the network gives for each cell. */
static bool score_position(const network_t *network, const board_t *board,
                           double scores[CELL_COUNT])
{
	double inputs[CELL_COUNT];

	board_to_inputs(board, inputs);

	return network_forward(network, inputs, CELL_COUNT, scores, CELL_COUNT);
}

bool evaluate_network(const network_t *network, unsigned long long seed, int games,
                      evaluation_t *result)
{
	policy_t network_policy;
	policy_t random_policy;
	game_t game;
	board_t positions[CELL_COUNT];
	double scores[CELL_COUNT];
	double total = 0.0;
	int moves = 0;

	if (network == NULL || result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&network_policy, (network_t *)network, seed, 0.0);
	policy_init(&random_policy, (network_t *)network, seed + 1, RANDOM_TEMPERATURE);

	for (int i = 0; i < games; i++) {
		game_result_t outcome;
		int network_is_x = (i % 2) == 0;
		int won;
		int drew;

		/* the network takes one colour and then the other, so that it
		 * does not get the first move every time */
		if (network_is_x)
			outcome = game_play(&game, &network_policy, &random_policy);
		else
			outcome = game_play(&game, &random_policy, &network_policy);

		if (network_is_x) result->as_x++;
		else result->as_o++;

		won = (network_is_x && outcome == GAME_X_WINS) ||
		      (!network_is_x && outcome == GAME_O_WINS);
		drew = (outcome == GAME_DRAW);

		if (won) {
			if (network_is_x) result->wins_as_x++;
			else result->wins_as_o++;
		} else if (drew) {
			result->draws++;
		} else {
			result->losses++;
		}

		if (outcome == GAME_X_WINS) result->x_wins++;

		/* and the probabilities it gave for the moves that were played */
		game_positions(&game, positions);

		for (int step = 0; step < game.moves; step++) {
			double probability;
			double actual;
			int bucket;

			if (!score_position(network, &positions[step], scores)) continue;

			probability = scores[game.history[step]];
			actual = (outcome == GAME_X_WINS) ? 1.0 : 0.0;

			bucket = (int)(probability * EVALUATION_BUCKETS);
			if (bucket < 0) bucket = 0;
			if (bucket >= EVALUATION_BUCKETS) bucket = EVALUATION_BUCKETS - 1;

			result->bucket_counts[bucket]++;
			result->bucket_predicted[bucket] += probability;
			result->bucket_observed[bucket] += actual;

			total += (probability - actual) * (probability - actual);
			moves++;
			result->moves++;

			if (outcome == GAME_X_WINS) result->moves_in_x_wins++;
		}
	}

	result->games = games;
	result->wins = result->wins_as_x + result->wins_as_o;

	if (moves > 0) result->brier_score = total / moves;

	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++) {
		if (result->bucket_counts[bucket] == 0) continue;

		result->bucket_predicted[bucket] /= result->bucket_counts[bucket];
		result->bucket_observed[bucket] /= result->bucket_counts[bucket];
	}

	return true;
}
