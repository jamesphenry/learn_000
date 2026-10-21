#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "analysis.h"
#include "policy.h"

/* Two worths that are this close are the same, as far as the analysis is
 * concerned. */
#define ANALYSIS_SAME 1e-9

/* The worth the perfect player gives the position a move leaves behind: it
 * is worth that much to the opponent, who is to move in it. */
static double worth_after(const game_t *game, const oracle_t *oracle,
                           const game_position_t *position, game_move_t move,
                           bool *played)
{
	game_position_t after = *position;

	*played = false;

	if (!game->play(&after, move)) return 0.0;

	*played = true;

	return oracle->worth(oracle->state, &after);
}

bool analysis_fixed_set(const game_t *game, const oracle_t *oracle,
                         const network_t *network, unsigned long long seed,
                         analysis_t *result)
{
	game_position_t *positions = NULL;
	policy_t policy;
	unsigned long long seen = 0;
	int count = 0;
	int measured = 0;
	double truth_total = 0.0;
	double worth_total = 0.0;
	double error_total = 0.0;
	double win_total = 0.0;
	double loss_total = 0.0;
	double value_lost = 0.0;
	int wins = 0;
	int losses = 0;
	int calls_win = 0;
	int truth_wins = 0;
	int moves = 0;
	int best_moves = 0;
	int distinct = 0;
	int winning_positions = 0;
	int winning_moves = 0;

	if (game == NULL || oracle == NULL || network == NULL || result == NULL) return false;

	/* a game that cannot say which positions there are has no fixed set */
	if (game->positions == NULL) return false;

	if (!oracle_ready(oracle)) return false;

	memset(result, 0, sizeof(*result));

	positions = malloc(sizeof(*positions) * ANALYSIS_MAX_POSITIONS);
	if (positions == NULL) return false;

	count = game->positions(positions, ANALYSIS_MAX_POSITIONS);

	policy_init(&policy, game, (network_t *)network, seed, 0.0);

	for (int i = 0; i < count; i++) {
		const game_position_t *position = &positions[i];
		double worth;
		double truth;
		double best_after = 2.0;
		double after;
		game_move_t move;
		bool played;

		if (!policy_worth_of(network, game, position, &worth)) continue;

		/* what the model says about the position, and what it is really
		 * worth */
		truth = oracle->worth(oracle->state, position);

		worth_total += worth;
		truth_total += truth;
		error_total += fabs(worth - truth);
		measured++;

		if (worth > 0.5) calls_win++;
		if (truth > 0.5) truth_wins++;

		if (truth > 0.5) {
			win_total += worth;
			wins++;
		} else {
			loss_total += worth;
			losses++;
		}

		/* a position that is over has no moves to play */
		if (position->n_moves < 1) continue;

		/* the best move there is: the one that leaves the opponent the
		 * least */
		for (int m = 0; m < position->n_moves; m++) {
			double other = worth_after(game, oracle, position,
			                           position->moves[m], &played);

			if (!played) continue;

			if (other < best_after) best_after = other;
		}

		/* and the move the model would play */
		if (!policy_select_move(&policy, position, &move)) continue;

		after = worth_after(game, oracle, position, move, &played);
		if (!played) continue;

		moves++;

		/* the worth the move gave away: what the position was worth to
		 * the mover, less what the move left behind */
		value_lost += truth + after - 1.0;

		if (after <= best_after + ANALYSIS_SAME) best_moves++;

		/* and the moves it has not played before */
		if (move >= 0 && move < GAME_MAX_MOVES &&
		    (seen & (1ULL << move)) == 0) {
			seen |= 1ULL << move;
			distinct++;
		}

		/* a position the player to move can win, and a move that leaves
		 * the opponent in a losing position */
		if (truth > 0.5) {
			winning_positions++;

			if (after <= ANALYSIS_SAME) winning_moves++;
		}
	}

	result->positions = measured;
	result->truth_error = measured > 0 ? error_total / measured : 0.0;
	result->mean_worth = measured > 0 ? worth_total / measured : 0.0;
	result->mean_truth = measured > 0 ? truth_total / measured : 0.0;
	result->win_worth = wins > 0 ? win_total / wins : 0.0;
	result->loss_worth = losses > 0 ? loss_total / losses : 0.0;
	result->calls_win = calls_win;
	result->truth_wins = truth_wins;
	result->moves = moves;
	result->best_moves = best_moves;
	result->distinct_moves = distinct;
	result->value_lost = value_lost;
	result->winning_positions = winning_positions;
	result->winning_moves = winning_moves;

	free(positions);

	return true;
}

bool analysis_movement(const network_t *before, const network_t *after,
                       movement_t *result)
{
	double hidden_total = 0.0;
	double hidden_count = 0.0;
	double output_total = 0.0;
	double output_count = 0.0;
	double all_total = 0.0;
	double all_count = 0.0;

	if (before == NULL || after == NULL || result == NULL) return false;
	if (before->n_layers != after->n_layers) return false;

	memset(result, 0, sizeof(*result));

	for (int layer = 0; layer < before->n_layers; layer++) {
		const layer_t *from = &before->layers[layer];
		const layer_t *to = &after->layers[layer];
		double total = 0.0;
		double count = 0.0;

		if (from->n_inputs != to->n_inputs) return false;
		if (from->n_neurons != to->n_neurons) return false;

		for (int neuron = 0; neuron < from->n_neurons; neuron++) {
			for (int input = 0; input < from->n_inputs; input++) {
				double change = fabs(from->weights[neuron][input] -
				                     to->weights[neuron][input]);

				total += change;
				count += 1.0;
			}

			{
				double change = fabs(from->biases[neuron] -
				                     to->biases[neuron]);

				total += change;
				count += 1.0;
			}
		}

		all_total += total;
		all_count += count;

		if (layer == 0) {
			hidden_total = total;
			hidden_count = count;
		} else {
			output_total = total;
			output_count = count;
		}
	}

	result->hidden = hidden_count > 0 ? hidden_total / hidden_count : 0.0;
	result->output = output_count > 0 ? output_total / output_count : 0.0;
	result->all = all_count > 0 ? all_total / all_count : 0.0;

	return true;
}

bool analysis_curve(const arena_settings_t *settings, const game_t *game,
                    const oracle_t *oracle, init_scheme_t scheme,
                    const int *budgets, int budgets_count, int seeds,
                    analysis_curve_t *curve)
{
	if (settings == NULL || game == NULL || oracle == NULL || budgets == NULL) return false;
	if (curve == NULL) return false;
	if (budgets_count < 1 || budgets_count > ANALYSIS_MAX_BUDGETS) return false;
	if (seeds < 1 || seeds > ANALYSIS_MAX_SEEDS) return false;

	memset(curve, 0, sizeof(*curve));

	curve->budgets = budgets_count;
	curve->seeds = seeds;

	for (int budget = 0; budget < budgets_count; budget++) {
		if (budgets[budget] < 0) return false;

		curve->budget[budget] = budgets[budget];

		for (int seed = 0; seed < seeds; seed++) {
			arena_settings_t run = *settings;
			network_t network;
			network_t before;
			rng_t rng;

			/* each seed is the one before it with one added to it */
			run.training_games = budgets[budget];
			run.training_seed = settings->training_seed + (unsigned long long)seed;

			rng_seed(&rng, run.training_seed);

			/* the model as it is before it has learned anything */
			if (!arena_build(game, &network, &run, scheme, &rng)) return false;

			before = network;

			if (!arena_teach(&run, game, oracle, &network, NULL)) return false;

			if (!analysis_movement(&before, &network, &curve->movement[budget][seed]))
				return false;

			if (!analysis_fixed_set(game, oracle, &network, run.evaluation_seed,
			                        &curve->measure[budget][seed]))
				return false;
		}
	}

	return true;
}
