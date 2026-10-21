#include <math.h>
#include <string.h>
#include "evaluate.h"
#include "game.h"
#include "judge.h"
#include "minimax.h"
#include "policy.h"

bool evaluate_against(const network_t *first, const network_t *second,
                      unsigned long long seed, int games, double temperature,
                      match_result_t *result)
{
	policy_t first_policy;
	policy_t second_policy;
	game_t game;
	judgement_t verdict;
	game_result_t outcome;
	int first_wins = 0;
	int draws = 0;
	int second_wins = 0;
	int flagged = 0;

	if (first == NULL || second == NULL || result == NULL || games < 1) return false;

	memset(result, 0, sizeof(*result));

	policy_init(&first_policy, (network_t *)first, seed, temperature);
	policy_init(&second_policy, (network_t *)second, seed + 1, temperature);

	for (int i = 0; i < games; i++) {
		int first_is_x = (i % 2) == 0;

		/* each network takes one colour and then the other, so that
		 * neither of them gets the first move every time */
		if (first_is_x)
			outcome = game_play(&game, &first_policy, &second_policy);
		else
			outcome = game_play(&game, &second_policy, &first_policy);

		if (first_is_x) result->first_as_x++;

		/* and the referee looks over it */
		if (!judge_game(&game, &verdict)) {
			flagged++;
			continue;
		}

		if (outcome == GAME_DRAW) draws++;
		else if ((outcome == GAME_X_WINS) == first_is_x) {
			first_wins++;

			if (first_is_x) result->first_wins_as_x++;
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

bool evaluate_skill(const network_t *network, unsigned long long seed, int games,
                    double opponent_temperature, skill_t *result)
{
	policy_t policy;
	policy_t opponent;
	game_t game;
	board_t positions[CELL_COUNT];
	minimax_t oracle;
	int moves = 0;
	int kept = 0;
	int gave_away = 0;
	int threw_wins = 0;
	double value_lost = 0.0;

	if (network == NULL || result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&policy, (network_t *)network, seed, 0.0);
	policy_init(&opponent, (network_t *)network, seed + 1, opponent_temperature);
	minimax_init(&oracle);

	for (int i = 0; i < games; i++) {
		/* the network takes one colour and then the other */
		if (i % 2 == 0)
			game_play(&game, &policy, &opponent);
		else
			game_play(&game, &opponent, &policy);

		game_positions(&game, positions);

		for (int step = 0; step < game.moves; step++) {
			minimax_result_t perfect;
			minimax_result_t then;
			board_t after;
			int lost;

			perfect = minimax_at(&oracle, &positions[step]);

			/* the game was over, so no move was left to make */
			if (perfect.move < 0) continue;

			/* the position the move left behind */
			after = positions[step];
			after.cells[game.history[step]] = encode_mark(after.current_player);
			then = minimax_at(&oracle, &after);

			/* the perfect player knows the position is worth perfect.value
			 * to the player to move, and the move that was played left it
			 * worth minus then.value to that same player, because the
			 * other one is to move now */
			lost = perfect.value + then.value;

			moves++;
			value_lost += lost;

			if (lost == 0) {
				kept++;
			} else {
				gave_away++;

				/* a win that was there and was not taken */
				if (perfect.value == 1) threw_wins++;
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

/* Whether the game is over, judging by the board alone. */
static bool board_is_finished(const board_t *board)
{
	if (detect_win(board, 'X')) return true;
	if (detect_win(board, 'O')) return true;

	return board_is_full(board);
}

bool evaluate_against_perfect(const network_t *network, unsigned long long seed,
                              int games, perfect_result_t *result)
{
	policy_t policy;
	minimax_t oracle;
	int wins = 0;
	int draws = 0;
	int losses = 0;

	if (network == NULL || result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&policy, (network_t *)network, seed, 0.0);
	minimax_init(&oracle);

	for (int i = 0; i < games; i++) {
		board_t board;
		char winner;
		int network_is_x = (i % 2) == 0;

		board_init(&board);

		while (!board_is_finished(&board)) {
			board_cell_t move;

			if ((board.current_player == 'X') == (network_is_x != 0)) {
				if (!policy_select_move(&policy, &board, &move)) break;
			} else {
				minimax_result_t perfect = minimax_at(&oracle, &board);

				if (perfect.move < 0) break;
				move = (board_cell_t)perfect.move;
			}

			place_mark(&board, move, board.current_player);
			board.current_player = (board.current_player == 'X') ? 'O' : 'X';
		}

		winner = detect_win(&board, 'X') ? 'X' :
		         detect_win(&board, 'O') ? 'O' : 0;

		if (network_is_x) result->as_x++;
		else result->as_o++;

		if (winner == 0) {
			draws++;
		} else if (winner == (network_is_x ? 'X' : 'O')) {
			wins++;

			if (network_is_x) result->wins_as_x++;
			else result->wins_as_o++;
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

bool evaluate_initialisation(const network_t *network, initialisation_t *result)
{
	network_t *mutable_network = (network_t *)network;
	board_t board;
	double inputs[CELL_COUNT];

	if (network == NULL || result == NULL) return false;
	if (network->n_layers < 1) return false;

	memset(result, 0, sizeof(*result));

	result->hidden_lowest = 1e9;
	result->hidden_highest = -1e9;
	result->output_lowest = 1e9;
	result->output_highest = -1e9;

	/* over every board with a mark on it and another mark off it */
	for (int first = 0; first < CELL_COUNT; first++) {
		for (int second = 0; second < CELL_COUNT; second++) {
			double hidden[LAYER_MAX_NEURONS];
			double output = 0.0;

			if (first == second) continue;

			board_init(&board);
			place_mark(&board, (board_cell_t)first, 'X');
			place_mark(&board, (board_cell_t)second, 'O');
			board_to_inputs(&board, inputs);

			for (int neuron = 0; neuron < network->layers[0].n_neurons; neuron++) {
				double sum = neuron_weighted_sum(inputs,
				                                network->layers[0].weights[neuron],
				                                network->layers[0].n_inputs,
				                                network->layers[0].biases[neuron]);

				hidden[neuron] = neuron_activate(sum, network->layers[0].activation);

				if (sum < result->hidden_lowest) result->hidden_lowest = sum;
				if (sum > result->hidden_highest) result->hidden_highest = sum;
			}

			for (int i = 0; i < network->layers[0].n_neurons; i++)
				output += hidden[i] * network->layers[1].weights[0][i];

			output += network->layers[1].biases[0];

			if (output < result->output_lowest) result->output_lowest = output;
			if (output > result->output_highest) result->output_highest = output;
		}
	}

	/* and what it says about an empty board */
	board_init(&board);
	board_to_inputs(&board, inputs);
	network_forward(mutable_network, inputs, CELL_COUNT, &result->worth, 1);

	/* the pull of the loss on it: the error of a sigmoid is its output
	 * less the target */
	result->gradient_to_one = fabs(result->worth - 1.0);
	result->gradient_to_zero = fabs(result->worth - 0.0);

	return true;
}

bool evaluate_network(const network_t *network, unsigned long long seed, int games,
                      evaluation_t *result)
{
	policy_t network_policy;
	policy_t random_policy;
	game_t game;
	board_t positions[CELL_COUNT];
	minimax_t oracle;
	double total = 0.0;
	double truth_total = 0.0;
	int moves = 0;

	if (network == NULL || result == NULL || games < 1) return false;

	/* the caller may hand over a result that has been used before */
	memset(result, 0, sizeof(*result));

	policy_init(&network_policy, (network_t *)network, seed, 0.0);
	policy_init(&random_policy, (network_t *)network, seed + 1, POLICY_RANDOM_TEMPERATURE);
	minimax_init(&oracle);

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

		/* and the worth it gave for the positions it was in */
		game_positions(&game, positions);

		for (int step = 0; step < game.moves; step++) {
			double worth;
			double actual;
			int bucket;

			if (!policy_position_worth(network, &positions[step], &worth)) continue;

			actual = game_worth_of_result(positions[step].current_player, outcome);

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
			truth_total += fabs(worth - minimax_worth(&oracle, &positions[step]));

			if (outcome == GAME_X_WINS) result->moves_in_x_wins++;
			if (outcome == GAME_DRAW) result->moves_in_draws++;
			if (actual == 1.0) result->moves_won++;
		}
	}

	result->games = games;
	result->wins = result->wins_as_x + result->wins_as_o;

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
