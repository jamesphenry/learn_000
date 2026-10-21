#include <math.h>
#include "policy.h"

bool policy_init(policy_t *policy, network_t *network, unsigned long long seed,
                 double temperature)
{
	if (policy == NULL || network == NULL) return false;

	policy->network = network;
	policy->temperature = temperature;
	rng_seed(&policy->rng, seed);

	return true;
}

/* The cells that are still empty. */
static int legal_cells(const board_t *board, board_cell_t *cells)
{
	int count = 0;

	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (board->cells[cell] == VALUE_EMPTY)
			cells[count++] = (board_cell_t)cell;
	}

	return count;
}

bool policy_select_move(policy_t *policy, const board_t *board, board_cell_t *move)
{
	double inputs[CELL_COUNT];
	double scores[CELL_COUNT];
	double weights[CELL_COUNT];
	board_cell_t cells[CELL_COUNT];
	int count;
	double biggest;
	double total = 0.0;
	double pick;

	if (policy == NULL || board == NULL || move == NULL) return false;
	if (policy->network == NULL) return false;

	board_to_inputs(board, inputs);
	if (!network_forward(policy->network, inputs, CELL_COUNT, scores, CELL_COUNT))
		return false;

	count = legal_cells(board, cells);
	if (count == 0) return false;

	biggest = scores[cells[0]];
	for (int i = 1; i < count; i++) {
		if (scores[cells[i]] > biggest)
			biggest = scores[cells[i]];
	}

	if (policy->temperature <= 0.0) {
		/* take the best cell, and break a tie at random */
		int ties = 0;

		for (int i = 0; i < count; i++) {
			if (scores[cells[i]] == biggest)
				ties++;
		}

		int choice = rng_next_int(&policy->rng, ties);

		for (int i = 0; i < count; i++) {
			if (scores[cells[i]] != biggest) continue;

			if (choice == 0) {
				*move = cells[i];
				return true;
			}

			choice--;
		}

		return false;
	}

	/* a softmax over the legal cells, shifted so the largest one is zero */
	for (int i = 0; i < count; i++) {
		weights[i] = exp((scores[cells[i]] - biggest) / policy->temperature);
		total += weights[i];
	}

	pick = rng_next_double(&policy->rng) * total;

	for (int i = 0; i < count; i++) {
		pick -= weights[i];

		if (pick <= 0.0) {
			*move = cells[i];
			return true;
		}
	}

	*move = cells[count - 1];
	return true;
}
