#include <math.h>
#include "policy.h"

/*
 * The worth of a position to X: one if X wins it, a half if it is drawn and
 * nought if O wins it.
 */
bool policy_position_worth(const network_t *network, const board_t *board, double *worth)
{
	double inputs[CELL_COUNT];

	if (network == NULL || board == NULL || worth == NULL) return false;

	board_to_inputs(board, inputs);

	return network_forward(network, inputs, CELL_COUNT, worth, 1);
}

bool policy_init(policy_t *policy, network_t *network, unsigned long long seed,
                 double temperature)
{
	if (policy == NULL || network == NULL) return false;

	/* the policy looks ahead, so it needs a network that answers with the
	 * worth of a position */
	if (network_output_size(network) != 1) return false;

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
	double scores[CELL_COUNT] = { 0.0 };
	double preference[CELL_COUNT] = { 0.0 };
	double weights[CELL_COUNT];
	board_cell_t cells[CELL_COUNT];
	int count;
	double biggest;
	double total = 0.0;
	double pick;

	if (policy == NULL || board == NULL || move == NULL) return false;
	if (policy->network == NULL) return false;

	count = legal_cells(board, cells);
	if (count == 0) return false;

	/* the worth of the position each cell would leave behind */
	for (int i = 0; i < count; i++) {
		board_t next = *board;

		if (!place_mark(&next, cells[i], board->current_player)) return false;

		next.current_player = (board->current_player == 'X') ? 'O' : 'X';

		if (!policy_position_worth(policy->network, &next, &scores[cells[i]])) return false;
	}

	/* each successor is worth what the opponent would get out of it, so the
	 * player to move wants the smallest of them */
	for (int i = 0; i < count; i++)
		preference[i] = -scores[cells[i]];

	biggest = preference[0];
	for (int i = 1; i < count; i++) {
		if (preference[i] > biggest)
			biggest = preference[i];
	}

	if (policy->temperature <= 0.0) {
		/* take the best cell, and break a tie at random */
		int ties = 0;

		for (int i = 0; i < count; i++) {
			if (preference[i] == biggest)
				ties++;
		}

		int choice = rng_next_int(&policy->rng, ties);

		for (int i = 0; i < count; i++) {
			if (preference[i] != biggest) continue;

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
		weights[i] = exp((preference[i] - biggest) / policy->temperature);
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
