#include <string.h>
#include "game.h"

bool oracle_start(oracle_t *oracle)
{
	if (oracle == NULL) return false;

	/* a perfect player that needs no memory of its own says so by leaving
	 * make_state alone, and is then asked with a null state */
	if (oracle->make_state == NULL) {
		oracle->state = NULL;

		return true;
	}

	oracle->state = oracle->make_state();

	return oracle->state != NULL;
}

bool oracle_ready(const oracle_t *oracle)
{
	if (oracle == NULL) return false;

	/* the perfect player was started, or it needs no memory of its own */
	if (oracle->make_state != NULL && oracle->state == NULL) return false;

	return true;
}

void oracle_stop(oracle_t *oracle)
{
	if (oracle == NULL) return;

	if (oracle->state != NULL && oracle->free_state != NULL)
		oracle->free_state(oracle->state);

	oracle->state = NULL;
}

bool game_play(const game_t *game, game_chooser_t first, void *first_context,
               game_chooser_t second, void *second_context, game_trace_t *trace)
{
	game_position_t position;

	if (game == NULL || trace == NULL) return false;

	game->start(&position);

	trace->n_offered = 0;
	trace->n_played = 0;

	while (!game->over(&position) && trace->n_offered < GAME_MAX_HISTORY) {
		game_move_t move;
		int turn = game->to_move(&position);

		/* the game says whose turn it is, so a game where the same player
		 * keeps playing, like Nim, needs no special case here */
		if (turn == 0) {
			if (first == NULL) break;

			move = first(first_context, &position);
		} else {
			if (second == NULL) break;

			move = second(second_context, &position);
		}

		if (move < 0) break;

		trace->offered[trace->n_offered++] = move;

		/* a move that is not allowed ends the game, but it is still on
		 * the record, so that a referee can see it */
		if (!game->play(&position, move)) break;

		trace->n_played++;
	}

	trace->position = position;

	return true;
}

int game_replay(const game_t *game, const game_trace_t *trace,
                game_position_t positions[GAME_MAX_HISTORY], game_position_t *final)
{
	game_position_t position;
	int played = 0;

	if (game == NULL || trace == NULL || positions == NULL) return 0;

	game->start(&position);

	for (int step = 0; step < trace->n_offered; step++) {
		game_position_t before = position;

		if (!game->play(&position, trace->offered[step])) break;
		if (played >= GAME_MAX_HISTORY) break;

		positions[played++] = before;
	}

	if (final != NULL) *final = position;

	return played;
}

double game_worth_of_outcome(game_outcome_t outcome, int player)
{
	if (outcome == GAME_DRAW) return 0.5;

	if ((outcome == GAME_FIRST_WINS) == (player == 0)) return 1.0;

	return 0.0;
}
