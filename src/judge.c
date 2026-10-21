#include <string.h>
#include "judge.h"

bool judge_game(const game_t *game, const oracle_t *oracle, const game_trace_t *trace,
                judgement_t *verdict)
{
	game_position_t position;
	int played = 0;

	if (game == NULL || oracle == NULL || trace == NULL || verdict == NULL) return false;

	memset(verdict, 0, sizeof(*verdict));
	verdict->attempts = trace->n_offered;

	/* the game is played again from the start, and every move that was
	 * offered is checked as it was offered */
	game->start(&position);

	for (int step = 0; step < trace->n_offered; step++) {
		if (!game->play(&position, trace->offered[step])) {
			verdict->unexpected++;
			break;
		}

		played++;
	}

	verdict->moves = played;

	/* the result is worked out by the perfect player as well, and the two
	 * are compared */
	if (oracle->outcome != NULL && oracle->state != NULL) {
		if (oracle->outcome(oracle->state, &trace->position) !=
		    game->outcome(&trace->position))
			verdict->result_wrong++;
	}

	return verdict->unexpected == 0 && verdict->result_wrong == 0;
}
