#include <string.h>
#include "judge.h"
#include "minimax.h"

bool judge_move_is_allowed(const game_t *game, board_cell_t move)
{
	if (game == NULL) return false;
	if (move < 0 || move >= CELL_COUNT) return false;
	if (game->board.cells[move] != VALUE_EMPTY) return false;

	return !game_is_over(game);
}

bool judge_game(const game_t *game, judgement_t *verdict)
{
	game_t replay;
	game_result_t expected;
	char winner;

	if (game == NULL || verdict == NULL) return false;

	memset(verdict, 0, sizeof(*verdict));
	verdict->attempts = game->attempt_count;

	/* the game is played again from the start, and every move that was
	 * offered is checked as it was offered */
	game_init(&replay);

	for (int step = 0; step < game->attempt_count; step++) {
		board_cell_t move = game->attempts[step];

		if (!judge_move_is_allowed(&replay, move))
			verdict->unexpected++;

		if (!game_play_move(&replay, move))
			break;
	}

	verdict->moves = replay.moves;

	/* the result is worked out by the perfect player as well, and the two
	 * are compared */
	winner = minimax_winner(&game->board);

	if (winner == 'X') expected = GAME_X_WINS;
	else if (winner == 'O') expected = GAME_O_WINS;
	else expected = GAME_DRAW;

	if (expected != game_result(game)) verdict->result_wrong++;

	return verdict->unexpected == 0 && verdict->result_wrong == 0;
}
