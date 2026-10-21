#include "game.h"

void game_init(game_t *game)
{
	board_init(&game->board);
	game->moves = 0;
	game->attempt_count = 0;
}

char game_turn(const game_t *game)
{
	return game->board.current_player;
}

bool game_play_move(game_t *game, board_cell_t cell)
{
	/* the move is written down whether it was allowed or not, so that a
	 * referee can see it */
	if (game->attempt_count < GAME_MAX_ATTEMPTS)
		game->attempts[game->attempt_count] = cell;
	game->attempt_count++;

	if (game_is_over(game)) return false;

	if (!place_mark(&game->board, cell, game->board.current_player))
		return false;

	game->history[game->moves] = cell;
	game->moves++;
	game->board.current_player = (game->board.current_player == 'X') ? 'O' : 'X';

	return true;
}

bool game_is_over(const game_t *game)
{
	if (detect_win(&game->board, 'X')) return true;
	if (detect_win(&game->board, 'O')) return true;

	return board_is_full(&game->board);
}

game_result_t game_result(const game_t *game)
{
	if (detect_win(&game->board, 'X')) return GAME_X_WINS;
	if (detect_win(&game->board, 'O')) return GAME_O_WINS;

	return GAME_DRAW;
}

game_result_t game_play(game_t *game, policy_t *x_policy, policy_t *o_policy)
{
	game_init(game);

	while (!game_is_over(game)) {
		policy_t *policy = (game_turn(game) == 'X') ? x_policy : o_policy;
		board_cell_t move;

		if (!policy_select_move(policy, &game->board, &move)) break;

		game_play_move(game, move);
	}

	return game_result(game);
}

double game_worth_of_result(char player_to_move, game_result_t outcome)
{
	if (outcome == GAME_DRAW) return 0.5;

	if ((outcome == GAME_X_WINS) == (player_to_move == 'X')) return 1.0;

	return 0.0;
}

void game_positions(const game_t *game, board_t positions[CELL_COUNT])
{
	game_t replay;

	game_init(&replay);

	for (int step = 0; step < game->moves; step++) {
		positions[step] = replay.board;

		if (!game_play_move(&replay, game->history[step])) break;
	}
}
