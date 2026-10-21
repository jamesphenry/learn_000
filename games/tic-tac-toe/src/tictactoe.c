#include <stdlib.h>
#include "tictactoe.h"

/*
 * The board lives in the state of a position, so the framework never has to
 * know what a tic-tac-toe position looks like.
 */
static board_t *board_of(game_position_t *position)
{
	return (board_t *)position->state.bytes;
}

static const board_t *board_in(const game_position_t *position)
{
	return (const board_t *)position->state.bytes;
}

/* The cells that are still empty. */
static void refresh_moves(game_position_t *position)
{
	const board_t *board = board_in(position);
	int count = 0;

	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (board->cells[cell] == VALUE_EMPTY)
			position->moves[count++] = cell;
	}

	position->n_moves = count;
}

static void tictactoe_start(game_position_t *position)
{
	board_init(board_of(position));

	refresh_moves(position);
}

static bool tictactoe_over(const game_position_t *position)
{
	const board_t *board = board_in(position);

	if (detect_win(board, 'X')) return true;
	if (detect_win(board, 'O')) return true;

	return board_is_full(board);
}

static bool tictactoe_play(game_position_t *position, game_move_t move)
{
	board_t *board = board_of(position);

	if (tictactoe_over(position)) return false;
	if (move < 0 || move >= CELL_COUNT) return false;
	if (board->cells[move] != VALUE_EMPTY) return false;

	if (!place_mark(board, (board_cell_t)move, board->current_player)) return false;

	board->current_player = (board->current_player == 'X') ? 'O' : 'X';

	refresh_moves(position);

	return true;
}

static void tictactoe_features(const game_position_t *position, double *out)
{
	board_to_inputs(board_in(position), out);
}

static int tictactoe_to_move(const game_position_t *position)
{
	return board_in(position)->current_player == 'X' ? 0 : 1;
}

static game_outcome_t tictactoe_outcome(const game_position_t *position)
{
	const board_t *board = board_in(position);

	if (detect_win(board, 'X')) return GAME_FIRST_WINS;
	if (detect_win(board, 'O')) return GAME_SECOND_WINS;

	return GAME_DRAW;
}

/*
 * A few positions to look at before a network has learned anything: the
 * empty board, and every board with one mark on it and another mark off it.
 */
static int tictactoe_samples(game_position_t *out, int wanted)
{
	int count = 0;

	if (out == NULL || wanted < 1) return 0;

	tictactoe_start(&out[count++]);

	for (int first = 0; first < CELL_COUNT && count < wanted; first++) {
		for (int second = 0; second < CELL_COUNT && count < wanted; second++) {
			board_t *board;

			if (first == second) continue;

			tictactoe_start(&out[count]);
			board = board_of(&out[count]);

			place_mark(board, (board_cell_t)first, 'X');
			place_mark(board, (board_cell_t)second, 'O');

			count++;
		}
	}

	return count;
}

void tictactoe_game(game_t *game)
{
	if (game == NULL) return;

	game->n_features = CELL_COUNT;
	game->start = tictactoe_start;
	game->play = tictactoe_play;
	game->features = tictactoe_features;
	game->over = tictactoe_over;
	game->to_move = tictactoe_to_move;
	game->outcome = tictactoe_outcome;
	game->samples = tictactoe_samples;
}

/*
 * The perfect player. It works out the worth of every position there is and
 * keeps it, so the table belongs to the perfect player and is made when the
 * oracle starts and given back when it stops.
 */
static void *tictactoe_make_state(void)
{
	minimax_t *minimax = malloc(sizeof(*minimax));

	if (minimax != NULL) minimax_init(minimax);

	return minimax;
}

static void tictactoe_free_state(void *state)
{
	free(state);
}

static double tictactoe_worth(void *state, const game_position_t *position)
{
	return minimax_worth((minimax_t *)state, board_in(position));
}

static game_move_t tictactoe_choose(void *state, const game_position_t *position)
{
	minimax_result_t result = minimax_at((minimax_t *)state, board_in(position));

	return result.move;
}

/*
 * The result, worked out by the perfect player from the lines of three
 * alone, so that it can disagree with the board module.
 */
static game_outcome_t tictactoe_oracle_outcome(void *state, const game_position_t *final)
{
	char winner;

	/* the winner is worked out from the lines of three alone, so the
	 * perfect player needs nothing it has worked out */
	(void)state;

	winner = minimax_winner(board_in(final));

	if (winner == 'X') return GAME_FIRST_WINS;
	if (winner == 'O') return GAME_SECOND_WINS;

	return GAME_DRAW;
}

void tictactoe_oracle(oracle_t *oracle)
{
	if (oracle == NULL) return;

	oracle->state = NULL;
	oracle->worth = tictactoe_worth;
	oracle->choose = tictactoe_choose;
	oracle->outcome = tictactoe_oracle_outcome;
	oracle->make_state = tictactoe_make_state;
	oracle->free_state = tictactoe_free_state;
}
