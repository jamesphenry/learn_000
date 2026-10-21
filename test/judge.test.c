#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/game.c"
#include "../src/judge.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../games/tic-tac-toe/src/board.c"
#include "../games/tic-tac-toe/src/minimax.c"
#include "../games/tic-tac-toe/src/tictactoe.c"

static game_t game;
static oracle_t oracle;

/* The moves that were offered, and where the game ended up. */
static void make_trace(game_trace_t *trace, const game_move_t *moves, int count)
{
	game_position_t position;

	game.start(&position);

	trace->n_offered = 0;
	trace->n_played = 0;

	for (int i = 0; i < count; i++) {
		trace->offered[trace->n_offered++] = moves[i];

		if (!game.play(&position, moves[i])) break;

		trace->n_played++;
	}

	trace->position = position;
}

static char *test_a_legal_game_passes(void)
{
	game_trace_t trace;
	judgement_t verdict;
	policy_t first_policy;
	policy_t second_policy;
	network_t network;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_RELU);
	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);

	policy_init(&first_policy, &game, &network, 1, 0.0);
	policy_init(&second_policy, &game, &network, 2, 0.0);

	game_play(&game, policy_choose, &first_policy, policy_choose, &second_policy, &trace);

	_it_should("be judged good", judge_game(&game, &oracle, &trace, &verdict));
	_it_should("have no unexpected moves", verdict.unexpected == 0);
	_it_should("not disagree about the result", verdict.result_wrong == 0);
	_it_should("count the moves that were offered", verdict.attempts == verdict.moves);
	_it_should("count the moves that were played", verdict.moves > 0);

	return NULL;
}

static char *test_a_move_that_is_not_allowed_is_found(void)
{
	game_trace_t trace;
	judgement_t verdict;
	const game_move_t moves[] = { CELL_TOP_LEFT, CELL_CENTER, CELL_TOP_LEFT };

	make_trace(&trace, moves, 3);

	_it_should("be judged bad", !judge_game(&game, &oracle, &trace, &verdict));
	_it_should("find the move that was not allowed", verdict.unexpected == 1);
	_it_should("count every move that was offered", verdict.attempts == 3);
	_it_should("count the moves that were played", verdict.moves == 2);

	return NULL;
}

static char *test_a_move_after_the_game_is_over_is_found(void)
{
	game_trace_t trace;
	judgement_t verdict;
	const game_move_t moves[] = {
		CELL_TOP_LEFT, CELL_CENTER, CELL_TOP, CELL_MIDDLE_LEFT, CELL_TOP_RIGHT, CELL_BOTTOM
	};

	make_trace(&trace, moves, 6);

	/* X has the top row, so the game is over */
	_it_should("be judged bad", !judge_game(&game, &oracle, &trace, &verdict));
	_it_should("find the move that was not allowed", verdict.unexpected == 1);
	_it_should("count the moves that were played", verdict.moves == 5);

	return NULL;
}

static char *test_a_move_off_the_board_is_found(void)
{
	game_trace_t trace;
	judgement_t verdict;
	const game_move_t moves[] = { CELL_COUNT };

	make_trace(&trace, moves, 1);

	_it_should("be judged bad", !judge_game(&game, &oracle, &trace, &verdict));
	_it_should("find the move that was not allowed", verdict.unexpected == 1);
	_it_should("count no moves that were played", verdict.moves == 0);

	return NULL;
}

static char *test_the_two_reckonings_agree(void)
{
	game_trace_t trace;
	judgement_t verdict;
	const game_move_t moves[] = {
		CELL_TOP_LEFT, CELL_CENTER, CELL_TOP, CELL_MIDDLE_LEFT, CELL_TOP_RIGHT
	};

	make_trace(&trace, moves, 5);

	/* X has the top row */
	_it_should("say that X won", game.outcome(&trace.position) == GAME_FIRST_WINS);
	_it_should("agree with the perfect player",
		judge_game(&game, &oracle, &trace, &verdict) && verdict.result_wrong == 0);

	return NULL;
}

static char *test_the_two_reckonings_agree_on_a_diagonal(void)
{
	game_trace_t trace;
	judgement_t verdict;
	const game_move_t moves[] = {
		CELL_TOP_RIGHT, CELL_TOP_LEFT, CELL_CENTER, CELL_TOP, CELL_BOTTOM_LEFT
	};

	make_trace(&trace, moves, 5);

	/* X has the diagonal from the top right to the bottom left */
	_it_should("say that X won", game.outcome(&trace.position) == GAME_FIRST_WINS);
	_it_should("agree with the perfect player",
		judge_game(&game, &oracle, &trace, &verdict) && verdict.result_wrong == 0);

	return NULL;
}

static char *test_a_move_is_checked_against_the_game(void)
{
	game_position_t position;

	game.start(&position);

	_it_should("allow an empty cell", game.play(&position, CELL_TOP_LEFT));
	_it_should("refuse a cell off the board", !game.play(&position, CELL_COUNT));

	game.start(&position);
	game.play(&position, CELL_TOP_LEFT);

	_it_should("refuse a cell that is taken", !game.play(&position, CELL_TOP_LEFT));
	_it_should("still allow an empty one", game.play(&position, CELL_CENTER));

	return NULL;
}

static char *run_tests()
{
	tictactoe_game(&game);
	tictactoe_oracle(&oracle);
	oracle_start(&oracle);

	_run_test(test_a_legal_game_passes);
	_run_test(test_a_move_that_is_not_allowed_is_found);
	_run_test(test_a_move_after_the_game_is_over_is_found);
	_run_test(test_a_move_off_the_board_is_found);
	_run_test(test_the_two_reckonings_agree);
	_run_test(test_the_two_reckonings_agree_on_a_diagonal);
	_run_test(test_a_move_is_checked_against_the_game);

	oracle_stop(&oracle);

	return NULL;
}
