#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"
#include "../src/game.c"
#include "../src/judge.c"
#include "../src/layer.c"
#include "../src/minimax.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"

static char *test_a_legal_game_passes(void)
{
	game_t game;
	judgement_t verdict;
	policy_t x_policy;
	policy_t o_policy;
	network_t network;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_SIGMOID);

	policy_init(&x_policy, &network, 1, 0.0);
	policy_init(&o_policy, &network, 2, 0.0);

	game_play(&game, &x_policy, &o_policy);

	_it_should("be judged good", judge_game(&game, &verdict));
	_it_should("have no unexpected moves", verdict.unexpected == 0);
	_it_should("not disagree about the result", verdict.result_wrong == 0);
	_it_should("count the moves that were offered", verdict.attempts == verdict.moves);
	_it_should("count the moves that were played", verdict.moves > 0);

	return NULL;
}

static char *test_a_move_that_is_not_allowed_is_found(void)
{
	game_t game;
	judgement_t verdict;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);

	/* the top left is taken, so this one is not allowed */
	game_play_move(&game, CELL_TOP_LEFT);

	_it_should("be judged bad", !judge_game(&game, &verdict));
	_it_should("find the move that was not allowed", verdict.unexpected == 1);
	_it_should("count every move that was offered", verdict.attempts == 3);
	_it_should("count the moves that were played", verdict.moves == 2);

	return NULL;
}

static char *test_a_move_after_the_game_is_over_is_found(void)
{
	game_t game;
	judgement_t verdict;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_TOP_RIGHT);

	/* X has the top row, so the game is over */
	game_play_move(&game, CELL_BOTTOM);

	_it_should("be judged bad", !judge_game(&game, &verdict));
	_it_should("find the move that was not allowed", verdict.unexpected == 1);
	_it_should("count the moves that were played", verdict.moves == 5);

	return NULL;
}

static char *test_a_move_off_the_board_is_found(void)
{
	game_t game;
	judgement_t verdict;

	game_init(&game);
	game_play_move(&game, (board_cell_t)CELL_COUNT);

	_it_should("be judged bad", !judge_game(&game, &verdict));
	_it_should("find the move that was not allowed", verdict.unexpected == 1);
	_it_should("count no moves that were played", verdict.moves == 0);

	return NULL;
}

static char *test_the_two_reckonings_agree(void)
{
	game_t game;
	judgement_t verdict;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_TOP_RIGHT);

	/* X has the top row */
	_it_should("say that X won", minimax_winner(&game.board) == 'X');
	_it_should("agree with the board module",
		judge_game(&game, &verdict) && verdict.result_wrong == 0);

	return NULL;
}

static char *test_the_two_reckonings_agree_on_a_diagonal(void)
{
	game_t game;
	judgement_t verdict;

	game_init(&game);
	game_play_move(&game, CELL_TOP_RIGHT);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_BOTTOM_LEFT);

	/* X has the diagonal from the top right to the bottom left */
	_it_should("say that X won", minimax_winner(&game.board) == 'X');
	_it_should("agree with the board module",
		judge_game(&game, &verdict) && verdict.result_wrong == 0);

	return NULL;
}

static char *test_a_move_is_checked_against_the_game(void)
{
	game_t game;

	game_init(&game);

	_it_should("allow an empty cell", judge_move_is_allowed(&game, CELL_TOP_LEFT));
	_it_should("refuse a cell off the board",
		!judge_move_is_allowed(&game, (board_cell_t)CELL_COUNT));

	game_play_move(&game, CELL_TOP_LEFT);

	_it_should("refuse a cell that is taken", !judge_move_is_allowed(&game, CELL_TOP_LEFT));
	_it_should("still allow an empty one", judge_move_is_allowed(&game, CELL_CENTER));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_a_legal_game_passes);
	_run_test(test_a_move_that_is_not_allowed_is_found);
	_run_test(test_a_move_after_the_game_is_over_is_found);
	_run_test(test_a_move_off_the_board_is_found);
	_run_test(test_the_two_reckonings_agree);
	_run_test(test_the_two_reckonings_agree_on_a_diagonal);
	_run_test(test_a_move_is_checked_against_the_game);

	return NULL;
}
