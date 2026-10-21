#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../../../src/game.c"
#include "../../../src/judge.c"
#include "../src/nim.c"

static game_t game;
static oracle_t oracle;

/* A position with the given heaps, and the given player to move. */
static void position_with(game_position_t *position, int a, int b, int c, int turn)
{
	nim_t nim = { {a, b, c}, turn };

	memset(position, 0, sizeof(*position));
	memcpy(position->state.bytes, &nim, sizeof(nim));

	refresh_moves(position);
}

/*
 * A search of the whole game tree, worked out from the rules alone, so that
 * the perfect player's rule can be checked against it. It says one for a
 * win for the player to move and nought for a loss.
 */
static int search_memo[NIM_MAX_HEAP + 1][NIM_MAX_HEAP + 1][NIM_MAX_HEAP + 1];

static int search_value(int a, int b, int c)
{
	int heaps[NIM_HEAPS] = { a, b, c };
	int value = search_memo[a][b][c];

	if (value >= 0) return value;

	for (int heap = 0; heap < NIM_HEAPS; heap++) {
		for (int take = 1; take <= heaps[heap]; take++) {
			int next[NIM_HEAPS];

			for (int i = 0; i < NIM_HEAPS; i++) next[i] = heaps[i];
			next[heap] -= take;

			/* the other player is to move in the position that
			 * follows, so a loss for them is a win for this
			 * player */
			if (search_value(next[0], next[1], next[2]) == 0) {
				search_memo[a][b][c] = 1;

				return 1;
			}
		}
	}

	/* no stones left in any of the heaps: the player to move has lost */
	search_memo[a][b][c] = 0;

	return 0;
}

/* The chooser the perfect player uses: it asks the oracle. */
static game_move_t perfect_choose(void *context, const game_position_t *position)
{
	oracle_t *oracle = (oracle_t *)context;

	return oracle->choose(oracle->state, position);
}

static char *test_a_game_starts_with_its_heaps(void)
{
	game_position_t position;

	game.start(&position);

	_it_should("start with the first heap", nim_state_of(&position)->heaps[0] == 3);
	_it_should("start with the second heap", nim_state_of(&position)->heaps[1] == 4);
	_it_should("start with the third heap", nim_state_of(&position)->heaps[2] == 5);
	_it_should("start with the first player to move", game.to_move(&position) == 0);
	_it_should("offer every move there is", position.n_moves == 3 + 4 + 5);
	return NULL;
}

static char *test_a_move_takes_stones_from_a_heap(void)
{
	game_position_t position;

	game.start(&position);

	_it_should("take the move", game.play(&position, NIM_MOVE(1, 2)));
	_it_should("take the stones out of the heap",
		nim_state_of(&position)->heaps[1] == 2);
	_it_should("leave the other heaps alone", nim_state_of(&position)->heaps[0] == 3);
	_it_should("pass the turn to the other player", game.to_move(&position) == 1);
	_it_should("offer one move fewer", position.n_moves == 3 + 2 + 5);
	return NULL;
}

static char *test_it_refuses_an_illegal_move(void)
{
	game_position_t position;

	game.start(&position);

	_it_should("refuse to take more stones than the heap holds",
		!game.play(&position, NIM_MOVE(0, 4)));
	_it_should("refuse to take no stones at all",
		!game.play(&position, NIM_MOVE(0, 0)));
	_it_should("refuse a heap that is not there",
		!game.play(&position, NIM_MOVE(3, 1)));
	_it_should("leave the heaps alone", nim_state_of(&position)->heaps[0] == 3);
	return NULL;
}

static char *test_the_game_ends_when_there_are_no_stones_left(void)
{
	game_position_t position;

	position_with(&position, 2, 0, 0, 0);

	_it_should("not be over while there are stones", !game.over(&position));

	game.play(&position, NIM_MOVE(0, 2));

	_it_should("be over when there are none", game.over(&position));
	_it_should("have no moves left", position.n_moves == 0);
	_it_should("say the first player took the last stone and won",
		game.outcome(&position) == GAME_FIRST_WINS);
	_it_should("refuse another move", !game.play(&position, NIM_MOVE(0, 1)));
	return NULL;
}

static char *test_the_features_are_the_sizes_of_the_heaps_as_a_fraction(void)
{
	game_position_t position;
	double features[NIM_HEAPS];

	position_with(&position, 1, 2, 3, 0);

	game.features(&position, features);

	_it_should("give the first heap", features[0] == 1.0 / NIM_MAX_HEAP);
	_it_should("give the second heap", features[1] == 2.0 / NIM_MAX_HEAP);
	_it_should("give the third heap", features[2] == 3.0 / NIM_MAX_HEAP);
	return NULL;
}

static char *test_the_perfect_player_knows_what_a_position_is_worth(void)
{
	game_position_t position;

	oracle_start(&oracle);

	/* the exclusive or of three, four and five is two, so the player to
	 * move can win */
	position_with(&position, 3, 4, 5, 0);

	_it_should("say the position is worth one",
		oracle.worth(oracle.state, &position) == 1.0);

	/* and a position whose heaps all have the same bits in them is a loss
	 * for the player to move */
	position_with(&position, 3, 4, 7, 0);

	_it_should("say the position is worth nought",
		oracle.worth(oracle.state, &position) == 0.0);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_the_perfect_player_leaves_a_position_that_is_worth_nothing(void)
{
	game_position_t position;
	game_move_t move;
	int xor = 0;

	oracle_start(&oracle);

	position_with(&position, 3, 4, 5, 0);

	move = oracle.choose(oracle.state, &position);

	_it_should("have a move to make", move >= 0);

	game.play(&position, move);

	for (int heap = 0; heap < NIM_HEAPS; heap++)
		xor ^= nim_state_of(&position)->heaps[heap];

	_it_should("leave a position that is worth nought", xor == 0);
	_it_should("and the oracle agrees",
		oracle.worth(oracle.state, &position) == 0.0);

	oracle_stop(&oracle);

	return NULL;
}

/*
 * The rule the perfect player plays by, checked against a search of the
 * whole game tree, over every position there is.
 */
static char *test_the_rule_agrees_with_a_search(void)
{
	int wrong_worth = 0;
	int wrong_search = 0;

	memset(search_memo, -1, sizeof(search_memo));

	oracle_start(&oracle);

	for (int a = 0; a <= NIM_MAX_HEAP; a++) {
		for (int b = 0; b <= NIM_MAX_HEAP; b++) {
			for (int c = 0; c <= NIM_MAX_HEAP; c++) {
				game_position_t position;
				double expected = (a ^ b ^ c) != 0 ? 1.0 : 0.0;

				position_with(&position, a, b, c, 0);

				if (oracle.worth(oracle.state, &position) != expected)
					wrong_worth++;

				if (search_value(a, b, c) != (expected > 0.5 ? 1 : 0))
					wrong_search++;
			}
		}
	}

	_it_should("say the right worth for every position there is", wrong_worth == 0);
	_it_should("agree with a search of the whole game tree", wrong_search == 0);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_the_perfect_player_agrees_with_the_game(void)
{
	game_position_t position;

	oracle_start(&oracle);

	/* a game that is played out: the perfect player takes the last stone
	 * of the last heap */
	position_with(&position, 1, 0, 0, 0);
	game.play(&position, NIM_MOVE(0, 1));

	_it_should("say the same as the game",
		oracle.outcome(oracle.state, &position) == game.outcome(&position));

	oracle_stop(&oracle);

	return NULL;
}

static char *test_the_referee_finds_nothing_wrong(void)
{
	game_trace_t trace;
	judgement_t verdict;

	oracle_start(&oracle);

	game_play(&game, perfect_choose, &oracle, perfect_choose, &oracle, &trace);

	_it_should("play the game out", trace.n_played > 0);
	_it_should("have no games flagged", judge_game(&game, &oracle, &trace, &verdict));
	_it_should("count every move", verdict.moves == trace.n_played);
	_it_should("find no unexpected moves", verdict.unexpected == 0);
	_it_should("find no wrong results", verdict.result_wrong == 0);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_it_offers_positions_to_look_at(void)
{
	game_position_t samples[GAME_MAX_SAMPLES];
	int count;

	count = game.samples(samples, GAME_MAX_SAMPLES);

	/* the position the game starts in, and every position one move away
	 * from it */
	_it_should("offer them", count == 1 + 3 + 4 + 5);
	_it_should("not offer more than were asked for", count <= GAME_MAX_SAMPLES);
	return NULL;
}

static char *test_the_perfect_player_plays_on_when_it_cannot_win(void)
{
	game_position_t position;
	game_move_t move;

	oracle_start(&oracle);

	/* the exclusive or of three, four and seven is zero, so the position is
	 * a loss for the player to move */
	position_with(&position, 3, 4, 7, 0);

	move = oracle.choose(oracle.state, &position);

	_it_should("have a move to make", move >= 0);
	_it_should("and the move is allowed", game.play(&position, move));

	oracle_stop(&oracle);

	return NULL;
}

/*
 * The move the perfect player plays, checked over every position there is:
 * a position it can win is left as a loss for the opponent, and a position it
 * cannot win is still played on.
 */
static char *test_the_move_is_right_in_every_position(void)
{
	int no_move = 0;
	int wrong_move = 0;

	oracle_start(&oracle);

	for (int a = 0; a <= NIM_MAX_HEAP; a++) {
		for (int b = 0; b <= NIM_MAX_HEAP; b++) {
			for (int c = 0; c <= NIM_MAX_HEAP; c++) {
				game_position_t position;
				game_position_t after;
				game_move_t move;
				const nim_t *heaps;
				int xor = a ^ b ^ c;

				if (a + b + c == 0) continue;

				position_with(&position, a, b, c, 0);

				move = oracle.choose(oracle.state, &position);

				if (move < 0) {
					no_move++;
					continue;
				}

				after = position;

				if (!game.play(&after, move)) {
					wrong_move++;
					continue;
				}

				heaps = nim_state_of(&after);

				/* the opponent is left in a position that is worth
				 * nought, when there was one to leave */
				if (xor != 0 &&
				    (heaps->heaps[0] ^ heaps->heaps[1] ^ heaps->heaps[2]) != 0)
					wrong_move++;
			}
		}
	}

	_it_should("always have a move to make", no_move == 0);
	_it_should("always leave a position worth nought", wrong_move == 0);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_the_nim_sum_is_the_exclusive_or_of_the_heaps(void)
{
	game_position_t position;

	/* the exclusive or of three, four and five is two */
	position_with(&position, 3, 4, 5, 0);

	_it_should("say two", nim_sum(nim_state_of(&position)) == 2);

	/* and of three, four and seven it is zero */
	position_with(&position, 3, 4, 7, 0);

	_it_should("say zero", nim_sum(nim_state_of(&position)) == 0);

	/* and of one, two and four it is seven */
	position_with(&position, 1, 2, 4, 0);

	_it_should("say seven", nim_sum(nim_state_of(&position)) == 7);

	/* and a position with no stones in it has a nim-sum of zero */
	position_with(&position, 0, 0, 0, 0);

	_it_should("say zero", nim_sum(nim_state_of(&position)) == 0);

	return NULL;
}

/*
 * The positions the game can reach: every one of them, no more and no fewer.
 */
static char *test_it_offers_every_position_it_can_reach(void)
{
	game_position_t positions[NIM_MAX_POSITIONS];
	int count;
	int unreachable = 0;
	int wrong_moves = 0;
	int found_start = 0;
	int repeated = 0;

	count = game.positions(positions, NIM_MAX_POSITIONS);

	_it_should("offer every position there is", count == NIM_MAX_POSITIONS);
	_it_should("not offer more than were asked for", count <= NIM_MAX_POSITIONS);

	for (int i = 0; i < count; i++) {
		const nim_t *nim = nim_state_of(&positions[i]);

		/* no heap is bigger than the one the game starts with */
		if (nim->heaps[0] > NIM_START_HEAP_0 ||
		    nim->heaps[1] > NIM_START_HEAP_1 ||
		    nim->heaps[2] > NIM_START_HEAP_2) unreachable++;

		/* and it offers one move for every stone there is */
		if (positions[i].n_moves != nim->heaps[0] + nim->heaps[1] + nim->heaps[2])
			wrong_moves++;

		/* the position the game starts in is one of them */
		if (nim->heaps[0] == NIM_START_HEAP_0 &&
		    nim->heaps[1] == NIM_START_HEAP_1 &&
		    nim->heaps[2] == NIM_START_HEAP_2) found_start++;

		/* and no position is offered twice */
		for (int j = i + 1; j < count; j++) {
			const nim_t *other = nim_state_of(&positions[j]);

			if (nim->heaps[0] == other->heaps[0] &&
			    nim->heaps[1] == other->heaps[1] &&
			    nim->heaps[2] == other->heaps[2]) repeated++;
		}
	}

	_it_should("offer no position that cannot be reached", unreachable == 0);
	_it_should("offer the moves of every position", wrong_moves == 0);
	_it_should("offer the position the game starts in", found_start == 1);
	_it_should("offer no position twice", repeated == 0);

	return NULL;
}

static char *run_tests()
{
	nim_game(&game);
	nim_oracle(&oracle);

	_run_test(test_a_game_starts_with_its_heaps);
	_run_test(test_a_move_takes_stones_from_a_heap);
	_run_test(test_it_refuses_an_illegal_move);
	_run_test(test_the_game_ends_when_there_are_no_stones_left);
	_run_test(test_the_features_are_the_sizes_of_the_heaps_as_a_fraction);
	_run_test(test_the_nim_sum_is_the_exclusive_or_of_the_heaps);
	_run_test(test_it_offers_every_position_it_can_reach);
	_run_test(test_the_perfect_player_knows_what_a_position_is_worth);
	_run_test(test_the_perfect_player_leaves_a_position_that_is_worth_nothing);
	_run_test(test_the_perfect_player_plays_on_when_it_cannot_win);
	_run_test(test_the_move_is_right_in_every_position);
	_run_test(test_the_rule_agrees_with_a_search);
	_run_test(test_the_perfect_player_agrees_with_the_game);
	_run_test(test_the_referee_finds_nothing_wrong);
	_run_test(test_it_offers_positions_to_look_at);

	return NULL;
}
