#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>

#include "../src/game.c"

/*
 * A toy game, so that the framework can be tested without a game in sight: a
 * pile of stones, and the players take one at a time. The player who takes
 * the last one wins.
 *
 * The state lives in the state of a position, and the only move there is
 * move zero.
 */
#define TOY_STONES 5

typedef struct {
	int stones;
	int turn;
} toy_t;

static toy_t *toy_of(game_position_t *position)
{
	return (toy_t *)position->state.bytes;
}

static const toy_t *toy_in(const game_position_t *position)
{
	return (const toy_t *)position->state.bytes;
}

static void toy_start(game_position_t *position)
{
	toy_t *toy = toy_of(position);

	toy->stones = TOY_STONES;
	toy->turn = 0;

	position->n_moves = 1;
	position->moves[0] = 0;
}

static bool toy_play(game_position_t *position, game_move_t move)
{
	toy_t *toy = toy_of(position);

	if (move != 0) return false;
	if (toy->stones < 1) return false;

	toy->stones--;
	toy->turn = 1 - toy->turn;

	position->n_moves = (toy->stones > 0) ? 1 : 0;

	return true;
}

static void toy_features(const game_position_t *position, double *out)
{
	out[0] = (double)toy_in(position)->stones;
}

static bool toy_over(const game_position_t *position)
{
	return toy_in(position)->stones < 1;
}

static int toy_to_move(const game_position_t *position)
{
	return toy_in(position)->turn;
}

static game_outcome_t toy_outcome(const game_position_t *position)
{
	/* the player to move has run out of stones, so the other one won */
	if (toy_in(position)->turn == 0) return GAME_SECOND_WINS;

	return GAME_FIRST_WINS;
}

static int toy_samples(game_position_t *out, int wanted)
{
	int count = 0;

	if (out == NULL || wanted < 1) return 0;

	for (int stones = TOY_STONES; stones >= 0 && count < wanted; stones--) {
		toy_start(&out[count]);
		toy_of(&out[count])->stones = stones;
		count++;
	}

	return count;
}

static void toy_game(game_t *game)
{
	game->n_features = 1;
	game->start = toy_start;
	game->play = toy_play;
	game->features = toy_features;
	game->over = toy_over;
	game->to_move = toy_to_move;
	game->outcome = toy_outcome;
	game->samples = toy_samples;
}

/*
 * The perfect player for the toy game: with an odd number of stones the
 * player to move takes the last one, and with an even number the other one
 * does.
 */
static double toy_worth(void *state, const game_position_t *position)
{
	(void)state;

	return (toy_in(position)->stones % 2 == 1) ? 1.0 : 0.0;
}

static game_move_t toy_choose(void *state, const game_position_t *position)
{
	(void)state;

	if (toy_over(position)) return -1;

	return 0;
}

static game_outcome_t toy_oracle_outcome(void *state, const game_position_t *final)
{
	(void)state;

	return toy_outcome(final);
}

static void *toy_make_state(void)
{
	return malloc(sizeof(int));
}

static void toy_free_state(void *state)
{
	free(state);
}

/* A perfect player that asks for memory of its own and cannot have it. */
static void *toy_make_state_that_fails(void)
{
	return NULL;
}

/*
 * The perfect player for the toy game. It needs no memory of its own: it
 * works the answer out from the pile of stones every time.
 */
static void toy_oracle(oracle_t *oracle)
{
	oracle->state = NULL;
	oracle->worth = toy_worth;
	oracle->choose = toy_choose;
	oracle->outcome = toy_oracle_outcome;
	oracle->make_state = NULL;
	oracle->free_state = NULL;
}

/* The chooser the tests use: it always takes the only move there is. */
static game_move_t always_take(void *context, const game_position_t *position)
{
	(void)context;

	if (position->n_moves < 1) return -1;

	return position->moves[0];
}

/* The chooser that refuses to move. */
static game_move_t never_move(void *context, const game_position_t *position)
{
	(void)context;
	(void)position;

	return -1;
}

static char *test_a_game_is_played_to_the_end(void)
{
	game_t game;
	game_trace_t trace;

	toy_game(&game);

	game_play(&game, always_take, NULL, always_take, NULL, &trace);

	_it_should("have played every stone", trace.n_played == TOY_STONES);
	_it_should("have offered every move", trace.n_offered == TOY_STONES);
	_it_should("have come to an end", game.over(&trace.position));
	_it_should("have gone to the first player, who took the last stone",
		game.outcome(&trace.position) == GAME_FIRST_WINS);

	return NULL;
}

static char *test_the_turn_goes_back_and_forth(void)
{
	game_t game;
	game_position_t position;

	toy_game(&game);

	game.start(&position);

	_it_should("start with the first player", game.to_move(&position) == 0);

	game.play(&position, 0);

	_it_should("pass the turn to the other player", game.to_move(&position) == 1);

	return NULL;
}

static char *test_a_game_stops_when_a_side_cannot_move(void)
{
	game_t game;
	game_trace_t trace;

	toy_game(&game);

	game_play(&game, never_move, NULL, always_take, NULL, &trace);

	_it_should("have played no moves at all", trace.n_played == 0);
	_it_should("have offered no moves at all", trace.n_offered == 0);

	return NULL;
}

static char *test_a_game_is_played_again_from_the_start(void)
{
	game_t game;
	game_trace_t trace;
	game_position_t positions[GAME_MAX_HISTORY];
	game_position_t final;
	int played;

	toy_game(&game);

	game_play(&game, always_take, NULL, always_take, NULL, &trace);

	played = game_replay(&game, &trace, positions, &final);

	_it_should("have found every position", played == TOY_STONES);
	_it_should("start again with the whole pile", toy_in(&positions[0])->stones == TOY_STONES);
	_it_should("have the turn back again", toy_in(&positions[0])->turn == 0);
	_it_should("end where the game ended", toy_in(&final)->stones == 0);

	return NULL;
}

static char *test_a_move_that_is_not_allowed_is_refused(void)
{
	game_t game;
	game_position_t position;

	toy_game(&game);

	game.start(&position);
	game.play(&position, 0);
	game.play(&position, 0);

	/* a move that is not move zero is not allowed */
	_it_should("refuse a move that is not there", !game.play(&position, 7));
	_it_should("leave the pile alone", toy_in(&position)->stones == TOY_STONES - 2);

	return NULL;
}

static char *test_the_worth_of_a_result(void)
{
	_it_should("say a win is worth one", game_worth_of_outcome(GAME_FIRST_WINS, 0) == 1.0);
	_it_should("say a win is worth nought to the other one",
		game_worth_of_outcome(GAME_FIRST_WINS, 1) == 0.0);
	_it_should("say a draw is worth a half", game_worth_of_outcome(GAME_DRAW, 0) == 0.5);
	_it_should("say a draw is worth a half to the other one too",
		game_worth_of_outcome(GAME_DRAW, 1) == 0.5);
	_it_should("say a loss is worth nought", game_worth_of_outcome(GAME_SECOND_WINS, 0) == 0.0);
	_it_should("say a loss is worth one to the other one",
		game_worth_of_outcome(GAME_SECOND_WINS, 1) == 1.0);

	return NULL;
}

static char *test_the_perfect_player_knows_the_game(void)
{
	game_t game;
	oracle_t oracle;
	game_position_t position;

	toy_game(&game);
	toy_oracle(&oracle);

	oracle_start(&oracle);

	game.start(&position);

	_it_should("say the first player wins the pile of five",
		oracle.worth(oracle.state, &position) == 1.0);

	game.play(&position, 0);

	_it_should("say the second player wins the pile of four",
		oracle.worth(oracle.state, &position) == 0.0);

	_it_should("play the only move there is", oracle.choose(oracle.state, &position) == 0);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_a_perfect_player_that_needs_no_memory(void)
{
	oracle_t oracle;

	toy_oracle(&oracle);

	_it_should("be ready without any memory of its own", oracle_ready(&oracle));
	_it_should("be started without any memory of its own", oracle_start(&oracle));
	_it_should("still be ready", oracle_ready(&oracle));
	_it_should("have no memory of its own", oracle.state == NULL);

	oracle_stop(&oracle);

	return NULL;
}

static char *test_a_perfect_player_that_asks_for_memory(void)
{
	oracle_t oracle;

	toy_oracle(&oracle);
	oracle.make_state = toy_make_state;
	oracle.free_state = toy_free_state;

	_it_should("not be ready before it is started", !oracle_ready(&oracle));
	_it_should("be started", oracle_start(&oracle));
	_it_should("be ready once it is started", oracle_ready(&oracle));
	_it_should("have the memory it asked for", oracle.state != NULL);

	oracle_stop(&oracle);

	_it_should("have given the memory back", oracle.state == NULL);

	return NULL;
}

static char *test_a_perfect_player_that_cannot_have_its_memory(void)
{
	oracle_t oracle;

	toy_oracle(&oracle);
	oracle.make_state = toy_make_state_that_fails;

	_it_should("not be started", !oracle_start(&oracle));
	_it_should("not be ready", !oracle_ready(&oracle));

	return NULL;
}

static char *run_tests(void)
{
	_run_test(test_a_game_is_played_to_the_end);
	_run_test(test_the_turn_goes_back_and_forth);
	_run_test(test_a_game_stops_when_a_side_cannot_move);
	_run_test(test_a_game_is_played_again_from_the_start);
	_run_test(test_a_move_that_is_not_allowed_is_refused);
	_run_test(test_the_worth_of_a_result);
	_run_test(test_the_perfect_player_knows_the_game);
	_run_test(test_a_perfect_player_that_needs_no_memory);
	_run_test(test_a_perfect_player_that_asks_for_memory);
	_run_test(test_a_perfect_player_that_cannot_have_its_memory);

	return NULL;
}
