#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "arena.h"
#include "board.h"
#include "evaluate.h"
#include "game.h"
#include "judge.h"
#include "minimax.h"
#include "network.h"
#include "nim.h"
#include "policy.h"
#include "tictactoe.h"
#include "trainer.h"

int greet(char *message)
{
	return puts(message);
}

#ifndef TEST
#define UNUSED(x) (void)(x)

/*
 * A game in progress, and a network with hand picked weights that spots
 * the threat: the first hidden neuron notices the two X on the top row, and
 * the second one notices that the top right cell is still empty. The one
 * neuron that reads both of them then gives that cell a score.
 */
static void score_a_game(void)
{
	board_t board;
	network_t network;
	double inputs[CELL_COUNT];
	double worth;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_CENTER, 'O');

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[0][1] = 1.0;
	network.layers[0].weights[1][2] = -1.0;
	network.layers[0].biases[1] = 0.5;

	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);
	network.layers[1].weights[0][0] = 1.0;
	network.layers[1].weights[0][1] = -1.0;

	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);

	print_board(&board);

	printf("the position is worth %g to X\n", worth);
}

/*
 * The network the tests use, one change at a time, to see how far each
 * change reaches. The first hidden neuron adds the first two cells together
 * and the second one reads the third cell, and every output neuron reads
 * both of them, so a change to a hidden value moves every score and a change
 * to an output weight moves only one. See test/network.test.c.
 */
static void build_the_experiment_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network->layers[0].weights[0][0] = 1.0;
	network->layers[0].weights[0][1] = 1.0;
	network->layers[0].weights[1][2] = 1.0;

	network_add_layer(network, 2, 1, ACTIVATION_SIGMOID);
	network->layers[1].weights[0][0] = 4.0;
	network->layers[1].weights[0][1] = 1.0;
}

static void print_worth(const char *what, double worth)
{
	printf("%-42s%8g\n", what, worth);
}

static void run_the_experiments(void)
{
	network_t network;
	board_t board;
	double inputs[CELL_COUNT];
	double worth;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	build_the_experiment_network(&network);
	board_to_inputs(&board, inputs);

	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("X at the top left and the top middle", worth);

	network.layers[1].weights[0][0] = 10.0;
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("an output weight: 4 to 10", worth);
	network.layers[1].weights[0][0] = 4.0;

	network.layers[0].weights[0][1] = 3.0;
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("a hidden weight: 1 to 3", worth);
	network.layers[0].weights[0][1] = 1.0;

	network.layers[1].biases[0] = 3.0;
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("an output bias: 0 to 3", worth);
	network.layers[1].biases[0] = 0.0;

	network.layers[0].biases[0] = 1.0;
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("a hidden bias: 0 to 1", worth);
	network.layers[0].biases[0] = 0.0;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("a board with only X at the top left", worth);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, &worth, 1);
	print_worth("a board with X and O on the top row", worth);
}

/*
 * A game played by the network on both sides. The network was told to like
 * the first two numbers and to dislike the third, so the two of them end up
 * fighting over the third.
 */
static void play_a_game(const game_t *game, void (*show)(const game_position_t *))
{
	network_t network;
	policy_t policy;
	game_trace_t trace;

	network_init(&network);
	network_add_layer(&network, game->n_features, 2, ACTIVATION_CLAMP);
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[0][1] = 1.0;
	network.layers[0].weights[1][2] = -1.0;
	network.layers[0].biases[1] = 0.5;

	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);
	network.layers[1].weights[0][0] = 1.0;
	network.layers[1].weights[0][1] = -1.0;

	policy_init(&policy, game, &network, 1, 0.1);
	game_play(game, policy_choose, &policy, policy_choose, &policy, &trace);

	if (show != NULL) show(&trace.position);

	printf("%d moves, and the game %s\n", trace.n_played,
	       game->outcome(&trace.position) == GAME_FIRST_WINS ? "went to the first player" :
	       game->outcome(&trace.position) == GAME_SECOND_WINS ? "went to the second player" :
	       "was drawn");
}

/* The board of a game of tic-tac-toe. */
static void show_the_board(const game_position_t *position)
{
	print_board((const board_t *)position->state.bytes);
}

/* The heaps of a game of Nim. */
static void show_the_heaps(const game_position_t *position)
{
	const nim_t *nim = nim_state_of(position);

	printf("the heaps are %d, %d and %d, and %d stones are left\n",
	       nim->heaps[0], nim->heaps[1], nim->heaps[2],
	       nim->heaps[0] + nim->heaps[1] + nim->heaps[2]);
}

/*
 * A network that watches the top row, and a game where X takes it and wins.
 * The trainer plays the game over and over, and the chance that the player
 * to move wins by taking the top right cell goes up.
 */
static void train_a_network(const game_t *game)
{
	network_t network;
	trainer_t trainer;
	game_trace_t trace;
	game_position_t position;
	double features[GAME_MAX_FEATURES];
	double before;
	double after;
	const game_move_t moves[] = {
		CELL_TOP_LEFT, CELL_CENTER, CELL_TOP, CELL_MIDDLE_LEFT, CELL_TOP_RIGHT
	};

	network_init(&network);
	network_add_layer(&network, game->n_features, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);

	/* the hidden layer watches the two cells on the left of the top row */
	network.layers[0].weights[0][0] = 1.0;
	network.layers[1].weights[1][1] = 1.0;

	/* and the output reads both of them */
	network.layers[1].weights[0][0] = 1.0;
	network.layers[1].weights[0][1] = 1.0;

	trainer_init(&trainer, &network, 0.1);

	/* a game where the first player takes the top row and wins */
	game->start(&position);

	trace.n_offered = 0;
	trace.n_played = 0;

	for (int i = 0; i < 5; i++) {
		trace.offered[trace.n_offered++] = moves[i];
		game->play(&position, moves[i]);
		trace.n_played++;
	}

	trace.position = position;

	/* the position as it is before the top right is taken */
	game->start(&position);
	game->play(&position, CELL_TOP_LEFT);
	game->play(&position, CELL_CENTER);
	game->play(&position, CELL_TOP);
	game->features(&position, features);
	network_forward(&network, features, game->n_features, &before, 1);

	for (int i = 0; i < 20; i++)
		trainer_learn_from_outcome(&trainer, game, &trace);

	game->features(&position, features);
	network_forward(&network, features, game->n_features, &after, 1);

	printf("\nthe position was worth %g, and after training %g\n", before, after);
}

static void measure_a_network(const game_t *game, const oracle_t *oracle)
{
	network_t network;
	policy_t policy;
	trainer_t trainer;
	evaluation_t before;
	evaluation_t after;

	network_init(&network);
	network_add_layer(&network, game->n_features, 8, ACTIVATION_RELU);
	network_add_layer(&network, 8, 1, ACTIVATION_SIGMOID);

	for (int neuron = 0; neuron < network.layers[0].n_neurons; neuron++) {
		for (int input = 0; input < network.layers[0].n_inputs; input++)
			network.layers[0].weights[neuron][input] = 0.5;
	}

	policy_init(&policy, game, &network, 1, 0.3);
	trainer_init(&trainer, &network, 0.05);

	evaluate_network(game, oracle, &network, 1, 200, &before);
	trainer_run(&trainer, game, &policy, 1000);
	evaluate_network(game, oracle, &network, 1, 200, &after);

	printf("\nagainst an opponent that plays at random, over %d games\n", after.games);
	printf("  before training: won %d, drew %d, lost %d, and was out by %g\n",
	       before.wins, before.draws, before.losses, before.brier_score);
	printf("  after training:  won %d, drew %d, lost %d, and was out by %g\n",
	       after.wins, after.draws, after.losses, after.brier_score);

	printf("\nwhat it said, and how often the player to move won:\n");
	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++) {
		if (after.bucket_counts[bucket] == 0) continue;

		printf("  said %g, the mover won %g of the time, over %d moves\n",
		       after.bucket_predicted[bucket], after.bucket_observed[bucket],
		       after.bucket_counts[bucket]);
	}
}

/*
 * The perfect player, asked about a position the network was also asked
 * about. The network gave the top right a chance of 0.88; the truth is
 * that it wins outright.
 */
static void ask_the_minimax(void)
{
	board_t board;
	minimax_result_t result;
	minimax_t oracle;

	minimax_init(&oracle);
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_CENTER, 'O');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_MIDDLE_LEFT, 'O');

	result = minimax_at(&oracle, &board);

	printf("\nthe same position, seen by the perfect player\n");
	printf("  it plays cell %d, and the position is worth %d\n", result.move, result.value);
	printf("  so the player to move wins with a certainty of %g\n",
	       minimax_worth(&oracle, &board));
}

/*
 * A referee, looking at a game in which a move was not allowed: the first
 * move of the game is played, and then offered again.
 */
static void judge_a_game(const game_t *game, const oracle_t *oracle)
{
	game_trace_t trace;
	judgement_t verdict;
	game_position_t position;
	game_move_t move;

	game->start(&position);
	move = position.moves[0];

	trace.n_offered = 0;
	trace.n_played = 0;
	trace.offered[trace.n_offered++] = move;
	game->play(&position, move);
	trace.offered[trace.n_offered++] = move;
	trace.position = position;

	judge_game(game, oracle, &trace, &verdict);

	printf("\nthe referee looked over a game\n");
	printf("  %d moves were offered and %d of them were played\n",
	       verdict.attempts, verdict.moves);
	printf("  %d of the moves should not have been allowed\n", verdict.unexpected);
	printf("  the two reckonings of the result %s\n",
	       verdict.result_wrong ? "disagree" : "agree");
}

/*
 * The three models, started off in the three ways, given the same games
 * and the same number of training steps, and measured on games they have
 * not seen.
 */
static void run_the_arena(const game_t *game, const oracle_t *oracle)
{
	arena_settings_t settings;
	arena_survey_t survey;
	const char *names[3] = { "zero", "random", "maximum" };
	const int runs = 5;

	arena_settings_init(&settings);

	printf("\nthe training arena\n");
	printf("  %d games to learn from, %d to measure with, %d between the models,"
	       " over %d runs with a seed of their own\n\n",
	       settings.training_games, settings.evaluation_games, settings.match_games,
	       runs);

	if (!arena_survey(&settings, game, oracle, runs, &survey)) {
		printf("  the arena could not be run\n");
		return;
	}

	printf("  %-4s %-6s %8s %8s %8s   %s\n", "run", "seed", names[0], names[1],
	       names[2], "the models against each other");

	for (int run = 0; run < survey.runs; run++) {
		arena_report_t *report = &survey.reports[run];
		int wins[ARENA_SCHEMES] = { 0, 0, 0 };

		for (int i = 0; i < ARENA_SCHEMES; i++)
			for (int j = i + 1; j < ARENA_SCHEMES; j++) {
				wins[i] += report->matches[i][j].first_wins;
				wins[j] += report->matches[i][j].second_wins;
			}

		printf("  %-4d %-6llu", run + 1, report->training_seed);
		for (int i = 0; i < ARENA_SCHEMES; i++)
			printf(" %8.3f", report->results[i].truth_after);

		printf("   ");
		for (int i = 0; i < ARENA_SCHEMES; i++)
			printf("%s %d  ", names[i], wins[i]);
		printf("\n");
	}

	printf("\n  what they looked like before they learned anything\n");
	printf("  %-8s %16s %16s %9s %10s %10s\n", "started", "the hidden sums",
	       "the output's sum", "it says", "to a win", "to a loss");

	for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
		initialisation_t start = survey.reports[0].results[scheme].initialisation;

		printf("  %-8s %6.2f to %-6.2f %6.2f to %-6.2f %9.3f %10.3f %10.3f\n",
		       names[scheme],
		       start.hidden_lowest, start.hidden_highest,
		       start.output_lowest, start.output_highest, start.worth,
		       start.gradient_to_one, start.gradient_to_zero);
	}

	printf("\n  over all %d runs\n", survey.runs);

	for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
		double against_random = 0.0;
		double against_itself = 0.0;

		for (int run = 0; run < survey.runs; run++) {
			against_random += survey.reports[run].results[scheme].value_lost_after;
			against_itself += survey.reports[run].results[scheme].value_lost_own_after;
		}

		against_random /= survey.runs;
		against_itself /= survey.runs;

		printf("  %-8s won the round robin %d times out of %d;  it won %d and"
		       " lost %d games against the other models;  against the perfect"
		       " player it won %d, drew %d and lost %d\n",
		       names[scheme], survey.round_robin_wins[scheme], survey.runs,
		       survey.games_won[scheme], survey.games_lost[scheme],
		       survey.perfect_wins[scheme], survey.perfect_draws[scheme],
		       survey.perfect_losses[scheme]);

		printf("           it gave away %.3f a move against a player that plays"
		       " at random, and %.3f in the games it plays against itself\n",
		       against_random, against_itself);
	}
}

/*
 * Two models, each with a seed of its own, and how they fare against each
 * other: before either of them has learned anything, after each has learned
 * from its own games, and after they have learned against each other.
 *
 * Neither is taught by the perfect player here. Each learns from the result
 * of the games it plays, which is what a model has to do when there is
 * nobody to tell it the answer.
 */
static void run_the_duel(const game_t *game, const oracle_t *oracle)
{
	arena_settings_t settings;
	arena_duel_t duel;

	arena_settings_init(&settings);
	settings.training_games = 50000;

	printf("\ntwo models, each with a seed of its own\n");
	printf("  %d games of training each, %d games when they meet\n\n",
	       settings.training_games, settings.match_games);

	if (!arena_duel(&settings, game, oracle, 42, 1978, &duel)) {
		printf("  the duel could not be run\n");
		return;
	}

	const char *stages[ARENA_DUEL_STAGES] = {
		"before either has learned anything",
		"after each has learned from its own games",
		"after they have learned against each other"
	};

	printf("  %-30s %10s %10s\n", "", "seed 42", "seed 1978");

	for (int stage = 0; stage < ARENA_DUEL_STAGES; stage++) {
		arena_stage_result_t *at = &duel.stages[stage];

		printf("\n  %s\n", stages[stage]);
		printf("    %-28s %10.3f %10.3f\n", "gave away a move",
		       at->value_lost[0], at->value_lost[1]);
		printf("    %-28s %10.3f %10.3f\n", "out by the truth",
		       at->truth[0], at->truth[1]);
		printf("    %-28s %10d %10d\n", "drew with the perfect player",
		       at->perfect_draws[0], at->perfect_draws[1]);
		printf("    %-28s %10d %10d\n", "lost to the perfect player",
		       at->perfect_losses[0], at->perfect_losses[1]);
		printf("    %-28s %10d %10d\n", "seed 42 won",
		       at->meeting.first_wins, at->meeting.second_wins);
		printf("    %-28s %10d %10d\n", "the games were drawn",
		       at->meeting.draws, at->meeting.draws);
	}

	printf("\n  the games they learned against each other: %d won, %d drawn,"
	       " %d lost, of %d\n", duel.together_games.first_wins,
	       duel.together_games.draws, duel.together_games.second_wins,
	       duel.together_games.games);
}

/*
 * Where a model is in one of the three rankings of a league.
 */
static int place_of(const arena_league_t *league, const int *ranking, int model)
{
	for (int place = 0; place < league->models; place++) {
		if (ranking[place] == model) return place + 1;
	}

	return 0;
}

/*
 * The whole table of a league: every model, in the order it finished, with
 * the place it took in the other two rankings beside it.
 */
static void print_the_league_table(const arena_league_t *league)
{
	printf("\n  %-6s %6s %8s %10s %8s %6s %6s %6s %8s %8s\n",
	       "place", "seed", "taught", "gave away", "points", "won", "drawn",
	       "lost", "training", "general");

	for (int place = 0; place < league->models; place++) {
		int model = league->ranking_league[place];
		const league_model_t *record = &league->model[model];

		printf("  %-6d %6llu %8s %10.3f %8d %6d %6d %6d %8d %8d\n",
		       place + 1, record->seed,
		       record->teacher == LEAGUE_TRUTH ? "truth" : "self",
		       record->training_value_lost, record->points,
		       record->wins, record->draws, record->losses,
		       place_of(league, league->ranking_training, model),
		       place_of(league, league->ranking_general, model));
	}
}

/*
 * A league: a number of models, each with a seed of its own, every model
 * playing every other one, and the points added up.
 *
 * Half of them are taught by the perfect player and half learn from their
 * own games, so the league holds opponents of two kinds. Three rankings come
 * out of it, and they do not agree: how well a model learned, how well it
 * did in the league, and how well it did against models of the other kind.
 */
static void run_the_league(const game_t *game, const oracle_t *oracle,
                           const char *name, int models, bool table)
{
	arena_settings_t settings;
	arena_league_t league;
	const char *names[3] = { "training", "league", "general" };

	arena_settings_init(&settings);
	settings.training_games = 20000;

	printf("\nthe league of %s\n", name);
	printf("  %d models, %d games of training each, %d games a match\n\n",
	       models, settings.training_games, settings.match_games);

	if (!arena_league(&settings, game, oracle, models, &league)) {
		printf("  the league could not be run\n");
		return;
	}

	printf("  %-12s %6s %8s %10s %8s\n", "ranking", "seed", "taught",
	       "gave away", "points");

	int leaders[3] = {
		league.ranking_training[0],
		league.ranking_league[0],
		league.ranking_general[0]
	};

	for (int i = 0; i < 3; i++) {
		int model = leaders[i];

		printf("  %-12s %6llu %8s %10.3f %8d\n", names[i],
		       league.model[model].seed,
		       league.model[model].teacher == LEAGUE_TRUTH ? "truth" : "self",
		       league.model[model].training_value_lost,
		       league.model[model].points);
	}

	printf("\n  the champion is model %d, with the seed %llu: %d won, %d drawn,"
	       " %d lost\n", league.champion, league.model[league.champion].seed,
	       league.model[league.champion].wins, league.model[league.champion].draws,
	       league.model[league.champion].losses);

	if (table) print_the_league_table(&league);
}

int main(int argc, char *argv[], char *env[]) {
	game_t game;
	oracle_t oracle;
	int models = argc > 1 ? atoi(argv[1]) : 12;
	bool table = argc > 2 && strcmp(argv[2], "table") == 0;

	UNUSED(env);

	greet("Hello, World!");

	/* the game of tic-tac-toe, and the perfect player that goes with it */
	tictactoe_game(&game);
	tictactoe_oracle(&oracle);

	if (!oracle_start(&oracle)) {
		printf("the perfect player could not be started\n");
		return EXIT_FAILURE;
	}

	printf("\nthe game of tic-tac-toe\n");

	score_a_game();
	run_the_experiments();

	printf("\n");
	play_a_game(&game, show_the_board);

	printf("\n");
	train_a_network(&game);

	printf("\n");
	measure_a_network(&game, &oracle);

	printf("\n");
	ask_the_minimax();

	printf("\n");
	judge_a_game(&game, &oracle);

	printf("\n");
	run_the_arena(&game, &oracle);

	printf("\n");
	run_the_duel(&game, &oracle);

	printf("\n");
	run_the_league(&game, &oracle, "tic-tac-toe", models, table);

	oracle_stop(&oracle);

	/* and the very same demonstration, with the game of Nim, to show that
	 * none of it knows what game it is playing */
	nim_game(&game);
	nim_oracle(&oracle);

	if (!oracle_start(&oracle)) {
		printf("the perfect player could not be started\n");
		return EXIT_FAILURE;
	}

	printf("\nthe game of Nim\n");

	printf("\n");
	play_a_game(&game, show_the_heaps);

	printf("\n");
	measure_a_network(&game, &oracle);

	printf("\n");
	judge_a_game(&game, &oracle);

	printf("\n");
	run_the_arena(&game, &oracle);

	printf("\n");
	run_the_league(&game, &oracle, "Nim", models, table);

	oracle_stop(&oracle);

	return EXIT_SUCCESS;
}

#endif
