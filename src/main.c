#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include "arena.h"
#include "board.h"
#include "game.h"
#include "judge.h"
#include "minimax.h"
#include "network.h"
#include "evaluate.h"
#include "trainer.h"
#include "policy.h"

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
 * the top right cell, and O does the opposite, so the two of them end up
 * fighting over it.
 */
static void play_a_game(void)
{
	network_t network;
	policy_t policy;
	game_t game;
	game_result_t result;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[0][1] = 1.0;
	network.layers[0].weights[1][2] = -1.0;
	network.layers[0].biases[1] = 0.5;

	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);
	network.layers[1].weights[0][0] = 1.0;
	network.layers[1].weights[0][1] = -1.0;

	policy_init(&policy, &network, 1, 0.1);
	result = game_play(&game, &policy, &policy);

	print_board(&game.board);

	printf("%d moves, and the game %s\n", game.moves,
	       result == GAME_X_WINS ? "went to X" :
	       result == GAME_O_WINS ? "went to O" : "was drawn");
}

/*
 * A network that watches the top row, and a game where X takes it and wins.
 * The trainer plays the game over and over, and the chance that X wins by
 * taking the top right cell goes up.
 */
static void train_a_network(void)
{
	network_t network;
	trainer_t trainer;
	game_t game;
	board_t board;
	double inputs[CELL_COUNT];
	double before;
	double after;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, 1, ACTIVATION_SIGMOID);

	/* the hidden layer watches the two cells on the left of the top row */
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[1][1] = 1.0;

	/* and the output reads both of them */
	network.layers[1].weights[0][0] = 1.0;
	network.layers[1].weights[0][1] = 1.0;

	trainer_init(&trainer, &network, 0.1);

	/* the board as it is before the top right is taken */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, &before, 1);

	/* a game where X takes the top row and wins */
	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_TOP_RIGHT);

	for (int i = 0; i < 20; i++)
		trainer_train_game(&trainer, &game);

	network_forward(&network, inputs, CELL_COUNT, &after, 1);

	printf("\nthe position was worth %g, and after training %g\n", before, after);
}

static void measure_a_network(void)
{
	network_t network;
	policy_t policy;
	trainer_t trainer;
	evaluation_t before;
	evaluation_t after;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 8, ACTIVATION_RELU);
	network_add_layer(&network, 8, 1, ACTIVATION_SIGMOID);

	for (int neuron = 0; neuron < network.layers[0].n_neurons; neuron++) {
		for (int input = 0; input < network.layers[0].n_inputs; input++)
			network.layers[0].weights[neuron][input] = 0.5;
	}

	policy_init(&policy, &network, 1, 0.3);
	trainer_init(&trainer, &network, 0.05);

	evaluate_network(&network, 1, 200, &before);
	trainer_run(&trainer, &policy, 1000);
	evaluate_network(&network, 1, 200, &after);

	printf("\nagainst an opponent that plays at random, over %d games\n", after.games);
	printf("  before training: won %d, drew %d, lost %d, and was out by %g\n",
	       before.wins, before.draws, before.losses, before.brier_score);
	printf("  after training:  won %d, drew %d, lost %d, and was out by %g\n",
	       after.wins, after.draws, after.losses, after.brier_score);

	printf("\nwhat it said, and how often X won:\n");
	for (int bucket = 0; bucket < EVALUATION_BUCKETS; bucket++) {
		if (after.bucket_counts[bucket] == 0) continue;

		printf("  said %g, X won %g of the time, over %d moves\n",
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
	printf("  so X wins with a certainty of %g\n", minimax_x_win_chance(&oracle, &board));
}

/*
 * A referee, looking at a game in which a move was not allowed.
 */
static void judge_a_game(void)
{
	game_t game;
	judgement_t verdict;

	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);

	/* the top left is taken, so this one is not allowed */
	game_play_move(&game, CELL_TOP_LEFT);

	judge_game(&game, &verdict);

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
static void run_the_arena(void)
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

	if (!arena_survey(&settings, runs, &survey)) {
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
		       " player it drew %d and lost %d\n",
		       names[scheme], survey.round_robin_wins[scheme], survey.runs,
		       survey.games_won[scheme], survey.games_lost[scheme],
		       survey.perfect_draws[scheme], survey.perfect_losses[scheme]);

		printf("           it gave away %.3f a move against a player that plays"
		       " at random, and %.3f in the games it plays against itself\n",
		       against_random, against_itself);
	}
}

int main(int argc, char *argv[], char *env[]) {
	UNUSED(argc);
	UNUSED(argv);
	UNUSED(env);

	greet("Hello, World!");
	score_a_game();
	run_the_experiments();

	printf("\n");
	play_a_game();

	printf("\n");
	train_a_network();

	printf("\n");
	measure_a_network();

	printf("\n");
	ask_the_minimax();

	printf("\n");
	judge_a_game();

	printf("\n");
	run_the_arena();

	return EXIT_SUCCESS;
}

#endif
