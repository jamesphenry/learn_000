#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include "board.h"
#include "game.h"
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
	double scores[CELL_COUNT];

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

	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_CLAMP);
	network.layers[1].weights[2][0] = 1.0;
	network.layers[1].weights[2][1] = -1.0;

	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);

	print_board(&board);

	printf("scores\n");
	for (int row = 0; row < BOARD_SIZE; row++) {
		printf(" %g | %g | %g \n", scores[row * BOARD_SIZE + 0],
			scores[row * BOARD_SIZE + 1], scores[row * BOARD_SIZE + 2]);

		if (row < 2) {
			printf("---+---+---\n");
		}
	}
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

	network_add_layer(network, 2, CELL_COUNT, ACTIVATION_CLAMP);
	for (int cell = 0; cell < CELL_COUNT; cell++) {
		network->layers[1].weights[cell][0] = (double)(cell + 1);
		network->layers[1].weights[cell][1] = 1.0;
	}
}

static void print_scores(const char *what, const double *scores)
{
	printf("%-38s", what);

	for (int cell = 0; cell < CELL_COUNT; cell++)
		printf("%6g", scores[cell]);

	printf("\n");
}

static void run_the_experiments(void)
{
	network_t network;
	board_t board;
	double inputs[CELL_COUNT];
	double scores[CELL_COUNT];

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	build_the_experiment_network(&network);
	board_to_inputs(&board, inputs);

	printf("\n%-38s", "");
	for (int cell = 0; cell < CELL_COUNT; cell++)
		printf("%6d", cell);
	printf("\n");

	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("X at the top left and the top middle", scores);

	network.layers[1].weights[3][0] = 10.0;
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("an output weight of cell 3: 4 to 10", scores);
	network.layers[1].weights[3][0] = 4.0;

	network.layers[0].weights[0][1] = 3.0;
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("a hidden weight: 1 to 3", scores);
	network.layers[0].weights[0][1] = 1.0;

	network.layers[1].biases[5] = 3.0;
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("an output bias of cell 5: 0 to 3", scores);
	network.layers[1].biases[5] = 0.0;

	network.layers[0].biases[0] = 1.0;
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("a hidden bias: 0 to 1", scores);
	network.layers[0].biases[0] = 0.0;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("a board with only X at the top left", scores);

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');
	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, scores, CELL_COUNT);
	print_scores("a board with X and O on the top row", scores);
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

	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_CLAMP);
	network.layers[1].weights[2][0] = 1.0;
	network.layers[1].weights[2][1] = -1.0;

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
	double before[CELL_COUNT];
	double after[CELL_COUNT];

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2, ACTIVATION_CLAMP);
	network_add_layer(&network, 2, CELL_COUNT, ACTIVATION_SIGMOID);

	/* the hidden layer watches the two cells on the left of the top row */
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[1][1] = 1.0;

	/* and the output for the top right cell reads both of them */
	network.layers[1].weights[CELL_TOP_RIGHT][0] = 1.0;
	network.layers[1].weights[CELL_TOP_RIGHT][1] = 1.0;

	trainer_init(&trainer, &network, 0.1);

	/* the board as it is before the top right is taken */
	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	board_to_inputs(&board, inputs);
	network_forward(&network, inputs, CELL_COUNT, before, CELL_COUNT);

	/* a game where X takes the top row and wins */
	game_init(&game);
	game_play_move(&game, CELL_TOP_LEFT);
	game_play_move(&game, CELL_CENTER);
	game_play_move(&game, CELL_TOP);
	game_play_move(&game, CELL_MIDDLE_LEFT);
	game_play_move(&game, CELL_TOP_RIGHT);

	for (int i = 0; i < 20; i++)
		trainer_train_game(&trainer, &game);

	network_forward(&network, inputs, CELL_COUNT, after, CELL_COUNT);

	printf("\nthe top right cell was scored %g, and after training %g\n",
	       before[CELL_TOP_RIGHT], after[CELL_TOP_RIGHT]);
}

static void measure_a_network(void)
{
	network_t network;
	policy_t policy;
	trainer_t trainer;
	evaluation_t before;
	evaluation_t after;

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 8, ACTIVATION_CLAMP);
	network_add_layer(&network, 8, CELL_COUNT, ACTIVATION_SIGMOID);

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

	return EXIT_SUCCESS;
}

#endif
