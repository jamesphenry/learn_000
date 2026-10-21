#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include "board.h"
#include "game.h"
#include "network.h"
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
	network_add_layer(&network, CELL_COUNT, 2);
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[0][1] = 1.0;
	network.layers[0].weights[1][2] = -1.0;
	network.layers[0].biases[1] = 0.5;

	network_add_layer(&network, 2, CELL_COUNT);
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
	network_add_layer(network, CELL_COUNT, 2);
	network->layers[0].weights[0][0] = 1.0;
	network->layers[0].weights[0][1] = 1.0;
	network->layers[0].weights[1][2] = 1.0;

	network_add_layer(network, 2, CELL_COUNT);
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
	network_add_layer(&network, CELL_COUNT, 2);
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[0][1] = 1.0;
	network.layers[0].weights[1][2] = -1.0;
	network.layers[0].biases[1] = 0.5;

	network_add_layer(&network, 2, CELL_COUNT);
	network.layers[1].weights[2][0] = 1.0;
	network.layers[1].weights[2][1] = -1.0;

	policy_init(&policy, &network, 1, 0.1);
	result = game_play(&game, &policy, &policy);

	print_board(&game.board);

	printf("%d moves, and the game %s\n", game.moves,
	       result == GAME_X_WINS ? "went to X" :
	       result == GAME_O_WINS ? "went to O" : "was drawn");
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

	return EXIT_SUCCESS;
}

#endif
