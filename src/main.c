#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include "board.h"
#include "network.h"

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

int main(int argc, char *argv[], char *env[]) {
	UNUSED(argc);
	UNUSED(argv);
	UNUSED(env);

	greet("Hello, World!");
	score_a_game();

	return EXIT_SUCCESS;
}

#endif
