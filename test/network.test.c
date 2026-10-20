/*
 * Two layers chained by hand: the output buffer of the first is the input
 * buffer of the second. The weights and the biases are chosen so the result
 * can be worked out on paper, which is the point of the file. There is no
 * network type yet, so the chain is assembled by the caller.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"
#include "../src/layer.c"
#include "../src/neuron.c"

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

/* hidden: 2 inputs, 2 neurons.  out: 2 inputs, 1 neuron. */
static void build_the_two_layers(layer_t *hidden, layer_t *out)
{
	layer_init(hidden, 2, 2);
	hidden->weights[0][0] = 1.0;
	hidden->weights[0][1] = -1.0;
	hidden->biases[0] = 0.0;
	hidden->weights[1][0] = 1.0;
	hidden->weights[1][1] = 1.0;
	hidden->biases[1] = 0.0;

	layer_init(out, 2, 1);
	out->weights[0][0] = 2.0;
	out->weights[0][1] = 3.0;
	out->biases[0] = -1.0;
}

static char *test_two_layers_compose(void)
{
	layer_t hidden;
	layer_t out;
	double inputs[2] = {3.0, 1.0};
	double hidden_out[2];
	double outputs[1];

	build_the_two_layers(&hidden, &out);

	layer_forward(&hidden, inputs, 2, hidden_out, 2);
	layer_forward(&out, hidden_out, 2, outputs, 1);

	/* hidden: 3 - 1 = 2 and 3 + 1 = 4 */
	_it_should("give the hidden layer the two sums",
		nearly_equal(hidden_out[0], 2.0) && nearly_equal(hidden_out[1], 4.0));
	/* out: 2 * 2 + 3 * 4 - 1 = 15 */
	_it_should("give the output layer the result", nearly_equal(outputs[0], 15.0));

	return NULL;
}

static char *test_two_layers_compose_when_the_hidden_layer_clamps(void)
{
	layer_t hidden;
	layer_t out;
	double inputs[2] = {1.0, 3.0};
	double hidden_out[2];
	double outputs[1];

	build_the_two_layers(&hidden, &out);

	layer_forward(&hidden, inputs, 2, hidden_out, 2);
	layer_forward(&out, hidden_out, 2, outputs, 1);

	/* hidden: 1 - 3 = -2 becomes 0, and 1 + 3 = 4 */
	_it_should("let the hidden layer clamp its own sum",
		nearly_equal(hidden_out[0], 0.0) && nearly_equal(hidden_out[1], 4.0));
	/* out: 2 * 0 + 3 * 4 - 1 = 11 */
	_it_should("carry the clamped value into the next layer",
		nearly_equal(outputs[0], 11.0));

	return NULL;
}

static char *test_two_layers_compose_when_the_last_sum_is_negative(void)
{
	layer_t hidden;
	layer_t out;
	double inputs[2] = {-1.0, -1.0};
	double hidden_out[2];
	double outputs[1];

	build_the_two_layers(&hidden, &out);

	layer_forward(&hidden, inputs, 2, hidden_out, 2);
	layer_forward(&out, hidden_out, 2, outputs, 1);

	/* hidden: -1 + 1 = 0 and -1 - 1 = -2, both become 0 */
	_it_should("clamp both hidden sums", nearly_equal(hidden_out[0], 0.0) && nearly_equal(hidden_out[1], 0.0));
	/* out: 2 * 0 + 3 * 0 - 1 = -1, which becomes 0 */
	_it_should("clamp the last sum", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_the_board_feeds_a_two_layer_network(void)
{
	layer_t hidden;
	layer_t out;
	board_t board;
	double inputs[CELL_COUNT];
	double hidden_out[2];
	double outputs[CELL_COUNT];

	/* hidden: picks the first two cells out of the board */
	layer_init(&hidden, CELL_COUNT, 2);
	hidden.weights[0][0] = 1.0;
	hidden.weights[1][1] = 1.0;

	/* out: one neuron per cell, each reading the first hidden value */
	layer_init(&out, 2, CELL_COUNT);
	out.weights[0][0] = 2.0;
	out.weights[1][0] = -1.0;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	for (int cell = 0; cell < CELL_COUNT; cell++)
		inputs[cell] = (double)board.cells[cell];

	layer_forward(&hidden, inputs, CELL_COUNT, hidden_out, 2);
	layer_forward(&out, hidden_out, 2, outputs, CELL_COUNT);

	/* the board holds one X and no O, so the hidden layer sees 1 and 0 */
	_it_should("read the board into the hidden layer",
		nearly_equal(hidden_out[0], 1.0) && nearly_equal(hidden_out[1], 0.0));
	/* the first output is 2 * 1 = 2, the second is -1 * 1 = 0, the rest 0 */
	_it_should("give one output per cell", nearly_equal(outputs[0], 2.0));
	_it_should("leave the other outputs at zero",
		nearly_equal(outputs[1], 0.0) && nearly_equal(outputs[8], 0.0));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_two_layers_compose);
	_run_test(test_two_layers_compose_when_the_hidden_layer_clamps);
	_run_test(test_two_layers_compose_when_the_last_sum_is_negative);
	_run_test(test_the_board_feeds_a_two_layer_network);

	return NULL;
}
