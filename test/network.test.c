/*
 * Two layers chained by hand, and the same two layers chained by a network.
 * The weights and the biases are chosen so the result can be worked out on
 * paper, which is the point of the file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/board.c"
#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

/* hidden: 2 inputs, 2 neurons.  out: 2 inputs, 1 neuron. */
static void build_hidden_weights(layer_t *hidden)
{
	hidden->weights[0][0] = 1.0;
	hidden->weights[0][1] = -1.0;
	hidden->biases[0] = 0.0;
	hidden->weights[1][0] = 1.0;
	hidden->weights[1][1] = 1.0;
	hidden->biases[1] = 0.0;
}

static void build_out_weights(layer_t *out)
{
	out->weights[0][0] = 2.0;
	out->weights[0][1] = 3.0;
	out->biases[0] = -1.0;
}

static void build_the_two_layers(layer_t *hidden, layer_t *out)
{
	layer_init(hidden, 2, 2);
	build_hidden_weights(hidden);
	layer_init(out, 2, 1);
	build_out_weights(out);
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

/* --- the network --- */

static char *test_network_init(void)
{
	network_t network;

	_it_should("set up an empty network", network_init(&network));
	_it_should("have no layers", network.n_layers == 0);
	_it_should("take no inputs", network_input_size(&network) == 0);
	_it_should("give no outputs", network_output_size(&network) == 0);

	return NULL;
}

static char *test_network_add_layer(void)
{
	network_t network;

	network_init(&network);

	_it_should("take the first layer", network_add_layer(&network, 9, 16));
	_it_should("count the layer", network.n_layers == 1);
	_it_should("take the width of the first layer", network_input_size(&network) == 9);
	_it_should("give the width of the last layer", network_output_size(&network) == 16);

	_it_should("take the second layer", network_add_layer(&network, 16, 9));
	_it_should("count both layers", network.n_layers == 2);
	_it_should("still take the width of the first layer", network_input_size(&network) == 9);
	_it_should("give the width of the last layer", network_output_size(&network) == 9);

	return NULL;
}

static char *test_network_add_layer_refuses_a_mismatched_join(void)
{
	network_t network;

	network_init(&network);
	network_add_layer(&network, 9, 16);

	/* the second layer takes 3 inputs, but the first gives 16 */
	_it_should("refuse a layer that does not take what the last one gave",
		!network_add_layer(&network, 3, 9));
	_it_should("leave the network as it was", network.n_layers == 1);
	_it_should("still give the width of the last layer", network_output_size(&network) == 16);

	_it_should("take a layer that does", network_add_layer(&network, 16, 9));

	return NULL;
}

static char *test_network_add_layer_refuses_when_full(void)
{
	network_t network;

	network_init(&network);

	for (int i = 0; i < NETWORK_MAX_LAYERS; i++)
		_it_should("take a layer", network_add_layer(&network, 4, 4));

	_it_should("refuse one layer too many", !network_add_layer(&network, 4, 4));
	_it_should("leave the count alone", network.n_layers == NETWORK_MAX_LAYERS);

	return NULL;
}

static char *test_network_forward_matches_the_manual_chain(void)
{
	network_t network;
	layer_t hidden;
	layer_t out;
	double inputs[2] = {3.0, 1.0};
	double manual_hidden[2];
	double manual_out[1];
	double network_out[1];

	/* the same two layers, put together by hand */
	build_the_two_layers(&hidden, &out);
	layer_forward(&hidden, inputs, 2, manual_hidden, 2);
	layer_forward(&out, manual_hidden, 2, manual_out, 1);

	/* and the same two layers, put together by the network */
	network_init(&network);
	network_add_layer(&network, 2, 2);
	build_hidden_weights(&network.layers[0]);
	network_add_layer(&network, 2, 1);
	build_out_weights(&network.layers[1]);
	network_forward(&network, inputs, 2, network_out, 1);

	_it_should("give the same answer as the manual chain",
		nearly_equal(network_out[0], manual_out[0]));
	_it_should("give the answer worked out by hand", nearly_equal(network_out[0], 15.0));

	return NULL;
}

static char *test_network_forward_clamps_at_each_layer(void)
{
	network_t network;
	double inputs[2] = {1.0, 3.0};
	double outputs[1];

	network_init(&network);
	network_add_layer(&network, 2, 2);
	build_hidden_weights(&network.layers[0]);
	network_add_layer(&network, 2, 1);
	build_out_weights(&network.layers[1]);

	network_forward(&network, inputs, 2, outputs, 1);

	/* hidden: 1 - 3 = -2 becomes 0, out: 2 * 0 + 3 * 4 - 1 = 11 */
	_it_should("clamp inside the network", nearly_equal(outputs[0], 11.0));

	return NULL;
}

static char *test_network_forward_with_one_layer(void)
{
	network_t network;
	double inputs[2] = {3.0, 1.0};
	double outputs[1];

	network_init(&network);
	network_add_layer(&network, 2, 1);
	build_out_weights(&network.layers[0]);

	network_forward(&network, inputs, 2, outputs, 1);

	/* 2 * 3 + 3 * 1 - 1 = 8 */
	_it_should("run a single layer", nearly_equal(outputs[0], 8.0));

	return NULL;
}

static char *test_the_board_feeds_a_network(void)
{
	network_t network;
	board_t board;
	double inputs[CELL_COUNT];
	double outputs[CELL_COUNT];

	network_init(&network);
	network_add_layer(&network, CELL_COUNT, 2);
	network.layers[0].weights[0][0] = 1.0;
	network.layers[0].weights[1][1] = 1.0;
	network_add_layer(&network, 2, CELL_COUNT);
	network.layers[1].weights[0][0] = 2.0;
	network.layers[1].weights[1][0] = -1.0;

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');

	for (int cell = 0; cell < CELL_COUNT; cell++)
		inputs[cell] = (double)board.cells[cell];

	network_forward(&network, inputs, CELL_COUNT, outputs, CELL_COUNT);

	_it_should("give one output per cell", nearly_equal(outputs[0], 2.0));
	_it_should("leave the other outputs at zero",
		nearly_equal(outputs[1], 0.0) && nearly_equal(outputs[8], 0.0));

	return NULL;
}

static char *test_network_forward_null_arguments(void)
{
	network_t network;
	double inputs[2] = {1.0, 1.0};
	double outputs[1] = {0.0};

	network_init(&network);
	network_add_layer(&network, 2, 1);
	build_out_weights(&network.layers[0]);

	_it_should("refuse a null network", !network_forward(NULL, inputs, 2, outputs, 1));
	_it_should("refuse null inputs", !network_forward(&network, NULL, 2, outputs, 1));
	_it_should("refuse null outputs", !network_forward(&network, inputs, 2, NULL, 1));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_network_forward_uninitialised_network(void)
{
	network_t network;
	double inputs[2] = {1.0, 1.0};
	double outputs[1] = {0.0};

	memset(&network, 0xFF, sizeof(network));

	_it_should("refuse a network that was never initialised",
		!network_forward(&network, inputs, 2, outputs, 1));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_network_forward_with_a_broken_join(void)
{
	network_t network;
	double inputs[2] = {1.0, 1.0};
	double outputs[1] = {0.0};

	network_init(&network);
	network_add_layer(&network, 2, 2);
	network_add_layer(&network, 2, 1);
	build_out_weights(&network.layers[1]);

	/* the first layer now gives 3, but the second one takes 2 */
	network.layers[0].n_neurons = 3;

	_it_should("refuse a network whose join does not hold",
		!network_forward(&network, inputs, 2, outputs, 1));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_network_forward_buffers_too_small(void)
{
	network_t network;
	double inputs[2] = {1.0, 1.0};
	double outputs[2] = {0.0, 0.0};

	network_init(&network);
	network_add_layer(&network, 2, 1);
	build_out_weights(&network.layers[0]);

	_it_should("refuse too few inputs", !network_forward(&network, inputs, 1, outputs, 1));
	_it_should("refuse too few outputs", !network_forward(&network, inputs, 2, outputs, 0));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_network_forward_with_swapped_layers(void)
{
	network_t network;
	double inputs[3] = {1.0, 1.0, 1.0};
	double outputs[3] = {0.0, 0.0, 0.0};

	network_init(&network);
	network_add_layer(&network, 3, 3);
	network_add_layer(&network, 3, 1);

	/* swap the layers round: the one that gives 1 now feeds the one that
	 * takes 3, and every layer is still perfectly valid on its own */
	layer_t temp = network.layers[0];
	network.layers[0] = network.layers[1];
	network.layers[1] = temp;

	_it_should("refuse a network whose layers do not fit together",
		!network_forward(&network, inputs, 3, outputs, 3));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_network_forward_with_a_changed_layer_count(void)
{
	network_t network;
	double inputs[3] = {1.0, 1.0, 1.0};
	double outputs[3] = {0.0, 0.0, 0.0};

	network_init(&network);
	network_add_layer(&network, 3, 3);
	network_add_layer(&network, 3, 1);

	/* the count now says there is one layer, but there are two */
	network.n_layers = 1;

	_it_should("refuse a network whose layer count was changed",
		!network_forward(&network, inputs, 3, outputs, 3));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_two_layers_compose);
	_run_test(test_two_layers_compose_when_the_hidden_layer_clamps);
	_run_test(test_two_layers_compose_when_the_last_sum_is_negative);
	_run_test(test_the_board_feeds_a_two_layer_network);
	_run_test(test_network_init);
	_run_test(test_network_add_layer);
	_run_test(test_network_add_layer_refuses_a_mismatched_join);
	_run_test(test_network_add_layer_refuses_when_full);
	_run_test(test_network_forward_matches_the_manual_chain);
	_run_test(test_network_forward_clamps_at_each_layer);
	_run_test(test_network_forward_with_one_layer);
	_run_test(test_the_board_feeds_a_network);
	_run_test(test_network_forward_null_arguments);
	_run_test(test_network_forward_uninitialised_network);
	_run_test(test_network_forward_with_a_broken_join);
	_run_test(test_network_forward_with_swapped_layers);
	_run_test(test_network_forward_with_a_changed_layer_count);
	_run_test(test_network_forward_buffers_too_small);

	return NULL;
}
