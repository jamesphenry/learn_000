#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/layer.c"
#include "../src/neuron.c"

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

static char *test_layer_init(void)
{
	layer_t layer;

	_it_should("accept a layer that fits", layer_init(&layer, 9, 9, ACTIVATION_CLAMP));
	_it_should("remember the number of inputs", layer.n_inputs == 9);
	_it_should("remember the number of neurons", layer.n_neurons == 9);
	_it_should("start every weight at zero", layer.weights[3][4] == 0.0);
	_it_should("start every bias at zero", layer.biases[3] == 0.0);

	_it_should("reject more inputs than fit", !layer_init(&layer, LAYER_MAX_INPUTS + 1, 1, ACTIVATION_CLAMP));
	_it_should("reject more neurons than fit", !layer_init(&layer, 1, LAYER_MAX_NEURONS + 1, ACTIVATION_CLAMP));
	_it_should("reject a negative size", !layer_init(&layer, -1, 1, ACTIVATION_CLAMP));

	return NULL;
}

static char *test_layer_forward_one_neuron(void)
{
	layer_t layer;
	double inputs[3] = {1.0, 2.0, 3.0};
	double outputs[1] = {0.0};

	layer_init(&layer, 3, 1, ACTIVATION_CLAMP);
	layer.weights[0][0] = 1.0;
	layer.weights[0][1] = 2.0;
	layer.weights[0][2] = 3.0;
	layer.biases[0] = 0.5;

	_it_should("report that it ran", layer_forward(&layer, inputs, 3, outputs, 1));
	_it_should("add up the weighted inputs and the bias", nearly_equal(outputs[0], 14.5));

	return NULL;
}

static char *test_layer_forward_applies_the_activation(void)
{
	layer_t layer;
	double inputs[2] = {-1.0, 1.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 2, 2, ACTIVATION_CLAMP);

	layer.weights[0][0] = 1.0;
	layer.weights[0][1] = 0.0;
	layer.weights[1][0] = 0.0;
	layer.weights[1][1] = 1.0;

	layer_forward(&layer, inputs, 2, outputs, 2);

	_it_should("put a negative sum on the floor", nearly_equal(outputs[0], NEURON_FLOOR));
	_it_should("leave a positive sum unchanged", nearly_equal(outputs[1], 1.0));

	return NULL;
}

static char *test_layer_forward_identity(void)
{
	layer_t layer;
	double inputs[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
	double outputs[9] = {0.0};

	layer_init(&layer, 9, 9, ACTIVATION_CLAMP);

	for (int i = 0; i < 9; i++)
		layer.weights[i][i] = 1.0;

	layer_forward(&layer, inputs, 9, outputs, 9);

	_it_should("pass the input through", nearly_equal(outputs[0], 1.0));
	_it_should("leave the other outputs at zero", nearly_equal(outputs[5], 0.0));

	return NULL;
}

static char *test_layer_forward_each_neuron_has_its_own_bias(void)
{
	layer_t layer;
	double inputs[1] = {0.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 1, 2, ACTIVATION_CLAMP);

	layer.biases[0] = 1.0;
	layer.biases[1] = 2.0;

	layer_forward(&layer, inputs, 1, outputs, 2);

	_it_should("give the first neuron its own bias", nearly_equal(outputs[0], 1.0));
	_it_should("give the second neuron its own bias", nearly_equal(outputs[1], 2.0));

	return NULL;
}

static char *test_layer_forward_null_arguments(void)
{
	layer_t layer;
	double inputs[2] = {1.0, 1.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 2, 2, ACTIVATION_CLAMP);

	_it_should("refuse a null layer", !layer_forward(NULL, inputs, 2, outputs, 2));
	_it_should("refuse null inputs", !layer_forward(&layer, NULL, 2, outputs, 2));
	_it_should("refuse null outputs", !layer_forward(&layer, inputs, 2, NULL, 2));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_layer_forward_uninitialised_layer(void)
{
	layer_t layer;
	double inputs[2] = {1.0, 1.0};
	double outputs[2] = {0.0, 0.0};

	memset(&layer, 0xFF, sizeof(layer));

	_it_should("refuse a layer that was never initialised",
		!layer_forward(&layer, inputs, 2, outputs, 2));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_layer_forward_changed_dimensions(void)
{
	layer_t layer;
	double inputs[2] = {1.0, 1.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 2, 2, ACTIVATION_CLAMP);

	layer.n_inputs = 1;

	_it_should("refuse a layer whose dimensions were changed",
		!layer_forward(&layer, inputs, 2, outputs, 2));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_layer_forward_buffers_too_small(void)
{
	layer_t layer;
	double inputs[2] = {1.0, 1.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 2, 2, ACTIVATION_CLAMP);

	_it_should("refuse too few inputs", !layer_forward(&layer, inputs, 1, outputs, 2));
	_it_should("refuse too few outputs", !layer_forward(&layer, inputs, 2, outputs, 1));
	_it_should("leave the outputs alone", nearly_equal(outputs[0], 0.0));

	return NULL;
}

static char *test_layer_with_a_sigmoid(void)
{
	layer_t layer;
	double inputs[2] = {0.0, 0.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 2, 2, ACTIVATION_SIGMOID);
	layer.weights[0][0] = 1.0;

	layer_forward(&layer, inputs, 2, outputs, 2);

	_it_should("answer with a probability", nearly_equal(outputs[0], 0.5));
	_it_should("answer with the same probability everywhere", nearly_equal(outputs[1], 0.5));

	inputs[0] = 10.0;
	layer_forward(&layer, inputs, 2, outputs, 2);
	_it_should("answer near one for a large sum", outputs[0] > 0.99);

	inputs[0] = -10.0;
	layer_forward(&layer, inputs, 2, outputs, 2);
	_it_should("answer near zero for a small sum", outputs[0] < 0.01);

	return NULL;
}

static char *test_the_activation_is_part_of_the_layer(void)
{
	layer_t layer;
	double inputs[2] = {0.0, 0.0};
	double outputs[2] = {0.0, 0.0};

	layer_init(&layer, 2, 2, ACTIVATION_SIGMOID);

	/* the layer was set up to answer with a probability, and something
	 * changed its mind */
	layer.activation = ACTIVATION_CLAMP;

	_it_should("refuse a layer whose activation was changed",
		!layer_forward(&layer, inputs, 2, outputs, 2));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_layer_init);
	_run_test(test_layer_with_a_sigmoid);
	_run_test(test_the_activation_is_part_of_the_layer);
	_run_test(test_layer_forward_one_neuron);
	_run_test(test_layer_forward_applies_the_activation);
	_run_test(test_layer_forward_identity);
	_run_test(test_layer_forward_each_neuron_has_its_own_bias);
	_run_test(test_layer_forward_null_arguments);
	_run_test(test_layer_forward_uninitialised_layer);
	_run_test(test_layer_forward_changed_dimensions);
	_run_test(test_layer_forward_buffers_too_small);

	return NULL;
}
