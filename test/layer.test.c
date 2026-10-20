#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>

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

	_it_should("accept a layer that fits", layer_init(&layer, 9, 9));
	_it_should("remember the number of inputs", layer.n_inputs == 9);
	_it_should("remember the number of neurons", layer.n_neurons == 9);
	_it_should("start every weight at zero", layer.weights[3][4] == 0.0);
	_it_should("start every bias at zero", layer.biases[3] == 0.0);

	_it_should("reject more inputs than fit", !layer_init(&layer, LAYER_MAX_INPUTS + 1, 1));
	_it_should("reject more neurons than fit", !layer_init(&layer, 1, LAYER_MAX_NEURONS + 1));
	_it_should("reject a negative size", !layer_init(&layer, -1, 1));

	return NULL;
}

static char *test_layer_forward_one_neuron(void)
{
	layer_t layer;
	double inputs[3] = {1.0, 2.0, 3.0};
	double outputs[1];

	layer_init(&layer, 3, 1);
	layer.weights[0][0] = 1.0;
	layer.weights[0][1] = 2.0;
	layer.weights[0][2] = 3.0;
	layer.biases[0] = 0.5;

	layer_forward(&layer, inputs, outputs);

	/* 1 * 1 + 2 * 2 + 3 * 3 + 0.5 = 1 + 4 + 9 + 0.5 = 14.5 */
	_it_should("add up the weighted inputs and the bias",
		nearly_equal(outputs[0], 14.5));

	return NULL;
}

static char *test_layer_forward_applies_the_activation(void)
{
	layer_t layer;
	double inputs[2] = {-1.0, 1.0};
	double outputs[2];

	layer_init(&layer, 2, 2);

	/* the first neuron sees a negative sum, the second a positive one */
	layer.weights[0][0] = 1.0;
	layer.weights[0][1] = 0.0;
	layer.weights[1][0] = 0.0;
	layer.weights[1][1] = 1.0;

	layer_forward(&layer, inputs, outputs);

	_it_should("turn a negative sum into zero", nearly_equal(outputs[0], 0.0));
	_it_should("leave a positive sum unchanged", nearly_equal(outputs[1], 1.0));

	return NULL;
}

static char *test_layer_forward_identity(void)
{
	layer_t layer;
	double inputs[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
	double outputs[9];

	layer_init(&layer, 9, 9);

	/* one neuron per input, each copying its own input */
	for (int i = 0; i < 9; i++)
		layer.weights[i][i] = 1.0;

	layer_forward(&layer, inputs, outputs);

	_it_should("pass the input through", nearly_equal(outputs[0], 1.0));
	_it_should("leave the other outputs at zero", nearly_equal(outputs[5], 0.0));

	return NULL;
}

static char *test_layer_forward_each_neuron_has_its_own_bias(void)
{
	layer_t layer;
	double inputs[1] = {0.0};
	double outputs[2];

	layer_init(&layer, 1, 2);

	/* no weights, so each neuron can only answer with its own bias */
	layer.biases[0] = 1.0;
	layer.biases[1] = 2.0;

	layer_forward(&layer, inputs, outputs);

	_it_should("give the first neuron its own bias", nearly_equal(outputs[0], 1.0));
	_it_should("give the second neuron its own bias", nearly_equal(outputs[1], 2.0));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_layer_init);
	_run_test(test_layer_forward_one_neuron);
	_run_test(test_layer_forward_applies_the_activation);
	_run_test(test_layer_forward_identity);
	_run_test(test_layer_forward_each_neuron_has_its_own_bias);

	return NULL;
}
