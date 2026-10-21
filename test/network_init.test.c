#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/layer.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/rng.c"

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

static void build_the_network(network_t *network)
{
	network_init(network);
	network_add_layer(network, 4, 3, ACTIVATION_CLAMP);
	network_add_layer(network, 3, 2, ACTIVATION_SIGMOID);
}

static char *test_a_new_network_starts_at_zero(void)
{
	network_t network;

	build_the_network(&network);

	_it_should("have zero weights", network.layers[0].weights[1][2] == 0.0);
	_it_should("have zero biases", network.layers[1].biases[0] == 0.0);

	return NULL;
}

static char *test_the_maximum_fill_gives_every_parameter_the_same_value(void)
{
	network_t network;

	build_the_network(&network);
	_it_should("fill the network", network_fill_maximum(&network, 0.75));

	_it_should("give the first weight the value", network.layers[0].weights[0][0] == 0.75);
	_it_should("give a weight in the last layer the value",
		network.layers[1].weights[1][2] == 0.75);
	_it_should("give the bias the value", network.layers[1].biases[1] == 0.75);

	return NULL;
}

static char *test_the_random_fill_stays_inside_the_scale(void)
{
	network_t network;
	rng_t rng;
	build_the_network(&network);
	rng_seed(&rng, 42);
	_it_should("fill the network", network_fill_random(&network, &rng, 0.1));

	for (int layer = 0; layer < network.n_layers; layer++) {
		for (int neuron = 0; neuron < network.layers[layer].n_neurons; neuron++) {
			if (network.layers[layer].biases[neuron] > 0.1 ||
			    network.layers[layer].biases[neuron] < -0.1)
				_it_should("keep the bias inside the scale", false);

			for (int input = 0; input < network.layers[layer].n_inputs; input++) {
				double weight = network.layers[layer].weights[neuron][input];

				if (weight > 0.1 || weight < -0.1)
					_it_should("keep the weight inside the scale", false);
			}
		}
	}

	_it_should("keep the bias inside the scale", true);
	_it_should("keep the weight inside the scale", true);
	_it_should("give the weights different values",
		network.layers[0].weights[0][0] != network.layers[0].weights[0][1]);

	return NULL;
}

static char *test_the_random_fill_is_repeatable(void)
{
	network_t first;
	network_t second;
	rng_t rng;

	build_the_network(&first);
	build_the_network(&second);

	rng_seed(&rng, 7);
	network_fill_random(&first, &rng, 0.5);
	rng_seed(&rng, 7);
	network_fill_random(&second, &rng, 0.5);

	_it_should("give the same weights", nearly_equal(
		first.layers[0].weights[1][1], second.layers[0].weights[1][1]));
	_it_should("give the same bias", nearly_equal(
		first.layers[1].biases[0], second.layers[1].biases[0]));

	return NULL;
}

static char *test_it_refuses_nothing(void)
{
	network_t network;
	rng_t rng;

	build_the_network(&network);
	rng_seed(&rng, 1);

	_it_should("refuse a null network", !network_fill_random(NULL, &rng, 0.1));
	_it_should("refuse a null generator", !network_fill_random(&network, NULL, 0.1));
	_it_should("refuse a null network", !network_fill_maximum(NULL, 0.1));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_a_new_network_starts_at_zero);
	_run_test(test_the_maximum_fill_gives_every_parameter_the_same_value);
	_run_test(test_the_random_fill_stays_inside_the_scale);
	_run_test(test_the_random_fill_is_repeatable);
	_run_test(test_it_refuses_nothing);

	return NULL;
}
