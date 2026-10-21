#include "network.h"

static unsigned int network_check(int n_layers, int input_size, int output_size)
{
	return NETWORK_CHECK ^ (unsigned int)n_layers ^ ((unsigned int)input_size << 8) ^
	       ((unsigned int)output_size << 16);
}

bool network_init(network_t *network)
{
	if (network == NULL) return false;

	network->n_layers = 0;
	network->input_size = 0;
	network->output_size = 0;
	network->check = network_check(0, 0, 0);

	return true;
}

bool network_add_layer(network_t *network, int n_inputs, int n_neurons)
{
	if (network == NULL) return false;
	if (network->n_layers >= NETWORK_MAX_LAYERS) return false;

	if (network->n_layers == 0) {
		/* the first layer decides how wide the whole network is */
		network->input_size = n_inputs;
	} else {
		/* every layer must take exactly what the one before it produced */
		if (n_inputs != network->layers[network->n_layers - 1].n_neurons) return false;
	}

	if (!layer_init(&network->layers[network->n_layers], n_inputs, n_neurons)) return false;

	network->n_layers++;
	network->output_size = n_neurons;
	network->check = network_check(network->n_layers, network->input_size, network->output_size);

	return true;
}

bool network_forward(const network_t *network, const double *inputs, size_t n_inputs,
                     double *outputs, size_t n_outputs)
{
	double first[LAYER_MAX_NEURONS];
	double second[LAYER_MAX_NEURONS];
	const double *current = inputs;
	double *next = first;

	if (network == NULL || inputs == NULL || outputs == NULL) return false;
	if (network->check != network_check(network->n_layers, network->input_size, network->output_size))
		return false;
	if (network->n_layers < 1) return false;
	if (n_inputs < (size_t)network->input_size) return false;
	if (n_outputs < (size_t)network->output_size) return false;

	for (int i = 0; i < network->n_layers; i++) {
		/* the join has to hold, however the network was put together */
		if (i > 0 && network->layers[i - 1].n_neurons != network->layers[i].n_inputs)
			return false;

		if (i == network->n_layers - 1)
			return layer_forward(&network->layers[i], current, LAYER_MAX_NEURONS,
			                     outputs, n_outputs);

		if (!layer_forward(&network->layers[i], current, LAYER_MAX_NEURONS,
		                   next, LAYER_MAX_NEURONS))
			return false;

		current = next;
		next = (next == first) ? second : first;
	}

	return false;
}

int network_input_size(const network_t *network)
{
	if (network == NULL) return 0;

	return network->input_size;
}

int network_output_size(const network_t *network)
{
	if (network == NULL) return 0;

	return network->output_size;
}
