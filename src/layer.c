#include "layer.h"
#include "neuron.h"

static unsigned int layer_check(int n_inputs, int n_neurons)
{
	return LAYER_CHECK ^ (unsigned int)n_inputs ^ ((unsigned int)n_neurons << 16);
}

bool layer_init(layer_t *layer, int n_inputs, int n_neurons)
{
	if (n_inputs < 0 || n_inputs > LAYER_MAX_INPUTS) return false;
	if (n_neurons < 0 || n_neurons > LAYER_MAX_NEURONS) return false;

	layer->n_inputs = n_inputs;
	layer->n_neurons = n_neurons;
	layer->check = layer_check(n_inputs, n_neurons);

	for (int neuron = 0; neuron < n_neurons; neuron++) {
		layer->biases[neuron] = 0.0;

		for (int input = 0; input < n_inputs; input++)
			layer->weights[neuron][input] = 0.0;
	}

	return true;
}

bool layer_forward(const layer_t *layer, const double *inputs, size_t n_inputs,
                   double *outputs, size_t n_outputs)
{
	if (layer == NULL || inputs == NULL || outputs == NULL) return false;
	if (layer->check != layer_check(layer->n_inputs, layer->n_neurons)) return false;
	if (n_inputs < (size_t)layer->n_inputs) return false;
	if (n_outputs < (size_t)layer->n_neurons) return false;

	for (int neuron = 0; neuron < layer->n_neurons; neuron++) {
		double sum = neuron_weighted_sum(inputs, layer->weights[neuron],
		                                 layer->n_inputs, layer->biases[neuron]);

		outputs[neuron] = neuron_relu(sum);
	}

	return true;
}
