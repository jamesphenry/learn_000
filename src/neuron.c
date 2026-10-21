#include <math.h>
#include "neuron.h"

double neuron_weighted_sum(const double *inputs, const double *weights,
			    int count, double bias)
{
	double sum = 0.0;

	for (int i = 0; i < count; i++)
		sum += inputs[i] * weights[i];

	return sum + bias;
}

double neuron_relu(double value)
{
	return fmax(0.0, value);
}

double neuron_sigmoid(double value)
{
	return 1.0 / (1.0 + exp(-value));
}

double neuron_activate(double value, activation_t activation)
{
	if (activation == ACTIVATION_SIGMOID) return neuron_sigmoid(value);

	return neuron_relu(value);
}

double neuron_derivative(double value, activation_t activation)
{
	if (activation == ACTIVATION_SIGMOID) {
		double probability = neuron_sigmoid(value);

		return probability * (1.0 - probability);
	}

	return value > 0.0 ? 1.0 : 0.0;
}
