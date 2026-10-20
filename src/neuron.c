#include <math.h>
#include "neuron.h"

double neuron_weighted_sum(const double inputs[NEURON_INPUTS],
			    const double weights[NEURON_INPUTS],
			    double bias)
{
	double sum = 0.0;

	for (int i = 0; i < NEURON_INPUTS; i++)
		sum += inputs[i] * weights[i];

	return sum + bias;
}

double neuron_relu(double value)
{
	return fmax(0.0, value);
}
