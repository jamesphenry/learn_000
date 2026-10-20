#ifndef NEURON_H
#define NEURON_H

/*
 * The weighted sum of the inputs plus the bias:
 *
 *     sum(inputs[i] * weights[i]) + bias
 *
 * No activation function is applied.
 */
double neuron_weighted_sum(const double *inputs, const double *weights,
			    int count, double bias);

/*
 * The activation of the neuron: a rectified linear unit.
 *
 *     value > 0 ? value : 0
 *
 * A positive value passes through unchanged, anything else becomes zero.
 */
double neuron_relu(double value);

#endif // NEURON_H
