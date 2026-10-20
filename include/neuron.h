#ifndef NEURON_H
#define NEURON_H

// The number of inputs a single neuron takes. It matches the number of cells
// on the board, so a board can be fed to a neuron without any conversion.
#define NEURON_INPUTS 9

/*
 * The weighted sum of the inputs plus the bias:
 *
 *     sum(inputs[i] * weights[i]) + bias
 *
 * No activation function is applied.
 */
double neuron_weighted_sum(const double inputs[NEURON_INPUTS],
			    const double weights[NEURON_INPUTS],
			    double bias);

#endif // NEURON_H
