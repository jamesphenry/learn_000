#ifndef NEURON_H
#define NEURON_H

/* What a neuron does with its sum. */
typedef enum {
	ACTIVATION_RELU,
	ACTIVATION_SIGMOID
} activation_t;

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

/*
 * The activation of a neuron that answers with a probability:
 *
 *     1 / (1 + e^-value)
 *
 * A large positive value becomes one and a large negative value becomes
 * zero. It replaces the rectified linear unit rather than following it,
 * because a unit that has already thrown the negative half away can never
 * answer with less than a half.
 */
double neuron_sigmoid(double value);

/* The activation, chosen by name. */
double neuron_activate(double value, activation_t activation);

/*
 * How much the activation changes for a small change in the sum. A relu
 * passes a positive sum on unchanged and stops a negative one dead, and a
 * sigmoid is flattest at its extremes.
 */
double neuron_derivative(double value, activation_t activation);

#endif // NEURON_H
