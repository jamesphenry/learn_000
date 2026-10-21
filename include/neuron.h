#ifndef NEURON_H
#define NEURON_H

/* What a neuron does with its sum. */
typedef enum {
	ACTIVATION_CLAMP,
	ACTIVATION_SIGMOID,
	ACTIVATION_RELU
} activation_t;

/*
 * The floor of the clamped activation. A sum that is a little negative
 * keeps its value, so the opponent's marks stay visible to the layer above
 * instead of being thrown away.
 */
#define NEURON_FLOOR (-1.0)

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
 * The activation of the neuron: a rectified linear unit with a floor of -1
 * rather than 0.
 *
 *     value < -1 ? -1 : value
 *
 * A positive value passes through unchanged, and a negative one keeps its
 * value down to the floor.
 */
double neuron_clamp(double value);

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

/*
 * The activation of a neuron that throws the negative half away:
 *
 *     value < 0 ? 0 : value
 *
 * The opponent's marks are still seen by the weights, which can be
 * negative, so nothing is lost by throwing the negative half of the sum
 * away. What it buys is a sharper unit: the clamp passes everything
 * above its floor on unchanged, which leaves the layer above it with
 * very little to tell one position from another.
 */
double neuron_relu(double value);

/* The activation, chosen by name. */
double neuron_activate(double value, activation_t activation);

/*
 * How much the activation changes for a small change in the sum. The clamp
 * passes a sum above the floor on unchanged and stops one below it dead,
 * and a sigmoid is flattest at its extremes.
 */
double neuron_derivative(double value, activation_t activation);

#endif // NEURON_H
