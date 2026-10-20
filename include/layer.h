#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>
#include <stddef.h>

// A layer holds at most this many inputs and this many neurons.
#define LAYER_MAX_INPUTS 64
#define LAYER_MAX_NEURONS 64

// Marks a layer as set up by layer_init. The dimensions are folded into it,
// so a layer that was never initialised, or whose dimensions were changed
// afterwards, does not match.
#define LAYER_CHECK 0x4C415945u

typedef struct {
	int n_inputs;
	int n_neurons;
	unsigned int check;
	double weights[LAYER_MAX_NEURONS][LAYER_MAX_INPUTS];
	double biases[LAYER_MAX_NEURONS];
} layer_t;

bool layer_init(layer_t *layer, int n_inputs, int n_neurons);

bool layer_forward(const layer_t *layer, const double *inputs, size_t n_inputs,
                   double *outputs, size_t n_outputs);

#endif // LAYER_H
