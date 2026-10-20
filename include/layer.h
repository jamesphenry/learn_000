#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>

// A layer holds at most this many inputs and this many neurons. Keeping the
// weights inside the struct means a layer needs no allocation, can be copied
// with a single assignment, and can be shared between games.
#define LAYER_MAX_INPUTS 64
#define LAYER_MAX_NEURONS 64

typedef struct {
	int n_inputs;
	int n_neurons;
	double weights[LAYER_MAX_NEURONS][LAYER_MAX_INPUTS];
	double biases[LAYER_MAX_NEURONS];
} layer_t;

bool layer_init(layer_t *layer, int n_inputs, int n_neurons);
void layer_forward(const layer_t *layer, const double *inputs, double *outputs);

#endif // LAYER_H
