#ifndef NETWORK_H
#define NETWORK_H

#include <stdbool.h>
#include <stddef.h>
#include "layer.h"

// A network holds at most this many layers. Two is enough for the shape
// tic-tac-toe needs, 9 -> hidden -> 9. The forward pass is a loop, so raising
// this needs no other change.
#define NETWORK_MAX_LAYERS 2

// Marks a network as set up by network_init. Its shape is folded in, so a
// network that was never set up, or whose shape was changed afterwards, does
// not match.
#define NETWORK_CHECK 0x4E455457u

typedef struct {
	int n_layers;
	int input_size;
	int output_size;
	unsigned int check;
	layer_t layers[NETWORK_MAX_LAYERS];
} network_t;

bool network_init(network_t *network);
bool network_add_layer(network_t *network, int n_inputs, int n_neurons);
bool network_forward(const network_t *network, const double *inputs, size_t n_inputs,
                     double *outputs, size_t n_outputs);
int network_input_size(const network_t *network);
int network_output_size(const network_t *network);

#endif // NETWORK_H
