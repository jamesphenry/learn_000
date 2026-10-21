#include "trainer.h"

/*
 * The error signal of one neuron: how much the loss changes if its sum
 * moves. For the output layer with a sigmoid and a cross entropy the
 * derivative of the sigmoid is already folded in, so the error of the
 * neuron that was played is just its output less the target, and the
 * others are left alone.
 */
static void train_step(trainer_t *trainer, const board_t *board, board_cell_t action,
                       double target)
{
	network_t *network = trainer->network;
	int last = network->n_layers - 1;
	double inputs[CELL_COUNT];
	double sums[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];
	double outputs[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];
	double error[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];

	board_to_inputs(board, inputs);

	/* the forward pass, keeping the sums so the backward pass knows which
	 * of them were switched off */
	for (int layer = 0; layer < network->n_layers; layer++) {
		const layer_t *l = &network->layers[layer];
		const double *in = (layer == 0) ? inputs : outputs[layer - 1];

		for (int neuron = 0; neuron < l->n_neurons; neuron++) {
			double sum = l->biases[neuron];

			for (int i = 0; i < l->n_inputs; i++)
				sum += in[i] * l->weights[neuron][i];

			sums[layer][neuron] = sum;
			outputs[layer][neuron] = neuron_activate(sum, l->activation);
		}
	}

	for (int neuron = 0; neuron < network->layers[last].n_neurons; neuron++)
		error[last][neuron] = 0.0;

	error[last][action] = outputs[last][action] - target;

	/* and back through the layers */
	for (int layer = last; layer >= 0; layer--) {
		layer_t *l = &network->layers[layer];
		const double *in = (layer == 0) ? inputs : outputs[layer - 1];

		/* the error for the layer before, through the weights as they
		 * are now, before they are changed */
		if (layer > 0) {
			const layer_t *previous = &network->layers[layer - 1];

			for (int i = 0; i < l->n_inputs; i++) {
				double sum = 0.0;

				for (int neuron = 0; neuron < l->n_neurons; neuron++)
					sum += error[layer][neuron] * l->weights[neuron][i];

				error[layer - 1][i] = sum *
					neuron_derivative(sums[layer - 1][i], previous->activation);
			}
		}

		/* and then the weights move against the error */
		for (int neuron = 0; neuron < l->n_neurons; neuron++) {
			double e = error[layer][neuron];

			l->biases[neuron] -= trainer->learning_rate * e;

			for (int i = 0; i < l->n_inputs; i++)
				l->weights[neuron][i] -= trainer->learning_rate * e * in[i];
		}
	}
}

bool trainer_init(trainer_t *trainer, network_t *network, double learning_rate)
{
	if (trainer == NULL || network == NULL) return false;

	trainer->network = network;
	trainer->learning_rate = learning_rate;

	return true;
}

bool trainer_train_game(trainer_t *trainer, const game_t *game)
{
	network_t *network;
	game_t replay;
	double target;
	int last;

	if (trainer == NULL || game == NULL) return false;

	network = trainer->network;
	if (network == NULL) return false;
	if (network->n_layers < 1) return false;

	/* the loss only makes sense for a network that answers with a
	 * probability */
	last = network->n_layers - 1;
	if (network->layers[last].activation != ACTIVATION_SIGMOID) return false;

	target = (game_result(game) == GAME_X_WINS) ? 1.0 : 0.0;

	/* the game is played again from the start, so that every position is
	 * seen as it was when the move was made */
	game_init(&replay);

	for (int step = 0; step < game->moves; step++) {
		board_cell_t action = game->history[step];

		train_step(trainer, &replay.board, action, target);

		if (!game_play_move(&replay, action)) break;
	}

	return true;
}

int trainer_run(trainer_t *trainer, policy_t *policy, int games)
{
	game_t game;
	int played = 0;

	if (trainer == NULL || policy == NULL) return 0;

	for (int i = 0; i < games; i++) {
		game_play(&game, policy, policy);

		if (!trainer_train_game(trainer, &game)) break;

		played++;
	}

	return played;
}
