#include "trainer.h"
#include "policy.h"

/*
 * The error signal of one neuron: how much the loss changes if its sum
 * moves. For the output layer with a sigmoid and a cross entropy the
 * derivative of the sigmoid is already folded in, so the error of the
 * neuron is just its output less the target.
 */
static void train_step(trainer_t *trainer, const game_t *game,
                       const game_position_t *position, double target)
{
	network_t *network = trainer->network;
	int last = network->n_layers - 1;
	double inputs[GAME_MAX_FEATURES];
	double sums[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];
	double outputs[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];
	double error[NETWORK_MAX_LAYERS][LAYER_MAX_NEURONS];

	game->features(position, inputs);

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

	error[last][0] = outputs[last][0] - target;

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

	/* the loss only makes sense for a network that answers with the worth
	 * of a position, which is one number and a probability */
	if (network->n_layers < 1) return false;
	if (network->layers[network->n_layers - 1].activation != ACTIVATION_SIGMOID) return false;
	if (network->layers[network->n_layers - 1].n_neurons != 1) return false;

	trainer->network = network;
	trainer->learning_rate = learning_rate;

	return true;
}

bool trainer_train_position(trainer_t *trainer, const game_t *game,
                            const game_position_t *position, double target)
{
	if (trainer == NULL || game == NULL || position == NULL) return false;
	if (trainer->network == NULL) return false;

	train_step(trainer, game, position, target);

	return true;
}

bool trainer_learn_from_oracle(trainer_t *trainer, const game_t *game,
                               const oracle_t *oracle, const game_trace_t *trace)
{
	game_position_t positions[GAME_MAX_HISTORY];
	game_position_t final;
	int played;

	if (trainer == NULL || game == NULL || oracle == NULL || trace == NULL) return false;
	if (trainer->network == NULL) return false;
	if (oracle->worth == NULL || !oracle_ready(oracle)) return false;

	/* the game is played again from the start, so that every position is
	 * seen as it was when the move was made */
	played = game_replay(game, trace, positions, &final);

	/* and each position is worth what the perfect player knows it to be
	 * worth */
	for (int step = 0; step < played; step++) {
		double target = oracle->worth(oracle->state, &positions[step]);

		train_step(trainer, game, &positions[step], target);
	}

	return true;
}

bool trainer_learn_from_outcome(trainer_t *trainer, const game_t *game,
                                const game_trace_t *trace)
{
	game_position_t positions[GAME_MAX_HISTORY];
	game_position_t final;
	game_outcome_t outcome;
	int played;

	if (trainer == NULL || game == NULL || trace == NULL) return false;
	if (trainer->network == NULL) return false;

	played = game_replay(game, trace, positions, &final);

	/* and the game is over, so what it came to is known */
	outcome = game->outcome(&final);

	/* and each position is worth what happened to the player who was to
	 * move in it */
	for (int step = 0; step < played; step++) {
		double target = game_worth_of_outcome(outcome, game->to_move(&positions[step]));

		train_step(trainer, game, &positions[step], target);
	}

	return true;
}

int trainer_run(trainer_t *trainer, const game_t *game, policy_t *policy, int games)
{
	game_trace_t trace;
	int played = 0;

	if (trainer == NULL || game == NULL || policy == NULL) return 0;

	for (int i = 0; i < games; i++) {
		if (!game_play(game, policy_choose, policy, policy_choose, policy, &trace)) break;

		if (!trainer_learn_from_outcome(trainer, game, &trace)) break;

		played++;
	}

	return played;
}
