#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <math.h>

#include "../src/board.c"
#include "../src/neuron.c"

/* Floats are compared with a tolerance, so hand written expectations are not
 * thrown off by the last bit of a rounding error. */
static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

static char *test_all_zero(void)
{
	double inputs[CELL_COUNT] = {0};
	double weights[CELL_COUNT] = {0};

	_it_should("return zero for zero inputs and zero weights",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 0.0));

	return NULL;
}

static char *test_zero_inputs_with_bias(void)
{
	double inputs[CELL_COUNT] = {0};
	double weights[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	_it_should("return the bias when every input is zero",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 3.5), 3.5));
	_it_should("return a negative bias when every input is zero",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, -2.5), -2.5));

	return NULL;
}

static char *test_all_weights_one(void)
{
	double inputs[CELL_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	double weights[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	/* 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 45 */
	_it_should("add up the inputs when every weight is one",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 45.0));
	_it_should("add the bias to the sum of the inputs",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 10.0), 55.0));

	return NULL;
}

static char *test_weights_are_applied(void)
{
	double inputs[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	double weights[CELL_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	/* 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 45 */
	_it_should("weight each input by its own weight",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 45.0));

	return NULL;
}

static char *test_single_weight(void)
{
	double inputs[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	double weights[CELL_COUNT] = {2, 0, 0, 0, 0, 0, 0, 0, 0};

	_it_should("ignore the inputs whose weight is zero",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 2.0));

	return NULL;
}

static char *test_negative_inputs(void)
{
	double inputs[CELL_COUNT] = {-1, 1, -1, 1, -1, 1, -1, 1, -1};
	double weights[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	/* -1 + 1 - 1 + 1 - 1 + 1 - 1 + 1 - 1 = -1 */
	_it_should("add up inputs that are negative",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), -1.0));

	return NULL;
}

static char *test_negative_inputs_and_weights(void)
{
	double inputs[CELL_COUNT] = {-3, -2, -1, 0, 1, 2, 3, 4, 5};
	double weights[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	/* -3 - 2 - 1 + 0 + 1 + 2 + 3 + 4 + 5 = 9 */
	_it_should("add up inputs that are negative",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 9.0));

	return NULL;
}

static char *test_negative_weights(void)
{
	double inputs[CELL_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	double weights[CELL_COUNT] = {1, -1, 1, -1, 1, -1, 1, -1, 1};

	/* 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 = 5 */
	_it_should("apply weights that are negative",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 5.0));

	return NULL;
}

static char *test_fractional_values(void)
{
	double inputs[CELL_COUNT] = {0.5, 0.25, 0, 0, 0, 0, 0, 0, 0};
	double weights[CELL_COUNT] = {2, 4, 0, 0, 0, 0, 0, 0, 0};

	/* 0.5 * 2 + 0.25 * 4 = 1 + 1 = 2 */
	_it_should("multiply fractional inputs by their weights",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 2.0));

	return NULL;
}

static char *test_rounding(void)
{
	double inputs[CELL_COUNT] = {0.1, 0.2, 0, 0, 0, 0, 0, 0, 0};
	double weights[CELL_COUNT] = {1, 1, 0, 0, 0, 0, 0, 0, 0};

	_it_should("stay close to the expected sum when the inputs do not divide evenly",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 0.3));

	return NULL;
}

/* The board feeds a neuron directly: the cells are already numbers. */
static char *test_board_feeds_a_neuron(void)
{
	board_t board;
	double inputs[CELL_COUNT];
	double weights[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'O');

	for (int cell = 0; cell < CELL_COUNT; cell++)
		inputs[cell] = (double)board.cells[cell];

	/* +1 from the X and -1 from the O cancel out */
	_it_should("take the cells of a board as inputs",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 0.0));

	board_init(&board);
	place_mark(&board, CELL_TOP_LEFT, 'X');
	place_mark(&board, CELL_TOP, 'X');
	place_mark(&board, CELL_TOP_RIGHT, 'X');

	for (int cell = 0; cell < CELL_COUNT; cell++)
		inputs[cell] = (double)board.cells[cell];

	/* three X on the top row, so the sum is 3 */
	_it_should("sum the marks of a board",
		nearly_equal(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0), 3.0));

	return NULL;
}

static char *test_input_count_matches_the_board(void)
{
	_it_should("take one input per cell on the board", CELL_COUNT == CELL_COUNT);

	return NULL;
}

static char *test_clamp(void)
{
	_it_should("leave a positive value unchanged",
		nearly_equal(neuron_clamp(3.5), 3.5));
	_it_should("leave zero at zero",
		nearly_equal(neuron_clamp(0.0), 0.0));
	_it_should("turn a value below the floor into the floor",
		nearly_equal(neuron_clamp(-2.5), NEURON_FLOOR));
	_it_should("leave a value above the floor alone",
		nearly_equal(neuron_clamp(-0.5), -0.5));

	return NULL;
}

static char *test_clamp_edges(void)
{
	_it_should("leave a large positive value unchanged",
		nearly_equal(neuron_clamp(1000.0), 1000.0));
	_it_should("leave a small negative value alone",
		nearly_equal(neuron_clamp(-0.0001), -0.0001));
	_it_should("turn negative zero into zero",
		nearly_equal(neuron_clamp(-0.0), 0.0));

	/* A NaN is compared directly, a tolerance would never match it. */
	_it_should("turn a NaN value into the floor", neuron_clamp(NAN) == NEURON_FLOOR);

	return NULL;
}

/* The weighted sum feeds the activation, which is the whole neuron. */
static char *test_clamp_after_the_weighted_sum(void)
{
	double inputs[CELL_COUNT] = {0};
	double weights[CELL_COUNT] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	/* -1 * 1 = -1, and the activation puts it on the floor */
	inputs[0] = -1.0;
	_it_should("put a negative sum on the floor",
		nearly_equal(neuron_clamp(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0)), NEURON_FLOOR));

	/* 0 * 1 = 0, and the activation leaves a sum of zero alone */
	inputs[0] = 0.0;
	_it_should("leave a sum of zero at zero",
		nearly_equal(neuron_clamp(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0)), 0.0));

	/* 1 * 1 = 1, and the activation leaves a positive sum alone */
	inputs[0] = 1.0;
	_it_should("leave a positive sum unchanged",
		nearly_equal(neuron_clamp(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0)), 1.0));

	/* one X and eight O: 1 - 8 = -7, still a negative sum */
	inputs[0] = 1.0;
	for (int i = 1; i < CELL_COUNT; i++)
		inputs[i] = -1.0;
	_it_should("put a negative sum of mixed inputs on the floor",
		nearly_equal(neuron_clamp(neuron_weighted_sum(inputs, weights, CELL_COUNT, 0.0)), NEURON_FLOOR));

	/* nine X and a bias of -9: 9 - 9 = 0 */
	for (int i = 0; i < CELL_COUNT; i++)
		inputs[i] = 1.0;
	_it_should("leave a sum of zero at zero when the bias cancels the inputs",
		nearly_equal(neuron_clamp(neuron_weighted_sum(inputs, weights, CELL_COUNT, -9.0)), 0.0));

	return NULL;
}

static char *test_sigmoid(void)
{
	_it_should("give a half at zero", nearly_equal(neuron_sigmoid(0.0), 0.5));
	_it_should("give nearly one for a large positive value", neuron_sigmoid(10.0) > 0.99);
	_it_should("give nearly zero for a large negative value", neuron_sigmoid(-10.0) < 0.01);
	_it_should("stay a probability even for a huge value",
		neuron_sigmoid(-1000.0) >= 0.0 && neuron_sigmoid(1000.0) <= 1.0);
	_it_should("not be the same as a clamp", neuron_sigmoid(-2.5) != neuron_clamp(-2.5));

	return NULL;
}

static char *test_activate(void)
{
	_it_should("use the clamp when asked for it",
		nearly_equal(neuron_activate(-2.5, ACTIVATION_CLAMP), NEURON_FLOOR));
	_it_should("use the sigmoid when asked for it",
		nearly_equal(neuron_activate(0.0, ACTIVATION_SIGMOID), 0.5));

	return NULL;
}

static char *test_derivative(void)
{
	_it_should("be one above the floor", nearly_equal(neuron_derivative(1.0, ACTIVATION_CLAMP), 1.0));
	_it_should("be zero below the floor", nearly_equal(neuron_derivative(-2.0, ACTIVATION_CLAMP), 0.0));
	_it_should("be zero at the floor", nearly_equal(neuron_derivative(NEURON_FLOOR, ACTIVATION_CLAMP), 0.0));
	_it_should("be one just above the floor", nearly_equal(neuron_derivative(-0.5, ACTIVATION_CLAMP), 1.0));
	_it_should("be a quarter in the middle of a sigmoid",
		nearly_equal(neuron_derivative(0.0, ACTIVATION_SIGMOID), 0.25));

	return NULL;
}

static char *run_tests()
{
	_run_test(test_all_zero);
	_run_test(test_zero_inputs_with_bias);
	_run_test(test_all_weights_one);
	_run_test(test_weights_are_applied);
	_run_test(test_single_weight);
	_run_test(test_negative_inputs);
	_run_test(test_negative_inputs_and_weights);
	_run_test(test_negative_weights);
	_run_test(test_fractional_values);
	_run_test(test_rounding);
	_run_test(test_clamp);
	_run_test(test_sigmoid);
	_run_test(test_activate);
	_run_test(test_derivative);
	_run_test(test_clamp_edges);
	_run_test(test_clamp_after_the_weighted_sum);
	_run_test(test_board_feeds_a_neuron);
	_run_test(test_input_count_matches_the_board);

	return NULL;
}
