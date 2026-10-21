#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>

#include "../src/rng.c"

static char *test_the_same_seed_gives_the_same_sequence(void)
{
	rng_t first;
	rng_t second;

	rng_seed(&first, 12345);
	rng_seed(&second, 12345);

	for (int i = 0; i < 20; i++) {
		if (rng_next_double(&first) != rng_next_double(&second))
			_it_should("give the same sequence", false);
	}

	_it_should("give the same sequence", true);

	return NULL;
}

static char *test_different_seeds_give_different_sequences(void)
{
	rng_t first;
	rng_t second;

	rng_seed(&first, 1);
	rng_seed(&second, 2);

	int differences = 0;

	for (int i = 0; i < 20; i++) {
		if (rng_next_double(&first) != rng_next_double(&second))
			differences++;
	}

	_it_should("give a different sequence", differences > 15);

	return NULL;
}

static char *test_the_values_are_between_zero_and_one(void)
{
	rng_t rng;

	rng_seed(&rng, 7);

	for (int i = 0; i < 1000; i++) {
		double value = rng_next_double(&rng);

		if (value < 0.0 || value >= 1.0)
			_it_should("stay inside the range", false);
	}

	_it_should("stay inside the range", true);

	return NULL;
}

static char *test_a_zero_seed_still_varies(void)
{
	rng_t rng;

	rng_seed(&rng, 0);

	double first = rng_next_double(&rng);
	double second = rng_next_double(&rng);

	_it_should("not stay at zero", first != 0.0);
	_it_should("give a different value each time", first != second);

	return NULL;
}

static char *test_the_values_are_spread_out(void)
{
	rng_t rng;

	rng_seed(&rng, 99);

	int low = 0;
	int high = 0;

	for (int i = 0; i < 1000; i++) {
		double value = rng_next_double(&rng);

		if (value < 0.25) low++;
		if (value > 0.75) high++;
	}

	_it_should("give values in the lower half", low > 100);
	_it_should("give values in the upper half", high > 100);

	return NULL;
}

static char *test_next_int_stays_inside_its_limit(void)
{
	rng_t rng;

	rng_seed(&rng, 5);

	for (int i = 0; i < 1000; i++) {
		int value = rng_next_int(&rng, 9);

		if (value < 0 || value >= 9)
			_it_should("stay inside the limit", false);
	}

	_it_should("stay inside the limit", true);
	_it_should("give zero for a limit of zero", rng_next_int(&rng, 0) == 0);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_the_same_seed_gives_the_same_sequence);
	_run_test(test_different_seeds_give_different_sequences);
	_run_test(test_the_values_are_between_zero_and_one);
	_run_test(test_a_zero_seed_still_varies);
	_run_test(test_the_values_are_spread_out);
	_run_test(test_next_int_stays_inside_its_limit);

	return NULL;
}
