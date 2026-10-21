#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <string.h>

#include "../src/arena.c"
#include "../src/board.c"
#include "../src/evaluate.c"
#include "../src/game.c"
#include "../src/judge.c"
#include "../src/layer.c"
#include "../src/minimax.c"
#include "../src/network.c"
#include "../src/neuron.c"
#include "../src/policy.c"
#include "../src/rng.c"
#include "../src/trainer.c"

static bool nearly_equal(double actual, double expected)
{
	double diff = actual > expected ? actual - expected : expected - actual;

	return diff < 1e-9;
}

static void build_small_settings(arena_settings_t *settings)
{
	arena_settings_init(settings);
	settings->hidden_neurons = 4;
	settings->training_games = 20;
	settings->evaluation_games = 20;
}

static char *test_the_settings_start_with_sensible_values(void)
{
	arena_settings_t settings;

	arena_settings_init(&settings);

	_it_should("have a hidden layer", settings.hidden_neurons > 0);
	_it_should("play some games", settings.training_games > 0 && settings.evaluation_games > 0);
	_it_should("keep the two seeds apart", settings.training_seed != settings.evaluation_seed);
	_it_should("have a learning rate", settings.learning_rate > 0.0);
	_it_should("have a temperature", settings.temperature > 0.0);

	return NULL;
}

static char *test_it_refuses_bad_arguments(void)
{
	arena_settings_t settings;
	arena_result_t result;

	build_small_settings(&settings);

	_it_should("refuse no settings", !arena_run(NULL, INIT_ZERO, &result));
	_it_should("refuse nowhere to put the result", !arena_run(&settings, INIT_ZERO, NULL));

	settings.evaluation_games = 0;
	_it_should("refuse to measure no games", !arena_run(&settings, INIT_ZERO, &result));

	settings.evaluation_games = 20;
	settings.hidden_neurons = 0;
	_it_should("refuse a hidden layer of nought", !arena_run(&settings, INIT_ZERO, &result));

	return NULL;
}

static char *test_a_model_that_is_never_trained_can_be_measured(void)
{
	arena_settings_t settings;
	arena_result_t result;

	build_small_settings(&settings);
	settings.training_games = 0;

	_it_should("run", arena_run(&settings, INIT_ZERO, &result));
	_it_should("train on no games", result.trained_games == 0);

	/* the model has not changed, so both measurements are of the same model
	 * on the same games */
	_it_should("measure it all the same",
		nearly_equal(result.win_rate_before, result.win_rate_after));
	_it_should("measure it the same to the last bit",
		nearly_equal(result.brier_before, result.brier_after) &&
		nearly_equal(result.truth_before, result.truth_after));

	return NULL;
}

static char *test_a_run_reports_a_result(void)
{
	arena_settings_t settings;
	arena_result_t result;

	build_small_settings(&settings);

	_it_should("run", arena_run(&settings, INIT_RANDOM, &result));
	_it_should("train on the games it was given", result.trained_games == 20);
	_it_should("give a win rate between nought and a hundred",
		result.win_rate_after >= 0.0 && result.win_rate_after <= 100.0);
	_it_should("give a brier score between nought and one",
		result.brier_after >= 0.0 && result.brier_after <= 1.0);
	_it_should("give a truth error between nought and one",
		result.truth_after >= 0.0 && result.truth_after <= 1.0);

	return NULL;
}

static char *test_the_same_run_gives_the_same_result(void)
{
	arena_settings_t settings;
	arena_result_t first;
	arena_result_t second;

	build_small_settings(&settings);

	arena_run(&settings, INIT_RANDOM, &first);
	arena_run(&settings, INIT_RANDOM, &second);

	_it_should("give the same win rate",
		nearly_equal(first.win_rate_after, second.win_rate_after));
	_it_should("give the same truth error",
		nearly_equal(first.truth_after, second.truth_after));

	return NULL;
}

static char *test_the_way_a_model_starts_off_matters(void)
{
	arena_settings_t settings;
	arena_result_t zero;
	arena_result_t random;
	arena_result_t maximum;

	build_small_settings(&settings);

	arena_run(&settings, INIT_ZERO, &zero);
	arena_run(&settings, INIT_RANDOM, &random);
	arena_run(&settings, INIT_MAXIMUM, &maximum);

	/* every pair has to differ, or one of the three is not being started
	 * the way it says it is */
	_it_should("give the three models different truths",
		!nearly_equal(zero.truth_before, random.truth_before) &&
		!nearly_equal(random.truth_before, maximum.truth_before) &&
		!nearly_equal(zero.truth_before, maximum.truth_before));
	_it_should("give the three models different scores",
		!nearly_equal(zero.brier_before, random.brier_before) &&
		!nearly_equal(random.brier_before, maximum.brier_before) &&
		!nearly_equal(zero.brier_before, maximum.brier_before));

	return NULL;
}

static char *test_a_model_that_learns_gets_better(void)
{
	arena_settings_t settings;
	arena_result_t result;

	build_small_settings(&settings);
	settings.training_games = 200;

	arena_run(&settings, INIT_RANDOM, &result);

	/* a model that has learned something should be closer to the truth */
	_it_should("get closer to the truth", result.truth_after < result.truth_before);

	return NULL;
}

static char *test_a_run_says_what_the_model_gave_away(void)
{
	arena_settings_t settings;
	arena_result_t result;

	build_small_settings(&settings);

	arena_run(&settings, INIT_RANDOM, &result);

	/* the worth a position can be given away is between nought and two:
	 * a win given away is one and a loss given away is one */
	_it_should("say how much worth the model gave away",
		result.value_lost_after >= 0.0 && result.value_lost_after <= 2.0);
	_it_should("count the games against the perfect player",
		result.perfect_after.games == settings.evaluation_games);
	_it_should("count every one of them once",
		result.perfect_after.wins + result.perfect_after.draws +
		result.perfect_after.losses == settings.evaluation_games);
	_it_should("take each colour the same number of times",
		result.perfect_after.as_x == result.perfect_after.as_o);

	return NULL;
}

static char *test_the_perfect_player_teaches_better_than_its_own_games(void)
{
	arena_settings_t settings;
	arena_result_t truth;
	arena_result_t self;

	build_small_settings(&settings);
	settings.training_games = 500;

	settings.learning = ARENA_LEARN_TRUTH;
	arena_run(&settings, INIT_RANDOM, &truth);

	settings.learning = ARENA_LEARN_SELF;
	arena_run(&settings, INIT_RANDOM, &self);

	/* the perfect player says what each position is worth, so a model
	 * taught by it should end up closer to the truth than one that has to
	 * make do with the result of its own games */
	_it_should("get closer to the truth", truth.truth_after < self.truth_after);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_the_settings_start_with_sensible_values);
	_run_test(test_it_refuses_bad_arguments);
	_run_test(test_a_model_that_is_never_trained_can_be_measured);
	_run_test(test_a_run_reports_a_result);
	_run_test(test_the_same_run_gives_the_same_result);
	_run_test(test_the_way_a_model_starts_off_matters);
	_run_test(test_a_model_that_learns_gets_better);
	_run_test(test_a_run_says_what_the_model_gave_away);
	_run_test(test_the_perfect_player_teaches_better_than_its_own_games);

	return NULL;
}
