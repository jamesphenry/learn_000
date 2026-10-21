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

static char *test_the_models_play_each_other(void)
{
	arena_settings_t settings;
	arena_report_t report;

	build_small_settings(&settings);
	settings.match_games = 20;

	_it_should("run", arena_run_all(&settings, &report));

	for (int first = 0; first < ARENA_SCHEMES; first++) {
		_it_should("train every model the same",
			report.results[first].trained_games == settings.training_games);

		for (int second = first + 1; second < ARENA_SCHEMES; second++) {
			_it_should("play every pair of models",
				report.matches[first][second].games == settings.match_games);
			_it_should("count every game once",
				report.matches[first][second].first_wins +
				report.matches[first][second].draws +
				report.matches[first][second].second_wins == settings.match_games);
			_it_should("take each colour the same number of times",
				report.matches[first][second].first_as_x ==
				settings.match_games / 2);
			_it_should("have no games flagged",
				report.matches[first][second].flagged == 0);
		}
	}

	return NULL;
}

static char *test_the_same_report_comes_again(void)
{
	arena_settings_t settings;
	arena_report_t first;
	arena_report_t second;

	build_small_settings(&settings);

	arena_run_all(&settings, &first);
	arena_run_all(&settings, &second);

	for (int i = 0; i < ARENA_SCHEMES; i++)
		for (int j = i + 1; j < ARENA_SCHEMES; j++) {
			_it_should("give the same wins",
				first.matches[i][j].first_wins == second.matches[i][j].first_wins);
			_it_should("give the same draws",
				first.matches[i][j].draws == second.matches[i][j].draws);
		}

	return NULL;
}

static char *test_the_model_that_learned_beats_the_one_that_did_not(void)
{
	arena_settings_t settings;
	arena_report_t report;

	build_small_settings(&settings);
	settings.training_games = 500;
	settings.match_games = 40;
	settings.learning = ARENA_LEARN_TRUTH;

	_it_should("run", arena_run_all(&settings, &report));

	/* the model that was started at random has learned the worth of a
	 * position and the one that was started at zero never can, so the
	 * second should get the better of it */
	_it_should("give the learned model the better of it",
		report.matches[INIT_ZERO][INIT_RANDOM].first_wins <
		report.matches[INIT_ZERO][INIT_RANDOM].second_wins);

	return NULL;
}

static char *test_a_survey_runs_the_arena_several_times(void)
{
	arena_settings_t settings;
	arena_survey_t survey;

	build_small_settings(&settings);

	_it_should("run", arena_survey(&settings, 3, &survey));
	_it_should("keep a report for every run", survey.runs == 3);

	for (int run = 0; run < survey.runs; run++) {
		for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
			_it_should("train every model in every run",
				survey.reports[run].results[scheme].trained_games ==
				settings.training_games);
		}
	}

	return NULL;
}

static char *test_each_run_has_a_seed_of_its_own(void)
{
	arena_settings_t settings;
	arena_survey_t survey;

	build_small_settings(&settings);

	arena_survey(&settings, 3, &survey);

	for (int run = 1; run < survey.runs; run++)
		_it_should("use a different seed in every run",
			survey.reports[run].training_seed !=
			survey.reports[run - 1].training_seed);

	return NULL;
}

static char *test_the_survey_counts_the_round_robin(void)
{
	arena_settings_t settings;
	arena_survey_t survey;
	int expected_won[ARENA_SCHEMES] = { 0, 0, 0 };
	int expected_lost[ARENA_SCHEMES] = { 0, 0, 0 };
	int expected_draws[ARENA_SCHEMES] = { 0, 0, 0 };
	int expected_winners[ARENA_SCHEMES] = { 0, 0, 0 };
	int winners = 0;

	build_small_settings(&settings);
	settings.match_games = 10;

	arena_survey(&settings, 3, &survey);

	/* what the reports say, worked out again here */
	for (int run = 0; run < survey.runs; run++) {
		arena_report_t *report = &survey.reports[run];
		int wins[ARENA_SCHEMES] = { 0, 0, 0 };
		int best = -1;

		for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++)
			expected_draws[scheme] += report->results[scheme].perfect_after.draws;

		for (int i = 0; i < ARENA_SCHEMES; i++)
			for (int j = i + 1; j < ARENA_SCHEMES; j++) {
				wins[i] += report->matches[i][j].first_wins;
				wins[j] += report->matches[i][j].second_wins;

				expected_won[i] += report->matches[i][j].first_wins;
				expected_lost[i] += report->matches[i][j].second_wins;
				expected_won[j] += report->matches[i][j].second_wins;
				expected_lost[j] += report->matches[i][j].first_wins;
			}

		/* the model that won the most of the matches won the round robin,
		 * unless two of them tied */
		int best_wins = -1;
		int tied = 0;

		for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
			if (wins[scheme] > best_wins) {
				best_wins = wins[scheme];
				best = scheme;
				tied = 0;
			} else if (wins[scheme] == best_wins) {
				tied = 1;
			}
		}

		if (!tied) expected_winners[best]++;
	}

	for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
		winners += survey.round_robin_wins[scheme];

		_it_should("count the rounds it won",
			survey.round_robin_wins[scheme] == expected_winners[scheme]);
		_it_should("count the games it won",
			survey.games_won[scheme] == expected_won[scheme]);
		_it_should("count the games it lost",
			survey.games_lost[scheme] == expected_lost[scheme]);
		_it_should("count the draws against the perfect player",
			survey.perfect_draws[scheme] == expected_draws[scheme]);
	}

	/* a run has one winner, or none when the models tie */
	_it_should("count at most one winner for each run", winners <= survey.runs);

	return NULL;
}

static char *test_the_same_survey_comes_again(void)
{
	arena_settings_t settings;
	arena_survey_t first;
	arena_survey_t second;

	build_small_settings(&settings);

	arena_survey(&settings, 2, &first);
	arena_survey(&settings, 2, &second);

	for (int run = 0; run < first.runs; run++) {
		for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
			_it_should("give the same truth error",
				nearly_equal(first.reports[run].results[scheme].truth_after,
				             second.reports[run].results[scheme].truth_after));
		}
	}

	return NULL;
}

static char *test_a_survey_refuses_bad_arguments(void)
{
	arena_settings_t settings;
	arena_survey_t survey;

	build_small_settings(&settings);

	_it_should("refuse no settings", !arena_survey(NULL, 3, &survey));
	_it_should("refuse nowhere to put the survey", !arena_survey(&settings, 3, NULL));
	_it_should("refuse no runs", !arena_survey(&settings, 0, &survey));
	_it_should("refuse more runs than it holds",
		!arena_survey(&settings, ARENA_MAX_RUNS + 1, &survey));

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
	_run_test(test_the_models_play_each_other);
	_run_test(test_the_same_report_comes_again);
	_run_test(test_the_model_that_learned_beats_the_one_that_did_not);
	_run_test(test_a_survey_runs_the_arena_several_times);
	_run_test(test_each_run_has_a_seed_of_its_own);
	_run_test(test_the_survey_counts_the_round_robin);
	_run_test(test_the_same_survey_comes_again);
	_run_test(test_a_survey_refuses_bad_arguments);

	return NULL;
}
