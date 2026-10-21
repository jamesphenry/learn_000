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

static char *test_a_duel_is_three_meetings(void)
{
	arena_settings_t settings;
	arena_duel_t duel;

	build_small_settings(&settings);
	settings.match_games = 20;

	_it_should("run", arena_duel(&settings, 42, 1978, &duel));

	/* and they meet three times */
	_it_should("meet before either has learned",
		duel.stages[DUEL_BEFORE].meeting.games == settings.match_games);
	_it_should("meet after each has learned on its own",
		duel.stages[DUEL_OWN_GAMES].meeting.games == settings.match_games);
	_it_should("meet after they have learned together",
		duel.stages[DUEL_TOGETHER].meeting.games == settings.match_games);

	/* and the games they trained on are counted */
	_it_should("count the games they trained on",
		duel.together_games.games == settings.training_games);
	_it_should("count every one of them once",
		duel.together_games.first_wins + duel.together_games.draws +
		duel.together_games.second_wins == settings.training_games);

	/* and what they were like before either of them had learned anything
	 * is the same as what the arena measures for a model that has never
	 * seen a game */
	_it_should("measure the truth the way the arena does",
		duel.stages[DUEL_BEFORE].truth[0] > 0.2 &&
		duel.stages[DUEL_BEFORE].truth[0] < 0.4);

	return NULL;
}

static char *test_the_two_models_of_a_duel_start_off_differently(void)
{
	arena_settings_t settings;
	arena_duel_t duel;

	build_small_settings(&settings);

	arena_duel(&settings, 42, 1978, &duel);

	/* each is started off with its own seed, so they are not the same model */
	_it_should("start the two models off differently",
		duel.stages[DUEL_BEFORE].initialisation[0].hidden_lowest !=
		duel.stages[DUEL_BEFORE].initialisation[1].hidden_lowest ||
		duel.stages[DUEL_BEFORE].initialisation[0].worth !=
		duel.stages[DUEL_BEFORE].initialisation[1].worth);

	return NULL;
}

/*
 * A model that has never seen a game is measured the same way by the duel
 * and by the arena, so the two must agree.
 */
static char *test_the_duel_measures_a_model_the_way_the_arena_does(void)
{
	arena_settings_t settings;
	arena_duel_t duel;
	arena_result_t alone;

	build_small_settings(&settings);
	settings.training_games = 0;
	settings.training_seed = 42;

	arena_duel(&settings, 42, 42, &duel);
	arena_run(&settings, INIT_RANDOM, &alone);

	_it_should("measure the truth the same way",
		nearly_equal(duel.stages[DUEL_BEFORE].truth[0], alone.truth_before));
	_it_should("measure the worth given away the same way",
		nearly_equal(duel.stages[DUEL_BEFORE].value_lost[0], alone.value_lost_before));

	return NULL;
}

static char *test_both_models_learn_from_the_games_they_play_together(void)
{
	arena_settings_t settings;
	arena_duel_t duel;

	build_small_settings(&settings);
	settings.training_games = 500;

	arena_duel(&settings, 42, 1978, &duel);

	/* the games they play against each other change both of them: each
	 * ends up worth something different from what it was worth when it had
	 * only played against itself */
	_it_should("change the first model",
		duel.stages[DUEL_TOGETHER].truth[0] !=
		duel.stages[DUEL_OWN_GAMES].truth[0]);
	_it_should("change the second model",
		duel.stages[DUEL_TOGETHER].truth[1] !=
		duel.stages[DUEL_OWN_GAMES].truth[1]);

	return NULL;
}

static char *test_the_same_duel_comes_again(void)
{
	arena_settings_t settings;
	arena_duel_t first;
	arena_duel_t second;

	build_small_settings(&settings);
	settings.match_games = 20;

	arena_duel(&settings, 42, 1978, &first);
	arena_duel(&settings, 42, 1978, &second);

	_it_should("give the same first meeting",
		first.stages[DUEL_BEFORE].meeting.first_wins ==
		second.stages[DUEL_BEFORE].meeting.first_wins &&
		first.stages[DUEL_BEFORE].meeting.draws ==
		second.stages[DUEL_BEFORE].meeting.draws &&
		first.stages[DUEL_BEFORE].meeting.second_wins ==
		second.stages[DUEL_BEFORE].meeting.second_wins);
	_it_should("give the same truth error",
		nearly_equal(first.stages[DUEL_TOGETHER].truth[0],
		             second.stages[DUEL_TOGETHER].truth[0]));
	_it_should("give the same truth error for the second",
		nearly_equal(first.stages[DUEL_TOGETHER].truth[1],
		             second.stages[DUEL_TOGETHER].truth[1]));

	return NULL;
}

static char *test_a_duel_refuses_bad_arguments(void)
{
	arena_settings_t settings;
	arena_duel_t duel;

	build_small_settings(&settings);

	_it_should("refuse no settings", !arena_duel(NULL, 42, 1978, &duel));
	_it_should("refuse nowhere to put the duel", !arena_duel(&settings, 42, 1978, NULL));

	return NULL;
}

static char *test_a_league_plays_every_pair(void)
{
	arena_settings_t settings;
	arena_league_t league;

	build_small_settings(&settings);
	settings.match_games = 10;

	_it_should("run", arena_league(&settings, 8, &league));

	_it_should("hold every model", league.models == 8);

	/* every model plays every other one, and the points are added up */
	for (int i = 0; i < league.models; i++) {
		int games = league.model[i].wins + league.model[i].draws +
		            league.model[i].losses;

		_it_should("play every other model", games == (league.models - 1) *
			settings.match_games);
		_it_should("count its points from its results",
			league.model[i].points == 3 * league.model[i].wins + league.model[i].draws);

		/* and it played models of both kinds */
		_it_should("have played the models of its own kind",
			league.model[i].same_wins + league.model[i].same_draws > 0);
		_it_should("have played the models of the other kind",
			league.model[i].other_wins + league.model[i].other_draws > 0);

		/* and the same held back opponents as every other model, which is
		 * what makes the third ranking a fair comparison */
		_it_should("have played the held back models of its own kind",
			league.model[i].held_same_games > 0);
		_it_should("have played the held back models of the other kind",
			league.model[i].held_other_games > 0);
		_it_should("have played as many held back games as every other model",
			league.model[i].held_same_games + league.model[i].held_other_games ==
			league.model[0].held_same_games + league.model[0].held_other_games);
		_it_should("cannot have played more held back games than there are",
			league.model[i].held_same_games + league.model[i].held_other_games <=
			ARENA_HELD_OUT * settings.match_games);


		_it_should("count its points against its own kind from those games",
			league.model[i].same_points == 3 * league.model[i].same_wins +
			league.model[i].same_draws);
		_it_should("count its points against the other kind from those games",
			league.model[i].other_points == 3 * league.model[i].other_wins +
			league.model[i].other_draws);
	}

	_it_should("crown a champion", league.champion >= 0 && league.champion < league.models);
	_it_should("crown the model with the most points",
		league.model[league.champion].points ==
		league.model[league.ranking_league[0]].points);

	return NULL;
}

static char *test_every_model_in_a_league_has_a_seed_of_its_own(void)
{
	arena_settings_t settings;
	arena_league_t league;

	build_small_settings(&settings);

	arena_league(&settings, 8, &league);

	for (int i = 0; i < league.models; i++)
		_it_should("give each model a seed of its own",
			league.model[i].seed != league.model[(i + 1) % league.models].seed);

	return NULL;
}

static char *test_the_three_rankings_of_a_league_are_in_order(void)
{
	arena_settings_t settings;
	arena_league_t league;

	build_small_settings(&settings);

	arena_league(&settings, 8, &league);

	for (int i = 1; i < league.models; i++) {
		/* the less worth it gave away while learning, the better it
		 * learned, so the order is the other way up */
		_it_should("put the models in the order of how well they learned",
			league.model[league.ranking_training[i - 1]].training_value_lost <=
			league.model[league.ranking_training[i]].training_value_lost);
		_it_should("put the models in the order of their points",
			league.model[league.ranking_league[i - 1]].points >=
			league.model[league.ranking_league[i]].points);
		_it_should("put the models in the order of how they did against the"
			   " held back models of the other kind",
			league.model[league.ranking_general[i - 1]].held_other_points >=
			league.model[league.ranking_general[i]].held_other_points);
	}

	/* and the three rankings say three different things: how well a model
	 * learned is not how well it plays */
	_it_should("rank the models differently by training and by the league",
		league.ranking_training[0] != league.ranking_league[0]);

	return NULL;
}

/*
 * A model that has learned the game beats one that has not, so the models
 * taught by the perfect player earn more against the other kind than against
 * their own, and the others earn less. The models here have to have learned
 * something for that to be true.
 */
static char *test_the_held_back_opponents_are_of_both_kinds(void)
{
	arena_settings_t settings;
	arena_league_t league;

	build_small_settings(&settings);
	settings.training_games = 2000;
	settings.hidden_neurons = 8;

	arena_league(&settings, 6, &league);

	for (int i = 0; i < league.models; i++) {
		if (league.model[i].teacher == LEAGUE_TRUTH)
			_it_should("earn at least as much against the other kind as"
				   " against its own",
				league.model[i].held_other_points >=
				league.model[i].held_same_points);
		else
			_it_should("earn less against the other kind than against its own",
				league.model[i].held_other_points < league.model[i].held_same_points);
	}

	return NULL;
}

static char *test_the_held_back_models_are_not_in_the_league(void)
{
	arena_settings_t settings;
	arena_league_t league;

	build_small_settings(&settings);

	arena_league(&settings, 8, &league);

	for (int held = 0; held < ARENA_HELD_OUT; held++) {
		int in_the_league = 0;

		for (int i = 0; i < league.models; i++)
			if (league.model[i].seed == league.held_seed[held]) in_the_league = 1;

		_it_should("hold a model back that is not in the league", !in_the_league);
	}

	return NULL;
}

static char *test_the_same_league_comes_again(void)
{
	arena_settings_t settings;
	arena_league_t first;
	arena_league_t second;

	build_small_settings(&settings);

	arena_league(&settings, 8, &first);
	arena_league(&settings, 8, &second);

	_it_should("give the same champion", first.champion == second.champion);
	_it_should("give the same points",
		first.model[first.champion].points == second.model[second.champion].points);
	_it_should("give the same truth for the champion",
		nearly_equal(first.model[first.champion].truth,
		             second.model[second.champion].truth));

	return NULL;
}

static char *test_a_league_refuses_bad_arguments(void)
{
	arena_settings_t settings;
	arena_league_t league;

	build_small_settings(&settings);

	_it_should("refuse no settings", !arena_league(NULL, 8, &league));
	_it_should("refuse nowhere to put the league", !arena_league(&settings, 8, NULL));
	_it_should("refuse one model", !arena_league(&settings, 1, &league));
	_it_should("refuse more models than it holds",
		!arena_league(&settings, ARENA_MAX_MODELS + 1, &league));

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
	_run_test(test_a_duel_is_three_meetings);
	_run_test(test_the_two_models_of_a_duel_start_off_differently);
	_run_test(test_the_duel_measures_a_model_the_way_the_arena_does);
	_run_test(test_both_models_learn_from_the_games_they_play_together);
	_run_test(test_the_same_duel_comes_again);
	_run_test(test_a_duel_refuses_bad_arguments);
	_run_test(test_a_league_plays_every_pair);
	_run_test(test_every_model_in_a_league_has_a_seed_of_its_own);
	_run_test(test_the_three_rankings_of_a_league_are_in_order);
	_run_test(test_the_held_back_opponents_are_of_both_kinds);
	_run_test(test_the_held_back_models_are_not_in_the_league);
	_run_test(test_the_same_league_comes_again);
	_run_test(test_a_league_refuses_bad_arguments);

	return NULL;
}
