#include <stdlib.h>
#include <string.h>
#include "arena.h"
#include "evaluate.h"
#include "game.h"
#include "policy.h"
#include "trainer.h"

void arena_settings_init(arena_settings_t *settings)
{
	if (settings == NULL) return;

	settings->hidden_neurons = 16;
	settings->training_games = 20000;
	settings->evaluation_games = 200;
	settings->match_games = 200;
	settings->training_seed = 1;
	settings->evaluation_seed = 2;
	settings->learning_rate = 0.05;
	settings->temperature = 0.3;
	settings->init_scale = 0.5;
	settings->init_value = 1.0;
	settings->learning = ARENA_LEARN_TRUTH;
}

/* Builds the network, and starts it off in the way that was asked for. */
bool arena_build(const game_t *game, network_t *network,
                 const arena_settings_t *settings, init_scheme_t scheme, rng_t *rng)
{
	network_init(network);

	if (!network_add_layer(network, game->n_features, settings->hidden_neurons,
	                       ACTIVATION_RELU))
		return false;

	if (!network_add_layer(network, settings->hidden_neurons, 1, ACTIVATION_SIGMOID))
		return false;

	switch (scheme) {
	case INIT_RANDOM:
		return network_fill_random(network, rng, settings->init_scale);

	case INIT_MAXIMUM:
		return network_fill_maximum(network, settings->init_value);

	case INIT_ZERO:
		/* a network starts with everything at zero on its own */
		return true;
	}

	return false;
}

/*
 * Learns what the perfect player knows about every position it plays. The
 * games are its own, so the positions are the ones it visits, and the
 * perfect player says what each of them is worth.
 */
static int arena_learn_from_the_truth(trainer_t *trainer, const game_t *game,
                                      const oracle_t *oracle, policy_t *policy, int games)
{
	game_trace_t trace;
	int learned = 0;

	for (int i = 0; i < games; i++) {
		if (!game_play(game, policy_choose, policy, policy_choose, policy, &trace))
			break;

		if (!trainer_learn_from_oracle(trainer, game, oracle, &trace)) break;

		learned++;
	}

	return learned;
}

/* What a model looks like at one moment. */
typedef struct {
	evaluation_t evaluation;
	skill_t skill;      /* against an opponent that plays at random */
	skill_t own_skill;  /* against itself */
	perfect_result_t perfect;
} arena_look_t;

/* Measures a model: how it does against an opponent that plays at random,
 * how much worth it gives away, and how it does against the perfect player. */
static void arena_measure(const arena_settings_t *settings, const game_t *game,
                           const oracle_t *oracle, const network_t *network,
                           arena_look_t *look)
{
	memset(look, 0, sizeof(*look));

	evaluate_network(game, oracle, network, settings->evaluation_seed,
	                 settings->evaluation_games, &look->evaluation);
	evaluate_skill(game, oracle, network, settings->evaluation_seed,
	               settings->evaluation_games, POLICY_RANDOM_TEMPERATURE,
	               &look->skill);
	evaluate_skill(game, oracle, network, settings->evaluation_seed,
	               settings->evaluation_games, 0.0, &look->own_skill);
	evaluate_against_perfect(game, oracle, network, settings->evaluation_seed,
	                         settings->evaluation_games, &look->perfect);
}

/* Fills in what a model did, from what it looked like before and after. */
static void arena_result_of(const arena_look_t *before, const arena_look_t *after,
                             int trained, arena_result_t *result)
{
	result->win_rate_before = 100.0 * before->evaluation.wins / before->evaluation.games;
	result->win_rate_after = 100.0 * after->evaluation.wins / after->evaluation.games;
	result->brier_before = before->evaluation.brier_score;
	result->brier_after = after->evaluation.brier_score;
	result->truth_before = before->evaluation.truth_error;
	result->truth_after = after->evaluation.truth_error;
	result->value_lost_before = before->skill.moves > 0 ?
		before->skill.value_lost / before->skill.moves : 0.0;
	result->value_lost_after = after->skill.moves > 0 ?
		after->skill.value_lost / after->skill.moves : 0.0;
	result->value_lost_own_before = before->own_skill.moves > 0 ?
		before->own_skill.value_lost / before->own_skill.moves : 0.0;
	result->value_lost_own_after = after->own_skill.moves > 0 ?
		after->own_skill.value_lost / after->own_skill.moves : 0.0;
	result->perfect_before = before->perfect;
	result->perfect_after = after->perfect;
	result->trained_games = trained;
}

/*
 * Teaches a model the way the settings say, and says how many games it
 * learned from.
 *
 * The perfect player teaches it, when the settings say so; otherwise it
 * learns from the result of the games it plays against itself.
 */
bool arena_teach(const arena_settings_t *settings, const game_t *game,
                 const oracle_t *oracle, network_t *network, int *trained)
{
	policy_t policy;
	trainer_t trainer;
	int learned = 0;

	trainer_init(&trainer, network, settings->learning_rate);
	policy_init(&policy, game, network, settings->training_seed, settings->temperature);

	if (settings->learning == ARENA_LEARN_TRUTH)
		learned = arena_learn_from_the_truth(&trainer, game, oracle, &policy,
		                                    settings->training_games);
	else
		learned = trainer_run(&trainer, game, &policy, settings->training_games);

	if (trained != NULL) *trained = learned;

	return true;
}
/*
 * Builds one model, measures it, teaches it, and measures it again. The
 * games it learns from and the games it is measured with are apart from one
 * another, so the measurement is on games the model has not seen.
 */
static bool arena_train_one(const arena_settings_t *settings, const game_t *game,
                             const oracle_t *oracle, init_scheme_t scheme,
                             rng_t *rng, network_t *network, arena_result_t *result)
{
	arena_look_t before;
	arena_look_t after;
	int trained = 0;

	if (!arena_build(game, network, settings, scheme, rng)) return false;

	/* what the model is like before it has learned anything */
	arena_measure(settings, game, oracle, network, &before);
	evaluate_initialisation(game, network, &result->initialisation);

	/* and then it learns */
	if (!arena_teach(settings, game, oracle, network, &trained)) return false;

	/* and what it is like afterwards, on games it has not seen */
	arena_measure(settings, game, oracle, network, &after);

	arena_result_of(&before, &after, trained, result);

	return true;
}

bool arena_train(const arena_settings_t *settings, const game_t *game,
                 const oracle_t *oracle, init_scheme_t scheme, rng_t *rng,
                 network_t *network, int *trained)
{
	if (settings == NULL || game == NULL || oracle == NULL || network == NULL) return false;
	if (rng == NULL) return false;
	if (settings->training_games < 0 || settings->hidden_neurons < 1) return false;

	if (!arena_build(game, network, settings, scheme, rng)) return false;

	return arena_teach(settings, game, oracle, network, trained);
}

bool arena_run(const arena_settings_t *settings, const game_t *game,
               const oracle_t *oracle, init_scheme_t scheme, arena_result_t *result)
{
	network_t network;
	rng_t rng;

	if (settings == NULL || game == NULL || oracle == NULL || result == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(result, 0, sizeof(*result));

	rng_seed(&rng, settings->training_seed);

	return arena_train_one(settings, game, oracle, scheme, &rng, &network, result);
}

bool arena_run_all(const arena_settings_t *settings, const game_t *game,
                   const oracle_t *oracle, arena_report_t *report)
{
	network_t networks[ARENA_SCHEMES];
	rng_t rng;

	if (settings == NULL || game == NULL || oracle == NULL || report == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->match_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(report, 0, sizeof(*report));
	report->training_seed = settings->training_seed;

	rng_seed(&rng, settings->training_seed);

	/* every model is given the same conditions, and the only thing that
	 * differs between them is how they were started off */
	for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
		if (!arena_train_one(settings, game, oracle, (init_scheme_t)scheme, &rng,
		                     &networks[scheme], &report->results[scheme]))
			return false;
	}

	/* and then they play each other, each of them taking one side and
	 * then the other */
	for (int first = 0; first < ARENA_SCHEMES; first++) {
		for (int second = first + 1; second < ARENA_SCHEMES; second++) {
			if (!evaluate_against(game, oracle, &networks[first], &networks[second],
			                      settings->evaluation_seed, settings->match_games,
			                      0.0, &report->matches[first][second]))
				return false;
		}
	}

	return true;
}

/*
 * The model that won the most of the matches in a run: the one that came
 * out on top of the round robin. -1 when the models tied.
 */
static int arena_winner_of_the_round_robin(const arena_report_t *report)
{
	int best = -1;
	int best_wins = -1;
	int tied = 0;

	for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
		int wins = 0;

		for (int other = 0; other < ARENA_SCHEMES; other++) {
			if (other == scheme) continue;

			if (scheme < other)
				wins += report->matches[scheme][other].first_wins;
			else
				wins += report->matches[other][scheme].second_wins;
		}

		if (wins > best_wins) {
			best_wins = wins;
			best = scheme;
			tied = 0;
		} else if (wins == best_wins) {
			tied = 1;
		}
	}

	return tied ? -1 : best;
}

/* Fills in what the two models were like at one stage of a duel. */
static void arena_stage(const arena_settings_t *settings, const game_t *game,
                        const oracle_t *oracle, const network_t *first,
                        const arena_look_t *first_look, const network_t *second,
                        const arena_look_t *second_look, arena_stage_result_t *stage)
{
	stage->value_lost[0] = first_look->skill.moves > 0 ?
		first_look->skill.value_lost / first_look->skill.moves : 0.0;
	stage->value_lost[1] = second_look->skill.moves > 0 ?
		second_look->skill.value_lost / second_look->skill.moves : 0.0;
	stage->truth[0] = first_look->evaluation.truth_error;
	stage->truth[1] = second_look->evaluation.truth_error;
	stage->perfect_draws[0] = first_look->perfect.draws;
	stage->perfect_draws[1] = second_look->perfect.draws;
	stage->perfect_losses[0] = first_look->perfect.losses;
	stage->perfect_losses[1] = second_look->perfect.losses;

	evaluate_against(game, oracle, first, second, settings->evaluation_seed,
	                 settings->match_games, 0.0, &stage->meeting);
}

/*
 * Plays a game between the two models and tallies it.
 */
static void arena_tally_a_game(const game_t *game, const game_trace_t *trace,
                               int first_is_first, match_result_t *tally)
{
	game_outcome_t outcome = game->outcome(&trace->position);

	if (outcome == GAME_DRAW) {
		tally->draws++;
	} else if ((outcome == GAME_FIRST_WINS) == (first_is_first != 0)) {
		tally->first_wins++;

		if (first_is_first) tally->first_wins_as_first++;
	} else {
		tally->second_wins++;
	}

	tally->games++;
}

bool arena_duel(const arena_settings_t *settings, const game_t *game,
                const oracle_t *oracle, unsigned long long first_seed,
                unsigned long long second_seed, arena_duel_t *duel)
{
	network_t first;
	network_t second;
	policy_t first_policy;
	policy_t second_policy;
	trainer_t first_trainer;
	trainer_t second_trainer;
	arena_look_t first_before;
	arena_look_t first_after;
	arena_look_t second_before;
	arena_look_t second_after;
	rng_t first_rng;
	rng_t second_rng;

	if (settings == NULL || game == NULL || oracle == NULL || duel == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->match_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(duel, 0, sizeof(*duel));

	/* the two models, each started off with its own seed */
	rng_seed(&first_rng, first_seed);
	rng_seed(&second_rng, second_seed);

	if (!arena_build(game, &first, settings, INIT_RANDOM, &first_rng)) return false;
	if (!arena_build(game, &second, settings, INIT_RANDOM, &second_rng)) return false;

	arena_measure(settings, game, oracle, &first, &first_before);
	arena_measure(settings, game, oracle, &second, &second_before);
	evaluate_initialisation(game, &first, &duel->stages[DUEL_BEFORE].initialisation[0]);
	evaluate_initialisation(game, &second, &duel->stages[DUEL_BEFORE].initialisation[1]);

	/* and how they fare against each other before either has learned
	 * anything */
	arena_stage(settings, game, oracle, &first, &first_before, &second, &second_before,
	            &duel->stages[DUEL_BEFORE]);

	/* each of them learns from the games it plays against itself */
	trainer_init(&first_trainer, &first, settings->learning_rate);
	trainer_init(&second_trainer, &second, settings->learning_rate);
	policy_init(&first_policy, game, &first, first_seed, settings->temperature);
	policy_init(&second_policy, game, &second, second_seed, settings->temperature);

	trainer_run(&first_trainer, game, &first_policy, settings->training_games);
	trainer_run(&second_trainer, game, &second_policy, settings->training_games);

	arena_measure(settings, game, oracle, &first, &first_after);
	arena_measure(settings, game, oracle, &second, &second_after);

	/* and they meet again */
	arena_stage(settings, game, oracle, &first, &first_after, &second, &second_after,
	            &duel->stages[DUEL_OWN_GAMES]);

	/* and now they learn against each other */
	for (int i = 0; i < settings->training_games; i++) {
		game_trace_t trace;
		int first_is_first = (i % 2) == 0;

		if (first_is_first)
			game_play(game, policy_choose, &first_policy, policy_choose,
			          &second_policy, &trace);
		else
			game_play(game, policy_choose, &second_policy, policy_choose,
			          &first_policy, &trace);

		arena_tally_a_game(game, &trace, first_is_first, &duel->together_games);

		/* both of them learn the result of the game they played */
		trainer_learn_from_outcome(&first_trainer, game, &trace);
		trainer_learn_from_outcome(&second_trainer, game, &trace);
	}

	arena_measure(settings, game, oracle, &first, &first_after);
	arena_measure(settings, game, oracle, &second, &second_after);

	/* and they meet once more */
	arena_stage(settings, game, oracle, &first, &first_after, &second, &second_after,
	            &duel->stages[DUEL_TOGETHER]);

	return true;
}

/*
 * Puts the models in order by a measure, best first. The measure is a
 * number where bigger is better.
 */
static void league_rank(arena_league_t *league, int *ranking,
                        double (*measure)(const league_model_t *))
{
	for (int i = 0; i < league->models; i++)
		ranking[i] = i;

	/* a plain sort, so that two models that are the same keep the order
	 * they were given */
	for (int i = 1; i < league->models; i++) {
		int model = ranking[i];
		int j = i - 1;

		while (j >= 0 && measure(&league->model[ranking[j]]) <
		                  measure(&league->model[model])) {
			ranking[j + 1] = ranking[j];
			j--;
		}

		ranking[j + 1] = model;
	}
}

/* How well it learned from its own games: the less worth it gave away in
 * the positions it visited while learning, the better. */
static double measure_training(const league_model_t *model)
{
	return -model->training_value_lost;
}

/* How well it did in the league, against every other model. */
static double measure_league(const league_model_t *model)
{
	return model->points;
}

/* How well it did against the models held back from the league, which are
 * the same for every model. */
static double measure_general(const league_model_t *model)
{
	return model->held_other_points;
}

/*
 * Builds one model and teaches it, and says how well it learned.
 *
 * The temperature is how much the policy explores while it is learning, and
 * it is what makes two models of the same kind play differently.
 */
static bool league_build_one(const arena_settings_t *settings, const game_t *game,
                             const oracle_t *oracle, unsigned long long seed,
                             league_teacher_t teacher, double temperature,
                             network_t *network, policy_t *policy, trainer_t *trainer,
                             league_model_t *record)
{
	arena_look_t look;
	rng_t rng;

	rng_seed(&rng, seed);

	if (!arena_build(game, network, settings, INIT_RANDOM, &rng)) return false;

	trainer_init(trainer, network, settings->learning_rate);
	policy_init(policy, game, network, seed, temperature);

	if (teacher == LEAGUE_TRUTH)
		arena_learn_from_the_truth(trainer, game, oracle, policy, settings->training_games);
	else
		trainer_run(trainer, game, policy, settings->training_games);

	arena_measure(settings, game, oracle, network, &look);

	record->seed = seed;
	record->teacher = teacher;
	record->training_value_lost = look.own_skill.moves > 0 ?
		look.own_skill.value_lost / look.own_skill.moves : 0.0;
	record->truth = look.evaluation.truth_error;
	record->value_lost = look.skill.moves > 0 ?
		look.skill.value_lost / look.skill.moves : 0.0;
	record->perfect_draws = look.perfect.draws;
	record->perfect_losses = look.perfect.losses;

	return true;
}

bool arena_league(const arena_settings_t *settings, const game_t *game,
                  const oracle_t *oracle, int models, arena_league_t *league)
{
	network_t *network = NULL;
	policy_t *policy = NULL;
	trainer_t *trainer = NULL;
	network_t held_network[ARENA_HELD_OUT];
	policy_t held_policy[ARENA_HELD_OUT];
	rng_t rng;

	if (settings == NULL || game == NULL || oracle == NULL || league == NULL) return false;
	if (models < 2 || models > ARENA_MAX_MODELS) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->match_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(league, 0, sizeof(*league));

	/* the models are far too big to hold on the stack all at once */
	network = malloc(sizeof(*network) * (size_t)models);
	policy = malloc(sizeof(*policy) * (size_t)models);
	trainer = malloc(sizeof(*trainer) * (size_t)models);

	if (network == NULL || policy == NULL || trainer == NULL) {
		free(network);
		free(policy);
		free(trainer);
		return false;
	}

	league->models = models;

	/* half the models are taught by the perfect player and half learn from
	 * their own games, so the league holds opponents of two kinds */
	for (int i = 0; i < models; i++) {
		league_teacher_t teacher = (i % 2 == 0) ? LEAGUE_TRUTH : LEAGUE_SELF;
		unsigned long long seed = settings->training_seed + (unsigned long long)i;

		if (!league_build_one(settings, game, oracle, seed, teacher, settings->temperature,
		                      &network[i], &policy[i], &trainer[i], &league->model[i])) {
			free(network);
			free(policy);
			free(trainer);
			return false;
		}
	}

	/* four models are held back from the league and never play in it: one
	 * of each kind, each trained to play in two different ways, so that
	 * every model meets the same opponents of the same kinds */
	for (int held = 0; held < ARENA_HELD_OUT; held++) {
		league_teacher_t teacher = (held % 2 == 0) ? LEAGUE_TRUTH : LEAGUE_SELF;
		double temperature = (held / 2 == 0) ?
			settings->temperature : settings->temperature * 4.0;
		league_model_t record;
		trainer_t held_trainer;

		league->held_seed[held] = settings->training_seed +
		                          (unsigned long long)models +
		                          (unsigned long long)held;

		if (!league_build_one(settings, game, oracle, league->held_seed[held],
		                      teacher, temperature, &held_network[held],
		                      &held_policy[held], &held_trainer, &record)) {
			free(network);
			free(policy);
			free(trainer);
			return false;
		}
	}

	/* and every model is measured against them */
	for (int i = 0; i < models; i++) {
		for (int held = 0; held < ARENA_HELD_OUT; held++) {
			match_result_t match;
			int same_kind = (league->model[i].teacher ==
			                 (held % 2 == 0) ? LEAGUE_TRUTH : LEAGUE_SELF);

			evaluate_against(game, oracle, &network[i], &held_network[held],
			                 settings->evaluation_seed, settings->match_games,
			                 0.0, &match);

			if (same_kind) {
				league->model[i].held_same_points += 3 * match.first_wins + match.draws;
				league->model[i].held_same_games += settings->match_games;
			} else {
				league->model[i].held_other_points += 3 * match.first_wins + match.draws;
				league->model[i].held_other_games += settings->match_games;
			}
		}
	}

	/* and every model plays every other one */
	rng_seed(&rng, settings->evaluation_seed);

	for (int i = 0; i < models; i++) {
		for (int j = i + 1; j < models; j++) {
			match_result_t match;
			int first_points;
			int second_points;
			int same_kind;

			evaluate_against(game, oracle, &network[i], &network[j], settings->evaluation_seed,
			                 settings->match_games, 0.0, &match);

			first_points = 3 * match.first_wins + match.draws;
			second_points = 3 * match.second_wins + match.draws;

			league->model[i].wins += match.first_wins;
			league->model[i].draws += match.draws;
			league->model[i].losses += match.second_wins;
			league->model[i].points += first_points;

			league->model[j].wins += match.second_wins;
			league->model[j].draws += match.draws;
			league->model[j].losses += match.first_wins;
			league->model[j].points += second_points;

			same_kind = (league->model[i].teacher == league->model[j].teacher);

			if (same_kind) {
				league->model[i].same_wins += match.first_wins;
				league->model[i].same_draws += match.draws;
				league->model[i].same_points += first_points;

				league->model[j].same_wins += match.second_wins;
				league->model[j].same_draws += match.draws;
				league->model[j].same_points += second_points;
			} else {
				league->model[i].other_wins += match.first_wins;
				league->model[i].other_draws += match.draws;
				league->model[i].other_points += first_points;

				league->model[j].other_wins += match.second_wins;
				league->model[j].other_draws += match.draws;
				league->model[j].other_points += second_points;
			}
		}
	}

	league_rank(league, league->ranking_training, measure_training);
	league_rank(league, league->ranking_league, measure_league);
	league_rank(league, league->ranking_general, measure_general);

	/* the champion is the one at the top of the league table */
	league->champion = league->ranking_league[0];

	free(network);
	free(policy);
	free(trainer);

	return true;
}

bool arena_survey(const arena_settings_t *settings, const game_t *game,
                  const oracle_t *oracle, int runs, arena_survey_t *survey)
{
	if (settings == NULL || game == NULL || oracle == NULL || survey == NULL) return false;
	if (runs < 1 || runs > ARENA_MAX_RUNS) return false;

	memset(survey, 0, sizeof(*survey));

	for (int run = 0; run < runs; run++) {
		arena_settings_t one = *settings;
		int winner;

		/* each run has a seed of its own, so the models are started off
		 * differently and learn from different games */
		one.training_seed = settings->training_seed + (unsigned long long)run;

		if (!arena_run_all(&one, game, oracle, &survey->reports[run])) return false;

		survey->runs++;

		winner = arena_winner_of_the_round_robin(&survey->reports[run]);

		if (winner >= 0) survey->round_robin_wins[winner]++;

		for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
			int other;

			/* how the model did against the perfect player */
			survey->perfect_wins[scheme] +=
				survey->reports[run].results[scheme].perfect_after.wins;
			survey->perfect_draws[scheme] +=
				survey->reports[run].results[scheme].perfect_after.draws;
			survey->perfect_losses[scheme] +=
				survey->reports[run].results[scheme].perfect_after.losses;

			/* and how many of the round robin it lost */
			for (other = 0; other < ARENA_SCHEMES; other++) {
				int lost;

				if (other == scheme) continue;

				if (scheme < other) {
					lost = survey->reports[run].matches[scheme][other].second_wins;
					survey->games_won[scheme] +=
						survey->reports[run].matches[scheme][other].first_wins;
				} else {
					lost = survey->reports[run].matches[other][scheme].first_wins;
					survey->games_won[scheme] +=
						survey->reports[run].matches[other][scheme].second_wins;
				}

				survey->games_lost[scheme] += lost;
			}
		}
	}

	return true;
}
