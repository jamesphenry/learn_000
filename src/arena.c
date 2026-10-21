#include <string.h>
#include "arena.h"
#include "evaluate.h"
#include "game.h"
#include "minimax.h"
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
static bool arena_build(network_t *network, const arena_settings_t *settings,
                        init_scheme_t scheme, rng_t *rng)
{
	network_init(network);

	if (!network_add_layer(network, CELL_COUNT, settings->hidden_neurons, ACTIVATION_RELU))
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
static int arena_learn_from_the_truth(trainer_t *trainer, policy_t *policy, int games)
{
	game_t game;
	board_t positions[CELL_COUNT];
	minimax_t oracle;
	int learned = 0;

	minimax_init(&oracle);

	for (int i = 0; i < games; i++) {
		game_play(&game, policy, policy);
		game_positions(&game, positions);

		for (int step = 0; step < game.moves; step++) {
			if (!trainer_train_position(trainer, &positions[step],
			                            minimax_x_win_chance(&oracle, &positions[step])))
				return learned;
		}

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
static void arena_measure(const arena_settings_t *settings, const network_t *network,
                           arena_look_t *look)
{
	memset(look, 0, sizeof(*look));

	evaluate_network(network, settings->evaluation_seed, settings->evaluation_games,
	                 &look->evaluation);
	evaluate_skill(network, settings->evaluation_seed, settings->evaluation_games,
	               POLICY_RANDOM_TEMPERATURE, &look->skill);
	evaluate_skill(network, settings->evaluation_seed, settings->evaluation_games,
	               0.0, &look->own_skill);
	evaluate_against_perfect(network, settings->evaluation_seed,
	                         settings->evaluation_games, &look->perfect);
}

/*
 * Builds one model, measures it, teaches it, and measures it again. The
 * games it learns from and the games it is measured with are apart from one
 * another, so the measurement is on games the model has not seen.
 */
static bool arena_train_one(const arena_settings_t *settings, init_scheme_t scheme,
                             rng_t *rng, network_t *network, arena_result_t *result)
{
	policy_t policy;
	trainer_t trainer;
	arena_look_t before;
	arena_look_t after;
	int trained;

	if (!arena_build(network, settings, scheme, rng)) return false;

	/* what the model is like before it has learned anything */
	arena_measure(settings, network, &before);
	evaluate_initialisation(network, &result->initialisation);

	/* and then it learns */
	trainer_init(&trainer, network, settings->learning_rate);
	policy_init(&policy, network, settings->training_seed, settings->temperature);

	if (settings->learning == ARENA_LEARN_TRUTH)
		trained = arena_learn_from_the_truth(&trainer, &policy, settings->training_games);
	else
		trained = trainer_run(&trainer, &policy, settings->training_games);

	/* and what it is like afterwards, on games it has not seen */
	arena_measure(settings, network, &after);

	result->win_rate_before = 100.0 * before.evaluation.wins / before.evaluation.games;
	result->win_rate_after = 100.0 * after.evaluation.wins / after.evaluation.games;
	result->brier_before = before.evaluation.brier_score;
	result->brier_after = after.evaluation.brier_score;
	result->truth_before = before.evaluation.truth_error;
	result->truth_after = after.evaluation.truth_error;
	result->value_lost_before = before.skill.moves > 0 ?
		before.skill.value_lost / before.skill.moves : 0.0;
	result->value_lost_after = after.skill.moves > 0 ?
		after.skill.value_lost / after.skill.moves : 0.0;
	result->value_lost_own_before = before.own_skill.moves > 0 ?
		before.own_skill.value_lost / before.own_skill.moves : 0.0;
	result->value_lost_own_after = after.own_skill.moves > 0 ?
		after.own_skill.value_lost / after.own_skill.moves : 0.0;
	result->perfect_before = before.perfect;
	result->perfect_after = after.perfect;
	result->trained_games = trained;

	return true;
}

bool arena_run(const arena_settings_t *settings, init_scheme_t scheme,
               arena_result_t *result)
{
	network_t network;
	rng_t rng;

	if (settings == NULL || result == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(result, 0, sizeof(*result));

	rng_seed(&rng, settings->training_seed);

	return arena_train_one(settings, scheme, &rng, &network, result);
}

bool arena_run_all(const arena_settings_t *settings, arena_report_t *report)
{
	network_t networks[ARENA_SCHEMES];
	rng_t rng;

	if (settings == NULL || report == NULL) return false;
	if (settings->training_games < 0 || settings->evaluation_games < 1) return false;
	if (settings->match_games < 1) return false;
	if (settings->hidden_neurons < 1) return false;

	memset(report, 0, sizeof(*report));
	report->training_seed = settings->training_seed;

	rng_seed(&rng, settings->training_seed);

	/* every model is given the same conditions, and the only thing that
	 * differs between them is how they were started off */
	for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
		if (!arena_train_one(settings, (init_scheme_t)scheme, &rng, &networks[scheme],
		                     &report->results[scheme]))
			return false;
	}

	/* and then they play each other, each of them taking one colour and
	 * then the other */
	for (int first = 0; first < ARENA_SCHEMES; first++) {
		for (int second = first + 1; second < ARENA_SCHEMES; second++) {
			if (!evaluate_against(&networks[first], &networks[second],
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

bool arena_survey(const arena_settings_t *settings, int runs, arena_survey_t *survey)
{
	if (settings == NULL || survey == NULL) return false;
	if (runs < 1 || runs > ARENA_MAX_RUNS) return false;

	memset(survey, 0, sizeof(*survey));

	for (int run = 0; run < runs; run++) {
		arena_settings_t one = *settings;
		int winner;

		/* each run has a seed of its own, so the models are started off
		 * differently and learn from different games */
		one.training_seed = settings->training_seed + (unsigned long long)run;

		if (!arena_run_all(&one, &survey->reports[run])) return false;

		survey->runs++;

		winner = arena_winner_of_the_round_robin(&survey->reports[run]);

		if (winner >= 0) survey->round_robin_wins[winner]++;

		for (int scheme = 0; scheme < ARENA_SCHEMES; scheme++) {
			int other;

			/* how the model did against the perfect player */
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
