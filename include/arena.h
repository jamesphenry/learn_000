#ifndef ARENA_H
#define ARENA_H

#include <stdbool.h>
#include "evaluate.h"
#include "network.h"
#include "rng.h"

/* The three ways a network can be started off. */
typedef enum {
	INIT_ZERO,   /* every weight and bias at zero */
	INIT_RANDOM, /* small random values */
	INIT_MAXIMUM /* every weight and bias at the same value */
} init_scheme_t;

/* How many ways there are to start a network off. */
#define ARENA_SCHEMES 3

/* The two ways a model can be taught. */
typedef enum {
	ARENA_LEARN_TRUTH, /* the perfect player says what each position is worth */
	ARENA_LEARN_SELF   /* the model learns the result of its own games */
} arena_learning_t;

/* The conditions every model in a run is given, so that the only thing that
 * differs between them is how they were started. */
typedef struct {
	int hidden_neurons;      /* how many neurons the hidden layer has */
	int training_games;      /* how many games to learn from */
	int evaluation_games;    /* how many games to measure with */
	int match_games;         /* how many games the models play each other */
	unsigned long long training_seed;
	unsigned long long evaluation_seed;
	double learning_rate;
	double temperature;      /* how much the policy explores while playing */
	double init_scale;       /* how big the random values are */
	double init_value;       /* the value every weight and bias starts at */
	arena_learning_t learning; /* who teaches the model */
} arena_settings_t;

/* What a model did. */
typedef struct {
	/* what it looked like before it learned anything */
	initialisation_t initialisation;

	double win_rate_before;
	double win_rate_after;
	double brier_before;
	double brier_after;
	double truth_before;
	double truth_after;
	/* the worth the model gave away, on each move it played, against an
	 * opponent that plays at random */
	double value_lost_before;
	double value_lost_after;
	/* and the same in the games it plays against itself, which are the
	 * positions it knows best */
	double value_lost_own_before;
	double value_lost_own_after;
	/* and how it did against the perfect player */
	perfect_result_t perfect_before;
	perfect_result_t perfect_after;
	int trained_games;
} arena_result_t;

/*
 * What all the models did, and how they fare against each other.
 */
typedef struct {
	/* the seed the models were trained with */
	unsigned long long training_seed;

	/* what each model did on its own, in the order of the ways of
	 * starting off */
	arena_result_t results[ARENA_SCHEMES];

	/* how the models fare against each other. The match at [i][j] is the
	 * one between the models started off in the ways i and j, and only the
	 * upper half is filled in: a model does not play itself. */
	match_result_t matches[ARENA_SCHEMES][ARENA_SCHEMES];
} arena_report_t;

/* How many runs a survey holds. */
#define ARENA_MAX_RUNS 10

/*
 * What the models did over a number of runs, each with its own seed.
 */
typedef struct {
	int runs;

	/* what each run found */
	arena_report_t reports[ARENA_MAX_RUNS];

	/* how many of the runs each model won the round robin in */
	int round_robin_wins[ARENA_SCHEMES];

	/* and the games it won and lost against the other models, over all the
	 * runs */
	int games_won[ARENA_SCHEMES];
	int games_lost[ARENA_SCHEMES];

	/* and the games each model drew and lost against the perfect player,
	 * over all the runs */
	int perfect_draws[ARENA_SCHEMES];
	int perfect_losses[ARENA_SCHEMES];
} arena_survey_t;

/* The conditions a run is given, with sensible values for tic-tac-toe. */
void arena_settings_init(arena_settings_t *settings);

/*
 * Builds a network, measures it, trains it, and measures it again. The
 * games it learns from and the games it is measured with are apart from one
 * another, so the measurement is on games the model has not seen.
 */
bool arena_run(const arena_settings_t *settings, init_scheme_t scheme,
               arena_result_t *result);

/*
 * Builds all three models, gives them the same conditions, measures each of
 * them, and then has them play each other.
 */
bool arena_run_all(const arena_settings_t *settings, arena_report_t *report);

/*
 * Runs the arena several times, each time with a seed of its own, and says
 * what the models did over all of them. A ranking that holds from one seed
 * to the next is worth more than one that was measured once.
 */
bool arena_survey(const arena_settings_t *settings, int runs, arena_survey_t *survey);

/* The three stages of a duel. */
typedef enum {
	DUEL_BEFORE,    /* before either of them has learned anything */
	DUEL_OWN_GAMES, /* after each has learned from its own games */
	DUEL_TOGETHER,  /* after they have learned against each other */
	ARENA_DUEL_STAGES
} arena_stage_t;

/* What the two models were like at one stage of a duel. */
typedef struct {
	/* the worth each model gave away, on each move it played: the first
	 * is the one with the first seed */
	double value_lost[2];
	/* how far each was from the truth */
	double truth[2];
	/* and how each did against the perfect player */
	int perfect_draws[2];
	int perfect_losses[2];

	/* and how they fared against each other */
	match_result_t meeting;

	/* and what each of them looked like when it started off */
	initialisation_t initialisation[2];
} arena_stage_result_t;

/*
 * Two models, each with a seed of its own, and how they fare against each
 * other: before either of them has learned anything, after each has
 * learned from its own games, and after they have learned against each
 * other.
 *
 * Neither model is taught by the perfect player here. Each learns from the
 * result of the games it plays, which is what a model has to do when there
 * is nobody to tell it the answer.
 */
typedef struct {
	/* what happened at each stage */
	arena_stage_result_t stages[ARENA_DUEL_STAGES];

	/* the games they played against each other while learning together */
	match_result_t together_games;
} arena_duel_t;

bool arena_duel(const arena_settings_t *settings, unsigned long long first_seed,
                unsigned long long second_seed, arena_duel_t *duel);

/* How many models a league holds. */
#define ARENA_MAX_MODELS 128

/* How many opponents are held back from the league for every model to be
 * measured against: one of each kind, trained to play in two different
 * ways. */
#define ARENA_HELD_OUT 4

/* How a model in a league is taught. */
typedef enum {
	LEAGUE_SELF,  /* from the result of the games it plays against itself */
	LEAGUE_TRUTH  /* by the perfect player, which says what a position is worth */
} league_teacher_t;

/* One model in a league. */
typedef struct {
	unsigned long long seed;  /* the seed it was started off with */
	league_teacher_t teacher;  /* and how it was taught */

	/* how well it learned from its own games: the worth it gave away in the
	 * positions it visited while learning */
	double training_value_lost;

	/* how far it is from the truth, over the positions it reaches against
	 * an opponent that plays at random */
	double truth;
	double value_lost;

	/* the games it drew and lost against the perfect player */
	int perfect_draws;
	int perfect_losses;

	/* its record in the league, against every other model */
	int wins;
	int draws;
	int losses;
	int points;         /* three for a win, one for a draw */

	/* and the same, split by the kind of opponent in the league */
	int same_wins, same_draws, same_points;
	int other_wins, other_draws, other_points;

	/* and its record against the opponents held back from the league,
	 * which are the same for every model: one of its own kind and one of
	 * the other kind, each trained to play in two different ways */
	int held_same_points;   /* against the held back models of its own kind */
	int held_other_points;  /* and against the held back models of the other kind */
	int held_same_games;    /* how many games that was */
	int held_other_games;
} league_model_t;

/* A whole league. */
typedef struct {
	int models;
	league_model_t model[ARENA_MAX_MODELS];

	/* the model with the most points */
	int champion;

	/* and the seeds of the opponents that were held back from the league,
	 * which are not the seeds of any model in it */
	unsigned long long held_seed[ARENA_HELD_OUT];

	/* and the models in three different orders, because they say three
	 * different things:
	 *
	 *   training   how well it learned from its own games
	 *   league     how well it did against every other model
	 *   general    how well it did against held back models of another kind
	 */
	int ranking_training[ARENA_MAX_MODELS];
	int ranking_league[ARENA_MAX_MODELS];
	int ranking_general[ARENA_MAX_MODELS];
} arena_league_t;

/*
 * Builds a number of models, each with a seed of its own, and plays a
 * league with them: every model plays every other one, and the points they
 * earn are added up.
 *
 * Half of them are taught by the perfect player and half learn from their
 * own games, so that the league holds opponents of two different kinds.
 *
 * A league rather than a knockout, because a knockout of models that draw
 * most of their games against each other is decided by luck.
 *
 * Four models are held back from the league and never play in it, and every
 * model is measured against them. They are the same for every model, which
 * is what makes the third ranking a fair comparison: the points a model
 * earns against the other kind in the league depend on which models happened
 * to be in the league and how strong they were, while the points it earns
 * against a held back opponent depend only on the model and the opponent.
 */
bool arena_league(const arena_settings_t *settings, int models,
                  arena_league_t *league);


#endif // ARENA_H
