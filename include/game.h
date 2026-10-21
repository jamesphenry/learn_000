#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <stddef.h>

/* The most moves a position may have, and the most numbers a model may take
 * in. */
#define GAME_MAX_MOVES 64
#define GAME_MAX_FEATURES 64

/* How many moves a game may take, how much room the state of a position
 * takes, and how many positions a game offers to look at before a network
 * has learned anything. */
#define GAME_MAX_HISTORY 128
#define GAME_STATE_BYTES 256
#define GAME_MAX_SAMPLES 128

/*
 * A move is an index. What it means is the game's own business: in
 * tic-tac-toe it is a cell, in Nim it is a heap and how many stones to take
 * from it.
 */
typedef int game_move_t;

/*
 * How a game ended, in terms of the two players and the order they went in
 * the game. A game that has not finished has no outcome yet.
 */
typedef enum {
	GAME_DRAW,
	GAME_FIRST_WINS,
	GAME_SECOND_WINS
} game_outcome_t;

/*
 * The state of a position: a blob of bytes the game owns and understands.
 * The union keeps it lined up, whatever the game chooses to put in it.
 */
typedef union {
	unsigned char bytes[GAME_STATE_BYTES];
	double align_double;
	void *align_pointer;
	long double align_long_double;
} game_state_t;

/*
 * A position in a game: the state, and the moves that may be played in it.
 *
 * The framework copies positions about and never looks inside the state.
 */
typedef struct {
	int n_moves;                     /* how many moves may be played */
	game_move_t moves[GAME_MAX_MOVES];
	game_state_t state;
} game_position_t;

/*
 * A game, as the framework sees it.
 *
 * The features of a position are given from the point of view of the player
 * to move, and the worth of a position is in the same terms, so that a model
 * learns one thing about the game rather than two.
 */
typedef struct {
	/* how many numbers the model takes */
	int n_features;

	/* start a game */
	void (*start)(game_position_t *position);

	/* play a move; false when the move was not allowed */
	bool (*play)(game_position_t *position, game_move_t move);

	/* the numbers the model takes */
	void (*features)(const game_position_t *position, double *out);

	/* whether the game is over */
	bool (*over)(const game_position_t *position);

	/* which player is to move: zero for the one who started the game */
	int (*to_move)(const game_position_t *position);

	/* how the game ended */
	game_outcome_t (*outcome)(const game_position_t *position);

	/*
	 * A few positions to look at before a network has learned anything.
	 * It fills in as many as it can and says how many it gave.
	 */
	int (*samples)(game_position_t *out, int wanted);

	/*
	 * Every position the game can reach, so that two models can be
	 * measured on the same positions. It fills in as many as it can and
	 * says how many it gave.
	 *
	 * A game that cannot say which positions there are leaves this null,
	 * and is measured on the positions its models happen to visit.
	 */
	int (*positions)(game_position_t *out, int wanted);
} game_t;

/*
 * The perfect player for a game: what a position is worth, what move it
 * would play, and how a game that has been played ended.
 *
 * The perfect player may need room of its own to remember what it has worked
 * out, so it is given some by oracle_start and gives it back at
 * oracle_stop. The state belongs to whoever holds the oracle.
 *
 * A perfect player that needs no memory of its own leaves make_state null,
 * and is then asked with a null state.
 */
typedef struct {
	/* the perfect player's own memory */
	void *state;

	/*
	 * The worth of a position to the player to move, if both play
	 * perfectly: one for a win, a half for a draw, nought for a loss.
	 */
	double (*worth)(void *state, const game_position_t *position);

	/* the move the perfect player would play, or -1 when the game is over */
	game_move_t (*choose)(void *state, const game_position_t *position);

	/* how the game ended, according to the perfect player */
	game_outcome_t (*outcome)(void *state, const game_position_t *final);

	/* how the perfect player gets the memory it needs, and how it lets go
	 * of it */
	void *(*make_state)(void);
	void (*free_state)(void *state);
} oracle_t;

/*
 * A game being played, with every move that was offered.
 */
typedef struct {
	game_position_t position;               /* where the game ended up */
	game_move_t offered[GAME_MAX_HISTORY];  /* every move offered, allowed or not */
	int n_offered;
	int n_played;                          /* the moves that were played */
} game_trace_t;

/*
 * A callback that chooses a move in a position. It returns -1 when it
 * cannot choose one.
 */
typedef game_move_t (*game_chooser_t)(void *context, const game_position_t *position);

/* Empties a game, so that a part of it a game does not fill in is empty
 * rather than whatever was there before. */
void game_init(game_t *game);

/* Gives the perfect player the memory it needs. */
bool oracle_start(oracle_t *oracle);

/* Takes the memory back again. */
void oracle_stop(oracle_t *oracle);

/*
 * True when the perfect player can be asked questions: it needs no memory of
 * its own, or it has been given some.
 */
bool oracle_ready(const oracle_t *oracle);

/*
 * Plays a game, asking each side for its moves in turn, and records every
 * move that was offered.
 *
 * The game ends when it is over, when a side cannot choose a move, or when
 * a move is offered that is not allowed.
 */
bool game_play(const game_t *game, game_chooser_t first, void *first_context,
               game_chooser_t second, void *second_context, game_trace_t *trace);

/*
 * Plays the moves of a trace again from the start, and gives each position
 * as it was when the move was played, along with the position the game
 * ended up in. Says how many positions there were.
 */
int game_replay(const game_t *game, const game_trace_t *trace,
                game_position_t positions[GAME_MAX_HISTORY], game_position_t *final);

/*
 * The worth of a result for a player: one if they won, a half if the game
 * was drawn and nought if they lost. The player is zero for the one who
 * started the game and one for the other.
 */
double game_worth_of_outcome(game_outcome_t outcome, int player);

#endif // GAME_H
