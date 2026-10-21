#include <string.h>
#include "nim.h"

/*
 * The game of Nim: three heaps of stones, and the players take turns
 * taking any number of stones from one of them. The player who takes the
 * last stone wins, so the player who has no stone left to take has lost.
 *
 * The heaps live in the state of a position, so the framework never has to
 * know what a Nim position looks like.
 */
static nim_t *nim_of(game_position_t *position)
{
	return (nim_t *)position->state.bytes;
}

const nim_t *nim_state_of(const game_position_t *position)
{
	return (const nim_t *)position->state.bytes;
}

int nim_heap_of(game_move_t move)
{
	return move / (NIM_MAX_HEAP + 1);
}

int nim_count_of(game_move_t move)
{
	return move % (NIM_MAX_HEAP + 1);
}

/* The moves that may be played: take any number of stones from any heap. */
static void refresh_moves(game_position_t *position)
{
	const nim_t *nim = nim_state_of(position);
	int count = 0;

	for (int heap = 0; heap < NIM_HEAPS; heap++) {
		for (int take = 1; take <= nim->heaps[heap]; take++) {
			if (count >= GAME_MAX_MOVES) break;

			position->moves[count++] = NIM_MOVE(heap, take);
		}
	}

	position->n_moves = count;
}

static void nim_start(game_position_t *position)
{
	nim_t *nim = nim_of(position);

	nim->heaps[0] = NIM_START_HEAP_0;
	nim->heaps[1] = NIM_START_HEAP_1;
	nim->heaps[2] = NIM_START_HEAP_2;
	nim->turn = 0;

	refresh_moves(position);
}

static bool nim_over(const game_position_t *position)
{
	const nim_t *nim = nim_state_of(position);

	for (int heap = 0; heap < NIM_HEAPS; heap++)
		if (nim->heaps[heap] > 0) return false;

	return true;
}

static bool nim_play(game_position_t *position, game_move_t move)
{
	nim_t *nim = nim_of(position);
	int heap;
	int take;

	if (nim_over(position)) return false;

	heap = nim_heap_of(move);
	take = nim_count_of(move);

	if (heap < 0 || heap >= NIM_HEAPS) return false;
	if (take < 1 || take > nim->heaps[heap]) return false;

	nim->heaps[heap] -= take;
	nim->turn = 1 - nim->turn;

	refresh_moves(position);

	return true;
}

/*
 * The features are the sizes of the heaps, as a fraction of the biggest heap
 * there is, so that they lie between nought and one. The tic-tac-toe board is
 * given to the model in the same range, and a model that is given numbers of
 * very different sizes is not the same experiment twice.
 */
static void nim_features(const game_position_t *position, double *out)
{
	const nim_t *nim = nim_state_of(position);

	for (int heap = 0; heap < NIM_HEAPS; heap++)
		out[heap] = (double)nim->heaps[heap] / (double)NIM_MAX_HEAP;
}

static int nim_to_move(const game_position_t *position)
{
	return nim_state_of(position)->turn;
}

static game_outcome_t nim_outcome(const game_position_t *position)
{
	const nim_t *nim = nim_state_of(position);

	/* the game is over when there are no stones left, and the player to
	 * move has lost it, because they are the one who has no move left */
	if (!nim_over(position)) return GAME_DRAW;

	if (nim->turn == 0) return GAME_SECOND_WINS;

	return GAME_FIRST_WINS;
}

/*
 * A few positions to look at before a network has learned anything: the
 * position the game starts in, and every position one move away from it.
 */
static int nim_samples(game_position_t *out, int wanted)
{
	const int start[NIM_HEAPS] = {
		NIM_START_HEAP_0, NIM_START_HEAP_1, NIM_START_HEAP_2
	};
	int count = 0;

	if (out == NULL || wanted < 1) return 0;

	nim_start(&out[count++]);

	for (int heap = 0; heap < NIM_HEAPS && count < wanted; heap++) {
		for (int take = 1; take <= start[heap] && count < wanted; take++) {
			nim_start(&out[count]);
			nim_play(&out[count], NIM_MOVE(heap, take));
			count++;
		}
	}

	return count;
}

/*
 * Every position the game can reach: the heaps only ever get smaller, so a
 * position is reachable when no heap is bigger than the one the game starts
 * with.
 *
 * The heaps belong to no player, so who is to move makes no difference to
 * what a position is worth, and every position is given with the first
 * player to move.
 */
static int nim_positions(game_position_t *out, int wanted)
{
	int count = 0;

	if (out == NULL || wanted < 1) return 0;

	for (int a = 0; a <= NIM_START_HEAP_0 && count < wanted; a++)
		for (int b = 0; b <= NIM_START_HEAP_1 && count < wanted; b++)
			for (int c = 0; c <= NIM_START_HEAP_2 && count < wanted; c++) {
				nim_t *nim = nim_of(&out[count]);

				nim->heaps[0] = a;
				nim->heaps[1] = b;
				nim->heaps[2] = c;
				nim->turn = 0;

				refresh_moves(&out[count]);

				count++;
			}

	return count;
}

void nim_game(game_t *game)
{
	if (game == NULL) return;

	game_init(game);

	game->n_features = NIM_HEAPS;
	game->start = nim_start;
	game->play = nim_play;
	game->features = nim_features;
	game->over = nim_over;
	game->to_move = nim_to_move;
	game->outcome = nim_outcome;
	game->samples = nim_samples;
	game->positions = nim_positions;
}

/*
 * The perfect player. It knows the one rule of Nim: the exclusive or of the
 * heaps is zero when the position is a loss for the player to move, and any
 * other position can be turned into one by taking stones from the right
 * heap.
 */
int nim_sum(const nim_t *nim)
{
	int value = 0;

	for (int heap = 0; heap < NIM_HEAPS; heap++)
		value ^= nim->heaps[heap];

	return value;
}

static double nim_worth(void *state, const game_position_t *position)
{
	(void)state;

	/* the player to move wins if the exclusive or of the heaps is not
	 * zero */
	return nim_sum(nim_state_of(position)) != 0 ? 1.0 : 0.0;
}

static game_move_t nim_choose(void *state, const game_position_t *position)
{
	const nim_t *nim = nim_state_of(position);
	int xor = nim_sum(nim);

	(void)state;

	if (nim_over(position)) return -1;

	/* the heap to take from is the one whose size, exclusive ored with
	 * the exclusive or of all of them, is smaller than it is */
	for (int heap = 0; heap < NIM_HEAPS; heap++) {
		int left = nim->heaps[heap] ^ xor;

		if (left < nim->heaps[heap])
			return NIM_MOVE(heap, nim->heaps[heap] - left);
	}

	/* the position is a loss for the player to move, and every move is as
	 * good as any other, so one is played anyway: the game goes on until
	 * the stones are gone */
	for (int heap = 0; heap < NIM_HEAPS; heap++)
		if (nim->heaps[heap] > 0) return NIM_MOVE(heap, 1);

	return -1;
}

/*
 * The result, worked out from the exclusive or of the heaps alone, so that
 * it can disagree with the game it is judging.
 */
static game_outcome_t nim_oracle_outcome(void *state, const game_position_t *final)
{
	const nim_t *nim = nim_state_of(final);

	(void)state;

	/* the game is over when there are no stones left in any of the heaps */
	for (int heap = 0; heap < NIM_HEAPS; heap++)
		if (nim->heaps[heap] > 0) return GAME_DRAW;

	/* and the player to move has lost it, because the exclusive or of the
	 * heaps is zero */
	if (nim_sum(nim) != 0) return GAME_DRAW;

	if (nim->turn == 0) return GAME_SECOND_WINS;

	return GAME_FIRST_WINS;
}

void nim_oracle(oracle_t *oracle)
{
	if (oracle == NULL) return;

	oracle->state = NULL;
	oracle->worth = nim_worth;
	oracle->choose = nim_choose;
	oracle->outcome = nim_oracle_outcome;
	oracle->make_state = NULL;
	oracle->free_state = NULL;
}
