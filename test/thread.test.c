#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test-runner.h"
#include <stdbool.h>
#include <pthread.h>

#include "../src/board.c"

#define THREAD_COUNT 16
#define ROUNDS 50

struct job {
	board_t board;
	board_cell_t cell;
	int ok;
};

static void *play_a_game(void *arg)
{
	struct job *job = arg;

	board_init(&job->board);
	place_mark(&job->board, job->cell, 'X');

	/* My board must hold exactly one mark, and it must be mine. */
	int marks = 0;
	for (int cell = 0; cell < CELL_COUNT; cell++) {
		if (job->board.cells[cell] != VALUE_EMPTY)
			marks++;
	}

	job->ok = (marks == 1 && job->board.cells[job->cell] == VALUE_X);

	return NULL;
}

static char *test_games_run_in_parallel(void)
{
	struct job jobs[THREAD_COUNT];
	int all_ok = 1;

	for (int round = 0; round < ROUNDS && all_ok; round++) {
		pthread_t threads[THREAD_COUNT];

		for (int i = 0; i < THREAD_COUNT; i++) {
			jobs[i].cell = (board_cell_t)(i % CELL_COUNT);
			jobs[i].ok = 0;
			pthread_create(&threads[i], NULL, play_a_game, &jobs[i]);
		}

		for (int i = 0; i < THREAD_COUNT; i++) {
			pthread_join(threads[i], NULL);
			if (!jobs[i].ok)
				all_ok = 0;
		}
	}

	_it_should("run every game in parallel without interference", all_ok);

	return NULL;
}

static char *run_tests()
{
	_run_test(test_games_run_in_parallel);

	return NULL;
}
