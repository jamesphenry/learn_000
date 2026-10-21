#ifndef RNG_H
#define RNG_H

/*
 * A small generator whose state lives in the struct. Every game can have
 * its own, so games running side by side do not disturb each other, and a
 * run with the same seed is repeated exactly.
 */
typedef struct {
	unsigned long long state;
} rng_t;

void rng_seed(rng_t *rng, unsigned long long seed);
double rng_next_double(rng_t *rng);
int rng_next_int(rng_t *rng, int limit);

#endif // RNG_H
