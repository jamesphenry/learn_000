#include "rng.h"

/* xorshift64*, which needs nothing but a shift and a multiply. */
void rng_seed(rng_t *rng, unsigned long long seed)
{
	/* a state of zero would stay zero for ever */
	rng->state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

double rng_next_double(rng_t *rng)
{
	unsigned long long x = rng->state;

	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	rng->state = x;

	/* the top 53 bits, which is what a double holds exactly */
	return (double)(x * 0x2545F4914F6CDD1DULL >> 11) * (1.0 / 9007199254740992.0);
}

int rng_next_int(rng_t *rng, int limit)
{
	if (limit <= 0) return 0;

	return (int)(rng_next_double(rng) * (double)limit);
}
