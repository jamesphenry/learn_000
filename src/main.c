#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include "board.h"

int greet(char *message)
{
	return puts(message);
}

#ifndef TEST
#define UNUSED(x) (void)(x)

int main(int argc, char *argv[], char *env[]) {
	UNUSED(argc);
	UNUSED(argv);
	UNUSED(env);

	greet("Hello, World!");
	print_board();

	return EXIT_SUCCESS;
}

#endif
