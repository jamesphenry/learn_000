#include <stdio.h>
#include <stdlib.h>
#include "main.h"
#include <stdbool.h>

char current_player = 'X'; // Keeps track of whose turn it is ('X' or 'O')

char board[3][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'}
};

int greet(char *message)
{
	return puts(message);
}

void print_board()
{
	for (int i = 0; i < 3; i++) {
    printf(" %c | %c | %c \n", board[i][0], board[i][1], board[i][2]);
    
    // Only print the divider after the first and second rows
    if (i < 2) {
        printf("---+---+---\n");
    }
}
}

bool place_mark(int row, int col, char player_mark) {
    if (row < 0 || row > 2 || col < 0 || col > 2) return false;
    if (board[row][col] == 'X' || board[row][col] == 'O') return false;

    board[row][col] = player_mark;
    return true;
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
