#include <stdio.h>
#include "board.h"

char current_player = 'X'; // Keeps track of whose turn it is ('X' or 'O')

char board[3][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'}
};

void print_board(void)
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
    if (player_mark != 'X' && player_mark != 'O') return false;
    if (board[row][col] == 'X' || board[row][col] == 'O') return false;

    board[row][col] = player_mark;
    return true;
}

bool detect_win(char player_mark)
{
	for (int row = 0; row < 3; row++) {
		if (board[row][0] == player_mark && board[row][1] == player_mark &&
		    board[row][2] == player_mark)
			return true;
	}

	for (int col = 0; col < 3; col++) {
		if (board[0][col] == player_mark && board[1][col] == player_mark &&
		    board[2][col] == player_mark)
			return true;
	}

	if (board[0][0] == player_mark && board[1][1] == player_mark && board[2][2] == player_mark)
		return true;

	if (board[0][2] == player_mark && board[1][1] == player_mark && board[2][0] == player_mark)
		return true;

	return false;
}

bool detect_current_win(void)
{
	return detect_win(current_player);
}

bool board_is_full(void)
{
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			if (board[row][col] != 'X' && board[row][col] != 'O')
				return false;
		}
	}

	return true;
}

bool detect_draw(void)
{
	if (!board_is_full()) return false;

	return !detect_win('X') && !detect_win('O');
}
