#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>

// The cells of the board. Empty cells hold '1' to '9', taken cells hold 'X' or 'O'.
extern char board[3][3];

// Keeps track of whose turn it is ('X' or 'O').
extern char current_player;

void print_board(void);
bool place_mark(int row, int col, char player_mark);
bool detect_win(char player_mark);
bool detect_current_win(void);
bool board_is_full(void);
bool detect_draw(void);

#endif // BOARD_H
