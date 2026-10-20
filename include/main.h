#ifndef MAIN_H
#define MAIN_H

#include <stdbool.h>

// Function declarations go here
bool place_mark(int row, int col, char player_mark);
bool detect_win(char player_mark);
bool detect_current_win(void);
bool board_is_full(void);
bool detect_draw(void);

#endif // MAIN_H
