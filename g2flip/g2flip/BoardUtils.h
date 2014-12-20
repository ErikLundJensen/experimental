#ifndef BOARD_UTILS_H
#define BOARD_UTILS_H

#include "FlipTest.h"
#include "Flip.h"

void initPositions(uint64_t* p,int numberOfPositions);

uint64_t getBoard(char* board, char c);
int getRow(char* s, char c);
char* toBoard(uint64_t* b, int labels, int color, char* board);
char* toRow(unsigned int r, int color, char* row);
char* toBoard_Pattern(uint64_t b, int labels, char* board);

#endif