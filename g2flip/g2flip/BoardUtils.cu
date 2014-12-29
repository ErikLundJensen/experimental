#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include "vector_types.h"

#include "flip.h"

char* numbers[] = {"1","2","3","4","5","6","7","8" };

/*
char* testBoard[] = {	"X    X  " , //1		
					    " O   O  " , //2
						"  O  O  " , //3
						"   O O X" , //4
						"    OOOO" , //5
						"XOOOO OX" , //6
						"    OOO " , //7
						"   X X X"  }; //8
*/
/*
char* testBoard[] = {	"       X" , //1		
					    " OOOOOXX" , //2
						"XXXXOOXX" , //3
						"XOOOXOXX" , //4
						"XOOOOOXX" , //6
						"XOOOOOXX" , //5
						"     O  " , //7
						"        "  }; //8
*/
char* testBoard[] = { "        ", //1		
					  "        ", //2
					  "        ", //3
					  "   OX   ", //4
					  "   XO   ", //5
					  "        ", //6
					  "        ", //7
					  "        " }; //8


/*
char* testBoard[] = { "        ", //1		
					  "        ", //2
					  "  XO X  ", //3
					  " XXXXXO ", //4
					  "XOXXXXX ", //5
					  "O O     ", //6
					  "        ", //7
					  "        " }; //8
*/

uint64_t getBoard(char* board, char c);
char* toBoard(ulonglong2 b, int labels, int color, char* board);
		
void setupPosition(ulonglong2* p, int index){		
	char board[100];
	board[0] = 0;
	for(int i=0; i<8; i++){
		strcat_s(board, testBoard[i]);
	}

	p[index].x = getBoard(board, 'X');
	p[index].y = getBoard(board, 'O');

	p[index].x |= 1ULL << (2 + 3*8);
	p[index].y |= 1ULL << (2 + 3*8);

}


void initPositions(ulonglong2* p, int numberOfPositions){	
	for(int i = 0; i< numberOfPositions; i++){
		setupPosition(p, i);
	}
	char board[500];
	board[0] = 0;
	//printf("Board\n%s\n", toBoard(p[0], 1, 0, board));
}

uint64_t getBoard(char* board, char c){
	uint64_t b = 0ULL;	
	for(int i=0; i<64 ;i++){			
		if (board[i]==c)
			b |= 1ULL << i;
	}
	return b;		
}

int getRow(char* s, char c){
	int r = 0;
	int bit = 1;
	for(int i = 0 ; i < 8 ; i++){
		if (s[i]==c){
			r |= bit;
		}
		bit <<= 1;
	}
	return r;
}

char* toBoard(ulonglong2 b, int labels, int color, char* board){
	if (labels){
		strcat(board, " ABCDEFGH\n");
		strcat(board, " 12345678");
	}
	char* me = "X";
	char* opp = "O";

	if(color==1){
		me = "O";
		opp = "X";
	}

	uint64_t black = b.x;
	uint64_t white = b.y;

	for(int i=0; i<64 ;i++){
		uint64_t p = 1ULL << i;

		if ( (i%8) == 0 && labels){
			strcat(board, "\n");
			strcat(board, numbers[i/8]);
		}
		if ( (black&p) != 0 && (white&p) != 0) 
			strcat(board, "*");
		else if ( (black&p) != 0)
			strcat(board,me);
		else if ( (white&p) != 0)
			strcat(board,opp);
		else
			strcat(board," ");
	}
	if (labels)
		strcat(board,"\n");
		
	return board;
}

char* toBoard_Pattern(uint64_t b, int labels, char* board){
	if (labels){
		strcat(board, " ABCDEFGH\n");
		strcat(board, " 12345678");
	}	

	for(int i=0; i<64 ;i++){
		uint64_t p = 1ULL << i;

		if ( (i%8) == 0 && labels){
			strcat(board, "\n");
			strcat(board, numbers[i/8]);
		}
		if ( (b&p) != 0) 
			strcat(board, "f");		
		else
			strcat(board," ");
	}
	if (labels)
		strcat(board,"\n");
		
	return board;
}

char* toRow(unsigned int r, int color, char row[]){	
	for(int i = 0 ; i < 8; i++){
		if ((r & (1u << i)) !=0) {
			if (color){
				strcat(row, "O");
			}else{
				strcat(row, "X");
			}
		}else{
			strcat(row, " ");
		}
	}
	return row;
}

char* formatTranscript(int* transcript, int labels, int color, char* board){
	if (labels){
		strcat(board, "  ------------------------\n");
		strcat(board, " |A |B |C |D |E |F |G |H |");
	}

	for (int i = 0; i < 64; i++){
		if ((i % 8) == 0 && labels){
			strcat(board, "\n");
			strcat(board, " |--+--+--+--+--+--+--+--|\n");
			strcat(board, numbers[i / 8]);			
			strcat(board, "|");
		}
		char buffer[50];
		sprintf(buffer, "%2d|", transcript[i]);
		strcat(board, buffer);		
	}
	if (labels){
		strcat(board, "\n");
		strcat(board, " -------------------------\n");
	}
	return board;
}