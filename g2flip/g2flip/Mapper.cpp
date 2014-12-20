#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include "flip.h"

uint64_t getDirection(int pos, int direction);

/**
 * from pos/diagonal direction to 64 bits
 * @return
 */
uint64_t* initFilter(){
	int boardSize = 64;
	int directions = 2;	
	int area = sizeof(uint64_t) * boardSize * directions;
	printf("Allocating const memory: %d b\n", area );
	uint64_t* filter = (uint64_t* ) malloc(area);
	memset(filter, 0, area);

	for(int i = 0 ; i < boardSize ; i++){
		for(int z = 0; z < directions ; z++){
			filter[i + z*boardSize] = getDirection(i,z+2);
		}
	}
	return filter;
}

/**
 * Get pattern for a specific location at board 
 * @param pos
 * @param direction
 * @return
 */

uint64_t getDirection(int pos, int direction){
	uint64_t p = 0ULL;
	int x = pos & 7;
	int y = pos >> 3;

	int dx = 0; 
	int dy = 0;
		
	switch(direction){
		case 0: dy=1; break;
		case 1: dx=1; break;
		case 2: dy=1;dx=1; break;
		case 3: dy=1;dx=-1; break;
	}
		
	int rX;
	int rY;		
	for(int z=0;z<2;z++){
		rX = x;
		rY = y;
		while(rX>=0 && rX<8 && rY>=0 && rY<8){
			p |=  1ULL << (rX + 8*rY);
			rX += dx;
			rY += dy;
		}
		dx = -dx;
		dy = -dy;
	}
	return p;
}
