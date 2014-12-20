// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include "mapper.h"
#include "BoardUtils.h"

#define __device__
#define __inline__
#define __global__

#define __min(a,b) a<b ? a : b
#define __max(a,b) a>b ? a : b

#define threadIdx 0

int __ffs(uint64_t location){
	uint64_t v = 1;
	for(int i=0; i<64; i++){
		if (location == v){
			return i;
		}
		v <<= 1;
	}
	return 0;
}

unsigned int __brev(unsigned int s){
	unsigned v = 0;	
	for(int i = 0; i< 32; i++){
		if ((1u << i) & s){
			v |= 1u << (31-i);
		}
	}
	return v;
}



// TODO: implementer øvige flip..() metoder, 
const int FILTER_AREA = sizeof(uint64_t) * 64 * 2;



__device__ __inline__ unsigned int flipRightTest(int x, unsigned int me, unsigned int opp){	 	
	unsigned int a = x+1;
	unsigned int c = opp >> a;
	c++;
	c &= (me >> a);
	c==0 ? 0 : c--;
	c <<= a;
	return c;
}		
	
__device__ __inline__ unsigned int flipLeftTest(int x, unsigned int me, unsigned int opp){
	return __brev(flipRightTest(7-x, __brev(me) >> 24 , __brev(opp) >> 24)) >> 24;	
}

__device__	__inline__ unsigned int flipHorizontalTest(uint64_t me, uint64_t opp, int x, int yHigh){    
	unsigned int lineMe = (me >> yHigh) & 255 ;
	unsigned int lineOpp = (opp >> yHigh) & 255;
				
	return flipLeftTest(x, lineMe, lineOpp) | 		 
		   flipRightTest(x, lineMe, lineOpp);
}

__device__	__inline__ uint64_t horizontalPatternTest(int y, int xorPattern) {
	return ((uint64_t) xorPattern) << (y<<3);
}

__device__	__inline__ unsigned int flipVerticalTest(uint64_t me, uint64_t opp, int x, int y) {
	unsigned int lineMe = 0U;
	unsigned int lineOpp = 0U;
				
	uint64_t pos = 1ULL << x;
	uint64_t z= 1ULL;  // 7*8 = 56 inst
	do{			
		lineMe  |= (unsigned int) __min(z, (me & pos));
		lineOpp |= (unsigned int) __min(z, (opp & pos));
		pos <<=8;
		z <<= 1;
	}while(z<=128);

	return flipLeftTest(y, lineMe, lineOpp) | 		 
		   flipRightTest(y, lineMe, lineOpp);		
}

__device__	__inline__ uint64_t verticalPatternTest(int x, unsigned int xorPattern) {
	int pos = x;
	uint64_t pattern = 0ULL;
		
	int z = 1;	
	do{			
		pattern |= ((uint64_t) (xorPattern & z)) << pos;
		pos += 7;
		z <<= 1;
	}while(z<=128);

	return pattern;
}

	// DEBUG
	// char board[200];
	// char row[100];	
	// board[0] = 0;
	// uint64_t pos[2];
	// pos[0] = me2;
	// pos[1] = opp2;
	// printf("shifted\n %s\n",toBoard(pos, true, 0, board));
	// END OF DEBUG out

__device__	__inline__ uint64_t flipDiagonalUpTest(uint64_t me, uint64_t opp, int x, int y) {
	unsigned short dnr = x+y;
	unsigned short over = __max(7u, dnr);
	unsigned short under = __min(7u, dnr);
	uint64_t me2 =  me  >> ((over - 7 ) << 3 );
	me2 <<= ((7-under) << 3 );
	uint64_t opp2 =  opp  >> ((over - 7 ) << 3 );
	opp2 <<= ((7-under) << 3 );

	uint64_t i = 128ULL;	
	uint64_t s = 1ULL << 7;	
	unsigned int lineMe = 0u;
	unsigned int lineOpp = 0u;

	for(short a = 0; a<8; a++){				
		lineMe  |= (unsigned int) (__min( (me2 & s), i));
		lineOpp |= (unsigned int) (__min( (opp2 & s), i));
		i >>= 1;
		s <<=7;			
	}

	unsigned int flipped = flipLeftTest(x, lineMe, lineOpp) | 		 
						   flipRightTest(x, lineMe, lineOpp);
	
	i = 128ULL;	
	s = 1ULL << 7;
	uint64_t pattern = 0ULL;
	for(short a = 0; a<8; a++){
		pattern |= flipped & i ? s : 0ULL;
		i >>= 1;
		s <<= 7;
	}
	
	pattern <<= ((over - 7 ) << 3 );
	pattern >>= ((7-under) << 3 );

	return pattern;
}


__device__	__inline__ uint64_t flipDiagonalDownTest(uint64_t me, uint64_t opp, int x, int y) {
	unsigned short dnr = x + 7 -y;
	unsigned short over = __max(7u, dnr);
	unsigned short under = __min(7u, dnr);
	uint64_t me2 =  me  << ((over - 7 ) << 3 );
	me2 >>= ((7-under) << 3 );
	uint64_t opp2 =  opp  << ((over - 7 ) << 3 );
	opp2 >>= ((7-under) << 3 );

	uint64_t i = 1ULL;	
	uint64_t s = 1ULL;	
	unsigned int lineMe = 0u;
	unsigned int lineOpp = 0u;

	for(short a = 0; a<8; a++){				
		lineMe  |= (unsigned int) (__min( (me2 & s), i));
		lineOpp |= (unsigned int) (__min( (opp2 & s), i));
		i <<= 1;
		s <<=9;			
	}

	unsigned int flipped = flipLeftTest(x, lineMe, lineOpp) | 		 
						   flipRightTest(x, lineMe, lineOpp);
	
	i = 1ULL;	
	s = 1ULL;
	uint64_t pattern = 0ULL;
	for(short a = 0; a<8; a++){
		pattern |= flipped & i ? s : 0ULL;
		i <<= 1;
		s <<= 9;
	}
	
	pattern >>= ((over - 7 ) << 3 );
	pattern <<= ((7-under) << 3 );

	return pattern;
}


/**
* Location of move is stored as "black and white in same field"
*/
__global__ void optionTest(uint64_t* positions, uint64_t* result, uint64_t* filter)
{	
	uint64_t boardMe = positions[threadIdx * 2];
	uint64_t boardOpp = positions[threadIdx *2 + 1];

	uint64_t location = boardMe & boardOpp;
	boardMe ^= location;
	boardOpp ^= location;

	int bitNr = __ffs(location);
	int x = bitNr & 7;
	int y = bitNr >> 3;
	int yHigh = bitNr & (8+16+32);

	uint64_t xorPattern = 0ULL;

	xorPattern |= horizontalPatternTest(y, flipHorizontalTest(boardMe, boardOpp, x, yHigh));
	xorPattern |= verticalPatternTest(x, flipVerticalTest(boardMe, boardOpp, x, y));
	xorPattern |= flipDiagonalUpTest(boardMe, boardOpp, x, y);
	xorPattern |= flipDiagonalDownTest(boardMe, boardOpp, x, y);

	if (xorPattern){		
		result[threadIdx * 2] = boardMe ^ (xorPattern | 1ULL<<bitNr);			
		result[threadIdx * 2 + 1] = boardOpp ^ xorPattern;						
	}
	return;
}


int testKernelTest()
{	
	int threads = 32;
	int grid = 1;
    int numberOfPositions = 32;	

	// Host memory
    uint64_t *hPositions = initPositions(numberOfPositions);        
	uint64_t *hFilters = initFilter();
		
    optionTest(hPositions, &hPositions[numberOfPositions*2], hFilters);

	char board[500];
	board[0] = 0;
	printf("Before\n%s\n", toBoard(hPositions, 1, 0, board));
	board[0] = 0;
	printf("After\n%s\n", toBoard(&hPositions[numberOfPositions * 2], 1, 0, board));

	for(int i=0; i < numberOfPositions ; i++){
		printf("%d: I=%lld,%lld R=%lld,%lld \n", i , hPositions[i*2], hPositions[i*2 + 1], hPositions[i*2 + numberOfPositions*2], hPositions[i*2 + numberOfPositions*2 +1]);
	}

    // Clean up memory
    free(hPositions);
	free(hFilters);

	return 0;
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Othello Using CUDA] - Starting...\n");    

    // Use a larger block size for Fermi and above
    //int block_size = 16;    

    int result = testKernelTest();

    exit(result);
}
