// System includes
#include <stdio.h>
#include "windows.h"
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <vector_types.h>
#include "mapper.h"

#define ITERATIONS 100
// Actual thread size
#define BLOCK_SIZE 256
#define GRID_SIZE 4096
//#define GRID_SIZE 16384
#define DEFAULT_DEPTH 60

#define MAX_STREAMS 2

const uint64_t NotH1H8 = 18374403900871474942ULL;
const uint64_t NotA1A8 = 9187201950435737471ULL;

void initPositions(ulonglong2* p,int numberOfPositions);

uint64_t getBoard(char* board, char c);
int getRow(char* s, char c);
char* toBoard(ulonglong2 b, int labels, int color, char* board);
char* toRow(unsigned int r, int color, char* row);
char* toBoard_Pattern(uint64_t b, int labels, char* board);

#define DO_FLIP 1

// TODO:
// implement node structure
// refactor so takeback is possible
// Count disc differance at end of game. Store value in node. 
// Use alpha-beta pruning
// multi-threaded så begge devices anvendes
// fix fejl ved flere streams
// 

void onErrorExit(char *msg, cudaError_t error, int line){
	if (error != cudaSuccess){
		printf("%s cuda error code %d, line(%d): %s\n", msg, error, line, cudaGetErrorString(error));   
		exit(EXIT_FAILURE); 
	}
}

/************************************************************************************
 *
 *	Utility methods for calculating lines excluding H1-H8 and A1-A8
 *
 ************************************************************************************/
uint64_t getNotH1H8(){
	uint64_t v = 0LL;
	for(int y = 0; y < 8; y++){
		v |= (1LL<< (y*8)); 
	} 
	return ~v;
}

uint64_t getNotA1A8(){
	uint64_t v = 0LL;
	for(int y = 0; y < 8; y++){
		v |= (128LL<< (y*8)); 
	} 
	return ~v;
}

/************************************************************************************
 *
 *	Find legal moves and add them to the board
 *
 ************************************************************************************/
__device__ void getLegalMoves(ulonglong2 *board, int isWhiteToPlay){
	uint64_t me;
	uint64_t op;
	if (isWhiteToPlay){
		me = board[0].y;
		op = board[0].x;
	}else{		
		op = board[0].y;
		me = board[0].x;
	}	

	uint64_t f = 0LL; // legal moves (output)
	uint64_t e;		// empty fields
	uint64_t v;		// temp variable
	f = 0LL;
	e = ~(me | op);
	/////////////////////////////////
	// down
	v = me;
	v <<= 8;		// 1.
	v &= op;
	v <<= 8;		// 2.
	f = e & v;
	v &= op;
	v <<= 8;		// 3.
	f |= (e & v);
	v &= op;
	v <<= 8;		// 4.
	f |= (e & v);
	v &= op;
	v <<= 8;		// 5.
	f |= (e & v);
	v &= op;
	v <<= 8;		// 6.
	f |= (e & v);
	v &= op;
	v <<= 8;		// 7.
	f |= (e & v);
	
	/////////////////////////////////
	// up
	v = me;
	v >>= 8;		// 1.
	v &= op;
	v >>= 8;		// 2.
	f |= (e & v);
	v &= op;
	v >>= 8;		// 3.
	f |= (e & v);
	v &= op;
	v >>= 8;		// 4.
	f |= (e & v);
	v &= op;
	v >>= 8;		// 5.
	f |= (e & v);
	v &= op;
	v >>= 8;		// 6.
	f |= (e & v);
	v &= op;
	v >>= 8;		// 7.
	f |= (e & v);

	/////////////////////////////////
	// right
	v = me;
	v <<= 1;		// 1.
	v &= NotH1H8;
	v &= op;
	v <<= 1;		// 2.
	v &= NotH1H8;
	f |= e & v;
#pragma __unroll
	for(int a = 0; a<5; a++){
		v &= op;
		v <<= 1;		// 3-7
		v &= NotH1H8;
		f |= (e & v);
	}
	
	/////////////////////////////////
	// left
	v = me;
	v >>= 1;		// 1.
	v &= NotA1A8;
	v &= op;
	v >>= 1;		// 2.
	v &= NotA1A8;
	f |= e & v;
#pragma __unroll
	for(int a = 0; a<5; a++){
		v &= op;
		v >>= 1;		// 3-7
		v &= NotA1A8;
		f |= (e & v);
	}

	/////////////////////////////////
	// left-up
	v = me;
	v >>= 9;		// 1.
	v &= NotA1A8;
	v &= op;
	v >>= 9;		// 2.
	v &= NotA1A8;
	f |= (e & v);
#pragma __unroll
	for(int a = 0; a<5; a++){
		v &= op;
		v >>= 9;		// 3-7
		v &= NotA1A8;
		f |= (e & v);
	}
	
	/////////////////////////////////
	// right-up
	v = me;
	v >>= 7;		// 1.
	v &= NotA1A8;
	v &= op;
	v >>= 7;		// 2.
	v &= NotA1A8;
	f |= e & v;
#pragma __unroll
	for(int a = 0; a<5; a++){
		v &= op;
		v >>= 7;		// 3-7
		v &= NotA1A8;
		f |= (e & v);
	}

	/////////////////////////////////
	// left-down
	v = me;
	v <<= 7;		// 1.
	v &= NotA1A8;
	v &= op;
	v <<= 7;		// 2.
	v &= NotA1A8;
	f |= e & v;
#pragma __unroll
	for(int a = 0; a<5; a++){
		v &= op;
		v <<= 7;		// 3-7
		v &= NotA1A8;
		f |= (e & v);
	}

	/////////////////////////////////
	// right-down
	v = me;
	v <<= 9;		// 1.
	v &= NotA1A8;
	v &= op;
	v <<= 9;		// 2.
	v &= NotA1A8;
	f |= e & v;
#pragma __unroll
	for(int a = 0; a<5; a++){
		v &= op;
		v <<= 9;		// 3-7
		v &= NotA1A8;
		f |= (e & v);
	}

	board[0].x |= f;
	board[0].y |= f;
	
	return;
}

/************************************************************************************
 *
 *	Flip methods
 *
 ************************************************************************************/
__device__ __inline__ unsigned int flipRight(int x, unsigned int me, unsigned int opp){	 	
	unsigned int a = x+1;
	unsigned int c = opp >> a;
	c++;
	c &= (me >> a);
	c==0 ? 0 : c--;
	c <<= a;
	return c;
}		
	
__device__ __inline__ unsigned int flipLeft(int x, unsigned int me, unsigned int opp){
	return __brev(flipRight(7-x, __brev(me) >> 24 , __brev(opp) >> 24)) >> 24;	
}

__device__	__inline__ unsigned int flipHorizontal(uint64_t me, uint64_t opp, int x, int yHigh){    
	unsigned int lineMe = (me >> yHigh) & 255 ;
	unsigned int lineOpp = (opp >> yHigh) & 255;
				
	return flipLeft(x, lineMe, lineOpp) | 		 
		   flipRight(x, lineMe, lineOpp);
}

__device__	__inline__ uint64_t horizontalPattern(int y, int xorPattern) {
	return ((uint64_t) xorPattern) << (y<<3);
}

__device__	__inline__ unsigned int flipVertical(uint64_t me, uint64_t opp, int x, int y) {
	unsigned int lineMe = 0U;
	unsigned int lineOpp = 0U;
				
	uint64_t pos = 1ULL << x;
	uint64_t z= 1ULL;  // 7*8 = 56 inst
#pragma unroll	
	do{			
		lineMe  |= (unsigned int) __min(z, (me & pos));
		lineOpp |= (unsigned int) __min(z, (opp & pos));
		pos <<=8;
		z <<= 1;
	}while(z<=128);

	return flipLeft(y, lineMe, lineOpp) | 		 
		   flipRight(y, lineMe, lineOpp);		
}

__device__	__inline__ uint64_t verticalPattern(int x, unsigned int xorPattern) {
	int pos = x;
	uint64_t pattern = 0ULL;
		
	int z = 1;	
#pragma unroll
	do{			
		pattern |= ((uint64_t) (xorPattern & z)) << pos;
		pos += 7;
		z <<= 1;
	}while(z<=128);

	return pattern;
}

__device__	__inline__ uint64_t flipDiagonalUp(uint64_t me, uint64_t opp, int x, int y) {
	unsigned short dnr = x+y;
	unsigned short over = __max(7u, dnr);
	unsigned short under = __min(7u, dnr);
	uint64_t me2 =  me  >> ((over - 7u ) << 3 );
	me2 <<= ((7u - under) << 3 );
	uint64_t opp2 =  opp  >> ((over - 7u ) << 3 );
	opp2 <<= ((7u - under) << 3 );

	uint64_t i = 128ULL;	
	uint64_t s = 1ULL << 7;	
	unsigned int lineMe = 0u;
	unsigned int lineOpp = 0u;

#pragma unroll
	for(short a = 0; a<8; a++){				
		lineMe  |= (unsigned int) (__min( (me2 & s), i));
		lineOpp |= (unsigned int) (__min( (opp2 & s), i));
		i >>= 1;
		s <<=7;			
	}

	unsigned int flipped = flipLeft(x, lineMe, lineOpp) | 		 
						   flipRight(x, lineMe, lineOpp);
	
	i = 128ULL;	
	s = 1ULL << 7;
	uint64_t pattern = 0ULL;
	for(short a = 0; a<8; a++){
		pattern |= flipped & i ? s : 0ULL;
		i >>= 1;
		s <<= 7;
	}
	
	pattern <<= ((over - 7u ) << 3 );
	pattern >>= ((7u - under) << 3 );

	return pattern;
}

__device__	__inline__ uint64_t flipDiagonalDown(uint64_t me, uint64_t opp, int x, int y) {
	unsigned short dnr = x + 7u - y;
	unsigned short over = __max(7u, dnr);
	unsigned short under = __min(7u, dnr);
	uint64_t me2 =  me  << ((over - 7u ) << 3 );
	me2 >>= ((7u - under) << 3 );
	uint64_t opp2 =  opp  << ((over - 7u ) << 3 );
	opp2 >>= ((7u - under) << 3 );

	uint64_t i = 1ULL;	
	uint64_t s = 1ULL;	
	unsigned int lineMe = 0u;
	unsigned int lineOpp = 0u;

#pragma unroll
	for(short a = 0; a<8; a++){				
		lineMe  |= (unsigned int) (__min( (me2 & s), i));
		lineOpp |= (unsigned int) (__min( (opp2 & s), i));
		i <<= 1;
		s <<=9;			
	}

	unsigned int flipped = flipLeft(x, lineMe, lineOpp) | 		 
						   flipRight(x, lineMe, lineOpp);

	unsigned v = 1u;	
	s = 1ULL;
	uint64_t pattern = 0ULL;

#pragma unroll	
	for(short a = 0; a<8; a++){
		unsigned int t = (flipped & v) >> a;
		i = s * t;		
		pattern |= i;
		v <<= 1;
		s <<= 9;
	}
	
	pattern >>= ((over - 7u ) << 3 );
	pattern <<= ((7u - under) << 3 );
	
	return pattern;
}

/************************************************************************************
 *
 * Select option to play. Return bit nr to identify location
 *
 ************************************************************************************/
__device__ __inline__ int selectOption(uint64_t options){

	// TODO: Optimize and do select by pruning etc.
	// Select the middle options of the possible options	
	int numberOfOptions = __popcll(options);
	if (numberOfOptions > 3){
		numberOfOptions >>= 1;
		for (int a=0; a<numberOfOptions; a++){
			int optionDeselected = __ffsll(options);
			options ^= (1ULL << (optionDeselected-1));
		}
	}
	return __ffsll(options);
}

/************************************************************************************
 *
 * Location of move to play is stored as "black and white in same field"
 *
 ************************************************************************************/
__device__ __inline__ int option(ulonglong2* positions, ulonglong2* result, int isWhiteToPlay){
	uint64_t boardMe;
	uint64_t boardOpp;
	
	// TODO: move globalIdx and pos out of this method
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (isWhiteToPlay){		
		boardOpp = positions[globalIdx].x;
		boardMe = positions[globalIdx].y;
	}else{		
		boardOpp = positions[globalIdx].y;
		boardMe = positions[globalIdx].x;
	}

	uint64_t location = boardMe & boardOpp;
	if (location==0ULL) return 0;

	boardMe ^= location;
	boardOpp ^= location;
	
	int bitNr = selectOption(location);
	bitNr--;
	int x = bitNr & 7;
	int y = bitNr >> 3;
	int yHigh = bitNr & (8+16+32);
	
	uint64_t xorPattern = 0ULL;
	
	xorPattern |= horizontalPattern(y, flipHorizontal(boardMe, boardOpp, x, yHigh));
	xorPattern |= verticalPattern(x, flipVertical(boardMe, boardOpp, x, y));
	xorPattern |= flipDiagonalUp(boardMe, boardOpp, x, y);
	xorPattern |= flipDiagonalDown(boardMe, boardOpp, x, y);	


	if (xorPattern){		
		boardMe ^= (xorPattern | 1ULL<<bitNr);		
		boardOpp ^= xorPattern;	
		if (isWhiteToPlay){
			result[globalIdx].x = boardOpp;
			result[globalIdx].y = boardMe;
		}else{
			result[globalIdx].x = boardMe;
			result[globalIdx].y = boardOpp;
		}
		getLegalMoves(&result[globalIdx], isWhiteToPlay ^ 1);
	}

	return bitNr;
}

/************************************************************************************
 *
 * Location of move to play is stored as "black and white in same field"
 * TODO: count disc differance at end of game. Store value in node. Use alpha-beta pruning
 *
 ************************************************************************************/
__global__ void play(ulonglong2* positions, ulonglong2* result, int isWhiteToPlay, int depth)
{	   
	int endOfGame = 0;
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int a=0; a< depth ; a++){
		int chosenMove = option(positions, result, isWhiteToPlay);
		if (chosenMove!=0){
			positions = result;
			isWhiteToPlay ^= 1;
		}else{
			// pass
			endOfGame++;
			if (endOfGame>1) break;			
			a--;
		}
	}
	int diffForBlack = __popcll(positions[globalIdx].y) - __popcll(positions[globalIdx].x);
}

/************************************************************************************
 *
 * Kernel starter
 *
 ************************************************************************************/
int testKernel(int streams, int device, int depth)
{	
	int block_size = BLOCK_SIZE;
	dim3 threads(block_size);
    dim3 grid(GRID_SIZE);

    cudaError_t error;
	cudaStream_t stream[MAX_STREAMS];
		
	int device_count;
	error = cudaGetDeviceCount(&device_count);
	onErrorExit("getDeviceCount", error, __LINE__);
			
	error = cudaSetDevice(device);
	onErrorExit("setDevice", error, __LINE__);

	error = cudaGetDevice(&device); 
	onErrorExit("setDevice", error, __LINE__);
	printf("device %d \n", device);	

	for(int i=0; i < streams; i++){
		error = cudaStreamCreate(&stream[i]);
		onErrorExit("stream create", error, __LINE__);
	}
	
    int numberOfPositions = block_size * GRID_SIZE;	
	int streamSize = numberOfPositions;

	//int area = sizeof(ulonglong2) * streamSize;
	//printf("Allocating host memory x %d: %d b\n", streams, area);
	ulonglong2* hPositions[MAX_STREAMS];
	ulonglong2* dPositions[MAX_STREAMS];
	ulonglong2* hFlipped[MAX_STREAMS];
	ulonglong2* dFlipped[MAX_STREAMS];

    // Allocate host and device memory
	for(int i = 0; i < streams; i++){
		//printf("Alloc mem... stream %d\n",i);
		error = cudaMallocHost((void **) &hPositions[i], sizeof(ulonglong2) * streamSize);		
		onErrorExit("memory", error, __LINE__);
		memset(hPositions[i], 0, sizeof(ulonglong2) * streamSize);
		
		error = cudaMallocHost((void **) &hFlipped[i], sizeof(ulonglong2) * streamSize);
		onErrorExit("memory", error, __LINE__);
		memset(hFlipped[i], 0, sizeof(ulonglong2) * streamSize);

		//printf("Init positions... stream %d\n",i);
		initPositions(hPositions[i], streamSize);

		error = cudaMalloc((void **) &dPositions[i], sizeof(ulonglong2) * streamSize);
		onErrorExit("memory", error, __LINE__);

		error = cudaMalloc((void **) &dFlipped[i], sizeof(ulonglong2) * streamSize);
		onErrorExit("memory", error, __LINE__);
	}    

    //printf("Invoke CUDA Kernel...\n");
	long long numberOfMoves = (long long) numberOfPositions * depth * ITERATIONS;
	printf("Number of moves: %5.3f giga moves\n",  numberOfMoves / 1000000000.0);

	SYSTEMTIME time;	
	GetSystemTime(&time);
	printf("Start: %ld:%ld.%ld\n", time.wMinute, time.wSecond, time.wMilliseconds);
	long long startMs = (long long)(time.wMinute * 60 * 1000 + time.wSecond * 1000 + time.wMilliseconds);
	//WORD startMs = (time.wSecond * 1000) + time.wMilliseconds;	
	
	for(int z = 0; z< ITERATIONS; z++){		
		int i = z % streams;		
		cudaStreamSynchronize(stream[i]);
		//printf("%d",i);
		// copy host memory to device	
		error = cudaMemcpyAsync(dPositions[i], hPositions[i], sizeof(ulonglong2) * streamSize, cudaMemcpyHostToDevice, stream[i]);
		onErrorExit("memory", error, __LINE__);
	
		play<<< grid, threads, 0, stream[i] >>>(dPositions[i], dFlipped[i], 0, depth);

		// Copy result from device to host
#ifdef DO_FLIP
		// Copy resulting black&white positions
		error = cudaMemcpyAsync(&hPositions[i][streamSize], &dPositions[i][streamSize], sizeof(ulonglong2) * streamSize, cudaMemcpyDeviceToHost, stream[i]);
#else
		// Copy only xor patterns
		// error = cudaMemcpyAsync(hFlipped[i], dFlipped[i], sizeof(uint64_t) * streamSize , cudaMemcpyDeviceToHost, stream[i]);
#endif
		onErrorExit("memory", error, __LINE__);				
	}

	for(int i=0; i < streams; i++){
		cudaStreamSynchronize(stream[i]);
	}	

	GetSystemTime(&time);
	long long endTime = (long long)(time.wMinute * 60 * 1000 + time.wSecond * 1000 + time.wMilliseconds);
	long long endMs = endTime - startMs;
	
	char board[500];
	for(int i=0; i < streams; i++){
		board[0] = 0;
//		printf("Before %d\n%s\n", i, toBoard(hPositions[i][0], 1, 0, board));
		board[0] = 0;
#ifdef DO_FLIP
		printf("After %d\n%s\n", i, toBoard(hFlipped[i][0], 1, 0, board));
#else
//		printf("After %d\n%s\n", i, toBoard_Pattern(hFlipped[i][0], 1, board));		
#endif
	}

//	for(int i=0; i < numberOfPositions ; i++){
//		printf("%d: I=%I64U,%I64U R=%I64U,%I64U \n", i , hPositions[i*2], hPositions[i*2 + 1], hPositions[i*2 + numberOfPositions*2], hPositions[i*2 + numberOfPositions*2 +1]);
//	}
	
	printf("End:   %ld:%ld.%ld\n", time.wMinute, time.wSecond, time.wMilliseconds);
	printf("Total: %ld ms\n%5.3f mega moves per second\n", endMs, ((float) (numberOfMoves / endMs)) / 1000.0);


    // Clean up memory
	for(int i=0; i < streams; i++){
		error = cudaFreeHost(hPositions[i]);
		onErrorExit("memory", error, __LINE__);
		error = cudaFreeHost(hFlipped[i]);
		onErrorExit("memory", error, __LINE__);
		error = cudaFree(dPositions[i]);
		onErrorExit("memory", error, __LINE__);
		error = cudaFree(dFlipped[i]);
		onErrorExit("memory", error, __LINE__);
		error = cudaStreamDestroy(stream[i]);
		onErrorExit("destroy stream", error, __LINE__);
	}

    cudaDeviceReset();
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

	int streams = 1;
	int device = 0;
	int nowait = 0;
	int depth = DEFAULT_DEPTH;

	for (int i = 1; i < argc; i++){
		if (strncmp(argv[i],"-Dstreams", 9)==0){
			streams = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Ddevice", 8)==0){
			device = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Ddepth", 8)==0){
			depth = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Dnowait", 8)==0){
			nowait = 1;
		}
	}

	//for(int a = 1; a<depth; a++){		  	
	//	int result = testKernel(streams, device, a);
	//}
	int result = testKernel(streams, device, depth);

	if (!nowait){
		getchar();
	}
    exit(0);
}
