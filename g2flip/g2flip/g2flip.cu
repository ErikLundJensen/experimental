// System includes
#include <stdio.h>
#include "windows.h"
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <vector_types.h>
#include "mapper.h"


// GTX 295 configuration and mega moves per second
//Threads Grid	Total mm / s
//128	2048	810			<--- Best configuration (giving 262144 parallel games)
// 32      1      1			<--- latency is 31ms for 1-10 moves
//  8      1      1			<--- latency is 16ms for 1-10 moves
// Select CUDA device type
#define gtx295 1

#define ITERATIONS 100

// Early break in loops increases throughput by 17% even though threads gets more diverted
#define earlybreak 1

// Use float for counters reduces throughput by about 7% even though several tera flops are available
//#define usefloat 1

// Copy result after each game iteration
#define COPY_RESULT_TO_HOST 1

// Print transcript from one of the played games
//#define DO_TRANSCRIPT 1

#ifdef gtx295
// Actual thread number
#define BLOCK_SIZE 8
#define GRID_SIZE 32768
#else
#define BLOCK_SIZE 8
#define GRID_SIZE 65536
#endif

#define DEFAULT_DEPTH 60

#define MAX_STREAMS 2
#define MAX_DEVICES 1

// Parameters used for switching between float and int for counters
#ifdef usefloat
#define zero 0.0
#define one 1.0
#define three 3.0
#define five 5.0
#define six 6.0
#define eight 8.0
#define sixtyfour 64.0
#define counter float
#else
#define zero 0
#define one 1
#define three 3
#define five 5
#define six 6
#define eight 8
#define sixtyfour 64
#define counter int
#endif

#define IS_MASTER if (threadIdx.x == 1)

// Left edge
const uint64_t NotA1A8 = 18374403900871474942ULL;
// Right edge
const uint64_t NotH1H8 = 9187201950435737471ULL;
// No side edges
const uint64_t NotEDGE = NotH1H8 & NotA1A8;
__constant__ const int DIRECTIONS[] = { -9, -8, -7, -1, 1, 7, 8, 9 };

void initPositions(ulonglong2* p,int numberOfPositions);

uint64_t getBoard(char* board, char c);
int getRow(char* s, char c);
char* toBoard(ulonglong2 b, int labels, int color, char* board);
char* toRow(unsigned int r, int color, char* row);
char* toBoard_Pattern(uint64_t b, int labels, char* board);
char* formatTranscript(int* transcript, int labels, int color, char* board);
int readFromFile(char *filename, ulonglong2* positions, int maxPositions);

// Overall configuration of execution
int streams = 1;
int depth = DEFAULT_DEPTH;
int initFromFile = 0;

double statistics[MAX_DEVICES];

// TODO:
// implement node structure
// refactor so takeback is possible
// Count disc differance at end of game. Store value in node. 
// Use alpha-beta pruning
// multi-threaded s� begge devices anvendes
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
 *	Utility methods for calculating lines excluding H1-H8 and A1-A8.
 *  Only invoked once
 *
 ************************************************************************************/
uint64_t getNotH1H8(){
	uint64_t v = 0LL;
	for(int y = 0; y < 64; y+=8){
		v |= (1LL<< y); 
	} 
	return ~v;
}

uint64_t getNotA1A8(){
	uint64_t v = 0LL;
	for (int y = 0; y < 64; y += 8){
		v |= (128LL<< y); 
	} 
	return ~v;
}

__device__ __inline__ uint64_t getLegalMoves(int direction, uint64_t e, uint64_t me, uint64_t op){
	uint64_t v;		// temp variable	
	uint64_t xor_pattern = 0ULL;

	int d = abs(direction);
	v = me;
	if (direction < 0) {
		v <<= d;
	}else{
		v >>= d;	
	}
	if (d != 8) v &= NotEDGE;	
#pragma __unroll
	for (int a = 0; a < 6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		if (direction < 0) {
			v <<= d;
		}
		else{
			v >>= d;
		}

		xor_pattern |= (e & v);	
		if (d != 8) v &= NotEDGE;
	}
	return xor_pattern;
}

/************************************************************************************
 *
 *	Find legal moves and store them in options array
 *
 ************************************************************************************/
__device__ void getLegalAllMoves(ulonglong2 *board, int isWhiteToPlay, uint64_t *options){
	__shared__ uint64_t me;
	__shared__ uint64_t op;
	__shared__ uint64_t e;

	IS_MASTER{
		if (isWhiteToPlay){
			me = board[0].y;
			op = board[0].x;
		}
		else{
			op = board[0].y;
			me = board[0].x;
		}
		e = ~(me | op);
	}
	__syncthreads();

	int a = threadIdx.x & 7;
	// Safe guard! TODO remove
	if (a > 7){
		a = 7;
	}
	options[a] = getLegalMoves(DIRECTIONS[a], e, me, op);
}

/************************************************************************************
 *
 * Select option to play. Return bit nr to identify location
 *
 ************************************************************************************/
__device__ __inline__ int selectOption(uint64_t options){

	// TODO: Optimize and do select by pruning etc.
	// Select the middle options of the possible options	
	counter numberOfOptions = (counter) __popcll(options);
	if (numberOfOptions > three){
#ifdef usefloat
		numberOfOptions /= 2.0;
#else
		numberOfOptions >>= 1;
#endif		
		for (counter a = zero; a<numberOfOptions; a++){
			int optionDeselected = __ffsll(options);
			options ^= (1ULL << (optionDeselected - 1));
		}
	}
	return __ffsll(options);
}


__device__ __inline__ uint64_t flip(int direction, uint64_t location, uint64_t option, uint64_t boardOpp){
	uint64_t xorPattern = 0ULL;
	uint64_t d = abs(direction);
	if (option & location){
		uint64_t v = location;
#pragma __unroll
		for (int a = 0; a < 6; a++){
			if (direction > 0) {
				v <<= d;
			}
			else{
				v >>= d;
			}
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}
	return xorPattern;
}


__device__ int makeMove(int globalIdx, ulonglong2* positions, ulonglong2* result, int isWhiteToPlay, uint64_t *options){
	__shared__ uint64_t location;
	__shared__ uint64_t boardOpp;
	__shared__ uint64_t xorPattern[8];
	__shared__ int bitNr;

	IS_MASTER{
		bitNr = 0;
		location = 0ULL;
		for (int a = 0; a < 8; a++){
			location |= options[a];
		}

		if (location != 0ULL){
			if (isWhiteToPlay){
				boardOpp = positions[globalIdx].x;
			}
			else{
				boardOpp = positions[globalIdx].y;
			}

			bitNr = selectOption(location);
			if (bitNr > 0){				
				location = 1ULL << (bitNr-1);
			}
		}
	}
	__syncthreads();
	if (location == 0ULL) return 0;	
	
	int a = threadIdx.x & 7;
	// Safe guard! TODO remove
	if (a > 7){
		a = 7;
	}
	xorPattern[a] = flip(DIRECTIONS[a], location, options[a], boardOpp);
	__syncthreads();

	IS_MASTER{
		uint64_t pattern = 0ULL;
		for (int b = 0; b < 8; b++){
			pattern |= xorPattern[b];
		}
		result[globalIdx].x = positions[globalIdx].x ^ pattern;
		result[globalIdx].y = positions[globalIdx].y ^ pattern;

		if (isWhiteToPlay){
			result[globalIdx].y |= location;
		}
		else{
			result[globalIdx].x |= location;
		}
	}
	return bitNr;
}

__device__ __inline__ void swap(ulonglong2* p1, ulonglong2* p2){
	ulonglong2* temp = p1;
	p1 = p2;
	p2 = temp;
}

/************************************************************************************
 *
 * TODO: count disc differance at end of game. Store value in node. Use alpha-beta pruning
 *
 ************************************************************************************/
__global__ void play(ulonglong2* positions, ulonglong2* result, int isWhiteToPlayParm, counter depth, int* transcript)
{	  
	__shared__ counter endOfGame;
	__shared__ int isWhiteToPlay;
	__shared__ int globalIdx;
	__shared__ uint64_t options[8];
	__shared__ int movenr;

	IS_MASTER{
		endOfGame = zero;
		isWhiteToPlay = isWhiteToPlayParm;
		globalIdx = blockIdx.x * blockDim.x;
		movenr = 0;
	}
	__syncthreads();
	
	do{
		getLegalAllMoves(&positions[globalIdx], isWhiteToPlay, options);
		__syncthreads();
		int chosenMove = makeMove(globalIdx, positions, result, isWhiteToPlay, options);

		IS_MASTER{
			isWhiteToPlay ^= 1;
			if (chosenMove != 0){
				positions[globalIdx] = result[globalIdx];
				endOfGame = zero;
#ifdef DO_TRANSCRIPT
				transcript[chosenMove - 1] = movenr + 1;
#endif
				movenr++;
			}
			else{
				// pass				
				endOfGame++;				
			}			
		}
		__syncthreads();
		if (endOfGame > one) break;
	} while (movenr < depth);
	__syncthreads();
	// Copy options to result
	/*
	IS_MASTER{
		for (int a = 0; a < 8; a++){
			result[globalIdx].x |= options[a];
			result[globalIdx].y |= options[a];
		}
	}
	*/
	__syncthreads();
	//int diffForBlack = __popcll(positions[globalIdx].y) - __popcll(positions[globalIdx].x);

}

/************************************************************************************
 *
 * Kernel starter
 *
 ************************************************************************************/
int testKernel(int streams, int device, int depth, int initFromFile)
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
	
    int numberOfPositions = block_size * GRID_SIZE / 8;	
	int streamSize = numberOfPositions;

	//int area = sizeof(ulonglong2) * streamSize;
	//printf("Allocating host memory x %d: %d b\n", streams, area);
	ulonglong2* hPositions[MAX_STREAMS];
	ulonglong2* dPositions[MAX_STREAMS];
	ulonglong2* hFlipped[MAX_STREAMS];
	ulonglong2* dFlipped[MAX_STREAMS];
	int* hTranscript[MAX_STREAMS];
	int* dTranscript[MAX_STREAMS];

	// Allocate host and device memory

	for(int i = 0; i < streams; i++){
		//printf("Alloc mem... stream %d\n",i);
		error = cudaMallocHost((void **) &hPositions[i], sizeof(ulonglong2) * streamSize);		
		onErrorExit("memory", error, __LINE__);
		memset(hPositions[i], 0, sizeof(ulonglong2) * streamSize);
		
		error = cudaMallocHost((void **) &hFlipped[i], sizeof(ulonglong2) * streamSize);
		onErrorExit("memory", error, __LINE__);
		memset(hFlipped[i], 0, sizeof(ulonglong2) * streamSize);

		error = cudaMallocHost((void **)&hTranscript[i], sizeof(int) * streamSize * 64);
		onErrorExit("memory", error, __LINE__);
		memset(hTranscript[i], 0, sizeof(int) * streamSize * 64);

		//printf("Init positions... stream %d\n",i);
		if (initFromFile != 0){
			int loaded = readFromFile("\\temp\\games.txt", hPositions[i], numberOfPositions);
			printf("Loaded %d positions\n", loaded);
		}
		else{
			initPositions(hPositions[i], streamSize);
		}

		error = cudaMalloc((void **) &dPositions[i], sizeof(ulonglong2) * streamSize);
		onErrorExit("memory", error, __LINE__);

		error = cudaMalloc((void **) &dFlipped[i], sizeof(ulonglong2) * streamSize);
		onErrorExit("memory", error, __LINE__);

		error = cudaMalloc((void **)&dTranscript[i], sizeof(int) * streamSize * 64);
		onErrorExit("memory", error, __LINE__);

	}    

    //printf("Invoke CUDA Kernel...\n");
	long long numberOfMoves = (long long) numberOfPositions * depth * ITERATIONS;
	printf("Number of moves: %5.3f giga moves\n",  numberOfMoves / 1000000000.0);

	counter maxDepth = (counter) depth;

	SYSTEMTIME time;	
	GetSystemTime(&time);
	printf("Start: %ld:%ld.%ld\n", time.wMinute, time.wSecond, time.wMilliseconds);
	long long startMs = (long long)(time.wMinute * 60 * 1000 + time.wSecond * 1000 + time.wMilliseconds);
	//WORD startMs = (time.wSecond * 1000) + time.wMilliseconds;	
	
	for(int z = 0; z< ITERATIONS; z++){		
		int i = z % streams;		
		
		// copy host memory to device	
		error = cudaMemcpyAsync(dPositions[i], hPositions[i], sizeof(ulonglong2) * streamSize, cudaMemcpyHostToDevice, stream[i]);
		
		onErrorExit("memory", error, __LINE__);
		cudaStreamSynchronize(stream[i]);

		play<<< grid, threads, 0, stream[i] >>>(dPositions[i], dFlipped[i], 0, maxDepth, dTranscript[i]);
		
		// Copy result from device to host		
#ifdef DO_TRANSCRIPT
		// FIX-ME: Won't work with multiple streams...
		error = cudaMemcpyAsync(&hTranscript[i][0], &dTranscript[i][0], sizeof(int) * streamSize * 64, cudaMemcpyDeviceToHost, stream[i]);
		onErrorExit("memory", error, __LINE__);
#endif

#ifdef COPY_RESULT_TO_HOST
		// Copy result 
		error = cudaMemcpyAsync(hFlipped[i], dFlipped[i], sizeof(ulonglong2) * streamSize, cudaMemcpyDeviceToHost, stream[i]);
		onErrorExit("memory", error, __LINE__);				
#endif
	}
	for(int i=0; i < streams; i++){
		cudaStreamSynchronize(stream[i]);
	}

	GetSystemTime(&time);
	long long endTime = (long long)(time.wMinute * 60 * 1000 + time.wSecond * 1000 + time.wMilliseconds);	
	long long endMs = endTime - startMs;
	
	char board[4000];
	for (int i = 0; i < streams; i++){
		// Prepare print-buffer// Prepare print-buffer
		board[0] = 0;
		printf("Before %d\n%s\n", i, toBoard(hPositions[i][0], 1, 0, board));

		// Prepare print-buffer	
		board[0] = 0;
		printf("After %d\n%s\n", i, toBoard(hFlipped[i][0], 1, 0, board));

#ifdef DO_TRANSCRIPT
		// Prepare print-buffer		
		board[0] = 0;
		printf("Transcript %d:\n%s\n", i, formatTranscript(hTranscript[i], 1, 0, board));		
//		printf("After %d\n%s\n", i, toBoard_Pattern(hFlipped[i][0], 1, board));		
#endif
	}

//	for(int i=0; i < numberOfPositions ; i++){
//		printf("%d: I=%I64U,%I64U R=%I64U,%I64U \n", i , hPositions[i*2], hPositions[i*2 + 1], hPositions[i*2 + numberOfPositions*2], hPositions[i*2 + numberOfPositions*2 +1]);
//	}
	
	printf("Thread end:   %ld:%ld.%ld\n", time.wMinute, time.wSecond, time.wMilliseconds);
	double megaMovesPerSecond = 0.0;
	if (endMs > 0){
		megaMovesPerSecond = ((double)(numberOfMoves / endMs)) / 1000.0;
	}
	printf("Thread total: %ld ms\nThread: %5.3f mega moves per second\n", endMs, megaMovesPerSecond);
	printf("Time per iteration: %ld ns\n", (endMs * 1000) / ITERATIONS);

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
		error = cudaFree(dTranscript[i]);
		onErrorExit("memory", error, __LINE__);
		error = cudaStreamDestroy(stream[i]);
		onErrorExit("destroy stream", error, __LINE__);
	}

    cudaDeviceReset();
	statistics[device] = megaMovesPerSecond;
	return 0;
}


DWORD WINAPI testKernalThreadable(LPVOID lpParam){
	return testKernel(streams, *(int*) lpParam, depth, initFromFile);	
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Othello Using CUDA] - Starting...\n");    

	HANDLE threadHandles[MAX_DEVICES];

	int devices = 1;
	int nowait = 0;

	for (int i = 1; i < argc; i++){
		if (strncmp(argv[i],"-Dstreams", 9)==0){
			streams = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Ddevices", 8)==0){
			devices = atoi(argv[++i]);
			if (devices > MAX_DEVICES){
				printf("Maximum number of devices is %d", MAX_DEVICES);
				exit(1);
			}
		}
		if (strncmp(argv[i],"-Ddepth", 8)==0){
			depth = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Dnowait", 8)==0){
			nowait = 1;
		}
		if (strncmp(argv[i], "-Dfile", 6) == 0){
			initFromFile = 1;
		}
	}

	//for(int a = 1; a<depth; a++){		  	
	//	int result = testKernel(streams, device, a);
	//}

	int deviceNr[MAX_DEVICES];
	
	for (int i = 0; i < devices; i++){
		deviceNr[i] = i;
		threadHandles[i] = CreateThread(NULL, 0, testKernalThreadable, &deviceNr[i], 0, NULL);
	}

	WaitForMultipleObjects(devices, threadHandles, TRUE, INFINITE);

	double mmps = 0.0;
	for (int i = 0; i < devices; i++){
		CloseHandle(threadHandles[i]);
		mmps += statistics[i];
	}
	printf("Total: %5.3f mega moves per second\n", mmps);

	if (!nowait){
		getchar();
	}
    exit(0);
}
