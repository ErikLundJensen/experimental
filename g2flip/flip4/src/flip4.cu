// System includes
#include <stdio.h>
//#include "windows.h"
#include <time.h>
#include <sys/time.h>

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
//#define gtx295 1

#define ITERATIONS 2000

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
#define BLOCK_SIZE 128
#define GRID_SIZE 2048
#else
#define BLOCK_SIZE 32
#define GRID_SIZE 512
#endif

#define DEFAULT_DEPTH 60

#define MAX_STREAMS 2
#define MAX_DEVICES 2

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

#define DIR_DOWN 0
#define DIR_UP 1
#define DIR_LEFT 2
#define DIR_RIGHT 3
#define DIR_LEFTDOWN 4
#define DIR_LEFTUP 5
#define DIR_RIGHTDOWN 6
#define DIR_RIGHTUP 7

const uint64_t NotA1A8 = 18374403900871474942ULL;
const uint64_t NotH1H8 = 9187201950435737471ULL;
const uint64_t NotEDGE = NotH1H8 & NotA1A8;

void initPositions(ulonglong2* p,int numberOfPositions);

uint64_t getBoard(char* board, char c);
int getRow(char* s, char c);
char* toBoard(ulonglong2 b, int labels, int color, char* board);
char* toRow(unsigned int r, int color, char* row);
char* toBoard_Pattern(uint64_t b, int labels, char* board);
char* formatTranscript(int* transcript, int labels, int color, char* board);
int readFromFile(char *filename, ulonglong2* positions, int maxPositions);

long long getCurrentTs(){
	struct timeval tp;
    gettimeofday(&tp, NULL);
    long long ms = (long long) tp.tv_sec * 1000L + tp.tv_usec / 1000;
    return ms;
}

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

/************************************************************************************
 *
 *	Find legal moves and store them in options array
 *
 ************************************************************************************/
__device__ void getLegalMoves(ulonglong2 *board, int isWhiteToPlay, uint64_t *options){
	uint64_t me;
	uint64_t op;

	// Clear options
//#pragma __unroll
	for(int a = 0; a<8; a++){
		options[a] = 0LL;
	}

	if (isWhiteToPlay){
		me = board[0].y;
		op = board[0].x;
	}else{		
		op = board[0].y;
		me = board[0].x;
	}	

	uint64_t e;		// empty fields
	uint64_t v;		// temp variable	
	e = ~(me | op);
	/////////////////////////////////
	// down
	v = me;
	v <<= 8;		// 1.
//#pragma __unroll
	for (int a = 0; a < 6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v <<= 8;		// 2-7
		options[DIR_DOWN] |= (e & v);
	}
	
	/////////////////////////////////
	// up	
	v = me;
	v >>= 8;		// 1.
//#pragma __unroll
	for (int a = 0; a < 6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v >>= 8;
		options[DIR_UP] |= (e & v);
	}
	
	
	/////////////////////////////////
	// right		
	v = me;
	v <<= 1;		// 1.
	v &= NotEDGE;
//#pragma __unroll
	for(int a = 0; a<6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v <<= 1;		// 2-7		
		options[DIR_RIGHT] |= (e & v);
		v &= NotEDGE;
	}
	
	/////////////////////////////////
	// left
	
	v = me;
	v >>= 1;		// 1.	
	v &= NotEDGE;
//#pragma __unroll
	for(int a = 0; a<6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v >>= 1;		// 2-7		
		options[DIR_LEFT] |= (e & v);
		v &= NotEDGE;
	}
	
	/////////////////////////////////
	// left-up
	v = me;
	v >>= 9;		// 1.
	v &= NotEDGE;
//#pragma __unroll
	for(int a = 0; a<6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v >>= 9;		// 2-7
		options[DIR_LEFTUP] |= (e & v);
		v &= NotEDGE;
	}
	
	/////////////////////////////////
	// right-up
	v = me;
	v >>= 7;		// 1.
	v &= NotEDGE;
//#pragma __unroll
	for(int a = 0; a<6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v >>= 7;		// 2-7		
		options[DIR_RIGHTUP] |= (e & v);
		v &= NotEDGE;
	}
	
	/////////////////////////////////
	// left-down
	v = me;
	v <<= 7;		// 1.
	v &= NotEDGE;
//#pragma __unroll
	for(int a = 0; a<6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v <<= 7;		// 2-7
		options[DIR_LEFTDOWN] |= (e & v);
		v &= NotEDGE;
	}
	
	/////////////////////////////////
	// right-down
	v = me;
	v <<= 9;		// 1.
	v &= NotEDGE;	
//#pragma __unroll
	for(int a = 0; a<6; a++){
		v &= op;
#ifdef earlybreak
		if (v == 0) break;
#endif
		v <<= 9;		// 2-7
		options[DIR_RIGHTDOWN] |= (e & v);
		v &= NotEDGE;
	}
	return;
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



__device__ __inline__ int makeMove(int globalIdx, ulonglong2* positions, ulonglong2* result, int isWhiteToPlay, uint64_t *options){
	uint64_t location = 0LL;
//#pragma __unroll
   	for(int a = 0; a<8; a++){
   		location |= options[a];
   	}

	if (location == 0ULL) return 0;

	uint64_t boardOpp;
	
	if (isWhiteToPlay){		
		boardOpp = positions[globalIdx].x;
	}else{		
		boardOpp = positions[globalIdx].y;
	}

	int bitNr = selectOption(location);
	if (bitNr == 0){
		return 0;
	}
	bitNr--;
	location = 1ULL << bitNr;

	uint64_t xorPattern = 0LL;
	if (options[DIR_UP] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v <<= 8;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;			
		}
	}

	if (options[DIR_DOWN] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v >>= 8;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}

	if (options[DIR_LEFT] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v <<= 1;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}

	if (options[DIR_RIGHT]& location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v >>= 1;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}

	if (options[DIR_LEFTUP] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v <<= 9;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}

	if (options[DIR_LEFTDOWN] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v >>= 7;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}

	if (options[DIR_RIGHTUP] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v <<= 7;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}
	if (options[DIR_RIGHTDOWN] & location){
		uint64_t v = location;
//#pragma __unroll
		for(int a=0; a<6; a++){
			v >>= 9;
			if (!(boardOpp & v)) break;
			xorPattern ^= v;
		}
	}		

	result[globalIdx].x = positions[globalIdx].x ^ xorPattern;
	result[globalIdx].y = positions[globalIdx].y ^ xorPattern;

	if (isWhiteToPlay){				
		result[globalIdx].y |= location;
	}else{
		result[globalIdx].x |= location;		
	}

	return bitNr + 1;
}

__device__ __inline__ void swap(ulonglong2* p1, ulonglong2* p2){
	ulonglong2* temp = p1;
	p1 = p2;
	p2 = temp;
}

/************************************************************************************
 *
 * TODO: count disc difference at end of game. Store value in node. Use alpha-beta pruning
 *
 ************************************************************************************/
__global__ void play(ulonglong2* positions, ulonglong2* result, int isWhiteToPlay, counter depth, int* transcript)
{	  
	counter endOfGame = zero;
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

	uint64_t options[8];
	for(counter a=zero; a < depth ; a++){
		getLegalMoves(&positions[globalIdx], isWhiteToPlay, options);
		int chosenMove = makeMove(globalIdx, positions, result, isWhiteToPlay, options);
		isWhiteToPlay ^= 1;
		if (chosenMove!=0){
			positions[globalIdx] = result[globalIdx];
			
			endOfGame = zero;
#ifdef DO_TRANSCRIPT
			transcript[chosenMove-1] = a +1;
#endif
		}else{
			// pass
			endOfGame++;			
			if (endOfGame>one) break;
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
	
    int numberOfPositions = block_size * GRID_SIZE;	
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
			int loaded = readFromFile("../games.txt", hPositions[i], numberOfPositions);
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

#ifdef WIN
	SYSTEMTIME startTime;
	GetSystemTime(&startTime);
	printf("Start: %ld:%ld.%ld\n", startTime.wMinute, startTime.wSecond, startTime.wMilliseconds);
	long long startMs = (long long)(startTime.wMinute * 60 * 1000 + startTime.wSecond * 1000 + startTime.wMilliseconds);
#else
	time_t startTime = time(NULL);
	printf("Start: ");
	printf(ctime(&startTime));
	long long startMs = getCurrentTs();
#endif
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
		// Copy only xor patterns
		error = cudaMemcpyAsync(hFlipped[i], dFlipped[i], sizeof(uint64_t) * streamSize , cudaMemcpyDeviceToHost, stream[i]);
		onErrorExit("memory", error, __LINE__);				
#endif
	}
	for(int i=0; i < streams; i++){
		cudaStreamSynchronize(stream[i]);
	}	

#ifdef WIN
	GetSystemTime(&startTime);
	long long endTime = (long long)(startTime.wMinute * 60 * 1000 + startTime.wSecond * 1000 + startTime.wMilliseconds);
	long long endMs = endTime - startMs;
#else
	time_t endTime = time(NULL);
	long long endMs = getCurrentTs() - startMs;
#endif

	char board[4000];
	for(int i=0; i < streams; i++){
		// Prepare print-buffer// Prepare print-buffer
		board[0] = 0;
		printf("Before %d\n%s\n", i, toBoard(hPositions[i][10], 1, 0, board));

		// Prepare print-buffer		
		board[0] = 0;
		printf("After %d\n%s\n", i, toBoard(hFlipped[i][10], 1, 0, board));				

#ifdef DO_TRANSCRIPT
		// Prepare print-buffer		
		board[0] = 0;
		printf("Transcript %d:\n%s\n", i, formatTranscript(hTranscript[i], 1, 0, board));		
//		printf("After %d\n%s\n", i, toBoard_Pattern(hFlipped[i][0], 1, board));		
#endif
	}

//	for(int i=0; i < numberOfPositions ; i++){
//		// printf("%d: I=%I64U,%I64U R=%I64U,%I64U \n", i , hPositions[0][i].x, hPositions[0][i].y, );
//		board[0] = 0;
//		printf("%d\n%s\n", i, toBoard(hFlipped[0][i], 1, 0, board));
//	}
	
#ifdef WIN
	printf("Thread end:   %ld:%ld.%ld\n", time.wMinute, time.wSecond, time.wMilliseconds);
#else
	printf("Thread end: ");
	printf(ctime(&endTime));
#endif
	if (endMs<1){
		printf("Warning: Duration < 1 ms\n");
		endMs = 1;
	}
	double megaMovesPerSecond = ((double)(numberOfMoves / endMs)) / 1000.0;
	printf("Thread total: %ld ms\nThread: %5.3f mega moves per second\n", endMs, megaMovesPerSecond);

	double gigaMovesPerSecond = ((double)(numberOfMoves / endMs)) / 1000.0 / 1000.0;
	printf("Thread total: %ld ms\nThread: %5.3f giga moves per second\n", endMs, gigaMovesPerSecond);


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

//
//DWORD WINAPI testKernalThreadable(LPVOID lpParam){
//	return testKernel(streams, *(int*) lpParam, depth, initFromFile);
//}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Othello Using CUDA] - Starting...\n");    

	//HANDLE threadHandles[MAX_DEVICES];

	int devices = 2;
	int nowait = 0;
	int device = 1;

	for (int i = 1; i < argc; i++){
		if (strncmp(argv[i],"-Dstreams", 9)==0){
			streams = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Ddevices", 9)==0){
			devices = atoi(argv[++i]);
			if (devices > MAX_DEVICES){
				printf("Maximum number of devices is %d", MAX_DEVICES);
				exit(1);
			}
		}else if (strncmp(argv[i],"-Ddevice", 8)==0){
			device = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Ddepth", 7)==0){
			depth = atoi(argv[++i]);
		}
		if (strncmp(argv[i],"-Dnowait", 7)==0){
			nowait = 1;
		}
		if (strncmp(argv[i], "-Dfile", 6) == 0){
			initFromFile = 1;
		}
	}

	int result = testKernel(streams, device, depth, initFromFile);


//	int deviceNr[MAX_DEVICES];
//
//	for (int i = 0; i < devices; i++){
//		deviceNr[i] = i;
//		threadHandles[i] = CreateThread(NULL, 0, testKernalThreadable, &deviceNr[i], 0, NULL);
//	}
//
//	WaitForMultipleObjects(devices, threadHandles, TRUE, INFINITE);
//
//	double mmps = 0.0;
//	for (int i = 0; i < devices; i++){
//		CloseHandle(threadHandles[i]);
//		mmps += statistics[i];
//	}
//	printf("Total: %5.3f mega moves per second\n", mmps);

	if (!nowait){
//		getchar();
	}
    exit(0);
}
