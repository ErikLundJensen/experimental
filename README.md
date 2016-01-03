experimental
============

This branch of the code tests parallel flipping of discs in all 8 directions. This version puts 8 threads in each block which is far from optimal, however, it eases the synchronization between threads.

This variation runs up to 400 mill moves/s on gtx 690.

Tested at the slower GTX 295 we get a lower latency:
The lattency has dropped to 125 ns for making 2 moves in one game. Playing 10 moves takes 219 ns and a full game 704 ns. (Note we are talking nano seconds here)
The overhead of starting the kernel is around 125 ns and 9 ns for each move.

Starting 32K games with 2  moves give 0.05 ns per move.
Starting 32K games with 60 moves give 0.0066 ns per move. (149m/s)

The price for getting lower latency is reducing the overall through-put.
Optimizing the implementation to play more moves per thread block complicates the sharing of states, however, it would improve through-put without increasing latency.


----





Experimental code

g2flip is a CUDA implementation of Othello board game. The implementation covers:<br>
1) Find legal moves<br>
2) Select pseudo-random move<br>
3) Flip discs<br>
4) Count the number of dics if end-of-game<br>

Before each game the initial positions is copied to the GPU memory. After end-of-game copy resulting positions or transcript to host memory (regular CPU managed memory)
The invocation of copy is included in the performance benchmark.

The test setup runs through from start position to end-of-game (including handling pass).

The result of the performance test is:
<B>2.8 giga moves per second</B> (where "move" is defined as above)

Running at overclocked watercooled NVIDIA Geforce GTX 690.

The latest gain (from 1.17G moves/s to 2.8G moves/s) came from simplification of algorithm for flipping discs -- the cost was addition memory usage (64 bytes from the stack).
The current source/implementation does not look like a real optimization for the GPU, as it contains a lot of branching, however, the implementation is anyhow real fast.

It is quite problematic to combine this implementation with existing Othello programs running on CPU, as the amount of data is hard to get through the PCI bus.
Calculation given by 20 bytes per position gives 56 GB/s in and out which is beyond the current speed of PCI 3.0 x16.

One solution to this limitation is running the alpha-beta pruning at the GPU using CUDA, however, that is a complex task as branching is bad for the GPU.
Launching a Cuda kernel for just a few moves or positions has a large overhead (minimum 16ms om GTX 295). The current implementation handles 2 million positions in parallel to get to the maxium speed at the GTX 690.

One way to optimize the algorithm could be to lower the latency by using multiple threads per position. This requires that 8 threads is handling the same position which leads to some syncronization of the threads and will be a quite complex implementation. This could lead to a lower overall through-put, however, the latency will be decreased.

GPU-side alpha beta pruning is a must to get any benefit of the speed of the GPU.

Looking into the next generation of NVIDIA products (Pascal architecture) GPU memory will be even faster and come in sizes up to 16GB. This indeed points to the direction of implementing alpha beta pruning in CUDA.
A simple implementation of the alpha beta pruning which does not take advantage of the massive parallelisme will still be faster than copying the positions to host memory.


Copyright notes:
The part of the CUDA source code for "find legal moves" is ported from Edax to CUDA without much optimization. That part could be further optimized for GPU, however, as the bottleneck is the PCI bus it does not help bringing moves per second even higher.
The ported Edax algorithm is implemented in the method getLegalMoves().

The Edax source code is under "GNU General Public License". See also http://abulmo.perso.neuf.fr/edax/4.3/index.htm 
Edax is written by Richard Delorme.

