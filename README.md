experimental
============

Experimental code

g2flip is a CUDA implementation of Othello board game. The implementation covers:<br>
1) Find legal moves<br>
2) Select pseudo-random move<br>
3) Flip discs<br>
4) Count the number of dics if end-of-game<br>

The test setup runs through from start position to end-of-game (including handling pass).

The result of the performance test is:
1.17 giga moves per second (where "move" is defined as above)

Running at overclocked watercooled NVIDIA Geforce GTX 690.

It is quite problematic to combine this implementation with existing Othello programs running on CPU, as the amount of data is hard to get through the PCI bus.
Calculating 20 bytes per position gives 23 GB/s in and out which is beyond the current speed of PCI 3.0 x16.

One solution to this limitation is running the alpha-beta pruning at the GPU using CUDA, however, that is a complex task as branching is bad for the GPU.


Copyright notes:
The part of the CUDA source code for "find legal moves" is ported from Edax to CUDA without much optimization. That part could be further optimized for GPU, however, as the bottleneck is the PCI bus it does not help bringing moves per second even higher.
The ported Edax algorithm is implemented in the method getLegalMoves().

The Edax source code is under "GNU General Public License". See also http://abulmo.perso.neuf.fr/edax/4.3/index.htm 
Edax is written by Richard Delorme.

