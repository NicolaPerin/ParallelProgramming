## Distributed and accelerated version of the Jacobi iteration

We perform a 1 dimensional domain decomposition of a square grid, with no periodic boundary conditions. Each chunk of the grid is then offloaded to a gpu via OpenACC. We use `MPI_Sendrecv()`, `#pragma acc update host()` and `#pragma acc update device()` to exchange the ghost rows between gpus at each iteration.

On the host we can avoid having to copy the new grid to the old by swapping the pointers at each iteration. On the other hand this does not work with OpenACC, so we use a different workaround: at each iteration we switch the arguments matrix and matrix_new in the function in which the evolution of the grid is performed (check utility.c).

The goal is to compute the efficiency of the non-accelerated and accelerated distributed versions of the code, and compare the runtime in the two cases.


