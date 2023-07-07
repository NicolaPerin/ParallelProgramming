## Distributed and accelerated version of the Jacobi iteration

We perform a 1 dimensional domain decomposition of a square grid, with no periodic boundary conditions. Each chunk of the grid is then offloaded to a gpu via OpenACC. We use MPI_Sendrecv(), #pragma acc update host() and #pragma acc update device() to exchange the ghost rows between gpus at each iteration.

The goal is to compute the efficiency of the non-accelerated and accelerated distributed versions of the code, and compare the runtime in the two cases.
