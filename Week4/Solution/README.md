## Distributed and accelerated version of the Jacobi iteration

I performed a 1 dimensional domain decomposition of a square grid, with no periodic boundary conditions. Each chunk of the grid is then offloaded to a gpu via OpenACC. I used `MPI_Sendrecv()`, `#pragma acc update host()` and `#pragma acc update device()` to exchange the ghost rows between gpus at each iteration.

On the host I can avoid having to copy the new grid to the old by swapping the pointers at each iteration. On the other hand this does not work with OpenACC, so I used a different workaround: at each iteration I switch the arguments matrix and matrix_new in the function in which the evolution of the grid is performed (check utility.c).

The grid is saved to file only at the end; I did not exploit the parallel file system so the processes take turns in sending their chunk of the grid to the first one which saves the grid to a binary file.

The goal is to compute the efficiency of the non-accelerated and accelerated distributed versions of the code, and compare the runtime in the two cases.

To compile and run the program you need to load the following modules:
  - spectrum_mpi/10.4.0--binary
  - hpc-sdk/2021--binary

and then run either `make` or `make acc`. To run the program use mpirun Jacobi.x N it print, where N is the grid size (square), it is the number of iterations and print is either 0 or 1 depending on whether or not you want to print the grid to terminal. If the grid is printed to terminal it's not written to file and vice versa.
To generate a png of the grid run `make plot`.

## Solution

The following are the final snapshots for a grid of size 1000x1000 after 10k and 500k iterations:

After 10000 iterations:

<img src="Plots/plot_1k_10k.png" alt="Image Description" width="800" height="600">

After 500000 iterations:

<img src="Plots/jacobi_1k_500k.png" alt="Image Description" width="800" height="600">

## Benchmarks
Here I show the scalability for 1000 iterations on the largest grid I could allocate on the gpus of a single node of the cluster (4x NVIDIA Tesla V100 16Gb). For a correct comparison the size is the same on the non accelerated version benchmark even though the host memory is much larger (256Gb). The scalability for smaller sizes is much worse, especially on the gpus. You can find those plots in the Plots folder.

### CPU scalability
<img src="Plots/mpi_64000.png" alt="Image Description" width="600" height="450">

### GPU scalability
<img src="Plots/acc_64000.png" alt="Image Description" width="600" height="450">

### Comparison
<img src="Plots/Comparison_64000.png" alt="Image Description" width="600" height="450">
