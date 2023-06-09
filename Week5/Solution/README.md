## Distributed version of the FFTW library

We distribute the data on the n1 direction among all the process, and perform a 2D FFT on the n2-n3 plan. After that, we reorder the data inside a buffer using `MPI_Alltoallw()` routine in such a way that the orginal matrix is now distributed among the process along the n2 direction. It's now possible to perform the fft on the n1 direction and then go back to the original data order. In this way there's no need to change the plot routines.
To perform the FFT we use the `fftw_plan_many_dft()`, which allows to use only one call to perform the fft on a 2D or 1D plan. For both the 2D plan and the 1D plan, the data are considered contiguos.
To send the data we create an MPI_Datatype, called column_block, which represents the matrix block that each process send and receive during the alltoall routine. The block will contain `n1_loc * n2_loc * n3` elements and, since we work in a row-major access, the next line is after `n2_loc * n3` elements. The advantages of this approach is that both before and after the reordering, the data are contiguos through a row-major access.

The goal is to compare the runtime of the fftw-mpi library with the manually distributed code.

To compile the program on the Marconi100 cluster we have to load the following modules (using *autoload*):
  - fftw/3.3.8--spectrum_mpi--10.3.1--binary

and then run `make`. To run the program use `mpirun fftw.x dt nstep n1 n2 n3`. To generate the plots run `make plot`.

## Solution

The following animation shows the solution of the diffusion equation using a grid of size 256 x 256 x 512:

![Alt Text](animate.gif)

## Benchmarks

Here are reported the measured runtimes for the two versions of the code as well as the efficiency in every case, for a grid of size 512 x 512 x 1024 and 100 time steps:

| | FFTW_MPI | | custom | |
| -------------- | -------------- | -------------- | -------------- | -------------- |
| Number of nodes | runtime (s) | efficiency | runtime (s) | efficiency |
| 1 | 1048.32 | 1.000 | 899.280 | 1.000 | 
| 2 | 662.313 | 0.791 | 566.119 | 0.794 | 
| 4 | 455.914 | 0.575 | 388.254 | 0.579 | 
| 8 | 285.021 | 0.460 | 251.448 | 0.447 |


The following graph shows the scalability comparison of the two versions of the code:

<img src="Comparison_FFTW.png" alt="Image Description" width="600" height="450">


