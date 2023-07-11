## Distributed version of the FFTW library

We distribute the data on the n1 direction among all the process, and perform a 2D FFT on the n2-n3 plan. After that, we reorder the data inside a buffer using `MPI_Alltoallw()` routine in such a way that the orginal matrix is now distributed among the process along the n2 direction. It's now possible to perform the fft on the n1 direction and then go back to the original data order. In this way there's no need to change the plot routines.
To perform the FFT we use the `fftw_plan_many_dft()`, which allows to use only one call to perform the fft on a 2D or 1D plan. For both the 2D plan and the 1D plan, the data are considered contiguos.
