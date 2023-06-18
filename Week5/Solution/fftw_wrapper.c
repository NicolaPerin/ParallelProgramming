#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "utilities.h"

double seconds() {
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

// Removed n1 from the arguments because it's not used in the body of the function
int index_f (int i1, int i2, int i3, int n2, int n3) { return n3*n2*i1 + n3*i2 + i3; } 

void init_fftw(fftw_dist_handler *fft, int n1, int n2, int n3, MPI_Comm comm) {
  
  // I initialize again the MPI environment
  int wsz, rank, i;
  MPI_Comm_size( comm, &wsz );
  MPI_Comm_rank( comm, &rank );
  fft->mpi_comm = comm; 

  // I want a symmetric distribution of the data among the wsz processes
  if((( n1 % wsz ) || (n2 % wsz)) && !rank) {
    fprintf( stdout, "\nN1 dimension must be multiple of the number of processes. Aborting...\n\n" );
    MPI_Abort(comm, 1);
  } 

  fft->n1 = n1; 
  fft->n2 = n2; 
  fft->n3 = n3; 

  // Compute the local dimensions of the grid (manually)
  fft->local_n1 = fft->n1 / wsz; 
  fft->local_n1_offset = fft->local_n1 * rank; 

  fft->local_n2 = n2 / wsz; 
  fft->local_n2_offset = fft->local_n2 * rank; 

  fft->global_size_grid = n1 * n2 * n3; 
  fft->local_size_grid = fft->local_n1 * fft->n2 * fft->n3; 

// This is the dimension of the volume that the all_to_all routine will send
  fft->alltoall_size = fft->local_n1 * fft->local_n2 * fft->n3; 

  // Allocate data to transform and make plans
  fft->data = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof(fftw_complex)); 
  fft->data_redistributed = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof(fftw_complex)); 

  /*
   * Create the fft plans using the fftw advanced interface.
   * The first plan will work on the n2 and n3 dimension while the last one only on n1.
   * The advanced interface need some extra parameters: howmany, inembed, istridem onemebed, ostride, odist
   */

  // The advanced interface requires an array containing the dimensions on which perform the fft
  int dims[] = {n2, n3}; 
  int dim_1[] = {n1}; 

  // First the 2 dimensional plan
  fft->fw_plan_2d = fftw_plan_many_dft(2, dims, fft->local_n1, fft->data, dims, 1, 
                                      fft->n2 * fft->n3, fft->data, dims, 1, fft->n2 * fft->n3, FFTW_FORWARD, FFTW_ESTIMATE);

  // Then the 1 dimensional
  fft->fw_plan_1d = fftw_plan_many_dft(1, dim_1, fft->local_n2 * fft->n3, fft->data_redistributed, dim_1, fft->local_n2 * fft->n3, 1, 
                                      fft->data_redistributed, dim_1, fft->local_n2*fft->n3, 1, FFTW_FORWARD, FFTW_ESTIMATE);

  //For both plan we also need the inverse transformation
  fft->bw_plan_2d = fftw_plan_many_dft(2, dims, fft->local_n1, fft->data, dims, 1, 
                                      fft->n2*fft->n3, fft->data, dims, 1, fft->n2*fft->n3, FFTW_BACKWARD, FFTW_ESTIMATE);

  fft->bw_plan_1d = fftw_plan_many_dft(1, dim_1, fft->local_n2*fft->n3, fft->data_redistributed, dim_1, fft->local_n2*fft->n3, 1, 
                                      fft->data_redistributed, dim_1, fft->local_n2*fft->n3, 1, FFTW_BACKWARD, FFTW_ESTIMATE);

  // Datatype which selects the correct data to send
  MPI_Datatype column_block;

  MPI_Type_vector(fft->local_n1, /*number of blocks*/
                  fft->local_n2 * fft->n3, /*block size*/
                  fft->n2 * fft->n3, /*stride*/
                  MPI_C_DOUBLE_COMPLEX, /*old datatype*/
                  &column_block); /*new datatype*/

  MPI_Type_commit(&column_block);

  // Counts, displacements and types for the alltoall
  fft->sendcounts = (int *)malloc(wsz * sizeof(int));
  fft->senddispls = (int *)malloc(wsz * sizeof(int));
  fft->recvcounts = (int *)malloc(wsz * sizeof(int));
  fft->recvdispls = (int *)malloc(wsz * sizeof(int));
  fft->sendtype = (MPI_Datatype *)malloc(wsz * sizeof(MPI_Datatype));
  fft->recvtype = (MPI_Datatype *)malloc(wsz * sizeof(MPI_Datatype));

  fft->senddispls[0] = 0; fft->recvdispls[0] = 0;
  fft->sendtype[0] = column_block;
  fft->recvtype[0] = MPI_C_DOUBLE_COMPLEX;

  for (i = 0; i < wsz - 1; i++) {
    fft->sendcounts[i] = 1;
    fft->recvcounts[i] = fft->alltoall_size;
    fft->senddispls[i + 1] = fft->senddispls[i] + fft->local_n2 * fft->n3 * sizeof(fftw_complex);
    fft->recvdispls[i + 1] = fft->recvdispls[i] + fft->alltoall_size * sizeof(fftw_complex);
    fft->sendtype[i + 1] = column_block;
    fft->recvtype[i + 1] = MPI_C_DOUBLE_COMPLEX;
  }

  fft->sendcounts[wsz - 1] = 1;
  fft->recvcounts[wsz - 1] = fft->alltoall_size;
  
}

void close_fftw(fftw_dist_handler *fft ) { 

    fftw_destroy_plan(fft->bw_plan_2d);
    fftw_destroy_plan(fft->bw_plan_1d);
    fftw_destroy_plan(fft->fw_plan_2d);
    fftw_destroy_plan(fft->fw_plan_1d);

    fftw_free(fft->data);
    fftw_free(fft->data_redistributed);

    free(fft->sendcounts);
    free(fft->senddispls);
    free(fft->recvcounts);
    free(fft->recvdispls);
    free(fft->sendtype);
    free(fft->recvtype);

    fftw_cleanup();
}

/* This subroutine uses fftw to calculate 3-dimensional discrete FFTs.
 * The data in direct space is assumed to be real-valued
 * The data in reciprocal space is complex. 
 * direct_to_reciprocal indicates in which direction the FFT is to be calculated
 * 
 * Note that for real data in direct space (like here), we have
 * F(N-j) = conj(F(j)) where F is the array in reciprocal space.
 * Here, we do not make use of this property.
 * Also, we do not use the special (time-saving) routines of FFTW which
 * allow one to save time and memory for such real-to-complex transforms.
 *
 * f: array in direct space
 * F: array in reciprocal space
 * 
 * F(k) = \sum_{l=0}^{N-1} exp(- 2 \pi I k*l/N) f(l)
 * f(l) = 1/N \sum_{k=0}^{N-1} exp(+ 2 \pi I k*l/N) F(k)
 */

void fft_3d(fftw_dist_handler* fft, double *data_direct, fftw_complex* data_rec, bool direct_to_reciprocal) {

  double fac;
  int local_size_grid = fft->local_size_grid;
  fftw_complex * data = fft->data;
  fftw_complex * data_redistributed = fft->data_redistributed;
  
  int wsz;
  MPI_Comm_size(fft->mpi_comm, &wsz);
    
  // Now distinguish in which direction the FFT is performed
  if( direct_to_reciprocal ) {

    // Since fft->data is fftw_complex, we need to make data_direct complex
    for(int i = 0; i < local_size_grid; i++) data[i]  = data_direct[i] + 0.0 * I;

    // Perform the first fft on the n2-n3 plan locally
    fftw_execute(fft->fw_plan_2d);

    /* 
    * Perform all_to_all to have the data distributed along n2 direction
    * Alltoallw: generalized all-to-all communication allowing different datatypes, 
    * counts and displacements for each partner
    */
    MPI_Alltoallw(data, /*send buffer*/
                  fft->sendcounts, /*number of elements to send to each processor*/
                  fft->senddispls, /*send displacements*/
                  fft->sendtype, /*array of datatypes. Entry j specifies the type of data to send to process j*/
                  data_redistributed, /*receive buffer*/
                  fft->recvcounts, /*number of elements that can be received from each processor*/
                  fft->recvdispls, /*receive displacements*/
                  fft->recvtype, /*Entry i specifies the type of data received from process i*/
                  MPI_COMM_WORLD);

    // Perform fft on n1 direction
    fftw_execute(fft->fw_plan_1d);

    // Perform an Alltoall communication to get the data to the original ordering
    MPI_Alltoallw(data_redistributed,
                  fft->recvcounts,
                  fft->recvdispls,
                  fft->recvtype,
                  data,
                  fft->sendcounts,
                  fft->senddispls,
                  fft->sendtype,
                  MPI_COMM_WORLD);

    // Copy the data into the data_rec array
    memcpy(data_rec, data, fft->local_size_grid * sizeof(fftw_complex));

  } else { // basically the same thing but the bw plans are executed on the reciprocal space data
    // Copy the complex data_rec into the data array
    memcpy(data, data_rec, fft->local_size_grid * sizeof(fftw_complex));

    // Perform the reverse transform on n2 and n3
    fftw_execute(fft->bw_plan_2d);

    // Identical to the direct transformation
    MPI_Alltoallw(data, fft->sendcounts, fft->senddispls, fft->sendtype,
                  data_redistributed, fft->recvcounts, fft->recvdispls, fft->recvtype, MPI_COMM_WORLD);

    // Perform the reverse transform on n1
    fftw_execute(fft->bw_plan_1d);

    // Identical to the direct transformation
    MPI_Alltoallw(data_redistributed, fft->recvcounts, fft->recvdispls, fft->recvtype,
                  data, fft->sendcounts, fft->senddispls, fft->sendtype, MPI_COMM_WORLD);

    // Normalize the data
    fac = 1.0 / ( fft->global_size_grid );
    // creal returns the real part of a complex number
    for(int i = 0; i < fft->local_size_grid; ++i ) data_direct[i] = creal(data[i])*fac;
  }
}
