#include <stdbool.h>
#include <string.h>
#include "utilities.h"

double seconds(){
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

int index_f ( int i1, int i2, int i3, int n1, int n2, int n3){

  return n3*n2*i1 + n3*i2 + i3; 
}

void init_fftw(fftw_mpi_handler *fft, int n1, int n2, int n3, MPI_Comm comm){

  fftw_mpi_init();
  fft->mpi_comm = comm;

  fft->global_size_grid = n1 * n2 * n3;
  fft->local_size_grid = fftw_mpi_local_size_3d( n1, n2, n3, fft->mpi_comm, &fft->local_n1, &fft->local_n1_offset);
  fft->fftw_data = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof( fftw_complex ) );

  fft->fw_plan = fftw_mpi_plan_dft_3d( n1, n2, n3, fft->fftw_data, fft->fftw_data, fft->mpi_comm, FFTW_FORWARD, FFTW_ESTIMATE );
  fft->bw_plan = fftw_mpi_plan_dft_3d( n1, n2, n3, fft->fftw_data, fft->fftw_data, fft->mpi_comm, FFTW_BACKWARD, FFTW_ESTIMATE );

}

void close_fftw( fftw_mpi_handler *fft ){

    fftw_destroy_plan( fft->bw_plan );
    fftw_destroy_plan( fft->fw_plan );
    fftw_free( fft->fftw_data );
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
 * 
 */

void fft_3d( fftw_mpi_handler* fft, double *data_direct, fftw_complex* data_rec, bool direct_to_reciprocal ){

    double fac;
    int i;
    
    if( direct_to_reciprocal ){
     
      for(i = 0; i < fft->local_size_grid; i++){

      	  fft->fftw_data[i]  = data_direct[i] + 0.0 * I;
      	}
      
      fftw_mpi_execute_dft( fft->fw_plan, fft->fftw_data, fft->fftw_data );
            
      memcpy( data_rec, fft->fftw_data, fft->local_size_grid * sizeof(fftw_complex) );

    }
    else{
     
      memcpy(fft->fftw_data, data_rec, fft->local_size_grid * sizeof(fftw_complex) );
      
      fftw_mpi_execute_dft(fft->bw_plan, fft->fftw_data, fft->fftw_data);
     
      fac = 1.0 / fft->global_size_grid;
      
      for( i = 0; i < fft->local_size_grid; ++i ){
	
	data_direct[i] = creal( fft->fftw_data[i] ) * fac;
      }
    }
}

