#include <stdbool.h>
#include "utilities.h"

void derivative( fftw_mpi_handler* fft, int n1, int n2, int n3, double L1, double L2, double L3, int ipol, double* data, double* deriv ) {

    fftw_complex *aux;
    double G;
    int i, i1, i2, i3, index;

    aux = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof(fftw_complex) );

    // First get the FFT of data
    fft_3d( fft, data, aux, true );

    if( ipol == 1 ){
	
      G = 2.0 * pi / L1;
      for( i1 = 0; i1 < fft->local_n1; ++i1 ){
	
    	i = i1 + fft->local_n1_offset;
    	if( i > n1/2 ) i = i - n1;
    	if( i == n1/2 ) i = 0;
	
    	for( i2 = 0; i2 < n2; ++i2 ){
    	  for( i3 = 0; i3 < n3; ++i3 ){
    	    index = index_f( i1, i2, i3, fft->local_n1, n2, n3 );
    	    aux[index] *= 0.0 + G * i * I;
    	  }
    	}
      }
    }
    
    if( ipol == 2 ){
      
      G = 2.0 * pi / L2;
      for( i2 = 0; i2 < n2; ++i2 ){
	
    	i = i2;
    	if( i > n2/2 ) i = i -n2;
    	if( i == n2/2 ) i = 0;
	
    	for( i1 = 0; i1 < fft->local_n1; ++i1 ){
    	  for( i3 = 0; i3 < n3; ++i3 ){
    	    index = index_f( i1, i2, i3, fft->local_n1, n2, n3 );
    	    aux[index] *= 0.0 + G * i * I;
    	  }
    	}
      }
    }
      
    if( ipol == 3 ){
      
      G = 2.0 * pi / L3;
      for( i3 = 0; i3 < n3; ++i3 ){
	  
    	i = i3;
    	if( i > n3/2 ) i = i -n3;
    	if( i == n3/2 ) i = 0;
	
    	for( i1 = 0; i1 < fft->local_n1; ++i1 ){
    	  for( i2 = 0; i2 < n2; ++i2 ){
    	    index = index_f(i1, i2, i3, fft->local_n1, n2, n3);
    	    aux[index] *= 0.0 + G * i * I;
    	  }
    	}
      }
    }
    
    // Now go back to real space
    fft_3d( fft, deriv, aux, false);
    fftw_free(aux);
}
