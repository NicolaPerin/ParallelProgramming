/* 
 * This code calcutes the solution of the diffusion equaution in 3D, using time forward 
 * discretization for the time derivative, and using discrete fourier transform to calculate
 * spatial derivatives. 
 * 
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "utilities.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

int main(){

    // Dimensions of the system
    double L1 = 10., L2 = 10., L3 = 20.;
    // Grid size
    int n1 = 48, n2 = 48, n3 = 96;
    // time step for time integration
    double dt = 2.e-3; 
    // number of time steps
    int nstep = 101; 
    // Radius of diffusion channel
    double rad_diff = 0.7;
    // Radius of starting concentration
    double rad_conc = 0.6;
    double start, end;
  
    double *diffusivity, *conc, *dconc, *aux1, *aux2;
   
    int i1, i2, i3, ipol, istep, index;
  
    double f1conc, f2conc, f3conc, f1diff, f2diff, f3diff, fac, ss;
    double x1, x2 , x3, rr, r2mean;
    fftw_handler fft_h;

    diffusivity = (double*)malloc(n1*n2*n3*sizeof(double));
    conc = (double*)malloc(n1*n2*n3*sizeof(double));
    dconc = (double*)malloc(n1*n2*n3*sizeof(double));
    aux1 = (double*)malloc(n1*n2*n3*sizeof(double));
    aux2 = (double*)malloc(n1*n2*n3*sizeof(double));

 // 
 // Define the diffusivity inside the system and 
 // the starting concentration
 //
 // ss is to integrate (and normalize) the concentration
 // 

    ss = 0.0;

    for (i3 = 0; i3 < n3; ++i3)
      {  
	x3=L3*((double)i3)/n3;
	f3diff = exp( -pow((x3-0.5*L3)/rad_diff,2));
	f3conc = exp( -pow((x3-0.5*L3)/rad_conc,2));
	
	
        for (i2 = 0; i2 < n2; ++i2)
	  {
            x2=L2*((double)i2)/n2;
            f2diff = exp( -pow((x2-0.5*L2)/rad_diff,2));
            f2conc = exp( -pow((x2-0.5*L2)/rad_conc,2));
	    
	
	    for (i1 = 0; i1 < n1; ++i1)
	      {
		x1=L1*((double)i1)/n1;
		f1diff = exp( -pow((x1-0.5*L1)/rad_diff,2));
		f1conc = exp( -pow((x1-0.5*L1)/rad_conc,2));
		
		index = index_f(i1, i2, i3, n1, n2, n3);
		diffusivity[index]  = MAX( f1diff * f2diff, f2diff * f3diff);
		conc[index] = f1conc * f2conc * f3conc;
		ss += conc[index]; 
		
	      }   
	  }
      }
    
    plot_data_2d("diffusivity", n1, n2, n3, 1, diffusivity);
    plot_data_2d("diffusivity", n1, n2, n3, 2, diffusivity);
    plot_data_2d("diffusivity", n1, n2, n3, 3, diffusivity);
    
    
    fac= L1*L2*L3/(n1*n2*n3);
  // Now normalize the concentration
    ss = 1.0/(ss*fac);
    for (i1=0; i1< n1*n2*n3; ++i1)
      conc[i1]*=ss;
      
   // initialize the fftw system 
    init_fftw(&fft_h, n1, n2, n3);
   
  // Now everything is defined: system size, diffusivity inside the system, and
  // the starting concentration
  //
  // Start the dynamics
  //
 
    start = seconds();
    for (istep = 1; istep <= nstep; ++istep)
      {
        for (i1=0; i1< n1*n2*n3; ++i1)
	  dconc[i1] = 0.0;
        for (ipol =1; ipol<=3; ++ipol )
	  {
	    derivative(&fft_h, n1, n2, n3, L1, L2, L3, ipol, conc, aux1);
	    for (i1=0; i1< n1*n2*n3; ++i1)
	      {
                aux1[i1] *= diffusivity[i1];
	      }
	    derivative(&fft_h, n1, n2, n3, L1, L2, L3, ipol, aux1, aux2);
            // summing up contributions from the three spatial directions
            for (i1=0; i1< n1*n2*n3; ++i1)
	      dconc[i1] += aux2[i1];
	  } 
        for (i1=0; i1< n1*n2*n3; ++i1)
	  conc[i1] += dt*dconc[i1];
	
        if (istep%30 == 1)
	  {
            // Check the normalization of conc
             ss = 0.;
            r2mean = 0.;
            for (i3 = 0; i3 < n3; ++i3)
	      {
                x3=L3*((double)i3)/n3 - 0.5*L3;
                for (i2 = 0; i2 < n2; ++i2)
		  {
                    x2=L2*((double)i2)/n2 - 0.5*L2;
                    for (i1 = 0; i1 < n1; ++i1)
		      {
			x1=L1*((double)i1)/n1 - 0.5*L1;
			rr = pow( x1, 2)  + pow( x2, 2) + pow( x3, 2);
			index = index_f(i1, i2, i3, n1, n2, n3); 
			ss += conc[index]; 
			r2mean += conc[index]*rr;
		      }   
		  }
	      }
            ss *= fac;
            r2mean *= fac;
            end = seconds();
            printf(" %d %17.15f %17.15f Elapsed time per iteration %f \n", istep, r2mean, ss, (end-start)/istep);

            plot_data_2d("concentration", n1, n2, n3, 2, conc);
            plot_data_1d("1d_conc", n1, n2, n3, 3, conc);
	  }
	
      } 
    
    close_fftw(&fft_h);
    free(diffusivity);
    free(conc);
    free(dconc);
    free(aux1);
    free(aux2);

    return 0;
} 
