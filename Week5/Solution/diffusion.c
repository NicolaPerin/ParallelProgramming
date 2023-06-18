#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

int main(int argc, char* argv[]) {

    double L1 = 10., L2 = 10., L3 = 20.; // Dimensions of the system
    int n1 = atoi(argv[3]), n2 = atoi(argv[4]), n3 = atoi(argv[5]); // Grid size
    double dt = atof(argv[1]); // time step for time integration
    int nstep = atoi(argv[2]); // number of time steps
    int interval = 10;
    double rad_diff = 0.7; // Radius of diffusion channel
    double rad_conc = 0.6; // Radius of starting concentration
    double start, end;

    double *diffusivity, *concentration, *dconc, *aux1, *aux2;

    int ii, i1, i2, i3, ipol, istep, index;

    double f1conc, f2conc, f3conc, f1diff, f2diff, f3diff, fac;
    double x1, x2 , x3, rr;
    double ss, r2mean, global_ss, global_r2mean;

    fftw_dist_handler fft_h;
    int rank, wsz;
    int n1_local, n1_local_offset, local_size_grid, global_size_grid;

    // Initialize MPI environment
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &wsz );

    // Initialize the fftw system and local dimension as the value returned from the parallel FFT grid initializzation
    init_fftw(&fft_h, n1, n2, n3, MPI_COMM_WORLD);
    n1_local = fft_h.local_n1;
    n1_local_offset = fft_h.local_n1_offset;
    local_size_grid = fft_h.local_size_grid;
    global_size_grid = fft_h.global_size_grid;

    // Allocate distributed memory arrays
    diffusivity = (double*) malloc(local_size_grid * sizeof(double));
    concentration = (double*) malloc(local_size_grid * sizeof(double));
    dconc = (double*) malloc(local_size_grid * sizeof(double));
    aux1 = (double*) malloc(local_size_grid * sizeof(double));
    aux2 = (double*) malloc(local_size_grid * sizeof(double));

    // Define the diffusivity inside the system and the starting concentration
    ss = 0.0; // ss is to integrate (and normalize) the concentration

    for (i3 = 0; i3 < n3; ++i3) {

        x3 = L3 * ((double)i3) / n3;
        f3diff = exp(-pow((x3 - 0.5 * L3) / rad_diff, 2));
        f3conc = exp(-pow((x3 - 0.5 * L3) / rad_conc, 2));

        for (i2 = 0; i2 < n2; ++i2) {

            x2 = L2 * ((double)i2) / n2;
            f2diff = exp(-pow((x2 - 0.5 * L2) / rad_diff, 2));
            f2conc = exp(-pow((x2 - 0.5 * L2) / rad_conc, 2));

            for (i1 = 0; i1 < n1_local; ++i1) {

                x1 = L1 * ((double) (i1 + n1_local_offset)) / n1;
                f1diff = exp(-pow((x1 - 0.5 * L1) / rad_diff, 2));
                f1conc = exp(-pow((x1 - 0.5 * L1) / rad_conc, 2));
                index = index_f(i1, i2, i3, n2, n3); // removed n1_local
                diffusivity[index] = MAX(f1diff * f2diff, f2diff * f3diff);
                concentration[index] = f1conc * f2conc * f3conc;
                ss += concentration[index];
            }
        }
    }

    // Plot the diffusivity in all 3 directions
    plot_data_2d("diffusivity1", n1, n2, n3, n1_local, n1_local_offset, 1, diffusivity, rank, wsz);
    plot_data_2d("diffusivity2", n1, n2, n3, n1_local, n1_local_offset, 2, diffusivity, rank, wsz);
    plot_data_2d("diffusivity3", n1, n2, n3, n1_local, n1_local_offset, 3, diffusivity, rank, wsz);

    /*
     * Now normalize the concentration (DONE!)
     * the global ss must be computed and propagated to all processes
     * ss = 1.0/(ss*fac);
     */
    fac = L1 * L2 * L3 / global_size_grid;
    MPI_Allreduce(&ss, &global_ss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_ss = 1.0 / (global_ss * fac);

    for(ii = 0; ii < local_size_grid; ++ii) concentration[ii] *= global_ss;

    // Parallelize the output routines
    plot_data_2d("concentration_init", n1, n2, n3, n1_local, n1_local_offset, 2, concentration, rank, wsz);
    /*
     * Now everything is defined: system size, diffusivity inside the system, and the starting concentration
     * Start the dynamics
     */

    start = seconds();
    // I initialize dconc
    for (ii = 0; ii < local_size_grid; ++ii) dconc[ii] = 0.0;

    for (istep = 1; istep <= nstep; ++istep) {
        // I need to make the transform in all 3 direction and sum the contribution

        for (ipol = 1; ipol <= 3; ++ipol) {

            derivative(&fft_h, n1, n2, n3, L1, L2, L3, ipol, concentration, aux1);
            for (ii = 0; ii < local_size_grid; ++ii) aux1[ii] *= diffusivity[ii];
            derivative(&fft_h, n1, n2, n3, L1, L2, L3, ipol, aux1, aux2);
            // summing up contributions from the three spatial directions
            for (ii = 0; ii < local_size_grid; ++ii) dconc[ii] += aux2[ii];
        }

        // I update the concentration array and reset dconc
        for (ii = 0; ii < local_size_grid; ++ii) {
            concentration[ii] += dt * dconc[ii];
            dconc[ii] = 0.0; // Perché l'ha messo qui poi
        }

        if (istep % 30 == 1) {
            // Check the normalization of concentration
            ss = 0.; r2mean = 0.; global_ss = 0.; global_r2mean = 0.;

            // The concentration array is distributed, so only a part of it is on each processor
            for (i3 = 0; i3 < n3; ++i3) {
                x3 = L3 * ((double)i3) / n3 - 0.5 * L3;
                for (i2 = 0; i2 < n2; ++i2) {
                    x2 = L2 * ((double)i2) / n2 - 0.5 * L2;

                    for (i1 = 0; i1 < n1_local; ++i1) {
                        x1 = L1 * ((double) (i1 + n1_local_offset)) / n1 - 0.5 * L1;
                        rr = pow(x1, 2) + pow(x2, 2) + pow(x3, 2);
                        index = index_f(i1, i2, i3, n2, n3); // Removed n1_local
                        ss += concentration[index];
                        r2mean += concentration[index] * rr;
                    }
                }
            }
            // Global values of ss and r2mean must be globally computed and distributed to all processes
            MPI_Allreduce(&ss, &global_ss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&r2mean, &global_r2mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            global_ss *= fac;
            global_r2mean *= fac;

            end = seconds();
            if( rank == 0 ) printf(" %d %17.15f %17.15f Elapsed time per iteration %f \n ", istep, global_r2mean, global_ss, ( end - start ) / istep );
            // Use parallel version of output routines
            char title[80]; // Title of the plot
            sprintf(title, "concentration_%d", 1 + (istep - 1) / interval);
            plot_data_2d(title, n1, n2, n3, n1_local, n1_local_offset, 2, concentration, rank, wsz);
        }
    }

    close_fftw(&fft_h);
    free(diffusivity);
    free(concentration);
    free(dconc);
    free(aux1);
    free(aux2);

    MPI_Finalize();
    return 0;
}
