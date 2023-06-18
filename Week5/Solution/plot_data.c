#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

int FileExists(const char *filename) {    
    FILE *fp = fopen (filename, "r");
    if (fp!=NULL) fclose (fp);
    return (fp!=NULL);
}

void plot_data_2d(char* name, int n1, int n2, int n3, 
                    int n1_local, int n1_local_offset, 
                    int dir, double* data, 
                    int rank, int wsz) {
    int i1, i2, i3, i, index, owner = wsz + 1;
    int *sizes, *displ;
    double* buffer, *buffer1d, *local_buffer;
    FILE *fp;
    char buf[256];

    if ((n1/2 > n1_local_offset) &&  (n1/2 <= n1_local_offset + n1_local)) owner = rank;

    snprintf(buf, sizeof(buf), "%s.dat", name); 

    fp = fopen (buf, "w");
  /*
   * HINT: Assuming you sliced your system along i3, iif idir==1 or idir==2 you 
   *       need to take the correct slice of the plane from each process. If idir==3, you need  
   *       to understand which process holds the plane you are interested in. 
   */
    if (dir == 1) {
        i1 = n1/2 - 1; // OK
        if (rank == owner) {
            fp = fopen (buf, "w"); // OK
            for (i2 = 0; i2 < n2; ++i2) { // OK
                for (i3 = 0; i3 < n3; ++i3) { // OK
                    index = index_f(i1 - n1_local_offset, i2, i3, n2, n3); // MODIFIED
                    fprintf(fp, " %14.6f ", data[index] ); // OK
                }
                fprintf(fp, "\n"); // OK
            }
            fclose(fp);
        }

    } else if (dir == 2) {
        i2 = n2/2 - 1; // OK
        sizes = (int*)malloc(wsz * sizeof(int));
        displ = (int*)malloc(wsz * sizeof(int));
        buffer = (double*)malloc(n1 * n3 * sizeof(double));
        buffer1d = (double*)malloc(n1 * sizeof(double));
        local_buffer = (double*)malloc(n1_local * sizeof(double));

        // Gather the local sizes from the other processes
        MPI_Gather(&n1_local, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Compute the displacements based on the local sizes
        if (rank  ==  0) {
            displ[0] = 0;
            for (i = 1; i < wsz; i++) {
                displ[i] = sizes[i-1] + displ[i-1];
            }
        }

        for (i3 = 0; i3 < n3; ++i3) {
            for (i1 = 0; i1 < n1_local; ++i1) { 
                index = index_f(i1, i2, i3, n2, n3); // OK
                local_buffer[i1] = data[index]; // OK
            }

        // Gather the data
        MPI_Gatherv(local_buffer, n1_local, MPI_DOUBLE, buffer1d, sizes, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (i1 = 0; i1 < n1; i1++) buffer[i1 * n3 + i3] = buffer1d[i1];
        }

        if (rank == 0) { 
            fp = fopen (buf, "w");
            for (i1 = 0; i1 < n1; ++i1) {
                for (i3 = 0; i3 < n3; ++i3) {
                    fprintf(fp, " %14.6f ", buffer[i1*n3 + i3]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        free(sizes);
        free(displ);
        free(buffer);
        free(buffer1d);
        free(local_buffer);

    } else if (dir == 3) {

        i3 = n3/2 - 1; // OK

        sizes = (int*)malloc(wsz * sizeof(int));
        displ = (int*)malloc(wsz * sizeof(int));
        buffer = (double*)malloc(n1 * n3 * sizeof(double));
        buffer1d = (double*)malloc(n1 * sizeof(double));
        local_buffer = (double*)malloc(n1_local * sizeof(double));

        // Gather the local sizes from the other processes
        MPI_Gather(&n1_local, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Compute the displacements based on the local sizes
        if (rank  ==  0) {
            displ[0] = 0;
            for (i = 1; i < wsz; i++) {
                displ[i] = sizes[i-1] + displ[i-1];
            }
        }

        for (i2 = 0; i2 < n2; ++i2) {
            for (i1 = 0; i1 < n1_local; ++i1) {
                index = index_f(i1, i2, i3, n2, n3); // OK
                local_buffer[i1] = data[index]; // OK
            }

            // Gather the data
            MPI_Gatherv(local_buffer, n1_local, MPI_DOUBLE, buffer1d, sizes, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (i1 = 0; i1 < n1; i1++) buffer[i1 * n2 + i2] = buffer1d[i1];
        }

        if (rank == 0) {
            fp = fopen (buf, "w");
            for (i1 = 0; i1 < n1; ++i1) {
                for (i2 = 0; i2 < n2; ++i2) {
                    fprintf(fp, " %14.6f ", buffer[i1*n2 + i2]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        free(sizes);
        free(displ);
        free(buffer);
        free(buffer1d);
        free(local_buffer);

    } else fprintf(stderr, " Wrong value for argument 5 in plot_data_2d \n");
}
