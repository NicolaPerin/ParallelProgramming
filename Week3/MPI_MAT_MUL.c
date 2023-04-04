#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "utility.h"

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_GPU
#include "gpu.cu"
#endif

int main(int argc, char** argv) {

    MPI_Init( &argc, &argv );

    int N = atoi(argv[1]); // get the size of the matrix as cmd line arg
    int world_size, rank, n_loc, rest; // MPI stuff
    int *row_counts, *displ_B, *elem_counts, *displ_B_col; // how many rows per process

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double computation = 0, communication = 0, time1, time2, time3;
    double *A, *B, *C, *B_col; // declare the three matrices + the receive buffer

    n_loc = N / world_size; // ex: 10 / 4 = 2
    rest = N % world_size; // ex: 10 % 4 = 2

    if (rank == 0) { printf("\nN: %d, world_size: %d, n_loc: %d, rest: %d\n", N, world_size, n_loc, rest); }

#ifdef USE_CBLAS
    if (rank == 0) printf("USING DGEMM!\n");
#elif USE_GPU
    if (rank == 0) printf("USING CUBLAS_DGEMM!\n");
#endif

    row_counts = (int *) malloc( world_size * sizeof(int) );
    init_row_counts(row_counts, n_loc, rest, rank, world_size); // ex: 3 3 2 2

    // Allocate the space for the matrices, each process now knows how many rows it has
    A = (double *) malloc( row_counts[rank] * N * sizeof(double) );
    B = (double *) malloc( row_counts[rank] * N * sizeof(double) );
    C = (double *) malloc( row_counts[rank] * N * sizeof(double) );
    // Our block of columns of B has to be large enough to store the larger blocks -> N * (n_loc + 1)
    B_col = (double *) malloc( N * (n_loc + 1) * sizeof(double) );

#ifdef USE_GPU
    double *d_A, *d_B_col, *d_C;
    gpu_initialization(row_counts, n_loc, rank, N, d_A, d_B_col, d_C);
#endif

    // Initalization of the matrices
    init_A( A, row_counts[rank], N, rank, rest ); // Increasing integer numbers
    init_B( B, row_counts[rank], N, rank, rest ); // Identity
    memset( C, 0, row_counts[rank] * N * sizeof(double) ); // Zero

    displ_B = (int *) malloc( world_size * sizeof(int) );
    init_displ_B(displ_B, row_counts, rank, world_size); // ex: 0 3 6 8
    elem_counts = (int *) malloc(  world_size * sizeof(int) );
    displ_B_col = (int *) malloc ( world_size * sizeof(int) );

    // Loop over the number of processes: create the datatype and gather the little blocks into every B_col
    for (int p = 0; p < world_size; p++) {

        MPI_Barrier(MPI_COMM_WORLD); // Force syncronization
        time1 = MPI_Wtime();

        // Initialize displacements and counts, create datatype, Allgatherv
        MPICalls(row_counts, displ_B, elem_counts, displ_B_col, B, B_col, rank, p, N, world_size);

        MPI_Barrier(MPI_COMM_WORLD);
        time2 = MPI_Wtime();

#ifdef USE_CBLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_counts[rank],
                    row_counts[p], N, 1.0, A, N, B_col, row_counts[p], 0.0, C + displ_B[p], N);
#elif USE_GPU
        gpu_computation(rank, p, n_loc, N, row_counts, displ_B, A, B_col, C, d_A, d_B_col, d_C);
#else
        matrixMultiply(C, A, B_col, row_counts, displ_B, rank, p, N); // Finally
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        time3 = MPI_Wtime();

        communication += time2 - time1;
        computation += time3 - time2;
    }

    if (rank == 0) printf("%.5g %.5g\n", communication, computation);

    free(A); free(B); free(C); free(B_col);
    free(row_counts); free(displ_B); free(elem_counts); free(displ_B_col);

    MPI_Finalize();
    return 0;
}
/*// Print the final matrix
    if (rank == 0) {
        printf("\nC\n");
        printMatrix(C, row_counts[rank], N);
        for (int count = 1; count < world_size; count++) {
            MPI_Recv(C, row_counts[count] * N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printMatrix(C, row_counts[count], N);
        }
    } else MPI_Send(C, row_counts[rank] * N, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);*/
