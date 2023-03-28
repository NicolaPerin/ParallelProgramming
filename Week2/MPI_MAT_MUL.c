#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "utility.h"

#ifdef USE_BLAS
#include <cblas.h>
#endif

int main(int argc, char** argv) {

    MPI_Init( &argc, &argv );

    int N = atoi(argv[1]); // get the size of the matrix as cmd line arg
    int world_size, rank, n_loc, rest; // MPI stuff
    int *row_counts, *displ_B, *elem_counts, *displ_B_col; // how many rows per process

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *A, *B, *C, *B_col; // declare the three matrices + the receive buffer

    n_loc = N / world_size; // ex: 10 / 4 = 2
    rest = N % world_size; // ex: 10 % 4 = 2

    if (rank == 0) { printf("\nN: %d, world_size: %d, n_loc: %d, rest: %d\n", N, world_size, n_loc, rest); }

    row_counts = (int *) malloc( world_size * sizeof(int) );
    init_row_counts(row_counts, n_loc, rest, rank, world_size); // ex: 3 3 2 2

    // Allocate the space for the matrices, each process now knows how many rows it has
    A = (double *) malloc( row_counts[rank] * N * sizeof(double) );
    B = (double *) malloc( row_counts[rank] * N * sizeof(double) );
    C = (double *) malloc( row_counts[rank] * N * sizeof(double) );
    // Our block of columns of B has to be large enough to store the larger blocks -> N * (n_loc + 1)
    B_col = (double *) malloc( N * (n_loc + 1) * sizeof(double) );

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
        init_elem_counts(elem_counts, row_counts, p, world_size); // Must change with p, not the rank!!!
        init_displ_B_col(displ_B_col, elem_counts, world_size); // Same
        
        MPI_Datatype my_block;
        MPI_Type_vector(row_counts[rank], row_counts[p], N, MPI_DOUBLE, &my_block);
        MPI_Type_commit(&my_block);
        MPI_Allgatherv(B + displ_B[p], 1, my_block, B_col, elem_counts, displ_B_col, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Type_free(&my_block);

#ifdef USE_BLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_counts[rank], 
                    row_counts[p], N, 1.0, A, N, B_col, row_counts[p], 0.0, C + displ_B[p], N);
#else
        matrixMultiply(C, A, B_col, row_counts, displ_B, rank, p, N); // Finally
#endif
    }

    // Print the final matrix
    if (rank == 0) {
        printf("\nC\n");
        printMatrix(C, row_counts[rank], N);
        for (int count = 1; count < world_size; count++) {
            MPI_Recv(C, row_counts[count] * N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printMatrix(C, row_counts[count], N);
        }
    } else MPI_Send(C, row_counts[rank] * N, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

    free(A); free(B); free(C); free(B_col);
    free(row_counts); free(displ_B); free(elem_counts); free(displ_B_col);

    MPI_Finalize();
    return 0;
}