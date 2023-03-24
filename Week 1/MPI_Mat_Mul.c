#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

void printMatrix(const double* A, const int n_loc, const int N) {

  for (int i = 0; i < n_loc; i++ ) {
      for (int j = 0; j < N; j++ ) {
          fprintf( stdout, "%.3g ", A[i*N + j] );
      }
      fprintf( stdout, "\n");
  }
}

void initCounts(int *scounts, int *displ_send, const int world_size,
                const int rank, const int n_loc, const int rest) {
    if (rank == 0) { printf("scounts: "); }
    for (int i = 0; i < world_size; i++) {
        if (i < rest) { scounts[i] = n_loc + 1; }
        else { scounts[i] = n_loc; }
        if (rank == 0) { printf("%d ", scounts[i]); }
    }

    if (rank == 0) { printf("\ndispl_send: 0 "); }
    displ_send[0] = 0;
    for (int i = 0; i < world_size - 1; i++) {
        displ_send[i+1] = displ_send[i] + scounts[i];
        if (rank == 0) { printf("%d ", displ_send[i+1]); }
    }
}

int main(int argc, char** argv) {

    MPI_Init( &argc, &argv );

    int N = atoi(argv[1]); // get the size of the matrix as cmd line arg
    int *scounts, *displ_send, *displ_rec, *rcounts;
    double *A, *B, *C, *B_loc;

    MPI_Datatype my_block;
    int world_size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_loc = N / world_size;
    int rest = N % world_size;
    int offset = 0;

    if (rank == 0) { printf("\nN: %d, world_size: %d, n_loc: %d, rest: %d\n", N, world_size, n_loc, rest); }

    scounts = (int *) malloc( world_size * sizeof(int) );
    rcounts = (int *) malloc( world_size * sizeof(int) );
    displ_send = (int *) malloc( world_size * sizeof(int) );
    displ_rec = (int *) malloc( world_size * sizeof(int) );

    initCounts(scounts, displ_send, world_size, rank, n_loc, rest);

    if (rank < rest) { n_loc++; offset = rank * n_loc * N; }
    else { offset = rest * (n_loc + 1) * N + (rank - rest) * n_loc * N; }

    // Compute the number of elements in the receive buffer for each process as #n of columns * #n of rows
    if (rank == 0) { printf("\nrcounts: "); }
    for (int i = 0; i < world_size; i++) { rcounts[i] = scounts[i] * n_loc;
                                            if (rank == 0) { printf("%d ", rcounts[i]); }}
    // Compute where to insert the received elements in the buffer
    if (rank == 0) { printf("\ndispl_rec: 0 "); }
    displ_rec[0] = 0;
    for (int i = 0; i < world_size - 1; i++) { displ_rec[i+1] = displ_rec[i] + rcounts[i];
                                            if (rank == 0) { printf("%d ", displ_rec[i+1]); }}

    if (rank == 0) { printf("\n------------\n"); }

    // initialize A with increasing integer numbers (no need to use memset in this case)
    A = (double *) malloc( N * n_loc * sizeof(double) ); // rectangular chunk of N columns and n_loc rows
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < N; j++) {
            int index = i*N + j;
            A[index] = offset + index + 1.0;
        }
    }

    if (rank < rest) offset = 0; // reset offset to initalize B
    else offset = rest;
    // initialize B as the identity (as in: MPI_Identity.c)
    B = (double *) malloc( N * n_loc * sizeof(double) );
    memset( B, 0.0, N * n_loc * sizeof(double) );
    for (int i = 0; i < n_loc; i++) {
        int j = i + ( n_loc * rank ) + offset;
        B[ j + ( i * N ) ] = 1.0;
    }

    // Initialize C as zero
    C = (double *) malloc( N * n_loc * sizeof(double) );
    memset( C, 0.0, N * n_loc * sizeof(double) );

    // initialize the receive buffer for B
    B_loc = (double *) malloc( n_loc * N * sizeof(double) );
    memset( B_loc, 0, n_loc * N * sizeof(double) );

    if (rank == 0) { printf("\nFin qui tutto bene (inizializzazione)\n"); } // Debug

    for (int p = 0; p < world_size; p++) {

        // the parallelism between the processes is along the blocks of rows and each block has n_loc rows,
        // (n_loc has previously been increased by 1 if rank < rest) so we use n_loc as #n of rows per block
        // the loop is over the blocks of columns, so the number of columns per block must depend on the iterator
        // we used scounts to store the number of elements per column, so it's scounts[p]
        MPI_Type_vector(n_loc, scounts[p], N, MPI_DOUBLE, &my_block);
        MPI_Type_commit(&my_block);
        if (rank == p) { printf("\nFin qui tutto bene (commit datatype)\n"); }
        MPI_Allgatherv(B + displ_send[p], 1, my_block, B_loc, rcounts, displ_rec, MPI_DOUBLE, MPI_COMM_WORLD);
        if (rank == p) { printf("\nFin qui tutto bene (allgather)\n"); }
        if (rank == p) {
            printf("\nblock of columns\n");
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < n_loc; j++) {
                    fprintf( stdout, "%.3g ", B_loc[i*n_loc + j] );
                }
                fprintf(stdout, "\n");
            }
        }

        // Do the multiplication (elements are accessed using linear indexing)
        for (int i = 0; i < n_loc; i++) { // A is n_loc x N
            for (int j = 0; j < n_loc; j++) { // B_loc is N x n_loc
                for (int k = 0; k < N; k++) { // C_p is n_loc x n_loc -> C is n_loc x N (same as A)
                        C[i * N + j + p * n_loc] += A[i * N + k ] * B_loc[k * n_loc + j];
                }
            }
        }
    }

   // Print the final matrix
   if (rank == 0) {
        printf("\nC\n");
        printMatrix(C, n_loc, N);
        for (int count = 1; count < world_size; count++) {
            if (count == rest) n_loc = n_loc - 1;
            MPI_Recv(C, n_loc * N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printMatrix(C, n_loc, N);
        }
    } else MPI_Send(C, n_loc * N, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

    free(scounts);
    free(rcounts);
    free(displ_send);
    free(A);
    free(B);
    free(C);
    MPI_Type_free(&my_block);

    MPI_Finalize();
    return 0;
}

/*
if (rank == 0) {
        printf("\nB\n");
        printMatrix(B, n_loc, N);
        for (int count = 1; count < world_size; count++) {
            if (count == rest) n_loc = n_loc - 1;
            MPI_Recv(B, n_loc * N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printMatrix(B, n_loc, N);
        }

    } else MPI_Send(B, n_loc * N, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
*/