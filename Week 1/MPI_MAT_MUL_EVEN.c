#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

void print_matrix(double* A, int n_loc, int N) {

  for (int i = 0; i < n_loc; i++ ) {
      for (int j = 0; j < N; j++ ) {
          fprintf( stdout, "%.3g ", A[i*N + j] );
      }
      fprintf( stdout, "\n");
  }
}

int main(int argc, char** argv) {

    MPI_Init( &argc, &argv );

    int N = atoi(argv[1]); // get the size of the matrix as cmd line arg
    int i_loc = 0, j_glob = 0, offset = 0;
    double *A, *B, *C, *B_loc;

    MPI_Datatype my_block;

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n_loc = N / world_size;
    int rest = N - world_size * n_loc; // equivalently: N % world_size

    if (world_rank == 0) printf("N: %d, world_size: %d, n_loc: %d, rest: %d\n", N, world_size, n_loc, rest);

    // initialize A with increasing integer numbers (no need to use memset in this case)
    A = (double *) malloc( N * n_loc * sizeof(double) ); // rectangular chunk of N columns and n_loc rows
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (i + n_loc * world_rank) * N + j + 1.0;
        }
    }

    // initialize B as the identity (as in: MPI_Identity.c)
    B = (double *) malloc( N * n_loc * sizeof(double) );
    memset( B, 0.0, N * n_loc * sizeof(double) );
    for (int i = 0; i < n_loc; i++) {
        j_glob = i + ( n_loc * world_rank );
        B[ j_glob + ( i * N ) ] = 1.0;
    }
    
    // Initialize C as zero
    C = (double *) malloc( N * n_loc * sizeof(double) );
    memset( C, 0.0, N * n_loc * sizeof(double) );
    
    /*
    The rows of A re local to each process
    The columns of B are NOT local to each process (-> communication!)
    The elements of C are local to each process

    Iteratively multiply a row block by a column block -> the column block has to be shared among the processes
    Every process sends its part of the column block, everybody has to see it -> allgather

    Since B is the identity, C = AB = A -> check that A and C are the same!
    */

    MPI_Type_vector(n_loc, n_loc, N, MPI_DOUBLE, &my_block); // n_loc rows of n_loc elements with a stride equal to N
    MPI_Type_commit(&my_block);

    // initialize the receive buffer for B
    B_loc = (double *) malloc( n_loc * N * sizeof(double) );
    memset( B_loc, 0, n_loc * N * sizeof(double) );

    for (int p = 0; p < world_size; p++) {
        MPI_Allgather(&B[n_loc * p], 1, my_block, B_loc, n_loc*n_loc, MPI_DOUBLE, MPI_COMM_WORLD); /*
        if (world_rank == p) { // check that the blocks of columns are correct
            printf("\nblock of columns\n");
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < n_loc; j++) {
                    fprintf( stdout, "%.3g ", B_loc[i*n_loc + j] );
                }
                fprintf(stdout, "\n");
            }
        } */

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
   if (world_rank == 0) {
        printf("\nC\n");
        print_matrix(C, n_loc, N);
        for (int count = 1; count < world_size; count++) {
            MPI_Recv(C, n_loc * N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_matrix(C, n_loc, N);
        }
    } else MPI_Send(C, n_loc * N, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD);


    free(A);
    free(B);
    free(C);
    MPI_Type_free(&my_block);
    MPI_Finalize();
    return 0;
}