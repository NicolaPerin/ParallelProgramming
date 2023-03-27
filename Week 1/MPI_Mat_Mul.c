#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

void printMatrix(const double* M, const int rows, const int cols) {
  for (int i = 0; i < rows; i++ ) {
      for (int j = 0; j < cols; j++ ) {
          fprintf( stdout, "%.3g ", M[i*cols + j] );
      }
      fprintf( stdout, "\n");
  }
}

void init_row_counts(int *row_counts, const int n_loc, const int rest, const int rank, const int world_size) {

    if (rank == 0) { printf("row_counts: "); }
    for (int i = 0; i < world_size; i++) {
        if (i < rest) { row_counts[i] = n_loc + 1; }
        else { row_counts[i] = n_loc; }
        if (rank == 0) { printf("%d ", row_counts[i]); }
    }
    if (rank == 0) { printf("\n"); }
}

void init_displ_B(int *displ_B, const int *row_counts, const int rank, const int world_size) {

    if (rank == 0) { printf("displ_B: 0 "); }
    displ_B[0] = 0;
    for (int i = 0; i < world_size - 1; i++) {
        displ_B[i+1] = displ_B[i] + row_counts[i]; // cumulative sum
        if (rank == 0) { printf("%d ", displ_B[i+1]); }
    }
    if (rank == 0) { printf("\n"); }
}

void init_elem_counts(int *elem_counts, const int *row_counts, const int p, const int world_size) {
    for (int i = 0; i < world_size; i++) {
        elem_counts[i] = row_counts[i] * row_counts[p];
    }
}

void init_displ_B_col(int *displ_B_col, const int *elem_counts, const int world_size) {
    displ_B_col[0] = 0;
    for (int i = 0; i < world_size - 1; i++) {
        displ_B_col[i+1] = displ_B_col[i] + elem_counts[i];
    }
}

void init_A(double *A, const int rows, const int cols, const int rank, const int rest) {
    double offset;
    if (rank < rest) offset = rank * rows * cols;
    else offset = rest * (rows + 1.0) * cols + (rank - rest) * rows * cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i*cols + j;
            A[index] = offset + index + 1.0;
        }
    }
}

void init_B(double *B, const int rows, const int cols, const int rank, const int rest) {
    double offset;
    if (rank < rest) offset = 0;
    else offset = rest;

    memset( B, 0, rows * cols * sizeof(double) );
    for (int i = 0; i < rows; i++) {
        int j = i + ( rows * rank ) + offset;
        B[ j + ( i * cols ) ] = 1.0;
    }
}

// Do the multiplication (elements are accessed using linear indexing)
void matrixMultiply(double *C, const double *A, const double *B_col, const int *row_counts,
                    const int* displ_B, const int rank, const int p, const int N) {
    for (int i = 0; i < row_counts[rank]; i++) { // A is row_counts[rank] x N
        for (int j = 0; j < row_counts[p] ; j++) { // B_loc is N x row_counts[rank]
            for (int k = 0; k < N; k++) { // C_p is row_counts[rank] x row_counts[rank] -> C is n_loc x N (same as A)
                    C[i * N + j + displ_B[p]] += A[i * N + k] * B_col[k* row_counts[p] + j];
            }
        }
    }
}

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

    /*
    Each process initializes a block of rows of A, B and C.
    The number of rows in general is not equal for all processes:
    we redistribute rest rows among the first rest processes, ex:
    2 2 2 2 + rest 2 becomes 3 3 2 2
    */
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

    /*
    Now we need:
        - the positions of B at which we start copying the elements into the buffer B_col (displ_B)
        - the number of elements (rows * cols) that the buffer receives from each process
        - the positions of B_col at which we start inserting the received elements
    */
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

        matrixMultiply(C, A, B_col, row_counts, displ_B, rank, p, N); // Finally
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
