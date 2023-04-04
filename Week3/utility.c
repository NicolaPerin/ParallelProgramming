#include "utility.h"

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

void MPICalls(const int* row_counts, const int* displ_B, int* elem_counts, int* displ_B_col,
              const double* B, double *B_col, const int rank, const int p, const int N, const int world_size) {

    init_elem_counts(elem_counts, row_counts, p, world_size); // Must change with p, not the rank!!!
    init_displ_B_col(displ_B_col, elem_counts, world_size); // Same

    MPI_Datatype my_block;
    MPI_Type_vector(row_counts[rank], row_counts[p], N, MPI_DOUBLE, &my_block);
    MPI_Type_commit(&my_block);
    MPI_Allgatherv(B + displ_B[p], 1, my_block, B_col, elem_counts, displ_B_col, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Type_free(&my_block);
}
