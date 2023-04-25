#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printMatrix(const double* matrix, const int n_loc, const int N);
void yellow() { printf("\033[1;33m"); }
void reset() { printf("\033[0m"); }

int main(int argc, char** argv) {
     
    int rank, wsz, N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsz);

    N = atoi(argv[1]); // size of the grid
    int n_loc = (N + 2) / wsz;
    int rest = (N + 2) % wsz;

    int* rows = (int *) malloc( wsz * sizeof(int) ); // N° of actual rows
    int* offset = (int *) malloc( wsz * sizeof(int) ); // For initialization
    for (int i = 0; i < wsz; i++) rows[i] = (i < rest) ? n_loc + 1 : n_loc; // first processes get the remainder
    offset[0] = 0; for (int i = 1; i < wsz; i++) offset[i] = offset[i-1] + rows[i-1];

    if (rank == 0) {
        printf("N (int): %d\nN (ext): %d\nn_loc: %d\nrest: %d\n\n", N, N + 2, n_loc, rest);
        for (int i = 0; i < wsz; i++) printf("%d %d\n", rows[i], offset[i]);
    }

    // allocation includes space for ghost rows
    double* matrix = (double*)malloc( (rows[rank] + 2) * (N + 2) * sizeof(double) );
    memset( matrix, 0, (rows[rank] + 2) * (N + 2) * sizeof(double) ); // set to zero to check for bugs later

    // Initialize local matrix
    for (int i = 1; i <= rows[rank]; i++)
        for (int j = 0; j < N + 2; j++)
            *(matrix + i * (N + 2) + j) = (i + offset[rank]) * (N + 2) + j;

    // Exchange ghost rows
    int above = (rank == 0) ? MPI_PROC_NULL : rank - 1; // First process doesn't send its upper row to anybody
    int below = (rank == wsz - 1) ? MPI_PROC_NULL : rank + 1; // Last process doesn't sent its lower row to anybody

    // Exchange upper row
    MPI_Sendrecv(matrix + N + 2, N + 2, // Send the first (actual) row (the second)
                 MPI_DOUBLE, above, 0, // to the process above
                 matrix + (rows[rank] + 1 ) * (N + 2), N + 2, // Receive it and put it in the lower ghost row (last row)
                 MPI_DOUBLE, below, 0, // from the process below (same tag)
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("OK 1 rank %d\n", rank); // Debug

    // Exchange lower row
    MPI_Sendrecv(matrix + rows[rank] * (N + 2), N + 2, // Send the last (actual) row (the second to last)
                 MPI_DOUBLE, below, 0, // to the process below with tag = rank + wsz + 1
                 matrix, N + 2, // Receive the row and put it in the upper ghost row (the first row)
                 MPI_DOUBLE, above, 0, // from the process above (same tag)
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("OK 2 rank %d\n", rank); // Debug

    MPI_Barrier(MPI_COMM_WORLD); // Debug

    // Print in order
    if (rank == 0) {
        yellow(); printf("Rank %d:\n", rank); reset();
        printMatrix(matrix, rows[rank] + 2, N + 2);
        for (int count = 1; count < wsz; count++) {
            MPI_Recv(matrix, (rows[count] + 2) * (N + 2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            yellow(); printf("Rank %d:\n", count); reset();
            printMatrix(matrix, rows[count] + 2, N + 2);
        }
    } else MPI_Send(matrix, (rows[rank] + 2) * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

    free(matrix);
    MPI_Finalize();
}

void printMatrix(const double* matrix, const int rows, const int cols) {
    // Print local grid
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%g ", *(matrix + i * cols + j));
        }
        printf("\n");
    }
}
