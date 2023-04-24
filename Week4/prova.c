#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printMatrix(const int* matrix, const int n_loc, const int N);

void yellow() { printf("\033[1;33m"); }

void reset() { printf("\033[0m"); }

int main(int argc, char** argv) {
    int N = 24; // size of the grid
    int rank, wsz;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsz);

    int n_loc = N / wsz;
    int true_n_loc = rank == 0 || rank == wsz - 1 ? n_loc + 1 : n_loc + 2;
    int* matrix = (int*)malloc((true_n_loc + 2) * N * sizeof(int)); // +2 for ghost rows
    memset( matrix, 0, true_n_loc * N * sizeof(int) );

    // Initialize local grid
    for (int i = rank == 0 ? 0 : 1; i <= n_loc; i++) {
        for (int j = 0; j < N; j++) {
            *(matrix + i * N + j) = rank + 1;
        }
    }

    // Exchange ghost rows
    int source_upper = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int dest_upper = source_upper;
    int source_lower = rank == wsz - 1 ? MPI_PROC_NULL : rank + 1;
    int dest_lower = source_lower;

    MPI_Sendrecv(matrix + (rank == 0 ? 0 : 1) * N, N, MPI_INT, dest_upper, 0,
                 matrix + (rank == 0 ? n_loc : 0) * N, N, MPI_INT, source_upper, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(matrix + n_loc * N, N, MPI_INT, dest_lower, 0,
                 matrix + ((rank == wsz - 1 || rank == 0 )? n_loc : n_loc + 1) * N, N, MPI_INT, source_lower, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

 // Print local grid
    if (rank == 0) {
        yellow(); printf("Rank %d:\n", rank); reset();
        printMatrix(matrix, true_n_loc, N);
        for (int count = 1; count < wsz; count++) {
            int recv_rows = count == wsz - 1 ? n_loc + 1 : n_loc + 2;
            MPI_Recv(matrix, recv_rows * N, MPI_INT, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            yellow(); printf("Rank %d:\n", count); reset();
            printMatrix(matrix, recv_rows, N);
        }
    } else MPI_Send(matrix, true_n_loc * N, MPI_INT, 0, rank, MPI_COMM_WORLD);

    free(matrix);
    MPI_Finalize();
}

void printMatrix(const int* matrix, const int n_loc, const int N) {
    // Print local grid
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", *(matrix + i * N + j));
        }
        printf("\n");
    }
}
