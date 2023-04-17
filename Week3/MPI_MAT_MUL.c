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
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0) printf("Error: Incorrect number of arguments\nTwo arguments needed: matrix size and print option.\n");
        exit(1);
    }

    int N = atoi(argv[1]); // get the size of the matrix as cmd line arg
    int print = atoi(argv[2]); // you may want to print the matrix on the terminal
    int n_loc, rest; // self explanatory
    int *row_counts, *displ_B, *elem_counts, *displ_B_col; // how much stuff each process gets

    double computation = 0, communication = 0, time1, time2, time3;
    double *A, *B, *C, *B_col; // declare the three matrices + the receive buffer

    n_loc = N / world_size; // ex: 10 / 4 = 2
    rest = N % world_size; // ex: 10 % 4 = 2

#ifdef USE_CBLAS
    if (rank == 0) printf("USING DGEMM!\n");
#endif

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

#ifdef USE_GPU
    if (rank == 0) printf("USING GPU!\n");
    cublasHandle_t handle;
    double *d_A, *d_B_col, *d_C;

    // Create and start timer
    cudaEvent_t start, stop;
    float elapsedTotalTime, maxElapsedTotalTime, elapsedTime = 0.0, maxElapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocation of device memory and copy of A to d_A
    alloc(row_counts, n_loc, rank, N, A, &d_A, &d_B_col, &d_C, &handle);
#endif

    // Loop over the number of processes: create the datatype and gather the little blocks into every B_col
    for (int p = 0; p < world_size; p++) {

        MPI_Barrier(MPI_COMM_WORLD); // Force syncronization
        time1 = MPI_Wtime();

        // Initialize displacements and counts, create datatype, Allgatherv
        MPICalls(row_counts, displ_B, elem_counts, displ_B_col, B, B_col, rank, p, N, world_size);

        MPI_Barrier(MPI_COMM_WORLD);
        time2 = MPI_Wtime();

#ifdef USE_CBLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    row_counts[rank], row_counts[p], N, // m, n, k
                    1.0, A, N, B_col, row_counts[p], 0.0, C + displ_B[p], N);
#elif USE_GPU
        gpu_computation(rank, p, n_loc, N, row_counts, displ_B, A, B_col, C, d_A, d_B_col, d_C, handle, &elapsedTime);
#else
        matrixMultiply(C, A, B_col, row_counts, displ_B, rank, p, N); // Finally
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        time3 = MPI_Wtime();

        communication += time2 - time1;
        computation += time3 - time2;
    }

#ifdef USE_CBLAS
    // Append the profiling data in a file
    if (rank == 0) {
    FILE *fp = fopen("dgemm_scalability.txt", "a");
    fprintf(fp, "%d %d %g %g\n", N, world_size, communication, computation);
    fclose(fp); }
#elif USE_GPU
    // Copy C from device to host memory
    cudaError_t err3 = cudaMemcpy(C, d_C, row_counts[rank] * N * sizeof(double), cudaMemcpyDeviceToHost);
    if (err3 != cudaSuccess) printf("Error on copying d_C to C: %s\n", cudaGetErrorString(err3));
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B_col); cudaFree(d_C);

    // Stop and destroy timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTotalTime, start, stop);

    // Compute the maximum elapsed time across all processes
    MPI_Reduce(&elapsedTime, &maxElapsedTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsedTotalTime, &maxElapsedTotalTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total time: %f ms\n", maxElapsedTotalTime);
    if (rank == 0) printf("cublasDgemm time: %f ms\n", maxElapsedTime);

    // Append the profiling data in a file
    if (rank == 0) {
    FILE *fp = fopen("gpu_scalability.txt", "a");
    fprintf(fp, "%d %d %f %f\n", N, world_size, maxElapsedTotalTime, maxElapsedTime);
    fclose(fp); }
#else
    if (rank == 0) printf("Communication time %.5g\nComputation time %.5g\n", communication, computation);
#endif

    // Print the final matrix
    if (print == 1 && N <= 100) { // This is to prevent you from accidentally print a very large matrix
        if (rank == 0) { printf("\nN: %d, world_size: %d, n_loc: %d, rest: %d\n", N, world_size, n_loc, rest); }
        if (rank == 0) {
            printf("\nResult:\n");
            printMatrix(C, row_counts[rank], N);
            for (int count = 1; count < world_size; count++) {
                MPI_Recv(C, row_counts[count] * N, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printMatrix(C, row_counts[count], N);
            }
        } else MPI_Send(C, row_counts[rank] * N, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }

    free(A); free(B); free(C); free(B_col);
    free(row_counts); free(displ_B); free(elem_counts); free(displ_B_col);

    MPI_Finalize();
    return 0;
}
