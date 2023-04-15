#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>
#include <stdio.h>

// Matrix dimensions
const int N = 5040;

// Initialize
void init_matrix_A(double *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = i + 1;
    }
}

void init_matrix_B(double *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == j) {
                mat[i * cols + j] = 1;
            } else {
                mat[i * cols + j] = 0;
            }
        }
    }
}

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the GPU to use for this process
    cudaSetDevice(rank);

    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set alpha and beta for cublasDgemm
    const double alpha = 1.0;
    const double beta = 0.0;

    // Compute the size of the sub-matrices for each process
    int sub_N = N / size;

    // Host matrices
    double *A, *B, *C;
    B = (double *)malloc(N * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        memset( C, 0, N * N * sizeof(double) );

        // Initialize host matrices with random values
        init_matrix_A(A, N, N);
        init_matrix_B(B, N, N);
    }

    // Allocate CPU memory for the sub-matrix on each process
    double *sub_A = (double *)malloc(sub_N * N * sizeof(double));

    // Scatter the matrix A from the root process to all processes
    MPI_Scatter(A, sub_N * N, MPI_DOUBLE,
            sub_A, sub_N * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

    // Allocate GPU memory for the sub-matrix on each process
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sub_N * N * sizeof(double));
    cudaMalloc((void **)&d_B, N * N * sizeof(double));
    cudaMalloc((void **)&d_C, sub_N * N * sizeof(double));

    // Transfer the sub-matrix from CPU memory to GPU memory
    cudaMemcpy(d_A, sub_A, sub_N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Broadcast matrix B from rank 0 to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

    // Transfer the matrix B from CPU memory to GPU memory
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Compute partial matrix-matrix product on this GPU
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                sub_N, N, N,
                &alpha,
                d_A, sub_N,
                d_B, N,
                &beta,
                d_C, sub_N);

    // Allocate CPU memory for the sub-matrix on each process
    double *sub_C = (double *)malloc(sub_N * N * sizeof(double));

    // Transfer the sub-matrix from GPU memory to CPU memory
    cudaMemcpy(sub_C, d_C, sub_N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Gather the sub-matrices from all processes to the root process
    MPI_Gather(sub_C, sub_N * N, MPI_DOUBLE,
           C, sub_N * N, MPI_DOUBLE,
           0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sum = 0.0;
        for (int i = 0; i < N*N; i++) {
            sum += C[i] - (i + 1.0);
        }
        printf("Sum (should be 0): %g\n", sum);
    }

    // Clean up resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
        cublasDestroy(handle);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
