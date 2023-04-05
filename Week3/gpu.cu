#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_initialization(const int* row_counts, const int n_loc, const int rank, const int N, const double *A,
                        double *d_A, double *d_B_col, double *d_C, cublasHandle_t handle) {
    // Allocate memory for B_col on the device
    cudaMalloc( (void **)&d_A, row_counts[rank] * N * sizeof(double) );
    cudaMalloc( (void **)&d_B_col, N * (n_loc + 1) * sizeof(double) );
    cudaMalloc( (void **)&d_C, row_counts[rank] * N * sizeof(double) );
    // copy A and B_loc to the device (C is already allocated on the gpu)
    cudaMemcpy(d_A, A, row_counts[rank] * N * sizeof(double), cudaMemcpyHostToDevice);
    // Select the GPU to use
    cudaSetDevice(rank);
    // Create a cuBLAS handle
    cublasCreate(&handle);
}

void gpu_computation(const int rank, const int p, const int n_loc, const int N, const int *row_counts,
                     const int *displ_B, const double *A, const double *B_col, double *C, double *d_A,
                     double* d_B_col, double *d_C, cublasHandle_t handle) {

    // copy B_loc to the device (A is already allocated on the gpu)
    cudaMemcpy(d_B_col, B_col, N * (n_loc + 1) * sizeof(double), cudaMemcpyHostToDevice);
    // cublas multiplication
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, row_counts[rank], row_counts[p], &alpha, d_B_col, N, d_A, N, &beta, d_C + displ_B[p], N);
}
