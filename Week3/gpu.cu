#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_initialization(const int* row_counts, const int n_loc, const int rank, const int N, double *d_A, double *d_B_col, double *d_C) {
    // Allocate memory for B_col on the device
    cudaMalloc( (void **)&d_A, row_counts[rank] * N * sizeof(double) );
    cudaMalloc( (void **)&d_B_col, N * (n_loc + 1) * sizeof(double) );
    cudaMalloc( (void **)&d_C, row_counts[rank] * N * sizeof(double) );
}

void gpu_computation(const int rank, const int p, const int n_loc, const int N, const int *row_counts, const int *displ_B,
                     const double *A, const double *B_col, double *C, double *d_A, double* d_B_col, double *d_C) {
    // Select the GPU to use
    cudaSetDevice(rank);
    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    // copy A and B_loc to the device (C is already allocated on the gpu)
    cudaMemcpy(d_A, A, row_counts[rank] * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_col, B_col, N * (n_loc + 1) * sizeof(double), cudaMemcpyHostToDevice);
    // cublas multiplication
    const double alpha = 1, beta = 0; // lmao why do i have to do this
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row_counts[rank], row_counts[p], N,
                &alpha, d_A, N, d_B_col, row_counts[p], &beta, d_C + displ_B[p], N);
    // Copy C from device to host memory
    cudaMemcpy(C, d_C, row_counts[rank] * N * sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
}
