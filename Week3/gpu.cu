#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void alloc(const int* row_counts, const int n_loc, const int rank, const int N, const double *A,
                        double **d_A, double **d_B_col, double **d_C) {
    // Allocate memory on the device
    cudaMalloc( (void **)d_A, row_counts[rank] * N * sizeof(double) );
    cudaError_t err = cudaMalloc((void **)d_B_col, N * (n_loc + 1) * sizeof(double));
    if (err != cudaSuccess) printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
    cudaMalloc( (void **)d_C, row_counts[rank] * N * sizeof(double) );
    // copy A to the device (C is already allocated on the gpu)
    cudaError_t err1 = cudaMemcpy(*d_A, A, row_counts[rank] * N * sizeof(double), cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess) { printf("Error on copying A to d_A: %s\n", cudaGetErrorString(err1)); }
}

void gpu_computation(const int rank, const int p, const int n_loc, const int N, const int *row_counts,
                     const int *displ_B, const double *A, const double *B_col, double *C, double *d_A,
                     double* d_B_col, double *d_C, cublasHandle_t handle) {

    if (B_col == NULL) printf("B_col pointer is null\n");
    if (d_B_col == NULL) printf("d_B_col pointer is null\n");
    // copy B_loc to the device (A is already allocated on the gpu)
    cudaError_t err2 = cudaMemcpy(d_B_col, B_col, N * (n_loc + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (err2 != cudaSuccess) { printf("Error on copying B_col to d_B_col: %s\n", cudaGetErrorString(err2)); }
    // cublas multiplication
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row_counts[p], row_counts[rank], N, &alpha, d_B_col, row_counts[p], d_A, N, &beta, d_C + displ_B[p], N);
}
