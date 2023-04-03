#include <stdio.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define N_cols 2048
#define N (N_cols * N_cols)

void initA(int *A) {
    for (int i = 0; i < N_cols; i++) {
        for (int j = 0; j < N_cols; j++) {
            A[i * N_cols + j] = i * N_cols + j + 1;
        }
    }
}

void printMatrix(const int *A) {
    for (int i = 0; i < N_cols; i++) {
        for (int j = 0; j < N_cols; j++) {
            printf("%d ", A[i*N_cols + j]);
        }
        printf("\n");
    }
}

int checkCorrectness(const int* A) {
    int sum = 0;
    for (int i = 0; i < N_cols; i++) {
        for (int j = 0; j < N_cols; j++) {
            sum += A[j*N_cols + i] - (i * N_cols + j + 1);
        }
    }
    return sum; // Should be zero
}

__global__ void transpose( int *dev_A, const int dim) {
    int temp;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int j = idx % dim;  
    int i = idx / dim;

    int idx_trasp = j * dim + i;

    /*Check indexing is correct
    printf("I am Thread id: %d Block id: %d\n", threadIdx.x, blockIdx.x );
    printf("i: %d j: %d value: %d\n", i, j , dev_A[idx] );*/

    if (j > i) {
      temp = dev_A[idx];
      dev_A[idx] = dev_A[idx_trasp];
      dev_A[idx_trasp] = temp;
    }

}

int main( int argc, char * argv[] ) {

    int *A;
    int *dev_A;

    A = (int *)malloc(N * sizeof(int));
    cudaMalloc( (void**)&dev_A, N * sizeof(int) );

    initA(A); // Initialize A with increasing integers

    // Profiling (C)
    clock_t start, end;

    // Profiling (CUDA)
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);

    start = clock();
    cudaEventRecord(c_start);

    cudaMemcpy(dev_A, A, N * sizeof(int), cudaMemcpyHostToDevice); // Copy to gpu

    transpose <<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> (dev_A, N_cols);  // Do the transposition

    cudaMemcpy(A, dev_A, N * sizeof(int), cudaMemcpyDeviceToHost); // Copy the transposed matrix back to cpu

    end = clock();
    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Elapsed time (C): %f ms\n", time_taken);

    float milliseconds = 0; // CUDA wants float
    cudaEventElapsedTime(&milliseconds, c_start, c_stop);
    printf("Elapsed time (CUDA): %f ms\n", milliseconds);

    //printMatrix(A); // For small matrices

    start = clock();
    if (!checkCorrectness(A)) printf("OK!\n"); // For big matrices
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time to check correctness (CPU): %f ms\n", time_taken);

    cudaFree(dev_A);
    free(A);
    return 0;
}