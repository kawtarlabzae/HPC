#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CHANGE 1: Increase N to 2048 to exceed L3 Cache size (96MB total data)
#define N 2048

// Helper: Fill matrix with random values
void initialize_matrix(double *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

// Helper: Clear result matrix
void clear_matrix(double *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = 0.0;
    }
}

// ----------------------------------------------------------------------
// Block Matrix Multiplication Function
// ----------------------------------------------------------------------
void mat_mul_block(double *A, double *B, double *C, int n, int b_size) {
    // ii, jj, kk are the start indices of the BLOCKS
    for (int ii = 0; ii < n; ii += b_size) {
        for (int kk = 0; kk < n; kk += b_size) {
            for (int jj = 0; jj < n; jj += b_size) {

                // Handle edge cases if N is not divisible by b_size
                int i_limit = (ii + b_size > n) ? n : ii + b_size;
                int k_limit = (kk + b_size > n) ? n : kk + b_size;
                int j_limit = (jj + b_size > n) ? n : jj + b_size;

                // Standard multiplication INSIDE the block (i-k-j optimization)
                for (int i = ii; i < i_limit; i++) {
                    for (int k = kk; k < k_limit; k++) {
                        double r = A[i * n + k]; // Load A once per inner loop
                        for (int j = jj; j < j_limit; j++) {
                            C[i * n + j] += r * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Allocate memory
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    printf("Initializing matrices...\n");
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    // List of block sizes to experiment with
    int block_sizes[] = {16, 32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("\nMatrix Size: %d x %d\n", N, N);
    printf("Total Data Size: %.2f MB\n", 3.0 * N * N * sizeof(double) / (1024*1024));
    printf("----------------------------------------------------------------------\n");
    printf("| Block Size | Time (sec) | Bandwidth (MB/s) | Performance (GFLOPS) |\n");
    printf("----------------------------------------------------------------------\n");

    // Metrics setup
    double data_size_bytes = 3.0 * N * N * sizeof(double); // For Bandwidth
    double total_ops = 2.0 * N * N * N;                   // For GFLOPS

    for (int x = 0; x < num_sizes; x++) {
        int b_size = block_sizes[x];

        clear_matrix(C, N); // Reset C for fair testing

        clock_t start = clock();
        mat_mul_block(A, B, C, N, b_size);
        clock_t end = clock();

        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Calculate Metrics
        double bw_mb = (data_size_bytes / time_taken) / (1024.0 * 1024.0);
        double gflops = (total_ops / time_taken) / 1e9;

        printf("| %-10d | %-10.4f | %-16.2f | %-20.2f |\n", b_size, time_taken, bw_mb, gflops);
    }
    printf("----------------------------------------------------------------------\n");

    free(A); free(B); free(C);
    return 0;
}
