#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

// fill with random numbers
void initialize_matrix(double *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

// reset result matrix to zero
void clear_matrix(double *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = 0.0;
    }
}

int main() {
    // allocate memory
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    initialize_matrix(A, N);
    initialize_matrix(B, N);

    clock_t start, end;
    double t_direct, t_sum, t_opt;

    // ---------------------------------------------------------
    // 1. Direct Write (I-J-K) - Slowest
    // ---------------------------------------------------------
    clear_matrix(C, N);
    printf("1. Running Direct Write (modifying C inside loop)...\n");

    start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // constantly writing to memory here
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    end = clock();
    t_direct = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f sec\n\n", t_direct);

    // ---------------------------------------------------------
    // 2. Standard (I-J-K) - A little optimization using variable sum
    // ---------------------------------------------------------
    clear_matrix(C, N);
    printf("2. Running Standard (using local sum)...\n");
    
    start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0; // use register for math
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum; // write to memory once
        }
    }
    end = clock();
    t_sum = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f sec\n\n", t_sum);

    // ---------------------------------------------------------
    // 3. Optimized (I-K-J) - Best Cache Usage
    // ---------------------------------------------------------
    clear_matrix(C, N);
    printf("3. Running Optimized (I-K-J)...\n");

    start = clock();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k]; // keep this const for inner loop
            for (int j = 0; j < N; j++) {
                // sequential access is faster
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
    end = clock();
    t_opt = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f sec\n", t_opt);

    // ---------------------------------------------------------
    // Analysis
    // ---------------------------------------------------------
    double data_size = 3.0 * N * N * sizeof(double);
    
    printf("\n--- Summary ---\n");
    printf("Direct Write: %.2f sec\n", t_direct);
    printf("Standard:     %.2f sec\n", t_sum);
    printf("Optimized:    %.2f sec\n", t_opt);
    
    printf("\nBandwidth (Optimized): %.2f MB/s\n", (data_size / t_opt) / (1024*1024));

    free(A); free(B); free(C);
    return 0;
}
