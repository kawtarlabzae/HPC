#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // Matrix size (try 1024 or 2048)

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

int main() {
    // 1. Setup Memory
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    initialize_matrix(A, N);
    initialize_matrix(B, N);

    clock_t start, end;
    double t_direct, t_var, t_opt;
    double total_bytes = 3.0 * N * N * sizeof(double); // For bandwidth calculation

    printf("Matrix Size: %d x %d\n", N, N);
    printf("--------------------------------------------------------------\n");
    printf("| Version             | Time (sec) | Bandwidth (MB/s) |\n");
    printf("--------------------------------------------------------------\n");

    // =========================================================
    // 1. Standard: Direct Write (i-j-k)
    // =========================================================
    clear_matrix(C, N);
    start = clock();
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // BAD: We are writing to RAM (C[i*N+j]) in every single step
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    
    end = clock();
    t_direct = ((double)(end - start)) / CLOCKS_PER_SEC;
    double bw_direct = (total_bytes / t_direct) / (1024 * 1024);
    printf("| 1. Direct Write     | %-10.4f | %-16.2f |\n", t_direct, bw_direct);


    // =========================================================
    // 2. A Little Optimized: Variable Replacement (i-j-k)
    // =========================================================
    clear_matrix(C, N);
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0; // GOOD: Use a CPU register for math
            
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            
            C[i * N + j] = sum; // Write to RAM only ONCE per pixel
        }
    }

    end = clock();
    t_var = ((double)(end - start)) / CLOCKS_PER_SEC;
    double bw_var = (total_bytes / t_var) / (1024 * 1024);
    printf("| 2. Variable Sum     | %-10.4f | %-16.2f |\n", t_var, bw_var);


    // =========================================================
    // 3. Fully Optimized: Loop Reordering (i-k-j)
    // =========================================================
    clear_matrix(C, N);
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k]; // Pre-load A
            
            for (int j = 0; j < N; j++) {
                // BEST: Accessing C and B sequentially (Stride-1)
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    end = clock();
    t_opt = ((double)(end - start)) / CLOCKS_PER_SEC;
    double bw_opt = (total_bytes / t_opt) / (1024 * 1024);
    printf("| 3. Loop Reorder     | %-10.4f | %-16.2f |\n", t_opt, bw_opt);
    
    printf("--------------------------------------------------------------\n");

    // =========================================================
    // 4. Speedup Analysis
    // =========================================================
    // Speedup = Time_Old / Time_New
    double speedup_vs_direct = t_direct / t_opt;
    double speedup_vs_var    = t_var / t_opt;

    printf("\n--- Speedup Analysis ---\n");
    printf("Speedup vs Direct Write:  %.2f x faster\n", speedup_vs_direct);
    printf("Speedup vs Variable Sum:  %.2f x faster\n", speedup_vs_var);

    free(A); free(B); free(C);
    return 0;
}
