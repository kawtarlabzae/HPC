#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

// Logic: A[i][j] = i + j
void init_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)(i + j);
        }
    }
}

void print_matrix(int n, double *A) {
    // Safety check to avoid spamming the screen if N is huge
    if (n > 20) {
        printf("[Matrix too large to print]\n");
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", A[i*n + j]);
        }
        printf("\n");
    }
}

int main() {
    double *A;
    double sum = 0.0;
    double start, end;

    A = (double*) malloc(N * N * sizeof(double));

    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Use OpenMP wall-clock time for parallel measurements
    start = omp_get_wtime();

    #pragma omp parallel
    {
        // 1. MASTER: Only the master thread initializes the memory
        #pragma omp master
        {
            init_matrix(N, A);
        }
        // IMPORTANT: Master does not imply a barrier. 
        // We must wait here to ensure initialization is done before printing/summing.
        #pragma omp barrier

        // 2. SINGLE: Only one thread prints (doesn't matter which one)
        #pragma omp single
        {
            print_matrix(N, A);
        }
        // Implicit barrier exists at the end of 'single', so we don't need one here.

        // 3. PARALLEL FOR: All threads share the work of summing
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < N*N; i++) {
            sum += A[i];
        }
    }

    end = omp_get_wtime();

    printf("Sum = %lf\n", sum);
    printf("Execution time (OpenMP) = %lf seconds\n", end - start);

    free(A);
    return 0;
}
