#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 1000000

int main() {
    double *A = malloc(N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    double sum = 0.0;
    double sum_sq = 0.0; 
    double max = 0.0;
    double stddev = 0.0;
    double start_time, end_time; // Timing variables

    // Initialization
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }

    // --- START TIMER ---
    start_time = omp_get_wtime();

    // Force 3 threads so all sections run simultaneously
    omp_set_num_threads(3);

    #pragma omp parallel sections
    {
        // --- SECTION 1: Compute Sum ---
        #pragma omp section
        {
            double local_sum = 0.0;
            for (int i = 0; i < N; i++) {
                local_sum += A[i];
            }
            // Critical not needed if we just write to a shared var once at end
            #pragma omp critical
            sum = local_sum;
        }

        // --- SECTION 2: Compute Max ---
        #pragma omp section
        {
            double local_max = -1.0; 
            for (int i = 0; i < N; i++) {
                if (A[i] > local_max) local_max = A[i];
            }
            #pragma omp critical
            if (local_max > max) max = local_max;
        }

        // --- SECTION 3: Compute Sum of Squares (for Std Dev) ---
        #pragma omp section
        {
            double local_sq = 0.0;
            for (int i = 0; i < N; i++) {
                local_sq += A[i] * A[i];
            }
            #pragma omp critical
            sum_sq = local_sq;
        }
    }

    // Final math (Sequential, but instant)
    double mean = sum / N;
    double variance = (sum_sq / N) - (mean * mean);
    stddev = sqrt(variance);

    // --- STOP TIMER ---
    end_time = omp_get_wtime();

    // Print results
    printf("Sum      = %f\n", sum);
    printf("Max      = %f\n", max);
    printf("Std Dev  = %f\n", stddev);
    printf("Time     = %f seconds\n", end_time - start_time);

    free(A);
    return 0;
}
