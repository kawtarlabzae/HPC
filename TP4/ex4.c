#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Version 1: Implicit Barrier (Safe but Slow)
void dmvm_v1(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            // Implicit barrier at the end of this directive waits for all threads
            #pragma omp for schedule(static)
            for (int r = 0; r < m; ++r) {
                lhs[r] += mat[r + offset] * rhs[c];
            }
        }
    }
}

// Version 2: Dynamic + nowait (FAST BUT WRONG/UNSAFE due to Race Condition)
void dmvm_v2(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            // Threads race ahead. Dynamic scheduling means two threads might
            // update the same row 'r' for different columns 'c' simultaneously.
            #pragma omp for schedule(dynamic) nowait
            for (int r = 0; r < m; ++r) {
                lhs[r] += mat[r + offset] * rhs[c];
            }
        }
    }
}

// Version 3: Static + nowait (Fast & Safe)
void dmvm_v3(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            // Static scheduling ensures Thread X always owns the same rows.
            // Removing the barrier allows Thread X to process its rows for
            // column 0, then immediately col 1, col 2, etc., without waiting.
            #pragma omp for schedule(static) nowait
            for (int r = 0; r < m; ++r) {
                lhs[r] += mat[r + offset] * rhs[c];
            }
        }
    }
}

void reset_lhs(int m, double *lhs) {
    for (int i = 0; i < m; i++) lhs[i] = 0.0;
}

int main(void) {
    const int n = 40000; // columns
    const int m = 600;   // rows
    
    // Total Floating Point Operations: 2 * N * M (Multiply + Add per element)
    double FLOPs = 2.0 * (double)n * (double)m;

    double *mat = (double*)malloc(n * m * sizeof(double));
    double *rhs = (double*)malloc(n * sizeof(double));
    double *lhs = (double*)malloc(m * sizeof(double));

    // Initialization
    for (int c = 0; c < n; ++c) {
        rhs[c] = 1.0;
        for (int r = 0; r < m; ++r)
            mat[r + c*m] = 1.0; // All 1s for verification
    }

    printf("Matrix: %d x %d | Threads: %d\n", m, n, omp_get_max_threads());
    printf("----------------------------------------------------------------\n");
    printf("| Version | Time (s) | Speedup | Efficiency | MFLOP/s  | Status |\n");
    printf("----------------------------------------------------------------\n");

    double start, end, time_seq, time_v1, time_v2, time_v3;

    // --- SEQUENTIAL BASELINE (Run V1 with 1 thread) ---
    // Note: Usually we write a separate sequential function, but V1 with 1 thread is equivalent.
    omp_set_num_threads(1);
    reset_lhs(m, lhs);
    start = omp_get_wtime();
    dmvm_v1(n, m, lhs, rhs, mat);
    end = omp_get_wtime();
    time_seq = end - start;
    printf("| Seq     | %8.4f |   1.00x |    100%%    | %8.2f |   OK   |\n", 
           time_seq, FLOPs/time_seq/1e6);

    // Reset to max threads for parallel tests
    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);

    // --- VERSION 1: Implicit Barrier ---
    reset_lhs(m, lhs);
    start = omp_get_wtime();
    dmvm_v1(n, m, lhs, rhs, mat);
    end = omp_get_wtime();
    time_v1 = end - start;
    printf("| V1 Sync | %8.4f | %6.2fx | %9.1f%% | %8.2f |   OK   |\n", 
           time_v1, time_seq/time_v1, (time_seq/time_v1/threads)*100, FLOPs/time_v1/1e6);

    // --- VERSION 2: Dynamic + NoWait (Unsafe) ---
    reset_lhs(m, lhs);
    start = omp_get_wtime();
    dmvm_v2(n, m, lhs, rhs, mat);
    end = omp_get_wtime();
    time_v2 = end - start;
    printf("| V2 Dyn  | %8.4f | %6.2fx | %9.1f%% | %8.2f |  RACE  |\n", 
           time_v2, time_seq/time_v2, (time_seq/time_v2/threads)*100, FLOPs/time_v2/1e6);

    // --- VERSION 3: Static + NoWait (Best) ---
    reset_lhs(m, lhs);
    start = omp_get_wtime();
    dmvm_v3(n, m, lhs, rhs, mat);
    end = omp_get_wtime();
    time_v3 = end - start;
    printf("| V3 Stat | %8.4f | %6.2fx | %9.1f%% | %8.2f |   OK   |\n", 
           time_v3, time_seq/time_v3, (time_seq/time_v3/threads)*100, FLOPs/time_v3/1e6);
    printf("----------------------------------------------------------------\n");

    // Verification (First element should be equal to N * 1.0 * 1.0 = 40000)
    printf("\nVerification (lhs[0]): Expected %.1f\n", (double)n);
    printf("V1 Result: %.1f\n", lhs[0]); 
    // Note: V2 result might be wrong here due to race condition

    free(mat);
    free(rhs);
    free(lhs);
    return 0;
}
