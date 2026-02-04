#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// If T isn't defined by the compiler line, default to double
#ifndef T
#define T double
#endif

// We need a larger N to measure fast integer operations
#define N 10000000

int main() {
    // 1. Allocate Memory
    T *a = malloc(N * sizeof(T));
    T sum = 0;
    double start, end;

    // 2. Initialize
    for (int i = 0; i < N; i++) a[i] = (T)1;

    // Helper macro to print based on type
    // We cast to double for floats, and long long for ints to handle everything safely
    #ifdef IS_INT
        #define PRINT_RES(u, s, t) printf("U %-2d: Sum = %-12lld | Time = %.4f ms\n", u, (long long)s, t)
    #else
        #define PRINT_RES(u, s, t) printf("U %-2d: Sum = %-12.1f | Time = %.4f ms\n", u, (double)s, t)
    #endif

    printf("Benchmarking N = %d elements...\n", N);

    // --- Unroll Factor 1 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i++) { sum += a[i]; }
    end = (double)clock() / CLOCKS_PER_SEC;
    PRINT_RES(1, sum, (end - start) * 1000);

    // --- Unroll Factor 2 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 2) { sum += a[i] + a[i+1]; }
    end = (double)clock() / CLOCKS_PER_SEC;
    PRINT_RES(2, sum, (end - start) * 1000);

    // --- Unroll Factor 4 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 4) { sum += a[i] + a[i+1] + a[i+2] + a[i+3]; }
    end = (double)clock() / CLOCKS_PER_SEC;
    PRINT_RES(4, sum, (end - start) * 1000);

    // --- Unroll Factor 8 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 8) { 
        sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    PRINT_RES(8, sum, (end - start) * 1000);

    // --- Unroll Factor 16 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 16) { 
        sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7] +
               a[i+8] + a[i+9] + a[i+10] + a[i+11] + a[i+12] + a[i+13] + a[i+14] + a[i+15]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    PRINT_RES(16, sum, (end - start) * 1000);

    // --- Unroll Factor 32 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 32) { 
        sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7] +
               a[i+8] + a[i+9] + a[i+10] + a[i+11] + a[i+12] + a[i+13] + a[i+14] + a[i+15] +
               a[i+16] + a[i+17] + a[i+18] + a[i+19] + a[i+20] + a[i+21] + a[i+22] + a[i+23] +
               a[i+24] + a[i+25] + a[i+26] + a[i+27] + a[i+28] + a[i+29] + a[i+30] + a[i+31]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    PRINT_RES(32, sum, (end - start) * 1000);

    free(a);
    return 0;
}
