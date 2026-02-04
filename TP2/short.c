#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000 // 100 Million

int main() {
    // 1. Array is SHORT (2 bytes)
    short *a = malloc(N * sizeof(short));
    
    // 2. Accumulator is LONG (8 bytes) to prevent overflow
    long long sum = 0;
    
    double start, end;

    printf("Initializing %d shorts...\n", N);
    for (int i = 0; i < N; i++) a[i] = 1;

    // --- Unroll Factor 1 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i++) { 
        sum += a[i]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("U 1 : Sum = %lld | Time = %.4f ms\n", sum, (end - start) * 1000);

    // --- Unroll Factor 2 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 2) { 
        sum += a[i] + a[i+1]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("U 2 : Sum = %lld | Time = %.4f ms\n", sum, (end - start) * 1000);

    // --- Unroll Factor 4 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 4) { 
        sum += a[i] + a[i+1] + a[i+2] + a[i+3]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("U 4 : Sum = %lld | Time = %.4f ms\n", sum, (end - start) * 1000);

    // --- Unroll Factor 8 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 8) { 
        sum += a[i] + a[i+1] + a[i+2] + a[i+3] + 
               a[i+4] + a[i+5] + a[i+6] + a[i+7]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("U 8 : Sum = %lld | Time = %.4f ms\n", sum, (end - start) * 1000);

    // --- Unroll Factor 16 ---
    sum = 0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i += 16) { 
        sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7] +
               a[i+8] + a[i+9] + a[i+10] + a[i+11] + a[i+12] + a[i+13] + a[i+14] + a[i+15]; 
    }
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("U 16: Sum = %lld | Time = %.4f ms\n", sum, (end - start) * 1000);

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
    printf("U 32: Sum = %lld | Time = %.4f ms\n", sum, (end - start) * 1000);

    free(a);
    return 0;
}
