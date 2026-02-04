#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Default to float if no type is given
#ifndef T
#define T float
#endif

// Helper macro to print correct format based on type
#ifdef IS_INT
    #define PRINT_RES(s, t) printf("Sum = %lld, Time = %f ms\n", (long long)s, t)
#else
    #define PRINT_RES(s, t) printf("Sum = %f, Time = %f ms\n", (double)s, t)
#endif

#define N 10000000

int main() {
    // 1. Allocate memory dynamically using generic Type T
    T *a = malloc(N * sizeof(T));

    // Safety check
    if (a == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    T sum = 0;
    double start, end;

    // Initialize the array
    for (int i = 0; i < N; i++) {
        a[i] = (T)1; // Cast 1 to the correct type
    }

    // Start timing
    start = (double)clock() / CLOCKS_PER_SEC;

    // Perform summation
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }

    // End timing
    end = (double)clock() / CLOCKS_PER_SEC;

    // Print results using our smart macro
    PRINT_RES(sum, (end - start) * 1000);

    // Free the allocated memory
    free(a);

    return 0;
}
