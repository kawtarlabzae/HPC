#include <stdio.h>
#include <time.h>

#define N 1000000000

int main() {
    double a = 1.1, b = 1.2;
    double x = 0.0, y = 0.0;
    clock_t start, end;

    start = clock();
    
    // Inefficient Loop: 
    // The CPU recalculates (a * b) 200 million times.
    for (int i = 0; i < N; i++) {
        x = a * b + x; // stream 1
        y = a * b + y; // independent stream 2
    }
    
    end = clock();

    printf("Original: x = %f, y = %f, time = %f s\n", 
           x, y, (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
