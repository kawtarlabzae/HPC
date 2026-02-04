#include <stdio.h>
#include <time.h>

#define N 1000000000

int main() {
    double a = 1.1, b = 1.2;
    double x = 0.0, y = 0.0;
    clock_t start, end;

    start = clock();
    
    double product = a * b; 
    
    for (int i = 0; i < N; i += 1) {
        x = x + product;
        y = y + product;}        
    
    end = clock();

    printf("Optimized: x = %f, y = %f, time = %f s\n", 
           x, y, (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
