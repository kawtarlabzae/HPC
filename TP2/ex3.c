#include <stdio.h>
#include <stdlib.h>

//#define N 100000000

/* ===== Sequential Part ===== */
void add_noise(double *a) {
    a[0] = 1.0;
    for (int i = 1; i < N; i++) {
        a[i] = a[i-1] * 1.0000001;
    }
}

/* ===== Initialization ===== */
void init_b(double *b) {
    for (int i = 0; i < N; i++) {
        b[i] = i * 0.5;
    }
}

/* ===== Compute addition ===== */
void compute_addition(double *a, double *b, double *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

/* ===== Reduction ===== */
double reduction(double *c) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += c[i];
    }
    return sum;
}

int main() {
    double *a = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    double *c = malloc(N * sizeof(double));

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 1. Sequential part
    add_noise(a);

    // 2. Parallelizable parts
    init_b(b);
    compute_addition(a, b, c);

    // 3. Reduction
    double sum = reduction(c);
    
    printf("Sum = %f\n", sum);

    free(a);
    free(b);
    free(c);
    return 0;
}
