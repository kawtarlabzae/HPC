#include <stdio.h>
#include <stdlib.h>

#define N 512 // Matrix size

/* ===== Generate noise (Strictly Sequential) ===== */
void generate_noise(double *noise) {
    noise[0] = 1.0;
    for (int i = 1; i < N; i++) {
        noise[i] = noise[i-1] * 1.0000001;
    }
}

/* ===== Matrices Initialization ===== */
void init_matrix(double *M) {
    for (int i = 0; i < N*N; i++) {
        M[i] = (double)(i % 100) * 0.01;
    }
}

/* ===== Matrix Multiplication (Highly Parallelizable) ===== */
void matmul(double *A, double *B, double *C, double *noise) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = noise[i];
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main() {
    // Allocate memory
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    double *noise = malloc(N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL || noise == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 1. Sequential Part
    generate_noise(noise);

    // 2. Parallelizable Parts
    init_matrix(A);
    init_matrix(B);
    matmul(A, B, C, noise);

    printf("C[0] = %f\n", C[0]);

    // Cleanup
    free(A);
    free(B);
    free(C);
    free(noise);
    return 0;
}
