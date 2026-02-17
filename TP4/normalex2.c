#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000

void init_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)(i + j);
        }
    }
}

void print_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", A[i*n + j]);
        }
        printf("\n");
    }
}

double sum_matrix(int n, double *A) {
    double sum = 0.0;
    for (int i = 0; i < n*n; i++) {
        sum += A[i];
    }
    return sum;
}

int main() {
    double *A;
    double sum;
    double start, end;

    A = (double*) malloc(N * N * sizeof(double));

    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    start = (double) clock() / CLOCKS_PER_SEC;

    /* Initialization */
    init_matrix(N, A);

    /* Printing (can be commented if N is large) */
    /* print_matrix(N, A); */

    /* Sum computation */
    sum = sum_matrix(N, A);

    end = (double) clock() / CLOCKS_PER_SEC;

    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n", end - start);

    free(A);
    return 0;
}
