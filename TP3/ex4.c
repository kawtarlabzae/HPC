#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n = 1000;
    int m = 1000;
    double start_time, run_time;

    double *a = (double *)malloc(n * n * sizeof(double));
    double *b = (double *)malloc(n * m * sizeof(double));
    double *c = (double *)malloc(n * m * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
        }
    }

    start_time = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }

    run_time = omp_get_wtime() - start_time;

    printf("Execution time: %f seconds\n", run_time);

    free(a);
    free(b);
    free(c);

    return 0;
}
