#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

int main () {
    int i;
    double starttime, run_time;
    double x, pi, sum = 0.0;

    step = 1.0 / (double) num_steps;

    starttime = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) private(x)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    run_time = omp_get_wtime() - starttime;

    pi = step * sum;
    
    printf("value : %f\n", pi);
    printf("Execution time: %f seconds\n", run_time);
}
