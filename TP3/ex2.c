#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

int main() {
    double pi, global_sum = 0.0;
    double start_time, run_time;

    step = 1.0 / (double) num_steps;

    start_time = omp_get_wtime();

    #pragma omp parallel reduction(+:global_sum)
    {
        int i, id, nthreads;
        double x;

        id = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        for (i = id; i < num_steps; i += nthreads) {
            x = (i + 0.5) * step;
            global_sum += 4.0 / (1.0 + x * x);
        }
    }

    pi = step * global_sum;

    run_time = omp_get_wtime() - start_time;

    printf("\n pi with %ld steps is %f", num_steps, pi);
    printf("\n Execution time : %f seconds\n", run_time);

    return 0;
}
