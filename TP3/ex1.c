#include <stdio.h>
#include <omp.h>

int main() {
    int nthreads, rank;

    #pragma omp parallel private(rank)
    {
        rank = omp_get_thread_num();
        printf("Hello from the rank %d thread\n", rank);

        if (rank == 0) {
            nthreads = omp_get_num_threads();
        }
    }

    printf("Parallel execution of hello_world with %d threads\n", nthreads);

    return 0;
}
