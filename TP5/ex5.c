#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <N_iterations>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long long N = atoll(argv[1]);
    if (N <= 0) {
        if (rank == 0) printf("N must be strictly positive.\n");
        MPI_Finalize();
        return 1;
    }

    double serial_time = 0.0;
    double pi_serial = 0.0;

    // --- 1. SERIAL COMPUTATION (For Speedup Baseline) ---
    if (rank == 0) {
        double start_serial = MPI_Wtime();
        double sum_serial = 0.0;
        for (long long i = 0; i < N; ++i) {
            double x = (i + 0.5) / (double)N;
            sum_serial += 4.0 / (1.0 + x * x);
        }
        pi_serial = sum_serial / (double)N;
        serial_time = MPI_Wtime() - start_serial;
    }

    // --- 2. PARALLEL COMPUTATION ---
    long long base = N / num_procs;
    long long remainder = N % num_procs;

    long long local_N = base + (rank < remainder ? 1 : 0);
    long long start_i;

    if (rank < remainder) {
        start_i = rank * (base + 1);
    } else {
        start_i = remainder * (base + 1) + (rank - remainder) * base;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel = MPI_Wtime();

    double local_sum = 0.0;
    for (long long i = start_i; i < start_i + local_N; ++i) {
        double x = (i + 0.5) / (double)N;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double parallel_time = MPI_Wtime() - start_parallel;

    // --- 3. PRINT RESULTS ---
    if (rank == 0) {
        double pi_parallel = global_sum / (double)N;
        double error = fabs(pi_parallel - M_PI);

        printf("\n--- PI CALCULATION RESULTS (N=%lld) ---\n", N);
        printf("Calculated Pi (Parallel): %.15f\n", pi_parallel);
        printf("Error compared to M_PI:   %e\n", error);
        printf("\n--- PERFORMANCE ---\n");
        printf("Serial Time:   %f seconds\n", serial_time);
        printf("Parallel Time: %f seconds\n", parallel_time);

        double speedup = serial_time / parallel_time;
        printf("Speedup:       %f\n", speedup);
        printf("Efficiency:    %f\n", speedup / num_procs);
        printf("---------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}
