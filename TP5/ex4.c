#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void matrixVectorMult(double* A, double* b, double* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            x[i] += A[i * size + j] * b[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int size = atoi(argv[1]);
    if (size <= 0) {
        if (rank == 0) printf("Matrix size must be positive.\n");
        MPI_Finalize();
        return 1;
    }

    double *A = NULL, *x_serial = NULL, *x_parallel = NULL;
    double *b = malloc(size * sizeof(double));
    double serial_time = 0.0;

    if (rank == 0) {
        A = malloc(size * size * sizeof(double));
        x_serial = malloc(size * sizeof(double));
        x_parallel = malloc(size * sizeof(double));

        srand48(42);

        int limit = (size < 100) ? size : 100;
        for (int j = 0; j < limit; ++j) A[0 * size + j] = drand48();

        if (size > 1 && size > 100) {
            int copy_len = (size - 100 < 100) ? (size - 100) : 100;
            for (int j = 0; j < copy_len; ++j) A[1 * size + (100 + j)] = A[0 * size + j];
        }

        for (int i = 0; i < size; ++i) A[i * size + i] = drand48();
        for (int i = 0; i < size; ++i) b[i] = drand48();

        double start_serial = MPI_Wtime();
        matrixVectorMult(A, b, x_serial, size);
        double end_serial = MPI_Wtime();
        serial_time = end_serial - start_serial;
    }

    MPI_Bcast(b, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *sendcounts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));
    int *recvcounts = malloc(num_procs * sizeof(int));
    int *recv_displs = malloc(num_procs * sizeof(int));

    int offset = 0;
    for (int i = 0; i < num_procs; ++i) {
        int local_rows = size / num_procs + (i < (size % num_procs) ? 1 : 0);
        sendcounts[i] = local_rows * size;
        displs[i] = offset * size;
        recvcounts[i] = local_rows;
        recv_displs[i] = offset;
        offset += local_rows;
    }

    int my_rows = size / num_procs + (rank < (size % num_procs) ? 1 : 0);
    double *local_A = malloc(my_rows * size * sizeof(double));
    double *local_x = malloc(my_rows * sizeof(double));

    double start_parallel = MPI_Wtime();

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, my_rows * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < my_rows; ++i) {
        local_x[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            local_x[i] += local_A[i * size + j] * b[j];
        }
    }

    MPI_Gatherv(local_x, my_rows, MPI_DOUBLE, x_parallel, recvcounts, recv_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_parallel = MPI_Wtime();
    double parallel_time = end_parallel - start_parallel;

    if (rank == 0) {
        printf("\n--- RESULTS ---\n");
        printf("Serial Time:   %f seconds\n", serial_time);
        printf("Parallel Time: %f seconds\n", parallel_time);
        double speedup = serial_time / parallel_time;
        printf("Speedup:       %f\n", speedup);
        printf("Efficiency:    %f\n", speedup / num_procs);
        printf("---------------\n");

        free(A);
        free(x_serial);
        free(x_parallel);
    }

    free(b);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recv_displs);
    free(local_A);
    free(local_x);

    MPI_Finalize();
    return 0;
}
