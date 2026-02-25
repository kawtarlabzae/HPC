#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, value;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    value = 0;
    do {
        if (rank == 0) scanf("%d", &value);
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (value >= 0)
            printf("I, process %d, received %d of process 0\n", rank, value);
    } while (value >= 0);
    MPI_Finalize();
}
