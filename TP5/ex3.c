#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int value = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("Please run this program with at least 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        fflush(stdout);
        scanf("%d", &value);
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent initial value %d to Process 1\n", value);
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += rank;
        printf("Process 0 received the final value %d from Process %d to close the ring.\n", value, size - 1);
    } else {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += rank;
        printf("Process %d received data, added its rank. New value: %d\n", rank, value);
        int next_process = (rank + 1) % size;
        MPI_Send(&value, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
