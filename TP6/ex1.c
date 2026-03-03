#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure the program is run with at least 2 processes to allow for sending and receiving
    if (size < 2) {
        if (rank == 0) {
            printf("Please run with at least 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {

        int a[4][5];


        int count = 1;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                a[i][j] = count++;
            }
        }


        printf("Matrix A (Process 0):\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%2d ", a[i][j]);
            }
            printf("\n");
        }
        printf("\n");


        MPI_Send(&a[0][0], 20, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        int at[5][4];

        MPI_Datatype column_type, transpose_type;

        // It expects 5 items. The stride is 4, meaning it skips 4 memory spaces, the 1 if for blocklenght
        // how many blocks does it take before skipping (using the stride)

        MPI_Type_vector(5, 1, 4, MPI_INT, &column_type);

         // It places 4 of these columns together. The stride is sizeof(int),
        // meaning each new column starts exactly one integer to the right of the previous one.
        MPI_Type_create_hvector(4, 1, sizeof(int), column_type, &transpose_type);

        // The custom datatype must be committed to the MPI environment before it can be used
        MPI_Type_commit(&transpose_type);

        MPI_Recv(&at[0][0], 1, transpose_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        printf("Transposed Matrix AT (Process 1):\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%2d ", at[i][j]);
            }
            printf("\n");
        }

        // Custom datatypes are freed from memory once they are no longer needed to prevent memory leaks
        MPI_Type_free(&column_type);
        MPI_Type_free(&transpose_type);
    }

    MPI_Finalize();
    return 0;
}
