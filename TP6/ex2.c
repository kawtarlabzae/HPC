#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N_FEATURES 2
#define TOTAL_SAMPLES 1000
#define EPOCHS 1000
#define LEARNING_RATE 0.01
#define THRESHOLD 1.0e-02

// Our training sample structure
typedef struct {
    double x[N_FEATURES];
    double y;
} Sample;

// A dummy function to generate synthetic linear data on Process 0
void generate_data(Sample *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i].x[0] = (double)rand() / RAND_MAX;
        data[i].x[1] = (double)rand() / RAND_MAX;
        // True relation: y = 2.0 * x[0] - 1.0 * x[1] + noise
        data[i].y = 2.0 * data[i].x[0] - 1.0 * data[i].x[1] + ((double)rand() / RAND_MAX * 0.1);
    }
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    // ------------------------------------------------------------------------
    // 1. Define the Derived Type for the Sample Struct
    // ------------------------------------------------------------------------
    int blocklengths[2] = {N_FEATURES, 1}; // 1 array of N_FEATURES, 1 scalar
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[2];

    // We create a dummy instance just to measure where the compiler placed things in memory
    Sample dummy;
    MPI_Aint base_address;

    // Get the exact memory addresses of the struct itself, and its internal variables
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.x, &offsets[0]);
    MPI_Get_address(&dummy.y, &offsets[1]);

    // Calculate the absolute byte distance from the start of the struct to the variables
    offsets[0] = MPI_Aint_diff(offsets[0], base_address);
    offsets[1] = MPI_Aint_diff(offsets[1], base_address);

    // Build and commit the blueprint so MPI knows how to navigate the struct's memory
    MPI_Datatype MPI_SAMPLE;
    MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_SAMPLE);
    MPI_Type_commit(&MPI_SAMPLE);

    // ------------------------------------------------------------------------
    // Data Distribution Prep
    // ------------------------------------------------------------------------
    Sample *dataset = NULL;
    if (rank == 0) {
        dataset = (Sample*)malloc(TOTAL_SAMPLES * sizeof(Sample));
        generate_data(dataset, TOTAL_SAMPLES); // 2. Generate full dataset
    }

    // Calculate how many samples each process gets.
    // If TOTAL_SAMPLES doesn't divide perfectly by 'size', we calculate counts and displacements for MPI_Scatterv
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = TOTAL_SAMPLES % size;
    int sum = 0;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = TOTAL_SAMPLES / size + (i < rem ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    int local_n = sendcounts[rank];
    Sample *local_data = (Sample*)malloc(local_n * sizeof(Sample));

    // 3. Scatter the dataset to all processes using our custom MPI_SAMPLE type
    // Process 0 slices the 'dataset' array and deals it out.
    // Everyone receives their specific chunk into 'local_data'.
    MPI_Scatterv(dataset, sendcounts, displs, MPI_SAMPLE,
                 local_data, local_n, MPI_SAMPLE, 0, MPI_COMM_WORLD);

    // Initialize the weight vector (everyone starts with the exact same initial guess)
    double w[N_FEATURES] = {0.0, 0.0};

    // ------------------------------------------------------------------------
    // The Training Loop (Distributed Gradient Descent)
    // ------------------------------------------------------------------------
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        double local_loss_sum = 0.0;
        double local_grad_sum[N_FEATURES] = {0.0};

        // 4. Each process computes its local loss and gradient over its assigned chunk of data
        for (int i = 0; i < local_n; i++) {
            double y_pred = 0.0;
            for (int j = 0; j < N_FEATURES; j++) {
                y_pred += w[j] * local_data[i].x[j];
            }

            double error = y_pred - local_data[i].y;
            local_loss_sum += error * error; // Squared error

            for (int j = 0; j < N_FEATURES; j++) {
                local_grad_sum[j] += error * local_data[i].x[j];
            }
        }

        // 5. Aggregate gradients and losses
        double global_loss_sum = 0.0;
        double global_grad_sum[N_FEATURES] = {0.0};

        // MPI_Allreduce gathers the local sums from all processes, adds them together (MPI_SUM),
        // and broadcasts the final total back to every process simultaneously.
        MPI_Allreduce(&local_loss_sum, &global_loss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_grad_sum, global_grad_sum, N_FEATURES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double mse = global_loss_sum / TOTAL_SAMPLES;

        // 6. All processes update their local copy of the weight vector synchronously
        for (int j = 0; j < N_FEATURES; j++) {
            double avg_grad = global_grad_sum[j] / TOTAL_SAMPLES;
            w[j] = w[j] - LEARNING_RATE * avg_grad;
        }

        // 7. Print loss and weight updates every 10 epochs from process 0
        if (rank == 0 && epoch % 10 == 0) {
            printf("Epoch %d | Loss (MSE): %f | w[0]: %.4f, w[1]: %.4f\n",
                   epoch, mse, w[0], w[1]);
        }

        // 8. Stop early if the global loss becomes smaller than the predefined threshold
        if (mse < THRESHOLD) {
            if (rank == 0) {
                printf("Early stopping at epoch %d — loss %f < %e\n", epoch, mse, THRESHOLD);
            }
            break;
        }
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Training time: %.3f seconds (MPI)\n", end_time - start_time);
    }

    // Clean up memory and MPI types
    if (rank == 0) free(dataset);
    free(local_data);
    free(sendcounts);
    free(displs);
    MPI_Type_free(&MPI_SAMPLE);
    MPI_Finalize();

    return 0;
}
