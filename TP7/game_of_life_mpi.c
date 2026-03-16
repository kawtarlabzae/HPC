/* ================================================================
 * TP7 - MPI Communicators
 * Exercise 1: Conway's Game of Life — Parallel MPI Implementation
 * ================================================================
 *
 * PROBLEM:
 *   Simulate Conway's Game of Life on a 2D grid distributed across
 *   multiple MPI processes using a 2D Cartesian topology with
 *   periodic (wrap-around) boundary conditions.
 *
 * RULES:
 *   1. A live cell with < 2 live neighbours dies (underpopulation)
 *   2. A live cell with 2 or 3 live neighbours survives
 *   3. A live cell with > 3 live neighbours dies (overpopulation)
 *   4. A dead cell with exactly 3 live neighbours becomes alive
 *
 * COMPILATION:
 *   mpicc -O2 -Wall -o game_of_life game_of_life_mpi.c
 *
 * EXECUTION:
 *   mpirun -n 4 ./game_of_life
 *   mpirun -n 9 ./game_of_life
 * ================================================================ */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Constants
 * --------------------------------------------------------------- */
#define ALIVE 1
#define DEAD  0

/* ---------------------------------------------------------------
 * IDX: Map local grid (i, j) to 1D array index.
 * The local array has nx_halo × ny_halo cells, stored row-major.
 *   - i = 0 and i = nx_halo-1 are the north/south ghost rows
 *   - j = 0 and j = ny_halo-1 are the west/east ghost columns
 *   - Interior cells: i in [1..local_nx], j in [1..local_ny]
 * --------------------------------------------------------------- */
#define IDX(i, j, ny_h) ((i) * (ny_h) + (j))

/* ================================================================
 * count_neighbors
 *   Return the number of live cells among the 8 Moore neighbours
 *   of cell (i, j).  Ghost cells must be up-to-date before this
 *   is called.
 * ================================================================ */
static int count_neighbors(const int *grid, int i, int j, int ny_halo)
{
    int count = 0;
    /* iterate over all 8 surrounding cells (3x3 minus centre) */
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            if (di != 0 || dj != 0)
                count += grid[IDX(i + di, j + dj, ny_halo)];
    return count;
}

/* ================================================================
 * apply_rules
 *   Apply Conway's rules to every interior cell, writing results
 *   into new_grid.  Ghost cells are NOT updated here.
 * ================================================================ */
static void apply_rules(const int *grid, int *new_grid,
                         int local_nx, int local_ny, int ny_halo)
{
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            int nbrs = count_neighbors(grid, i, j, ny_halo);
            int cell = grid[IDX(i, j, ny_halo)];

            if (cell == ALIVE)
                /* Survive with 2 or 3 neighbours, otherwise die */
                new_grid[IDX(i, j, ny_halo)] = (nbrs == 2 || nbrs == 3) ? ALIVE : DEAD;
            else
                /* Dead cell with exactly 3 neighbours comes alive */
                new_grid[IDX(i, j, ny_halo)] = (nbrs == 3) ? ALIVE : DEAD;
        }
    }
}

/* ================================================================
 * print_local_grid
 *   Print this process's interior subgrid (without ghost cells).
 *   Call sequentially (with MPI_Barrier) to avoid interleaving.
 * ================================================================ */
static void print_local_grid(const int *grid, int cart_rank, int gen,
                              int local_nx, int local_ny, int ny_halo)
{
    printf("Rank %d - Generation %d:\n", cart_rank, gen);
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++)
            printf("%d ", grid[IDX(i, j, ny_halo)]);
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(int argc, char *argv[])
{
    int rank, size;

    /* ---- MPI Initialisation ---- */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Problem parameters (can be changed via argv) ---- */
    int global_nx = 12;   /* total rows of the global grid    */
    int global_ny = 12;   /* total columns of the global grid */
    int G         = 10;   /* number of generations to simulate */

    if (argc >= 3) {
        global_nx = atoi(argv[1]);
        global_ny = atoi(argv[2]);
    }
    if (argc >= 4) G = atoi(argv[3]);

    /* ============================================================
     * STEP 1 – Compute 2D process grid dimensions
     * ============================================================
     * MPI_Dims_create fills dims[] with the most balanced factori-
     * sation of 'size' into 2 dimensions.
     *   dims[0] = Px  (processes along x / rows)
     *   dims[1] = Py  (processes along y / columns)
     * ============================================================ */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Px = dims[0];
    int Py = dims[1];

    /* ============================================================
     * STEP 2 – Create 2D Cartesian communicator
     * ============================================================
     * periods = {1, 1}  →  PERIODIC in both dimensions.
     * This means processes at the boundary are wrapped around, so
     * every process has exactly 4 neighbours (north, south, east,
     * west) — implementing the wrap-around boundary conditions.
     * reorder = 1 allows MPI to reorder ranks for better locality.
     * ============================================================ */
    int periods[2] = {1, 1};   /* periodic: wrap-around */
    int reorder    = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    /* Get our rank inside the new Cartesian communicator */
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    /* ============================================================
     * STEP 3 – Determine our 2D coordinates in the process grid
     * ============================================================
     * coords[0] = our row    position in the Px × Py process grid
     * coords[1] = our column position in the Px × Py process grid
     * ============================================================ */
    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int coord_x = coords[0];   /* position along rows    */
    int coord_y = coords[1];   /* position along columns */

    if (rank == 0) {
        printf("=== Conway's Game of Life — MPI Parallel ===\n");
        printf("Global grid : %d x %d cells\n", global_nx, global_ny);
        printf("Generations : %d\n", G);
        printf("Processes   : %d (topology %d x %d)\n\n", size, Px, Py);
    }
    MPI_Barrier(cart_comm);

    /* ============================================================
     * STEP 4 – Compute local subdomain size
     * ============================================================
     * Distribute global_nx rows over Px processes and global_ny
     * columns over Py processes.  Remainder cells go to the first
     * rem_x (resp. rem_y) processes.
     * ============================================================ */
    int rem_x    = global_nx % Px;
    int rem_y    = global_ny % Py;
    int local_nx = global_nx / Px + (coord_x < rem_x ? 1 : 0);
    int local_ny = global_ny / Py + (coord_y < rem_y ? 1 : 0);

    /* Global starting index of our subdomain (0-based) */
    int start_x = coord_x * (global_nx / Px) + (coord_x < rem_x ? coord_x : rem_x);
    int start_y = coord_y * (global_ny / Py) + (coord_y < rem_y ? coord_y : rem_y);

    /* ============================================================
     * STEP 5 – Allocate local grids WITH halo (ghost cell) border
     * ============================================================
     * Each dimension is extended by 2 (one ghost cell on each side):
     *
     *   nx_halo = local_nx + 2
     *   ny_halo = local_ny + 2
     *
     * Memory layout (ny_halo columns per row):
     *
     *   row 0              ← north ghost row  (filled by north neighbour)
     *   rows 1..local_nx   ← interior cells   (owned by this process)
     *   row local_nx+1     ← south ghost row  (filled by south neighbour)
     *
     *   col 0              ← west ghost col
     *   cols 1..local_ny   ← interior cols
     *   col local_ny+1     ← east ghost col
     *
     * calloc initialises everything to 0.
     * ============================================================ */
    int nx_halo = local_nx + 2;
    int ny_halo = local_ny + 2;

    int *grid     = (int *)calloc(nx_halo * ny_halo, sizeof(int));
    int *new_grid = (int *)calloc(nx_halo * ny_halo, sizeof(int));
    if (!grid || !new_grid) {
        fprintf(stderr, "Rank %d: allocation failure\n", cart_rank);
        MPI_Abort(cart_comm, 1);
    }

    /* ============================================================
     * STEP 6 – Find 4 direct neighbours with MPI_Cart_shift
     * ============================================================
     * MPI_Cart_shift(comm, direction, displacement, &source, &dest)
     *
     *   direction = 0   →  move along x (row direction)
     *   direction = 1   →  move along y (column direction)
     *   displacement= 1 →  positive direction
     *
     *   source (returned): rank of the process that is "behind" us
     *                      (i.e., our north / west neighbour)
     *   dest   (returned): rank of the process that is "ahead" of us
     *                      (i.e., our south / east neighbour)
     *
     * With periodic BC all neighbours are valid (no MPI_PROC_NULL).
     * ============================================================ */
    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  /* x direction */
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);   /* y direction */

    /* Print topology info sequentially */
    for (int r = 0; r < size; r++) {
        if (cart_rank == r) {
            printf("Rank %d  coords [%d,%d]  local=%dx%d  "
                   "N=%d S=%d W=%d E=%d\n",
                   cart_rank, coord_x, coord_y,
                   local_nx, local_ny,
                   north, south, west, east);
            fflush(stdout);
        }
        MPI_Barrier(cart_comm);
    }
    if (rank == 0) printf("\n");

    /* ============================================================
     * STEP 7 – Initialise local grid with random values
     * ============================================================
     * Use a seed derived from the process's global start position
     * so that the initial pattern is deterministic and reproducible
     * regardless of the number of processes.
     * Only interior cells [1..local_nx] × [1..local_ny] are set.
     * ============================================================ */
    unsigned int seed = 42 + (unsigned int)(start_x * global_ny + start_y);
    srand(seed);
    for (int i = 1; i <= local_nx; i++)
        for (int j = 1; j <= local_ny; j++)
            grid[IDX(i, j, ny_halo)] = rand() % 2;

    /* ============================================================
     * STEP 8 – Create MPI derived type for column exchange
     * ============================================================
     * Rows are contiguous in memory (easy to send with MPI_INT).
     * Columns are NOT contiguous: consecutive elements of column j
     * are separated by ny_halo integers in memory.
     *
     * MPI_Type_vector creates a strided vector type:
     *   count       = nx_halo   (total rows including ghosts)
     *   blocklength = 1         (one element per row)
     *   stride      = ny_halo   (row width = distance between rows)
     *
     * By spanning ALL rows (nx_halo, not just local_nx), we fill
     * the 4 corner ghost cells automatically when we exchange the
     * columns AFTER the row exchange has already populated the
     * ghost rows.
     * ============================================================ */
    MPI_Datatype full_col_type;
    MPI_Type_vector(nx_halo,    /* count:  all rows including ghosts */
                    1,          /* blocklength: one cell per row     */
                    ny_halo,    /* stride: full row width             */
                    MPI_INT, &full_col_type);
    MPI_Type_commit(&full_col_type);

    /* ============================================================
     * MAIN GENERATION LOOP
     * ============================================================
     * Each generation follows this sequence:
     *
     *   1. Exchange north/south ghost rows
     *   2. Exchange east/west ghost columns (incl. corners)
     *   3. Apply Conway's rules on interior cells
     *   4. Swap grid pointers (new_grid → grid, no data copy)
     *
     * Communication pattern for MPI_Sendrecv:
     *
     *   MPI_Sendrecv(sendbuf, ..., dest,   sendtag,
     *                recvbuf, ..., source, recvtag, ...)
     *
     *   Northward sends use tag 0; southward sends use tag 1.
     *   Westward  sends use tag 2; eastward  sends use tag 3.
     *   This avoids tag collisions between the two calls in each
     *   direction while still being consistent across processes.
     * ============================================================ */
    for (int gen = 1; gen <= G; gen++) {

        /* ====================================================
         * Ghost row exchange (north ↔ south along x)
         * ====================================================
         *
         * First Sendrecv:
         *   Send  : my top interior row (i=1) → north neighbour
         *   Recv  : south neighbour's top row → my bottom ghost (i=nx_halo-1)
         *
         * Why different source/dest in one Sendrecv?
         *   MPI_Sendrecv can send to one neighbour and receive from
         *   ANOTHER in the same call.  This ring-style exchange
         *   ensures that after both calls every process has its
         *   north and south ghost rows correctly filled.
         * ==================================================== */

        /* Send top row to north; receive bottom ghost from south */
        MPI_Sendrecv(
            &grid[IDX(1, 1, ny_halo)],         /* send: row 1, cols 1..local_ny */
            local_ny, MPI_INT, north, 0,
            &grid[IDX(local_nx+1, 1, ny_halo)],/* recv: bottom ghost row        */
            local_ny, MPI_INT, south, 0,
            cart_comm, MPI_STATUS_IGNORE
        );

        /* Send bottom row to south; receive top ghost from north */
        MPI_Sendrecv(
            &grid[IDX(local_nx, 1, ny_halo)],  /* send: row local_nx            */
            local_ny, MPI_INT, south, 1,
            &grid[IDX(0, 1, ny_halo)],          /* recv: top ghost row           */
            local_ny, MPI_INT, north, 1,
            cart_comm, MPI_STATUS_IGNORE
        );

        /* ====================================================
         * Ghost column exchange (west ↔ east along y)
         * ====================================================
         *
         * We use full_col_type which spans ALL rows (0..nx_halo-1),
         * so the ghost row entries (row 0 and row nx_halo-1) are
         * included.  This fills the 4 corner ghost cells without
         * needing separate diagonal communications.
         *
         * First Sendrecv:
         *   Send  : my left interior col (j=1)       → west neighbour
         *   Recv  : east neighbour's col j=1          → my right ghost
         * ==================================================== */

        /* Send left col to west; receive right ghost from east */
        MPI_Sendrecv(
            &grid[IDX(0, 1, ny_halo)],          /* send: full col j=1            */
            1, full_col_type, west, 2,
            &grid[IDX(0, local_ny+1, ny_halo)], /* recv: full right ghost col    */
            1, full_col_type, east, 2,
            cart_comm, MPI_STATUS_IGNORE
        );

        /* Send right col to east; receive left ghost from west */
        MPI_Sendrecv(
            &grid[IDX(0, local_ny, ny_halo)],   /* send: full col j=local_ny     */
            1, full_col_type, east, 3,
            &grid[IDX(0, 0, ny_halo)],           /* recv: full left ghost col     */
            1, full_col_type, west, 3,
            cart_comm, MPI_STATUS_IGNORE
        );

        /* ====================================================
         * Apply Conway's rules to all interior cells
         * (ghost cells already contain valid neighbour data)
         * ==================================================== */
        apply_rules(grid, new_grid, local_nx, local_ny, ny_halo);

        /* ====================================================
         * Swap pointers: new_grid becomes the current grid
         * No memcpy needed — just swap the pointer values.
         * ==================================================== */
        int *tmp  = grid;
        grid      = new_grid;
        new_grid  = tmp;
    }

    /* ============================================================
     * STEP 9 – Print final state
     * ============================================================
     * Serialise output with MPI_Barrier to avoid interleaving.
     * Each process prints its local subgrid in rank order.
     * ============================================================ */
    if (rank == 0)
        printf("=== Final state after %d generations ===\n", G);
    MPI_Barrier(cart_comm);

    for (int r = 0; r < size; r++) {
        if (cart_rank == r)
            print_local_grid(grid, cart_rank, G, local_nx, local_ny, ny_halo);
        MPI_Barrier(cart_comm);
    }

    /* ---- Cleanup ---- */
    MPI_Type_free(&full_col_type);
    free(grid);
    free(new_grid);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
