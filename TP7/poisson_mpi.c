/* ================================================================
 * TP7 - MPI Communicators
 * Exercise 2: Parallel Poisson Solver (Jacobi Iteration)
 * ================================================================
 *
 * PROBLEM:
 *   Solve  −Δu = f  on [0,1]×[0,1]  with u=0 on ∂Ω
 *   f(x,y) = 2·(x²−x + y²−y)
 *   Exact:  u(x,y) = x·y·(x−1)·(y−1)
 *
 * METHOD:
 *   Finite-difference discretisation + Jacobi iterative solver
 *   distributed over a 2D MPI Cartesian process topology.
 *
 * COMPILATION:
 *   mpicc -O2 -Wall -o poisson poisson_mpi.c -lm
 *
 * EXECUTION:
 *   mpirun -n 4 ./poisson           (default grid 12×10)
 *   mpirun -n 4 ./poisson 100 80    (custom grid 100×80)
 * ================================================================ */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ================================================================
 * Global grid parameters
 * ================================================================
 * These variables are set once (in main, from the MPI domain
 * decomposition) and then used by the helper functions that
 * mirror compute.c (initialization, compute, output_results).
 * ================================================================ */
int ntx, nty;   /* total interior points in x and y (global grid) */
int sx, ex;     /* this process's x range: interior rows [sx..ex] */
int sy, ey;     /* this process's y range: interior cols [sy..ey] */

/*
 * IDX  –  Map global (i,j) indices to a 1D local array index.
 *
 * The local array has (ex−sx+3) × (ey−sy+3) elements:
 *   · (ex−sx+1) interior rows  + 1 ghost row  on each side
 *   · (ey−sy+1) interior cols  + 1 ghost col  on each side
 *
 * Index ranges visible through IDX:
 *   i  ∈  [sx−1 .. ex+1]   (sx−1 and ex+1 are ghost rows)
 *   j  ∈  [sy−1 .. ey+1]   (sy−1 and ey+1 are ghost cols)
 *
 * Row width (number of columns in local storage):
 *   ey−sy+3  =  (ey−sy+1 interior) + 2 ghost
 */
#define IDX(i, j) ( ((i)-(sx-1)) * (ey-sy+3) + ((j)-(sy-1)) )

/* Right-hand side  f(xi, yj) */
static double *f;

/* Precomputed Jacobi coefficients (see initialization) */
static double coef[3];

/* ================================================================
 * initialization
 * ================================================================
 * Allocate all local arrays (including ghost-cell borders) and
 * fill the right-hand side f and the exact solution u_exact for
 * interior points owned by this process.
 *
 * Grid spacing:
 *   hx = 1/(ntx+1)   hy = 1/(nty+1)
 * Interior points:
 *   xi = i·hx  for i=1..ntx        boundary: x0=0, x_{ntx+1}=1
 *   yj = j·hy  for j=1..nty
 *
 * Jacobi coefficients (derived from 5-point finite-difference):
 *   coef[0] = 0.5·hx²·hy² / (hx²+hy²)    [overall normalisation]
 *   coef[1] = 1/hx²                        [weight of x-neighbours]
 *   coef[2] = 1/hy²                        [weight of y-neighbours]
 *
 * Iteration formula:
 *   u^{n+1}_{i,j} = coef[0] · ( coef[1]·(u^n_{i+1,j}+u^n_{i-1,j})
 *                              + coef[2]·(u^n_{i,j+1}+u^n_{i,j-1})
 *                              − f_{i,j} )
 * ================================================================ */
void initialization(double **pu, double **pu_new, double **pu_exact)
{
    double hx, hy;
    int    i, j;
    double x, y;

    /* Allocate local arrays (calloc → all zeros = Dirichlet BC) */
    *pu       = (double *)calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
    *pu_new   = (double *)calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
    *pu_exact = (double *)calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
    f         = (double *)calloc((ex-sx+3) * (ey-sy+3), sizeof(double));

    if (!*pu || !*pu_new || !*pu_exact || !f) {
        fprintf(stderr, "[initialization] Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Grid spacing */
    hx = 1.0 / (ntx + 1.0);
    hy = 1.0 / (nty + 1.0);

    /*
     * Jacobi iteration coefficients:
     *
     *  coef[0] ensures the formula equals the analytical expression
     *          for ui,j after isolating it from the discrete Laplacian.
     *          (See Laplacian.pdf Section 5-6 for derivation.)
     */
    coef[0] = (0.5 * hx * hx * hy * hy) / (hx * hx + hy * hy);
    coef[1] = 1.0 / (hx * hx);
    coef[2] = 1.0 / (hy * hy);

    /* Initialise f and u_exact for every interior point this process owns */
    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            x = i * hx;
            y = j * hy;
            /* Source term: f(x,y) = 2·(x²−x + y²−y)  =  −Δu_exact */
            f[IDX(i, j)] = 2.0 * (x*x - x + y*y - y);
            /* Exact solution: u(x,y) = x·y·(x−1)·(y−1) */
            (*pu_exact)[IDX(i, j)] = x * y * (x - 1.0) * (y - 1.0);
        }
    }
}

/* ================================================================
 * compute
 * ================================================================
 * Perform one Jacobi iteration: read from u[], write to u_new[].
 *
 * Only interior cells [sx..ex] × [sy..ey] are updated.
 * Ghost cells (sx−1, ex+1, sy−1, ey+1) must already contain the
 * correct neighbour values (filled by MPI halo exchange).
 *
 * 5-point stencil:
 *
 *        u[i-1,j]
 *           |
 *  u[i,j-1]—u[i,j]—u[i,j+1]
 *           |
 *        u[i+1,j]
 * ================================================================ */
void compute(double *u, double *u_new)
{
    int i, j;
    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            u_new[IDX(i, j)] =
                coef[0] * (
                    coef[1] * (u[IDX(i+1, j)] + u[IDX(i-1, j)]) +  /* x-neighbours */
                    coef[2] * (u[IDX(i, j+1)] + u[IDX(i, j-1)]) -  /* y-neighbours */
                    f[IDX(i, j)]                                      /* source term  */
                );
        }
    }
}

/* ================================================================
 * output_results
 * ================================================================
 * Print a column-by-column comparison of exact vs computed solution
 * for the first global row (i=1).
 * IMPORTANT: only call this from a process that owns row i=1,
 *            i.e. where sx == 1.
 * ================================================================ */
void output_results(const double *u, const double *u_exact)
{
    int j;
    printf("Exact solution u_exact - Computed solution u\n");
    for (j = sy; j <= ey; j++)
        printf("%12.5e - %12.5e\n",
               u_exact[IDX(1, j)],
               u[IDX(1, j)]);
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

    /* ---- Problem parameters ---- */
    ntx = 12;   /* default: interior points in x */
    nty = 10;   /* default: interior points in y */
    if (argc >= 3) {
        ntx = atoi(argv[1]);
        nty = atoi(argv[2]);
    }

    int    max_iter  = 100000;    /* maximum Jacobi iterations  */
    double tolerance = 1.0e-6;   /* convergence threshold (max-norm) */

    /* ============================================================
     * STEP 1 – Compute 2D process grid dimensions
     * ============================================================
     * MPI_Dims_create distributes 'size' processes as evenly as
     * possible in 2D.  For 4 processes → dims = {2, 2}.
     * ============================================================ */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Px = dims[0];   /* processes along x (rows)    */
    int Py = dims[1];   /* processes along y (columns) */

    if (rank == 0) {
        printf("Poisson execution with %d MPI processes\n", size);
        printf("Domain size: ntx=%d nty=%d\n", ntx, nty);
        printf("Topology dimensions: %d along x, %d along y\n", Px, Py);
        printf("-----------------------------------------\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ============================================================
     * STEP 2 – Create 2D Cartesian communicator
     * ============================================================
     * periods = {0, 0}  →  NON-PERIODIC (Dirichlet BC).
     * Ghost cells at domain boundaries stay 0 (from calloc),
     * which correctly enforces u=0 on ∂Ω.
     * MPI_PROC_NULL is returned for neighbours that do not exist
     * (boundary processes); MPI_Sendrecv with MPI_PROC_NULL is a
     * no-op, so boundary ghost cells remain 0.
     * ============================================================ */
    int periods[2] = {0, 0};   /* non-periodic: Dirichlet boundaries */
    int reorder    = 0;        /* keep original rank order for readability */
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    /* Get rank in Cartesian communicator */
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    /* ============================================================
     * STEP 3 – Determine 2D coordinates in the process grid
     * ============================================================
     * coords[0] = position along x (row index in process grid)
     * coords[1] = position along y (col index in process grid)
     * ============================================================ */
    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int coord_x = coords[0];
    int coord_y = coords[1];

    /* ============================================================
     * STEP 4 – Compute subdomain boundaries [sx..ex] × [sy..ey]
     * ============================================================
     * Global interior indices:  1 ≤ i ≤ ntx  and  1 ≤ j ≤ nty
     *
     * We distribute ntx points over Px processes:
     *   base size  =  ntx / Px
     *   remainder  =  ntx % Px  (first rem_x processes get +1)
     *
     * Global start index for coord_x:
     *   sx = coord_x * base + min(coord_x, rem_x) + 1
     *
     * Example (ntx=12, Px=2):
     *   coord_x=0: local_nx=6, sx=1,  ex=6
     *   coord_x=1: local_nx=6, sx=7,  ex=12
     * ============================================================ */
    int rem_x    = ntx % Px;
    int rem_y    = nty % Py;
    int local_nx = ntx / Px + (coord_x < rem_x ? 1 : 0);
    int local_ny = nty / Py + (coord_y < rem_y ? 1 : 0);

    sx = coord_x * (ntx / Px) + (coord_x < rem_x ? coord_x : rem_x) + 1;
    sy = coord_y * (nty / Py) + (coord_y < rem_y ? coord_y : rem_y) + 1;
    ex = sx + local_nx - 1;
    ey = sy + local_ny - 1;

    /* ============================================================
     * STEP 5 – Find 4 direct neighbours with MPI_Cart_shift
     * ============================================================
     * MPI_Cart_shift(comm, direction, disp, &source, &dest):
     *   direction=0, disp=+1  →  source=north (coord_x−1),
     *                             dest  =south (coord_x+1)
     *   direction=1, disp=+1  →  source=west  (coord_y−1),
     *                             dest  =east  (coord_y+1)
     *
     * Convention used in this code:
     *   north = smaller x index  (above in the grid picture)
     *   south = larger  x index  (below)
     *   west  = smaller y index  (left)
     *   east  = larger  y index  (right)
     *
     * Boundary processes get MPI_PROC_NULL for missing neighbours.
     * ============================================================ */
    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  /* x direction */
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);   /* y direction */

    /* Print topology info in rank order */
    for (int r = 0; r < size; r++) {
        if (cart_rank == r) {
            printf("Rank in the topology: %d   "
                   "Array indices: x from %d to %d, y from %d to %d\n",
                   cart_rank, sx, ex, sy, ey);
            printf("Process %d has neighbors: N %d  E %d  S %d  W %d\n",
                   cart_rank, north, east, south, west);
            fflush(stdout);
        }
        MPI_Barrier(cart_comm);
    }
    if (rank == 0) printf("\n");

    /* ============================================================
     * STEP 6 – Initialise local arrays
     * ============================================================
     * Calls initialization() which:
     *   · allocates u, u_new, u_exact, f  (with ghost borders)
     *   · fills f[i,j] and u_exact[i,j] for interior points
     *   · leaves ghost rows/cols as 0 (= Dirichlet BC u=0)
     * ============================================================ */
    double *u, *u_new, *u_exact;
    initialization(&u, &u_new, &u_exact);

    /* ============================================================
     * Create MPI derived types for non-contiguous column exchange
     * ============================================================
     * In row-major storage, a column is NOT contiguous:
     *   consecutive elements of column j in row i and row i+1
     *   are separated by (ey−sy+3) doubles in memory.
     *
     * MPI_Type_vector(count, blocklength, stride, base, &type):
     *   count       = local_nx        (one element per interior row)
     *   blocklength = 1               (one double per block)
     *   stride      = ey−sy+3         (row width including ghosts)
     *
     * This type selects the local_nx elements of a single column.
     * ============================================================ */
    int row_width = ey - sy + 3;   /* width of local storage (incl. ghosts) */

    MPI_Datatype col_type;
    MPI_Type_vector(local_nx,   /* number of rows (interior only) */
                    1,          /* one double per row             */
                    row_width,  /* stride = full row width        */
                    MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    /* ============================================================
     * JACOBI ITERATION LOOP
     * ============================================================
     *
     * At every iteration:
     *
     *   ① Halo exchange (ghost rows, then ghost columns)
     *      Each ghost boundary gets the adjacent interior border
     *      from the corresponding neighbour process.
     *
     *   ② Jacobi update: compute(u, u_new)
     *      Applies the 5-point stencil to every interior point.
     *
     *   ③ Residual: max-norm of the update ‖u_new − u‖∞
     *      MPI_Allreduce with MPI_MAX gives the global maximum.
     *
     *   ④ Swap u ↔ u_new pointers (zero-cost, no memcpy).
     *
     * Communication tags:
     *   Northward data (process → north neighbour) : tag 10
     *   Southward data (process → south neighbour) : tag 20
     *   Westward  data (process → west  neighbour) : tag 30
     *   Eastward  data (process → east  neighbour) : tag 40
     *
     * Using different tags for each direction ensures that a
     * process sending to its south neighbour and receiving from
     * its north neighbour in the SAME MPI_Sendrecv call will
     * not accidentally match the wrong message.
     * ============================================================ */
    double global_error = 1.0;
    int    iter         = 0;
    double t_start      = MPI_Wtime();

    while (global_error > tolerance && iter < max_iter) {
        iter++;

        /* ====================================================
         * ① Ghost row exchange  (north ↔ south along x)
         * ====================================================
         *
         * Call A:
         *   send  my top interior row (i=sx)   → north  [tag 10]
         *   recv  bottom ghost row   (i=ex+1)  ← south  [tag 20]
         *
         *   My south does Call B:
         *     send  its bottom row (i=ex)      → south's south  [tag 20]
         *     recv  its top ghost  (i=sx-1)    ← south's north  [tag 10]
         *   … but it also does its own Call A where it sends its
         *   top row to me (its north) with tag 10.  This matches
         *   my Call B recv from north with tag 10 (see below). ✓
         * ==================================================== */

        /* Call A: send top row northward; recv south's top row into bottom ghost */
        MPI_Sendrecv(
            &u[IDX(sx,    sy)], local_ny, MPI_DOUBLE, north, 10,  /* → north */
            &u[IDX(ex+1,  sy)], local_ny, MPI_DOUBLE, south, 10,  /* ← south */
            cart_comm, MPI_STATUS_IGNORE
        );

        /* Call B: send bottom row southward; recv north's bottom row into top ghost */
        MPI_Sendrecv(
            &u[IDX(ex,    sy)], local_ny, MPI_DOUBLE, south, 20,  /* → south */
            &u[IDX(sx-1,  sy)], local_ny, MPI_DOUBLE, north, 20,  /* ← north */
            cart_comm, MPI_STATUS_IGNORE
        );

        /* ====================================================
         * ① Ghost column exchange  (west ↔ east along y)
         * ====================================================
         *
         * Uses col_type to send/receive non-contiguous column data.
         *
         * Call C: send left interior col (j=sy)  → west  [tag 30]
         *         recv right ghost col   (j=ey+1) ← east  [tag 30]
         *
         * Call D: send right interior col (j=ey) → east  [tag 40]
         *         recv left ghost col    (j=sy-1) ← west  [tag 40]
         * ==================================================== */

        /* Call C: send westmost col to west; recv eastmost ghost from east */
        MPI_Sendrecv(
            &u[IDX(sx, sy)],   1, col_type, west, 30,   /* → west  */
            &u[IDX(sx, ey+1)], 1, col_type, east, 30,   /* ← east  */
            cart_comm, MPI_STATUS_IGNORE
        );

        /* Call D: send eastmost col to east; recv westmost ghost from west */
        MPI_Sendrecv(
            &u[IDX(sx, ey)],   1, col_type, east, 40,   /* → east  */
            &u[IDX(sx, sy-1)], 1, col_type, west, 40,   /* ← west  */
            cart_comm, MPI_STATUS_IGNORE
        );

        /* ====================================================
         * ② Jacobi update: u_new ← stencil(u)
         * ==================================================== */
        compute(u, u_new);

        /* ====================================================
         * ③ Compute local residual ‖u_new − u‖∞
         *   then reduce globally with MPI_Allreduce (MPI_MAX)
         *   so all processes know whether to stop.
         * ==================================================== */
        double local_error = 0.0;
        for (int i = sx; i <= ex; i++) {
            for (int j = sy; j <= ey; j++) {
                double diff = fabs(u_new[IDX(i, j)] - u[IDX(i, j)]);
                if (diff > local_error) local_error = diff;
            }
        }

        /*
         * MPI_Allreduce with MPI_MAX:
         *   every process contributes its local_error;
         *   every process receives the global maximum → global_error.
         * This is used as the convergence criterion.
         */
        MPI_Allreduce(&local_error, &global_error, 1,
                      MPI_DOUBLE, MPI_MAX, cart_comm);

        /* ====================================================
         * ④ Swap u ↔ u_new  (pointer swap, no data copy)
         * ==================================================== */
        double *tmp = u;
        u     = u_new;
        u_new = tmp;

        /* Print progress every 100 iterations (rank 0 only) */
        if (rank == 0 && iter % 100 == 0)
            printf("Iteration %d  global_error = %.5e\n", iter, global_error);
    }

    double t_end = MPI_Wtime();

    /* ---- Convergence report ---- */
    if (rank == 0) {
        if (global_error <= tolerance)
            printf("Converged after %d iterations in %f seconds\n",
                   iter, t_end - t_start);
        else
            printf("Did NOT converge: %d iterations, error = %.5e\n",
                   iter, global_error);
        fflush(stdout);
    }
    MPI_Barrier(cart_comm);

    /* ============================================================
     * STEP 7 – Output results
     * ============================================================
     * Compare computed vs exact solution along x=1 (first row).
     * All processes that own row i=1 (i.e., sx==1) print their
     * y-slice in order.
     * ============================================================ */
    /*
     * Print exact vs computed solution along the first row (i=1).
     * The TP expected output shows only the y-values owned by the
     * process at coord [0,0] (cart_rank 0).  We print all y-slices
     * of the i=1 row in west→east order so the full row is shown.
     *
     * All processes with sx==1 participate, printing in y-order.
     */
    for (int r = 0; r < size; r++) {
        if (cart_rank == r && sx == 1) {
            if (sy == 1)   /* first y-slice: print the column header once */
                printf("Exact solution u_exact - Computed solution u\n");
            /* Print only the y-columns this process owns */
            for (int j = sy; j <= ey; j++)
                printf("%12.5e - %12.5e\n",
                       u_exact[IDX(1, j)], u[IDX(1, j)]);
        }
        MPI_Barrier(cart_comm);
    }

    /* ---- Cleanup ---- */
    MPI_Type_free(&col_type);
    free(u);
    free(u_new);
    free(u_exact);
    free(f);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
