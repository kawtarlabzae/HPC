#!/bin/bash
cd /mnt/c/msys64/TPHPC/TP1/hpl-2.3/bin/Linux_Intel64

# ========================================
# FORCE SINGLE-CORE EXECUTION
# ========================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "HPL SINGLE-CORE BENCHMARK EXPERIMENTS"
echo "========================================"
echo ""

# ========================================
# SYSTEM INFORMATION
# ========================================
echo "=== SYSTEM INFORMATION ==="
CPU_MODEL=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d':' -f2 | xargs)
echo "CPU Model: $CPU_MODEL"
echo ""

echo "=== THREADING CONFIGURATION ==="
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo ""

echo "=== BLAS LIBRARY ==="
ldd ./xhpl | grep -iE "blas|mkl|lapack" || echo "Using standard BLAS"
echo ""

# ========================================
# BENCHMARK CONFIGURATION
# ========================================
# Matrix sizes
SIZES=(1000 5000 10000 20000)
# Block sizes
BLOCKS=(1 2 4 8 16 32 64 128 256)

# Results file
RESULTS="hpl_single_core_results.csv"
echo "N,NB,Time_s,GFLOPS,Status" > $RESULTS

total_runs=$((${#SIZES[@]} * ${#BLOCKS[@]}))
echo "Starting HPL benchmark experiments..."
echo "Total runs: $total_runs (this will take a while)"
echo "All runs executed on SINGLE CORE only"
echo ""

run_count=0

# ========================================
# MAIN BENCHMARK LOOP
# ========================================
for N in "${SIZES[@]}"; do
    for NB in "${BLOCKS[@]}"; do
        run_count=$((run_count + 1))
        
        echo "========================================" 
        echo "[$run_count/$total_runs] Running N=$N, NB=$NB..."
        echo "========================================" 
        
        # Create HPL.dat for this configuration
        cat > HPL.dat << EOF
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$N           Ns
1            # of NBs
$NB          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
EOF
        
        # Run HPL silently (hide Column= progress messages)
        ./xhpl 2>&1 | grep -v "Column=" > output_single_${N}_${NB}.txt
        
        # Extract the results line (starts with WR)
        RESULT_LINE=$(grep "^WR" output_single_${N}_${NB}.txt)
        
        if [ -n "$RESULT_LINE" ]; then
            # Parse: columns are space-separated
            # Format: WR11C2R4  N  NB  P  Q  Time  Gflops
            TIME=$(echo $RESULT_LINE | awk '{print $6}')
            GFLOPS=$(echo $RESULT_LINE | awk '{print $7}')
        else
            TIME="ERROR"
            GFLOPS="ERROR"
        fi
        
        # Check if test passed
        if grep -q "PASSED" output_single_${N}_${NB}.txt; then
            STATUS="PASSED"
        else
            STATUS="FAILED"
        fi
        
        # Save to results file
        echo "$N,$NB,$TIME,$GFLOPS,$STATUS" >> $RESULTS
        
        # Display results
        echo ""
        echo "  Time: ${TIME} s"
        echo "  Performance: ${GFLOPS} GFLOPS"
        echo "  Status: $STATUS"
        echo ""
        
        # Brief pause between runs
        sleep 1
    done
done

echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS"
echo "Detailed outputs: output_single_N_NB.txt files"
echo ""
