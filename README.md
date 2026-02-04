# Parallel and Distributed Computing

This repository contains the practical labs (TPs) and source code for the **Parallel and Distributed Computing** module. The projects focus on high-performance computing concepts, code optimization, and benchmarking.

## Repository Structure

### `TP1/`
Contains the source code, scripts, and documentation for the first lab session:
* **`ex1.c`**: Examining multipple strides in the code.
* **`ex2.c`**: Implementations of standard and optimized matrix multiplication.
* **`mxm_bloc.c`**: Blocked matrix multiplication for cache optimization analysis.
* **`memory_debug_modif.c`**: Modified code with corrected memory management, verified clean using Valgrind.
* **`run_hpl_single_core_full.sh`**: Bash script to automate HPL benchmarks on a single core.
* **`TP1_HPC_LABZAE_Kawtar.pdf`**: The detailed lab report analyzing the results.

### `TP2/`
Focuses on  Loop Optimizations,  Instruction Scheduling, performance analysis, and understanding scalability laws (Amdahl vs. Gustafson).
* **`TP2_Report.pdf`**: Comprehensive report containing speedup curves, Valgrind/Callgrind analysis, and theoretical scaling comparisons.
* **`ex2_opt.c` / `ex2_orig.c`**: Solution odes for ex2.
* **`bench_ex1.c` / `ex1_generic.c`**: Solution codes for ex1.
* **`ex3.c`**: Solution odes for ex3.
* **`ex4.c`**: Solution odes for ex4.
---
**Author:** Kawtar Labzae
