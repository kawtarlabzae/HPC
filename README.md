# Parallel and Distributed Computing (HPC)

This repository contains the practical labs (TPs) and source code for the **Parallel and Distributed Computing** module. The projects focus on high-performance computing concepts, code optimization, and benchmarking.

## Repository Structure

### `TP1/`
Contains the source code, scripts, and documentation for the first lab session:
* **`ex1.c` / `ex2.c`**: Implementations of standard and optimized matrix multiplication.
* **`mxm_bloc.c`**: Blocked matrix multiplication for cache optimization analysis.
* **`memory_debug_modif.c`**: Modified code with corrected memory management, verified clean using Valgrind (no memory leaks).
* **`run_hpl_single_core_full.sh`**: Bash script to automate HPL benchmarks on a single core.
* **`TP1_HPC_LABZAE_Kawtar.pdf`**: The detailed lab report analyzing the results.

## Overview
The goal of this work is to analyze CPU performance, implement cache-friendly algorithms (like loop tiling), and compare measured performance against theoretical hardware limits using tools like the HPL benchmark.

---
**Author:** Kawtar Labzae
