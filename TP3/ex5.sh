#!/bin/bash

gcc -fopenmp ex5.c -o jacobi_omp -O3

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Threads Time" > results.txt

echo "Running tests..."
for t in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$t
    echo "Running with $t threads..."
    
    # Run the program and extract the CPU time line
    output=$(./jacobi_omp | grep "CPU time")
    
    # Extract just the number (assumes format "... : 1.234E+01 sec.")
    time_val=$(echo $output | awk '{print $4}')
    
    echo "$t $time_val" >> results.txt
done

echo "Done. Results saved to results.txt"
cat results.txt
