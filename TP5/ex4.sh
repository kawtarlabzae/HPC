#!/bin/bash

echo "Compiling..."
mpicc -o matrixMul ex4.c -lm

# Matrix size to test (Make it large enough so the CPU actually has to work)
N=4000

echo "Processes,Speedup,Efficiency" > results.csv

echo "Starting tests for Matrix Size: $N"

# Loop through different numbers of processes
for p in 1 2 4 8 10 16 20
do
    echo "Running with $p processes..."

    # Run the program and capture the output
    output=$(mpirun -n $p matrixMul $N)

    # Extract the Speedup and Efficiency numbers using grep and awk
    speedup=$(echo "$output" | grep "Speedup:" | awk '{print $2}')
    efficiency=$(echo "$output" | grep "Efficiency:" | awk '{print $2}')

    # Save to the CSV file
    echo "$p,$speedup,$efficiency"
done
