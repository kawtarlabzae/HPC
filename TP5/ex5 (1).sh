#!/bin/bash

# Compile the C code
echo "Compiling ex5.c..."
mpicc -o pi_approxi ex5.c -lm

# Set iterations to 1 Billion
N=1000000000
echo "Starting Pi Calculation tests with N=$N iterations..."
echo "--------------------------------------------------------"
echo -e "Processes\tSpeedup\t\tEfficiency"
echo "--------------------------------------------------------"

# Loop through different numbers of processes
for p in 1 2 4 8 10 16 20
do
    # Run the program and capture its output
    output=$(mpirun -n $p ./pi_approxi $N)

    # Extract the Speedup and Efficiency numbers using grep and awk
    speedup=$(echo "$output" | grep "Speedup:" | awk '{print $2}')
    efficiency=$(echo "$output" | grep "Efficiency:" | awk '{print $2}')

    # Print the values directly to the terminal screen
    echo -e "$p\t\t$speedup\t$efficiency"
done

echo "--------------------------------------------------------"
echo "Tests complete!"
