#!/bin/bash

gcc -fopenmp ex4.c -o matrix_mult

if [ $? -ne 0 ]; then
    exit 1
fi

printf "%-10s %-10s %-10s %-15s\n" "Threads" "Schedule" "Chunk" "Time"

for threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$threads

    for sched in static dynamic guided; do
        for chunk in 1 10 100; do
            
            export OMP_SCHEDULE="$sched,$chunk"
            
            output=$(./matrix_mult | awk '{print $3}')
            
            printf "%-10s %-10s %-10s %-15s\n" "$threads" "$sched" "$chunk" "$output"
        done
    done
done
