for i in 2 3 4 5 6 7; do
    echo "Running with $i processes..."
    mpirun -n $i MachineLearning
done
