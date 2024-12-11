#!/bin/bash
# Loop over methods and environments
for s in 0 42 2024
do
    for m in A2C TD3 SAC
    do
        for e in Pendulum-v1 BipedalWalker-v3
        do
            log_file="./output/${m}_${e}_${s}.log"  # Create a unique log filename for each combination
            echo "Running training with method $m on environment $e. Logging to $log_file"
            python3.7 ./train.py --env "$e" --method "$m" --seed $s > "$log_file" 2>&1
        done
    done
done

