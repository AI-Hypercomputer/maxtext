#!/bin/bash

max_attempts=5
current_attempt=1

# Loop until setup succeeds or reaches the maximum number of attempts
while [ $current_attempt -le $max_attempts ]; do
    echo "Attempt to run setup.sh $current_attempt:"
    bash setup.sh "$@"

    # Check the exit status of run_setup
    if [ $? -eq 0 ]; then
        echo "Success for running setup on attempt $current_attempt!"
        exit 0
    else
        echo "Failed to run setup on attempt $current_attempt."
        ((current_attempt++))
        sleep 5  # Short delay before next attempt
    fi
done

echo "All attempts to run setup failed. Exiting."
exit 1