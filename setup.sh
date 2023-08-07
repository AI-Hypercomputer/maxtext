#!/bin/bash

# Set the maximum number of attempts
max_attempts=5
current_attempt=1

# Function to run the flaky.sh script and check the exit status
run_flaky_script() {
    bash single_setup.sh
}

# Loop until the script succeeds or reaches the maximum number of attempts
while [ $current_attempt -le $max_attempts ]; do
    echo "Attempt to run single_setup.sh $current_attempt:"
    run_flaky_script

    # Check the exit status of the flaky.sh script
    if [ $? -eq 0 ]; then
        echo "Success for running single_setup on attempt $current_attempt!"
        exit 0
    else
        echo "Failed to run single_setup on attempt $current_attempt"
        ((current_attempt++))
        sleep 5  # Add a short delay before the next attempt (optional)
    fi
done

echo "All attempts on single_setup failed. Exiting."
exit 1