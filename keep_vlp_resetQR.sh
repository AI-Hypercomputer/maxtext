function resetVLP() {
    local QR_ID=$1 
    TPU_NODE=$QR_ID-0 # The ID of the 0-th node in the queued resource.
    echo "checking $TPU_NODE"
    PID=$(gcloud compute tpus tpu-vm ssh root@${TPU_NODE} --worker=0 --command="sudo lsof -w /dev/vfio/0" --project=${PROJECT} --zone=${ZONE})
    if [[ -z "$PID" ]]; then
        echo "TPU is not in use, resetting the queued resource."
        yes | (gcloud alpha compute tpus queued-resources reset "$QR_ID" --project=${PROJECT} --zone=${ZONE})
    else
        echo "TPU $QR_ID is in use."
    fi
}

function main() {
    gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time 
    # Run the command and store the output in a variable
    output=$(gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time)

    # Initialize an empty array to store the names
    active_names=()

    # Loop through each line of the output
    while IFS= read -r line; do
        # Check if the line contains "ACTIVE" in the STATE column
        if [[ $line == *"ACTIVE"* ]]; then
            # Extract the name and add it to the array
            name=$(echo "$line" | awk '{print $1}')
            active_names+=("$name")
        fi
    done <<< "$output"
    
    # Print the array to verify the names
    printf '%s\n' "${active_names[@]}"

    for QR_ID in "${active_names[@]}"; do
        echo "$QR_ID"
        resetVLP $QR_ID &
    done
    
    echo "Sleeping $POLLING_FREQUENCY seconds for main"
    sleep "$POLLING_FREQUENCY"
}

# PROJECT=tpu-prod-env-multipod
PROJECT=tpu-prod-env-vlp-2nic
export ZONE=us-east5-b
export POLLING_FREQUENCY=300

while true
do
    main
done 