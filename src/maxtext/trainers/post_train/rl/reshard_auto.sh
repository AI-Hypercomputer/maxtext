#!/bin/bash

# Define your configurations: "trainer_chips:number_of_sampler_chips_per_replica"
configs=(
    "1:1"
    "4:1"
    "4:2"
    "4:4"
    "8:1"
    "8:2"
    "8:4"
    "8:8"
    "16:1"
    "16:2"
    "16:4"
    "16:8"
    "16:16"
    "32:2"
    "32:4"
    "32:8"
    "32:16"
    "32:32"
)

# Global variables
base_output_directory="gs://sanbao-bucket/mlperf_rl/reshard"
store_path="./reshard"
project="cloud-tpu-multipod-dev"
zone="us-central1"
cluster="zxhe-super-xpk-bid"
store_cvs_file="${store_path}/reshard_stats_tp.csv"

mkdir -p ${store_path}

# Function to handle errors and ensure cleanup
handle_error() {
    echo "Error occurred during config ${workload_name}. Cleaning up..."
    python ~/xpk/xpk.py workload delete --workload "${workload_name}" --cluster "${cluster}" --project "${project}" --zone "${zone}"
    # Continue to next iteration rather than exiting the whole script
}

for config in "${configs[@]}"; do
    # Split the config string into variables
    IFS=":" read -r trainer_chips sampler_chips <<< "$config"
    
    # Generate a unique workload name based on config and date
    # timestamp=$(date +"%m-%d")
    timestamp="tp0313"
    workload_name="sanbao-${trainer_chips}-${sampler_chips}-${timestamp}"
    
    echo "----------------------------------------------------------"
    echo "Running Config: Trainer=${trainer_chips}, Sampler=${sampler_chips}"
    echo "Workload Name: ${workload_name}"
    echo "----------------------------------------------------------"

    # Trap errors specifically for this iteration
    trap 'handle_error' ERR

    # 1. Create the YAML
    python ./maxtext/src/maxtext/trainers/post_train/rl/create_yaml.py \
        --metadata_name "${workload_name}" \
        --trainer_chips "${trainer_chips}" \
        --number_of_sampler_chips_per_replica "${sampler_chips}" \
        --sampler_replicas 1 \
        --base_output_directory "${base_output_directory}" \
        --HF_TOKEN "${HF_TOKEN}" \
        --store_directory "${store_path}"

    # 2. Apply Kubernetes YAML
    echo "Applying Kubernetes YAML..."
    kubectl apply -f "${store_path}/${workload_name}.yaml"

    # 3. Wait for workload to run
    echo "Waiting 10 minutes for workload execution..."
    sleep 600

    # 5. Extract Timing Data
    echo "Extracting timing data..."
    python ./maxtext/src/maxtext/trainers/post_train/rl/extract_time.py \
        --pod_name "${workload_name}" \
        --store_cvs_file "${store_cvs_file}"

    echo "Finished: ${workload_name}. Data in ${store_cvs_file}"
    
    # Small buffer before starting the next config
    sleep 120

    # 4. Cleanup Workload
    echo "Deleting workload..."
    python ~/xpk/xpk.py workload delete --workload "${workload_name}" --cluster "${cluster}" --project "${project}" --zone "${zone}"
    
    # Clear trap for next iteration
    trap - ERR
done

echo "All configurations completed."


store_cvs_file="${store_path}/reshard_stats_ep.csv"

mkdir -p ${store_path}

# Function to handle errors and ensure cleanup
handle_error() {
    echo "Error occurred during config ${workload_name}. Cleaning up..."
    python ~/xpk/xpk.py workload delete --workload "${workload_name}" --cluster "${cluster}" --project "${project}" --zone "${zone}"
    # Continue to next iteration rather than exiting the whole script
}

for config in "${configs[@]}"; do
    # Split the config string into variables
    IFS=":" read -r trainer_chips sampler_chips <<< "$config"
    
    # Generate a unique workload name based on config and date
    # timestamp=$(date +"%m-%d")
    timestamp="ep0313"
    workload_name="sanbao-${trainer_chips}-${sampler_chips}-${timestamp}"
    
    echo "----------------------------------------------------------"
    echo "Running Config: Trainer=${trainer_chips}, Sampler=${sampler_chips}"
    echo "Workload Name: ${workload_name}"
    echo "----------------------------------------------------------"

    # Trap errors specifically for this iteration
    trap 'handle_error' ERR

    # 1. Create the YAML
    python ./maxtext/src/maxtext/trainers/post_train/rl/create_yaml.py \
        --metadata_name "${workload_name}" \
        --trainer_chips "${trainer_chips}" \
        --number_of_sampler_chips_per_replica "${sampler_chips}" \
        --sampler_replicas 1 \
        --base_output_directory "${base_output_directory}" \
        --HF_TOKEN "${HF_TOKEN}" \
        --store_directory "${store_path}" \
        --enable_tp False

    # 2. Apply Kubernetes YAML
    echo "Applying Kubernetes YAML..."
    kubectl apply -f "${store_path}/${workload_name}.yaml"

    # 3. Wait for workload to run
    echo "Waiting 10 minutes for workload execution..."
    sleep 600

    # 5. Extract Timing Data
    echo "Extracting timing data..."
    python ./maxtext/src/maxtext/trainers/post_train/rl/extract_time.py \
        --pod_name "${workload_name}" \
        --store_cvs_file "${store_cvs_file}"

    echo "Finished: ${workload_name}. Data in ${store_cvs_file}"
    
    # Small buffer before starting the next config
    sleep 120

    # 4. Cleanup Workload
    echo "Deleting workload..."
    python ~/xpk/xpk.py workload delete --workload "${workload_name}" --cluster "${cluster}" --project "${project}" --zone "${zone}"
    
    # Clear trap for next iteration
    trap - ERR
done

echo "All configurations completed."