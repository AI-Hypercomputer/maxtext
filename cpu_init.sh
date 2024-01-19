# Description:
# bash setup.sh MODE={stable,nightly,head,libtpu-only} LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu} DEVICE={tpu,gpu}


# You need to specificy a MODE, default value stable. 
# You have the option to provide a LIBTPU_GCS_PATH that points to a libtpu.so provided to you by Google. 
# In libtpu-only MODE, the LIBTPU_GCS_PATH is mandatory.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.13


# Enable "exit immediately if any command fails" option
set -e

#NEED TO GCLOUD CONFIG SET BEFORE THIS#########

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ -z "$CPU_PREFIX" ]]; then
        echo -e "\n\nError: You must provide a cpu prefix.\n\n"
        exit 1
fi

if [[ -z "$NUM_NODES" ]]; then
        export NUM_NODES=1
fi

#for ((i = 0; i < $NUM_NODES; i++))
#do
#    echo "\n\nCreating node $CPU_PREFIX-$i...\n\n"
#    gcloud compute instances create $CPU_PREFIX-$i --boot-disk-size=100 --machine-type=n2-standard-64 --image-project=ubuntu-os-cloud --image=ubuntu-2204-jammy-v20240112 --scopes=cloud-platform,compute-rw,logging-write,monitoring-read,monitoring-write,storage-full 
#    echo "\n\nNode $CPU_PREFIX-$i created!\n\n"
#done

python3 multihost_runner.py --TPU_PREFIX=prii-cpu-create-test --COMMAND="sudo apt update; sudo apt install python3-pip; bash setup.sh"