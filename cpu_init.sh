# Description:
# bash cpu_init.sh CPU_PREFIX=$CPU_PREFIX NUM_NODES=$NUM_NODES

# Make sure you have already run `gcloud config set project $PROJECT_NAME`
# and `gcloud config set compute/zone $ZONE`

# Enable "exit immediately if any command fails" option
set -e

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

for ((i = 0; i < $NUM_NODES; i++))
do
    echo "\n\nCreating node $CPU_PREFIX-$i...\n\n"
    gcloud compute instances create $CPU_PREFIX-$i --boot-disk-size=100 --machine-type=n2-standard-64 --image-project=ubuntu-os-cloud --image=ubuntu-2204-jammy-v20240112 --scopes=cloud-platform,compute-rw,logging-write,monitoring-read,monitoring-write,storage-full
    wait $!
    echo "\n\nNode $CPU_PREFIX-$i created!\n\n"
done

wait $!
python3 multihost_runner.py --CPU_PREFIX=prii-cpu-create-test --COMMAND="sudo apt update; sudo apt install python3-pip; bash setup.sh"