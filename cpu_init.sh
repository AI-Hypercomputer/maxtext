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
        echo -e "Error: You must provide a cpu prefix."
        exit 1
fi

if [[ -z "$NUM_NODES" ]]; then
        export NUM_NODES=1
fi

for ((i = 0; i < $NUM_NODES; i++))
do
    echo "Creating node $CPU_PREFIX-$i..."
    gcloud compute instances create $CPU_PREFIX-$i --boot-disk-size=100 --machine-type=n2-standard-64 --image-project=ubuntu-os-cloud --image=ubuntu-2204-jammy-v20240112 --scopes=cloud-platform,compute-rw,logging-write,monitoring-read,monitoring-write,storage-full
    echo "Node $CPU_PREFIX-$i created!"
done
