export PROJECT=cloud-tpu-multipod-dev #<your_project_id>
export ZONE=europe-west4-b #<zone>
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE

export CLUSTER_NAME=v5p-1024-shared #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>

xpk cluster create \
--default-pool-cpu-machine-type=n1-standard-32 \
--cluster ${CLUSTER_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
--reservation=cloudtpu-20240716121201-595617744


xpk workload create \
--cluster ${CLUSTER_NAME} \
--workload hello-world-test \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--command "echo Hello World"