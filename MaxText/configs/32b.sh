echo "Running 32b.sh"
# Example command to invoke this script
# bash 32b.sh ${RUN_NAME} ${OUTPUT_PATH} ${DATASET_PATH} ${PLATFORM}

RUN_NAME=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}
PLATFORM=${4} # can be 'gke' or 'gce', default is 'gce'

if [ "${PLATFORM}" == "gke" ]
then 
    bash gke_rto_setup.sh
else
    bash rto_setup.sh
fi

# For DNS lookup when running on large number of VMs
echo '142.250.123.95 www.googleapis.com' | tee -a /etc/hosts
echo '142.251.4.128 storage.googleapis.com' | tee -a /etc/hosts

TFLOP_THRESHOLD=0 # set to 0 since we are not actually running as a test.
bash end_to_end/test_tflops_32b_params.sh ${RUN_NAME} ${TFLOP_THRESHOLD} ${OUTPUT_PATH} ${DATASET_PATH}
