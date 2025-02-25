#.!/bin/bash

set -e

LOG_FILE_IN_GCS=$1
filename=$(basename $LOG_FILE_IN_GCS)
output_file=$(date "+%Y-%m-%d-%H:%M:%S")_${filename}

CNS_PATH=/cns/pi-d/home/${USER}/tensorboard/multislice/
fileutil mkdir -p ${CNS_PATH}
/google/data/ro/projects/cloud/bigstore/mpm/fileutil_bs/stable/bin/fileutil_bs cp /bigstore/${LOG_FILE_IN_GCS} ${CNS_PATH}/$output_file
echo file to put into xprof: ${CNS_PATH}/$output_file
