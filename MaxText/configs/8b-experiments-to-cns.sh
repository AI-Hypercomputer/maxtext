#!/bin/bash

# Name of the input text file
#input_file="mattdavidow-7-25-run-names.txt"
input_file=$1
CNS_PATH=/cns/pi-d/home/${USER}/int8-metrics/

# Check if the file exists
if [ ! -f "$input_file" ]; then
  echo "Error: File '$input_file' not found."
  exit 1
fi

# Initialize an empty array to store the list of strings
list=()

# Read each line of the file and append it to the list array
while IFS= read -r run_name; do
  list+=("$run_name")
  gcs=mattdavidow-maxtext-br/${run_name}.txt
  /google/data/ro/projects/cloud/bigstore/mpm/fileutil_bs/stable/bin/fileutil_bs cp /bigstore/${gcs} ${CNS_PATH}/
done < "$input_file"

# also move the run_name file
/google/data/ro/projects/cloud/bigstore/mpm/fileutil_bs/stable/bin/fileutil_bs cp ${input_file} ${CNS_PATH}/

# Print the list of strings
echo "List of strings:"
for item in "${list[@]}"; do
  echo "$item"
done