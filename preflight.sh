#!/bin/bash
echo "Running preflight.sh"
# Command Flags:
# PLATFORM (Required, must be "gke" or "gce")
#
# Example to invoke this script:
# bash preflight.sh PLATFORM=gke

# Stop execution if any command exits with error
set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [ -z $PLATFORM ]; then
    echo "Error: PLATFORM flag is missing."
    exit 1
fi

if [[ $PLATFORM == "gce" ]]; then
    # Set up network for running on gce
    echo "Setting up network for GCE"
    sudo bash rto_setup.sh
else
    # Set up network for running on gke
    echo "Setting up network for GKE"
    bash rto_setup.sh
fi