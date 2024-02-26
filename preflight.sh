#!/bin/bash
echo "Running preflight.sh"
# Command Flags:
# PLATFORM (Required, must be "gke" or "gce")
#
# Example to invoke this script:
# bash preflight.sh PLATFORM=[GCE or GKE]

# Warning:
# For any dependencies, please add them into `setup.sh` or `maxtext_dependencies.Dockerfile`. 
# You should not install any dependencies in this file.

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

# Check if sudo is available
if command -v sudo >/dev/null 2>&1; then
    # sudo is available, use it
    echo "running rto_setup.sh with sudo"

    # apply network settings.
    sudo bash rto_setup.sh
else
    # sudo is not available, run the script without sudo
    echo "running rto_setup.sh without sudo"

    # apply network settings.
    bash rto_setup.sh
fi