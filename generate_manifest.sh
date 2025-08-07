#!/bin/bash

# This script generates a manifest of currently installed Python packages, along with their versions.
# The manifest is named with a timestamp for easy versioning and tracking.

export PREFIX='default'

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

# Set the Manifest file name with the date for versioning
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
MANIFEST_FILE="${PREFIX}_manifest_${TIMESTAMP}.txt"

# Freeze packages installed and their version to the Manifest file, with sorted and commented Manifest
pip freeze | sort > "$MANIFEST_FILE"

# Write commit details to the Manifest file
if [[ -n "$COMMIT_HASH" ]]; then
    echo "# Commit_hash: $COMMIT_HASH" | cat - "$MANIFEST_FILE" > temp && mv temp "$MANIFEST_FILE"
fi

# Add a header comment to the Manifest file
echo "# Python Packages Frozen at: ${TIMESTAMP}" | cat - "$MANIFEST_FILE" > temp && mv temp "$MANIFEST_FILE"
