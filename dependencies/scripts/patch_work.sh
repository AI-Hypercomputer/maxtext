
#!/bin/bash

# 1. Define the target directory
SITE_PACKAGES="/usr/local/lib/python*/site-packages"
TEMP_DIR="temp_patch_work"

# Ensure the script stops if any command fails
set -e

echo "Navigate to site-packages: $SITE_PACKAGES"
cd "$SITE_PACKAGES"

# 2. Create a temporary directory for cloning
echo "Creating temporary directory..."
# Remove it first if it exists from a previous failed run to ensure a clean slate
if [ -d "$TEMP_DIR" ]; then rm -rf "$TEMP_DIR"; fi
mkdir "$TEMP_DIR"
cd "$TEMP_DIR"

# 3. Clone the repositories
echo "Cloning repositories..."
git clone https://github.com/vllm-project/vllm.git
git clone -b make-moe-work https://github.com/abhinavclemson/tunix.git
git clone https://github.com/vllm-project/tpu-inference.git

# Go back up to site-packages
cd ..

# 4. Copy files
# We use 'cp -rf' to force overwrite existing files recursively.
# We assume the destination folders (./tunix, ./vllm) already exist as installed packages.
# If they don't exist, we create them.

echo "Patching Tunix..."
mkdir -p ./tunix
cp -rf "$TEMP_DIR/tunix/tunix/"* ./tunix/

echo "Patching TPU-Inference..."
# Note: Verify if the installed package name is 'tpu_inference' (underscore) or 'tpu-inference' (dash). 
# Based on your prompt, we are using 'tpu-inference'.
mkdir -p ./tpu_inference
cp -rf "$TEMP_DIR/tpu-inference/tpu_inference/"* ./tpu_inference/

echo "Patching vLLM..."
mkdir -p ./vllm
cp -rf "$TEMP_DIR/vllm/vllm/"* ./vllm/

# 5. Cleanup
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Done! Packages have been patched."