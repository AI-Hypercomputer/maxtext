set -e
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_SUSPEND=1
export NEEDRESTART_MODE=l

apt-get update && apt-get install -y sudo
(sudo bash || bash) <<'EOF'
apt update && \
apt install -y numactl lsb-release gnupg curl net-tools iproute2 procps lsof git ethtool && \
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update -y && apt -y install gcsfuse
rm -rf /var/lib/apt/lists/*
EOF

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ -z "$DEVICE" ]]; then
        export DEVICE="tpu"
fi

if [[ $JAX_VERSION == NONE ]]; then
  unset JAX_VERSION
fi

if [[ $LIBTPU_GCS_PATH == NONE ]]; then
  unset LIBTPU_GCS_PATH
fi

if [[ -n $JAX_VERSION && ! ($MODE == "stable" || -z $MODE || ($MODE == "nightly" && $DEVICE == "gpu")) ]]; then
     echo -e "\n\nError: You can only specify a JAX_VERSION with stable mode (plus nightly mode on GPU).\n\n"
     exit 1
fi


# Save the script folder path of maxtext
run_name_folder_path=$(pwd)

# Install dependencies from requirements.txt
cd $run_name_folder_path && pip install --upgrade pip
pip3 install --no-cache-dir -U -r requirements_stable_stack_additional.txt


