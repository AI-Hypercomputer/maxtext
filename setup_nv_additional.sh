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


# Save the script folder path of maxtext
run_name_folder_path=$(pwd)

# Install dependencies from requirements.txt
cd $run_name_folder_path && pip install --upgrade pip
pip3 install --no-cache-dir -U -r requirements_maxtext_nv_image.txt


