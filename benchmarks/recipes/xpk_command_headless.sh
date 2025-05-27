# Constants
# PROJECT=cloud-tpu-multipod-dev
# CLUSTER=v4-128-bodaborg-us-central2-b
# ZONE=us-central2-b
# TPU_TYPE=v4-128
# NUM_SLICES=1

# PROJECT=cloud-tpu-multipod-dev
# CLUSTER=v5p-32-bodaborg-us-east5-a
# ZONE=us-east5-a
# TPU_TYPE=v5p-32
# NUM_SLICES=1

CLUSTER=v4-8-maxtext
PROJECT=tpu-prod-env-multipod
ZONE=us-central2-b
TPU_TYPE=v4-8
NUM_SLICES=1

# Images
PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/proxy_server:latest
SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/server:latest
REMOTE_PYTHON_SIDECAR_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/remote_python_sidecar_server:latest

python3 xpk.py workload delete \
  --workload ksadi-headless-rp \
  --project="$PROJECT" \
  --cluster=$CLUSTER \
  --zone=$ZONE

python3 xpk.py workload create-pathways \
  --workload ksadi-headless-rp \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --headless \
  --project="$PROJECT" \
  --cluster=$CLUSTER \
  --zone=$ZONE \
  --proxy-server-image=$PROXY_IMAGE \
  --server-image=$SERVER_IMAGE \
  --remote-python-sidecar-image=$REMOTE_PYTHON_SIDECAR_IMAGE
