# Weight Transfer Installation & Deployment Guide

This document provides instructions for installing `tpu-raiden-jax`, baking the dependencies into a Docker image, verifying the installation, and submitting multi-slice jobs via XPK.

---

## 1. Installation Options for `tpu-raiden-jax`

### Option 1: Direct Installation from Google Artifact Registry (Googlers Only)

```bash
# Step 1: Install Artifact Registry Auth Helper
pip install keyrings.google-artifactregistry-auth --extra-index-url https://pypi.org/simple

# Step 2: Install tpu-raiden-jax
pip install tpu-raiden-jax \
  --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/ \
  --extra-index-url https://pypi.org/simple \
  --no-deps
```

### Option 2: Building from Source using Bazel

```bash
# Step 1: Clone tpu-raiden repository
git clone https://github.com/google/tpu-raiden.git
cd tpu-raiden

# Step 2: Build C++ extension and JAX bindings
bash build.sh --jax_only

# Step 3: Install compiled wheel package
pip install --no-deps --force-reinstall dist/tpu_raiden_jax-*.whl
```

---

## 2. Docker Image Setup & Container Baking

### Step 1: Build Base Dependency Image
```bash
bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=stable WORKFLOW=post-training LOCAL_IMAGE_NAME=maxtext-raiden:latest
```

### Step 2: Bake `tpu-raiden-jax` into Image with GCP Credentials
```bash
CID=$(docker create \
  -v ~/.config/gcloud:/root/.config/gcloud \
  -v $(pwd):/deps \
  maxtext-raiden:latest \
  python3 /deps/src/dependencies/scripts/install_post_train_extra_deps.py)

docker start -a $CID
docker commit $CID maxtext-raiden:latest
```

---

## 3. Verification

Verify that `tpu-raiden-jax` and its native C++ extension are functional inside the Docker container:

```bash
docker run --rm maxtext-raiden:latest \
  python3 -c "import tpu_raiden; from tpu_raiden.api.jax import weight_synchronizer; print('Native C++ Extension Available:', weight_synchronizer._weight_synchronizer is not None)"
```

---

## 4. XPK Multi-Slice Deployment

Push the Docker image to Google Container Registry and submit a weight transfer job across TPU v5p slices:

```bash
# Step 1: Tag and Push Docker Image to GCR
docker tag maxtext-raiden:latest gcr.io/<your-gcp-project>/maxtext-raiden:latest
docker push gcr.io/<your-gcp-project>/maxtext-raiden:latest

# Step 2: Submit Workload with XPK
PROJECT_ID=<your-gcp-project> \
CLUSTER_NAME=<your-gke-cluster> \
ZONE=<your-cluster-zone> \
DOCKER_IMAGE=gcr.io/<your-gcp-project>/maxtext-raiden:latest \
USE_BASE_DOCKER_IMAGE=true \
WORKLOAD_NAME=raiden-bench-$(date +%M%S) \
bash src/maxtext/experimental/weight_transfer/run_weight_transfer_xpk.sh
```
