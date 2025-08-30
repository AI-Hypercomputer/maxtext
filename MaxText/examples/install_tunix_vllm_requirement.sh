# !/bin/bash
set -e
set -x

python -m ensurepip --default-pip

gcloud auth login
gcloud auth application-default login
pip install keyring keyrings.google-artifactregistry-auth

cd ~/maxtext
bash setup.sh

pip uninstall -y jax jaxlib libtpu

pip install aiohttp==3.12.15
VLLM_TARGET_DEVICE="tpu" pip install --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html \
    --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html \
    vllm==0.10.1rc2.dev129+g800349c2a.tpu

pip install --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    tpu-commons==0.1.0

pip install nest_asyncio
pip install numba==0.61.2

