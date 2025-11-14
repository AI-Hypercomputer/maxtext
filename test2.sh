HOME=/home/shuningjin_google_com
VENV=$HOME/venv-rl
uv venv --python 3.12 --seed $VENV
source $VENV/bin/activate

# 1. Install base auth and utility packages
# pip install --upgrade pip
pip install aiohttp==3.12.15
pip install keyring keyrings.google-artifactregistry-auth
pip install numba==0.61.2

# 2. Install Tunix (Editable mode)
# git clone https://github.com/google/tunix.git ~/tunix
pip install -e $HOME/tunix

# 3. Install vLLM (TPU backend)
# We must set the target device and point to specific wheels for JAX/Libtpu compatibility
# git clone https://github.com/vllm-project/vllm.git ~/vllm
export VLLM_TARGET_DEVICE="tpu"
pip install -e $HOME/vllm \
    --pre \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html \
    --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html 

# 4. Install TPU-Inference
# git clone https://github.com/vllm-project/tpu-inference.git ~/tpu-inference
pip install -e $HOME/tpu-inference \
    --pre \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html


# Only run this if you want the bleeding edge JAX nightly
# pip uninstall -y jax jaxlib libtpu
# pip install --pre -U jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
# pip install -U --pre libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html