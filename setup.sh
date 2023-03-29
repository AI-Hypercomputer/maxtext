#!/bin/bash
pip3 install -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# install flax
cd
git clone --branch=main https://github.com/google/flax.git
pip3 install -e flax

gsutil cp gs://cloud-tpu-v2-images-artifacts/libtpu/v5litepod-preview/jax/libtpu.so ~/libtpu.so