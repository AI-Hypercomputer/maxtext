FROM ghcr.io/nvidia/jax:base

WORKDIR /root
ENV NVTE_FRAMEWORK=jax


RUN git clone https://github.com/NVIDIA/TransformerEngine
WORKDIR /root/TransformerEngine
RUN git checkout 23caab3fab07b212df65b002eeb05834e6f6c85e
RUN git submodule update --init --recursive
RUN python setup.py bdist_wheel
