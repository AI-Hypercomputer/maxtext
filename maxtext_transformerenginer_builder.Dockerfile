FROM ghcr.io/nvidia/jax:base

WORKDIR /root
ENV NVTE_FRAMEWORK=jax


RUN git clone https://github.com/NVIDIA/TransformerEngine
WORKDIR /root/TransformerEngine
RUN git checkout 0fbc76af3733ae997394eaf82b78ff9c0498fe9
RUN git submodule update --init --recursive
RUN python setup.py bdist_wheel
