FROM ghcr.io/nvidia/jax:base

WORKDIR /root
ENV NVTE_FRAMEWORK=jax


RUN git clone https://github.com/NVIDIA/TransformerEngine
WORKDIR /root/TransformerEngine
RUN git checkout 297459bd08e1b791ca7a2872cfa8582220477782
RUN git submodule update --init --recursive
RUN python setup.py bdist_wheel
