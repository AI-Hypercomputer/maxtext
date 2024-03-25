FROM ghcr.io/nvidia/jax:base

WORKDIR /root
COPY ./constraints.txt .
ENV NVTE_FRAMEWORK=jax

RUN git clone https://github.com/NVIDIA/TransformerEngine
WORKDIR /root/TransformerEngine
RUN git checkout 0fbc76af3733ae997394eaf82b78ff9c0498fe9
RUN git submodule update --init --recursive
RUN python setup.by bdist_wheel
