#FROM ghcr.io/nvidia/upstream-maxtext:nightly-2024-02-07
FROM ghcr.io/nvidia/jax:maxtext
#FROM ghcr.io/nvidia/upstream-maxtext:nightly-2024-02-07

RUN git clone -b nina_nsys https://github.com/google/maxtext flash_maxtext

RUN pip install aqtp==0.5.0
#RUN cd flash_maxtext && bash setup.sh DEVICE=gpu


COPY train_maxtext.sh /opt/maxtext/train_maxtext.sh
