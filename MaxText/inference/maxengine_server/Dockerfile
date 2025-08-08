# SPDX-License-Identifier: Apache-2.0

# Ubuntu:24.04
# Use Ubuntu 24.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags?page=1&name=24.04
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAXTEXT_VERSION=main
ENV JETSTREAM_VERSION=main

RUN apt -y update && apt install -y --no-install-recommends \
    ca-certificates \
    git \
    python3.12 \
    python3-pip

RUN update-alternatives --install \
    /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN git clone https://github.com/AI-Hypercomputer/maxtext.git && \
git clone https://github.com/AI-Hypercomputer/JetStream.git

RUN cd maxtext/ && \
git checkout ${MAXTEXT_VERSION} && \
bash setup.sh

RUN cd /JetStream && \
git checkout ${JETSTREAM_VERSION} && \
python3 -m pip install -e .

COPY maxengine_server_entrypoint.sh /usr/bin/

RUN chmod +x /usr/bin/maxengine_server_entrypoint.sh

ENTRYPOINT ["/usr/bin/maxengine_server_entrypoint.sh"]