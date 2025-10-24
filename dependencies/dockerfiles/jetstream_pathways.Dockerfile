# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ubuntu:24.04
# Use Ubuntu 24.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags\?page\=1\&name\=24.04
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && apt install -y --no-install-recommends apt-transport-https ca-certificates gnupg git python3.12 python3-pip curl nano vim

RUN update-alternatives --install     /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y

RUN python3 -m pip install --upgrade pip

ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://localhost:38681
ENV XCLOUD_ENVIRONMENT=GCP

ENV MAXTEXT_VERSION=main
ENV JETSTREAM_VERSION=main

RUN git clone https://github.com/AI-Hypercomputer/JetStream.git && \
git clone https://github.com/AI-Hypercomputer/maxtext.git

RUN cd maxtext/ && \
git checkout ${MAXTEXT_VERSION} && \
bash ./tools/setup/setup.sh

RUN cd /JetStream && \
git checkout ${JETSTREAM_VERSION} && \
python3 -m pip install -e .

RUN python3 -m pip install setuptools fastapi uvicorn

RUN apt -y update && apt-get -y install python3-dev && apt-get -y install build-essential

COPY jetstream_pathways_entrypoint.sh /usr/bin/
RUN chmod +x /usr/bin/jetstream_pathways_entrypoint.sh

ENTRYPOINT ["jetstream_pathways_entrypoint.sh"]
