FROM ubuntu:22.04

ARG tpu
USER root

RUN apt-get update -y && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get upgrade -y && apt-get update -y && \
  apt-get install -y --upgrade python3.10 python3-pip python-is-python3 \
  coreutils rsync openssh-client curl vim git
RUN pip install --upgrade pip

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN pip install cryptography gitpython
RUN pip install memray py-spy
RUN pip install jupyter
RUN if [ -z "$tpu" ] ; then pip install "jax[cpu]" ; else pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html ; fi

RUN git clone https://github.com/AI-Hypercomputer/maxtext.git
RUN pip install -r maxtext/requirements.txt
RUN pip install ray[default]==2.37.0
