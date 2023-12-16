# Dockerfile for xpk examples in xpk_example_dag.py,
# and is saved at gcr.io/cloud-ml-auto-solutions/xpk_jax_test:latest.
FROM python:3.10
RUN pip install --upgrade pip
RUN pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir --upgrade clu tensorflow tensorflow-datasets
RUN git clone https://github.com/google/flax.git /tmp/flax
RUN pip install --no-cache-dir flax
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
# Package kubectl & gke-gcloud-auth-plugin needed for KubernetesPodOperator
RUN gcloud components install kubectl
RUN gcloud components install gke-gcloud-auth-plugin
ENV TFDS_DATA_DIR=gs://xl-ml-test-us-central2/tfds-data
ENV JAX_PLATFORM_NAME=TPU
