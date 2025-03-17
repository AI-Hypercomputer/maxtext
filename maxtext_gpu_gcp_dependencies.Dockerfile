ARG BASEIMAGE=ghcr.io/nvidia/jax:base
FROM $BASEIMAGE

RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
  && echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get install --yes google-cloud-cli \
  && apt-get install --yes dnsutils \
  && mkdir /gcs

# The Jax stable stack stable release still depends on 12-6 but the nightly
# release depends on 12-8, therefore install both and fail silently if doesn't
# cannot install one of them.
RUN apt-get update && apt-get install -y libcusparse-12-6 || true
RUN apt-get update && apt-get install -y libcusparse-12-8 || true

# Clean up the docker temp to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory in the container
WORKDIR /deps
