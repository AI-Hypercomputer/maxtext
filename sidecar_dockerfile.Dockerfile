# Use the JAX image with the custom-built sidecar as the base.
FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:2026_03_31-python_3.12-jax_0.9.2


COPY . /deps
WORKDIR /deps

RUN pip install -r src/dependencies/requirements/generated_requirements/tpu-requirements.txt

# Note: The ENTRYPOINT and CMD are inherited from the base image, so they do not
# need to be redefined here. I.e. the sidecar will be launched automatically.
ENV PYTHONPATH=/deps/src