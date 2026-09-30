ARG BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar@sha256:0beb0db984784efa78b948cfcd56575e1d052a582f9ce06dbadd0525d830e52f
FROM ${BASE_IMAGE}
ARG REQUIREMENTS_FILE
WORKDIR /app
COPY ${REQUIREMENTS_FILE} .
COPY maxtext /app/maxtext
ENV PYTHONPATH=/app/maxtext/src:${PYTHONPATH}
RUN echo "--> Installing user-provided dependencies..." && \
    uv pip install -r ${REQUIREMENTS_FILE} -c /opt/venv/server_constraints.txt && \
    echo "--> Verifying JAX patch integrity..." && \
    python -c "from jax._src.lib import _jax; _jax.colocated_python_cpu_client" && \
    python -c "import orbax.checkpoint, tensorstore, gcsfs; print(orbax.checkpoint.__version__)" && \
    uv cache clean
# ENTRYPOINT/CMD inherited (sidecar gRPC server).
