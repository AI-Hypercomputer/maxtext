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
    python -c "from jax._src.lib import _jax; from pathways_sidecar import sidecar_main_lib; from jax.experimental.colocated_python import serialization; assert getattr(_jax, 'colocated_python_cpu_client', None) or serialization._get_cpu_device_map()" && \
    python -c "import orbax.checkpoint, tensorstore, gcsfs; print(orbax.checkpoint.__version__)" && \
    uv cache clean
# The base image's ENTRYPOINT is `sh -c 'while true; do python /app/main.py "$@"; done'`.
# That shell is PID 1 with no TERM handler, so the kubelet's SIGTERM is dropped
# (pid_namespaces(7)) and the container is only SIGKILLed when the pod grace period
# expires, holding the gRPC port the whole time. Run the server as a child of a shell
# that traps and forwards TERM; k8s `restartPolicy: Always` restarts the container if
# the server exits. Docker clears an inherited CMD when ENTRYPOINT is redefined, so the
# base image's flags are restated.
ENTRYPOINT ["/bin/sh", "-c", "python /app/main.py \"$@\" & child=$!; trap 'kill -TERM \"$child\"; wait \"$child\"' TERM INT; wait \"$child\"", "sh"]
CMD ["--port=50051", "--logtostderr", "--stderrthreshold=0", "--v=1"]
