FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:20260423-python_3.12-jax_0.10.0

RUN apt-get update && apt-get install -y git

WORKDIR /app

# Copy the current directory (MaxText repo) into the image
COPY . /app/maxtext/

# Install MaxText dependencies
# We assume requirements.txt is in maxtext/src/dependencies/requirements/generated_requirements/tpu-requirements.txt based on repo structure
RUN uv pip install --upgrade pip setuptools wheel
RUN uv pip install -r maxtext/src/dependencies/requirements/generated_requirements/tpu-requirements.txt -c /opt/venv/server_constraints.txt

# Ensure MaxText src is in PYTHONPATH
ENV PYTHONPATH=/app/maxtext/src:$PYTHONPATH

WORKDIR /app/maxtext
