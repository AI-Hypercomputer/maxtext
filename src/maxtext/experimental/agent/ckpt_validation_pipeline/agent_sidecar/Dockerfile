FROM python:3.10-slim

WORKDIR /app

# Install git and necessary dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir google-cloud-storage google-genai requests google-auth

# Copy the agent_sidecar code
COPY src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar /app/agent_sidecar
COPY src/maxtext/utils /app/maxtext/utils

# Set python path so maxtext utils can be imported
ENV PYTHONPATH="/app/maxtext/utils:/app"

# Command to run the Cloud Run Job
CMD ["python", "agent_sidecar/main.py"]
