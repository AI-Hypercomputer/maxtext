FROM python:3.10-slim

WORKDIR /app

# Install necessary dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent_sidecar code
COPY src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar /app/agent_sidecar
COPY src/maxtext/utils /app/maxtext/utils

# Set python path so maxtext utils can be imported
ENV PYTHONPATH="/app/maxtext/utils:/app"

# Command to run the sidecar
CMD ["python", "agent_sidecar/main.py"]
