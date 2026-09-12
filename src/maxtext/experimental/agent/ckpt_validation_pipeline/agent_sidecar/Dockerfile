FROM python:3.12-slim

WORKDIR /app

# Install git, curl, gpg, and GitHub CLI (gh)
RUN apt-get update && apt-get install -y git curl ca-certificates gpg && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y gh google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache for heavy installations
COPY src/dependencies/requirements/generated_requirements/tpu-requirements.txt /tmp/tpu-requirements.txt
COPY pyproject.toml /app/pyproject.toml

# Automatically install all MaxText TPU requirements and local package
RUN pip install --no-cache-dir google-cloud-storage google-cloud-pubsub google-genai requests google-auth pyink pylint && \
    pip install --no-cache-dir -r /tmp/tpu-requirements.txt

# Now copy the full source code (any python code changes will invalidate this layer but skip the pip install)
COPY . /app
RUN pip install --no-cache-dir --no-deps -e .

# Set git global identity for commits and associate local repo with origin/main history
RUN git config --global user.email "overwatch-agent@google.com" && \
    git config --global user.name "Overwatch Agent" 

ENV PYTHONPATH="/app/src/maxtext/utils:/app/src:/app"

CMD ["python", "src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/main.py"]
