FROM python:3.12-slim

WORKDIR /app

# Install git, curl, gpg, and GitHub CLI (gh)
RUN apt-get update && apt-get install -y git curl ca-certificates gpg && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Copy entire repository first so requirement files and source code are available
COPY . /app

# Automatically install all MaxText TPU requirements and local package
RUN pip install --no-cache-dir google-cloud-storage google-genai requests google-auth pyink pylint && \
    pip install --no-cache-dir -r src/dependencies/requirements/generated_requirements/tpu-requirements.txt || true && \
    pip install --no-cache-dir --no-deps -e . || true

# Set git global identity for commits
RUN git config --global user.email "overwatch-agent@google.com" && \
    git config --global user.name "Overwatch Agent"

ENV PYTHONPATH="/app/src/maxtext/utils:/app/src:/app"

CMD ["python", "src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/main.py"]
