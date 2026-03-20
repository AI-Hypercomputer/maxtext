# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM-TPU server lifecycle management (launch, health-poll, teardown)."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

import requests

logger = logging.getLogger(__name__)

_HEALTH_ENDPOINT = "/health"
_MODELS_ENDPOINT = "/v1/models"


class VllmServerManager:
  """Manages a vLLM tpu-inference server subprocess.

  Starts the server, polls until it is healthy, and tears it down on exit.

  Usage::

      with VllmServerManager(hf_model_path, host="localhost", port=8000) as mgr:
          base_url = mgr.base_url
          # request

  Args:
    hf_model_path: Path to the huggingface model directory (local or GCS).
    host: Hostname or IP the server binds to.
    port: Port the server listens on.
    tensor_parallel_size: Number of chips for tensor parallelism.
    max_model_len: Maximum context length.
    dtype: Torch dtype string passed to vLLM.
    max_num_batched_tokens: Total tokens processed per scheduler step. Default None.
    max_num_seqs: Maximum number of sequences the scheduler holds concurrently. Default None.
    startup_timeout: Seconds to wait for the server to become healthy.
    extra_vllm_args: Additional CLI arguments forwarded verbatim to vLLM.
    env: Optional dict of environment variable overrides for the subprocess.
  """

  def __init__(
      self,
      hf_model_path: str,
      host: str = "localhost",
      port: int = 8000,
      tensor_parallel_size: int = 4,
      max_model_len: int = 4096,
      dtype: str = "bfloat16",
      max_num_batched_tokens: int | None = None,
      max_num_seqs: int | None = None,
      startup_timeout: int = 600,
      extra_vllm_args: list[str] | None = None,
      env: dict[str, str] | None = None,
  ):
    self.hf_model_path = hf_model_path
    self.host = host
    self.port = port
    self.tensor_parallel_size = tensor_parallel_size
    self.max_model_len = max_model_len
    self.dtype = dtype
    self.max_num_batched_tokens = max_num_batched_tokens
    self.max_num_seqs = max_num_seqs
    self.startup_timeout = startup_timeout
    self.extra_vllm_args = extra_vllm_args or []
    self.env = env
    self._proc: subprocess.Popen | None = None

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"

  def start(self) -> None:
    """Launch the vLLM server subprocess."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", self.hf_model_path,
        "--host", self.host,
        "--port", str(self.port),
        "--tensor-parallel-size", str(self.tensor_parallel_size),
        "--max-model-len", str(self.max_model_len),
        "--dtype", self.dtype,
        "--device", "tpu",
        "--disable-log-requests",
    ]
    if self.max_num_batched_tokens is not None:
      cmd += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]
    if self.max_num_seqs is not None:
      cmd += ["--max-num-seqs", str(self.max_num_seqs)]
    cmd += self.extra_vllm_args

    proc_env = os.environ.copy()
    if self.env:
      proc_env.update(self.env)

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    # pylint: disable=consider-using-with
    self._proc = subprocess.Popen(cmd, env=proc_env)
    self._wait_until_healthy()

  def _wait_until_healthy(self) -> None:
    deadline = time.time() + self.startup_timeout
    health_url = f"{self.base_url}{_HEALTH_ENDPOINT}"
    while time.time() < deadline:
      try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
          logger.info("vLLM server is healthy at %s", self.base_url)
          return
      except requests.exceptions.ConnectionError:
        pass
      if self._proc is not None and self._proc.poll() is not None:
        raise RuntimeError(
            f"vLLM server process exited with code {self._proc.returncode} "
            "before becoming healthy."
        )
      time.sleep(5)
    raise TimeoutError(
        f"vLLM server did not become healthy within {self.startup_timeout}s."
    )

  def stop(self) -> None:
    """Gracefully terminate the vLLM server subprocess."""
    if self._proc is None:
      return
    if self._proc.poll() is None:
      logger.info("Sending SIGTERM to vLLM server (pid=%d).", self._proc.pid)
      self._proc.send_signal(signal.SIGTERM)
      try:
        self._proc.wait(timeout=30)
      except subprocess.TimeoutExpired:
        logger.warning("vLLM server did not exit after 30 s, sending SIGKILL.")
        self._proc.kill()
        self._proc.wait()
    logger.info("vLLM server stopped.")
    self._proc = None

  def __enter__(self) -> "VllmServerManager":
    self.start()
    return self

  def __exit__(self, *_) -> None:
    self.stop()
