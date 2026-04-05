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

"""vLLM-TPU server lifecycle (in-process LLM + thin HTTP wrapper)."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HEALTH_ENDPOINT = "/health"

def _build_app(llm: Any) -> Any:
  """Return a FastAPI app that wraps an in-process vLLM LLM instance."""
  import fastapi  # pylint: disable=import-outside-toplevel
  from vllm.sampling_params import SamplingParams  # pylint: disable=import-outside-toplevel

  app = fastapi.FastAPI()

  @app.get("/health")
  def health():
    return {"status": "ok"}

  @app.post("/v1/completions")
  async def completions(request: fastapi.Request):
    body = await request.json()

    raw_prompt = body.get("prompt", "")
    prompts = raw_prompt if isinstance(raw_prompt, list) else [raw_prompt]
    model_name = body.get("model", "")
    max_tokens = int(body.get("max_tokens") or 256)
    temperature = float(body.get("temperature") or 0.0)
    logprobs_n = body.get("logprobs")  # int | None
    echo = bool(body.get("echo", False))
    stop = body.get("stop")

    sp_kwargs: dict = {"max_tokens": max_tokens, "temperature": temperature}
    if logprobs_n is not None:
      sp_kwargs["logprobs"] = int(logprobs_n)
    if echo and logprobs_n is not None:
      sp_kwargs["prompt_logprobs"] = int(logprobs_n)
    if stop:
      sp_kwargs["stop"] = [stop] if isinstance(stop, str) else list(stop)

    outputs = llm.generate(prompts, SamplingParams(**sp_kwargs))
    tokenizer = llm.get_tokenizer()

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, output in enumerate(outputs):
      gen = output.outputs[0]
      total_prompt_tokens += len(output.prompt_token_ids)
      total_completion_tokens += len(gen.token_ids)

      logprobs_payload = None
      if logprobs_n is not None:
        tok_strings: list[str] = []
        tok_lps: list[float | None] = []
        tok_offsets: list[int] = []
        running_offset = 0

        if echo:
          prompt_lps = output.prompt_logprobs or []
          for pos, tok_id in enumerate(output.prompt_token_ids):
            tok_str = tokenizer.decode([tok_id])
            tok_strings.append(tok_str)
            tok_offsets.append(running_offset)
            running_offset += len(tok_str)
            lp_dict = prompt_lps[pos] if pos < len(prompt_lps) else None
            lp_val = lp_dict[tok_id].logprob if (lp_dict and tok_id in lp_dict) else None
            tok_lps.append(lp_val)

        gen_lps = gen.logprobs or []
        for pos, tok_id in enumerate(gen.token_ids):
          tok_str = tokenizer.decode([tok_id])
          tok_strings.append(tok_str)
          tok_offsets.append(running_offset)
          running_offset += len(tok_str)
          lp_dict = gen_lps[pos] if pos < len(gen_lps) else None
          lp_val = lp_dict[tok_id].logprob if (lp_dict and tok_id in lp_dict) else None
          tok_lps.append(lp_val)

        logprobs_payload = {
            "tokens": tok_strings,
            "token_logprobs": tok_lps,
            "top_logprobs": None,
            "text_offset": tok_offsets,
        }

      text_out = (prompts[idx] + gen.text) if echo else gen.text
      choices.append({
          "text": text_out,
          "index": idx,
          "logprobs": logprobs_payload,
          "finish_reason": gen.finish_reason or "stop",
      })

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }

  @app.post("/v1/chat/completions")
  async def chat_completions(request: fastapi.Request):  # pylint: disable=unused-variable
    """OpenAI-compatible chat completions endpoint.

    Used by evalchemy and lm-eval chat tasks.
    """
    body = await request.json()
    messages = body.get("messages", [])
    model_name = body.get("model", "")
    max_tokens = int(body.get("max_tokens") or 256)
    temperature = float(body.get("temperature") or 0.0)
    stop = body.get("stop")

    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sp_kwargs: dict = {"max_tokens": max_tokens, "temperature": temperature}
    if stop:
      sp_kwargs["stop"] = [stop] if isinstance(stop, str) else list(stop)

    outputs = llm.generate([prompt], SamplingParams(**sp_kwargs))
    gen = outputs[0].outputs[0]
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(gen.token_ids)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": gen.text},
            "finish_reason": gen.finish_reason or "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

  return app


class VllmServerManager:
  """Manages an in-process vLLM-TPU LLM with an OpenAI-compatible HTTP layer.

  Args:
    model_path: HF model ID or local path. 
    checkpoint_path: MaxText orbax checkpoint path.
    maxtext_model_name: MaxText model name (e.g. "llama3.1-8b").
    host: Hostname the HTTP server binds to (rank-0 only).
    port: Port the HTTP server listens on.
    tensor_parallel_size: Tensor parallelism.
    max_model_len: Maximum sequence length.
    dtype: Activation dtype string passed to vLLM (e.g. "bfloat16").
    max_num_batched_tokens: Tokens per scheduler step (None = vLLM default).
    max_num_seqs: Max concurrent sequences (None = vLLM default).
    startup_timeout: Seconds to wait for /health to return healthy.
    env: Optional environment-variable overrides.
  """

  def __init__(
      self,
      model_path: str,
      checkpoint_path: str | None = None,
      maxtext_model_name: str | None = None,
      host: str = "localhost",
      port: int = 8000,
      tensor_parallel_size: int = 4,
      max_model_len: int = 4096,
      dtype: str = "bfloat16",
      max_num_batched_tokens: int | None = None,
      max_num_seqs: int | None = None,
      startup_timeout: int = 600,
      env: dict[str, str] | None = None,
  ):
    if checkpoint_path and not maxtext_model_name:
      raise ValueError("maxtext_model_name is required when checkpoint_path is set.")
    self.model_path = model_path
    self.checkpoint_path = checkpoint_path
    self.maxtext_model_name = maxtext_model_name
    self.host = host
    self.port = port
    self.tensor_parallel_size = tensor_parallel_size
    self.max_model_len = max_model_len
    self.dtype = dtype
    self.max_num_batched_tokens = max_num_batched_tokens
    self.max_num_seqs = max_num_seqs
    self.startup_timeout = startup_timeout
    self.env = env

    self._llm: Any | None = None
    self._uvicorn_server: Any | None = None
    self._server_thread: threading.Thread | None = None

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"

  def start(self) -> None:
    """Initialise the in-process vLLM LLM and start the HTTP server."""
    # pylint: disable=import-outside-toplevel
    from vllm import LLM

    # Disable V1 multiprocessing so EngineCore runs in-process instead.
    # V1 engine architecture is otherwise preserved (tpu-inference plugin works),
    # and JAX/TPU is initialised exactly once inside LLM() in this process.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    if self.env:
      os.environ.update(self.env)

    vllm_kwargs: dict = {
        "model": self.model_path,
        "tensor_parallel_size": self.tensor_parallel_size,
        "max_model_len": self.max_model_len,
        "dtype": self.dtype,
    }
    if self.max_num_batched_tokens is not None:
      vllm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
    if self.max_num_seqs is not None:
      vllm_kwargs["max_num_seqs"] = self.max_num_seqs

    if self.checkpoint_path:
      vllm_kwargs["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
      vllm_kwargs["additional_config"] = {
          "maxtext_config": {
              "model_name": self.maxtext_model_name,
              "load_parameters_path": self.checkpoint_path,
              "log_config": False,
          }
      }
    else:
      vllm_kwargs["load_format"] = "auto"

    logger.info(
        "Initializing in-process vLLM (tp=%d, max_len=%d)...",
        self.tensor_parallel_size,
        self.max_model_len,
    )
    self._llm = LLM(**vllm_kwargs)

    import jax as _jax  # pylint: disable=import-outside-toplevel
    logger.info("Rank %d: vLLM LLM ready.", _jax.process_index())

    if _jax.process_index() == 0:
      import uvicorn  # pylint: disable=import-outside-toplevel

      app = _build_app(self._llm)
      config = uvicorn.Config(
          app,
          host=self.host,
          port=self.port,
          log_level="warning",
          workers=1,
      )
      self._uvicorn_server = uvicorn.Server(config)
      self._server_thread = threading.Thread(
          target=self._uvicorn_server.run,
          daemon=True,
          name="vllm-http-server",
      )
      self._server_thread.start()
      self._wait_until_healthy()

  def _wait_until_healthy(self) -> None:
    deadline = time.time() + self.startup_timeout
    health_url = f"{self.base_url}{_HEALTH_ENDPOINT}"
    while time.time() < deadline:
      try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
          logger.info("vLLM HTTP server is healthy at %s", self.base_url)
          return
      except requests.exceptions.ConnectionError:
        pass
      if self._server_thread is not None and not self._server_thread.is_alive():
        raise RuntimeError("vLLM HTTP server thread died before becoming healthy.")
      time.sleep(2)
    raise TimeoutError(
        f"vLLM HTTP server did not become healthy within {self.startup_timeout}s."
    )

  def stop(self) -> None:
    """Stop the HTTP server and release the LLM."""
    if self._uvicorn_server is not None:
      logger.info("Stopping vLLM HTTP server...")
      self._uvicorn_server.should_exit = True
      if self._server_thread is not None:
        self._server_thread.join(timeout=30)
        if self._server_thread.is_alive():
          logger.warning("vLLM HTTP server thread did not exit within 30 s.")
    self._llm = None
    self._uvicorn_server = None
    self._server_thread = None
    logger.info("VllmServerManager stopped.")

  def __enter__(self) -> "VllmServerManager":
    self.start()
    return self

  def __exit__(self, *_) -> None:
    self.stop()
