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

import asyncio
from collections.abc import Mapping
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)


_HEALTH_ENDPOINT = "/health"
_AUTO_REQUESTS_PER_ACCELERATOR = 4


def _is_harmony_model(llm: Any) -> bool:
  """Whether the in-process vLLM model uses GPT-OSS Harmony messages."""
  model_config = getattr(llm, "model_config", None)
  hf_config = getattr(model_config, "hf_config", None)
  return getattr(hf_config, "model_type", None) == "gpt_oss"


def _get_harmony_stop_token_ids() -> list[int]:
  """Return vLLM's EOS-like assistant action tokens for GPT-OSS."""
  from vllm.entrypoints.openai.parser.harmony_utils import (  # pylint: disable=import-outside-toplevel
      get_stop_tokens_for_assistant_actions,
  )

  return list(get_stop_tokens_for_assistant_actions())


def _parse_harmony_chat_output(token_ids: list[int]) -> tuple[str | None, str | None, bool]:
  """Parse GPT-OSS token IDs with the parser used by vLLM's OpenAI server."""
  from vllm.entrypoints.openai.parser.harmony_utils import (  # pylint: disable=import-outside-toplevel
      parse_chat_output,
  )

  return parse_chat_output(token_ids)


def _render_harmony_chat_prompt(messages: list[dict[str, Any]], reasoning_effort: str | None) -> list[int]:
  """Render GPT-OSS messages exactly as vLLM's OpenAI chat server does."""
  from vllm.entrypoints.openai.parser.harmony_utils import (  # pylint: disable=import-outside-toplevel
      get_system_message,
      parse_chat_inputs_to_harmony_messages,
      render_for_completion,
  )

  if reasoning_effort == "none":
    raise ValueError("Harmony does not support reasoning_effort='none'.")
  harmony_messages = [
      get_system_message(
          reasoning_effort=reasoning_effort,
          browser_description=None,
          python_description=None,
          with_custom_tools=True,
      )
  ]
  harmony_messages.extend(parse_chat_inputs_to_harmony_messages(messages))
  return list(render_for_completion(harmony_messages))


def _top_logprobs_dict(tokenizer: Any, lp_dict: dict | None) -> "dict[str, float] | None":
  if not lp_dict:
    return None
  return {tokenizer.decode([tid]): lp.logprob for tid, lp in lp_dict.items()}


# --- Memory diagnostics (OOM / leak debugging) -------------------------------
# The crash surfaces client-side as APIConnectionError after retries, which can
# mean either a hard abort (HBM/host OOM, PJRT fatal) or a stall. These helpers
# log the three signals that distinguish them: host RSS (host-RAM leak / OOM-kill),
# aggregated JAX HBM (device-memory growth), and vLLM's own KV-cache usage (a KV
# block leak is INVISIBLE to JAX memory_stats because the pool is pre-allocated).


def _host_rss_bytes() -> int:
  """Resident set size of this process in bytes; -1 if unavailable."""
  try:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) * 1024  # kB -> bytes
  except OSError:
    pass
  try:
    import resource  # pylint: disable=import-outside-toplevel

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kB on Linux, bytes on macOS.
    return rss if sys.platform == "darwin" else rss * 1024
  except Exception:  # pylint: disable=broad-except
    return -1


def _hbm_stats() -> dict:
  """Aggregate JAX HBM stats summed across all local devices; {} if unavailable."""
  try:
    import jax  # pylint: disable=import-outside-toplevel

    devs = jax.local_devices()
    in_use = peak = limit = 0
    for d in devs:
      ms = d.memory_stats() or {}
      in_use += ms.get("bytes_in_use", 0)
      peak += ms.get("peak_bytes_in_use", 0)
      limit += ms.get("bytes_limit", 0)
    return {"in_use": in_use, "peak": peak, "limit": limit, "ndev": len(devs)}
  except Exception:  # pylint: disable=broad-except
    return {}


def _kv_cache_usage(llm: Any) -> "float | None":
  """Best-effort vLLM GPU KV-cache usage fraction (0..1); None if unavailable.

  vLLM API for this differs across versions, so the probe is fully guarded and
  only reads metrics (no mutating calls). vLLM also logs KV usage itself at INFO,
  which is the fallback if this returns None.
  """
  try:
    engine = getattr(llm, "llm_engine", None)
    get_metrics = getattr(engine, "get_metrics", None)
    if not callable(get_metrics):
      return None
    for m in get_metrics():
      name = getattr(m, "name", "")
      if "kv_cache_usage" in name or "gpu_cache_usage" in name:
        return float(getattr(m, "value"))
    return None
  except Exception:  # pylint: disable=broad-except
    return None


def _mem_line(llm: Any) -> str:
  """One-line memory snapshot for logging."""
  hbm = _hbm_stats()
  return (
      f"host_rss={_host_rss_bytes()} "
      f"hbm_in_use={hbm.get('in_use', -1)} hbm_peak={hbm.get('peak', -1)} "
      f"hbm_limit={hbm.get('limit', -1)} ndev={hbm.get('ndev', -1)} "
      f"kv_usage={_kv_cache_usage(llm)}"
  )


class _ChatBatchQueue:
  """Coalesces concurrent /v1/chat/completions requests into one llm.generate() call.

  vLLM's LLM.generate() is a single blocking call. Calling it once per HTTP
  request -- the original chat_completions handler -- means uvicorn
  (workers=1, no executor offload) can only ever have one sequence in
  flight, no matter how many client requests arrive concurrently: the
  request being generated blocks the entire event loop, so every other
  request just queues. That serializes what should be batched decode,
  which both tanks throughput (no continuous-batching amortization across
  chips) and, under concurrent load from many eval-harness client threads,
  makes later-queued requests sit long enough to exceed the client's read
  timeout -- and retries only re-queue behind the same bottleneck, making
  it worse.

  This buffers incoming requests for up to max_wait_s (or until max_batch
  accumulate, whichever first) and generates them together in one call,
  the same way /v1/completions already batches a prompt list sent in a
  single request.

  Everything here (submit and flush) runs on the single uvicorn event-loop
  thread; nothing is offloaded to an executor. That is deliberate: it keeps
  exactly one llm.generate() call in flight at a time, so there is no
  question of whether vLLM's synchronous engine tolerates concurrent calls
  from multiple threads. The tradeoff is that a batch's generation still
  blocks new requests from being accepted into the *next* batch until it
  completes -- there is no cross-batch pipelining. That leaves some
  throughput headroom on the table, but is still a large win over strict
  one-at-a-time, since each flush lets vLLM amortize decode across every
  prompt in the batch instead of running them fully serially.
  """

  def __init__(self, llm: Any, max_wait_s: float, max_batch: int, max_pending: int):
    self._llm = llm
    self._max_wait_s = max_wait_s
    self._max_batch = max_batch
    self._max_pending = max_pending
    self._pending: list[tuple[Any, Any, "asyncio.Future"]] = []
    self._timer: "asyncio.TimerHandle | None" = None

  async def submit(self, prompt: Any, sampling_params: Any) -> Any:
    """Queue one rendered prompt and await its vLLM RequestOutput."""
    if len(self._pending) >= self._max_pending:
      raise _ChatQueueFullError(f"Chat request queue is full ({self._max_pending} pending requests).")
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    item = (prompt, sampling_params, future)
    self._pending.append(item)
    if len(self._pending) >= self._max_batch:
      if self._timer is not None:
        self._timer.cancel()
      self._timer = None
      self._flush()
    elif self._timer is None:
      self._timer = loop.call_later(self._max_wait_s, self._flush)
    try:
      return await future
    except asyncio.CancelledError:
      if item in self._pending:
        self._pending.remove(item)
        if not self._pending and self._timer is not None:
          self._timer.cancel()
          self._timer = None
      raise

  def _flush(self) -> None:
    self._timer = None
    batch, self._pending = self._pending, []
    if not batch:
      return
    prompts = [item[0] for item in batch]
    sampling_params = [item[1] for item in batch]
    try:
      outputs = self._llm.generate(prompts, sampling_params)
    except BaseException as exc:  # noqa: B902  pylint: disable=broad-except
      if isinstance(exc, Exception):
        logger.exception("Batched generate() failed for %d requests [%s]: %s", len(batch), type(exc).__name__, exc)
        for _, _, future in batch:
          if not future.done():
            future.set_exception(exc)
        return
      # Hard aborts (XLA/PJRT RESOURCE_EXHAUSTED and similar) are not
      # Exception subclasses. Raising from here would just be logged and
      # swallowed by asyncio's default callback-exception handler when this
      # runs as a call_later callback (the common case), silently leaving
      # the engine in an undefined state while the server keeps accepting
      # requests. Fail loudly and hard instead.
      logger.critical(
          "Fatal error in batched generate() [%s]: %s -- terminating process.",
          type(exc).__name__,
          exc,
          exc_info=exc,
      )
      os._exit(1)  # pylint: disable=protected-access
    for (_, _, future), output in zip(batch, outputs):
      if not future.done():
        future.set_result(output)


class _ChatQueueFullError(RuntimeError):
  """Raised when chat admission control rejects an excess request."""


def _build_app(
    llm: Any,
    chat_batch_wait_s: float = 0.02,
    chat_batch_max_size: int = 64,
    request_concurrency: int = 16,
) -> Any:
  """Return a FastAPI app that wraps an in-process vLLM LLM instance."""
  import fastapi  # pylint: disable=import-outside-toplevel
  from vllm.sampling_params import SamplingParams  # pylint: disable=import-outside-toplevel

  globals()["fastapi"] = fastapi

  app = fastapi.FastAPI()
  app.state.request_count = 0
  app.state.use_harmony = _is_harmony_model(llm)
  app.state.harmony_stop_token_ids = _get_harmony_stop_token_ids() if app.state.use_harmony else []
  configured_max_model_len = getattr(getattr(llm, "model_config", None), "max_model_len", None)
  app.state.max_model_len = (
      configured_max_model_len
      if isinstance(configured_max_model_len, int) and configured_max_model_len > 0
      else None
  )
  app.state.chat_batch_queue = _ChatBatchQueue(
      llm,
      max_wait_s=chat_batch_wait_s,
      max_batch=chat_batch_max_size,
      max_pending=request_concurrency,
  )

  @app.get("/health")
  def health():
    return {"status": "ok"}

  @app.post("/v1/completions")
  async def completions(request: fastapi.Request):
    body = await request.json()

    raw_prompt = body.get("prompt", "")
    # raw_prompt can be: str, list[int] (single token-ID prompt), list[str] (batch of
    # text prompts), or list[list[int]] (batch of token-ID prompts).  Normalise to a
    # list where each element is either a str or a list[int].
    if isinstance(raw_prompt, list) and raw_prompt and isinstance(raw_prompt[0], int):
      prompts = [raw_prompt]  # single prompt sent as token IDs
    else:
      prompts = raw_prompt if isinstance(raw_prompt, list) else [raw_prompt]
    model_name = body.get("model", "")
    max_tokens = int(body["max_tokens"]) if body.get("max_tokens") is not None else 256
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
        tok_tops: list[dict[str, float] | None] = []
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
            tok_tops.append(_top_logprobs_dict(tokenizer, lp_dict))

        gen_lps = gen.logprobs or []
        for pos, tok_id in enumerate(gen.token_ids):
          tok_str = tokenizer.decode([tok_id])
          tok_strings.append(tok_str)
          tok_offsets.append(running_offset)
          running_offset += len(tok_str)
          lp_dict = gen_lps[pos] if pos < len(gen_lps) else None
          lp_val = lp_dict[tok_id].logprob if (lp_dict and tok_id in lp_dict) else None
          tok_lps.append(lp_val)
          tok_tops.append(_top_logprobs_dict(tokenizer, lp_dict))

        logprobs_payload = {
            "tokens": tok_strings,
            "token_logprobs": tok_lps,
            "top_logprobs": tok_tops,
            "text_offset": tok_offsets,
        }

      if echo:
        p = prompts[idx]
        text_out = (tokenizer.decode(p) if isinstance(p, list) else p) + gen.text
      else:
        text_out = gen.text
      choices.append(
          {
              "text": text_out,
              "index": idx,
              "logprobs": logprobs_payload,
              "finish_reason": gen.finish_reason or "stop",
          }
      )

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
    try:
      max_tokens = int(body["max_tokens"]) if body.get("max_tokens") is not None else 256
      temperature = float(body.get("temperature") or 0.0)
    except (TypeError, ValueError) as exc:
      raise fastapi.HTTPException(status_code=400, detail="max_tokens and temperature must be numeric.") from exc
    stop = body.get("stop")
    # Sent by the sampler via extra_body; OpenAI client hoists it to top-level body.
    reasoning_effort = body.get("reasoning_effort")
    include_reasoning = bool(body.get("include_reasoning", True))
    include_raw_output = bool(body.get("include_raw_output", False))

    try:
      if not isinstance(messages, list):
        raise ValueError("messages must be a list.")
      if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

      if app.state.use_harmony:
        prompt_token_ids = _render_harmony_chat_prompt(messages, reasoning_effort)
      else:
        tokenizer = llm.get_tokenizer()
        template_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )

      if isinstance(prompt_token_ids, Mapping):
        prompt_token_ids = prompt_token_ids["input_ids"]
      prompt_token_ids = list(prompt_token_ids)
      if app.state.max_model_len is not None:
        if len(prompt_token_ids) >= app.state.max_model_len:
          raise ValueError(
              f"Input length ({len(prompt_token_ids)}) must be shorter than the model's maximum "
              f"context length ({app.state.max_model_len})."
          )
        # Match vLLM's OpenAI serving behavior: cap the requested output budget
        # to the context space remaining after chat rendering.
        max_tokens = min(max_tokens, app.state.max_model_len - len(prompt_token_ids))
      prompt = {"prompt_token_ids": list(prompt_token_ids)}

      sp_kwargs: dict = {"max_tokens": max_tokens, "temperature": temperature}
      if stop:
        sp_kwargs["stop"] = [stop] if isinstance(stop, str) else list(stop)
      if app.state.use_harmony:
        requested_stop_token_ids = body.get("stop_token_ids") or []
        if isinstance(requested_stop_token_ids, int):
          requested_stop_token_ids = [requested_stop_token_ids]
        if not isinstance(requested_stop_token_ids, list) or not all(
            isinstance(token_id, int) for token_id in requested_stop_token_ids
        ):
          raise ValueError("stop_token_ids must be an integer or a list of integers.")
        sp_kwargs["stop_token_ids"] = list(
            dict.fromkeys([*requested_stop_token_ids, *app.state.harmony_stop_token_ids])
        )
      sampling_params = SamplingParams(**sp_kwargs)
    except (TypeError, ValueError, KeyError) as exc:
      raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

    try:
      # Queued and generated together with other concurrent chat requests --
      # see _ChatBatchQueue -- instead of one llm.generate() call per request.
      output = await app.state.chat_batch_queue.submit(prompt, sampling_params)
    except _ChatQueueFullError as exc:
      raise fastapi.HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except BaseException as exc:  # noqa: B902  pylint: disable=broad-except
      # Catch BaseException (not just Exception): XLA/PJRT RESOURCE_EXHAUSTED and
      # other hard aborts are NOT Exception subclasses. Fatal aborts raised
      # while *this* request's batch is flushed synchronously (the batch-full
      # path in _ChatBatchQueue.submit) surface here and are re-raised
      # unchanged; the far more common case -- a fatal abort during a
      # timer-triggered flush, with no request coroutine on the stack to catch
      # it -- is instead handled inside _ChatBatchQueue._flush itself (a raise
      # from a call_later callback would just be logged and swallowed by
      # asyncio, not crash the process). Ordinary errors convert to HTTP 500.
      logger.exception("chat_completions generate failed [%s]: %s", type(exc).__name__, exc)
      if isinstance(exc, Exception):
        raise fastapi.HTTPException(status_code=500, detail=str(exc)) from exc
      raise

    gen = output.outputs[0]
    prompt_tokens = len(output.prompt_token_ids)
    completion_tokens = len(gen.token_ids)
    raw_output = gen.text
    reasoning = None
    final_content = raw_output
    if app.state.use_harmony:
      try:
        reasoning, final_content, _ = _parse_harmony_chat_output(list(gen.token_ids))
      except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to parse GPT-OSS Harmony output token IDs: %s", exc)
        logger.debug("Unparsed GPT-OSS raw output: %r", raw_output)
        raise fastapi.HTTPException(status_code=500, detail="Failed to parse GPT-OSS Harmony output.") from exc
      logger.debug(
          "Parsed GPT-OSS Harmony output: raw=%r reasoning=%r final=%r",
          raw_output,
          reasoning,
          final_content,
      )

    # Per-request diagnostics. Two failure modes look identical from the client
    # (both surface as APIConnectionError after retries), so log signals that
    # discriminate them server-side:
    #   - prompt/completion token counts -> length-triggered failure (a long
    #     prompt forcing a new XLA shape / KV pressure on the dying request).
    #   - _mem_line -> host RSS (host-RAM leak), HBM growth, vLLM KV usage.
    #   - finish_reason "length" => generation hit max_tokens without stopping
    #     (likely no "Answer:" emitted -> silently scored 0).
    # Logged on a 100-request heartbeat and on every truncated generation.
    app.state.request_count += 1
    if app.state.request_count % 100 == 0 or gen.finish_reason == "length":
      logger.info(
          "req=%d prompt_tok=%d completion_tok=%d finish=%s %s",
          app.state.request_count,
          prompt_tokens,
          completion_tokens,
          gen.finish_reason,
          _mem_line(llm),
      )

    message: dict[str, Any] = {"role": "assistant", "content": final_content}
    if include_reasoning and reasoning is not None:
      # Matches the field emitted by vLLM's OpenAI-compatible ChatMessage.
      message["reasoning"] = reasoning

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": gen.finish_reason or "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if include_raw_output:
      # Non-standard, opt-in diagnostics extension. Raw Harmony output must not
      # leak into message.content because graders should see final content only.
      response["diagnostics"] = {"raw_output": raw_output}
    return response

  return app


class VllmServerManager:
  """Manages an in-process vLLM-TPU LLM with an OpenAI-compatible HTTP layer.

  Args:
    model_path: HF model ID or local path.
    checkpoint_path: MaxText orbax checkpoint path.
    maxtext_model_name: MaxText model name (e.g. "llama3.1-8b").
    host: Hostname the HTTP server binds to (rank-0 only).
    port: Port the HTTP server listens on.
    tensor_parallel_size: Total number of chips.
    expert_parallel_size: Chips allocated to the expert mesh axis (EP).
    max_model_len: Maximum sequence length.
    dtype: Activation dtype string passed to vLLM (e.g. "bfloat16").
    max_num_batched_tokens: Tokens per scheduler step (None = vLLM default).
    max_num_seqs: Max concurrent sequences (None = vLLM default).
    startup_timeout: Seconds to wait for /health to return healthy.
    hbm_memory_utilization: Fraction of HBM reserved for KV cache.
    chat_batch_wait_s: Max seconds /v1/chat/completions buffers a request
      before flushing whatever has accumulated (see _ChatBatchQueue).
    chat_batch_max_size: Max requests /v1/chat/completions batches into one
      llm.generate() call before flushing early.
    concurrency: Maximum accepted in-flight chat requests. None selects an
      automatic value from CPU and accelerator counts and server limits.
    env: Optional environment-variable overrides.
    additional_vllm_kwargs: Extra kwargs merged into the vLLM LLM() constructor.
  """

  def __init__(
      self,
      model_path: str,
      checkpoint_path: str | None = None,
      maxtext_model_name: str | None = None,
      host: str = "localhost",
      port: int = 8000,
      tensor_parallel_size: int = 4,
      expert_parallel_size: int = 1,
      data_parallel_size: int = 1,
      max_model_len: int = 4096,
      dtype: str = "bfloat16",
      max_num_batched_tokens: int | None = None,
      max_num_seqs: int | None = None,
      startup_timeout: int = 600,
      hbm_memory_utilization: float = 0.3,
      chat_batch_wait_s: float = 0.02,
      chat_batch_max_size: int = 64,
      concurrency: int | None = None,
      env: dict[str, str] | None = None,
      additional_vllm_kwargs: dict | None = None,
  ):
    if checkpoint_path and not maxtext_model_name:
      raise ValueError("maxtext_model_name is required when checkpoint_path is set.")
    if tensor_parallel_size % expert_parallel_size != 0:
      raise ValueError(
          f"tensor_parallel_size ({tensor_parallel_size}) is not divisible by "
          f"expert_parallel_size ({expert_parallel_size})."
      )
    self.model_path = model_path
    self.checkpoint_path = checkpoint_path
    self.maxtext_model_name = maxtext_model_name
    self.host = host
    self.port = port
    self.tensor_parallel_size = tensor_parallel_size
    self.expert_parallel_size = expert_parallel_size
    self.data_parallel_size = data_parallel_size
    self.max_model_len = max_model_len
    self.dtype = dtype
    self.max_num_batched_tokens = max_num_batched_tokens
    self.max_num_seqs = max_num_seqs
    self.startup_timeout = startup_timeout
    self.hbm_memory_utilization = hbm_memory_utilization
    self.chat_batch_wait_s = chat_batch_wait_s
    self.chat_batch_max_size = chat_batch_max_size
    if concurrency is not None and concurrency <= 0:
      raise ValueError("concurrency must be positive when specified.")
    self._requested_concurrency = concurrency
    self._concurrency: int | None = None
    self.env = env
    self.additional_vllm_kwargs = additional_vllm_kwargs or {}

    self._llm: Any | None = None
    self._uvicorn_server: Any | None = None
    self._server_thread: threading.Thread | None = None
    self._mem_monitor_thread: threading.Thread | None = None
    self._mem_monitor_stop = threading.Event()

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"

  @property
  def concurrency(self) -> int:
    """Resolved request concurrency; available after the server starts."""
    if self._concurrency is None:
      raise RuntimeError("Request concurrency is not resolved until the server starts.")
    return self._concurrency

  def _resolve_concurrency(self, cpu_count: int, accelerator_count: int) -> int:
    """Resolve the sole request-pressure knob from host and accelerator capacity."""
    if self._requested_concurrency is not None:
      return self._requested_concurrency
    limits = [
        max(1, cpu_count),
        # Four queued sequences per accelerator is a conservative starting
        # point for TPU decode; max_num_seqs/chat batch caps still take priority.
        max(1, accelerator_count * _AUTO_REQUESTS_PER_ACCELERATOR),
        max(1, self.chat_batch_max_size),
    ]
    if self.max_num_seqs is not None:
      limits.append(max(1, self.max_num_seqs))
    return min(limits)

  def start(self) -> None:
    """Initialize the in-process vLLM LLM and start the HTTP server."""

    # Disable V1 multiprocessing to make EngineCore run in-process.
    # JAX initialized exactly once inside LLM() in this process.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
    from vllm import LLM  # pylint: disable=import-outside-toplevel

    if self.env:
      os.environ.update(self.env)

    # total chips = ici_tensor_parallelism * ici_expert_parallelism.
    ici_tp = self.tensor_parallel_size // self.expert_parallel_size
    ici_ep = self.expert_parallel_size

    vllm_kwargs: dict = {
        "model": self.model_path,
        "tensor_parallel_size": ici_tp,
        "data_parallel_size": self.data_parallel_size,
        "max_model_len": self.max_model_len,
        "dtype": self.dtype,
        "gpu_memory_utilization": self.hbm_memory_utilization,
    }
    if self.max_num_batched_tokens is not None:
      vllm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
    if self.max_num_seqs is not None:
      vllm_kwargs["max_num_seqs"] = self.max_num_seqs

    if self.checkpoint_path:
      vllm_kwargs["additional_config"] = {
          "maxtext_config": {
              "model_name": self.maxtext_model_name,
              "load_parameters_path": self.checkpoint_path,
              "log_config": False,
              "ici_tensor_parallelism": ici_tp,
              "ici_expert_parallelism": ici_ep,
          },
          "sharding": {
              "sharding_strategy": {},
          },
      }
      if ici_ep > 1:
        vllm_kwargs["additional_config"]["sharding"]["sharding_strategy"]["expert_parallelism"] = ici_ep
    else:
      vllm_kwargs["load_format"] = "auto"

    if self.additional_vllm_kwargs:
      for _k, _v in self.additional_vllm_kwargs.items():
        if _k == "additional_config" and isinstance(_v, dict) and isinstance(vllm_kwargs.get("additional_config"), dict):
          for _sub_k, _sub_v in _v.items():
            if isinstance(_sub_v, dict) and isinstance(vllm_kwargs["additional_config"].get(_sub_k), dict):
              vllm_kwargs["additional_config"][_sub_k].update(_sub_v)
            else:
              vllm_kwargs["additional_config"][_sub_k] = _sub_v
        else:
          vllm_kwargs[_k] = _v

    logger.info(
        "Initializing in-process vLLM (tp=%d, ep=%d, dp=%d, max_len=%d)",
        ici_tp,
        ici_ep,
        self.data_parallel_size,
        self.max_model_len,
    )
    self._llm = LLM(**vllm_kwargs)

    import jax as _jax  # pylint: disable=import-outside-toplevel

    detected_accelerators = max(1, _jax.device_count())
    active_accelerators = min(
        detected_accelerators,
        max(1, self.tensor_parallel_size * self.data_parallel_size),
    )
    self._concurrency = self._resolve_concurrency(
        cpu_count=os.cpu_count() or 1,
        accelerator_count=active_accelerators,
    )
    logger.info(
        "Request concurrency=%d (%s; cpu=%d active_accelerators=%d detected_accelerators=%d "
        "max_num_seqs=%s chat_batch_max_size=%d)",
        self._concurrency,
        "explicit" if self._requested_concurrency is not None else "auto",
        os.cpu_count() or 1,
        active_accelerators,
        detected_accelerators,
        self.max_num_seqs,
        self.chat_batch_max_size,
    )

    logger.info("Rank %d: vLLM LLM ready.", _jax.process_index())

    # Time-based memory monitor (all ranks) catches OOM/leak trajectory even when
    # requests stall or the process is about to be OOM-killed. Off unless
    # EVAL_MEM_MONITOR_SEC is set to a positive interval.
    self._start_mem_monitor(_jax.process_index())

    if _jax.process_index() == 0:
      import uvicorn  # pylint: disable=import-outside-toplevel

      app = _build_app(
          self._llm,
          chat_batch_wait_s=self.chat_batch_wait_s,
          chat_batch_max_size=self.chat_batch_max_size,
          request_concurrency=self.concurrency,
      )
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

  def _start_mem_monitor(self, rank: int) -> None:
    """Start a daemon thread logging memory every EVAL_MEM_MONITOR_SEC seconds.

    No-op unless the env var is set to a positive value. Logs host RSS, HBM, and
    vLLM KV usage on a fixed interval so a leak/OOM trajectory is captured even
    if no request completes (stall) or the process is OOM-killed mid-request.
    """
    try:
      interval = float(os.environ.get("EVAL_MEM_MONITOR_SEC", "0") or 0)
    except ValueError:
      interval = 0.0
    if interval <= 0:
      return

    def _loop() -> None:
      while not self._mem_monitor_stop.wait(interval):
        logger.info("MEM_MONITOR rank=%d %s", rank, _mem_line(self._llm))

    self._mem_monitor_thread = threading.Thread(target=_loop, daemon=True, name="mem-monitor")
    self._mem_monitor_thread.start()
    logger.info("Memory monitor started (rank=%d, every %.0fs).", rank, interval)

  def _wait_until_healthy(self) -> None:
    """Wait until the HTTP server returns 200 OK on /health."""
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
    raise TimeoutError(f"vLLM HTTP server did not become healthy within {self.startup_timeout}s.")

  def stop(self) -> None:
    """Stop the HTTP server and release the LLM."""
    self._mem_monitor_stop.set()
    if self._mem_monitor_thread is not None:
      self._mem_monitor_thread.join(timeout=5)
      self._mem_monitor_thread = None
    if self._uvicorn_server is not None:
      logger.info("Stopping vLLM HTTP server.")
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
