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

"""Shape-controlled XProf profiling runner for vLLM-TPU inference.

The default workload captures the shapes used for the GPT-OSS optimization
study:

  prefill  Q=1024 per request, batch=8, KV=1024.
  decode   Q=1 per request, batch=896, KV=9216. The 896 query tokens pad to
           the TPU's 1024-token execution bucket.

Raw token-ID prompts are submitted directly to the in-process ``LLM``. This
avoids chat-template length uncertainty and HTTP queue batch ambiguity. The
prefill compile pass and traced pass use disjoint prefixes so prefix caching
cannot turn the traced prefill into a cache hit. Decode deliberately uses the
same prompts for its untraced and traced passes, then verifies vLLM's reported
prefix-cache hits; this makes the trace contain Q=1 decode steps at the target
KV length instead of another long prefill.

The requested decode shape must fit the configured KV cache. The runner fails
before allocating the workload when vLLM's resolved block count proves that
the requested active batch cannot fit; it never silently relabels a sequence
of smaller scheduler waves as batch=896.

Usage::

  python -m maxtext.eval.runner.run --runner profile \
      --model_name gpt-oss-20b \
      --hf_path openai/gpt-oss-20b \
      --base_output_directory gs://<bucket>/ \
      --run_name profile_run \
      --max_model_len 8192 \
      --tensor_parallel_size 8 \
      --hf_token $HF_TOKEN

View the traces with XProf (or TensorBoard's profile plugin):

  pip install xprof
  xprof --logdir <base_output_directory>/<run_name>/xprof --port 8791
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from maxtext.eval.runner.common import (
    add_server_args,
    build_server_manager,
    resolve_token,
)

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 1800
_PROFILE_CORPUS = (
    "A research crew studies an unfamiliar planet, records each observation, "
    "checks competing explanations, and reports the evidence precisely. "
)


@dataclass
class PhaseResult:
  """Client-side stats for one traced phase."""

  name: str
  num_requests: int
  wall_s: float
  prompt_tokens: int
  completion_tokens: int
  prompt_tokens_min: int
  prompt_tokens_max: int
  completion_tokens_min: int
  completion_tokens_max: int
  finish_reasons: dict[str, int]
  requested_max_tokens: int
  configured_prompt_words: int
  trace_dir: str
  target_q_len: int = 0
  target_kv_len: int = 0
  padded_query_tokens: int = 0
  cached_tokens_min: int = 0
  cached_tokens_max: int = 0
  cache_resume_tokens_max: int = 0

  @property
  def completion_tok_per_s(self) -> float:
    return self.completion_tokens / self.wall_s if self.wall_s > 0 else 0.0

  @property
  def prompt_tok_per_s(self) -> float:
    return self.prompt_tokens / self.wall_s if self.wall_s > 0 else 0.0

  @property
  def decode_tok_per_s(self) -> float:
    """Backward-compatible alias for completion-token throughput."""
    return self.completion_tok_per_s


def _send_chat_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> dict:
  """POST one chat request and return usage plus its finish reason."""
  payload: dict = {
      "model": model,
      "messages": [{"role": "user", "content": prompt}],
      "max_tokens": max_tokens,
      "temperature": 0.0,
  }
  if reasoning_effort:
    payload["reasoning_effort"] = reasoning_effort
  resp = requests.post(
      f"{base_url}/v1/chat/completions",
      json=payload,
      timeout=_REQUEST_TIMEOUT_S,
  )
  resp.raise_for_status()
  body = resp.json()
  usage = dict(body.get("usage", {}))
  choices = body.get("choices") or []
  usage["finish_reason"] = choices[0].get("finish_reason") if choices else None
  return usage


def _run_phase(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    reasoning_effort: str | None,
) -> tuple[float, list[dict]]:
  """Fire all prompts concurrently; return wall time and per-request usage."""
  start = time.monotonic()
  with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
    usages = list(
        pool.map(
            lambda p: _send_chat_request(base_url, model, p, max_tokens, reasoning_effort),
            prompts,
        )
    )
  wall = time.monotonic() - start
  return wall, usages


def _build_exact_token_prompts(
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    marker_offset: int,
) -> list[list[int]]:
  """Build exact-length, cache-distinct token prompts with natural-ish content.

  The first token is unique for each request. Prefix-cache block hashes are
  chained, so differing at token zero prevents requests in the same batch from
  sharing cached blocks. ``marker_offset`` makes compile and traced prefill
  batches disjoint as well.
  """
  if seq_len <= 0 or batch_size <= 0:
    raise ValueError("seq_len and batch_size must be positive.")

  corpus_ids = list(tokenizer.encode(_PROFILE_CORPUS, add_special_tokens=False))
  if not corpus_ids:
    raise ValueError("Tokenizer produced no tokens for the profiling corpus.")

  vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
  special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
  marker_ids = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
  marker_end = marker_offset + batch_size
  if marker_end > len(marker_ids):
    raise ValueError(
        f"Tokenizer has only {len(marker_ids)} non-special marker IDs, but marker range "
        f"[{marker_offset}, {marker_end}) was requested."
    )

  body_len = seq_len - 1
  repeats = (body_len + len(corpus_ids) - 1) // len(corpus_ids)
  body = (corpus_ids * repeats)[:body_len]
  return [[marker_ids[marker_offset + index], *body] for index in range(batch_size)]


def _run_llm_batch(llm, prompts: list[list[int]], max_tokens: int) -> tuple[float, list[dict]]:
  """Run one exact token-ID batch and return per-request vLLM metadata."""
  from vllm.sampling_params import SamplingParams  # pylint: disable=import-outside-toplevel

  sampling_params = SamplingParams(
      max_tokens=max_tokens,
      temperature=0.0,
      ignore_eos=True,
  )
  token_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
  start = time.monotonic()
  outputs = llm.generate(token_prompts, sampling_params, use_tqdm=False)
  wall = time.monotonic() - start

  usages = []
  for output in outputs:
    generation = output.outputs[0]
    usages.append(
        {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(generation.token_ids),
            "finish_reason": generation.finish_reason,
            "cached_tokens": int(getattr(output, "num_cached_tokens", 0) or 0),
        }
    )
  return wall, usages


def _reset_prefix_cache(llm) -> None:
  """Reset vLLM's prefix cache and require confirmation from the engine."""
  reset_result = llm.reset_prefix_cache()
  if reset_result is False:
    raise RuntimeError("vLLM refused to reset the prefix cache before the traced prefill.")


def _profile_llm_phase(
    *,
    name: str,
    trace_dir: str,
    llm,
    compile_prompts: list[list[int]],
    traced_prompts: list[list[int]],
    max_tokens: int,
    target_q_len: int,
    target_kv_len: int,
    padded_query_tokens: int,
    require_cached_prefix: bool,
) -> PhaseResult:
  """Compile a shape, capture it, and validate the intended cache behavior."""
  import jax  # pylint: disable=import-outside-toplevel

  logger.info(
      "[%s] compile pass: batch=%d q=%d kv=%d max_tokens=%d",
      name,
      len(compile_prompts),
      target_q_len,
      target_kv_len,
      max_tokens,
  )
  _run_llm_batch(llm, compile_prompts, max_tokens)

  if not require_cached_prefix:
    _reset_prefix_cache(llm)

  logger.info("[%s] traced pass -> %s", name, trace_dir)
  jax.profiler.start_trace(trace_dir)
  try:
    wall, usages = _run_llm_batch(llm, traced_prompts, max_tokens)
  finally:
    jax.profiler.stop_trace()

  prompt_token_counts = [int(usage["prompt_tokens"]) for usage in usages]
  completion_token_counts = [int(usage["completion_tokens"]) for usage in usages]
  cached_token_counts = [int(usage["cached_tokens"]) for usage in usages]
  finish_reasons = Counter(str(usage.get("finish_reason") or "unknown") for usage in usages)

  if any(count != target_kv_len for count in prompt_token_counts):
    raise RuntimeError(
        f"{name} prompt shape mismatch: expected {target_kv_len}, "
        f"observed {min(prompt_token_counts)}-{max(prompt_token_counts)} tokens."
    )
  if require_cached_prefix:
    # vLLM intentionally recomputes the final prompt token to obtain logits.
    # Because computed tokens must be scheduler-block aligned, this can resume
    # an entire final block (128 tokens for the GPT-OSS hybrid cache), e.g.
    # floor((1024 - 1) / 128) * 128 = 896 cached tokens. Validate against that
    # exact expected boundary instead of an arbitrary cache-hit percentage.
    scheduler_block_size = _scheduler_block_size(llm)
    if scheduler_block_size is not None:
      minimum_required_hit = (target_kv_len - 1) // scheduler_block_size * scheduler_block_size
    else:
      minimum_required_hit = target_kv_len * 9 // 10
    if not cached_token_counts or min(cached_token_counts) < minimum_required_hit:
      raise RuntimeError(
          f"{name} did not start from the requested warmed KV prefix: expected at least "
          f"{minimum_required_hit} cached tokens per request, observed "
          f"{min(cached_token_counts, default=0)}-{max(cached_token_counts, default=0)}. "
          "The trace contains substantial prefill work and must not be interpreted as Q=1 decode."
      )
  elif any(cached_token_counts):
    raise RuntimeError(
        f"{name} was intended to be a cold prefill, but vLLM reported cache hits of "
        f"{min(cached_token_counts)}-{max(cached_token_counts)} tokens."
    )

  result = PhaseResult(
      name=name,
      num_requests=len(usages),
      wall_s=wall,
      prompt_tokens=sum(prompt_token_counts),
      completion_tokens=sum(completion_token_counts),
      prompt_tokens_min=min(prompt_token_counts, default=0),
      prompt_tokens_max=max(prompt_token_counts, default=0),
      completion_tokens_min=min(completion_token_counts, default=0),
      completion_tokens_max=max(completion_token_counts, default=0),
      finish_reasons=dict(finish_reasons),
      requested_max_tokens=max_tokens,
      configured_prompt_words=target_kv_len,
      trace_dir=trace_dir,
      target_q_len=target_q_len,
      target_kv_len=target_kv_len,
      padded_query_tokens=padded_query_tokens,
      cached_tokens_min=min(cached_token_counts, default=0),
      cached_tokens_max=max(cached_token_counts, default=0),
      cache_resume_tokens_max=max(
          (target_kv_len - cached_tokens for cached_tokens in cached_token_counts),
          default=0,
      ),
  )
  logger.info(
      "[%s] wall=%.2fs batch=%d q=%d kv=%d padded_query_tokens=%d "
      "cached_tokens=%d-%d cache_resume_tokens_max=%d completion_tokens=%d finish_reasons=%s",
      name,
      result.wall_s,
      result.num_requests,
      result.target_q_len,
      result.target_kv_len,
      result.padded_query_tokens,
      result.cached_tokens_min,
      result.cached_tokens_max,
      result.cache_resume_tokens_max,
      result.completion_tokens,
      result.finish_reasons,
  )
  return result


def _resolved_kv_capacity_tokens(llm) -> int | None:
  """Return vLLM's resolved logical KV token capacity when exposed."""
  engine = getattr(llm, "llm_engine", None)
  vllm_config = getattr(engine, "vllm_config", None)
  cache_config = getattr(vllm_config, "cache_config", None)

  # The in-process V1 engine exposes the scheduler's complete KVCacheConfig.
  # Use vLLM's own hybrid-cache concurrency calculation when possible; a
  # num_blocks * block_size estimate is inaccurate for sliding/local layers.
  core_client = getattr(engine, "engine_core", None)
  engine_core = getattr(core_client, "engine_core", None)
  scheduler = getattr(engine_core, "scheduler", None)
  kv_cache_config = getattr(scheduler, "kv_cache_config", None)
  if vllm_config is not None and kv_cache_config is not None:
    try:
      from vllm.v1.core.kv_cache_utils import (  # pylint: disable=import-outside-toplevel
          get_max_concurrency_for_kv_cache_config,
      )

      max_concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
      return int(max_concurrency * vllm_config.model_config.max_model_len)
    except (ImportError, AttributeError, TypeError, ValueError):
      logger.warning("Could not use vLLM's hybrid KV-capacity calculation; using the block-count fallback.")

  num_blocks = getattr(cache_config, "num_gpu_blocks", None)
  block_size = getattr(cache_config, "block_size", None)
  if isinstance(num_blocks, int) and num_blocks > 0 and isinstance(block_size, int) and block_size > 0:
    return num_blocks * block_size
  return None


def _scheduler_block_size(llm) -> int | None:
  """Return the scheduler's token-alignment block size when exposed."""
  engine = getattr(llm, "llm_engine", None)
  core_client = getattr(engine, "engine_core", None)
  engine_core = getattr(core_client, "engine_core", None)
  scheduler = getattr(engine_core, "scheduler", None)
  block_size = getattr(scheduler, "block_size", None)
  if isinstance(block_size, int) and block_size > 0:
    return block_size
  return None


def _profile_phase(
    name: str,
    trace_dir: str,
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    reasoning_effort: str | None,
    configured_prompt_words: int,
) -> PhaseResult:
  """Run one phase: untraced compile pass, then the same workload under a trace."""
  import jax  # pylint: disable=import-outside-toplevel

  logger.info("[%s] compile pass (untraced): %d request(s), max_tokens=%d", name, len(prompts), max_tokens)
  _run_phase(base_url, model, prompts, max_tokens, reasoning_effort)

  logger.info("[%s] traced pass -> %s", name, trace_dir)
  jax.profiler.start_trace(trace_dir)
  try:
    wall, usages = _run_phase(base_url, model, prompts, max_tokens, reasoning_effort)
  finally:
    jax.profiler.stop_trace()

  prompt_token_counts = [int(usage.get("prompt_tokens") or 0) for usage in usages]
  completion_token_counts = [int(usage.get("completion_tokens") or 0) for usage in usages]
  finish_reasons = Counter(str(usage.get("finish_reason") or "unknown") for usage in usages)

  result = PhaseResult(
      name=name,
      num_requests=len(prompts),
      wall_s=wall,
      prompt_tokens=sum(prompt_token_counts),
      completion_tokens=sum(completion_token_counts),
      prompt_tokens_min=min(prompt_token_counts, default=0),
      prompt_tokens_max=max(prompt_token_counts, default=0),
      completion_tokens_min=min(completion_token_counts, default=0),
      completion_tokens_max=max(completion_token_counts, default=0),
      finish_reasons=dict(finish_reasons),
      requested_max_tokens=max_tokens,
      configured_prompt_words=configured_prompt_words,
      trace_dir=trace_dir,
  )
  logger.info(
      "[%s] wall=%.2fs prompt_tok=%d range=%d-%d completion_tok=%d range=%d-%d "
      "prompt_throughput=%.1f tok/s completion_throughput=%.1f tok/s finish_reasons=%s",
      name,
      result.wall_s,
      result.prompt_tokens,
      result.prompt_tokens_min,
      result.prompt_tokens_max,
      result.completion_tokens,
      result.completion_tokens_min,
      result.completion_tokens_max,
      result.prompt_tok_per_s,
      result.completion_tok_per_s,
      result.finish_reasons,
  )
  return result


def _print_summary(results: list[PhaseResult], profile_dir: str) -> None:
  """Print per-phase stats and pointers for reading the traces in XProf."""
  lines = ["", "=" * 78, "XProf profiling summary", "=" * 78]
  for r in results:
    lines.append(
        f"  {r.name:<8} requests={r.num_requests:<4} Q={r.target_q_len:<5} "
        f"KV={r.target_kv_len:<6} padded_Q_tokens={r.padded_query_tokens:<5} wall={r.wall_s:8.2f}s "
        f"prompt_tok={r.prompt_tokens:<7} completion_tok={r.completion_tokens:<7} "
        f"prompt_rate={r.prompt_tok_per_s:8.1f} tok/s "
        f"completion_rate={r.completion_tok_per_s:8.1f} tok/s"
    )
    lines.append(
        f"           actual_prompt_tok={r.prompt_tokens_min}-{r.prompt_tokens_max} "
        f"cached_tok={r.cached_tokens_min}-{r.cached_tokens_max} "
        f"resume_tok_max={r.cache_resume_tokens_max} "
        f"actual_completion_tok={r.completion_tokens_min}-{r.completion_tokens_max} "
        f"finish_reasons={r.finish_reasons}"
    )
  lines += [
      "",
      f"Traces written under: {profile_dir}",
      "View with:  pip install xprof && xprof --logdir " + profile_dir + " --port 8791",
      "  (or: tensorboard --logdir " + profile_dir + " with the profile plugin)",
      "",
      "Where to look for the bottleneck:",
      "  Trace Viewer   Gaps between device steps => host-side (scheduler/HTTP/tokenize) bound;",
      "                 back-to-back steps => device bound.",
      "  HLO Op Stats   Which fused ops dominate: attention (prefill) vs MLP weight loads (decode).",
      "  Memory Viewer  HBM bandwidth utilization; decode near peak bandwidth = expected for LLMs.",
      "  Decode begins with one block-aligned cache-resume step; the following model steps are Q=1.",
      "  The submitted decode batch is exact; verify the device track also shows one scheduler wave.",
      "=" * 78,
  ]
  print("\n".join(lines))


def run_profile(cfg: dict, hf_token: str | None = None) -> dict:
  """Boot the vLLM server and capture per-phase XProf traces.

  Args:
    cfg: Configuration dict. Required keys: model_name, hf_path,
      max_model_len, results_path (used to derive the trace dir). Optional:
      profile_dir, prefill_seq_len, prefill_batch_size, decode_batch_size,
      decode_kv_len, decode_profile_tokens, decode_padded_tokens, and all
      server keys handled by build_server_manager.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Dict with per-phase stats and the profile dir. Empty on non-rank-0 hosts.
  """
  profile_dir = cfg.get("profile_dir") or f"{cfg['results_path'].rstrip('/')}/xprof"
  prefill_seq_len = int(cfg.get("prefill_seq_len") or 1024)
  prefill_batch_size = int(cfg.get("prefill_batch_size") or 8)
  decode_batch_size = int(cfg.get("decode_batch_size") or 896)
  decode_kv_len = int(cfg.get("decode_kv_len") or 9216)
  decode_profile_tokens = int(cfg.get("decode_profile_tokens") or 8)
  decode_padded_tokens = int(cfg.get("decode_padded_tokens") or 1024)
  token = resolve_token(cfg, hf_token)

  positive_values = {
      "prefill_seq_len": prefill_seq_len,
      "prefill_batch_size": prefill_batch_size,
      "decode_batch_size": decode_batch_size,
      "decode_kv_len": decode_kv_len,
      "decode_profile_tokens": decode_profile_tokens,
      "decode_padded_tokens": decode_padded_tokens,
  }
  invalid = [name for name, value in positive_values.items() if value <= 0]
  if invalid:
    raise ValueError(f"Profiling shape values must be positive: {', '.join(invalid)}.")
  if decode_padded_tokens < decode_batch_size:
    raise ValueError(
        f"decode_padded_tokens ({decode_padded_tokens}) must cover one Q token for each of "
        f"the {decode_batch_size} decode requests."
    )
  if decode_padded_tokens & (decode_padded_tokens - 1):
    raise ValueError("decode_padded_tokens must be a power of two TPU token bucket.")
  if int(cfg["max_model_len"]) < decode_kv_len + decode_profile_tokens:
    raise ValueError(
        f"max_model_len ({cfg['max_model_len']}) must be at least decode_kv_len + "
        f"decode_profile_tokens ({decode_kv_len + decode_profile_tokens})."
    )

  required_prefill_tokens = prefill_seq_len * prefill_batch_size
  configured_max_batched_tokens = cfg.get("max_num_batched_tokens")
  if configured_max_batched_tokens is None:
    cfg["max_num_batched_tokens"] = required_prefill_tokens
  elif int(configured_max_batched_tokens) < required_prefill_tokens:
    raise ValueError(
        f"max_num_batched_tokens ({configured_max_batched_tokens}) is smaller than the exact prefill "
        f"batch ({prefill_batch_size} * {prefill_seq_len} = {required_prefill_tokens})."
    )
  configured_max_num_seqs = cfg.get("max_num_seqs")
  if configured_max_num_seqs is None:
    cfg["max_num_seqs"] = decode_batch_size
  elif int(configured_max_num_seqs) < decode_batch_size:
    raise ValueError(
        f"max_num_seqs ({configured_max_num_seqs}) is smaller than decode_batch_size "
        f"({decode_batch_size})."
    )

  # TPU attention otherwise pads every phase to max_num_seqs. Compile exactly
  # the two requested request-count buckets instead.
  os.environ["ATTN_BUCKETIZED_NUM_REQS"] = "1"
  os.environ["ATTN_CUSTOM_NUM_REQS_BUCKETS"] = ",".join(
      str(value) for value in sorted({prefill_batch_size, decode_batch_size})
  )

  results: list[PhaseResult] = []
  is_rank0 = False
  with build_server_manager(cfg, token) as server:
    import jax as _jax
    from jax.experimental import multihost_utils as _multihost_utils

    is_rank0 = _jax.process_index() == 0

    # Validate on every rank before rank 0 enters the workload, so a bad
    # distributed configuration cannot strand other ranks at the final sync.
    llm = server._llm  # pylint: disable=protected-access
    if llm is None:
      raise RuntimeError("The in-process vLLM engine was not initialized.")
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if not bool(getattr(cache_config, "enable_prefix_caching", False)):
      raise RuntimeError(
          "Exact Q=1 decode profiling requires vLLM prefix caching so the target KV state can be "
          "warmed outside the trace. Enable prefix caching for this profiling run."
      )

    resolved_capacity = _resolved_kv_capacity_tokens(llm)
    block_size = int(getattr(cache_config, "block_size", 1) or 1)
    blocks_per_decode_request = (decode_kv_len + decode_profile_tokens + block_size - 1) // block_size
    required_decode_kv_tokens = decode_batch_size * blocks_per_decode_request * block_size
    if resolved_capacity is not None and required_decode_kv_tokens > resolved_capacity:
      raise ValueError(
          f"Requested active decode shape requires {required_decode_kv_tokens:,} KV tokens "
          f"after {block_size}-token block rounding, but vLLM resolved only approximately "
          f"{resolved_capacity:,} logical KV tokens. Reduce decode_batch_size/decode_kv_len or "
          "increase KV-cache capacity; the runner will not profile scheduler waves and label them "
          f"as batch {decode_batch_size}."
      )

    if is_rank0:
      from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

      logger.info("Loading tokenizer from %s for exact token-ID profiling prompts.", cfg["hf_path"])
      tokenizer = AutoTokenizer.from_pretrained(cfg["hf_path"], token=token)
      prefill_compile_prompts = _build_exact_token_prompts(
          tokenizer,
          seq_len=prefill_seq_len,
          batch_size=prefill_batch_size,
          marker_offset=0,
      )
      prefill_trace_prompts = _build_exact_token_prompts(
          tokenizer,
          seq_len=prefill_seq_len,
          batch_size=prefill_batch_size,
          marker_offset=prefill_batch_size,
      )
      decode_prompts = _build_exact_token_prompts(
          tokenizer,
          seq_len=decode_kv_len,
          batch_size=decode_batch_size,
          marker_offset=prefill_batch_size * 2,
      )

      results.append(
          _profile_llm_phase(
              name="prefill",
              trace_dir=f"{profile_dir}/prefill",
              llm=llm,
              compile_prompts=prefill_compile_prompts,
              traced_prompts=prefill_trace_prompts,
              max_tokens=1,
              target_q_len=prefill_seq_len,
              target_kv_len=prefill_seq_len,
              padded_query_tokens=required_prefill_tokens,
              require_cached_prefix=False,
          )
      )
      # The cold-prefill trace leaves its distinct prompts cache-resident.
      # Remove them before reserving the much larger decode prefix batch.
      _reset_prefix_cache(llm)
      results.append(
          _profile_llm_phase(
              name="decode",
              trace_dir=f"{profile_dir}/decode",
              llm=llm,
              compile_prompts=decode_prompts,
              traced_prompts=decode_prompts,
              max_tokens=decode_profile_tokens,
              target_q_len=1,
              target_kv_len=decode_kv_len,
              padded_query_tokens=decode_padded_tokens,
              require_cached_prefix=True,
          )
      )

    # Same multihost pattern as the eval runners: non-rank-0 hosts keep their
    # in-process LLM alive until rank-0 finishes sending requests.
    _multihost_utils.sync_global_devices("profile_complete")

  if not is_rank0:
    return {}

  _print_summary(results, profile_dir)
  return {
      "profile_dir": profile_dir,
      "phases": {
          r.name: {
              "num_requests": r.num_requests,
              "wall_s": round(r.wall_s, 3),
              "prompt_tokens": r.prompt_tokens,
              "completion_tokens": r.completion_tokens,
              "prompt_tokens_min": r.prompt_tokens_min,
              "prompt_tokens_max": r.prompt_tokens_max,
              "completion_tokens_min": r.completion_tokens_min,
              "completion_tokens_max": r.completion_tokens_max,
              "finish_reasons": r.finish_reasons,
              "requested_max_tokens": r.requested_max_tokens,
              "configured_prompt_words": r.configured_prompt_words,
              "target_q_len": r.target_q_len,
              "target_kv_len": r.target_kv_len,
              "padded_query_tokens": r.padded_query_tokens,
              "cached_tokens_min": r.cached_tokens_min,
              "cached_tokens_max": r.cached_tokens_max,
              "cache_resume_tokens_max": r.cache_resume_tokens_max,
              "prompt_tok_per_s": round(r.prompt_tok_per_s, 1),
              "completion_tok_per_s": round(r.completion_tok_per_s, 1),
              "decode_tok_per_s": round(r.decode_tok_per_s, 1),
              "trace_dir": r.trace_dir,
          }
          for r in results
      },
  }


def _build_arg_parser() -> argparse.ArgumentParser:
  """Build argument parser."""
  parser = argparse.ArgumentParser(
      description="MaxText vLLM inference XProf profiling runner.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  add_server_args(parser)
  parser.add_argument(
      "--profile_dir",
      help="Trace output directory (default: {base_output_directory}/{run_name}/eval_results/xprof).",
  )
  parser.add_argument(
      "--prefill_seq_len",
      type=int,
      default=1024,
      help="Exact Q/KV token length per request in the prefill phase.",
  )
  parser.add_argument(
      "--prefill_batch_size",
      type=int,
      default=8,
      help="Exact number of active requests in the prefill phase.",
  )
  parser.add_argument(
      "--decode_batch_size",
      type=int,
      default=896,
      help="Exact requested active sequence count for Q=1 decode.",
  )
  parser.add_argument(
      "--decode_kv_len",
      type=int,
      default=9216,
      help="Exact warmed KV-prefix length for the decode trace.",
  )
  parser.add_argument(
      "--decode_profile_tokens",
      type=int,
      default=8,
      help="Number of forced Q=1 decode steps to capture after warming the KV prefix.",
  )
  parser.add_argument(
      "--decode_padded_tokens",
      type=int,
      default=1024,
      help="TPU token bucket expected to contain decode_batch_size one-token queries.",
  )
  return parser


def main() -> None:
  parser = _build_arg_parser()
  args = parser.parse_args()

  logging.basicConfig(
      level=getattr(logging, args.log_level),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  results_path = f"{args.base_output_directory.rstrip('/')}/{args.run_name}/eval_results"
  cfg = {k: v for k, v in vars(args).items() if k not in ("log_level", "hf_token")}
  cfg["results_path"] = results_path

  output = run_profile(cfg, hf_token=args.hf_token)
  if output:
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
  main()
