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

"""XProf profiling runner for the MaxText vLLM inference server.

Boots the same in-process vLLM server the eval runners use (including the
/v1/chat/completions micro-batching path), then captures one xplane trace per
workload phase so prefill, single-stream decode, and batched-decode
bottlenecks can be separated in XProf:

  prefill  One request with a long prompt and max_tokens=1. The trace is
           dominated by prompt processing; long gaps or slow attention here
           mean prefill-bound.
  decode   One request with a short prompt generating many tokens. Shows the
           per-step decode latency for a single sequence (the "150 tok/s"
           number): weight-load bound MLP/attention steps, host gaps between
           steps, etc.
  batch    N concurrent chat requests, mirroring eval-harness traffic through
           the server's batching queue. Compare per-step time here against
           the decode phase: if a step with batch=N costs barely more than
           batch=1, the chips are memory-bandwidth-bound and higher
           concurrency is nearly free; if host gaps dominate, the bottleneck
           is scheduling/HTTP, not the TPU.

Each phase runs once untraced first so XLA compilation for that phase's
shapes is excluded from the trace (otherwise compile time swamps the
timeline).

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
import json
import logging
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
# Repeating a common short word approximates one token per word without
# needing the tokenizer client-side; exact prompt length is not important for
# profiling, only that it lands in a "long prompt" compile bucket.
_FILLER_WORD = "many"


@dataclass
class PhaseResult:
  """Client-side stats for one traced phase."""

  name: str
  num_requests: int
  wall_s: float
  prompt_tokens: int
  completion_tokens: int
  trace_dir: str

  @property
  def decode_tok_per_s(self) -> float:
    return self.completion_tokens / self.wall_s if self.wall_s > 0 else 0.0


def _send_chat_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> dict:
  """POST one /v1/chat/completions request; return its usage dict."""
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
  return resp.json().get("usage", {})


def _run_phase(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    reasoning_effort: str | None,
) -> tuple[float, int, int]:
  """Fire all prompts concurrently; return (wall_s, prompt_tokens, completion_tokens)."""
  start = time.monotonic()
  with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
    usages = list(
        pool.map(
            lambda p: _send_chat_request(base_url, model, p, max_tokens, reasoning_effort),
            prompts,
        )
    )
  wall = time.monotonic() - start
  prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usages)
  completion_tokens = sum(u.get("completion_tokens", 0) for u in usages)
  return wall, prompt_tokens, completion_tokens


def _profile_phase(
    name: str,
    trace_dir: str,
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    reasoning_effort: str | None,
) -> PhaseResult:
  """Run one phase: untraced compile pass, then the same workload under a trace."""
  import jax  # pylint: disable=import-outside-toplevel

  logger.info("[%s] compile pass (untraced): %d request(s), max_tokens=%d", name, len(prompts), max_tokens)
  _run_phase(base_url, model, prompts, max_tokens, reasoning_effort)

  logger.info("[%s] traced pass -> %s", name, trace_dir)
  jax.profiler.start_trace(trace_dir)
  try:
    wall, prompt_tokens, completion_tokens = _run_phase(base_url, model, prompts, max_tokens, reasoning_effort)
  finally:
    jax.profiler.stop_trace()

  result = PhaseResult(
      name=name,
      num_requests=len(prompts),
      wall_s=wall,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      trace_dir=trace_dir,
  )
  logger.info(
      "[%s] wall=%.2fs prompt_tok=%d completion_tok=%d decode_throughput=%.1f tok/s",
      name,
      result.wall_s,
      result.prompt_tokens,
      result.completion_tokens,
      result.decode_tok_per_s,
  )
  return result


def _print_summary(results: list[PhaseResult], profile_dir: str) -> None:
  """Print per-phase stats and pointers for reading the traces in XProf."""
  lines = ["", "=" * 78, "XProf profiling summary", "=" * 78]
  for r in results:
    lines.append(
        f"  {r.name:<8} requests={r.num_requests:<4} wall={r.wall_s:8.2f}s "
        f"prompt_tok={r.prompt_tokens:<7} completion_tok={r.completion_tokens:<7} "
        f"decode={r.decode_tok_per_s:8.1f} tok/s"
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
      "  Compare the 'decode' and 'batch' traces: if per-step time barely grows with batch size,",
      "  raise concurrency (--chat_batch_max_size / eval client threads) for more throughput.",
      "=" * 78,
  ]
  print("\n".join(lines))


def run_profile(cfg: dict, hf_token: str | None = None) -> dict:
  """Boot the vLLM server and capture per-phase XProf traces.

  Args:
    cfg: Configuration dict. Required keys: model_name, hf_path,
      max_model_len, results_path (used to derive the trace dir). Optional:
      profile_dir, prefill_prompt_words, decode_tokens, batch_concurrency,
      reasoning_effort, and all server keys handled by build_server_manager.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Dict with per-phase stats and the profile dir. Empty on non-rank-0 hosts.
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.eval.runner.warmup import warmup_server

  model_name = cfg["model_name"]
  profile_dir = cfg.get("profile_dir") or f"{cfg['results_path'].rstrip('/')}/xprof"
  prefill_words = int(cfg.get("prefill_prompt_words") or 1024)
  decode_tokens = int(cfg.get("decode_tokens") or 256)
  batch_concurrency = int(cfg.get("batch_concurrency") or 32)
  reasoning_effort = cfg.get("reasoning_effort")
  token = resolve_token(cfg, hf_token)

  short_prompt = "Write a long story about a spaceship crew exploring a new planet."
  long_prompt = " ".join([_FILLER_WORD] * prefill_words) + "\nSummarize the text above in one word."

  results: list[PhaseResult] = []
  is_rank0 = False
  with build_server_manager(cfg, token) as server:
    import jax as _jax
    from jax.experimental import multihost_utils as _multihost_utils

    is_rank0 = _jax.process_index() == 0

    if is_rank0:
      if not cfg.get("skip_warmup"):
        warmup_server(base_url=server.base_url, model=model_name)

      results.append(
          _profile_phase(
              "prefill",
              f"{profile_dir}/prefill",
              server.base_url,
              model_name,
              prompts=[long_prompt],
              max_tokens=1,
              reasoning_effort=reasoning_effort,
          )
      )
      results.append(
          _profile_phase(
              "decode",
              f"{profile_dir}/decode",
              server.base_url,
              model_name,
              prompts=[short_prompt],
              max_tokens=decode_tokens,
              reasoning_effort=reasoning_effort,
          )
      )
      results.append(
          _profile_phase(
              "batch",
              f"{profile_dir}/batch",
              server.base_url,
              model_name,
              prompts=[short_prompt] * batch_concurrency,
              max_tokens=decode_tokens,
              reasoning_effort=reasoning_effort,
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
      "--prefill_prompt_words",
      type=int,
      default=1024,
      help="Approximate prompt length (in words ~ tokens) for the prefill phase.",
  )
  parser.add_argument(
      "--decode_tokens",
      type=int,
      default=256,
      help="Tokens to generate per request in the decode and batch phases.",
  )
  parser.add_argument(
      "--batch_concurrency",
      type=int,
      default=32,
      help="Concurrent requests in the batch phase (mirror your eval client's thread count).",
  )
  parser.add_argument(
      "--reasoning_effort",
      type=str,
      choices=["low", "medium", "high"],
      default=None,
      help="Reasoning effort passed to /v1/chat/completions (gpt-oss and other reasoning models).",
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
