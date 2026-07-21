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

"""Capture prefill and decode XProf traces for one vLLM model.
python -m maxtext.eval.runner.run --runner profile \
  --model /home/yujiedeng_google_com/ws/data/gpt-oss-20b-teacher \
  --output_dir ./xprof_traces
  
Example::

  python -m maxtext.eval.runner.run --runner profile \
      --model openai/gpt-oss-20b \
      --output_dir /tmp/xprof \
      --tensor_parallel_size 8

  xprof --logdir /tmp/xprof --port 8791
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_PROMPT_TEXT = "A research team explores a new planet and records its observations. "


def _prompt(tokenizer, length: int, marker: int) -> list[int]:
  """Return an exact-length token prompt."""
  if length <= 0:
    raise ValueError("Prompt length must be positive.")
  body = tokenizer.encode(_PROMPT_TEXT, add_special_tokens=False)
  if not body:
    raise ValueError("Tokenizer produced an empty prompt.")
  repeated = (body * ((length + len(body) - 1) // len(body)))[:length]
  repeated[0] = marker
  return repeated


def _run(llm, prompts: list[list[int]], max_tokens: int) -> tuple[float, list]:
  """Run a token-ID batch and return elapsed time and vLLM outputs."""
  from vllm import SamplingParams  # pylint: disable=import-outside-toplevel

  params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
  inputs = [{"prompt_token_ids": prompt} for prompt in prompts]
  start = time.monotonic()
  outputs = llm.generate(inputs, params, use_tqdm=False)
  return time.monotonic() - start, outputs


def _trace(output_dir: str, llm, prompts: list[list[int]], max_tokens: int) -> dict:
  """Capture one already-compiled workload in XProf."""
  import jax  # pylint: disable=import-outside-toplevel

  jax.profiler.start_trace(output_dir)
  try:
    wall_s, outputs = _run(llm, prompts, max_tokens)
  finally:
    jax.profiler.stop_trace()

  generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
  return {
      "trace_dir": output_dir,
      "batch_size": len(prompts),
      "prompt_length": len(prompts[0]),
      "generated_tokens": generated_tokens,
      "wall_s": round(wall_s, 3),
      "tokens_per_s": round(generated_tokens / wall_s, 1) if wall_s else 0.0,
  }


def run_profile(
    model: str,
    output_dir: str,
    *,
    max_model_len: int = 8320,
    tensor_parallel_size: int = 2,
    batch_size: int = 64,
    prefill_length: int = 8192,
    decode_context_length: int = 8192,
    decode_tokens: int = 128,
) -> dict:
  """Load one model and write separate prefill and decode XProf traces."""
  from vllm import LLM  # pylint: disable=import-outside-toplevel

  if min(batch_size, prefill_length, decode_context_length, decode_tokens) <= 0:
    raise ValueError("Batch size and token counts must be positive.")
  if max_model_len < decode_context_length + decode_tokens:
    raise ValueError("max_model_len must cover decode_context_length + decode_tokens.")

  os.environ.pop("VLLM_TORCH_PROFILER_DIR", None)
  llm = LLM(
      model=model,
      tensor_parallel_size=tensor_parallel_size,
      max_model_len=max_model_len,
      dtype="bfloat16",
      trust_remote_code=True,
      enforce_eager=True,
      enable_prefix_caching=True,
  )
  tokenizer = llm.get_tokenizer()
  markers = [token_id for token_id in range(len(tokenizer)) if token_id not in tokenizer.all_special_ids]
  if len(markers) < batch_size * 3:
    raise ValueError("Tokenizer does not have enough non-special tokens for profiling prompts.")

  def prompts(length: int, offset: int) -> list[list[int]]:
    return [_prompt(tokenizer, length, markers[offset + index]) for index in range(batch_size)]

  # Compile prefill once, then clear the cache so the traced prompts are cold.
  _run(llm, prompts(prefill_length, 0), max_tokens=1)
  llm.reset_prefix_cache()
  prefill = _trace(
      f"{output_dir.rstrip('/')}/prefill",
      llm,
      prompts(prefill_length, batch_size),
      max_tokens=1,
  )

  # Warm the complete decode prompt. Reusing it makes the traced call decode
  # from the cached KV state instead of profiling another prefill.
  llm.reset_prefix_cache()
  decode_prompts = prompts(decode_context_length, batch_size * 2)
  _run(llm, decode_prompts, max_tokens=decode_tokens)
  decode = _trace(
      f"{output_dir.rstrip('/')}/decode",
      llm,
      decode_prompts,
      max_tokens=decode_tokens,
  )

  result = {"model": model, "output_dir": output_dir, "prefill": prefill, "decode": decode}
  print(json.dumps(result, indent=2))
  print(f"View with: xprof --logdir {output_dir} --port 8791")
  return result


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory.")
  parser.add_argument("--output_dir", default="xprof", help="XProf output directory.")
  parser.add_argument("--max_model_len", type=int, default=8320)
  parser.add_argument("--tensor_parallel_size", type=int, default=2)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--prefill_length", type=int, default=8192)
  parser.add_argument("--decode_context_length", type=int, default=8192)
  parser.add_argument("--decode_tokens", type=int, default=128)
  parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  return parser


def main() -> None:
  args = _build_arg_parser().parse_args()
  logging.basicConfig(level=getattr(logging, args.log_level))
  kwargs = vars(args)
  kwargs.pop("log_level")
  run_profile(**kwargs)


if __name__ == "__main__":
  main()
