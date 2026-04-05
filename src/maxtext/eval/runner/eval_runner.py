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

"""CLI entry point for model evaluation.

MaxTextForCausalLM mode (preferred):
Load weights directly from the MaxText checkpoint, no HuggingFace weight
conversion required. Flag --hf_path supplies the tokenizer (HF model ID
or local tokenizer dir).

  python -m maxtext.eval.runner.eval_runner \
      --config src/maxtext/eval/configs/mlperf.yml \
      --base_config src/maxtext/configs/base.yml  \
      --base_output_directory gs://<gcs_bucket>/ \
      --run_name my_run \
      --checkpoint_path gs://<gcs_bucket>/checkpoint/0/items \
      --model_name llama3.1-8b \
      --hf_path meta-llama/Llama-3.1-8B-Instruct

HuggingFace safetensors mode:
Use --hf_mode and point --hf_path to an existing HF model directory.

  python -m maxtext.eval.runner.eval_runner \
      --config src/maxtext/eval/configs/mlperf.yml \
      --hf_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --model_name tinyllama \
      --hf_mode \
      --base_output_directory /tmp/eval/ \
      --run_name smoke_test \
      --tensor_parallel_size 1
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
  with open(config_path) as f:
    return yaml.safe_load(f) or {}


def _merge_config(base: dict, overrides: dict) -> dict:
  merged = dict(base)
  for k, v in overrides.items():
    if v is not None:
      merged[k] = v
  return merged


def _derive_from_maxtext_config(maxtext_config_path: str) -> dict:
  raw = _load_config(maxtext_config_path)
  prefill_len = raw.get("max_prefill_predict_length")
  target_len = raw.get("max_target_length")

  derived: dict = {}
  if target_len is not None:
    derived["max_model_len"] = int(target_len)
    logger.info(
        "Derived max_model_len=%d from MaxText config max_target_length.",
        derived["max_model_len"],
    )
  if prefill_len is not None and target_len is not None:
    derived["max_tokens_default"] = int(target_len) - int(prefill_len)
    logger.info(
        "Derived max_tokens_default=%d from max_target_length - max_prefill_predict_length.",
        derived["max_tokens_default"],
    )
  for key in ("base_output_directory", "run_name"):
    if raw.get(key):
      derived[key] = raw[key]
  return derived


def _build_results_path(cfg: dict) -> str:
  base_output_directory = cfg.get("base_output_directory", "").rstrip("/")
  run_name = cfg.get("run_name", "")
  if not base_output_directory or not run_name:
    raise ValueError(
        "Cannot build eval results_path."
    )
  return f"{base_output_directory}/{run_name}/eval_results"



def run_eval(cfg: dict, hf_token: str | None = None) -> dict:
  """Execute all the evaluation steps.

  Args:
    cfg: Configuration dict.

  Returns:
    Results dict as written to the JSON output file.
  """
  # pylint: disable=import-outside-toplevel
  from transformers import AutoTokenizer

  from maxtext.eval.datasets.registry import get_dataset
  from maxtext.eval.reporting.json_reporter import write_results
  from maxtext.eval.runner.async_client import generate_batch
  from maxtext.eval.runner.server_manager import VllmServerManager
  from maxtext.eval.runner.warmup import warmup_server
  from maxtext.eval.scoring.registry import get_scorer

  benchmark = cfg["benchmark"]
  model_name = cfg["model_name"]
  hf_path = cfg["hf_path"]
  results_path = cfg["results_path"]
  num_samples = cfg.get("num_samples")
  max_tokens = int(cfg.get("max_tokens", 1024))
  temperature = float(cfg.get("temperature", 0.0))
  concurrency = int(cfg.get("concurrency", 64))
  tensor_parallel_size = int(cfg.get("tensor_parallel_size", 4))
  if "max_model_len" not in cfg:
    raise ValueError(
        "Error: max_model_len is required."
    )
  max_model_len = int(cfg["max_model_len"])
  server_host = cfg.get("server_host", "localhost")
  server_port = int(cfg.get("server_port", 8000))
  max_num_batched_tokens = cfg.get("max_num_batched_tokens")
  if max_num_batched_tokens is not None:
    max_num_batched_tokens = int(max_num_batched_tokens)
  max_num_seqs = cfg.get("max_num_seqs")
  if max_num_seqs is not None:
    max_num_seqs = int(max_num_seqs)
  gcs_results_path = cfg.get("gcs_results_path")
  token = hf_token or os.environ.get("HF_TOKEN") or None
  checkpoint_path = cfg.get("checkpoint_path")
  hf_mode = cfg.get("hf_mode", False)

  # Determine loading mode.
  use_maxtext_adapter = bool(checkpoint_path) and not hf_mode

  # Load tokenizer for prompt formatting.
  logger.info("Loading tokenizer from %s.", hf_path)
  tokenizer = AutoTokenizer.from_pretrained(hf_path, token=token)

  # Prepare dataset.
  logger.info("Loading benchmark dataset: %s", benchmark)
  dataset = get_dataset(benchmark)
  requests = dataset.sample_requests(num_samples=num_samples, tokenizer=tokenizer)
  logger.info("Loaded %d samples.", len(requests))

  prompts = [r.prompt for r in requests]
  references = [r.reference for r in requests]

  # Start vLLM server.
  server_env = {"HF_TOKEN": token} if token else None
  additional_vllm_kwargs = {}
  if cfg.get("enable_expert_parallel"):
    additional_vllm_kwargs["enable_expert_parallel"] = True

  with VllmServerManager(
      model_path=hf_path,
      checkpoint_path=checkpoint_path if use_maxtext_adapter else None,
      maxtext_model_name=model_name if use_maxtext_adapter else None,
      host=server_host,
      port=server_port,
      tensor_parallel_size=tensor_parallel_size,
      max_model_len=max_model_len,
      max_num_batched_tokens=max_num_batched_tokens,
      max_num_seqs=max_num_seqs,
      env=server_env,
      additional_vllm_kwargs=additional_vllm_kwargs or None,
  ) as server:
    base_url = server.base_url

    # Warmup server.
    warmup_server(base_url=base_url, model=model_name, sample_requests=requests)

    # Generate responses.
    logger.info("Generating responses for %d prompts.", len(prompts))
    t0 = time.time()
    results = generate_batch(
        prompts=prompts,
        base_url=base_url,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        concurrency=concurrency,
    )
    elapsed = time.time() - t0
    logger.info("Generation completed in %.1fs (%.1f samples/s).", elapsed, len(prompts) / elapsed)

  # Score.
  responses = [r.text for r in results]
  errors = [r for r in results if r.error]
  if errors:
    logger.warning("%d generation errors (out of %d).", len(errors), len(results))

  scorer = get_scorer(benchmark)
  scores = scorer(responses, references)
  logger.info("Scores: %s", scores)

  # Write results
  generation_stats = {
      "total_samples": len(prompts),
      "num_errors": len(errors),
      "elapsed_s": round(elapsed, 2),
      "samples_per_second": round(len(prompts) / elapsed, 2),
      "total_prompt_tokens": sum(r.prompt_tokens for r in results),
      "total_completion_tokens": sum(r.completion_tokens for r in results),
  }
  output = write_results(
      benchmark=benchmark,
      model_name=model_name,
      scores=scores,
      generation_stats=generation_stats,
      config=cfg,
      results_path=results_path,
  )

  # Optional GCS Upload.
  if gcs_results_path:
    from maxtext.eval.reporting.gcs_reporter import upload_results  # pylint: disable=import-outside-toplevel
    upload_results(output["local_path"], gcs_results_path)

  return output


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="MaxText model evaluation runner.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--config", required=True, help="Path to eval config file.")
  parser.add_argument("--base_config", help="Path to maxtext config.")
  parser.add_argument("--benchmark", help="Benchmark name.")
  parser.add_argument("--checkpoint_path", help="MaxText checkpoint path.")
  parser.add_argument("--model_name", help="MaxText model name.")
  parser.add_argument("--hf_path", help="HF model ID or tokenizer dir.")
  parser.add_argument("--base_output_directory", help="Base output directory.")
  parser.add_argument("--run_name", help="Run name/identifier.")
  parser.add_argument("--gcs_results_path", help="Optional GCS path to upload results.")
  parser.add_argument("--num_samples", type=int, help="Number of eval samples.")
  parser.add_argument("--max_tokens", type=int, help="Max tokens per generation.")
  parser.add_argument("--temperature", type=float, help="Sampling temperature.")
  parser.add_argument("--concurrency", type=int, help="HTTP request concurrency.")
  parser.add_argument("--tensor_parallel_size", type=int, help="vLLM tensor parallelism.")
  parser.add_argument("--max_model_len", type=int, help="vLLM max context length.")
  parser.add_argument("--server_host", help="vLLM server host.")
  parser.add_argument("--server_port", type=int, help="vLLM server port.")
  parser.add_argument("--hf_mode", action="store_true", help="Use HF safetensors mode.")
  parser.add_argument(
      "--enable_expert_parallel",
      action="store_true",
      help=(
          "Enable expert parallelism in vLLM. Required for MoE models such as "
          "qwen3-30b-a3b, qwen3-235b-a22b, deepseek-v3, etc. Without this flag "
          "tpu-inference omits the 'expert' mesh axis and MaxText's MoE sharding "
          "raises KeyError."
      ),
  )
  parser.add_argument("--hf_token", help="HuggingFace token for gated models.")
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Logging level.",
  )
  return parser


def main() -> None:
  parser = _build_arg_parser()
  args = parser.parse_args()

  logging.basicConfig(
      level=getattr(logging, args.log_level),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  base_cfg = _load_config(args.config)

  if args.base_config:
    maxtext_derived = _derive_from_maxtext_config(args.base_config)
    for k, v in maxtext_derived.items():
      if k not in base_cfg:
        base_cfg[k] = v

  cli_overrides = {
      k: v for k, v in vars(args).items()
      if k not in ("config", "base_config", "log_level", "hf_token")
  }
  cfg = _merge_config(base_cfg, cli_overrides)

  if "max_tokens" not in cfg and "max_tokens_default" in cfg:
    cfg["max_tokens"] = cfg["max_tokens_default"]
    logger.info("Using max_tokens=%d derived from MaxText config.", cfg["max_tokens"])

  if "results_path" not in cfg:
    cfg["results_path"] = _build_results_path(cfg)
    logger.info("Results will be written to %s", cfg["results_path"])

  required = ["benchmark", "model_name", "hf_path"]
  missing = [f for f in required if not cfg.get(f)]
  if missing:
    parser.error(f"Missing required config field(s): {missing}")

  run_eval(cfg, hf_token=args.hf_token)


if __name__ == "__main__":
  main()
