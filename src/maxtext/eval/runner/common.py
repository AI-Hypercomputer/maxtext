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

"""Shared helpers for MaxText eval runners."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from maxtext.eval.runner.server_manager import VllmServerManager


def resolve_token(cfg: dict, hf_token: str | None) -> str | None:
  """Return HF token from explicit arg or HF_TOKEN env var."""
  return hf_token or os.environ.get("HF_TOKEN") or None


def build_server_manager(cfg: dict, token: str | None) -> "VllmServerManager":
  """Build a VllmServerManager from a merged config dict.

  Args:
    cfg: Merged configuration dict. Required key: max_model_len. Common
      optional keys: tensor_parallel_size, server_host, server_port,
      max_num_batched_tokens, max_num_seqs, hf_mode, enable_expert_parallel.
    token: HuggingFace token (or None).

  Returns:
    A VllmServerManager instance ready for use as a context manager.
  """
  from maxtext.eval.runner.server_manager import VllmServerManager  # pylint: disable=import-outside-toplevel

  hf_path = cfg["hf_path"]
  model_name = cfg["model_name"]
  checkpoint_path = cfg.get("checkpoint_path")
  hf_mode = cfg.get("hf_mode", False)
  use_maxtext_adapter = bool(checkpoint_path) and not hf_mode

  tensor_parallel_size = int(cfg.get("tensor_parallel_size", 4))
  max_model_len = int(cfg["max_model_len"])
  server_host = cfg.get("server_host", "localhost")
  server_port = int(cfg.get("server_port", 8000))

  max_num_batched_tokens = cfg.get("max_num_batched_tokens")
  if max_num_batched_tokens is not None:
    max_num_batched_tokens = int(max_num_batched_tokens)
  max_num_seqs = cfg.get("max_num_seqs")
  if max_num_seqs is not None:
    max_num_seqs = int(max_num_seqs)

  expert_parallel_size = int(cfg.get("expert_parallel_size") or 1)
  data_parallel_size = int(cfg.get("data_parallel_size") or 1)
  hbm_memory_utilization = float(cfg.get("hbm_memory_utilization") or 0.3)

  server_env = {"HF_TOKEN": token} if token else None

  return VllmServerManager(
      model_path=hf_path,
      checkpoint_path=checkpoint_path if use_maxtext_adapter else None,
      maxtext_model_name=model_name if use_maxtext_adapter else None,
      host=server_host,
      port=server_port,
      tensor_parallel_size=tensor_parallel_size,
      expert_parallel_size=expert_parallel_size,
      data_parallel_size=data_parallel_size,
      max_model_len=max_model_len,
      max_num_batched_tokens=max_num_batched_tokens,
      max_num_seqs=max_num_seqs,
      hbm_memory_utilization=hbm_memory_utilization,
      env=server_env,
  )


def maybe_upload_to_gcs(output: dict, gcs_results_path: str | None) -> None:
  """Upload the results JSON to GCS if gcs_results_path is provided."""
  if gcs_results_path:
    from maxtext.eval.reporting.gcs_reporter import upload_results  # pylint: disable=import-outside-toplevel
    upload_results(output["local_path"], gcs_results_path)


def add_server_args(parser: argparse.ArgumentParser) -> None:
  """Add the server/model CLI args shared by all eval runner parsers."""
  parser.add_argument("--checkpoint_path", help="MaxText orbax checkpoint path (/0/items).")
  parser.add_argument("--model_name", required=True, help="MaxText model name (e.g. llama3.1-8b).")
  parser.add_argument("--hf_path", required=True, help="HF model ID or local tokenizer dir.")
  parser.add_argument(
      "--base_output_directory",
      required=True,
      help="Base output directory (local path or gs://<bucket>/).",
  )
  parser.add_argument("--run_name", required=True, help="Run name/identifier.")
  parser.add_argument("--max_model_len", type=int, required=True, help="vLLM max context length.")
  parser.add_argument(
      "--tensor_parallel_size", type=int, default=4, help="vLLM tensor parallelism."
  )
  parser.add_argument("--server_host", default="localhost", help="vLLM server bind host.")
  parser.add_argument("--server_port", type=int, default=8000, help="vLLM server port.")
  parser.add_argument(
      "--max_num_batched_tokens", type=int, help="vLLM tokens per scheduler step."
  )
  parser.add_argument("--max_num_seqs", type=int, help="vLLM max concurrent sequences.")
  parser.add_argument("--hf_mode", action="store_true", help="HF safetensors mode.")
  parser.add_argument(
      "--expert_parallel_size",
      type=int,
      default=0,
      help=(
          "Chips allocated to the expert mesh axis (EP). "
      ),
  )
  parser.add_argument(
      "--data_parallel_size",
      type=int,
      default=1,
      help="Number of model replicas (DP).",
  )
  parser.add_argument(
      "--hbm_memory_utilization",
      type=float,
      default=0.3,
      help=(
          "Fraction of HBM reserved for KV cache."
      ),
  )
  parser.add_argument("--hf_token", help="HuggingFace token for gated models.")
  parser.add_argument(
      "--gcs_results_path", help="Optional secondary GCS path to upload the results JSON."
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Logging level.",
  )
