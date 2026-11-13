# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verifies architectural and mathematical prefill logit parity against reference goldens or on-the-fly models."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import sys
import time
import traceback

EXIT_DETERMINISTIC_GATE_FAIL = 42

DETERMINISTIC_CODE_ERRORS = (
    TypeError,
    AttributeError,
    NameError,
    SyntaxError,
    ImportError,
    KeyError,
    ValueError,
    ZeroDivisionError,
    NotImplementedError,
)
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import absl
import jax
import jax.numpy as jnp
import jsonlines
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning

try:
  from google.cloud import storage
except ImportError:
  storage = None
from transformers import AutoModelForCausalLM, AutoTokenizer

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import lora_utils, max_logging, maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_TEST_ASSETS_ROOT
from tests.utils import deepseek_v41_frozen_topk
from tests.utils import deepseek_v41_preemption_banking as banking

try:
  import torch

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  HAS_TORCH = False

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log

# User ruling (2026-11-01 19:50 PST, Option A): strict >=8-ULP router and indexer top-k set equality
# is gated in the per-layer teacher-forced v5p harness (reference h_in at every layer). In 40-layer E2E
# FrozenTopK runs where continuous residual activations drift across depth, forced-routing text logit
# gates (KL, cos_sim, top1, top10) remain strict PASS/FAIL gates while E2E router/indexer agreement is
# computed, logged, and persisted in output_summary_json as a diagnostic (never silently dropped).

# --- GOLDEN BATCH FOR ZERO-GENERATION VERIFICATION ---
GOLDEN_BATCH_TEXT = """The James Webb Space Telescope (JWST) is an infrared space observatory launched in 2021
to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old,
distant, or faint for the Hubble Space Telescope.

To calculate the gravitational force at the Sun-Earth L2 Lagrange point, we use Newton's law of universal gravitation:
F = G * (m1 * m2) / r^2

Here is a Python script to model the baseline orbital parameters:

def calculate_orbital_velocity(mass_kg, radius_m):
    G = 6.67430e-11 # gravitational constant
    velocity = (G * mass_kg / radius_m) ** 0.5
    return {"velocity_m_s": round(velocity, 2), "stable": True}

# Payload Configuration
payload_config = {
    "instruments": ["NIRCam", "NIRSpec", "MIRI", "FGS/NIRISS"],
    "operating_temp_k": 50,
    "sunshield_deployed": True
}

Testing multilingual and unicode tokenizer fallbacks.
Chinese (Simplified): 韦伯太空望远镜 (Webb Space Telescope).
Arabic: تلسكوب جيمس ويب الفضائي
Russian: Космический телескоп Джеймса Уэбба
Emoji byte-pair encoding test: 🚀🔭✨🌌🌍
"""

# Reference implementations for models without a transformers implementation: model type -> loader module.
_REFERENCE_MODELS = {
    "deepseek_v41": "tests.utils.deepseek_v41_reference.runner",
}


LENGTH_BUCKET_THRESHOLDS: dict[str, dict[str, object]] = {
    "<= 512": {
        "bucket": "<= 512",
        "max_seq_len": 512,
        "max_kl": 0.025,
        "mean_kl": 2e-4,
        "top1_agreement": 0.99,
        "p95_kl": 5e-3,
        "p99_kl": 0.002,
        "mean_cos_sim": 0.999,
        "batch_rel_tol": 5e-4,
    },
    "513 - 1024": {
        "bucket": "513 - 1024",
        "max_seq_len": 1024,
        "max_kl": 0.05,
        "mean_kl": 5e-4,
        "top1_agreement": 0.985,
        "p95_kl": 5e-3,
        "p99_kl": 0.002,
        "mean_cos_sim": 0.999,
        "batch_rel_tol": 5e-4,
    },
    "1025 - 2048": {
        "bucket": "1025 - 2048",
        "max_seq_len": 2048,
        "max_kl": 0.08,
        "mean_kl": 1e-3,
        "top1_agreement": 0.98,
        "p95_kl": 5e-3,
        "p99_kl": 0.002,
        "mean_cos_sim": 0.999,
        "batch_rel_tol": 5e-4,
    },
    "2049 - 4096": {
        "bucket": "2049 - 4096",
        "max_seq_len": 4096,
        "max_kl": 0.08,
        "mean_kl": 1e-3,
        "top1_agreement": 0.98,
        "p95_kl": 5e-3,
        "p99_kl": 0.005,
        "mean_cos_sim": 0.999,
        "batch_rel_tol": 5e-4,
    },
}


def get_length_bucket_thresholds(seq_len: int) -> dict[str, object]:
  """Returns the approved length-bucketed masked-prefill thresholds for `seq_len`."""
  if seq_len <= 512:
    return dict(LENGTH_BUCKET_THRESHOLDS["<= 512"])
  if seq_len <= 1024:
    return dict(LENGTH_BUCKET_THRESHOLDS["513 - 1024"])
  if seq_len <= 2048:
    return dict(LENGTH_BUCKET_THRESHOLDS["1025 - 2048"])
  return dict(LENGTH_BUCKET_THRESHOLDS["2049 - 4096"])


def _local_checkpoint_dir(path):
  """Returns a local directory with the checkpoint at `path`, copying a gs:// checkpoint to a temp dir first."""
  if not path.startswith("gs://"):
    return path
  import subprocess  # pylint: disable=import-outside-toplevel
  import tempfile  # pylint: disable=import-outside-toplevel

  local = tempfile.mkdtemp(prefix="hf_reference_")
  max_logging.log(f"Copying {path} to {local} for the reference model.")
  subprocess.run(["gcloud", "storage", "cp", "-r", path.rstrip("/") + "/*", local], check=True)
  return local


def _detect_reference_model(hf_model_path):
  """Picks a reference implementation from the checkpoint's config.json `model_type`, if it needs one."""
  if not hf_model_path or hf_model_path.startswith("gs://"):
    return ""
  config_path = os.path.join(hf_model_path, "config.json")
  if not os.path.exists(config_path):
    return ""
  with open(config_path, encoding="utf-8") as f:
    cfg_json = json.load(f)
  model_type = cfg_json["model_type"] if "model_type" in cfg_json else ""
  return model_type if model_type in _REFERENCE_MODELS else ""


class _ReferenceModelOutput:

  def __init__(self, logits):
    self.logits = logits


class _ReferenceModel:
  """Adapts a reference model to the `hf_model(**inputs).logits` interface."""

  def __init__(self, model):
    self.model = model

  def __call__(self, input_ids, **unused_kwargs):
    return _ReferenceModelOutput(self.model(input_ids, full_logits=True).float())


def _load_reference_model(reference_model, hf_model_path, torch_dtype, config, tokenizer):
  """Loads the registered reference model."""
  import importlib  # pylint: disable=import-outside-toplevel

  if reference_model not in _REFERENCE_MODELS:
    raise ValueError(f"Unknown reference model {reference_model}; choose from {sorted(_REFERENCE_MODELS)}")
  runner = importlib.import_module(_REFERENCE_MODELS[reference_model])
  dtype = "fp32" if torch_dtype == torch.float32 else "bf16"
  max_logging.log(f"Loading the {reference_model} reference model ({dtype}) from {hf_model_path}")
  model, _ = runner.load_reference_model(
      _local_checkpoint_dir(hf_model_path), dtype=dtype, max_seq_len=config.max_target_length, tokenizer=tokenizer
  )
  return _ReferenceModel(model)


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ("true", "t", "yes", "1"):
    return True
  if v.lower() in ("false", "f", "no", "0"):
    return False
  raise argparse.ArgumentTypeError("Boolean value expected.")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  if storage is None:
    raise RuntimeError("google-cloud-storage is required to upload to GCS.")
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


def get_top_k_tokens_scores(logits_tensor, tokenizer_instance, k=10, description=""):
  """Get the top-k tokens and their scores from a given logits tensor."""
  max_logging.log(f"\n--- {description} top {k} tokens ---")
  collected_tokens = []
  topk_results = torch.topk(logits_tensor[0], k=k)
  for i in range(k):
    tok_id = topk_results.indices[i].item()
    score = topk_results.values[i].item()
    tok = tokenizer_instance.decode(tok_id) if tokenizer_instance is not None else str(tok_id)
    collected_tokens.append({"id": int(tok_id), "token": tok.strip(), "score": float(score)})

  table_str = f"| {'Token ID':<10} | {'Token':<20} | {'Score':<10} |\n"
  table_str += f"|{'-'*12}|{'-'*22}|{'-'*12}|\n"
  for d in collected_tokens:
    table_str += f"| {d['id']:<10} | {d['token']:<20} | {d['score']:<10.4f} |\n"
  max_logging.log(table_str)
  return collected_tokens


def compare_top_tokens(converted_tokens, golden_tokens):
  """Compares two lists of top tokens and calculates similarity metrics."""
  converted_ids = {token["id"] for token in converted_tokens}
  golden_ids = {token["id"] for token in golden_tokens}

  overlap_count = len(converted_ids & golden_ids)
  union_count = len(converted_ids | golden_ids)
  jaccard_similarity = overlap_count / union_count if union_count else 0.0

  min_len = min(len(converted_tokens), len(golden_tokens))
  rank_matches = sum(1 for a, b in zip(converted_tokens[:min_len], golden_tokens[:min_len]) if a["id"] == b["id"])
  rank_agreement = (rank_matches / min_len) * 100 if min_len > 0 else 0.0

  metrics = {
      "overlap_count": f"{overlap_count}/{len(golden_ids)}",
      "jaccard_similarity": f"{jaccard_similarity:.4f}",
      "rank_agreement_percentage": f"{rank_agreement:.2f}%",
  }

  max_logging.log("\n--- Similarity Metrics of Top Tokens ---")
  table_str = f"| {'Metric':<30} | {'Value':<20} |\n"
  table_str += f"|{'-'*32}|{'-'*22}|\n"
  for key, value in metrics.items():
    table_str += f"| {key:<30} | {str(value):<20} |\n"
  max_logging.log(table_str)
  return metrics


def _gather_and_free_logits(raw_logits: Any, seq_len: int, vocab_size: int) -> tuple[np.ndarray, int]:
  """Slices active sequence positions, allgathers to host float32, and frees device buffers."""
  # Slice sequence axis 1 first so axis-0 sharding divisibility across multihost FSDP is preserved.
  sliced_logits = raw_logits[:, :seq_len, :]
  if isinstance(sliced_logits, jax.Array) and sliced_logits.sharding is not None:
    rows_per_batch_shard = int(sliced_logits.sharding.shard_shape(sliced_logits.shape)[0])
  else:
    rows_per_batch_shard = int(sliced_logits.shape[0])

  gathered_logits = jax.experimental.multihost_utils.process_allgather(sliced_logits, tiled=True)
  host_logits = np.array(gathered_logits, dtype=np.float32, copy=True)
  if host_logits.ndim == 4:
    host_logits = np.reshape(host_logits, (-1, seq_len, vocab_size))

  for arr in (gathered_logits, sliced_logits, raw_logits):
    if isinstance(arr, jax.Array) and not arr.is_deleted():
      arr.delete()
  return host_logits, rows_per_batch_shard


# pylint: disable=too-many-locals
def _prepare_log_probs_and_probs(
    logits_2d: np.ndarray,
    clip_logits_epsilon: float | None = None,
    need_probs: bool = True,
    chunk_size: int = 256,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
  """Precomputes probabilities, log probabilities, and top1 tokens in float64 for fast branch screening."""
  num_tokens, vocab_size = logits_2d.shape
  top1 = np.argmax(logits_2d, axis=-1)
  log_prob = np.empty((num_tokens, vocab_size), dtype=np.float64)
  prob = np.empty((num_tokens, vocab_size), dtype=np.float64) if need_probs else None

  for t0 in range(0, num_tokens, chunk_size):
    t1 = min(t0 + chunk_size, num_tokens)
    f64 = np.asarray(logits_2d[t0:t1], dtype=np.float64)
    shifted = f64 - np.max(f64, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    lp_chunk = shifted - np.log(sum_exp)
    log_prob[t0:t1] = lp_chunk
    if need_probs:
      p_chunk = np.exp(lp_chunk)
      if clip_logits_epsilon is not None:
        eps_f64 = float(clip_logits_epsilon)
        p_clipped = np.clip(p_chunk, a_min=eps_f64, a_max=None)
        p_renorm = p_clipped / np.sum(p_clipped, axis=-1, keepdims=True)
        prob[t0:t1] = p_renorm
        log_prob[t0:t1] = np.log(p_renorm)
      else:
        prob[t0:t1] = p_chunk

  return prob, log_prob, top1


# pylint: disable=too-many-locals
def _resolve_tie_branch_dir(
    tie_branch_dir: str | Path,
    sample_name: str = "",
    max_ulp: float = 8.0,
) -> Path:
  """Resolves local or gs:// tie branch directory, downloading matching gs:// files to local cache."""
  dir_str = str(tie_branch_dir).strip()
  if not dir_str:
    return Path("")
  if dir_str.startswith("gs://"):
    cache_dir = Path("/tmp/cached_tie_branches")
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = dir_str[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    if storage is None:
      raise RuntimeError("google-cloud-storage is required to access GCS.")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    manifest_blob_name = f"{prefix}/tie_branches_manifest.json" if prefix else "tie_branches_manifest.json"
    manifest_blob = bucket.blob(manifest_blob_name)
    local_manifest = cache_dir / "tie_branches_manifest.json"
    if manifest_blob.exists():
      manifest_blob.download_to_filename(str(local_manifest))
    else:
      return Path("")

    if not sample_name:
      return cache_dir

    # Clean previous sample npz files from /tmp/cached_tie_branches to prevent disk/RAM accumulation
    current_marker = cache_dir / f".sample_{sample_name}"
    if not current_marker.exists():
      for p in cache_dir.glob("*.npz"):
        try:
          p.unlink(missing_ok=True)
        except OSError:
          pass
      for marker in cache_dir.glob(".sample_*"):
        try:
          marker.unlink(missing_ok=True)
        except OSError:
          pass
      current_marker.touch()

    try:
      with open(local_manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    except (OSError, json.JSONDecodeError):
      return cache_dir

    s_prefix = f"s{sample_name.split('_')[1]}" if sample_name.startswith("sample_") else sample_name
    to_download: list[tuple[str, Path]] = []
    for site_tag, entry in manifest.items():
      entry_sample = str(entry.get("sample", ""))
      match_sample = bool(
          entry_sample == sample_name
          or entry_sample == s_prefix
          or site_tag.startswith(f"{s_prefix}_")
          or site_tag.startswith(f"{sample_name}_")
      )
      if not match_sample:
        continue

      commit = str(entry.get("generator_commit", "")).strip()
      if not commit:
        continue
      if int(entry.get("num_swapped_sites", 0)) != 1:
        continue
      if len(entry.get("experts", [])) != 2:
        continue
      gap_ulp = float(entry.get("gap_ulp", -1.0))
      if gap_ulp < 0.0 or gap_ulp > max_ulp:
        continue

      fname = str(entry.get("file", f"{site_tag}.npz"))
      local_file = cache_dir / fname
      if not local_file.exists():
        blob_name = f"{prefix}/{fname}" if prefix else fname
        to_download.append((blob_name, local_file))

    if to_download:

      def _fetch(blob_and_path):
        bname, lpath = blob_and_path
        blob_obj = bucket.blob(bname)
        if blob_obj.exists():
          blob_obj.download_to_filename(str(lpath))

      with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_fetch, to_download))

    return cache_dir
  return Path(dir_str)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
_TIE_BRANCH_CACHE: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = {}


def load_and_validate_tie_branches(
    tie_branch_dir: str | Path,
    sample_name: str,
    max_ulp: float = 8.0,
) -> dict[str, dict[str, Any]]:
  """Loads and strictly validates all <= max_ulp tie branches for sample_name using multi-threading."""
  resolved_dir = _resolve_tie_branch_dir(tie_branch_dir, sample_name=sample_name, max_ulp=max_ulp)
  if not resolved_dir or not resolved_dir.exists():
    return {}
  manifest_path = resolved_dir / "tie_branches_manifest.json"
  if not manifest_path.exists():
    max_logging.log(f"Option A-prime: tie_branches_manifest.json not found in {resolved_dir}")
    return {}

  # Clean stale entries from other samples in _TIE_BRANCH_CACHE to conserve host RAM
  stale_keys = [k for k in _TIE_BRANCH_CACHE if k[1] != sample_name]
  for k in stale_keys:
    _TIE_BRANCH_CACHE.pop(k, None)

  cache_key = (str(resolved_dir.resolve()), sample_name, float(max_ulp), manifest_path.stat().st_mtime_ns)
  if cache_key in _TIE_BRANCH_CACHE:
    return _TIE_BRANCH_CACHE[cache_key]

  with open(manifest_path, "r", encoding="utf-8") as fp:
    manifest = json.load(fp)

  s_prefix = f"s{sample_name.split('_')[1]}" if sample_name.startswith("sample_") else sample_name
  eligible_items: list[tuple[str, dict[str, Any], Path, str, set[int], float, str, str]] = []

  for site_tag, entry in manifest.items():
    entry_sample = str(entry.get("sample", ""))
    match_sample = bool(
        entry_sample == sample_name
        or entry_sample == s_prefix
        or site_tag.startswith(f"{s_prefix}_")
        or site_tag.startswith(f"{sample_name}_")
    )
    if not match_sample:
      continue

    branch_id = str(entry.get("branch_id") or site_tag)

    commit = str(entry.get("generator_commit", "")).strip()
    if not commit:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: empty generator_commit")
      continue

    if "num_swapped_sites" not in entry:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: missing num_swapped_sites")
      continue
    num_sites = int(entry["num_swapped_sites"])
    if num_sites != 1:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: num_swapped_sites={num_sites} != 1")
      continue

    experts = entry.get("experts", [])
    if len(experts) != 2:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: len(experts)={len(experts)} != 2")
      continue
    exp_set = {int(e) for e in experts}

    if "gap_ulp" not in entry:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: missing gap_ulp")
      continue
    gap_ulp = float(entry["gap_ulp"])
    if gap_ulp < 0.0 or gap_ulp > max_ulp:
      max_logging.log(
          f"Option A-prime: branch {site_tag} rejected: gap_ulp={gap_ulp} exceeds max_ulp={max_ulp} (or < 0.0)"
      )
      continue

    fname = str(entry.get("file", f"{site_tag}.npz"))
    npz_path = resolved_dir / fname
    if not npz_path.exists():
      max_logging.log(f"Option A-prime: branch {site_tag} file not found: {npz_path}")
      continue

    expected_sha = str(entry.get("sha256", "")).strip()
    if not expected_sha or len(expected_sha) != 64:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: invalid expected sha256 {expected_sha!r}")
      continue

    ref_k = int(entry.get("ref_rank_k_expert", list(exp_set)[0]))
    ref_k1 = int(entry.get("ref_rank_k_plus_1_expert", list(exp_set)[1]))
    if {ref_k, ref_k1} != exp_set:
      max_logging.log(f"Option A-prime: branch {site_tag} rejected: ref_rank experts != {exp_set}")
      continue

    if "primary_experts" in entry and "swapped_experts" in entry:
      p_exps = set(int(x) for x in entry["primary_experts"])
      s_exps = set(int(x) for x in entry["swapped_experts"])
      if (p_exps ^ s_exps) != exp_set:
        max_logging.log(f"Option A-prime: branch {site_tag} rejected: expert symmetric diff != {exp_set}")
        continue

    eligible_items.append((site_tag, entry, npz_path, expected_sha, exp_set, gap_ulp, branch_id, commit))

  def _load_single_branch(item):
    site_tag, entry, npz_path, expected_sha, exp_set, gap_ulp, branch_id, commit = item
    with open(npz_path, "rb") as fp:
      actual_sha = hashlib.sha256(fp.read()).hexdigest()
    if actual_sha != expected_sha:
      max_logging.log(f"Option A-prime: branch {site_tag} sha256 mismatch: {actual_sha} != {expected_sha}")
      return None

    with np.load(npz_path) as data:
      if "tie_flip_logits" not in data:
        max_logging.log(f"Option A-prime: branch {site_tag} missing tie_flip_logits")
        return None

      if "tie_flip_sites" in data:
        tfs = np.asarray(data["tie_flip_sites"])
        if len(tfs) != 1:
          max_logging.log(f"Option A-prime: branch {site_tag} rejected: tie_flip_sites length {len(tfs)} != 1")
          return None
        tfs_row = tfs[0]
        if len(tfs_row) >= 4:
          npz_exps = {int(tfs_row[2]), int(tfs_row[3])}
          if npz_exps != exp_set:
            max_logging.log(f"Option A-prime: branch {site_tag} rejected: npz experts {npz_exps} != manifest {exp_set}")
            return None

      raw_logits = np.asarray(data["tie_flip_logits"], dtype=np.float32)
      branch_logits = raw_logits[0] if (raw_logits.ndim == 3 and raw_logits.shape[0] == 1) else raw_logits

    return (
        site_tag,
        {
            "branch_id": branch_id,
            "best_branch_id": branch_id,
            "site_tag": site_tag,
            "sample": entry.get("sample", "") or sample_name,
            "layer": int(entry.get("layer", -1)),
            "pos": int(entry.get("pos", -1)),
            "experts": [int(e) for e in entry.get("experts", [])],
            "gap_ulp": gap_ulp,
            "sha256": actual_sha,
            "generator_commit": commit,
            "logits": branch_logits,
        },
    )

  valid_branches: dict[str, dict[str, Any]] = {}
  if eligible_items:
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
      results = list(pool.map(_load_single_branch, eligible_items))
    for res in results:
      if res is not None:
        valid_branches[res[0]] = res[1]

  _TIE_BRANCH_CACHE[cache_key] = valid_branches
  return valid_branches


# pylint: disable=too-many-positional-arguments,too-many-locals
def evaluate_against_ref(
    cand_logits: np.ndarray,
    ref_logits: np.ndarray,
    tokens: np.ndarray | None,
    bucket_gates: dict[str, Any],
    effective_max_kl: float,
    effective_min_cos: float,
    clip_logits_epsilon: float | None = None,
    atol_linf: float | None = None,
    cos_sim_diagnostic_only: bool = False,
    include_top10_and_legacy_kl: bool = False,
    log_metrics: bool = False,
    precomputed_golden_prob=None,
    precomputed_golden_log_prob=None,
    precomputed_model_log_prob=None,
    precomputed_golden_entropy=None,
    precomputed_golden_std=None,
    precomputed_golden_norm=None,
    precomputed_golden_top1=None,
    precomputed_model_std=None,
    precomputed_model_norm=None,
    precomputed_model_top1=None,
    precomputed_model_shifted=None,
    precomputed_model_lse=None,
    precomputed_kl: tuple[float, float] | None = None,
) -> tuple[bool, dict[str, Any]]:
  """Evaluates cand_logits against ref_logits using exact Copy-0 bucket thresholds."""
  metrics = check_mathematical_parity(
      model_logits=cand_logits,
      golden_logits=ref_logits,
      atol_linf=atol_linf,
      min_cos_sim=None,
      max_kl=None,
      clip_logits_epsilon=clip_logits_epsilon,
      tokens=tokens,
      include_top10_and_legacy_kl=include_top10_and_legacy_kl,
      log_metrics=log_metrics,
      precomputed_golden_prob=precomputed_golden_prob,
      precomputed_golden_log_prob=precomputed_golden_log_prob,
      precomputed_model_log_prob=precomputed_model_log_prob,
      precomputed_golden_entropy=precomputed_golden_entropy,
      precomputed_golden_std=precomputed_golden_std,
      precomputed_golden_norm=precomputed_golden_norm,
      precomputed_golden_top1=precomputed_golden_top1,
      precomputed_model_std=precomputed_model_std,
      precomputed_model_norm=precomputed_model_norm,
      precomputed_model_top1=precomputed_model_top1,
      precomputed_model_shifted=precomputed_model_shifted,
      precomputed_model_lse=precomputed_model_lse,
      precomputed_kl=precomputed_kl,
  )
  passes = (
      (metrics["max_kl"] <= effective_max_kl)
      and (metrics["mean_kl"] <= float(bucket_gates["mean_kl"]))
      and (metrics["top1_agreement"] >= float(bucket_gates["top1_agreement"]))
      and (metrics["p95_kl"] <= float(bucket_gates["p95_kl"]))
      and (metrics["p99_kl"] <= float(bucket_gates["p99_kl"]))
      and (True if cos_sim_diagnostic_only else (metrics["mean_cos_sim"] >= effective_min_cos))
  )
  if atol_linf is not None and "l_inf_pass" in metrics:
    passes = passes and bool(metrics["l_inf_pass"])
  return bool(passes), metrics


# pylint: disable=too-many-locals,too-many-positional-arguments
def check_batch_slot_determinism(
    full_train_logits,
    seq_len,
    min_vocab_size,
    rows_per_batch_shard=1,
    rerun_train_logits=None,
    allow_reference_tie_branches=False,
    tie_branch_dir="",
    tie_branch_max_ulp=8.0,
    sample_name="",
    primary_golden_logits=None,
    bucket_gates=None,
    effective_max_kl=None,
    effective_min_cos=None,
    tokens=None,
    clip_logits_epsilon=None,
    atol_linf=None,
    cos_sim_diagnostic_only=False,
    start_index: int = 0,
    token_size: int | None = None,
    pos_mask: np.ndarray | None = None,
    preloaded_branches: dict[str, dict[str, Any]] | None = None,
):
  """Checks rerun bitwise determinism and PR #4447 mean batch-slot drift, logging row spike diagnostics."""
  t_start = time.perf_counter()
  t_load_s = 0.0
  if rerun_train_logits is None:
    raise ValueError("rerun_train_logits is required to verify compiled rerun determinism.")

  run1_full = np.ascontiguousarray(full_train_logits[:, :seq_len, :], dtype=np.float32)
  run2_full = np.ascontiguousarray(rerun_train_logits[:, :seq_len, :], dtype=np.float32)
  if run1_full.shape != run2_full.shape:
    raise ValueError(f"Rerun logits shape {run2_full.shape} does not match first run {run1_full.shape}.")

  both_finite = bool(np.all(np.isfinite(run1_full)) and np.all(np.isfinite(run2_full)))
  rerun_max_abs_diff = float(np.max(np.abs(run1_full - run2_full)))
  rerun_bitwise_identical = bool(
      both_finite and (rerun_max_abs_diff == 0.0) and np.array_equal(run1_full.view(np.uint32), run2_full.view(np.uint32))
  )
  rerun_argmax_identical = bool(
      both_finite and np.array_equal(np.argmax(run1_full, axis=-1), np.argmax(run2_full, axis=-1))
  )

  logits_np = run1_full[:, :, :min_vocab_size]
  max_logit_magnitude = float(np.max(np.abs(logits_np)))
  rel_tolerance = 5e-4
  abs_tolerance = rel_tolerance * max(max_logit_magnitude, 1.0)
  mean_diff_threshold = 1e-4
  num_rows = int(logits_np.shape[0])
  shard_rows = max(1, int(rows_per_batch_shard))

  if num_rows > 1:
    row0 = logits_np[0]
    max_batch_diff = 0.0
    sum_batch_diff = 0.0
    total_elements = (num_rows - 1) * row0.size
    for r in range(1, num_rows):
      d = np.abs(logits_np[r] - row0)
      max_batch_diff = max(max_batch_diff, float(np.max(d)))
      sum_batch_diff += float(np.sum(d, dtype=np.float64))
    mean_batch_diff = float(sum_batch_diff / total_elements)
  else:
    max_batch_diff = 0.0
    mean_batch_diff = 0.0

  intra_shard_max_diff = 0.0
  intra_shard_sum_diff = 0.0
  intra_shard_count = 0
  if shard_rows > 1:
    num_shards = (num_rows + shard_rows - 1) // shard_rows
    for k in range(num_shards):
      start_idx = k * shard_rows
      end_idx = min(start_idx + shard_rows, num_rows)
      if end_idx - start_idx > 1:
        leader = logits_np[start_idx]
        for f_idx in range(start_idx + 1, end_idx):
          d = np.abs(logits_np[f_idx] - leader)
          intra_shard_max_diff = max(intra_shard_max_diff, float(np.max(d)))
          intra_shard_sum_diff += float(np.sum(d, dtype=np.float64))
          intra_shard_count += leader.size

  if intra_shard_count > 0:
    intra_shard_mean_diff = float(intra_shard_sum_diff / intra_shard_count)
    intra_shard_checked = True
  else:
    intra_shard_max_diff = 0.0
    intra_shard_mean_diff = 0.0
    intra_shard_checked = False

  # Diagnostic: shard leaders k * shard_rows (k >= 1) vs row 0.
  cross_shard_max_diff = 0.0
  cross_shard_sum_diff = 0.0
  cross_shard_count = 0
  row0 = logits_np[0]
  k = 1
  while k * shard_rows < num_rows:
    d = np.abs(logits_np[k * shard_rows] - row0)
    cross_shard_max_diff = max(cross_shard_max_diff, float(np.max(d)))
    cross_shard_sum_diff += float(np.sum(d, dtype=np.float64))
    cross_shard_count += row0.size
    k += 1

  if cross_shard_count > 0:
    cross_shard_mean_diff = float(cross_shard_sum_diff / cross_shard_count)
  else:
    cross_shard_max_diff = 0.0
    cross_shard_mean_diff = 0.0

  copy0_primary_pass = True
  any_copy_primary_pass = True
  all_copies_valid = True
  tie_branch_batch_gate_passed = True
  primary_passed_copies = []
  primary_matched_copies = []
  tie_branch_matched_copies = []
  unmatched_copies = []
  per_copy_report = []
  valid_branches = {}
  t_metric_eval_s = 0.0

  if allow_reference_tie_branches:
    if primary_golden_logits is None or bucket_gates is None or effective_max_kl is None or effective_min_cos is None:
      raise ValueError(
          "allow_reference_tie_branches=True requires primary_golden_logits, "
          "bucket_gates, effective_max_kl, and effective_min_cos"
      )

    if preloaded_branches is not None:
      valid_branches = preloaded_branches
    else:
      t_load_start = time.perf_counter()
      valid_branches = load_and_validate_tie_branches(tie_branch_dir, sample_name=sample_name, max_ulp=tie_branch_max_ulp)
      t_load_s = time.perf_counter() - t_load_start
    t_eval_start = time.perf_counter()

    # Step 1: Fast bitwise identity (exact duplicate elimination only, no subsampling)
    unique_trajectories = []
    copy_to_unique = {}
    for c in range(num_rows):
      cand = logits_np[c]
      matched_u = None
      for u in unique_trajectories:
        other = logits_np[u]
        if (
            cand[0, 0] == other[0, 0]
            and cand[-1, -1] == other[-1, -1]
            and cand[0, -1] == other[0, -1]
            and cand[-1, 0] == other[-1, 0]
        ):
          if np.array_equal(cand, other):
            matched_u = u
            break
      if matched_u is not None:
        copy_to_unique[c] = matched_u
      else:
        unique_trajectories.append(c)
        copy_to_unique[c] = c

    num_unique = len(unique_trajectories)
    t_len = logits_np.shape[1]

    if primary_golden_logits.shape[0] > t_len or (
        pos_mask is not None and primary_golden_logits.shape[0] == len(pos_mask)
    ):
      pri_slice = primary_golden_logits[start_index:token_size, :min_vocab_size]
      pri_ref = (
          pri_slice[pos_mask] if (pos_mask is not None and pri_slice.shape[0] == len(pos_mask)) else pri_slice[:t_len]
      )
    else:
      pri_ref = primary_golden_logits[:t_len, :min_vocab_size]

    cand_sliced_list = [logits_np[u][:t_len, :min_vocab_size] for u in unique_trajectories]
    branch_list = list(valid_branches.values())
    num_branches = len(branch_list)

    # Step 2: Exact float64 Batched DGEMM Token Chunking
    br_refs = []
    for br in branch_list:
      br_logits = br["logits"]
      if br_logits.shape[0] > t_len or (pos_mask is not None and br_logits.shape[0] == len(pos_mask)):
        br_slice = br_logits[start_index:token_size, :min_vocab_size]
        br_ref = br_slice[pos_mask] if (pos_mask is not None and br_slice.shape[0] == len(pos_mask)) else br_slice[:t_len]
      else:
        br_ref = br_logits[:t_len, :min_vocab_size]
      br_refs.append(br_ref)

    if num_branches > 0:

      def _check_diff(b):
        return np.any(br_refs[b] != pri_ref, axis=-1)

      with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, num_branches)) as pool:
        diff_mask_rows = list(pool.map(_check_diff, range(num_branches)))
      diff_mask = np.asarray(diff_mask_rows)
    else:
      diff_mask = np.zeros((0, t_len), dtype=bool)

    T_CHUNK = 32
    num_chunks = (t_len + T_CHUNK - 1) // T_CHUNK

    def _process_token_chunk(chunk_i):
      t0 = chunk_i * T_CHUNK
      t1 = min(t0 + T_CHUNK, t_len)

      chunk_max_kl_prim = np.zeros(num_unique, dtype=np.float64)
      chunk_sum_kl_prim = np.zeros(num_unique, dtype=np.float64)
      chunk_max_kl_br = np.zeros((num_unique, num_branches), dtype=np.float64)
      chunk_sum_kl_br = np.zeros((num_unique, num_branches), dtype=np.float64)

      for t in range(t0, t1):
        p_f64 = np.asarray(pri_ref[t], dtype=np.float64)
        p_m = np.max(p_f64)
        p_s = p_f64 - p_m
        p_e = np.exp(p_s)
        p_sum = np.sum(p_e)
        p_prim_t = p_e / p_sum
        neg_ent_prim_t = np.sum(p_prim_t * p_s) - np.log(p_sum)

        active_b = np.flatnonzero(diff_mask[:, t]) if num_branches > 0 else np.array([], dtype=int)

        c_f64 = np.asarray([cand[t] for cand in cand_sliced_list], dtype=np.float64)
        c_m = np.max(c_f64, axis=-1, keepdims=True)
        c_s = c_f64 - c_m
        c_e = np.exp(c_s)
        c_sum = np.sum(c_e, axis=-1, keepdims=True)
        log_q_t = c_s - np.log(c_sum)

        if len(active_b) == 0:
          sum_p_log_q = np.dot(log_q_t, p_prim_t)
          kl_prim_t = np.maximum(0.0, neg_ent_prim_t - sum_p_log_q)

          chunk_max_kl_prim = np.maximum(chunk_max_kl_prim, kl_prim_t)
          chunk_sum_kl_prim += kl_prim_t

          if num_branches > 0:
            chunk_max_kl_br = np.maximum(chunk_max_kl_br, kl_prim_t[:, None])
            chunk_sum_kl_br += kl_prim_t[:, None]
        else:
          p_br_list = []
          neg_ent_br_list = []
          for b_idx in active_b:
            b_f64 = np.asarray(br_refs[b_idx][t], dtype=np.float64)
            b_m = np.max(b_f64)
            b_s = b_f64 - b_m
            b_e = np.exp(b_s)
            b_sum = np.sum(b_e)
            p_b = b_e / b_sum
            p_br_list.append(p_b)
            neg_ent_br_list.append(np.sum(p_b * b_s) - np.log(b_sum))

          p_br_t = np.asarray(p_br_list, dtype=np.float64)
          neg_ent_br_t = np.asarray(neg_ent_br_list, dtype=np.float64)

          P_t = np.vstack([p_prim_t[None, :], p_br_t])
          sum_p_log_q = np.matmul(P_t, log_q_t.T)

          kl_prim_t = np.maximum(0.0, neg_ent_prim_t - sum_p_log_q[0])
          chunk_max_kl_prim = np.maximum(chunk_max_kl_prim, kl_prim_t)
          chunk_sum_kl_prim += kl_prim_t

          kl_active_t = np.maximum(0.0, neg_ent_br_t[:, None] - sum_p_log_q[1:]).T

          kl_br_t = np.repeat(kl_prim_t[:, None], num_branches, axis=1)
          for k, b_idx in enumerate(active_b):
            kl_br_t[:, b_idx] = kl_active_t[:, k]

          chunk_max_kl_br = np.maximum(chunk_max_kl_br, kl_br_t)
          chunk_sum_kl_br += kl_br_t

      return chunk_max_kl_prim, chunk_sum_kl_prim, chunk_max_kl_br, chunk_sum_kl_br

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, num_chunks))) as pool:
      chunk_results = list(pool.map(_process_token_chunk, range(num_chunks)))

    total_max_kl_prim = np.zeros(num_unique, dtype=np.float64)
    total_sum_kl_prim = np.zeros(num_unique, dtype=np.float64)
    total_max_kl_br = np.zeros((num_unique, num_branches), dtype=np.float64)
    total_sum_kl_br = np.zeros((num_unique, num_branches), dtype=np.float64)

    for c_max_p, c_sum_p, c_max_b, c_sum_b in chunk_results:
      total_max_kl_prim = np.maximum(total_max_kl_prim, c_max_p)
      total_sum_kl_prim += c_sum_p
      total_max_kl_br = np.maximum(total_max_kl_br, c_max_b)
      total_sum_kl_br += c_sum_b

    total_mean_kl_prim = total_sum_kl_prim / t_len
    total_mean_kl_br = total_sum_kl_br / t_len if num_branches > 0 else np.zeros((num_unique, 0), dtype=np.float64)

    # Step 2c: Fast primary evaluate_against_ref with precomputed_kl
    primary_eval_results = {}

    def _eval_primary_fast(u_tuple):
      idx, u = u_tuple
      c_cand = cand_sliced_list[idx]
      u_max_kl = float(total_max_kl_prim[idx])
      u_mean_kl = float(total_mean_kl_prim[idx])
      pri_pass, pri_metrics = evaluate_against_ref(
          c_cand,
          pri_ref,
          tokens=tokens,
          bucket_gates=bucket_gates,
          effective_max_kl=effective_max_kl,
          effective_min_cos=effective_min_cos,
          clip_logits_epsilon=clip_logits_epsilon,
          atol_linf=atol_linf,
          cos_sim_diagnostic_only=cos_sim_diagnostic_only,
          include_top10_and_legacy_kl=False,
          log_metrics=False,
          precomputed_kl=(u_max_kl, u_mean_kl),
      )
      return u, (bool(pri_pass), pri_metrics)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, num_unique))) as pool:
      pri_results = list(pool.map(_eval_primary_fast, list(enumerate(unique_trajectories))))
    for u, res in pri_results:
      primary_eval_results[u] = res

    # Step 3: Exact float64 Branch Attribution
    best_tie_for_idx = {idx: None for idx in range(num_unique)}

    for b_idx, br in enumerate(branch_list):
      br_id = str(br.get("branch_id") or br.get("branch_name"))
      gap_ulp = float(br.get("gap_ulp", float("inf")))
      site = str(br.get("site_tag") or br.get("site", "") or br_id)
      layer = int(br.get("layer", -1))
      pos = int(br.get("pos", -1))
      experts = list(br.get("experts", []))
      br_sha = str(br.get("sha256", ""))
      br_commit = str(br.get("generator_commit", ""))
      br_ref = br_refs[b_idx]

      for idx in range(num_unique):
        m_kl = float(total_max_kl_br[idx, b_idx])
        mn_kl = float(total_mean_kl_br[idx, b_idx])
        passed = bool((m_kl <= effective_max_kl) and (mn_kl <= float(bucket_gates["mean_kl"])))
        curr_best = best_tie_for_idx[idx]
        is_better = False
        if curr_best is None:
          is_better = True
        elif passed and not curr_best["passed"]:
          is_better = True
        elif passed == curr_best["passed"]:
          if m_kl < curr_best["max_kl"]:
            is_better = True
          elif m_kl == curr_best["max_kl"] and mn_kl < curr_best["mean_kl"]:
            is_better = True

        if is_better:
          best_tie_for_idx[idx] = {
              "branch_id": br_id,
              "branch_ref": br_ref,
              "passed": passed,
              "max_kl": m_kl,
              "mean_kl": mn_kl,
              "site": site,
              "layer": layer,
              "pos": pos,
              "experts": experts,
              "gap_ulp": gap_ulp,
              "sha256": br_sha,
              "generator_commit": br_commit,
          }

    # Step 4: Fast winning branch evaluate_against_ref with precomputed_kl
    def _eval_one_unique_copy(item):
      idx, u = item
      cand = cand_sliced_list[idx]
      primary_passed, pri_metrics = primary_eval_results[u]
      primary_max_kl = float(pri_metrics["max_kl"])
      primary_mean_kl = float(pri_metrics["mean_kl"])
      primary_l2 = float(pri_metrics.get("l2", 0.0))
      primary_l_inf = float(pri_metrics.get("l_inf", 0.0))

      best_tie = best_tie_for_idx[idx]
      best_branch_passed = False
      best_branch_id = None
      best_branch_site = None
      best_branch_layer = None
      best_branch_pos = None
      best_branch_experts = None
      best_branch_gap_ulp = None
      best_branch_sha256 = None
      best_branch_commit = None
      best_branch_max_kl = None
      best_branch_mean_kl = None
      best_branch_l2 = None
      best_branch_l_inf = None
      best_branch_metrics = None

      if best_tie is not None:
        best_branch_id = best_tie["branch_id"]
        best_branch_site = best_tie["site"]
        best_branch_layer = best_tie["layer"]
        best_branch_pos = best_tie["pos"]
        best_branch_experts = best_tie["experts"]
        best_branch_gap_ulp = best_tie["gap_ulp"]
        best_branch_sha256 = best_tie["sha256"]
        best_branch_commit = best_tie["generator_commit"]
        br_ref = best_tie["branch_ref"]

        br_pass, br_metrics = evaluate_against_ref(
            cand,
            br_ref,
            tokens=tokens,
            bucket_gates=bucket_gates,
            effective_max_kl=effective_max_kl,
            effective_min_cos=effective_min_cos,
            clip_logits_epsilon=clip_logits_epsilon,
            atol_linf=atol_linf,
            cos_sim_diagnostic_only=cos_sim_diagnostic_only,
            include_top10_and_legacy_kl=False,
            log_metrics=False,
            precomputed_kl=(best_tie["max_kl"], best_tie["mean_kl"]),
        )
        best_branch_passed = bool(br_pass)
        best_branch_max_kl = float(br_metrics["max_kl"])
        best_branch_mean_kl = float(br_metrics["mean_kl"])
        best_branch_l2 = float(br_metrics.get("l2", 0.0))
        best_branch_l_inf = float(br_metrics.get("l_inf", 0.0))
        best_branch_metrics = br_metrics

      if primary_passed and best_branch_passed:
        if primary_max_kl <= best_branch_max_kl:
          matched = "primary"
        else:
          matched = "tie_branch"
      elif primary_passed:
        matched = "primary"
      elif best_branch_passed:
        matched = "tie_branch"
      else:
        matched = "neither"

      passed = bool(matched in ("primary", "tie_branch"))
      report_data = {
          "passed": passed,
          "matched": matched,
          "primary_passed": primary_passed,
          "primary_max_kl": primary_max_kl,
          "primary_mean_kl": primary_mean_kl,
          "primary_l2": primary_l2,
          "primary_l_inf": primary_l_inf,
          "best_branch_passed": best_branch_passed,
          "best_branch_id": best_branch_id,
          "best_branch_max_kl": best_branch_max_kl,
          "best_branch_mean_kl": best_branch_mean_kl,
          "best_branch_l2": best_branch_l2,
          "best_branch_l_inf": best_branch_l_inf,
          "branch_id": best_branch_id,
          "site": best_branch_site,
          "layer": best_branch_layer,
          "pos": best_branch_pos,
          "experts": best_branch_experts,
          "gap_ulp": best_branch_gap_ulp,
          "sha256": best_branch_sha256,
          "generator_commit": best_branch_commit,
          "primary_metrics": pri_metrics,
          "best_branch_metrics": best_branch_metrics,
      }
      return u, report_data

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, num_unique)) as pool:
      u_eval_results = list(pool.map(_eval_one_unique_copy, list(enumerate(unique_trajectories))))
    unique_reports = {}
    for u, report_data in u_eval_results:
      unique_reports[u] = report_data

    # Step 5: Assemble per_copy_report for all rows
    for c in range(num_rows):
      u = copy_to_unique[c]
      entry = dict(unique_reports[u])
      entry["copy"] = c
      per_copy_report.append(entry)

      log_msg = (
          f"Copy {c:2d} (unique={u:2d}): matched={entry['matched']} (passed={entry['passed']}, "
          f"primary_passed={entry['primary_passed']}, best_branch_passed={entry['best_branch_passed']}, "
          f"best_branch_id={entry['best_branch_id']}) "
          f"primary: max_kl={entry['primary_max_kl']:.6f}, mean_kl={entry['primary_mean_kl']:.6e}, "
          f"l2={entry['primary_l2']:.4f}, l_inf={entry['primary_l_inf']:.6f}"
      )
      if entry["best_branch_max_kl"] is not None:
        log_msg += (
            f" | tie_branch: max_kl={entry['best_branch_max_kl']:.6f}, mean_kl={entry['best_branch_mean_kl']:.6e}, "
            f"l2={entry['best_branch_l2']:.4f}, l_inf={entry['best_branch_l_inf']:.6f}"
        )
      if entry["site"] is not None:
        log_msg += f" (site={entry['site']}, gap_ulp={entry['gap_ulp']})"
      max_logging.log(log_msg)

    copy0_primary_pass = bool(per_copy_report[0]["primary_passed"]) if per_copy_report else False
    any_copy_primary_pass = any(bool(entry["primary_passed"]) for entry in per_copy_report)
    all_copies_valid = all(entry["matched"] in ("primary", "tie_branch") and entry["passed"] for entry in per_copy_report)
    tie_branch_batch_gate_passed = bool(any_copy_primary_pass and all_copies_valid)
    primary_passed_copies = [entry["copy"] for entry in per_copy_report if entry["primary_passed"]]
    primary_matched_copies = [entry["copy"] for entry in per_copy_report if entry["matched"] == "primary"]
    tie_branch_matched_copies = [entry["copy"] for entry in per_copy_report if entry["matched"] == "tie_branch"]
    unmatched_copies = [entry["copy"] for entry in per_copy_report if entry["matched"] == "neither"]

    batch_gate_log_line = (
        f"[TIE_BRANCH_BATCH_GATE] sample={sample_name} "
        f"tie_branch_batch_gate_passed={tie_branch_batch_gate_passed} "
        f"any_copy_primary_pass={any_copy_primary_pass} all_copies_valid={all_copies_valid} "
        f"copy0_primary_pass={copy0_primary_pass} "
        f"primary_passed_copies={primary_passed_copies} "
        f"primary_matched_copies={primary_matched_copies} "
        f"tie_branch_matched_copies={tie_branch_matched_copies} "
        f"unmatched_copies={unmatched_copies}"
    )
    max_logging.log(batch_gate_log_line)
    if jax.process_index() == 0:
      print(batch_gate_log_line, flush=True)

    t_metric_eval_s = time.perf_counter() - t_eval_start
    t_total_check_s = time.perf_counter() - t_start
    timing_log_line = (
        f"[TIE_BRANCH_TIMING] sample={sample_name} num_branches={len(valid_branches)} "
        f"tie_load_s={t_load_s:.3f} metric_eval_s={t_metric_eval_s:.3f} total_check_s={t_total_check_s:.3f}"
    )
    max_logging.log(timing_log_line)
    if jax.process_index() == 0:
      print(timing_log_line, flush=True)

    passed = bool(rerun_bitwise_identical and rerun_argmax_identical and tie_branch_batch_gate_passed)
  else:
    passed = bool(rerun_bitwise_identical and rerun_argmax_identical and (mean_batch_diff <= mean_diff_threshold))

  if passed and max_batch_diff > abs_tolerance:
    max_logging.log(
        f"WARNING: MaxText diagnostic max batch diff {max_batch_diff:.4e} (intra_max={intra_shard_max_diff:.4e}, "
        f"cross_max={cross_shard_max_diff:.4e}) exceeds diagnostic rel tolerance {abs_tolerance:.4e}, "
        f"but passed rerun determinism (rerun_bitwise_identical={rerun_bitwise_identical}, "
        f"rerun_max_abs_diff={rerun_max_abs_diff:.4e}, rerun_argmax_identical={rerun_argmax_identical}) "
        f"and mean_batch_diff={mean_batch_diff:.4e} <= {mean_diff_threshold:.4e} (PR #4447)."
    )
  else:
    max_logging.log(
        f"Batch & rerun determinism check ({logits_np.shape[0]} rows, rows_per_shard={rows_per_batch_shard}): "
        f"rerun_bitwise_identical={rerun_bitwise_identical}, rerun_max_abs_diff={rerun_max_abs_diff:.4e}, "
        f"rerun_argmax_identical={rerun_argmax_identical}, mean_diff={mean_batch_diff:.4e}, "
        f"diag[max_diff={max_batch_diff:.4e}, intra_max={intra_shard_max_diff:.4e}, "
        f"cross_max={cross_shard_max_diff:.4e}, tol={abs_tolerance:.4e} (5e-4 x max|logit|={max_logit_magnitude:.4f})]"
    )

  return {
      "num_batch_rows_checked": int(logits_np.shape[0]),
      "rows_per_batch_shard": int(rows_per_batch_shard),
      "rerun_max_abs_diff": rerun_max_abs_diff,
      "rerun_bitwise_identical": rerun_bitwise_identical,
      "rerun_argmax_identical": rerun_argmax_identical,
      "max_batch_diff": max_batch_diff,
      "mean_batch_diff": mean_batch_diff,
      "intra_shard_checked": intra_shard_checked,
      "intra_shard_max_diff": intra_shard_max_diff,
      "intra_shard_mean_diff": intra_shard_mean_diff,
      "cross_shard_max_diff": cross_shard_max_diff,
      "cross_shard_mean_diff": cross_shard_mean_diff,
      "max_logit_magnitude": max_logit_magnitude,
      "batch_det_tol": abs_tolerance,
      "mean_diff_threshold": mean_diff_threshold,
      "passed": passed,
      "allow_reference_tie_branches": bool(allow_reference_tie_branches),
      "copy0_primary_pass": bool(copy0_primary_pass),
      "any_copy_primary_pass": bool(any_copy_primary_pass),
      "all_copies_valid": bool(all_copies_valid),
      "tie_branch_batch_gate_passed": bool(tie_branch_batch_gate_passed),
      "primary_passed_copies": primary_passed_copies,
      "primary_matched_copies": primary_matched_copies,
      "tie_branch_matched_copies": tie_branch_matched_copies,
      "unmatched_copies": unmatched_copies,
      "per_copy_tie_branch_report": per_copy_report,
      "tie_branches_evaluated_count": len(valid_branches),
      "metric_eval_s": t_metric_eval_s,
  }


# pylint: disable=too-many-positional-arguments
def evaluate_sample_gates(
    parity_metrics: dict[str, Any],
    batch_det: dict[str, Any],
    bucket_gates: dict[str, Any],
    effective_max_kl: float,
    effective_min_cos: float,
    cos_sim_diagnostic_only: bool = False,
    is_multimodal: bool = False,
) -> dict[str, bool]:
  """Evaluates whether all mathematical parity and rerun/batch determinism gates pass for a sample."""
  if is_multimodal and cos_sim_diagnostic_only:
    raise ValueError(
        "cos_sim_diagnostic_only=True is forbidden for multimodal samples (M1..M8); "
        "only text-only sample_6_verbatim_repeat_stress may enable it."
    )
  if batch_det.get("allow_reference_tie_branches", False):
    batch_determinism_pass = bool(
        batch_det.get("rerun_bitwise_identical", False)
        and batch_det.get("rerun_argmax_identical", False)
        and batch_det.get("passed", False)
    )
  else:
    mean_diff_limit = float(batch_det["mean_diff_threshold"]) if "mean_diff_threshold" in batch_det else 1e-4
    batch_determinism_pass = bool(
        batch_det["rerun_bitwise_identical"]
        and batch_det["rerun_argmax_identical"]
        and (float(batch_det["mean_batch_diff"]) <= mean_diff_limit)
    )
  return {
      "max_kl_pass": bool(parity_metrics["max_kl"] <= effective_max_kl),
      "mean_kl_pass": bool(parity_metrics["mean_kl"] <= float(bucket_gates["mean_kl"])),
      "top1_agreement_pass": bool(parity_metrics["top1_agreement"] >= float(bucket_gates["top1_agreement"])),
      "p95_kl_pass": bool(parity_metrics["p95_kl"] <= float(bucket_gates["p95_kl"])),
      "p99_kl_pass": bool(parity_metrics["p99_kl"] <= float(bucket_gates["p99_kl"])),
      "mean_cos_sim_pass": (
          True if cos_sim_diagnostic_only else bool(parity_metrics["mean_cos_sim"] >= effective_min_cos)
      ),
      "batch_determinism_pass": batch_determinism_pass,
  }


def _compute_per_token_kl(
    golden_logits_sliced,
    model_logits_sliced,
    clip_logits_epsilon=None,
    precomputed_golden_prob=None,
    precomputed_golden_log_prob=None,
    precomputed_model_log_prob=None,
    precomputed_golden_entropy=None,
    precomputed_model_shifted=None,
    precomputed_model_lse=None,
) -> np.ndarray:
  """Computes per-position D_KL(P_golden || P_model) in host float64."""
  if clip_logits_epsilon is None:
    if (
        precomputed_golden_entropy is not None
        and precomputed_golden_prob is not None
        and precomputed_model_shifted is not None
        and precomputed_model_lse is not None
    ):
      raw_kl = precomputed_golden_entropy - (
          np.sum(precomputed_golden_prob * precomputed_model_shifted, axis=-1, dtype=np.float64)
          - precomputed_model_lse[:, 0]
      )
      return np.asarray(np.maximum(raw_kl, 0.0), dtype=np.float64)
    if (
        precomputed_golden_entropy is not None
        and precomputed_golden_prob is not None
        and precomputed_model_log_prob is not None
    ):
      raw_kl = precomputed_golden_entropy - np.sum(
          precomputed_golden_prob * precomputed_model_log_prob, axis=-1, dtype=np.float64
      )
      return np.asarray(np.maximum(raw_kl, 0.0), dtype=np.float64)

  if (
      precomputed_golden_prob is not None
      and precomputed_golden_log_prob is not None
      and precomputed_model_log_prob is not None
  ):
    golden_prob = precomputed_golden_prob
    golden_log_prob = precomputed_golden_log_prob
    model_log_prob = precomputed_model_log_prob
  else:
    ref_f64 = np.asarray(golden_logits_sliced, dtype=np.float64)
    mod_f64 = np.asarray(model_logits_sliced, dtype=np.float64)
    ref_shifted = ref_f64 - np.max(ref_f64, axis=-1, keepdims=True)
    mod_shifted = mod_f64 - np.max(mod_f64, axis=-1, keepdims=True)
    golden_log_prob = ref_shifted - np.log(np.sum(np.exp(ref_shifted), axis=-1, keepdims=True))
    model_log_prob = mod_shifted - np.log(np.sum(np.exp(mod_shifted), axis=-1, keepdims=True))
    golden_prob = np.exp(golden_log_prob)

  if clip_logits_epsilon is not None:
    eps_f64 = float(clip_logits_epsilon)
    model_prob = np.exp(model_log_prob)
    p_g = np.clip(golden_prob, a_min=eps_f64, a_max=None)
    p_t = np.clip(model_prob, a_min=eps_f64, a_max=None)
    p_g = p_g / np.sum(p_g, axis=-1, keepdims=True)
    p_t = p_t / np.sum(p_t, axis=-1, keepdims=True)
    raw_kl = np.sum(
        np.where(p_g > 0.0, p_g * (np.log(p_g) - np.log(p_t)) - p_g + p_t, 0.0),
        axis=-1,
        dtype=np.float64,
    )
    return np.asarray(np.maximum(raw_kl, 0.0), dtype=np.float64)
  num_t = golden_prob.shape[0]
  raw_kl = np.empty(num_t, dtype=np.float64)
  for t0 in range(0, num_t, 256):
    t1 = min(t0 + 256, num_t)
    p_chunk = golden_prob[t0:t1]
    raw_kl[t0:t1] = np.sum(
        np.where(p_chunk > 0.0, p_chunk * (golden_log_prob[t0:t1] - model_log_prob[t0:t1]), 0.0),
        axis=-1,
        dtype=np.float64,
    )
  return np.asarray(np.maximum(raw_kl, 0.0), dtype=np.float64)


def _compute_per_token_kl_legacy_fp32(golden_logits_sliced, model_logits_sliced, clip_logits_epsilon=None) -> np.ndarray:
  """Computes legacy float32 per-position D_KL(P_golden || P_model) for diagnostic dual-reporting."""
  ref_j = jnp.asarray(golden_logits_sliced, dtype=jnp.float32)
  mod_j = jnp.asarray(model_logits_sliced, dtype=jnp.float32)
  if clip_logits_epsilon is not None:
    golden_prob = jax.nn.softmax(ref_j, axis=-1)
    model_prob = jax.nn.softmax(mod_j, axis=-1)
    golden_prob = jnp.clip(golden_prob, min=clip_logits_epsilon)
    model_prob = jnp.clip(model_prob, min=clip_logits_epsilon)
    golden_prob = golden_prob / jnp.sum(golden_prob, axis=-1, keepdims=True)
    model_prob = model_prob / jnp.sum(model_prob, axis=-1, keepdims=True)
    return np.asarray(jnp.sum(jax.scipy.special.kl_div(golden_prob, model_prob), axis=-1), dtype=np.float64)
  golden_log_prob = jax.nn.log_softmax(ref_j, axis=-1)
  model_log_prob = jax.nn.log_softmax(mod_j, axis=-1)
  golden_prob = jnp.exp(golden_log_prob)
  return np.asarray(jnp.sum(golden_prob * (golden_log_prob - model_log_prob), axis=-1), dtype=np.float64)


def _validate_and_envelope_tie_flip_kl(
    kl_primary: np.ndarray,
    model_logits_sliced,
    golden_shape: tuple[int, ...],
    tie_flip_golden_logits,
    clip_logits_epsilon=None,
) -> tuple[float, int, bool]:
  """Computes min(KL_primary[t], KL_tie_flip[t]) envelope for single-token max_kl when tie_flip_logits is provided."""
  raw_max_kl = float(np.max(kl_primary))
  raw_argmax_pos = int(np.argmax(kl_primary))
  if tie_flip_golden_logits is None:
    return raw_max_kl, raw_argmax_pos, False

  if hasattr(tie_flip_golden_logits, "detach"):
    tie_np = tie_flip_golden_logits.detach().to(torch.float32).cpu().numpy()
  else:
    tie_np = np.asarray(tie_flip_golden_logits, dtype=np.float32)

  if tuple(tie_np.shape) != tuple(golden_shape):
    raise ValueError(f"tie_flip_logits shape {tuple(tie_np.shape)} must match golden_logits shape {tuple(golden_shape)}")
  if not bool(np.all(np.isfinite(tie_np))):
    raise ValueError("tie_flip_logits contains non-finite values (NaN or Inf)")

  vocab_slice = int(model_logits_sliced.shape[-1])
  tie_sliced = np.asarray(tie_np[..., :vocab_slice], dtype=np.float32)
  kl_tie_flip = _compute_per_token_kl(tie_sliced, model_logits_sliced, clip_logits_epsilon=clip_logits_epsilon)
  kl_envelope = np.minimum(kl_primary, kl_tie_flip)
  return float(np.max(kl_envelope)), int(np.argmax(kl_envelope)), True


def _compute_tie_flip_max_kl_legacy_fp32(
    kl_primary_legacy_fp32: np.ndarray,
    model_logits_sliced,
    tie_flip_golden_logits,
    clip_logits_epsilon=None,
) -> float:
  """Computes legacy float32 min(KL_primary[t], KL_tie_flip[t]) max_kl for diagnostic dual-reporting."""
  if tie_flip_golden_logits is None:
    return float(np.max(kl_primary_legacy_fp32))
  if hasattr(tie_flip_golden_logits, "detach"):
    tie_np = tie_flip_golden_logits.detach().to(torch.float32).cpu().numpy()
  else:
    tie_np = np.asarray(tie_flip_golden_logits, dtype=np.float32)
  vocab_slice = int(model_logits_sliced.shape[-1])
  tie_sliced = jnp.asarray(tie_np[..., :vocab_slice], dtype=jnp.float32)
  kl_tie_flip_legacy_fp32 = _compute_per_token_kl_legacy_fp32(
      tie_sliced, model_logits_sliced, clip_logits_epsilon=clip_logits_epsilon
  )
  return float(np.max(np.minimum(kl_primary_legacy_fp32, kl_tie_flip_legacy_fp32)))


# pylint: disable=too-many-positional-arguments,too-many-locals
def check_mathematical_parity(
    model_logits,
    golden_logits,
    atol_linf=None,
    min_cos_sim=0.999,
    max_kl=None,
    clip_logits_epsilon=None,
    tokens=None,
    tie_flip_golden_logits=None,
    include_top10_and_legacy_kl: bool = True,
    log_metrics: bool = True,
    precomputed_golden_prob=None,
    precomputed_golden_log_prob=None,
    precomputed_model_log_prob=None,
    precomputed_golden_entropy=None,
    precomputed_golden_std=None,
    precomputed_golden_norm=None,
    precomputed_golden_top1=None,
    precomputed_model_std=None,
    precomputed_model_norm=None,
    precomputed_model_top1=None,
    precomputed_model_shifted=None,
    precomputed_model_lse=None,
    precomputed_kl: tuple[float, float] | None = None,
):
  """Tier 1B: Logit Parity Metrics over a sequence."""
  if hasattr(model_logits, "detach"):
    model_logits = np.asarray(model_logits.detach().to(torch.float32).cpu().numpy())
  if hasattr(golden_logits, "detach"):
    golden_logits = np.asarray(golden_logits.detach().to(torch.float32).cpu().numpy())

  golden_raw_shape = tuple(golden_logits.shape)
  token_size = min(model_logits.shape[-1], golden_logits.shape[-1])
  model_logits_sliced = np.asarray(model_logits[..., :token_size], dtype=np.float32)
  golden_logits_sliced = np.asarray(golden_logits[..., :token_size], dtype=np.float32)

  model_std = precomputed_model_std if precomputed_model_std is not None else float(np.std(model_logits_sliced))
  golden_std = precomputed_golden_std if precomputed_golden_std is not None else float(np.std(golden_logits_sliced))

  assert model_std > 1e-4, (
      f"FATAL: MaxText logits have collapsed! Standard deviation is {model_std:.4e}. "
      "The forward pass is returning dead data."
  )
  assert golden_std > 1e-4, (
      f"FATAL: HF golden logits have collapsed! Standard deviation is {golden_std:.4e}. "
      "The HF forward pass is returning dead data."
  )

  # Metric 1: L-infinity Norm (Max Absolute Error) & L2 Norm
  diff = model_logits_sliced - golden_logits_sliced
  abs_diff = np.abs(diff)
  l_inf_norm = float(np.max(abs_diff))
  l2_norm = float(np.linalg.norm(diff))
  mean_abs_diff = float(np.mean(abs_diff))
  if log_metrics:
    max_abs_per_pos = np.asarray(np.max(abs_diff, axis=-1), dtype=np.float64)
    p95_abs_diff = float(np.percentile(max_abs_per_pos, 95))
    max_abs_diff_idx = tuple(int(i) for i in np.unravel_index(np.argmax(abs_diff), abs_diff.shape))
  else:
    p95_abs_diff = 0.0
    max_abs_diff_idx = (0, 0)

  # Metric 2: Cosine Similarity
  norm_model = (
      precomputed_model_norm if precomputed_model_norm is not None else np.linalg.norm(model_logits_sliced, axis=-1)
  )
  norm_golden = (
      precomputed_golden_norm if precomputed_golden_norm is not None else np.linalg.norm(golden_logits_sliced, axis=-1)
  )
  dot_product = np.sum(model_logits_sliced * golden_logits_sliced, axis=-1)
  cos_sim_per_token = np.asarray(dot_product / (norm_model * norm_golden + 1e-8), dtype=np.float64)

  mean_cos_sim = float(np.mean(cos_sim_per_token))
  min_cos_sim_val = float(np.min(cos_sim_per_token))
  argmin_cos_pos = int(np.argmin(cos_sim_per_token)) if log_metrics else 0

  # Metric 3: KL Divergence D_KL(P_golden || P_model) — mean/median/p95/p99 strictly on primary golden
  if precomputed_kl is not None and not include_top10_and_legacy_kl:
    max_kl_div = float(precomputed_kl[0])
    mean_kl_div = float(precomputed_kl[1])
    kl_div_per_token = np.array([max_kl_div], dtype=np.float64)
    raw_primary_max_kl = max_kl_div
    raw_primary_argmax_kl_pos = 0
    median_kl_div = mean_kl_div
    p95_kl_div = max_kl_div
    p99_kl_div = max_kl_div
    argmax_kl_pos = 0
    tie_flip_envelope_used = False
  else:
    kl_div_per_token = _compute_per_token_kl(
        golden_logits_sliced,
        model_logits_sliced,
        clip_logits_epsilon=clip_logits_epsilon,
        precomputed_golden_prob=precomputed_golden_prob,
        precomputed_golden_log_prob=precomputed_golden_log_prob,
        precomputed_model_log_prob=precomputed_model_log_prob,
        precomputed_golden_entropy=precomputed_golden_entropy,
        precomputed_model_shifted=precomputed_model_shifted,
        precomputed_model_lse=precomputed_model_lse,
    )
    mean_kl_div = float(np.mean(kl_div_per_token))
    raw_primary_max_kl = float(np.max(kl_div_per_token))
    raw_primary_argmax_kl_pos = int(np.argmax(kl_div_per_token))
    median_kl_div = float(np.median(kl_div_per_token))
    p95_kl_div = float(np.percentile(kl_div_per_token, 95))
    p99_kl_div = float(np.percentile(kl_div_per_token, 99))
    max_kl_div, argmax_kl_pos, tie_flip_envelope_used = _validate_and_envelope_tie_flip_kl(
        kl_div_per_token,
        model_logits_sliced,
        golden_raw_shape,
        tie_flip_golden_logits,
        clip_logits_epsilon=clip_logits_epsilon,
    )

  if include_top10_and_legacy_kl:
    kl_div_per_token_legacy_fp32 = _compute_per_token_kl_legacy_fp32(
        golden_logits_sliced, model_logits_sliced, clip_logits_epsilon=clip_logits_epsilon
    )
    mean_kl_div_legacy_fp32 = float(np.mean(kl_div_per_token_legacy_fp32))
    raw_primary_max_kl_legacy_fp32 = float(np.max(kl_div_per_token_legacy_fp32))
    median_kl_div_legacy_fp32 = float(np.median(kl_div_per_token_legacy_fp32))
    p95_kl_div_legacy_fp32 = float(np.percentile(kl_div_per_token_legacy_fp32, 95))
    p99_kl_div_legacy_fp32 = float(np.percentile(kl_div_per_token_legacy_fp32, 99))
    max_kl_div_legacy_fp32 = _compute_tie_flip_max_kl_legacy_fp32(
        kl_div_per_token_legacy_fp32,
        model_logits_sliced,
        tie_flip_golden_logits,
        clip_logits_epsilon=clip_logits_epsilon,
    )
  else:
    kl_div_per_token_legacy_fp32 = kl_div_per_token
    mean_kl_div_legacy_fp32 = mean_kl_div
    raw_primary_max_kl_legacy_fp32 = raw_primary_max_kl
    median_kl_div_legacy_fp32 = median_kl_div
    p95_kl_div_legacy_fp32 = p95_kl_div
    p99_kl_div_legacy_fp32 = p99_kl_div
    max_kl_div_legacy_fp32 = max_kl_div

  argmax_kl_token_id = int(tokens[argmax_kl_pos]) if tokens is not None and argmax_kl_pos < len(tokens) else -1

  # Metric 4: Top-1 and Top-10 Agreement Across All Positions
  model_top1 = (
      precomputed_model_top1
      if precomputed_model_top1 is not None
      else np.asarray(np.argmax(model_logits_sliced, axis=-1))
  )
  golden_top1 = (
      precomputed_golden_top1
      if precomputed_golden_top1 is not None
      else np.asarray(np.argmax(golden_logits_sliced, axis=-1))
  )
  top1_matches = int(np.sum(model_top1 == golden_top1))
  seq_len = int(model_logits_sliced.shape[0])
  top1_agreement = float(top1_matches / seq_len)

  if include_top10_and_legacy_kl:
    _, model_top10 = jax.lax.top_k(model_logits_sliced, 10)
    _, golden_top10 = jax.lax.top_k(golden_logits_sliced, 10)
    model_top10_np = np.asarray(model_top10)
    golden_top10_np = np.asarray(golden_top10)
    jaccards = []
    overlaps = []
    for pos_i in range(seq_len):
      s_m = set(model_top10_np[pos_i].tolist())
      s_g = set(golden_top10_np[pos_i].tolist())
      inter = len(s_m & s_g)
      union = len(s_m | s_g)
      overlaps.append(inter / 10.0)
      jaccards.append(inter / union if union else 1.0)
    top10_overlap_mean = float(np.mean(overlaps))
    top10_jaccard_mean = float(np.mean(jaccards))
    last_token_top10_matches = len(
        set(np.asarray(jax.lax.top_k(model_logits_sliced[-1], min(10, int(model_logits_sliced.shape[-1])))[1]).tolist())
        & set(
            np.asarray(jax.lax.top_k(golden_logits_sliced[-1], min(10, int(golden_logits_sliced.shape[-1])))[1]).tolist()
        )
    )
  else:
    top10_overlap_mean = 1.0
    top10_jaccard_mean = 1.0
    last_token_top10_matches = 10

  if log_metrics:
    max_logging.log("\n--- Tier 1B: Logit Parity Metrics ---")
    max_logging.log(f"L-infinity Norm (Max Abs Diff): {l_inf_norm:.4e} at index {max_abs_diff_idx}")
    if atol_linf is not None:
      max_logging.log(f"  Threshold check: {l_inf_norm:.4e} < {atol_linf}")

    max_logging.log(f"Cosine Similarity (Mean):       {mean_cos_sim:.6f}")
    max_logging.log(f"Cosine Similarity (Min token):  {min_cos_sim_val:.6f} at pos {argmin_cos_pos}")
    if min_cos_sim is not None:
      max_logging.log(f"  Threshold check (Mean): {mean_cos_sim:.6f} >= {min_cos_sim}")

    max_logging.log(
        f"KL Divergence (Max token):      {max_kl_div:.6e} at pos {argmax_kl_pos} (token_id={argmax_kl_token_id})"
    )
    if tie_flip_envelope_used:
      max_logging.log(
          f"KL Divergence (Raw primary max): {raw_primary_max_kl:.6e} at pos {raw_primary_argmax_kl_pos} "
          "(tie_flip_envelope_used=True)"
      )
    max_logging.log(f"KL Divergence (Mean):           {mean_kl_div:.6e}")
    max_logging.log(f"KL Divergence (Median / P95 / P99): {median_kl_div:.6e} / {p95_kl_div:.6e} / {p99_kl_div:.6e}")
    max_logging.log(
        f"Top-1 Agreement:                {top1_agreement:.6f} ({top1_matches}/{seq_len}), "
        f"Top-10 Overlap Mean: {top10_overlap_mean:.4f}, Top-10 Jaccard Mean: {top10_jaccard_mean:.4f}"
    )
    if max_kl is not None:
      max_logging.log(f"  Threshold check: {max_kl_div:.6e} <= {max_kl}")

  if atol_linf is not None:
    assert l_inf_norm <= atol_linf, f"L-infinity norm {l_inf_norm:.4e} exceeds threshold {atol_linf}"
  if min_cos_sim is not None:
    assert mean_cos_sim >= min_cos_sim, f"Mean Cosine Similarity {mean_cos_sim:.6f} below threshold {min_cos_sim}"
  if max_kl is not None:
    assert max_kl_div <= max_kl, f"Max KL Divergence {max_kl_div:.6e} exceeds threshold {max_kl}"

  if log_metrics:
    max_logging.log("Model parity metrics computed successfully")
  return {
      "seq_len": seq_len,
      "max_kl": max_kl_div,
      "max_kl_legacy_fp32": max_kl_div_legacy_fp32,
      "raw_primary_max_kl": raw_primary_max_kl,
      "raw_primary_max_kl_legacy_fp32": raw_primary_max_kl_legacy_fp32,
      "raw_primary_argmax_kl_pos": raw_primary_argmax_kl_pos,
      "tie_flip_envelope_used": tie_flip_envelope_used,
      "mean_kl": mean_kl_div,
      "mean_kl_legacy_fp32": mean_kl_div_legacy_fp32,
      "median_kl": median_kl_div,
      "median_kl_legacy_fp32": median_kl_div_legacy_fp32,
      "p95_kl": p95_kl_div,
      "p95_kl_legacy_fp32": p95_kl_div_legacy_fp32,
      "p99_kl": p99_kl_div,
      "p99_kl_legacy_fp32": p99_kl_div_legacy_fp32,
      "argmax_kl_pos": argmax_kl_pos,
      "argmax_kl_token_id": argmax_kl_token_id,
      "top1_agreement": top1_agreement,
      "top1_matches": top1_matches,
      "last_token_top10_matches": last_token_top10_matches,
      "top10_overlap_mean": top10_overlap_mean,
      "top10_jaccard_mean": top10_jaccard_mean,
      "mean_cos_sim": mean_cos_sim,
      "min_cos_sim": min_cos_sim_val,
      "argmin_cos_pos": argmin_cos_pos,
      "l_inf": l_inf_norm,
      "l2": l2_norm,
      "mean_abs_diff": mean_abs_diff,
      "p95_abs_diff": p95_abs_diff,
      "max_abs_diff_idx": list(max_abs_diff_idx),
  }


def _resolve_sample_name(golden_data_point: Mapping[str, Any], fallback_label: str) -> str:
  """Resolves sample name from `name` or `sample_id` (falling back to `fallback_label`)."""
  if "name" in golden_data_point and golden_data_point["name"] is not None:
    name_val = golden_data_point["name"]
    name_str = str(np.asarray(name_val).item()) if isinstance(name_val, np.ndarray) else str(name_val)
    if name_str:
      return name_str
  if "sample_id" in golden_data_point and golden_data_point["sample_id"] is not None:
    sid_val = golden_data_point["sample_id"]
    sid_str = str(np.asarray(sid_val).item()) if isinstance(sid_val, np.ndarray) else str(sid_val)
    if sid_str:
      return sid_str
  return fallback_label


def _has_ds41_mm_images(golden_data_point: Mapping[str, Any]) -> bool:
  """Returns True when `golden_data_point` carries DeepSeek-V4.1 `patches` or `encoder_images`."""
  return ("patches" in golden_data_point and golden_data_point["patches"] is not None) or (
      "encoder_images" in golden_data_point and golden_data_point["encoder_images"] is not None
  )


def _resolve_ds41_image_token_id(config: Any) -> int | None:
  """Returns configured `ds41_image_token_id` when present on `config`, else None."""
  if "get_keys" in dir(config):
    if "ds41_image_token_id" not in config.get_keys():
      return None
  elif "ds41_image_token_id" not in dir(config):
    return None
  raw_val = config.ds41_image_token_id
  return int(raw_val) if raw_val is not None else None


def _validate_sample_token_types(
    golden_data_point: Mapping[str, Any],
    *,
    sample_name: str,
    sample_tokens: Any,
    seq_len: int,
    start_index: int,
    token_size: int,
    is_mm_sample: bool,
    ds41_image_token_id: int | None,
) -> np.ndarray | None:
  """Validates release `token_types` (`{-1, 0, 1, 2, 3}`) and image-token alignment."""
  has_token_types = "token_types" in golden_data_point and golden_data_point["token_types"] is not None
  tt_full = None
  if has_token_types:
    raw_tt = np.asarray(golden_data_point["token_types"])
    if np.issubdtype(raw_tt.dtype, np.bool_) or not np.issubdtype(raw_tt.dtype, np.integer):
      raise ValueError(
          f"Sample {sample_name!r} token_types must have integer dtype with values in "
          f"{{-1, 0, 1, 2, 3}}, got dtype={raw_tt.dtype}"
      )
    raw_tt_arr = raw_tt.reshape(-1)
    if raw_tt_arr.shape[0] != seq_len:
      raise ValueError(f"Sample {sample_name!r} token_types length {raw_tt_arr.shape[0]} != tokens length {seq_len}")
    if not np.all((raw_tt_arr >= -1) & (raw_tt_arr <= 3)):
      raise ValueError(
          f"Sample {sample_name!r} token_types must contain only {{-1, 0, 1, 2, 3}}, "
          f"got {np.unique(raw_tt_arr).tolist()}"
      )
    tt_full = raw_tt_arr.astype(np.int32)
    if not is_mm_sample and np.any(tt_full >= 0):
      raise ValueError(
          f"Text-only sample {sample_name!r} has no images but declares image-span entries "
          f"(token_types >= 0): {np.unique(tt_full[tt_full >= 0]).tolist()}"
      )
  if is_mm_sample:
    if tt_full is None:
      raise ValueError(
          f"Multimodal sample {sample_name!r} has patches/encoder_images but is missing non-None token_types"
      )
    mm_mask = tt_full >= 0
    text_mask = tt_full == -1
    if not np.any(mm_mask):
      raise ValueError(f"Multimodal sample {sample_name!r} has images but no image-span entries (token_types >= 0)")
    if not np.any(text_mask) or not np.any(tt_full[start_index:token_size] == -1):
      raise ValueError(f"Multimodal sample {sample_name!r} has no evaluated text positions (token_types == -1)")
    if ds41_image_token_id is not None:
      tokens_1d = np.asarray(sample_tokens).reshape(-1)[:seq_len]
      expected_mm_mask = tokens_1d == int(ds41_image_token_id)
      if not np.array_equal(mm_mask, expected_mm_mask):
        mismatch_idx = np.flatnonzero(mm_mask != expected_mm_mask).tolist()
        raise ValueError(
            f"Multimodal sample {sample_name!r} token_types >= 0 mask disagrees with "
            f"tokens == ds41_image_token_id ({ds41_image_token_id}) at {len(mismatch_idx)} "
            f"positions (first 8: {mismatch_idx[:8]})"
        )
  return tt_full


def _record_from_npz(npz_file: Path, *, require_option_c: bool = False) -> dict[str, object]:
  """Loads a single `.npz` golden file (text-only or multimodal `M1..M8` / `L1..L4`)."""
  with np.load(npz_file, allow_pickle=False) as loaded:
    raw_logits = np.asarray(loaded["logits"], dtype=np.float32)
    squeeze_b1 = raw_logits.ndim == 3 and raw_logits.shape[0] == 1
    logits = raw_logits[0] if squeeze_b1 else raw_logits
    has_mm = ("patches" in loaded) or ("encoder_images" in loaded)
    sample_id = str(np.asarray(loaded["sample_id"]).item()) if "sample_id" in loaded else npz_file.stem
    prompt_str = str(np.asarray(loaded["prompt"]).item()) if "prompt" in loaded else sample_id
    diag_val = (
        bool(np.asarray(loaded["cos_sim_diagnostic_only"]).item())
        if "cos_sim_diagnostic_only" in loaded
        else (not has_mm and sample_id == "sample_6_verbatim_repeat_stress")
    )
    record: dict[str, object] = {
        "name": sample_id,
        "sample_id": sample_id,
        "prompt": prompt_str,
        "tokens": np.asarray(loaded["tokens"], dtype=np.int32).tolist(),
        "logits": logits,
        "cos_sim_diagnostic_only": diag_val,
    }
    if "tie_flip_logits" in loaded and loaded["tie_flip_logits"] is not None:
      raw_tie_flip = np.asarray(loaded["tie_flip_logits"], dtype=np.float32)
      if raw_tie_flip.shape != raw_logits.shape:
        raise ValueError(
            f"{npz_file.name}: tie_flip_logits shape {raw_tie_flip.shape} != logits shape {raw_logits.shape}"
        )
      if not np.all(np.isfinite(raw_tie_flip)):
        raise ValueError(f"{npz_file.name}: tie_flip_logits contains non-finite (NaN/Inf) entries")
      record["tie_flip_logits"] = raw_tie_flip[0] if squeeze_b1 else raw_tie_flip
    if "patches" in loaded:
      record["patches"] = np.asarray(loaded["patches"], dtype=np.float32)
    if "encoder_images" in loaded:
      record["encoder_images"] = np.asarray(loaded["encoder_images"], dtype=np.float32)
    if "image_masks" in loaded:
      record["image_masks"] = np.asarray(loaded["image_masks"])
    if "encoder_image_masks" in loaded:
      record["encoder_image_masks"] = np.asarray(loaded["encoder_image_masks"])
    if "image_mask" in loaded:
      record["image_mask"] = np.asarray(loaded["image_mask"])
    if "grid_hw" in loaded:
      record["grid_hw"] = np.asarray(loaded["grid_hw"], dtype=np.int32)
    if "grids" in loaded:
      record["grids"] = np.asarray(loaded["grids"], dtype=np.int32)
    if "token_types" in loaded:
      record["token_types"] = np.asarray(loaded["token_types"])
    if "segment_ids" in loaded:
      record["segment_ids"] = np.asarray(loaded["segment_ids"], dtype=np.int32)
    if "num_images" in loaded:
      record["num_images"] = int(np.asarray(loaded["num_images"]).item())
    eff_req_c = bool(require_option_c) or (has_mm and deepseek_v41_frozen_topk.is_mm_v3_golden_path(npz_file))
    record.update(
        deepseek_v41_frozen_topk.extract_and_validate_option_c_fields(
            loaded,
            sample_name=sample_id,
            expected_seq_len=len(record["tokens"]),
            is_multimodal=has_mm,
            require_option_c=eff_req_c,
        )
    )
    return record


def _normalize_golden_record(
    rec: Mapping[str, Any],
    fallback_label: str,
    *,
    require_option_c: bool = False,
) -> dict[str, object]:
  """Ensures both `name` and `sample_id` keys are populated on loaded golden records."""
  out: dict[str, object] = {**rec}
  resolved = _resolve_sample_name(out, fallback_label)
  if "name" not in out or out["name"] is None:
    out["name"] = resolved
  if "sample_id" not in out or out["sample_id"] is None:
    out["sample_id"] = resolved
  if "tie_flip_logits" in out and out["tie_flip_logits"] is not None:
    raw_logits = np.asarray(out["logits"], dtype=np.float32)
    raw_tie_flip = np.asarray(out["tie_flip_logits"], dtype=np.float32)
    if raw_tie_flip.shape != raw_logits.shape:
      raise ValueError(
          f"Sample {resolved!r}: tie_flip_logits shape {raw_tie_flip.shape} != logits shape {raw_logits.shape}"
      )
    if not np.all(np.isfinite(raw_tie_flip)):
      raise ValueError(f"Sample {resolved!r}: tie_flip_logits contains non-finite (NaN/Inf) entries")
    if raw_logits.ndim == 3 and raw_logits.shape[0] == 1:
      out["logits"] = raw_logits[0]
      out["tie_flip_logits"] = raw_tie_flip[0]
    else:
      out["tie_flip_logits"] = raw_tie_flip
  has_mm = ("patches" in out and out["patches"] is not None) or (
      "encoder_images" in out and out["encoder_images"] is not None
  )
  tok_len = len(np.asarray(out["tokens"]).reshape(-1))
  out.update(
      deepseek_v41_frozen_topk.extract_and_validate_option_c_fields(
          out,
          sample_name=resolved,
          expected_seq_len=tok_len,
          is_multimodal=has_mm,
          require_option_c=bool(require_option_c),
      )
  )
  return out


def load_golden_dataset(golden_logits_path: str, *, require_option_c: bool = False) -> list[dict[str, object]]:
  """Loads golden prefill samples from .jsonl, .pickle/.pkl, .npz, or a directory of .npz files."""
  p = Path(golden_logits_path)
  eff_req_c = bool(require_option_c) or deepseek_v41_frozen_topk.is_mm_v3_golden_path(golden_logits_path)
  if p.is_dir():
    npz_files = sorted(p.glob("*.npz"))
    if not npz_files:
      raise ValueError(f"No .npz files found in directory {golden_logits_path}")
    return [_record_from_npz(npz_file, require_option_c=eff_req_c) for npz_file in npz_files]
  if p.suffix == ".jsonl":
    max_logging.log("loading goldens from jsonl file")
    with jsonlines.open(p, "r") as f:
      return [_normalize_golden_record(item, f"sample_{idx}", require_option_c=eff_req_c) for idx, item in enumerate(f)]
  if p.suffix in (".pickle", ".pkl"):
    max_logging.log("loading goldens from pickle file")
    with open(p, "rb") as f:
      loaded_list = pickle.load(f)
    return [
        _normalize_golden_record(item, f"sample_{idx}", require_option_c=eff_req_c)
        for idx, item in enumerate(loaded_list)
    ]
  if p.suffix == ".npz":
    max_logging.log("loading golden from npz file")
    return [_record_from_npz(p, require_option_c=eff_req_c)]
  raise ValueError("golden_logits_path must end with .jsonl, .pickle, .pkl, .npz, or be a directory of .npz files")


def _extract_tie_flip_slice(
    golden_data_point: Mapping[str, Any],
    golden_logits: np.ndarray,
    start_index: int,
    token_size: int,
    min_vocab_size: int,
) -> np.ndarray | None:
  """Extracts, validates, and slices optional tie_flip_logits from a golden sample record."""
  if "tie_flip_logits" not in golden_data_point or golden_data_point["tie_flip_logits"] is None:
    return None
  tie_arr = np.asarray(golden_data_point["tie_flip_logits"], dtype=np.float32)
  if tie_arr.ndim == 3 and tie_arr.shape[0] == 1 and golden_logits.ndim == 2:
    tie_arr = tie_arr[0]
  if tuple(tie_arr.shape) != tuple(golden_logits.shape):
    raise ValueError(
        f"tie_flip_logits shape {tuple(tie_arr.shape)} must match golden_logits shape {tuple(golden_logits.shape)}"
    )
  if not bool(np.all(np.isfinite(tie_arr))):
    raise ValueError("tie_flip_logits contains non-finite values (NaN or Inf)")
  return tie_arr[start_index:token_size, :min_vocab_size]


def prepare_ds41_mm_batch_inputs(
    raw_patches: np.ndarray,
    global_batch_size: int,
    max_images_per_sample: int = 4,
    *,
    image_masks: np.ndarray | None = None,
    num_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Pads `[N_img, P, 588]` patches to `[B, max_images, P, 588]` (`float32`) + `[B, max_images]` (`int32`) masks."""
  patches_f32 = np.asarray(raw_patches, dtype=np.float32)
  if patches_f32.ndim != 3:
    raise ValueError(f"Expected 3D patches [N_img, P, D], got shape {patches_f32.shape}")
  n_img, num_patches, patch_dim = patches_f32.shape
  if n_img < 1 or n_img > max_images_per_sample:
    raise ValueError(
        f"Sample has invalid n_img={n_img}; expected 1 <= n_img <= max_images_per_sample={max_images_per_sample}"
    )
  padded = np.zeros((max_images_per_sample, num_patches, patch_dim), dtype=np.float32)
  padded[:n_img] = patches_f32
  mask_1d = np.zeros((max_images_per_sample,), dtype=np.int32)
  if image_masks is not None:
    raw_mask_arr = np.asarray(image_masks).reshape(-1)
    if raw_mask_arr.shape[0] < 1 or raw_mask_arr.shape[0] > max_images_per_sample:
      raise ValueError(
          f"image_masks length {raw_mask_arr.shape[0]} must satisfy "
          f"1 <= len <= max_images_per_sample={max_images_per_sample}"
      )
    if not np.all((raw_mask_arr == 0) | (raw_mask_arr == 1)):
      raise ValueError(f"image_masks must be strictly binary in {{0, 1}}, got {raw_mask_arr.tolist()}")
    mask_1d[: raw_mask_arr.shape[0]] = raw_mask_arr.astype(np.int32)
    if n_img < max_images_per_sample and np.any(mask_1d[n_img:] != 0):
      raise ValueError(f"image_masks {mask_1d.tolist()} has active entries in zero-padded image slots >= n_img={n_img}")
    mask_sum = int(np.sum(mask_1d))
    if mask_sum < 1:
      raise ValueError(f"image_masks {mask_1d.tolist()} must activate at least 1 image slot (sum={mask_sum})")
    if num_images is not None:
      num_valid = int(num_images)
      if num_valid < 1 or num_valid > n_img:
        raise ValueError(f"Invalid num_images={num_valid} for n_img={n_img}")
      if mask_sum != num_valid:
        raise ValueError(
            f"Sum of image validity mask {mask_sum} ({mask_1d.tolist()}) does not match num_images={num_valid}"
        )
  elif num_images is not None:
    num_valid = int(num_images)
    if num_valid < 1 or num_valid > max_images_per_sample or num_valid > n_img:
      raise ValueError(f"Invalid num_images={num_valid} for n_img={n_img}, max_images_per_sample={max_images_per_sample}")
    mask_1d[:num_valid] = 1
  else:
    mask_1d[:n_img] = 1
  encoder_images = np.broadcast_to(
      padded[None, ...], (global_batch_size, max_images_per_sample, num_patches, patch_dim)
  ).copy()
  encoder_image_masks = np.broadcast_to(mask_1d[None, ...], (global_batch_size, max_images_per_sample)).copy()
  return encoder_images, encoder_image_masks


def get_data(golden_data_point, config, max_images_per_sample: int = 4):
  """Get the golden data for the test indexed at golden_data_index."""
  max_logging.log(f"config.global_batch_size_to_train_on={config.global_batch_size_to_train_on}")
  encoder_image_masks = None
  if _has_ds41_mm_images(golden_data_point):
    if not config.use_multimodal:
      raise ValueError("Multimodal golden sample with 'patches'/'encoder_images' requires config.use_multimodal=True")
    raw_patches = (
        golden_data_point["patches"]
        if ("patches" in golden_data_point and golden_data_point["patches"] is not None)
        else golden_data_point["encoder_images"]
    )
    explicit_mask = None
    for mask_key in ("encoder_image_masks", "image_masks", "image_mask"):
      if mask_key in golden_data_point and golden_data_point[mask_key] is not None:
        explicit_mask = np.asarray(golden_data_point[mask_key])
        break
    num_images_val = (
        int(np.asarray(golden_data_point["num_images"]).item())
        if ("num_images" in golden_data_point and golden_data_point["num_images"] is not None)
        else None
    )
    pixel_values, encoder_image_masks = prepare_ds41_mm_batch_inputs(
        np.asarray(raw_patches, dtype=np.float32),
        int(config.global_batch_size_to_train_on),
        max_images_per_sample=max_images_per_sample,
        image_masks=explicit_mask,
        num_images=num_images_val,
    )
  elif config.use_multimodal and "pixel_values" in golden_data_point:
    pixel_values = np.asarray(golden_data_point["pixel_values"], dtype=np.float32)
    max_logging.log(f"pixel_values.shape = {pixel_values.shape}")
    model_prefix = config.model_name.split("-")[0]
    if model_prefix in ["gemma3", "gemma4"]:
      if pixel_values.ndim == 2:
        h, w = config.image_size_for_vit
        p = config.patch_size_for_vit
        c = pixel_values.shape[-1] // (p * p)
        pixel_values = np.reshape(pixel_values, (h // p, w // p, p, p, c))
        pixel_values = np.transpose(pixel_values, (0, 2, 1, 3, 4))
        pixel_values = np.reshape(pixel_values, (h, w, c))
      else:
        pixel_values = np.transpose(pixel_values, (1, 2, 0))
    elif model_prefix in ["llama4"]:
      pixel_values = pixel_values[None, :]
    pixel_values = np.stack([pixel_values for _ in range(config.global_batch_size_to_train_on)])
  else:
    pixel_values = None

  original_ids = np.asarray(golden_data_point["tokens"], dtype=np.int32)
  seq_len = len(original_ids)

  if seq_len > config.max_target_length:
    raise ValueError(
        f"Golden data sequence length ({seq_len}) is greater than max_target_length ({config.max_target_length})"
    )

  s = (config.global_batch_size_to_train_on, config.max_target_length)
  padded_ids = np.pad(original_ids, (0, config.max_target_length - seq_len), "constant", constant_values=0)
  ids = np.stack([padded_ids for _ in range(config.global_batch_size_to_train_on)])

  logits = np.asarray(golden_data_point["logits"], dtype=np.float32)
  if "formatted_prompt" in golden_data_point:
    prompt = str(golden_data_point["formatted_prompt"])
  elif "prompt" in golden_data_point:
    prompt = str(golden_data_point["prompt"])
  else:
    prompt = ""
  sample_name = _resolve_sample_name(golden_data_point, f"sample_{seq_len}")
  max_logging.log(
      f'sample="{sample_name}" seq_len={seq_len} raw ids (first 20)={original_ids[:20]}, logits.shape={logits.shape}'
  )
  if prompt:
    max_logging.log(f'prompt preview="{prompt[:120]}..."')

  decoder_segment_ids = np.zeros(s, dtype=np.int32)
  if "segment_ids" in golden_data_point and golden_data_point["segment_ids"] is not None:
    raw_seg = np.asarray(golden_data_point["segment_ids"])
    if np.issubdtype(raw_seg.dtype, np.bool_) or not np.issubdtype(raw_seg.dtype, np.integer):
      raise ValueError(f"segment_ids must have integer dtype, got {raw_seg.dtype}")
    seg_1d = raw_seg.reshape(-1).astype(np.int32)
    if seg_1d.shape[0] != seq_len:
      raise ValueError(f"segment_ids length {seg_1d.shape[0]} != tokens length {seq_len}")
    decoder_segment_ids[:, :seq_len] = seg_1d[None, :]
  else:
    decoder_segment_ids[:, :seq_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR
  decoder_positions = np.stack(
      [np.arange(config.max_target_length, dtype=np.int32) for _ in range(config.global_batch_size_to_train_on)]
  )
  return ids, decoder_segment_ids, decoder_positions, logits, seq_len, pixel_values, encoder_image_masks


def check_banked_sample_for_test(
    samples_dir: str | Path | None,
    sample_name: str,
    stage_name: str = "r3",
    gcs_results_path: str | None = None,
) -> tuple[str, dict[str, Any] | None, str]:
  """Helper to check banked sample from CLI flag string or Path."""
  local_samples_path = Path(samples_dir) if samples_dir else None
  if local_samples_path is None:
    return banking.STATUS_MISSING, None, "samples_dir not provided"
  gcs_samples_prefix = f"{gcs_results_path.rstrip('/')}/samples" if gcs_results_path else None
  return banking.check_banked_sample(sample_name, stage_name, local_samples_path, gcs_samples_prefix=gcs_samples_prefix)


# pylint: disable=redefined-outer-name
def _main_impl(config, test_args, *, preloaded_model: Any = None):
  """Test the Whole Model of model_name."""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  hf_token = config.hf_access_token
  tokenizer_load_paths = []
  if test_args.hf_model_path:
    tokenizer_load_paths.append(test_args.hf_model_path)
  if config.tokenizer_path:
    tokenizer_load_paths.append(config.tokenizer_path)

  tokenizer = None
  for path in tokenizer_load_paths:
    try:
      max_logging.log(f"Loading tokenizer from {path}.")
      tokenizer = AutoTokenizer.from_pretrained(path, token=hf_token, trust_remote_code=test_args.trust_remote_code)
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Failed to load tokenizer from {path}: {e}")

  max_images_per_sample = int(test_args.max_images_per_sample)
  mm_golden_dir = str(test_args.mm_golden_dir)
  output_json_path = str(test_args.output_json)
  require_frozen_topk_gate = bool(
      getattr(test_args, "require_frozen_topk_gate", False)
      or deepseek_v41_frozen_topk.is_mm_v3_golden_path(mm_golden_dir)
      or deepseek_v41_frozen_topk.is_mm_v3_golden_path(test_args.golden_logits_path)
  )
  router_ulp_multiplier = float(
      getattr(
          test_args,
          "router_agreement_ulp_multiplier",
          deepseek_v41_frozen_topk.DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER,
      )
  )

  if not test_args.run_hf_model:
    max_logging.log("Initializing MaxText model")
    if preloaded_model is not None:
      model = preloaded_model
    else:
      model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      if config.lora.enable_lora:
        model = lora_utils.apply_lora_to_model(model, mesh, config)
        if config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(model, config)
    state = None
    nnx_graphdef, nnx_state = nnx.split(model)

    @jax.jit
    def eval_step_nnx(
        step_state,
        ids,
        decoder_positions,
        decoder_segment_ids,
        images=None,
        image_masks=None,
    ):
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        nnx_model = nnx.merge(nnx_graphdef, step_state)
        return nnx_model(
            decoder_input_tokens=ids,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            encoder_images=images,
            encoder_image_masks=image_masks,
            enable_dropout=False,
        )

    golden_data: list[dict[str, object]] = []
    if test_args.golden_logits_path == "" and mm_golden_dir == "":
      input_golden_data_path = os.path.join(
          MAXTEXT_TEST_ASSETS_ROOT,
          "golden_logits",
          f"golden_data_{config.model_name}.jsonl",
      )
      golden_data.extend(load_golden_dataset(input_golden_data_path))
    else:
      if test_args.golden_logits_path != "":
        golden_data.extend(load_golden_dataset(test_args.golden_logits_path, require_option_c=require_frozen_topk_gate))
      if mm_golden_dir != "":
        golden_data.extend(load_golden_dataset(mm_golden_dir, require_option_c=require_frozen_topk_gate))

    max_logging.log(f"loaded {len(golden_data)} golden data points")
    all_data_to_save = []
    sample_reports = []
    failed_samples = []

    for golden_data_index, golden_data_point in enumerate(golden_data):
      sample_name = _resolve_sample_name(golden_data_point, f"sample_{golden_data_index}")
      samples_dir = str(getattr(test_args, "samples_dir", ""))
      stage_name = str(getattr(test_args, "stage_name", "r3"))
      gcs_results_path = str(getattr(test_args, "gcs_results_path", ""))
      local_samples_path = Path(samples_dir) if samples_dir else None
      gcs_samples_prefix = f"{gcs_results_path.rstrip('/')}/samples" if gcs_results_path else None

      banked_sample_status = banking.STATUS_MISSING
      banked_sample_data = None
      if local_samples_path is not None:
        banked_sample_status, banked_sample_data, sample_reason = check_banked_sample_for_test(
            samples_dir, sample_name, stage_name=stage_name, gcs_results_path=gcs_results_path
        )
        max_logging.log(
            f"[preemption_banking] Sample {sample_name} check: status={banked_sample_status} reason={sample_reason}"
        )

      if banked_sample_status == banking.STATUS_FAILED:
        attempt = banked_sample_data.get("attempt_id") if banked_sample_data else "unknown"
        pod = banked_sample_data.get("pod") if banked_sample_data else "unknown"
        max_logging.log(
            f"[preemption_banking] Sample {sample_name} recorded FAILURE on attempt {attempt} "
            f"(pod {pod}): recording failed sample without re-running."
        )
        rec = dict(banked_sample_data.get("record", banked_sample_data)) if banked_sample_data else {}
        rec.setdefault("name", sample_name)
        rec["passed"] = False
        is_mm = bool(_has_ds41_mm_images(golden_data_point) or str(sample_name).startswith("M"))
        rec.setdefault("is_multimodal", is_mm)
        rec.setdefault("image_token_count", 0)
        rec.setdefault("max_kl", float(banked_sample_data.get("max_kl", 0.0)) if banked_sample_data else 0.0)
        rec.setdefault("mean_kl", float(banked_sample_data.get("mean_kl", 0.0)) if banked_sample_data else 0.0)
        sample_reports.append(rec)
        failed_samples.append(rec)
        continue

      if banked_sample_status in (banking.STATUS_PASSED, banking.STATUS_DEFERRED) and banked_sample_data is not None:
        max_logging.log(
            f"[preemption_banking] Skipping sample {sample_name}: already banked as {banked_sample_status} "
            f"(attempt={banked_sample_data.get('attempt_id')}, pod={banked_sample_data.get('pod')})."
        )
        rec = banked_sample_data.get("record", banked_sample_data)
        sample_reports.append(rec)
        continue
      max_logging.log(f"\n--- Comparing masked prefill forward pass for index {golden_data_index} ({sample_name}) ---")
      ids, decoder_segment_ids, decoder_positions, golden_logits, seq_len, images, image_masks = get_data(
          golden_data_point, config, max_images_per_sample=max_images_per_sample
      )
      token_size = int(test_args.token_size) if test_args.token_size else seq_len
      start_index = 1 if test_args.skip_first_token else 0

      # Validate any optional tie_flip_logits upfront before running model forward passes.
      if "tie_flip_logits" in golden_data_point and golden_data_point["tie_flip_logits"] is not None:
        _ = _extract_tie_flip_slice(
            golden_data_point,
            golden_logits,
            start_index,
            token_size,
            int(golden_logits.shape[-1]),
        )

      has_images = _has_ds41_mm_images(golden_data_point)
      is_multimodal_sample = has_images
      tt_full = _validate_sample_token_types(
          golden_data_point,
          sample_name=sample_name,
          sample_tokens=ids[0, :seq_len],
          seq_len=seq_len,
          start_index=start_index,
          token_size=token_size,
          is_mm_sample=is_multimodal_sample,
          ds41_image_token_id=_resolve_ds41_image_token_id(config),
      )

      cos_sim_diagnostic_only = (
          bool(golden_data_point["cos_sim_diagnostic_only"]) if "cos_sim_diagnostic_only" in golden_data_point else False
      )
      if is_multimodal_sample and cos_sim_diagnostic_only:
        raise ValueError(
            "cos_sim_diagnostic_only=True is forbidden for multimodal samples (M1..M8); "
            "only text-only sample_6_verbatim_repeat_stress may enable it."
        )

      max_logging.log("maxtext forward pass (run 1)")
      any_option_c_extra = any(
          k in golden_data_point and golden_data_point[k] is not None
          for k in deepseek_v41_frozen_topk.OPTION_C_REQUIRED_ROUTING_KEYS[1:]
      )
      use_option_c_gate = bool(
          is_multimodal_sample
          and state is None
          and deepseek_v41_frozen_topk.has_option_c_routing_fields(golden_data_point)
      )
      if is_multimodal_sample and (require_frozen_topk_gate or any_option_c_extra) and not use_option_c_gate:
        missing_opt_c = [
            k
            for k in deepseek_v41_frozen_topk.OPTION_C_REQUIRED_ROUTING_KEYS
            if k not in golden_data_point or golden_data_point[k] is None
        ]
        raise ValueError(
            f"Multimodal sample {sample_name!r}: Option C FrozenTopK gate requires all 7 routing arrays "
            f"{deepseek_v41_frozen_topk.OPTION_C_REQUIRED_ROUTING_KEYS}; missing {missing_opt_c}."
        )
      natural_raw_logits_slice_np: np.ndarray | None = None
      router_agreement: dict[str, Any] | None = None
      unforced_router_agreement_summary: dict[str, Any] | None = None
      if use_option_c_gate:
        missing_idx_fields = [
            k
            for k in ("ref_indexer_layer_ids", "ref_indexer_topk", "ref_indexer_margins", "ref_indexer_boundary_ulp")
            if k not in golden_data_point or golden_data_point[k] is None
        ]
        if missing_idx_fields:
          raise ValueError(
              f"Multimodal sample {sample_name!r}: Option C FrozenTopK gate requires all 4 indexer routing fields; "
              f"missing {missing_idx_fields}"
          )
        raw_logits_run1, raw_logits_run2, forced_captured = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
            model,
            config,
            mesh,
            golden_data_point["ref_topk_experts"],
            sample_name=sample_name,
            ids=ids,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            images=images,
            image_masks=image_masks,
            seq_len=seq_len,
            raw_indexer_topk=golden_data_point["ref_indexer_topk"],
            raw_indexer_layer_ids=golden_data_point["ref_indexer_layer_ids"],
            return_rerun=True,
            return_captured=True,
        )
        if router_ulp_multiplier > 0 and (
            forced_captured is None
            or not hasattr(forced_captured, "tpu_moe_topk")
            or forced_captured.tpu_moe_topk is None
            or not hasattr(forced_captured, "tpu_indexer_topk")
            or forced_captured.tpu_indexer_topk is None
        ):
          raise ValueError(
              f"Multimodal sample {sample_name!r}: Option C FrozenTopK router-agreement diagnostic requires "
              "captured pre-override tpu_moe_topk and tpu_indexer_topk arrays when "
              f"router_agreement_ulp_multiplier={router_ulp_multiplier} > 0."
          )
        full_train_logits, rows_per_batch_shard = _gather_and_free_logits(raw_logits_run1, seq_len, config.vocab_size)
        del raw_logits_run1
        rerun_train_logits, _ = _gather_and_free_logits(raw_logits_run2, seq_len, config.vocab_size)
        del raw_logits_run2
        nat_res = deepseek_v41_frozen_topk.run_natural_routing_forward_with_trace_nnx(
            model,
            config,
            mesh,
            sample_name=sample_name,
            ids=ids,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            images=images,
            image_masks=image_masks,
        )
        nat_full, _ = _gather_and_free_logits(nat_res.logits, seq_len, config.vocab_size)
        nat_min_vocab = min(int(nat_full.shape[-1]), int(golden_logits.shape[-1]))
        natural_raw_logits_slice_np = np.asarray(nat_full[0, start_index:token_size, :nat_min_vocab])
        del nat_full
        unforced_ra = deepseek_v41_frozen_topk.check_router_agreement(
            sample_name=sample_name,
            seq_len=seq_len,
            token_types=tt_full,
            tpu_moe_topk=nat_res.tpu_moe_topk,
            ref_topk_experts=golden_data_point["ref_topk_experts"],
            ref_gate_margins=golden_data_point["ref_gate_margins"],
            ref_gate_boundary_ulp=golden_data_point["ref_gate_boundary_ulp"],
            tpu_indexer_topk=nat_res.tpu_indexer_topk,
            ref_indexer_layer_ids=golden_data_point["ref_indexer_layer_ids"],
            ref_indexer_topk=golden_data_point["ref_indexer_topk"],
            ref_indexer_margins=golden_data_point["ref_indexer_margins"],
            ref_indexer_boundary_ulp=golden_data_point["ref_indexer_boundary_ulp"],
            ulp_multiplier=router_ulp_multiplier,
            raise_on_failure=False,
        )
        unforced_router_agreement_summary = {
            "passed": bool(unforced_ra["passed"]),
            "diagnostic_only": True,
            "ulp_multiplier": float(unforced_ra["ulp_multiplier"]),
            "high_margin_moe_failure_count": int(unforced_ra["high_margin_moe_failure_count"]),
            "low_margin_moe_flip_count": int(unforced_ra["low_margin_moe_flip_count"]),
            "high_margin_indexer_failure_count": int(unforced_ra["high_margin_indexer_failure_count"]),
            "low_margin_indexer_flip_count": int(unforced_ra["low_margin_indexer_flip_count"]),
            "moe_router": dict(unforced_ra["moe_router"]),
            "sparse_attention_indexer": dict(unforced_ra["sparse_attention_indexer"]),
        }
        router_agreement = deepseek_v41_frozen_topk.check_router_agreement(
            sample_name=sample_name,
            seq_len=seq_len,
            token_types=tt_full,
            tpu_moe_topk=forced_captured.tpu_moe_topk,
            ref_topk_experts=golden_data_point["ref_topk_experts"],
            ref_gate_margins=golden_data_point["ref_gate_margins"],
            ref_gate_boundary_ulp=golden_data_point["ref_gate_boundary_ulp"],
            tpu_indexer_topk=forced_captured.tpu_indexer_topk,
            ref_indexer_layer_ids=golden_data_point["ref_indexer_layer_ids"],
            ref_indexer_topk=golden_data_point["ref_indexer_topk"],
            ref_indexer_margins=golden_data_point["ref_indexer_margins"],
            ref_indexer_boundary_ulp=golden_data_point["ref_indexer_boundary_ulp"],
            ulp_multiplier=router_ulp_multiplier,
            raise_on_failure=False,
        )
        router_agreement["diagnostic_only"] = True
        router_log_line = f"DS41_MM_ROUTER_AGREEMENT: {json.dumps({'sample': sample_name, **router_agreement})}"
        max_logging.log(router_log_line)
        if jax.process_index() == 0:
          print(router_log_line, flush=True)
      else:
        if state is None:
          raw_logits_run1 = eval_step_nnx(nnx_state, ids, decoder_positions, decoder_segment_ids, images, image_masks)
        else:
          raw_logits_run1 = model.apply(
              state.params,
              ids,
              decoder_positions,
              decoder_segment_ids,
              encoder_images=images,
              encoder_image_masks=image_masks,
              enable_dropout=False,
              rngs={"aqt": init_rng},
          )
        full_train_logits, rows_per_batch_shard = _gather_and_free_logits(raw_logits_run1, seq_len, config.vocab_size)
        del raw_logits_run1

        max_logging.log("maxtext forward pass (run 2 for rerun determinism)")
        if state is None:
          raw_logits_run2 = eval_step_nnx(nnx_state, ids, decoder_positions, decoder_segment_ids, images, image_masks)
        else:
          raw_logits_run2 = model.apply(
              state.params,
              ids,
              decoder_positions,
              decoder_segment_ids,
              encoder_images=images,
              encoder_image_masks=image_masks,
              enable_dropout=False,
              rngs={"aqt": init_rng},
          )
        rerun_train_logits, _ = _gather_and_free_logits(raw_logits_run2, seq_len, config.vocab_size)
        del raw_logits_run2

      min_vocab_size = min(full_train_logits.shape[-1], golden_logits.shape[-1])
      seg_slice = np.asarray(decoder_segment_ids[0, start_index:token_size], dtype=np.int32)
      valid_mask = seg_slice > 0
      if not bool(np.any(valid_mask)):
        raise ValueError(f"Sample {sample_name!r} has zero non-padding positions in slice [{start_index}:{token_size}]")

      train_logits_slice = full_train_logits[0, start_index:token_size, :min_vocab_size]
      golden_logits_slice = golden_logits[start_index:token_size, :min_vocab_size]
      tie_flip_logits_slice = _extract_tie_flip_slice(
          golden_data_point, golden_logits, start_index, token_size, min_vocab_size
      )
      eval_tokens = ids[0, start_index:token_size]
      train_slice_np = np.asarray(train_logits_slice)
      golden_slice_np = np.asarray(golden_logits_slice)
      eval_tokens_np = np.asarray(eval_tokens)

      bucket_gates = get_length_bucket_thresholds(int(np.sum(valid_mask)))
      effective_max_kl = (
          test_args.max_kl_div
          if (test_args.max_kl_div is not None and not test_args.enforce_length_bucket_gates)
          else float(bucket_gates["max_kl"])
      )
      effective_min_cos = (
          test_args.min_cos_sim
          if (test_args.min_cos_sim is not None and not test_args.enforce_length_bucket_gates)
          else float(bucket_gates["mean_cos_sim"])
      )

      preloaded_branches = None
      if getattr(test_args, "allow_reference_tie_branches", False):
        preloaded_branches = load_and_validate_tie_branches(
            getattr(test_args, "tie_branch_dir", ""),
            sample_name=sample_name,
            max_ulp=getattr(test_args, "tie_branch_max_ulp", 8.0),
        )

      run1_slice_3d = full_train_logits[:, start_index:token_size, :]
      run2_slice_3d = rerun_train_logits[:, start_index:token_size, :]
      run1_valid_3d = run1_slice_3d[:, valid_mask, :]
      run2_valid_3d = run2_slice_3d[:, valid_mask, :]
      batch_det = check_batch_slot_determinism(
          run1_valid_3d,
          int(run1_valid_3d.shape[1]),
          min_vocab_size,
          rows_per_batch_shard=rows_per_batch_shard,
          rerun_train_logits=run2_valid_3d,
          allow_reference_tie_branches=getattr(test_args, "allow_reference_tie_branches", False),
          tie_branch_dir=getattr(test_args, "tie_branch_dir", ""),
          tie_branch_max_ulp=getattr(test_args, "tie_branch_max_ulp", 8.0),
          sample_name=sample_name,
          primary_golden_logits=golden_slice_np[valid_mask] if golden_slice_np is not None else None,
          bucket_gates=bucket_gates,
          effective_max_kl=effective_max_kl,
          effective_min_cos=effective_min_cos,
          tokens=eval_tokens_np[valid_mask] if eval_tokens_np is not None else None,
          clip_logits_epsilon=test_args.clip_logits_epsilon,
          atol_linf=None if is_multimodal_sample else test_args.atol,
          cos_sim_diagnostic_only=cos_sim_diagnostic_only,
          start_index=start_index,
          token_size=token_size,
          pos_mask=valid_mask,
          preloaded_branches=preloaded_branches,
      )

      primary_copy_idx = 0
      pri_copies = (
          batch_det.get("primary_passed_copies", [])
          if (getattr(test_args, "allow_reference_tie_branches", False) and not is_multimodal_sample)
          else []
      )
      if 0 in pri_copies:
        primary_copy_idx = 0
      elif pri_copies:
        primary_copy_idx = pri_copies[0]
        msg = (
            f"[TIE_BRANCH_PRIMARY_COPY] sample={sample_name}: Copy 0 took tie branch; "
            f"evaluating sample primary parity on copy {primary_copy_idx} "
            f"(primary_passed_copies={pri_copies})"
        )
        max_logging.log(msg)
        if jax.process_index() == 0:
          print(msg, flush=True)
        train_slice_np = np.asarray(full_train_logits[primary_copy_idx, start_index:token_size, :min_vocab_size])

      text_batch_det: dict[str, Any] | None = None
      image_batch_det: dict[str, Any] | None = None
      text_mask: np.ndarray | None = None
      image_mask_pos: np.ndarray | None = None
      if is_multimodal_sample:
        assert tt_full is not None
        raw_tt = tt_full[start_index:token_size]
        text_mask = valid_mask & (raw_tt == -1)
        image_mask_pos = valid_mask & (raw_tt != -1)
        if not bool(np.any(text_mask)):
          raise ValueError(f"Multimodal sample {sample_name!r} has zero valid text_positions (token_types == -1)")
        if not bool(np.any(image_mask_pos)):
          raise ValueError(f"Multimodal sample {sample_name!r} has zero valid image_positions (token_types != -1)")
        run1_text_3d = run1_slice_3d[:, text_mask, :]
        run2_text_3d = run2_slice_3d[:, text_mask, :]
        text_batch_det = check_batch_slot_determinism(
            run1_text_3d,
            int(run1_text_3d.shape[1]),
            min_vocab_size,
            rows_per_batch_shard=rows_per_batch_shard,
            rerun_train_logits=run2_text_3d,
            allow_reference_tie_branches=getattr(test_args, "allow_reference_tie_branches", False),
            tie_branch_dir=getattr(test_args, "tie_branch_dir", ""),
            tie_branch_max_ulp=getattr(test_args, "tie_branch_max_ulp", 8.0),
            sample_name=sample_name,
            primary_golden_logits=golden_slice_np[text_mask] if golden_slice_np is not None else None,
            bucket_gates=bucket_gates,
            effective_max_kl=effective_max_kl,
            effective_min_cos=effective_min_cos,
            tokens=eval_tokens_np[text_mask] if eval_tokens_np is not None else None,
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            atol_linf=test_args.atol,
            cos_sim_diagnostic_only=cos_sim_diagnostic_only,
            start_index=start_index,
            token_size=token_size,
            pos_mask=text_mask,
            preloaded_branches=preloaded_branches,
        )
        mm_pri_copies = (
            text_batch_det.get("primary_passed_copies", [])
            if getattr(test_args, "allow_reference_tie_branches", False)
            else []
        )
        if 0 in mm_pri_copies:
          primary_copy_idx = 0
        elif mm_pri_copies:
          primary_copy_idx = mm_pri_copies[0]
          msg = (
              f"[TIE_BRANCH_PRIMARY_COPY] multimodal sample={sample_name}: Copy 0 took tie branch; "
              f"evaluating sample primary parity on copy {primary_copy_idx} "
              f"(primary_passed_copies={mm_pri_copies})"
          )
          max_logging.log(msg)
          if jax.process_index() == 0:
            print(msg, flush=True)
          train_slice_np = np.asarray(full_train_logits[primary_copy_idx, start_index:token_size, :min_vocab_size])
        run1_img_3d = run1_slice_3d[:, image_mask_pos, :]
        run2_img_3d = run2_slice_3d[:, image_mask_pos, :]
        image_batch_det = check_batch_slot_determinism(
            run1_img_3d,
            int(run1_img_3d.shape[1]),
            min_vocab_size,
            rows_per_batch_shard=rows_per_batch_shard,
            rerun_train_logits=run2_img_3d,
        )
      del rerun_train_logits

      # Last-token top-10 diagnostic (evaluated on gated text_positions for multimodal samples)
      last_scope_mask = text_mask if (is_multimodal_sample and text_mask is not None) else valid_mask
      if HAS_TORCH and torch is not None:
        mt_last_token_logits = torch.tensor(train_slice_np[last_scope_mask][-1:, :])
        hf_last_token_logits = torch.tensor(golden_slice_np[last_scope_mask][-1:, :])
        tokens_maxtext = get_top_k_tokens_scores(mt_last_token_logits, tokenizer, k=10, description="MaxText model")
        tokens_hf = get_top_k_tokens_scores(hf_last_token_logits, tokenizer, k=10, description="Reference model")
        top10_last_metrics = compare_top_tokens(converted_tokens=tokens_maxtext, golden_tokens=tokens_hf)
      else:
        top10_last_metrics = {"overlap_count": 10, "jaccard_similarity": 1.0, "rank_agreement_percentage": 100.0}

      def _attach_batch_det(parity_dict: dict[str, Any], bdet: dict[str, Any]) -> dict[str, Any]:
        return {
            **parity_dict,
            "rerun_max_abs_diff": bdet["rerun_max_abs_diff"],
            "rerun_bitwise_identical": bdet["rerun_bitwise_identical"],
            "rerun_argmax_identical": bdet["rerun_argmax_identical"],
            "rows_per_batch_shard": bdet["rows_per_batch_shard"],
            "intra_shard_checked": bdet["intra_shard_checked"],
            "intra_shard_max_diff": bdet["intra_shard_max_diff"],
            "intra_shard_mean_diff": bdet["intra_shard_mean_diff"],
            "cross_shard_max_diff": bdet["cross_shard_max_diff"],
            "cross_shard_mean_diff": bdet["cross_shard_mean_diff"],
            "max_batch_diff": bdet["max_batch_diff"],
            "mean_batch_diff": bdet["mean_batch_diff"],
            "max_logit_magnitude": bdet["max_logit_magnitude"],
            "batch_det_tol": bdet["batch_det_tol"],
            "allow_reference_tie_branches": bool(bdet.get("allow_reference_tie_branches", False)),
            "copy0_primary_pass": bool(bdet.get("copy0_primary_pass", True)),
            "any_copy_primary_pass": bool(bdet.get("any_copy_primary_pass", True)),
            "all_copies_valid": bool(bdet.get("all_copies_valid", True)),
            "tie_branch_batch_gate_passed": bool(bdet.get("tie_branch_batch_gate_passed", True)),
            "primary_matched_copies": bdet.get("primary_matched_copies", []),
            "tie_branch_matched_copies": bdet.get("tie_branch_matched_copies", []),
            "unmatched_copies": bdet.get("unmatched_copies", []),
            "per_copy_tie_branch_report": bdet.get("per_copy_tie_branch_report", []),
            "tie_branches_evaluated_count": int(bdet.get("tie_branches_evaluated_count", 0)),
        }

      tie_flip_all = (
          None
          if (is_multimodal_sample or getattr(test_args, "allow_reference_tie_branches", False))
          else (np.asarray(tie_flip_logits_slice)[valid_mask] if tie_flip_logits_slice is not None else None)
      )
      parity_metrics = check_mathematical_parity(
          model_logits=train_slice_np[valid_mask],
          golden_logits=golden_slice_np[valid_mask],
          atol_linf=None if is_multimodal_sample else test_args.atol,
          min_cos_sim=None,
          max_kl=None,
          clip_logits_epsilon=test_args.clip_logits_epsilon,
          tokens=eval_tokens_np[valid_mask],
          tie_flip_golden_logits=tie_flip_all,
      )
      all_positions_metrics = {
          **_attach_batch_det(parity_metrics, batch_det),
          "last_token_top10_overlap": top10_last_metrics["overlap_count"],
          "last_token_top10_jaccard": float(top10_last_metrics["jaccard_similarity"]),
          "last_token_rank_agreement": top10_last_metrics["rank_agreement_percentage"],
      }

      gates = evaluate_sample_gates(
          parity_metrics=parity_metrics,
          batch_det=batch_det,
          bucket_gates=bucket_gates,
          effective_max_kl=effective_max_kl,
          effective_min_cos=effective_min_cos,
          cos_sim_diagnostic_only=cos_sim_diagnostic_only,
          is_multimodal=is_multimodal_sample,
      )

      if tt_full is not None:
        tt_slice = tt_full[start_index:token_size]
        image_token_count = int(np.sum(valid_mask & (tt_slice != -1)))
        text_token_count = int(np.sum(valid_mask & (tt_slice == -1)))
      else:
        image_token_count = 0
        text_token_count = int(np.sum(valid_mask))

      text_positions_metrics: dict[str, Any] | None = None
      text_positions_gates: dict[str, bool] | None = None
      image_positions_metrics: dict[str, Any] | None = None
      frozen_topk_diagnostic: dict[str, Any] | None = None
      natural_routing_metrics: dict[str, Any] | None = None
      failure_reasons: list[str] = []

      if is_multimodal_sample:
        assert text_mask is not None
        assert image_mask_pos is not None
        assert text_batch_det is not None
        assert image_batch_det is not None
        tie_flip_logits_text = (
            None
            if getattr(test_args, "allow_reference_tie_branches", False)
            else (np.asarray(tie_flip_logits_slice)[text_mask] if tie_flip_logits_slice is not None else None)
        )
        text_parity = check_mathematical_parity(
            model_logits=train_slice_np[text_mask],
            golden_logits=golden_slice_np[text_mask],
            atol_linf=test_args.atol,
            min_cos_sim=None,
            max_kl=None,
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tokens=eval_tokens_np[text_mask],
            tie_flip_golden_logits=tie_flip_logits_text,
        )
        text_positions_metrics = _attach_batch_det(text_parity, text_batch_det)
        image_parity = check_mathematical_parity(
            model_logits=train_slice_np[image_mask_pos],
            golden_logits=golden_slice_np[image_mask_pos],
            atol_linf=None,
            min_cos_sim=None,
            max_kl=None,
            clip_logits_epsilon=test_args.clip_logits_epsilon,
            tokens=eval_tokens_np[image_mask_pos],
            tie_flip_golden_logits=None,
        )
        image_positions_metrics = _attach_batch_det(image_parity, image_batch_det)
        text_positions_gates = evaluate_sample_gates(
            parity_metrics=text_positions_metrics,
            batch_det=text_batch_det,
            bucket_gates=bucket_gates,
            effective_max_kl=effective_max_kl,
            effective_min_cos=effective_min_cos,
            cos_sim_diagnostic_only=cos_sim_diagnostic_only,
            is_multimodal=True,
        )
        sample_passed = bool(all(text_positions_gates.values()))
        failure_reasons = [f"text_positions.{k}=False" for k, v in text_positions_gates.items() if not v]

        if use_option_c_gate and natural_raw_logits_slice_np is not None:
          nat_text_metrics = check_mathematical_parity(
              model_logits=natural_raw_logits_slice_np[text_mask],
              golden_logits=golden_slice_np[text_mask],
              atol_linf=None,
              min_cos_sim=None,
              max_kl=None,
              clip_logits_epsilon=test_args.clip_logits_epsilon,
              tokens=eval_tokens_np[text_mask],
              tie_flip_golden_logits=tie_flip_logits_text,
          )
          nat_image_metrics = check_mathematical_parity(
              model_logits=natural_raw_logits_slice_np[image_mask_pos],
              golden_logits=golden_slice_np[image_mask_pos],
              atol_linf=None,
              min_cos_sim=None,
              max_kl=None,
              clip_logits_epsilon=test_args.clip_logits_epsilon,
              tokens=eval_tokens_np[image_mask_pos],
              tie_flip_golden_logits=None,
          )
          nat_all_metrics = check_mathematical_parity(
              model_logits=natural_raw_logits_slice_np[valid_mask],
              golden_logits=golden_slice_np[valid_mask],
              atol_linf=None,
              min_cos_sim=None,
              max_kl=None,
              clip_logits_epsilon=test_args.clip_logits_epsilon,
              tokens=eval_tokens_np[valid_mask],
              tie_flip_golden_logits=None,
          )
          natural_routing_metrics = {
              "text_positions_metrics": nat_text_metrics,
              "image_positions_metrics": nat_image_metrics,
              "all_positions_metrics": nat_all_metrics,
          }
          frozen_topk_diagnostic = {
              "ran": True,
              "mode": "option_c_primary_gate",
              "text_positions_metrics": text_positions_metrics,
              "image_positions_metrics": image_positions_metrics,
              "all_positions_metrics": all_positions_metrics,
              "natural_routing_metrics": natural_routing_metrics,
              "router_agreement": router_agreement,
              "unforced_router_agreement_summary": unforced_router_agreement_summary,
          }
          if router_ulp_multiplier > 0 and (router_agreement is None or unforced_router_agreement_summary is None):
            raise ValueError(
                f"Multimodal sample {sample_name!r}: Option C router_agreement diagnostic must be computed "
                f"when router_agreement_ulp_multiplier={router_ulp_multiplier} > 0."
            )
          diag_line = (
              f"DS41_MM_FROZEN_TOPK_DIAGNOSTIC: sample={sample_name} "
              f"unforced_text_max_kl={float(nat_text_metrics['max_kl']):.8g} -> "
              f"forced_text_max_kl={float(text_positions_metrics['max_kl']):.8g}, "
              f"unforced_text_mean_kl={float(nat_text_metrics['mean_kl']):.8g} -> "
              f"forced_text_mean_kl={float(text_positions_metrics['mean_kl']):.8g} "
              f"unforced_router_agreement_summary={json.dumps(unforced_router_agreement_summary)}"
          )
          max_logging.log(diag_line)
          if jax.process_index() == 0:
            print(diag_line, flush=True)
          # User ruling (2026-11-01 19:50 PST, Option A): strict >=8-ULP router and indexer top-k set equality
          # is gated in the per-layer teacher-forced v5p harness (reference h_in at every layer). In 40-layer E2E
          # FrozenTopK runs where continuous residual activations drift across depth, forced-routing text logit
          # gates (KL, cos_sim, top1, top10) remain strict PASS/FAIL gates while E2E router/indexer agreement is
          # computed, logged, and persisted in output_summary_json as a diagnostic (never silently dropped).
          if router_agreement is not None and not bool(router_agreement["passed"]):
            diag_reasons = "; ".join(deepseek_v41_frozen_topk.format_router_agreement_failure_reasons(router_agreement))
            router_diag_line = f"[DIAGNOSTIC] sample={sample_name} E2E router_agreement: {diag_reasons}"
            max_logging.log(router_diag_line)
            if jax.process_index() == 0:
              print(router_diag_line, flush=True)

        scope_log_payload = {
            "sample": sample_name,
            "passed": sample_passed,
            "text_positions_metrics": text_positions_metrics,
            "image_positions_metrics": image_positions_metrics,
            "all_positions_metrics": all_positions_metrics,
            "natural_routing_metrics": natural_routing_metrics,
            "router_agreement": router_agreement,
        }
        scope_log_line = f"DS41_MM_PREFILL_SCOPE_METRICS: {json.dumps(scope_log_payload)}"
        max_logging.log(scope_log_line)
        if jax.process_index() == 0:
          print(scope_log_line, flush=True)

        if not use_option_c_gate:
          has_ref_topk = "ref_topk_experts" in golden_data_point and golden_data_point["ref_topk_experts"] is not None
          if not sample_passed and has_ref_topk and state is None:
            raw_diag_logits = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
                model,
                config,
                mesh,
                golden_data_point["ref_topk_experts"],
                sample_name=sample_name,
                ids=ids,
                decoder_positions=decoder_positions,
                decoder_segment_ids=decoder_segment_ids,
                images=images,
                image_masks=image_masks,
                seq_len=seq_len,
            )
            diag_full_logits, _ = _gather_and_free_logits(raw_diag_logits, seq_len, config.vocab_size)
            del raw_diag_logits
            diag_slice_np = np.asarray(diag_full_logits[0, start_index:token_size, :min_vocab_size])
            del diag_full_logits
            diag_text_metrics = check_mathematical_parity(
                model_logits=diag_slice_np[text_mask],
                golden_logits=golden_slice_np[text_mask],
                atol_linf=None,
                min_cos_sim=None,
                max_kl=None,
                clip_logits_epsilon=test_args.clip_logits_epsilon,
                tokens=eval_tokens_np[text_mask],
                tie_flip_golden_logits=tie_flip_logits_text,
            )
            diag_image_metrics = check_mathematical_parity(
                model_logits=diag_slice_np[image_mask_pos],
                golden_logits=golden_slice_np[image_mask_pos],
                atol_linf=None,
                min_cos_sim=None,
                max_kl=None,
                clip_logits_epsilon=test_args.clip_logits_epsilon,
                tokens=eval_tokens_np[image_mask_pos],
                tie_flip_golden_logits=None,
            )
            diag_all_metrics = check_mathematical_parity(
                model_logits=diag_slice_np[valid_mask],
                golden_logits=golden_slice_np[valid_mask],
                atol_linf=None,
                min_cos_sim=None,
                max_kl=None,
                clip_logits_epsilon=test_args.clip_logits_epsilon,
                tokens=eval_tokens_np[valid_mask],
                tie_flip_golden_logits=None,
            )
            frozen_topk_diagnostic = {
                "ran": True,
                "text_positions_metrics": diag_text_metrics,
                "image_positions_metrics": diag_image_metrics,
                "all_positions_metrics": diag_all_metrics,
            }
            diag_line = (
                f"DS41_MM_FROZEN_TOPK_DIAGNOSTIC: sample={sample_name} "
                f"unforced_text_max_kl={float(text_positions_metrics['max_kl']):.8g} -> "
                f"forced_text_max_kl={float(diag_text_metrics['max_kl']):.8g}, "
                f"unforced_text_mean_kl={float(text_positions_metrics['mean_kl']):.8g} -> "
                f"forced_text_mean_kl={float(diag_text_metrics['mean_kl']):.8g}"
            )
            max_logging.log(diag_line)
            if jax.process_index() == 0:
              print(diag_line, flush=True)
          else:
            frozen_topk_diagnostic = {
                "ran": False,
                "reason": "text_positions_passed" if sample_passed else "missing_ref_topk_experts",
            }
      else:
        sample_passed = bool(all(gates.values()))
        failure_reasons = [f"all_positions.{k}=False" for k, v in gates.items() if not v]

      gated_scope = text_positions_metrics if is_multimodal_sample else parity_metrics
      assert gated_scope is not None
      gated_bdet = text_batch_det if is_multimodal_sample else batch_det
      assert gated_bdet is not None

      sample_record = {
          "index": golden_data_index,
          "name": sample_name,
          "is_multimodal": is_multimodal_sample,
          "image_token_count": image_token_count,
          "text_token_count": text_token_count,
          "seq_len": parity_metrics["seq_len"],
          "bucket": bucket_gates["bucket"],
          "cos_sim_diagnostic_only": cos_sim_diagnostic_only,
          "max_kl": gated_scope["max_kl"],
          "max_kl_legacy_fp32": gated_scope["max_kl_legacy_fp32"],
          "mean_kl": gated_scope["mean_kl"],
          "mean_kl_legacy_fp32": gated_scope["mean_kl_legacy_fp32"],
          "p95_kl": gated_scope["p95_kl"],
          "p95_kl_legacy_fp32": gated_scope["p95_kl_legacy_fp32"],
          "p99_kl": gated_scope["p99_kl"],
          "p99_kl_legacy_fp32": gated_scope["p99_kl_legacy_fp32"],
          "last_token_top10_matches": int(gated_scope["last_token_top10_matches"]),
          "thresholds": {
              "max_kl": effective_max_kl,
              "mean_kl": bucket_gates["mean_kl"],
              "top1_agreement": bucket_gates["top1_agreement"],
              "p95_kl": bucket_gates["p95_kl"],
              "p99_kl": bucket_gates["p99_kl"],
              "mean_cos_sim": effective_min_cos,
              "mean_batch_diff": gated_bdet["mean_diff_threshold"],
              "rerun_bitwise_identical": True,
              "batch_det_tol": gated_bdet["batch_det_tol"],
          },
          "allow_reference_tie_branches": bool(gated_bdet.get("allow_reference_tie_branches", False)),
          "copy0_primary_pass": bool(gated_bdet.get("copy0_primary_pass", True)),
          "any_copy_primary_pass": bool(gated_bdet.get("any_copy_primary_pass", True)),
          "all_copies_valid": bool(gated_bdet.get("all_copies_valid", True)),
          "tie_branch_batch_gate_passed": bool(gated_bdet.get("tie_branch_batch_gate_passed", True)),
          "primary_matched_copies": gated_bdet.get("primary_matched_copies", []),
          "tie_branch_matched_copies": gated_bdet.get("tie_branch_matched_copies", []),
          "unmatched_copies": gated_bdet.get("unmatched_copies", []),
          "per_copy_tie_branch_report": gated_bdet.get("per_copy_tie_branch_report", []),
          "tie_branches_evaluated_count": int(gated_bdet.get("tie_branches_evaluated_count", 0)),
          "metrics": all_positions_metrics,
          "all_positions_metrics": all_positions_metrics,
          "gates": gates,
          "text_positions_metrics": text_positions_metrics,
          "text_positions_gates": text_positions_gates,
          "image_positions_metrics": image_positions_metrics,
          "frozen_topk_diagnostic": frozen_topk_diagnostic,
          "natural_routing_metrics": natural_routing_metrics,
          "router_agreement": router_agreement,
          "unforced_router_agreement_summary": unforced_router_agreement_summary,
          "failure_reasons": failure_reasons,
          "passed": sample_passed,
      }
      if jax.process_index() == 0 and local_samples_path is not None:
        status_to_write = banking.STATUS_PASSED if sample_passed else banking.STATUS_FAILED
        banking.write_banked_sample(
            sample_id=sample_name,
            stage_name=stage_name,
            status=status_to_write,
            local_samples_dir=local_samples_path,
            metrics=sample_record,
            gcs_samples_prefix=gcs_samples_prefix,
        )

      sample_reports.append(sample_record)
      max_logging.log(f"MASKED_PREFILL_SAMPLE_METRICS: {json.dumps(sample_record)}")
      if not sample_passed:
        failed_samples.append(sample_record)
        if local_samples_path is not None:
          max_logging.log(
              f"Sample {sample_name} FAILED verification: {failure_reasons}. " "Failed sample JSON banked to GCS."
          )

      if jax.process_index() == 0 and test_args.output_logits_path:
        gd = golden_data[golden_data_index]
        if "formatted_prompt" in gd:
          saved_prompt = gd["formatted_prompt"]
        elif "prompt" in gd:
          saved_prompt = gd["prompt"]
        else:
          saved_prompt = ""
        data_to_save = {
            "name": sample_name,
            "prompt": saved_prompt,
            "tokens": ids[0, :seq_len].tolist(),
            "report": sample_record,
        }
        all_data_to_save.append(data_to_save)
      if getattr(test_args, "tie_branch_dir", "") and str(test_args.tie_branch_dir).startswith("gs://"):
        cache_dir = Path("/tmp/cached_tie_branches")
        if cache_dir.exists():
          s_prefix = f"s{sample_name.split('_')[1]}" if sample_name.startswith("sample_") else sample_name
          for p in cache_dir.glob("*.npz"):
            if p.name.startswith(f"{s_prefix}_") or p.name.startswith(f"{sample_name}_"):
              try:
                p.unlink()
              except OSError:
                pass
      del full_train_logits

    text_reports = [r for r in sample_reports if not r.get("is_multimodal", False)]
    mm_reports = [r for r in sample_reports if r.get("is_multimodal", False)]
    all_samples_passed = len(failed_samples) == 0
    sample_prov = banking.assemble_stage_provenance(sample_reports)
    prov_table = banking.format_provenance_table(sample_prov)
    max_logging.log(f"\n{prov_table}\n")
    if jax.process_index() == 0:
      print(f"\n{prov_table}\n", flush=True)

    summary = {
        "passed": all_samples_passed,
        "sample_provenance": sample_prov,
        "all_passed": all_samples_passed,
        "num_samples": len(sample_reports),
        "num_passed": sum(1 for r in sample_reports if r["passed"]),
        "max_kl": max(float(r.get("max_kl", 0.0)) for r in sample_reports) if sample_reports else 0.0,
        "max_kl_legacy_fp32": (
            max(float(r.get("max_kl_legacy_fp32", r.get("max_kl", 0.0))) for r in sample_reports)
            if sample_reports
            else 0.0
        ),
        "mean_kl": (float(np.mean([float(r.get("mean_kl", 0.0)) for r in sample_reports])) if sample_reports else 0.0),
        "mean_kl_legacy_fp32": (
            float(np.mean([float(r.get("mean_kl_legacy_fp32", r.get("mean_kl", 0.0))) for r in sample_reports]))
            if sample_reports
            else 0.0
        ),
        "p95_kl": max(float(r.get("p95_kl", 0.0)) for r in sample_reports) if sample_reports else 0.0,
        "p95_kl_legacy_fp32": (
            max(float(r.get("p95_kl_legacy_fp32", r.get("p95_kl", 0.0))) for r in sample_reports)
            if sample_reports
            else 0.0
        ),
        "p99_kl": max(float(r.get("p99_kl", 0.0)) for r in sample_reports) if sample_reports else 0.0,
        "p99_kl_legacy_fp32": (
            max(float(r.get("p99_kl_legacy_fp32", r.get("p99_kl", 0.0))) for r in sample_reports)
            if sample_reports
            else 0.0
        ),
        "max_kl_div": max(float(r.get("max_kl", 0.0)) for r in sample_reports) if sample_reports else 0.0,
        "max_kl_div_legacy_fp32": (
            max(float(r.get("max_kl_legacy_fp32", r.get("max_kl", 0.0))) for r in sample_reports)
            if sample_reports
            else 0.0
        ),
        "mean_kl_div": (
            float(np.mean([float(r.get("mean_kl", 0.0)) for r in sample_reports])) if sample_reports else 0.0
        ),
        "mean_kl_div_legacy_fp32": (
            float(np.mean([float(r.get("mean_kl_legacy_fp32", r.get("mean_kl", 0.0))) for r in sample_reports]))
            if sample_reports
            else 0.0
        ),
        "text_only_summary": {
            "num_samples": len(text_reports),
            "num_passed": sum(1 for r in text_reports if r["passed"]),
            "all_passed": all(r["passed"] for r in text_reports),
            "sample_names": [r["name"] for r in text_reports],
        },
        "multimodal_summary": {
            "num_samples": len(mm_reports),
            "num_passed": sum(1 for r in mm_reports if r["passed"]),
            "all_passed": all(r["passed"] for r in mm_reports),
            "total_image_tokens": sum(int(r.get("image_token_count", 0)) for r in mm_reports),
            "sample_names": [r["name"] for r in mm_reports],
        },
        "samples": sample_reports,
    }
    max_logging.log(f"MASKED_PREFILL_SUMMARY: {json.dumps(summary)}")

    if jax.process_index() == 0 and output_json_path:
      out_dir = os.path.dirname(output_json_path)
      if out_dir:
        os.makedirs(out_dir, exist_ok=True)
      with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if jax.process_index() == 0 and test_args.output_logits_path and all_data_to_save:
      output_dir = os.path.dirname(test_args.output_logits_path)
      if output_dir:
        os.makedirs(output_dir, exist_ok=True)
      with jsonlines.open(test_args.output_logits_path, "w") as f:
        for item in all_data_to_save:
          f.write(item)
      max_logging.log(f"Saved {len(all_data_to_save)} sample reports to {test_args.output_logits_path}")

    assert not failed_samples, (
        f"{len(failed_samples)}/{len(sample_reports)} samples failed masked-prefill gates: "
        f"{[r['name'] for r in failed_samples]}"
    )
    return sample_reports

  else:
    if test_args.hf_model_path == "":
      raise ValueError("run_hf_model requires hf_model_path")

    hf_token = config.hf_access_token
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }

    dtype_key = jnp.dtype(config.dtype).name.lower()
    torch_dtype = dtype_mapping[dtype_key] if dtype_key in dtype_mapping else torch.bfloat16
    max_logging.log(f"Loading HF/reference model with dtype: {torch_dtype} (derived from config.dtype: {config.dtype})")

    reference_model = (
        test_args.reference_model if test_args.reference_model else _detect_reference_model(test_args.hf_model_path)
    )
    if reference_model:
      hf_model = _load_reference_model(reference_model, test_args.hf_model_path, torch_dtype, config, tokenizer)
    else:
      hf_model = AutoModelForCausalLM.from_pretrained(
          test_args.hf_model_path,
          torch_dtype=torch_dtype,
          token=hf_token,
          trust_remote_code=test_args.trust_remote_code,
      )
    hf_lora_path = config.hf_lora_adapter_path
    if hf_lora_path:
      max_logging.log(f"Loading HF PEFT LoRA adapter from {hf_lora_path}")
      try:
        from peft import PeftModel  # pylint: disable=import-outside-toplevel
      except ImportError as exc:
        raise ImportError("peft library is required to load HF LoRA adapter. Run `pip install peft`.") from exc
      hf_model = PeftModel.from_pretrained(hf_model, hf_lora_path)

    pad_token_prefixes = ["llama3.1", "mixtral"]
    if tokenizer is not None and any(config.model_name.startswith(prefix) for prefix in pad_token_prefixes):
      tokenizer.pad_token = tokenizer.eos_token

    maxtext_model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    if config.lora.enable_lora:
      maxtext_model = lora_utils.apply_lora_to_model(maxtext_model, mesh, config)
      if config.lora.lora_restore_path:
        lora_utils.restore_lora_from_path(maxtext_model, config)
    maxtext_state = None

    target_seq_len = int(test_args.token_size) if test_args.token_size else 512
    base_tokens = tokenizer.encode(GOLDEN_BATCH_TEXT, add_special_tokens=False)
    repeat_factor = (target_seq_len // len(base_tokens)) + 2
    golden_text_expanded = (GOLDEN_BATCH_TEXT + "\n\n") * repeat_factor

    inputs = tokenizer(
        golden_text_expanded,
        return_tensors="pt",
        padding="max_length",
        max_length=target_seq_len,
        truncation=True,
    )

    actual_seq_len = inputs["input_ids"].shape[1]
    if actual_seq_len < target_seq_len:
      raise RuntimeError(f"Tokenizer produced only {actual_seq_len} tokens; expected {target_seq_len}.")
    max_logging.log(f"\n--- Running Zero-Generation Verification on {actual_seq_len} tokens ---")

    mt_ids_raw = jnp.asarray(inputs["input_ids"], dtype=jnp.int32)
    hf_attention_mask = jnp.asarray(inputs["attention_mask"], dtype=jnp.int32)
    pad_len = config.max_target_length - actual_seq_len
    if pad_len < 0:
      raise ValueError(f"actual_seq_len ({actual_seq_len}) exceeds max_target_length ({config.max_target_length})")
    mt_ids_padded = jnp.pad(mt_ids_raw, ((0, 0), (0, pad_len)), constant_values=0)
    attention_mask_padded = jnp.pad(hf_attention_mask, ((0, 0), (0, pad_len)), constant_values=0)

    padding_side = str(tokenizer.padding_side)
    real_token_count = int(jnp.sum(hf_attention_mask))
    if padding_side == "left":
      real_start = actual_seq_len - real_token_count
      real_end = actual_seq_len
    else:
      real_start = 0
      real_end = real_token_count
    last_real_idx = real_end - 1

    if mt_ids_padded.shape[0] != config.global_batch_size_to_train_on:
      tile_factor = config.global_batch_size_to_train_on // mt_ids_padded.shape[0]
      mt_ids = jnp.tile(mt_ids_padded, (tile_factor, 1))
      attention_mask_full = jnp.tile(attention_mask_padded, (tile_factor, 1))
    else:
      mt_ids = mt_ids_padded
      attention_mask_full = attention_mask_padded

    mt_decoder_segment_ids = attention_mask_full * DECODING_ACTIVE_SEQUENCE_INDICATOR
    mt_decoder_positions = jnp.broadcast_to(
        jnp.arange(config.max_target_length, dtype=jnp.int32),
        (config.global_batch_size_to_train_on, config.max_target_length),
    )

    with torch.no_grad():
      hf_logits_torch = hf_model(**inputs).logits.detach().to(torch.float32).cpu()

    if maxtext_state is None:
      nnx_graphdef, nnx_state = nnx.split(maxtext_model)

      @jax.jit
      def eval_step_nnx(step_state, ids, decoder_positions, decoder_segment_ids):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          nnx_model = nnx.merge(nnx_graphdef, step_state)
          return nnx_model(
              decoder_input_tokens=ids,
              decoder_positions=decoder_positions,
              decoder_segment_ids=decoder_segment_ids,
              enable_dropout=False,
          )

      mt_raw_run1 = eval_step_nnx(nnx_state, mt_ids, mt_decoder_positions, mt_decoder_segment_ids)
      mt_logits_jax_sliced, rows_per_batch_shard = _gather_and_free_logits(mt_raw_run1, actual_seq_len, config.vocab_size)
      del mt_raw_run1
      mt_raw_run2 = eval_step_nnx(nnx_state, mt_ids, mt_decoder_positions, mt_decoder_segment_ids)
      mt_logits_rerun, _ = _gather_and_free_logits(mt_raw_run2, actual_seq_len, config.vocab_size)
      del mt_raw_run2
    else:
      mt_raw_run1 = maxtext_model.apply(
          maxtext_state.params,
          mt_ids,
          mt_decoder_positions,
          mt_decoder_segment_ids,
          enable_dropout=False,
          rngs={"aqt": init_rng},
      )
      mt_logits_jax_sliced, rows_per_batch_shard = _gather_and_free_logits(mt_raw_run1, actual_seq_len, config.vocab_size)
      del mt_raw_run1
      mt_raw_run2 = maxtext_model.apply(
          maxtext_state.params,
          mt_ids,
          mt_decoder_positions,
          mt_decoder_segment_ids,
          enable_dropout=False,
          rngs={"aqt": init_rng},
      )
      mt_logits_rerun, _ = _gather_and_free_logits(mt_raw_run2, actual_seq_len, config.vocab_size)
      del mt_raw_run2

    mt_logits_torch = torch.from_numpy(mt_logits_jax_sliced)

    batch_det = check_batch_slot_determinism(
        mt_logits_jax_sliced,
        actual_seq_len,
        min(mt_logits_jax_sliced.shape[-1], hf_logits_torch.shape[-1]),
        rows_per_batch_shard=rows_per_batch_shard,
        rerun_train_logits=mt_logits_rerun,
    )
    del mt_logits_rerun
    assert batch_det["passed"], f"Batch-slot determinism failed: {batch_det}"

    hf_last_token_logits = hf_logits_torch[:, last_real_idx, :]
    mt_last_token_logits = mt_logits_torch[:, last_real_idx, :]
    tokens_maxtext = get_top_k_tokens_scores(mt_last_token_logits, tokenizer, k=10, description="MaxText model")
    tokens_hf = get_top_k_tokens_scores(hf_last_token_logits, tokenizer, k=10, description="HF model")
    compare_top_tokens(converted_tokens=tokens_maxtext, golden_tokens=tokens_hf)

    start_index = real_start + (1 if test_args.skip_first_token else 0)
    end_index = real_end
    check_mathematical_parity(
        model_logits=mt_logits_torch[0, start_index:end_index],
        golden_logits=hf_logits_torch[0, start_index:end_index],
        atol_linf=test_args.atol,
        min_cos_sim=test_args.min_cos_sim,
        max_kl=test_args.max_kl_div,
        clip_logits_epsilon=test_args.clip_logits_epsilon,
    )
    return None


def main(config, test_args, *, preloaded_model: Any = None):
  """Test the Whole Model of model_name."""
  try:
    return _main_impl(config, test_args, preloaded_model=preloaded_model)
  except DETERMINISTIC_CODE_ERRORS as exc:
    cli_err_msg = f"[DETERMINISTIC_CODE_ERROR] Unhandled {type(exc).__name__}: {exc}"
    max_logging.log(cli_err_msg)
    print(cli_err_msg, file=sys.stderr)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(EXIT_DETERMINISTIC_GATE_FAIL)


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  parser = argparse.ArgumentParser()
  parser.add_argument("--atol", type=float, required=False, default=None)
  parser.add_argument("--min_cos_sim", type=float, required=False, default=0.999)
  parser.add_argument("--token_size", type=int, required=False)
  parser.add_argument("--max_kl_div", type=float, required=False, default=0.08)
  parser.add_argument("--golden_logits_path", type=str, required=False, default="")
  parser.add_argument("--mm_golden_dir", type=str, required=False, default="")
  parser.add_argument("--max_images_per_sample", type=int, required=False, default=4)
  parser.add_argument("--output_json", type=str, required=False, default="")
  parser.add_argument("--samples_dir", type=str, required=False, default="")
  parser.add_argument("--stage_name", type=str, required=False, default="r3")
  parser.add_argument("--gcs_results_path", type=str, required=False, default="")
  parser.add_argument("--hf_model_path", type=str, required=False, default="")
  parser.add_argument("--run_hf_model", type=str2bool, required=False, default=False)
  parser.add_argument("--reference_model", type=str, required=False, default="")
  parser.add_argument("--output_logits_path", type=str, required=False, default="")
  parser.add_argument("--gcs_output_logits_path", type=str, required=False, default="")
  parser.add_argument("--clip_logits_epsilon", type=float, required=False, default=None)
  parser.add_argument("--enforce_length_bucket_gates", type=str2bool, required=False, default=True)
  parser.add_argument(
      "--skip_first_token",
      action="store_true",
      required=False,
      default=False,
      help="Skip the first token during comparison to ignore BOS/init mismatches.",
  )
  parser.add_argument(
      "--ckpt_type",
      type=str,
      required=False,
      default="linen",
      choices=["linen", "nnx"],
      help="Checkpoint format to load: 'linen' (default) or 'nnx'.",
  )
  parser.add_argument(
      "--trust_remote_code", type=str2bool, required=False, default=True, help="from_pretrained: trust_remote_code"
  )
  parser.add_argument(
      "--allow_reference_tie_branches",
      type=str2bool,
      required=False,
      default=False,
      help="Allow B>1 batch copies to match validated <=8 ULP reference tie branches (Option A-prime).",
  )
  parser.add_argument(
      "--tie_branch_dir",
      type=str,
      required=False,
      default="",
      help="Directory or gs:// path holding tie_branches_manifest.json and alternate branch .npz files.",
  )
  parser.add_argument(
      "--tie_branch_max_ulp",
      type=float,
      required=False,
      default=8.0,
      help="Maximum allowed gap in ULPs for an alternate reference tie branch.",
  )
  parser.add_argument(
      "--require_frozen_topk_gate",
      type=str2bool,
      required=False,
      default=False,
      help="Require Option C FrozenTopK gate + router-agreement fields for multimodal samples.",
  )
  parser.add_argument(
      "--router_agreement_ulp_multiplier",
      type=float,
      required=False,
      default=deepseek_v41_frozen_topk.DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER,
      help="ULP multiplier k for Option C high-margin router agreement check.",
  )

  test_args, remaining_args = parser.parse_known_args()
  model_args = [sys.argv[0]] + remaining_args

  cfg = pyconfig.initialize_pydantic(model_args)
  assert (
      test_args.atol is not None or test_args.max_kl_div is not None or test_args.enforce_length_bucket_gates
  ), "At least one of --atol, --max_kl_div, or --enforce_length_bucket_gates must be specified."

  if cfg.use_multimodal:
    assert not test_args.run_hf_model, (
        "Multimodal does not support running hf model on-the-fly, please generate hf golden logits "
        "using tests/assets/logits_generation/generate_hf_golden_logits.py"
    )
  try:
    main(cfg, test_args)
  except AssertionError as e:
    if "failed masked-prefill gates" in str(e):
      max_logging.log(f"Deterministic gate failure: {e}; exiting with code 42")
      sys.exit(42)
    raise
  except DETERMINISTIC_CODE_ERRORS as exc:
    cli_err_msg = f"[DETERMINISTIC_CODE_ERROR] Unhandled {type(exc).__name__}: {exc}"
    max_logging.log(cli_err_msg)
    print(cli_err_msg, file=sys.stderr)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(42)
