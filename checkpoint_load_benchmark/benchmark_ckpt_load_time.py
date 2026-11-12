"""Unified MaxText Checkpoint Load-Time Benchmark Suite (`to_maxtext` vs. `safetensors_dynamic`).

Compares end-to-end and warm checkpoint loading latency, sub-phase breakdowns, and peak Host RAM /
TPU HBM usage between MaxText's offline Orbax conversion pipeline (`to_maxtext`) and on-the-fly
distributed SafeTensors loading (`safetensors_dynamic`) across:
  - Scan Modes: `scanned` (`scan_layers=True`) and `unscanned` (`scan_layers=False`)
  - Slice Topologies: Single-Slice (`num_slices=1`) and Multi-Slice (`num_slices>=2`)
  - Hardware Generations:
      * TPU v5p (`bodaborg-v5p-nap`, `ct5p-hightpu-4t`, `2x2x1`): 4 devices/slice, 368 GB HBM/slice, 270 GB RAM/slice
      * TPU v7 / Ironwood (`bodaborg-tpu7x-nap`, `tpu7x-standard-4t`, `2x2x1`): 8 TPU cores/slice, 813.80 GB HBM/slice, 1,014.50 GB RAM/slice

Pipelines Measured:
  1. `to_maxtext` (3-Step Offline Orbax Pipeline):
     - Step 1 (`hf_to_gcs_s`): Download HuggingFace `.safetensors` shards to GCS and local staging dir
     - Step 2 (`convert_to_orbax_s`): Offline CPU weight conversion (`to_maxtext.py`) to GCS Orbax checkpoint
     - Step 3 (`load_orbax_s`): Restore Orbax checkpoint into the MaxText mesh (`source_checkpoint_layout="orbax"`)

  2. `safetensor` (`safetensors_dynamic`, 2-Step Direct Pipeline):
     - Step 1 (`hf_to_gcs_s`): Stream HuggingFace `.safetensors` shards directly to shared GCS `hf_cache`
     - Step 2 (`safetensors_and_convert_s`): Distributed sharded load + in-memory layout transformation
       (`source_checkpoint_layout="safetensors_dynamic"`), decomposed into:
         * `[1/3]` (`mappings_s`): Parameter mapping derivation
         * `[2/3]` (`load_sharded_hf_s`): Distributed byte-ranged `.safetensors` read via `ocp.v1.load_pytree`
         * `[3/3]` (`transform_hf_to_mt_s`): In-memory weight transformation / layer stacking (`_transform_matched_weights`)

Supported Environment Variables:
  - `REMOTE_SLICE1_ADDR`: Comma-separated `host:port` list of remote slice worker daemons (`--remote-slice-daemon`)
  - `KEEP_HF_CACHE=1`: Preserve the shared GCS `hf_cache` across runs to measure warm load or reuse across branches
  - `SKIP_FAILED_RESULTS=1`: When `--skip-completed` is set, also skip (do not retry) cached JSON results with
    `status == "FAILED"`; by default those are re-run
  - `SHARED_HF_CACHE_BASE_DIR`: Reuse an existing shared GCS `hf_cache` directory (e.g., when comparing `origin/main` vs. PRs)
  - `REUSE_SHARED_ST_GCS_CACHE=1`: Copy `.safetensors` shards directly from the `safetensors_dynamic` GCS cache in `to_maxtext` Step 1
  - `PHASE_TIMEOUT_S`: Per-phase wall-clock timeout in seconds, default `14400` (e.g., `360` to capture multi-host
    hangs on `origin/main`)
  - `STAGE_SPARSE_DEEPSEEK_OOM=1`: Stage sparse BF16 `.safetensors` headers on local disk for ultra-large DeepSeek models

Usage Examples:
  # Local or single-slice run:
  PYTHONPATH=src python3 checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models qwen3-0.6b

  # Multi-slice TPU v7 (tpu7x) run across 1 and 2 slices (8 and 16 TPU cores):
  PYTHONPATH=src python3 checkpoint_load_benchmark/benchmark_ckpt_load_time.py \
      --models qwen3-235b-a22b deepseek4-284b qwen3.5-397b-a17b qwen3-480b-a35b deepseek3-671b \
      --scan-modes scanned unscanned \
      --num-slices-list 1 2 \
      --devices-per-slice 8 \
      --hardware tpu \
      --reuse-model-hf-cache \
      --skip-completed \
      --allow-step-failure
"""

import argparse
import concurrent.futures
import contextlib
import csv
import datetime
import functools
import json
import mmap
import multiprocessing
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
import time

_ENTER_WALL_EPOCH = time.time()
_ENTER_MONO = time.perf_counter()

from typing import Any

# Prevent PyTorch (if installed in container) from autoloading torch_xla and locking libtpu before JAX distributed init
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _import_bench_modules():
  """Imports local `bench.*` instrumentation packages cleanly from `checkpoint_load_benchmark`."""
  bench_root = os.path.dirname(os.path.abspath(__file__))
  if bench_root not in sys.path:
    sys.path.insert(0, bench_root)
  from bench import harness_v2  # pylint: disable=import-outside-toplevel
  from bench import metadata  # pylint: disable=import-outside-toplevel
  from bench import schema_v2  # pylint: disable=import-outside-toplevel
  from bench import stages  # pylint: disable=import-outside-toplevel

  return harness_v2, metadata, schema_v2, stages


def _import_ocp_v1():
  """Imports Orbax v1 checkpoint API (`from orbax.checkpoint import v1` or `experimental.v1`)."""
  try:
    from orbax.checkpoint import v1 as ocp_v1  # pylint: disable=import-outside-toplevel

    return ocp_v1
  except ImportError:
    try:
      from orbax.checkpoint.experimental import v1 as ocp_v1  # pylint: disable=import-outside-toplevel

      return ocp_v1
    except ImportError:
      return None


# Register OmegaConf resolver for HF_TOKEN if present in configs
try:
  from omegaconf import OmegaConf

  if not OmegaConf.has_resolver("HF_TOKEN"):
    OmegaConf.register_new_resolver("HF_TOKEN", lambda: os.environ.get("HF_TOKEN", ""))
except ImportError:
  pass


def _register_transformers_fallbacks() -> None:
  """Ensures newer HuggingFace Config classes fall back to PretrainedConfig on older transformers."""
  try:
    import transformers
    from transformers.configuration_utils import PretrainedConfig as _PTConfig

    for _cfg_cls in ("Qwen3OmniMoeConfig", "Qwen3NextConfig", "Olmo3Config"):
      if not hasattr(transformers, _cfg_cls):
        setattr(transformers, _cfg_cls, _PTConfig)
  except ImportError:
    pass


DEFAULT_MODELS = ["qwen3-0.6b"]
DEFAULT_SCAN_MODES = ["scanned", "unscanned"]
DEFAULT_GCS_BUCKET = "gs://mesa-maxtext/benchmark_ckpt_load_time"
DEFAULT_HF_TOKEN_FILE = "/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/hf_token.txt"


def get_repo_root() -> str:
  """Resolves the MaxText repository root directory regardless of current working directory."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  for candidate in (script_dir, os.path.dirname(script_dir), os.getcwd()):
    if os.path.isdir(os.path.join(candidate, "src", "maxtext")):
      return candidate
  return script_dir


def resolve_dcn_axis(model_name: str, dcn_axis_arg: str = "auto") -> str:
  """Resolves the DCN parallelism axis for multi-slice execution (`dcn_data_parallelism` vs `dcn_fsdp_parallelism`)."""
  if dcn_axis_arg == "fsdp":
    return "dcn_fsdp_parallelism"
  if dcn_axis_arg == "data":
    return "dcn_data_parallelism"
  if model_name in (
      "gpt-oss-120b",
      "llama3.1-70b",
      "llama3.1-405b",
      "mixtral-8x22b",
      "qwen3-235b-a22b",
      "qwen3.5-397b-a17b",
      "qwen3-480b-a35b",
      "deepseek3-671b",
      "deepseek3.2-671b",
      "deepseek4-284b",
      "deepseek4-1600b",
  ):
    return "dcn_fsdp_parallelism"
  return "dcn_data_parallelism"


def get_hf_token() -> str:
  token = os.environ.get("HF_TOKEN") or os.environ.get("HF_AUTH_TOKEN") or ""
  if not token:
    token_file = os.environ.get("HF_TOKEN_FILE", DEFAULT_HF_TOKEN_FILE)
    if os.path.isfile(token_file):
      with open(token_file, "r", encoding="utf-8") as f:
        token = f.read().strip()
      if token:
        os.environ["HF_TOKEN"] = token
  return token


def resolve_hardware(hw_arg: str) -> str:
  if hw_arg in ("cpu", "tpu"):
    return hw_arg
  if os.path.exists("/dev/vfio/0") or os.path.exists("/dev/accel0") or os.environ.get("TPU_ACCELERATOR_TYPE"):
    return "tpu"
  return "cpu"


def check_tpu_backend(
    hardware: str,
    backend: str,
    device_count: int,
    expected_device_count: int,
) -> tuple[bool, str]:
  """Verifies that --hardware tpu initialized JAX on TPU with expected_device_count global devices."""
  if hardware != "tpu":
    return True, ""
  if backend != "tpu":
    return False, f"Expected JAX backend 'tpu' for --hardware tpu, got {backend!r} ({device_count} devices)"
  if device_count != expected_device_count:
    return (
        False,
        f"Expected {expected_device_count} TPU devices (num_slices * devices_per_slice), got {device_count}",
    )
  return True, ""


def _verify_and_measure_jax_backend(args: argparse.Namespace, jax_mod: Any) -> dict[str, Any]:
  """Runs the TPU check right after JAX init and returns measured backend/device metadata."""
  backend = str(jax_mod.default_backend())
  device_count = int(jax_mod.device_count())
  local_device_count = int(jax_mod.local_device_count())
  devices = jax_mod.devices()
  device_kind = str(devices[0].device_kind) if devices else "unknown"
  expected_devices = int(getattr(args, "num_slices", 1)) * int(getattr(args, "devices_per_slice", 4))
  ok, reason = check_tpu_backend(getattr(args, "hardware", "auto"), backend, device_count, expected_devices)
  if not ok:
    print(
        f"[NOT-TPU] backend={backend} device_count={device_count} expected={expected_devices}: {reason}",
        flush=True,
    )
    raise RuntimeError(f"TPU check failed: {reason}")
  print(
      f"[TPU CHECK OK] backend={backend} device_kind={device_kind!r} "
      f"global_devices={device_count} local_devices={local_device_count}",
      flush=True,
  )
  return {
      "jax_backend_platform": backend,
      "tpu_device_kind": device_kind,
      "total_tpu_devices_measured": device_count,
      "tpu_devices_per_host_measured": local_device_count,
  }


def get_concurrent_gb(model_name: str, scan_mode: str = "scanned", num_slices: int = 1) -> int:
  """Returns checkpoint_storage_concurrent_gb sized to fit the largest single leaf while minimizing transient RAM."""
  if scan_mode == "unscanned":
    return 24
  if model_name == "llama3.1-70b":
    # The largest scanned leaf is 37.58 GB. On tpu7x, a 2-slice restore with dcn_fsdp_parallelism=2 still requests
    # the whole leaf per process (`Requested more bytes than we reserved space for: 37580963840 > 20000000000`).
    return 38
  if model_name in ("mixtral-8x7b", "gpt-oss-120b", "qwen3-next-80b-a3b"):
    return 45
  if model_name in ("mixtral-8x22b", "deepseek4-284b"):
    return 96
  if model_name in ("qwen3-235b-a22b",):
    return 160
  if model_name in ("qwen3.5-397b-a17b", "llama3.1-405b"):
    return 210
  if model_name in ("qwen3-480b-a35b",):
    return 330
  if model_name in ("deepseek3-671b", "deepseek3.2-671b", "deepseek4-1600b"):
    return 460
  return 20


def get_loader_flags(model_name: str) -> list[str]:
  """Returns loader flags for `to_maxtext` conversion."""
  if model_name.startswith("gemma3"):
    return ["--lazy_load_tensors=False", "--eager_load_method=transformers"]
  return ["--lazy_load_tensors=True"]


def get_tokenizer_flags(model_name: str, hf_ids: dict[str, str]) -> list[str]:
  """Returns tokenizer config flags so model config initialization succeeds."""
  if "gemma4" in model_name:
    return [
        "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_gemma4.model",
        "tokenizer_type=sentencepiece",
    ]
  if "gemma3" in model_name:
    return [
        "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3",
        "tokenizer_type=sentencepiece",
    ]
  if "gemma" in model_name:
    return [
        "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma",
        "tokenizer_type=sentencepiece",
    ]
  if "llama" in model_name:
    return [
        "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken",
        "tokenizer_type=tiktoken",
    ]
  if "mistral" in model_name:
    return [
        "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.mistral-v3",
        "tokenizer_type=sentencepiece",
    ]
  if model_name.startswith("deepseek4"):
    return [
        "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.llama2",
        "tokenizer_type=sentencepiece",
        "attention=dot_product",
    ]
  if model_name.startswith("deepseek3.2"):
    return [
        f"tokenizer_path={hf_ids[model_name]}",
        "tokenizer_type=huggingface",
        "attention=dot_product",
    ]
  if model_name in hf_ids:
    return [f"tokenizer_path={hf_ids[model_name]}", "tokenizer_type=huggingface"]
  return []


def build_abstract_state_for_load(config):
  """Creates the MaxText mesh and abstract state used by `checkpointing.load_state_if_possible`."""
  from maxtext.utils import maxtext_utils
  from maxtext.utils import model_creation_utils

  mesh = maxtext_utils.get_mesh_from_config(config, None)
  create_model_partial, _ = model_creation_utils.create_nnx_abstract_model(config, mesh, None)
  unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, create_model_partial, is_training=False)
  return mesh, unboxed_abstract_state


def _stage_sparse_bf16_safetensors(model_name: str, target_dir: str) -> tuple[int, int, float]:
  """Creates sharded sparse BF16 .safetensors files (~4 GB per shard) matching HF_SHAPE[model_name]."""
  import math
  import struct
  from maxtext.checkpoint_conversion.utils import hf_model_configs
  from maxtext.checkpoint_conversion.utils import hf_shape

  os.makedirs(target_dir, exist_ok=True)
  marker = os.path.join(target_dir, ".sparse_sharded_v2_ready")
  index_path = os.path.join(target_dir, "model.safetensors.index.json")
  cfg_dict = hf_model_configs.HF_MODEL_CONFIGS[model_name].to_dict()
  shapes_map = hf_shape.HF_SHAPE[model_name](cfg_dict)
  clean_shapes = {k: list(v) for k, v in shapes_map.items() if isinstance(k, str)}
  total_bytes = sum(int(math.prod(shp)) * 2 for shp in clean_shapes.values())
  total_gb = total_bytes / 1e9
  if os.path.exists(marker) and os.path.exists(index_path):
    num_st = len([f for f in os.listdir(target_dir) if f.endswith(".safetensors")])
    return num_st, len(clean_shapes), total_gb

  for fn in os.listdir(target_dir):
    if fn.endswith(".safetensors") or fn in ("model.safetensors.index.json", ".sparse_ready", ".sparse_sharded_v2_ready"):
      try:
        os.remove(os.path.join(target_dir, fn))
      except OSError:
        pass

  with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as cf:
    json.dump(cfg_dict, cf, indent=2)

  max_shard_bytes = 4 * 10**9
  max_tensors_per_shard = 500
  shards: list[list[tuple[str, list[int], int]]] = []
  cur_shard: list[tuple[str, list[int], int]] = []
  cur_bytes = 0
  for k, shp in clean_shapes.items():
    nbytes = int(math.prod(shp)) * 2
    if cur_shard and (cur_bytes + nbytes > max_shard_bytes or len(cur_shard) >= max_tensors_per_shard):
      shards.append(cur_shard)
      cur_shard = []
      cur_bytes = 0
    cur_shard.append((k, shp, nbytes))
    cur_bytes += nbytes
  if cur_shard:
    shards.append(cur_shard)

  num_shards = len(shards)
  weight_map: dict[str, str] = {}
  for idx, shard_items in enumerate(shards, start=1):
    shard_fname = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
    header_dict = {}
    offset = 0
    for k, shp, nbytes in shard_items:
      header_dict[k] = {"dtype": "BF16", "shape": shp, "data_offsets": [offset, offset + nbytes]}
      weight_map[k] = shard_fname
      offset += nbytes
    header_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    pad_len = (8 - (len(header_bytes) % 8)) % 8
    if pad_len:
      header_bytes += b" " * pad_len
    st_path = os.path.join(target_dir, shard_fname)
    with open(st_path, "wb") as sf:
      sf.write(struct.pack("<Q", len(header_bytes)))
      sf.write(header_bytes)
      sf.flush()
      os.truncate(sf.fileno(), 8 + len(header_bytes) + offset)

  with open(index_path, "w", encoding="utf-8") as ix_f:
    json.dump(
        {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        },
        ix_f,
    )

  with open(marker, "w", encoding="utf-8") as mf:
    mf.write(f"{total_gb:.2f}\n")
  return num_shards, len(clean_shapes), total_gb


def _should_stage_sparse_bf16(model_name: str) -> bool:
  if os.environ.get("STAGE_SPARSE_ALL_LARGE") == "1" and model_name in (
      "qwen3-235b-a22b",
      "qwen3.5-397b-a17b",
      "qwen3-480b-a35b",
      "llama3.1-405b",
      "deepseek3-671b",
      "deepseek3.2-671b",
      "deepseek4-284b",
      "deepseek4-1600b",
  ):
    return True
  return os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1" and model_name in (
      "deepseek3-671b",
      "deepseek3.2-671b",
      "deepseek4-284b",
      "deepseek4-1600b",
  )


def _list_hf_repo_files(repo_id: str, token: str | None) -> list[str]:
  """Lists repository files on HF Hub using a single lightweight API call with 429 backoff."""
  import random
  import huggingface_hub

  is_gated = repo_id.startswith(("meta-llama/", "google/"))
  for attempt in range(8):
    try:
      use_tok = token if (is_gated or attempt % 2 == 1) else None
      api = huggingface_hub.HfApi(token=use_tok)
      return list(api.list_repo_files(repo_id=repo_id))
    except Exception as ex:  # pylint: disable=broad-exception-caught
      if attempt < 7:
        wait_s = min(15 * (attempt + 1), 60) + random.uniform(1.0, 5.0)
        print(f"[HF API] list_repo_files({repo_id}) retry {attempt+1}/8 in {wait_s:.1f}s ({ex})", flush=True)
        time.sleep(wait_s)
      else:
        raise
  return []


def fast_build_gcs_cache_worker(fpath: str, gcs_cache_dir: str, hf_access_token: str | None) -> None:
  """Streams a single .safetensors shard from HF Hub to GCS using 1 HTTP GET request (instead of 800 5MB range reads)."""
  import random
  from google.cloud import storage
  import huggingface_hub
  import requests

  parts = fpath.strip("/").split("/")
  repo_id = "/".join(parts[:2])
  filename = "/".join(parts[2:])
  url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=filename)

  no_scheme = gcs_cache_dir.replace("gs://", "")
  bucket_name, _, blob_prefix = no_scheme.partition("/")
  blob_name = os.path.join(blob_prefix, os.path.basename(fpath))

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  if blob.exists() and (blob.size or 0) > 0:
    return

  time.sleep(random.uniform(0.0, 2.0))
  is_gated = repo_id.startswith(("meta-llama/", "google/"))
  t0 = time.time()
  max_retries = 8
  for attempt in range(max_retries):
    try:
      headers = {}
      if hf_access_token and (is_gated or attempt % 2 == 1):
        headers["Authorization"] = f"Bearer {hf_access_token}"
      with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as resp:
        if resp.status_code == 429:
          retry_after = float(resp.headers.get("Retry-After", 20 * (attempt + 1)))
          wait_s = min(retry_after, 90.0) + random.uniform(1.0, 5.0)
          print(
              f"[Worker] 429 on {os.path.basename(fpath)}, backing off {wait_s:.1f}s (attempt {attempt+1}/{max_retries})",
              flush=True,
          )
          time.sleep(wait_s)
          continue
        resp.raise_for_status()
        resp.raw.decode_content = True
        with blob.open("wb", chunk_size=32 * 1024 * 1024) as gcs_out:
          shutil.copyfileobj(resp.raw, gcs_out, length=16 * 1024 * 1024)
      print(f"[Worker] Cached {os.path.basename(fpath)} in {time.time() - t0:.1f}s", flush=True)
      return
    except Exception as e:  # pylint: disable=broad-exception-caught
      if attempt < max_retries - 1:
        wait_s = min(12 * (attempt + 1), 60) + random.uniform(1.0, 5.0)
        print(
            f"[Worker] Error fetching {os.path.basename(fpath)}: {e}. Retrying in {wait_s:.1f}s ({attempt+1}/{max_retries})",
            flush=True,
        )
        time.sleep(wait_s)
      else:
        raise


def run_worker_to_maxtext_step1_hf_to_gcs(args: argparse.Namespace) -> dict[str, Any]:
  """Step 1 for `to_maxtext`: Transfer HF checkpoint from HF Hub to GCS (and stage locally for conversion)."""
  import huggingface_hub
  import requests
  from maxtext.checkpoint_conversion.utils import load_dynamic
  from maxtext.utils import globals as maxtext_globals

  load_dynamic.build_gcs_cache_worker = fast_build_gcs_cache_worker
  repo_id = maxtext_globals.HF_IDS[args.model]
  token = get_hf_token() or None
  recorder = getattr(args, "_recorder", None)

  if _should_stage_sparse_bf16(args.model):
    t_gcs_start = time.perf_counter()
    print(f"[to_maxtext Step 1] Inspecting HF Hub repository {repo_id} for {args.model} ...", flush=True)
    try:
      all_rfiles = _list_hf_repo_files(repo_id, token)
      files = [f"{repo_id}/{fn}" for fn in all_rfiles if fn.endswith(".safetensors")]
      if files:
        print(
            f"[to_maxtext Step 1] Found {len(files)} .safetensors shards on HF Hub for {repo_id}. "
            f"Inspecting native shard header for quantization format...",
            flush=True,
        )
        import struct

        first_fn = files[0].split("/", 2)[2]
        hdr_url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=first_fn)
        with requests.get(hdr_url, headers={"Range": "bytes=0-65535"}, timeout=30) as rf:
          raw_hdr = rf.content
          hlen = struct.unpack("<Q", raw_hdr[:8])[0]
          if len(raw_hdr) < 8 + hlen:
            with requests.get(hdr_url, headers={"Range": f"bytes=0-{8 + hlen}"}, timeout=30) as rf2:
              raw_hdr = rf2.content
          hdr = json.loads(raw_hdr[8 : 8 + hlen].decode("utf-8"))
        dtypes_seen = sorted({v.get("dtype", "") for k, v in hdr.items() if k != "__metadata__"})
        scale_keys = [k for k in hdr if "scale" in k]
        print(
            f"[to_maxtext Step 1] Native HF shard {os.path.basename(files[0])}: dtypes={dtypes_seen}, "
            f"scale_inv keys={len(scale_keys)} (e.g. {scale_keys[:2]}). "
            f"Native FP8 (F8_E4M3 + weight_scale_inv) requires BF16 dequantization before MaxText loading.",
            flush=True,
        )
      else:
        print(
            f"[to_maxtext Step 1] No public .safetensors shards found on HF Hub for {repo_id} "
            f"(unreleased / gated repository; native V4 format uses FP4 U8 packed + block scale).",
            flush=True,
        )
    except Exception as ex:  # pylint: disable=broad-exception-caught
      print(f"[to_maxtext Step 1] HF Hub query for {repo_id}: {ex}", flush=True)
    hf_to_gcs_s = time.perf_counter() - t_gcs_start

    t_local_start = time.perf_counter()
    num_st, num_tensors, total_gb = _stage_sparse_bf16_safetensors(args.model, args.local_hf_dir)
    hf_to_local_s = time.perf_counter() - t_local_start
    print(
        f"[to_maxtext Step 1] Staged sparse BF16 .safetensors for {args.model} at {args.local_hf_dir} "
        f"({num_st} shards, {num_tensors} tensors, {total_gb:.2f} GB BF16 footprint) in {hf_to_local_s:.2f}s",
        flush=True,
    )
    return {
        "hf_to_gcs_s": hf_to_gcs_s,
        "hf_to_local_s": hf_to_local_s,
        "num_safetensors_files": num_st,
        "num_tensors": num_tensors,
        "bf16_total_gb": round(total_gb, 2),
        "local_hf_dir": args.local_hf_dir,
    }

  # 1a. Measure HF -> GCS transfer using the same mechanism as load_dynamic.py
  from etils import epath

  all_repo_files = _list_hf_repo_files(repo_id, token)
  files = [f"{repo_id}/{fn}" for fn in all_repo_files if fn.endswith(".safetensors")]
  json_files = [fn for fn in all_repo_files if fn.endswith(".json")]
  if not files:
    raise RuntimeError(f"No .safetensors files found on HF Hub for {repo_id}")

  # SHARED_HF_CACHE_BASE_DIR (`<run dir>/safetensors_dynamic` of another run) lets a new run dir reuse that run's
  # cache, as the safetensors_dynamic step already does; the cold Step 1 time is then looked up in that run's logs.
  shared_cache_root = os.environ.get("SHARED_HF_CACHE_BASE_DIR", "").strip().rstrip("/")
  shared_st_cache_dir = (
      f"{shared_cache_root or args.gcs_run_dir + '/safetensors_dynamic'}/{args.model}/shared/hf_cache/"
      f"{repo_id.replace('/', '_')}"
  )
  pre_staged_gcs_dir = os.environ.get("BENCH_SAFETENSORS_GCS_DIR", "").strip().rstrip("/")
  log_roots = [f"{args.gcs_run_dir}/logs/{args.model}"]
  if shared_cache_root.endswith("/safetensors_dynamic"):
    log_roots.append(f"{shared_cache_root.removesuffix('/safetensors_dynamic')}/logs/{args.model}")
  reused_existing = False
  if pre_staged_gcs_dir:
    gcs_cache_dir = pre_staged_gcs_dir
    hf_to_gcs_s = 0.0
    reused_existing = True
    if recorder is not None:
      with recorder.stage("to_maxtext_step1_verify_gcs_cache_seconds"):
        pass
    print(f"[to_maxtext Step 1] Using pre-staged GCS dir {gcs_cache_dir}", flush=True)
  elif os.environ.get("REUSE_SHARED_ST_GCS_CACHE") == "1" or getattr(args, "reuse_model_hf_cache", False):
    # Reuse the shared cache when every shard is in it. Requiring a cold HF->GCS time in the scanned 1-slice
    # safetensors_dynamic record made two workers re-download qwen3.5-397b-a17b (795 GB) from HF Hub into the cache
    # that other runs were reading, because that record was an infra failure without a Step 1 time.
    _t_ver0 = time.perf_counter()
    try:
      from google.cloud import storage as gcs_storage

      c_bucket, _, c_prefix = shared_st_cache_dir.removeprefix("gs://").partition("/")
      cached = {
          b.name.rsplit("/", 1)[-1]: b.size
          for b in gcs_storage.Client().list_blobs(c_bucket, prefix=c_prefix.rstrip("/") + "/")
      }
      if all(cached.get(os.path.basename(f), 0) > 0 for f in files):
        # The cold download is the largest Step 1 time recorded for this model; warm cache checks take under 1 s.
        recorded = []
        for log_root in log_roots:
          for res in epath.Path(log_root).glob("*/*_result.json"):
            if res.name in ("safetensors_dynamic_result.json", "to_maxtext_step1_result.json"):
              try:
                recorded.append(float(json.loads(res.read_text()).get("hf_to_gcs_s", 0.0) or 0.0))
              except Exception:  # pylint: disable=broad-exception-caught
                pass
        cold_s = max(recorded, default=0.0)
        gcs_cache_dir = shared_st_cache_dir
        hf_to_gcs_s = cold_s if cold_s >= 1.0 else 0.0
        reused_existing = True
        cold_note = f"recorded cold HF->GCS={hf_to_gcs_s:.2f}s" if hf_to_gcs_s else "no cold HF->GCS time in this run dir"
        print(
            f"[to_maxtext Step 1] Reusing populated HF->GCS cache {gcs_cache_dir} ({cold_note}, "
            f"{len(files)} safetensors files)",
            flush=True,
        )
    except Exception as ex:  # pylint: disable=broad-exception-caught
      print(f"[to_maxtext Step 1] Could not check the shared HF->GCS cache ({ex}); downloading from HF Hub", flush=True)
    if recorder is not None:
      recorder.record_value("to_maxtext_step1_verify_gcs_cache_seconds", time.perf_counter() - _t_ver0)

  if not reused_existing:
    gcs_cache_dir = (
        shared_st_cache_dir
        if getattr(args, "reuse_model_hf_cache", False)
        else f"{args.gcs_run_dir}/to_maxtext/{args.model}/{args.scan_mode}/hf_cache/{repo_id.replace('/', '_')}"
    )
    print(f"[to_maxtext Step 1] Downloading {repo_id}/*.safetensors from HF Hub -> {gcs_cache_dir} ...", flush=True)
    t_gcs_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(files)))) as executor:
      futures = [executor.submit(fast_build_gcs_cache_worker, fpath, gcs_cache_dir, token) for fpath in files]
      for f in concurrent.futures.as_completed(futures):
        f.result()
    hf_to_gcs_s = time.perf_counter() - t_gcs_start
    print(f"[to_maxtext Step 1] HF -> GCS completed in {hf_to_gcs_s:.2f}s ({len(files)} safetensors files)", flush=True)

  # 1b. Stage HF checkpoint to a clean local directory so `to_maxtext` Step 2 converts without HF network I/O
  local_hf_dir = args.local_hf_dir
  local_hf_cache = f"{local_hf_dir}_cache"
  if os.path.exists(local_hf_dir):
    shutil.rmtree(local_hf_dir)
  if os.path.exists(local_hf_cache):
    shutil.rmtree(local_hf_cache)
  os.makedirs(local_hf_dir, exist_ok=True)

  print(f"[to_maxtext Step 1] Staging {repo_id} to local directory {local_hf_dir} for Step 2 conversion...", flush=True)
  t_local_start = time.perf_counter()
  is_gated = repo_id.startswith(("meta-llama/", "google/"))
  for jf in json_files:
    jf_url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=jf)
    jf_dst = os.path.join(local_hf_dir, jf)
    os.makedirs(os.path.dirname(jf_dst), exist_ok=True)
    for attempt in range(6):
      try:
        hdrs = {"Authorization": f"Bearer {token}"} if (token and (is_gated or attempt % 2 == 1)) else {}
        with requests.get(jf_url, headers=hdrs, timeout=30) as jresp:
          if jresp.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
          jresp.raise_for_status()
          with open(jf_dst, "wb") as jout:
            jout.write(jresp.content)
        break
      except Exception:  # pylint: disable=broad-exception-caught
        if attempt == 5:
          raise
        time.sleep(10 * (attempt + 1))
  from etils import epath

  with open(os.path.join(local_hf_dir, ".gcs_cache_dir"), "w", encoding="utf-8") as gf:
    gf.write(gcs_cache_dir)

  def _copy_from_gcs_cache(fpath: str) -> None:
    fname = os.path.basename(fpath)
    src = epath.Path(f"{gcs_cache_dir}/{fname}")
    dst = os.path.join(local_hf_dir, fname)
    with src.open("rb") as fsrc, open(dst, "wb") as fdst:
      shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)

  free_disk_gb = shutil.disk_usage(local_hf_dir).free / 1e9
  try:
    from google.cloud import storage as gcs_storage

    c_bucket, _, c_prefix = gcs_cache_dir.removeprefix("gs://").partition("/")
    blob_sizes = {
        b.name.rsplit("/", 1)[-1]: b.size
        for b in gcs_storage.Client().list_blobs(c_bucket, prefix=c_prefix.rstrip("/") + "/")
    }
    total_gb = sum(blob_sizes.get(os.path.basename(f), 4.2e9) for f in files) / 1e9
  except Exception:  # pylint: disable=broad-exception-caught
    total_gb = len(files) * 4.2
  # `to_maxtext` holds the whole converted state (about the checkpoint size) in host RAM before the Orbax save. On
  # tpu7x (1,014.5 GB RAM, 986.7 GB allocatable), a conversion pod was evicted at 944 GB container memory, so a
  # conversion whose state alone does not fit in ~93% of host RAM cannot finish here. Record that instead of running
  # for hours (qwen3-480b-a35b: 960 GB).
  _, avail_ram_gb, host_ram_gb = _read_host_mem_gb()
  ram_ratio, _, _, _ = _platform_runtime_limits(getattr(args, "platform", None))
  usable_ram_gb = min(ram_ratio * host_ram_gb, avail_ram_gb + 20.0)
  if total_gb + 40.0 > usable_ram_gb:
    raise RuntimeError(
        f"to_maxtext conversion of {args.model} holds the whole converted state (~{total_gb:.1f} GB) in host RAM "
        f"before the Orbax save, but this host has ~{usable_ram_gb:.1f} GB usable ({host_ram_gb:.1f} GB total, "
        f"{avail_ram_gb:.1f} GB currently available). Convert on a host with more RAM."
    )
  # Staged shards on /ramdisk live in host RAM. `to_maxtext` then holds the whole converted state in host RAM before
  # the Orbax save (about the checkpoint size again), so stage only when both fit, plus a margin for transients. On
  # tpu7x (944.8 GiB RAM, ~100 GB boot disk), a scanned qwen3-235b-a22b conversion (470 GB) with all shards staged
  # was OOM-killed at leaf 4 of 15.
  if local_hf_dir.startswith("/ramdisk"):
    stage_budget_gb = min(free_disk_gb, avail_ram_gb) - total_gb - 60.0
  else:
    stage_budget_gb = free_disk_gb - 20.0
  if total_gb <= stage_budget_gb:
    local_copy_mode = "staged_all"
    print(
        f"[to_maxtext Step 1] Staging all {len(files)} shards ({total_gb:.1f} GB) from {gcs_cache_dir} "
        f"(staging budget {stage_budget_gb:.1f} GB)",
        flush=True,
    )
    t_cp0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(1, len(files)))) as pool:
      list(pool.map(_copy_from_gcs_cache, files))
    t_cp_s = time.perf_counter() - t_cp0
    if recorder is not None:
      recorder.record_value("to_maxtext_step1_copy_gcs_to_local_ramdisk_seconds", t_cp_s)
      recorder.record_value("to_maxtext_step1_copy_gcs_to_local_ramdisk_gb", round(total_gb, 4))
      if t_cp_s > 0:
        recorder.record_value("to_maxtext_step1_copy_gcs_to_local_ramdisk_gb_per_second", round(total_gb / t_cp_s, 4))
  elif args.scan_mode == "scanned":
    # A scanned leaf stacks each MoE weight expert by expert across all layers, so every leaf reads every shard.
    # Streaming shards through a small LRU then downloads the checkpoint once per expert: on tpu7x, a scanned
    # qwen3-235b-a22b conversion wrote 1 of 15 leaves in 2 h. Scanned conversion needs the checkpoint staged.
    raise RuntimeError(
        f"Scanned to_maxtext conversion of {args.model} needs all {len(files)} shards ({total_gb:.1f} GB) staged on "
        f"one host, and the converted state (about {total_gb:.1f} GB more) is held in host RAM before the save. "
        f"Only {stage_budget_gb:.1f} GB is left for staging ({free_disk_gb:.1f} GB free in "
        f"{os.path.dirname(local_hf_dir)}). Convert on a host with more local storage or RAM."
    )
  else:
    local_copy_mode = "json_only_lru_stream"
    # Unscanned leaves read one layer at a time (a few adjacent shards), so a small LRU of streamed shards works.
    print(
        f"[to_maxtext Step 1] Model {args.model} ({len(files)} shards, {total_gb:.1f} GB) exceeds the staging budget "
        f"({stage_budget_gb:.1f} GB); staged index/config JSONs and enabled bounded LRU shard streaming from "
        f"{gcs_cache_dir}",
        flush=True,
    )
  hf_to_local_s = time.perf_counter() - t_local_start
  t_cln0 = time.perf_counter()
  shutil.rmtree(os.path.join(local_hf_dir, ".cache"), ignore_errors=True)
  shutil.rmtree(local_hf_cache, ignore_errors=True)
  if recorder is not None:
    recorder.record_value("to_maxtext_step1_cleanup_seconds", time.perf_counter() - t_cln0)
    recorder.record_value("to_maxtext_step1_total_seconds", hf_to_local_s)
  print(f"[to_maxtext Step 1] Local staging completed in {hf_to_local_s:.2f}s", flush=True)

  return {
      "hf_to_gcs_s": hf_to_gcs_s,
      "hf_to_local_s": hf_to_local_s,
      "to_maxtext_step1_local_copy_mode": local_copy_mode,
      "to_maxtext_step1_copy_gcs_to_local_ramdisk_gb": round(total_gb if local_copy_mode == "staged_all" else 0.0, 4),
      "num_safetensors_files": len(files),
      "gcs_cache_dir": gcs_cache_dir,
      "local_hf_dir": local_hf_dir,
  }


def _parallel_gcs_download(src_uri: str, dst: str, workers: int, chunk_bytes: int = 128 * 1024 * 1024) -> int:
  """Downloads one GCS object to `dst` as concurrent byte-range reads; returns the object size in bytes.

  A single sequential stream reached ~55 MB/s from a tpu7x host, which made streaming the shards of a 807 GB
  checkpoint through `to_maxtext` take ~4.3 h. Each worker thread uses its own client (one HTTP session each).
  """
  import concurrent.futures
  import threading

  from google.cloud import storage as gcs_storage

  bucket_name, _, blob_name = src_uri.removeprefix("gs://").partition("/")
  size = gcs_storage.Client().bucket(bucket_name).get_blob(blob_name).size
  with open(dst, "wb") as f:
    f.truncate(size)
  local = threading.local()

  def _read_range(start: int) -> None:
    if not hasattr(local, "blob"):
      local.blob = gcs_storage.Client().bucket(bucket_name).blob(blob_name)
    data = local.blob.download_as_bytes(start=start, end=min(start + chunk_bytes, size) - 1, checksum=None)
    fd = os.open(dst, os.O_WRONLY)
    try:
      os.pwrite(fd, data, start)
    finally:
      os.close(fd)

  with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
    list(pool.map(_read_range, range(0, size, chunk_bytes)))
  if os.path.getsize(dst) != size:
    raise RuntimeError(f"Downloaded {os.path.getsize(dst)} of {size} bytes of {src_uri}")
  return size


def run_worker_to_maxtext_step2_convert(args: argparse.Namespace) -> dict[str, Any]:
  """Step 2 for `to_maxtext`: Convert HF checkpoint to an Orbax checkpoint on GCS."""
  harness_v2, _, _, _ = _import_bench_modules()
  recorder = getattr(args, "_recorder", None)
  mon = getattr(args, "_monitoring", None)
  t_step2_entry = time.perf_counter()

  with recorder.stage("setup_python_imports_seconds") if recorder else contextlib.nullcontext():
    import jax

    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    jax.config.update("jax_platforms", "cpu")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    from maxtext.checkpoint_conversion import to_maxtext
    from maxtext.checkpoint_conversion.utils import utils as conv_utils
    from maxtext.common import checkpointing

  if mon is not None:
    mon.register(jax.monitoring)
  if recorder is not None:
    harness_v2.install_to_maxtext_step2_hooks(recorder, to_maxtext_mod=to_maxtext, conv_utils_mod=conv_utils, jax_mod=jax)

  sub_timings: dict[str, float] = {}

  orig_get_model_info = to_maxtext.get_maxtext_model_info
  orig_shard_jax_weights = conv_utils.shard_jax_weights
  orig_save_checkpoint = checkpointing.save_checkpoint
  orig_wait_until_finished = checkpointing.wait_until_finished

  def timed_get_model_info(*a, **kw):
    with (
        recorder.stage("to_maxtext_step2_build_abstract_model_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      t0 = time.perf_counter()
      res = orig_get_model_info(*a, **kw)
      sub_timings["init_abstract_model_s"] = time.perf_counter() - t0
      return res

  def timed_shard_jax_weights(*a, **kw):
    with (
        recorder.stage("to_maxtext_step2_transform_total_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      t0 = time.perf_counter()
      res = orig_shard_jax_weights(*a, **kw)
      sub_timings["transform_and_shard_s"] = time.perf_counter() - t0
      return res

  save_start_holder = {}

  def timed_save_checkpoint(*a, **kw):
    save_start_holder["t0"] = time.perf_counter()
    with (
        recorder.stage("to_maxtext_step2_orbax_save_total_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      return orig_save_checkpoint(*a, **kw)

  def timed_wait_until_finished(*a, **kw):
    with recorder.stage("to_maxtext_step2_orbax_save_total_seconds") if recorder else contextlib.nullcontext():
      res = orig_wait_until_finished(*a, **kw)
    if "t0" in save_start_holder:
      sub_timings["orbax_save_to_gcs_s"] = time.perf_counter() - save_start_holder["t0"]
      if recorder is not None:
        recorder.record_value("to_maxtext_step2_orbax_save_total_seconds", sub_timings["orbax_save_to_gcs_s"])
    return res

  to_maxtext.get_maxtext_model_info = timed_get_model_info
  conv_utils.shard_jax_weights = timed_shard_jax_weights
  checkpointing.save_checkpoint = timed_save_checkpoint
  checkpointing.wait_until_finished = timed_wait_until_finished

  if _should_stage_sparse_bf16(args.model):
    _stage_sparse_bf16_safetensors(args.model, args.local_hf_dir)
    import jax.numpy as jnp
    import numpy as np

    _orig_jnp_asarray = jnp.asarray

    def _materialized_jnp_asarray(a, *ja, **jkw):
      if isinstance(a, np.ndarray) and a.nbytes >= 1024 * 1024:
        a = np.array(a, copy=True)
        a.view(np.uint8).reshape(-1)[::4096] = 1
      res = _orig_jnp_asarray(a, *ja, **jkw)
      res.block_until_ready()
      return res

    jnp.asarray = _materialized_jnp_asarray
  elif not os.path.isdir(args.local_hf_dir) or not os.listdir(args.local_hf_dir):
    print(
        f"[to_maxtext Step 2] Local staging dir {args.local_hf_dir} missing after GCS log restore; re-staging from GCS cache...",
        flush=True,
    )
    # Step 1 ran on another pod (or this one was replaced): the local copy is part of to_maxtext's cost, so record it.
    sub_timings["hf_to_local_restage_s"] = run_worker_to_maxtext_step1_hf_to_gcs(args)["hf_to_local_s"]

  from etils import epath

  gcs_marker = os.path.join(args.local_hf_dir, ".gcs_cache_dir")
  if os.path.exists(gcs_marker) and hasattr(to_maxtext, "LazyHFLoader"):
    with open(gcs_marker, "r", encoding="utf-8") as gf:
      staged_gcs_cache_dir = gf.read().strip()
    import collections
    import threading

    _orig_get_tensor = to_maxtext.LazyHFLoader.get_tensor
    _lru_shards: collections.OrderedDict[str, str] = collections.OrderedDict()
    _fetch_lock = threading.Lock()
    # TM_STREAM_DOWNLOAD_WORKERS=1 reproduces the original single-stream fetch (used for the first 235B run).
    stream_workers = int(os.environ.get("TM_STREAM_DOWNLOAD_WORKERS", "16"))
    stream_stats = {"bytes": 0, "seconds": 0.0, "shards": 0}
    sub_timings["stream_download_workers"] = stream_workers

    def _streaming_get_tensor(self, key: str, dtype: str | None = None):
      with (
          recorder.stage("to_maxtext_step2_read_hf_tensors_to_host_ram_seconds") if recorder else contextlib.nullcontext()
      ):
        if self.is_local:
          resolved_key = to_maxtext.resolve_scale_key(key, self.shard_map)
          shard_name = self.shard_map.get(resolved_key) or self.shard_map.get(None)
          if shard_name:
            local_shard = os.path.join(self.model_id, shard_name)
            with _fetch_lock:
              if not os.path.exists(local_shard):
                while len(_lru_shards) >= 3:
                  _, old_path = _lru_shards.popitem(last=False)
                  try:
                    os.remove(old_path)
                  except OSError:
                    pass
                with (
                    recorder.stage("to_maxtext_step2_stream_files_from_gcs_seconds")
                    if recorder
                    else contextlib.nullcontext()
                ):
                  t_fetch = time.perf_counter()
                  if stream_workers > 1:
                    for attempt in range(3):
                      try:
                        stream_stats["bytes"] += _parallel_gcs_download(
                            f"{staged_gcs_cache_dir}/{shard_name}", local_shard, stream_workers
                        )
                        break
                      except Exception:  # pylint: disable=broad-exception-caught
                        # Never leave a partial shard behind: the next lookup would treat it as cached.
                        try:
                          os.remove(local_shard)
                        except OSError:
                          pass
                        if attempt == 2:
                          raise
                  else:
                    src_p = epath.Path(f"{staged_gcs_cache_dir}/{shard_name}")
                    with src_p.open("rb") as fsrc, open(local_shard, "wb") as fdst:
                      shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
                    stream_stats["bytes"] += os.path.getsize(local_shard)
                  stream_stats["seconds"] += time.perf_counter() - t_fetch
                  stream_stats["shards"] += 1
                  sub_timings["stream_download_s"] = stream_stats["seconds"]
                  sub_timings["stream_download_gb"] = stream_stats["bytes"] / 1e9
                  sub_timings["stream_download_shards"] = stream_stats["shards"]
              _lru_shards[shard_name] = local_shard
              _lru_shards.move_to_end(shard_name)
        return _orig_get_tensor(self, key, dtype=dtype)

    to_maxtext.LazyHFLoader.get_tensor = _streaming_get_tensor

  scan_bool = "True" if args.scan_mode == "scanned" else "False"
  orbax_output_dir = f"{args.gcs_run_dir}/to_maxtext/{args.model}/{args.scan_mode}/orbax_ckpt"
  epath.Path(orbax_output_dir).rmtree(missing_ok=True)
  token = get_hf_token()
  loader_flags = get_loader_flags(args.model)
  lazy_load = "--lazy_load_tensors=True" in loader_flags
  eager_method = "transformers" if "--eager_load_method=transformers" in loader_flags else "safetensors"

  model_args = [
      "to_maxtext.py",
      "src/maxtext/configs/base.yml",
      f"model_name={args.model}",
      f"base_output_directory={orbax_output_dir}",
      f"hf_access_token={token}",
      "hardware=cpu",
      "skip_jax_distributed_system=True",
      f"scan_layers={scan_bool}",
      f"checkpoint_storage_concurrent_gb={get_concurrent_gb(args.model, args.scan_mode, 1)}",
      *(["attention=dot_product"] if args.model.startswith(("deepseek3.2", "deepseek4")) else []),
  ]
  sys.argv = list(model_args)

  print(f"[to_maxtext Step 2] Converting {args.model} ({args.scan_mode}) -> {orbax_output_dir} ...", flush=True)
  if recorder is not None:
    recorder.record_value("to_maxtext_step2_startup_and_config_seconds", time.perf_counter() - t_step2_entry)
    if "hf_to_local_restage_s" in sub_timings:
      recorder.record_value("to_maxtext_step2_recopy_local_files_seconds", sub_timings["hf_to_local_restage_s"])
  t0 = time.perf_counter()
  with recorder.stage("to_maxtext_step2_total_seconds", log_start=True) if recorder else contextlib.nullcontext():
    to_maxtext.main(
        args=model_args,
        lazy_load_tensors=lazy_load,
        eager_load_method=eager_method,
        hf_model_path=args.local_hf_dir,
        save_dtype="bfloat16",
        simulated_cpu_devices_count=16,
    )
  convert_to_orbax_s = time.perf_counter() - t0
  if recorder is not None and sub_timings.get("stream_download_gb"):
    recorder.record_value("to_maxtext_step2_stream_files_from_gcs_gb", round(sub_timings["stream_download_gb"], 4))
  header_info = harness_v2.inspect_gcs_or_local_safetensors(args.local_hf_dir)
  if not header_info and os.path.exists(gcs_marker):
    with open(gcs_marker, "r", encoding="utf-8") as gf:
      header_info = harness_v2.inspect_gcs_or_local_safetensors(gf.read().strip())
  print(f"[to_maxtext Step 2] Convert to Orbax checkpoint completed in {convert_to_orbax_s:.2f}s", flush=True)

  return {
      "convert_to_orbax_s": convert_to_orbax_s,
      "orbax_items_path": f"{orbax_output_dir}/0/items",
      "header_info": header_info,
      "pipeline_config": {
          "to_maxtext_orbax_save_dtype": "bfloat16",
          "checkpoint_storage_concurrent_gb": get_concurrent_gb(args.model, args.scan_mode, 1),
      },
      **sub_timings,
  }


def find_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    return int(s.getsockname()[1])


def _find_libtpu_path() -> str | None:
  """Returns the path to `libtpu.so` if configured or installed, without importing JAX."""
  env_path = os.environ.get("TPU_LIBRARY_PATH")
  if env_path and os.path.isfile(env_path):
    return env_path
  try:
    import libtpu  # pylint: disable=import-outside-toplevel

    lib_path = libtpu.get_library_path()
    if lib_path and os.path.isfile(lib_path):
      return lib_path
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return None


def _libtpu_supports_flag(libtpu_path: str, flag_name: str) -> bool:
  """Checks via mmap whether `libtpu_path` contains gflag `flag_name`."""
  if not libtpu_path or not os.path.isfile(libtpu_path) or os.path.getsize(libtpu_path) == 0:
    return False
  needle = flag_name.encode("ascii")
  try:
    with open(libtpu_path, "rb") as f:
      with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.find(needle) != -1
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _ensure_tpu_insecure_grpc_init_args(num_slices: int, total_procs: int) -> list[str]:
  """Appends multi-host / multi-slice insecure gRPC flags to `LIBTPU_INIT_ARGS` when needed."""
  # GKE Workload Identity pods lack GCE metadata ALTS credentials between SliceBuilder/MegaScale peers.
  if total_procs <= 1 or os.environ.get("KEEP_ALTS_GRPC") == "1":
    return []
  libtpu_path = _find_libtpu_path()
  if not libtpu_path:
    return []
  existing_raw = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
  try:
    existing_tokens = shlex.split(existing_raw) if existing_raw else []
  except ValueError:
    existing_tokens = existing_raw.split()

  desired_flags = ["slicebuilder_use_insecure_grpc"]
  if num_slices > 1:
    desired_flags.append("megascale_use_insecure_grpc")

  added: list[str] = []
  for flag_name in desired_flags:
    if any(tok == f"--{flag_name}" or tok.startswith(f"--{flag_name}=") for tok in existing_tokens):
      continue
    if _libtpu_supports_flag(libtpu_path, flag_name):
      added.append(f"--{flag_name}=true")

  if added:
    os.environ["LIBTPU_INIT_ARGS"] = " ".join([existing_raw, *added]).strip()
  return added


def init_worker_runtime(args: argparse.Namespace) -> str:
  """Initializes JAX platform and (when multi-slice or multi-host) JAX distributed runtime."""
  harness_v2, _, _, _ = _import_bench_modules()
  recorder = getattr(args, "_recorder", None)
  sampler = getattr(args, "_sampler", None)
  mon = getattr(args, "_monitoring", None)
  hw = resolve_hardware(args.hardware)
  num_slices = getattr(args, "num_slices", 1)
  total_procs = num_slices * getattr(args, "num_hosts_per_slice", 1)
  if hw == "tpu":
    if total_procs > 1:
      added_flags = _ensure_tpu_insecure_grpc_init_args(num_slices=num_slices, total_procs=total_procs)
      if added_flags and getattr(args, "slice_id", 0) == 0:
        print(f"[TPU runtime] Added LIBTPU_INIT_ARGS: {' '.join(added_flags)}", flush=True)
    if num_slices > 1:
      coord_ip = args.coordinator_address.split(":")[0]
      slice_idx = args.slice_id // max(1, getattr(args, "num_hosts_per_slice", 1))
      os.environ["MEGASCALE_NUM_SLICES"] = str(args.num_slices)
      os.environ["MEGASCALE_SLICE_ID"] = str(slice_idx)
      os.environ["MEGASCALE_PORT"] = "8477"
      os.environ["MEGASCALE_COORDINATOR_ADDRESS"] = f"{coord_ip}:8477"
    elif total_procs == 1:
      for _ms_key in ("MEGASCALE_NUM_SLICES", "MEGASCALE_SLICE_ID", "MEGASCALE_PORT", "MEGASCALE_COORDINATOR_ADDRESS"):
        os.environ.pop(_ms_key, None)

  with recorder.stage("setup_python_imports_seconds") if recorder else contextlib.nullcontext():
    import jax
    from jax.experimental import multihost_utils
    from maxtext.utils import maxtext_utils
    from maxtext.utils import model_creation_utils

  if recorder is not None:
    from bench import stages as _stages

    _stages.wrap_sync_global_devices(multihost_utils, recorder)
    harness_v2.install_setup_hooks(recorder, maxtext_utils_mod=maxtext_utils, model_creation_mod=model_creation_utils)
  if mon is not None:
    mon.register(jax.monitoring)
  if sampler is not None and hw == "tpu":
    sampler._jax_mod = jax  # pylint: disable=protected-access

  try:
    from jax.experimental import hijax as _hjx

    if not hasattr(_hjx, "HiPrimitive") and hasattr(_hjx, "Primitive"):
      _hjx.HiPrimitive = _hjx.Primitive
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  if hw == "cpu":
    jax.config.update("jax_platforms", "cpu")
  if total_procs > 1:
    with (
        recorder.stage("setup_jax_distributed_initialize_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      jax.distributed.initialize(
          coordinator_address=args.coordinator_address,
          num_processes=total_procs,
          process_id=args.slice_id,
      )
    print(
        f"[Process {args.slice_id}/{total_procs} (num_slices={args.num_slices}, hosts_per_slice={getattr(args, 'num_hosts_per_slice', 1)})] "
        f"JAX distributed initialized: process_index={jax.process_index()}, process_count={jax.process_count()}, "
        f"local_devices={jax.local_device_count()}, global_devices={jax.device_count()}",
        flush=True,
    )
  with recorder.stage("setup_tpu_backend_initialize_seconds", log_start=True) if recorder else contextlib.nullcontext():
    _ = jax.default_backend()
    _ = jax.local_devices()
    _ = jax.device_count()
  _register_transformers_fallbacks()
  return hw


def _get_parallelism_flags(args: argparse.Namespace, concurrent_gb: int) -> list[str]:
  """Builds MaxText parallelism CLI flags for single-host, multi-host single-slice, and multi-slice."""
  num_slices = getattr(args, "num_slices", 1)
  num_hosts_per_slice = getattr(args, "num_hosts_per_slice", 1)
  total_procs = num_slices * num_hosts_per_slice
  dcn_axis = resolve_dcn_axis(args.model, getattr(args, "dcn_parallelism_axis", "auto"))
  ici_expert = int(os.environ.get("ICI_EXPERT_PARALLELISM", "1"))
  ici_tensor = int(os.environ.get("ICI_TENSOR_PARALLELISM", "1"))
  extra_ici = []
  if ici_expert > 1:
    extra_ici.append(f"ici_expert_parallelism={ici_expert}")
  if ici_tensor > 1:
    extra_ici.append(f"ici_tensor_parallelism={ici_tensor}")
  if num_slices > 1:
    return [
        "skip_jax_distributed_system=False",
        f"num_slices={num_slices}",
        f"{dcn_axis}={num_slices}",
        "ici_fsdp_parallelism=-1",
        f"checkpoint_storage_concurrent_gb={concurrent_gb}",
        *extra_ici,
    ]
  if total_procs > 1:
    return [
        "skip_jax_distributed_system=False",
        "num_slices=1",
        "ici_fsdp_parallelism=-1",
        f"checkpoint_storage_concurrent_gb={concurrent_gb}",
        *extra_ici,
    ]
  return [
      "skip_jax_distributed_system=True",
      "ici_fsdp_parallelism=-1",
      f"checkpoint_storage_concurrent_gb={concurrent_gb}",
      *extra_ici,
  ]


def run_worker_to_maxtext_step3_load_orbax(args: argparse.Namespace) -> dict[str, Any]:
  """Step 3 for `to_maxtext`: Load the converted Orbax checkpoint from GCS into MaxText."""
  harness_v2, _, schema_v2, _ = _import_bench_modules()
  recorder = getattr(args, "_recorder", None)
  sampler = getattr(args, "_sampler", None)
  import jax

  hw = init_worker_runtime(args)
  with recorder.stage("setup_tpu_check_seconds", log_start=True) if recorder else contextlib.nullcontext():
    tpu_info = _verify_and_measure_jax_backend(args, jax)
  with recorder.stage("setup_python_imports_seconds") if recorder else contextlib.nullcontext():
    from flax.core.spmd import logical_axis_rules
    from maxtext.common import checkpointing
    from maxtext.configs import pyconfig
    from maxtext.utils import globals as maxtext_globals
    import orbax.checkpoint as ocp

    ocp_v1 = _import_ocp_v1()

  if recorder is not None:
    harness_v2.install_to_maxtext_step3_hooks(recorder, checkpointing_mod=checkpointing, ocp_v1_mod=ocp_v1, ocp_mod=ocp)

  scan_bool = "True" if args.scan_mode == "scanned" else "False"
  orbax_items_path = f"{args.gcs_run_dir}/to_maxtext/{args.model}/{args.scan_mode}/orbax_ckpt/0/items"
  token = get_hf_token()
  num_slices = getattr(args, "num_slices", 1)
  slice_suffix = f"_slices{num_slices}" if num_slices > 1 else ""
  if getattr(args, "num_hosts_per_slice", 1) > 1:
    slice_suffix += f"_hosts{args.num_hosts_per_slice}"

  concurrent_gb = get_concurrent_gb(args.model, args.scan_mode, num_slices)
  multi_slice_flags = _get_parallelism_flags(args, concurrent_gb)
  weight_dtype_flag = (
      [f"weight_dtype={args.weight_dtype}"]
      if getattr(args, "weight_dtype", "auto") not in ("", "auto", "default")
      else ([] if args.model in ("qwen3-0.6b", "gemma2-2b") else ["weight_dtype=bfloat16"])
  )

  config_argv = (
      [
          "benchmark_load_orbax.py",
          "src/maxtext/configs/base.yml",
          "run_name=benchmark_load_orbax",
          f"model_name={args.model}",
          f"scan_layers={scan_bool}",
          f"hardware={hw}",
          "enable_checkpointing=False",
          "async_checkpointing=False",
          "per_device_batch_size=1",
          "dataset_type=synthetic",
          "source_checkpoint_layout=orbax",
          f"load_parameters_path={orbax_items_path}",
          f"base_output_directory={args.gcs_run_dir}/to_maxtext/{args.model}/{args.scan_mode}{slice_suffix}/load_run",
          f"hf_access_token={token}",
      ]
      + weight_dtype_flag
      + multi_slice_flags
      + get_tokenizer_flags(args.model, maxtext_globals.HF_IDS)
  )

  t_init0 = time.perf_counter()
  with recorder.stage("setup_maxtext_config_build_seconds", log_start=True) if recorder else contextlib.nullcontext():
    config = pyconfig.initialize(config_argv)
  mesh, unboxed_abstract_state = build_abstract_state_for_load(config)
  init_abstract_state_s = time.perf_counter() - t_init0
  print(
      f"[to_maxtext Step 3] Abstract state initialized in {init_abstract_state_s:.2f}s "
      f"(num_slices={config.num_slices}, global_devices={jax.device_count()}, mesh_shape={mesh.shape})",
      flush=True,
  )

  try:
    from orbax.checkpoint.experimental.v1._src.synchronization import multihost as _ocp_mh

    _ocp_mh._DEFAULT_BARRIER_TIMEOUT = max(
        getattr(_ocp_mh, "_DEFAULT_BARRIER_TIMEOUT", 300),
        int(os.environ.get("ORBAX_BARRIER_TIMEOUT_S", "1800")),
    )
  except ImportError:
    pass
  if jax.process_count() > 1:
    from jax.experimental import multihost_utils

    with (
        recorder.stage("setup_wait_for_all_hosts_before_load_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      multihost_utils.sync_global_devices("before_load")
  if recorder is not None:
    proc_start_s = (
        recorder.snapshot()
        .get("stages", {})
        .get("setup_process_start_to_harness_code_seconds", {})
        .get("inclusive_s", 0.0)
    )
    recorder.record_value("setup_total_before_load_seconds", proc_start_s + max(0.0, time.perf_counter() - _ENTER_MONO))

  print(f"[to_maxtext Step 3] Loading Orbax checkpoint from {orbax_items_path} ...", flush=True)
  t_load0 = time.perf_counter()
  with recorder.stage("to_maxtext_step3_total_seconds", log_start=True) if recorder else contextlib.nullcontext():
    with logical_axis_rules(config.logical_axis_rules):
      _, raw_params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path=config.load_parameters_path,
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          abstract_unboxed_pre_state=unboxed_abstract_state,
          enable_single_replica_ckpt_restoring=False,
          dataset_type=config.dataset_type,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
          enable_orbax_v1=config.enable_orbax_v1,
          checkpoint_conversion_fn=config.checkpoint_conversion_fn,
          source_checkpoint_layout="orbax",
          expansion_factor_real_data=config.expansion_factor_real_data,
          maxtext_config=config,
      )
      with recorder.stage("to_maxtext_step3_post_load_checks_seconds") if recorder else contextlib.nullcontext():
        jax.block_until_ready(raw_params)
    if jax.process_count() > 1:
      from jax.experimental import multihost_utils

      with recorder.stage("to_maxtext_step3_final_check_and_sync_seconds") if recorder else contextlib.nullcontext():
        multihost_utils.sync_global_devices("orbax_step3_load_complete")
  load_orbax_s = time.perf_counter() - t_load0

  if raw_params is None:
    raise RuntimeError(f"Failed to restore parameters from Orbax checkpoint at {orbax_items_path}")
  if sampler is not None:
    sampler.record_hbm_milestone("after_load", jax)

  num_leaves = len(jax.tree_util.tree_leaves(raw_params))
  bytes_read_gb = harness_v2.compute_local_shard_gb(raw_params)
  model_tier = next(
      (m.get("tier", "") for m in schema_v2.load_models_v2()["models"] if m["model_name"] == args.model), ""
  )
  want_cs = getattr(args, "verify_checksums", False) or (
      os.environ.get("BENCH_CHECK_CHECKSUMS", "auto") == "1"
      or (os.environ.get("BENCH_CHECK_CHECKSUMS", "auto") == "auto" and model_tier == "A")
  )
  checksums = None
  if want_cs:
    with recorder.stage("post_load_checksum_verify_seconds") if recorder else contextlib.nullcontext():
      checksums = harness_v2.compute_parameter_checksums(raw_params, jax)

  pipeline_config = {
      "device_mesh_shape": str(mesh.shape),
      "ici_fsdp_parallelism": getattr(config, "ici_fsdp_parallelism", -1),
      "dcn_parallelism_axis": resolve_dcn_axis(args.model, getattr(args, "dcn_parallelism_axis", "auto")),
      "dcn_parallelism_size": getattr(config, "dcn_fsdp_parallelism", 1)
      if getattr(config, "dcn_fsdp_parallelism", 1) > 1
      else getattr(config, "dcn_data_parallelism", 1),
      "checkpoint_storage_concurrent_gb": concurrent_gb,
      "maxtext_load_weight_dtype": str(getattr(config, "weight_dtype", "")),
      "harness_environment_flags": ",".join(
          f"{k}={os.environ[k]}"
          for k in sorted(os.environ)
          if k.startswith(("BENCH_", "ORBAX_", "PHASE_TIMEOUT", "SAFETENSORS_"))
      ),
  }

  print(
      f"[to_maxtext Step 3] Loaded Orbax checkpoint ({num_leaves} parameter leaves, "
      f"num_slices={num_slices}, global_devices={jax.device_count()}) in {load_orbax_s:.2f}s",
      flush=True,
  )
  return {
      "load_orbax_s": load_orbax_s,
      "init_abstract_state_s": init_abstract_state_s,
      "num_param_leaves": num_leaves,
      "bytes_read_gb": bytes_read_gb,
      "num_slices": num_slices,
      "global_devices": jax.device_count(),
      "pipeline_config": pipeline_config,
      "parameter_checksums": checksums,
      **tpu_info,
  }


_HF_DOWNLOAD_MAX_WORKERS = 16


class _DelegatingProxy:
  """Forwards attributes to a wrapped module while overriding specific names."""

  def __init__(self, target: Any, **overrides: Any):
    self._target = target
    self._overrides = overrides

  def __getattr__(self, name: str) -> Any:
    if name in self._overrides:
      return self._overrides[name]
    return getattr(self._target, name)


@contextlib.contextmanager
def _cap_load_dynamic_download_workers(load_dynamic_mod: Any, max_workers: int = _HF_DOWNLOAD_MAX_WORKERS):
  """Caps `load_dynamic.concurrent.futures.ProcessPoolExecutor` max_workers without modifying src/."""
  orig_concurrent = load_dynamic_mod.concurrent
  orig_futures = orig_concurrent.futures
  orig_pool_cls = orig_futures.ProcessPoolExecutor

  def _capped_pool(*args: Any, **kwargs: Any):
    if "max_workers" in kwargs:
      mw = kwargs["max_workers"]
      kwargs["max_workers"] = max_workers if mw is None else min(mw, max_workers)
    elif args:
      mw = args[0]
      args = (max_workers if mw is None else min(mw, max_workers), *args[1:])
    else:
      kwargs["max_workers"] = max_workers
    return orig_pool_cls(*args, **kwargs)

  capped_futures = _DelegatingProxy(orig_futures, ProcessPoolExecutor=_capped_pool)
  load_dynamic_mod.concurrent = _DelegatingProxy(orig_concurrent, futures=capped_futures)
  try:
    yield
  finally:
    load_dynamic_mod.concurrent = orig_concurrent


def run_worker_safetensors_dynamic(args: argparse.Namespace) -> dict[str, Any]:
  """Runs `safetensors_dynamic` and records (1) HF -> GCS and (2) safetensors and convert to MaxText."""
  harness_v2, _, schema_v2, _ = _import_bench_modules()
  recorder = getattr(args, "_recorder", None)
  sampler = getattr(args, "_sampler", None)
  import jax

  hw = init_worker_runtime(args)
  with recorder.stage("setup_tpu_check_seconds", log_start=True) if recorder else contextlib.nullcontext():
    tpu_info = _verify_and_measure_jax_backend(args, jax)
  with recorder.stage("setup_python_imports_seconds") if recorder else contextlib.nullcontext():
    from flax.core.spmd import logical_axis_rules
    import jax.numpy as jnp
    import numpy as np
    from maxtext.checkpoint_conversion.utils import load_dynamic
    from maxtext.checkpoint_conversion.utils import tensor_handling
    from maxtext.common import checkpointing
    from maxtext.configs import pyconfig
    from maxtext.utils import globals as maxtext_globals

    ocp_v1 = _import_ocp_v1()

  import ctypes
  import gc

  def _trim_malloc():
    gc.collect()
    try:
      ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
      pass

  try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.INFO)
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  load_dynamic.build_gcs_cache_worker = fast_build_gcs_cache_worker

  from etils import epath

  scan_bool = "True" if args.scan_mode == "scanned" else "False"
  repo_id = maxtext_globals.HF_IDS[args.model]
  token = get_hf_token()
  num_slices = getattr(args, "num_slices", 1)
  slice_suffix = f"_slices{num_slices}" if num_slices > 1 else ""
  if getattr(args, "num_hosts_per_slice", 1) > 1:
    slice_suffix += f"_hosts{args.num_hosts_per_slice}"
  shared_cache_base = os.environ.get("SHARED_HF_CACHE_BASE_DIR", "").strip()
  if shared_cache_base:
    base_out_dir = f"{shared_cache_base.rstrip('/')}/{args.model}/shared"
    should_clean_cache = False
  elif getattr(args, "reuse_model_hf_cache", False):
    base_out_dir = f"{args.gcs_run_dir}/safetensors_dynamic/{args.model}/shared"
    tm_step1_json = epath.Path(f"{args.gcs_run_dir}/logs/{args.model}/scanned/to_maxtext_step1_result.json")
    tm_already_cached = False
    if tm_step1_json.exists():
      try:
        tm_prev = json.loads(tm_step1_json.read_text())
        if tm_prev.get("status") != "FAILED" and tm_prev.get("hf_to_gcs_s", 0.0) > 0:
          tm_already_cached = True
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    should_clean_cache = (
        args.scan_mode == "scanned"
        and num_slices == 1
        and not tm_already_cached
        and os.environ.get("KEEP_HF_CACHE", "0") != "1"
    )
  else:
    base_out_dir = f"{args.gcs_run_dir}/safetensors_dynamic/{args.model}/{args.scan_mode}{slice_suffix}"
    should_clean_cache = os.environ.get("KEEP_HF_CACHE", "0") != "1"
  if should_clean_cache or jax.process_count() > 1:
    with recorder.stage("setup_delete_old_gcs_cache_seconds", log_start=True) if recorder else contextlib.nullcontext():
      if jax.process_index() == 0 and should_clean_cache:
        epath.Path(f"{base_out_dir}/hf_cache").rmtree(missing_ok=True)
      if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("safetensors_clean_cache_complete")

  concurrent_gb = get_concurrent_gb(args.model, args.scan_mode, num_slices)
  multi_slice_flags = _get_parallelism_flags(args, concurrent_gb)
  weight_dtype_flag = (
      [f"weight_dtype={args.weight_dtype}"]
      if getattr(args, "weight_dtype", "auto") not in ("", "auto", "default")
      else ([] if args.model in ("qwen3-0.6b", "gemma2-2b") else ["weight_dtype=bfloat16"])
  )

  pre_staged_gcs_dir = os.environ.get("BENCH_SAFETENSORS_GCS_DIR", "").strip()
  if _should_stage_sparse_bf16(args.model):
    local_sparse_dir = f"/tmp/hf_local_{args.model}"
    num_st, num_t, tot_gb = _stage_sparse_bf16_safetensors(args.model, local_sparse_dir)
    print(
        f"[safetensor] Ensured sparse BF16 .safetensors for {args.model} at {local_sparse_dir} "
        f"({num_st} shards, {num_t} tensors, {tot_gb:.2f} GB BF16)",
        flush=True,
    )
    load_params_uri = local_sparse_dir
  elif pre_staged_gcs_dir:
    load_params_uri = pre_staged_gcs_dir
  else:
    load_params_uri = (
        args.local_hf_dir
        if os.environ.get("USE_LOCAL_HF_FOR_SAFETENSORS") == "1"
        and jax.process_count() == 1
        and os.path.isdir(args.local_hf_dir)
        else f"hf://{repo_id}"
    )
  config_argv = (
      [
          "benchmark_safetensors_dynamic.py",
          "src/maxtext/configs/base.yml",
          "run_name=benchmark_safetensors_dynamic",
          f"model_name={args.model}",
          f"scan_layers={scan_bool}",
          f"hardware={hw}",
          "enable_checkpointing=False",
          "async_checkpointing=False",
          "per_device_batch_size=1",
          "dataset_type=synthetic",
          "source_checkpoint_layout=safetensors_dynamic",
          f"load_parameters_path={load_params_uri}",
          f"base_output_directory={base_out_dir}",
          f"hf_access_token={token}",
      ]
      + weight_dtype_flag
      + multi_slice_flags
      + get_tokenizer_flags(args.model, maxtext_globals.HF_IDS)
  )

  t_init0 = time.perf_counter()
  with recorder.stage("setup_maxtext_config_build_seconds", log_start=True) if recorder else contextlib.nullcontext():
    config = pyconfig.initialize(config_argv)
  mesh, unboxed_abstract_state = build_abstract_state_for_load(config)
  init_abstract_state_s = time.perf_counter() - t_init0
  print(
      f"[safetensor] Abstract state initialized in {init_abstract_state_s:.2f}s "
      f"(num_slices={config.num_slices}, global_devices={jax.device_count()}, mesh_shape={mesh.shape})",
      flush=True,
  )

  header_info: dict[str, Any] = {}

  if jax.process_count() > 1:
    from jax.experimental import multihost_utils

    with (
        recorder.stage("setup_wait_for_all_hosts_before_load_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      multihost_utils.sync_global_devices("before_load")
  if recorder is not None:
    proc_start_s = (
        recorder.snapshot()
        .get("stages", {})
        .get("setup_process_start_to_harness_code_seconds", {})
        .get("inclusive_s", 0.0)
    )
    recorder.record_value("setup_total_before_load_seconds", proc_start_s + max(0.0, time.perf_counter() - _ENTER_MONO))

  timings: dict[str, float] = {}
  slowest_param = (
      harness_v2.install_safetensors_dynamic_hooks(
          recorder,
          load_dynamic_mod=load_dynamic,
          tensor_handling_mod=tensor_handling,
          ocp_v1_mod=ocp_v1,
          jax_mod=jax,
      )
      if recorder is not None
      else {"slowest_s": 0.0, "slowest_name": ""}
  )

  orig_get_mappings = load_dynamic.get_hf_config_and_mappings
  orig_load_sharded = load_dynamic.load_sharded_hf_state
  orig_transform = load_dynamic.transform_hf_state_to_mt_state

  def timed_get_mappings(*a, **kw):
    # Called immediately after Step 1 (HF -> GCS caching + barrier) finishes in load_safetensors_dynamic_state
    timings["t_step1_end"] = time.perf_counter()
    step1_elapsed = timings["t_step1_end"] - t_start
    print(f"[safetensor] Step 1 (HF->GCS) completed in {step1_elapsed:.2f}s", flush=True)
    if recorder is not None:
      recorder.record_value("safetensors_dynamic_step1_total_seconds", step1_elapsed)
      wait_s = recorder.snapshot().get("barriers", {}).get("dynamic_hf_download_complete", {}).get("total_s", 0.0)
      recorder.record_value("safetensors_dynamic_step1_other_hosts_wait_seconds", wait_s)
      recorder.record_value("safetensors_dynamic_step1_verify_gcs_cache_seconds", max(0.0, step1_elapsed - wait_s))
    t0 = time.perf_counter()
    with (
        recorder.stage("safetensors_dynamic_step2_build_hf_to_maxtext_mappings_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      res = orig_get_mappings(*a, **kw)
    timings["derive_mappings_s"] = time.perf_counter() - t0
    return res

  def timed_load_sharded(path, *a, **kw):
    t0 = time.perf_counter()
    with (
        recorder.stage("safetensors_dynamic_step2_load_total_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      res = _timed_load_sharded_inner(path, *a, **kw)
    timings["load_sharded_hf_s"] = time.perf_counter() - t0
    if recorder is not None:
      needed_gb = harness_v2.compute_local_shard_gb(res)
      recorder.add_metric("safetensors_dynamic_step2_load_total_seconds", "bytes_needed_by_this_host_gb", needed_gb)
      timings["bytes_needed_by_this_host_gb"] = needed_gb
    if sampler is not None:
      sampler.record_hbm_milestone("after_read", jax)
    print(f"[safetensor] load_sharded_hf_state completed in {timings['load_sharded_hf_s']:.2f}s", flush=True)
    return res

  def _timed_load_sharded_inner(path, *a, **kw):
    use_custom_byte_reader = os.environ.get("CUSTOM_BYTE_READER", "0") == "1"
    if use_custom_byte_reader:
      import collections
      import itertools
      import math
      import struct

      st_dir = epath.Path(path)
      st_files = sorted(st_dir.glob("*.safetensors")) if st_dir.is_dir() else [st_dir]
      if not st_files:
        raise RuntimeError(f"No .safetensors files found at {path}")

      def _read_st_hdr(fp):
        with fp.open("rb") as f:
          raw8 = f.read(8)
          hlen = struct.unpack("<Q", raw8)[0]
          hdr = json.loads(f.read(hlen).decode("utf-8"))
        return fp, hdr, 8 + hlen

      with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(st_files)))) as ex:
        file_entries = list(ex.map(_read_st_hdr, st_files))

      st_dtypes = {
          "BOOL": np.bool_,
          "I8": np.int8,
          "U8": np.uint8,
          "I16": np.int16,
          "U16": np.uint16,
          "I32": np.int32,
          "U32": np.uint32,
          "I64": np.int64,
          "U64": np.uint64,
          "F16": np.float16,
          "F32": np.float32,
          "F64": np.float64,
          "BF16": jnp.bfloat16,
          "F8_E5M2": jnp.float8_e5m2,
          "F8_E4M3": jnp.float8_e4m3fn,
      }
      fsdp_size = mesh.shape.get("fsdp", 1)
      this_process = jax.process_index()
      is_sparse_oom = os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1"
      dcn_axis = resolve_dcn_axis(args.model, getattr(args, "dcn_parallelism_axis", "auto"))

      def _make_sharded_spec(k: str, shp: tuple[int, ...]) -> list[str | None]:
        spec: list[str | None] = [None] * len(shp)
        if fsdp_size <= 1:
          return spec
        if dcn_axis == "dcn_data_parallelism" or is_sparse_oom:
          for ax, dim in enumerate(shp):
            if dim % fsdp_size == 0 and (not k.endswith("gate_up_proj") or ax < len(shp) - 1):
              spec[ax] = "fsdp"
              break
        else:
          chosen_ax = None
          if len(shp) == 3 and k.endswith("gate_up_proj") and shp[1] % fsdp_size == 0:
            chosen_ax = 1
          elif len(shp) == 3 and k.endswith("down_proj") and shp[2] % fsdp_size == 0:
            chosen_ax = 2
          elif (
              len(shp) == 2
              and k.endswith(
                  (
                      "q_proj.weight",
                      "k_proj.weight",
                      "v_proj.weight",
                      "gate_proj.weight",
                      "up_proj.weight",
                      "w1.weight",
                      "w3.weight",
                      "embed_tokens.weight",
                      "wq_a.weight",
                      "wkv.weight",
                  )
              )
              and shp[1] % fsdp_size == 0
          ):
            chosen_ax = 1
          elif (
              len(shp) == 2 and k.endswith(("o_proj.weight", "down_proj.weight", "w2.weight")) and shp[0] % fsdp_size == 0
          ):
            chosen_ax = 0
          else:
            for ax, dim in enumerate(shp):
              if dim % fsdp_size == 0 and (not k.endswith("gate_up_proj") or ax < len(shp) - 1):
                chosen_ax = ax
                break
          if chosen_ax is not None:
            spec[chosen_ax] = "fsdp"
        return spec

      def _index_domain_to_byte_runs(bounds, global_shape, itemsize, tensor_base):
        ndim = len(global_shape)
        if ndim == 0:
          return [(tensor_base, itemsize)]
        strides = [itemsize] * ndim
        acc = itemsize
        for i in range(ndim - 1, -1, -1):
          strides[i] = acc
          acc *= global_shape[i]
        full_trailing = 0
        for i in range(ndim - 1, -1, -1):
          if bounds[i] == (0, global_shape[i]):
            full_trailing += 1
          else:
            break
        boundary = ndim - full_trailing - 1
        if boundary < 0:
          return [(tensor_base, strides[0] * global_shape[0])]
        run_length = (bounds[boundary][1] - bounds[boundary][0]) * strides[boundary]
        run_base = tensor_base + bounds[boundary][0] * strides[boundary]
        outer_ranges = [range(bounds[i][0], bounds[i][1]) for i in range(boundary)]
        runs = []
        for outer_idx in itertools.product(*outer_ranges):
          off = run_base
          for d, coord in enumerate(outer_idx):
            off += coord * strides[d]
          runs.append((off, run_length))
        runs.sort()
        return runs

      res: dict[str, Any] = {}
      max_batch_bytes = 12 * 10**9
      chunk_cap = 64 * 1024 * 1024

      for fp, hdr, data_start in file_entries:
        tensor_items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
        batches = []
        cur_batch = []
        cur_bytes = 0
        for k, info in tensor_items:
          t_bytes = max(1, (info["data_offsets"][1] - info["data_offsets"][0]) // max(1, jax.process_count()))
          if cur_batch and cur_bytes + t_bytes > max_batch_bytes:
            batches.append(cur_batch)
            cur_batch = []
            cur_bytes = 0
          cur_batch.append((k, info))
          cur_bytes += t_bytes
        if cur_batch:
          batches.append(cur_batch)

        for batch in batches:
          batch_reads = []  # (offset, length, dst_u8_slice)
          batch_builders = []
          for k, info in batch:
            shp = tuple(int(x) for x in info["shape"])
            dt = np.dtype(st_dtypes[info["dtype"]])
            spec = _make_sharded_spec(k, shp)
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
            tensor_base = data_start + int(info["data_offsets"][0])
            idx_to_devs = collections.defaultdict(list)
            for dev, idx in sharding.devices_indices_map(shp).items():
              if dev.process_index == this_process:
                bnds = tuple(
                    (0 if s.start is None else int(s.start), shp[i] if s.stop is None else int(s.stop))
                    for i, s in enumerate(idx)
                )
                idx_to_devs[bnds].append(dev)
            shard_specs = []
            for bnds, devs in idx_to_devs.items():
              shard_shape = tuple(stop - start for start, stop in bnds)
              numel = math.prod(shard_shape) if shard_shape else 1
              flat_u8 = np.empty(numel * dt.itemsize, dtype=np.uint8)
              shard_buf = flat_u8.view(dt).reshape(shard_shape)
              runs = _index_domain_to_byte_runs(bnds, shp, dt.itemsize, tensor_base)
              w_off = 0
              for r_off, r_len in runs:
                batch_reads.append((r_off, r_len, flat_u8[w_off : w_off + r_len]))
                w_off += r_len
              shard_specs.append((shard_buf, devs))

            def _make_builder(t_name, t_shp, t_dt, t_sharding, t_specs):
              def _build():
                arrs = []
                while t_specs:
                  s_buf, s_devs = t_specs.pop(0)
                  for d in s_devs:
                    arrs.append(jax.device_put(s_buf, jax.sharding.SingleDeviceSharding(d)))
                  del s_buf
                res[t_name] = jax.make_array_from_single_device_arrays(t_shp, t_sharding, arrs, dtype=t_dt)

              return _build

            batch_builders.append(_make_builder(k, shp, dt, sharding, shard_specs))

          if batch_reads:
            batch_reads.sort(key=lambda r: r[0])
            coalesced = []
            c_start, c_end, c_members = batch_reads[0][0], batch_reads[0][0] + batch_reads[0][1], [batch_reads[0]]
            for r in batch_reads[1:]:
              r_start, r_end = r[0], r[0] + r[1]
              if r_start - c_end <= 65536:
                c_end = max(c_end, r_end)
                c_members.append(r)
              else:
                coalesced.append((c_start, c_end, c_members))
                c_start, c_end, c_members = r_start, r_end, [r]
            coalesced.append((c_start, c_end, c_members))

            chunk_tasks = []
            for b_start, b_end, members in coalesced:
              m_idx = 0
              for ck_start in range(b_start, b_end, chunk_cap):
                ck_end = min(ck_start + chunk_cap, b_end)
                while m_idx < len(members) and members[m_idx][0] + members[m_idx][1] <= ck_start:
                  m_idx += 1
                writes = []
                for j in range(m_idx, len(members)):
                  r_off, r_len, r_dst = members[j]
                  if r_off >= ck_end:
                    break
                  lo = max(r_off, ck_start)
                  hi = min(r_off + r_len, ck_end)
                  if lo < hi:
                    writes.append((r_dst[lo - r_off : hi - r_off], lo - ck_start, hi - ck_start))
                chunk_tasks.append((ck_start, ck_end - ck_start, writes))

            def _read_ck(task, _fp=fp):
              ck_off, ck_len, ck_writes = task
              with _fp.open("rb") as rf:
                rf.seek(ck_off)
                data = rf.read(ck_len)
                if is_sparse_oom and hasattr(os, "posix_fadvise"):
                  try:
                    os.posix_fadvise(rf.fileno(), ck_off, ck_len, os.POSIX_FADV_DONTNEED)
                  except Exception:
                    pass
              mv = memoryview(data)
              for w_dst, w_lo, w_hi in ck_writes:
                w_dst[:] = np.frombuffer(mv[w_lo:w_hi], dtype=np.uint8)
              del mv, data

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(chunk_tasks)))) as ck_ex:
              list(ck_ex.map(_read_ck, chunk_tasks))
            del chunk_tasks, coalesced, batch_reads

          for b_fn in batch_builders:
            b_fn()
          del batch_builders
          _trim_malloc()
    else:
      if (
          hasattr(load_dynamic, "_leading_axis_shardings")
          and jax.process_count() == 1
          and not a
          and "num_groups" not in kw
      ):
        res = orig_load_sharded(path, num_groups=1)
      else:
        res = orig_load_sharded(path, *a, **kw)
    _trim_malloc()
    return res

  def timed_transform(*a, **kw):
    t0 = time.perf_counter()
    with (
        recorder.stage("safetensors_dynamic_step2_transform_total_seconds", log_start=True)
        if recorder
        else contextlib.nullcontext()
    ):
      res = orig_transform(*a, **kw)
    if recorder is not None:
      with recorder.stage("safetensors_dynamic_step2_harness_memory_cleanup_seconds"):
        _trim_malloc()
    else:
      _trim_malloc()
    timings["transform_hf_to_mt_s"] = time.perf_counter() - t0
    return res

  load_dynamic.get_hf_config_and_mappings = timed_get_mappings
  load_dynamic.load_sharded_hf_state = timed_load_sharded
  load_dynamic.transform_hf_state_to_mt_state = timed_transform

  print(f"[safetensor] Running load_state_if_possible(source_checkpoint_layout='safetensors_dynamic') ...", flush=True)
  t_start = time.perf_counter()
  with (
      recorder.stage("safetensors_dynamic_step2_total_seconds", log_start=True) if recorder else contextlib.nullcontext()
  ):
    with _cap_load_dynamic_download_workers(load_dynamic), logical_axis_rules(config.logical_axis_rules):
      _, raw_params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path=config.load_parameters_path,
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          abstract_unboxed_pre_state=unboxed_abstract_state,
          enable_single_replica_ckpt_restoring=False,
          dataset_type=config.dataset_type,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
          enable_orbax_v1=config.enable_orbax_v1,
          checkpoint_conversion_fn=config.checkpoint_conversion_fn,
          source_checkpoint_layout="safetensors_dynamic",
          expansion_factor_real_data=config.expansion_factor_real_data,
          maxtext_config=config,
      )
      with (
          recorder.stage("safetensors_dynamic_step2_final_weight_check_and_sync_seconds")
          if recorder
          else contextlib.nullcontext()
      ):
        jax.block_until_ready(raw_params)
    if jax.process_count() > 1:
      from jax.experimental import multihost_utils

      with (
          recorder.stage("safetensors_dynamic_step2_final_weight_check_and_sync_seconds")
          if recorder
          else contextlib.nullcontext()
      ):
        multihost_utils.sync_global_devices("safetensors_dynamic_load_complete")
  t_end = time.perf_counter()

  if raw_params is None:
    raise RuntimeError(f"Failed to restore parameters via safetensors_dynamic for {args.model}")
  if sampler is not None:
    sampler.record_hbm_milestone("after_load", jax)

  if recorder is not None and not header_info and jax.process_index() == 0:
    header_info = harness_v2.inspect_gcs_or_local_safetensors(
        load_params_uri
        if (load_params_uri.startswith("gs://") or os.path.isdir(load_params_uri))
        else f"{base_out_dir}/hf_cache/{repo_id.replace('/', '_')}"
    )

  hf_to_gcs_s = timings["t_step1_end"] - t_start
  safetensors_and_convert_s = t_end - timings["t_step1_end"]
  total_s = t_end - t_start
  num_leaves = len(jax.tree_util.tree_leaves(raw_params))

  model_tier = next(
      (m.get("tier", "") for m in schema_v2.load_models_v2()["models"] if m["model_name"] == args.model), ""
  )
  want_cs = getattr(args, "verify_checksums", False) or (
      os.environ.get("BENCH_CHECK_CHECKSUMS", "auto") == "1"
      or (os.environ.get("BENCH_CHECK_CHECKSUMS", "auto") == "auto" and model_tier == "A")
  )
  checksums = None
  if want_cs:
    with recorder.stage("post_load_checksum_verify_seconds") if recorder else contextlib.nullcontext():
      checksums = harness_v2.compute_parameter_checksums(raw_params, jax)

  pipeline_config = {
      "device_mesh_shape": str(mesh.shape),
      "ici_fsdp_parallelism": getattr(config, "ici_fsdp_parallelism", -1),
      "dcn_parallelism_axis": resolve_dcn_axis(args.model, getattr(args, "dcn_parallelism_axis", "auto")),
      "dcn_parallelism_size": getattr(config, "dcn_fsdp_parallelism", 1)
      if getattr(config, "dcn_fsdp_parallelism", 1) > 1
      else getattr(config, "dcn_data_parallelism", 1),
      "checkpoint_storage_concurrent_gb": concurrent_gb,
      "safetensors_read_concurrent_bytes": int(os.environ.get("SAFETENSORS_READ_CONCURRENT_BYTES", "0") or 0) or "",
      "maxtext_load_weight_dtype": str(getattr(config, "weight_dtype", "")),
      "harness_environment_flags": ",".join(
          f"{k}={os.environ[k]}"
          for k in sorted(os.environ)
          if k.startswith(("BENCH_", "ORBAX_", "PHASE_TIMEOUT", "SAFETENSORS_", "CUSTOM_BYTE_READER"))
      ),
  }

  print(
      f"[safetensor] Completed: (1) HF->GCS={hf_to_gcs_s:.2f}s, "
      f"(2) safetensors & convert to MaxText={safetensors_and_convert_s:.2f}s "
      f"(total={total_s:.2f}s, {num_leaves} parameter leaves, "
      f"num_slices={num_slices}, global_devices={jax.device_count()})",
      flush=True,
  )
  return {
      "hf_to_gcs_s": hf_to_gcs_s,
      "safetensors_and_convert_s": safetensors_and_convert_s,
      "derive_mappings_s": timings.get("derive_mappings_s", 0.0),
      "load_sharded_hf_s": timings.get("load_sharded_hf_s", 0.0),
      "transform_hf_to_mt_s": timings.get("transform_hf_to_mt_s", 0.0),
      "init_abstract_state_s": init_abstract_state_s,
      "bytes_needed_by_this_host_gb": timings.get("bytes_needed_by_this_host_gb"),
      "safetensors_dynamic_step2_slowest_parameter_seconds": round(slowest_param["slowest_s"], 4),
      "safetensors_dynamic_step2_slowest_parameter_name": slowest_param["slowest_name"],
      "num_param_leaves": num_leaves,
      "num_slices": num_slices,
      "global_devices": jax.device_count(),
      "header_info": header_info,
      "pipeline_config": pipeline_config,
      "parameter_checksums": checksums,
      **tpu_info,
  }


_SITE_CUSTOMIZE_BODY = """\
import dataclasses
import resource
import sys
import types

# 0. Disable core dumps so SIGABRT/OOM exits immediately without writing multi-GB core files
try:
  resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
  pass

# 1. Stub drjax if not installed
sys.modules.setdefault("drjax", types.ModuleType("drjax"))

# 2. HuggingFace transformers Config fallbacks for older transformers
try:
  import transformers
  from transformers.configuration_utils import PretrainedConfig as _PTConfig
  for _c in ("Qwen3OmniMoeConfig", "Qwen3NextConfig", "Olmo3Config"):
    if not hasattr(transformers, _c):
      setattr(transformers, _c, _PTConfig)
except Exception:
  pass

# 3. JAX / Tokamax compatibility shims for older local packages
try:
  import jax.experimental.xla_metadata as _xm
  if not hasattr(_xm, "must_fuse_call"):
    _xm.must_fuse_call = lambda f, *a, **kw: f(*a, **kw)
except Exception:
  pass

try:
  import tokamax._src.ops.attention as _ta
  if not hasattr(_ta, "pallas_triton"):
    from tokamax._src.ops.attention import pallas_triton_flash_attention as _ptfa
    _ta.pallas_triton = _ptfa
    sys.modules["tokamax._src.ops.attention.pallas_triton"] = _ptfa
except Exception:
  pass

# 4. pathwaysutils.elastic.elastic compatibility shim
try:
  import pathwaysutils.elastic as _pwe
  if not hasattr(_pwe, "elastic"):
    _mod = types.ModuleType("pathwaysutils.elastic.elastic")
    _mod.is_error_due_to_slice_down = lambda *a, **kw: False
    _pwe.elastic = _mod
    sys.modules["pathwaysutils.elastic.elastic"] = _mod
except Exception:
  pass

# 5. orbax.checkpoint.v1 compatibility shims for orbax-checkpoint 0.11.x
try:
  from etils import epath
  import orbax.checkpoint
  from orbax.checkpoint import v1 as ocp
  from orbax.checkpoint.experimental.v1._src.context import options as _ocp_opts

  if not hasattr(orbax.checkpoint, "pathways"):
    _pw = types.ModuleType("orbax.checkpoint.pathways")
    class _CheckpointingImpl:
      NO_DISPATCHER = "NO_DISPATCHER"
      @classmethod
      def from_options(cls, **kw):
        return "COLOCATED"
    _pw.CheckpointingImpl = _CheckpointingImpl
    orbax.checkpoint.pathways = _pw
    sys.modules["orbax.checkpoint.pathways"] = _pw

  if not hasattr(ocp.training, "pathways"):
    _tpw = types.ModuleType("orbax.checkpoint.v1.training.pathways")
    _tpw.Snapshotter = object
    ocp.training.pathways = _tpw

  def _unfreeze(cls):
    if dataclasses.is_dataclass(cls):
      cls.__setattr__ = object.__setattr__
    for attr in dir(cls):
      if not attr.startswith("_"):
        sub = getattr(cls, attr, None)
        if isinstance(sub, type) and dataclasses.is_dataclass(sub):
          _unfreeze(sub)

  for _name in dir(_ocp_opts):
    _obj = getattr(_ocp_opts, _name)
    if isinstance(_obj, type) and dataclasses.is_dataclass(_obj):
      _unfreeze(_obj)

  if not hasattr(_ocp_opts.ArrayOptions.Loading, "use_load_and_broadcast"):
    _ocp_opts.ArrayOptions.Loading.use_load_and_broadcast = False
    _ocp_opts.ArrayOptions.Loading.load_and_broadcast_options = types.SimpleNamespace(
        replica_axis_index=0, broadcast_memory_limit_bytes=0
    )

  class _MemoryProxy:
    def __init__(self, ctx):
      self._ctx = ctx
    @property
    def write_concurrent_bytes(self):
      return self._ctx._array_options.saving.concurrent_bytes
    @write_concurrent_bytes.setter
    def write_concurrent_bytes(self, val):
      self._ctx._array_options.saving.concurrent_bytes = val
    @property
    def read_concurrent_bytes(self):
      return self._ctx._array_options.loading.concurrent_bytes
    @read_concurrent_bytes.setter
    def read_concurrent_bytes(self, val):
      self._ctx._array_options.loading.concurrent_bytes = val

  if not hasattr(ocp.Context, "array"):
    ocp.Context.array = property(lambda self: self._array_options)
    ocp.Context.asynchronous = property(lambda self: self._async_options)
    ocp.Context.pytree = property(lambda self: self._pytree_options)
    ocp.Context.memory = property(lambda self: _MemoryProxy(self))
    ocp.Context.deletion = property(
        lambda self: types.SimpleNamespace(
            gcs_deletion_options=types.SimpleNamespace(todelete_full_path=None)
        )
    )
    ocp.Context.pathways = property(lambda self: types.SimpleNamespace(checkpointing_impl=None))
    ocp.Context.checkpoint_layout = property(
        lambda self: self._checkpoint_layout,
        lambda self, val: setattr(self, "_checkpoint_layout", val),
    )

  if not hasattr(ocp, "load"):
    def _ocp_load(path, abstract_pytree=None, *, checkpointable_name=None):
      p = epath.Path(path)
      if checkpointable_name is None and (p / "items").exists():
        checkpointable_name = "items"
      elif checkpointable_name is None and (p / "pytree").exists():
        checkpointable_name = "pytree"
      return ocp.load_pytree(path, abstract_pytree, checkpointable_name=checkpointable_name)
    ocp.load = _ocp_load

  if not hasattr(ocp, "metadata"):
    def _ocp_metadata(path, checkpointable_name=None):
      p = epath.Path(path)
      if checkpointable_name is not None:
        cm = ocp.checkpointables_metadata(p)
        return ocp.CheckpointMetadata(
            metadata=cm.metadata[checkpointable_name],
            init_timestamp_nsecs=cm.init_timestamp_nsecs,
            commit_timestamp_nsecs=cm.commit_timestamp_nsecs,
            custom_metadata=cm.custom_metadata,
        )
      if (p / "pytree").exists():
        return ocp.pytree_metadata(p)
      cm = ocp.checkpointables_metadata(p.parent)
      return ocp.CheckpointMetadata(
          metadata=cm.metadata[p.name],
          init_timestamp_nsecs=cm.init_timestamp_nsecs,
          commit_timestamp_nsecs=cm.commit_timestamp_nsecs,
          custom_metadata=cm.custom_metadata,
      )
    ocp.metadata = _ocp_metadata

  if not hasattr(ocp, "save"):
    def _ocp_save(path, pytree, *, checkpointable_name="pytree", overwrite=False, custom_metadata=None):
      return ocp.save_checkpointables(
          path,
          {checkpointable_name: pytree},
          overwrite=overwrite,
          custom_metadata=custom_metadata,
      )
    ocp.save = _ocp_save

  _orig_ckptr_init = ocp.training.Checkpointer.__init__
  _orig_save_ckpt = ocp.training.Checkpointer.save_checkpointables
  _orig_save_ckpt_async = ocp.training.Checkpointer.save_checkpointables_async
  _orig_load_ckpt = ocp.training.Checkpointer.load_checkpointables

  def _patched_ckptr_init(self, directory, *, context=None, logger=None, **kwargs):
    self._custom_context = context
    if context is not None:
      with context:
        _orig_ckptr_init(self, directory, **kwargs)
    else:
      _orig_ckptr_init(self, directory, **kwargs)

  def _patched_save_ckpt(self, *a, **kw):
    ctx = getattr(self, "_custom_context", None)
    if ctx is not None:
      with ctx:
        return _orig_save_ckpt(self, *a, **kw)
    return _orig_save_ckpt(self, *a, **kw)

  def _patched_save_ckpt_async(self, *a, **kw):
    ctx = getattr(self, "_custom_context", None)
    if ctx is not None:
      with ctx:
        return _orig_save_ckpt_async(self, *a, **kw)
    return _orig_save_ckpt_async(self, *a, **kw)

  def _patched_load_ckpt(self, *a, **kw):
    ctx = getattr(self, "_custom_context", None)
    if ctx is not None:
      with ctx:
        return _orig_load_ckpt(self, *a, **kw)
    return _orig_load_ckpt(self, *a, **kw)

  ocp.training.Checkpointer.__init__ = _patched_ckptr_init
  ocp.training.Checkpointer.save_checkpointables = _patched_save_ckpt
  ocp.training.Checkpointer.save_checkpointables_async = _patched_save_ckpt_async
  ocp.training.Checkpointer.load_checkpointables = _patched_load_ckpt
  if not hasattr(ocp.training.Checkpointer, "wait"):
    ocp.training.Checkpointer.wait = lambda self: self._manager.wait_until_finished()
except Exception:
  pass

# 6. Support multi-slice hybrid device mesh on platforms without device.slice_index (uses process_index per slice)
try:
  from jax.experimental import mesh_utils as _mu
  _orig_create_hybrid = _mu.create_hybrid_device_mesh

  def _patched_create_hybrid(mesh_shape, dcn_mesh_shape, devices=None, *, process_is_granule=False, **kw):
    import jax

    devs = devices if devices is not None else jax.devices()
    if devs and (
        not hasattr(devs[0], "slice_index")
        or (len({getattr(d, "slice_index", 0) for d in devs}) == 1 and len({d.process_index for d in devs}) > 1)
    ):
      process_is_granule = True
    return _orig_create_hybrid(
        mesh_shape, dcn_mesh_shape, devices=devices, process_is_granule=process_is_granule, **kw
    )

  _mu.create_hybrid_device_mesh = _patched_create_hybrid
except Exception:
  pass

# 7. Bound transient host RAM during Orbax ArrayHandler.deserialize for large (>50 GB) models
try:
  import ctypes
  import gc
  import math
  import numpy as np
  from orbax.checkpoint._src.serialization import type_handlers as _th

  _orig_ah_deserialize = _th.ArrayHandler.deserialize

  async def _batched_ah_deserialize(self, infos, args=None):
    if args is None or len(infos) <= 1:
      return await _orig_ah_deserialize(self, infos, args=args)
    leaf_bytes = []
    for a in args:
      shp = getattr(a, "shape", None)
      dt = getattr(a, "dtype", None)
      if shp and dt is not None:
        try:
          n_procs = max(1, len({d.process_index for d in a.sharding.device_set})) if getattr(a, "sharding", None) else 1
        except Exception:
          n_procs = 1
        leaf_bytes.append((math.prod(shp) * np.dtype(dt).itemsize) // n_procs)
      else:
        leaf_bytes.append(0)
    if sum(leaf_bytes) <= 45 * 10**9:
      return await _orig_ah_deserialize(self, infos, args=args)

    results = [None] * len(infos)
    batch_idx, batch_infos, batch_args = [], [], []
    batch_b = 0
    max_batch_b = 38 * 10**9
    for i, (inf, arg, lb) in enumerate(zip(infos, args, leaf_bytes)):
      if batch_infos and batch_b + lb > max_batch_b:
        part = await _orig_ah_deserialize(self, batch_infos, args=batch_args)
        for idx_p, val_p in zip(batch_idx, part):
          results[idx_p] = val_p
        del part
        gc.collect()
        try:
          ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
          pass
        batch_idx, batch_infos, batch_args, batch_b = [], [], [], 0
      batch_idx.append(i)
      batch_infos.append(inf)
      batch_args.append(arg)
      batch_b += lb
    if batch_infos:
      part = await _orig_ah_deserialize(self, batch_infos, args=batch_args)
      for idx_p, val_p in zip(batch_idx, part):
        results[idx_p] = val_p
      del part
      gc.collect()
      try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
      except Exception:
        pass
    return results

  _th.ArrayHandler.deserialize = _batched_ah_deserialize
except Exception:
  pass

# 8. Single-GET streaming for HuggingFace Hub -> GCS caching (1 resolver request per shard instead of 800)
try:
  import os as _os
  import random as _random
  import shutil as _shutil
  import time as _time
  import huggingface_hub as _hfh
  import requests as _requests

  _orig_hffs_glob = _hfh.HfFileSystem.glob

  def _fast_hffs_glob(self, path, ** kwargs):
    if isinstance(path, str) and path.endswith("/*.safetensors") and not kwargs:
      repo_id = path[: -len("/*.safetensors")].strip("/")
      if repo_id.count("/") == 1:
        is_gated = repo_id.startswith(("meta-llama/", "google/"))
        tok = getattr(self, "token", None) or _os.environ.get("HF_TOKEN")
        for attempt in range(8):
          try:
            use_tok = tok if (is_gated or attempt % 2 == 1) else None
            api = _hfh.HfApi(token=use_tok)
            return [f"{repo_id}/{fn}" for fn in api.list_repo_files(repo_id=repo_id) if fn.endswith(".safetensors")]
          except Exception:
            if attempt < 7:
              _time.sleep(min(15 * (attempt + 1), 60) + _random.uniform(1.0, 5.0))
            else:
              break
    return _orig_hffs_glob(self, path, **kwargs)

  _hfh.HfFileSystem.glob = _fast_hffs_glob
except Exception:
  pass
"""

# Each worker runs SITE_CUSTOMIZE_CONTENT twice: at interpreter start, from the generated `sitecustomize.py` on
# PYTHONPATH, and again from `main` (for processes started without it). The sentinel makes the second run a no-op,
# so the patches above wrap the original Orbax, mesh and HF functions only once.
_SITE_CUSTOMIZE_SENTINEL = "_ckpt_load_benchmark_site_customize_applied"


def _once_per_process(body: str, sentinel: str) -> str:
  """Returns Python source that runs `body` only the first time it is executed in a process.

  The first run sets the attribute `sentinel` on `sys`, and later runs see it and skip `body`.
  """
  guard = f"import sys\n\nif not getattr(sys, {sentinel!r}, False):\n  setattr(sys, {sentinel!r}, True)\n"
  return guard + textwrap.indent(body, "  ")


SITE_CUSTOMIZE_CONTENT = _once_per_process(_SITE_CUSTOMIZE_BODY, _SITE_CUSTOMIZE_SENTINEL)


def _platform_runtime_limits(platform_name: str | None = None) -> tuple[float, float, float, float]:
  """Returns `(step2_ram_margin_ratio, min_avail_gb, sparse_min_avail_gb, default_phase_timeout_s)`."""
  pname = platform_name or os.environ.get("BENCH_PLATFORM", "v7x")
  _import_bench_modules()
  from bench import platforms  # pylint: disable=import-outside-toplevel

  profile = platforms.PROFILES.get(pname)
  if profile is None:
    return 0.93, 8.5, 24.0, 14400.0
  return (
      profile.step2_ram_margin_ratio,
      profile.min_avail_host_ram_gb,
      profile.sparse_min_avail_host_ram_gb,
      float(profile.default_phase_timeout_s),
  )


def _read_host_mem_gb() -> tuple[float, float, float]:
  """Returns (used_gb, avail_gb, total_gb) in decimal GB (10**9 bytes) from /proc/meminfo and cgroups."""
  try:
    mem_total_kb = 0
    mem_avail_kb = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as mf:
      for line in mf:
        if line.startswith("MemTotal:"):
          mem_total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
          mem_avail_kb = int(line.split()[1])
        if mem_total_kb and mem_avail_kb:
          break
    total_gb = (mem_total_kb * 1024) / 1e9
    avail_gb = (mem_avail_kb * 1024) / 1e9
    used_gb = max(0.0, total_gb - avail_gb)
    for cg_cur_path, cg_max_path in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
      if os.path.exists(cg_cur_path):
        try:
          with open(cg_cur_path, "r", encoding="utf-8") as cf:
            cg_used_gb = int(cf.read().strip()) / 1e9
          if os.path.exists(cg_max_path):
            with open(cg_max_path, "r", encoding="utf-8") as mf_cg:
              raw_max = mf_cg.read().strip()
            if raw_max.isdigit():
              cg_max_gb = int(raw_max) / 1e9
              if 10.0 < cg_max_gb < total_gb:
                total_gb = cg_max_gb
          if cg_used_gb > used_gb:
            used_gb = cg_used_gb
            avail_gb = max(0.0, total_gb - used_gb)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        break
    return used_gb, avail_gb, total_gb
  except Exception:  # pylint: disable=broad-exception-caught
    return 0.0, 250.0, 250.0


def _failure_tag(returncode: int, watchdog_state: dict[str, Any]) -> str:
  """Returns why a worker died: a PHASE_TIMEOUT_S watchdog kill, a host OOM kill, or "" for other failures.

  Both watchdogs use SIGKILL (exit code -9), so the exit code alone cannot tell a timeout from an OOM kill.
  """
  if watchdog_state.get("timeout_triggered"):
    return f" (TIMEOUT: exceeded PHASE_TIMEOUT_S={watchdog_state.get('timeout_s', 0.0):.0f}s)"
  if returncode in (-9, 137) or watchdog_state.get("oom_triggered"):
    return " (OOMKilled / Exit Code 137)"
  return ""


def _start_oom_watchdog(
    proc: subprocess.Popen,
    min_avail_gb: float | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
  """Monitors host RAM while `proc` runs and kills ONLY `proc` if MemAvailable drops below `min_avail_gb`.

  This prevents the Kubernetes container cgroup from hitting its hard memory limit and killing PID 1.
  """
  import signal
  import threading

  _, default_min_avail, sparse_min_avail, default_timeout_s = _platform_runtime_limits(platform_name)
  if min_avail_gb is None:
    min_avail_gb = default_min_avail
  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1":
    min_avail_gb = max(min_avail_gb, sparse_min_avail)
  max_wall_s = float(os.environ.get("PHASE_TIMEOUT_S", str(int(default_timeout_s))))
  t_watch0 = time.time()

  state: dict[str, Any] = {
      "peak_ram_gb": 0.0,
      "total_ram_gb": 0.0,
      "oom_triggered": False,
      "timeout_triggered": False,
      "timeout_s": max_wall_s,
      "oom_message": "",
      "stop": False,
  }

  def _watch():
    while not state["stop"] and proc.poll() is None:
      used_gb, avail_gb, total_gb = _read_host_mem_gb()
      state["total_ram_gb"] = total_gb
      if used_gb > state["peak_ram_gb"]:
        state["peak_ram_gb"] = used_gb
      elapsed_s = time.time() - t_watch0
      if elapsed_s > max_wall_s:
        msg = (
            f"\n[TIMEOUT_WATCHDOG] Worker subprocess PID {proc.pid} exceeded PHASE_TIMEOUT_S={max_wall_s:.0f}s "
            f"(elapsed={elapsed_s:.1f}s, Peak RAM={state['peak_ram_gb']:.2f}/{total_gb:.2f} GB). "
            f"Sending SIGKILL to worker subprocess.\n"
        )
        state["timeout_triggered"] = True
        state["oom_message"] = msg
        sys.stdout.write(msg)
        sys.stdout.flush()
        try:
          os.kill(proc.pid, signal.SIGKILL)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        break
      if total_gb > 0 and avail_gb < min_avail_gb:
        msg = (
            f"\n[OOM_WATCHDOG] Host RAM exhausted! Peak RAM Usage: {used_gb:.2f}/{total_gb:.2f} GB "
            f"({(used_gb / total_gb) * 100:.1f}%, MemAvailable={avail_gb:.2f} GB < {min_avail_gb:.2f} GB threshold). "
            f"Sending SIGKILL (OOMKilled / exit code 137) to worker subprocess PID {proc.pid}.\n"
        )
        state["oom_triggered"] = True
        state["oom_message"] = msg
        sys.stdout.write(msg)
        sys.stdout.flush()
        try:
          os.kill(proc.pid, signal.SIGKILL)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        break
      time.sleep(0.05)

  t = threading.Thread(target=_watch, daemon=True)
  t.start()
  state["thread"] = t
  return state


def _start_self_oom_watchdog(
    min_avail_gb: float | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
  """In-process watchdog inside a worker subprocess so both local and remote slice workers self-terminate before container cgroup OOM."""
  import signal
  import threading

  try:
    with open("/proc/self/oom_score_adj", "w", encoding="utf-8") as oom_f:
      oom_f.write("1000\n")
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  _, default_min_avail, sparse_min_avail, default_timeout_s = _platform_runtime_limits(platform_name)
  if min_avail_gb is None:
    min_avail_gb = default_min_avail + 0.5
  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1":
    min_avail_gb = max(min_avail_gb, sparse_min_avail + 1.0)
  max_wall_s = float(os.environ.get("PHASE_TIMEOUT_S", str(int(default_timeout_s))))

  state: dict[str, Any] = {"peak_ram_gb": 0.0, "total_ram_gb": 0.0, "stop": False}

  def _self_watch():
    pid = os.getpid()
    t_watch0 = time.time()
    while not state["stop"]:
      used_gb, avail_gb, total_gb = _read_host_mem_gb()
      state["total_ram_gb"] = total_gb
      if used_gb > state["peak_ram_gb"]:
        state["peak_ram_gb"] = used_gb
      elapsed_s = time.time() - t_watch0
      if elapsed_s > max_wall_s:
        msg = (
            f"\n[TIMEOUT_WATCHDOG] Worker PID {pid} exceeded PHASE_TIMEOUT_S={max_wall_s:.0f}s "
            f"(elapsed={elapsed_s:.1f}s, Peak RAM={state['peak_ram_gb']:.2f}/{total_gb:.2f} GB). "
            f"Sending SIGKILL to worker PID {pid}.\n"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
        os.kill(pid, signal.SIGKILL)
        break
      if total_gb > 0 and avail_gb < min_avail_gb:
        msg = (
            f"\n[OOM_WATCHDOG] Host RAM exhausted! Peak RAM Usage: {used_gb:.2f}/{total_gb:.2f} GB "
            f"({(used_gb / total_gb) * 100:.1f}%, MemAvailable={avail_gb:.2f} GB < {min_avail_gb:.2f} GB threshold). "
            f"Sending SIGKILL (OOMKilled / exit code 137) to worker PID {pid}.\n"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
        os.kill(pid, signal.SIGKILL)
        break
      time.sleep(0.05)

  t = threading.Thread(target=_self_watch, daemon=True)
  t.start()
  return state


def run_remote_slice_daemon(port: int) -> None:
  """Runs a lightweight TCP listener that executes remote slice worker subprocesses on a companion node."""
  import threading

  def _handle_client(conn: socket.socket, addr):
    try:
      with conn:
        f_in = conn.makefile("r", encoding="utf-8", errors="replace")
        header = f_in.readline()
        if not header:
          return
        req = json.loads(header)
        if req.get("shutdown"):
          print(f"[RemoteDaemon] Received shutdown request from {addr}. Exiting cleanly.", flush=True)
          conn.sendall(b"__REMOTE_EXIT_CODE__:0\n")
          os._exit(0)
        cmd = req["cmd"]
        c_env = os.environ.copy()
        if req.get("env"):
          c_env.update(req["env"])
        r_work_dir = req.get("local_work_dir", "/tmp/benchmark_remote")
        os.makedirs(r_work_dir, exist_ok=True)
        site_dir = os.path.join(r_work_dir, "_sitecustomize")
        os.makedirs(site_dir, exist_ok=True)
        with open(os.path.join(site_dir, "sitecustomize.py"), "w", encoding="utf-8") as sf:
          sf.write(SITE_CUSTOMIZE_CONTENT)
        subprocess.run(["pkill", "-9", "-f", "--", "--worker-phase"], check=False)
        print(f"[RemoteDaemon] Starting command from {addr}: {' '.join(cmd)}", flush=True)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=c_env, text=True) as proc:
          r_wd = _start_oom_watchdog(proc, platform_name=c_env.get("BENCH_PLATFORM"))
          assert proc.stdout is not None
          for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            conn.sendall(line.encode("utf-8", errors="replace"))
          proc.wait(timeout=14400)
          r_wd["stop"] = True
          if (r_wd.get("oom_triggered") or r_wd.get("timeout_triggered")) and r_wd.get("oom_message"):
            conn.sendall(r_wd["oom_message"].encode("utf-8", errors="replace"))
          if r_wd.get("peak_ram_gb", 0.0) > 0:
            conn.sendall(
                f"INFO:watchdog:Remote Slice Peak Host RAM: {r_wd['peak_ram_gb']:.2f}/{r_wd.get('total_ram_gb', 251.69):.2f} GB\n".encode(
                    "utf-8", errors="replace"
                )
            )
          rc = proc.returncode
        conn.sendall(f"\n__REMOTE_EXIT_CODE__:{rc}\n".encode("utf-8"))
        print(f"[RemoteDaemon] Finished command from {addr} with exit code {rc}", flush=True)
    except Exception as ex:  # pylint: disable=broad-exception-caught
      print(f"[RemoteDaemon] Error handling client {addr}: {ex}", flush=True)
      try:
        conn.sendall(f"\n[RemoteDaemon Error]: {ex}\n__REMOTE_EXIT_CODE__:1\n".encode("utf-8"))
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"[RemoteDaemon] Listening on 0.0.0.0:{port} (pod IP: {local_ip}:{port}) ...", flush=True)
    while True:
      conn, addr = srv.accept()
      threading.Thread(target=_handle_client, args=(conn, addr), daemon=True).start()


def _load_code_version() -> dict[str, Any]:
  """Returns the CODE_VERSION.json at the root of the code tree this harness runs from, the parent of its directory.

  scripts/build_code_tarball.py writes it into the code tarball. Without it (e.g. in a git checkout), or when it cannot
  be read or is not a JSON object, the result says so, with `git_commit` "unknown", instead of failing the phase.
  """
  path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CODE_VERSION.json")
  try:
    with open(path, "r", encoding="utf-8") as f:
      code_version = json.load(f)
  except FileNotFoundError:
    return {"git_commit": "unknown", "source": "CODE_VERSION.json not found"}
  except OSError as e:
    return {"git_commit": "unknown", "source": f"CODE_VERSION.json could not be read: {e}"}
  except ValueError as e:
    return {"git_commit": "unknown", "source": f"CODE_VERSION.json is not valid JSON: {e}"}
  if not isinstance(code_version, dict):
    return {"git_commit": "unknown", "source": "CODE_VERSION.json is not a JSON object"}
  return code_version


def _write_result_json(result_json: str, data: dict[str, Any]) -> None:
  """Writes `data` as the phase result JSON, after adding to it the version of the code that ran (`code_version`)."""
  data["code_version"] = _load_code_version()
  with open(result_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)


_NOT_TPU_MARKER_RE = re.compile(r"\[NOT-TPU\]\s+backend=(\S+)\s+device_count=([0-9]+)\s+expected=([0-9]+):\s*(.*)")


def _parse_not_tpu_marker(log_texts: list[str]) -> dict[str, Any] | None:
  """Returns parsed NOT-TPU details from worker logs, if the TPU verification check failed."""
  for text in log_texts:
    match = _NOT_TPU_MARKER_RE.search(text)
    if match:
      return {
          "jax_backend_platform": match.group(1),
          "total_tpu_devices_measured": int(match.group(2)),
          "expected_tpu_devices": int(match.group(3)),
          "reason": match.group(4).strip(),
      }
  return None


def _persist_worker_upload_max_s(result_json: str, worker_cached_times: list[float]) -> dict[str, Any]:
  """Returns the phase result JSON, with the slowest worker upload written into it as `worker_upload_max_s`.

  The worker writes the result JSON before the parent has parsed the upload times from the worker log, so the parent
  adds their maximum before it mirrors the JSON to GCS. Without upload times, the JSON stays as the worker wrote it.
  """
  with open(result_json, "r", encoding="utf-8") as rf:
    data = json.load(rf)
  if worker_cached_times:
    data["worker_upload_max_s"] = max(worker_cached_times)
    with open(result_json, "w", encoding="utf-8") as wf:
      json.dump(data, wf, indent=2)
  return data


def _phase_scan_dir_name(phase: str, scan_mode: str, num_slices: int, num_hosts_per_slice: int) -> str:
  """Returns the log subdirectory name under `logs/<model>/` for a phase and slice layout."""
  total_procs = num_slices * num_hosts_per_slice
  is_multislice_phase = total_procs > 1 and phase in ("to_maxtext_step3", "safetensors_dynamic")
  scan_dir_name = scan_mode
  if is_multislice_phase and num_slices > 1:
    scan_dir_name += f"_slices{num_slices}"
  if is_multislice_phase and num_hosts_per_slice > 1:
    scan_dir_name += f"_hosts{num_hosts_per_slice}"
  return scan_dir_name


def _resolve_slice_topology(
    platform_name: str,
    explicit_topology: str | None,
    devices_per_slice: int,
    num_hosts_per_slice: int,
) -> str:
  """Resolves a valid `PlatformProfile` slice topology string from explicit CLI arg or device/host counts."""
  _import_bench_modules()
  from bench import platforms  # pylint: disable=import-outside-toplevel

  profile = platforms.PROFILES.get(platform_name)
  if profile is None:
    return explicit_topology or f"{devices_per_slice}dev"
  if explicit_topology and explicit_topology in profile.hosts_per_slice_by_topology:
    return explicit_topology
  for topo in profile.hosts_per_slice_by_topology:
    if profile.devices_per_slice(topo) == devices_per_slice and profile.hosts_per_slice(topo) == num_hosts_per_slice:
      return topo
  for topo in profile.hosts_per_slice_by_topology:
    if profile.devices_per_slice(topo) == devices_per_slice:
      return topo
  return next(iter(profile.hosts_per_slice_by_topology))


def _collect_phase_logs(log_dir: str, phase: str) -> list[str]:
  """Reads local and remote slice log files belonging to `phase`."""
  if not os.path.isdir(log_dir):
    return []
  log_texts: list[str] = []
  for fname in sorted(os.listdir(log_dir)):
    if fname == f"{phase}.log" or (fname.startswith(f"{phase}_slice") and fname.endswith(".log")):
      try:
        with open(os.path.join(log_dir, fname), "r", encoding="utf-8", errors="replace") as lf:
          log_texts.append(lf.read())
      except OSError:
        pass
  return log_texts


def _extract_pod_and_node_names(
    merged_stages: dict[str, Any],
    env_meta: dict[str, Any],
) -> tuple[list[str], list[str]]:
  """Collects ordered unique pod and node names from per-host stage payloads with env fallback."""
  pods: list[str] = []
  nodes: list[str] = []
  for host in merged_stages.get("hosts", []):
    pod = host.get("pod_name") or host.get("host_name") or ""
    if pod and pod not in pods:
      pods.append(str(pod))
    node = host.get("node_name") or ""
    if node and node not in nodes:
      nodes.append(str(node))
  if not pods and env_meta.get("pod_name"):
    pods.append(str(env_meta["pod_name"]))
  if not nodes and env_meta.get("node_name"):
    nodes.append(str(env_meta["node_name"]))
  return pods, nodes


def _build_log_urls(gcs_run_dir: str, model: str, scan_dir_name: str, phase: str) -> dict[str, str]:
  """Builds canonical GCS log URIs for a phase when `gcs_run_dir` is a `gs://` path."""
  if not gcs_run_dir.startswith("gs://"):
    return {}
  prefix = f"{gcs_run_dir.rstrip('/')}/logs/{model}/{scan_dir_name}"
  return {
      "gcs_logs_prefix": prefix,
      "gcs_result_json_uri": f"{prefix}/{phase}_result.json",
      "gcs_main_log_uri": f"{prefix}/{phase}.log",
  }


def _finalize_v2_phase_result(
    *,
    phase: str,
    model: str,
    scan_mode: str,
    platform_name: str,
    slice_topology: str,
    num_slices: int,
    num_hosts_per_slice: int,
    devices_per_slice: int,
    raw_data: dict[str, Any],
    run_status: str,
    started_at_utc: str,
    finished_at_utc: str,
    wall_clock_s: float,
    cache_state: str,
    env: dict[str, str],
    repo_root: str,
    gcs_run_dir: str,
    scan_dir_name: str,
    log_dir: str,
    worker_cached_times: list[float],
    failure_err_text: str = "",
    failure_exit_code: int | None = None,
    watchdog_state: dict[str, Any] | None = None,
    skip_reason: str = "",
) -> dict[str, Any]:
  """Merges worker/remote logs, metadata, and raw timings into a validated v2 phase record."""
  harness_v2, metadata, schema_v2, _ = _import_bench_modules()
  log_texts = _collect_phase_logs(log_dir, phase)
  merged_stages = harness_v2.collect_merged_stages_from_logs(log_texts)
  env_meta = metadata.collect_env_cluster_metadata(env)
  code_ver = metadata.load_code_version_metadata(repo_root)
  hw_info = metadata.collect_host_hardware_info()
  pkg_versions = metadata.collect_package_versions()
  pod_names, node_names = _extract_pod_and_node_names(merged_stages, env_meta)

  raw = dict(raw_data)
  header_info = raw.pop("header_info", None)
  pipeline_config = raw.pop("pipeline_config", None)
  parameter_checksums = raw.pop("parameter_checksums", None)

  failure_info: dict[str, Any] | None = None
  if run_status == "FAILED":
    not_tpu_info = _parse_not_tpu_marker(log_texts)
    fcat, fsig, pre, esum = harness_v2.classify_failure(
        failure_err_text,
        log_texts,
        exit_code=failure_exit_code,
        not_tpu_info=not_tpu_info,
    )
    failed_stage = harness_v2.infer_failed_stage(merged_stages, log_texts)
    failure_info = {
        "failure_category": fcat,
        "failed_stage_name": failed_stage,
        "process_exit_code": failure_exit_code if failure_exit_code is not None else "",
        "process_signal": fsig,
        "error_summary": esum,
        "was_preempted": pre,
    }
    if not_tpu_info is not None:
      raw.update(not_tpu_info)
      for timing_key in (
          "hf_to_gcs_s",
          "hf_to_local_s",
          "convert_to_orbax_s",
          "load_orbax_s",
          "safetensors_and_convert_s",
          "worker_upload_max_s",
      ):
        raw.pop(timing_key, None)
    elif watchdog_state and watchdog_state.get("peak_ram_gb", 0.0) > 0:
      raw.setdefault("peak_host_ram_gb", round(watchdog_state["peak_ram_gb"], 2))
      raw.setdefault("total_host_ram_gb", round(watchdog_state.get("total_ram_gb", 251.69), 2))
    raw.setdefault("status", "FAILED")
    raw.setdefault("error", failure_err_text)
  elif run_status == "SKIPPED":
    raw.setdefault("status", "SKIPPED")
    if skip_reason:
      failure_info = {"error_summary": skip_reason}
  else:
    raw.setdefault("status", "OK")
    if worker_cached_times:
      raw["worker_upload_max_s"] = max(worker_cached_times)

  v2_rec = harness_v2.build_v2_phase_record(
      phase=phase,
      model_name=model,
      scan_mode=scan_mode,
      platform_name=platform_name,
      num_slices=num_slices,
      num_hosts_per_slice=num_hosts_per_slice,
      topology=slice_topology,
      raw_worker_data=raw,
      merged_stages=merged_stages,
      env_meta=env_meta,
      code_ver=code_ver,
      hw_info=hw_info,
      pkg_versions=pkg_versions,
      started_at_utc=started_at_utc,
      finished_at_utc=finished_at_utc,
      wall_clock_s=wall_clock_s,
      run_status=run_status,
      cache_state=cache_state,
      pipeline_config=pipeline_config,
      header_info=header_info,
      failure_info=failure_info,
      pod_names=pod_names,
      node_names=node_names,
      log_urls=_build_log_urls(gcs_run_dir, model, scan_dir_name, phase),
      parameter_checksums=parameter_checksums,
  )
  val_errors = schema_v2.validate_v2_phase_record(v2_rec)
  if val_errors:
    print(f"[WARN v2 schema validation] {phase} ({model}/{scan_mode}): {val_errors}", flush=True)
  v2_rec["status"] = raw.get("status", run_status)
  return v2_rec


def _write_skipped_v2_result(
    *,
    phase: str,
    model: str,
    scan_mode: str,
    platform_name: str,
    slice_topology: str,
    num_slices: int,
    num_hosts_per_slice: int,
    devices_per_slice: int,
    local_work_dir: str,
    gcs_run_dir: str,
    skip_reason: str,
    raw_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Writes and mirrors a valid v2 `SKIPPED` result JSON so skipped cells are preserved cleanly."""
  repo_root = get_repo_root()
  scan_dir_name = _phase_scan_dir_name(phase, scan_mode, num_slices, num_hosts_per_slice)
  log_dir = os.path.join(local_work_dir, "logs", model, scan_dir_name)
  os.makedirs(log_dir, exist_ok=True)
  result_json = os.path.join(log_dir, f"{phase}_result.json")
  now_utc = _import_bench_modules()[0].utc_now_iso()
  raw_data = {
      "status": "SKIPPED",
      "num_slices": num_slices,
      "num_hosts_per_slice": num_hosts_per_slice,
      "global_devices": num_slices * devices_per_slice,
  }
  if raw_extra:
    raw_data.update(raw_extra)
  v2_rec = _finalize_v2_phase_result(
      phase=phase,
      model=model,
      scan_mode=scan_mode,
      platform_name=platform_name,
      slice_topology=slice_topology,
      num_slices=num_slices,
      num_hosts_per_slice=num_hosts_per_slice,
      devices_per_slice=devices_per_slice,
      raw_data=raw_data,
      run_status="SKIPPED",
      started_at_utc=now_utc,
      finished_at_utc=now_utc,
      wall_clock_s=0.0,
      cache_state=os.environ.get("BENCH_CACHE_STATE", "empty"),
      env=dict(os.environ),
      repo_root=repo_root,
      gcs_run_dir=gcs_run_dir,
      scan_dir_name=scan_dir_name,
      log_dir=log_dir,
      worker_cached_times=[],
      skip_reason=skip_reason,
  )
  _write_result_json(result_json, v2_rec)
  if gcs_run_dir.startswith("gs://"):
    try:
      from etils import epath  # pylint: disable=import-outside-toplevel

      gcs_res = epath.Path(f"{gcs_run_dir.rstrip('/')}/logs/{model}/{scan_dir_name}/{phase}_result.json")
      gcs_res.parent.mkdir(parents=True, exist_ok=True)
      gcs_res.write_bytes(epath.Path(result_json).read_bytes())
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[WARN] Failed to mirror SKIPPED JSON to GCS: {exc}", flush=True)
  return v2_rec


def _maybe_verify_and_persist_checksums(
    *,
    model: str,
    scan_mode: str,
    num_slices: int,
    num_hosts_per_slice: int,
    m_step3: dict[str, Any],
    s_res: dict[str, Any],
    local_work_dir: str,
    gcs_run_dir: str,
) -> None:
  """Verifies parameter checksums across both pipelines and persists match/mismatch onto `s_res`."""
  tm_cs = m_step3.get("parameter_checksums")
  sd_cs = s_res.get("parameter_checksums")
  if m_step3.get("status") != "OK" or s_res.get("status") != "OK" or not tm_cs or not sd_cs:
    return
  harness_v2 = _import_bench_modules()[0]
  mismatches = harness_v2.compare_parameter_checksums(tm_cs, sd_cs)
  cs_status = "MISMATCH" if mismatches else "MATCH"
  print(f"[CHECKSUM VERIFY] {model} ({scan_mode}, slices={num_slices}): {cs_status}", flush=True)
  for mm in mismatches:
    print(f"  [CHECKSUM MISMATCH] {mm}", flush=True)
  s_res["checksum_verification"] = cs_status
  if mismatches:
    s_res["checksum_mismatches"] = mismatches
  sd_dir = _phase_scan_dir_name("safetensors_dynamic", scan_mode, num_slices, num_hosts_per_slice)
  sd_json = os.path.join(local_work_dir, "logs", model, sd_dir, "safetensors_dynamic_result.json")
  if not os.path.exists(sd_json):
    return
  _write_result_json(sd_json, s_res)
  if not gcs_run_dir.startswith("gs://"):
    return
  try:
    from etils import epath  # pylint: disable=import-outside-toplevel

    epath.Path(f"{gcs_run_dir.rstrip('/')}/logs/{model}/{sd_dir}/safetensors_dynamic_result.json").write_bytes(
        epath.Path(sd_json).read_bytes()
    )
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def invoke_worker_subprocess(
    phase: str,
    model: str,
    scan_mode: str,
    gcs_run_dir: str,
    local_work_dir: str,
    hardware: str = "auto",
    num_slices: int = 1,
    num_hosts_per_slice: int = 1,
    weight_dtype: str = "auto",
    dcn_parallelism_axis: str = "auto",
    devices_per_slice: int = 4,
    remote_slice_addrs: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    skip_completed: bool = False,
    reuse_model_hf_cache: bool = False,
    allow_step_failure: bool = False,
    platform_name: str | None = None,
    slice_topology: str | None = None,
    verify_checksums: bool = False,
) -> dict[str, Any]:
  """Runs a single benchmark phase in clean subprocess(es) and returns its v2 JSON metrics.

  Multi-process phases (to_maxtext Step 3 and safetensors_dynamic) run one JAX process per TPU host:
  num_slices * num_hosts_per_slice processes. Process 0 runs locally and process k > 0 runs on
  remote_slice_addrs[k - 1], so the endpoints are listed host by host within each slice.
  """
  repo_root = get_repo_root()
  os.makedirs(local_work_dir, exist_ok=True)
  harness_v2, _, _, _ = _import_bench_modules()
  resolved_platform = platform_name or os.environ.get("BENCH_PLATFORM") or "v7x"
  resolved_topology = _resolve_slice_topology(resolved_platform, slice_topology, devices_per_slice, num_hosts_per_slice)
  total_procs = num_slices * num_hosts_per_slice
  is_multislice_phase = total_procs > 1 and phase in ("to_maxtext_step3", "safetensors_dynamic")
  scan_dir_name = _phase_scan_dir_name(phase, scan_mode, num_slices, num_hosts_per_slice)
  log_dir = os.path.join(local_work_dir, "logs", model, scan_dir_name)
  os.makedirs(log_dir, exist_ok=True)

  result_json = os.path.join(log_dir, f"{phase}_result.json")
  log_file = os.path.join(log_dir, f"{phase}.log")
  hf_stage_root = "/ramdisk" if os.path.isdir("/ramdisk") else local_work_dir
  local_hf_dir = os.path.join(hf_stage_root, "hf_local", model)

  if skip_completed and not os.path.exists(result_json):
    try:
      from etils import epath

      gcs_res = epath.Path(f"{gcs_run_dir}/logs/{model}/{scan_dir_name}/{phase}_result.json")
      if gcs_res.exists():
        epath.Path(result_json).write_bytes(gcs_res.read_bytes())
        gcs_log = epath.Path(f"{gcs_run_dir}/logs/{model}/{scan_dir_name}/{phase}.log")
        if gcs_log.exists():
          epath.Path(log_file).write_bytes(gcs_log.read_bytes())
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  if skip_completed and os.path.exists(result_json):
    with open(result_json, "r", encoding="utf-8") as rf:
      data = json.load(rf)
    if data.get("status") != "FAILED" or os.environ.get("SKIP_FAILED_RESULTS") == "1":
      print(
          f"[SKIP COMPLETED]: {phase} | Model: {model} | Scan Mode: {scan_mode} "
          f"| Slices: {num_slices} -> {result_json}",
          flush=True,
      )
      if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as lf:
          worker_cached_times = [float(m.group(1)) for m in re.finditer(r"\[Worker\] Cached .* in ([0-9.]+)s", lf.read())]
        if worker_cached_times:
          data["worker_upload_max_s"] = max(worker_cached_times)
      return data

  if reuse_model_hf_cache and phase == "to_maxtext_step1" and scan_mode != "scanned" and not os.path.exists(result_json):
    scanned_json = os.path.join(local_work_dir, "logs", model, "scanned", "to_maxtext_step1_result.json")
    scanned_log = os.path.join(local_work_dir, "logs", model, "scanned", "to_maxtext_step1.log")
    if os.path.exists(scanned_json):
      with open(scanned_json, "r", encoding="utf-8") as rf:
        data = json.load(rf)
      if data.get("status") != "FAILED":
        shutil.copyfile(scanned_json, result_json)
        if os.path.exists(scanned_log):
          shutil.copyfile(scanned_log, log_file)
        if "scan_layers_mode" in data:
          data["scan_layers_mode"] = scan_mode
          _write_result_json(result_json, data)
        print(
            f"[REUSE MODEL HF STAGE]: {phase} | Model: {model} | Scan Mode: {scan_mode} -> {result_json}",
            flush=True,
        )
        if os.path.exists(log_file):
          with open(log_file, "r", encoding="utf-8") as lf:
            worker_cached_times = [
                float(m.group(1)) for m in re.finditer(r"\[Worker\] Cached .* in ([0-9.]+)s", lf.read())
            ]
          if worker_cached_times:
            data["worker_upload_max_s"] = max(worker_cached_times)
        return data

  base_cmd = [
      sys.executable,
      os.path.abspath(__file__),
      "--worker-phase",
      phase,
      "--model",
      model,
      "--scan-mode",
      scan_mode,
      "--hardware",
      hardware,
      "--platform",
      resolved_platform,
      "--slice-topology",
      resolved_topology,
      "--weight-dtype",
      weight_dtype,
      "--dcn-parallelism-axis",
      dcn_parallelism_axis,
      "--num-slices",
      str(num_slices if is_multislice_phase else 1),
      "--num-hosts-per-slice",
      str(num_hosts_per_slice if is_multislice_phase else 1),
      "--devices-per-slice",
      str(devices_per_slice),
      "--gcs-run-dir",
      gcs_run_dir,
      "--local-hf-dir",
      local_hf_dir,
      "--result-json",
      result_json,
  ]
  if reuse_model_hf_cache:
    base_cmd.append("--reuse-model-hf-cache")
  if verify_checksums:
    base_cmd.append("--verify-checksums")

  env = os.environ.copy()
  env.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
  env.setdefault("PYTHONUNBUFFERED", "1")
  cache_state = harness_v2.prepare_clean_jax_compilation_cache()
  env["BENCH_CACHE_STATE"] = cache_state
  env["BENCH_PLATFORM"] = resolved_platform
  env["BENCH_SLICE_TOPOLOGY"] = resolved_topology
  site_dir = os.path.join(local_work_dir, "_sitecustomize")
  os.makedirs(site_dir, exist_ok=True)
  sitecustomize_path = os.path.join(site_dir, "sitecustomize.py")
  with open(sitecustomize_path, "w", encoding="utf-8") as sf:
    sf.write(SITE_CUSTOMIZE_CONTENT)
  src_dir = os.path.join(repo_root, "src")
  existing_pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = f"{site_dir}:{src_dir}:{existing_pythonpath}" if existing_pythonpath else f"{site_dir}:{src_dir}"
  if extra_env:
    env.update(extra_env)

  print(f"\n{'-' * 80}")
  print(
      f"[PHASE]: {phase} | Model: {model} | Scan Mode: {scan_mode} | Slices: {num_slices} "
      f"| Hosts/slice: {num_hosts_per_slice}"
  )
  print(f"[LOG]:   {log_file}")
  print(f"{'-' * 80}", flush=True)

  def _mirror_logs_to_gcs():
    try:
      from etils import epath

      gcs_log_dir = epath.Path(f"{gcs_run_dir}/logs/{model}/{scan_dir_name}")
      gcs_log_dir.mkdir(parents=True, exist_ok=True)
      for fname in os.listdir(log_dir):
        fpath = os.path.join(log_dir, fname)
        if os.path.isfile(fpath):
          (gcs_log_dir / fname).write_bytes(epath.Path(fpath).read_bytes())
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"[WARN] Failed to mirror logs to GCS: {e}", flush=True)

  def _finalize_and_persist(
      raw_payload: dict[str, Any],
      *,
      status_str: str,
      err_str: str = "",
      exit_code: int | None = None,
  ) -> dict[str, Any]:
    v2_data = _finalize_v2_phase_result(
        phase=phase,
        model=model,
        scan_mode=scan_mode,
        platform_name=resolved_platform,
        slice_topology=resolved_topology,
        num_slices=num_slices if is_multislice_phase else 1,
        num_hosts_per_slice=num_hosts_per_slice if is_multislice_phase else 1,
        devices_per_slice=devices_per_slice,
        raw_data=raw_payload,
        run_status=status_str,
        started_at_utc=started_at_utc,
        finished_at_utc=harness_v2.utc_now_iso(),
        wall_clock_s=max(0.0, time.perf_counter() - t_phase_wall0),
        cache_state=cache_state,
        env=env,
        repo_root=repo_root,
        gcs_run_dir=gcs_run_dir,
        scan_dir_name=scan_dir_name,
        log_dir=log_dir,
        worker_cached_times=worker_cached_times,
        failure_err_text=err_str,
        failure_exit_code=exit_code,
        watchdog_state=watchdog_state,
    )
    _write_result_json(result_json, v2_data)
    _mirror_logs_to_gcs()
    return v2_data

  worker_cached_times: list[float] = []
  watchdog_state: dict[str, Any] = {}
  failed_exit_code: int | None = None
  started_at_utc = harness_v2.utc_now_iso()
  t_phase_wall0 = time.perf_counter()
  try:
    if not is_multislice_phase:
      if resolve_hardware(hardware) == "cpu" and "XLA_FLAGS" not in env:
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices_per_slice}"
      with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"Command: {' '.join(base_cmd)}\n\n")
        lf.flush()
        env["BENCH_PROCESS_START_EPOCH"] = f"{time.time():.6f}"
        with subprocess.Popen(
            base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=repo_root, text=True
        ) as proc:
          watchdog_state = _start_oom_watchdog(proc, platform_name=resolved_platform)
          assert proc.stdout is not None
          for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            m = re.search(r"\[Worker\] Cached .* in ([0-9.]+)s", line)
            if m:
              worker_cached_times.append(float(m.group(1)))
          proc.wait(timeout=14400)
          watchdog_state["stop"] = True
          if (watchdog_state.get("oom_triggered") or watchdog_state.get("timeout_triggered")) and watchdog_state.get(
              "oom_message"
          ):
            lf.write(watchdog_state["oom_message"])
          if watchdog_state.get("peak_ram_gb", 0.0) > 0:
            peak_msg = (
                f"INFO:watchdog:Subprocess Peak Host RAM: {watchdog_state['peak_ram_gb']:.2f}/"
                f"{watchdog_state.get('total_ram_gb', 251.69):.2f} GB\n"
            )
            sys.stdout.write(peak_msg)
            sys.stdout.flush()
            lf.write(peak_msg)
          lf.flush()

      if proc.returncode != 0:
        failed_exit_code = proc.returncode
        oom_tag = _failure_tag(proc.returncode, watchdog_state)
        raise RuntimeError(
            f"Worker phase '{phase}' failed with exit code {proc.returncode}{oom_tag} "
            f"(Peak RAM: {watchdog_state.get('peak_ram_gb', 0.0):.2f}/{watchdog_state.get('total_ram_gb', 251.69):.2f} GB). "
            f"See log: {log_file}"
        )
    else:
      remote_endpoints = list(remote_slice_addrs or [])
      if not remote_endpoints:
        env_remote1 = os.environ.get("REMOTE_SLICE1_ADDR", "").strip()
        if env_remote1:
          remote_endpoints = [a.strip() for a in env_remote1.split(",") if a.strip()]
      coord_port = find_free_port()
      if remote_endpoints:
        local_ip = socket.gethostbyname(socket.gethostname())
        coord_addr = f"{local_ip}:{coord_port}"
      else:
        coord_addr = f"127.0.0.1:{coord_port}"
      if resolve_hardware(hardware) == "cpu" and "XLA_FLAGS" not in env:
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices_per_slice}"
      import threading  # pylint: disable=import-outside-toplevel

      procs: list[subprocess.Popen] = []
      remote_threads: list[Any] = []
      remote_rc: dict[int, int] = {}
      remote_killed_local = threading.Event()
      slice_log_files = []
      try:
        # `slice_id` is the JAX process id (one per host); the slice index is slice_id // num_hosts_per_slice.
        env["BENCH_PROCESS_START_EPOCH"] = f"{time.time():.6f}"
        for slice_id in range(total_procs):
          cmd_i = base_cmd + ["--slice-id", str(slice_id), "--coordinator-address", coord_addr]
          if slice_id == 0:
            p = subprocess.Popen(
                cmd_i, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=repo_root, text=True
            )
            procs.append(p)
          elif remote_endpoints and (slice_id - 1) < len(remote_endpoints):
            target_remote_addr = remote_endpoints[slice_id - 1]
            slf = open(os.path.join(log_dir, f"{phase}_slice{slice_id}.log"), "w", encoding="utf-8")
            slf.write(f"Remote Slice {slice_id} ({target_remote_addr}) Command: {' '.join(cmd_i)}\n\n")
            slf.flush()
            slice_log_files.append(slf)

            def _run_remote(s_id, target_addr, c_list, c_env, log_f):
              host, port_s = target_addr.split(":", 1)
              rc = 1
              clean_env = {
                  k: v
                  for k, v in c_env.items()
                  if not (
                      k.endswith("_SERVICE_HOST")
                      or "_SERVICE_PORT" in k
                      or "_TCP" in k
                      or (
                          k.endswith("_PORT")
                          and not k.startswith("TPU_")
                          and not k.startswith("DAEMON_")
                          and not k.startswith("MEGASCALE_")
                      )
                      # Per-host TPU and Indexed Job identity: each remote pod keeps its own values.
                      or k
                      in (
                          "HOSTNAME",
                          "POD_NAME",
                          "NODE_NAME",
                          "TPU_WORKER_ID",
                          "TPU_WORKER_HOSTNAMES",
                          "TPU_PROCESS_ADDRESSES",
                          "JOB_COMPLETION_INDEX",
                          "JOBSET_JOB_INDEX",
                      )
                  )
              }
              try:
                import base64
                import zlib

                script_abs = os.path.abspath(__file__)
                with open(script_abs, "rb") as f_self:
                  b64_self = base64.b64encode(zlib.compress(f_self.read(), level=9)).decode("ascii")
                sync_cmd = [
                    "python3",
                    "-c",
                    (
                        f"import base64, os, zlib; p = {script_abs!r}; "
                        f"os.makedirs(os.path.dirname(p), exist_ok=True); "
                        f"open(p, 'wb').write(zlib.decompress(base64.b64decode('{b64_self}')))"
                    ),
                ]
                with socket.create_connection((host, int(port_s)), timeout=60) as sync_sock:
                  sync_payload = json.dumps({"cmd": sync_cmd, "env": clean_env, "local_work_dir": local_work_dir}) + "\n"
                  sync_sock.sendall(sync_payload.encode("utf-8"))
                  with sync_sock.makefile("r", encoding="utf-8", errors="replace") as sf_sync:
                    for _ in sf_sync:
                      pass
              except Exception:  # pylint: disable=broad-exception-caught
                pass
              try:
                with socket.create_connection((host, int(port_s)), timeout=14400) as sock:
                  run_env = dict(clean_env)
                  run_env["BENCH_PROCESS_START_EPOCH"] = f"{time.time():.6f}"
                  payload = json.dumps({"cmd": c_list, "env": run_env, "local_work_dir": local_work_dir}) + "\n"
                  sock.sendall(payload.encode("utf-8"))
                  with sock.makefile("r", encoding="utf-8", errors="replace") as sf_in:
                    for rline in sf_in:
                      if rline.startswith("__REMOTE_EXIT_CODE__:"):
                        rc = int(rline.strip().split(":", 1)[1])
                      else:
                        log_f.write(rline)
                        log_f.flush()
              except Exception as ex:  # pylint: disable=broad-exception-caught
                log_f.write(f"\n[REMOTE SLICE ERROR]: {ex}\n")
                log_f.flush()
              remote_rc[s_id] = rc
              if rc != 0:
                remote_killed_local.set()
                for lp in procs:
                  if lp.poll() is None:
                    try:
                      lp.kill()
                    except Exception:  # pylint: disable=broad-exception-caught
                      pass

            t = threading.Thread(target=_run_remote, args=(slice_id, target_remote_addr, cmd_i, env, slf), daemon=True)
            t.start()
            remote_threads.append((slice_id, t))
          else:
            slf = open(os.path.join(log_dir, f"{phase}_slice{slice_id}.log"), "w", encoding="utf-8")
            slf.write(f"Command: {' '.join(cmd_i)}\n\n")
            slf.flush()
            slice_log_files.append(slf)
            p = subprocess.Popen(cmd_i, stdout=slf, stderr=subprocess.STDOUT, env=env, cwd=repo_root, text=True)
            procs.append(p)

        with open(log_file, "w", encoding="utf-8") as lf:
          lf.write(
              f"Multi-slice Command (num_slices={num_slices}, hosts_per_slice={num_hosts_per_slice}, "
              f"coord={coord_addr}): {' '.join(base_cmd)}\n\n"
          )
          lf.flush()
          watchdog_state = _start_oom_watchdog(procs[0], platform_name=resolved_platform)
          assert procs[0].stdout is not None
          for line in procs[0].stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            m = re.search(r"\[Worker\] Cached .* in ([0-9.]+)s", line)
            if m:
              worker_cached_times.append(float(m.group(1)))
          for idx, p in enumerate(procs):
            p.wait(timeout=14400)
          watchdog_state["stop"] = True
          if (watchdog_state.get("oom_triggered") or watchdog_state.get("timeout_triggered")) and watchdog_state.get(
              "oom_message"
          ):
            lf.write(watchdog_state["oom_message"])
          if watchdog_state.get("peak_ram_gb", 0.0) > 0:
            peak_msg = (
                f"INFO:watchdog:Subprocess Peak Host RAM: {watchdog_state['peak_ram_gb']:.2f}/"
                f"{watchdog_state.get('total_ram_gb', 251.69):.2f} GB\n"
            )
            sys.stdout.write(peak_msg)
            sys.stdout.flush()
            lf.write(peak_msg)
          lf.flush()
          if remote_killed_local.is_set():
            failed_remote = [(s_id, rc) for s_id, rc in sorted(remote_rc.items()) if rc != 0]
            if failed_remote:
              s_id, rc = failed_remote[0]
              failed_exit_code = rc
              r_ep = remote_endpoints[s_id - 1] if (s_id - 1) < len(remote_endpoints) else "remote"
              raise RuntimeError(
                  f"Multi-slice worker phase '{phase}' remote slice {s_id} ({r_ep}) failed with exit code {rc} "
                  f"(aborted local slice 0). See logs in: {log_dir}"
              )
          for idx, p in enumerate(procs):
            if p.returncode != 0:
              failed_exit_code = p.returncode
              oom_tag = _failure_tag(p.returncode, watchdog_state)
              raise RuntimeError(
                  f"Multi-slice worker phase '{phase}' local slice {idx} failed with exit code {p.returncode}{oom_tag} "
                  f"(Peak RAM: {watchdog_state.get('peak_ram_gb', 0.0):.2f}/{watchdog_state.get('total_ram_gb', 251.69):.2f} GB). "
                  f"See logs in: {log_dir}"
              )
          for s_id, t in remote_threads:
            t.join(timeout=14400)
            rc = remote_rc.get(s_id, 1)
            if rc != 0:
              failed_exit_code = rc
              r_ep = remote_endpoints[s_id - 1] if (s_id - 1) < len(remote_endpoints) else "remote"
              raise RuntimeError(
                  f"Multi-slice worker phase '{phase}' remote slice {s_id} ({r_ep}) failed with exit code {rc}. "
                  f"See logs in: {log_dir}"
              )
      finally:
        for _, t in remote_threads:
          t.join(timeout=15.0)
        for slf in slice_log_files:
          slf.close()
  except RuntimeError as e:
    if os.path.exists(result_json):
      try:
        with open(result_json, "r", encoding="utf-8") as rf_ok:
          existing_ok = json.load(rf_ok)
        if existing_ok.get("status") != "FAILED" and any(
            existing_ok.get(k, 0.0) > 0.0
            for k in ("safetensors_and_convert_s", "load_orbax_s", "convert_to_orbax_s", "hf_to_gcs_s")
        ):
          print(
              f"[WARN] Subprocess exited non-zero during teardown ({e}), but valid result JSON already written: {result_json}",
              flush=True,
          )
          return _finalize_and_persist(existing_ok, status_str="OK")
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    if not allow_step_failure:
      raise
    print(f"[STEP FAILED (continuing due to --allow-step-failure)]: {e}", flush=True)
    parsed_step1_s = 0.0
    if os.path.exists(log_file):
      with open(log_file, "r", encoding="utf-8", errors="replace") as lf_err:
        log_txt = lf_err.read()
        m_s1 = re.search(r"\[safetensor\] Step 1 \(HF->GCS\) completed in ([0-9.]+)s", log_txt)
        if m_s1:
          parsed_step1_s = float(m_s1.group(1))
        m_orb = re.search(
            r"\[to_maxtext Step 3\] Loaded Orbax checkpoint \(([0-9]+) parameter leaves, num_slices=([0-9]+), global_devices=([0-9]+)\) in ([0-9.]+)s",
            log_txt,
        )
        if m_orb and phase == "to_maxtext_step3":
          m_init = re.search(r"\[to_maxtext Step 3\] Abstract state initialized in ([0-9.]+)s", log_txt)
          rec_orb = {
              "load_orbax_s": float(m_orb.group(4)),
              "init_abstract_state_s": float(m_init.group(1)) if m_init else 0.0,
              "num_param_leaves": int(m_orb.group(1)),
              "num_slices": int(m_orb.group(2)),
              "global_devices": int(m_orb.group(3)),
              "peak_host_ram_gb": round(watchdog_state.get("peak_ram_gb", 0.0), 2),
              "total_host_ram_gb": round(watchdog_state.get("total_ram_gb", 251.69), 2),
          }
          return _finalize_and_persist(rec_orb, status_str="OK")
        m_st = re.search(
            r"\[safetensor\] Completed: \(1\) HF->GCS=([0-9.]+)s, \(2\) safetensors & convert to MaxText=([0-9.]+)s \(total=([0-9.]+)s, ([0-9]+) parameter leaves, num_slices=([0-9]+), global_devices=([0-9]+)\)",
            log_txt,
        )
        if m_st and phase == "safetensors_dynamic":
          m_init = re.search(r"\[safetensor\] Abstract state initialized in ([0-9.]+)s", log_txt)
          m_shd = re.search(r"\[safetensor\] load_sharded_hf_state completed in ([0-9.]+)s", log_txt)
          m_tr = re.search(r"CPU Transformations completed in ([0-9.]+)s", log_txt)
          rec_st = {
              "hf_to_gcs_s": float(m_st.group(1)),
              "safetensors_and_convert_s": float(m_st.group(2)),
              "derive_mappings_s": 0.0,
              "load_sharded_hf_s": float(m_shd.group(1)) if m_shd else 0.0,
              "transform_hf_to_mt_s": float(m_tr.group(1)) if m_tr else 0.0,
              "init_abstract_state_s": float(m_init.group(1)) if m_init else 0.0,
              "total_s": float(m_st.group(3)),
              "num_param_leaves": int(m_st.group(4)),
              "num_slices": int(m_st.group(5)),
              "global_devices": int(m_st.group(6)),
              "worker_upload_max_s": max(worker_cached_times) if worker_cached_times else 0.0,
              "peak_host_ram_gb": round(watchdog_state.get("peak_ram_gb", 0.0), 2),
              "total_host_ram_gb": round(watchdog_state.get("total_ram_gb", 251.69), 2),
          }
          return _finalize_and_persist(rec_st, status_str="OK")
    fail_raw: dict[str, Any] = {
        "status": "FAILED",
        "error": str(e),
        "num_slices": num_slices if is_multislice_phase else 1,
        "num_hosts_per_slice": num_hosts_per_slice if is_multislice_phase else 1,
        "global_devices": (num_slices if is_multislice_phase else 1) * devices_per_slice,
    }
    if parsed_step1_s > 0.0:
      fail_raw["hf_to_gcs_s"] = parsed_step1_s
    return _finalize_and_persist(fail_raw, status_str="FAILED", err_str=str(e), exit_code=failed_exit_code)
  finally:
    if (
        os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1"
        and os.path.isdir(local_hf_dir)
        and hasattr(os, "posix_fadvise")
    ):
      for fn in os.listdir(local_hf_dir):
        if fn.endswith(".safetensors"):
          try:
            with open(os.path.join(local_hf_dir, fn), "rb") as sf_evict:
              os.posix_fadvise(sf_evict.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
    _mirror_logs_to_gcs()

  data = _persist_worker_upload_max_s(result_json, worker_cached_times)
  return _finalize_and_persist(data, status_str="OK")


def print_summary_table(results: list[dict[str, Any]], csv_path: str) -> None:
  """Prints a formatted markdown comparison table and writes results to CSV."""
  fieldnames = [
      "model",
      "scan_mode",
      "num_slices",
      "num_hosts_per_slice",
      "global_devices",
      "to_maxtext_1_hf_to_gcs_s",
      "to_maxtext_1b_hf_to_local_s",
      "to_maxtext_2_convert_to_orbax_s",
      "to_maxtext_2a_init_abstract_model_s",
      "to_maxtext_2b_transform_and_shard_s",
      "to_maxtext_2c_orbax_save_to_gcs_s",
      "to_maxtext_3_load_orbax_s",
      "to_maxtext_total_s",
      "safetensor_1_hf_to_gcs_s",
      "safetensor_1_worker_upload_s",
      "safetensor_1_warm_cache_s",
      "safetensor_2_safetensors_and_convert_s",
      "safetensor_2a_derive_mappings_s",
      "safetensor_2b_load_sharded_hf_s",
      "safetensor_2c_transform_hf_to_mt_s",
      "safetensor_total_s",
      "speedup_total_vs_to_maxtext_x",
      "load_time_diff_safetensor_step2_vs_orbax_step3_s",
  ]
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
      writer.writerow({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in row.items() if k in fieldnames})

  print(f"\n{'=' * 100}")
  print("RUNTIME (LOAD TIME) COMPARISON SUMMARY")
  print(f"{'=' * 100}")
  for r in results:
    print(
        f"\nModel: {r['model']} | Scan Mode: {r['scan_mode']} "
        f"| Slices: {r.get('num_slices', 1)} (Global Devices: {r.get('global_devices', 1)})"
    )
    print("-" * 80)
    print("  [to_maxtext]")
    print(
        f"    1. HF -> GCS                              : {r['to_maxtext_1_hf_to_gcs_s']:8.2f} s"
        f"  (worker upload: {r.get('to_maxtext_1_worker_upload_s', 0.0):.2f}s,"
        f" local stage: {r['to_maxtext_1b_hf_to_local_s']:.2f}s)"
    )
    print(
        f"    2. Convert to an Orbax checkpoint         : {r['to_maxtext_2_convert_to_orbax_s']:8.2f} s"
        f"  (init: {r['to_maxtext_2a_init_abstract_model_s']:.2f}s,"
        f" transform+shard: {r['to_maxtext_2b_transform_and_shard_s']:.2f}s,"
        f" orbax save: {r['to_maxtext_2c_orbax_save_to_gcs_s']:.2f}s)"
    )
    print(f"    3. Load the Orbax checkpoint              : {r['to_maxtext_3_load_orbax_s']:8.2f} s")
    print(f"    --------------------------------------------------------------")
    print(f"    Total (1 + 2 + 3)                         : {r['to_maxtext_total_s']:8.2f} s")
    print()
    print("  [safetensor (safetensors_dynamic)]")
    print(
        f"    1. HF -> GCS                              : {r['safetensor_1_hf_to_gcs_s']:8.2f} s"
        f"  (worker upload: {r.get('safetensor_1_worker_upload_s', 0.0):.2f}s)"
    )
    print(
        f"    2. Safetensors and convert to MaxText     : {r['safetensor_2_safetensors_and_convert_s']:8.2f} s"
        f"  ([1/3] mappings: {r['safetensor_2a_derive_mappings_s']:.2f}s,"
        f" [2/3] GCS load: {r['safetensor_2b_load_sharded_hf_s']:.2f}s,"
        f" [3/3] transform: {r['safetensor_2c_transform_hf_to_mt_s']:.2f}s)"
    )
    print(f"    --------------------------------------------------------------")
    print(f"    Total (1 + 2)                             : {r['safetensor_total_s']:8.2f} s")
    print()
    print(f"  [Comparison]")
    print(f"    End-to-End Speedup (to_maxtext / safetensor) : {r['speedup_total_vs_to_maxtext_x']:.2f}x")
    print(
        f"    Warm Load (safetensor Step 2 vs Orbax Step 3): {r['safetensor_2_safetensors_and_convert_s']:.2f}s"
        f" vs {r['to_maxtext_3_load_orbax_s']:.2f}s"
        f" ({r['load_time_diff_safetensor_step2_vs_orbax_step3_s']:+.2f}s)"
    )
  print(f"\nCSV report written to: {csv_path}")
  print(f"{'=' * 100}\n", flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark load time comparison: to_maxtext vs safetensor.")
  parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to benchmark.")
  parser.add_argument(
      "--scan-modes",
      nargs="+",
      default=DEFAULT_SCAN_MODES,
      choices=["scanned", "unscanned"],
      help="Scan modes to benchmark.",
  )
  parser.add_argument(
      "--num-slices-list",
      nargs="+",
      type=int,
      default=[1],
      help="List of slice counts (num_slices) to benchmark (e.g. 1 2).",
  )
  parser.add_argument(
      "--hardware",
      default="auto",
      choices=["auto", "cpu", "tpu"],
      help="Hardware backend for MaxText checkpoint loading (auto detects TPU if available).",
  )
  parser.add_argument("--gcs-base", default=DEFAULT_GCS_BUCKET, help="Base GCS bucket path for benchmark artifacts.")
  parser.add_argument(
      "--csv-report",
      default="benchmark_ckpt_load_time_summary.csv",
      help="Path to write the summary CSV.",
  )
  parser.add_argument(
      "--run-id",
      default=None,
      help="Optional run ID to use (or resume when combined with --skip-completed).",
  )
  parser.add_argument(
      "--skip-completed",
      action="store_true",
      help="Skip worker phases whose result JSON already exists in the run workdir.",
  )
  parser.add_argument(
      "--reuse-model-hf-cache",
      action="store_true",
      help=(
          "Measure cold HF->GCS once per model (in scanned / num_slices=1) and reuse the populated "
          "GCS hf_cache for subsequent scan_mode/num_slices runs of the same model."
      ),
  )
  parser.add_argument(
      "--local-work-dir",
      default=None,
      help="Optional custom local working directory for staging and logs.",
  )
  parser.add_argument(
      "--weight-dtype",
      default="auto",
      help="Weight dtype for MaxText checkpoint loading (auto uses bfloat16 for large models).",
  )
  parser.add_argument(
      "--dcn-parallelism-axis",
      default="auto",
      choices=["auto", "data", "fsdp"],
      help=(
          "Multi-slice DCN parallelism axis: 'auto' uses dcn_fsdp_parallelism for >=70B models "
          "and dcn_data_parallelism for smaller models; 'fsdp' or 'data' forces that axis."
      ),
  )
  parser.add_argument(
      "--devices-per-slice",
      type=int,
      default=4,
      help=(
          "Number of TPU devices in one slice, counted across all of its hosts (devices per host x "
          "--num-hosts-per-slice, e.g. 8 for a tpu7x 2x2x1 and 16 for a tpu7x 2x2x2). Failed and skipped results "
          "record num_slices x this as global_devices. Also the device count when simulating CPU devices "
          "(default: 4, matching v5p-8)."
      ),
  )
  parser.add_argument(
      "--remote-slice-addrs",
      nargs="*",
      default=None,
      help=(
          "Optional host:port endpoint(s) of remote slice daemon(s) for multi-node cluster runs "
          "(also configurable via REMOTE_SLICE1_ADDR env var)."
      ),
  )
  parser.add_argument(
      "--remote-slice-daemon",
      action="store_true",
      help="Run as a remote slice worker daemon on a companion node.",
  )
  parser.add_argument(
      "--daemon-port",
      type=int,
      default=55555,
      help="TCP port for --remote-slice-daemon.",
  )
  parser.add_argument(
      "--allow-step-failure",
      action="store_true",
      help="Record OOM or step failures in result JSON and continue remaining phases.",
  )
  parser.add_argument(
      "--skip-to-maxtext",
      action="store_true",
      help="Skip to_maxtext Steps 1-3 and only run safetensors_dynamic.",
  )
  parser.add_argument(
      "--skip-safetensors",
      action="store_true",
      help="Skip safetensors_dynamic and only run to_maxtext Steps 1-3 (e.g. to add to_maxtext to finished runs).",
  )
  parser.add_argument(
      "--tm-convert-only",
      action="store_true",
      help=(
          "Run only to_maxtext Steps 1-2 (HF->GCS and CPU conversion to Orbax) and skip Step 3 and safetensors_dynamic,"
          " e.g. to convert on one slice a checkpoint that only fits in HBM on two."
      ),
  )
  parser.add_argument(
      "--num-hosts-per-slice",
      type=int,
      default=1,
      help="Number of TPU hosts per slice (e.g., 1 for 2x2x1, 2 for 2x2x2).",
  )
  parser.add_argument(
      "--platform",
      default=os.environ.get("BENCH_PLATFORM", "v7x"),
      choices=["v5e", "v6e", "v7x"],
      help="Target TPU platform profile (v5e, v6e, or v7x).",
  )
  parser.add_argument(
      "--slice-topology",
      default=os.environ.get("BENCH_SLICE_TOPOLOGY", ""),
      help="Per-slice TPU topology label (e.g., 2x2x1, 2x4, 4x4, 16x16).",
  )
  parser.add_argument(
      "--verify-checksums",
      action="store_true",
      help="Compute and compare post-load sample parameter checksums between to_maxtext and safetensors_dynamic.",
  )
  # Internal worker arguments
  parser.add_argument("--worker-phase", default=None, help=argparse.SUPPRESS)
  parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
  parser.add_argument("--scan-mode", default=None, help=argparse.SUPPRESS)
  parser.add_argument("--num-slices", type=int, default=1, help=argparse.SUPPRESS)
  parser.add_argument("--slice-id", type=int, default=0, help=argparse.SUPPRESS)
  parser.add_argument("--coordinator-address", default="127.0.0.1:12345", help=argparse.SUPPRESS)
  parser.add_argument("--gcs-run-dir", default=None, help=argparse.SUPPRESS)
  parser.add_argument("--local-hf-dir", default=None, help=argparse.SUPPRESS)
  parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)

  args = parser.parse_args()
  repo_root = get_repo_root()
  src_dir = os.path.join(repo_root, "src")
  if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

  if args.remote_slice_daemon:
    os.chdir(repo_root)
    run_remote_slice_daemon(args.daemon_port)
    return

  if args.worker_phase:
    import traceback

    os.chdir(repo_root)
    exec(SITE_CUSTOMIZE_CONTENT, globals())  # pylint: disable=exec-used
    harness_v2, _, _, stages = _import_bench_modules()
    recorder = stages.StageRecorder()
    start_epoch_s = os.environ.get("BENCH_PROCESS_START_EPOCH", "").strip()
    if start_epoch_s:
      try:
        recorder.record_value(
            "setup_process_start_to_harness_code_seconds",
            max(0.0, _ENTER_WALL_EPOCH - float(start_epoch_s)),
        )
      except ValueError:
        pass
    recorder.record_value("setup_python_imports_seconds", max(0.0, time.perf_counter() - _ENTER_MONO))
    mon = stages.MonitoringCollector(recorder)
    sampler = harness_v2.ResourceSampler(recorder, interval_s=0.5)
    sampler.start()
    args._recorder = recorder  # pylint: disable=protected-access
    args._monitoring = mon  # pylint: disable=protected-access
    args._sampler = sampler  # pylint: disable=protected-access

    self_wd = _start_self_oom_watchdog(platform_name=getattr(args, "platform", None))
    worker_exit_code = 0
    worker_failed_stage = ""
    res: dict[str, Any] | None = None
    try:
      if args.worker_phase == "to_maxtext_step1":
        res = run_worker_to_maxtext_step1_hf_to_gcs(args)
      elif args.worker_phase == "to_maxtext_step2":
        res = run_worker_to_maxtext_step2_convert(args)
      elif args.worker_phase == "to_maxtext_step3":
        res = run_worker_to_maxtext_step3_load_orbax(args)
      elif args.worker_phase == "safetensors_dynamic":
        res = run_worker_safetensors_dynamic(args)
      else:
        raise ValueError(f"Unknown worker phase: {args.worker_phase}")
    except Exception:  # pylint: disable=broad-exception-caught
      worker_exit_code = 1
      worker_failed_stage = recorder.current_stage() or ""
      traceback.print_exc()
    finally:
      t_td0 = time.perf_counter()
      self_wd["stop"] = True
      if self_wd.get("peak_ram_gb", 0.0) > 0:
        print(
            f"INFO:watchdog:Worker Peak Host RAM: {self_wd['peak_ram_gb']:.2f}/{self_wd.get('total_ram_gb', 251.69):.2f} GB",
            flush=True,
        )
        if res is not None:
          res["peak_host_ram_gb"] = round(self_wd["peak_ram_gb"], 2)
          res["total_host_ram_gb"] = round(self_wd.get("total_ram_gb", 251.69), 2)
      if res is not None and args.slice_id == 0:
        _write_result_json(args.result_json, res)
      mon.unregister()
      recorder.restore_all()
      sampler.stop()
      mem_payload = sampler.memory_payload()
      mon_snapshot = mon.snapshot()
      recorder.record_value("teardown_write_results_and_exit_seconds", max(0.0, time.perf_counter() - t_td0))
      num_hosts_per_slice = max(1, int(getattr(args, "num_hosts_per_slice", 1)))
      host_payload = {
          "process_index": int(getattr(args, "slice_id", 0)),
          "slice_id": int(getattr(args, "slice_id", 0)) // num_hosts_per_slice,
          "host_name": stages.default_host_name(),
          "pod_name": os.environ.get("POD_NAME") or os.environ.get("HOSTNAME") or "",
          "node_name": os.environ.get("NODE_NAME") or "",
          "failed_stage": worker_failed_stage,
          "active_stage": "" if worker_exit_code == 0 else (recorder.current_stage() or ""),
          "monitoring": mon_snapshot,
          "memory": mem_payload,
          **recorder.snapshot(),
      }
      stages.emit_host_stages_line(host_payload)
      sys.stdout.flush()
      sys.stderr.flush()
      if args.slice_id != 0:
        time.sleep(0.5)
      os._exit(worker_exit_code)  # pylint: disable=protected-access

  run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  gcs_run_dir = f"{args.gcs_base.rstrip('/')}/run_{run_id}"
  local_work_dir = args.local_work_dir or f"/tmp/benchmark_ckpt_load_time_{run_id}"
  csv_report_path = os.path.abspath(args.csv_report)
  resolved_topology = _resolve_slice_topology(
      args.platform, args.slice_topology, args.devices_per_slice, args.num_hosts_per_slice
  )

  print(f"Starting checkpoint load time benchmark (run_id={run_id})")
  print(f"  Repo root    : {repo_root}")
  print(f"  Platform     : {args.platform} (topology={resolved_topology})")
  print(f"  Models       : {args.models}")
  print(f"  Scan modes   : {args.scan_modes}")
  print(f"  Num slices   : {args.num_slices_list}")
  print(f"  DCN axis     : {args.dcn_parallelism_axis}")
  print(f"  Hardware     : {args.hardware} (resolved: {resolve_hardware(args.hardware)})")
  print(f"  Weight dtype : {args.weight_dtype}")
  print(f"  Checksums    : {args.verify_checksums}")
  print(f"  GCS run dir  : {gcs_run_dir}")
  print(f"  Local workdir: {local_work_dir}")
  print(f"  CSV report   : {csv_report_path}")

  if args.skip_completed and gcs_run_dir.startswith("gs://"):
    try:
      from etils import epath

      for model in args.models:
        gcs_model_log_dir = epath.Path(f"{gcs_run_dir}/logs/{model}")
        if gcs_model_log_dir.exists():
          for sub in gcs_model_log_dir.iterdir():
            if sub.is_dir():
              local_sub = os.path.join(local_work_dir, "logs", model, sub.name)
              os.makedirs(local_sub, exist_ok=True)
              for f in sub.iterdir():
                local_f = os.path.join(local_sub, f.name)
                if not os.path.exists(local_f):
                  epath.Path(local_f).write_bytes(f.read_bytes())
                  print(f"[RESTORE FROM GCS]: {f} -> {local_f}", flush=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"[WARN] Failed to restore logs from GCS: {e}", flush=True)

  all_results: list[dict[str, Any]] = []

  def _skip_result(
      *,
      phase_name: str,
      model_name: str,
      scan_mode_name: str,
      slices_count: int,
      hosts_count: int,
      reason: str,
  ) -> dict[str, Any]:
    s_dir = _phase_scan_dir_name(phase_name, scan_mode_name, slices_count, hosts_count)
    existing_path = os.path.join(local_work_dir, "logs", model_name, s_dir, f"{phase_name}_result.json")
    if args.skip_completed and os.path.exists(existing_path):
      with open(existing_path, "r", encoding="utf-8") as rf_ex:
        return json.load(rf_ex)
    return _write_skipped_v2_result(
        phase=phase_name,
        model=model_name,
        scan_mode=scan_mode_name,
        platform_name=args.platform,
        slice_topology=resolved_topology,
        num_slices=slices_count,
        num_hosts_per_slice=hosts_count,
        devices_per_slice=args.devices_per_slice,
        local_work_dir=local_work_dir,
        gcs_run_dir=gcs_run_dir,
        skip_reason=reason,
    )

  for model in args.models:
    cold_st_step1_s: float | None = None
    cold_st_worker_s: float = 0.0
    m_step1_by_scan: dict[str, dict[str, Any]] = {}
    m_step2_by_scan: dict[str, dict[str, Any]] = {}

    # Pass 1: Run to_maxtext Step 1 (HF -> GCS + local stage) and Step 2 (Convert to Orbax)
    # across all scan_modes first, then remove the local HF staging directory to free disk/RAM
    # before running Step 3 and safetensors_dynamic across single/multi-slice configurations.
    skip_to_maxtext = args.skip_to_maxtext or os.environ.get("SKIP_TO_MAXTEXT", "0") == "1"
    for scan_mode in args.scan_modes:
      if skip_to_maxtext:
        m_step1_by_scan[scan_mode] = _skip_result(
            phase_name="to_maxtext_step1",
            model_name=model,
            scan_mode_name=scan_mode,
            slices_count=1,
            hosts_count=1,
            reason="--skip-to-maxtext",
        )
        m_step2_by_scan[scan_mode] = _skip_result(
            phase_name="to_maxtext_step2",
            model_name=model,
            scan_mode_name=scan_mode,
            slices_count=1,
            hosts_count=1,
            reason="--skip-to-maxtext",
        )
        continue
      m_step1_by_scan[scan_mode] = invoke_worker_subprocess(
          phase="to_maxtext_step1",
          model=model,
          scan_mode=scan_mode,
          gcs_run_dir=gcs_run_dir,
          local_work_dir=local_work_dir,
          hardware=args.hardware,
          weight_dtype=args.weight_dtype,
          dcn_parallelism_axis=args.dcn_parallelism_axis,
          devices_per_slice=args.devices_per_slice,
          remote_slice_addrs=args.remote_slice_addrs,
          skip_completed=args.skip_completed,
          reuse_model_hf_cache=args.reuse_model_hf_cache,
          allow_step_failure=args.allow_step_failure,
          platform_name=args.platform,
          slice_topology=resolved_topology,
          verify_checksums=args.verify_checksums,
      )
      if m_step1_by_scan[scan_mode].get("status") == "FAILED" and args.allow_step_failure:
        print(
            f"[SKIP to_maxtext_step2]: Step 1 failed for {model} ({scan_mode}); skipping Orbax conversion.",
            flush=True,
        )
        m_step2_by_scan[scan_mode] = _skip_result(
            phase_name="to_maxtext_step2",
            model_name=model,
            scan_mode_name=scan_mode,
            slices_count=1,
            hosts_count=1,
            reason="SKIPPED_DUE_TO_STEP1_FAILURE",
        )
        continue
      m_step2_by_scan[scan_mode] = invoke_worker_subprocess(
          phase="to_maxtext_step2",
          model=model,
          scan_mode=scan_mode,
          gcs_run_dir=gcs_run_dir,
          local_work_dir=local_work_dir,
          hardware=args.hardware,
          weight_dtype=args.weight_dtype,
          dcn_parallelism_axis=args.dcn_parallelism_axis,
          devices_per_slice=args.devices_per_slice,
          remote_slice_addrs=args.remote_slice_addrs,
          extra_env={"XLA_FLAGS": "--xla_force_host_platform_device_count=16"},
          skip_completed=args.skip_completed,
          reuse_model_hf_cache=args.reuse_model_hf_cache,
          allow_step_failure=args.allow_step_failure,
          platform_name=args.platform,
          slice_topology=resolved_topology,
          verify_checksums=args.verify_checksums,
      )

    # Clean up staged local HF weights once Step 2 conversions for this model are done
    for _stage_root in (local_work_dir, "/ramdisk"):
      local_hf_model_dir = os.path.join(_stage_root, "hf_local", model)
      if os.environ.get("USE_LOCAL_HF_FOR_SAFETENSORS") != "1":
        shutil.rmtree(local_hf_model_dir, ignore_errors=True)
      shutil.rmtree(f"{local_hf_model_dir}_cache", ignore_errors=True)

    # Pass 2: Run to_maxtext Step 3 (Load Orbax) and safetensors_dynamic across num_slices and scan_modes
    # Run all single-slice configurations first before multi-slice configurations.
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    s_res_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for num_slices in args.num_slices_list:
      for scan_mode in args.scan_modes:
        m_step1 = m_step1_by_scan[scan_mode]
        m_step2 = m_step2_by_scan[scan_mode]
        if skip_to_maxtext or args.tm_convert_only or (m_step2.get("status") == "FAILED" and args.allow_step_failure):
          skip_why = (
              "--skip-to-maxtext"
              if skip_to_maxtext
              else ("--tm-convert-only" if args.tm_convert_only else "SKIPPED_DUE_TO_STEP2_FAILURE")
          )
          if not skip_to_maxtext and not args.tm_convert_only:
            print(
                f"[SKIP to_maxtext_step3]: Step 2 failed for {model} ({scan_mode}); skipping Orbax load.",
                flush=True,
            )
          m_step3 = _skip_result(
              phase_name="to_maxtext_step3",
              model_name=model,
              scan_mode_name=scan_mode,
              slices_count=num_slices,
              hosts_count=args.num_hosts_per_slice,
              reason=skip_why,
          )
        else:
          m_step3 = invoke_worker_subprocess(
              phase="to_maxtext_step3",
              model=model,
              scan_mode=scan_mode,
              gcs_run_dir=gcs_run_dir,
              local_work_dir=local_work_dir,
              hardware=args.hardware,
              num_slices=num_slices,
              num_hosts_per_slice=args.num_hosts_per_slice,
              weight_dtype=args.weight_dtype,
              dcn_parallelism_axis=args.dcn_parallelism_axis,
              devices_per_slice=args.devices_per_slice,
              remote_slice_addrs=args.remote_slice_addrs,
              skip_completed=args.skip_completed,
              reuse_model_hf_cache=args.reuse_model_hf_cache,
              allow_step_failure=args.allow_step_failure,
              platform_name=args.platform,
              slice_topology=resolved_topology,
              verify_checksums=args.verify_checksums,
          )

        if (
            scan_mode == "unscanned"
            and ("scanned", num_slices) in s_res_by_key
            and s_res_by_key[("scanned", num_slices)].get("status") == "FAILED"
            and s_res_by_key[("scanned", num_slices)].get("load_sharded_hf_s", 0.0) == 0.0
            and args.allow_step_failure
        ):
          print(
              f"[SKIP safetensors_dynamic]: {model} (unscanned, num_slices={num_slices}) skipped because "
              f"scanned on num_slices={num_slices} already failed with OOM/timeout.",
              flush=True,
          )
          s_res = _skip_result(
              phase_name="safetensors_dynamic",
              model_name=model,
              scan_mode_name=scan_mode,
              slices_count=num_slices,
              hosts_count=args.num_hosts_per_slice,
              reason=f"Skipped unscanned because scanned failed on num_slices={num_slices}",
          )
        elif args.skip_safetensors or args.tm_convert_only:
          s_res = _skip_result(
              phase_name="safetensors_dynamic",
              model_name=model,
              scan_mode_name=scan_mode,
              slices_count=num_slices,
              hosts_count=args.num_hosts_per_slice,
              reason="--tm-convert-only" if args.tm_convert_only else "--skip-safetensors",
          )
        else:
          s_res = invoke_worker_subprocess(
              phase="safetensors_dynamic",
              model=model,
              scan_mode=scan_mode,
              gcs_run_dir=gcs_run_dir,
              local_work_dir=local_work_dir,
              hardware=args.hardware,
              num_slices=num_slices,
              num_hosts_per_slice=args.num_hosts_per_slice,
              weight_dtype=args.weight_dtype,
              dcn_parallelism_axis=args.dcn_parallelism_axis,
              devices_per_slice=args.devices_per_slice,
              remote_slice_addrs=args.remote_slice_addrs,
              skip_completed=args.skip_completed,
              reuse_model_hf_cache=args.reuse_model_hf_cache,
              allow_step_failure=args.allow_step_failure,
              platform_name=args.platform,
              slice_topology=resolved_topology,
              verify_checksums=args.verify_checksums,
          )
        s_res_by_key[(scan_mode, num_slices)] = s_res
        _maybe_verify_and_persist_checksums(
            model=model,
            scan_mode=scan_mode,
            num_slices=num_slices,
            num_hosts_per_slice=args.num_hosts_per_slice,
            m_step3=m_step3,
            s_res=s_res,
            local_work_dir=local_work_dir,
            gcs_run_dir=gcs_run_dir,
        )

        if s_res.get("worker_upload_max_s", 0.0) > 0.0:
          cold_st_step1_s = s_res.get("hf_to_gcs_s", 0.0)
          cold_st_worker_s = s_res.get("worker_upload_max_s", 0.0)
          st_step1_s = s_res.get("hf_to_gcs_s", 0.0)
          st_worker_s = cold_st_worker_s
          st_warm_cache_s = 0.0
        elif cold_st_step1_s is None and m_step1.get("hf_to_gcs_s", 0.0) > s_res.get("hf_to_gcs_s", 0.0):
          cold_st_step1_s = m_step1.get("hf_to_gcs_s", 0.0)
          cold_st_worker_s = m_step1.get("worker_upload_max_s", 0.0)
          st_step1_s = cold_st_step1_s
          st_worker_s = cold_st_worker_s
          st_warm_cache_s = s_res.get("hf_to_gcs_s", 0.0)
        elif cold_st_step1_s is None:
          cold_st_step1_s = s_res.get("hf_to_gcs_s", 0.0)
          cold_st_worker_s = s_res.get("worker_upload_max_s", 0.0)
          st_step1_s = s_res.get("hf_to_gcs_s", 0.0)
          st_worker_s = cold_st_worker_s
          st_warm_cache_s = 0.0
        else:
          st_step1_s = cold_st_step1_s
          st_worker_s = cold_st_worker_s
          st_warm_cache_s = s_res.get("hf_to_gcs_s", 0.0)

        to_maxtext_total_s = (
            m_step1.get("hf_to_gcs_s", 0.0) + m_step2.get("convert_to_orbax_s", 0.0) + m_step3.get("load_orbax_s", 0.0)
        )
        safetensor_total_s = st_step1_s + s_res.get("safetensors_and_convert_s", 0.0)

        row = {
            "model": model,
            "scan_mode": scan_mode,
            "num_slices": num_slices,
            "num_hosts_per_slice": args.num_hosts_per_slice,
            "global_devices": s_res.get("global_devices", m_step3.get("global_devices", 1)),
            "to_maxtext_1_hf_to_gcs_s": m_step1.get("hf_to_gcs_s", 0.0),
            "to_maxtext_1_worker_upload_s": m_step1.get("worker_upload_max_s", 0.0),
            "to_maxtext_1b_hf_to_local_s": m_step1.get("hf_to_local_s", 0.0),
            "to_maxtext_2_convert_to_orbax_s": m_step2.get("convert_to_orbax_s", 0.0),
            "to_maxtext_2a_init_abstract_model_s": m_step2.get("init_abstract_model_s", 0.0),
            "to_maxtext_2b_transform_and_shard_s": m_step2.get("transform_and_shard_s", 0.0),
            "to_maxtext_2c_orbax_save_to_gcs_s": m_step2.get("orbax_save_to_gcs_s", 0.0),
            "to_maxtext_3_load_orbax_s": m_step3.get("load_orbax_s", 0.0),
            "to_maxtext_total_s": to_maxtext_total_s,
            "safetensor_1_hf_to_gcs_s": st_step1_s,
            "safetensor_1_worker_upload_s": st_worker_s,
            "safetensor_1_warm_cache_s": st_warm_cache_s,
            "safetensor_2_safetensors_and_convert_s": s_res.get("safetensors_and_convert_s", 0.0),
            "safetensor_2a_derive_mappings_s": s_res.get("derive_mappings_s", 0.0),
            "safetensor_2b_load_sharded_hf_s": s_res.get("load_sharded_hf_s", 0.0),
            "safetensor_2c_transform_hf_to_mt_s": s_res.get("transform_hf_to_mt_s", 0.0),
            "safetensor_total_s": safetensor_total_s,
            "speedup_total_vs_to_maxtext_x": (to_maxtext_total_s / safetensor_total_s if safetensor_total_s > 0 else 0.0),
            "load_time_diff_safetensor_step2_vs_orbax_step3_s": (
                s_res.get("safetensors_and_convert_s", 0.0) - m_step3.get("load_orbax_s", 0.0)
            ),
        }
        rows_by_key[(scan_mode, num_slices)] = row

    for scan_mode in args.scan_modes:
      for num_slices in args.num_slices_list:
        all_results.append(rows_by_key[(scan_mode, num_slices)])

  print_summary_table(all_results, csv_report_path)


if __name__ == "__main__":
  main()
