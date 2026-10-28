"""Benchmark runtime (load time) comparison between `to_maxtext` and `safetensor` (`safetensors_dynamic`).

For `to_maxtext`, records:
  1. HF -> GCS execution time
  2. Convert to an Orbax checkpoint execution time
  3. Load the Orbax checkpoint execution time

For `safetensor` (`safetensors_dynamic`), records:
  1. HF -> GCS execution time
  2. Safetensors and convert to MaxText execution time

Usage:
  PYTHONPATH=src /usr/local/google/home/mesakh/Documents/mesakh/py312/bin/python \
      benchmark_ckpt_load_time.py --models qwen3-0.6b
"""

import argparse
import concurrent.futures
import csv
import datetime
import functools
import json
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

# Register OmegaConf resolver for HF_TOKEN if present in configs
try:
  from omegaconf import OmegaConf

  if not OmegaConf.has_resolver("HF_TOKEN"):
    OmegaConf.register_new_resolver("HF_TOKEN", lambda: os.environ.get("HF_TOKEN", ""))
except ImportError:
  pass

# Ensure newer HuggingFace Config classes fall back to PretrainedConfig on older transformers
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
  if (
      os.path.exists("/dev/vfio/0")
      or os.path.exists("/dev/accel0")
      or os.environ.get("TPU_ACCELERATOR_TYPE")
  ):
    return "tpu"
  return "cpu"


def get_concurrent_gb(model_name: str, scan_mode: str = "scanned", num_slices: int = 1) -> int:
  """Returns checkpoint_storage_concurrent_gb sized to fit the largest single leaf while minimizing transient RAM."""
  if scan_mode == "unscanned":
    return 16
  if model_name == "llama3.1-70b":
    return 20 if num_slices > 1 else 38
  if model_name in ("mixtral-8x7b", "gpt-oss-120b", "qwen3-next-80b-a3b"):
    return 45
  if model_name in ("mixtral-8x22b",):
    return 96
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
  unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(
      config, mesh, create_model_partial, is_training=False
  )
  return mesh, unboxed_abstract_state


def _stage_sparse_bf16_safetensors(model_name: str, target_dir: str) -> tuple[int, int, float]:
  """Creates sparse BF16 .safetensors files matching HF_SHAPE[model_name] for cluster OOM testing."""
  import math
  import struct
  from maxtext.checkpoint_conversion.utils import hf_model_configs
  from maxtext.checkpoint_conversion.utils import hf_shape

  os.makedirs(target_dir, exist_ok=True)
  marker = os.path.join(target_dir, ".sparse_ready")
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
    if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
      try:
        os.remove(os.path.join(target_dir, fn))
      except OSError:
        pass

  with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as cf:
    json.dump(cfg_dict, cf, indent=2)

  header_dict = {}
  offset = 0
  for k, shp in clean_shapes.items():
    nbytes = int(math.prod(shp)) * 2
    header_dict[k] = {"dtype": "BF16", "shape": shp, "data_offsets": [offset, offset + nbytes]}
    offset += nbytes

  header_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
  pad_len = (8 - (len(header_bytes) % 8)) % 8
  if pad_len:
    header_bytes += b" " * pad_len
  st_path = os.path.join(target_dir, "model-00001-of-00001.safetensors")
  with open(st_path, "wb") as sf:
    sf.write(struct.pack("<Q", len(header_bytes)))
    sf.write(header_bytes)
    sf.flush()
    os.truncate(sf.fileno(), 8 + len(header_bytes) + offset)

  with open(index_path, "w", encoding="utf-8") as ix_f:
    json.dump(
        {
            "metadata": {"total_size": offset},
            "weight_map": {k: "model-00001-of-00001.safetensors" for k in clean_shapes},
        },
        ix_f,
    )

  with open(marker, "w", encoding="utf-8") as mf:
    mf.write(f"{total_gb:.2f}\n")
  return 1, len(clean_shapes), total_gb


def run_worker_to_maxtext_step1_hf_to_gcs(args: argparse.Namespace) -> dict[str, Any]:
  """Step 1 for `to_maxtext`: Transfer HF checkpoint from HF Hub to GCS (and stage locally for conversion)."""
  import huggingface_hub
  from maxtext.checkpoint_conversion.utils import load_dynamic
  from maxtext.utils import globals as maxtext_globals

  repo_id = maxtext_globals.HF_IDS[args.model]
  token = get_hf_token() or None

  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1" and args.model in (
      "deepseek3-671b",
      "deepseek3.2-671b",
      "deepseek4-284b",
      "deepseek4-1600b",
  ):
    t_gcs_start = time.perf_counter()
    fs = huggingface_hub.HfFileSystem(token=token)
    print(f"[to_maxtext Step 1] Inspecting HF Hub repository {repo_id} for {args.model} ...", flush=True)
    try:
      files = fs.glob(f"{repo_id}/*.safetensors")
      if files:
        print(
            f"[to_maxtext Step 1] Found {len(files)} .safetensors shards on HF Hub for {repo_id}. "
            f"Inspecting native shard header for quantization format...",
            flush=True,
        )
        import struct

        with fs.open(files[0], "rb") as rf:
          hlen = struct.unpack("<Q", rf.read(8))[0]
          hdr = json.loads(rf.read(hlen).decode("utf-8"))
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
        f"({num_tensors} tensors, {total_gb:.2f} GB BF16 footprint) in {hf_to_local_s:.2f}s",
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
  gcs_cache_dir = f"{args.gcs_run_dir}/to_maxtext/{args.model}/{args.scan_mode}/hf_cache/{repo_id.replace('/', '_')}"
  print(f"[to_maxtext Step 1] Downloading {repo_id}/*.safetensors from HF Hub -> {gcs_cache_dir} ...", flush=True)

  t_gcs_start = time.perf_counter()
  fs = huggingface_hub.HfFileSystem(token=token)
  files = fs.glob(f"{repo_id}/*.safetensors")
  if not files:
    raise RuntimeError(f"No .safetensors files found on HF Hub for {repo_id}")

  with concurrent.futures.ProcessPoolExecutor(
      max_workers=min(32, max(1, len(files))), mp_context=multiprocessing.get_context("spawn")
  ) as executor:
    futures = [
        executor.submit(load_dynamic.build_gcs_cache_worker, fpath, gcs_cache_dir, token)
        for fpath in files
    ]
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
  huggingface_hub.snapshot_download(
      repo_id=repo_id,
      local_dir=local_hf_dir,
      token=token,
      allow_patterns=["*.json"],
  )
  from etils import epath

  def _copy_from_gcs_cache(fpath: str) -> None:
    fname = os.path.basename(fpath)
    src = epath.Path(f"{gcs_cache_dir}/{fname}")
    dst = os.path.join(local_hf_dir, fname)
    with src.open("rb") as fsrc, open(dst, "wb") as fdst:
      shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)

  with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(1, len(files)))) as pool:
    list(pool.map(_copy_from_gcs_cache, files))
  hf_to_local_s = time.perf_counter() - t_local_start
  shutil.rmtree(os.path.join(local_hf_dir, ".cache"), ignore_errors=True)
  shutil.rmtree(local_hf_cache, ignore_errors=True)
  print(f"[to_maxtext Step 1] Local staging completed in {hf_to_local_s:.2f}s", flush=True)

  return {
      "hf_to_gcs_s": hf_to_gcs_s,
      "hf_to_local_s": hf_to_local_s,
      "num_safetensors_files": len(files),
      "gcs_cache_dir": gcs_cache_dir,
      "local_hf_dir": local_hf_dir,
  }


def run_worker_to_maxtext_step2_convert(args: argparse.Namespace) -> dict[str, Any]:
  """Step 2 for `to_maxtext`: Convert HF checkpoint to an Orbax checkpoint on GCS."""
  import jax

  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  jax.config.update("jax_platforms", "cpu")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  from maxtext.checkpoint_conversion import to_maxtext
  from maxtext.checkpoint_conversion.utils import utils as conv_utils
  from maxtext.common import checkpointing

  sub_timings: dict[str, float] = {}

  orig_get_model_info = to_maxtext.get_maxtext_model_info
  orig_shard_jax_weights = conv_utils.shard_jax_weights
  orig_save_checkpoint = checkpointing.save_checkpoint
  orig_wait_until_finished = checkpointing.wait_until_finished

  def timed_get_model_info(*a, **kw):
    t0 = time.perf_counter()
    res = orig_get_model_info(*a, **kw)
    sub_timings["init_abstract_model_s"] = time.perf_counter() - t0
    return res

  def timed_shard_jax_weights(*a, **kw):
    t0 = time.perf_counter()
    res = orig_shard_jax_weights(*a, **kw)
    sub_timings["transform_and_shard_s"] = time.perf_counter() - t0
    return res

  save_start_holder = {}

  def timed_save_checkpoint(*a, **kw):
    save_start_holder["t0"] = time.perf_counter()
    return orig_save_checkpoint(*a, **kw)

  def timed_wait_until_finished(*a, **kw):
    res = orig_wait_until_finished(*a, **kw)
    if "t0" in save_start_holder:
      sub_timings["orbax_save_to_gcs_s"] = time.perf_counter() - save_start_holder["t0"]
    return res

  to_maxtext.get_maxtext_model_info = timed_get_model_info
  conv_utils.shard_jax_weights = timed_shard_jax_weights
  checkpointing.save_checkpoint = timed_save_checkpoint
  checkpointing.wait_until_finished = timed_wait_until_finished

  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1" and args.model in (
      "deepseek3-671b",
      "deepseek3.2-671b",
      "deepseek4-284b",
      "deepseek4-1600b",
  ):
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

  from etils import epath

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
      *(["attention=dot_product"] if args.model.startswith(("deepseek3.2", "deepseek4")) else []),
  ]
  sys.argv = list(model_args)

  print(f"[to_maxtext Step 2] Converting {args.model} ({args.scan_mode}) -> {orbax_output_dir} ...", flush=True)
  t0 = time.perf_counter()
  to_maxtext.main(
      args=model_args,
      lazy_load_tensors=lazy_load,
      eager_load_method=eager_method,
      hf_model_path=args.local_hf_dir,
      save_dtype="bfloat16",
      simulated_cpu_devices_count=16,
  )
  convert_to_orbax_s = time.perf_counter() - t0
  print(f"[to_maxtext Step 2] Convert to Orbax checkpoint completed in {convert_to_orbax_s:.2f}s", flush=True)

  return {
      "convert_to_orbax_s": convert_to_orbax_s,
      "orbax_items_path": f"{orbax_output_dir}/0/items",
      **sub_timings,
  }


def find_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    return int(s.getsockname()[1])


def init_worker_runtime(args: argparse.Namespace) -> str:
  """Initializes JAX platform and (when num_slices > 1) JAX multi-slice distributed runtime."""
  import jax

  hw = resolve_hardware(args.hardware)
  if hw == "cpu":
    jax.config.update("jax_platforms", "cpu")
  if getattr(args, "num_slices", 1) > 1:
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_slices,
        process_id=args.slice_id,
    )
    print(
        f"[Slice {args.slice_id}/{args.num_slices}] JAX distributed initialized: "
        f"process_index={jax.process_index()}, process_count={jax.process_count()}, "
        f"local_devices={jax.local_device_count()}, global_devices={jax.device_count()}",
        flush=True,
    )
  return hw


def run_worker_to_maxtext_step3_load_orbax(args: argparse.Namespace) -> dict[str, Any]:
  """Step 3 for `to_maxtext`: Load the converted Orbax checkpoint from GCS into MaxText."""
  import jax

  hw = init_worker_runtime(args)
  from flax.core.spmd import logical_axis_rules
  from maxtext.common import checkpointing
  from maxtext.configs import pyconfig
  from maxtext.utils import globals as maxtext_globals

  scan_bool = "True" if args.scan_mode == "scanned" else "False"
  orbax_items_path = f"{args.gcs_run_dir}/to_maxtext/{args.model}/{args.scan_mode}/orbax_ckpt/0/items"
  token = get_hf_token()
  num_slices = getattr(args, "num_slices", 1)
  slice_suffix = f"_slices{num_slices}" if num_slices > 1 else ""

  concurrent_gb = get_concurrent_gb(args.model, args.scan_mode, num_slices)
  dcn_axis = resolve_dcn_axis(args.model, getattr(args, "dcn_parallelism_axis", "auto"))
  multi_slice_flags = (
      [
          "skip_jax_distributed_system=False",
          f"num_slices={num_slices}",
          f"{dcn_axis}={num_slices}",
          "ici_fsdp_parallelism=-1",
          f"checkpoint_storage_concurrent_gb={concurrent_gb}",
      ]
      if num_slices > 1
      else [
          "skip_jax_distributed_system=True",
          f"checkpoint_storage_concurrent_gb={concurrent_gb}",
      ]
  )
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
  config = pyconfig.initialize(config_argv)
  mesh, unboxed_abstract_state = build_abstract_state_for_load(config)
  init_abstract_state_s = time.perf_counter() - t_init0
  print(
      f"[to_maxtext Step 3] Abstract state initialized in {init_abstract_state_s:.2f}s "
      f"(num_slices={config.num_slices}, global_devices={jax.device_count()}, mesh_shape={mesh.shape})",
      flush=True,
  )

  print(f"[to_maxtext Step 3] Loading Orbax checkpoint from {orbax_items_path} ...", flush=True)
  t_load0 = time.perf_counter()
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
    jax.block_until_ready(raw_params)
  if num_slices > 1:
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices("orbax_step3_load_complete")
  load_orbax_s = time.perf_counter() - t_load0

  if raw_params is None:
    raise RuntimeError(f"Failed to restore parameters from Orbax checkpoint at {orbax_items_path}")

  num_leaves = len(jax.tree_util.tree_leaves(raw_params))
  print(
      f"[to_maxtext Step 3] Loaded Orbax checkpoint ({num_leaves} parameter leaves, "
      f"num_slices={num_slices}, global_devices={jax.device_count()}) in {load_orbax_s:.2f}s",
      flush=True,
  )
  return {
      "load_orbax_s": load_orbax_s,
      "init_abstract_state_s": init_abstract_state_s,
      "num_param_leaves": num_leaves,
      "num_slices": num_slices,
      "global_devices": jax.device_count(),
  }


def run_worker_safetensors_dynamic(args: argparse.Namespace) -> dict[str, Any]:
  """Runs `safetensors_dynamic` and records (1) HF -> GCS and (2) safetensors and convert to MaxText."""
  import jax

  hw = init_worker_runtime(args)
  from flax.core.spmd import logical_axis_rules
  import jax.numpy as jnp
  import numpy as np
  from maxtext.checkpoint_conversion.utils import load_dynamic
  from maxtext.checkpoint_conversion.utils import tensor_handling
  from maxtext.common import checkpointing
  from maxtext.configs import pyconfig
  from maxtext.utils import globals as maxtext_globals

  orig_apply_hook_fns = tensor_handling.apply_hook_fns

  def multihost_safe_apply_hook_fns(weight, target_shape, hook_fns):
    if hook_fns is None:
      return weight
    if not isinstance(hook_fns, list):
      hook_fns = [hook_fns]
    for hook_fn in hook_fns:
      try:
        weight = hook_fn(weight, target_shape)
      except RuntimeError as e:
        if isinstance(weight, jax.Array) and "non-addressable" in str(e):
          probe = hook_fn(np.ones((1, 1), dtype=np.float32), (1, 1))
          scale = np.dtype("float32").type(probe[0, 0])
          slices = tuple(slice(0, min(s, t)) for s, t in zip(weight.shape, target_shape))
          pad_width = tuple((0, max(0, t - s)) for s, t in zip(weight.shape, target_shape))
          orig_dtype = weight.dtype
          weight = jnp.pad(weight[slices], pad_width)
          if float(scale) != 1.0:
            weight = weight * scale
          weight = weight.astype(orig_dtype)
        else:
          raise
    if isinstance(weight, jax.Array) and jax.process_count() > 1:
      weight.block_until_ready()
    return weight

  tensor_handling.apply_hook_fns = multihost_safe_apply_hook_fns

  import ctypes
  import gc

  def _trim_malloc():
    gc.collect()
    try:
      ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
      pass

  def low_mem_binary_chunked_stack(tensors, axis: int, _top: bool = True):
    if _top:
      _trim_malloc()
    if not tensors:
      raise ValueError("Cannot stack empty list of tensors.")
    if len(tensors) == 1:
      t = tensors.pop()
      res = jnp.expand_dims(t, axis=axis)
      res.block_until_ready()
      return res
    if len(tensors) == 2:
      t0, t1 = tensors[0], tensors[1]
      tensors.clear()
      res = jnp.stack([t0, t1], axis=axis)
      res.block_until_ready()
      if isinstance(t0, jax.Array) and not t0.is_deleted():
        t0.delete()
      if isinstance(t1, jax.Array) and not t1.is_deleted():
        t1.delete()
      del t0, t1
      return res
    mid = len(tensors) // 2
    left_list = tensors[:mid]
    right_list = tensors[mid:]
    tensors.clear()
    left = low_mem_binary_chunked_stack(left_list, axis=axis, _top=False)
    right = low_mem_binary_chunked_stack(right_list, axis=axis, _top=False)
    res = jnp.concatenate([left, right], axis=axis)
    res.block_until_ready()
    if isinstance(left, jax.Array) and not left.is_deleted():
      left.delete()
    if isinstance(right, jax.Array) and not right.is_deleted():
      right.delete()
    del left, right
    if _top:
      _trim_malloc()
    return res

  tensor_handling._binary_chunked_stack = low_mem_binary_chunked_stack

  from etils import epath

  scan_bool = "True" if args.scan_mode == "scanned" else "False"
  repo_id = maxtext_globals.HF_IDS[args.model]
  token = get_hf_token()
  num_slices = getattr(args, "num_slices", 1)
  slice_suffix = f"_slices{num_slices}" if num_slices > 1 else ""
  if getattr(args, "reuse_model_hf_cache", False):
    base_out_dir = f"{args.gcs_run_dir}/safetensors_dynamic/{args.model}/shared"
    should_clean_cache = args.scan_mode == "scanned" and num_slices == 1
  else:
    base_out_dir = f"{args.gcs_run_dir}/safetensors_dynamic/{args.model}/{args.scan_mode}{slice_suffix}"
    should_clean_cache = True
  if jax.process_index() == 0 and should_clean_cache:
    epath.Path(f"{base_out_dir}/hf_cache").rmtree(missing_ok=True)
  if num_slices > 1:
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices("safetensors_clean_cache_complete")

  concurrent_gb = get_concurrent_gb(args.model, args.scan_mode, num_slices)
  dcn_axis = resolve_dcn_axis(args.model, getattr(args, "dcn_parallelism_axis", "auto"))
  multi_slice_flags = (
      [
          "skip_jax_distributed_system=False",
          f"num_slices={num_slices}",
          f"{dcn_axis}={num_slices}",
          "ici_fsdp_parallelism=-1",
          f"checkpoint_storage_concurrent_gb={concurrent_gb}",
      ]
      if num_slices > 1
      else [
          "skip_jax_distributed_system=True",
          f"checkpoint_storage_concurrent_gb={concurrent_gb}",
      ]
  )
  weight_dtype_flag = (
      [f"weight_dtype={args.weight_dtype}"]
      if getattr(args, "weight_dtype", "auto") not in ("", "auto", "default")
      else ([] if args.model in ("qwen3-0.6b", "gemma2-2b") else ["weight_dtype=bfloat16"])
  )

  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1" and args.model in (
      "deepseek3-671b",
      "deepseek3.2-671b",
      "deepseek4-284b",
      "deepseek4-1600b",
  ):
    local_sparse_dir = args.local_hf_dir if os.path.exists(os.path.dirname(args.local_hf_dir)) else f"/tmp/hf_local_{args.model}"
    _, num_t, tot_gb = _stage_sparse_bf16_safetensors(args.model, local_sparse_dir)
    print(
        f"[safetensor] Ensured sparse BF16 .safetensors for {args.model} at {local_sparse_dir} "
        f"({num_t} tensors, {tot_gb:.2f} GB BF16)",
        flush=True,
    )
    load_params_uri = local_sparse_dir
  else:
    load_params_uri = (
        args.local_hf_dir
        if os.environ.get("USE_LOCAL_HF_FOR_SAFETENSORS") == "1" and num_slices == 1 and os.path.isdir(args.local_hf_dir)
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
  config = pyconfig.initialize(config_argv)
  mesh, unboxed_abstract_state = build_abstract_state_for_load(config)
  init_abstract_state_s = time.perf_counter() - t_init0
  print(
      f"[safetensor] Abstract state initialized in {init_abstract_state_s:.2f}s "
      f"(num_slices={config.num_slices}, global_devices={jax.device_count()}, mesh_shape={mesh.shape})",
      flush=True,
  )

  timings: dict[str, float] = {}

  orig_get_mappings = load_dynamic.get_hf_config_and_mappings
  orig_load_sharded = load_dynamic.load_sharded_hf_state
  orig_transform = load_dynamic.transform_hf_state_to_mt_state

  def timed_get_mappings(*a, **kw):
    # Called immediately after Step 1 (HF -> GCS caching + barrier) finishes in load_safetensors_dynamic_state
    timings["t_step1_end"] = time.perf_counter()
    print(f"[safetensor] Step 1 (HF->GCS) completed in {timings['t_step1_end'] - t_start:.2f}s", flush=True)
    t0 = time.perf_counter()
    res = orig_get_mappings(*a, **kw)
    timings["derive_mappings_s"] = time.perf_counter() - t0
    return res

  def timed_load_sharded(path, *a, **kw):
    t0 = time.perf_counter()
    use_custom_byte_reader = not hasattr(load_dynamic, "_leading_axis_shardings") and (
        num_slices > 1
        or args.model
        in (
            "mixtral-8x7b",
            "llama3.1-70b",
            "gpt-oss-120b",
            "deepseek3-671b",
            "deepseek3.2-671b",
            "deepseek4-284b",
            "deepseek4-1600b",
        )
    )
    if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM", "0") == "1" and args.model in (
        "deepseek3-671b",
        "deepseek3.2-671b",
        "deepseek4-284b",
        "deepseek4-1600b",
    ):
      use_custom_byte_reader = True
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
              and k.endswith((
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
              ))
              and shp[1] % fsdp_size == 0
          ):
            chosen_ax = 1
          elif len(shp) == 2 and k.endswith(("o_proj.weight", "down_proj.weight", "w2.weight")) and shp[0] % fsdp_size == 0:
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
      res = orig_load_sharded(path, *a, **kw)
    _trim_malloc()
    timings["load_sharded_hf_s"] = time.perf_counter() - t0
    return res

  def timed_transform(*a, **kw):
    t0 = time.perf_counter()
    _orig_dp = jax.device_put
    _orig_get_load_fn = load_dynamic.get_hf_loading_function
    _step_cnt = [0]

    def _low_mem_get_load_fn(hf_source, tensor_getter, hook_fn, target_leaf, maxtext_config, mt_key):
      popped = []
      def _track_getter(key):
        arr = tensor_getter(key)
        if isinstance(arr, jax.Array):
          popped.append(arr)
        return arr
      inner_fn = _orig_get_load_fn(hf_source, _track_getter, hook_fn, target_leaf, maxtext_config, mt_key)
      def _wrapped_fn():
        out = inner_fn()
        out_id = id(out)
        for p_arr in popped:
          if id(p_arr) != out_id and isinstance(p_arr, jax.Array) and not p_arr.is_deleted():
            try:
              p_arr.delete()
            except Exception:
              pass
        popped.clear()
        _step_cnt[0] += 1
        if _step_cnt[0] % 8 == 0:
          _trim_malloc()
        return out
      return _wrapped_fn

    load_dynamic.get_hf_loading_function = _low_mem_get_load_fn
    if jax.process_count() > 1:
      def _synced_dp(*da, **dkw):
        r = _orig_dp(*da, **dkw)
        try:
          jax.block_until_ready(r)
        except Exception:
          pass
        return r
      jax.device_put = _synced_dp
    try:
      res = orig_transform(*a, **kw)
    finally:
      jax.device_put = _orig_dp
      load_dynamic.get_hf_loading_function = _orig_get_load_fn
    _trim_malloc()
    timings["transform_hf_to_mt_s"] = time.perf_counter() - t0
    return res

  load_dynamic.get_hf_config_and_mappings = timed_get_mappings
  load_dynamic.load_sharded_hf_state = timed_load_sharded
  load_dynamic.transform_hf_state_to_mt_state = timed_transform

  print(f"[safetensor] Running load_state_if_possible(source_checkpoint_layout='safetensors_dynamic') ...", flush=True)
  t_start = time.perf_counter()
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
        source_checkpoint_layout="safetensors_dynamic",
        expansion_factor_real_data=config.expansion_factor_real_data,
        maxtext_config=config,
    )
    jax.block_until_ready(raw_params)
  if num_slices > 1:
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices("safetensors_dynamic_load_complete")
  t_end = time.perf_counter()

  if raw_params is None:
    raise RuntimeError(f"Failed to restore parameters via safetensors_dynamic for {args.model}")

  hf_to_gcs_s = timings["t_step1_end"] - t_start
  safetensors_and_convert_s = t_end - timings["t_step1_end"]
  total_s = t_end - t_start
  num_leaves = len(jax.tree_util.tree_leaves(raw_params))

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
      "num_param_leaves": num_leaves,
      "num_slices": num_slices,
      "global_devices": jax.device_count(),
  }


SITE_CUSTOMIZE_CONTENT = '''\
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
'''


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


def _start_oom_watchdog(proc: subprocess.Popen, min_avail_gb: float = 8.5) -> dict[str, Any]:
  """Monitors host RAM while `proc` runs and kills ONLY `proc` if MemAvailable drops below `min_avail_gb`.

  This prevents the Kubernetes container cgroup from hitting its hard memory limit and killing PID 1.
  """
  import signal
  import threading

  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1":
    min_avail_gb = max(min_avail_gb, 24.0)

  state: dict[str, Any] = {
      "peak_ram_gb": 0.0,
      "total_ram_gb": 0.0,
      "oom_triggered": False,
      "oom_message": "",
      "stop": False,
  }

  def _watch():
    while not state["stop"] and proc.poll() is None:
      used_gb, avail_gb, total_gb = _read_host_mem_gb()
      state["total_ram_gb"] = total_gb
      if used_gb > state["peak_ram_gb"]:
        state["peak_ram_gb"] = used_gb
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


def _start_self_oom_watchdog(min_avail_gb: float = 9.0) -> dict[str, Any]:
  """In-process watchdog inside a worker subprocess so both local and remote slice workers self-terminate before container cgroup OOM."""
  import signal
  import threading

  try:
    with open("/proc/self/oom_score_adj", "w", encoding="utf-8") as oom_f:
      oom_f.write("1000\n")
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1":
    min_avail_gb = max(min_avail_gb, 25.0)

  state: dict[str, Any] = {"peak_ram_gb": 0.0, "total_ram_gb": 0.0, "stop": False}

  def _self_watch():
    pid = os.getpid()
    while not state["stop"]:
      used_gb, avail_gb, total_gb = _read_host_mem_gb()
      state["total_ram_gb"] = total_gb
      if used_gb > state["peak_ram_gb"]:
        state["peak_ram_gb"] = used_gb
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
        cmd = req["cmd"]
        c_env = req["env"]
        r_work_dir = req.get("local_work_dir", "/tmp/benchmark_remote")
        os.makedirs(r_work_dir, exist_ok=True)
        site_dir = os.path.join(r_work_dir, "_sitecustomize")
        os.makedirs(site_dir, exist_ok=True)
        with open(os.path.join(site_dir, "sitecustomize.py"), "w", encoding="utf-8") as sf:
          sf.write(SITE_CUSTOMIZE_CONTENT)
        subprocess.run(["pkill", "-9", "-f", "--", "--worker-phase"], check=False)
        print(f"[RemoteDaemon] Starting command from {addr}: {' '.join(cmd)}", flush=True)
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=c_env, text=True
        ) as proc:
          r_wd = _start_oom_watchdog(proc)
          assert proc.stdout is not None
          for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            conn.sendall(line.encode("utf-8", errors="replace"))
          proc.wait(timeout=14400)
          r_wd["stop"] = True
          if r_wd.get("oom_triggered") and r_wd.get("oom_message"):
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


def invoke_worker_subprocess(
    phase: str,
    model: str,
    scan_mode: str,
    gcs_run_dir: str,
    local_work_dir: str,
    hardware: str = "auto",
    num_slices: int = 1,
    weight_dtype: str = "auto",
    dcn_parallelism_axis: str = "auto",
    devices_per_slice: int = 4,
    remote_slice_addrs: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    skip_completed: bool = False,
    reuse_model_hf_cache: bool = False,
    allow_step_failure: bool = False,
) -> dict[str, Any]:
  """Runs a single benchmark phase in clean subprocess(es) and returns its JSON metrics."""
  repo_root = get_repo_root()
  os.makedirs(local_work_dir, exist_ok=True)
  is_multislice_phase = num_slices > 1 and phase in ("to_maxtext_step3", "safetensors_dynamic")
  scan_dir_name = f"{scan_mode}_slices{num_slices}" if is_multislice_phase else scan_mode
  log_dir = os.path.join(local_work_dir, "logs", model, scan_dir_name)
  os.makedirs(log_dir, exist_ok=True)

  result_json = os.path.join(log_dir, f"{phase}_result.json")
  log_file = os.path.join(log_dir, f"{phase}.log")
  local_hf_dir = os.path.join(local_work_dir, "hf_local", model)

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
    print(
        f"[SKIP COMPLETED]: {phase} | Model: {model} | Scan Mode: {scan_mode} "
        f"| Slices: {num_slices} -> {result_json}",
        flush=True,
    )
    with open(result_json, "r", encoding="utf-8") as rf:
      data = json.load(rf)
    if os.path.exists(log_file):
      with open(log_file, "r", encoding="utf-8") as lf:
        worker_cached_times = [
            float(m.group(1))
            for m in re.finditer(r"\[Worker\] Cached .* in ([0-9.]+)s", lf.read())
        ]
      if worker_cached_times:
        data["worker_upload_max_s"] = max(worker_cached_times)
    return data

  if (
      reuse_model_hf_cache
      and phase == "to_maxtext_step1"
      and scan_mode != "scanned"
      and not os.path.exists(result_json)
  ):
    scanned_json = os.path.join(local_work_dir, "logs", model, "scanned", "to_maxtext_step1_result.json")
    scanned_log = os.path.join(local_work_dir, "logs", model, "scanned", "to_maxtext_step1.log")
    if os.path.exists(scanned_json):
      shutil.copyfile(scanned_json, result_json)
      if os.path.exists(scanned_log):
        shutil.copyfile(scanned_log, log_file)
      print(
          f"[REUSE MODEL HF STAGE]: {phase} | Model: {model} | Scan Mode: {scan_mode} -> {result_json}",
          flush=True,
      )
      with open(result_json, "r", encoding="utf-8") as rf:
        data = json.load(rf)
      if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as lf:
          worker_cached_times = [
              float(m.group(1))
              for m in re.finditer(r"\[Worker\] Cached .* in ([0-9.]+)s", lf.read())
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
      "--weight-dtype",
      weight_dtype,
      "--dcn-parallelism-axis",
      dcn_parallelism_axis,
      "--num-slices",
      str(num_slices if is_multislice_phase else 1),
      "--gcs-run-dir",
      gcs_run_dir,
      "--local-hf-dir",
      local_hf_dir,
      "--result-json",
      result_json,
  ]
  if reuse_model_hf_cache:
    base_cmd.append("--reuse-model-hf-cache")

  env = os.environ.copy()
  env.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
  site_dir = os.path.join(local_work_dir, "_sitecustomize")
  os.makedirs(site_dir, exist_ok=True)
  sitecustomize_path = os.path.join(site_dir, "sitecustomize.py")
  with open(sitecustomize_path, "w", encoding="utf-8") as sf:
    sf.write(SITE_CUSTOMIZE_CONTENT)
  src_dir = os.path.join(repo_root, "src")
  existing_pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = (
      f"{site_dir}:{src_dir}:{existing_pythonpath}" if existing_pythonpath else f"{site_dir}:{src_dir}"
  )
  if extra_env:
    env.update(extra_env)

  print(f"\n{'-' * 80}")
  print(f"[PHASE]: {phase} | Model: {model} | Scan Mode: {scan_mode} | Slices: {num_slices}")
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

  worker_cached_times: list[float] = []
  watchdog_state: dict[str, Any] = {}
  try:
    if not is_multislice_phase:
      if resolve_hardware(hardware) == "cpu" and "XLA_FLAGS" not in env:
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices_per_slice}"
      with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"Command: {' '.join(base_cmd)}\n\n")
        lf.flush()
        with subprocess.Popen(
            base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=repo_root, text=True
        ) as proc:
          watchdog_state = _start_oom_watchdog(proc)
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
          if watchdog_state.get("oom_triggered") and watchdog_state.get("oom_message"):
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
        oom_tag = " (OOMKilled / Exit Code 137)" if (proc.returncode in (-9, 137) or watchdog_state.get("oom_triggered")) else ""
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
      procs: list[subprocess.Popen] = []
      remote_threads: list[Any] = []
      remote_rc: dict[int, int] = {}
      slice_log_files = []
      try:
        for slice_id in range(num_slices):
          cmd_i = base_cmd + ["--slice-id", str(slice_id), "--coordinator-address", coord_addr]
          if slice_id == 0:
            p = subprocess.Popen(
                cmd_i, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=repo_root, text=True
            )
            procs.append(p)
          elif remote_endpoints and (slice_id - 1) < len(remote_endpoints):
            import threading

            target_remote_addr = remote_endpoints[slice_id - 1]
            slf = open(os.path.join(log_dir, f"{phase}_slice{slice_id}.log"), "w", encoding="utf-8")
            slf.write(f"Remote Slice {slice_id} ({target_remote_addr}) Command: {' '.join(cmd_i)}\n\n")
            slf.flush()
            slice_log_files.append(slf)

            def _run_remote(s_id, target_addr, c_list, c_env, log_f):
              host, port_s = target_addr.split(":", 1)
              rc = 1
              try:
                import base64

                script_abs = os.path.abspath(__file__)
                with open(script_abs, "rb") as f_self:
                  b64_self = base64.b64encode(f_self.read()).decode("ascii")
                sync_cmd = [
                    "python3",
                    "-c",
                    (
                        f"import base64, os; p = {script_abs!r}; "
                        f"os.makedirs(os.path.dirname(p), exist_ok=True); "
                        f"open(p, 'wb').write(base64.b64decode('{b64_self}'))"
                    ),
                ]
                with socket.create_connection((host, int(port_s)), timeout=60) as sync_sock:
                  sync_payload = json.dumps({"cmd": sync_cmd, "env": c_env, "local_work_dir": local_work_dir}) + "\n"
                  sync_sock.sendall(sync_payload.encode("utf-8"))
                  with sync_sock.makefile("r", encoding="utf-8", errors="replace") as sf_sync:
                    for _ in sf_sync:
                      pass
              except Exception:  # pylint: disable=broad-exception-caught
                pass
              try:
                with socket.create_connection((host, int(port_s)), timeout=14400) as sock:
                  payload = json.dumps({"cmd": c_list, "env": c_env, "local_work_dir": local_work_dir}) + "\n"
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
          lf.write(f"Multi-slice Command (num_slices={num_slices}, coord={coord_addr}): {' '.join(base_cmd)}\n\n")
          lf.flush()
          watchdog_state = _start_oom_watchdog(procs[0])
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
          if watchdog_state.get("oom_triggered") and watchdog_state.get("oom_message"):
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
          for idx, p in enumerate(procs):
            if p.returncode != 0:
              oom_tag = " (OOMKilled / Exit Code 137)" if (p.returncode in (-9, 137) or watchdog_state.get("oom_triggered")) else ""
              raise RuntimeError(
                  f"Multi-slice worker phase '{phase}' local slice {idx} failed with exit code {p.returncode}{oom_tag} "
                  f"(Peak RAM: {watchdog_state.get('peak_ram_gb', 0.0):.2f}/{watchdog_state.get('total_ram_gb', 251.69):.2f} GB). "
                  f"See logs in: {log_dir}"
              )
          for s_id, t in remote_threads:
            t.join(timeout=14400)
            rc = remote_rc.get(s_id, 1)
            if rc != 0:
              raise RuntimeError(
                  f"Multi-slice worker phase '{phase}' remote slice {s_id} ({remote_slice1}) failed with exit code {rc}. "
                  f"See logs in: {log_dir}"
              )
      finally:
        for slf in slice_log_files:
          slf.close()
  except RuntimeError as e:
    if not allow_step_failure:
      raise
    print(f"[STEP FAILED (continuing due to --allow-step-failure)]: {e}", flush=True)
    parsed_step1_s = 0.0
    if os.path.exists(log_file):
      with open(log_file, "r", encoding="utf-8", errors="replace") as lf_err:
        m_s1 = re.search(r"\[safetensor\] Step 1 \(HF->GCS\) completed in ([0-9.]+)s", lf_err.read())
        if m_s1:
          parsed_step1_s = float(m_s1.group(1))
    fail_data: dict[str, Any] = {
        "status": "FAILED",
        "error": str(e),
        "hf_to_gcs_s": parsed_step1_s,
        "hf_to_local_s": 0.0,
        "convert_to_orbax_s": 0.0,
        "load_orbax_s": 0.0,
        "safetensors_and_convert_s": 0.0,
        "worker_upload_max_s": max(worker_cached_times) if worker_cached_times else 0.0,
        "num_slices": num_slices,
        "global_devices": num_slices * 4,
    }
    with open(result_json, "w", encoding="utf-8") as rf:
      json.dump(fail_data, rf, indent=2)
    _mirror_logs_to_gcs()
    return fail_data
  finally:
    if os.environ.get("STAGE_SPARSE_DEEPSEEK_OOM") == "1" and os.path.isdir(local_hf_dir) and hasattr(os, "posix_fadvise"):
      for fn in os.listdir(local_hf_dir):
        if fn.endswith(".safetensors"):
          try:
            with open(os.path.join(local_hf_dir, fn), "rb") as sf_evict:
              os.posix_fadvise(sf_evict.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
    _mirror_logs_to_gcs()

  with open(result_json, "r", encoding="utf-8") as rf:
    data = json.load(rf)
  if worker_cached_times:
    data["worker_upload_max_s"] = max(worker_cached_times)
  _mirror_logs_to_gcs()
  return data


def print_summary_table(results: list[dict[str, Any]], csv_path: str) -> None:
  """Prints a formatted markdown comparison table and writes results to CSV."""
  fieldnames = [
      "model",
      "scan_mode",
      "num_slices",
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
      help="Number of devices per slice when simulating CPU devices (default: 4, matching v5p-8).",
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
    os.chdir(repo_root)
    exec(SITE_CUSTOMIZE_CONTENT, globals())  # pylint: disable=exec-used
    self_wd = _start_self_oom_watchdog()
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
    self_wd["stop"] = True
    if self_wd.get("peak_ram_gb", 0.0) > 0:
      print(
          f"INFO:watchdog:Worker Peak Host RAM: {self_wd['peak_ram_gb']:.2f}/{self_wd.get('total_ram_gb', 251.69):.2f} GB",
          flush=True,
      )
      res["peak_host_ram_gb"] = round(self_wd["peak_ram_gb"], 2)
      res["total_host_ram_gb"] = round(self_wd.get("total_ram_gb", 251.69), 2)

    if args.slice_id == 0:
      with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return

  run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  gcs_run_dir = f"{args.gcs_base.rstrip('/')}/run_{run_id}"
  local_work_dir = args.local_work_dir or f"/tmp/benchmark_ckpt_load_time_{run_id}"
  csv_report_path = os.path.abspath(args.csv_report)

  print(f"Starting checkpoint load time benchmark (run_id={run_id})")
  print(f"  Repo root    : {repo_root}")
  print(f"  Models       : {args.models}")
  print(f"  Scan modes   : {args.scan_modes}")
  print(f"  Num slices   : {args.num_slices_list}")
  print(f"  DCN axis     : {args.dcn_parallelism_axis}")
  print(f"  Hardware     : {args.hardware} (resolved: {resolve_hardware(args.hardware)})")
  print(f"  Weight dtype : {args.weight_dtype}")
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

  for model in args.models:
    cold_st_step1_s: float | None = None
    cold_st_worker_s: float = 0.0
    m_step1_by_scan: dict[str, dict[str, Any]] = {}
    m_step2_by_scan: dict[str, dict[str, Any]] = {}

    # Pass 1: Run to_maxtext Step 1 (HF -> GCS + local stage) and Step 2 (Convert to Orbax)
    # across all scan_modes first, then remove the local HF staging directory to free disk/RAM
    # before running Step 3 and safetensors_dynamic across single/multi-slice configurations.
    for scan_mode in args.scan_modes:
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
      )
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
      )

    # Clean up staged local HF weights once Step 2 conversions for this model are done
    local_hf_model_dir = os.path.join(local_work_dir, "hf_local", model)
    if os.environ.get("USE_LOCAL_HF_FOR_SAFETENSORS") != "1":
      shutil.rmtree(local_hf_model_dir, ignore_errors=True)
    shutil.rmtree(f"{local_hf_model_dir}_cache", ignore_errors=True)

    # Pass 2: Run to_maxtext Step 3 (Load Orbax) and safetensors_dynamic across num_slices and scan_modes
    # Run all single-slice configurations first before multi-slice configurations.
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for num_slices in args.num_slices_list:
      for scan_mode in args.scan_modes:
        m_step1 = m_step1_by_scan[scan_mode]
        m_step2 = m_step2_by_scan[scan_mode]
        if m_step2.get("status") == "FAILED" and args.allow_step_failure:
          print(
              f"[SKIP to_maxtext_step3]: Step 2 failed for {model} ({scan_mode}); skipping Orbax load.",
              flush=True,
          )
          m_step3 = {
              "status": "SKIPPED_DUE_TO_STEP2_FAILURE",
              "load_orbax_s": 0.0,
              "num_slices": num_slices,
              "global_devices": num_slices * args.devices_per_slice,
          }
        else:
          m_step3 = invoke_worker_subprocess(
              phase="to_maxtext_step3",
              model=model,
              scan_mode=scan_mode,
              gcs_run_dir=gcs_run_dir,
              local_work_dir=local_work_dir,
              hardware=args.hardware,
              num_slices=num_slices,
              weight_dtype=args.weight_dtype,
              dcn_parallelism_axis=args.dcn_parallelism_axis,
              devices_per_slice=args.devices_per_slice,
              remote_slice_addrs=args.remote_slice_addrs,
              skip_completed=args.skip_completed,
              reuse_model_hf_cache=args.reuse_model_hf_cache,
              allow_step_failure=args.allow_step_failure,
          )

        s_res = invoke_worker_subprocess(
            phase="safetensors_dynamic",
            model=model,
            scan_mode=scan_mode,
            gcs_run_dir=gcs_run_dir,
            local_work_dir=local_work_dir,
            hardware=args.hardware,
            num_slices=num_slices,
            weight_dtype=args.weight_dtype,
            dcn_parallelism_axis=args.dcn_parallelism_axis,
            devices_per_slice=args.devices_per_slice,
            remote_slice_addrs=args.remote_slice_addrs,
            skip_completed=args.skip_completed,
            reuse_model_hf_cache=args.reuse_model_hf_cache,
            allow_step_failure=args.allow_step_failure,
        )

        if s_res.get("worker_upload_max_s", 0.0) > 0.0 or cold_st_step1_s is None:
          cold_st_step1_s = s_res["hf_to_gcs_s"]
          cold_st_worker_s = s_res.get("worker_upload_max_s", 0.0)
          st_step1_s = s_res["hf_to_gcs_s"]
          st_worker_s = cold_st_worker_s
          st_warm_cache_s = 0.0
        else:
          st_step1_s = cold_st_step1_s
          st_worker_s = cold_st_worker_s
          st_warm_cache_s = s_res["hf_to_gcs_s"]

        to_maxtext_total_s = (
            m_step1["hf_to_gcs_s"] + m_step2["convert_to_orbax_s"] + m_step3["load_orbax_s"]
        )
        safetensor_total_s = st_step1_s + s_res["safetensors_and_convert_s"]

        row = {
            "model": model,
            "scan_mode": scan_mode,
            "num_slices": num_slices,
            "global_devices": s_res.get("global_devices", m_step3.get("global_devices", 1)),
            "to_maxtext_1_hf_to_gcs_s": m_step1["hf_to_gcs_s"],
            "to_maxtext_1_worker_upload_s": m_step1.get("worker_upload_max_s", 0.0),
            "to_maxtext_1b_hf_to_local_s": m_step1["hf_to_local_s"],
            "to_maxtext_2_convert_to_orbax_s": m_step2["convert_to_orbax_s"],
            "to_maxtext_2a_init_abstract_model_s": m_step2.get("init_abstract_model_s", 0.0),
            "to_maxtext_2b_transform_and_shard_s": m_step2.get("transform_and_shard_s", 0.0),
            "to_maxtext_2c_orbax_save_to_gcs_s": m_step2.get("orbax_save_to_gcs_s", 0.0),
            "to_maxtext_3_load_orbax_s": m_step3["load_orbax_s"],
            "to_maxtext_total_s": to_maxtext_total_s,
            "safetensor_1_hf_to_gcs_s": st_step1_s,
            "safetensor_1_worker_upload_s": st_worker_s,
            "safetensor_1_warm_cache_s": st_warm_cache_s,
            "safetensor_2_safetensors_and_convert_s": s_res["safetensors_and_convert_s"],
            "safetensor_2a_derive_mappings_s": s_res.get("derive_mappings_s", 0.0),
            "safetensor_2b_load_sharded_hf_s": s_res.get("load_sharded_hf_s", 0.0),
            "safetensor_2c_transform_hf_to_mt_s": s_res.get("transform_hf_to_mt_s", 0.0),
            "safetensor_total_s": safetensor_total_s,
            "speedup_total_vs_to_maxtext_x": (
                to_maxtext_total_s / safetensor_total_s if safetensor_total_s > 0 else 0.0
            ),
            "load_time_diff_safetensor_step2_vs_orbax_step3_s": (
                s_res["safetensors_and_convert_s"] - m_step3["load_orbax_s"]
            ),
        }
        rows_by_key[(scan_mode, num_slices)] = row

    for scan_mode in args.scan_modes:
      for num_slices in args.num_slices_list:
        all_results.append(rows_by_key[(scan_mode, num_slices)])

  print_summary_table(all_results, csv_report_path)


if __name__ == "__main__":
  main()
