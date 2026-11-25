# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-slice TPU hardware verifier for full 284B DeepSeek-V4 & 671B DeepSeek-V3 shard-digest parity."""

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import time

os.environ.setdefault("EPATH_USE_TF", "false")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _configure_megascale_env(num_slices: int, slice_index: int, coord_host: str) -> str:
  """Resolves `coord_host` and configures `LIBTPU_INIT_ARGS` + `MEGASCALE_*` before TPU backend startup."""
  coord_ip = None
  for _ in range(90):
    try:
      coord_ip = socket.gethostbyname(coord_host)
      break
    except OSError:
      time.sleep(2)
  assert coord_ip is not None, f"Could not resolve {coord_host}"

  cur = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
  for flag in ("slicebuilder_use_insecure_grpc", "megascale_use_insecure_grpc"):
    if f"--{flag}" not in cur:
      cur = f"{cur} --{flag}=true".strip()
  os.environ["LIBTPU_INIT_ARGS"] = cur
  os.environ["MEGASCALE_NUM_SLICES"] = str(num_slices)
  os.environ["MEGASCALE_SLICE_ID"] = str(slice_index)
  os.environ["MEGASCALE_PORT"] = "8477"
  os.environ["MEGASCALE_COORDINATOR_ADDRESS"] = f"{coord_ip}:8477"
  return coord_ip


if "COORD_HOST" in os.environ:
  _BOOT_COORD_IP = _configure_megascale_env(
      num_slices=int(os.environ.get("NUM_SLICES", "4")),
      slice_index=int(os.environ.get("SLICE_INDEX", "0")),
      coord_host=os.environ["COORD_HOST"],
  )
else:
  _BOOT_COORD_IP = ""

# pylint: disable=wrong-import-position
from etils import epath
import jax

try:
  from jax.experimental import hijax as _hjx

  if not hasattr(_hjx, "HiPrimitive") and hasattr(_hjx, "Primitive"):
    _hjx.HiPrimitive = _hjx.Primitive
except Exception:  # pylint: disable=broad-exception-caught
  pass

from flax import nnx
import flax.traverse_util
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
import numpy as np

_SYNC_COUNTER = 0


def _sync(tag: str) -> None:
  global _SYNC_COUNTER  # pylint: disable=global-statement
  _SYNC_COUNTER += 1
  jax.experimental.multihost_utils.sync_global_devices(f"repro_ms_{_SYNC_COUNTER}_{tag}")


def _log(msg: str) -> None:
  if jax.process_index() == 0:
    print(f"[TPU-Verify-MS][{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _peak_hbm_gb() -> float:
  try:
    stats = jax.local_devices()[0].memory_stats()
    return stats.get("peak_bytes_in_use", 0) / 1e9 if stats else 0.0
  except Exception:  # pylint: disable=broad-exception-caught
    return 0.0


def _git_commit_from_tree(repo_root: str) -> str:
  version_file = os.path.join(repo_root, "CODE_VERSION.json")
  if os.path.exists(version_file):
    try:
      with open(version_file, "r", encoding="utf-8") as f:
        return str(json.load(f).get("git_commit", "unknown"))
    except (OSError, ValueError):
      pass
  return os.environ.get("BENCH_GIT_COMMIT", "unknown")


def _digest_and_free(model) -> dict[str, list[list]]:
  """Computes exact raw-byte blake2b digests for every local addressable shard, then frees HBM."""
  _, state = nnx.split(model)
  flat = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep=".")
  digests: dict[str, list[list]] = {}
  for k in sorted(flat):
    v = flat[k]
    if not isinstance(v, jax.Array) or jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key):
      continue
    shard_sigs = []
    for s in v.addressable_shards:
      arr = np.asarray(s.data)
      raw_bytes = np.ascontiguousarray(arr).tobytes()
      digest = hashlib.blake2b(raw_bytes, digest_size=16).hexdigest()
      shard_sigs.append([repr(s.index), list(arr.shape), str(arr.dtype), digest])
    digests[k] = shard_sigs
  del flat, state, model
  gc.collect()
  return digests


def _step_cache_paths(step_cache_dir: str, label: str) -> list[epath.Path]:
  return [epath.Path(f"{step_cache_dir}/{label}_rank{r}.json") for r in range(jax.process_count())]


def _load_and_digest(
    base_yml: str,
    model_name: str,
    ckpt_path: str,
    streaming: bool,
    label: str,
    step_cache_dir: str,
    num_slices: int,
    resume: bool,
):
  """Loads `model_name`, computes per-shard blake2b digests, and persists step metadata to GCS."""
  paths = _step_cache_paths(step_cache_dir, label)
  my_path = paths[jax.process_index()]
  if resume and all(p.exists() for p in paths):
    _sync(f"cache_hit_{label}")
    cached = json.loads(my_path.read_text())
    _log(f"[RESUME] Loaded cached step '{label}' from GCS (dt={cached['dt']:.2f}s, peak={cached['peak_gb']:.1f} GB)")
    return cached["digests"], float(cached["dt"]), float(cached["peak_gb"])

  _sync(f"start_{label}")
  t0 = time.perf_counter()
  devices_per_slice = max(1, jax.device_count() // num_slices)
  argv = [
      "",
      base_yml,
      f"model_name={model_name}",
      "source_checkpoint_layout=safetensors_dynamic",
      f"safetensors_dynamic_streaming={str(streaming).lower()}",
      f"load_parameters_path={ckpt_path}",
      "weight_dtype=bfloat16",
      "dtype=bfloat16",
      "scan_layers=true",
      "attention=dot_product",
      "per_device_batch_size=1",
      "max_target_length=32",
      f"num_slices={num_slices}",
      f"dcn_fsdp_parallelism={num_slices}",
      f"ici_fsdp_parallelism={devices_per_slice}",
      "skip_jax_distributed_system=true",
  ]
  cfg = pyconfig.initialize(argv)
  model, _ = model_creation_utils.from_pretrained(cfg)
  _sync(f"loaded_{label}")
  dt = time.perf_counter() - t0
  peak_gb = _peak_hbm_gb()
  digests = _digest_and_free(model)
  my_path.write_text(json.dumps({"dt": dt, "peak_gb": peak_gb, "digests": digests}))
  _sync(f"saved_{label}")
  return digests, dt, peak_gb


def _is_router_bias_leaf(key: str) -> bool:
  return key.endswith(("gate.bias", "moe_bias", "e_score_correction_bias"))


def _assert_digests_equal(a: dict, b: dict, label: str, allow_legacy_bf16_router_bias: bool = False) -> list[str]:
  assert set(a.keys()) == set(b.keys()), f"{label}: key mismatch {set(a) ^ set(b)}"
  mismatches = [k for k in sorted(a) if a[k] != b[k]]
  if allow_legacy_bf16_router_bias:
    unexpected = [k for k in mismatches if not _is_router_bias_leaf(k)]
    assert not unexpected, f"{label}: unexpected byte-digest mismatch on process {jax.process_index()}: {unexpected[:5]}"
  else:
    assert not mismatches, f"{label}: byte-digest mismatch on process {jax.process_index()}: {mismatches[:5]}"
  _sync(f"verified_{label}")
  return mismatches


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for multi-slice TPU DeepSeek parity verification."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", choices=["dsv4", "dsv3", "both"], default="both")
  parser.add_argument(
      "--num-slices",
      type=int,
      default=int(os.environ.get("NUM_SLICES", "4")),
      help="Number of TPU slices in the JobSet (defaults to NUM_SLICES env or 4).",
  )
  parser.add_argument("--repo-root", default="/tmp/maxtext")
  parser.add_argument("--gcs-root", default="gs://mesa-maxtext-us-central1/mesa-bench-v2")
  parser.add_argument("--out-gcs", required=True)
  parser.add_argument(
      "--resume",
      action="store_true",
      help="Resume from completed step-digest checkpoints under the output directory if present.",
  )
  args = parser.parse_args(argv)

  slice_index = int(os.environ.get("SLICE_INDEX", "0"))
  coord_host = os.environ.get("COORD_HOST", "")
  if not coord_host:
    raise RuntimeError("COORD_HOST environment variable is required for multi-slice TPU initialization.")
  coord_ip = _BOOT_COORD_IP or _configure_megascale_env(args.num_slices, slice_index, coord_host)
  if os.environ.get("MEGASCALE_NUM_SLICES") != str(args.num_slices):
    os.environ["MEGASCALE_NUM_SLICES"] = str(args.num_slices)

  jax.distributed.initialize(
      coordinator_address=f"{coord_ip}:8476",
      num_processes=args.num_slices,
      process_id=slice_index,
  )
  _log(
      f"Initialized JAX backend={jax.default_backend()} processes={jax.process_count()}"
      f" total_devices={jax.device_count()} local_devices={len(jax.local_devices())} suite={args.suite}"
  )
  assert jax.default_backend() == "tpu", f"Expected TPU backend, got {jax.default_backend()}"

  os.chdir(args.repo_root)
  base_yml = os.path.join(args.repo_root, "src/maxtext/configs/base.yml")
  gcs_root = args.gcs_root.rstrip("/")
  hf_quant = f"{gcs_root}/hf_cache"
  hf_bf16 = f"{gcs_root}/hf_cache_bf16"
  out_path = epath.Path(args.out_gcs)
  step_cache_dir = str(out_path.parent / f"step_cache_{out_path.stem}")
  results = {}

  if args.suite in ("dsv4", "both"):
    _log("=== Full deepseek4-284b (46 shards, FP8 + MXFP4 + F8_E8M0) ===")
    d_v4_stream, t_v4_stream, hbm_v4_stream = _load_and_digest(
        base_yml,
        "deepseek4-284b",
        f"{hf_quant}/deepseek4-284b",
        True,
        "dsv4_quant_stream",
        step_cache_dir,
        args.num_slices,
        args.resume,
    )
    _log(
        f"deepseek4-284b quant streaming=True loaded {len(d_v4_stream)} leaves in {t_v4_stream:.2f}s"
        f" (peak HBM {hbm_v4_stream:.1f} GB)"
    )

    d_v4_bf16, t_v4_bf16, hbm_v4_bf16 = _load_and_digest(
        base_yml,
        "deepseek4-284b",
        f"{hf_bf16}/deepseek4-284b",
        True,
        "dsv4_bf16_stream",
        step_cache_dir,
        args.num_slices,
        args.resume,
    )
    _log(f"deepseek4-284b offline BF16 streaming=True loaded in {t_v4_bf16:.2f}s (peak HBM {hbm_v4_bf16:.1f} GB)")
    v4_bias_diffs = _assert_digests_equal(
        d_v4_stream, d_v4_bf16, "dsv4_quant_stream_vs_bf16", allow_legacy_bf16_router_bias=True
    )
    _log(
        f"[PASS] deepseek4-284b quant(streaming=True) vs offline BF16:"
        f" {len(d_v4_stream) - len(v4_bias_diffs)}/{len(d_v4_stream)} leaves byte-identical"
        f" (legacy BF16-rounded router bias leaves: {v4_bias_diffs})"
    )

    d_v4_def, t_v4_def, hbm_v4_def = _load_and_digest(
        base_yml,
        "deepseek4-284b",
        f"{hf_quant}/deepseek4-284b",
        False,
        "dsv4_quant_default",
        step_cache_dir,
        args.num_slices,
        args.resume,
    )
    _log(f"deepseek4-284b quant streaming=False loaded in {t_v4_def:.2f}s (peak HBM {hbm_v4_def:.1f} GB)")
    _assert_digests_equal(d_v4_def, d_v4_stream, "dsv4_quant_default_vs_quant_stream")
    _log(
        f"[PASS] deepseek4-284b quant(streaming=False) == quant(streaming=True)"
        f" ({len(d_v4_def)}/{len(d_v4_def)} leaves 100% byte-identical including FP32 gate.bias)"
    )

    results["deepseek4-284b"] = {
        "leaves": len(d_v4_stream),
        "exact_weight_leaves_vs_bf16": len(d_v4_stream) - len(v4_bias_diffs),
        "legacy_bf16_router_bias_leaves": v4_bias_diffs,
        "default_vs_streaming_exact_leaves": len(d_v4_def),
        "quant_streaming_s": round(t_v4_stream, 2),
        "quant_default_s": round(t_v4_def, 2),
        "bf16_streaming_s": round(t_v4_bf16, 2),
        "peak_hbm_streaming_gb": round(hbm_v4_stream, 2),
        "peak_hbm_default_gb": round(hbm_v4_def, 2),
        "bit_exact_weights": True,
    }

  if args.suite in ("dsv3", "both"):
    _log("=== Full deepseek3-671b (163 shards, FP8 + F32 _scale_inv) ===")
    d_v3_stream, t_v3_stream, hbm_v3_stream = _load_and_digest(
        base_yml,
        "deepseek3-671b",
        f"{hf_quant}/deepseek3-671b",
        True,
        "dsv3_quant_stream",
        step_cache_dir,
        args.num_slices,
        args.resume,
    )
    _log(
        f"deepseek3-671b quant streaming=True loaded {len(d_v3_stream)} leaves in {t_v3_stream:.2f}s"
        f" (peak HBM {hbm_v3_stream:.1f} GB)"
    )

    d_v3_bf16, t_v3_bf16, hbm_v3_bf16 = _load_and_digest(
        base_yml,
        "deepseek3-671b",
        f"{hf_bf16}/deepseek3-671b",
        True,
        "dsv3_bf16_stream",
        step_cache_dir,
        args.num_slices,
        args.resume,
    )
    _log(f"deepseek3-671b offline BF16 streaming=True loaded in {t_v3_bf16:.2f}s (peak HBM {hbm_v3_bf16:.1f} GB)")
    v3_bias_diffs = _assert_digests_equal(
        d_v3_stream, d_v3_bf16, "dsv3_quant_stream_vs_bf16", allow_legacy_bf16_router_bias=True
    )
    _log(
        f"[PASS] deepseek3-671b quant(streaming=True) vs offline BF16:"
        f" {len(d_v3_stream) - len(v3_bias_diffs)}/{len(d_v3_stream)} leaves byte-identical"
        f" (legacy BF16-rounded router bias leaves: {v3_bias_diffs})"
    )

    results["deepseek3-671b"] = {
        "leaves": len(d_v3_stream),
        "exact_weight_leaves_vs_bf16": len(d_v3_stream) - len(v3_bias_diffs),
        "legacy_bf16_router_bias_leaves": v3_bias_diffs,
        "quant_streaming_s": round(t_v3_stream, 2),
        "bf16_streaming_s": round(t_v3_bf16, 2),
        "peak_hbm_streaming_gb": round(hbm_v3_stream, 2),
        "bit_exact_weights": True,
    }

  _sync("suite_complete")
  if jax.process_index() == 0:
    payload = {
        "suite": args.suite,
        "git_commit": _git_commit_from_tree(args.repo_root),
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    _log(f"Wrote summary JSON to {args.out_gcs}:\n{json.dumps(payload, indent=2)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
