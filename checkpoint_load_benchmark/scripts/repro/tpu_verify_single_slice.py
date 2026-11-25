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

"""Single-slice TPU hardware verification suite (4-model regression & multi-host DeepSeek parity)."""

import argparse
import json
import os
import socket
import sys
import time

os.environ.setdefault("EPATH_USE_TF", "false")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
_NUM_HOSTS = int(os.environ.get("NUM_HOSTS", "1"))
if _NUM_HOSTS > 1:
  _cur = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
  if "--slicebuilder_use_insecure_grpc" not in _cur:
    os.environ["LIBTPU_INIT_ARGS"] = f"{_cur} --slicebuilder_use_insecure_grpc=true".strip()
else:
  for _k in ("MEGASCALE_NUM_SLICES", "MEGASCALE_SLICE_ID", "MEGASCALE_PORT", "MEGASCALE_COORDINATOR_ADDRESS"):
    os.environ.pop(_k, None)

# pylint: disable=wrong-import-position
from etils import epath
import jax
import jax.numpy as jnp

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

DEFAULT_REGRESSION_MODELS = ("qwen3-0.6b", "gemma3-4b", "deepseek2-16b", "gpt-oss-20b")
_SYNC_COUNTER = 0


def _sync(tag: str) -> None:
  global _SYNC_COUNTER  # pylint: disable=global-statement
  _SYNC_COUNTER += 1
  if _NUM_HOSTS > 1:
    jax.experimental.multihost_utils.sync_global_devices(f"repro_1s_{_SYNC_COUNTER}_{tag}")


def _log(msg: str) -> None:
  if jax.process_index() == 0:
    print(f"[TPU-Verify-1S][{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def _load_model_state(base_yml: str, model_name: str, ckpt_path: str, streaming: bool, extra_flags: list[str]):
  """Loads `model_name` via `from_pretrained` and returns `(weights_dict, elapsed_s, peak_hbm_gb)`."""
  _sync(f"start_{model_name}_{streaming}")
  t0 = time.perf_counter()
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
      "per_device_batch_size=1",
      "max_target_length=32",
      "skip_jax_distributed_system=true",
      *extra_flags,
  ]
  cfg = pyconfig.initialize(argv)
  model, _ = model_creation_utils.from_pretrained(cfg)
  _sync(f"loaded_{model_name}_{streaming}")
  dt = time.perf_counter() - t0
  _, state = nnx.split(model)
  flat = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep=".")
  weights = {
      k: v
      for k, v in flat.items()
      if isinstance(v, jax.Array) and not jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key)
  }
  return weights, dt, _peak_hbm_gb()


def _assert_states_equal(a: dict[str, jax.Array], b: dict[str, jax.Array], label: str) -> None:
  """Asserts that every local addressable shard of `a` and `b` matches bit-for-bit."""
  assert set(a.keys()) == set(b.keys()), f"{label}: key mismatch {set(a) ^ set(b)}"
  mismatches = []
  for k in sorted(a):
    # Compare shard-local addressable buffers on every host so we never gather 600+ GB onto host RAM.
    sa = a[k].addressable_shards
    sb = b[k].addressable_shards
    assert len(sa) == len(sb), f"{label} {k}: shard count {len(sa)} != {len(sb)}"
    for x, y in zip(sa, sb):
      assert x.index == y.index, f"{label} {k}: shard index {x.index} != {y.index}"
      if not bool(jnp.array_equal(x.data, y.data, equal_nan=True)):
        diff = float(jnp.max(jnp.abs(x.data.astype(jnp.float32) - y.data.astype(jnp.float32))))
        mismatches.append((k, diff))
        break
  assert not mismatches, f"{label}: mismatches on process {jax.process_index()}: {mismatches[:5]}"
  _sync(f"verified_{label}")


def run_regression_suite(base_yml: str, gcs_root: str, models: list[str]) -> dict:
  """Verifies `streaming=False` vs `streaming=True` bit-exact parity across the regression models."""
  results = {}
  hf_cache = f"{gcs_root.rstrip('/')}/hf_cache"
  extra = [f"ici_fsdp_parallelism={jax.device_count()}"]
  for model_name in models:
    path = f"{hf_cache}/{model_name}"
    _log(f"--- Regression test: {model_name} from {path} ---")
    w_def, t_def, hbm_def = _load_model_state(base_yml, model_name, path, streaming=False, extra_flags=extra)
    w_str, t_str, hbm_str = _load_model_state(base_yml, model_name, path, streaming=True, extra_flags=extra)
    _assert_states_equal(w_def, w_str, f"{model_name}_default_vs_streaming")
    _log(
        f"[PASS] {model_name}: {len(w_def)} leaves bit-exact (default={t_def:.2f}s HBM={hbm_def:.1f}GB,"
        f" streaming={t_str:.2f}s HBM={hbm_str:.1f}GB)"
    )
    results[model_name] = {
        "leaves": len(w_def),
        "default_s": round(t_def, 2),
        "streaming_s": round(t_str, 2),
        "peak_hbm_gb": round(max(hbm_def, hbm_str), 2),
        "bit_exact": True,
    }
    del w_def, w_str
  return results


def run_multihost_deepseek_suite(base_yml: str, gcs_root: str) -> dict:
  """Verifies full `deepseek4-284b` and `deepseek3-671b` quantized vs offline BF16 on a multi-host slice."""
  results = {}
  root = gcs_root.rstrip("/")
  hf_quant = f"{root}/hf_cache"
  hf_bf16 = f"{root}/hf_cache_bf16"
  num_devices = jax.device_count()
  extra = [f"ici_fsdp_parallelism={num_devices}"]

  _log(f"=== [1/2] Full deepseek4-284b across {jax.process_count()} hosts ({num_devices} devices) ===")
  w_v4_stream, t_v4_stream, hbm_v4_stream = _load_model_state(
      base_yml, "deepseek4-284b", f"{hf_quant}/deepseek4-284b", streaming=True, extra_flags=extra
  )
  _log(
      f"deepseek4-284b quant streaming=True loaded {len(w_v4_stream)} leaves in {t_v4_stream:.2f}s"
      f" (peak HBM {hbm_v4_stream:.1f} GB)"
  )

  w_v4_bf16, t_v4_bf16, hbm_v4_bf16 = _load_model_state(
      base_yml, "deepseek4-284b", f"{hf_bf16}/deepseek4-284b", streaming=True, extra_flags=extra
  )
  _log(f"deepseek4-284b offline BF16 streaming=True loaded in {t_v4_bf16:.2f}s (peak HBM {hbm_v4_bf16:.1f} GB)")
  _assert_states_equal(w_v4_stream, w_v4_bf16, "dsv4_quant_stream_vs_bf16")
  del w_v4_bf16

  w_v4_def, t_v4_def, hbm_v4_def = _load_model_state(
      base_yml, "deepseek4-284b", f"{hf_quant}/deepseek4-284b", streaming=False, extra_flags=extra
  )
  _log(f"deepseek4-284b quant streaming=False loaded in {t_v4_def:.2f}s (peak HBM {hbm_v4_def:.1f} GB)")
  _assert_states_equal(w_v4_def, w_v4_stream, "dsv4_quant_def_vs_stream")
  results["deepseek4-284b"] = {
      "leaves": len(w_v4_stream),
      "quant_streaming_s": round(t_v4_stream, 2),
      "quant_default_s": round(t_v4_def, 2),
      "bf16_streaming_s": round(t_v4_bf16, 2),
      "peak_hbm_streaming_gb": round(hbm_v4_stream, 2),
      "peak_hbm_default_gb": round(hbm_v4_def, 2),
      "bit_exact_vs_bf16": True,
  }
  del w_v4_stream, w_v4_def

  _log(f"=== [2/2] Full deepseek3-671b across {jax.process_count()} hosts ({num_devices} devices) ===")
  w_v3_stream, t_v3_stream, hbm_v3_stream = _load_model_state(
      base_yml, "deepseek3-671b", f"{hf_quant}/deepseek3-671b", streaming=True, extra_flags=extra
  )
  _log(
      f"deepseek3-671b quant streaming=True loaded {len(w_v3_stream)} leaves in {t_v3_stream:.2f}s"
      f" (peak HBM {hbm_v3_stream:.1f} GB)"
  )

  w_v3_bf16, t_v3_bf16, hbm_v3_bf16 = _load_model_state(
      base_yml, "deepseek3-671b", f"{hf_bf16}/deepseek3-671b", streaming=True, extra_flags=extra
  )
  _log(f"deepseek3-671b offline BF16 streaming=True loaded in {t_v3_bf16:.2f}s (peak HBM {hbm_v3_bf16:.1f} GB)")
  _assert_states_equal(w_v3_stream, w_v3_bf16, "dsv3_quant_stream_vs_bf16")
  results["deepseek3-671b"] = {
      "leaves": len(w_v3_stream),
      "quant_streaming_s": round(t_v3_stream, 2),
      "bf16_streaming_s": round(t_v3_bf16, 2),
      "peak_hbm_streaming_gb": round(hbm_v3_stream, 2),
      "bit_exact_vs_bf16": True,
  }
  del w_v3_stream, w_v3_bf16
  return results


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for single-slice TPU verification."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", required=True, choices=["regression", "deepseek_full"])
  parser.add_argument("--repo-root", default="/tmp/maxtext")
  parser.add_argument("--gcs-root", default="gs://mesa-maxtext-us-central1/mesa-bench-v2")
  parser.add_argument("--out-gcs", required=True)
  parser.add_argument(
      "--models",
      nargs="+",
      default=list(DEFAULT_REGRESSION_MODELS),
      help="Model list when --suite=regression.",
  )
  args = parser.parse_args(argv)

  if _NUM_HOSTS > 1:
    proc_id = int(os.environ["JOB_COMPLETION_INDEX"])
    coord_host = os.environ["COORD_HOST"]
    coord_ip = None
    for _ in range(90):
      try:
        coord_ip = socket.gethostbyname(coord_host)
        break
      except OSError:
        time.sleep(2)
    assert coord_ip is not None, f"Could not resolve {coord_host}"
    jax.distributed.initialize(
        coordinator_address=f"{coord_ip}:8476",
        num_processes=_NUM_HOSTS,
        process_id=proc_id,
    )
  _log(
      f"Initialized JAX backend={jax.default_backend()} processes={jax.process_count()}"
      f" total_devices={jax.device_count()} local_devices={len(jax.local_devices())}"
  )
  assert jax.default_backend() == "tpu", f"Expected TPU backend, got {jax.default_backend()}"

  os.chdir(args.repo_root)
  base_yml = os.path.join(args.repo_root, "src/maxtext/configs/base.yml")
  if args.suite == "regression":
    res = run_regression_suite(base_yml, args.gcs_root, args.models)
  else:
    res = run_multihost_deepseek_suite(base_yml, args.gcs_root)

  _sync("suite_complete")
  if jax.process_index() == 0:
    payload = {
        "suite": args.suite,
        "git_commit": _git_commit_from_tree(args.repo_root),
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "results": res,
    }
    epath.Path(args.out_gcs).write_text(json.dumps(payload, indent=2))
    _log(f"Wrote summary JSON to {args.out_gcs}:\n{json.dumps(payload, indent=2)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
