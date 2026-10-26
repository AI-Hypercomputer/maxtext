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

"""TPU Ahead-of-Time (AOT) compilation and HBM pre-flight checker for MaxText."""

import argparse
import re
import shlex
import subprocess
import sys

# Usable per-device HBM limits (GiB) by TPU generation prefix.
TOPOLOGY_HBM_GIB: dict[str, float] = {
    "v4": 31.7,
    "v5e": 15.7,
    "v5litepod": 15.7,
    "v5p": 94.7,
    "v6e": 31.2,
    "tpu7x": 191.5,
}
DEFAULT_HBM_LIMIT_GIB = 94.7


def infer_hbm_limit_gib(topology: str, override_gib: float | None = None) -> float:
  """Returns the per-device HBM ceiling (GiB) for a TPU topology string (e.g. 'v5p-128', 'v6e-16')."""
  if override_gib is not None and override_gib > 0:
    return float(override_gib)
  topo_lower = topology.lower().strip()
  for prefix, limit_gib in TOPOLOGY_HBM_GIB.items():
    if topo_lower.startswith(prefix):
      return limit_gib
  return DEFAULT_HBM_LIMIT_GIB


def build_base_config_argv(
    base_yml: str,
    model_name: str,
    topology: str,
    num_slices: int,
    seq_len: int,
    per_device_batch_size: float,
    extra_args: list[str],
) -> list[str]:
  """Constructs pyconfig argv for AOT compilation."""
  return [
      base_yml,
      f"model_name={model_name}",
      f"compile_topology={topology}",
      f"compile_topology_num_slices={num_slices}",
      f"max_target_length={seq_len}",
      f"max_prefill_predict_length={max(1, seq_len // 2)}",
      f"per_device_batch_size={per_device_batch_size}",
      *extra_args,
  ]


def run_train_aot(argv: list[str], hbm_limit_gib: float) -> tuple[bool, float]:
  """Runs maxtext.train_compile as a subprocess and parses peak HBM usage if reported."""
  cmd = [sys.executable, "-m", "maxtext.train_compile", *argv]
  proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
  combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
  if proc.returncode != 0:
    print(combined, file=sys.stderr)
    return False, 0.0

  # Parse 'Total memory allocated ... X.XX GiB' or bytes if printed by train_compile.
  gib_matches = re.findall(r"([0-9]+\.[0-9]+)\s*GiB", combined)
  peak_gib = max((float(x) for x in gib_matches), default=0.0)
  return peak_gib <= hbm_limit_gib, peak_gib


def run_forward_aot(
    base_yml: str,
    model_name: str,
    topology: str,
    num_slices: int,
    seq_len: int,
    per_device_batch_size: float,
    extra_args: list[str],
    hbm_limit_gib: float,
) -> tuple[bool, float]:
  """Compiles a model-agnostic forward pass for the target TPU topology and returns (passed, total_gib)."""
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.models import models  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT  # pylint: disable=import-outside-toplevel
  from flax import linen as nn  # pylint: disable=import-outside-toplevel
  import os  # pylint: disable=import-outside-toplevel

  argv = [
      "aot_preflight.py",
      *build_base_config_argv(base_yml, model_name, topology, num_slices, seq_len, per_device_batch_size, extra_args),
      "ici_fsdp_parallelism=-1",
      "remat_policy=full",
      "scan_layers=false",
      "weight_dtype=bfloat16",
      "matmul_precision=default",
      f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
  ]
  cfg = pyconfig.initialize(argv)
  topo_mesh = maxtext_utils.get_topology_mesh(cfg)
  quant = models.quantizations.configure_quantization(cfg)
  model = models.transformer_as_linen(cfg, mesh=topo_mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
  rng = jax.random.PRNGKey(0)

  init_vars = jax.jit(
      lambda r: model.init(
          {"params": r, "dropout": r, "aqt": r},
          jnp.zeros((1, seq_len), dtype=jnp.int32),
          jnp.zeros((1, seq_len), dtype=jnp.int32),
          model_mode=MODEL_MODE_TRAIN,
      )
  ).eval_shape(rng)
  logical_specs = nn.get_partition_spec(init_vars)
  param_shardings = nn.logical_to_mesh_sharding(logical_specs, topo_mesh, rules=cfg.logical_axis_rules)
  shaped_vars = jax.tree_util.tree_map(
      lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
      init_vars,
      param_shardings,
  )
  rep = NamedSharding(topo_mesh, P())
  tok_struct = jax.ShapeDtypeStruct((1, seq_len), jnp.int32, sharding=rep)

  def fwd(v, t, p):
    logits = model.apply(v, t, p, enable_dropout=False, model_mode=MODEL_MODE_TRAIN, rngs={"params": rng})
    return logits.astype(jnp.float32)

  compiled = (
      jax.jit(fwd, in_shardings=(param_shardings, rep, rep), out_shardings=rep)
      .lower(shaped_vars, tok_struct, tok_struct)
      .compile()
  )
  mem = compiled.memory_analysis()
  total_bytes = mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes - mem.alias_size_in_bytes
  total_gib = total_bytes / (1024**3)
  print(f"AOT Forward Peak HBM ({topology} x {num_slices}): {total_gib:.2f} GiB / {hbm_limit_gib:.2f} GiB")
  return total_gib <= hbm_limit_gib, total_gib


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("forward", "train"), default="forward", help="AOT check mode")
  parser.add_argument("--config", default="src/maxtext/configs/base.yml", help="Path to base.yml")
  parser.add_argument("--model-name", required=True, help="MaxText model_name")
  parser.add_argument("--topology", default="v5p-128", help="Target TPU compile_topology (e.g. v5p-128, v6e-16)")
  parser.add_argument("--num-slices", type=int, default=1, help="Number of slices")
  parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for AOT compile")
  parser.add_argument("--per-device-batch-size", type=float, default=1.0, help="Per-device batch size")
  parser.add_argument("--hbm-limit-gib", type=float, default=None, help="Override per-device HBM limit (GiB)")
  parser.add_argument("--extra-args", default="", help="Space-separated extra MaxText config key=value overrides")
  args = parser.parse_args()

  extra = shlex.split(args.extra_args) if args.extra_args else []
  limit_gib = infer_hbm_limit_gib(args.topology, args.hbm_limit_gib)

  if args.mode == "train":
    argv = build_base_config_argv(
        args.config, args.model_name, args.topology, args.num_slices, args.seq_len, args.per_device_batch_size, extra
    )
    ok, gib = run_train_aot(argv, limit_gib)
  else:
    ok, gib = run_forward_aot(
        args.config,
        args.model_name,
        args.topology,
        args.num_slices,
        args.seq_len,
        args.per_device_batch_size,
        extra,
        limit_gib,
    )

  if not ok:
    print(f"FAIL: AOT compilation failed or exceeded HBM limit ({gib:.2f} GiB > {limit_gib:.2f} GiB).", file=sys.stderr)
    return 1
  print(f"PASS: AOT HBM check succeeded ({gib:.2f} GiB <= {limit_gib:.2f} GiB).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
