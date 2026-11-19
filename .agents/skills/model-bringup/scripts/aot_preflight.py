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

"""AOT-compiles the MaxText forward pass for a target TPU topology and reports per-device HBM usage.

Usage:
  python3 aot_preflight.py src/maxtext/configs/base.yml model_name=...
  compile_topology=v5p-128 --seqs=512,128

  `--model-name=X` and `--topology=Y` are shorthands for `model_name=X` and
  `compile_topology=Y`. The HBM limit is inferred from the topology unless
  `--hbm-limit-gib=Z` is passed. Script flags must use the `--flag=value` form.
"""

import sys

DEFAULT_HBM_LIMIT_GIB = 94.74
# Usable per-device HBM (GiB) by TPU generation, matched as a prefix of
# `compile_topology` (e.g. "v6e-16").
TOPOLOGY_HBM_GIB = (
    ("v4", 31.7),
    ("v5litepod", 15.7),
    ("v5e", 15.7),
    ("v5p", 94.74),
    ("v6e", 31.2),
    ("tpu7x", 191.5),
)
# Script shorthand flag -> MaxText config key it sets.
CONFIG_SHORTHANDS = {
    "--model-name": "model_name",
    "--topology": "compile_topology",
}


def infer_hbm_limit_gib(topology: str) -> float:
  """Returns the usable per-device HBM in GiB for a `compile_topology`."""
  topo = topology.lower().strip()
  for prefix, gib in TOPOLOGY_HBM_GIB:
    if topo.startswith(prefix):
      return gib
  print(
      f"WARNING: unknown compile_topology {topology!r}; using"
      f" {DEFAULT_HBM_LIMIT_GIB} GiB. Pass --hbm-limit-gib to override.",
      file=sys.stderr,
  )
  return DEFAULT_HBM_LIMIT_GIB


def split_argv(argv: list[str]) -> tuple[list[str], dict[str, str]]:
  """Splits argv into MaxText config argv and this script's `--flag` values."""
  own: dict[str, str] = {}
  mt_args: list[str] = []
  for a in argv[1:]:
    if not a.startswith("--"):
      mt_args.append(a)
    elif "=" in a:
      key, value = a.split("=", 1)
      own[key] = value
    else:
      sys.exit(f"ERROR: script flags must use --flag=value, got {a!r}")
  for flag, key in CONFIG_SHORTHANDS.items():
    if flag not in own:
      continue
    if any(a.startswith(f"{key}=") for a in mt_args):
      sys.exit(f"ERROR: pass either {flag}= or {key}=, not both")
    mt_args.append(f"{key}={own[flag]}")
  if not any(a.startswith("compile_topology_num_slices=") for a in mt_args):
    mt_args.append("compile_topology_num_slices=1")
  return [argv[0]] + mt_args, own


def main(argv: list[str]) -> None:
  """Runs AOT forward compilation for the requested sequence lengths and topology."""
  if any(a in ("-h", "--help") for a in argv[1:]):
    print((__doc__ or "").strip())
    return
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  from jax.sharding import NamedSharding  # pylint: disable=import-outside-toplevel
  from jax.sharding import PartitionSpec as P  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.trainers.pre_train import train_compile  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils import sharding  # pylint: disable=import-outside-toplevel

  mt_argv, own = split_argv(argv)
  config = pyconfig.initialize_pydantic(mt_argv)
  train_compile.validate_config(config)
  mesh = train_compile.get_topology_mesh(config)
  create_fn, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)
  abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, create_fn, True)
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    graphdef, _ = nnx.split(nnx.eval_shape(create_fn))
  batch = config.global_batch_size_to_train_on
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  repl = NamedSharding(mesh, P())
  ntok = int(own.get("--ntok", "8"))

  def fwd(st, ids, pos, seg):
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      m = nnx.merge(graphdef, st)
      logits = m(
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          encoder_images=None,
          enable_dropout=False,
      )
    return logits[:1, :ntok].astype(jnp.float32)

  seqs = [int(s) for s in own.get("--seqs", str(config.max_target_length)).split(",")]
  if "--hbm-limit-gib" in own:
    limit = float(own["--hbm-limit-gib"])
  else:
    limit = infer_hbm_limit_gib(config.compile_topology)
  failed = 0
  for seq in seqs:
    x = jax.ShapeDtypeStruct((batch, seq), jnp.int32, sharding=data_sharding)
    with jax.set_mesh(mesh):
      compiled = jax.jit(fwd, out_shardings=repl).lower(abstract_state, x, x, x).compile()
    ma = compiled.memory_analysis()
    tot = (ma.argument_size_in_bytes + ma.output_size_in_bytes + ma.temp_size_in_bytes - ma.alias_size_in_bytes) / 2**30
    status = "PASS" if tot < limit else "FAIL_OOM"
    failed += tot >= limit
    print(
        f"AOT_FWD [{status}] seq={seq} batch={batch}"
        f" arg={ma.argument_size_in_bytes/2**30:.2f}G"
        f" out={ma.output_size_in_bytes/2**30:.2f}G"
        f" temp={ma.temp_size_in_bytes/2**30:.2f}G"
        f" alias={ma.alias_size_in_bytes/2**30:.2f}G total={tot:.2f}G"
        f" limit={limit:.2f}G",
        flush=True,
    )
  sys.exit(1 if failed else 0)


if __name__ == "__main__":
  main(sys.argv)
