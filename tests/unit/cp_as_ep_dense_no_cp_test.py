# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The cp-as-ep-dense-no-cp and cp-as-ep-dense-fsdp custom mesh rules on a CPU mesh.

Both rules are cp-as-ep with the dense weights ('embed', 'embed_attn', 'q_lora', 'kv_lora') taken off the context axis:
cp-as-ep-dense-no-cp shards them over fsdp x expert, cp-as-ep-dense-fsdp over fsdp only, and both route the dense
weights' gradients through a reduce-scatter over every token axis ('embed_grad', linears.route_dense_weight_grad).
Only the sharding changes, so the model must compute the same function.

The model is a tiny hybrid Qwen3.5 (GatedDeltaNet kernel path, gated full attention, routed and shared-expert MoE) on
16 CPU devices, fsdp2 x context4 x expert2, with sequence-sharded GatedDeltaNet context parallelism and ring of experts
over the expert-parallel group context x expert, as in the Qwen3.5-397B configuration. Every arm runs with the same
parameter values (the cp-as-ep initialization, placed on each rule's shardings). The checks run in a subprocess so the
forced device count takes effect before JAX initializes, with JAX's persistent compilation cache off.

  - Shardings: exactly the parameters whose embed dimension is 'embed' or 'embed_attn' move (attention and
    shared-expert projections, the shared-expert gate and the router), each losing 'context' (and 'expert' under
    cp-as-ep-dense-fsdp) and nothing else; the GatedDeltaNet projections ('embed_gdn') and every other parameter keep
    their cp-as-ep sharding.
  - Collectives: the compiled program all-gathers over fsdp x expert (4 devices) or fsdp (2) where cp-as-ep gathers
    over all 16.
  - Numerics: the loss is bit-identical to cp-as-ep, and every parameter gradient agrees to reduction order,
    max|new - old| <= 2**-6 * max|old| per leaf: the activations and the gradient partial sums are bfloat16 (the ragged
    sort supports no other activation dtype), so a changed reduction order moves a gradient by bfloat16 roundings. A
    float32 variant (dense MLP instead of the MoE block) must agree to 1e-5.
  - Positive controls, each of which must fail the numerics gate:
      the rule without its RoutedMoE change, which keeps ("context", "expert") as the expert-parallel axes for these
        rules as for cp-as-ep: expert parallelism over 'expert' alone while the expert weights are sharded over
        context x expert;
      one reshared weight (the first attention query kernel) scaled by 1 + 1e-3.
"""

# pylint: disable=protected-access

import collections
import contextlib
import dataclasses
import os
import re
import subprocess
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from maxtext.utils import globals as maxtext_globals

_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
_FSDP, _CONTEXT, _EXPERT = 2, 4, 2
_NUM_DEVICES = _FSDP * _CONTEXT * _EXPERT
# Four context ranks of two 64-token GatedDeltaNet kernel chunks each.
_SEQ = 512
# Reduction-order gate on every gradient leaf: max|new - old| <= tolerance * max|old|. The gradient partial sums are
# bfloat16, so a changed reduction order moves an element by bfloat16 roundings of the partial sums: four bfloat16
# ulps (2**-6) at the leaf's largest magnitude, log2 of the 16 summands (1.5 measured). The float32 variant gets a
# float32 reduction-order tolerance.
_BF16_TOL = 2.0**-6
_F32_TOL = 1e-5
# The float32 variant: a dense MLP instead of the MoE block (the ragged sort only supports bfloat16 activations).
_F32_DENSE = {"num_experts": 1, "num_experts_per_tok": 1, "dtype": "float32"}
_REFERENCE_RULE = "cp-as-ep"
# rule -> the mesh axes of the dense weights' embed dimension.
_RULES = {"cp-as-ep-dense-no-cp": ("fsdp", "expert"), "cp-as-ep-dense-fsdp": ("fsdp",)}
_PASS_TOKEN = "CP_AS_EP_DENSE_NO_CP_CHECKS_PASSED"


@pytest.mark.cpu_only
def test_cp_as_ep_dense_no_cp_on_cpu_mesh():
  """Runs `main` below in a child process with 16 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _config(rule, **overrides):
  """The tiny hybrid Qwen3.5 on the fsdp2 x context4 x expert2 mesh under `rule`, with `overrides` applied."""
  argv = [
      None,
      _BASE_CONFIG_PATH,
      "run_name=cp_as_ep_dense_no_cp_test",
      "model_name=qwen3.5-397b-a17b",
      "override_model_config=true",
      "hardware=cpu",
      f"custom_mesh_and_rule={rule}",
      f"ici_fsdp_parallelism={_FSDP}",
      f"ici_context_parallelism={_CONTEXT}",
      f"ici_expert_parallelism={_EXPERT}",
      "ici_tensor_parallelism=1",
      "context_parallel_load_balance=False",
      "attention=dot_product",
      "use_gdn_kernel=true",
      "gdn_cp_mode=seq",
      "sparse_matmul=true",
      "megablox=true",
      # Tokamax gmm v2 is TPU-only; every arm runs the same GMM, so CPU uses megablox.
      "use_tokamax_gmm=false",
      "use_gmm_v2=false",
      "use_ring_of_experts=true",
      "use_ragged_sort=true",
      "use_custom_sort_vjp=false",
      "ragged_buffer_factor=2.0",
      "num_moe_token_chunks=2",
      "ring_of_experts_local_routing=true",
      # The ragged sort only supports bfloat16 activations.
      "dtype=bfloat16",
      "weight_dtype=float32",
      "scan_layers=True",
      "remat_policy=custom",
      "decoder_layer_input=device",
      "context=device",
      "packing=false",
      "enable_checkpointing=false",
      "enable_dropout=false",
      "dataset_type=synthetic",
      "base_num_decoder_layers=8",  # two blocks of [3 GatedDeltaNet + 1 full attention]
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=64",
      "mrope_section=[3,3,2]",  # rotary_dim / 2 = 64 * 0.25 / 2
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=16",
      "num_experts_per_tok=2",
      "shared_experts=1",
      # The GatedDeltaNet kernel path needs head dims that are multiples of 128 and gdn_chunk_size=64.
      "gdn_key_head_dim=128",
      "gdn_value_head_dim=128",
      "gdn_num_key_heads=2",
      "gdn_num_value_heads=4",
      "gdn_chunk_size=64",
      "vocab_size=384",
      f"max_target_length={_SEQ}",
      "per_device_batch_size=0.5",
      "use_multimodal=false",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
  ]
  argv = [a for a in argv if a is None or a.split("=", 1)[0] not in overrides]
  return pyconfig.initialize(argv + [f"{key}={value}" for key, value in overrides.items()])


def _inputs(cfg):
  """Random tokens in one segment per row, and a fixed random cotangent on the logits.

  The cotangent makes the backward per-token sensitive: at random init the cross-entropy is nearly flat.
  """
  batch = cfg.micro_batch_size_to_train_on
  tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, _SEQ + 1), 0, cfg.vocab_size)
  ones = jnp.ones((batch, _SEQ), jnp.int32)
  data = {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32), (batch, _SEQ)),
      "inputs_segmentation": ones,
      "targets_segmentation": ones,
  }
  cotangent = jax.random.normal(jax.random.PRNGKey(3), (batch, _SEQ, cfg.vocab_size), jnp.float32)
  return data, cotangent


@dataclasses.dataclass
class _Result:
  loss: np.ndarray
  grads: dict[str, np.ndarray]
  specs: dict[str, tuple]  # parameter -> the PartitionSpec of its sharding
  values: dict[str, np.ndarray]  # parameter -> its value
  hlo: str


def _run(cfg, data, cotangent, values=None, control=None) -> _Result:
  """Loss <logits, cotangent> and its parameter gradients; `values` replaces the initial parameter values."""
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.utils import maxtext_utils_nnx, model_creation_utils

  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with control or contextlib.nullcontext():
    # As the trainer builds it: the parameters are created directly on the rule's shardings.
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      create_model, abstract_model = model_creation_utils.create_nnx_abstract_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=jax.random.PRNGKey(0)
      )
      model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_model, mesh=mesh)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    keyed = jax.tree_util.tree_leaves_with_path(params)
    specs = {jax.tree_util.keystr(path): tuple(leaf.sharding.spec) for path, leaf in keyed}
    if values is not None:
      assert values.keys() == specs.keys()
      params = jax.tree_util.tree_map_with_path(
          lambda path, leaf: jax.device_put(values[jax.tree_util.keystr(path)].astype(leaf.dtype), leaf.sharding),
          params,
      )

    def loss_fn(params):
      merged = nnx.merge(graphdef, params, rest, copy=True)
      logits = merged(
          decoder_input_tokens=data["inputs"],
          decoder_positions=data["inputs_position"],
          decoder_segment_ids=data["inputs_segmentation"],
          enable_dropout=False,
          decoder_target_tokens=data["targets"],
          decoder_target_mask=data["targets_segmentation"],
      )
      return jnp.sum(logits.astype(jnp.float32) * cotangent)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      compiled = jax.jit(jax.value_and_grad(loss_fn)).lower(params).compile()
      loss, grads = compiled(params)
  as_np = lambda tree: {jax.tree_util.keystr(p): np.asarray(x) for p, x in jax.tree_util.tree_leaves_with_path(tree)}
  return _Result(np.asarray(loss), as_np(grads), specs, as_np(params), compiled.as_text())


# ---------------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------------


def _numerics(label, result, reference, tol=_BF16_TOL) -> bool:
  """Bit-identical loss and every gradient leaf within reduction order of the reference; prints the report."""
  assert result.grads.keys() == reference.grads.keys()
  loss_equal = bool(np.array_equal(result.loss, reference.loss))
  worst, worst_name, worst_l2, not_bit_equal, grads_ok = 0.0, "", 0.0, 0, True
  for name, ref in reference.grads.items():
    new, ref = result.grads[name].astype(np.float64), ref.astype(np.float64)
    diff, scale = float(np.max(np.abs(new - ref))), float(np.max(np.abs(ref)))
    # Every comparison is False on NaN, so a NaN fails.
    grads_ok &= bool(diff <= tol * scale) and bool(np.isfinite(new).all())
    rel = diff / scale if scale > 0 else diff
    if rel > worst or not worst_name:
      worst, worst_name = rel, name
      worst_l2 = float(np.linalg.norm(new - ref)) / max(float(np.linalg.norm(ref)), 1e-30)
    not_bit_equal += int(not np.array_equal(new, ref))
  passed = loss_equal and grads_ok
  print(
      f"{label}: numerics {'PASS' if passed else 'FAIL'}: loss bit-identical={loss_equal}"
      f" (d={float(result.loss) - float(reference.loss):.6e}); gradients within {tol:.3g} x max|ref|={grads_ok},"
      f" worst max|d|/max|ref|={worst:.3e} (||d||/||ref||={worst_l2:.3e}) at {worst_name},"
      f" {not_bit_equal} of {len(reference.grads)} leaves not bit-identical",
      flush=True,
  )
  return passed


def _mesh_axes(entry):
  """The mesh axes of one PartitionSpec entry, as a tuple."""
  if entry is None:
    return ()
  return tuple(entry) if isinstance(entry, (tuple, list)) else (entry,)


def _sharding_moves(rule, result, reference) -> bool:
  """Exactly the dense weights move, each from cp-as-ep's fsdp x context x expert to the rule's axes."""
  dense_axes, moved, wrong = _RULES[rule], [], []
  for name, ref_spec in reference.specs.items():
    spec = result.specs[name]
    if spec == ref_spec:
      continue
    moved.append(name)
    # The only change allowed: one dimension sharded over fsdp x context x expert becomes the rule's dense axes.
    changes = [(_mesh_axes(a), _mesh_axes(b)) for a, b in zip(ref_spec, spec) if a != b]
    if len(spec) != len(ref_spec) or changes != [(("fsdp", "context", "expert"), dense_axes)]:
      wrong.append((name, ref_spec, spec))
  required = (
      "['query']",
      "['key']",
      "['value']",
      "['out']",
      "shared_expert']['wi_0",
      "shared_expert']['wo",
      "shared_expert_gate",
      "routed_experts']['gate",
  )
  missing = [r for r in required if not any(r in name for name in moved)]
  kept = ("token_embedder", "routed_experts']['wi", "routed_experts']['wo", "in_proj_qkvz", "in_proj_ba", "out_proj")
  forbidden = [n for n in moved if any(k in n for k in kept)]
  print(f"{rule}: {len(moved)} of {len(reference.specs)} parameters change sharding:", flush=True)
  for name in moved:
    print(f"    {name}: {reference.specs[name]} -> {result.specs[name]}", flush=True)
  if wrong or missing or forbidden:
    print(f"{rule}: unexpected sharding changes {wrong}; required but unmoved {missing}; moved {forbidden}", flush=True)
  return not (wrong or missing or forbidden)


# The result shape is one token, or a parenthesised tuple of them when XLA combined several collectives into one.
_COLLECTIVE = re.compile(r"=\s*(?:\([^=]*?\)|\S+)\s+(all-gather|reduce-scatter|all-reduce|all-to-all)(?:-start)?\(")


def _group_size(line) -> int:
  """The replica-group size of one HLO collective instruction (explicit, iota or mesh form)."""
  if "replica_groups={}" in line:  # one group of every device
    return _NUM_DEVICES
  match = re.search(r"replica_groups=\{(\{[^}]*\})", line)
  if match:
    return len(match.group(1).strip("{}").split(","))
  match = re.search(r"replica_groups=\[(\d+),(\d+)\]<=", line)
  if match:
    return int(match.group(2))
  match = re.search(r"replica_groups=mesh\[([^\]]*)\](?:,\s*device_ids=\((?:[^()]|\([^()]*\))*\))?\s*\{([^}]*)\}", line)
  if match:
    sizes = dict(re.findall(r"'(\w+)'=(\d+)", match.group(1)))
    return int(np.prod([int(sizes[a]) for a in re.findall(r"'(\w+)'", match.group(2))]))
  return -1


def _collectives(hlo_text) -> collections.Counter:
  """(collective, replica-group size) -> count, over the compiled program."""
  counts = collections.Counter()
  for line in hlo_text.splitlines():
    match = _COLLECTIVE.search(line)
    if match:
      counts[(match.group(1), _group_size(line))] += 1
  return counts


# ---------------------------------------------------------------------------
# Positive controls.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _moe_without_rule_change():
  """RoutedMoE as before this change: expert parallelism over 'expert' alone under the new rules."""
  real = moe.RoutedMoE.__init__

  def init(self, *args, **kwargs):
    real(self, *args, **kwargs)
    assert self._expert_parallelism_name == ("context", "expert"), self._expert_parallelism_name
    self._expert_parallelism_name = "expert"

  with mock.patch.object(moe.RoutedMoE, "__init__", init):
    yield


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  jax.config.update("jax_enable_compilation_cache", False)
  reference_cfg = _config(_REFERENCE_RULE)
  data, cotangent = _inputs(reference_cfg)
  reference = _run(reference_cfg, data, cotangent)
  print(f"{_REFERENCE_RULE}: collectives {sorted(_collectives(reference.hlo).items())}", flush=True)
  checks, results = {}, {}
  for rule, dense_axes in _RULES.items():
    result = results[rule] = _run(_config(rule), data, cotangent, values=reference.values)
    gathers = _collectives(result.hlo)
    print(f"{rule}: collectives {sorted(gathers.items())}", flush=True)
    dense_group = int(np.prod([{"fsdp": _FSDP, "expert": _EXPERT}[a] for a in dense_axes]))
    checks[f"{rule}: exactly the dense weights move, off context"] = _sharding_moves(rule, result, reference)
    # Every all-gather over all devices that goes away comes back over the dense weights' smaller group.
    ref_gathers = _collectives(reference.hlo)
    gone = ref_gathers[("all-gather", _NUM_DEVICES)] - gathers[("all-gather", _NUM_DEVICES)]
    checks[f"{rule}: {_NUM_DEVICES}-device all-gathers become {dense_group}-device ones"] = (
        0 < gone <= gathers[("all-gather", dense_group)] - ref_gathers[("all-gather", dense_group)]
        # No all-gather this test cannot size beyond those of cp-as-ep.
        and gathers[("all-gather", -1)] <= ref_gathers[("all-gather", -1)]
    )
    checks[f"{rule}: numerics"] = _numerics(rule, result, reference)

  # The bfloat16 differences are reduction order: with float32 activations they shrink to float32 roundings.
  reference32 = _run(_config(_REFERENCE_RULE, **_F32_DENSE), data, cotangent)
  result32 = _run(_config("cp-as-ep-dense-no-cp", **_F32_DENSE), data, cotangent, values=reference32.values)
  checks["float32 dense variant: cp-as-ep-dense-no-cp numerics"] = _numerics(
      "float32 dense cp-as-ep-dense-no-cp", result32, reference32, tol=_F32_TOL
  )

  # Positive controls. Each must fail the numerics gate (or fail to run); `controls` records whether it was caught.
  controls = {}
  rule = "cp-as-ep-dense-no-cp"
  try:
    wrong = _run(_config(rule), data, cotangent, values=reference.values, control=_moe_without_rule_change())
    controls["the rule without its RoutedMoE change"] = not _numerics("control: no RoutedMoE change", wrong, reference)
  except Exception as error:  # pylint: disable=broad-exception-caught
    print(f"control: no RoutedMoE change: raised {type(error).__name__}: {str(error)[:400]}", flush=True)
    controls["the rule without its RoutedMoE change"] = True
  query = next(name for name in reference.values if re.search(r"\['query'\].*\['kernel'\]", name))
  perturbed = dict(reference.values, **{query: reference.values[query] * np.float32(1 + 1e-3)})
  wrong = _run(_config(rule), data, cotangent, values=perturbed)
  controls[f"{query} scaled by 1 + 1e-3"] = not _numerics("control: perturbed query", wrong, reference)

  for name, ok in {**checks, **{f"positive control caught: {k}": v for k, v in controls.items()}}.items():
    print(f"  [{'ok' if ok else 'FAIL'}] {name}", flush=True)
  failed = [name for name, ok in checks.items() if not ok]
  undetected = [name for name, caught in controls.items() if not caught]
  if failed or undetected:
    print(f"checks that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
