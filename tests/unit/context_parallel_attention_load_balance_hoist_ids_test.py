# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""context_parallel_attention_load_balance_hoist_ids: the attention load-balance permute of the ids, out of the loop.

Under context_parallel_attention_load_balance every attention layer permutes its positions and segment ids into
DUAL_CHUNK_SWAP order, two ppermutes per array. They depend only on the batch, so jax.lax.scan already computes the
forward's permutes once, outside the layer scan, but the remat replay in the backward permutes them again in every
attention layer. With the key the layer names the permuted arrays and remat_policy=custom saves the name; the saved
value is a loop invariant, which the scan hoists, and the replay reads it.

On a CPU mesh of 8 devices, in a child process (see main), for a tiny hybrid Qwen3.5 (GatedDeltaNet, gated full
attention and MoE on a context 4 x expert 2 cp-as-ep mesh) with two packed segments per row and without segment ids,
and for a tiny dense decoder on an fsdp 2 x context 4 mesh, all with remat_policy=custom and scan_layers:
  - Values: the logits and every parameter gradient of <logits, cotangent> are bit-identical with the key on and off.
  - Program: in the jaxpr of that VJP, the backward (reverse) layer scan of the key-off program carries the int32
    ppermutes of the positions and segment ids, two per array, and that of the key-on program carries none. Outside
    every scan both programs have the same int32 ppermutes, the forward's, computed once; the float ppermutes are
    the same in every region. With the key the remat save list gains the name, and only then does the program name
    it.
  - Positive controls, each of which must fail one of those checks: a hoist that saves the natural-order positions
    (values), and a remat policy that never saves the name (program).
In this process: the save lists of maxtext_utils.get_save_and_offload_names, and the config validator, which rejects
the key where it would do nothing.
"""

import collections
import contextlib
import os
import re
import subprocess
import sys
import types
import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import maxtext_utils

_KEY = "context_parallel_attention_load_balance_hoist_ids"
_PASS_TOKEN = "CONTEXT_PARALLEL_ATTENTION_LOAD_BALANCE_HOIST_IDS_CHECKS_PASSED"
_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
_SEQ = 256


def _hybrid_argv(hoist: bool, **overrides):
  """A tiny hybrid Qwen3.5 (attention_load_balance_test.py's) with custom remat on a context 4 x expert 2 mesh."""
  args = {
      "run_name": "attention_lb_hoist_ids_hybrid",
      "model_name": "qwen3.5-397b-a17b",
      "override_model_config": "true",
      "hardware": "cpu",
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": "1",
      "ici_context_parallelism": "4",
      "ici_expert_parallelism": "2",
      "ici_tensor_parallelism": "1",
      "context_parallel_load_balance": "False",
      "context_parallel_attention_load_balance": "True",
      "attention": "dot_product",
      "use_gdn_kernel": "false",
      "sparse_matmul": "false",
      "megablox": "false",
      "use_ring_of_experts": "false",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "scan_layers": "True",
      "enable_checkpointing": "false",
      "enable_dropout": "false",
      "dataset_type": "synthetic",
      "base_num_decoder_layers": "8",  # two blocks of [3 GatedDeltaNet + 1 full attention]
      "base_emb_dim": "64",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "64",
      "mrope_section": "[3,3,2]",
      "base_mlp_dim": "32",
      "base_moe_mlp_dim": "32",
      "num_experts": "8",
      "num_experts_per_tok": "2",
      "shared_experts": "1",
      "gdn_key_head_dim": "16",
      "gdn_value_head_dim": "16",
      "gdn_num_key_heads": "2",
      "gdn_num_value_heads": "4",
      "gdn_chunk_size": "16",
      "vocab_size": "384",
      "max_target_length": str(_SEQ),
      "per_device_batch_size": "0.5",
      "use_multimodal": "false",
      "skip_jax_distributed_system": "True",
      "enable_tensorboard": "False",
      "remat_policy": "custom",
      "decoder_layer_input": "device",
      _KEY: str(hoist),
  }
  args.update({k: str(v) for k, v in overrides.items()})
  return [None, _BASE_CONFIG_PATH] + [f"{k}={v}" for k, v in args.items()]


def _dense_argv(hoist: bool, **overrides):
  """A tiny dense decoder (attention and MLP only) with custom remat on an fsdp 2 x context 4 mesh."""
  args = {
      "run_name": "attention_lb_hoist_ids_dense",
      "hardware": "cpu",
      "ici_fsdp_parallelism": "2",
      "ici_context_parallelism": "4",
      "ici_tensor_parallelism": "1",
      "context_parallel_load_balance": "False",
      "context_parallel_attention_load_balance": "True",
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "scan_layers": "True",
      "enable_checkpointing": "false",
      "enable_dropout": "false",
      "dataset_type": "synthetic",
      "base_num_decoder_layers": "2",
      "base_emb_dim": "64",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "base_mlp_dim": "64",
      "vocab_size": "384",
      "max_target_length": str(_SEQ),
      "per_device_batch_size": "0.25",
      "skip_jax_distributed_system": "True",
      "enable_tensorboard": "False",
      "remat_policy": "custom",
      "decoder_layer_input": "device",
      _KEY: str(hoist),
  }
  args.update({k: str(v) for k, v in overrides.items()})
  return [None, _BASE_CONFIG_PATH] + [f"{k}={v}" for k, v in args.items()]


# ----------------------------------------------------------------------------------------------------------------------
# In this process: the remat save lists and the validator.


def _remat_cfg(**fields):
  return types.SimpleNamespace(
      **{"remat_policy": "custom", "tensors_on_device": None, "tensors_to_offload": None, **fields}
  )


class SaveNamesTest(absltest.TestCase):
  """remat_policy=custom saves ATTENTION_LB_IDS on device with the key, and nothing changes without it."""

  def test_custom_gains_the_name(self):
    off = _remat_cfg(tensors_on_device=["decoder_layer_input", "context"], tensors_to_offload=["out_proj"])
    on = _remat_cfg(tensors_on_device=["decoder_layer_input", "context"], tensors_to_offload=["out_proj"], **{_KEY: True})
    self.assertEqual(maxtext_utils.get_save_and_offload_names(off), (["decoder_layer_input", "context"], ["out_proj"]))
    self.assertEqual(
        maxtext_utils.get_save_and_offload_names(on),
        (["decoder_layer_input", "context", maxtext_utils.ATTENTION_LB_IDS], ["out_proj"]),
    )
    self.assertEqual(maxtext_utils.ATTENTION_LB_IDS, "attention_lb_ids")

  def test_after_the_gdn_names(self):
    cfg = _remat_cfg(tensors_on_device=["gdn_states"], use_gdn_kernel=True, **{_KEY: True})
    save, offload = maxtext_utils.get_save_and_offload_names(cfg)
    self.assertEqual(save[-1], maxtext_utils.ATTENTION_LB_IDS)
    self.assertIn("gdn_t_inv", save)
    self.assertEqual(offload, [])

  def test_key_off_or_missing_changes_nothing(self):
    for cfg in (_remat_cfg(tensors_on_device=["context"]), _remat_cfg(tensors_on_device=["context"], **{_KEY: False})):
      self.assertEqual(maxtext_utils.get_save_and_offload_names(cfg), (["context"], []))

  def test_config_lists_are_not_mutated(self):
    on_device = ["decoder_layer_input"]
    maxtext_utils.get_save_and_offload_names(_remat_cfg(tensors_on_device=on_device, **{_KEY: True}))
    self.assertEqual(on_device, ["decoder_layer_input"])

  def test_other_policies_take_no_name(self):
    for policy in ("full", "minimal", "save_qkv_proj", "none", "qkv_proj_offloaded", "minimal_offloaded"):
      save, offload = maxtext_utils.get_save_and_offload_names(_remat_cfg(remat_policy=policy, **{_KEY: True}))
      self.assertNotIn(maxtext_utils.ATTENTION_LB_IDS, save + offload, policy)


def _validator_ns(**overrides):
  """The fields the validator reads, for a config that passes it."""
  fields = {
      _KEY: True,
      "context_parallel_attention_load_balance": True,
      "context_sharding": "context",
      "ici_context_parallelism": 4,
      "dcn_context_parallelism": 1,
      "remat_policy": "custom",
      "scan_layers": True,
  }
  fields.update(overrides)
  return types.SimpleNamespace(**fields)


def _validate(ns):
  return config_types.HardwareAndMesh.validate_context_parallel_attention_load_balance_hoist_ids(ns)


class ValidatorTest(parameterized.TestCase):
  """The validator rejects the key where nothing is hoisted, one case per requirement."""

  @parameterized.named_parameters(
      ("cp4", {}),
      ("cp2", {"ici_context_parallelism": 2}),
      ("dcn_cp2", {"ici_context_parallelism": 1, "dcn_context_parallelism": 2}),
      ("ici2_dcn2", {"ici_context_parallelism": 2, "dcn_context_parallelism": 2}),
      ("cp_as_expert_axis", {"context_sharding": "expert", "ici_expert_parallelism": 2, "dcn_expert_parallelism": 1}),
  )
  def test_accepted(self, overrides):
    ns = _validator_ns(**overrides)
    self.assertIs(_validate(ns), ns)

  @parameterized.named_parameters(
      (
          "no_attention_load_balance",
          {"context_parallel_attention_load_balance": False},
          "context_parallel_attention_load_balance=True",
      ),
      ("cp1", {"ici_context_parallelism": 1}, "an even context parallelism of at least 2 (got 1;"),
      ("cp3", {"ici_context_parallelism": 3}, "an even context parallelism of at least 2 (got 3;"),
      ("cp_unset", {"ici_context_parallelism": -1}, "an even context parallelism of at least 2 (got -1;"),
      ("remat_full", {"remat_policy": "full"}, "remat_policy='custom'"),
      ("remat_minimal", {"remat_policy": "minimal"}, "remat_policy='custom'"),
      ("remat_none", {"remat_policy": "none"}, "remat_policy='custom'"),
      ("remat_offloaded_preset", {"remat_policy": "minimal_offloaded"}, "remat_policy='custom'"),
      ("no_scan", {"scan_layers": False}, "scan_layers=True"),
  )
  def test_rejected(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"{_KEY}=True requires: .*{re.escape(requirement)}"):
      _validate(_validator_ns(**overrides))
    # The same config with the key off passes.
    _validate(_validator_ns(**{**overrides, _KEY: False}))

  def test_names_every_unmet_requirement(self):
    with self.assertRaises(ValueError) as raised:
      _validate(_validator_ns(remat_policy="full", scan_layers=False))
    self.assertIn("remat_policy='custom'", str(raised.exception))
    self.assertIn("scan_layers=True", str(raised.exception))

  def test_runs_on_both_config_classes(self):
    """A model validator of the HardwareAndMesh section, so it runs on MaxTextConfig and on RLConfig."""
    name = "validate_context_parallel_attention_load_balance_hoist_ids"
    for cls in (config_types.MaxTextConfig, config_types.RLConfig):
      self.assertIn(name, cls.__pydantic_decorators__.model_validators, cls.__name__)

  def test_pyconfig(self):
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      self.assertFalse(getattr(pyconfig.initialize(_hybrid_argv(False)), _KEY))
      self.assertTrue(getattr(pyconfig.initialize(_hybrid_argv(True)), _KEY))
      self.assertTrue(getattr(pyconfig.initialize(_dense_argv(True)), _KEY))
      with self.assertRaisesRegex(ValueError, f"{_KEY}=True requires: remat_policy='custom'"):
        pyconfig.initialize(_hybrid_argv(True, remat_policy="full"))
      with self.assertRaisesRegex(ValueError, f"{_KEY}=True requires: scan_layers=True"):
        pyconfig.initialize(_dense_argv(True, scan_layers=False))
      with self.assertRaisesRegex(ValueError, f"{_KEY}=True requires: context_parallel_attention_load_balance=True"):
        pyconfig.initialize(_hybrid_argv(True, context_parallel_attention_load_balance=False))
    self.assertFalse(getattr(config_types.HardwareAndMesh(), _KEY))


@pytest.mark.cpu_only
def test_values_and_program_in_a_child_process():
  """Runs `main` below in a fresh process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


# ----------------------------------------------------------------------------------------------------------------------
# Child process (main): values and program of the VJP on 8 CPU devices.


def _ppermute_census(closed_jaxpr) -> dict[str, int]:
  """ppermute equations by (region, operand dtype); region = the innermost enclosing scan (fwd_scan / rev_scan) or
  outside, where the backward of a scan is the scan with reverse=True."""
  counts = collections.Counter()

  def subjaxprs(params):
    for value in params.values():
      for item in value if isinstance(value, (tuple, list)) else (value,):
        if isinstance(item, jax_core.ClosedJaxpr):
          yield item.jaxpr
        elif isinstance(item, jax_core.Jaxpr):
          yield item

  def walk(jaxpr, region):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "ppermute":
        counts[f"{region}:{eqn.invars[0].aval.dtype}"] += 1
      inner = ("rev_scan" if eqn.params.get("reverse") else "fwd_scan") if eqn.primitive.name == "scan" else region
      for sub in subjaxprs(eqn.params):
        walk(sub, inner)

  walk(closed_jaxpr.jaxpr, "outside")
  return dict(sorted(counts.items()))


def _named(closed_jaxpr, name: str) -> int:
  """How many checkpoint_name equations in the jaxpr carry `name`."""
  return len(re.findall(rf"name\[name={re.escape(name)}\]", str(closed_jaxpr)))


def _batch(cfg, segments: bool):
  """Tokens, targets and positions for one micro-batch; with `segments`, two packed segments per row."""
  batch = cfg.micro_batch_size_to_train_on
  tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, _SEQ + 1), 0, cfg.vocab_size)
  boundary = (_SEQ * 5) // 8 if segments else _SEQ
  segmentation = jnp.where(jnp.arange(_SEQ) < boundary, 1, 2).astype(jnp.int32)
  positions = jnp.where(jnp.arange(_SEQ) < boundary, jnp.arange(_SEQ), jnp.arange(_SEQ) - boundary).astype(jnp.int32)
  data = {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": jnp.broadcast_to(positions, (batch, _SEQ)),
      "inputs_segmentation": jnp.broadcast_to(segmentation, (batch, _SEQ)) if segments else None,
      "targets_segmentation": jnp.broadcast_to(segmentation, (batch, _SEQ)) if segments else None,
  }
  cotangent = jax.random.normal(jax.random.PRNGKey(3), (batch, _SEQ, cfg.vocab_size), jnp.float32)
  return data, cotangent


def _vjp(cfg, data, cotangent, patch=None):
  """Logits, parameter gradients of <logits, cotangent>, and the jaxpr of one tiny model's jitted VJP."""
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.utils import maxtext_utils_nnx, model_creation_utils

  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=jax.random.PRNGKey(0))
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def logits_of(params):
    merged = nnx.merge(graphdef, params, rest, copy=True)
    logits = merged(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        enable_dropout=False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    return logits.astype(jnp.float32)

  def value_and_grads(params):
    logits, vjp = jax.vjp(logits_of, params)
    return logits, vjp(cotangent)[0]

  with patch or contextlib.nullcontext(), nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    traced = jax.jit(value_and_grads).trace(params)
    logits, grads = traced.lower().compile()(params)
  leaves = {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}
  return np.asarray(logits), leaves, traced.jaxpr


def _bitwise_mismatches(a, b) -> list[str]:
  """Names of the arrays (logits, then gradients) that differ in dtype, shape or bytes."""
  (logits_a, grads_a), (logits_b, grads_b) = a, b
  assert grads_a.keys() == grads_b.keys()
  pairs = [("logits", logits_a, logits_b)] + [(n, grads_a[n], grads_b[n]) for n in sorted(grads_a)]
  return [n for n, x, y in pairs if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes()]


def _program_failures(off_census, on_census, permuted_int_arrays: int) -> list[str]:
  """What the key-on program gets wrong against the key-off one, by the ppermute census."""
  expected_rev = 2 * permuted_int_arrays
  failures = []
  if off_census.get("rev_scan:int32", 0) != expected_rev:
    failures.append(
        f"key off: {off_census.get('rev_scan:int32', 0)} int32 ppermutes in the backward scan, not {expected_rev}"
    )
  if on_census.get("rev_scan:int32", 0) != 0:
    failures.append(f"key on: {on_census['rev_scan:int32']} int32 ppermutes left in the backward scan")
  if on_census.get("outside:int32", 0) != off_census.get("outside:int32", 0) or not off_census.get("outside:int32"):
    failures.append(
        f"int32 ppermutes outside every scan: off {off_census.get('outside:int32')} on {on_census.get('outside:int32')}"
    )
  floats_off = {k: v for k, v in off_census.items() if not k.endswith("int32")}
  floats_on = {k: v for k, v in on_census.items() if not k.endswith("int32")}
  if floats_off != floats_on:
    failures.append(f"float ppermutes differ: off {floats_off} on {floats_on}")
  return failures


def _check(label, ok, detail=""):
  print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}", flush=True)
  return ok


def _run_case(name, argv_fn, segments, permuted_int_arrays, controls=False):
  """Key off vs on for one model: values bit for bit, the program by its ppermute census, the save list."""
  # pylint: disable=import-outside-toplevel
  from jax.ad_checkpoint import checkpoint_name
  from maxtext.layers import attentions

  off_cfg, on_cfg = pyconfig.initialize(argv_fn(False)), pyconfig.initialize(argv_fn(True))
  data, cotangent = _batch(off_cfg, segments)
  off_logits, off_grads, off_jaxpr = _vjp(off_cfg, data, cotangent)
  on_logits, on_grads, on_jaxpr = _vjp(on_cfg, data, cotangent)
  off_census, on_census = _ppermute_census(off_jaxpr), _ppermute_census(on_jaxpr)
  ok = True
  mismatches = _bitwise_mismatches((off_logits, off_grads), (on_logits, on_grads))
  ok &= _check(f"{name} values", not mismatches, f"{1 + len(off_grads)} arrays, bit-identical except {mismatches}")
  failures = _program_failures(off_census, on_census, permuted_int_arrays)
  ok &= _check(f"{name} program", not failures, f"off {off_census} on {on_census} {failures}")
  off_save, _ = maxtext_utils.get_save_and_offload_names(off_cfg)
  on_save, _ = maxtext_utils.get_save_and_offload_names(on_cfg)
  ok &= _check(
      f"{name} save list", on_save == off_save + [maxtext_utils.ATTENTION_LB_IDS], f"off {off_save} on {on_save}"
  )
  named = (_named(off_jaxpr, maxtext_utils.ATTENTION_LB_IDS), _named(on_jaxpr, maxtext_utils.ATTENTION_LB_IDS))
  ok &= _check(f"{name} named only with the key", named[0] == 0 and named[1] > 0, f"name equations off/on {named}")
  if not controls:
    return ok

  # Positive control 1: a hoist that saves the natural-order positions must fail the values check.
  real_order = attentions.Attention._to_load_balanced_order  # pylint: disable=protected-access

  def natural_positions(self, inputs_q, inputs_kv, inputs_positions, decoder_segment_ids, *, use_shared_kv):
    q, kv, _, segment_ids = real_order(
        self, inputs_q, inputs_kv, inputs_positions, decoder_segment_ids, use_shared_kv=use_shared_kv
    )
    return q, kv, checkpoint_name(inputs_positions, maxtext_utils.ATTENTION_LB_IDS), segment_ids

  patch = unittest.mock.patch.object(attentions.Attention, "_to_load_balanced_order", natural_positions)
  bad_logits, bad_grads, _ = _vjp(on_cfg, data, cotangent, patch=patch)
  bad = _bitwise_mismatches((off_logits, off_grads), (bad_logits, bad_grads))
  ok &= _check(f"{name} control natural-order positions", len(bad) == 1 + len(off_grads), f"{len(bad)} arrays differ")

  # Positive control 2: a policy that never saves the name must fail the program check.
  real_names = maxtext_utils.get_save_and_offload_names

  def drop_name(config):
    save, offload = real_names(config)
    return [n for n in save if n != maxtext_utils.ATTENTION_LB_IDS], offload

  patch = unittest.mock.patch.object(maxtext_utils, "get_save_and_offload_names", drop_name)
  _, _, dropped_jaxpr = _vjp(on_cfg, data, cotangent, patch=patch)
  dropped = _program_failures(off_census, _ppermute_census(dropped_jaxpr), permuted_int_arrays)
  ok &= _check(f"{name} control policy without the name", bool(dropped), f"{dropped}")
  return ok


def main():
  assert len(jax.devices()) >= 8 and jax.devices()[0].platform == "cpu", jax.devices()
  ok = True
  # Packed: positions and segment ids are permuted (2 int32 arrays); unpacked: positions only.
  ok &= _run_case("hybrid packed", _hybrid_argv, segments=True, permuted_int_arrays=2, controls=True)
  ok &= _run_case("hybrid unpacked", _hybrid_argv, segments=False, permuted_int_arrays=1)
  ok &= _run_case("dense packed", _dense_argv, segments=True, permuted_int_arrays=2)
  if not ok:
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  if len(sys.argv) == 1:
    main()
  else:
    absltest.main()
