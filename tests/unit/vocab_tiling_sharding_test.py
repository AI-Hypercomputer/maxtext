# Copyright 2026 Google LLC
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

"""Tests for how vocabulary tiling shards the tokens of the loss head.

`vocab_tiling_nnx_loss` reshapes the final hidden states from [batch, length, embed] to
[num_vocab_tiling, batch * length / num_vocab_tiling, embed], and shards the flattened dimension
with the `activation_embed_and_logits_batch_sequence` rule. When that rule leaves out a mesh axis
that shards the batch or the length, every device along that axis computes the logits and the
cross entropy of the tokens the other devices own too. The loss is still right, so nothing
fails. `cp-as-ep` left out `context` this way: at context parallelism 4, every device computed
the loss head for 4x its own tokens.

Two kinds of test:

  - Coverage, from the config alone: in every shipped rule set, the flattened rule includes every
    mesh axis that the batch and length rules use.
  - Numerics, on 8 CPU devices: a shrunk Qwen3.5-397B-A17B (GDN, full attention and MoE layers)
    under `cp-as-ep` on an fsdp1 x context4 x expert2 mesh. Loss and every parameter gradient,
    through the whole model, match between the rule before and after the fix to f32
    reduction-order tolerance, and both match the untiled loss. The compiled program shows each
    device computing a quarter of the logits rows it did before. The same comparison fails on
    three deliberately wrong inputs, so it can fail.

The numerics test runs in a subprocess, because the CPU backend fixes its device count when it
initializes, which another test module has usually done by the time pytest imports this one.
"""

import glob
import os
import re
import subprocess
import sys
import tempfile

from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.utils.vocabulary_tiling import vocab_tiling_nnx_loss

pytestmark = pytest.mark.cpu_only

_FLAT_RULE = "activation_embed_and_logits_batch_sequence"
_HIDDEN_AXES = ("activation_embed_and_logits_batch", "activation_length", "activation_embed")
_FLAT_AXES = (_FLAT_RULE, "activation_embed")

_REQUIRED_DEVICES = 8
_SENTINEL = "VOCAB_TILING_SHARDING_CHECKS_PASSED"

# f32, so the only difference a sharding rule may make is the order of the reductions.
_RTOL = 1e-5


def _as_list(axes):
  if axes is None:
    return []
  return [axes] if isinstance(axes, str) else list(axes)


def _load_rule_set(path):
  with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  rules = tuple((rule[0], tuple(_as_list(rule[1]))) for rule in cfg["logical_axis_rules"])
  return rules, tuple(cfg["mesh_axes"])


def _shipped_rule_sets():
  """(name, path) for base.yml and every file under configs/custom_mesh_and_rule."""
  paths = [os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")]
  paths += sorted(glob.glob(os.path.join(MAXTEXT_CONFIGS_DIR, "custom_mesh_and_rule", "*.yml")))
  return [(os.path.splitext(os.path.basename(p))[0].replace("-", "_"), p) for p in paths]


def _with_flat_rule(rules, axes):
  """Returns `rules` with the flattened-token rule replaced by `axes`."""
  return tuple((name, tuple(axes)) if name == _FLAT_RULE else (name, value) for name, value in rules)


def _axes_missing_from_flat_rule(rules, mesh_axes):
  """Mesh axes that shard [batch, length] of the hidden states but not the flattened tokens."""
  hidden_spec = nn.logical_to_mesh_axes(_HIDDEN_AXES, rules)
  flat_spec = nn.logical_to_mesh_axes(_FLAT_AXES, rules)
  sharding_hidden = {a for dim in hidden_spec[:2] for a in _as_list(dim) if a in mesh_axes}
  sharding_flat = {a for a in _as_list(flat_spec[0]) if a in mesh_axes}
  return sharding_hidden - sharding_flat


class LossTokenShardingCoverageTest(parameterized.TestCase):
  """The flattened-token rule covers every axis that shards the batch or the length."""

  @parameterized.named_parameters(*_shipped_rule_sets())
  def test_flat_rule_covers_batch_and_length_axes(self, path):
    rules, mesh_axes = _load_rule_set(path)
    self.assertEmpty(
        _axes_missing_from_flat_rule(rules, mesh_axes),
        f"{os.path.basename(path)}: `{_FLAT_RULE}` leaves out mesh axes that shard the batch or the length of"
        " the hidden states, so every device along them computes the loss head for the others' tokens too.",
    )

  def test_check_flags_the_cp_as_ep_rule_before_the_fix(self):
    """Positive control: the rule `cp-as-ep` shipped before the fix leaves out `context`."""
    rules, mesh_axes = _load_rule_set(os.path.join(MAXTEXT_CONFIGS_DIR, "custom_mesh_and_rule", "cp-as-ep.yml"))
    old_rules = _with_flat_rule(rules, ("data", "stage", "fsdp", "expert"))
    self.assertEqual(_axes_missing_from_flat_rule(old_rules, mesh_axes), {"context"})

  def test_ep_as_cp_shards_the_flat_tokens_over_its_context_axis(self):
    """`ep-as-cp` has no `context` axis: `expert` shards the length, and the flattened rule has it."""
    rules, mesh_axes = _load_rule_set(os.path.join(MAXTEXT_CONFIGS_DIR, "custom_mesh_and_rule", "ep-as-cp.yml"))
    self.assertNotIn("context", mesh_axes)
    self.assertEqual(_as_list(nn.logical_to_mesh_axes(("activation_length",), rules)[0]), ["expert"])
    self.assertEqual(_as_list(nn.logical_to_mesh_axes((_FLAT_RULE,), rules)[0]), ["data", "stage", "fsdp", "expert"])


# ----------------------------------------------------------------------------------------------
# Numerics on 8 CPU devices.
# ----------------------------------------------------------------------------------------------

# The sharding flags of the Qwen3.5-397B-A17B cp-as-ep training configuration, on a model shrunk to run on CPU.
# The TPU kernels (splash attention, the GDN Pallas kernels, megablox) are replaced by their CPU
# implementations, and the MoE uses dense matmuls because ragged all-to-all is unimplemented on
# XLA:CPU. None of these changes touch the loss head.
_TINY_QWEN35_CP_AS_EP = (
    "model_name=qwen3.5-397b-a17b",
    "override_model_config=true",
    "custom_mesh_and_rule=cp-as-ep",
    "ici_fsdp_parallelism=1",
    "ici_context_parallelism=4",
    "ici_expert_parallelism=2",
    "ici_tensor_parallelism=1",
    "context_parallel_load_balance=false",
    "attention=dot_product",
    "use_gdn_kernel=false",
    "sparse_matmul=false",
    "megablox=false",
    "z_loss_multiplier=1e-4",
    "dtype=float32",
    "weight_dtype=float32",
    "matmul_precision=highest",
    "scan_layers=true",
    "enable_dropout=false",
    "enable_checkpointing=false",
    "dataset_type=synthetic",
    # One hybrid cycle: three GDN layers, then one full-attention layer. Every layer has an MoE.
    "base_num_decoder_layers=4",
    "base_emb_dim=64",
    "base_num_query_heads=4",
    "base_num_kv_heads=2",
    "head_dim=32",
    "base_mlp_dim=32",
    "base_moe_mlp_dim=32",
    "num_experts=8",
    "num_experts_per_tok=2",
    "shared_experts=1",
    "gdn_key_head_dim=16",
    "gdn_value_head_dim=16",
    "gdn_num_key_heads=2",
    "gdn_num_value_heads=4",
    "gdn_chunk_size=16",
    "vocab_size=384",
    "max_target_length=256",
    # Global batch 2: one sequence per shard of the batch axes, as in the 397B configuration.
    "per_device_batch_size=0.25",
    "use_multimodal=false",
    "use_mrope=false",
    "skip_jax_distributed_system=true",
    "enable_tensorboard=false",
)
_OLD_CP_AS_EP_FLAT_AXES = ("data", "stage", "fsdp", "expert")
_NEW_CP_AS_EP_FLAT_AXES = ("data", "stage", "fsdp", "expert", "context")


def _leaves(tree):
  return {jax.tree_util.keystr(p): np.asarray(v, np.float64) for p, v in jax.tree_util.tree_leaves_with_path(tree)}


def _rel(ref, other):
  ref_norm = np.linalg.norm(ref)
  if ref_norm == 0.0:
    return 0.0 if not np.any(other) else np.inf
  return float(np.linalg.norm(other - ref) / ref_norm)


def _per_leaf_rel(ref, other):
  """Largest relative difference over the loss and each gradient leaf on its own (l2 norm per leaf)."""
  (ref_loss, ref_grads), (other_loss, other_grads) = ref, other
  ref_leaves, other_leaves = _leaves(ref_grads), _leaves(other_grads)
  assert ref_leaves.keys() == other_leaves.keys() and ref_leaves
  return max(
      [_rel(np.float64(ref_loss), np.float64(other_loss))] + [_rel(v, other_leaves[k]) for k, v in ref_leaves.items()]
  )


def _whole_gradient_rel(ref, other):
  """Largest relative difference of the loss and of the whole gradient, all leaves as one vector."""
  (ref_loss, ref_grads), (other_loss, other_grads) = ref, other
  ref_leaves, other_leaves = _leaves(ref_grads), _leaves(other_grads)
  assert ref_leaves.keys() == other_leaves.keys() and ref_leaves
  ref_vec = np.concatenate([v.ravel() for _, v in sorted(ref_leaves.items())])
  other_vec = np.concatenate([other_leaves[k].ravel() for k, _ in sorted(ref_leaves.items())])
  return max(_rel(np.float64(ref_loss), np.float64(other_loss)), _rel(ref_vec, other_vec))


class CpAsEpLossRuleNumerics:
  """Builds the shrunk model once and compiles its loss and gradients under a given rule set."""

  def __init__(self, extra_args=()):

    def _config(num_vocab_tiling):
      return pyconfig.initialize(
          [
              "vocab_tiling_sharding_test.py",
              os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"),
              "run_name=vocab_tiling_sharding_test",
              f"base_output_directory={tempfile.mkdtemp()}",
              *_TINY_QWEN35_CP_AS_EP,
              f"num_vocab_tiling={num_vocab_tiling}",
              *extra_args,
          ]
      )

    self.config = _config(num_vocab_tiling=2)
    self.untiled_config = _config(num_vocab_tiling=1)
    self.mesh = maxtext_utils.get_mesh_from_config(self.config)
    self.rules = tuple((name, tuple(_as_list(value))) for name, value in self.config.logical_axis_rules)

    def _model(config):
      rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=jax.random.PRNGKey(0))
      return model_creation_utils.from_config(config, mesh=self.mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)

    with nn_partitioning.axis_rules(self.rules):
      self.graphdef, self.params, self.rest = nnx.split(_model(self.config), nnx.Param, ...)
      # The decoder reads num_vocab_tiling from its own config: when it is above 1 it returns no
      # logits, only the hidden states. The untiled reference therefore needs its own graph, and
      # shares the parameters of the tiled one.
      self.untiled_graphdef = nnx.split(_model(self.untiled_config), nnx.Param, ...)[0]

    batch, length = self.config.micro_batch_size_to_train_on, self.config.max_target_length
    tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, length + 1), 0, self.config.vocab_size)
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))
    self.data = {
        "inputs": tokens[:, :-1],
        "targets": tokens[:, 1:],
        "inputs_position": positions,
        "targets_position": positions,
        "inputs_segmentation": jnp.ones((batch, length), jnp.int32),
        "targets_segmentation": jnp.ones((batch, length), jnp.int32),
    }
    self.hidden_states = jax.random.normal(jax.random.PRNGKey(11), (batch, length, self.config.emb_dim), jnp.float32)

  def compile(self, rules, *, untiled=False):
    """Compiles value_and_grad of `train.loss_fn` under `rules`; returns (executable, HLO text)."""
    config = self.untiled_config if untiled else self.config
    graphdef = self.untiled_graphdef if untiled else self.graphdef

    def loss(params, rest, data):
      model = nnx.merge(graphdef, params, rest, copy=True)
      value, _ = train.loss_fn(model, config, dict(data), None, None, is_train=True)
      return value

    with nn_partitioning.axis_rules(rules), jax.set_mesh(self.mesh):
      compiled = jax.jit(jax.value_and_grad(loss)).lower(self.params, self.rest, self.data).compile()
    return compiled, compiled.as_text()

  def run(self, compiled, data=None):
    loss, grads = compiled(self.params, self.rest, self.data if data is None else data)
    return float(loss), jax.device_get(grads)

  def compile_head(self, rules, *, untiled=False):
    """Compiles the loss head alone, on fixed hidden states: loss and gradients for params and hidden states.

    Tiled, this is `vocab_tiling_nnx_loss`, the only function the flattened-token rule reaches.
    Untiled, it is full-vocabulary logits from the same output head, which no rule of the tiled
    path touches.
    """
    config = self.config

    def loss(params, rest, hidden_states, data):
      model = nnx.merge(self.graphdef, params, rest, copy=True)
      if untiled:
        logits = model.logits_from_hidden_states_for_vocab_tiling(hidden_states, True, MODEL_MODE_TRAIN)
        one_hot = jax.nn.one_hot(data["targets"], config.vocab_size)
        xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=config.z_loss_multiplier)
        return jnp.sum(xent * (data["targets_segmentation"] != 0))
      total_loss, _ = vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train=True)
      return total_loss

    with nn_partitioning.axis_rules(rules), jax.set_mesh(self.mesh):
      return (
          jax.jit(jax.value_and_grad(loss, argnums=(0, 2)))
          .lower(self.params, self.rest, self.hidden_states, self.data)
          .compile()
      )

  def run_head(self, compiled, data=None):
    loss, grads = compiled(self.params, self.rest, self.hidden_states, self.data if data is None else data)
    return float(loss), jax.device_get(grads)

  def logits_rows_per_device(self, hlo):
    """Row counts of the rank-2 [rows, vocab] logits tiles in a compiled per-device program."""
    return sorted({int(r) for r in re.findall(rf"f32\[(\d+),{self.config.vocab_size}\]", hlo)})


def _wrong_inputs(data, context, vocab_size):
  """Deliberately wrong batches, each a mistake a broken token layout would make.

  A sharding rule cannot be the wrong input here: a GSPMD sharding constraint never changes what
  is computed, only where, so any rule gives the same loss up to reduction order. These batches
  are what the loss would see if a layout bug paired hidden states with the wrong labels or
  dropped a rank's tokens.
  """
  targets, segmentation = np.asarray(data["targets"]), np.asarray(data["targets_segmentation"])
  batch, length = targets.shape

  moved = targets.copy()
  moved[-1, -1] = (moved[-1, -1] + 1) % vocab_size  # one token of the batch paired with the wrong label

  dropped = segmentation.copy()
  dropped[-1, length - length // context :] = 0  # the tokens of one context rank left out of the loss

  # Labels flattened context-major while the hidden states are flattened batch-major.
  context_major = targets.reshape(batch, context, length // context).transpose(1, 0, 2).reshape(batch, length)

  return {
      "one_label_moved": dict(data, targets=jnp.asarray(moved)),
      "one_context_rank_dropped": dict(data, targets_segmentation=jnp.asarray(dropped)),
      "labels_flattened_context_major": dict(data, targets=jnp.asarray(context_major)),
  }


def _check_equal(label, pairs, metric):
  for name, (ref, other) in pairs.items():
    diff = metric(ref, other)
    print(f"{label} {name}: {diff:.3e} (tolerance {_RTOL:.0e})", flush=True)
    assert diff <= _RTOL, f"{label} {name}: {diff:.3e} > {_RTOL:.0e}"


def _run_numerics_checks():
  """The subprocess body. Raises AssertionError on any failed check."""
  harness = CpAsEpLossRuleNumerics()
  cfg = harness.config
  shipped = dict(harness.rules)[_FLAT_RULE]
  assert shipped == _NEW_CP_AS_EP_FLAT_AXES, f"cp-as-ep.yml ships {shipped}, expected {_NEW_CP_AS_EP_FLAT_AXES}"
  assert dict(harness.mesh.shape) == {"data": 1, "stage": 1, "fsdp": 1, "context": 4, "expert": 2}
  context = harness.mesh.shape["context"]
  old_rules = _with_flat_rule(harness.rules, _OLD_CP_AS_EP_FLAT_AXES)
  wrong_inputs = _wrong_inputs(harness.data, context, cfg.vocab_size)

  # 1. The loss head alone, where the rule acts: loss, every parameter gradient and the gradient of
  #    the hidden states, each on its own, match between the old rule, the new rule and no tiling.
  head_old_exe, head_new_exe = harness.compile_head(old_rules), harness.compile_head(harness.rules)
  head_old, head_new = harness.run_head(head_old_exe), harness.run_head(head_new_exe)
  head_untiled = harness.run_head(harness.compile_head(harness.rules, untiled=True))
  _check_equal(
      "loss head, per leaf",
      {
          "new-vs-old": (head_old, head_new),
          "new-vs-untiled": (head_untiled, head_new),
          "old-vs-untiled": (head_untiled, head_old),
      },
      _per_leaf_rel,
  )

  # 2. The whole model: train.loss_fn through the GDN, attention and MoE layers. The whole gradient
  #    is compared as one vector. Per leaf it is not: the gradients of the GDN decay parameters
  #    (A_log, dt_bias) of the first layers amplify rounding in the loss head about 60x (scaling the
  #    LM-head kernel by one f32 ulp moves them by ~7e-6), so they differ by ~1e-5 between any two
  #    programs that sum the head in a different order, the old rule against no tiling included.
  #    The largest single-leaf difference is printed, not asserted.
  old_exe, old_hlo = harness.compile(old_rules)
  new_exe, new_hlo = harness.compile(harness.rules)
  old, new = harness.run(old_exe), harness.run(new_exe)
  untiled = harness.run(harness.compile(harness.rules, untiled=True)[0])
  print(f"loss old={old[0]!r} new={new[0]!r} untiled={untiled[0]!r}", flush=True)
  full_pairs = {"new-vs-old": (old, new), "new-vs-untiled": (untiled, new), "old-vs-untiled": (untiled, old)}
  _check_equal("whole model, whole gradient", full_pairs, _whole_gradient_rel)
  for name, (ref, other) in full_pairs.items():
    print(f"whole model, largest single leaf {name}: {_per_leaf_rel(ref, other):.3e} (not asserted)", flush=True)

  # 3. The fix takes effect: each device computes the logits of its own tokens only.
  tile_rows = cfg.micro_batch_size_to_train_on * cfg.max_target_length // cfg.num_vocab_tiling
  rows_old, rows_new = tile_rows // (jax.device_count() // context), tile_rows // jax.device_count()
  assert cfg.emb_dim not in (rows_old, rows_new), "the [emb, vocab] kernel would match the logits pattern"
  old_rows, new_rows = harness.logits_rows_per_device(old_hlo), harness.logits_rows_per_device(new_hlo)
  print(f"logits rows per device: old={old_rows} new={new_rows} (expected {rows_old} -> {rows_new})", flush=True)
  assert rows_old in old_rows and rows_new not in old_rows, old_rows
  assert rows_new in new_rows and rows_old not in new_rows, new_rows

  # 4. Both comparisons can fail: each wrong batch, through the new rule, must be caught by each.
  for name, wrong_data in wrong_inputs.items():
    head_diff = _per_leaf_rel(head_old, harness.run_head(head_new_exe, wrong_data))
    full_diff = _whole_gradient_rel(old, harness.run(new_exe, wrong_data))
    print(f"positive control {name}: loss head {head_diff:.3e}, whole model {full_diff:.3e}", flush=True)
    assert head_diff > _RTOL, f"positive control {name} passed the loss-head check ({head_diff:.3e})"
    assert full_diff > _RTOL, f"positive control {name} passed the whole-model check ({full_diff:.3e})"

  print(_SENTINEL, flush=True)


def test_cp_as_ep_loss_rule_numerics_on_cpu_mesh():
  """Loss and gradients are unchanged by the cp-as-ep loss rule fix, which quarters the logits rows."""
  if jax.device_count() >= _REQUIRED_DEVICES and jax.default_backend() == "cpu":
    _run_numerics_checks()
    return
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  assert _SENTINEL in result.stdout, f"the child did not finish its checks\n{report}"


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _run_numerics_checks()
