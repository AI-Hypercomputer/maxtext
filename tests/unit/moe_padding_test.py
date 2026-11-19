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

"""RoutedMoE with padding rows: the `token_mask` of `elastic_global_batch_size_mode='constant_padded'`.

A batch padded with all-zero rows (`token_mask` False on them) must be, for the real rows, the unpadded batch: the
same routing (expert ids and weights, exactly), the same outputs and gradients (up to float32 rounding: the padded
run is a different XLA program), the same load-balance loss and the same routed-bias update. The padded rows must
get no routing and a zero output. Covered on the dense path (with and without `capacity_factor`), the dropless
sparse path (megablox, in interpret mode off TPU) and a DeepSeek-style sigmoid router with the Megatron sequence
aux loss and loss-free balancing, plus the pure masked-mean and loss functions. The padding is placed per device
block and, on several devices, at the tail of the batch (whole batch shards without a real row), which exercises the
cross-shard reduction of the sparse path's Switch loss.
"""

import contextlib
import functools
import os
import subprocess
import sys
import types
import unittest

from absl.testing import parameterized
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils import linen_wrappers
from tests.utils.test_helpers import get_test_config_path

_SEQ_LEN = 16


def _mixtral_config(run_name, **overrides):
  """A tiny float32 Mixtral-style (softmax top-k router, Switch aux loss) config; `overrides` replace the defaults."""
  kwargs = {
      "run_name": run_name,
      "enable_checkpointing": False,
      "decoder_block": "mixtral",
      "base_emb_dim": 32,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 64,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      "sparse_matmul": False,
      "megablox": False,
      "per_device_batch_size": 2,
      "max_target_length": _SEQ_LEN,
      "load_balance_loss_weight": 0.01,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def _deepseek_config(run_name, **overrides):
  """A tiny float32 DeepSeek-V3 config: sigmoid router, Megatron seq aux loss and loss-free balancing."""
  kwargs = {
      "run_name": run_name,
      "enable_checkpointing": False,
      "model_name": "deepseek3-tiny",
      "override_model_config": True,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      # megablox runs its Pallas kernels in interpret mode off-TPU.
      "megablox": True,
      "sparse_matmul": True,
      "per_device_batch_size": 2,
      "max_target_length": _SEQ_LEN,
      "load_balance_loss_weight": 0.01,
      "moe_use_megatron_seq_aux_loss": True,
      "routed_bias_update_rate": 1e-3,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def _router_stub(num_experts=8, num_experts_per_tok=2):
  """A stand-in with the attributes `RoutedMoE.load_balance_loss` / `megatron_seq_aux_loss` read."""
  router = types.SimpleNamespace(
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      config=types.SimpleNamespace(load_balance_loss_weight=0.01),
  )
  # megatron_seq_aux_loss calls self.load_balance_loss, so bind it to the stub.
  router.load_balance_loss = functools.partial(moe.RoutedMoE.load_balance_loss, router)
  return router


class _Harness:
  """Builds one RoutedMoE from a config and runs it (with gradients) on padded and unpadded batches."""

  def __init__(self, cfg, global_loss_cfg=None):
    self.cfg = cfg
    self.mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    self.model = self._build(cfg)
    # A dense-path twin (same router, same parameters) whose unpadded Switch loss is the mean over the global batch;
    # see global_switch_loss.
    self.global_loss_model = None if global_loss_cfg is None else self._build(global_loss_cfg)
    rng_model, rng_inputs = jax.random.split(jax.random.PRNGKey(11))
    # Real rows: a multiple of the device count, so every batch sharding divides. Padding doubles the batch.
    self.real_rows = 2 * jax.device_count()
    self.inputs = jax.random.normal(rng_inputs, (self.real_rows, _SEQ_LEN, cfg.base_emb_dim), jnp.float32)
    with self._scope():
      self.params = self.model.init({"params": rng_model, "dropout": rng_model}, self.inputs)["params"]

  def _build(self, cfg):
    return linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
    )

  def global_switch_loss(self):
    """The Switch loss of the unpadded batch as a mean over the global batch, from the dense-path twin.

    On the sparse path the Switch loss is formed inside the shard_map and returned through `out_specs=P()` without
    a reduction (pre-existing): the forward value is shard 0's mean over its own rows, while the transpose of that
    output averages the shards, so the gradient is that of the global mean. With a `token_mask` the masked loss now
    reduces across the shards (`reduce_axis_names`), i.e. the padded run returns the global mean, which the dense
    path (no shard_map) computes directly from the same routing.
    """
    _, lb_loss, _, _ = self.run(self.inputs, model=self.global_loss_model)
    return lb_loss

  @contextlib.contextmanager
  def _scope(self):
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.cfg.logical_axis_rules):
      yield

  def padded(self, tail=False):
    """The batch with as many all-zero rows as real ones, and the matching token_mask.

    By default the rows are padded per device block (two real rows, then two padded ones, per device) so that, under
    batch sharding, every device keeps the real rows it had in the unpadded batch and holds some real rows. With
    `tail=True` all padded rows follow the real ones, as `_PaddedElasticIterator` and the synthetic pipeline place
    them: under batch sharding the leading devices then hold only real rows and the trailing devices only padding,
    which is what the sparse path's Switch loss has to reduce across shards for (`reduce_axis_names`).
    """
    if tail:
      padded_inputs = jnp.concatenate([self.inputs, jnp.zeros_like(self.inputs)], axis=0)
      row_is_real = jnp.arange(padded_inputs.shape[0]) < self.real_rows
      token_mask = jnp.broadcast_to(row_is_real[:, None], padded_inputs.shape[:2])
      return padded_inputs, token_mask
    per_device = self.real_rows // jax.device_count()
    blocks = self.inputs.reshape(jax.device_count(), per_device, *self.inputs.shape[1:])
    padded_inputs = jnp.concatenate([blocks, jnp.zeros_like(blocks)], axis=1).reshape(-1, *self.inputs.shape[1:])
    row_is_real = jnp.concatenate(
        [jnp.ones((jax.device_count(), per_device), bool), jnp.zeros((jax.device_count(), per_device), bool)], axis=1
    ).reshape(-1)
    token_mask = jnp.broadcast_to(row_is_real[:, None], padded_inputs.shape[:2])
    return padded_inputs, token_mask

  def run(self, inputs, token_mask=None, model=None):
    """Returns (output, lb_loss, bias_updates, grads) where grads are of a loss over the real rows only."""
    model = self.model if model is None else model
    with self._scope():

      def loss_fn(params):
        out, lb_loss, bias_updates = model.apply({"params": params}, inputs, token_mask=token_mask)
        # The CE loss of a padded run sees only the real rows, so the gradient must not see the padded ones either.
        real = out if token_mask is None else jnp.where(token_mask[..., None], out, 0.0)
        loss = jnp.sum(real**2) / self.inputs.size
        if lb_loss is not None:
          loss = loss + lb_loss
        return loss, (out, lb_loss, bias_updates)

      (_, (out, lb_loss, bias_updates)), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(self.params)
    return out, lb_loss, bias_updates, grads


def _assert_close(actual, expected):
  """Equal up to float32 rounding.

  The padded and unpadded batches are different XLA programs (the per-device batch differs), so matmul tiling and
  reduction order may differ and a few elements come out one float32 ulp apart; the routing itself (expert ids and
  weights) is checked exactly in test_padded_rows_route_nothing.
  """
  expected = np.asarray(expected)
  scale = float(np.max(np.abs(expected))) if expected.size else 0.0
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6 * scale)


def _assert_trees_close(actual, expected):
  jax.tree_util.tree_map(_assert_close, actual, expected)


class _PaddingTestCase(parameterized.TestCase):
  """Shared assertion that a padded batch and its unpadded counterpart agree on the real rows."""

  # Where `_check` places the padded rows; see `_Harness.padded`.
  tail_padding = False

  def _check(self, config_fn, run_name, overrides=None, global_loss_run_name=None, expect_bias_updates=False):
    """Runs the padded and unpadded batches through a RoutedMoE built from `config_fn` and compares them.

    Args:
      config_fn: `_mixtral_config` or `_deepseek_config`; called here, not at import time.
      run_name: The run name of the config under test.
      overrides: Config overrides passed to `config_fn` on top of its defaults.
      global_loss_run_name: For sparse Switch-loss configs, the run name of the dense-path twin (`config_fn`'s
        defaults) whose unpadded loss is the global-batch mean the padded run must reproduce; see
        `_Harness.global_switch_loss`. The unpadded sparse run's own forward value is shard 0's per-shard
        statistic, so on several devices it is not the reference (it is checked to be the global mean's shard-0
        counterpart only through the gradients, which the shard_map transpose averages over the shards).
      expect_bias_updates: Whether the config produces a routed-bias update (loss-free balancing).

    Returns:
      The `_Harness` the batches ran through.
    """
    cfg = config_fn(run_name, **(overrides or {}))
    global_loss_cfg = None if global_loss_run_name is None else config_fn(global_loss_run_name)
    harness = _Harness(cfg, global_loss_cfg=global_loss_cfg)
    out, lb_loss, bias_updates, grads = harness.run(harness.inputs)
    padded_inputs, token_mask = harness.padded(tail=self.tail_padding)
    padded_out, padded_lb_loss, padded_bias_updates, padded_grads = harness.run(padded_inputs, token_mask)

    row_is_real = token_mask[:, 0]
    self.assertTrue(bool(jnp.all(jnp.isfinite(padded_out))))
    _assert_close(padded_out[row_is_real], out)
    np.testing.assert_array_equal(padded_out[~row_is_real], 0.0)
    self.assertIsNotNone(lb_loss)
    self.assertGreater(float(lb_loss), 0.0)
    # The padded loss is the mean over the real rows of the global batch; only the reduction order can differ.
    reference_lb_loss = lb_loss
    if global_loss_cfg is not None:
      reference_lb_loss = harness.global_switch_loss()
      if jax.device_count() == 1:
        # One shard: the sparse path's per-shard value is the global mean.
        np.testing.assert_allclose(lb_loss, reference_lb_loss, rtol=1e-6)
    np.testing.assert_allclose(padded_lb_loss, reference_lb_loss, rtol=1e-6)
    if expect_bias_updates:
      # rate * sign(mean_load - load) from integer expert counts: identical.
      self.assertIsNotNone(bias_updates)
      np.testing.assert_array_equal(padded_bias_updates, bias_updates)
    else:
      self.assertIsNone(bias_updates)
      self.assertIsNone(padded_bias_updates)
    _assert_trees_close(padded_grads, grads)
    return harness


_SPARSE = {"sparse_matmul": True, "megablox": True}


class RoutedMoePaddingTest(_PaddingTestCase):
  """Padded and unpadded batches agree on the real rows."""

  # The generated names are "test" + testcase_name, e.g. test_dense.
  @parameterized.named_parameters(
      {"testcase_name": "_dense", "config_fn": _mixtral_config, "run_name": "padding_dense"},
      # Expert capacity is per sequence, so a padded row (no expert ids) drops nothing of the real rows.
      {
          "testcase_name": "_dense_with_capacity_factor",
          "config_fn": _mixtral_config,
          "run_name": "padding_dense_capacity",
          "overrides": {"capacity_factor": 1.0},
      },
      {
          "testcase_name": "_sparse_dropless",
          "config_fn": _mixtral_config,
          "run_name": "padding_sparse",
          "overrides": _SPARSE,
          "global_loss_run_name": "padding_sparse_ref",
      },
      {
          "testcase_name": "_sparse_direct_token_gather",
          "config_fn": _mixtral_config,
          "run_name": "padding_sparse_gather",
          "overrides": {**_SPARSE, "moe_use_direct_token_gather": True},
          "global_loss_run_name": "padding_sparse_gather_ref",
      },
      {
          "testcase_name": "_deepseek_sparse_megatron_loss_and_bias",
          "config_fn": _deepseek_config,
          "run_name": "padding_deepseek_sparse",
          "expect_bias_updates": True,
      },
      {
          "testcase_name": "_deepseek_dense_switch_loss_and_bias",
          "config_fn": _deepseek_config,
          "run_name": "padding_deepseek_dense",
          "overrides": {"sparse_matmul": False, "moe_use_megatron_seq_aux_loss": False},
          "expect_bias_updates": True,
      },
  )
  def test(self, **case):
    self._check(**case)

  def test_padded_rows_route_nothing(self):
    """The router output itself: padded tokens get expert id -1 and weight 0, real tokens are untouched."""
    cfg = _mixtral_config("padding_topk")
    harness = _Harness(cfg)
    padded_inputs, token_mask = harness.padded()
    gate_kernel = harness.params["gate"]["kernel"].value
    logits = jnp.einsum("bse,ex->bsx", padded_inputs, gate_kernel, precision=jax.lax.Precision.HIGHEST)
    # Drive get_topk through a stand-in with the attributes it reads.
    stub = types.SimpleNamespace(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        per_expert_scale=None,
        dtype=jnp.float32,
        is_hash_routing=False,
    )
    weights, indices = moe.RoutedMoE.get_topk(stub, logits, None, token_mask=token_mask)
    row_is_real = token_mask[:, 0]
    ref_weights, ref_indices = moe.RoutedMoE.get_topk(stub, logits[row_is_real], None)
    np.testing.assert_array_equal(indices[row_is_real], ref_indices)
    np.testing.assert_array_equal(weights[row_is_real], ref_weights)
    np.testing.assert_array_equal(indices[~row_is_real], -1)
    np.testing.assert_array_equal(weights[~row_is_real], 0.0)
    # -1 ids are not counted by the loss-free-balancing expert counts.
    np.testing.assert_array_equal(
        moe.calculate_expert_counts(indices, cfg.num_experts), moe.calculate_expert_counts(ref_indices, cfg.num_experts)
    )


@pytest.mark.tpu_only
class RoutedMoePaddingTpuTest(RoutedMoePaddingTest):
  """The same checks on TPU (unmarked tests run on CPU only)."""


_TAIL_PADDING_DEVICES = 4
_TAIL_PADDING_MARKER = "MOE_TAIL_PADDING_TESTS_PASSED"


class TailPaddingTest(_PaddingTestCase):
  """Padding placed at the tail of the batch, so whole batch shards hold no real row.

  Under batch sharding over several devices the trailing devices receive only padded rows. Inside the sparse path's
  shard_map the Switch loss is formed per shard, so without the cross-shard reduction of `load_balance_loss`
  (`reduce_axis_names`) the all-padding shards would contribute a loss of 0 with the same weight as the real shards,
  halving the forward loss and its gradient here. Needs several devices: run on TPU (`TailPaddingTpuTest`), or on CPU
  through `test_tail_padding_on_cpu_mesh`, which forces `_TAIL_PADDING_DEVICES` host devices in a subprocess.
  """

  tail_padding = True

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest("Needs at least 2 devices so that some batch shards hold only padded rows.")

  # The generated names are "test" + testcase_name, e.g. test_sparse_dropless_tail.
  @parameterized.named_parameters(
      # The Switch loss of the shard_map path: forward loss (the global mean) and gradients of the unpadded batch.
      {
          "testcase_name": "_sparse_dropless_tail",
          "config_fn": _mixtral_config,
          "run_name": "tail_sparse",
          "overrides": _SPARSE,
          "global_loss_run_name": "tail_sparse_ref",
      },
      {
          "testcase_name": "_sparse_direct_token_gather_tail",
          "config_fn": _mixtral_config,
          "run_name": "tail_sparse_gather",
          "overrides": {**_SPARSE, "moe_use_direct_token_gather": True},
          "global_loss_run_name": "tail_sparse_gather_ref",
      },
      # Outside the shard_map the loss already sees the global batch; the placement of the padding must not matter.
      {"testcase_name": "_dense_tail", "config_fn": _mixtral_config, "run_name": "tail_dense"},
      # The Megatron seq aux loss is computed after the shard_map on the global arrays; the bias counts are psum'd.
      {
          "testcase_name": "_deepseek_sparse_megatron_loss_and_bias_tail",
          "config_fn": _deepseek_config,
          "run_name": "tail_deepseek_sparse",
          "expect_bias_updates": True,
      },
  )
  def test(self, **case):
    self._check(**case)

  def test_load_balance_loss_reduces_over_the_batch_shards(self):
    """The pure function: in a shard_map, `reduce_axis_names` turns the per-shard masked loss into the global one."""
    router = _router_stub()
    loss = moe.RoutedMoE.load_balance_loss
    num_devices = jax.device_count()
    rows = 2 * num_devices
    logits = jax.random.normal(jax.random.PRNGKey(0), (rows, 16, 8), jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    _, top_k = jax.lax.top_k(logits, 2)
    reference = loss(router, top_k, probs)
    # Tail padding: as many padded rows as real ones, so the trailing half of the shards holds no real row.
    pad_top_k = jnp.concatenate([top_k, jnp.full_like(top_k, -1)])
    pad_probs = jnp.concatenate([probs, jnp.full_like(probs, 1.0 / 8)])
    mask = jnp.broadcast_to(jnp.arange(2 * rows)[:, None] < rows, (2 * rows, 16))
    mesh = Mesh(np.array(jax.devices()), ("batch",))
    spec = P("batch")

    def sharded_loss(reduce_axis_names, top_k_indices, probabilities):
      @functools.partial(
          jax.shard_map, mesh=mesh, in_specs=(spec, spec, spec), out_specs=P(), check_vma=reduce_axis_names is not None
      )
      def body(top_k_indices, probabilities, token_mask):
        return loss(router, top_k_indices, probabilities, token_mask=token_mask, reduce_axis_names=reduce_axis_names)

      return body(top_k_indices, probabilities, mask)

    # Reduced: the global mean, forward and backward, as the unsharded unpadded loss.
    np.testing.assert_allclose(sharded_loss(("batch",), pad_top_k, pad_probs), reference, rtol=1e-6)
    grad = jax.grad(lambda probabilities: loss(router, top_k, probabilities))(probs)
    pad_grad = jax.grad(functools.partial(sharded_loss, ("batch",), pad_top_k))(pad_probs)
    np.testing.assert_allclose(pad_grad[:rows], grad, rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(pad_grad[rows:], 0.0)

    # Unreduced (the pre-fix per-shard statistic, shard 0's value is returned): the leading shard holds only real rows,
    # so its mean differs from the global one in general; the all-padding trailing shard would give 0.
    def per_shard_loss(indices, probabilities, token_mask):
      return loss(router, indices, probabilities, token_mask=token_mask)[None]

    with jax.set_mesh(mesh):
      per_shard = jax.jit(
          jax.shard_map(
              per_shard_loss,
              mesh=mesh,
              in_specs=(spec, spec, spec),
              out_specs=P("batch"),
              check_vma=False,
          )
      )(pad_top_k, pad_probs, mask)
    np.testing.assert_array_equal(per_shard[num_devices // 2 :], 0.0)
    np.testing.assert_allclose(jnp.sum(per_shard) / (num_devices // 2), reference, rtol=1e-6)


@pytest.mark.tpu_only
class TailPaddingTpuTest(TailPaddingTest):
  """The tail-padding checks on the TPU's devices (unmarked tests are auto-marked cpu_only and skipped on TPU)."""


@pytest.mark.cpu_only
class TailPaddingCpuMeshTest(unittest.TestCase):
  """TailPaddingTest on a multi-device CPU mesh."""

  def test_tail_padding_on_cpu_mesh(self):
    """Runs TailPaddingTest in a subprocess with `_TAIL_PADDING_DEVICES` forced CPU devices."""
    env = os.environ.copy()
    xla_flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{xla_flags} --xla_force_host_platform_device_count={_TAIL_PADDING_DEVICES}".strip()
    env["JAX_PLATFORMS"] = "cpu"
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, __file__, "TailPaddingTest"], env=env, capture_output=True, text=True, check=False
    )
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    self.assertEqual(result.returncode, 0, output)
    self.assertIn(_TAIL_PADDING_MARKER, result.stdout, output)


class MaskedLossFunctionsTest(unittest.TestCase):
  """masked_sequence_mean and load_balance_loss / megatron_seq_aux_loss with a token_mask."""

  def setUp(self):
    super().setUp()
    self.router = _router_stub()
    self.logits = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 8), jnp.float32)
    self.probs = jax.nn.softmax(self.logits, axis=-1)
    _, self.top_k = jax.lax.top_k(self.logits, 2)

  def test_masked_sequence_mean(self):
    values = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 5), jnp.float32)
    np.testing.assert_array_equal(moe.masked_sequence_mean(values, jnp.ones((3, 8), bool)), jnp.mean(values, axis=1))
    mask = jnp.array([[True] * 8, [True] * 3 + [False] * 5, [False] * 8])
    got = moe.masked_sequence_mean(values, mask)
    np.testing.assert_allclose(got[0], jnp.mean(values[0], axis=0), rtol=1e-6)
    np.testing.assert_allclose(got[1], jnp.mean(values[1, :3], axis=0), rtol=1e-6)
    np.testing.assert_array_equal(got[2], 0.0)
    # Integer input: float32 like jnp.mean; bfloat16 input: bfloat16 like jnp.mean.
    self.assertEqual(moe.masked_sequence_mean(jnp.ones((2, 4, 3), jnp.int32), mask[:2, :4]).dtype, jnp.float32)
    self.assertEqual(moe.masked_sequence_mean(values.astype(jnp.bfloat16), mask).dtype, jnp.bfloat16)

  def test_load_balance_loss_ignores_padded_rows(self):
    loss = moe.RoutedMoE.load_balance_loss
    reference = loss(self.router, self.top_k, self.probs)
    # An all-True mask is the unmasked loss.
    np.testing.assert_array_equal(
        loss(self.router, self.top_k, self.probs, token_mask=jnp.ones((4, 16), bool)), reference
    )
    # Appending padded rows (ids -1, whatever probabilities) with a False mask leaves the loss unchanged (up to the
    # order of the batch reduction) ...
    pad_top_k = jnp.concatenate([self.top_k, jnp.full((2, 16, 2), -1, self.top_k.dtype)])
    pad_probs = jnp.concatenate([self.probs, jnp.full((2, 16, 8), 1.0 / 8, jnp.float32)])
    mask = jnp.arange(6)[:, None] < 4
    mask = jnp.broadcast_to(mask, (6, 16))
    np.testing.assert_allclose(loss(self.router, pad_top_k, pad_probs, token_mask=mask), reference, rtol=1e-6)
    # ... whereas without the mask the padding dilutes it by the row ratio.
    np.testing.assert_allclose(loss(self.router, pad_top_k, pad_probs), reference * 4 / 6, rtol=1e-6)
    # The precomputed density_prob path (masked mean) agrees with the probabilities path.
    np.testing.assert_allclose(
        loss(self.router, pad_top_k, None, density_prob=moe.masked_sequence_mean(pad_probs, mask), token_mask=mask),
        reference,
        rtol=1e-6,
    )
    # Gradients on the real rows are unchanged and the padded rows get none.
    grad = jax.grad(lambda probabilities: loss(self.router, self.top_k, probabilities))(self.probs)
    pad_grad = jax.grad(lambda probabilities: loss(self.router, pad_top_k, probabilities, token_mask=mask))(pad_probs)
    np.testing.assert_array_equal(pad_grad[:4], grad)
    np.testing.assert_array_equal(pad_grad[4:], 0.0)

  def test_partial_sequence_mask_is_the_loss_of_the_valid_prefix(self):
    """A mask over part of a sequence gives the loss of that sequence truncated to its valid tokens."""
    loss = moe.RoutedMoE.load_balance_loss
    mask = jnp.arange(16)[None, :] < 10
    mask = jnp.broadcast_to(mask, (4, 16))
    masked_top_k = jnp.where(mask[..., None], self.top_k, -1)
    np.testing.assert_allclose(
        loss(self.router, masked_top_k, self.probs, token_mask=mask),
        loss(self.router, self.top_k[:, :10], self.probs[:, :10]),
        rtol=1e-6,
    )

  def test_megatron_seq_aux_loss_ignores_padded_rows(self):
    seq_loss = moe.RoutedMoE.megatron_seq_aux_loss
    scores = jax.nn.sigmoid(self.logits)
    reference = seq_loss(self.router, scores)
    # Padded rows carry the scores of all-zero logits, sigmoid(0) = 0.5, as in the model.
    pad_scores = jnp.concatenate([scores, jnp.full((2, 16, 8), 0.5, jnp.float32)])
    mask = jnp.broadcast_to(jnp.arange(6)[:, None] < 4, (6, 16))
    np.testing.assert_allclose(seq_loss(self.router, pad_scores, token_mask=mask), reference, rtol=1e-6)
    grad = jax.grad(lambda router_scores: seq_loss(self.router, router_scores))(scores)
    pad_grad = jax.grad(lambda router_scores: seq_loss(self.router, router_scores, token_mask=mask))(pad_scores)
    np.testing.assert_array_equal(pad_grad[:4], grad)
    np.testing.assert_array_equal(pad_grad[4:], 0.0)


class RoutingTokenMaskTest(unittest.TestCase):
  """routing_token_mask only masks when a logical (unpadded) global batch size is set, in training, with segment ids."""

  def test_routing_token_mask(self):
    segment_ids = jnp.array([[1, 1, 1], [0, 0, 0]])
    # The constant-GBS modes (and any future mode padding the same way) set the logical global batch size.
    constant_padded = types.SimpleNamespace(
        elastic_global_batch_size_mode="constant_padded", logical_global_batch_size_to_train_on=16
    )
    constant_grad_accum = types.SimpleNamespace(
        elastic_global_batch_size_mode="constant_grad_accum", logical_global_batch_size_to_train_on=16
    )
    scaled = types.SimpleNamespace(elastic_global_batch_size_mode="scaled", logical_global_batch_size_to_train_on=None)
    train, prefill = ctypes.MODEL_MODE_TRAIN, ctypes.MODEL_MODE_PREFILL
    np.testing.assert_array_equal(
        moe.routing_token_mask(constant_padded, segment_ids, train), [[True, True, True], [False, False, False]]
    )
    np.testing.assert_array_equal(
        moe.routing_token_mask(constant_grad_accum, segment_ids, train), [[True, True, True], [False, False, False]]
    )
    self.assertIsNone(moe.routing_token_mask(constant_padded, None, train))
    self.assertIsNone(moe.routing_token_mask(constant_padded, segment_ids, prefill))
    self.assertIsNone(moe.routing_token_mask(scaled, segment_ids, train))
    self.assertIsNone(moe.routing_token_mask(types.SimpleNamespace(), segment_ids, train))


if __name__ == "__main__":
  # `python moe_padding_test.py [TestClass ...]`: TailPaddingCpuMeshTest reruns TailPaddingTest this way with
  # several forced CPU devices and looks for the marker.
  _result = unittest.main(exit=False).result
  if _result.wasSuccessful():
    print(_TAIL_PADDING_MARKER)
  sys.exit(0 if _result.wasSuccessful() else 1)
