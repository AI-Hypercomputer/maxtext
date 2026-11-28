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

"""Tests num_moe_token_chunks > 1 on the ragged all-to-all expert-parallel path (use_ring_of_experts=False).

Routing is per token, so splitting the sequence into token chunks must not change the MoE output, the gradients
or the full-batch expert-bias update.
"""

import re

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _config(run_name, **overrides):
  """Tiny float32 DeepSeek-V3 config on the ragged all-to-all path with expert parallelism 2."""
  kwargs = {
      "run_name": run_name,
      "enable_checkpointing": False,
      "model_name": "deepseek3-tiny",
      "override_model_config": True,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      "megablox": True,
      "sparse_matmul": True,
      # Keep global batch at 8 on 8-device TPUs (same token count as 4-device v6e-4).
      "per_device_batch_size": 1 if jax.device_count() >= 8 else 2,
      "max_target_length": 16,
      "load_balance_loss_weight": 0.01,
      "moe_use_megatron_seq_aux_loss": True,
      "routed_bias_update_rate": 1e-3,
      "ici_expert_parallelism": 2,
      "use_ring_of_experts": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def _build(cfg, mesh):
  return moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed_moe", None),
      intermediate_dim=cfg.moe_mlp_dim,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      rngs=nnx.Rngs(params=0),
  )


def _loss_and_grad_fn(graphdef, rest, inputs):
  def loss_fn(params):
    out, lb_loss, bias_updates = nnx.merge(graphdef, params, rest)(inputs)
    return jnp.mean(out**2) + lb_loss, (out, lb_loss, bias_updates)

  return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


def _make_inputs(cfg, seq_len=None):
  """Builds deterministic spread-magnitude inputs for RoutedMoE chunk parity tests."""
  length = cfg.max_target_length if seq_len is None else seq_len
  inputs = jax.random.normal(
      jax.random.PRNGKey(7),
      (cfg.per_device_batch_size * jax.device_count(), length, cfg.base_emb_dim),
  )
  # Spread token magnitudes so a token routed to or combined at the wrong position changes the output.
  return inputs * jnp.exp(jax.random.normal(jax.random.PRNGKey(8), inputs.shape[:2] + (1,)))


def _run_one(cfg, *, mesh=None, params=None, inputs=None, compiled_text=False, seq_len=None):
  """Runs (or compiles to HLO) RoutedMoE forward and backward for one config."""
  if mesh is None:
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  if inputs is None:
    inputs = _make_inputs(cfg, seq_len=seq_len)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = _build(cfg, mesh)
    graphdef, model_params, rest = nnx.split(model, nnx.Param, ...)
    if params is None:
      params = model_params
    fn = _loss_and_grad_fn(graphdef, rest, inputs)
    if compiled_text:
      return fn.lower(params).compile().as_text(), params, inputs, mesh
    (_, aux), grads = fn(params)
    return (aux, grads), params, inputs, mesh


def _run(cfg_ref, cfg, compiled_text=False, seq_len=None):
  """Runs RoutedMoE forward and backward for cfg_ref and cfg on shared parameters and inputs."""
  res_ref, params, inputs, mesh = _run_one(cfg_ref, compiled_text=compiled_text, seq_len=seq_len)
  res_tgt, _, _, _ = _run_one(cfg, mesh=mesh, params=params, inputs=inputs, compiled_text=compiled_text, seq_len=seq_len)
  return res_ref, res_tgt


def _num_ragged_all_to_all(hlo_text):
  return len(re.findall(r"ragged-all-to-all(?:-start)?\(", hlo_text))


class Ra2aTokenChunksConfigTest(absltest.TestCase):
  """Config validation for token chunks without ring of experts."""

  def test_token_chunks_validate_without_ring_of_experts(self):
    cfg = _config("ra2a_chunks_cfg", num_moe_token_chunks=2)
    self.assertEqual(cfg.num_moe_token_chunks, 2)
    self.assertFalse(cfg.use_ring_of_experts)

  def test_token_chunks_with_layer_fallback_validate_without_ring_of_experts(self):
    cfg = _config(
        "ra2a_chunks_layer_cfg",
        num_moe_token_chunks=2,
        use_ragged_sort=True,
        ragged_buffer_factor=1.0,
        moe_dropless_fallback="layer",
    )
    self.assertEqual(cfg.moe_dropless_fallback, "layer")

  def test_token_chunks_must_divide_sequence(self):
    with self.assertRaisesRegex(ValueError, "must evenly divide max_target_length=16"):
      _config("ra2a_chunks_bad_div", num_moe_token_chunks=3)

  def test_token_chunks_must_divide_per_shard_sequence(self):
    with self.assertRaisesRegex(ValueError, "must evenly divide the per-shard sequence length"):
      _config(
          "ra2a_chunks_bad_shard_div",
          max_target_length=12,
          ici_context_parallelism=2,
          ici_expert_parallelism=1,
          compile_topology="tpu7x-8",
          compile_topology_num_slices=1,
          num_moe_token_chunks=4,
      )


@pytest.mark.tpu_only
class Ra2aTokenChunksParityTest(parameterized.TestCase):
  """Ragged all-to-all RoutedMoE: num_moe_token_chunks=2 matches num_moe_token_chunks=1."""

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, "_ref_cache"):
      delattr(cls, "_ref_cache")
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      # XLA:CPU does not implement ragged-all-to-all.
      self.skipTest("The ragged all-to-all path needs a TPU.")
    if jax.device_count() < 2 or jax.device_count() % 2:
      self.skipTest("Needs an even number (>= 2) of devices for expert parallelism 2.")

  def _assert_chunk_parity(self, seq_len=None, **overrides):
    """Asserts output, aux loss, bias update and gradients match between 1 and 2 token chunks."""
    name = "_".join(f"{k}{v}" for k, v in sorted(overrides.items()))
    # moe_chunk_barrier is only active when num_moe_token_chunks > 1, so strip it from the 1-chunk reference key.
    ref_overrides = {k: v for k, v in overrides.items() if k != "moe_chunk_barrier"}
    ref_key = (jax.device_count(), seq_len, tuple(sorted(ref_overrides.items())))
    ref_cache = getattr(self.__class__, "_ref_cache", {})
    if ref_key not in ref_cache:
      cfg1 = _config(f"ra2a_chunks1_{name}", num_moe_token_chunks=1, **ref_overrides)
      res1, params, inputs, mesh = _run_one(cfg1, seq_len=seq_len)
      ref_cache[ref_key] = (jax.device_get(res1), params, inputs, mesh)
      setattr(self.__class__, "_ref_cache", ref_cache)
    ((out1, lb1, bias1), grads1), params, inputs, mesh = ref_cache[ref_key]
    cfg2 = _config(f"ra2a_chunks2_{name}", num_moe_token_chunks=2, **overrides)
    ((out2, lb2, bias2), grads2), _, _, _ = _run_one(cfg2, mesh=mesh, params=params, inputs=inputs, seq_len=seq_len)
    self.assertTrue(np.all(np.isfinite(out1)))
    np.testing.assert_allclose(out2, out1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lb2, lb1, rtol=1e-6)
    np.testing.assert_array_equal(bias2, bias1)
    for (path, g2), g1 in zip(
        jax.tree_util.tree_leaves_with_path(grads2), jax.tree_util.tree_leaves(grads1), strict=True
    ):
      np.testing.assert_allclose(g2, g1, rtol=1e-4, atol=1e-6, err_msg=jax.tree_util.keystr(path))

  @parameterized.named_parameters(
      ("dropless", {}),
      # ragged_buffer_factor = EP is the worst-case buffer, so nothing is dropped with either chunking.
      ("capped_buffer", {"ragged_buffer_factor": 2.0}),
      ("chunk_barrier", {"moe_chunk_barrier": True}),
  )
  def test_matches_unchunked(self, overrides):
    self._assert_chunk_parity(**overrides)

  @parameterized.named_parameters(
      ("dropless", {}),
      ("capped_buffer", {"ragged_buffer_factor": 2.0}),
      # The buffer is too small, so a chunk (or the whole layer) takes the dropless branch; both are dropless.
      ("layer_fallback", {"ragged_buffer_factor": 1.0, "moe_dropless_fallback": "layer"}),
  )
  def test_ragged_sort_matches_unchunked(self, overrides):
    # SparseCore ragged_gather_reduce_v2 requires hidden size >= 2048 on v6e (num_row_partitions <= 8) to fit
    # its row partitions in the SIMD lanes; keep 2048 on all TPUs so the fan-in weight scale stays within atol=1e-6.
    self._assert_chunk_parity(use_ragged_sort=True, base_emb_dim=2048, **overrides)

  def test_accumulate_chunk_wgrad_matches_unchunked(self):
    self._assert_chunk_parity(use_tokamax_gmm=True, use_gmm_v2=True, moe_accumulate_chunk_wgrad=True)

  def test_single_token_decode_clamps_to_one_chunk(self):
    """Autoregressive decode (seq_len=1) clamps effective chunks to 1 instead of producing empty chunks."""
    emb_dim = 1024 if jax.devices()[0].device_kind == "TPU7x" else 2048
    self._assert_chunk_parity(
        seq_len=1,
        use_ragged_sort=True,
        base_emb_dim=emb_dim,
        ragged_buffer_factor=1.0,
        moe_dropless_fallback="layer",
    )

  def test_unaligned_per_shard_sequence_rejected(self):
    cfg4 = _config("ra2a_unaligned4", num_moe_token_chunks=4)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg4), cfg4.mesh_axes)
    inputs = jax.ShapeDtypeStruct((cfg4.per_device_batch_size * jax.device_count(), 6, cfg4.base_emb_dim), jnp.float32)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg4.logical_axis_rules):
      model = nnx.eval_shape(lambda: _build(cfg4, mesh))
      graphdef, params, rest = nnx.split(model, nnx.Param, ...)
      with self.assertRaisesRegex(
          ValueError, r"Per-shard sequence length \(6\) must be divisible by num_moe_token_chunks \(4\)"
      ):
        jax.eval_shape(lambda p, r, x: nnx.merge(graphdef, p, r)(x), params, rest, inputs)

  def test_each_chunk_runs_its_own_ragged_all_to_all(self):
    """Two chunks compile to twice the ragged all-to-alls (dispatch and combine, forward and backward)."""
    cfg1 = _config("ra2a_hlo_chunks1", num_moe_token_chunks=1)
    cfg2 = _config("ra2a_hlo_chunks2", num_moe_token_chunks=2)
    hlo1, hlo2 = _run(cfg1, cfg2, compiled_text=True)
    n1, n2 = _num_ragged_all_to_all(hlo1), _num_ragged_all_to_all(hlo2)
    self.assertGreater(n1, 0)
    self.assertEqual(n2, 2 * n1)


if __name__ == "__main__":
  absltest.main()
