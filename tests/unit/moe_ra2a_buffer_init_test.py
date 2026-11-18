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
"""Tests the uninitialized tail of the ragged_all_to_all receive buffer in RoutedMoE (ring of experts off).

The dispatch receive buffer of `ra2a_and_route` (`jax.lax.empty`) holds fewer than `buffer_size` rows, for every
ragged_buffer_factor. Its tail rows stay uninitialized; `local_permute` must zero them in the gmm input, with
either sort, so they cannot poison reductions over all rows (e.g. the fp8 absmax of the quantized gmm backward).
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest
import qwix
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import nd_dense_init, variable_to_logically_partitioned
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _nan_empty(shape, dtype, **kwargs):
  """Stand-in for `jax.lax.empty` that makes uninitialized memory visible as NaN."""
  del kwargs
  return jnp.full(shape, jnp.nan, dtype=dtype)


class Ra2aBufferInitTest(parameterized.TestCase):
  """With `jax.lax.empty` returning NaN, the fp8 RoutedMoE forward and backward stay finite."""

  def _build(self, ragged_buffer_factor, force_dropless, gmm_v2, ragged_sort):
    """Builds an fp8_full (Qwix, absmax) RoutedMoE with EP=4, ring of experts off and the given buffer settings."""
    if gmm_v2:
      # tokamax gmm_v2 (in-kernel lhs quantization), as in moe_quantized_a2a_test.
      gmm = {
          "base_mlp_dim": 512,
          "base_moe_mlp_dim": 512,
          "weight_dtype": "float32",
          "per_device_batch_size": 1,
          "megablox": False,
          "use_tokamax_gmm": True,
          "use_gmm_v2": True,
          "check_vma": False,
          **{f"{w}_tile_{p}_batch_seq": 128 for w in ("wi", "wo") for p in ("fwd", "dlhs", "drhs")},
          **dict.fromkeys(("wi_tile_dlhs_embed_dim", "wo_tile_fwd_embed_dim", "wo_tile_dlhs_mlp_dim"), 256),
      }
    else:
      gmm = {
          "base_mlp_dim": 256,
          "base_moe_mlp_dim": 256,
          "per_device_batch_size": 2,
          "megablox": True,
          # gmm tiles must not exceed the small mlp dim / per-shard buffer.
          **{f"{w}_tile_{p}_mlp_dim": 256 for w in ("wi", "wo") for p in ("fwd", "dlhs", "drhs")},
          **{f"{w}_tile_{p}_batch_seq": 128 for w in ("wi", "wo") for p in ("fwd", "dlhs", "drhs")},
      }
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"moe_ra2a_buffer_init_{ragged_buffer_factor}_{force_dropless}_{gmm_v2}_{ragged_sort}",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        # emb dim >= 2048 (multiple of 1024) for the SparseCore ragged sort kernels.
        base_emb_dim=2048,
        dtype="bfloat16",
        sparse_matmul=True,
        max_target_length=64,
        float32_gate_logits=True,
        ici_expert_parallelism=4,
        use_ring_of_experts=False,
        use_ragged_sort=ragged_sort,
        ragged_buffer_factor=ragged_buffer_factor,
        quantization="fp8_full",
        use_qwix_quantization=True,
        quantize_router_proj=False,
        weight_quantization_calibration_method="absmax",
        act_quantization_calibration_method="absmax",
        bwd_quantization_calibration_method="absmax",
        **gmm,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = nnx_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        metadata_fn=variable_to_logically_partitioned,
        abstract_init=False,
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        force_dropless=force_dropless,
    )
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        weight_calibration_method=cfg.weight_quantization_calibration_method,
        act_calibration_method=cfg.act_quantization_calibration_method,
        bwd_calibration_method=cfg.bwd_quantization_calibration_method,
        op_names=("gmm", "ragged_dot"),
    )
    model = qwix.quantize_model(model, qwix.QtProvider([rule]))
    return cfg, mesh, model

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      # megablox, XLA sort. Worst-case (EP x balanced) buffer of the default dropless path.
      ("dropless_rbf_neg1", -1.0, False, False, False),
      # force_dropless overrides a tiny capped buffer with the worst-case buffer and routes with rbf=-1.
      ("force_dropless", 0.125, True, False, False),
      # A capped buffer is under-full too unless the routing overflows it.
      ("capped_rbf_2", 2.0, False, False, False),
      # tokamax gmm_v2: XLA sort, then the SparseCore ragged sort (which only gathers the valid prefix).
      ("gmm_v2_rbf_neg1", -1.0, False, True, False),
      ("gmm_v2_ragged_sort_rbf_neg1", -1.0, False, True, True),
  )
  def test_nan_empty_fp8_finite(self, ragged_buffer_factor, force_dropless, gmm_v2, ragged_sort):
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU (fp8 gmm and ragged_all_to_all).")
    cfg, mesh, model = self._build(ragged_buffer_factor, force_dropless, gmm_v2, ragged_sort)
    rng_model, rng_x = jax.random.split(jax.random.PRNGKey(2345))
    x = jax.random.uniform(
        rng_x,
        (int(cfg.per_device_batch_size) * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    # Records (received rows, sorted rows, routed row count) of every local_permute call on a received buffer.
    captured = []
    orig_local_permute = moe.RoutedMoE.local_permute

    def _spy_local_permute(*args, **kwargs):
      res = orig_local_permute(*args, **kwargs)
      if not kwargs.get("is_offset", False):
        sorted_inputs, _, local_group_size, _ = res
        jax.debug.callback(
            lambda r, s, n: captured.append((np.asarray(r, np.float32), np.asarray(s, np.float32), int(np.sum(n)))),
            args[0],
            sorted_inputs,
            local_group_size,
        )
      return res

    def loss_fn(params, inp):
      out, lb_loss, _ = model.apply({"params": params}, inp)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    with (
        mock.patch("jax.lax.empty", side_effect=_nan_empty),
        mock.patch.object(moe.RoutedMoE, "local_permute", side_effect=_spy_local_permute),
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg.logical_axis_rules),
    ):
      variables = model.init({"params": rng_model, "dropout": rng_model}, x)
      captured.clear()
      (loss, out), (grads, x_grad) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(
          variables["params"], x
      )
      jax.block_until_ready((loss, out, grads, x_grad))

    self.assertTrue(np.isfinite(np.asarray(loss)), f"loss={loss}")
    self.assertTrue(np.all(np.isfinite(np.asarray(out, np.float32))), "Non-finite MoE output.")
    self.assertTrue(np.all(np.isfinite(np.asarray(x_grad, np.float32))), "Non-finite input gradient.")
    for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
      self.assertTrue(
          np.all(np.isfinite(np.asarray(g, np.float32))), f"Non-finite gradient at {jax.tree_util.keystr(path)}."
      )
    for name in ("wi_0", "wi_1", "wo"):
      g_max = max(float(np.max(np.abs(np.asarray(g, np.float32)))) for g in jax.tree_util.tree_leaves(grads[name]))
      self.assertGreater(g_max, 0.0, f"{name} grad is all zero.")

    # The test only means something if some shard's receive buffer was under-full.
    underfull = [(r, s, n) for r, s, n in captured if n < s.shape[0]]
    self.assertNotEmpty(underfull, "No under-full receive buffer; the test does not exercise the tail rows.")
    for r, s, n in underfull:
      # No memset: the receive buffer tail is still `jax.lax.empty` (NaN here); local_permute zeroes it.
      self.assertTrue(np.all(np.isnan(r[n:])), "Tail rows of the received buffer were initialized.")
      np.testing.assert_array_equal(s[n:], 0.0, err_msg="Tail rows of the gmm input are not zero.")


if __name__ == "__main__":
  absltest.main()
