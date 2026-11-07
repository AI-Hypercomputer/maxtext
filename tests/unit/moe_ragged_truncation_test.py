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

"""Unit tests for RoutedMoE ragged_all_to_all truncation (Bug 1A & Bug 1B)."""

import os
import unittest
from unittest import mock

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax._src.interpreters import mlir
from jax._src.lax import parallel as lax_parallel
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.kernels.megablox import ops as mblx
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
import numpy as np
import qwix
from tests.utils import linen_wrappers
from tests.utils.test_helpers import get_test_config_path


def _cpu_ragged_all_to_all(  # pylint: disable=too-many-positional-arguments
    operand,
    output,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    *,
    axis_name,
    axis_index_groups=None,
):
  """CPU collective lowering for `ragged_all_to_all_p` preserving `output` in unwritten slots."""
  del send_sizes
  all_operands = jax.lax.all_gather(operand, axis_name=axis_name, axis_index_groups=axis_index_groups)
  output_offsets_ = jax.lax.all_to_all(
      output_offsets,
      axis_name=axis_name,
      split_axis=0,
      concat_axis=0,
      tiled=True,
      axis_index_groups=axis_index_groups,
  )
  input_offsets_ = jax.lax.all_to_all(
      input_offsets,
      axis_name=axis_name,
      split_axis=0,
      concat_axis=0,
      tiled=True,
      axis_index_groups=axis_index_groups,
  )
  num_slices = int(recv_sizes.shape[0])
  num_devices = int(all_operands.shape[0])
  slices_per_device = num_slices // num_devices
  rows = jnp.arange(output.shape[0], dtype=output_offsets_.dtype)
  result = output
  for i in range(num_slices):
    src_idx = i // slices_per_device
    out_start = output_offsets_[i]
    size = recv_sizes[i]
    in_start = input_offsets_[i]
    in_slice = (rows >= out_start) & (rows < out_start + size)
    src_rows = jnp.clip(in_start + (rows - out_start), 0, operand.shape[0] - 1)
    gathered = all_operands[src_idx, src_rows]
    mask = jnp.expand_dims(in_slice, tuple(range(1, output.ndim)))
    result = jnp.where(mask, gathered, result)
  return result


mlir.register_lowering(
    lax_parallel.ragged_all_to_all_p,
    mlir.lower_fun(_cpu_ragged_all_to_all, multiple_results=False),
    platform="cpu",
)


def _nan_empty(shape, dtype=jnp.float32, **kwargs):
  """Stand-in for `jax.lax.empty` filling uninitialized buffers with NaN."""
  del kwargs
  return jnp.full(shape, jnp.nan, dtype=dtype)


class MoeRaggedTruncationTest(unittest.TestCase):
  """Verifies `RoutedMoE` token-dropping `ragged_all_to_all` zero-init (Bug 1A) and `local_permute` (Bug 1B)."""

  def _build_cfg_and_model(self, ragged_buffer_factor: float):
    """Builds a 4-EP-shard, 8-expert (`local_expert_size=2`) `RoutedMoE` model on CPU."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"moe_ragged_truncation_rbf_{ragged_buffer_factor}",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        base_emb_dim=8,
        base_mlp_dim=16,
        base_moe_mlp_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        float32_gate_logits=True,
        sparse_matmul=True,
        megablox=False,
        use_tokamax_gmm=False,
        use_gmm_v2=False,
        per_device_batch_size=1.0,
        max_target_length=4,
        ici_fsdp_parallelism=1,
        ici_expert_parallelism=4,
        use_ring_of_experts=False,
        use_ragged_sort=False,
        ragged_buffer_factor=float(ragged_buffer_factor),
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = linen_wrappers.to_linen(
        moe.RoutedMoE,
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
    )
    return cfg, mesh, model

  def _construct_overflow_routing_inputs_and_params(self, cfg, mesh, model):
    """Constructs inputs/params where shard 0 (`E0, E1`) overflows `buffer_size=8` while `E0<8` and `E1<8`.

    Geometry (`num_ep=4`, `num_experts=8`, `local_expert_size=2`, `seq_len=4`, `top_k=2`):
    With `ragged_buffer_factor=1.0`, `local_batch = 4 * 2 = 8`, so `buffer_size = 8`.
    Pre-truncation totals sent to shard 0 (`E0, E1`):
      - `s=0`: `t=0->(E0,E1)`, `t=1->(E0,E1)`, `t=2->(E0,E2)`, `t=3->(E4,E5)` (`E0=3, E1=2`, sum=5)
      - `s=1`: `t=0->(E0,E1)`, `t=1->(E0,E1)`, `t=2->(E2,E3)`, `t=3->(E6,E7)` (`E0=2, E1=2`, sum=4)
      - `s=2`: `t=0->(E0,E1)`, `t=1->(E1,E2)`, `t=2->(E4,E5)`, `t=3->(E6,E7)` (`E0=1, E1=2`, sum=3)
      - `s=3`: `t=0->(E2,E3)`, `t=1->(E4,E5)`, `t=2->(E6,E7)`, `t=3->(E4,E6)` (`E0=0, E1=0`, sum=0)
    Thus on shard 0: `E0 = 6 < 8` and `E1 = 6 < 8` individually, while `E0 + E1 = 12 > 8`,
    and shards 1, 2, 3 receive `6, 7, 7 < 8` tokens respectively (no overflow, testing `valid_count < 8`).
    """
    basis = np.eye(8, dtype=np.float32)
    pair_for_basis = {
        0: (0, 1),
        1: (0, 2),
        2: (1, 2),
        3: (2, 3),
        4: (4, 5),
        5: (6, 7),
        6: (4, 6),
        7: (3, 5),
    }
    gate_kernel = np.full((8, 8), -50.0, dtype=np.float32)
    for b_idx, (e_first, e_second) in pair_for_basis.items():
      gate_kernel[b_idx, e_first] = 10.0
      gate_kernel[b_idx, e_second] = 9.0

    # Assign basis indices for each (shard s, token t), scaled by unique positive factors.
    token_basis_idx = [
        [0, 0, 1, 4],  # s=0: shard 0 gets E0=3, E1=2 (5 tokens)
        [0, 0, 3, 5],  # s=1: shard 0 gets E0=2, E1=2 (4 tokens; 2nd E1 dropped at cap 8)
        [0, 2, 4, 5],  # s=2: (E0,E1)[both dropped], (E1,E2)[E1 dropped, E2 kept]
        [3, 4, 5, 6],  # s=3: shard 0 gets E0=0, E1=0; shards 1,2,3 get 6,7,7 < 8
    ]
    x_np = np.zeros((4, 4, 8), dtype=np.float32)
    for s in range(4):
      for t in range(4):
        scale = float(1.0 + 0.25 * s + 0.05 * t)
        x_np[s, t, :] = scale * basis[token_basis_idx[s][t]]
    x = jnp.asarray(x_np, dtype=jnp.float32)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x)
    params = jax.tree_util.tree_map(np.asarray, nn.unbox(variables["params"]))
    params["gate"]["kernel"] = gate_kernel
    rng = np.random.default_rng(12345)
    params["wi_0"] = rng.normal(0.0, 0.25, size=params["wi_0"].shape).astype(np.float32)
    params["wi_1"] = rng.normal(0.0, 0.25, size=params["wi_1"].shape).astype(np.float32)
    params["wo"] = rng.normal(0.0, 0.25, size=params["wo"].shape).astype(np.float32)
    params_jax = jax.tree_util.tree_map(jnp.asarray, params)
    return x, params_jax, token_basis_idx, pair_for_basis

  def test_1_local_permute_multi_expert_partial_truncation_and_tail_sentinel(self):
    """Directly tests `RoutedMoE.local_permute` when each local expert < buffer_size while sum > buffer_size."""
    # 4 EP shards, local_expert_size = 2, buffer_size = 6.
    # Shard 0 column counts: E0 = 2+2+1+0 = 5 < 6, E1 = 1+2+2+0 = 5 < 6, total = 10 > 6.
    # Flat sizes on shard 0: [2, 1, 2, 2, 1, 2, 0, 0].
    # Prefix-clamped at buffer_size=6: [2, 1, 2, 1, 0, 0, 0, 0].
    global_group_sizes = jnp.array(
        [
            [2, 1, 1, 1, 1, 1, 0, 0],  # s=0
            [2, 2, 1, 1, 0, 0, 1, 1],  # s=1
            [1, 2, 0, 1, 1, 1, 1, 1],  # s=2
            [0, 0, 0, 0, 2, 1, 1, 1],  # s=3
        ],
        dtype=jnp.int32,
    )
    shard0_inputs = jnp.array(
        [
            [10.0, 0.0, 0.0, 0.0],
            [11.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0, 0.0],
            [12.0, 0.0, 0.0, 0.0],
            [13.0, 0.0, 0.0, 0.0],
            [21.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids = moe.RoutedMoE.local_permute(
        shard0_inputs,
        global_group_sizes,
        local_expert_size=2,
        shard_index=0,
        is_offset=False,
        ragged_buffer_factor=0.75,
    )
    np.testing.assert_array_equal(np.asarray(local_group_size), np.array([4, 2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(sorted_indices), np.array([0, 1, 3, 4, 2, 5], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(sorted_experts_ids), np.array([0, 0, 0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(sorted_inputs[:, 0]),
        np.array([10.0, 11.0, 12.0, 13.0, 20.0, 21.0], dtype=np.float32),
    )

    # Check shard 1 where valid_count = 5 < buffer_size = 6 (tail slot 5 gets sentinel 2).
    shard1_inputs = jnp.array(
        [
            [100.0, 0.0],  # s=0 E2 (local 0)
            [200.0, 0.0],  # s=0 E3 (local 1)
            [101.0, 0.0],  # s=1 E2 (local 0)
            [201.0, 0.0],  # s=1 E3 (local 1)
            [202.0, 0.0],  # s=2 E3 (local 1)
            [999.0, 0.0],  # unpopulated tail slot 5
        ],
        dtype=jnp.float32,
    )
    sorted_in_1, sorted_idx_1, local_gs_1, sorted_exp_1 = moe.RoutedMoE.local_permute(
        shard1_inputs,
        global_group_sizes,
        local_expert_size=2,
        shard_index=1,
        is_offset=False,
        ragged_buffer_factor=0.75,
    )
    np.testing.assert_array_equal(np.asarray(local_gs_1), np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(sorted_exp_1), np.array([0, 0, 1, 1, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(sorted_idx_1), np.array([0, 2, 1, 3, 4, 5], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(sorted_in_1[:, 0]),
        np.array([100.0, 101.0, 200.0, 201.0, 202.0, 999.0], dtype=np.float32),
    )

  def test_2_and_3_dirty_empty_nan_grad_and_analytical_prefix_truncated_parity(self):
    """Tests finite forward+backward grads under NaN `jax.lax.empty` + analytical parity."""
    cfg, mesh, model = self._build_cfg_and_model(ragged_buffer_factor=1.0)
    x, params, token_basis_idx, pair_for_basis = self._construct_overflow_routing_inputs_and_params(cfg, mesh, model)

    captured_dispatches = []
    orig_local_permute = moe.RoutedMoE.local_permute

    def _spy_local_permute(*args, **kwargs):
      res = orig_local_permute(*args, **kwargs)
      sorted_inputs, _, local_group_size, _ = res
      jax.debug.callback(
          lambda s_in, l_gs: captured_dispatches.append((np.array(s_in), np.array(l_gs))),
          sorted_inputs,
          local_group_size,
      )
      return res

    def loss_and_out(p, inp):
      out, lb_loss, _ = model.apply({"params": p}, inp)
      target = jnp.arange(1, out.size + 1, dtype=jnp.float32).reshape(out.shape) * 0.01
      loss = jnp.sum((out - target) * target) + (0.0 if lb_loss is None else jnp.sum(lb_loss))
      return loss, (out, lb_loss)

    with (
        mock.patch("jax.lax.empty", side_effect=_nan_empty),
        mock.patch.object(moe.RoutedMoE, "local_permute", side_effect=_spy_local_permute),
    ):
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        (loss, (out, lb_loss)), (param_grads, x_grad) = jax.jit(
            jax.value_and_grad(loss_and_out, argnums=(0, 1), has_aux=True)
        )(params, x)
        jax.block_until_ready((loss, out, param_grads, x_grad))

    # Verify dispatch ragged_all_to_all zero-initializes unpopulated tail rows [valid_count:buffer_size)
    # when ragged_buffer_factor > 0.0, preventing NaN propagation into Megablox custom-VJP drhs gradients.
    self.assertEqual(len(captured_dispatches), 4)
    underfull_shards = 0
    gmm_rule = qwix.QtRule(
        weight_qtype=jnp.int8,
        weight_calibration_method="absmax",
        act_qtype=jnp.int8,
        act_calibration_method="absmax",
        bwd_qtype=jnp.int8,
        bwd_calibration_method="absmax",
    )
    gmm_tiling = (8, 8, 8, 8, 8, 8, 8, 8, 8)
    rhs_probe = jnp.ones((2, 8, 8), dtype=jnp.float32)
    for s_in, l_gs in captured_dispatches:
      valid_n = int(l_gs.sum())
      self.assertTrue(
          bool(np.all(np.isfinite(s_in))),
          f"Dispatched sorted_inputs contains NaN/Inf in buffer (valid_n={valid_n})",
      )
      if valid_n < s_in.shape[0]:
        underfull_shards += 1
        np.testing.assert_array_equal(
            s_in[valid_n:],
            np.zeros_like(s_in[valid_n:]),
            err_msg="Unpopulated dispatch tail rows must be zero-initialized when ragged_buffer_factor > 0.0.",
        )

        def _gmm_probe_loss(lhs_arr, rhs_arr, gs_arr=jnp.asarray(l_gs, dtype=jnp.int32), n_valid=valid_n):
          gmm_out = mblx.gmm(
              lhs_arr,
              rhs_arr,
              gs_arr,
              jnp.float32,
              gmm_tiling,
              interpret=True,
              use_qwix_quantization=True,
              qwix_rule=gmm_rule,
          )
          return jnp.sum(gmm_out[:n_valid])

        _, (dl_probe, dr_probe) = jax.value_and_grad(_gmm_probe_loss, argnums=(0, 1))(
            jnp.asarray(s_in, dtype=jnp.float32),
            rhs_probe,
        )
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(dl_probe[:valid_n]))),
            "Megablox custom-VJP dlhs gradient contains NaN/Inf on under-full dispatch buffer",
        )
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(dr_probe))),
            "Megablox custom-VJP drhs weight gradient contains NaN/Inf on under-full dispatch buffer",
        )
    self.assertEqual(underfull_shards, 3)

    self.assertTrue(bool(jnp.isfinite(loss)), f"Expected finite scalar loss, got {loss}")
    self.assertTrue(
        bool(jnp.all(jnp.isfinite(out))),
        "Forward output contains NaN/Inf under dirty jax.lax.empty",
    )
    if lb_loss is not None:
      self.assertTrue(bool(jnp.all(jnp.isfinite(lb_loss))), "lb_loss contains NaN/Inf")
    self.assertTrue(
        bool(jnp.all(jnp.isfinite(x_grad))),
        "Input cotangent x_grad contains NaN/Inf",
    )
    for leaf_name in ("gate", "wi_0", "wi_1", "wo"):
      self.assertIn(leaf_name, param_grads)
      leaf_arrays = jax.tree_util.tree_leaves(param_grads[leaf_name])
      self.assertTrue(leaf_arrays, f"Empty leaf gradient for {leaf_name}")
      for arr in leaf_arrays:
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(arr))),
            f"Parameter gradient leaf {leaf_name} contains NaN/Inf under dirty jax.lax.empty",
        )
        self.assertGreater(
            float(jnp.linalg.norm(arr)),
            0.0,
            f"Parameter gradient leaf {leaf_name} unexpectedly all zero",
        )

    out_np = np.asarray(out, dtype=np.float32)
    np.testing.assert_array_equal(
        out_np[2, 0, :],
        np.zeros((8,), dtype=np.float32),
        err_msg="Fully truncated token (s=2, t=0) must produce exact 0.0 in combine output.",
    )

    dropped_assignments = {
        (1, 1, 1),  # s=1, t=1, expert E1 (9th token sent to shard 0 -> truncated at 8)
        (2, 0, 0),  # s=2, t=0, expert E0 (truncated)
        (2, 0, 1),  # s=2, t=0, expert E1 (truncated)
        (2, 1, 1),  # s=2, t=1, expert E1 (truncated; E2 on shard 1 survives!)
    }
    x_np = np.asarray(x, dtype=np.float32)
    gate_np = np.asarray(params["gate"]["kernel"], dtype=np.float32)
    wi0_np = np.asarray(params["wi_0"], dtype=np.float32)
    wi1_np = np.asarray(params["wi_1"], dtype=np.float32)
    wo_np = np.asarray(params["wo"], dtype=np.float32)

    def _silu(v):
      return v / (1.0 + np.exp(-v))

    ref_out = np.zeros_like(out_np)
    for s in range(4):
      for t in range(4):
        vec = x_np[s, t]
        logits = vec @ gate_np
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        e_first, e_second = pair_for_basis[token_basis_idx[s][t]]
        for e_id in (e_first, e_second):
          if (s, t, e_id) in dropped_assignments:
            continue
          w = probs[e_id]
          h0 = _silu(vec @ wi0_np[e_id])
          h1 = vec @ wi1_np[e_id]
          expert_y = (h0 * h1) @ wo_np[e_id]
          ref_out[s, t] += w * expert_y

    self.assertGreater(float(np.linalg.norm(out_np[2, 1])), 0.0)
    ref_inf = max(1.0, float(np.max(np.abs(ref_out))))
    np.testing.assert_allclose(
        out_np,
        ref_out,
        rtol=1e-5,
        atol=1e-5 * ref_inf,
        err_msg="RoutedMoE prefix-truncated output diverged from analytical reference.",
    )
    frob_rel = float(np.linalg.norm(out_np - ref_out) / np.linalg.norm(ref_out))
    self.assertLessEqual(frob_rel, 1e-5, f"Expected frob_rel <= 1e-5, got {frob_rel:.3e}")

  def test_4_exact_equality_with_dropless_when_no_shard_overflows_under_nan_empty(self):
    """When `ragged_buffer_factor=4.0` (no overflow), matches dropless (`-1.0`) with `rtol=0, atol=0`."""
    cfg_drop, mesh_drop, model_drop = self._build_cfg_and_model(ragged_buffer_factor=4.0)
    cfg_ref, mesh_ref, model_ref = self._build_cfg_and_model(ragged_buffer_factor=-1.0)
    x, params, _, _ = self._construct_overflow_routing_inputs_and_params(cfg_drop, mesh_drop, model_drop)

    def loss_drop(p, inp):
      out, lb_loss, _ = model_drop.apply({"params": p}, inp)
      target = jnp.arange(1, out.size + 1, dtype=jnp.float32).reshape(out.shape) * 0.01
      loss = jnp.sum((out - target) * target) + (0.0 if lb_loss is None else jnp.sum(lb_loss))
      return loss, out

    def loss_ref(p, inp):
      out, lb_loss, _ = model_ref.apply({"params": p}, inp)
      target = jnp.arange(1, out.size + 1, dtype=jnp.float32).reshape(out.shape) * 0.01
      loss = jnp.sum((out - target) * target) + (0.0 if lb_loss is None else jnp.sum(lb_loss))
      return loss, out

    with mock.patch("jax.lax.empty", side_effect=_nan_empty):
      with jax.set_mesh(mesh_drop), nn_partitioning.axis_rules(cfg_drop.logical_axis_rules):
        (l_drop, out_drop), (grads_drop, x_grad_drop) = jax.jit(
            jax.value_and_grad(loss_drop, argnums=(0, 1), has_aux=True)
        )(params, x)
      with jax.set_mesh(mesh_ref), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
        (l_ref, out_ref), (grads_ref, x_grad_ref) = jax.jit(jax.value_and_grad(loss_ref, argnums=(0, 1), has_aux=True))(
            params, x
        )

    np.testing.assert_allclose(np.asarray(l_drop), np.asarray(l_ref), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(out_drop), np.asarray(out_ref), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(x_grad_drop), np.asarray(x_grad_ref), rtol=0.0, atol=0.0)
    for leaf_name in ("gate", "wi_0", "wi_1", "wo"):
      np.testing.assert_allclose(
          np.asarray(jax.tree_util.tree_leaves(grads_drop[leaf_name])[0]),
          np.asarray(jax.tree_util.tree_leaves(grads_ref[leaf_name])[0]),
          rtol=0.0,
          atol=0.0,
      )

    # Negative control: perturbing one leaf (`wi_0`) must fail the same zero-tolerance check.
    perturbed_wi0 = np.asarray(grads_drop["wi_0"]).copy()
    perturbed_wi0.flat[0] += 1e-3
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(perturbed_wi0, np.asarray(grads_ref["wi_0"]), rtol=0.0, atol=0.0)


if __name__ == "__main__":
  unittest.main()
