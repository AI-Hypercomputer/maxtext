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

"""CPU and TPU tests for lagged (t-1) MoEReplicaPlanCache online hot-expert replication."""

import contextlib
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import orbax.checkpoint as ocp
import pytest

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import moe_replica_planner as mrp
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils import moe_replica_test_utils

_BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

_TILE_OVERRIDES = {
    f"{p}_tile_{s}_{d}": 256 if d == "mlp_dim" else 128
    for p in ("wi", "wo")
    for s in ("fwd", "dlhs", "drhs")
    for d in ("batch_seq", "embed_dim", "mlp_dim")
}

_PLAN_CACHE_ATTRS = (
    "rep_route_table",
    "rep_home",
    "rep_slot",
    "rep_send_slots",
    "rep_send_peer_index",
)


def _build_layer_cfg(*, hardware: str, megablox: bool, emb_dim: int, mlp_dim: int, **overrides):
  """Builds a `RoutedMoE` config on an `(fsdp, expert=2)` mesh matching all visible JAX devices."""
  ici_ep = 2
  ici_fsdp = max(len(jax.devices()) // ici_ep, 2)
  kwargs = {
      "run_name": "moe_rep_lagged_layer_test",
      "base_output_directory": "/tmp/moe_rep_lagged_test",
      "enable_checkpointing": False,
      "model_name": "mixtral-8x7b",
      "override_model_config": True,
      "attention": "dot_product",
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": False,
      "megablox": megablox,
      "sparse_matmul": True,
      "use_ring_of_experts": False,
      "use_ragged_sort": True,
      "ragged_sort_use_single_sparsecore": True,
      "moe_use_direct_token_gather": False,
      "base_emb_dim": emb_dim,
      "base_mlp_dim": mlp_dim,
      "base_moe_mlp_dim": mlp_dim,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "per_device_batch_size": 1,
      "max_target_length": 64,
      "load_balance_loss_weight": 0.0,
      "routed_bias": False,
      "dropout_rate": 0.0,
      "ici_fsdp_parallelism": ici_fsdp,
      "ici_expert_parallelism": ici_ep,
      "ici_tensor_parallelism": 1,
      "shard_embed_moe_on_fsdp": False,
      "skip_jax_distributed_system": True,
      "hardware": hardware,
  }
  if megablox:
    kwargs.update(_TILE_OVERRIDES)
  kwargs.update(overrides)
  return pyconfig.initialize(["", _BASE_CONFIG], **kwargs)


def _build_deepseek_train_cfg(tmp_dir: str, *, hardware: str, megablox: bool, emb_dim: int, mlp_dim: int, **overrides):
  """Builds a scanned DeepSeek-V3 + MTP=1 config for end-to-end `train_step` lagged-plan verification."""
  ici_ep = 2
  ici_fsdp = max(len(jax.devices()) // ici_ep, 2)
  kwargs = {
      "run_name": "moe_rep_lagged_train_test",
      "base_output_directory": tmp_dir,
      "enable_checkpointing": False,
      "model_name": "deepseek3-671b",
      "override_model_config": True,
      "base_emb_dim": emb_dim,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "base_mlp_dim": mlp_dim,
      "base_moe_mlp_dim": mlp_dim,
      "base_num_decoder_layers": 3,
      "first_num_dense_layers": 1,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "n_routing_groups": -1,
      "topk_routing_group": -1,
      "q_lora_rank": 0,
      "kv_lora_rank": 32,
      "qk_nope_head_dim": 32,
      "qk_rope_head_dim": 16,
      "v_head_dim": 32,
      "vocab_size": 256,
      "mtp_num_layers": 1,
      "mtp_loss_scaling_factor": 0.1,
      "mtp_reuse_input_embedding": True,
      "attention": "dot_product",
      "use_tokamax_splash": False,
      "scan_layers": True,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      "megablox": megablox,
      "sparse_matmul": True,
      "use_ring_of_experts": False,
      "use_ragged_sort": True,
      "ragged_sort_use_single_sparsecore": True,
      "moe_use_direct_token_gather": False,
      "moe_flat_fsdp_weights": False,
      "shard_embed_moe_on_fsdp": False,
      "ici_fsdp_parallelism": ici_fsdp,
      "ici_expert_parallelism": ici_ep,
      "ici_tensor_parallelism": 1,
      "per_device_batch_size": 1,
      "max_target_length": 64,
      "dataset_type": "synthetic",
      "learning_rate": 0.0,
      "routed_bias": True,
      "routed_bias_update_rate": 0.0,
      "load_balance_loss_weight": 0.0,
      "dropout_rate": 0.0,
      "enable_goodput_recording": False,
      "monitor_goodput": False,
      "skip_jax_distributed_system": True,
      "hardware": hardware,
      "steps": 2,
  }
  if megablox:
    kwargs.update(_TILE_OVERRIDES)
  kwargs.update(overrides)
  return pyconfig.initialize(["", _BASE_CONFIG], **kwargs)


class _LaggedReplicationSuiteMixin:
  """Shared verification suite run on simulated CPU meshes and native TPU meshes."""

  HARDWARE = "cpu"
  MEGABLOX = False
  EMB_DIM = 64
  MLP_DIM = 32
  USE_CPU_RA2A = True

  @contextlib.contextmanager
  def _maybe_emulate_ra2a(self):
    if self.USE_CPU_RA2A:
      with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=moe_replica_test_utils.emulated_ragged_all_to_all):
        yield
    else:
      yield

  def test_step0_warmup_fallback_and_step1_exact_plan_match(self):
    """Step 0 identity plan falls back droplessly; `_update_moe_replica_plans` sets Step 1's exact plan (`fb=False`)."""
    num_devs = len(jax.devices())
    if num_devs < 4 or num_devs % 4 != 0:
      self.skipTest(f"Requires a multiple of 4 devices, got {num_devs}.")
    ici_fsdp = num_devs // 2
    cfg_ref = _build_layer_cfg(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=0,
        ragged_buffer_factor=-1.0,
    )
    cfg_lag = _build_layer_cfg(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        expert_replica_lagged_plan=True,
        ragged_buffer_factor=1.25,
        moe_dropless_fallback="layer",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    sharding_x = jax.sharding.NamedSharding(mesh, P(("fsdp", "expert"), None, None))
    k_x, k_c = jax.random.split(jax.random.PRNGKey(41))
    x = jax.device_put(jax.random.normal(k_x, (num_devs, 64, self.EMB_DIM), jnp.bfloat16), sharding_x)
    cotangent = jax.device_put(jax.random.normal(k_c, (num_devs, 64, self.EMB_DIM), jnp.bfloat16), sharding_x)
    # 44 tokens -> [0, 1], 20 tokens -> [0, 2]: pre-rep shard-0 load 216 > 160 (rbf=1.25),
    # post-rep shard-0 load 152 <= 160. Summed across `ici_fsdp`, each EP shard has `ici_fsdp * [64, 44, 20, 0]`.
    seq_forced = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (44, 2)),
            jnp.broadcast_to(jnp.array([[0, 2]], jnp.int32), (20, 2)),
        ],
        axis=0,
    )
    forced = jax.device_put(jnp.broadcast_to(seq_forced[None], (num_devs, 64, 2)), sharding_x)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules), self._maybe_emulate_ra2a():
      layer_ref = moe.RoutedMoE(
          config=cfg_ref,
          num_experts=4,
          num_experts_per_tok=2,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=self.MLP_DIM,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.float32,
          rngs=nnx.Rngs(0),
      )
      layer_lag = moe.RoutedMoE(
          config=cfg_lag,
          num_experts=4,
          num_experts_per_tok=2,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=self.MLP_DIM,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.float32,
          rngs=nnx.Rngs(0),
      )
      nnx.update(layer_lag, nnx.state(layer_ref, nnx.Param))
      np.testing.assert_array_equal(np.asarray(layer_lag.rep_home[...]), np.array([[-1], [-1]], dtype=np.int32))

      @nnx.jit
      def run_fwd_bwd(m, h):
        def loss_fn(m_, h_):
          out, _, _ = m_(h_, forced_routed_experts=forced)
          sown = nnx.pop(m_, nnx.Intermediate).to_pure_dict()
          return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), (out, sown)

        (_, (out, sown)), (m_grad, x_grad) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, h)
        return out, x_grad, m_grad, sown

      out_ref, dx_ref, mg_ref, _ = run_fwd_bwd(layer_ref, x)

      # Step 0: identity warmup plan must NOT call compute_plan_iter inside the layer, MUST take dropless fallback,
      # and MUST match R=0 dropless output + gradients with zero dropped tokens.
      with mock.patch.object(
          mrp, "compute_plan_iter", side_effect=AssertionError("in-layer planner must not run in lagged mode")
      ):
        out_s0, dx_s0, mg_s0, sown_s0 = run_fwd_bwd(layer_lag, x)

      self.assertFalse(bool(np.any(np.asarray(sown_s0["moe_has_overflow"][0]))))
      self.assertTrue(bool(np.any(np.asarray(sown_s0["moe_dropless_fallback"][0]))))
      np.testing.assert_allclose(np.asarray(out_s0, np.float32), np.asarray(out_ref, np.float32), rtol=3e-2, atol=3e-2)
      np.testing.assert_allclose(np.asarray(dx_s0, np.float32), np.asarray(dx_ref, np.float32), rtol=3e-2, atol=3e-2)
      for attr in ("wi_0", "wi_1", "wo"):
        np.testing.assert_allclose(
            np.asarray(getattr(mg_s0, attr)[...], np.float32),
            np.asarray(getattr(mg_ref, attr)[...], np.float32),
            rtol=3e-2,
            atol=5e-2,
        )

      # Apply the once-per-step outside-scan plan update and verify exact match against manual compute_plan_iter.
      train._update_moe_replica_plans(layer_lag, sown_s0, cfg_lag)  # pylint: disable=protected-access
      expected_plan = mrp.compute_plan_iter(
          jnp.array([[64, 44, 20, 0], [64, 44, 20, 0]], dtype=jnp.int32) * ici_fsdp,
          num_ep=2,
          num_replicas=1,
          max_send=2,
      )
      self.assertGreater(int((np.asarray(layer_lag.rep_home[...]) >= 0).sum()), 0)
      np.testing.assert_array_equal(np.asarray(layer_lag.rep_route_table[...]), np.asarray(expected_plan.route_table))
      np.testing.assert_array_equal(np.asarray(layer_lag.rep_home[...]), np.asarray(expected_plan.rep_home))
      np.testing.assert_array_equal(np.asarray(layer_lag.rep_slot[...]), np.asarray(expected_plan.rep_slot))
      np.testing.assert_array_equal(np.asarray(layer_lag.rep_send_slots[...]), np.asarray(expected_plan.send_slots))
      np.testing.assert_array_equal(
          np.asarray(layer_lag.rep_send_peer_index[...]), np.asarray(expected_plan.send_peer_index)
      )

      # Step 1: cached t-1 plan keeps physical load at 152 <= 160, so took_fallback is False and parity holds.
      with mock.patch.object(
          mrp, "compute_plan_iter", side_effect=AssertionError("in-layer planner must not run in lagged mode")
      ):
        out_s1, dx_s1, mg_s1, sown_s1 = run_fwd_bwd(layer_lag, x)

      self.assertFalse(bool(np.any(np.asarray(sown_s1["moe_has_overflow"][0]))))
      self.assertFalse(bool(np.any(np.asarray(sown_s1["moe_dropless_fallback"][0]))))
      np.testing.assert_allclose(np.asarray(out_s1, np.float32), np.asarray(out_ref, np.float32), rtol=3e-2, atol=3e-2)
      np.testing.assert_allclose(np.asarray(dx_s1, np.float32), np.asarray(dx_ref, np.float32), rtol=3e-2, atol=3e-2)
      for attr in ("wi_0", "wi_1", "wo"):
        np.testing.assert_allclose(
            np.asarray(getattr(mg_s1, attr)[...], np.float32),
            np.asarray(getattr(mg_ref, attr)[...], np.float32),
            rtol=3e-2,
            atol=5e-2,
        )

  def test_scanned_train_step_lagged_plan_and_orbax_param_exclusion(self):
    """Verifies scanned + MTP `train_step` Step 0->1 plan update and `MoEReplicaPlanCache` exclusion from Orbax params."""
    num_devs = len(jax.devices())
    if num_devs < 4 or num_devs % 4 != 0:
      self.skipTest(f"Requires a multiple of 4 devices, got {num_devs}.")
    tmp_dir = self.create_tempdir().full_path
    batch = {
        "inputs": jax.random.randint(jax.random.PRNGKey(101), (num_devs, 64), 1, 256, dtype=jnp.int32),
        "inputs_position": jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32)[None, :], (num_devs, 64)),
        "inputs_segmentation": jnp.ones((num_devs, 64), dtype=jnp.int32),
        "targets": jax.random.randint(jax.random.PRNGKey(202), (num_devs, 64), 1, 256, dtype=jnp.int32),
        "targets_position": jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32)[None, :], (num_devs, 64)),
        "targets_segmentation": jnp.ones((num_devs, 64), dtype=jnp.int32),
    }

    def set_skewed_router(model):
      """Biases router so all tokens pick `[0, 1]`: Step 0 load `256 > 208` (`rbf=1.625`), Step 1 load `192 <= 208`."""
      for _, node in nnx.iter_graph(model):
        if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
          b = node.bias.value
          node.bias.value = (
              jnp.zeros_like(b).at[..., 0].set(8.0).at[..., 1].set(4.0).at[..., 2].set(-8.0).at[..., 3].set(-8.0)
          )

    cfg_r0 = _build_deepseek_train_cfg(
        os.path.join(tmp_dir, "r0"),
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=0,
        ragged_buffer_factor=-1.0,
    )
    cfg_lag = _build_deepseek_train_cfg(
        os.path.join(tmp_dir, "lag"),
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        expert_replica_lagged_plan=True,
        ragged_buffer_factor=1.625,
        moe_dropless_fallback="layer",
    )

    with self._maybe_emulate_ra2a():
      _, _, shardings_r0, _, mesh, _, _, _, _, _, state_r0 = train_utils.setup_train_loop(cfg_r0, None)
      set_skewed_router(state_r0.model)

      _, _, shardings_lag, _, _, _, _, _, _, _, state_lag = train_utils.setup_train_loop(cfg_lag, None)
      # Copy exact Param + MoEBiasVar weights (fresh buffers: p_step_r0 donates state_r0) while keeping MoEReplicaPlanCache.
      nnx.update(
          state_lag.model,
          jax.tree.map(jnp.copy, nnx.state(state_r0.model, nnx.Any(nnx.Param, moe.MoEBiasVar))),
      )

      # (c) Verify MoEReplicaPlanCache is excluded from nnx.Param splits and Orbax parameter checkpoints.
      _, param_state, _ = nnx.split(state_lag.model, nnx.Param, ...)
      param_pure = nnx.to_pure_dict(param_state)
      flat_param_keys = {"/".join(map(str, k)) for k in traverse_util.flatten_dict(param_pure)}
      for cache_attr in _PLAN_CACHE_ATTRS:
        self.assertFalse(any(cache_attr in path for path in flat_param_keys), msg=cache_attr)
      cache_only = nnx.to_pure_dict(nnx.state(state_lag.model, moe.MoEReplicaPlanCache))
      flat_cache_keys = {"/".join(map(str, k)) for k in traverse_util.flatten_dict(cache_only)}
      for cache_attr in _PLAN_CACHE_ATTRS:
        self.assertTrue(any(cache_attr in path for path in flat_cache_keys), msg=cache_attr)

      ckpt_path = os.path.join(tmp_dir, "orbax_params_ckpt")
      ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
          ckpt_path, {"params": {"params": param_pure}}, force=True
      )
      _, r0_param_abstract, _ = nnx.split(state_r0.model, nnx.Param, ...)
      restored_into_r0 = checkpointing.load_params_from_path(ckpt_path, r0_param_abstract, 8)
      restored_flat_keys = {"/".join(map(str, k)) for k in traverse_util.flatten_dict(nnx.to_pure_dict(restored_into_r0))}
      self.assertEqual(restored_flat_keys, flat_param_keys)

      sharded_batch = jax.tree.map(
          lambda a: jax.device_put(a, jax.sharding.NamedSharding(mesh, P(("fsdp", "expert"), None))), batch
      )

      jit_r0, pure_r0 = nnx.split(state_r0)
      p_shardings_r0, s_shardings_r0 = sharding.maybe_update_params_sharding_with_opt(cfg_r0, shardings_r0)
      p_step_r0, _ = train_utils.jit_train_and_eval_step(
          cfg_r0, jit_r0, mesh, pure_r0, s_shardings_r0, train.train_step, None, None, p_shardings_r0
      )

      jit_lag, pure_lag = nnx.split(state_lag)
      p_shardings_lag, s_shardings_lag = sharding.maybe_update_params_sharding_with_opt(cfg_lag, shardings_lag)
      p_step_lag, _ = train_utils.jit_train_and_eval_step(
          cfg_lag, jit_lag, mesh, pure_lag, s_shardings_lag, train.train_step, None, None, p_shardings_lag
      )

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_lag.logical_axis_rules):
        pure_r0, m_r0_s0 = p_step_r0(pure_r0, sharded_batch)
        pure_r0, m_r0_s1 = p_step_r0(pure_r0, sharded_batch)

        # Step 0: identity warmup plan overflows rbf=1.625 across all 3 MoE layers (2 scanned + 1 MTP) -> fallback!
        pure_lag, m_lag_s0 = p_step_lag(pure_lag, sharded_batch)
        fb_layers_s0 = int(np.asarray(m_lag_s0["scalar"]["learning/moe_dropless_fallback_layers"]))
        self.assertEqual(fb_layers_s0, 3)

        merged_after_s0 = nnx.merge(jit_lag, pure_lag)
        routed_moes = [node for _, node in nnx.iter_graph(merged_after_s0.model) if isinstance(node, moe.RoutedMoE)]
        self.assertEqual(sorted(tuple(m.rep_home[...].shape) for m in routed_moes), [(2, 1), (2, 2, 1)])
        for m_node in routed_moes:
          self.assertTrue(np.all(np.asarray(m_node.rep_home[...])[..., 1, 0] == 0))

        # Step 1: cached t-1 plan eliminates the overflow across all 3 MoE layers (fallback_layers == 0)!
        pure_lag, m_lag_s1 = p_step_lag(pure_lag, sharded_batch)
        fb_layers_s1 = int(np.asarray(m_lag_s1["scalar"]["learning/moe_dropless_fallback_layers"]))
        self.assertEqual(fb_layers_s1, 0)

      for m_lag, m_ref in ((m_lag_s0, m_r0_s0), (m_lag_s1, m_r0_s1)):
        np.testing.assert_allclose(
            float(np.asarray(m_lag["scalar"]["learning/loss"])),
            float(np.asarray(m_ref["scalar"]["learning/loss"])),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            float(np.asarray(m_lag["scalar"]["learning/grad_norm"])),
            float(np.asarray(m_ref["scalar"]["learning/grad_norm"])),
            rtol=2e-2,
            atol=2e-2,
        )


class RoutedMoEReplicationLaggedCpuTest(_LaggedReplicationSuiteMixin, parameterized.TestCase):
  """CPU tests (4 simulated host devices) for `expert_replica_lagged_plan=True`."""


@pytest.mark.tpu_only
class RoutedMoEReplicationLaggedTpuTest(_LaggedReplicationSuiteMixin, parameterized.TestCase):
  """Real TPU (`(fsdp, expert=2)`) tests for `expert_replica_lagged_plan=True`."""

  HARDWARE = "tpu"
  MEGABLOX = True
  EMB_DIM = 256
  MLP_DIM = 256
  USE_CPU_RA2A = False

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest("TPU only")


if __name__ == "__main__":
  absltest.main()
