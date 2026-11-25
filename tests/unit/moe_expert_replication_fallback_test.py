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

"""CPU and TPU tests for post-replication dropless fallback and cross-FSDP count reduction."""

import contextlib
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import qwix

from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import moe_replica_planner as mrp
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils import moe_replica_test_utils

_BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

_TILE_OVERRIDES = {
    f"{p}_tile_{s}_{d}": 256 if d == "mlp_dim" else 128
    for p in ("wi", "wo")
    for s in ("fwd", "dlhs", "drhs")
    for d in ("batch_seq", "embed_dim", "mlp_dim")
}


def _qwag_ref_gmm(
    lhs,
    rhs,
    group_sizes,
    preferred_element_type=jnp.bfloat16,
    existing_out=None,
    weight_gather_axes=None,
    **_unused_kwargs,
):
  """Differentiable GMM that gathers FSDP-sharded weights along `weight_gather_axes` inside `shard_map`."""
  for axis_name, dim_idx in weight_gather_axes or ():
    rhs = jax.lax.all_gather(rhs, axis_name=axis_name, axis=dim_idx, tiled=True)
  starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes.astype(jnp.int32))])
  rows = jnp.arange(lhs.shape[0], dtype=jnp.int32)
  out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=preferred_element_type) if existing_out is None else existing_out
  for g in range(rhs.shape[0]):
    in_group = (rows >= starts[g]) & (rows < starts[g + 1])
    proj = jnp.dot(lhs.astype(jnp.float32), rhs[g].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST).astype(
        preferred_element_type
    )
    out = jnp.where(in_group[:, None], proj, out)
  return out


def _build_routed_moe_config(*, hardware: str, megablox: bool, emb_dim: int, mlp_dim: int, **overrides):
  """Builds a minimal `RoutedMoE` config on an `(fsdp, expert=2)` mesh matching all visible JAX devices."""
  ici_ep = 2
  ici_fsdp = max(len(jax.devices()) // ici_ep, 2)
  kwargs = {
      "run_name": "moe_rep_fallback_test",
      "base_output_directory": "/tmp/moe_rep_fallback_test",
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


def _run_layer_fwd_bwd(cfg, mesh, x, cotangent, forced, *, ref_state=None, use_cpu_ra2a=False, force_qwag=False):
  """Runs one `RoutedMoE` forward + backward step and returns `(out, x_grad, m_grad, has_overflow, took_fallback)`."""
  with contextlib.ExitStack() as stack:
    stack.enter_context(jax.set_mesh(mesh))
    stack.enter_context(nn_partitioning.axis_rules(cfg.logical_axis_rules))
    if use_cpu_ra2a:
      stack.enter_context(
          mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=moe_replica_test_utils.emulated_ragged_all_to_all)
      )
    if force_qwag:
      flat_cfg = object.__getattribute__(cfg, "_flat_config")
      pyd_cfg = object.__getattribute__(cfg, "_pydantic_config")
      saved_overrides = {k: flat_cfg[k] for k in ("shard_embed_moe_on_fsdp", "hardware", "megablox")}

      def _restore_cfg():
        for k, v in saved_overrides.items():
          flat_cfg[k] = v
          object.__setattr__(pyd_cfg, k, v)

      for k, v in (("shard_embed_moe_on_fsdp", True), ("hardware", "tpu"), ("megablox", True)):
        flat_cfg[k] = v
        object.__setattr__(pyd_cfg, k, v)
      stack.callback(_restore_cfg)
      qwag_rule = qwix.QtRule(module_path=".*", weight_calibration_method="fixed,1")
      stack.enter_context(mock.patch.object(moe.ctypes, "moe_qwag_flag", return_value="weight_quantization"))
      stack.enter_context(mock.patch.object(moe.qpl, "get_current_rule", lambda op: qwag_rule if op == "gmm" else None))
      stack.enter_context(mock.patch.object(moe.mblx, "gmm", side_effect=_qwag_ref_gmm))

    layer = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.base_moe_mlp_dim,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    if ref_state is not None:
      nnx.update(layer, ref_state)
    saved_state = nnx.state(layer, nnx.Param)
    nnx.pop(layer, nnx.Intermediate)

    @nnx.jit
    def step(m, h):
      def loss_fn(m_, h_):
        out, _, _ = m_(h_, forced_routed_experts=forced)
        sown = nnx.pop(m_, nnx.Intermediate).to_pure_dict()
        ov = jnp.any(jnp.asarray(sown["moe_has_overflow"][0]))
        fb = (
            jnp.any(jnp.asarray(sown["moe_dropless_fallback"][0]))
            if "moe_dropless_fallback" in sown
            else jnp.bool_(False)
        )
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), (out, ov, fb)

      (_, (out, ov, fb)), (m_grad, x_grad) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, h)
      return out, x_grad, m_grad, ov, fb

    out, x_grad, m_grad, ov, fb = step(layer, x)
    return out, x_grad, m_grad, bool(np.asarray(ov)), bool(np.asarray(fb)), saved_state


class _FallbackReplicationSuiteMixin:
  """Shared verification logic executed on both simulated CPU meshes and real TPU meshes."""

  HARDWARE = "cpu"
  MEGABLOX = False
  EMB_DIM = 32
  MLP_DIM = 16
  USE_CPU_RA2A = True

  def _mesh_and_tensors(self, seed=17):
    """Creates an `(fsdp, expert=2)` mesh and sharded random activation/cotangent tensors."""
    num_devs = len(jax.devices())
    if num_devs < 4 or num_devs % 4 != 0:
      self.skipTest(f"Requires a multiple of 4 devices, got {num_devs}.")
    cfg0 = _build_routed_moe_config(
        hardware=self.HARDWARE, megablox=self.MEGABLOX, emb_dim=self.EMB_DIM, mlp_dim=self.MLP_DIM
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg0), cfg0.mesh_axes)
    sharding_x = jax.sharding.NamedSharding(mesh, P(("fsdp", "expert"), None, None))
    k_x, k_c = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.device_put(jax.random.normal(k_x, (num_devs, 64, self.EMB_DIM), jnp.bfloat16), sharding_x)
    cotangent = jax.device_put(jax.random.normal(k_c, (num_devs, 64, self.EMB_DIM), jnp.bfloat16), sharding_x)
    return mesh, x, cotangent

  def _assert_parity(self, out, dx, mg, out_ref, dx_ref, mg_ref, tol=3e-2, grad_atol=5e-2):
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(out_ref, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32), rtol=tol, atol=tol)
    for attr in ("wi_0", "wi_1", "wo"):
      np.testing.assert_allclose(
          np.asarray(getattr(mg, attr)[...], np.float32),
          np.asarray(getattr(mg_ref, attr)[...], np.float32),
          rtol=tol,
          atol=grad_atol,
          err_msg=attr,
      )

  def test_post_replication_overflow_suppression(self):
    """Replication absorbs pre-replication shard overflow (`216 > 160 -> 152 <= 160`) without triggering fallback."""
    mesh, x, cotangent = self._mesh_and_tensors(seed=11)
    num_devs = x.shape[0]
    sharding_f = jax.sharding.NamedSharding(mesh, P(("fsdp", "expert"), None, None))
    # Per EP shard (64 tokens, top-2 -> 128 balanced rows, rbf=1.25 -> buffer_size=160):
    # 44 tokens -> [0, 1], 20 tokens -> [0, 2] gives pre-rep EP shard-0 load 216 > 160,
    # while R=1 shifts 64 expert-0 rows from src=1 to dst=1, leaving [152, 104] <= 160.
    seq_forced = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (44, 2)),
            jnp.broadcast_to(jnp.array([[0, 2]], jnp.int32), (20, 2)),
        ],
        axis=0,
    )
    forced = jax.device_put(jnp.broadcast_to(seq_forced[None], (num_devs, 64, 2)), sharding_f)

    cfg_ref = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=0,
        ragged_buffer_factor=-1.0,
    )
    out_ref, dx_ref, mg_ref, ov_ref, _, state_ref = _run_layer_fwd_bwd(
        cfg_ref, mesh, x, cotangent, forced, use_cpu_ra2a=self.USE_CPU_RA2A
    )
    self.assertFalse(ov_ref)

    # At R=0 with rbf=1.25, pre-replication shard-0 load (216 > 160) must take the layer fallback.
    cfg_r0_fb = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=0,
        ragged_buffer_factor=1.25,
        moe_dropless_fallback="layer",
    )
    _, _, _, ov_r0, fb_r0, _ = _run_layer_fwd_bwd(
        cfg_r0_fb, mesh, x, cotangent, forced, ref_state=state_ref, use_cpu_ra2a=self.USE_CPU_RA2A
    )
    self.assertFalse(ov_r0)
    self.assertTrue(fb_r0)

    # At R=1 with rbf=1.25, post-replication physical load (152 <= 160) suppresses fallback and matches R=0 dropless.
    cfg_r1_fb = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        ragged_buffer_factor=1.25,
        moe_dropless_fallback="layer",
    )
    out_r1, dx_r1, mg_r1, ov_r1, fb_r1, _ = _run_layer_fwd_bwd(
        cfg_r1_fb, mesh, x, cotangent, forced, ref_state=state_ref, use_cpu_ra2a=self.USE_CPU_RA2A
    )
    self.assertFalse(ov_r1)
    self.assertFalse(fb_r1)
    self._assert_parity(out_r1, dx_r1, mg_r1, out_ref, dx_ref, mg_ref)

  def test_post_replication_overflow_triggers_dropless_fallback(self):
    """When post-replication physical load (`192 > 160`) still exceeds `rbf`, layer fallback runs with zero drops."""
    mesh, x, cotangent = self._mesh_and_tensors(seed=23)
    num_devs = x.shape[0]
    sharding_f = jax.sharding.NamedSharding(mesh, P(("fsdp", "expert"), None, None))
    # All 64 tokens -> [0, 1]: pre-rep [256, 0]; with R=1 (1 replica slot), dst=1 absorbs 64 rows of expert 0,
    # leaving post-rep physical load [192, 64], which still overflows buffer_size=160 at rbf=1.25.
    forced = jax.device_put(jnp.broadcast_to(jnp.array([[[0, 1]]], jnp.int32), (num_devs, 64, 2)), sharding_f)

    cfg_ref = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=0,
        ragged_buffer_factor=-1.0,
    )
    out_ref, dx_ref, mg_ref, _, _, state_ref = _run_layer_fwd_bwd(
        cfg_ref, mesh, x, cotangent, forced, use_cpu_ra2a=self.USE_CPU_RA2A
    )

    # Without fallback, R=1 at rbf=1.25 reports has_overflow=True and drops 32 rows.
    cfg_r1_drop = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        ragged_buffer_factor=1.25,
        moe_dropless_fallback=None,
    )
    out_drop, _, _, ov_drop, fb_drop, _ = _run_layer_fwd_bwd(
        cfg_r1_drop, mesh, x, cotangent, forced, ref_state=state_ref, use_cpu_ra2a=self.USE_CPU_RA2A
    )
    self.assertTrue(ov_drop)
    self.assertFalse(fb_drop)
    self.assertFalse(np.allclose(np.asarray(out_drop, np.float32), np.asarray(out_ref, np.float32), rtol=2e-2, atol=2e-2))

    # With moe_dropless_fallback="layer", R=1 triggers took_fallback=True and matches R=0 dropless exactly.
    cfg_r1_fb = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=self.MEGABLOX,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        ragged_buffer_factor=1.25,
        moe_dropless_fallback="layer",
    )
    out_fb, dx_fb, mg_fb, ov_fb, fb_taken, _ = _run_layer_fwd_bwd(
        cfg_r1_fb, mesh, x, cotangent, forced, ref_state=state_ref, use_cpu_ra2a=self.USE_CPU_RA2A
    )
    self.assertFalse(ov_fb)
    self.assertTrue(fb_taken)
    self._assert_parity(out_fb, dx_fb, mg_fb, out_ref, dx_ref, mg_ref)

  def test_cross_fsdp_psum_agrees_on_replica_plan_under_weight_gather(self):
    """2D `(fsdp, expert=2)` QWAG path psums `local_counts` across FSDP so all FSDP slices share one plan."""
    mesh, x, cotangent = self._mesh_and_tensors(seed=37)
    num_devs = x.shape[0]
    half = num_devs // 2
    sharding_f = jax.sharding.NamedSharding(mesh, P(("fsdp", "expert"), None, None))
    # First half of FSDP slices is shard-0-heavy: 64 -> [0, 1] (per-shard [64, 64, 0, 0]).
    # Second half of FSDP slices is shard-1-heavy: 40 -> [2, 3], 24 -> [0, 1] (per-shard [24, 24, 40, 40]).
    # Unsummed plans disagree (rep_home [-1, 0] vs [1, -1]); summed counts give a shared non-vacuous plan
    # rep_home=[-1, 0], leaving first-half FSDP physical load [192, 64].
    f0 = jnp.broadcast_to(jnp.array([[[0, 1]]], jnp.int32), (half, 64, 2))
    seq_f1 = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.array([[2, 3]], jnp.int32), (40, 2)),
            jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (24, 2)),
        ],
        axis=0,
    )
    f1 = jnp.broadcast_to(seq_f1[None], (half, 64, 2))
    forced = jax.device_put(jnp.concatenate([f0, f1], axis=0), sharding_f)

    plan_f0 = mrp.compute_plan_iter(jnp.array([[64, 64, 0, 0], [64, 64, 0, 0]], jnp.int32), 2, 1, 2)
    plan_f1 = mrp.compute_plan_iter(jnp.array([[24, 24, 40, 40], [24, 24, 40, 40]], jnp.int32), 2, 1, 2)
    scale_fsdp = num_devs // 4
    plan_sum = mrp.compute_plan_iter(jnp.array([[88, 88, 40, 40], [88, 88, 40, 40]], jnp.int32) * scale_fsdp, 2, 1, 2)
    self.assertFalse(np.array_equal(np.asarray(plan_f0.rep_home), np.asarray(plan_f1.rep_home)))
    np.testing.assert_array_equal(np.asarray(plan_sum.rep_home), np.array([[-1], [0]], dtype=np.int32))

    cfg_ref = _build_routed_moe_config(
        hardware=self.HARDWARE,
        megablox=True,
        emb_dim=self.EMB_DIM,
        mlp_dim=self.MLP_DIM,
        expert_replicas_per_device=0,
        ragged_buffer_factor=-1.0,
    )
    out_ref, dx_ref, mg_ref, _, _, state_ref = _run_layer_fwd_bwd(
        cfg_ref, mesh, x, cotangent, forced, use_cpu_ra2a=self.USE_CPU_RA2A, force_qwag=True
    )

    for rbf, expected_fb in ((1.625, False), (1.25, True)):
      with self.subTest(rbf=rbf):
        cfg_rep = _build_routed_moe_config(
            hardware=self.HARDWARE,
            megablox=True,
            emb_dim=self.EMB_DIM,
            mlp_dim=self.MLP_DIM,
            expert_replicas_per_device=1,
            expert_replica_max_send=2,
            ragged_buffer_factor=rbf,
            moe_dropless_fallback="layer",
        )
        out_rep, dx_rep, mg_rep, ov_rep, fb_rep, _ = _run_layer_fwd_bwd(
            cfg_rep,
            mesh,
            x,
            cotangent,
            forced,
            ref_state=state_ref,
            use_cpu_ra2a=self.USE_CPU_RA2A,
            force_qwag=True,
        )
        self.assertFalse(ov_rep)
        self.assertEqual(fb_rep, expected_fb)
        self._assert_parity(out_rep, dx_rep, mg_rep, out_ref, dx_ref, mg_ref)


class RoutedMoEReplicationFallbackCpuTest(_FallbackReplicationSuiteMixin, parameterized.TestCase):
  """CPU tests (4 simulated host devices) + config validator checks."""

  def test_config_validator_accepts_layer_dropless_fallback_with_replication(self):
    """Verifies `moe_dropless_fallback='layer'` is accepted with replication while `'step'` stays RoE-only."""
    cfg = _build_routed_moe_config(
        hardware="cpu",
        megablox=False,
        emb_dim=32,
        mlp_dim=16,
        expert_replicas_per_device=2,
        moe_dropless_fallback="layer",
        use_ragged_sort=True,
        ragged_buffer_factor=1.25,
    )
    self.assertEqual(cfg.expert_replicas_per_device, 2)
    self.assertEqual(cfg.moe_dropless_fallback, "layer")
    with self.assertRaisesRegex(ValueError, "use_ring_of_experts=True"):
      _build_routed_moe_config(
          hardware="cpu",
          megablox=False,
          emb_dim=32,
          mlp_dim=16,
          expert_replicas_per_device=2,
          moe_dropless_fallback="step",
          use_ragged_sort=True,
          ragged_buffer_factor=1.25,
      )


@pytest.mark.tpu_only
class RoutedMoEReplicationFallbackTpuTest(_FallbackReplicationSuiteMixin, parameterized.TestCase):
  """Real TPU (`(fsdp, expert=2)`) tests with native TPU `ragged_all_to_all`, `ragged_sort`, and `mblx.gmm`."""

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
