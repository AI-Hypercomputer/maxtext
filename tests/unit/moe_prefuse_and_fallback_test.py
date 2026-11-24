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

"""CPU unit tests for pre-fused MoE weights (FSDP/TP/Muon) and dropless fallback."""

from __future__ import annotations

import functools
import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.core.spmd import logical_axis_rules
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.models import deepseek
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding as sharding_lib
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import numpy as np
import optax
from optax.contrib._muon import MuonDimensionNumbers
import pytest

pytestmark = [pytest.mark.cpu_only]

BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")


def _count_primitive(jaxpr, prim_name: str) -> int:
  """Recursively counts equations matching `prim_name` across nested Jaxprs."""
  count = sum(eqn.primitive.name == prim_name for eqn in jaxpr.eqns)
  for eqn in jaxpr.eqns:
    for sub in eqn.params.values():
      if isinstance(sub, jax.extend.core.ClosedJaxpr):
        count += _count_primitive(sub.jaxpr, prim_name)
      elif isinstance(sub, jax.extend.core.Jaxpr):
        count += _count_primitive(sub, prim_name)
  return count


def _count_3d_concatenate(jaxpr) -> int:
  """Recursively counts 3D `concatenate` equations (e.g. `[E, K, N]` expert weights)."""
  count = sum(eqn.primitive.name == "concatenate" and getattr(eqn.outvars[0].aval, "ndim", 0) == 3 for eqn in jaxpr.eqns)
  for eqn in jaxpr.eqns:
    for sub in eqn.params.values():
      if isinstance(sub, jax.extend.core.ClosedJaxpr):
        count += _count_3d_concatenate(sub.jaxpr)
      elif isinstance(sub, jax.extend.core.Jaxpr):
        count += _count_3d_concatenate(sub)
  return count


def _cpu_ragged_all_to_all(
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
  """Differentiable pure-JAX reference for `jax.lax.ragged_all_to_all` on XLA:CPU."""
  del axis_index_groups
  me = jax.lax.axis_index(axis_name)
  all_operands = jax.lax.all_gather(operand, axis_name=axis_name)
  all_in_offsets = jax.lax.all_gather(input_offsets, axis_name=axis_name)
  all_send_sizes = jax.lax.all_gather(send_sizes, axis_name=axis_name)
  all_out_offsets = jax.lax.all_gather(output_offsets, axis_name=axis_name)

  rows = jnp.arange(output.shape[0])
  mask_shape = (output.shape[0],) + (1,) * (output.ndim - 1)
  res = output
  oob = jnp.asarray(False)
  for src in range(all_operands.shape[0]):
    in_off = all_in_offsets[src, me]
    sz = all_send_sizes[src, me]
    out_off = all_out_offsets[src, me]
    oob = (
        oob
        | (sz < 0)
        | (
            (sz > 0)
            & ((in_off < 0) | (in_off + sz > operand.shape[0]) | (out_off < 0) | (out_off + sz > output.shape[0]))
        )
    )
    mask = ((rows >= out_off) & (rows < out_off + sz)).reshape(mask_shape)
    src_idx = jnp.clip(rows - out_off + in_off, 0, operand.shape[0] - 1)
    res = jnp.where(mask, jnp.take(all_operands[src], src_idx, axis=0), res)
  size_mismatch = jnp.any(recv_sizes != all_send_sizes[:, me])
  return jnp.where(size_mismatch | oob, jnp.asarray(jnp.nan, dtype=res.dtype), res)


def _make_moe_config(**overrides):
  """Builds a small CPU Mixtral MoE config for sparse_matmul tests."""
  defaults = dict(  # pylint: disable=use-dict-literal
      run_name="moe_prefuse_and_fallback_test",
      enable_checkpointing=False,
      model_name="mixtral-8x7b",
      override_model_config=True,
      base_emb_dim=32,
      base_mlp_dim=16,
      base_moe_mlp_dim=16,
      num_experts=4,
      num_experts_per_tok=2,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      float32_gate_logits=True,
      megablox=False,
      sparse_matmul=True,
      per_device_batch_size=1,
      max_target_length=8,
      dropout_rate=0.0,
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  defaults.update(overrides)
  return pyconfig.initialize(["", BASE_CONFIG], **defaults)


def _make_deepseek_moe_config(**overrides):
  """Builds a minimal CPU deepseek3-671b config with MoE + Muon enabled."""
  defaults = dict(  # pylint: disable=use-dict-literal
      run_name="deepseek_moe_muon_test",
      enable_checkpointing=False,
      model_name="deepseek3-671b",
      override_model_config=True,
      base_emb_dim=32,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=16,
      q_lora_rank=16,
      kv_lora_rank=8,
      qk_nope_head_dim=8,
      qk_rope_head_dim=8,
      v_head_dim=16,
      base_mlp_dim=16,
      base_moe_mlp_dim=16,
      num_experts=4,
      num_experts_per_tok=2,
      shared_experts=1,
      routed_scaling_factor=2.5,
      routed_score_func="sigmoid",
      routed_bias=False,
      routed_bias_update_rate=0.0,
      n_routing_groups=-1,
      topk_routing_group=-1,
      first_num_dense_layers=0,
      vocab_size=64,
      base_num_decoder_layers=1,
      per_device_batch_size=1,
      max_target_length=8,
      max_prefill_predict_length=4,
      max_position_embeddings=64,
      original_max_position_embeddings=64,
      scan_layers=False,
      attention="dot_product",
      attention_type="mla",
      megablox=False,
      sparse_matmul=True,
      capacity_factor=-1,
      matmul_precision="highest",
      dtype="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      dropout_rate=0.0,
      opt_type="muon",
      learning_rate=1e-2,
      adam_weight_decay=0.01,
      muon_consistent_rms=0.2,
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  defaults.update(overrides)
  return pyconfig.initialize(["", BASE_CONFIG], **defaults)


class MoePrefuseAndFallbackTest(parameterized.TestCase):
  """Verifies pre-fused MoE FSDP/TP/Muon parity and dropless combine behavior."""

  def _build_routed_moe_pair(self, cfg_unfused, cfg_fused, mesh, batch_axis):
    """Creates sharded unfused/fused `RoutedMoE` layers with identical weights and loss closures."""
    layer_u = moe.RoutedMoE(
        config=cfg_unfused,
        num_experts=cfg_unfused.num_experts,
        num_experts_per_tok=cfg_unfused.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg_unfused.base_moe_mlp_dim,
        dtype=jnp.dtype(jnp.float32),
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    layer_f = moe.RoutedMoE(
        config=cfg_fused,
        num_experts=cfg_fused.num_experts,
        num_experts_per_tok=cfg_fused.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg_fused.base_moe_mlp_dim,
        dtype=jnp.dtype(jnp.float32),
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    layer_f.gate.kernel[...] = layer_u.gate.kernel[...]
    layer_f.wo[...] = layer_u.wo[...]
    assert layer_u.wi_0 is not None and layer_u.wi_1 is not None
    layer_f.wi[...] = jnp.concatenate([layer_u.wi_0[...], layer_u.wi_1[...]], axis=-1)

    gdef_u, params_u, other_u = nnx.split(layer_u, nnx.Param, ...)
    gdef_f, params_f, other_f = nnx.split(layer_f, nnx.Param, ...)
    shardings_u = sharding_lib.nnx_construct_named_sharding(params_u, mesh)
    shardings_f = sharding_lib.nnx_construct_named_sharding(params_f, mesh)
    params_u = jax.tree.map(
        lambda v, s: v.replace(jax.device_put(v.get_value(), s.get_value())),
        params_u,
        shardings_u,
        is_leaf=lambda v: isinstance(v, nnx.Variable),
    )
    params_f = jax.tree.map(
        lambda v, s: v.replace(jax.device_put(v.get_value(), s.get_value())),
        params_f,
        shardings_f,
        is_leaf=lambda v: isinstance(v, nnx.Variable),
    )
    x = jnp.asarray(
        np.random.RandomState(42).randn(4, cfg_unfused.max_target_length, cfg_unfused.base_emb_dim),
        dtype=jnp.float32,
    )
    x = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(batch_axis, None, None)))

    def loss_unfused(p, x_in):
      m = nnx.merge(gdef_u, p, other_u)
      out, lb, _ = m(x_in)
      return jnp.mean(out**2) + (0.0 if lb is None else lb), out

    def loss_fused(p, x_in):
      m = nnx.merge(gdef_f, p, other_f)
      out, lb, _ = m(x_in)
      return jnp.mean(out**2) + (0.0 if lb is None else lb), out

    return params_u, params_f, x, loss_unfused, loss_fused

  def _assert_routed_moe_parity(self, unfused_res, fused_res):
    """Asserts forward outputs, loss, and input/weight gradients match between unfused and fused MoE."""
    (loss_u, out_u), (dparams_u, dx_u) = unfused_res
    (loss_f, out_f), (dparams_f, dx_f) = fused_res
    np.testing.assert_allclose(np.asarray(loss_f), np.asarray(loss_u), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_f), np.asarray(out_u), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dx_f), np.asarray(dx_u), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(dparams_f["gate"]["kernel"][...]),
        np.asarray(dparams_u["gate"]["kernel"][...]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(dparams_f["wo"][...]),
        np.asarray(dparams_u["wo"][...]),
        rtol=1e-5,
        atol=1e-5,
    )
    expected_dwi = jnp.concatenate([dparams_u["wi_0"][...], dparams_u["wi_1"][...]], axis=-1)
    np.testing.assert_allclose(np.asarray(dparams_f["wi"][...]), np.asarray(expected_dwi), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("fsdp4_ep1_tp1", 4, 1, 1),
      ("fsdp2_ep2_tp1", 2, 2, 1),
      ("fsdp1_ep2_tp2", 1, 2, 2),
  )
  def test_prefuse_moe_weights_fsdp_tp_parity_and_single_all_gather(self, fsdp_deg: int, ep_deg: int, tp_deg: int):
    if jax.device_count() != 4:
      self.skipTest("Requires 4 simulated CPU devices.")

    cfg_unfused = _make_moe_config(
        prefuse_moe_weights=False,
        ici_fsdp_parallelism=fsdp_deg,
        ici_expert_parallelism=ep_deg,
        ici_tensor_parallelism=tp_deg,
    )
    cfg_fused = _make_moe_config(
        prefuse_moe_weights=True,
        moe_unsliced_prefused_wi=tp_deg == 1,
        ici_fsdp_parallelism=fsdp_deg,
        ici_expert_parallelism=ep_deg,
        ici_tensor_parallelism=tp_deg,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_unfused), cfg_unfused.mesh_axes)
    batch_axis = ("fsdp", "expert") if (fsdp_deg > 1 or ep_deg > 1) else None

    with (
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_unfused.logical_axis_rules),
        logical_axis_rules(cfg_unfused.logical_axis_rules),
    ):
      params_u, params_f, x, loss_unfused, loss_fused = self._build_routed_moe_pair(
          cfg_unfused, cfg_fused, mesh, batch_axis
      )

      jaxpr_u = jax.make_jaxpr(loss_unfused)(params_u, x)
      jaxpr_f = jax.make_jaxpr(loss_fused)(params_f, x)
      if tp_deg == 1:
        # Inspect unmocked Jaxpr before any CPU collective shim: with `moe_unsliced_prefused_wi=True`
        # on TP=1, pre-fused `wi` enters shard_map unsliced (`w1_kernel=None`), so zero 3D `[E, K, N]`
        # `concatenate` ops are emitted and total `concatenate` count equals unfused.
        self.assertEqual(_count_3d_concatenate(jaxpr_f.jaxpr), 0)
        self.assertEqual(
            _count_primitive(jaxpr_f.jaxpr, "concatenate"),
            _count_primitive(jaxpr_u.jaxpr, "concatenate"),
        )
      else:
        # When TP > 1, `wi` is sliced before TP resharding and re-concatenated per shard in `gmm_up`.
        self.assertEqual(_count_3d_concatenate(jaxpr_f.jaxpr), 1)

      grad_u_fn = jax.jit(jax.value_and_grad(loss_unfused, argnums=(0, 1), has_aux=True))
      grad_f_fn = jax.jit(jax.value_and_grad(loss_fused, argnums=(0, 1), has_aux=True))

      if ep_deg == 1:
        # Native unmocked execution & XLA HLO collective count check on 4 CPU devices (FSDP=4, EP=1, TP=1).
        unfused_res = grad_u_fn(params_u, x)
        fused_res = grad_f_fn(params_f, x)
        if tp_deg == 1 and fsdp_deg > 1:
          hlo_u = grad_u_fn.lower(params_u, x).compile().as_text()
          hlo_f = grad_f_fn.lower(params_f, x).compile().as_text()
          assert hlo_u is not None and hlo_f is not None
          ag_re = r"=\s*[^\n]*\ball-gather(?:-start)?\("
          red_args_re = r"=\s*[^\n]*\b(?:reduce-scatter|all-reduce)(?:-start)?\(([^)]*)\)"

          def _count_red_operands(hlo: str) -> int:
            return sum(1 for args in re.findall(red_args_re, hlo) for a in args.split(",") if a.strip())

          self.assertLess(len(re.findall(ag_re, hlo_f)), len(re.findall(ag_re, hlo_u)))
          # Count reduced operands rather than raw opcodes because XLA's combiner merges weight grads into one tuple.
          self.assertLess(_count_red_operands(hlo_f), _count_red_operands(hlo_u))
      else:
        with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=_cpu_ragged_all_to_all):
          unfused_res = grad_u_fn(params_u, x)
          fused_res = grad_f_fn(params_f, x)

      self._assert_routed_moe_parity(unfused_res, fused_res)

  def test_prefuse_moe_weights_expert_replication_parity(self):
    """Pre-fused TP=1 `wi` (`w1=None`) replicates in one call and matches unfused forward/backward."""
    if jax.device_count() != 4:
      self.skipTest("Requires 4 simulated CPU devices.")

    cfg_unfused = _make_moe_config(
        prefuse_moe_weights=False,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
        ici_tensor_parallelism=1,
        expert_replicas_per_device=1,
    )
    cfg_fused = _make_moe_config(
        prefuse_moe_weights=True,
        moe_unsliced_prefused_wi=True,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
        ici_tensor_parallelism=1,
        expert_replicas_per_device=1,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_unfused), cfg_unfused.mesh_axes)

    with (
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_unfused.logical_axis_rules),
        logical_axis_rules(cfg_unfused.logical_axis_rules),
    ):
      params_u, params_f, x, loss_unfused, loss_fused = self._build_routed_moe_pair(
          cfg_unfused, cfg_fused, mesh, ("fsdp", "expert")
      )

      with mock.patch.object(
          moe.moe_replica_weights,
          "extend_with_replicas",
          wraps=moe.moe_replica_weights.extend_with_replicas,
      ) as extend_mock:
        with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=_cpu_ragged_all_to_all):
          unfused_res = jax.jit(jax.value_and_grad(loss_unfused, argnums=(0, 1), has_aux=True))(params_u, x)
          unfused_calls = extend_mock.call_count
          extend_mock.reset_mock()
          fused_res = jax.jit(jax.value_and_grad(loss_fused, argnums=(0, 1), has_aux=True))(params_f, x)
          fused_calls = extend_mock.call_count

      self.assertEqual(unfused_calls, 3)
      self.assertEqual(fused_calls, 2)
      self._assert_routed_moe_parity(unfused_res, fused_res)

  @parameterized.named_parameters(
      ("force_dropless_standard_sort_f32", False, "float32", True, 1e-6),
      ("layer_fallback_ragged_sort_bf16", True, "bfloat16", False, 2e-2),
  )
  def test_ra2a_layer_dropless_fallback_and_force_dropless_combine(
      self,
      use_ragged_sort: bool,
      dtype_str: str,
      float32_gate_logits: bool,
      tol: float,
  ):
    if jax.device_count() != 4:
      self.skipTest("Requires 4 simulated CPU devices.")

    jnp_dtype = jnp.float32 if dtype_str == "float32" else jnp.bfloat16

    def build_model(mesh, rbf: float, fallback: str | None = None, force_dropless: bool = False):
      """Builds a `RoutedMoE` instance configured for ragged-all-to-all fallback comparisons."""
      cfg = _make_moe_config(
          run_name=f"ra2a_fallback_{use_ragged_sort}_{rbf}_{fallback}_{force_dropless}",
          ici_expert_parallelism=4,
          use_ring_of_experts=False,
          use_ragged_sort=use_ragged_sort,
          moe_use_direct_token_gather=False,
          prefuse_moe_weights=True,
          dtype=dtype_str,
          weight_dtype=dtype_str,
          float32_gate_logits=float32_gate_logits,
          ragged_buffer_factor=rbf,
          moe_dropless_fallback=fallback,
      )
      model = nnx_wrappers.to_linen(
          moe.RoutedMoE,
          name="MoeBlock",
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.base_moe_mlp_dim,
          dtype=jnp_dtype,
          weight_dtype=jnp_dtype,
          force_dropless=force_dropless,
      )
      return cfg, model

    cfg0 = _make_moe_config(ici_expert_parallelism=4, use_ring_of_experts=False)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg0), cfg0.mesh_axes)
    rng_x = np.random.RandomState(7)
    x = jnp.asarray(rng_x.randn(4, cfg0.max_target_length, cfg0.base_emb_dim), dtype=jnp_dtype)
    # Route every token's top-1 slot to expert 0 (owned by EP shard 0), forcing 2x shard-0 load > rbf=0.5.
    forced = jnp.broadcast_to(
        jnp.array([0, 1], dtype=jnp.int32),
        (4, cfg0.max_target_length, cfg0.num_experts_per_tok),
    )

    cfg_ref, model_ref = build_model(mesh, rbf=-1.0, fallback=None)
    _, model_drop = build_model(mesh, rbf=0.5, fallback=None)
    _, model_force = build_model(mesh, rbf=0.5, fallback=None, force_dropless=True)
    if use_ragged_sort:
      _, model_fb = build_model(mesh, rbf=0.5, fallback="layer")
    else:
      model_fb = None
      with self.assertRaisesRegex(ValueError, "requires use_ragged_sort=True"):
        build_model(mesh, rbf=0.5, fallback="layer")

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=_cpu_ragged_all_to_all):
        variables = model_ref.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
            x,
            forced_routed_experts=forced,
        )
        params = variables["params"]

        def loss_and_out(model, p, x_in, return_mutated: bool = False):
          if return_mutated:
            (out, lb, _), mutated = model.apply(
                {"params": p}, x_in, forced_routed_experts=forced, mutable=["intermediates"]
            )
          else:
            out, lb, _ = model.apply({"params": p}, x_in, forced_routed_experts=forced)
            mutated = None
          loss = jnp.mean(out.astype(jnp.float32) ** 2) + (0.0 if lb is None else lb.astype(jnp.float32))
          return loss, (out, mutated)

        (loss_ref, (out_ref, _)), (dp_ref, dx_ref) = jax.jit(
            jax.value_and_grad(functools.partial(loss_and_out, model_ref), argnums=(0, 1), has_aux=True)
        )(params, x)

        out_drop, _, _ = jax.jit(model_drop.apply)({"params": params}, x, forced_routed_experts=forced)
        self.assertFalse(jnp.allclose(out_drop.astype(jnp.float32), out_ref.astype(jnp.float32), rtol=5e-2, atol=5e-2))

        (loss_force, (out_force, _)), (dp_force, dx_force) = jax.jit(
            jax.value_and_grad(functools.partial(loss_and_out, model_force), argnums=(0, 1), has_aux=True)
        )(params, x)
        np.testing.assert_allclose(
            np.asarray(loss_force, dtype=np.float32), np.asarray(loss_ref, dtype=np.float32), rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            np.asarray(out_force, dtype=np.float32), np.asarray(out_ref, dtype=np.float32), rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            np.asarray(dx_force, dtype=np.float32), np.asarray(dx_ref, dtype=np.float32), rtol=tol, atol=tol
        )
        jax.tree.map(
            lambda gf, gr: np.testing.assert_allclose(
                np.asarray(gf, dtype=np.float32), np.asarray(gr, dtype=np.float32), rtol=tol, atol=tol
            ),
            dp_force,
            dp_ref,
        )

        if model_fb is not None:
          (loss_fb, (out_fb, mutated)), (dp_fb, dx_fb) = jax.jit(
              jax.value_and_grad(
                  functools.partial(loss_and_out, model_fb, return_mutated=True),
                  argnums=(0, 1),
                  has_aux=True,
              )
          )(params, x)
          took_fb = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_dropless_fallback")
          self.assertTrue(bool(jnp.any(jnp.array([jnp.any(v) for v in took_fb]))))
          np.testing.assert_allclose(
              np.asarray(loss_fb, dtype=np.float32), np.asarray(loss_ref, dtype=np.float32), rtol=tol, atol=tol
          )
          np.testing.assert_allclose(
              np.asarray(out_fb, dtype=np.float32), np.asarray(out_ref, dtype=np.float32), rtol=tol, atol=tol
          )
          np.testing.assert_allclose(
              np.asarray(dx_fb, dtype=np.float32), np.asarray(dx_ref, dtype=np.float32), rtol=tol, atol=tol
          )
          jax.tree.map(
              lambda gf, gr: np.testing.assert_allclose(
                  np.asarray(gf, dtype=np.float32), np.asarray(gr, dtype=np.float32), rtol=tol, atol=tol
              ),
              dp_fb,
              dp_ref,
          )

      # Outside the CPU mock, verify force_dropless=True skips combine-stage `_truncate_matrix` (`cumsum`)
      # and `jnp.zeros` init on the real `ragged_all_to_all` Jaxpr, matching the dropless reference.
      jaxpr_ref = jax.make_jaxpr(lambda p, x_in: model_ref.apply({"params": p}, x_in))(params, x)
      jaxpr_force = jax.make_jaxpr(lambda p, x_in: model_force.apply({"params": p}, x_in))(params, x)
      jaxpr_drop = jax.make_jaxpr(lambda p, x_in: model_drop.apply({"params": p}, x_in))(params, x)
      self.assertEqual(_count_primitive(jaxpr_force.jaxpr, "cumsum"), _count_primitive(jaxpr_ref.jaxpr, "cumsum"))
      self.assertLess(_count_primitive(jaxpr_force.jaxpr, "cumsum"), _count_primitive(jaxpr_drop.jaxpr, "cumsum"))
      self.assertEqual(_count_primitive(jaxpr_force.jaxpr, "empty"), _count_primitive(jaxpr_ref.jaxpr, "empty"))
      self.assertGreater(_count_primitive(jaxpr_force.jaxpr, "empty"), _count_primitive(jaxpr_drop.jaxpr, "empty"))

  @parameterized.named_parameters(
      ("maxtext_muon_trainable_wi", "maxtext_muon", False),
      ("maxtext_muon_frozen_wi", "maxtext_muon", True),
      ("optax_muon_trainable_wi", "optax_muon", False),
      ("optax_muon_frozen_wi", "optax_muon", True),
  )
  def test_prefused_wi_muon_weight_decay_and_trainable_mask_parity(self, muon_type: str, freeze_wi: bool):
    class _UnfusedMoeSub(nnx.Module):

      def __init__(self, w0, w1, wo, gate_k):
        self.wi_0 = nnx.Param(w0, out_sharding=("exp", "embed_moe", "mlp_moe"))
        self.wi_1 = nnx.Param(w1, out_sharding=("exp", "embed_moe", "mlp_moe"))
        self.wo = nnx.Param(wo, out_sharding=("exp", "mlp_moe", "embed_moe"))
        self.gate = _FakeGate(gate_k)

    class _PrefusedMoeSub(nnx.Module):

      def __init__(self, w0, w1, wo, gate_k):
        self.wi = nnx.Param(jnp.concatenate([w0, w1], axis=-1), out_sharding=("exp", "embed_moe", "mlp_moe"))
        self.wo = nnx.Param(wo, out_sharding=("exp", "mlp_moe", "embed_moe"))
        self.gate = _FakeGate(gate_k)

    class _FakeGate(nnx.Module):

      def __init__(self, gate_k):
        self.kernel = nnx.Param(gate_k, out_sharding=("embed", "mlp"))

    class _UnfusedModel(nnx.Module):

      def __init__(self, w0, w1, wo, gate_k, scale):
        self.MoeBlock_0 = _UnfusedMoeSub(w0, w1, wo, gate_k)  # pylint: disable=invalid-name
        self.scale = nnx.Param(scale, out_sharding=(None,))

    class _PrefusedModel(nnx.Module):

      def __init__(self, w0, w1, wo, gate_k, scale):
        self.MoeBlock_0 = _PrefusedMoeSub(w0, w1, wo, gate_k)  # pylint: disable=invalid-name
        self.scale = nnx.Param(scale, out_sharding=(None,))

    keys = jax.random.split(jax.random.key(41), 10)
    w0 = jax.random.normal(keys[0], (4, 8, 6), dtype=jnp.float32)
    w1 = jax.random.normal(keys[1], (4, 8, 6), dtype=jnp.float32)
    wo = jax.random.normal(keys[2], (4, 6, 8), dtype=jnp.float32)
    gate_k = jax.random.normal(keys[3], (8, 4), dtype=jnp.float32)
    scale = jnp.ones((8,), dtype=jnp.float32)
    wi_init = jnp.concatenate([w0, w1], axis=-1)

    model_u = _UnfusedModel(w0, w1, wo, gate_k, scale)
    model_f = _PrefusedModel(w0, w1, wo, gate_k, scale)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1, 1)), ("data", "model"))

    if freeze_wi:
      # Train only `wo` and `gate`; freeze `wi`/`wi_0`/`wi_1` and `scale` (`MaskedNode` on `wi`).
      # Allow optional `/value` suffix so `nnx.state` (init) and `nnx.as_pure` (update) both match.
      trainable_mask = [r"^MoeBlock_0/(wo|gate/kernel)(/value)?$"]
    else:
      # Train `wi`/`wi_0`/`wi_1` and `wo`; freeze `gate` and `scale` (`MaskedNode` on non-`wi` leaves).
      trainable_mask = [r"^MoeBlock_0/(wi|wi_0|wi_1|wo)(/value)?$"]

    cfg_u = _make_deepseek_moe_config(
        prefuse_moe_weights=False,
        muon_type=muon_type,
        muon_use_all_to_all=False,
        muon_weight_decay=0.05,
        adam_weight_decay=0.1,
        trainable_parameters_mask=trainable_mask,
    )
    cfg_f = _make_deepseek_moe_config(
        prefuse_moe_weights=True,
        muon_type=muon_type,
        muon_use_all_to_all=False,
        muon_weight_decay=0.05,
        adam_weight_decay=0.1,
        trainable_parameters_mask=trainable_mask,
    )
    tx_u = optimizers.get_optimizer(cfg_u, optax.constant_schedule(1e-2), model=model_u, mesh=mesh)
    tx_f = optimizers.get_optimizer(cfg_f, optax.constant_schedule(1e-2), model=model_f, mesh=mesh)
    opt_u = nnx.Optimizer(model_u, tx_u, wrt=nnx.Param)
    opt_f = nnx.Optimizer(model_f, tx_f, wrt=nnx.Param)

    f_shapes = [x.shape for x in jax.tree.leaves(nnx.as_pure(nnx.state(opt_f))) if hasattr(x, "shape")]
    self.assertNotIn((4, 8, 6), f_shapes)
    if freeze_wi:
      self.assertNotIn((4, 8, 12), f_shapes)
    else:
      self.assertIn((4, 8, 12), f_shapes)

    for step in range(2):
      step_keys = jax.random.split(jax.random.fold_in(keys[9], step), 5)
      g_w0 = jax.random.normal(step_keys[0], w0.shape, dtype=jnp.float32)
      g_w1 = jax.random.normal(step_keys[1], w1.shape, dtype=jnp.float32)
      g_wo = jax.random.normal(step_keys[2], wo.shape, dtype=jnp.float32)
      g_gate = jax.random.normal(step_keys[3], gate_k.shape, dtype=jnp.float32)
      g_scale = jax.random.normal(step_keys[4], scale.shape, dtype=jnp.float32)

      grads_u = nnx.clone(nnx.state(model_u, nnx.Param))
      grads_u["MoeBlock_0"]["wi_0"].set_value(g_w0)
      grads_u["MoeBlock_0"]["wi_1"].set_value(g_w1)
      grads_u["MoeBlock_0"]["wo"].set_value(g_wo)
      grads_u["MoeBlock_0"]["gate"]["kernel"].set_value(g_gate)
      grads_u["scale"].set_value(g_scale)

      grads_f = nnx.clone(nnx.state(model_f, nnx.Param))
      grads_f["MoeBlock_0"]["wi"].set_value(jnp.concatenate([g_w0, g_w1], axis=-1))
      grads_f["MoeBlock_0"]["wo"].set_value(g_wo)
      grads_f["MoeBlock_0"]["gate"]["kernel"].set_value(g_gate)
      grads_f["scale"].set_value(g_scale)

      opt_u.update(model_u, grads_u)
      opt_f.update(model_f, grads_f)

    expected_wi = jnp.concatenate(
        [model_u.MoeBlock_0.wi_0.get_value(), model_u.MoeBlock_0.wi_1.get_value()],
        axis=-1,
    )
    if freeze_wi:
      np.testing.assert_array_equal(model_f.MoeBlock_0.wi.get_value(), wi_init)
      np.testing.assert_array_equal(expected_wi, wi_init)
      self.assertFalse(jnp.allclose(model_f.MoeBlock_0.gate.kernel.get_value(), gate_k))
    else:
      self.assertFalse(jnp.allclose(model_f.MoeBlock_0.wi.get_value(), wi_init))
      np.testing.assert_array_equal(model_f.MoeBlock_0.gate.kernel.get_value(), gate_k)
    np.testing.assert_array_equal(model_f.scale.get_value(), scale)
    np.testing.assert_allclose(model_f.MoeBlock_0.wi.get_value(), expected_wi, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(model_f.MoeBlock_0.wo.get_value(), model_u.MoeBlock_0.wo.get_value(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        model_f.MoeBlock_0.gate.kernel.get_value(), model_u.MoeBlock_0.gate.kernel.get_value(), rtol=1e-6, atol=1e-6
    )

  @parameterized.named_parameters(
      ("optax_muon", "optax_muon"),
      ("maxtext_muon", "maxtext_muon"),
  )
  def test_prefuse_moe_weights_muon_param_scan_axis_last(self, muon_type: str):
    """Pre-fused MoE `wi` with `param_scan_axis=3` (`output_axis=(-2,)`) splits along the true output axis."""
    scan_meta = {nnx.PARTITION_NAME: "layers", "param_scan_axis": 3}

    class _ScannedUnfusedMoeSub(nnx.Module):

      def __init__(self, w0, w1, wo):
        self.wi_0 = nnx.Param(w0, **scan_meta)
        self.wi_1 = nnx.Param(w1, **scan_meta)
        self.wo = nnx.Param(wo, **scan_meta)

    class _ScannedPrefusedMoeSub(nnx.Module):

      def __init__(self, w0, w1, wo):
        self.wi = nnx.Param(jnp.concatenate([w0, w1], axis=2), **scan_meta)
        self.wo = nnx.Param(wo, **scan_meta)

    class _ScannedUnfusedModel(nnx.Module):

      def __init__(self, w0, w1, wo):
        self.MoeBlock_0 = _ScannedUnfusedMoeSub(w0, w1, wo)  # pylint: disable=invalid-name

    class _ScannedPrefusedModel(nnx.Module):

      def __init__(self, w0, w1, wo):
        self.MoeBlock_0 = _ScannedPrefusedMoeSub(w0, w1, wo)  # pylint: disable=invalid-name

    keys = jax.random.split(jax.random.key(53), 6)
    w0 = jax.random.normal(keys[0], (2, 8, 6, 4), dtype=jnp.float32)
    w1 = jax.random.normal(keys[1], (2, 8, 6, 4), dtype=jnp.float32)
    wo = jax.random.normal(keys[2], (2, 6, 8, 4), dtype=jnp.float32)

    model_u = _ScannedUnfusedModel(w0, w1, wo)
    model_f = _ScannedPrefusedModel(w0, w1, wo)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1, 1)), ("data", "model"))

    cfg_u = _make_deepseek_moe_config(
        prefuse_moe_weights=False,
        muon_type=muon_type,
        muon_use_all_to_all=False,
        muon_weight_decay=0.05,
        adam_weight_decay=0.1,
    )
    cfg_f = _make_deepseek_moe_config(
        prefuse_moe_weights=True,
        muon_type=muon_type,
        muon_use_all_to_all=False,
        muon_weight_decay=0.05,
        adam_weight_decay=0.1,
    )
    tx_u = optimizers.get_optimizer(cfg_u, optax.constant_schedule(1e-2), model=model_u, mesh=mesh)
    tx_f = optimizers.get_optimizer(cfg_f, optax.constant_schedule(1e-2), model=model_f, mesh=mesh)
    opt_u = nnx.Optimizer(model_u, tx_u, wrt=nnx.Param)
    opt_f = nnx.Optimizer(model_f, tx_f, wrt=nnx.Param)

    f_shapes = [x.shape for x in jax.tree.leaves(nnx.as_pure(nnx.state(opt_f))) if hasattr(x, "shape")]
    self.assertIn((2, 8, 12, 4), f_shapes)
    self.assertNotIn((2, 8, 6, 4), f_shapes)

    g_w0 = jax.random.normal(keys[3], w0.shape, dtype=jnp.float32)
    g_w1 = jax.random.normal(keys[4], w1.shape, dtype=jnp.float32)
    g_wo = jax.random.normal(keys[5], wo.shape, dtype=jnp.float32)

    grads_u = nnx.clone(nnx.state(model_u, nnx.Param))
    grads_u["MoeBlock_0"]["wi_0"].set_value(g_w0)
    grads_u["MoeBlock_0"]["wi_1"].set_value(g_w1)
    grads_u["MoeBlock_0"]["wo"].set_value(g_wo)

    grads_f = nnx.clone(nnx.state(model_f, nnx.Param))
    grads_f["MoeBlock_0"]["wi"].set_value(jnp.concatenate([g_w0, g_w1], axis=2))
    grads_f["MoeBlock_0"]["wo"].set_value(g_wo)

    opt_u.update(model_u, grads_u)
    opt_f.update(model_f, grads_f)

    expected_wi = jnp.concatenate([model_u.MoeBlock_0.wi_0.get_value(), model_u.MoeBlock_0.wi_1.get_value()], axis=2)
    np.testing.assert_allclose(model_f.MoeBlock_0.wi.get_value(), expected_wi, rtol=1e-5, atol=1e-5)

  def test_prefuse_moe_weights_optax_muon_odd_output_axis_raises(self):
    """Odd pre-fused MoE `wi` output axis raises ValueError instead of silent asymmetric split."""

    class _OddWiSub(nnx.Module):

      def __init__(self):
        self.wi = nnx.Param(
            jnp.ones((2, 4, 7, 4), dtype=jnp.float32),
            **{nnx.PARTITION_NAME: "layers", "param_scan_axis": 3},
        )

    class _OddWiModel(nnx.Module):

      def __init__(self):
        self.MoeBlock_0 = _OddWiSub()  # pylint: disable=invalid-name

    model = _OddWiModel()
    cfg = _make_deepseek_moe_config(
        prefuse_moe_weights=True,
        muon_type="optax_muon",
        muon_use_all_to_all=False,
    )
    tx = optimizers.get_optimizer(cfg, optax.constant_schedule(1e-2), model=model)
    with self.assertRaisesRegex(ValueError, "output axis 2 of size 7 is not divisible by 2"):
      nnx.Optimizer(model, tx, wrt=nnx.Param)

  def test_prefuse_moe_weights_optax_muon_pure_dict_and_scalar_output_axis(self):
    """optax_muon wrapper supports pure dict trees, scalar int output_axis, and params=None."""

    class _WiSub(nnx.Module):

      def __init__(self, w0, w1):
        self.wi = nnx.Param(jnp.concatenate([w0, w1], axis=-1))

    class _Model(nnx.Module):

      def __init__(self, w0, w1):
        self.MoeBlock_0 = _WiSub(w0, w1)  # pylint: disable=invalid-name

    keys = jax.random.split(jax.random.key(53), 4)
    w0 = jax.random.normal(keys[0], (2, 8, 6), dtype=jnp.float32)
    w1 = jax.random.normal(keys[1], (2, 8, 6), dtype=jnp.float32)
    g0 = jax.random.normal(keys[2], (2, 8, 6), dtype=jnp.float32)
    g1 = jax.random.normal(keys[3], (2, 8, 6), dtype=jnp.float32)
    model = _Model(w0, w1)
    cfg = _make_deepseek_moe_config(
        prefuse_moe_weights=True,
        muon_type="optax_muon",
        muon_use_all_to_all=False,
        muon_weight_decay=0.0,
        adam_weight_decay=0.0,
    )
    scalar_dn = MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
    with mock.patch.object(
        optimizers,
        "get_muon_weight_dimension_numbers",
        return_value={"MoeBlock_0": {"wi": scalar_dn}},
    ):
      tx = optimizers.get_optimizer(cfg, optax.constant_schedule(1e-2), model=model)

    pure_params = {"MoeBlock_0": {"wi": jnp.concatenate([w0, w1], axis=-1)}}
    pure_grads = {"MoeBlock_0": {"wi": jnp.concatenate([g0, g1], axis=-1)}}
    state = tx.init(pure_params)
    # Passing params=None forwards `None` cleanly to optax_muon's own validation (no tree-map crash).
    with self.assertRaisesRegex(ValueError, "requires the current value of parameters"):
      tx.update(pure_grads, state, None)
    updates, new_state = tx.update(pure_grads, state, pure_params)
    self.assertEqual(updates["MoeBlock_0"]["wi"].shape, (2, 8, 12))
    self.assertIn((2, 8, 12), [x.shape for x in jax.tree.leaves(new_state) if hasattr(x, "shape")])

    cfg_u = _make_deepseek_moe_config(
        prefuse_moe_weights=False,
        muon_type="optax_muon",
        muon_use_all_to_all=False,
        muon_weight_decay=0.0,
        adam_weight_decay=0.0,
    )
    with mock.patch.object(
        optimizers,
        "get_muon_weight_dimension_numbers",
        return_value={"MoeBlock_0": {"wi_0": scalar_dn, "wi_1": scalar_dn}},
    ):
      ref_tx = optimizers.get_optimizer(cfg_u, optax.constant_schedule(1e-2), model=model)
    ref_params = {"MoeBlock_0": {"wi_0": w0, "wi_1": w1}}
    ref_state = ref_tx.init(ref_params)
    ref_updates, _ = ref_tx.update({"MoeBlock_0": {"wi_0": g0, "wi_1": g1}}, ref_state, ref_params)
    expected = jnp.concatenate([ref_updates["MoeBlock_0"]["wi_0"], ref_updates["MoeBlock_0"]["wi_1"]], axis=-1)
    np.testing.assert_allclose(updates["MoeBlock_0"]["wi"], expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("maxtext_muon_no_a2a", "maxtext", False),
      ("maxtext_muon_a2a", "maxtext", True),
      ("optax_muon", "optax", False),
  )
  def test_deepseek_moe_decoder_layer_muon_prefuse_parity(self, muon_type: str, use_all_to_all: bool):
    if jax.device_count() != 4:
      self.skipTest("Requires 4 simulated CPU devices.")

    cfg_unfused = _make_deepseek_moe_config(
        prefuse_moe_weights=False,
        muon_type=muon_type,
        muon_use_all_to_all=use_all_to_all,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
    )
    cfg_fused = _make_deepseek_moe_config(
        prefuse_moe_weights=True,
        muon_type=muon_type,
        muon_use_all_to_all=use_all_to_all,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_unfused), cfg_unfused.mesh_axes)
    batch, seq = 4, 6
    x = jnp.asarray(np.random.RandomState(19).randn(batch, seq, 32), dtype=jnp.float32)
    pos = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    seg = jnp.ones((batch, seq), dtype=jnp.int32)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_unfused.logical_axis_rules):
      layer_u = deepseek.DeepSeekMoELayer(
          config=cfg_unfused,
          model_mode=MODEL_MODE_TRAIN,
          mesh=mesh,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      layer_f = deepseek.DeepSeekMoELayer(
          config=cfg_fused,
          model_mode=MODEL_MODE_TRAIN,
          mesh=mesh,
          rngs=nnx.Rngs(params=0, dropout=1),
      )

      # Copy all shared/attention/norm/router weights identically and fuse [wi_0 | wi_1] -> wi.
      _, state_u = nnx.split(layer_u)
      gdef_f, state_f = nnx.split(layer_f)
      pure_u = nnx.to_pure_dict(state_u)
      pure_f = nnx.to_pure_dict(state_f)
      for key, val in pure_u.items():
        if key != "DeepSeekMoeBlock_0":
          pure_f[key] = val
      pure_f["DeepSeekMoeBlock_0"]["shared_experts"] = pure_u["DeepSeekMoeBlock_0"]["shared_experts"]
      pure_f["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"] = pure_u["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]
      pure_f["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wo"] = pure_u["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wo"]
      pure_f["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi"] = jnp.concatenate(
          [
              pure_u["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"],
              pure_u["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_1"],
          ],
          axis=-1,
      )
      nnx.replace_by_pure_dict(state_f, pure_f)
      layer_f = nnx.merge(gdef_f, state_f)

      lr_sched = optax.constant_schedule(cfg_unfused.learning_rate)
      tx_u = optimizers.get_optimizer(cfg_unfused, lr_sched, model=layer_u, mesh=mesh)
      tx_f = optimizers.get_optimizer(cfg_fused, lr_sched, model=layer_f, mesh=mesh)
      opt_u = nnx.Optimizer(layer_u, tx_u, wrt=nnx.Param)
      opt_f = nnx.Optimizer(layer_f, tx_f, wrt=nnx.Param)

      # Confirm optimizer state keeps pre-fused `wi` momentum as a single (4, 32, 32) tensor.
      opt_f_shapes = [leaf.shape for leaf in jax.tree.leaves(nnx.as_pure(nnx.state(opt_f))) if hasattr(leaf, "shape")]
      self.assertIn((4, 32, 32), opt_f_shapes)

      with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=_cpu_ragged_all_to_all):

        @nnx.jit
        def run_step(m, opt, tgt):
          def step_loss(model):
            out, _ = model(x, seg, pos, deterministic=True, model_mode=MODEL_MODE_TRAIN)
            return jnp.mean((out - tgt) ** 2)

          grads = nnx.grad(step_loss)(m)
          opt.update(m, grads)

        @nnx.jit
        def eval_layer(m):
          out, _ = m(x, seg, pos, deterministic=True, model_mode=MODEL_MODE_TRAIN)
          return out

        for step in range(2):
          target = jnp.asarray(
              np.random.RandomState(100 + step).randn(batch, seq, 32),
              dtype=jnp.float32,
          )
          run_step(layer_u, opt_u, target)
          run_step(layer_f, opt_f, target)

        moe_u = layer_u.DeepSeekMoeBlock_0.MoeBlock_0
        assert moe_u.wi_0 is not None and moe_u.wi_1 is not None
        expected_wi = jnp.concatenate([moe_u.wi_0[...], moe_u.wi_1[...]], axis=-1)
        np.testing.assert_allclose(
            np.asarray(layer_f.DeepSeekMoeBlock_0.MoeBlock_0.wi[...]),
            np.asarray(expected_wi),
            rtol=1e-5,
            atol=1e-5,
        )

        out_u_post = eval_layer(layer_u)
        out_f_post = eval_layer(layer_f)
        np.testing.assert_allclose(np.asarray(out_f_post), np.asarray(out_u_post), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
