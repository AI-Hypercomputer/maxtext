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

"""Unit and multi-device parity tests for `maxtext.m3.core.moe`."""

import dataclasses
import functools
import math
import os
import re
from types import SimpleNamespace
import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2

# pylint: disable=too-many-positional-arguments
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.kernels.ragged import ragged_sort
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core import moe as m3_moe
from maxtext.m3.core.sharding import DefaultSharding, create_mesh, param_metadata
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3 import models as m3_models
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
from maxtext.m3.train import loop as m3_train_loop
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import optax
import pytest


def _make_routing_inputs(
    batch: int = 4,
    length: int = 6,
    embed: int = 16,
    num_experts: int = 4,
    top_k: int = 2,
    *,
    dtype=jnp.float32,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds deterministic `(x, weights, indices, probs)` routing inputs."""
  k0, k1, k2 = jax.random.split(jax.random.key(seed), 3)
  x = jax.random.normal(k0, (batch, length, embed), dtype=dtype)
  logits = jax.random.normal(k1, (batch, length, num_experts), dtype=jnp.float32)
  probs = jax.nn.softmax(logits, axis=-1)
  top_vals, indices = jax.lax.top_k(probs, top_k)
  weights = (top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)).astype(dtype)
  # Ensure deterministic tie-free unique indices per token even under random collisions.
  noise = jax.random.uniform(k2, logits.shape, minval=0.0, maxval=1e-3)
  _, indices = jax.lax.top_k(probs + noise, top_k)
  return x, weights, indices.astype(jnp.int32), probs


def _make_expert_weights(
    num_experts: int = 4,
    embed: int = 16,
    mlp: int = 12,
    *,
    dtype=jnp.float32,
    seed: int = 10,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Builds deterministic `(wi_0, wi_1, wo)` expert weights with `1 / sqrt(fan_in)` scaling."""
  k0, k1, k2 = jax.random.split(jax.random.key(seed), 3)
  in_scale = 1.0 / math.sqrt(embed)
  mlp_scale = 1.0 / math.sqrt(mlp)
  wi_0 = (jax.random.normal(k0, (num_experts, embed, mlp), dtype=jnp.float32) * in_scale).astype(dtype)
  wi_1 = (jax.random.normal(k1, (num_experts, embed, mlp), dtype=jnp.float32) * in_scale).astype(dtype)
  wo = (jax.random.normal(k2, (num_experts, mlp, embed), dtype=jnp.float32) * mlp_scale).astype(dtype)
  return wi_0, wi_1, wo


def _cpu_ragged_all_to_all_ref(
    operand: jax.Array,
    output: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    axis_name: str,
) -> jax.Array:
  """Pure-JAX collective reference matching `jax.lax.ragged_all_to_all` semantics on CPU."""
  ep_size = send_sizes.shape[0]
  in_len = operand.shape[0]
  out_len = output.shape[0]
  cap = max(1, min(int(in_len), int(out_len)))
  col = jnp.arange(cap, dtype=jnp.int32)[None, :]
  gather_idx = input_offsets[:, None] + col
  valid_send = col < send_sizes[:, None]
  safe_gather = jnp.clip(gather_idx, 0, max(int(in_len) - 1, 0))
  bcast_send = jnp.reshape(valid_send, (ep_size, cap) + (1,) * (operand.ndim - 1))
  staged = jnp.where(bcast_send, operand[safe_gather], jnp.zeros((), dtype=operand.dtype))
  recvd = jax.lax.all_to_all(staged, axis_name, split_axis=0, concat_axis=0)
  local_out_offsets = jax.lax.all_to_all(output_offsets[:, None], axis_name, split_axis=0, concat_axis=0)[:, 0]

  row_idx = jnp.arange(out_len, dtype=jnp.int32)
  out = output
  for src in range(ep_size):
    rel = row_idx - local_out_offsets[src]
    in_chunk = (rel >= 0) & (rel < recv_sizes[src])
    safe_rel = jnp.clip(rel, 0, cap - 1)
    bcast_chunk = jnp.reshape(in_chunk, (out_len,) + (1,) * (operand.ndim - 1))
    out = jnp.where(bcast_chunk, recvd[src, safe_rel], out)
  return out


class _ToyMoELayer(nnx.Module):
  """Minimal NNX MoE layer used to verify `ScannedLayers` + `MoEExecutor` integration."""

  def __init__(self, num_experts: int, embed: int, mlp: int, config, sharding: DefaultSharding, *, rngs: nnx.Rngs):
    self.num_experts = num_experts
    self.executor = m3_moe.moe_from_config(config, sharding)
    self.router = nnx.Linear(
        embed,
        num_experts,
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        kernel_metadata=param_metadata(sharding, "router", ("embed", None)),
        rngs=rngs,
    )
    expert_init = nnx.initializers.lecun_normal(batch_axis=(0,))
    self.wi_0 = nnx.Param(
        expert_init(rngs.params(), (num_experts, embed, mlp), jnp.float32),
        **param_metadata(sharding, "wi_0", ("experts", "expert_embed", "expert_mlp")),
    )
    self.wi_1 = nnx.Param(
        expert_init(rngs.params(), (num_experts, embed, mlp), jnp.float32),
        **param_metadata(sharding, "wi_1", ("experts", "expert_embed", "expert_mlp")),
    )
    self.wo = nnx.Param(
        expert_init(rngs.params(), (num_experts, mlp, embed), jnp.float32),
        **param_metadata(sharding, "wo", ("experts", "expert_mlp", "expert_embed")),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    probs = jax.nn.softmax(self.router(x), axis=-1)
    vals, idx = jax.lax.top_k(probs, 2)
    weights = vals / jnp.sum(vals, axis=-1, keepdims=True)
    return x + self.executor(x, weights, idx, self.wi_0[...], self.wi_1[...], self.wo[...])


class ValidateAndResolveMoETest(parameterized.TestCase):
  """Tests static `MoEExecutor` resolution, NNX graphdef staticness, and fail-closed validation."""

  def test_resolves_all_four_executor_kinds_and_aliases(self):
    self.assertIs(m3_moe.Executor, m3_moe.MoEExecutor)
    self.assertIs(m3_moe.executor_from_config, m3_moe.moe_from_config)

    dense_exec = m3_moe.moe_from_config(SimpleNamespace(sparse_matmul=False, capacity_factor=1.25))
    self.assertEqual(dense_exec.kind, "dense")
    self.assertEqual(dense_exec.capacity_factor, 1.25)
    self.assertIsNone(dense_exec.experts)

    ragged_exec = m3_moe.moe_from_config(
        SimpleNamespace(sparse_matmul=True, use_tokamax_gmm=False, moe_bwd_wgrad_overlap=True)
    )
    self.assertEqual(ragged_exec.kind, "ragged_dot")
    self.assertTrue(ragged_exec.bwd_wgrad_overlap)

    tok_exec = m3_moe.moe_from_config(
        SimpleNamespace(
            sparse_matmul=True,
            use_tokamax_gmm=True,
            use_ragged_sort=True,
            moe_bwd_wgrad_overlap=True,
            ragged_buffer_factor=1.5,
        )
    )
    self.assertEqual(tok_exec.batch, ("data", "fsdp", "fsdp_transpose", "expert"))
    self.assertEqual(tok_exec.experts, "expert")
    self.assertEqual(tok_exec.mlp, ("tensor", "tensor_sequence"))
    self.assertEqual(tok_exec.kind, "tokamax")
    self.assertTrue(tok_exec.use_ragged_sort)
    self.assertTrue(tok_exec.bwd_wgrad_overlap)
    self.assertEqual(tok_exec.ragged_buffer_factor, 1.5)
    self.assertIsInstance(hash(tok_exec), int)

    # Forward compatibility once legacy `megablox`/`use_tokamax_gmm` flags are removed (#6114):
    post_6114_exec = m3_moe.moe_from_config(SimpleNamespace(num_experts=4, sparse_matmul=True, ici_expert_parallelism=2))
    self.assertEqual(post_6114_exec.kind, "gmm_v2")
    self.assertEqual(post_6114_exec.experts, "expert")
    # Dense models (num_experts=1) with TP > 1 remain valid under validate_moe_config.
    m3_moe.validate_moe_config(
        SimpleNamespace(
            num_experts=1, sparse_matmul=True, use_tokamax_gmm=False, megablox=False, ici_tensor_parallelism=2
        )
    )
    # Direct moe_from_config without num_experts still fails closed on unsharded ragged_dot with TP > 1.
    with self.assertRaisesRegex(ValueError, "tensor parallelism.*gmm_v2 or tokamax"):
      m3_moe.moe_from_config(SimpleNamespace(sparse_matmul=True, use_tokamax_gmm=False, ici_tensor_parallelism=2))

  def test_survives_nnx_split_and_merge_as_static_metadata(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    cfg = SimpleNamespace(sparse_matmul=True, use_tokamax_gmm=True, moe_bwd_wgrad_overlap=True)
    with jax.set_mesh(mesh):
      layer = _ToyMoELayer(4, 16, 12, cfg, DefaultSharding(), rngs=nnx.Rngs(0))
      graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
      self.assertEqual(len(jax.tree.leaves(rest)), 0)
      self.assertEqual(len(jax.tree.leaves(params)), 4)
      restored = nnx.merge(graphdef, params, rest)
      self.assertEqual(restored.executor, layer.executor)

  @parameterized.named_parameters(
      ("zero_capacity", SimpleNamespace(num_experts=4, capacity_factor=0.0), ValueError, "capacity_factor"),
      (
          "zero_ragged_buffer",
          SimpleNamespace(num_experts=4, ragged_buffer_factor=0.0),
          ValueError,
          "ragged_buffer_factor",
      ),
      (
          "ring_of_experts",
          SimpleNamespace(num_experts=4, use_ring_of_experts=True),
          NotImplementedError,
          "use_ring_of_experts",
      ),
      (
          "ragged_sort_without_gmm",
          SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=False, use_ragged_sort=True),
          ValueError,
          "use_ragged_sort",
      ),
      (
          "overlap_without_sparse",
          SimpleNamespace(num_experts=4, sparse_matmul=False, moe_bwd_wgrad_overlap=True),
          ValueError,
          "moe_bwd_wgrad_overlap",
      ),
      (
          "ep_without_sharded_backend",
          SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=False, ici_expert_parallelism=2),
          ValueError,
          "Expert parallelism.*gmm_v2 or tokamax",
      ),
      (
          "tp_without_sharded_backend",
          SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=False, ici_tensor_parallelism=2),
          ValueError,
          "tensor parallelism.*gmm_v2 or tokamax",
      ),
      (
          "tsp_without_sharded_backend",
          SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=False, ici_tensor_sequence_parallelism=2),
          ValueError,
          "tensor parallelism.*gmm_v2 or tokamax",
      ),
      (
          "dcn_tp_without_sharded_backend",
          SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=False, dcn_tensor_parallelism=2),
          ValueError,
          "tensor parallelism.*gmm_v2 or tokamax",
      ),
      (
          "dcn_tsp_without_sharded_backend",
          SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=False, dcn_tensor_sequence_parallelism=2),
          ValueError,
          "tensor parallelism.*gmm_v2 or tokamax",
      ),
  )
  def test_fails_closed_on_invalid_configs(self, cfg, exc_type, pattern):
    with self.assertRaisesRegex(exc_type, pattern):
      m3_moe.validate_moe_config(cfg)
    with self.assertRaisesRegex(exc_type, pattern):
      m3_moe.moe_from_config(cfg)

  def test_dual_driver_entrypoints_validate_moe_config(self):
    bad_cfg = SimpleNamespace(
        use_m3_model=True,
        model_name="qwen3.8-flash-next-mini",
        num_experts=8,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_ring_of_experts=True,
    )
    mesh = create_mesh(devices=jax.devices()[:1])
    with self.assertRaisesRegex(NotImplementedError, "use_ring_of_experts"):
      model_creation_utils.create_model(bad_cfg, mesh, rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(NotImplementedError, "use_ring_of_experts"):
      m3_train_loop.train_loop(bad_cfg)

  def test_cpu_pyconfig_ep_tp_and_ragged_sort_fallback(self):
    base_yml = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    for overrides, expect_rsort in (
        (["ici_expert_parallelism=2"], False),
        (["ici_tensor_parallelism=2"], False),
        (["ici_expert_parallelism=2", "ici_tensor_parallelism=2", "use_ragged_sort=True"], True),
    ):
      cfg = pyconfig.initialize(
          ["", base_yml, "model_name=qwen3.8-flash-next-mini", "hardware=cpu", "per_device_batch_size=1"] + overrides
      )
      self.assertFalse(cfg.megablox)
      m3_moe.validate_moe_config(cfg)
      exec_cpu = m3_moe.moe_from_config(cfg, DefaultSharding())
      self.assertEqual(exec_cpu.kind, "gmm_v2")
      self.assertEqual(exec_cpu.experts, "expert")
      self.assertEqual(exec_cpu.mlp, ("tensor", "tensor_sequence"))
      self.assertEqual(exec_cpu.use_ragged_sort, expect_rsort)

    # Explicit hardware=tpu with megablox=False and use_tokamax_gmm=False still fails closed.
    tpu_disabled = SimpleNamespace(
        hardware="tpu",
        num_experts=8,
        sparse_matmul=True,
        megablox=False,
        use_tokamax_gmm=False,
        ici_expert_parallelism=2,
    )
    with self.assertRaisesRegex(ValueError, "Expert parallelism.*gmm_v2 or tokamax"):
      m3_moe.validate_moe_config(tpu_disabled)
    with self.assertRaisesRegex(ValueError, "Expert parallelism.*gmm_v2 or tokamax"):
      m3_moe.moe_from_config(tpu_disabled, DefaultSharding())

    if jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 devices; got {jax.device_count()}.")
    cfg_run = pyconfig.initialize(
        [
            "",
            base_yml,
            "model_name=qwen3.8-flash-next-mini",
            "hardware=cpu",
            "per_device_batch_size=1",
            "ici_expert_parallelism=2",
            "ici_tensor_parallelism=2",
            "use_ragged_sort=True",
        ]
    )
    exec_run = m3_moe.moe_from_config(cfg_run, DefaultSharding())
    mesh = create_mesh(ici={"fsdp": 1, "expert": 2, "tensor": 2}, devices=jax.devices()[:4])
    x, weights, indices, _ = _make_routing_inputs(batch=4, length=4, embed=16, num_experts=4, top_k=2)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12)
    w_gu_4d = jnp.stack([wi_0, wi_1], axis=2)
    cot = jax.random.normal(jax.random.key(77), x.shape, dtype=jnp.float32)

    def ref_loss_fn(xi, gui, dni):
      return jnp.sum(m3_moe.DENSE.fused(xi, weights, indices, gui, dni) * cot)

    _, ref_grads = jax.value_and_grad(ref_loss_fn, argnums=(0, 1, 2))(x, w_gu_4d, wo)
    ref_out = m3_moe.DENSE.fused(x, weights, indices, w_gu_4d, wo)

    with jax.set_mesh(mesh):
      got_out = exec_run.fused(x, weights, indices, w_gu_4d, wo)
      _, got_grads = jax.value_and_grad(
          lambda xi, gui, dni: jnp.sum(exec_run.fused(xi, weights, indices, gui, dni) * cot),
          argnums=(0, 1, 2),
      )(x, w_gu_4d, wo)
    np.testing.assert_allclose(np.asarray(got_out), np.asarray(ref_out), rtol=1e-5, atol=1e-5)
    for g_got, g_ref in zip(got_grads, ref_grads):
      np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-5, atol=1e-5)


class SingleDeviceParityTest(parameterized.TestCase):
  """Tests single-device forward & backward parity across all backends, projections, and dropping modes."""

  @parameterized.named_parameters(
      ("ragged_dot_f32", "ragged_dot", jnp.float32, None, 1e-5),
      ("tokamax_f32", "tokamax", jnp.float32, 1.1, 1e-5),
      ("tokamax_bf16", "tokamax", jnp.bfloat16, None, 2e-2),
  )
  def test_separate_and_fused_3d_4d_forward_and_grad_parity(self, kind, dtype, swiglu_limit, tol):
    x, weights, indices, _ = _make_routing_inputs(batch=3, length=5, embed=16, num_experts=4, top_k=2, dtype=dtype)
    # Include sentinel -1 slot and an empty expert (expert 3 unused on row 0) to test masking & empty groups.
    indices = indices.at[0, 0, 1].set(-1)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12, dtype=dtype)
    w_gu_3d = jnp.concatenate([wi_0, wi_1], axis=-1)
    w_gu_4d = jnp.stack([wi_0, wi_1], axis=2)

    mesh = create_mesh(devices=jax.devices()[:1])
    exec_test = m3_moe.MoEExecutor(kind=kind, bwd_wgrad_overlap=kind != "dense")
    cotangent = jax.random.normal(jax.random.key(99), x.shape, dtype=jnp.float32)

    def ref_sep_loss(x_in, w0, w1, w_out):
      out = m3_moe.DENSE(x_in, weights, indices, w0, w1, w_out, swiglu_limit=swiglu_limit)
      return jnp.sum(out.astype(jnp.float32) * cotangent)

    def test_sep_loss(x_in, w0, w1, w_out):
      out = exec_test(x_in, weights, indices, w0, w1, w_out, swiglu_limit=swiglu_limit)
      return jnp.sum(out.astype(jnp.float32) * cotangent)

    def test_fused3d_loss(x_in, w_gu, w_dn):
      out = exec_test.fused(x_in, weights, indices, w_gu, w_dn, swiglu_limit=swiglu_limit)
      return jnp.sum(out.astype(jnp.float32) * cotangent)

    def test_fused4d_loss(x_in, w_gu, w_dn):
      out = exec_test.fused(x_in, weights, indices, w_gu, w_dn, swiglu_limit=swiglu_limit)
      return jnp.sum(out.astype(jnp.float32) * cotangent)

    with jax.set_mesh(mesh):
      ref_out = m3_moe.DENSE(x, weights, indices, wi_0, wi_1, wo, swiglu_limit=swiglu_limit)
      sep_out = exec_test(x, weights, indices, wi_0, wi_1, wo, swiglu_limit=swiglu_limit)
      f3d_out = exec_test.fused(x, weights, indices, w_gu_3d, wo, swiglu_limit=swiglu_limit)
      f4d_out = exec_test.fused(x, weights, indices, w_gu_4d, wo, swiglu_limit=swiglu_limit)

      np.testing.assert_allclose(sep_out, ref_out, rtol=tol, atol=tol)
      np.testing.assert_allclose(f3d_out, ref_out, rtol=tol, atol=tol)
      np.testing.assert_allclose(f4d_out, ref_out, rtol=tol, atol=tol)

      _, ref_grads = jax.value_and_grad(ref_sep_loss, argnums=(0, 1, 2, 3))(x, wi_0, wi_1, wo)
      _, sep_grads = jax.value_and_grad(test_sep_loss, argnums=(0, 1, 2, 3))(x, wi_0, wi_1, wo)
      _, f3d_grads = jax.value_and_grad(test_fused3d_loss, argnums=(0, 1, 2))(x, w_gu_3d, wo)
      _, f4d_grads = jax.value_and_grad(test_fused4d_loss, argnums=(0, 1, 2))(x, w_gu_4d, wo)

      for g_got, g_ref in zip(sep_grads, ref_grads):
        np.testing.assert_allclose(g_got, g_ref, rtol=tol, atol=tol)

      ref_gu_3d = jnp.concatenate([ref_grads[1], ref_grads[2]], axis=-1)
      ref_gu_4d = jnp.stack([ref_grads[1], ref_grads[2]], axis=2)
      np.testing.assert_allclose(f3d_grads[0], ref_grads[0], rtol=tol, atol=tol)
      np.testing.assert_allclose(f3d_grads[1], ref_gu_3d, rtol=tol, atol=tol)
      np.testing.assert_allclose(f3d_grads[2], ref_grads[3], rtol=tol, atol=tol)
      np.testing.assert_allclose(f4d_grads[0], ref_grads[0], rtol=tol, atol=tol)
      np.testing.assert_allclose(f4d_grads[1], ref_gu_4d, rtol=tol, atol=tol)
      np.testing.assert_allclose(f4d_grads[2], ref_grads[3], rtol=tol, atol=tol)

      # Perturbation negative control: swapping gate and up weights must fire clearly.
      swapped_out = exec_test(x, weights, indices, wi_1, wi_0, wo, swiglu_limit=swiglu_limit)
      self.assertGreater(float(jnp.max(jnp.abs(swapped_out.astype(jnp.float32) - ref_out.astype(jnp.float32)))), 1e-3)

  def test_capacity_factor_and_ragged_buffer_dropping_with_negative_controls(self):
    x, weights, _, _ = _make_routing_inputs(batch=4, length=4, embed=16, num_experts=4, top_k=2)
    # Route every token's first choice to expert 0 so capacity_factor=0.5 drops half of expert 0's traffic.
    indices = jnp.stack(
        [jnp.zeros((4, 4), dtype=jnp.int32), jnp.ones((4, 4), dtype=jnp.int32)],
        axis=-1,
    )
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12)
    mesh = create_mesh(devices=jax.devices()[:1])

    with jax.set_mesh(mesh):
      dropless = m3_moe.DENSE(x, weights, indices, wi_0, wi_1, wo)
      dense_cap = m3_moe.MoEExecutor("dense", capacity_factor=0.5)(x, weights, indices, wi_0, wi_1, wo)
      ragged_cap = m3_moe.MoEExecutor("ragged_dot", capacity_factor=0.5)(x, weights, indices, wi_0, wi_1, wo)
      tok_cap = m3_moe.MoEExecutor("tokamax", capacity_factor=0.5)(x, weights, indices, wi_0, wi_1, wo)

      np.testing.assert_allclose(ragged_cap, dense_cap, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(tok_cap, dense_cap, rtol=1e-5, atol=1e-5)
      # Negative control: capacity dropping must differ from dropless under skewed routing.
      self.assertGreater(float(jnp.max(jnp.abs(dense_cap - dropless))), 1e-2)

      ragged_rbf = m3_moe.MoEExecutor("ragged_dot", ragged_buffer_factor=0.25)(x, weights, indices, wi_0, wi_1, wo)
      tok_rbf = m3_moe.MoEExecutor("tokamax", ragged_buffer_factor=0.25)(x, weights, indices, wi_0, wi_1, wo)
      np.testing.assert_allclose(tok_rbf, ragged_rbf, rtol=1e-5, atol=1e-5)
      self.assertGreater(float(jnp.max(jnp.abs(tok_rbf - dropless))), 1e-2)

  @parameterized.named_parameters(
      ("ragged_dot", "ragged_dot"),
      ("tokamax", "tokamax"),
  )
  def test_grouped_matmul_unassigned_tail_rows_zeroed_in_fwd_and_bwd(self, backend):
    key = jax.random.key(314)
    k1, k2, k3 = jax.random.split(key, 3)
    m, valid_n, k, n = 16, 10, 12, 8
    group_sizes = jnp.array([4, 0, 6, 0], dtype=jnp.int32)
    lhs_valid = jax.random.normal(k1, (valid_n, k), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (4, k, n), dtype=jnp.float32)
    cot_valid = jax.random.normal(k3, (valid_n, n), dtype=jnp.float32)

    lhs_padded = jnp.concatenate([lhs_valid, jnp.full((m - valid_n, k), 500.0, dtype=jnp.float32)], axis=0)
    cot_padded = jnp.concatenate([cot_valid, jnp.full((m - valid_n, n), -700.0, dtype=jnp.float32)], axis=0)

    ref_out, ref_vjp = jax.vjp(
        lambda l, r: m3_moe.grouped_matmul(l, r, group_sizes, backend="ragged_dot", is_tpu=False),
        lhs_valid,
        rhs,
    )
    ref_dlhs, ref_drhs = ref_vjp(cot_valid)

    use_tpu_path = backend == "tokamax"
    with contextlib.ExitStack() as stack:
      if use_tpu_path:

        def _fake_gmm_v2(lhs, rhs, group_sizes, *, transpose_rhs=False, preferred_element_type=None, **kw):
          del kw
          rhs_eff = rhs.swapaxes(1, 2) if transpose_rhs else rhs
          res = jax.lax.ragged_dot(lhs, rhs_eff, group_sizes, preferred_element_type=jnp.float32)
          return res.astype(preferred_element_type) if preferred_element_type is not None else res

        def _fake_tgmm_v2(lhs, rhs, group_sizes, *, num_actual_groups=None, preferred_element_type=None, **kw):
          del kw
          res = jax.lax.ragged_dot_general(
              lhs,
              rhs,
              group_sizes,
              ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
                  dot_dimension_numbers=(((0,), (0,)), ((), ())),
                  lhs_ragged_dimensions=(0,),
                  rhs_group_dimensions=(),
              ),
              preferred_element_type=jnp.float32,
          )
          if num_actual_groups is not None:
            res = res[:num_actual_groups]
          return res.astype(preferred_element_type) if preferred_element_type is not None else res

        stack.enter_context(mock.patch.object(gmm_v2, "gmm_v2", side_effect=_fake_gmm_v2))
        stack.enter_context(mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_fake_tgmm_v2))
      out, vjp_fn = jax.vjp(
          lambda l, r: m3_moe.grouped_matmul(
              l,
              r,
              group_sizes,
              backend="gmm_v2" if use_tpu_path else backend,
              is_tpu=use_tpu_path,
          ),
          lhs_padded,
          rhs,
      )
      dlhs, drhs = vjp_fn(cot_padded)

    np.testing.assert_allclose(out[:valid_n], ref_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(out[valid_n:]), np.zeros((m - valid_n, n), dtype=np.float32))
    np.testing.assert_allclose(dlhs[:valid_n], ref_dlhs, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(dlhs[valid_n:]), np.zeros((m - valid_n, k), dtype=np.float32))
    np.testing.assert_allclose(drhs, ref_drhs, rtol=1e-5, atol=1e-5)

    if backend == "tokamax":
      lhs_nan = jnp.concatenate([lhs_valid, jnp.full((m - valid_n, k), jnp.nan, dtype=jnp.float32)], axis=0)
      cot_nan = jnp.concatenate([cot_valid, jnp.full((m - valid_n, n), jnp.nan, dtype=jnp.float32)], axis=0)
      captured_pads = {}

      def mock_gmm(*args, **kwargs):
        lhs_arg = kwargs.get("lhs", args[0] if args else None)
        rhs_arg = kwargs.get("rhs", args[1] if len(args) > 1 else None)
        rhs_eff = rhs_arg.swapaxes(1, 2) if kwargs.get("transpose_rhs", False) else rhs_arg
        out_dtype = kwargs.get("preferred_element_type", lhs_arg.dtype)
        if "fwd_lhs_pad" not in captured_pads:
          captured_pads["fwd_lhs_pad"] = np.asarray(lhs_arg)
          return jnp.zeros((lhs_arg.shape[0], rhs_eff.shape[2]), dtype=out_dtype)
        captured_pads["bwd_grad_pad"] = np.asarray(lhs_arg)
        return jnp.zeros((lhs_arg.shape[0], rhs_eff.shape[2]), dtype=out_dtype)

      def mock_tgmm(*args, **kwargs):
        lhs_arg = kwargs.get("lhs", args[0] if args else None)
        rhs_arg = kwargs.get("rhs", args[1] if len(args) > 1 else None)
        num_groups = kwargs.get("num_actual_groups", rhs.shape[0])
        captured_pads["bwd_lhs_pad"] = np.asarray(lhs_arg)
        return jnp.zeros((num_groups, lhs_arg.shape[1], rhs_arg.shape[1]), dtype=rhs.dtype)

      with (
          mock.patch.object(gmm_v2, "gmm_v2", side_effect=mock_gmm),
          mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=mock_tgmm),
      ):
        tpu_out, tpu_vjp = jax.vjp(
            lambda l, r: m3_moe.grouped_matmul(l, r, group_sizes, backend="gmm_v2", is_tpu=True),
            lhs_nan,
            rhs,
        )
        tpu_dlhs, tpu_drhs = tpu_vjp(cot_nan)

      self.assertFalse(np.isnan(captured_pads["fwd_lhs_pad"]).any())
      self.assertFalse(np.isnan(captured_pads["bwd_lhs_pad"]).any())
      self.assertFalse(np.isnan(captured_pads["bwd_grad_pad"]).any())
      self.assertFalse(np.isnan(np.asarray(tpu_out)).any())
      self.assertFalse(np.isnan(np.asarray(tpu_dlhs)).any())
      self.assertFalse(np.isnan(np.asarray(tpu_drhs)).any())

  def test_grouped_unsharded_fused_without_active_mesh(self):
    x, weights, indices, _ = _make_routing_inputs(batch=2, length=3, embed=16, num_experts=4, top_k=2)
    indices = indices.at[0, 0, 1].set(-1)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12)
    w_gu_4d = jnp.stack([wi_0, wi_1], axis=2)
    ref = m3_moe.grouped_unsharded_fused(
        x, weights, indices, w_gu_4d, wo, swiglu_limit=1.2, executor=m3_moe.MoEExecutor("ragged_dot"), is_tpu=False
    )
    tok = m3_moe.grouped_unsharded_fused(
        x, weights, indices, w_gu_4d, wo, swiglu_limit=1.2, executor=m3_moe.MoEExecutor("tokamax"), is_tpu=False
    )
    np.testing.assert_allclose(tok, ref, rtol=1e-5, atol=1e-5)


class MultiDeviceEPAndTPTest(parameterized.TestCase):
  """Verifies multi-device `(FSDP >= 1, EP > 1, TP > 1)` parity, HLO collectives, and `ScannedLayers`."""

  def _ep_tp_layouts(self) -> list[dict[str, int]]:
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices for EP > 1 and TP > 1; got {n}.")
    if n >= 8:
      return [
          {"fsdp": 2, "expert": 2, "tensor": 2},
          {"fsdp": 1, "expert": 4, "tensor": 2},
      ]
    return [{"fsdp": 1, "expert": 2, "tensor": 2}]

  def _assert_ep_tp_layout(
      self,
      ici: dict[str, int],
      *,
      kind: str,
      capacity_factor: float,
      ragged_buffer_factor: float,
      overlap: bool,
      sharding: DefaultSharding,
      tensors: tuple[jax.Array, ...],
      ref_exec: m3_moe.MoEExecutor,
  ) -> None:
    """Checks forward/backward parity and lowered HLO collectives on one `ici` mesh."""
    x, weights, indices, cotangent, wi_0, wi_1, wo, w_gu_3d, w_gu_4d = tensors
    dp_groups = ici["fsdp"]
    ep_size = ici["expert"]

    def dp_ref_step(x_in, w0, w1, w_out):
      b_dp = x_in.shape[0] // dp_groups
      chunks = [
          ref_exec(
              x_in[g * b_dp : (g + 1) * b_dp],
              weights[g * b_dp : (g + 1) * b_dp],
              indices[g * b_dp : (g + 1) * b_dp],
              w0,
              w1,
              w_out,
          )
          for g in range(dp_groups)
      ]
      y = jnp.concatenate(chunks, axis=0)
      return jnp.sum(y * cotangent), y

    (_, ref_out), ref_sep_grads = jax.value_and_grad(dp_ref_step, argnums=(0, 1, 2, 3), has_aux=True)(x, wi_0, wi_1, wo)
    mesh = create_mesh(ici=ici, devices=jax.devices()[: (ici["fsdp"] * ici["expert"] * ici["tensor"])])
    exec_dist = m3_moe.MoEExecutor(
        kind=kind,
        batch=("data", "fsdp", "expert"),
        experts="expert",
        mlp="tensor",
        capacity_factor=capacity_factor if capacity_factor > 0 else -1.0,
        ragged_buffer_factor=ragged_buffer_factor if ragged_buffer_factor > 0 else -1.0,
        bwd_wgrad_overlap=overlap,
    )
    with jax.set_mesh(mesh):
      tok_s = NamedSharding(mesh, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      rk_s = NamedSharding(mesh, P(("data", "fsdp", "expert"), None, None))
      w0_s = NamedSharding(mesh, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_s = NamedSharding(mesh, sharding("wo", ("experts", "expert_mlp", "expert_embed")))
      gu3_s = NamedSharding(mesh, sharding("gate_up", ("experts", "expert_embed", None)))
      gu4_s = NamedSharding(mesh, sharding("gate_up_4d", ("experts", "expert_embed", None, "expert_mlp")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(weights, rk_s)
      idx_d = jax.device_put(indices, rk_s)
      cot_d = jax.device_put(cotangent, tok_s)
      wi0_d = jax.device_put(wi_0, w0_s)
      wi1_d = jax.device_put(wi_1, w0_s)
      wo_d = jax.device_put(wo, wo_s)
      gu3_d = jax.device_put(w_gu_3d, gu3_s)
      gu4_d = jax.device_put(w_gu_4d, gu4_s)

      @jax.jit
      def sep_step(x_in, w0, w1, w_out):
        y = exec_dist(x_in, w_d, idx_d, w0, w1, w_out)
        return jnp.sum(y * cot_d), y

      (_, dist_out), dist_grads = jax.jit(jax.value_and_grad(sep_step, argnums=(0, 1, 2, 3), has_aux=True))(
          x_d, wi0_d, wi1_d, wo_d
      )

      if ragged_buffer_factor < 0:
        np.testing.assert_allclose(np.asarray(dist_out), np.asarray(ref_out), rtol=1e-4, atol=1e-4)
        for g_got, g_ref in zip(dist_grads, ref_sep_grads):
          np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-4, atol=1e-4)

        if ici["tensor"] > 1:
          with self.assertRaises(ValueError):
            exec_dist.fused(x_d, w_d, idx_d, gu3_d, wo_d)
        else:
          _, f3_grads = jax.jit(
              jax.value_and_grad(
                  lambda x_in, gu, dn: jnp.sum(exec_dist.fused(x_in, w_d, idx_d, gu, dn) * cot_d),
                  argnums=(0, 1, 2),
              )
          )(x_d, gu3_d, wo_d)
          np.testing.assert_allclose(
              np.asarray(f3_grads[1]),
              np.asarray(jnp.concatenate([ref_sep_grads[1], ref_sep_grads[2]], axis=-1)),
              rtol=1e-4,
              atol=1e-4,
          )

        _, f4_grads = jax.jit(
            jax.value_and_grad(
                lambda x_in, gu, dn: jnp.sum(exec_dist.fused(x_in, w_d, idx_d, gu, dn) * cot_d),
                argnums=(0, 1, 2),
            )
        )(x_d, gu4_d, wo_d)
        np.testing.assert_allclose(
            np.asarray(f4_grads[1]),
            np.asarray(jnp.stack([ref_sep_grads[1], ref_sep_grads[2]], axis=2)),
            rtol=1e-4,
            atol=1e-4,
        )

      hlo = (
          jax.jit(jax.value_and_grad(sep_step, argnums=(0, 1, 2, 3), has_aux=True))
          .lower(x_d, wi0_d, wi1_d, wo_d)
          .compile()
          .as_text()
      )
      full_expert_ag = re.findall(r"f32\[8,[0-9,]+\].*all-gather\(", hlo)
      self.assertEmpty(
          full_expert_ag,
          msg=f"Found forbidden full-expert weight all-gather on EP={ep_size}: {full_expert_ag}",
      )
      if ici["fsdp"] == 1:
        any_float_ag = re.findall(r"f32\[[0-9,]+\].*all-gather\(", hlo)
        self.assertEmpty(any_float_ag, msg=f"Expected zero float all-gather when fsdp=1; got {any_float_ag}")
      if overlap:
        hlo_lower = (
            jax.jit(jax.value_and_grad(sep_step, argnums=(0, 1, 2, 3), has_aux=True))
            .lower(x_d, wi0_d, wi1_d, wo_d)
            .as_text()
        )
        self.assertTrue("optimization_barrier" in hlo_lower or "opt-barrier" in hlo_lower)

      hlo_f4 = (
          jax.jit(
              jax.value_and_grad(
                  lambda x_in, gu, dn: jnp.sum(exec_dist.fused(x_in, w_d, idx_d, gu, dn) * cot_d),
                  argnums=(0, 1, 2),
              )
          )
          .lower(x_d, gu4_d, wo_d)
          .compile()
          .as_text()
      )
      full_expert_ag_f4 = re.findall(r"f32\[8,[0-9,]+\].*all-gather\(", hlo_f4)
      self.assertEmpty(
          full_expert_ag_f4,
          msg=f"Found forbidden 4D fused full-expert weight all-gather on EP={ep_size}: {full_expert_ag_f4}",
      )

      # Positive control: prove .compile().as_text() DOES match all-gather when a weight has P(None, ...):
      replicated_s = NamedSharding(mesh, P("expert", None, None))
      hlo_pos = jax.jit(lambda v: jax.lax.with_sharding_constraint(v, replicated_s)).lower(wi0_d).compile().as_text()
      if ici["tensor"] > 1:
        self.assertTrue(
            "all-gather" in hlo_pos,
            msg=f"Positive control failed: expected all-gather in compile().as_text(); got: {hlo_pos}",
        )

  @parameterized.named_parameters(
      ("tokamax_dropless", "tokamax", -1.0, -1.0, False),
      ("tokamax_overlap", "tokamax", -1.0, -1.0, True),
      ("tokamax_capacity", "tokamax", 0.75, -1.0, True),
      ("tokamax_rbf", "tokamax", -1.0, 0.75, False),
  )
  def test_ep_and_tp_forward_and_all_leaf_grad_parity_and_zero_expert_all_gather(
      self, kind, capacity_factor, ragged_buffer_factor, overlap
  ):
    sharding = DefaultSharding()
    x, weights, indices, _ = _make_routing_inputs(batch=8, length=4, embed=16, num_experts=8, top_k=2)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=8, embed=16, mlp=16)
    w_gu_3d = jnp.concatenate([wi_0, wi_1], axis=-1)
    w_gu_4d = jnp.stack([wi_0, wi_1], axis=2)
    cotangent = jax.random.normal(jax.random.key(77), x.shape, dtype=jnp.float32)

    ref_exec = m3_moe.MoEExecutor(
        "ragged_dot",
        capacity_factor=capacity_factor,
        ragged_buffer_factor=ragged_buffer_factor,
    )

    tensors = (x, weights, indices, cotangent, wi_0, wi_1, wo, w_gu_3d, w_gu_4d)
    for ici in self._ep_tp_layouts():
      self._assert_ep_tp_layout(
          ici,
          kind=kind,
          capacity_factor=capacity_factor,
          ragged_buffer_factor=ragged_buffer_factor,
          overlap=overlap,
          sharding=sharding,
          tensors=tensors,
          ref_exec=ref_exec,
      )

  def test_scan_layers_preserve_paths_and_match_unscanned_under_ep_and_tp(self):
    ici = self._ep_tp_layouts()[0]
    mesh = create_mesh(ici=ici, devices=jax.devices()[: (ici["fsdp"] * ici["expert"] * ici["tensor"])])
    sharding = DefaultSharding()
    cfg_dist = SimpleNamespace(
        sparse_matmul=True,
        use_tokamax_gmm=True,
        moe_bwd_wgrad_overlap=True,
    )
    cfg_ref = SimpleNamespace(sparse_matmul=False)
    x = jax.random.normal(jax.random.key(12), (8, 4, 16), dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(13), (8, 4, 16), dtype=jnp.float32)

    with jax.set_mesh(mesh):
      unscanned_ref = nnx.Sequential(
          _ToyMoELayer(8, 16, 16, cfg_ref, sharding, rngs=nnx.Rngs(1)),
          _ToyMoELayer(8, 16, 16, cfg_ref, sharding, rngs=nnx.Rngs(2)),
      )
      unscanned_dist = nnx.Sequential(
          _ToyMoELayer(8, 16, 16, cfg_dist, sharding, rngs=nnx.Rngs(1)),
          _ToyMoELayer(8, 16, 16, cfg_dist, sharding, rngs=nnx.Rngs(2)),
      )
      scanned_dist = m3_scan.scan_layers(unscanned_dist, param_scan_axis=0)
      paths_before_jit = {path for path, _ in nnx.to_flat_state(nnx.state(scanned_dist, nnx.Param))}

      tok_s = NamedSharding(mesh, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      x_d = jax.device_put(x, tok_s)
      cot_d = jax.device_put(cotangent, tok_s)

      ref_loss, ref_grads = nnx.jit(nnx.value_and_grad(lambda m, inp: jnp.sum(m(inp) * cot_d)))(unscanned_ref, x_d)
      scan_loss, scan_grads = nnx.jit(nnx.value_and_grad(lambda m, inp: jnp.sum(m(inp) * cot_d)))(scanned_dist, x_d)
      paths_after_jit = {path for path, _ in nnx.to_flat_state(nnx.state(scanned_dist, nnx.Param))}
      self.assertEqual(paths_before_jit, paths_after_jit)

      np.testing.assert_allclose(float(scan_loss), float(ref_loss), rtol=1e-4, atol=1e-4)
      scan_grad_map = {p: v.get_value() for p, v in nnx.to_flat_state(scan_grads)}
      for p_ref, v_ref in nnx.to_flat_state(ref_grads):
        self.assertEqual(p_ref[0], "layers")
        layer_idx = p_ref[1]
        tail = p_ref[2:]
        self.assertIn(tail, scan_grad_map)
        np.testing.assert_allclose(
            np.asarray(jnp.take(scan_grad_map[tail], layer_idx, axis=0)),
            np.asarray(v_ref.get_value()),
            rtol=1e-4,
            atol=1e-4,
        )

  def test_explicit_mesh_ep_tp_forward_and_grad_parity(self):
    ici = self._ep_tp_layouts()[0]
    dev_count = ici["fsdp"] * ici["expert"] * ici["tensor"]
    mesh_exp = create_mesh(ici=ici, devices=jax.devices()[:dev_count], explicit=True)
    sharding = DefaultSharding()
    x, weights, indices, _ = _make_routing_inputs(batch=8, length=4, embed=16, num_experts=8, top_k=2)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=8, embed=16, mlp=16)
    cotangent = jax.random.normal(jax.random.key(88), x.shape, dtype=jnp.float32)

    ref_exec = m3_moe.MoEExecutor("ragged_dot")
    (_, ref_out), ref_grads = jax.jit(
        jax.value_and_grad(
            lambda x_in, w0, w1, w_out: (
                jnp.sum(ref_exec(x_in, weights, indices, w0, w1, w_out) * cotangent),
                ref_exec(x_in, weights, indices, w0, w1, w_out),
            ),
            argnums=(0, 1, 2, 3),
            has_aux=True,
        )
    )(x, wi_0, wi_1, wo)

    with jax.set_mesh(mesh_exp):
      cfg = SimpleNamespace(num_experts=8, sparse_matmul=True, moe_bwd_wgrad_overlap=True)
      exec_exp = m3_moe.moe_from_config(cfg, sharding)
      self.assertEqual(exec_exp.kind, "gmm_v2")
      tok_s = NamedSharding(mesh_exp, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      w0_s = NamedSharding(mesh_exp, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_s = NamedSharding(mesh_exp, sharding("wo", ("experts", "expert_mlp", "expert_embed")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(weights, tok_s)
      idx_d = jax.device_put(indices, tok_s)
      cot_d = jax.device_put(cotangent, tok_s)
      wi0_d = jax.device_put(wi_0, w0_s)
      wi1_d = jax.device_put(wi_1, w0_s)
      wo_d = jax.device_put(wo, wo_s)

      (_, exp_out), exp_grads = jax.jit(
          jax.value_and_grad(
              lambda x_in, w0, w1, w_out: (
                  jnp.sum(exec_exp(x_in, w_d, idx_d, w0, w1, w_out) * cot_d),
                  exec_exp(x_in, w_d, idx_d, w0, w1, w_out),
              ),
              argnums=(0, 1, 2, 3),
              has_aux=True,
          )
      )(x_d, wi0_d, wi1_d, wo_d)

      np.testing.assert_allclose(np.asarray(exp_out), np.asarray(ref_out), rtol=1e-4, atol=1e-4)
      for g_got, g_ref in zip(exp_grads, ref_grads):
        np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-4, atol=1e-4)

  def test_sequence_sharding_preserved_on_context_mesh_and_noop_when_disabled(self):
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices; got {n}.")

    sharding = DefaultSharding()
    x, weights, indices, _ = _make_routing_inputs(batch=4, length=6, embed=16, num_experts=4, top_k=2, seed=71)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12, seed=72)
    cotangent = jax.random.normal(jax.random.key(73), x.shape, dtype=jnp.float32)

    ref_out = m3_moe.DENSE(x, weights, indices, wi_0, wi_1, wo)
    _, ref_grads = jax.jit(
        jax.value_and_grad(
            lambda x_in, w0, w1, w_out: jnp.sum(m3_moe.DENSE(x_in, weights, indices, w0, w1, w_out) * cotangent),
            argnums=(0, 1, 2, 3),
        )
    )(x, wi_0, wi_1, wo)

    def _seq_all_gathers(hlo_text: str) -> list[str]:
      found = []
      for line in hlo_text.splitlines():
        if "=" not in line:
          continue
        rhs = line.split("=", 1)[1]
        # Exclude expert weight gathers (`f32[2,16,12]` on `wi_0`/`wi_1` `expert_embed` over `context`).
        if "all-gather(" in rhs and "dimensions={1}" in rhs and "f32[2,16,12]" not in rhs:
          found.append(line.strip())
      return found

    mesh_auto = create_mesh(ici={"context": 2, "expert": 2}, devices=jax.devices()[:4])
    with jax.set_mesh(mesh_auto):
      cfg = SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=True)
      exec_ctx = m3_moe.moe_from_config(cfg, sharding)
      self.assertEqual(exec_ctx.length, "context")
      tok_s = NamedSharding(mesh_auto, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      w0_s = NamedSharding(mesh_auto, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_s = NamedSharding(mesh_auto, sharding("wo", ("experts", "expert_mlp", "expert_embed")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(weights, tok_s)
      idx_d = jax.device_put(indices, tok_s)
      cot_d = jax.device_put(cotangent, tok_s)
      wi0_d = jax.device_put(wi_0, w0_s)
      wi1_d = jax.device_put(wi_1, w0_s)
      wo_d = jax.device_put(wo, wo_s)

      fwd_ctx = jax.jit(lambda x_in, w_in, i_in: exec_ctx(x_in, w_in, i_in, wi0_d, wi1_d, wo_d))
      hlo_ctx = fwd_ctx.lower(x_d, w_d, idx_d).compile().as_text()
      self.assertEmpty(_seq_all_gathers(hlo_ctx))

      # Positive control: dropping `length` forces sequence `all-gather`s along `dimensions={1}`.
      exec_no_len = dataclasses.replace(exec_ctx, length=None)
      fwd_no_len = jax.jit(lambda x_in, w_in, i_in: exec_no_len(x_in, w_in, i_in, wi0_d, wi1_d, wo_d))
      hlo_no_len = fwd_no_len.lower(x_d, w_d, idx_d).compile().as_text()
      self.assertGreaterEqual(len(_seq_all_gathers(hlo_no_len)), 3)

      (_, got_out), got_grads = jax.jit(
          jax.value_and_grad(
              lambda x_in, w0, w1, w_out: (
                  jnp.sum(exec_ctx(x_in, w_d, idx_d, w0, w1, w_out) * cot_d),
                  exec_ctx(x_in, w_d, idx_d, w0, w1, w_out),
              ),
              argnums=(0, 1, 2, 3),
              has_aux=True,
          )
      )(x_d, wi0_d, wi1_d, wo_d)
      np.testing.assert_allclose(np.asarray(got_out), np.asarray(ref_out), rtol=1e-4, atol=1e-4)
      for g_got, g_ref in zip(got_grads, ref_grads):
        np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-4, atol=1e-4)

      # Single-step `length == 1` inputs cleanly fall back to unsharded sequence axis without error.
      x1, w1, i1 = x[:, :1, :], weights[:, :1, :], indices[:, :1, :]
      ref_out_1 = m3_moe.DENSE(x1, w1, i1, wi_0, wi_1, wo)
      got_out_1 = jax.jit(lambda a, b, c: exec_ctx(a, b, c, wi0_d, wi1_d, wo_d))(
          jax.device_put(x1, NamedSharding(mesh_auto, P("expert", None, None))),
          jax.device_put(w1, NamedSharding(mesh_auto, P("expert", None, None))),
          jax.device_put(i1, NamedSharding(mesh_auto, P("expert", None, None))),
      )
      np.testing.assert_allclose(np.asarray(got_out_1), np.asarray(ref_out_1), rtol=1e-4, atol=1e-4)

    # Explicit mesh preserves `P('expert', 'context', None)` on output without resharding to `None`.
    dev_grid = np.asarray(jax.devices()[:4]).reshape(2, 2)
    mesh_exp = jax.sharding.Mesh(
        dev_grid,
        ("expert", "context"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    with jax.set_mesh(mesh_exp):
      exec_exp = m3_moe.MoEExecutor("gmm_v2", batch="expert", experts="expert", length="context")
      tok_exp = NamedSharding(mesh_exp, P("expert", "context", None))
      w_exp = NamedSharding(mesh_exp, P("expert", None, None))
      out_exp = jax.jit(
          lambda a, b, c, p0, p1, p2: (
              self.assertEqual(
                  jax.typeof(exec_exp(a, b, c, p0, p1, p2)).sharding.spec,
                  P("expert", "context", None),
              )
              or exec_exp(a, b, c, p0, p1, p2)
          )
      )(
          jax.device_put(x, tok_exp),
          jax.device_put(weights, tok_exp),
          jax.device_put(indices, tok_exp),
          jax.device_put(wi_0, w_exp),
          jax.device_put(wi_1, w_exp),
          jax.device_put(wo, w_exp),
      )
      np.testing.assert_allclose(np.asarray(out_exp), np.asarray(ref_out), rtol=1e-4, atol=1e-4)

    # Default-off check: when `context == 1`, HLO with `length="context"` matches `length=None`.
    mesh_no_ctx = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=jax.devices()[:4])
    with jax.set_mesh(mesh_no_ctx):
      tok_nc = NamedSharding(mesh_no_ctx, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      w_nc = NamedSharding(mesh_no_ctx, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_nc = NamedSharding(mesh_no_ctx, sharding("wo", ("experts", "expert_mlp", "expert_embed")))
      x_nc, w_nc_in, i_nc = jax.device_put(x, tok_nc), jax.device_put(weights, tok_nc), jax.device_put(indices, tok_nc)
      wi0_nc, wi1_nc, wo_nc_in = jax.device_put(wi_0, w_nc), jax.device_put(wi_1, w_nc), jax.device_put(wo, wo_nc)
      e_with = m3_moe.MoEExecutor("tokamax", batch=("fsdp", "expert"), experts="expert", length="context")
      e_without = m3_moe.MoEExecutor("tokamax", batch=("fsdp", "expert"), experts="expert", length=None)
      h_with = (
          jax.jit(lambda a, b, c: e_with(a, b, c, wi0_nc, wi1_nc, wo_nc_in))
          .lower(x_nc, w_nc_in, i_nc)
          .compile()
          .as_text()
      )
      h_without = (
          jax.jit(lambda a, b, c: e_without(a, b, c, wi0_nc, wi1_nc, wo_nc_in))
          .lower(x_nc, w_nc_in, i_nc)
          .compile()
          .as_text()
      )

      def _strip_hlo_debug_locs(hlo_str: str) -> str:
        body_starts = [pos for pos in (hlo_str.find("\n%"), hlo_str.find("\nENTRY ")) if pos >= 0]
        body = hlo_str[min(body_starts) + 1 :] if body_starts else hlo_str
        return re.sub(r" stack_frame_id=\d+", "", body)

      norm_with = _strip_hlo_debug_locs(h_with)
      norm_without = _strip_hlo_debug_locs(h_without)
      self.assertGreater(len(norm_with), 1000)
      self.assertEqual(norm_with, norm_without)


class MockedTPULoweringContractTest(parameterized.TestCase):
  """Exercises TPU-only `ragged_all_to_all`, `gmm_v2`/`tgmm_v2`, and `a2a_ragged_sort` branches on CPU."""

  def test_tpu_tokamax_v2_ragged_sort_and_ragged_all_to_all_call_contracts(self):
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices; got {n}.")
    mesh = create_mesh(ici={"fsdp": 1, "expert": 2, "tensor": 2}, devices=jax.devices()[:4])
    sharding = DefaultSharding()

    x, weights, indices, _ = _make_routing_inputs(batch=4, length=4, embed=16, num_experts=4, top_k=2)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12)
    cotangent = jax.random.normal(jax.random.key(55), x.shape, dtype=jnp.float32)

    calls = {"gmm_shapes": [], "tgmm_shapes": [], "sort": 0, "unsort": 0, "ra2a": 0}

    def fake_gmm_v2(
        *, lhs, rhs, group_sizes, rhs_scale, tile_info, preferred_element_type, maybe_quantize_lhs, transpose_rhs=False
    ):
      del rhs_scale
      self.assertIs(tile_info, gmm_v2.calculate_tiling)
      self.assertFalse(maybe_quantize_lhs)
      rhs_eff = rhs.swapaxes(1, 2) if transpose_rhs else rhs
      self.assertEqual(lhs.shape[0] % 256, 0)
      self.assertEqual(lhs.shape[1] % 128, 0)
      self.assertEqual(rhs_eff.shape[2] % 128, 0)
      calls["gmm_shapes"].append((lhs.shape, rhs_eff.shape))
      return jax.lax.ragged_dot(lhs, rhs_eff, group_sizes, preferred_element_type=jnp.float32).astype(
          preferred_element_type
      )

    def fake_tgmm_v2(
        *, lhs, rhs, group_sizes, num_actual_groups, rhs_scale, precision, preferred_element_type, tile_info
    ):
      del rhs_scale, precision
      self.assertIs(tile_info, tgmm_v2.calculate_tgmm_tiling)
      self.assertEqual(lhs.shape[0] % 256, 0)
      self.assertEqual(lhs.shape[1] % 128, 0)
      self.assertEqual(rhs.shape[1] % 128, 0)
      calls["tgmm_shapes"].append((lhs.shape, rhs.shape, num_actual_groups))
      _, vjp_fn = jax.vjp(
          lambda r_w: jax.lax.ragged_dot(lhs, r_w, group_sizes, preferred_element_type=jnp.float32),
          jnp.zeros((num_actual_groups, lhs.shape[1], rhs.shape[1]), dtype=lhs.dtype),
      )
      (drhs,) = vjp_fn(rhs.astype(jnp.float32))
      return drhs.astype(preferred_element_type)

    def fake_sort(operand, indices_perm, num_valid=None):
      calls["sort"] += 1
      out = m3_moe._permute_rows(operand, indices_perm)  # pylint: disable=protected-access
      if num_valid is not None:
        out = jnp.where((jnp.arange(operand.shape[0], dtype=jnp.int32) < num_valid)[:, None], out, 0)
      return out

    def fake_unsort(operand, inv_perm, num_valid=None):
      calls["unsort"] += 1
      out = m3_moe._permute_rows(operand, inv_perm)  # pylint: disable=protected-access
      if num_valid is not None:
        out = jnp.where((jnp.arange(operand.shape[0], dtype=jnp.int32) < num_valid)[:, None], out, 0)
      return out

    def fake_ra2a(operand, output, in_off, send_sz, out_off, recv_sz, *, axis_name):
      calls["ra2a"] += 1
      return _cpu_ragged_all_to_all_ref(operand, output, in_off, send_sz, out_off, recv_sz, axis_name=axis_name)

    ref_out = m3_moe.DENSE(x, weights, indices, wi_0, wi_1, wo)
    _, ref_grads = jax.value_and_grad(
        lambda x_in, w0, w1, w_out: jnp.sum(m3_moe.DENSE(x_in, weights, indices, w0, w1, w_out) * cotangent),
        argnums=(0, 1, 2, 3),
    )(x, wi_0, wi_1, wo)

    exec_tpu = m3_moe.MoEExecutor(
        kind="tokamax",
        batch=("data", "fsdp", "expert"),
        experts="expert",
        mlp="tensor",
        use_ragged_sort=True,
        bwd_wgrad_overlap=True,
    )

    with (
        mock.patch.object(m3_kernels, "is_tpu_mesh", return_value=True),
        mock.patch.object(m3_moe, "_can_use_sparsecore_ragged_sort", return_value=True),
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=fake_gmm_v2),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=fake_tgmm_v2),
        mock.patch.object(ragged_sort, "a2a_ragged_sort", side_effect=fake_sort),
        mock.patch.object(ragged_sort, "a2a_ragged_unsort", side_effect=fake_unsort),
        mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=fake_ra2a),
        jax.set_mesh(mesh),
    ):
      tok_s = NamedSharding(mesh, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      rk_s = NamedSharding(mesh, P(("data", "fsdp", "expert"), None, None))
      w0_s = NamedSharding(mesh, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_s = NamedSharding(mesh, sharding("wo", ("experts", "expert_mlp", "expert_embed")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(weights, rk_s)
      idx_d = jax.device_put(indices, rk_s)
      cot_d = jax.device_put(cotangent, tok_s)
      wi0_d = jax.device_put(wi_0, w0_s)
      wi1_d = jax.device_put(wi_1, w0_s)
      wo_d = jax.device_put(wo, wo_s)

      out, grads = jax.value_and_grad(
          lambda x_in, w0, w1, w_out: jnp.sum(exec_tpu(x_in, w_d, idx_d, w0, w1, w_out) * cot_d),
          argnums=(0, 1, 2, 3),
      )(x_d, wi0_d, wi1_d, wo_d)
      out_val = exec_tpu(x_d, w_d, idx_d, wi0_d, wi1_d, wo_d)

      self.assertTrue(jnp.isfinite(out))
      np.testing.assert_allclose(np.asarray(out_val), np.asarray(ref_out), rtol=1e-4, atol=1e-4)
      for g_got, g_ref in zip(grads, ref_grads):
        np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-4, atol=1e-4)

    self.assertGreaterEqual(len(calls["gmm_shapes"]), 6)
    self.assertEqual(len(calls["tgmm_shapes"]), 3)
    self.assertGreaterEqual(calls["sort"], 2)
    self.assertGreaterEqual(calls["unsort"], 2)
    self.assertGreaterEqual(calls["ra2a"], 4)

  def test_tpu_f32_matmul_precision_and_ep1_tp_f32_down_combine(self):
    if jax.device_count() < 2:
      self.skipTest("Requires >= 2 JAX devices.")
    mesh = create_mesh(ici={"data": 1, "fsdp": 1, "expert": 1, "tensor": 2}, devices=jax.devices()[:2])
    sharding = DefaultSharding()
    recorded: dict[str, list[object]] = {"gmm_prec": [], "tgmm_prec": [], "gmm_pet": []}

    def _mxu_operand(arr: jax.Array, prec_cfg: object) -> jax.Array:
      if prec_cfg in ("float32", "highest"):
        return arr.astype(jnp.float32)
      return arr.astype(jnp.bfloat16).astype(jnp.float32)

    def _make_fakes(*, honor_default_prec: bool):
      def _fgmm(
          *, lhs, rhs, group_sizes, rhs_scale, tile_info, preferred_element_type, maybe_quantize_lhs, transpose_rhs=False
      ):
        del rhs_scale, tile_info, maybe_quantize_lhs
        cur_prec = jax.config.jax_default_matmul_precision if honor_default_prec else None
        recorded["gmm_prec"].append(cur_prec)
        recorded["gmm_pet"].append(jnp.dtype(preferred_element_type))
        rhs_eff = rhs.swapaxes(1, 2) if transpose_rhs else rhs
        return jax.lax.ragged_dot(
            _mxu_operand(lhs, cur_prec),
            _mxu_operand(rhs_eff, cur_prec),
            group_sizes,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).astype(preferred_element_type)

      def _ftgmm(*, lhs, rhs, group_sizes, num_actual_groups, rhs_scale, precision, preferred_element_type, tile_info):
        del rhs_scale, precision, tile_info
        cur_prec = jax.config.jax_default_matmul_precision if honor_default_prec else None
        recorded["tgmm_prec"].append(cur_prec)
        l_in = _mxu_operand(lhs, cur_prec)
        r_in = _mxu_operand(rhs, cur_prec)
        gs = group_sizes[:num_actual_groups].astype(jnp.int32)
        ends = jnp.cumsum(gs)
        starts = ends - gs
        rows = jnp.arange(l_in.shape[0], dtype=jnp.int32)[None, :]
        mask = (rows >= starts[:, None]) & (rows < ends[:, None])
        return jnp.einsum(
            "gmk,mn->gkn",
            jnp.where(mask[:, :, None], l_in[None, :, :], 0.0),
            r_in,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).astype(preferred_element_type)

      return _fgmm, _ftgmm

    # 1. Float32 path activates jax_default_matmul_precision="float32" inside gmm_v2 & tgmm_v2.
    x_f32, w_f32, idx_f32, _ = _make_routing_inputs(
        batch=2, length=8, embed=128, num_experts=4, top_k=2, dtype=jnp.float32
    )
    wi0_f32, wi1_f32, wo_f32 = _make_expert_weights(num_experts=4, embed=128, mlp=256, dtype=jnp.float32)
    gu_f32 = jnp.stack([wi0_f32, wi1_f32], axis=2)
    cot_f32 = jax.random.normal(jax.random.key(91), x_f32.shape, dtype=jnp.float32)

    @jax.jit
    def ref_f32_fn(xi, gui, dni):
      def _loss(x_in, gu_in, dn_in):
        y = m3_moe.DENSE.fused(x_in, w_f32, idx_f32, gu_in, dn_in)
        return jnp.mean(jnp.sum(y.astype(jnp.float32) * cot_f32, axis=-1)), y

      (_, y_out), grads = jax.value_and_grad(_loss, argnums=(0, 1, 2), has_aux=True)(xi, gui, dni)
      return y_out, grads

    ref_f32_out, ref_f32_grads = ref_f32_fn(x_f32, gu_f32, wo_f32)
    ex_tp = m3_moe.MoEExecutor(kind="gmm_v2", batch="data", experts="expert", mlp="tensor")

    fgmm_ok, ftgmm_ok = _make_fakes(honor_default_prec=True)
    with (
        mock.patch.object(m3_kernels, "is_tpu_mesh", return_value=True),
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=fgmm_ok),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=ftgmm_ok),
        jax.set_mesh(mesh),
    ):
      tok_s = NamedSharding(mesh, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      gu_s = NamedSharding(mesh, sharding("gate_up_proj", ("experts", "expert_embed", None, "expert_mlp")))
      dn_s = NamedSharding(mesh, sharding("down_proj", ("experts", "expert_mlp", "expert_embed")))
      x_d = jax.device_put(x_f32, tok_s)
      gu_d = jax.device_put(gu_f32, gu_s)
      dn_d = jax.device_put(wo_f32, dn_s)
      cot_d = jax.device_put(cot_f32, tok_s)

      @jax.jit
      def run_f32(xi, gui, dni):
        def _loss(x_in, gu_in, dn_in):
          y = ex_tp.fused(x_in, w_f32, idx_f32, gu_in, dn_in)
          return jnp.mean(jnp.sum(y.astype(jnp.float32) * cot_d, axis=-1)), y

        (_, y_out), grads = jax.value_and_grad(_loss, argnums=(0, 1, 2), has_aux=True)(xi, gui, dni)
        return y_out, grads

      got_f32_out, got_f32_grads = run_f32(x_d, gu_d, dn_d)

    self.assertTrue(all(p == "float32" for p in recorded["gmm_prec"]))
    self.assertTrue(all(p == "float32" for p in recorded["tgmm_prec"]))
    np.testing.assert_allclose(np.asarray(got_f32_out), np.asarray(ref_f32_out), rtol=1e-3, atol=1e-3)
    for g_got, g_ref in zip(got_f32_grads, ref_f32_grads):
      np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-3, atol=1e-3)

    # Negative control: bf16-MXU rounding on float32 violates 1e-3 tolerance.
    fgmm_bad, ftgmm_bad = _make_fakes(honor_default_prec=False)
    with (
        mock.patch.object(m3_kernels, "is_tpu_mesh", return_value=True),
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=fgmm_bad),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=ftgmm_bad),
        jax.set_mesh(mesh),
    ):
      bad_f32_out = jax.jit(lambda xi, gui, dni: ex_tp.fused(xi, w_f32, idx_f32, gui, dni))(x_d, gu_d, dn_d)
    self.assertGreater(
        float(np.max(np.abs(np.asarray(bad_f32_out, dtype=np.float32) - np.asarray(ref_f32_out, dtype=np.float32)))), 1e-3
    )

    # 2. EP=1, TP=2 bfloat16 keeps gate_up in bf16 (DEFAULT) and down_proj output in float32 across TP psum.
    recorded["gmm_prec"].clear()
    recorded["gmm_pet"].clear()
    x_bf, w_bf, idx_bf, _ = _make_routing_inputs(batch=2, length=8, embed=128, num_experts=4, top_k=2, dtype=jnp.bfloat16)
    wi0_bf, wi1_bf, wo_bf = _make_expert_weights(num_experts=4, embed=128, mlp=256, dtype=jnp.bfloat16)
    gu_bf = jnp.stack([wi0_bf, wi1_bf], axis=2)
    with (
        mock.patch.object(m3_kernels, "is_tpu_mesh", return_value=True),
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=fgmm_ok),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=ftgmm_ok),
        jax.set_mesh(mesh),
    ):
      out_bf = jax.jit(lambda xi, gui, dni: ex_tp.fused(xi, w_bf, idx_bf, gui, dni))(
          jax.device_put(x_bf, tok_s), jax.device_put(gu_bf, gu_s), jax.device_put(wo_bf, dn_s)
      )
    self.assertEqual(out_bf.dtype, jnp.bfloat16)
    self.assertTrue(all(p is None for p in recorded["gmm_prec"]))
    self.assertEqual(recorded["gmm_pet"], [jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)])

  def test_cpu_ragged_all_to_all_ref_honors_non_contiguous_and_reordered_output_offsets(self):
    n = jax.device_count()
    if n < 2:
      self.skipTest(f"Requires >= 2 devices; got {n}.")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("ep",))

    # Rank 0 sends [10] to rank 0 @ offset 2 and [20, 30] to rank 1 @ offset 1.
    # Rank 1 sends [40] to rank 0 @ offset 0 and [50] to rank 1 @ offset 3.
    operand = jnp.array(
        [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [99.0, 9.0], [40.0, 4.0], [50.0, 5.0], [88.0, 8.0], [77.0, 7.0]],
        dtype=jnp.float32,
    )
    output_init = jnp.full((8, 2), -7.0, dtype=jnp.float32)
    in_off = jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)
    send_sz = jnp.array([[1, 2], [1, 1]], dtype=jnp.int32)
    out_off = jnp.array([[2, 1], [0, 3]], dtype=jnp.int32)
    recv_sz = jnp.array([[1, 1], [2, 1]], dtype=jnp.int32)
    cot = jax.random.normal(jax.random.key(91), (8, 2), dtype=jnp.float32)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("ep", None), P("ep", None), P("ep", None), P("ep", None), P("ep", None), P("ep", None)),
        out_specs=P("ep", None),
    )
    def run_ref(op, out, io, ss, oo, rs):
      return _cpu_ragged_all_to_all_ref(op[0] if op.ndim == 3 else op, out, io[0], ss[0], oo[0], rs[0], axis_name="ep")

    got = run_ref(operand, output_init, in_off, send_sz, out_off, recv_sz)
    expected = jnp.array(
        [
            [40.0, 4.0],
            [-7.0, -7.0],
            [10.0, 1.0],
            [-7.0, -7.0],
            [-7.0, -7.0],
            [20.0, 2.0],
            [30.0, 3.0],
            [50.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=0.0, atol=0.0)

    d_op, d_out = jax.grad(
        lambda op, out: jnp.sum(run_ref(op, out, in_off, send_sz, out_off, recv_sz) * cot),
        argnums=(0, 1),
    )(operand, output_init)
    expected_d_op = jnp.array(
        [cot[2], cot[5], cot[6], jnp.zeros(2), cot[0], cot[7], jnp.zeros(2), jnp.zeros(2)],
        dtype=jnp.float32,
    )
    expected_d_out = jnp.array(
        [jnp.zeros(2), cot[1], jnp.zeros(2), cot[3], cot[4], jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(d_op), np.asarray(expected_d_op), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(d_out), np.asarray(expected_d_out), rtol=0.0, atol=0.0)

  def test_sparsecore_ragged_sort_guard_and_fallback_parity(self):
    from jax.experimental.pallas import tpu as pltpu  # pylint: disable=import-outside-toplevel

    def _check_guard(num_rows, hidden_size, dtype, *, sc_info, num_lanes=128):
      fake_info = SimpleNamespace(num_lanes=num_lanes, sparse_core=sc_info)
      with (
          mock.patch.object(pltpu, "is_tpu_device", return_value=True),
          mock.patch.object(pltpu, "get_tpu_info", return_value=fake_info),
      ):
        return m3_moe._can_use_sparsecore_ragged_sort(num_rows, hidden_size, dtype)  # pylint: disable=protected-access

    # v5e (`sparse_core=None`) always falls back cleanly.
    self.assertFalse(_check_guard(32, 1024, jnp.bfloat16, sc_info=None))
    self.assertFalse(_check_guard(32, 2560, jnp.float32, sc_info=None))

    # tpu7x (`2 * 16 = 32` subcores, `num_simd_lanes = 16`): requires `D >= 1024` and `D % 256 == 0`.
    sc_7x = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16)
    for bad_d in (128, 256, 512, 768, 1152, 1408, 1920):
      self.assertFalse(_check_guard(32, bad_d, jnp.bfloat16, sc_info=sc_7x), msg=f"Expected False for D={bad_d}")
    for ok_d in (1024, 1280, 1536, 2048, 2560, 4096):
      self.assertTrue(_check_guard(32, ok_d, jnp.bfloat16, sc_info=sc_7x), msg=f"Expected True for D={ok_d}")
      self.assertTrue(_check_guard(32, ok_d, jnp.float32, sc_info=sc_7x), msg=f"Expected True for f32 D={ok_d}")
    # Odd `bfloat16` row count violates 32-bit HBM packing in `ragged_gather`, while even `bf16` and odd `f32` pass.
    self.assertFalse(_check_guard(3, 1024, jnp.bfloat16, sc_info=sc_7x))
    self.assertTrue(_check_guard(4, 1024, jnp.bfloat16, sc_info=sc_7x))
    self.assertTrue(_check_guard(3, 1024, jnp.float32, sc_info=sc_7x))
    self.assertFalse(_check_guard(32, 1024, jnp.float16, sc_info=sc_7x))

    # v6e (`2 * 16 = 32` subcores, `num_simd_lanes = 8`): requires `D >= 2048` and `D % 512 == 0`.
    sc_v6e = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=8)
    self.assertFalse(_check_guard(32, 1024, jnp.bfloat16, sc_info=sc_v6e))
    self.assertFalse(_check_guard(32, 1536, jnp.bfloat16, sc_info=sc_v6e))
    self.assertTrue(_check_guard(32, 2048, jnp.bfloat16, sc_info=sc_v6e))
    self.assertTrue(_check_guard(32, 2560, jnp.bfloat16, sc_info=sc_v6e))

    # v5p (`4 * 16 = 64` subcores, `num_simd_lanes = 8`): requires `D >= 4096` and `D % 1024 == 0`.
    sc_v5p = SimpleNamespace(num_cores=4, num_subcores=16, num_lanes=8)
    self.assertFalse(_check_guard(32, 2048, jnp.bfloat16, sc_info=sc_v5p))
    self.assertFalse(_check_guard(32, 3072, jnp.bfloat16, sc_info=sc_v5p))
    self.assertTrue(_check_guard(32, 4096, jnp.bfloat16, sc_info=sc_v5p))

    # Verify fallback execution preserves dtype (`float32` and `bfloat16`) and gradients for both EP=1 and EP=2.
    n = jax.device_count()
    if n < 4:
      return
    for ici in ({"fsdp": 2, "expert": 1, "tensor": 2}, {"fsdp": 1, "expert": 2, "tensor": 2}):
      mesh = create_mesh(ici=ici, devices=jax.devices()[:4])
      for dt, tol in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
        x, weights, indices, _ = _make_routing_inputs(
            batch=4, length=4, embed=16, num_experts=4, top_k=2, dtype=dt, seed=81
        )
        wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12, dtype=dt, seed=82)
        cot = jax.random.normal(jax.random.key(83), x.shape, dtype=dt)
        ref_out = m3_moe.DENSE(x, weights, indices, wi_0, wi_1, wo)

        def _loss_with(fn, a, w0, w1, w2, w_in=weights, i_in=indices, c_in=cot):
          return jnp.sum(fn(a, w_in, i_in, w0, w1, w2).astype(jnp.float32) * c_in.astype(jnp.float32))

        _, ref_grads = jax.value_and_grad(
            functools.partial(_loss_with, m3_moe.DENSE),
            argnums=(0, 1, 2, 3),
        )(x, wi_0, wi_1, wo)

        def fake_gmm_f(
            *,
            lhs,
            rhs,
            group_sizes,
            rhs_scale,
            tile_info,
            preferred_element_type,
            maybe_quantize_lhs,
            transpose_rhs=False,
        ):
          del rhs_scale, tile_info, maybe_quantize_lhs
          rhs_eff = rhs.swapaxes(1, 2) if transpose_rhs else rhs
          return jax.lax.ragged_dot(lhs, rhs_eff, group_sizes, preferred_element_type=jnp.float32).astype(
              preferred_element_type
          )

        def fake_tgmm_f(
            *,
            lhs,
            rhs,
            group_sizes,
            num_actual_groups=None,
            rhs_scale=None,
            precision=None,
            preferred_element_type,
            tile_info,
        ):
          del num_actual_groups, rhs_scale, precision, tile_info
          g = group_sizes.shape[0]
          ends = jnp.cumsum(group_sizes)
          gid = jnp.searchsorted(ends, jnp.arange(lhs.shape[0], dtype=jnp.int32), side="right")
          valid = jnp.arange(lhs.shape[0], dtype=jnp.int32) < ends[-1]
          one_hot = jax.nn.one_hot(jnp.minimum(gid, g - 1), g, dtype=jnp.float32) * valid[:, None]
          return jnp.einsum("mg,mk,mn->gkn", one_hot, lhs.astype(jnp.float32), rhs.astype(jnp.float32)).astype(
              preferred_element_type
          )

        def fake_ra2a_f(operand, output, in_off, send_sz, out_off, recv_sz, *, axis_name):
          return _cpu_ragged_all_to_all_ref(operand, output, in_off, send_sz, out_off, recv_sz, axis_name=axis_name)

        exec_fb = m3_moe.MoEExecutor(
            kind="gmm_v2",
            batch=("data", "fsdp", "expert"),
            experts="expert",
            mlp="tensor",
            use_ragged_sort=True,
        )
        with (
            mock.patch.object(m3_kernels, "is_tpu_mesh", return_value=True),
            mock.patch.object(pltpu, "is_tpu_device", return_value=True),
            mock.patch.object(pltpu, "get_tpu_info", return_value=SimpleNamespace(num_lanes=128, sparse_core=None)),
            mock.patch.object(gmm_v2, "gmm_v2", side_effect=fake_gmm_f),
            mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=fake_tgmm_f),
            mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=fake_ra2a_f),
            mock.patch.object(ragged_sort, "a2a_ragged_sort", side_effect=AssertionError("sort must not run")),
            mock.patch.object(ragged_sort, "a2a_ragged_unsort", side_effect=AssertionError("unsort must not run")),
            jax.set_mesh(mesh),
        ):
          got_out = exec_fb(x, weights, indices, wi_0, wi_1, wo)
          self.assertEqual(got_out.dtype, dt)
          _, got_grads = jax.value_and_grad(
              functools.partial(_loss_with, exec_fb),
              argnums=(0, 1, 2, 3),
          )(x, wi_0, wi_1, wo)
          np.testing.assert_allclose(np.asarray(got_out), np.asarray(ref_out), rtol=tol, atol=tol)
          for g_got, g_ref in zip(got_grads, ref_grads):
            np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=tol, atol=tol)


@pytest.mark.tpu_only
class TPUExecutionParityTest(parameterized.TestCase):
  """Executes real TPU `gmm_v2`/`tgmm_v2`, `ragged_all_to_all`, and SparseCore `ragged_sort` on TPU hardware."""

  def setUp(self):
    super().setUp()
    if not any(d.platform == "tpu" for d in jax.devices()):
      self.skipTest("Requires TPU hardware.")
    if jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 TPU devices; got {jax.device_count()}.")

  @parameterized.named_parameters(
      ("gmm_v2_f32", jnp.float32, 128, 256, False, 1e-3),
      ("gmm_v2_bf16_ragged_sort", jnp.bfloat16, 1024, 256, True, 2e-2),
  )
  def test_tpu_ep_tp_forward_and_grad_parity(self, dtype, embed, mlp, use_ragged_sort, tol):
    devices = jax.devices()
    n = len(devices)
    mesh = create_mesh(ici={"data": 1, "fsdp": n // 4, "expert": 2, "tensor": 2}, devices=devices)
    sharding = DefaultSharding()

    batch = n
    x, weights, indices, _ = _make_routing_inputs(
        batch=batch, length=16, embed=embed, num_experts=4, top_k=2, dtype=dtype
    )
    indices = indices.at[0, 0, 1].set(-1)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=embed, mlp=mlp, dtype=dtype)
    w_gu_4d = jnp.stack([wi_0, wi_1], axis=2)
    cotangent = jax.random.normal(jax.random.key(61), x.shape, dtype=jnp.float32)

    ref_mesh = create_mesh(devices=devices[:1])
    with jax.set_mesh(ref_mesh):

      @jax.jit
      def ref_step_fn(xi, gui, dni):
        def ref_loss_fn(x_in, gu_in, dn_in):
          y = m3_moe.DENSE.fused(x_in, weights, indices, gu_in, dn_in)
          return jnp.mean(jnp.sum(y.astype(jnp.float32) * cotangent, axis=-1)), y

        (_, y), grads = jax.value_and_grad(ref_loss_fn, argnums=(0, 1, 2), has_aux=True)(xi, gui, dni)
        return y, grads

      ref_out, ref_grads = ref_step_fn(x, w_gu_4d, wo)

    exec_tpu = m3_moe.MoEExecutor(
        kind="gmm_v2",
        batch=("data", "fsdp", "expert"),
        experts="expert",
        mlp="tensor",
        use_ragged_sort=use_ragged_sort,
        bwd_wgrad_overlap=True,
    )

    with jax.set_mesh(mesh):
      tok_s = NamedSharding(mesh, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      rk_s = NamedSharding(mesh, P(("data", "fsdp", "expert"), None, None))
      gu_s = NamedSharding(mesh, sharding("gate_up_proj", ("experts", "expert_embed", None, "expert_mlp")))
      dn_s = NamedSharding(mesh, sharding("down_proj", ("experts", "expert_mlp", "expert_embed")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(weights, rk_s)
      idx_d = jax.device_put(indices, rk_s)
      cot_d = jax.device_put(cotangent, tok_s)
      gu_d = jax.device_put(w_gu_4d, gu_s)
      dn_d = jax.device_put(wo, dn_s)

      @jax.jit
      def step_fn(xi, gui, dni):
        def loss_fn(x_in, gu_in, dn_in):
          y = exec_tpu.fused(x_in, w_d, idx_d, gu_in, dn_in)
          return jnp.mean(jnp.sum(y.astype(jnp.float32) * cot_d, axis=-1)), y

        (loss, y), grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(xi, gui, dni)
        return loss, y, grads

      compiled = step_fn.lower(x_d, gu_d, dn_d).compile()
      self.assertIn("ragged-all-to-all", compiled.as_text())
      loss, got_out, got_grads = compiled(x_d, gu_d, dn_d)
      _, pert_out, pert_grads = step_fn(x_d, gu_d + jnp.asarray(0.25, dtype=dtype), dn_d)

    self.assertTrue(bool(jnp.isfinite(loss)))
    np.testing.assert_allclose(np.asarray(got_out), np.asarray(ref_out), rtol=tol, atol=tol)
    for g_got, g_ref in zip(got_grads, ref_grads):
      np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=tol, atol=tol)
    self.assertGreater(
        float(np.max(np.abs(np.asarray(pert_out, dtype=np.float32) - np.asarray(ref_out, dtype=np.float32)))),
        1e-2,
    )
    for g_pert, g_ref in zip(pert_grads, ref_grads):
      self.assertGreater(
          float(np.max(np.abs(np.asarray(g_pert, dtype=np.float32) - np.asarray(g_ref, dtype=np.float32)))),
          tol,
      )


class LoadBalanceLossMaskingTest(parameterized.TestCase):
  """Verifies padding masking (`!= 0`) and packed `segment_ids` binarization with negative controls."""

  def test_load_balance_loss_masks_padding_and_binarizes_segment_ids(self):
    _, _, indices, probs = _make_routing_inputs(batch=2, length=4, embed=8, num_experts=4, top_k=2, seed=31)
    seg_ids = jnp.array([[1, 1, 2, 0], [1, 0, 0, 0]], dtype=jnp.int32)
    seg_ids_scaled = jnp.array([[1, 1, 9, 0], [7, 0, 0, 0]], dtype=jnp.int32)
    non_binary_mask = jnp.array([[3, 2, 5, 0], [4, 0, 0, 0]], dtype=jnp.int32)
    binary_mask = (seg_ids != 0).astype(jnp.int32)

    # Perturb padded positions brutally so any unmasked leakage fires immediately.
    probs_corrupted = probs.at[0, 3].set(jnp.array([1.0, 0.0, 0.0, 0.0])).at[1, 1:].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    indices_corrupted = indices.at[0, 3].set(jnp.array([0, 0])).at[1, 1:].set(jnp.array([0, 0]))

    loss_clean = m3_moe.load_balance_loss(indices, probs, segment_ids=seg_ids)
    loss_corrupted = m3_moe.load_balance_loss(indices_corrupted, probs_corrupted, segment_ids=seg_ids)
    loss_scaled_seg = m3_moe.load_balance_loss(indices, probs, segment_ids=seg_ids_scaled)
    loss_non_binary = m3_moe.load_balance_loss(indices, probs, mask=non_binary_mask)
    loss_binary = m3_moe.load_balance_loss(indices, probs, mask=binary_mask)

    np.testing.assert_allclose(loss_corrupted, loss_clean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loss_scaled_seg, loss_clean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loss_non_binary, loss_binary, rtol=1e-6, atol=1e-6)

    # Negative control: without `segment_ids`, corrupting padded tokens must change the loss.
    unmasked_clean = m3_moe.load_balance_loss(indices, probs)
    unmasked_corrupted = m3_moe.load_balance_loss(indices_corrupted, probs_corrupted)
    self.assertGreater(abs(float(unmasked_corrupted - unmasked_clean)), 1e-2)

  def test_qwen3_8_router_and_total_loss_binarize_segment_ids_and_mask_padding(self):
    _, _, indices, probs = _make_routing_inputs(batch=1, length=4, embed=8, num_experts=4, top_k=2, seed=41)
    seg_a = jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)
    seg_b = jnp.array([[1, 1, 8, 0]], dtype=jnp.int32)
    aux_a = [{"router_probs": probs, "router_indices": indices}]
    aux_corrupt = [
        {
            "router_probs": probs.at[0, 3].set(jnp.array([1.0, 0.0, 0.0, 0.0])),
            "router_indices": indices.at[0, 3].set(jnp.array([0, 0])),
        }
    ]
    lb_a = modeling_qwen3_8_flash_next.router_load_balancing_loss(aux_a, 4, segment_ids=seg_a)
    lb_b = modeling_qwen3_8_flash_next.router_load_balancing_loss(aux_a, 4, segment_ids=seg_b)
    lb_corrupt = modeling_qwen3_8_flash_next.router_load_balancing_loss(aux_corrupt, 4, segment_ids=seg_a)
    np.testing.assert_allclose(lb_b, lb_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(lb_corrupt, lb_a, rtol=1e-6, atol=1e-6)
    # Passing `segment_ids` as `mask` must also binarize `!= 0` instead of weighting segment 2 by 2x.
    lb_mask = modeling_qwen3_8_flash_next.router_load_balancing_loss(aux_a, 4, mask=seg_a)
    np.testing.assert_allclose(lb_mask, lb_a, rtol=1e-6, atol=1e-6)


class Qwen3_8LiveAdoptionTest(parameterized.TestCase):
  """Verifies live `Qwen3_8FlashNextModel` adoption on `(EP=2, TP=2)` for unscanned & scanned modes."""

  def _mini_qwen3_8_cfg(self, *, scan_layers: bool, use_tokamax_gmm: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        model_name="qwen3.8-flash-next",
        vocab_size=128,
        emb_dim=32,
        num_decoder_layers=4,
        inhomogeneous_layer_cycle_interval=4,
        hc_count=4,
        hc_lowrank=8,
        normalization_layer_epsilon=1e-6,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        gdn_num_key_heads=2,
        gdn_num_value_heads=4,
        gdn_key_head_dim=8,
        gdn_value_head_dim=8,
        gdn_chunk_size=4,
        gdn_conv_kernel_dim=2,
        gdn_output_gate_activation="silu",
        gdn_chunk_inverse="block_doubling",
        use_qk_norm_in_gdn=True,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_max_timescale=10000.0,
        mrope_section=(1, 1, 0),
        index_n_heads=2,
        index_kv_heads=1,
        index_head_dim=8,
        index_select_budget=4,
        index_select_block_size=2,
        qsa_indexer_mask="gather",
        qsa_attention_kernel="dense",
        mlp_dim=16,
        moe_mlp_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_ple_layer=True,
        ple_layer_id=1,
        eos_token_id=0,
        ple_eos_token_id=0,
        ple_embed_dim=32,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=64,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=2,
        ple_conv_kernel_size=2,
        ple_seed=7,
        scan_layers=scan_layers,
        remat_policy="none",
        mtp_num_layers=1,
        mtp_num_streams=4,
        mtp_loss_scaling_factor=0.1,
        load_balance_loss_weight=0.01,
        sparse_matmul=True,
        use_tokamax_gmm=use_tokamax_gmm,
        moe_bwd_wgrad_overlap=True,
    )

  @parameterized.named_parameters(
      ("unscanned_tokamax", False, True),
      ("scan_layers_tokamax", True, True),
  )
  def test_qwen3_8_sparse_moe_block_ep_tp_forward_and_grad_parity(self, scan_layers, use_tokamax_gmm):
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices; got {n}.")
    ici = {"fsdp": 2, "expert": 2, "tensor": 2} if n >= 8 else {"fsdp": 1, "expert": 2, "tensor": 2}
    dev_count = ici["fsdp"] * ici["expert"] * ici["tensor"]

    cfg = self._mini_qwen3_8_cfg(scan_layers=scan_layers, use_tokamax_gmm=use_tokamax_gmm)
    x = jax.random.normal(jax.random.key(3), (4, 8, cfg.emb_dim), dtype=jnp.float32)
    seg_ids = jnp.array([[1, 1, 1, 1, 2, 2, 0, 0]] * 4, dtype=jnp.int32)

    def _build_stack():
      sharding_obj = DefaultSharding()
      executor = m3_moe.moe_from_config(cfg, sharding_obj)
      blocks = nnx.Sequential(
          *[
              modeling_qwen3_8_flash_next.Qwen3_8SparseMoeBlock(
                  emb_dim=cfg.emb_dim,
                  moe_mlp_dim=cfg.moe_mlp_dim,
                  num_experts=cfg.num_experts,
                  num_experts_per_tok=cfg.num_experts_per_tok,
                  norm_topk_prob=cfg.norm_topk_prob,
                  routed_scaling_factor=cfg.routed_scaling_factor,
                  dtype=jnp.float32,
                  weight_dtype=jnp.float32,
                  sharding=sharding_obj,
                  shared_expert_intermediate_size=cfg.mlp_dim,
                  executor=executor,
                  rngs=nnx.Rngs(params=100 + i),
              )
              for i in range(2)
          ]
      )
      return m3_scan.scan_layers(blocks, param_scan_axis=0) if scan_layers else blocks

    def _stack_loss(stack, h):
      if scan_layers:
        out = stack(h)
        return jnp.mean(out**2)
      aux = []
      cur = h
      for layer in stack.layers:
        cur, _, idx, probs = layer(cur, return_router_topk=True)
        aux.append({"router_probs": probs, "router_indices": idx})
      lb = modeling_qwen3_8_flash_next.router_load_balancing_loss(aux, cfg.num_experts, segment_ids=seg_ids)
      return jnp.mean(cur**2) + 0.01 * lb

    mesh_1 = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh_1):
      ref_stack = _build_stack()
      ref_paths = {path for path, _ in nnx.to_flat_state(nnx.state(ref_stack, nnx.Param))}
      ref_loss, (ref_grads, ref_dx) = nnx.jit(nnx.value_and_grad(_stack_loss, argnums=(0, 1)))(ref_stack, x)

    mesh_dist = create_mesh(ici=ici, devices=jax.devices()[:dev_count])
    with jax.set_mesh(mesh_dist):
      dist_stack = _build_stack()
      dist_paths = {path for path, _ in nnx.to_flat_state(nnx.state(dist_stack, nnx.Param))}
      self.assertEqual(dist_paths, ref_paths)

      blk0 = dist_stack if scan_layers else dist_stack.layers[0]
      if scan_layers:
        self.assertEqual(
            blk0.experts.gate_up_proj.get_metadata()["out_sharding"],
            P(None, "expert", ("fsdp", "fsdp_transpose", "context"), None, ("tensor", "tensor_sequence")),
        )
        self.assertEqual(
            blk0.experts.down_proj.get_metadata()["out_sharding"],
            P(None, "expert", ("tensor", "tensor_sequence"), ("fsdp", "fsdp_transpose", "context")),
        )
      else:
        self.assertEqual(
            blk0.experts.gate_up_proj.get_metadata()["out_sharding"],
            P("expert", ("fsdp", "fsdp_transpose", "context"), None, ("tensor", "tensor_sequence")),
        )
        self.assertEqual(
            blk0.experts.down_proj.get_metadata()["out_sharding"],
            P("expert", ("tensor", "tensor_sequence"), ("fsdp", "fsdp_transpose", "context")),
        )

      x_s = jax.device_put(x, NamedSharding(mesh_dist, P(("data", "fsdp", "expert"), None, None)))
      step_fn = nnx.jit(nnx.value_and_grad(_stack_loss, argnums=(0, 1)))
      hlo = step_fn.lower(dist_stack, x_s).compile().as_text()
      for line in hlo.splitlines():
        if "all-gather(" in line:
          self.assertNotIn("f32[4,32,32]", line)
          self.assertNotIn("f32[4,16,32]", line)
          self.assertNotIn("f32[2,4,32,32]", line)
          self.assertNotIn("f32[2,4,16,32]", line)

      dist_loss, (dist_grads, dist_dx) = step_fn(dist_stack, x_s)
      np.testing.assert_allclose(float(dist_loss), float(ref_loss), rtol=2e-4, atol=2e-4)
      np.testing.assert_allclose(np.asarray(dist_dx), np.asarray(ref_dx), rtol=2e-4, atol=2e-4)
      for (p_ref, g_ref), (p_dist, g_dist) in zip(
          nnx.to_flat_state(ref_grads),
          nnx.to_flat_state(dist_grads),
      ):
        self.assertEqual(p_ref, p_dist)
        np.testing.assert_allclose(
            np.asarray(g_dist.get_value()),
            np.asarray(g_ref.get_value()),
            rtol=2e-4,
            atol=2e-4,
            err_msg=f"Gradient mismatch at leaf {p_dist}",
        )

  @parameterized.named_parameters(
      ("unscanned_tokamax", False, True),
      ("scan_layers_tokamax", True, True),
  )
  def test_qwen3_8_full_model_ep_forward_and_all_leaf_grad_parity(self, scan_layers, use_tokamax_gmm):
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices; got {n}.")

    cfg = self._mini_qwen3_8_cfg(scan_layers=scan_layers, use_tokamax_gmm=use_tokamax_gmm)
    tokens = jax.random.randint(jax.random.key(5), (8, 8), 1, cfg.vocab_size, dtype=jnp.int32)
    targets = jax.random.randint(jax.random.key(6), (8, 8), 1, cfg.vocab_size, dtype=jnp.int32)
    seg_ids = jnp.array([[1, 1, 1, 1, 2, 2, 0, 0]] * 8, dtype=jnp.int32)

    mesh_1 = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh_1):
      ref_model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, mesh=mesh_1, rngs=nnx.Rngs(params=0))
      ref_paths = {path for path, _ in nnx.to_flat_state(nnx.state(ref_model, nnx.Param))}
      ref_loss, ref_grads = nnx.jit(
          nnx.value_and_grad(
              lambda m: modeling_qwen3_8_flash_next.compute_total_loss(m, tokens, targets, segment_ids=seg_ids)[0]
          )
      )(ref_model)

    # Verify EP + FSDP + TP `out_sharding` metadata (experts + depthwise `conv1d`) and 100% all-leaf grad parity.
    ici_tp = {"fsdp": 2, "expert": 2, "tensor": 2} if n >= 8 else {"fsdp": 1, "expert": 2, "tensor": 2}
    tp_devs = ici_tp["fsdp"] * ici_tp["expert"] * ici_tp["tensor"]
    mesh_tp = create_mesh(ici=ici_tp, devices=jax.devices()[:tp_devs])
    with jax.set_mesh(mesh_tp):
      tp_model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, mesh=mesh_tp, rngs=nnx.Rngs(params=0))
      if scan_layers:
        gu_var = tp_model.decoder.cycles.gdn_0.mlp.experts.gate_up_proj
        dn_var = tp_model.decoder.cycles.gdn_0.mlp.experts.down_proj
        gdn_conv_var = tp_model.decoder.cycles.gdn_0.linear_attn.conv1d.kernel
        ple_conv_var = tp_model.decoder.ple.conv1d.kernel
        self.assertEqual(
            gu_var.get_metadata()["out_sharding"],
            P(None, "expert", ("fsdp", "fsdp_transpose", "context"), None, ("tensor", "tensor_sequence")),
        )
        self.assertEqual(
            dn_var.get_metadata()["out_sharding"],
            P(None, "expert", ("tensor", "tensor_sequence"), ("fsdp", "fsdp_transpose", "context")),
        )
        self.assertEqual(gdn_conv_var.get_metadata()["out_sharding"], P(None, None, None, ("tensor", "tensor_sequence")))
        self.assertEqual(ple_conv_var.get_metadata()["out_sharding"], P(None, None, ("tensor", "tensor_sequence")))
      else:
        gu_var = tp_model.decoder.layers_0.mlp.experts.gate_up_proj
        dn_var = tp_model.decoder.layers_0.mlp.experts.down_proj
        gdn_conv_var = tp_model.decoder.layers_0.linear_attn.conv1d.kernel
        ple_conv_var = tp_model.decoder.layers_1.ple.conv1d.kernel
        self.assertEqual(
            gu_var.get_metadata()["out_sharding"],
            P("expert", ("fsdp", "fsdp_transpose", "context"), None, ("tensor", "tensor_sequence")),
        )
        self.assertEqual(
            dn_var.get_metadata()["out_sharding"],
            P("expert", ("tensor", "tensor_sequence"), ("fsdp", "fsdp_transpose", "context")),
        )
        self.assertEqual(gdn_conv_var.get_metadata()["out_sharding"], P(None, None, ("tensor", "tensor_sequence")))
        self.assertEqual(ple_conv_var.get_metadata()["out_sharding"], P(None, None, ("tensor", "tensor_sequence")))

      tok_tp = NamedSharding(mesh_tp, P(("data", "fsdp", "expert"), None))
      tp_step_fn = nnx.jit(
          nnx.value_and_grad(lambda m, t, y, s: modeling_qwen3_8_flash_next.compute_total_loss(m, t, y, segment_ids=s)[0])
      )
      tp_loss, tp_grads = tp_step_fn(
          tp_model, jax.device_put(tokens, tok_tp), jax.device_put(targets, tok_tp), jax.device_put(seg_ids, tok_tp)
      )
      np.testing.assert_allclose(float(tp_loss), float(ref_loss), rtol=2e-4, atol=2e-4)

      if scan_layers:
        gdn_g = tp_grads.decoder.cycles.gdn_0.linear_attn.conv1d.kernel.get_value()
        ple_g = tp_grads.decoder.ple.conv1d.kernel.get_value()
      else:
        gdn_g = tp_grads.decoder.layers_0.linear_attn.conv1d.kernel.get_value()
        ple_g = tp_grads.decoder.layers_1.ple.conv1d.kernel.get_value()
      self.assertGreater(float(jnp.max(jnp.abs(gdn_g))), 1e-3)
      self.assertGreater(float(jnp.max(jnp.abs(ple_g))), 1e-3)

      for (p_ref, g_ref), (p_tp, g_tp) in zip(
          nnx.to_flat_state(ref_grads),
          nnx.to_flat_state(tp_grads),
      ):
        self.assertEqual(p_ref, p_tp)
        np.testing.assert_allclose(
            np.asarray(g_tp.get_value()),
            np.asarray(g_ref.get_value()),
            rtol=2e-4,
            atol=2e-4,
            err_msg=f"TP gradient mismatch at leaf {p_tp}",
        )

    # Verify 100% all-leaf backward parameter gradient parity across the entire model under FSDP + EP.
    ici_ep = {"fsdp": 2, "expert": 4, "tensor": 1} if n >= 8 else {"fsdp": 2, "expert": 2, "tensor": 1}
    ep_devs = ici_ep["fsdp"] * ici_ep["expert"] * ici_ep["tensor"]
    mesh_ep = create_mesh(ici=ici_ep, devices=jax.devices()[:ep_devs])
    with jax.set_mesh(mesh_ep):
      dist_model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, mesh=mesh_ep, rngs=nnx.Rngs(params=0))
      dist_paths = {path for path, _ in nnx.to_flat_state(nnx.state(dist_model, nnx.Param))}
      self.assertEqual(dist_paths, ref_paths)
      ep_conv = (
          dist_model.decoder.cycles.gdn_0.linear_attn.conv1d.kernel
          if scan_layers
          else dist_model.decoder.layers_0.linear_attn.conv1d.kernel
      )
      self.assertNotIn("out_sharding", ep_conv.get_metadata())

      tok_s = NamedSharding(mesh_ep, P(("data", "fsdp", "expert"), None))
      tok_d = jax.device_put(tokens, tok_s)
      tgt_d = jax.device_put(targets, tok_s)
      seg_d = jax.device_put(seg_ids, tok_s)

      step_fn = nnx.jit(
          nnx.value_and_grad(lambda m, t, y, s: modeling_qwen3_8_flash_next.compute_total_loss(m, t, y, segment_ids=s)[0])
      )
      dist_loss, dist_grads = step_fn(dist_model, tok_d, tgt_d, seg_d)
      np.testing.assert_allclose(float(dist_loss), float(ref_loss), rtol=2e-4, atol=2e-4)

      for (p_ref, g_ref), (p_dist, g_dist) in zip(
          nnx.to_flat_state(ref_grads),
          nnx.to_flat_state(dist_grads),
      ):
        self.assertEqual(p_ref, p_dist)
        np.testing.assert_allclose(
            np.asarray(g_dist.get_value()),
            np.asarray(g_ref.get_value()),
            rtol=2e-4,
            atol=2e-4,
            err_msg=f"Gradient mismatch at leaf {p_dist}",
        )

  def test_get_abstract_state_nnx_preserves_expert_and_default_shardings(self):
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices; got {n}.")

    ici = {"fsdp": 2, "expert": 2, "tensor": 2} if n >= 8 else {"fsdp": 1, "expert": 2, "tensor": 2}
    dev_count = ici["fsdp"] * ici["expert"] * ici["tensor"]
    mesh = create_mesh(ici=ici, devices=jax.devices()[:dev_count])

    cfg = self._mini_qwen3_8_cfg(scan_layers=True, use_tokamax_gmm=True)
    cfg.use_m3_model = True
    cfg.logical_axis_rules = ()
    cfg.shard_optimizer_over_data = False
    cfg.optimizer_memory_host_offload = False
    cfg.parameter_memory_host_offload = False

    def init_state_fn(use_mask: bool = False):
      nnx_model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(params=0))
      tx = optax.adam(1e-3)
      if use_mask:

        def mask_fn(p):
          del p
          return jax.tree.map_with_path(
              lambda path, _: "token_embedder" not in "".join(str(k) for k in path),
              nnx.state(nnx_model, nnx.Param),
          )

        tx = optax.masked(tx, mask=mask_fn)
      optimizer = nnx.Optimizer(nnx_model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

    _, state_specs, _ = maxtext_utils.get_abstract_state_nnx(
        cfg, mesh, lambda: init_state_fn(use_mask=False), is_training=True
    )
    expected_default = ("fsdp", "expert") if ici["fsdp"] > 1 else "expert"

    gu_spec = state_specs.model.decoder.cycles.gdn_0.mlp.experts.gate_up_proj.get_value()
    dn_spec = state_specs.model.decoder.cycles.gdn_0.mlp.experts.down_proj.get_value()
    emb_spec = state_specs.model.decoder.token_embedder.embedding.get_value()
    se_spec = state_specs.model.decoder.cycles.gdn_0.mlp.shared_expert.gate_proj.kernel.get_value()
    self.assertEqual(
        gu_spec, P(None, "expert", ("fsdp", "fsdp_transpose", "context"), None, ("tensor", "tensor_sequence"))
    )
    self.assertEqual(dn_spec, P(None, "expert", ("tensor", "tensor_sequence"), ("fsdp", "fsdp_transpose", "context")))
    self.assertEqual(emb_spec, P(expected_default, None))
    self.assertEqual(se_spec, P(None, expected_default, None))

    opt_mu = state_specs.optimizer.opt_state[0].mu
    self.assertEqual(opt_mu.decoder.cycles.gdn_0.mlp.experts.gate_up_proj.get_value(), gu_spec)
    self.assertEqual(opt_mu.decoder.cycles.gdn_0.mlp.experts.down_proj.get_value(), dn_spec)
    self.assertEqual(opt_mu.decoder.token_embedder.embedding.get_value(), emb_spec)
    self.assertEqual(opt_mu.decoder.cycles.gdn_0.mlp.shared_expert.gate_proj.kernel.get_value(), se_spec)

    # optax.MaskedNode case proving maxtext_utils.get_abstract_state_nnx succeeds
    # without AttributeError: 'MaskedNode' object has no attribute 'shape'.
    _, masked_specs, _ = maxtext_utils.get_abstract_state_nnx(
        cfg, mesh, lambda: init_state_fn(use_mask=True), is_training=True
    )
    self.assertIsNotNone(masked_specs)

  def test_qwen3_8_experts_single_token_on_tp_mesh_and_conv1d_divisibility(self):
    n = jax.device_count()
    if n < 4:
      self.skipTest(f"Requires >= 4 devices; got {n}.")

    x_1 = jax.random.normal(jax.random.key(61), (1, 1, 16), dtype=jnp.float32)
    idx_1 = jnp.array([[[0, 2]]], dtype=jnp.int32)
    w_1 = jnp.array([[[0.6, 0.4]]], dtype=jnp.float32)
    cot_1 = jax.random.normal(jax.random.key(62), (1, 1, 16), dtype=jnp.float32)

    mesh_1 = create_mesh(devices=jax.devices()[:1])
    ctx_cases = {}
    with jax.set_mesh(mesh_1):
      ref_exp = modeling_qwen3_8_flash_next.Qwen3_8Experts(
          num_experts=4, emb_dim=16, moe_mlp_dim=12, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(params=3)
      )
      ref_out = ref_exp(x_1, idx_1, w_1)
      ref_g_mod, ref_dx = nnx.jit(nnx.grad(lambda m, a: jnp.sum(m(a, idx_1, w_1) * cot_1), argnums=(0, 1)))(ref_exp, x_1)
      for b_sz in (2, 1):
        xb_1, wb_1, ib_1, _ = _make_routing_inputs(batch=b_sz, length=4, embed=16, num_experts=4, top_k=2, seed=60 + b_sz)
        ctx_cases[b_sz] = (
            np.asarray(xb_1),
            np.asarray(wb_1),
            np.asarray(ib_1),
            np.asarray(ref_exp(xb_1, ib_1, wb_1)),
        )

    # Item 4a: (B=1, S=1) on (fsdp=2, expert=1, tensor=2) sets `batch=None` when `ep_size == 1` even with `tp_size > 1`.
    mesh_tp = create_mesh(ici={"fsdp": 2, "expert": 1, "tensor": 2}, devices=jax.devices()[:4])
    with jax.set_mesh(mesh_tp):
      tp_exp = modeling_qwen3_8_flash_next.Qwen3_8Experts(
          num_experts=4, emb_dim=16, moe_mlp_dim=12, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(params=3)
      )
      got_out = nnx.jit(lambda m, a, i, w: m(a, i, w))(tp_exp, x_1, idx_1, w_1)
      got_g_mod, got_dx = nnx.jit(nnx.grad(lambda m, a: jnp.sum(m(a, idx_1, w_1) * cot_1), argnums=(0, 1)))(tp_exp, x_1)
      np.testing.assert_allclose(np.asarray(got_out), np.asarray(ref_out), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(got_dx), np.asarray(ref_dx), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(
          np.asarray(got_g_mod.gate_up_proj.get_value()),
          np.asarray(ref_g_mod.gate_up_proj.get_value()),
          rtol=1e-4,
          atol=1e-4,
      )
      np.testing.assert_allclose(
          np.asarray(got_g_mod.down_proj.get_value()),
          np.asarray(ref_g_mod.down_proj.get_value()),
          rtol=1e-4,
          atol=1e-4,
      )

      # Indivisible `conv1d` channel counts fail closed on `tensor > 1`.
      with self.assertRaisesRegex(ValueError, "conv1d channels"):
        modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
            emb_dim=8,
            num_k_heads=1,
            num_v_heads=1,
            head_k_dim=1,
            head_v_dim=1,
            output_gate_activation="silu",
            rngs=nnx.Rngs(params=0),
        )
      with self.assertRaisesRegex(ValueError, "conv1d channels"):
        modeling_qwen3_8_flash_next.Qwen3_8PLELayer(
            eos_token_id=0,
            emb_dim=3,
            hc_count=1,
            vocab_size=32,
            ple_embed_dim=8,
            ngram_size=2,
            heads_per_ngram=2,
            ngram_vocab_size_base=16,
            make_ngram_vocab_size_divisible_by=8,
            rngs=nnx.Rngs(params=0),
        )

    # Item 3: `Qwen3_8Experts` wires `length="context"` and handles both batch-divisible `(2, 4, 16)`
    # and token-flattened `(1, 4, 16)` inputs on `(context=2, expert=2)`.
    mesh_ctx = create_mesh(ici={"context": 2, "expert": 2}, devices=jax.devices()[:4])
    with jax.set_mesh(mesh_ctx):
      ctx_exp = modeling_qwen3_8_flash_next.Qwen3_8Experts(
          num_experts=4, emb_dim=16, moe_mlp_dim=12, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(params=3)
      )
      self.assertEqual(ctx_exp.executor.length, "context")
      for b_sz, (xb_np, wb_np, ib_np, expected_b) in ctx_cases.items():
        tok_spec = P("expert", "context", None) if b_sz == 2 else P(None, "context", None)
        tok_shd = NamedSharding(mesh_ctx, tok_spec)
        xb = jax.device_put(xb_np, tok_shd)
        wb = jax.device_put(wb_np, tok_shd)
        ib = jax.device_put(ib_np, tok_shd)
        got_b = nnx.jit(lambda m, a, i, w: m(a, i, w))(ctx_exp, xb, ib, wb)
        np.testing.assert_allclose(np.asarray(got_b), expected_b, rtol=1e-4, atol=1e-4)


class ValidationErrorTest(parameterized.TestCase):
  """Tests fail-closed runtime shape/mesh checks and optimization barrier VJP helpers."""

  def test_barriers_preserve_values_and_cotangents(self):
    x = jnp.array([1.0, -2.0, 3.5], dtype=jnp.float32)
    y_fwd, dx_fwd = jax.value_and_grad(lambda z: jnp.sum(m3_moe.forward_optimization_barrier(z) ** 2))(x)
    y_bwd, dx_bwd = jax.value_and_grad(lambda z: jnp.sum(m3_moe.backward_optimization_barrier(z) ** 2))(x)
    np.testing.assert_allclose(y_fwd, jnp.sum(x**2))
    np.testing.assert_allclose(y_bwd, jnp.sum(x**2))
    np.testing.assert_allclose(dx_fwd, 2.0 * x)
    np.testing.assert_allclose(dx_bwd, 2.0 * x)

  def test_runtime_shape_and_mesh_errors(self):
    x, weights, indices, probs = _make_routing_inputs(batch=3, length=4, embed=8, num_experts=4, top_k=2)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=8, mlp=6)

    # Sharded backend with no mesh in scope fails closed.
    with self.assertRaisesRegex(ValueError, "active mesh"):
      m3_moe.MoEExecutor("tokamax")(x, weights, indices, wi_0, wi_1, wo)

    with self.assertRaisesRegex(ValueError, "Unknown grouped matmul backend"):
      m3_moe.grouped_matmul(jnp.zeros((4, 8)), wi_0, jnp.ones((4,), dtype=jnp.int32), backend="invalid")

    with self.assertRaisesRegex(ValueError, "Expected 3D"):
      m3_moe.routed_experts(x[0], weights, indices, wi_0, wi_1, wo)

    with self.assertRaisesRegex(ValueError, "even"):
      m3_moe.routed_experts_fused(x, weights, indices, jnp.zeros((4, 8, 7)), wo)

    with self.assertRaisesRegex(ValueError, "size 2"):
      m3_moe.routed_experts_fused(x, weights, indices, jnp.zeros((4, 8, 3, 6)), wo)

    with self.assertRaisesRegex(ValueError, "mask shape"):
      m3_moe.load_balance_loss(indices, probs, mask=jnp.ones((2, 4)))

    with self.assertRaisesRegex(ValueError, "segment_ids shape"):
      m3_moe.load_balance_loss(indices, probs, segment_ids=jnp.ones((3, 2)))

    n = jax.device_count()
    if n >= 4:
      mesh = create_mesh(ici={"fsdp": 1, "expert": 2, "tensor": 2}, devices=jax.devices()[:4])
      with jax.set_mesh(mesh):
        # Batch 3 does not divide over 2 token shards (expert=2).
        with self.assertRaisesRegex(ValueError, "does not divide"):
          m3_moe.MoEExecutor("tokamax", batch="expert", experts="expert", mlp="tensor")(
              x, weights, indices, wi_0, wi_1, wo
          )
        # num_experts=3 not divisible by ep_size=2.
        w0_3, w1_3, wo_3 = _make_expert_weights(num_experts=3, embed=8, mlp=8)
        x4 = jnp.zeros((4, 4, 8), dtype=jnp.float32)
        w4 = jnp.ones((4, 4, 2), dtype=jnp.float32)
        i4 = jnp.zeros((4, 4, 2), dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "divisible by expert parallelism"):
          m3_moe.MoEExecutor("tokamax", batch="expert", experts="expert", mlp="tensor")(x4, w4, i4, w0_3, w1_3, wo_3)
        # mlp=5 not divisible by tp_size=2.
        w0_5, w1_5, wo_5 = _make_expert_weights(num_experts=4, embed=8, mlp=5)
        with self.assertRaisesRegex(ValueError, "divisible by tensor parallelism"):
          m3_moe.MoEExecutor("tokamax", batch="expert", experts="expert", mlp="tensor")(x4, w4, i4, w0_5, w1_5, wo_5)

  def test_qwen3_8_sparse_moe_block_context_parallel_emits_zero_all_gathers(self):
    """Verifies Qwen3_8SparseMoeBlock preserves 3D sequence sharding across (fsdp=2, context=2, expert=2)."""
    from jax.experimental.topologies import get_topology_desc  # pylint: disable=import-outside-toplevel

    try:
      devs = get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.skipTest(f"Virtual tpu7x:2x2x1 topology unavailable: {exc}")

    mesh = create_mesh(ici={"fsdp": 2, "context": 2, "expert": 2}, devices=devs)
    shd = DefaultSharding()
    abs_block = nnx.eval_shape(
        lambda: modeling_qwen3_8_flash_next.Qwen3_8SparseMoeBlock(
            emb_dim=256,
            moe_mlp_dim=128,
            num_experts=8,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=128,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.bfloat16,
            sharding=shd,
            rngs=nnx.Rngs(params=7),
        )
    )
    gdef, abs_st = nnx.split(abs_block)

    def _bind_param(path, leaf):
      key_strs = [str(getattr(k, "key", getattr(k, "name", k))) for k in path]
      spec = P("expert", *([None] * (leaf.ndim - 1))) if "experts" in key_strs else P()
      return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=NamedSharding(mesh, spec))

    abs_st_mesh = jax.tree_util.tree_map_with_path(_bind_param, abs_st)
    x_abs = jax.ShapeDtypeStruct(
        (4, 128, 256), jnp.bfloat16, sharding=NamedSharding(mesh, P(("fsdp", "expert"), "context", None))
    )
    with jax.set_mesh(mesh):
      comp_3d = jax.jit(lambda s_in, inp: nnx.merge(gdef, s_in)(inp)).lower(abs_st_mesh, x_abs).compile()
      comp_2d = (
          jax.jit(lambda s_in, inp: nnx.merge(gdef, s_in)(inp.reshape(-1, 256)).reshape(inp.shape))
          .lower(abs_st_mesh, x_abs)
          .compile()
      )
    ag_3d = [ln for ln in comp_3d.as_text().splitlines() if "=" in ln and "all-gather(" in ln.split("=", 1)[1]]
    ag_2d = [ln for ln in comp_2d.as_text().splitlines() if "=" in ln and "all-gather(" in ln.split("=", 1)[1]]
    self.assertEmpty(ag_3d, f"Expected 0 all-gathers for 3D SparseMoeBlock, got: {ag_3d}")
    self.assertGreaterEqual(len(ag_2d), 3, "2D flatten negative control should emit >= 3 sequence all-gathers")

  def test_grouped_unsharded_fused_cpu_host_with_is_tpu_true_falls_back_safely(self):
    """Verifies grouped_unsharded_fused(..., is_tpu=True) does not crash when local devices are CPU or empty."""
    x, weights, indices, _ = _make_routing_inputs(batch=2, length=4, embed=16, num_experts=4, top_k=2)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=8)
    w_gu = jnp.stack([wi_0, wi_1], axis=2)
    ref = m3_moe.routed_experts_fused(x, weights, indices, w_gu, wo, executor=m3_moe.DENSE)

    for kind in ("dense", "ragged_dot", "gmm_v2", "tokamax"):
      out_cpu = m3_moe.grouped_unsharded_fused(
          x, weights, indices, w_gu, wo, executor=m3_moe.MoEExecutor(kind=kind), is_tpu=True
      )
      np.testing.assert_allclose(np.asarray(out_cpu), np.asarray(ref), rtol=1e-4, atol=1e-4)

    with mock.patch.object(jax, "local_devices", return_value=[]):
      out_empty = m3_moe.grouped_unsharded_fused(
          jnp.asarray(np.asarray(x)),
          weights,
          indices,
          w_gu,
          wo,
          executor=m3_moe.MoEExecutor(kind="gmm_v2"),
          is_tpu=True,
      )
      np.testing.assert_allclose(np.asarray(out_empty), np.asarray(ref), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
