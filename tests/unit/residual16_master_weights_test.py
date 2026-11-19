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

"""Unit, Orbax round-trip, sharding, and config tests for residual16 master weight storage."""

import importlib.util
import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, reshard
import numpy as np
import optax
from optax.contrib._muon import muon as optax_muon
from orbax.checkpoint import v1 as ocp
from pydantic import ValidationError
import pytest

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.experimental.lineage import lineage_adapter
from maxtext.layers import attention_compressed
from maxtext.layers import attention_kda
from maxtext.layers import attention_mla
from maxtext.layers import attentions
from maxtext.layers import embeddings
from maxtext.layers import initializers
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import nnx_scan
from maxtext.models import gemma4
from maxtext.models import hy4
from maxtext.models import ming
from maxtext.models import qwen3
from maxtext.utils import lora_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import qk_clip_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path


class _TinyResidualModel(nnx.Module):
  """Compact 2-layer NNX model exercising 2D kernels, 1D biases, and optional router bias."""

  def __init__(self, rngs: nnx.Rngs, include_gate: bool = False):
    self.embed = nnx.Param(
        jax.random.normal(rngs.params(), (16, 8), dtype=jnp.float32),
        out_sharding=("fsdp", None),
    )
    self.linear1 = nnx.Linear(8, 16, dtype=jnp.bfloat16, rngs=rngs)
    self.linear2 = nnx.Linear(16, 8, dtype=jnp.bfloat16, use_bias=False, rngs=rngs)
    if include_gate:
      self.gate = nnx.Dict({"bias": nnx.Param(jnp.zeros((4,), dtype=jnp.float32), keep_fp32_master=True)})

  def __call__(self, tokens: jax.Array) -> jax.Array:
    x = self.embed.get_value().astype(jnp.bfloat16)[tokens]
    h = jax.nn.silu(self.linear1(x))
    return self.linear2(h)


class _TinyMLAModel(nnx.Module):
  """Minimal MLA projection stub for verifying QK-Clip updates both bf16 weights and residuals."""

  def __init__(self, rngs: nnx.Rngs):
    self.self_attention = nnx.Dict(
        {
            "wq_b": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (4, 2, 8), dtype=jnp.float32))}),
            "wkv_b": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (4, 2, 8), dtype=jnp.float32))}),
        }
    )


def _ckpt_config(
    load_parameters_path: str = "",
    load_full_state_path: str = "",
    fp32_master_storage: str = "f32",
    optimizer_memory_host_offload: bool = False,
    parameter_memory_host_offload: bool = False,
):
  return SimpleNamespace(
      enable_diloco=False,
      enable_checkpointing=True,
      enable_continuous_checkpointing=False,
      enable_emergency_checkpoint=False,
      enable_multi_tier_checkpointing=False,
      enable_autocheckpoint=False,
      save_checkpoint_on_start=True,
      checkpoint_period=1,
      local_checkpoint_period=0,
      async_checkpointing=False,
      dataset_type="tfds",
      lora=None,
      checkpoint_storage_target_data_file_size_bytes=checkpointing.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE,
      elastic_enabled=False,
      logical_axis_rules=[("fsdp", "fsdp")],
      load_parameters_path=load_parameters_path,
      load_full_state_path=load_full_state_path,
      checkpoint_storage_concurrent_gb=8,
      enable_single_replica_ckpt_restoring=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      enable_orbax_v1=False,
      checkpoint_conversion_fn=None,
      source_checkpoint_layout="orbax",
      expansion_factor_real_data=-1,
      scan_layers=False,
      param_scan_axis=0,
      shard_optimizer_over_data=False,
      optimizer_memory_host_offload=optimizer_memory_host_offload,
      parameter_memory_host_offload=parameter_memory_host_offload,
      fp32_master_storage=fp32_master_storage,
      weight_dtype=jnp.float32,
      routed_bias=False,
      float32_gate_logits=False,
      use_m3_model=False,
      enable_single_controller=False,
      colocated_python_checkpointing=False,
  )


def _ts_config(**overrides):
  """Builds a minimal train-state test config stub with default attributes."""
  base = {
      "fp32_master_storage": "f32",
      "weight_dtype": jnp.float32,
      "routed_bias": False,
      "float32_gate_logits": False,
  }
  base.update(overrides)
  return SimpleNamespace(**base)


class TestBitLevelResidual16Encoding(unittest.TestCase):
  """Exhaustive IEEE-754 edge-case, subnormal FTZ/NaN-guard, and random bit-exact verification."""

  def test_exact_roundtrip_across_ieee754_edge_cases_and_random_patterns(self):
    edge_u32 = np.array(
        [
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x00000001,  # smallest positive subnormal
            0x80000001,  # smallest negative subnormal
            0x00007FFF,  # just below positive subnormal midpoint
            0x80007FFF,  # just below negative subnormal midpoint
            0x00008000,  # positive subnormal with bit 15 == 1 (previously wrapped to -NaN on TPU)
            0x80008000,  # negative subnormal with bit 15 == 1 (previously wrapped to +NaN on TPU)
            0x00008001,  # just above positive subnormal midpoint
            0x005CB1CB,  # interior positive subnormal with bit 15 == 1
            0x8011CA80,  # interior negative subnormal with bit 15 == 1
            0x007FFFFF,  # largest positive subnormal
            0x807FFFFF,  # largest negative subnormal
            0x00800000,  # smallest positive normal (2^-126)
            0x80800000,  # smallest negative normal (-2^-126)
            0x00808000,  # smallest positive normal + half bf16 ulp
            0x80808000,  # smallest negative normal - half bf16 ulp
            0x3F800000,  # +1.0
            0xBF800000,  # -1.0
            0x3F807FFF,  # 1.0 + (half bf16 ulp - 1 f32 ulp)
            0x3F808000,  # 1.0 + exact half bf16 ulp (even bf16)
            0x3F808001,  # 1.0 + (half bf16 ulp + 1 f32 ulp)
            0x3F818000,  # exact half bf16 ulp (odd bf16)
            0xBF808000,  # negative exact tie
            0x33800000,  # 2^-24
            0x4B800000,  # 2^24
            0x7F7F7FFF,  # near max finite without bf16 overflow
            0x7F7F8000,  # rounds bf16 up to +Inf while residual stays negative!
            0x7F7FFFFF,  # +finfo(float32).max (bf16 rounds to +Inf, residual reconstructs max!)
            0xFF7FFFFF,  # -finfo(float32).max
            0x7F800000,  # +Inf
            0xFF800000,  # -Inf
        ],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(2026)
    rand_u32 = rng.integers(0, 0x100000000, size=200_000, dtype=np.uint32)
    rand_f32 = rand_u32.view(np.float32)
    rand_u32 = rand_u32[~np.isnan(rand_f32)]

    all_u32 = np.concatenate([edge_u32, rand_u32])
    x_f32 = jax.lax.bitcast_convert_type(jnp.asarray(all_u32), jnp.float32)
    w_bf16, res_i16 = train_state_nnx.split_f32_to_bf16_residual16(x_f32)
    self.assertEqual(w_bf16.dtype, jnp.bfloat16)
    self.assertEqual(res_i16.dtype, jnp.int16)
    self.assertFalse(bool(jnp.any(jnp.isnan(w_bf16))))

    rec_f32 = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
    self.assertEqual(rec_f32.dtype, jnp.float32)
    self.assertFalse(bool(jnp.any(jnp.isnan(rec_f32))))
    rec_u32 = np.asarray(jax.lax.bitcast_convert_type(rec_f32, jnp.uint32))

    is_subnormal = ((all_u32 & np.uint32(0x7F800000)) == np.uint32(0)) & (
        (all_u32 & np.uint32(0x007FFFFF)) != np.uint32(0)
    )
    self.assertGreater(int(np.sum(is_subnormal)), 0)
    # 0-ULP bit-for-bit round-trip for every normal float32, +/-0.0, +/-finfo.max, and +/-Inf:
    np.testing.assert_array_equal(rec_u32[~is_subnormal], all_u32[~is_subnormal])
    # Signed-zero FTZ (never NaN or bfloat16 subnormal) for IEEE-754 float32 subnormals:
    expected_subnormal_u32 = all_u32[is_subnormal] & np.uint32(0x80000000)
    np.testing.assert_array_equal(rec_u32[is_subnormal], expected_subnormal_u32)
    w_u16 = np.asarray(jax.lax.bitcast_convert_type(w_bf16, jnp.uint16))
    np.testing.assert_array_equal(w_u16[is_subnormal], (expected_subnormal_u32 >> np.uint32(16)).astype(np.uint16))
    np.testing.assert_array_equal(np.asarray(res_i16)[is_subnormal], np.zeros(int(np.sum(is_subnormal)), dtype=np.int16))

  def test_zero_bf16_negative_residual_guard_and_subnormal_checkpoint_restore_step1(self):
    # 1. Explicit guard on (+/-0.0 bf16, negative int16 residuals [-32768, -14080, -1]):
    w_zero_u16 = jnp.array([0x0000, 0x0000, 0x0000, 0x8000, 0x8000, 0x8000], dtype=jnp.uint16)
    w_zero_bf16 = jax.lax.bitcast_convert_type(w_zero_u16, jnp.bfloat16)
    neg_res_i16 = jnp.array([-32768, -14080, -1, -32768, -14080, -1], dtype=jnp.int16)
    expected_zero_u32 = np.array(
        [0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000], dtype=np.uint32
    )
    for fn in (
        train_state_nnx.reconstruct_f32_from_bf16_residual16,
        jax.jit(train_state_nnx.reconstruct_f32_from_bf16_residual16),
    ):
      rec_zero = fn(w_zero_bf16, neg_res_i16)
      self.assertFalse(bool(jnp.any(jnp.isnan(rec_zero))))
      np.testing.assert_array_equal(
          np.asarray(jax.lax.bitcast_convert_type(rec_zero, jnp.uint32)),
          expected_zero_u32,
      )

    # 2. End-to-end `_split_f32_concrete_for_restore`, `overlay_raw_params_with_residual16`,
    #    `reshape_restored_weights_for_residual16`, and `resplit_model_params_with_residual16`
    #    with subnormal float32 checkpoint weights (`0x00008000`, `0x8011CA80`, etc.):
    prev_eager = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev_eager)

    n_dev = min(len(jax.devices()), 2)
    mesh = Mesh(np.array(jax.devices()[:n_dev]), ("fsdp",))
    dev_sharding = NamedSharding(mesh, P("fsdp", None), memory_kind="device")
    sub_and_norm_u32 = np.tile(
        np.array(
            [
                0x00008000,  # +subnormal (bit 15 == 1)
                0x8011CA80,  # -subnormal (bit 15 == 1)
                0x00008001,  # +subnormal
                0x005CB1CB,  # +subnormal
                0x3DCCCCCD,  # +0.1 normal
                0xBDCCCCCD,  # -0.1 normal
                0x3E808000,  # +0.25 + half bf16 ulp normal
                0xBE808000,  # -0.25 - half bf16 ulp normal
            ],
            dtype=np.uint32,
        ),
        (16, 1),
    )
    ckpt_embed_f32 = jax.device_put(
        jax.lax.bitcast_convert_type(jnp.asarray(sub_and_norm_u32), jnp.float32),
        dev_sharding,
    )
    w_split, r_split = train_state_nnx._split_f32_concrete_for_restore(  # pylint: disable=protected-access
        ckpt_embed_f32, dev_sharding, dev_sharding
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(w_split))))
    rec_split = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_split, r_split)
    self.assertFalse(bool(jnp.any(jnp.isnan(rec_split))))
    rec_split_u32 = np.asarray(jax.lax.bitcast_convert_type(rec_split, jnp.uint32))
    expected_row_u32 = np.array(
        [
            0x00000000,
            0x80000000,
            0x00000000,
            0x00000000,
            0x3DCCCCCD,
            0xBDCCCCCD,
            0x3E808000,
            0xBE808000,
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(rec_split_u32, np.tile(expected_row_u32, (16, 1)))

    # Overlay onto Residual16Optimizer state and verify Step-1 loss, gradients, and updated weights are finite:
    cfg_r16 = _ts_config(fp32_master_storage="residual16")
    with jax.set_mesh(mesh):
      model = _TinyResidualModel(nnx.Rngs(29))
      model.embed.set_value(jax.device_put(model.embed.get_value(), dev_sharding))
      ts = train_state_nnx.create_train_state(model, optax.adamw(1e-3), cfg_r16)
      graphdef, state = nnx.split(ts)

      linear1_u32 = np.full((8, 16), 0x3D4CCCCD, dtype=np.uint32)
      linear1_u32[0, 0] = np.uint32(0x00008000)
      linear1_u32[0, 1] = np.uint32(0x8011CA80)
      ckpt_linear1_f32 = jax.lax.bitcast_convert_type(jnp.asarray(linear1_u32), jnp.float32)

      self.assertTrue(
          train_state_nnx.overlay_raw_params_with_residual16(
              state,
              {"embed": ckpt_embed_f32, "linear1": {"kernel": ckpt_linear1_f32}},
          )
      )
      ts_restored = nnx.merge(graphdef, state)
      tokens = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)

      @nnx.jit
      def _step1(ts_in, tok):
        def _loss_fn(mdl):
          return jnp.mean(mdl(tok).astype(jnp.float32) ** 2)

        loss_val, grads_val = nnx.value_and_grad(_loss_fn)(ts_in.model)
        grads_f32 = jax.tree.map(
            lambda g: g.astype(jnp.float32) if hasattr(g, "dtype") and g.dtype == jnp.bfloat16 else g,
            grads_val,
        )
        ts_in.apply_gradients(grads_f32)
        return loss_val, grads_f32

      loss1, grads1 = _step1(ts_restored, tokens)
      self.assertTrue(bool(jnp.isfinite(loss1)))
      for grad_leaf in jax.tree.leaves(grads1):
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad_leaf))))
      ckpt_after_step1 = train_state_nnx.to_checkpoint_dict(ts_restored)["params"]["params"]
      for param_leaf in jax.tree.leaves(ckpt_after_step1):
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(param_leaf)))))

      # Also verify `reshape_restored_weights_for_residual16` and `resplit_model_params_with_residual16`:
      _, fresh_state = nnx.split(
          train_state_nnx.create_train_state(_TinyResidualModel(nnx.Rngs(30)), optax.adamw(1e-3), cfg_r16)
      )
      reshaped = train_state_nnx.reshape_restored_weights_for_residual16(
          {"model": {"embed": ckpt_embed_f32}, "optimizer": {}},
          fresh_state,
      )
      rec_reshaped = train_state_nnx.reconstruct_f32_from_bf16_residual16(
          reshaped["model"]["embed"], reshaped["optimizer"]["master_residuals"]["embed"]
      )
      self.assertFalse(bool(jnp.any(jnp.isnan(rec_reshaped))))

      nnx.replace_by_pure_dict(fresh_state["model"], {"embed": ckpt_embed_f32})
      self.assertTrue(train_state_nnx.resplit_model_params_with_residual16(fresh_state))
      rec_resplit = train_state_nnx.to_checkpoint_dict(fresh_state)["params"]["params"]["embed"]
      self.assertFalse(bool(np.any(np.isnan(np.asarray(rec_resplit)))))

  def test_nan_patterns_never_wrap_to_zero_or_finite(self):
    nan_u32 = np.array([0x7FC00000, 0x7FC00001, 0x7FFFFFFF, 0xFFC00001, 0xFFFFFFFF], dtype=np.uint32)
    x_nan = jax.lax.bitcast_convert_type(jnp.asarray(nan_u32), jnp.float32)
    w_bf16, res_i16 = train_state_nnx.split_f32_to_bf16_residual16(x_nan)
    rec_f32 = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
    self.assertTrue(bool(jnp.all(jnp.isnan(w_bf16))))
    self.assertTrue(bool(jnp.all(jnp.isnan(rec_f32))))

  def test_exact_bf16_inputs_produce_zero_residual(self):
    bf16_vals = jnp.array([-3.5, -0.0, 0.0, 0.125, 1.0, 42.0], dtype=jnp.bfloat16)
    w_bf16, res_i16 = train_state_nnx.split_f32_to_bf16_residual16(bf16_vals.astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(res_i16), np.zeros_like(np.asarray(res_i16)))
    rec = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(rec, jnp.uint32)),
        np.asarray(jax.lax.bitcast_convert_type(bf16_vals.astype(jnp.float32), jnp.uint32)),
    )


class TestSmallLRConvergenceParity(unittest.TestCase):
  """Verifies residual16 accumulates sub-bf16-ULP updates where pure bf16 weights stall at zero."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def _run_small_lr_comparison(self, tx: optax.GradientTransformation):
    """Runs 40 small-LR steps comparing f32, residual16, and pure bf16 weights."""
    cfg_f32 = _ts_config(fp32_master_storage="f32")
    cfg_r16 = _ts_config(fp32_master_storage="residual16")

    m_f32 = _TinyResidualModel(nnx.Rngs(11))
    m_bf16_only = _TinyResidualModel(nnx.Rngs(11))
    # Pure bf16 weights without residual16 master weights:
    _, bf16_p, _ = nnx.split(m_bf16_only, nnx.Param, ...)
    nnx.update(
        m_bf16_only,
        jax.tree.map(
            lambda v: v.replace(value=v.get_value().astype(jnp.bfloat16)),
            bf16_p,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        ),
    )

    ts_f32 = train_state_nnx.create_train_state(m_f32, tx, cfg_f32)
    ts_bf16 = train_state_nnx.TrainStateNNX(m_bf16_only, nnx.Optimizer(m_bf16_only, tx, wrt=nnx.Param))

    init_bf16_embed = np.array(ts_bf16.model.embed.get_value())
    tokens = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)

    @nnx.jit
    def step_fn(ts):
      def loss_fn(model):
        out = model(tokens)
        return jnp.mean(out.astype(jnp.float32) ** 2)

      grads = nnx.grad(loss_fn)(ts.model)
      ts.apply_gradients(grads)

    # Align m_f32 and m_r16 initial weights to exact bf16-rounded values so step-0 forward is identical:
    _, f32_p, _ = nnx.split(ts_f32.model, nnx.Param, ...)
    nnx.update(
        ts_f32.model,
        jax.tree.map(
            lambda v: v.replace(value=v.get_value().astype(jnp.bfloat16).astype(jnp.float32)),
            f32_p,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        ),
    )
    m_r16_aligned = _TinyResidualModel(nnx.Rngs(11))
    _, r16_p, _ = nnx.split(m_r16_aligned, nnx.Param, ...)
    nnx.update(
        m_r16_aligned,
        jax.tree.map(
            lambda v: v.replace(value=v.get_value().astype(jnp.bfloat16).astype(jnp.float32)),
            r16_p,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        ),
    )
    ts_r16 = train_state_nnx.create_train_state(m_r16_aligned, tx, cfg_r16)
    self.assertIsInstance(ts_r16.optimizer, train_state_nnx.Residual16Optimizer)
    self.assertEqual(type(ts_r16.optimizer.opt_state), type(ts_f32.optimizer.opt_state))
    self.assertIn("embed", ts_r16.optimizer.master_residuals)

    for _ in range(40):
      step_fn(ts_f32)
      step_fn(ts_r16)
      step_fn(ts_bf16)

    # Pure bf16 stalls completely at lr=1e-5 around ~O(1) weights (bf16 ULP at 1.0 is 2^-7 = 7.8e-3 >> 1e-5):
    np.testing.assert_array_equal(np.asarray(ts_bf16.model.embed.get_value()), init_bf16_embed)

    # Reconstructed residual16 master weights advance and closely track f32 master weights:
    r16_ckpt = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16))
    f32_ckpt = train_state_nnx.to_checkpoint_dict(nnx.state(ts_f32))
    r16_rec_embed = np.asarray(r16_ckpt["params"]["params"]["embed"])
    f32_embed = np.asarray(f32_ckpt["params"]["params"]["embed"])
    self.assertGreater(np.max(np.abs(r16_rec_embed - init_bf16_embed.astype(np.float32))), 2e-4)
    np.testing.assert_allclose(r16_rec_embed, f32_embed, rtol=1e-5, atol=1e-6)

  def test_adamw_small_lr_updates_match_f32_while_bf16_stalls(self):
    self._run_small_lr_comparison(optax.adamw(learning_rate=1e-5, weight_decay=0.01))

  def test_muon_small_lr_updates_match_f32_while_bf16_stalls(self):
    self._run_small_lr_comparison(optax_muon(learning_rate=1e-5, weight_decay=0.01))

  def test_qk_clip_updates_both_working_weights_and_residuals(self):
    nnx.use_eager_sharding(False)
    cfg = _ts_config(
        fp32_master_storage="residual16",
        attention_type="mla",
        qk_clip_threshold=10.0,
        qk_nope_head_dim=4,
    )
    m_r16 = _TinyMLAModel(nnx.Rngs(3))
    m_f32 = _TinyMLAModel(nnx.Rngs(3))
    ts_r16 = train_state_nnx.create_train_state(m_r16, optax.adamw(1e-3), cfg)
    ts_f32 = train_state_nnx.TrainStateNNX(m_f32, nnx.Optimizer(m_f32, optax.adamw(1e-3), wrt=nnx.Param))

    intermediates = {"self_attention": {"max_logits": jnp.array([[25.0, 5.0]], dtype=jnp.float32)}}
    qk_clip_utils.apply_qk_clip_nnx(ts_r16, intermediates, cfg)
    qk_clip_utils.apply_qk_clip_nnx(ts_f32, intermediates, cfg)

    r16_rec = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16))["params"]["params"]
    f32_ref = train_state_nnx.to_checkpoint_dict(nnx.state(ts_f32))["params"]["params"]
    np.testing.assert_allclose(
        r16_rec["self_attention"]["wq_b"]["kernel"],
        f32_ref["self_attention"]["wq_b"]["kernel"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        r16_rec["self_attention"]["wkv_b"]["kernel"],
        f32_ref["self_attention"]["wkv_b"]["kernel"],
        rtol=0,
        atol=0,
    )


class TestOrbaxCheckpointCompatibility(unittest.TestCase):
  """Full Orbax save, on-disk schema/dtype inspection, cross-restore, params-only, and host-offload tests."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)
    self._tmpdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
    self._mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))

  def _make_manager(self, subdir: str):
    return checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._tmpdir, subdir),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )

  def _init_fn(self, cfg, seed: int = 0):
    tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)

    def fn():
      model = _TinyResidualModel(nnx.Rngs(seed), include_gate=True)
      return train_state_nnx.create_train_state(model, tx, cfg)

    return fn

  def _sharded_abstract_and_shardings(self, init_fn):
    repl = NamedSharding(self._mesh, P())
    abstract = nnx.eval_shape(lambda: nnx.state(init_fn()))
    sharded_abs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=repl) if hasattr(x, "shape") else x,
        abstract,
    )
    shardings = jax.tree.map(lambda _: repl, abstract)
    return sharded_abs, shardings

  def _setup_state(self, manager, init_fn, cfg):
    abstract, shardings = self._sharded_abstract_and_shardings(init_fn)
    with mock.patch.object(maxtext_utils, "get_abstract_state", return_value=(abstract, None, shardings)):
      state, *_ = maxtext_utils.setup_initial_state(None, cfg, self._mesh, manager, init_fn, True)
    return state

  def test_residual16_orbax_save_produces_standard_f32_checkpoint_and_cross_restores_bit_exact(self):
    cfg_r16 = _ckpt_config(fp32_master_storage="residual16")
    cfg_r16.routed_bias = True
    cfg_f32 = _ckpt_config(fp32_master_storage="f32")
    cfg_f32.routed_bias = True

    mgr = self._make_manager("r16_ckpt")
    init_r16 = self._init_fn(cfg_r16, seed=10)
    ts_r16 = init_r16()
    tokens = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)

    for _ in range(3):
      grads = nnx.grad(lambda m: jnp.mean(m(tokens).astype(jnp.float32) ** 2))(ts_r16.model)
      ts_r16.apply_gradients(grads)

    # Save checkpoint at step 3 from residual16 state:
    checkpointing.save_checkpoint(mgr, 3, ts_r16, config=cfg_r16, force=True)
    checkpointing.wait_until_finished(mgr)

    # 1. Inspect raw on-disk Orbax checkpoint metadata & values: must be 100% standard float32 MaxText layout
    # with ZERO `master_residuals` or `"inner"` keys leaked!
    step_path = os.path.join(self._tmpdir, "r16_ckpt", "3")
    raw_loaded = ocp.load(step_path, checkpointable_name="items")
    self.assertIn("params", raw_loaded)
    self.assertIn("opt_state", raw_loaded)
    self.assertNotIn("master_residuals", str(raw_loaded.keys()))
    if isinstance(raw_loaded["opt_state"], dict):
      self.assertNotIn("master_residuals", raw_loaded["opt_state"])
      self.assertNotIn("inner", raw_loaded["opt_state"])
    for leaf in jax.tree.leaves(raw_loaded["params"]["params"]):
      self.assertEqual(leaf.dtype, jnp.float32)

    # Advance original ts_r16 one more step for bit-exact continuation reference:
    next_tokens = jnp.array([[2, 4, 6, 8], [1, 3, 5, 7]], dtype=jnp.int32)
    grads_ref = nnx.grad(lambda m: jnp.mean(m(next_tokens).astype(jnp.float32) ** 2))(ts_r16.model)
    ts_r16.apply_gradients(grads_ref)
    expected_after_step4 = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16))["params"]["params"]

    # 2. Restore into residual16 (with a different init seed!) and verify bit-exact continued step:
    init_r16_fresh = self._init_fn(cfg_r16, seed=999)
    restored_r16_state = self._setup_state(mgr, init_r16_fresh, cfg_r16)
    self.assertIn("master_residuals", restored_r16_state["optimizer"])
    graphdef_r16 = nnx.graphdef(init_r16_fresh())
    ts_r16_resumed = nnx.merge(graphdef_r16, restored_r16_state)
    self.assertEqual(ts_r16_resumed.model.embed.get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_r16_resumed.model.gate["bias"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16_resumed.optimizer.master_residuals["embed"].get_value().dtype, jnp.int16)
    grads_resumed = nnx.grad(lambda m: jnp.mean(m(next_tokens).astype(jnp.float32) ** 2))(ts_r16_resumed.model)
    ts_r16_resumed.apply_gradients(grads_resumed)
    actual_r16_step4 = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16_resumed))["params"]["params"]
    for p_exp, p_got in zip(jax.tree.leaves(expected_after_step4), jax.tree.leaves(actual_r16_step4)):
      np.testing.assert_array_equal(
          np.asarray(jax.lax.bitcast_convert_type(p_got, jnp.uint32)),
          np.asarray(jax.lax.bitcast_convert_type(p_exp, jnp.uint32)),
      )

    # 3. Cross-restore residual16 checkpoint into f32 train state:
    init_f32_fresh = self._init_fn(cfg_f32, seed=777)
    restored_f32_state = self._setup_state(mgr, init_f32_fresh, cfg_f32)
    graphdef_f32 = nnx.graphdef(init_f32_fresh())
    ts_f32_resumed = nnx.merge(graphdef_f32, restored_f32_state)
    self.assertEqual(ts_f32_resumed.model.embed.get_value().dtype, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(ts_f32_resumed.model.embed.get_value(), jnp.uint32)),
        np.asarray(jax.lax.bitcast_convert_type(raw_loaded["params"]["params"]["embed"], jnp.uint32)),
    )

    # 4. Reverse cross-restore: save from f32 -> restore into residual16 (both full state and load_parameters_path):
    mgr_f32 = self._make_manager("f32_ckpt")
    checkpointing.save_checkpoint(mgr_f32, 5, ts_f32_resumed, config=cfg_f32, force=True)
    checkpointing.wait_until_finished(mgr_f32)

    restored_from_f32 = self._setup_state(mgr_f32, self._init_fn(cfg_r16, seed=555), cfg_r16)
    rec_from_f32 = train_state_nnx.to_checkpoint_dict(restored_from_f32)["params"]["params"]
    for p_exp, p_got in zip(
        jax.tree.leaves(train_state_nnx.to_checkpoint_dict(nnx.state(ts_f32_resumed))["params"]["params"]),
        jax.tree.leaves(rec_from_f32),
    ):
      np.testing.assert_array_equal(
          np.asarray(jax.lax.bitcast_convert_type(p_got, jnp.uint32)),
          np.asarray(jax.lax.bitcast_convert_type(p_exp, jnp.uint32)),
      )

    # 5. Params-only load (`load_parameters_path`) into residual16 preserves full float32 precision in residuals:
    f32_step5_path = os.path.join(self._tmpdir, "f32_ckpt", "5")
    cfg_params_only = _ckpt_config(load_parameters_path=f32_step5_path, fp32_master_storage="residual16")
    cfg_params_only.routed_bias = True
    params_only_state = self._setup_state(None, self._init_fn(cfg_params_only, seed=333), cfg_params_only)
    self.assertEqual(int(params_only_state["optimizer"]["step"].get_value()), 0)
    rec_params_only = train_state_nnx.to_checkpoint_dict(params_only_state)["params"]["params"]
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(rec_params_only["embed"], jnp.uint32)),
        np.asarray(jax.lax.bitcast_convert_type(ts_f32_resumed.model.embed.get_value(), jnp.uint32)),
    )

  def test_orbax_roundtrip_and_params_load_with_optimizer_and_parameter_host_offload(self):
    for opt_offload, param_offload in ((True, False), (True, True)):
      with self.subTest(opt_offload=opt_offload, param_offload=param_offload):
        tag = f"offload_{int(opt_offload)}_{int(param_offload)}"
        cfg = _ckpt_config(
            fp32_master_storage="residual16",
            optimizer_memory_host_offload=opt_offload,
            parameter_memory_host_offload=param_offload,
        )
        cfg.routed_bias = True

        init_fn = self._init_fn(cfg, seed=21)
        state, *_ = maxtext_utils.setup_initial_state(None, cfg, self._mesh, None, init_fn, True)
        self.assertEqual(state["optimizer"]["step"].get_value().sharding.memory_kind, "pinned_host")
        self.assertEqual(
            state["optimizer"]["master_residuals"]["embed"].get_value().sharding.memory_kind,
            "pinned_host",
        )
        expected_param_kind = "pinned_host" if param_offload else None
        if param_offload:
          self.assertEqual(state["model"]["embed"].get_value().sharding.memory_kind, expected_param_kind)

        # Advance step to 2 on pinned_host so full-state restore proves non-zero step restoration:
        step_sharding = state["optimizer"]["step"].get_value().sharding
        state["optimizer"]["step"].set_value(
            jax.jit(lambda: jnp.array(2, dtype=jnp.uint32), out_shardings=step_sharding)()
        )

        expected_f32 = train_state_nnx.to_checkpoint_dict(state)["params"]["params"]
        mgr = self._make_manager(tag)
        ts_to_save = nnx.merge(nnx.graphdef(init_fn()), state)
        checkpointing.save_checkpoint(mgr, 2, ts_to_save, config=cfg, force=True)
        checkpointing.wait_until_finished(mgr)

        # Full-state Orbax restore back onto offloaded shardings:
        restored_state, *_ = maxtext_utils.setup_initial_state(
            None, cfg, self._mesh, mgr, self._init_fn(cfg, seed=88), True
        )
        self.assertEqual(int(restored_state["optimizer"]["step"].get_value()), 2)
        self.assertEqual(restored_state["optimizer"]["step"].get_value().sharding.memory_kind, "pinned_host")
        self.assertEqual(
            restored_state["optimizer"]["master_residuals"]["embed"].get_value().sharding.memory_kind,
            "pinned_host",
        )
        if param_offload:
          self.assertEqual(restored_state["model"]["embed"].get_value().sharding.memory_kind, "pinned_host")
        restored_f32 = train_state_nnx.to_checkpoint_dict(restored_state)["params"]["params"]
        for p_exp, p_got in zip(jax.tree.leaves(expected_f32), jax.tree.leaves(restored_f32)):
          np.testing.assert_array_equal(
              np.asarray(jax.lax.bitcast_convert_type(p_got, jnp.uint32)),
              np.asarray(jax.lax.bitcast_convert_type(p_exp, jnp.uint32)),
          )

        # Params-only restore (`load_parameters_path`) onto offloaded shardings keeps fresh step == 0:
        step2_path = os.path.join(self._tmpdir, tag, "2")
        cfg_params = _ckpt_config(
            load_parameters_path=step2_path,
            fp32_master_storage="residual16",
            optimizer_memory_host_offload=opt_offload,
            parameter_memory_host_offload=param_offload,
        )
        cfg_params.routed_bias = True
        params_loaded_state, *_ = maxtext_utils.setup_initial_state(
            None, cfg_params, self._mesh, None, self._init_fn(cfg_params, seed=99), True
        )
        self.assertEqual(int(params_loaded_state["optimizer"]["step"].get_value()), 0)
        params_loaded_f32 = train_state_nnx.to_checkpoint_dict(params_loaded_state)["params"]["params"]
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(params_loaded_f32["embed"], jnp.uint32)),
            np.asarray(jax.lax.bitcast_convert_type(expected_f32["embed"], jnp.uint32)),
        )

  def test_lineage_restore_resplit_preserves_exact_f32_bits_and_shardings(self):
    class _LineageRouterBiasVar(nnx.Variable):
      pass

    class _MockLineageModel(nnx.Module):
      """Lineage-shaped model with nested `params`, `keep_fp32_master` param, and list `mtp_router_bias`."""

      def __init__(self, rngs: nnx.Rngs):
        self.params = nnx.data(
            {
                "embed": {
                    "table": nnx.Param(
                        jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32),
                        out_sharding=("fsdp", None),
                    )
                },
                "mtp": {
                    "0": {
                        "w": nnx.Param(
                            jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32),
                            out_sharding=("fsdp", None),
                        )
                    }
                },
                "fp32_pin": {
                    "scale": nnx.Param(
                        jnp.linspace(0.5, 1.5, 8, dtype=jnp.float32),
                        out_sharding=(None,),
                        keep_fp32_master=True,
                    )
                },
            }
        )
        self.router_bias = _LineageRouterBiasVar(jnp.zeros((2, 4), dtype=jnp.float32), out_sharding=(None, None))
        self.mtp_router_bias = nnx.data([_LineageRouterBiasVar(jnp.zeros((4,), dtype=jnp.float32), out_sharding=(None,))])

      def __call__(self, x: jax.Array) -> jax.Array:
        h = x @ self.params["embed"]["table"].get_value().astype(x.dtype)
        h = h @ self.params["mtp"]["0"]["w"].get_value().astype(x.dtype)
        return h * self.params["fp32_pin"]["scale"].get_value().astype(x.dtype)

    rng_np = np.random.default_rng(20261121)
    loaded_embed_u32 = rng_np.integers(0x3E800001, 0x3FC00000, size=(8, 8), dtype=np.uint32) | np.uint32(0x1235)
    loaded_mtp_u32 = rng_np.integers(0xBE800001, 0xBFC00000, size=(8, 8), dtype=np.uint32) | np.uint32(0x6789)
    loaded_pin_u32 = rng_np.integers(0x3F000001, 0x3FC00000, size=(8,), dtype=np.uint32) | np.uint32(0x2468)
    loaded_embed_f32 = jax.lax.bitcast_convert_type(jnp.asarray(loaded_embed_u32), jnp.float32)
    loaded_mtp_f32 = jax.lax.bitcast_convert_type(jnp.asarray(loaded_mtp_u32), jnp.float32)
    loaded_pin_f32 = jax.lax.bitcast_convert_type(jnp.asarray(loaded_pin_u32), jnp.float32)

    n_dev = len(jax.devices())
    mesh_shape = (2, 2) if n_dev >= 4 else ((2, 1) if n_dev >= 2 else (1, 1))
    devices_2d = np.array(jax.devices()[: mesh_shape[0] * mesh_shape[1]]).reshape(mesh_shape)
    mesh_2d = Mesh(devices_2d, ("data", "fsdp"))

    # Returns False when `master_residuals` is absent:
    ts_plain = train_state_nnx.create_train_state(_MockLineageModel(nnx.Rngs(1)), optax.adamw(1e-3), _ts_config())
    _, state_plain = nnx.split(ts_plain)
    self.assertFalse(train_state_nnx.resplit_model_params_with_residual16(state_plain))

    for mode in ("device", "pinned_host", "zero1", "pure_dict"):
      with self.subTest(mode=mode):
        cfg = _ts_config(
            fp32_master_storage="residual16",
            optimizer_memory_host_offload=mode == "pinned_host",
            shard_optimizer_over_data=mode == "zero1",
        )
        m = _MockLineageModel(nnx.Rngs(13))
        ts = train_state_nnx.create_train_state(m, optax.adamw(1e-3), cfg)
        graphdef, state = nnx.split(ts)

        w_sharding = NamedSharding(mesh_2d, P(None, "fsdp"), memory_kind="device")
        pin_sharding = NamedSharding(mesh_2d, P(None), memory_kind="device")
        res_mem = "pinned_host" if mode == "pinned_host" else "device"
        res_spec = P("data", "fsdp") if mode == "zero1" else P(None, "fsdp")
        res_sharding = NamedSharding(mesh_2d, res_spec, memory_kind=res_mem)

        state["model"]["params"]["embed"]["table"].set_value(
            jax.device_put(state["model"]["params"]["embed"]["table"].get_value(), w_sharding)
        )
        state["model"]["params"]["mtp"]["0"]["w"].set_value(
            jax.device_put(state["model"]["params"]["mtp"]["0"]["w"].get_value(), w_sharding)
        )
        state["model"]["params"]["fp32_pin"]["scale"].set_value(
            jax.device_put(state["model"]["params"]["fp32_pin"]["scale"].get_value(), pin_sharding)
        )
        state["optimizer"]["master_residuals"]["params"]["embed"]["table"].set_value(
            train_state_nnx._device_put_staged(  # pylint: disable=protected-access
                state["optimizer"]["master_residuals"]["params"]["embed"]["table"].get_value(),
                res_sharding,
            )
        )
        state["optimizer"]["master_residuals"]["params"]["mtp"]["0"]["w"].set_value(
            train_state_nnx._device_put_staged(  # pylint: disable=protected-access
                state["optimizer"]["master_residuals"]["params"]["mtp"]["0"]["w"].get_value(),
                res_sharding,
            )
        )

        # Simulate `lineage_adapter.restore_params`: overwrites `state["model"]` with float32 weights only.
        restored_lineage_dict = {
            "params": {
                "embed": {"table": jax.device_put(loaded_embed_f32, w_sharding)},
                "mtp": {"0": {"w": jax.device_put(loaded_mtp_f32, w_sharding)}},
                "fp32_pin": {"scale": jax.device_put(loaded_pin_f32, pin_sharding)},
            },
            "router_bias": jnp.ones((2, 4), dtype=jnp.float32),
            "mtp_router_bias": {0: jnp.full((4,), 2.0, dtype=jnp.float32)},
        }
        nnx.replace_by_pure_dict(state["model"], restored_lineage_dict)

        # Before re-splitting, stale random-init low bits corrupt reconstruction:
        stale_rec_embed = train_state_nnx.reconstruct_f32_from_bf16_residual16(
            state["model"]["params"]["embed"]["table"].get_value().astype(jnp.bfloat16),
            jax.device_put(
                state["optimizer"]["master_residuals"]["params"]["embed"]["table"].get_value(),
                w_sharding,
            ),
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(jax.lax.bitcast_convert_type(stale_rec_embed, jnp.uint32)),
                loaded_embed_u32,
            )
        )

        target_state = state.to_pure_dict() if mode == "pure_dict" else state
        self.assertTrue(train_state_nnx.resplit_model_params_with_residual16(target_state))
        if mode == "pure_dict":
          nnx.replace_by_pure_dict(state, target_state)

        embed_w = state["model"]["params"]["embed"]["table"].get_value()
        mtp_w = state["model"]["params"]["mtp"]["0"]["w"].get_value()
        pin_w = state["model"]["params"]["fp32_pin"]["scale"].get_value()
        embed_res = state["optimizer"]["master_residuals"]["params"]["embed"]["table"].get_value()
        mtp_res = state["optimizer"]["master_residuals"]["params"]["mtp"]["0"]["w"].get_value()

        self.assertEqual(embed_w.dtype, jnp.bfloat16)
        self.assertEqual(mtp_w.dtype, jnp.bfloat16)
        self.assertEqual(pin_w.dtype, jnp.float32)
        self.assertEqual(state["model"]["mtp_router_bias"][0].get_value().dtype, jnp.float32)
        self.assertEqual(embed_w.sharding, w_sharding)
        self.assertEqual(embed_res.dtype, jnp.int16)
        self.assertEqual(embed_res.sharding, res_sharding)
        self.assertEqual(mtp_res.sharding, res_sharding)

        ckpt_params = train_state_nnx.to_checkpoint_dict(state)["params"]["params"]["params"]
        for got_leaf, exp_u32, dst in (
            (ckpt_params["embed"]["table"], loaded_embed_u32, w_sharding),
            (ckpt_params["mtp"]["0"]["w"], loaded_mtp_u32, w_sharding),
            (ckpt_params["fp32_pin"]["scale"], loaded_pin_u32, pin_sharding),
        ):
          got_u32 = np.asarray(jax.lax.bitcast_convert_type(jax.device_put(got_leaf, dst), jnp.uint32))
          np.testing.assert_array_equal(got_u32, exp_u32)

        # Verify first optimizer update after re-split matches an f32 model initialized at the exact loaded weights:
        if mode == "device":
          ts_resplit = nnx.merge(graphdef, state)
          m_ref = _MockLineageModel(nnx.Rngs(13))
          m_ref.params["embed"]["table"].set_value(jax.device_put(loaded_embed_f32, w_sharding))
          m_ref.params["mtp"]["0"]["w"].set_value(jax.device_put(loaded_mtp_f32, w_sharding))
          m_ref.params["fp32_pin"]["scale"].set_value(jax.device_put(loaded_pin_f32, pin_sharding))
          ts_ref = train_state_nnx.create_train_state(m_ref, optax.adamw(1e-3), _ts_config(fp32_master_storage="f32"))
          x_lineage_in = jnp.ones((2, 8), dtype=jnp.bfloat16)
          grads_r16 = jax.tree.map(
              lambda g: g.astype(jnp.float32) if hasattr(g, "dtype") and g.dtype == jnp.bfloat16 else g,
              nnx.grad(lambda mdl, x=x_lineage_in: jnp.mean(mdl(x).astype(jnp.float32) ** 2))(ts_resplit.model),
          )
          grads_ref = nnx.grad(lambda mdl, x=x_lineage_in: jnp.mean(mdl(x).astype(jnp.float32) ** 2))(ts_ref.model)
          ts_resplit.apply_gradients(grads_r16)
          ts_ref.apply_gradients(grads_ref)
          after_r16 = train_state_nnx.to_checkpoint_dict(ts_resplit)["params"]["params"]["params"]
          np.testing.assert_allclose(
              np.asarray(jax.device_put(after_r16["embed"]["table"], w_sharding)),
              np.asarray(ts_ref.model.params["embed"]["table"].get_value()),
              rtol=1e-6,
              atol=1e-6,
          )

    # End-to-end `maxtext_utils.setup_initial_state` test on the `if raw_params and native:` Lineage restore branch:
    cfg_native = _ckpt_config(
        load_parameters_path="/tmp/mock_lineage_ckpt",
        fp32_master_storage="residual16",
        optimizer_memory_host_offload=True,
    )
    expected_embed_u32 = (
        loaded_embed_u32[:16, :8]
        if loaded_embed_u32.shape == (16, 8)
        else (rng_np.integers(0x3E800001, 0x3FC00000, size=(16, 8), dtype=np.uint32) | np.uint32(0x1357))
    )
    expected_embed_f32 = jax.lax.bitcast_convert_type(jnp.asarray(expected_embed_u32), jnp.float32)

    def _mock_lineage_restore_params(model_state, raw_params, config, mesh):
      del raw_params, config
      target_sharding = NamedSharding(mesh, P("fsdp", None), memory_kind="device")
      nnx.replace_by_pure_dict(model_state, {"embed": jax.device_put(expected_embed_f32, target_sharding)})

    with (
        mock.patch.object(maxtext_utils.lineage_adapter, "is_native", return_value=True),
        mock.patch.object(maxtext_utils.lineage_adapter, "maxtext_restore_target", return_value={"params": {}}),
        mock.patch.object(
            maxtext_utils.checkpointing, "load_params_from_path", return_value={"params": {"embed": expected_embed_f32}}
        ),
        mock.patch.object(
            maxtext_utils.lineage_adapter, "restore_params", side_effect=_mock_lineage_restore_params
        ) as spy_restore,
    ):
      native_state, *_ = maxtext_utils.setup_initial_state(
          None, cfg_native, self._mesh, None, self._init_fn(cfg_native, seed=77), True
      )
    self.assertEqual(spy_restore.call_count, 1)
    self.assertEqual(native_state["model"]["embed"].get_value().dtype, jnp.bfloat16)
    self.assertEqual(native_state["model"]["embed"].get_value().sharding.memory_kind, "device")
    self.assertEqual(native_state["optimizer"]["master_residuals"]["embed"].get_value().dtype, jnp.int16)
    self.assertEqual(
        native_state["optimizer"]["master_residuals"]["embed"].get_value().sharding.memory_kind,
        "pinned_host",
    )
    native_rec = train_state_nnx.to_checkpoint_dict(native_state)["params"]["params"]["embed"]
    np.testing.assert_array_equal(
        np.asarray(
            jax.lax.bitcast_convert_type(
                jax.device_put(native_rec, NamedSharding(self._mesh, P("fsdp", None), memory_kind="device")),
                jnp.uint32,
            )
        ),
        expected_embed_u32,
    )


class TestFp32ParamExemptionsAndSaveMemoryKind(unittest.TestCase):
  """Regression tests for shared_expert_gate / A_log / dt_bias exemptions, chunked save, and ZeRO-1."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def test_shared_expert_gate_a_log_dt_bias_and_empty_residuals_preserve_f32(self):
    class _HybridGdnMoeBlock(nnx.Module):
      """Minimal Qwen3-Next-style hybrid GatedDeltaNet + shared-expert MoE block."""

      def __init__(self, rngs: nnx.Rngs):
        self.linear_attn = nnx.Dict(
            {
                "A_log": nnx.Param(
                    jnp.linspace(-2.0, 2.5, 8, dtype=jnp.float32) + jnp.float32(1e-6),
                    keep_fp32_master=True,
                ),
                "dt_bias": nnx.Param(
                    jnp.linspace(0.5, 1.5, 8, dtype=jnp.float32) + jnp.float32(1e-6),
                    keep_fp32_master=True,
                ),
                "out_proj": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32))}),
            }
        )
        self.mlp = nnx.Dict(
            {
                "routed_experts": nnx.Dict(
                    {
                        "gate": nnx.Dict(
                            {
                                "kernel": nnx.Param(
                                    jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32),
                                    keep_fp32_master=True,
                                )
                            }
                        )
                    }
                ),
                "shared_expert_gate": nnx.Dict(
                    {
                        "kernel": nnx.Param(
                            jax.random.normal(rngs.params(), (8, 1), dtype=jnp.float32),
                            keep_fp32_master=True,
                        )
                    }
                ),
                "shared_expert": nnx.Dict(
                    {"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32))}
                ),
            }
        )

      def forward_gates(self, x: jax.Array, a: jax.Array):
        a_log = jnp.asarray(self.linear_attn["A_log"].get_value(), dtype=jnp.float32)
        dt_bias = jnp.asarray(self.linear_attn["dt_bias"].get_value(), dtype=jnp.float32)
        g = -jnp.exp(a_log) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias)
        sh_gate = jax.nn.sigmoid(
            x.astype(jnp.float32) @ jnp.asarray(self.mlp["shared_expert_gate"]["kernel"].get_value(), dtype=jnp.float32)
        )
        return g, sh_gate

    cfg_f32 = SimpleNamespace(
        fp32_master_storage="f32",
        weight_dtype=jnp.float32,
        float32_gate_logits=True,
        routed_bias=False,
    )
    cfg_r16 = SimpleNamespace(
        fp32_master_storage="residual16",
        weight_dtype=jnp.float32,
        float32_gate_logits=True,
        routed_bias=False,
    )

    m_f32 = _HybridGdnMoeBlock(nnx.Rngs(7))
    m_r16 = _HybridGdnMoeBlock(nnx.Rngs(7))
    ts_f32 = train_state_nnx.create_train_state(m_f32, optax.adamw(1e-3), cfg_f32)
    ts_r16 = train_state_nnx.create_train_state(m_r16, optax.adamw(1e-3), cfg_r16)

    # Exempt parameters remain float32 in state.model and have no int16 residual entries:
    self.assertEqual(ts_r16.model.linear_attn["A_log"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.linear_attn["dt_bias"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.mlp["routed_experts"]["gate"]["kernel"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.mlp["shared_expert_gate"]["kernel"].get_value().dtype, jnp.float32)
    self.assertNotIn("A_log", ts_r16.optimizer.master_residuals["linear_attn"])

    # Standard projection weights use bfloat16 working storage + int16 sibling master_residuals:
    self.assertEqual(ts_r16.model.linear_attn["out_proj"]["kernel"].get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_r16.model.mlp["shared_expert"]["kernel"].get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_r16.optimizer.master_residuals["linear_attn"]["out_proj"]["kernel"].get_value().dtype, jnp.int16)
    self.assertEqual(ts_r16.optimizer.master_residuals["mlp"]["shared_expert"]["kernel"].get_value().dtype, jnp.int16)

    # Step-0 forward recurrence decay `g` and `shared_expert_gate` output match f32 bit-for-bit:
    x = jax.random.normal(jax.random.PRNGKey(19), (2, 8), dtype=jnp.bfloat16)
    a = jax.random.normal(jax.random.PRNGKey(23), (2, 8), dtype=jnp.bfloat16)
    g_f32, sh_f32 = ts_f32.model.forward_gates(x, a)
    g_r16, sh_r16 = ts_r16.model.forward_gates(x, a)
    np.testing.assert_array_equal(np.asarray(g_r16), np.asarray(g_f32))
    np.testing.assert_array_equal(np.asarray(sh_r16), np.asarray(sh_f32))

    # Under weight_dtype='bfloat16', float32-initialized parameters remain exempt (`master_residuals == {}`):
    class _AllExemptUnderBf16WeightDtype(nnx.Module):

      def __init__(self):
        self.A_log = nnx.Param(jnp.ones((4,), dtype=jnp.float32))
        self.gate = nnx.Param(jnp.ones((4, 2), dtype=jnp.float32))

    cfg_bf16_wt = _ts_config(fp32_master_storage="residual16", weight_dtype="bfloat16", float32_gate_logits=True)
    m_exempt = _AllExemptUnderBf16WeightDtype()
    ts_exempt = train_state_nnx.create_train_state(m_exempt, optax.adamw(1e-3), cfg_bf16_wt)
    self.assertEqual(ts_exempt.optimizer.master_residuals, {})
    grads_exempt = nnx.grad(lambda m: jnp.mean(m.A_log.get_value() ** 2 + jnp.sum(m.gate.get_value() ** 2)))(
        ts_exempt.model
    )
    ts_exempt.apply_gradients(grads_exempt)
    ts_exempt.optimizer.apply_master_param_transform(ts_exempt.model, lambda _, p: p * 0.5)
    exempt_ckpt = train_state_nnx.to_checkpoint_dict(ts_exempt)
    self.assertEqual(exempt_ckpt["params"]["params"]["A_log"].dtype, jnp.float32)

  def test_keep_fp32_master_metadata_custom_and_scanned_layers(self):
    class _CustomFp32SubLayer(nnx.Module):
      """Module using structural `keep_fp32_master=True` on arbitrary parameter names."""

      def __init__(self, rngs: nnx.Rngs):
        self.alpha_fp32 = nnx.Param(
            jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32) + jnp.float32(1e-6),
            out_sharding=("fsdp", None),
            keep_fp32_master=True,
        )
        self.regular_proj = nnx.Param(
            jax.random.normal(rngs.params(), (4, 8), dtype=jnp.float32),
            out_sharding=(None, "fsdp"),
        )

      def __call__(self, x: jax.Array) -> jax.Array:
        scores = x.astype(jnp.float32) @ self.alpha_fp32.get_value()
        return (scores.astype(jnp.bfloat16) @ self.regular_proj.get_value().astype(jnp.bfloat16)).astype(jnp.bfloat16)

    cfg_r16 = _ts_config(fp32_master_storage="residual16", weight_dtype=jnp.float32)
    cfg_f32 = _ts_config(fp32_master_storage="f32", weight_dtype=jnp.float32)

    m_r16 = _CustomFp32SubLayer(nnx.Rngs(101))
    m_f32 = _CustomFp32SubLayer(nnx.Rngs(101))
    m_f32.regular_proj.set_value(m_f32.regular_proj.get_value().astype(jnp.bfloat16).astype(jnp.float32))
    m_r16.regular_proj.set_value(m_r16.regular_proj.get_value().astype(jnp.bfloat16).astype(jnp.float32))

    self.assertTrue(m_r16.alpha_fp32.get_metadata().get("keep_fp32_master"))
    self.assertEqual(m_r16.alpha_fp32.get_metadata().get("out_sharding"), ("fsdp", None))

    ts_r16 = train_state_nnx.create_train_state(m_r16, optax.adamw(1e-3), cfg_r16)
    ts_f32 = train_state_nnx.create_train_state(m_f32, optax.adamw(1e-3), cfg_f32)
    self.assertEqual(ts_r16.model.alpha_fp32.get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.regular_proj.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("alpha_fp32", ts_r16.optimizer.master_residuals)
    self.assertIn("regular_proj", ts_r16.optimizer.master_residuals)

    x_in = jax.random.normal(jax.random.PRNGKey(5), (2, 8), dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(ts_r16.model(x_in)), np.asarray(ts_f32.model(x_in)))
    g_r16 = jax.tree.map(
        lambda g: g.astype(jnp.float32) if g.dtype == jnp.bfloat16 else g,
        nnx.grad(lambda m: jnp.mean(m(x_in).astype(jnp.float32) ** 2))(ts_r16.model),
    )
    g_f32 = nnx.grad(lambda m: jnp.mean(m(x_in).astype(jnp.float32) ** 2))(ts_f32.model)
    ts_r16.apply_gradients(g_r16)
    ts_f32.apply_gradients(g_f32)
    np.testing.assert_array_equal(
        np.asarray(ts_r16.model.alpha_fp32.get_value()),
        np.asarray(ts_f32.model.alpha_fp32.get_value()),
    )

    # Also verify `nnx_scan.create_scanned_layers` preserves `keep_fp32_master=True` across stacked layers:
    scanned = nnx_scan.create_scanned_layers(
        _CustomFp32SubLayer,
        length=2,
        param_scan_axis=0,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(202),
    )
    self.assertTrue(scanned.alpha_fp32.get_metadata().get("keep_fp32_master"))
    ts_scanned = train_state_nnx.create_train_state(scanned, optax.adamw(1e-3), cfg_r16)
    self.assertEqual(ts_scanned.model.alpha_fp32.get_value().dtype, jnp.float32)
    self.assertEqual(ts_scanned.model.regular_proj.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("alpha_fp32", ts_scanned.optimizer.master_residuals)
    self.assertIn("regular_proj", ts_scanned.optimizer.master_residuals)

    # Structural proof: untagged float32 params named `A_log`, `dt_bias`, or `gate` ARE split into `(bfloat16, int16)`!
    class _UntaggedNameCollisionModel(nnx.Module):
      """Model proving `should_use_residual16_for_param` is purely metadata-driven rather than name-based."""

      def __init__(self):
        self.A_log = nnx.Param(jnp.linspace(-2.0, 2.0, 8, dtype=jnp.float32))  # pylint: disable=invalid-name
        self.dt_bias = nnx.Param(jnp.ones((8,), dtype=jnp.float32))
        self.routed = nnx.Dict({"gate": nnx.Param(jnp.ones((4, 8), dtype=jnp.float32))})

    cfg_untagged = _ts_config(
        fp32_master_storage="residual16",
        weight_dtype=jnp.float32,
        float32_gate_logits=True,
        routed_bias=True,
    )
    ts_untagged = train_state_nnx.create_train_state(_UntaggedNameCollisionModel(), optax.adamw(1e-3), cfg_untagged)
    self.assertEqual(ts_untagged.model.A_log.get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_untagged.model.dt_bias.get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_untagged.model.routed["gate"].get_value().dtype, jnp.bfloat16)
    self.assertIn("A_log", ts_untagged.optimizer.master_residuals)
    self.assertIn("dt_bias", ts_untagged.optimizer.master_residuals)
    self.assertIn("gate", ts_untagged.optimizer.master_residuals["routed"])

    # Also verify `lineage_adapter._param_tree` keeps `moe.router.{kernel,bias}` in float32 under `float32_gate_logits`
    # while splitting `moe.routed.gate` SwiGLU expert weights into `(bfloat16, int16)`:
    fake_lineage_params = {
        "decoder": {
            "sparse": {
                "moe": {
                    "router": {"kernel": jnp.ones((8, 4), dtype=jnp.float32)},
                    "routed": {"gate": jnp.ones((4, 8, 16), dtype=jnp.float32)},
                }
            }
        }
    }
    fake_shardings = jax.tree.map(lambda _: P(), fake_lineage_params)
    lineage_dict, _ = lineage_adapter._param_tree(  # pylint: disable=protected-access
        fake_lineage_params, fake_shardings, float32_gate_logits=True
    )

    class _LineageTreeHolder(nnx.Module):

      def __init__(self, tree_dict):
        self.params = nnx.data(tree_dict)

    ts_lin_tree = train_state_nnx.create_train_state(_LineageTreeHolder(lineage_dict), optax.adamw(1e-3), cfg_untagged)
    moe_sub = ts_lin_tree.model.params["decoder"]["sparse"]["moe"]
    self.assertTrue(moe_sub["router"]["kernel"].get_metadata().get("keep_fp32_master"))
    self.assertFalse(moe_sub["routed"]["gate"].get_metadata().get("keep_fp32_master"))
    self.assertEqual(moe_sub["router"]["kernel"].get_value().dtype, jnp.float32)
    self.assertEqual(moe_sub["routed"]["gate"].get_value().dtype, jnp.bfloat16)
    self.assertIn("routed", ts_lin_tree.optimizer.master_residuals["params"]["decoder"]["sparse"]["moe"])
    self.assertNotIn("router", ts_lin_tree.optimizer.master_residuals["params"]["decoder"]["sparse"]["moe"])

  def test_real_intentionally_fp32_modules_preserve_float32_under_residual16(self):
    mesh = jax.make_mesh((min(len(jax.devices()), 4), 1, 1, 1), ("data", "fsdp", "tensor", "sequence"))
    base_args = ["", get_test_config_path(), "enable_checkpointing=False", "dtype=bfloat16"]

    # 1. DeepSeek-V3.2 `Indexer.weights_proj` and HY4 `Hy4Indexer`, `Hy4HyperConnection`, `Hy4HyperHead`, `Hy4Attention`:
    cfg_mla = pyconfig.initialize(
        base_args
        + [
            "run_name=test_mla_hy4_fp32",
            "base_emb_dim=64",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "head_dim=32",
            "attention_type=mla",
            "attention=dot_product",
            "attention_sink=True",
            "q_lora_rank=32",
            "kv_lora_rank=32",
            "qk_nope_head_dim=16",
            "qk_rope_head_dim=16",
            "v_head_dim=32",
            "use_indexer=True",
            "indexer_n_heads=4",
            "indexer_head_dim=32",
            "indexer_topk=8",
            "mhc_expansion_rate=2",
            "weight_dtype=float32",
            "fp32_master_storage=residual16",
        ]
    )
    rope = embeddings.RotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=mesh,
        embedding_dims=16,
        cast_as_fprop_dtype=True,
        fprop_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )
    k_init = initializers.nd_dense_init(1.0, "fan_in", "normal")

    idx_mla = attention_mla.Indexer(cfg_mla, rngs=nnx.Rngs(1), rotary_embedding=rope, kernel_init=k_init)
    self.assertTrue(idx_mla.weights_proj.kernel.get_metadata().get("keep_fp32_master"))
    self.assertTrue(idx_mla.k_norm.scale.get_metadata().get("keep_fp32_master"))
    self.assertTrue(idx_mla.k_norm.bias.get_metadata().get("keep_fp32_master"))
    self.assertEqual(idx_mla.weights_proj.kernel.get_metadata().get("out_sharding"), ("embed_attn", "q_heads"))
    ts_idx_mla = train_state_nnx.create_train_state(idx_mla, optax.adamw(1e-3), cfg_mla)
    self.assertEqual(ts_idx_mla.model.weights_proj.kernel.get_value().dtype, jnp.float32)
    self.assertEqual(ts_idx_mla.model.k_norm.scale.get_value().dtype, jnp.float32)
    self.assertEqual(ts_idx_mla.model.k_norm.bias.get_value().dtype, jnp.float32)
    self.assertEqual(ts_idx_mla.model.wq_b.kernel.get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_idx_mla.model.wk.kernel.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("weights_proj", ts_idx_mla.optimizer.master_residuals)
    self.assertNotIn("k_norm", ts_idx_mla.optimizer.master_residuals)

    ds4_hh_r16 = mhc.DeepSeek4HyperHead(cfg_mla, mesh, nnx.Rngs(14))
    ds4_hh_f32 = mhc.DeepSeek4HyperHead(cfg_mla, mesh, nnx.Rngs(14))
    ts_ds4_hh = train_state_nnx.create_train_state(ds4_hh_r16, optax.adamw(1e-3), cfg_mla)
    for var in (ts_ds4_hh.model.hc_fn, ts_ds4_hh.model.hc_base, ts_ds4_hh.model.hc_scale):
      self.assertTrue(var.get_metadata().get("keep_fp32_master"))
      self.assertEqual(var.get_value().dtype, jnp.float32)
    self.assertEqual(ts_ds4_hh.optimizer.master_residuals, {})
    x_hc = jax.random.normal(jax.random.PRNGKey(44), (2, 8, 2, 64), dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(ts_ds4_hh.model(x_hc)), np.asarray(ds4_hh_f32(x_hc)))

    idx_hy4 = hy4.Hy4Indexer(cfg_mla, rngs=nnx.Rngs(2), rotary_embedding=rope, kernel_init=k_init)
    self.assertTrue(idx_hy4.weights_proj.kernel.get_metadata().get("keep_fp32_master"))
    self.assertTrue(idx_hy4.k_norm.scale.get_metadata().get("keep_fp32_master"))
    self.assertTrue(idx_hy4.k_norm.bias.get_metadata().get("keep_fp32_master"))
    ts_idx_hy4 = train_state_nnx.create_train_state(idx_hy4, optax.adamw(1e-3), cfg_mla)
    self.assertEqual(ts_idx_hy4.model.weights_proj.kernel.get_value().dtype, jnp.float32)
    self.assertEqual(ts_idx_hy4.model.k_norm.scale.get_value().dtype, jnp.float32)
    self.assertEqual(ts_idx_hy4.model.k_norm.bias.get_value().dtype, jnp.float32)
    self.assertEqual(ts_idx_hy4.model.wk.kernel.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("weights_proj", ts_idx_hy4.optimizer.master_residuals)
    self.assertNotIn("k_norm", ts_idx_hy4.optimizer.master_residuals)

    hc = hy4.Hy4HyperConnection(cfg_mla, mesh, nnx.Rngs(3))
    hh = hy4.Hy4HyperHead(cfg_mla, mesh, nnx.Rngs(4))
    ts_hc = train_state_nnx.create_train_state(hc, optax.adamw(1e-3), cfg_mla)
    ts_hh = train_state_nnx.create_train_state(hh, optax.adamw(1e-3), cfg_mla)
    for var in (
        ts_hc.model.fn,
        ts_hc.model.base,
        ts_hc.model.scale,
        ts_hh.model.hc_fn,
        ts_hh.model.hc_base,
        ts_hh.model.hc_scale,
    ):
      self.assertTrue(var.get_metadata().get("keep_fp32_master"))
      self.assertEqual(var.get_value().dtype, jnp.float32)

    hy4_attn = hy4.Hy4Attention(
        config=cfg_mla,
        num_query_heads=4,
        num_kv_heads=4,
        head_dim=32,
        max_target_length=16,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(2, 16, 64),
        inputs_kv_shape=(2, 16, 64),
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        is_full_indexer=True,
        rngs=nnx.Rngs(5),
    )
    self.assertTrue(hy4_attn.sinks.get_metadata().get("keep_fp32_master"))
    ts_hy4_attn = train_state_nnx.create_train_state(hy4_attn, optax.adamw(1e-3), cfg_mla)
    self.assertEqual(ts_hy4_attn.model.sinks.get_value().dtype, jnp.float32)
    self.assertEqual(ts_hy4_attn.model.indexer.weights_proj.kernel.get_value().dtype, jnp.float32)
    self.assertEqual(ts_hy4_attn.model.indexer.k_norm.scale.get_value().dtype, jnp.float32)
    self.assertEqual(ts_hy4_attn.model.gate_proj.kernel.get_value().dtype, jnp.bfloat16)

    # 2. KDA (`KimiDeltaAttention` `o_norm.scale`, `A_log`, `dt_bias`, and `q/k/v_conv1d`):
    kda_r16 = attention_kda.KimiDeltaAttention(
        hidden_size=32,
        num_heads=4,
        head_dim=8,
        conv_size=4,
        chunk_size=16,
        compute_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(6),
    )
    kda_f32 = attention_kda.KimiDeltaAttention(
        hidden_size=32,
        num_heads=4,
        head_dim=8,
        conv_size=4,
        chunk_size=16,
        compute_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(6),
    )
    # Give `o_norm.scale` non-bf16-aligned float32 low bits while aligning bf16 projections to test `o_norm` parity:
    non_bf16_scale = jax.lax.bitcast_convert_type(
        jnp.array([0x3F804321, 0x3F806543, 0x3F802468, 0x3F801357] * 2, dtype=jnp.uint32),
        jnp.float32,
    )
    for kda_m in (kda_r16, kda_f32):
      kda_m.o_norm.scale.set_value(non_bf16_scale)
      for proj_name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj", "g_proj", "out"):
        proj = getattr(kda_m, proj_name)
        proj.kernel.set_value(proj.kernel.get_value().astype(jnp.bfloat16).astype(jnp.float32))

    for fp32_var in (
        kda_r16.o_norm.scale,
        kda_r16.A_log,
        kda_r16.dt_bias,
        kda_r16.q_conv1d.kernel,
        kda_r16.k_conv1d.kernel,
        kda_r16.v_conv1d.kernel,
    ):
      self.assertTrue(fp32_var.get_metadata().get("keep_fp32_master"))
    self.assertEqual(kda_r16.A_log.get_metadata().get("out_sharding"), ("heads",))
    self.assertEqual(kda_r16.q_conv1d.kernel.get_metadata().get("out_sharding"), (None, None, "heads"))

    ts_kda_r16 = train_state_nnx.create_train_state(kda_r16, optax.adamw(1e-3), cfg_mla)
    ts_kda_f32 = train_state_nnx.create_train_state(
        kda_f32, optax.adamw(1e-3), _ts_config(fp32_master_storage="f32", weight_dtype=jnp.float32)
    )
    self.assertEqual(ts_kda_r16.model.o_norm.scale.get_value().dtype, jnp.float32)
    self.assertEqual(ts_kda_r16.model.A_log.get_value().dtype, jnp.float32)
    self.assertEqual(ts_kda_r16.model.dt_bias.get_value().dtype, jnp.float32)
    self.assertEqual(ts_kda_r16.model.q_conv1d.kernel.get_value().dtype, jnp.float32)
    self.assertEqual(ts_kda_r16.model.out.kernel.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("o_norm", ts_kda_r16.optimizer.master_residuals)
    self.assertNotIn("A_log", ts_kda_r16.optimizer.master_residuals)
    self.assertNotIn("q_conv1d", ts_kda_r16.optimizer.master_residuals)
    x_kda = jax.random.normal(jax.random.PRNGKey(33), (2, 16, 32), dtype=jnp.bfloat16)
    out_kda_r16, _, _ = ts_kda_r16.model(x_kda)
    out_kda_f32, _, _ = ts_kda_f32.model(x_kda)
    np.testing.assert_array_equal(np.asarray(out_kda_r16), np.asarray(out_kda_f32))

    # 3. `gpt-oss` / `Attention.sinks` (`weight_dtype=float32` keeps float32; `weight_dtype=bfloat16` splits `sinks`):
    cfg_sink = pyconfig.initialize(
        base_args
        + [
            "run_name=test_gpt_oss_sinks",
            "base_emb_dim=32",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=8",
            "attention_sink=True",
            "attention=dot_product",
            "weight_dtype=float32",
            "fp32_master_storage=residual16",
        ]
    )
    attn_f32_wt = attentions.Attention(
        config=cfg_sink,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_target_length=16,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(2, 16, 32),
        inputs_kv_shape=(2, 16, 32),
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(7),
    )
    self.assertTrue(attn_f32_wt.sinks.get_metadata().get("keep_fp32_master"))
    self.assertEqual(attn_f32_wt.sinks.get_metadata().get("out_sharding"), (None,))
    ts_attn_f32_wt = train_state_nnx.create_train_state(attn_f32_wt, optax.adamw(1e-3), cfg_sink)
    self.assertEqual(ts_attn_f32_wt.model.sinks.get_value().dtype, jnp.float32)
    self.assertEqual(ts_attn_f32_wt.model.query.kernel.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("sinks", ts_attn_f32_wt.optimizer.master_residuals)

    attn_bf16_wt = attentions.Attention(
        config=cfg_sink,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_target_length=16,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(2, 16, 32),
        inputs_kv_shape=(2, 16, 32),
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(7),
    )
    self.assertFalse(attn_bf16_wt.sinks.get_metadata().get("keep_fp32_master"))
    ts_attn_bf16_wt = train_state_nnx.create_train_state(
        attn_bf16_wt, optax.adamw(1e-3), _ts_config(fp32_master_storage="residual16", weight_dtype=jnp.bfloat16)
    )
    self.assertIn("sinks", ts_attn_bf16_wt.optimizer.master_residuals)

    # 4. `Qwen3NextGatedDeltaNet` (`A_log` & `dt_bias`), `Qwen3NextSparseMoeBlock` (`shared_expert_gate`),
    #    `MingLinearAttention` (`A_log` & `dt_bias`), and `GateLogit`:
    cfg_qwen = pyconfig.initialize(
        base_args
        + [
            "run_name=test_qwen_gdn_fp32",
            "decoder_block=qwen3_next",
            "base_emb_dim=32",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=8",
            "gdn_num_key_heads=2",
            "gdn_num_value_heads=4",
            "gdn_key_head_dim=8",
            "gdn_value_head_dim=8",
            "gdn_conv_kernel_dim=4",
            "inhomogeneous_layer_cycle_interval=4",
            "num_experts=4",
            "num_experts_per_tok=2",
            "base_moe_mlp_dim=32",
            "shared_experts=1",
            "float32_gate_logits=True",
            "weight_dtype=float32",
            "fp32_master_storage=residual16",
        ]
    )
    gdn = qwen3.Qwen3NextGatedDeltaNet(config=cfg_qwen, mesh=mesh, quant=None, rngs=nnx.Rngs(8))
    self.assertTrue(gdn.A_log.get_metadata().get("keep_fp32_master"))
    self.assertTrue(gdn.dt_bias.get_metadata().get("keep_fp32_master"))
    ts_gdn = train_state_nnx.create_train_state(gdn, optax.adamw(1e-3), cfg_qwen)
    self.assertEqual(ts_gdn.model.A_log.get_value().dtype, jnp.float32)
    self.assertEqual(ts_gdn.model.dt_bias.get_value().dtype, jnp.float32)
    self.assertEqual(ts_gdn.model.in_proj_qkvz.kernel.get_value().dtype, jnp.bfloat16)
    self.assertNotIn("A_log", ts_gdn.optimizer.master_residuals)
    self.assertNotIn("dt_bias", ts_gdn.optimizer.master_residuals)

    qwen_moe = qwen3.Qwen3NextSparseMoeBlock(config=cfg_qwen, mesh=mesh, quant=None, rngs=nnx.Rngs(9))
    self.assertTrue(qwen_moe.shared_expert_gate.kernel.get_metadata().get("keep_fp32_master"))
    self.assertTrue(qwen_moe.routed_experts.gate.kernel.get_metadata().get("keep_fp32_master"))
    ts_qwen_moe = train_state_nnx.create_train_state(qwen_moe, optax.adamw(1e-3), cfg_qwen)
    self.assertEqual(ts_qwen_moe.model.shared_expert_gate.kernel.get_value().dtype, jnp.float32)
    self.assertEqual(ts_qwen_moe.model.routed_experts.gate.kernel.get_value().dtype, jnp.float32)

    cfg_ming = pyconfig.initialize(
        base_args
        + [
            "run_name=test_ming_fp32",
            "decoder_block=ming",
            "base_emb_dim=32",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=8",
            "gdn_conv_kernel_dim=4",
            "inhomogeneous_layer_cycle_interval=2",
            "weight_dtype=float32",
            "fp32_master_storage=residual16",
        ]
    )
    ming_attn = ming.MingKimiDeltaAttention(config=cfg_ming, mesh=mesh, model_mode="train", quant=None, rngs=nnx.Rngs(10))
    self.assertTrue(ming_attn.A_log.get_metadata().get("keep_fp32_master"))
    self.assertTrue(ming_attn.dt_bias.get_metadata().get("keep_fp32_master"))
    ts_ming = train_state_nnx.create_train_state(ming_attn, optax.adamw(1e-3), cfg_ming)
    self.assertEqual(ts_ming.model.A_log.get_value().dtype, jnp.float32)
    self.assertEqual(ts_ming.model.dt_bias.get_value().dtype, jnp.float32)
    self.assertEqual(ts_ming.model.o_proj.kernel.get_value().dtype, jnp.bfloat16)

    gate_logit = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=4,
        model_name="gpt-oss-20b",
        mesh=mesh,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        use_bias=True,
        rngs=nnx.Rngs(11),
    )
    self.assertTrue(gate_logit.kernel.get_metadata().get("keep_fp32_master"))
    self.assertTrue(gate_logit.bias.get_metadata().get("keep_fp32_master"))

    # 5. DeepSeek-V4 `CompressedAttention.sinks` and `Gemma4MoE.pre_forward_scale_2`:
    cfg_ds4 = pyconfig.initialize(
        base_args
        + [
            "model_name=deepseek4-tiny",
            "run_name=test_ds4_sinks",
            "weight_dtype=float32",
            "fp32_master_storage=residual16",
        ]
    )
    attn_comp = attention_compressed.CompressedAttention(
        config=cfg_ds4,
        num_query_heads=cfg_ds4.num_query_heads,
        num_kv_heads=cfg_ds4.num_kv_heads,
        head_dim=cfg_ds4.head_dim,
        max_target_length=16,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(2, 16, cfg_ds4.emb_dim),
        inputs_kv_shape=(2, 16, cfg_ds4.emb_dim),
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        compress_ratio=0,
        q_lora_rank=cfg_ds4.q_lora_rank,
        rngs=nnx.Rngs(12),
    )
    self.assertTrue(attn_comp.sinks.get_metadata().get("keep_fp32_master"))
    ts_attn_comp = train_state_nnx.create_train_state(attn_comp, optax.adamw(1e-3), cfg_sink)
    self.assertEqual(ts_attn_comp.model.sinks.get_value().dtype, jnp.float32)
    self.assertNotIn("sinks", ts_attn_comp.optimizer.master_residuals)

    gemma4_moe = gemma4.Gemma4MoE(config=cfg_qwen, mesh=mesh, rngs=nnx.Rngs(15))
    self.assertTrue(gemma4_moe.pre_forward_scale_2.get_metadata().get("keep_fp32_master"))
    ts_gemma4_moe = train_state_nnx.create_train_state(gemma4_moe, optax.adamw(1e-3), cfg_qwen)
    self.assertEqual(ts_gemma4_moe.model.pre_forward_scale_2.get_value().dtype, jnp.float32)
    self.assertNotIn("pre_forward_scale_2", ts_gemma4_moe.optimizer.master_residuals)

  def test_to_linen_checkpoint_dict_places_reconstructed_f32_on_pinned_host_with_fallback(self):
    devices = np.array(jax.devices()[:2])
    mesh = Mesh(devices, ("fsdp",))
    cfg = _ts_config(
        fp32_master_storage="residual16",
        parameter_memory_host_offload=False,
    )
    with jax.set_mesh(mesh):
      model = _TinyResidualModel(nnx.Rngs(31))
      sharded_spec = NamedSharding(mesh, P("fsdp", None), memory_kind="device")
      model.embed.set_value(jax.device_put(model.embed.get_value(), sharded_spec))
      ts = train_state_nnx.create_train_state(model, optax.adamw(1e-3), cfg)

    self.assertEqual(ts.model.embed.get_value().sharding.memory_kind, "device")
    ckpt_dict = train_state_nnx.to_checkpoint_dict(ts)
    saved_embed = ckpt_dict["params"]["params"]["embed"]
    self.assertEqual(saved_embed.dtype, jnp.float32)
    self.assertEqual(saved_embed.sharding.memory_kind, "pinned_host")

    # Verify public `to_checkpoint_dict` fallback to `device` when `with_memory_kind("pinned_host")` raises ValueError:
    orig_with_mem = NamedSharding.with_memory_kind

    def _reject_pinned_host(self_sharding, kind):
      if kind == "pinned_host":
        raise ValueError("pinned_host unsupported on mock backend")
      return orig_with_mem(self_sharding, kind)

    with mock.patch.object(NamedSharding, "with_memory_kind", _reject_pinned_host):
      fallback_ckpt = train_state_nnx.to_checkpoint_dict(ts)
    fallback_embed = fallback_ckpt["params"]["params"]["embed"]
    self.assertEqual(fallback_embed.dtype, jnp.float32)
    self.assertEqual(fallback_embed.sharding.memory_kind, "device")
    np.testing.assert_array_equal(np.asarray(fallback_embed), np.asarray(saved_embed))

    # Verify `_transfer_sharding` fallback when raw `jax.device_put` raises CPU PjRt `{1,0:S(5)}` RuntimeError:
    real_device_put = jax.device_put

    def _fail_s5_device_put(arr, dst):
      if isinstance(arr, jax.Array) and getattr(getattr(arr, "sharding", None), "memory_kind", None) == "pinned_host":
        raise RuntimeError("UNIMPLEMENTED: Unsupported layout used for PjRt CPU buffers: {1,0:S(5)}")
      return real_device_put(arr, dst)

    host_embed = real_device_put(ts.model.embed.get_value(), sharded_spec.with_memory_kind("pinned_host"))
    with mock.patch.object(train_state_nnx.jax, "device_put", side_effect=_fail_s5_device_put):
      staged_dev = train_state_nnx._device_put_staged(host_embed, sharded_spec)  # pylint: disable=protected-access
    self.assertEqual(staged_dev.sharding.memory_kind, "device")
    np.testing.assert_array_equal(np.asarray(staged_dev), np.asarray(ts.model.embed.get_value()))

    # Verify non-CPU `RuntimeError`s in `_transfer_sharding` re-raise immediately without `jit`/`np.asarray` fallback:
    with (
        mock.patch.object(train_state_nnx.jax, "default_backend", return_value="tpu"),
        mock.patch.object(
            train_state_nnx.jax, "device_put", side_effect=RuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        ),
        mock.patch.object(train_state_nnx.jax, "jit") as spy_jit,
    ):
      with self.assertRaisesRegex(RuntimeError, "RESOURCE_EXHAUSTED"):
        train_state_nnx._transfer_sharding(ts.model.embed.get_value(), sharded_spec)  # pylint: disable=protected-access
      self.assertEqual(spy_jit.call_count, 0)

  def test_chunked_save_reconstruction_batches_syncs_and_handles_oversized_leaves(self):
    devices = np.array(jax.devices()[:2])
    mesh = Mesh(devices, ("fsdp",))
    sharded_2d = NamedSharding(mesh, P("fsdp", None), memory_kind="device")
    replicated_1d = NamedSharding(mesh, P(None), memory_kind="device")

    rng = np.random.default_rng(20261120)

    def _make_f32_bits_array(shape, sharding_spec):
      raw_u32 = rng.integers(0x00800000, 0x45800000, size=shape, dtype=np.uint32)
      raw_u32.flat[0] = np.uint32(0x3F808000)  # exact +halfway tie
      raw_u32.flat[1] = np.uint32(0xBF808000)  # exact -halfway tie
      raw_u32.flat[2] = np.uint32(0x00800000)  # smallest positive normal (2^-126)
      raw_u32.flat[3] = np.uint32(0x7F7FFFFF)  # +finfo(float32).max
      arr_f32 = jax.lax.bitcast_convert_type(jnp.asarray(raw_u32), jnp.float32)
      return jax.device_put(arr_f32, sharding_spec), raw_u32

    expected_u32: dict[str, np.ndarray] = {}

    class _MultiLeafModel(nnx.Module):
      """Model with 5 small sharded leaves (2 KiB/dev f32), 1 oversized leaf (16 KiB/dev f32), and 2 f32 exemptions."""

      def __init__(self):
        specs = [
            ("w0", (32, 32), sharded_2d),
            ("w1", (32, 32), sharded_2d),
            ("w2_huge", (128, 64), sharded_2d),
            ("w3", (32, 32), sharded_2d),
            ("w4", (32, 32), sharded_2d),
            ("w5", (32, 32), sharded_2d),
        ]
        weights = {}
        for name, shape, s_spec in specs:
          arr, u32 = _make_f32_bits_array(shape, s_spec)
          weights[name] = nnx.Param(arr)
          expected_u32[name] = u32
        self.weights = nnx.Dict(weights)
        a_log_arr, a_log_u32 = _make_f32_bits_array((8,), replicated_1d)
        gate_arr, gate_u32 = _make_f32_bits_array((8,), replicated_1d)
        self.linear_attn = nnx.Dict({"A_log": nnx.Param(a_log_arr, keep_fp32_master=True)})
        self.gate = nnx.Dict({"bias": nnx.Param(gate_arr, keep_fp32_master=True)})
        expected_u32["A_log"] = a_log_u32
        expected_u32["gate_bias"] = gate_u32

    cfg = _ts_config(
        fp32_master_storage="residual16",
        routed_bias=True,
    )
    with jax.set_mesh(mesh):
      model = _MultiLeafModel()
      ts = train_state_nnx.create_train_state(model, optax.adamw(1e-3), cfg)

    with (
        mock.patch.object(train_state_nnx, "_RECONSTRUCT_SAVE_CHUNK_BYTES", 5_000),
        mock.patch.object(train_state_nnx.jax, "block_until_ready", wraps=jax.block_until_ready) as spy_block,
    ):
      ckpt = train_state_nnx.to_checkpoint_dict(ts)

    chunk_lengths = [len(call.args[0]) for call in spy_block.call_args_list]
    self.assertEqual(chunk_lengths, [2, 1, 2, 1])

    saved_params = ckpt["params"]["params"]
    for name in ("w0", "w1", "w2_huge", "w3", "w4", "w5"):
      leaf = saved_params["weights"][name]
      self.assertEqual(leaf.dtype, jnp.float32)
      self.assertEqual(leaf.sharding.memory_kind, "pinned_host")
      got_u32 = np.asarray(jax.lax.bitcast_convert_type(jax.device_put(leaf, sharded_2d), jnp.uint32))
      np.testing.assert_array_equal(got_u32, expected_u32[name], err_msg=name)

    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(saved_params["linear_attn"]["A_log"], jnp.uint32)),
        expected_u32["A_log"],
    )
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(saved_params["gate"]["bias"], jnp.uint32)),
        expected_u32["gate_bias"],
    )

    with mock.patch.object(train_state_nnx.jax, "block_until_ready", wraps=jax.block_until_ready) as spy_default:
      ckpt_default = train_state_nnx.to_checkpoint_dict(ts)
    self.assertEqual([len(call.args[0]) for call in spy_default.call_args_list], [6])
    np.testing.assert_array_equal(
        np.asarray(ckpt_default["params"]["params"]["weights"]["w2_huge"]),
        np.asarray(saved_params["weights"]["w2_huge"]),
    )

    abs_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None)),
        nnx.state(ts),
    )
    with mock.patch.object(train_state_nnx.jax, "block_until_ready", wraps=jax.block_until_ready) as spy_abs:
      abs_ckpt = train_state_nnx.to_checkpoint_dict(abs_state)
    self.assertEqual(spy_abs.call_count, 0)
    self.assertEqual(abs_ckpt["params"]["params"]["weights"]["w2_huge"].dtype, jnp.float32)

  def test_explicit_mesh_zero1_sharding_and_none_tracer_sharding(self):
    devices = np.array(jax.devices()[:4]).reshape(2, 2)
    mesh_z1 = Mesh(devices, ("data", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    cfg_z1 = _ts_config(
        shard_optimizer_over_data=True,
        optimizer_memory_host_offload=True,
        parameter_memory_host_offload=False,
        fp32_master_storage="residual16",
        logical_axis_rules=[("fsdp", "fsdp")],
        use_m3_model=False,
    )

    def init_z1():
      m = _TinyResidualModel(nnx.Rngs(0))
      return train_state_nnx.create_train_state(m, optax.adamw(1e-3), cfg_z1)

    _, _, z1_shardings = maxtext_utils.get_abstract_state_nnx(cfg_z1, mesh_z1, init_z1, is_training=True)
    self.assertEqual(
        z1_shardings.optimizer["master_residuals"]["linear1"]["kernel"].get_value().spec,
        P("data", None),
    )
    self.assertEqual(
        z1_shardings.optimizer["master_residuals"]["linear1"]["kernel"].get_value().memory_kind,
        "pinned_host",
    )
    _, updated_z1 = sharding.maybe_update_params_sharding_with_opt(cfg_z1, z1_shardings)
    self.assertEqual(updated_z1.model["linear1"]["kernel"].get_value().spec, P("data", None))

    devices_2 = np.array(jax.devices()[:2]).reshape(2, 1)

    class _ExplicitStepModel(nnx.Module):
      """Model exercising both residual16 weights (`w`) and float32-exempt params (`linear_attn.A_log`)."""

      def __init__(self):
        self.w = nnx.Param(reshard(jnp.linspace(-1.5, 1.5, 32, dtype=jnp.float32).reshape(8, 4), P(None, None)))
        self.linear_attn = nnx.Dict(
            {"A_log": nnx.Param(reshard(jnp.linspace(-0.5, 0.5, 4, dtype=jnp.float32), P(None)), keep_fp32_master=True)}
        )

      def __call__(self, x: jax.Array) -> jax.Array:
        h = x @ self.w.get_value().astype(x.dtype)
        return h * jnp.exp(self.linear_attn["A_log"].get_value()).astype(h.dtype)

    cfg_r16 = _ts_config(fp32_master_storage="residual16")
    cfg_f32 = _ts_config(fp32_master_storage="f32")
    real_typeof = jax.typeof

    def _typeof_tracer_sharding_none(x):
      aval = real_typeof(x)
      if isinstance(x, jax.core.Tracer) and getattr(aval, "sharding", None) is not None:
        if aval.sharding.spec in (P(None, None), P(None)):
          return SimpleNamespace(shape=aval.shape, dtype=aval.dtype, sharding=None)
      return aval

    for axis_types in ((AxisType.Explicit, AxisType.Explicit), (AxisType.Explicit, AxisType.Auto)):
      mesh = Mesh(devices_2, ("data", "fsdp"), axis_types=axis_types)
      with self.subTest(axis_types=axis_types), jax.set_mesh(mesh):
        m_r16 = _ExplicitStepModel()
        m_f32 = _ExplicitStepModel()
        m_f32.w.set_value(m_f32.w.get_value().astype(jnp.bfloat16).astype(jnp.float32))
        m_r16.w.set_value(m_r16.w.get_value().astype(jnp.bfloat16).astype(jnp.float32))

        ts_r16 = train_state_nnx.create_train_state(m_r16, optax.adamw(1e-2), cfg_r16)
        ts_f32 = train_state_nnx.create_train_state(m_f32, optax.adamw(1e-2), cfg_f32)

        def _shard_data_dim0(arr):
          if isinstance(arr, jax.Array) and arr.ndim >= 1:
            return reshard(arr, P(*(("data",) + (None,) * (arr.ndim - 1))))
          return arr

        # Shard entire `ts_r16.optimizer` (`opt_state` + sibling `master_residuals`) over Explicit `"data"` (ZeRO-1):
        nnx.update(ts_r16.optimizer, jax.tree.map(_shard_data_dim0, nnx.state(ts_r16.optimizer)))
        graphdef_r16, state_r16 = nnx.split(ts_r16)
        graphdef_f32, state_f32 = nnx.split(ts_f32)
        model_graphdef_r16 = nnx.graphdef(ts_r16.model)
        model_graphdef_f32 = nnx.graphdef(ts_f32.model)

        x_in = reshard(jnp.ones((4, 8), dtype=jnp.bfloat16), P("data", None))

        def _make_step(graphdef_ts, graphdef_mdl):
          @jax.jit
          def _step(st, x):
            ts_local = nnx.merge(graphdef_ts, st)
            params_in = jax.tree.map(
                lambda v: reshard(v, P(*((None,) * v.ndim))) if isinstance(v, jax.Array) and v.ndim >= 1 else v,
                nnx.state(ts_local.model, nnx.Param),
            )
            nnx.update(ts_local.model, params_in)

            def _loss(p_state):
              mdl = nnx.merge(graphdef_mdl, p_state)
              return jnp.mean(mdl(x).astype(jnp.float32) ** 2)

            loss, grads = jax.value_and_grad(_loss)(params_in)
            grads_f32 = jax.tree.map(lambda g: g.astype(jnp.float32) if g.dtype == jnp.bfloat16 else g, grads)
            ts_local.apply_gradients(grads_f32)
            return loss, nnx.state(ts_local)

          return _step

        loss_r16, out_r16 = _make_step(graphdef_r16, model_graphdef_r16)(state_r16, x_in)
        loss_f32, out_f32 = _make_step(graphdef_f32, model_graphdef_f32)(state_f32, x_in)
        np.testing.assert_allclose(float(loss_r16), float(loss_f32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(out_r16["model"]["linear_attn"]["A_log"].get_value()),
            np.asarray(out_f32["model"]["linear_attn"]["A_log"].get_value()),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(out_r16["model"]["w"].get_value().sharding.spec, P("data", None))
        self.assertEqual(out_r16["model"]["linear_attn"]["A_log"].get_value().sharding.spec, P("data"))

        with mock.patch.object(train_state_nnx.jax, "typeof", side_effect=_typeof_tracer_sharding_none):
          loss_none, out_none = _make_step(graphdef_r16, model_graphdef_r16)(state_r16, x_in)
        np.testing.assert_array_equal(np.asarray(loss_none), np.asarray(loss_r16))
        np.testing.assert_array_equal(
            np.asarray(out_none["model"]["w"].get_value()),
            np.asarray(out_r16["model"]["w"].get_value()),
        )
        np.testing.assert_array_equal(
            np.asarray(out_none["optimizer"]["master_residuals"]["w"].get_value()),
            np.asarray(out_r16["optimizer"]["master_residuals"]["w"].get_value()),
        )
        np.testing.assert_array_equal(
            np.asarray(out_none["model"]["linear_attn"]["A_log"].get_value()),
            np.asarray(out_r16["model"]["linear_attn"]["A_log"].get_value()),
        )

        state_r16_repl = jax.tree.map(
            lambda v: reshard(v, P(*((None,) * v.ndim))) if isinstance(v, jax.Array) and v.ndim >= 1 else v,
            state_r16,
        )
        with mock.patch.object(train_state_nnx.jax, "typeof", return_value=SimpleNamespace(sharding=None)):
          loss_all_none, _ = _make_step(graphdef_r16, model_graphdef_r16)(state_r16_repl, x_in)
          out_dict = {}
          train_state_nnx._collect_mu_specs({"mu": {"a": jnp.ones((2,))}}, out_dict)  # pylint: disable=protected-access
          self.assertEqual(out_dict, {})
        for opt_node in (
            nnx.as_pure(state_r16["optimizer"]["opt_state"]),
            nnx.as_pure(ts_r16.optimizer.opt_state),
        ):
          collected_specs = {}
          train_state_nnx._collect_mu_specs(opt_node, collected_specs)  # pylint: disable=protected-access
          self.assertEqual(collected_specs[("linear_attn", "A_log")], P("data"))
        np.testing.assert_allclose(float(loss_all_none), float(loss_r16), rtol=1e-6, atol=1e-6)


class TestResidual16ConfigAndFailFast(unittest.TestCase):
  """Tests fp32_master_storage config validation and fail-fast entrypoints."""

  def test_pyconfig_and_pydantic_validation(self):
    cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "run_name=test_res16",
            "fp32_master_storage=residual16",
        ]
    )
    self.assertEqual(cfg.fp32_master_storage, "residual16")

    # Verify real MaxTextConfig from pyconfig initializes Residual16Optimizer end-to-end without stub fields:
    prev_eager = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    try:
      real_cfg_model = _TinyResidualModel(nnx.Rngs(5), include_gate=True)
      ts_real_cfg = train_state_nnx.create_train_state(real_cfg_model, optax.adamw(1e-3), cfg)
      self.assertEqual(ts_real_cfg.model.embed.get_value().dtype, jnp.bfloat16)
      self.assertIn("embed", ts_real_cfg.optimizer.master_residuals)
      tokens = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
      grads = nnx.grad(lambda m: jnp.mean(m(tokens).astype(jnp.float32) ** 2))(ts_real_cfg.model)
      ts_real_cfg.apply_gradients(grads)
      ckpt_real = train_state_nnx.to_checkpoint_dict(ts_real_cfg)
      self.assertEqual(ckpt_real["params"]["params"]["embed"].dtype, jnp.float32)
    finally:
      nnx.use_eager_sharding(prev_eager)

    # Rejects invalid fp32_master_storage literal:
    with self.assertRaises(ValidationError):
      pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "run_name=test_bad_storage",
              "fp32_master_storage=fp16",
          ]
      )

    # Rejects FP8 weight_dtype with fp32_master_storage=residual16:
    with self.assertRaisesRegex(ValidationError, "requires weight_dtype to be 'float32' or 'bfloat16'"):
      pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "run_name=test_bad_weight_dtype",
              "fp32_master_storage=residual16",
              "weight_dtype=float8_e4m3fn",
          ]
      )

    # RLConfig inherits fp32_master_storage and rejects residual16:
    self.assertIn("fp32_master_storage", config_types.RLConfig.model_fields)
    rl_yml_path = os.path.join(os.path.dirname(get_test_config_path()), "post_train", "rl.yml")
    with self.assertRaisesRegex(ValidationError, "fp32_master_storage='residual16' is not supported"):
      pyconfig.initialize_pydantic(
          ["", rl_yml_path, "model_name=qwen3-0.6b", "fp32_master_storage=residual16"],
          config_class=config_types.RLConfig,
      )

    # Rejects unsupported trainer combinations on MaxTextConfig (m3, LoRA, DiLoCo, SFT/DPO/GRPO):
    for extra_args, match_pat in (
        (["use_m3_model=True", "fp32_master_storage=residual16"], "use_m3_model"),
        (["lora.enable_lora=True", "fp32_master_storage=residual16"], "LoRA"),
        (["enable_diloco=True", "fp32_master_storage=residual16"], "DiLoCo"),
        (["use_sft=True", "fp32_master_storage=residual16"], "use_sft"),
        (["use_dpo=True", "packing=False", "fp32_master_storage=residual16"], "use_dpo"),
        (["use_grpo=True", "fp32_master_storage=residual16"], "use_grpo"),
    ):
      with self.subTest(extra_args=extra_args):
        with self.assertRaisesRegex(ValidationError, match_pat):
          pyconfig.initialize(["", get_test_config_path(), "run_name=test_unsupported"] + extra_args)

  def test_unsupported_entrypoints_fail_fast(self):
    # 1. lora_utils.setup_initial_lora_state rejects residual16:
    res16_ns = SimpleNamespace(fp32_master_storage="residual16")
    with self.assertRaisesRegex(ValueError, "LoRA does not support fp32_master_storage"):
      lora_utils.setup_initial_lora_state(None, optax.sgd(1e-3), res16_ns, None, None, "/tmp/fake_lora")

    # 2. Distillation rejects residual16 on student or teacher:
    if importlib.util.find_spec("tunix") is not None:
      from maxtext.trainers.post_train.distillation import train_distill  # pylint: disable=import-outside-toplevel

      ok_ns = SimpleNamespace(fp32_master_storage="f32", vocab_size=32)
      bad_student = SimpleNamespace(fp32_master_storage="residual16", vocab_size=32)
      bad_teacher = SimpleNamespace(fp32_master_storage="residual16", vocab_size=32)
      with self.assertRaisesRegex(ValueError, "Distillation \\(student_config\\) does not support fp32_master_storage"):
        train_distill.train_distill(bad_student, ok_ns)
      with self.assertRaisesRegex(ValueError, "Distillation \\(teacher_config\\) does not support fp32_master_storage"):
        train_distill.train_distill(ok_ns, bad_teacher)


@pytest.mark.tpu_only
class TestResidual16TpuCiExecution(unittest.TestCase):
  """TPU hardware suite ensuring PR CI (`tpu7x-unit`) validates bf16 FTZ, restore, and ZeRO-1 sharding on TPU."""

  def test_tpu_ieee754_roundtrip_and_subnormal_ftz(self):
    bit_suite = TestBitLevelResidual16Encoding()
    bit_suite.test_exact_roundtrip_across_ieee754_edge_cases_and_random_patterns()
    bit_suite.test_nan_patterns_never_wrap_to_zero_or_finite()
    bit_suite.test_exact_bf16_inputs_produce_zero_residual()

  def test_tpu_zero_bf16_negative_residual_guard_and_subnormal_restore_step1(self):
    case = TestBitLevelResidual16Encoding(
        methodName="test_zero_bf16_negative_residual_guard_and_subnormal_checkpoint_restore_step1"
    )
    res = unittest.TestResult()
    case.run(res)
    if res.errors or res.failures:
      _, tb = (res.errors + res.failures)[0]
      self.fail(tb)

  def test_tpu_explicit_mesh_zero1_and_small_lr_parity(self):
    for cls, method in (
        (TestSmallLRConvergenceParity, "test_adamw_small_lr_updates_match_f32_while_bf16_stalls"),
        (TestFp32ParamExemptionsAndSaveMemoryKind, "test_explicit_mesh_zero1_sharding_and_none_tracer_sharding"),
    ):
      if method == "test_explicit_mesh_zero1_sharding_and_none_tracer_sharding" and len(jax.devices()) < 4:
        continue
      case = cls(methodName=method)
      res = unittest.TestResult()
      case.run(res)
      if res.errors or res.failures:
        _, tb = (res.errors + res.failures)[0]
        self.fail(tb)


if __name__ == "__main__":
  unittest.main()
