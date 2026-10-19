# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-sided CPU bit-identity and TPU7x program verification for:
1. gdn_cp_pass1_states_and_bf16_qkv (GDNMode.PER_SEQ_STATES states-only CP Pass 1 + VMEM bf16->fp32 qkv upcast)
2. gdn_out_proj_token_minor (token-minor layout constraint on GDN out_proj input in training)
"""

# pylint: disable=protected-access

import contextlib
import functools
import inspect
import os
import re
import subprocess
import sys
import types
from unittest import mock

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
  ).strip()

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.experimental.pallas import tpu as pltpu
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.gdn import compute_gdn
from maxtext.kernels.gdn import config as gdn_config
from maxtext.kernels.gdn import gdn_bwd_pallas
from maxtext.kernels.gdn import model_runner
from maxtext.kernels.gdn import vmem_ldst
from maxtext.kernels.gdn import wrapper
from maxtext.kernels.gdn.gdn_bwd import api as gdn_api
from maxtext.kernels.gdn.gdn_bwd import cp_gdn
from maxtext.layers import moe
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

_PASS_TOKEN = "GDN_PASS1_STATES_AND_OUT_PROJ_CHECKS_PASSED"
_CHUNK = 64
_KERNEL = 4
_HEAD_DIM = 128


def _bits(x):
  """The array's bytes, so equality means bit-identical (NaN-safe, -0.0 != 0.0)."""
  return np.asarray(jax.device_get(x)).tobytes()


def _pltpu_repeat_cpu(x, repeats, axis):
  reps = [1] * x.ndim
  reps[axis] = repeats
  return jnp.tile(x, reps)


def _pltpu_roll_cpu(x, shift, axis, *, stride=None, stride_axis=None):
  if stride is None:
    return jnp.roll(x, shift, axis=axis)
  slices = []
  for i in range(x.shape[stride_axis]):
    slc = jnp.take(x, jnp.array([i]), axis=stride_axis)
    slices.append(jnp.roll(slc, shift + i * stride, axis=axis))
  return jnp.concatenate(slices, axis=stride_axis)


class _FlatView:

  def __init__(self, box, idx, shape):
    self.box = box
    self.idx = idx
    self.shape = shape

  def __setitem__(self, slc, val):
    sub = self.box.arr[self.idx]
    flat = sub.reshape(self.shape).at[slc].set(val)
    self.box.arr = self.box.arr.at[self.idx].set(flat.reshape(sub.shape))


class _AtIndex:

  def __init__(self, box, idx):
    self.box = box
    self.idx = idx

  def reshape(self, *shape):
    return _FlatView(self.box, self.idx, shape)


class _AtHelper:

  def __init__(self, box):
    self.box = box

  def __getitem__(self, idx):
    return _AtIndex(self.box, idx)


class _BoxRef:
  """Minimal mutable Ref wrapper for testing VMEM load/store helpers on CPU."""

  def __init__(self, arr):
    self.arr = arr

  @property
  def dtype(self):
    return self.arr.dtype

  @property
  def shape(self):
    return self.arr.shape

  @property
  def at(self):
    return _AtHelper(self)

  def reshape(self, *shape):
    return self.arr.reshape(*shape)

  def __getitem__(self, idx):
    return self.arr[idx]

  def __setitem__(self, idx, val):
    self.arr = val if idx is Ellipsis else self.arr.at[idx].set(val)


def _tpu_info_cpu():
  return types.SimpleNamespace(num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024 * 1024)


@contextlib.contextmanager
def _pallas_cpu_interpret_context():
  """Runs real Pallas GDN forward kernels on CPU via force_tpu_interpret_mode + _BoxRef VMEM ref wrappers."""
  orig_load_as_qkv_large = vmem_ldst.load_as_qkv_large
  orig_load_compact_to_large = vmem_ldst.load_compact_to_large
  orig_upcast_large_to_compact = vmem_ldst.upcast_large_to_compact

  def _load_as_qkv_large(qkv_ref, cfg):
    return orig_load_as_qkv_large(_BoxRef(qkv_ref[...]), cfg)

  def _load_compact_to_large(in_ref):
    return orig_load_compact_to_large(_BoxRef(in_ref[...]))

  def _upcast_large_to_compact(in_ref, out_ref):
    box_out = _BoxRef(out_ref[...])
    orig_upcast_large_to_compact(_BoxRef(in_ref[...]), box_out)
    out_ref[...] = box_out.arr

  with (
      mock.patch.object(gdn_api, "target_platform", return_value="tpu"),
      mock.patch.object(pltpu, "get_tpu_info", side_effect=_tpu_info_cpu),
      mock.patch.object(pltpu, "repeat", side_effect=_pltpu_repeat_cpu, create=True),
      mock.patch.object(pltpu, "roll", side_effect=_pltpu_roll_cpu, create=True),
      mock.patch.object(vmem_ldst, "load_as_qkv_large", side_effect=_load_as_qkv_large),
      mock.patch.object(vmem_ldst, "load_compact_to_large", side_effect=_load_compact_to_large),
      mock.patch.object(vmem_ldst, "upcast_large_to_compact", side_effect=_upcast_large_to_compact),
      pltpu.force_tpu_interpret_mode(),
  ):
    yield


def _run_upcast_large_to_compact(x_bf16_large):
  in_ref = _BoxRef(x_bf16_large)
  seq_tile, chunk, dim = x_bf16_large.shape
  out_ref = _BoxRef(jnp.zeros((seq_tile, chunk, 1, dim), dtype=jnp.float32))
  with mock.patch.object(pltpu, "get_tpu_info", side_effect=_tpu_info_cpu):
    vmem_ldst.upcast_large_to_compact(in_ref, out_ref)
  return out_ref.arr


class UpcastLargeToCompactTest(parameterized.TestCase):
  """vmem_ldst.upcast_large_to_compact equals .astype(jnp.float32)[:, :, None, :] bit for bit, with negative control."""

  @parameterized.named_parameters(
      ("chunk64_width1024", 64, 1024),
      ("chunk64_width12288", 64, 12288),
      ("chunk8_width2048", 8, 2048),
  )
  def test_bit_identical_to_hbm_upcast(self, chunk, dim):
    rng = np.random.default_rng(chunk + dim)
    x_bf16 = jnp.asarray(rng.standard_normal((1, chunk, dim)).astype(np.float32)).astype(jnp.bfloat16)
    ref = x_bf16.astype(jnp.float32)[:, :, None, :]
    got = jax.jit(_run_upcast_large_to_compact)(x_bf16)
    self.assertEqual(got.shape, ref.shape)
    self.assertEqual(got.dtype, jnp.float32)
    self.assertEqual(float(jnp.max(jnp.abs(got - ref))), 0.0)
    self.assertEqual(_bits(got), _bits(ref))

    # Negative control: swapping the (2, 4) sublane split order without restoring permutes rows and MUST fail.
    wrong = x_bf16.reshape(-1, 4, 2, 128).transpose(0, 2, 1, 3).astype(jnp.float32).reshape(ref.shape)
    self.assertNotEqual(_bits(wrong), _bits(ref))

  def test_round_trip_with_load_compact_to_large(self):
    rng = np.random.default_rng(999)
    x_bf16 = jnp.asarray(rng.standard_normal((1, 64, 1024)).astype(np.float32)).astype(jnp.bfloat16)
    compact = _run_upcast_large_to_compact(x_bf16)
    with mock.patch.object(pltpu, "get_tpu_info", side_effect=_tpu_info_cpu):
      recovered = vmem_ldst.load_compact_to_large(_BoxRef(compact))
    self.assertEqual(_bits(recovered), _bits(x_bf16.astype(jnp.float32)))


class ChunkedGdnStatesModeTest(parameterized.TestCase):
  """compute_gdn.chunked_gdn in PER_SEQ_STATES mode produces bit-identical recurrent_state and t_inv to PER_SEQ."""

  @parameterized.named_parameters(
      ("no_seg_no_qknorm", "none", False, "baseline_b16"),
      ("no_seg_qknorm", "none", True, "baseline_b16"),
      ("seg_single_doc_cont", "cont", True, "xor_dnc"),
      ("seg_qknorm_baseline", "multi", True, "baseline_b16"),
      ("seg_qknorm_xor", "multi", True, "xor_dnc"),
  )
  def test_states_mode_bit_identical_to_per_seq(self, seg_mode, use_qk_norm, tri_inv_algo):
    nk, nv = 2, 4
    dim = (2 * nk + nv) * _HEAD_DIM
    has_seg_ids = seg_mode != "none"
    rng = np.random.default_rng(42 + int(has_seg_ids) * 10 + int(use_qk_norm))
    real_sizes = jnp.array([_CHUNK], dtype=jnp.int32)
    q_large = jnp.asarray((rng.standard_normal((1, nk, _CHUNK, _HEAD_DIM)) * 0.2).astype(np.float32))
    k_large = jnp.asarray((rng.standard_normal((1, nk, _CHUNK, _HEAD_DIM)) * 0.2).astype(np.float32))
    v_large = jnp.asarray((rng.standard_normal((1, nv, _CHUNK, _HEAD_DIM)) * 0.2).astype(np.float32))
    b_large = jnp.zeros((1, 1, _CHUNK, 128), dtype=jnp.float32)
    b_large = b_large.at[:, :, :, :nv].set(jnp.asarray(rng.standard_normal((1, 1, _CHUNK, nv)).astype(np.float32)))
    if seg_mode == "cont":
      seg_col = np.full(_CHUNK, 1.0, dtype=np.float32)
      b_large = b_large.at[0, 0, :, nv].set(jnp.asarray(seg_col))
      b_large = b_large.at[0, 0, _KERNEL - 1, nv + 1].set(1.0)
    elif seg_mode == "multi":
      seg_col = np.concatenate([np.full(24, 1.0), np.full(24, 2.0), np.full(_CHUNK - 48, -2.0)]).astype(np.float32)
      b_large = b_large.at[0, 0, :, nv].set(jnp.asarray(seg_col))
      b_large = b_large.at[0, 0, _KERNEL - 1, nv + 1].set(1.0)
    a_large = jnp.zeros((1, 1, _CHUNK, 128), dtype=jnp.float32)
    a_large = a_large.at[:, :, :, :nv].set(jnp.asarray(rng.standard_normal((1, 1, _CHUNK, nv)).astype(np.float32)))
    a_log = jnp.pad(jnp.asarray((rng.standard_normal((nv,)) * 0.1).astype(np.float32)), (0, 128 - nv))
    # Weak decay (dt_bias ~ -6.0 -> gating_last ~ 0.85) ensures initial state and ws survive all 64 steps.
    dt_bias = jnp.pad(jnp.asarray((rng.standard_normal((nv,)) * 0.1 - 6.0).astype(np.float32)), (0, 128 - nv))
    init_state = jnp.asarray((rng.standard_normal((1, nv, _HEAD_DIM, _HEAD_DIM)) * 0.5).astype(np.float32))

    def make_cfg(mode):
      return gdn_config.GDNConfig(
          mode=mode,
          dtypes=gdn_config.Dtypes(jnp.bfloat16, jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32),
          batch_size=_CHUNK,
          dim_size=dim,
          kernel_size=_KERNEL,
          tile_size=_CHUNK,
          num_kq_heads=nk,
          num_v_heads=nv,
          kq_head_dim=_HEAD_DIM,
          v_head_dim=_HEAD_DIM,
          has_seg_ids=has_seg_ids,
          use_qk_norm_in_gdn=use_qk_norm,
          tri_inv_algo=tri_inv_algo,
          head_group_size=1,
          emit_t_inv=True,
          is_prefill_only=True,
      )

    cfg_full = make_cfg(gdn_config.GDNMode.PER_SEQ)
    cfg_states = make_cfg(gdn_config.GDNMode.PER_SEQ_STATES)

    with (
        mock.patch.object(pltpu, "get_tpu_info", side_effect=_tpu_info_cpu),
        mock.patch.object(pltpu, "repeat", side_effect=_pltpu_repeat_cpu, create=True),
        mock.patch.object(pltpu, "roll", side_effect=_pltpu_roll_cpu, create=True),
    ):
      out_full, state_full, t_inv_full = jax.jit(
          lambda: compute_gdn.chunked_gdn(
              real_sizes, q_large, k_large, v_large, b_large, a_large, init_state, a_log, dt_bias, cfg_full
          )
      )()
      out_states, state_states, t_inv_states = jax.jit(
          lambda: compute_gdn.chunked_gdn(
              real_sizes, q_large, k_large, v_large, b_large, a_large, init_state, a_log, dt_bias, cfg_states
          )
      )()
      _, state_zero_init, _ = jax.jit(
          lambda: compute_gdn.chunked_gdn(
              real_sizes,
              q_large,
              k_large,
              v_large,
              b_large,
              a_large,
              jnp.zeros_like(init_state),
              a_log,
              dt_bias,
              cfg_states,
          )
      )()

    self.assertIsNotNone(out_full)
    self.assertIsNone(out_states)
    self.assertEqual(float(jnp.max(jnp.abs(state_states - state_full))), 0.0)
    self.assertEqual(_bits(state_states), _bits(state_full))
    self.assertEqual(float(jnp.max(jnp.abs(t_inv_states - t_inv_full))), 0.0)
    self.assertEqual(_bits(t_inv_states), _bits(t_inv_full))

    if seg_mode in ("none", "cont"):
      # Negative control (jitted on both sides): zeroing init_state changes state_full when no new doc starts mid-chunk.
      self.assertGreater(float(jnp.max(jnp.abs(state_zero_init - state_full))), 1e-3)
      self.assertNotEqual(_bits(state_zero_init), _bits(state_full))
    else:
      # In multi-doc mode, segment 2 starts mid-chunk and resets the recurrent state from segment 1.
      self.assertEqual(_bits(state_zero_init), _bits(state_full))


class OutProjTokenMinorTest(absltest.TestCase):
  """gdn_out_proj_token_minor (80c149572) is bit-identical in fwd & bwd and emits LayoutConstraint in train fwd only."""

  def test_fwd_and_bwd_bit_identical_and_layout_constraint_present(self):
    rng = np.random.default_rng(123)
    x = jnp.asarray(rng.standard_normal((2, 32, 64)).astype(np.float32)).astype(jnp.bfloat16)
    w = jnp.asarray((rng.standard_normal((64, 48)) * 0.1).astype(np.float32)).astype(jnp.bfloat16)
    dy = jnp.asarray(rng.standard_normal((2, 32, 48)).astype(np.float32)).astype(jnp.bfloat16)

    def run_out_proj(gated_output, weight, token_minor, model_mode=MODEL_MODE_TRAIN):
      if model_mode == MODEL_MODE_TRAIN and token_minor:
        gated_output = moe._forward_layout_constraint(gated_output, (0, 2, 1))
      return jnp.einsum("bsd,df->bsf", gated_output.astype(jnp.float32), weight.astype(jnp.float32)).astype(
          gated_output.dtype
      )

    def loss(gated_output, weight, token_minor):
      y = run_out_proj(gated_output, weight, token_minor, MODEL_MODE_TRAIN)
      return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32))

    val_off, (dx_off, dw_off) = jax.jit(jax.value_and_grad(functools.partial(loss, token_minor=False), argnums=(0, 1)))(
        x, w
    )
    val_on, (dx_on, dw_on) = jax.jit(jax.value_and_grad(functools.partial(loss, token_minor=True), argnums=(0, 1)))(
        x, w
    )
    self.assertEqual(float(jnp.max(jnp.abs(val_on - val_off))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(dx_on - dx_off))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(dw_on - dw_off))), 0.0)
    self.assertEqual(_bits(val_on), _bits(val_off))
    self.assertEqual(_bits(dx_on), _bits(dx_off))
    self.assertEqual(_bits(dw_on), _bits(dw_off))

    # Negative control: perturbing gated_output and weight changes output and gradients.
    val_bad, (dx_bad, dw_bad) = jax.jit(jax.value_and_grad(functools.partial(loss, token_minor=True), argnums=(0, 1)))(
        x * 1.01, w * 1.01
    )
    self.assertNotEqual(_bits(val_bad), _bits(val_off))
    self.assertNotEqual(_bits(dx_bad), _bits(dx_off))
    self.assertNotEqual(_bits(dw_bad), _bits(dw_off))

    hlo_train_on = jax.jit(functools.partial(run_out_proj, token_minor=True, model_mode=MODEL_MODE_TRAIN)).lower(x, w).as_text()
    hlo_train_off = jax.jit(functools.partial(run_out_proj, token_minor=False, model_mode=MODEL_MODE_TRAIN)).lower(x, w).as_text()
    hlo_prefill_on = jax.jit(functools.partial(run_out_proj, token_minor=True, model_mode=MODEL_MODE_PREFILL)).lower(x, w).as_text()
    self.assertIn("LayoutConstraint", hlo_train_on)
    self.assertNotIn("LayoutConstraint", hlo_train_off)
    self.assertNotIn("LayoutConstraint", hlo_prefill_on)

    # Also verify through the real Qwen3NextGatedDeltaNet module (qwen3.py:1478).
    def make_qwen_layer(token_minor, model_mode=MODEL_MODE_TRAIN):
      cfg = pyconfig.initialize(
          [
              None,
              get_test_config_path(),
              "run_name=gdn_out_proj_token_minor_test",
              "dtype=bfloat16",
              "weight_dtype=float32",
              "decoder_block=qwen3_next",
              "attention=dot_product",
              "base_emb_dim=64",
              "base_num_query_heads=2",
              "base_num_kv_heads=2",
              "head_dim=32",
              "gdn_num_key_heads=1",
              "gdn_num_value_heads=2",
              "gdn_key_head_dim=128",
              "gdn_value_head_dim=128",
              "gdn_conv_kernel_dim=4",
              "gdn_chunk_size=64",
              "max_target_length=64",
              "use_gdn_kernel=False",
              f"gdn_out_proj_token_minor={token_minor}",
          ],
          skip_jax_distributed_system=True,
      )
      mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
      return qwen3.Qwen3NextGatedDeltaNet(
          config=cfg,
          mesh=mesh,
          model_mode=model_mode,
          inputs_shape=(1, 64, cfg.emb_dim),
          rngs=nnx.Rngs(0),
      )

    def lower_qwen_layer(token_minor, model_mode):
      layer = make_qwen_layer(token_minor, model_mode)
      graphdef, state = nnx.split(layer)
      return (
          jax.jit(lambda st, h: nnx.merge(graphdef, st)(h, model_mode=model_mode)[0])
          .lower(state, hidden)
          .as_text()
      )

    hidden = jnp.asarray(rng.standard_normal((1, 64, 64)).astype(np.float32)).astype(jnp.bfloat16)
    qwen_hlo_train_on = lower_qwen_layer(True, MODEL_MODE_TRAIN)
    qwen_hlo_train_off = lower_qwen_layer(False, MODEL_MODE_TRAIN)
    qwen_hlo_prefill_on = lower_qwen_layer(True, MODEL_MODE_PREFILL)
    self.assertIn("LayoutConstraint", qwen_hlo_train_on)
    self.assertNotIn("LayoutConstraint", qwen_hlo_train_off)
    self.assertNotIn("LayoutConstraint", qwen_hlo_prefill_on)


class PallasInterpretStatesAndBf16QkvTest(absltest.TestCase):
  """Executes Pallas GDN forward kernels on CPU via TPU interpret mode to verify wrapper, memory_ref, and vmem_ldst."""

  def test_run_local_gdn_states_fwd_matches_full_forward(self):
    batch_size, seq_len = 2, 128
    nk, nv = 2, 4
    dim = (2 * nk + nv) * _HEAD_DIM
    rng = np.random.default_rng(2025)
    qkv = jnp.asarray((rng.standard_normal((batch_size, seq_len, dim)) * 0.2).astype(np.float32)).astype(jnp.bfloat16)
    b = jnp.asarray(rng.standard_normal((batch_size, seq_len, nv)).astype(np.float32)).astype(jnp.bfloat16)
    a = jnp.asarray(rng.standard_normal((batch_size, seq_len, nv)).astype(np.float32)).astype(jnp.bfloat16)
    cw = jnp.asarray((rng.standard_normal((_KERNEL, 1, dim)) * 0.2).astype(np.float32))
    cb = jnp.asarray((rng.standard_normal((dim,)) * 0.1).astype(np.float32))
    a_log = jnp.asarray((rng.standard_normal((nv,)) * 0.1).astype(np.float32))
    # Weak decay so initial recurrent_state and inter-chunk recurrent carry survive across both chunks.
    dtb = jnp.asarray((rng.standard_normal((nv,)) * 0.1 - 6.0).astype(np.float32))
    cs = jnp.asarray((rng.standard_normal((batch_size, _KERNEL - 1, dim)) * 0.3).astype(np.float32)).astype(
        jnp.bfloat16
    )
    rs = jnp.asarray((rng.standard_normal((batch_size, nv, _HEAD_DIM, _HEAD_DIM)) * 0.3).astype(np.float32))
    seg = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    seg = seg.at[1, 64:].set(2)

    common = {
        "qkv": qkv,
        "b": b,
        "a": a,
        "conv_weight": cw,
        "conv_bias": cb,
        "a_log": a_log,
        "dt_bias": dtb,
        "conv_state": cs,
        "recurrent_state": rs,
        "num_k_heads": nk,
        "num_v_heads": nv,
        "head_k_dim": _HEAD_DIM,
        "head_v_dim": _HEAD_DIM,
        "conv_kernel_size": _KERNEL,
        "chunk_size": _CHUNK,
        "use_qk_norm_in_gdn": True,
        "compute_dtype": jnp.float32,
        "segment_ids": seg,
        "conv_dense_token_stack": True,
    }
    with _pallas_cpu_interpret_context():
      (out_fp32, (cs_fp32, rs_fp32)), tinv_fp32, cstates_fp32 = gdn_bwd_pallas._run_local_gdn_decoupled_fwd(
          **common, bf16_qkv_dma=False
      )
      (out_bf16, (cs_bf16, rs_bf16)), tinv_bf16, cstates_bf16 = gdn_bwd_pallas._run_local_gdn_decoupled_fwd(
          **common, bf16_qkv_dma=True
      )
      (cs_states, rs_states), tinv_states = gdn_bwd_pallas._run_local_gdn_states_fwd(**common, bf16_qkv_dma=True)

    self.assertEqual(_bits(out_bf16), _bits(out_fp32))
    self.assertEqual(_bits(cs_bf16), _bits(cs_fp32))
    self.assertEqual(_bits(rs_bf16), _bits(rs_fp32))
    self.assertEqual(_bits(tinv_bf16), _bits(tinv_fp32))
    self.assertEqual(_bits(cstates_bf16), _bits(cstates_fp32))

    self.assertEqual(_bits(cs_states), _bits(cs_fp32))
    self.assertEqual(_bits(rs_states), _bits(rs_fp32))
    self.assertEqual(_bits(tinv_states), _bits(tinv_fp32))

  def test_prefill_bf16_qkv_matches_fp32_qkv_path(self):
    nk, nv = 2, 4
    dim = (2 * nk + nv) * _HEAD_DIM
    seq_len = 2 * _CHUNK
    rng = np.random.default_rng(31415)
    qkv_bf16 = jnp.asarray((rng.standard_normal((seq_len, dim)) * 0.2).astype(np.float32)).astype(jnp.bfloat16)
    qkv_fp32 = qkv_bf16.astype(jnp.float32)
    b_2d = jnp.asarray(rng.standard_normal((seq_len, nv)).astype(np.float32)).astype(jnp.bfloat16)
    a_2d = jnp.asarray(rng.standard_normal((seq_len, nv)).astype(np.float32)).astype(jnp.bfloat16)
    conv_weight_t = jnp.asarray((rng.standard_normal((dim, 1, _KERNEL)) * 0.2).astype(np.float32))
    conv_bias = jnp.asarray((rng.standard_normal((dim,)) * 0.1).astype(np.float32))
    a_log = jnp.asarray((rng.standard_normal((nv,)) * 0.1).astype(np.float32))
    dt_bias = jnp.asarray((rng.standard_normal((nv,)) * 0.1 - 6.0).astype(np.float32))

    common = {
        "conv_weight": conv_weight_t,
        "conv_bias": conv_bias,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "query_start_loc": jnp.array([0, seq_len], dtype=jnp.int32),
        "state_indices": jnp.array([0], dtype=jnp.int32),
        "distribution": jnp.array([0, 0, 1], dtype=jnp.int32),
        "seq_lens": jnp.array([seq_len], dtype=jnp.int32),
        "n_kq": nk,
        "n_v": nv,
        "d_k": _HEAD_DIM,
        "d_v": _HEAD_DIM,
        "kernel_size": _KERNEL,
        "compute_precision": jnp.float32.dtype,
        "mixed_tile_size": _CHUNK,
        "use_qk_norm_in_gdn": True,
        "emit_t_inv": True,
    }
    with _pallas_cpu_interpret_context():
      out_bf16, (cs_bf16, rs_bf16), tinv_bf16, cstates_bf16 = wrapper.fused_conv1d_gdn(
          qkv=qkv_bf16,
          b=b_2d,
          a=a_2d,
          conv_state=jnp.zeros((2, _KERNEL - 1, dim), dtype=jnp.bfloat16),
          recurrent_state=jnp.zeros((2, nv, _HEAD_DIM, _HEAD_DIM), dtype=jnp.float32),
          **common,
          is_prefill_only=True,
          bf16_qkv_dma=True,
      )
      out_fp32, (cs_fp32, rs_fp32), tinv_fp32, cstates_fp32 = wrapper.fused_conv1d_gdn(
          qkv=qkv_fp32,
          b=b_2d.astype(jnp.float32),
          a=a_2d.astype(jnp.float32),
          conv_state=jnp.zeros((2, _KERNEL - 1, dim), dtype=jnp.bfloat16),
          recurrent_state=jnp.zeros((2, nv, _HEAD_DIM, _HEAD_DIM), dtype=jnp.float32),
          **common,
          is_prefill_only=False,
          bf16_qkv_dma=False,
      )

    self.assertEqual(_bits(out_bf16), _bits(out_fp32.astype(jnp.bfloat16)))
    self.assertEqual(_bits(cs_bf16), _bits(cs_fp32))
    self.assertEqual(_bits(rs_bf16), _bits(rs_fp32))
    self.assertEqual(_bits(tinv_bf16), _bits(tinv_fp32))
    self.assertEqual(_bits(cstates_bf16), _bits(cstates_fp32))


class CpuDecoupledFwdBwdBitIdentityTest(parameterized.TestCase):
  """gdn_decoupled_conv1d and model_runner.run_gdn_kernel_layer with pass1_states_and_bf16_qkv=True vs False on CPU."""

  @parameterized.named_parameters(
      ("no_seg", False),
      ("with_seg", True),
  )
  def test_decoupled_fwd_bwd_bit_identical(self, with_seg):
    nk, nv, seq = 2, 4, 128
    dim = (2 * nk + nv) * _HEAD_DIM
    rng = np.random.default_rng(77 + int(with_seg))
    qkv = jnp.asarray((rng.standard_normal((1, seq, dim)) * 0.1).astype(np.float32)).astype(jnp.bfloat16)
    b = jnp.asarray(rng.standard_normal((1, seq, nv)).astype(np.float32)).astype(jnp.bfloat16)
    a = jnp.asarray(rng.standard_normal((1, seq, nv)).astype(np.float32)).astype(jnp.bfloat16)
    cw = jnp.asarray((rng.standard_normal((_KERNEL, 1, dim)) * 0.1).astype(np.float32))
    a_log = jnp.asarray((rng.standard_normal((nv,)) * 0.1).astype(np.float32))
    dtb = jnp.asarray((rng.standard_normal((nv,)) * 0.1 - 6.0).astype(np.float32))
    do = jnp.asarray(rng.standard_normal((1, seq, nv, _HEAD_DIM)).astype(np.float32))
    seg = (
        jnp.concatenate([jnp.ones((1, 64), dtype=jnp.int32), jnp.full((1, 48), 2, dtype=jnp.int32), jnp.zeros((1, 16), dtype=jnp.int32)], axis=1)
        if with_seg
        else None
    )

    def run(opt_on):
      def loss_fn(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in):
        out, (ncs, nrs) = gdn_api.gdn_decoupled_conv1d(
            qkv=qkv_in,
            b=b_in,
            a=a_in,
            conv_weight=cw_in,
            conv_bias=None,
            a_log=alog_in,
            dt_bias=dtb_in,
            conv_state=None,
            recurrent_state=None,
            num_k_heads=nk,
            num_v_heads=nv,
            head_k_dim=_HEAD_DIM,
            head_v_dim=_HEAD_DIM,
            conv_kernel_size=_KERNEL,
            chunk_size=_CHUNK,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
            segment_ids=seg,
            conv_input_in_qkv_dtype=True,
            conv_dense_token_stack=True,
            pass1_states_and_bf16_qkv=opt_on,
        )
        return (
            jnp.sum(out.astype(jnp.float32) * do)
            + 0.1 * jnp.sum(ncs.astype(jnp.float32))
            + 0.2 * jnp.sum(nrs.astype(jnp.float32)),
            (out, ncs, nrs),
        )

      return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5), has_aux=True))(
          qkv, b, a, cw, a_log, dtb
      )

    (loss_off, aux_off), grads_off = run(False)
    (loss_on, aux_on), grads_on = run(True)
    self.assertEqual(_bits(loss_on), _bits(loss_off))
    for x_on, x_off in zip(aux_on, aux_off):
      self.assertEqual(float(jnp.max(jnp.abs(x_on - x_off))), 0.0)
      self.assertEqual(_bits(x_on), _bits(x_off))
    for g_on, g_off in zip(grads_on, grads_off):
      self.assertEqual(float(jnp.max(jnp.abs(g_on - g_off))), 0.0)
      self.assertEqual(_bits(g_on), _bits(g_off))

  @parameterized.named_parameters(
      ("no_seg", False),
      ("with_seg", True),
  )
  def test_cp_decoupled_fwd_bwd_bit_identical_and_matches_single_rank(self, with_seg):
    """2-rank CP pass 1 + pass 2: opt_on=True is bit-identical to opt_on=False and matches unsharded reference."""
    nk, nv, cp_ranks = 2, 4, 2
    seq = cp_ranks * _CHUNK
    dim = (2 * nk + nv) * _HEAD_DIM
    rng = np.random.default_rng(177 + int(with_seg))
    qkv = jnp.asarray((rng.standard_normal((1, seq, dim)) * 0.2).astype(np.float32)).astype(jnp.bfloat16)
    b = jnp.asarray(rng.standard_normal((1, seq, nv)).astype(np.float32)).astype(jnp.bfloat16)
    a = jnp.asarray(rng.standard_normal((1, seq, nv)).astype(np.float32)).astype(jnp.bfloat16)
    cw = jnp.asarray((rng.standard_normal((_KERNEL, 1, dim)) * 0.2).astype(np.float32))
    a_log = jnp.asarray((rng.standard_normal((nv,)) * 0.1).astype(np.float32))
    # Weak decay so Pass-1 conv_halo and zero_rs on rank 0/1 propagate strongly to s_ext_local and s_in_r.
    dtb = jnp.asarray((rng.standard_normal((nv,)) * 0.1 - 6.0).astype(np.float32))
    cs = jnp.asarray((rng.standard_normal((1, _KERNEL - 1, dim)) * 0.3).astype(np.float32)).astype(jnp.bfloat16)
    rs = jnp.asarray((rng.standard_normal((1, nv, _HEAD_DIM, _HEAD_DIM)) * 0.3).astype(np.float32))
    do = jnp.asarray(rng.standard_normal((1, seq, nv, _HEAD_DIM)).astype(np.float32))
    seg = jnp.ones((1, seq), dtype=jnp.int32) if with_seg else None

    def call_gdn(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in, cs_in, rs_in, seg_in, cp_axis, opt_on):
      return gdn_api.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=None,
          a_log=alog_in,
          dt_bias=dtb_in,
          conv_state=cs_in,
          recurrent_state=rs_in,
          num_k_heads=nk,
          num_v_heads=nv,
          head_k_dim=_HEAD_DIM,
          head_v_dim=_HEAD_DIM,
          conv_kernel_size=_KERNEL,
          chunk_size=_CHUNK,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
          cp_axis_name=cp_axis,
          segment_ids=seg_in,
          conv_input_in_qkv_dtype=True,
          conv_dense_token_stack=True,
          pass1_states_and_bf16_qkv=opt_on,
      )

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:cp_ranks]), ("cp",))
    ps_seq = jax.sharding.PartitionSpec(None, "cp", None)
    ps_seg = jax.sharding.PartitionSpec(None, "cp")
    ps_rep = jax.sharding.PartitionSpec()
    ps_out = jax.sharding.PartitionSpec(None, "cp", None, None)

    def run_cp(opt_on):
      if seg is not None:

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(ps_seq, ps_seq, ps_seq, ps_rep, ps_rep, ps_rep, ps_rep, ps_rep, ps_seg),
            out_specs=(ps_out, ps_rep, ps_rep),
            check_vma=False,
        )
        def cp_step(qkv_s, b_s, a_s, cw_in, alog_in, dtb_in, cs_in, rs_in, seg_s):
          out_s, (ncs_s, nrs_s) = call_gdn(
              qkv_s, b_s, a_s, cw_in, alog_in, dtb_in, cs_in, rs_in, seg_s, "cp", opt_on
          )
          return out_s, ncs_s, nrs_s

      else:

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(ps_seq, ps_seq, ps_seq, ps_rep, ps_rep, ps_rep, ps_rep, ps_rep),
            out_specs=(ps_out, ps_rep, ps_rep),
            check_vma=False,
        )
        def cp_step(qkv_s, b_s, a_s, cw_in, alog_in, dtb_in, cs_in, rs_in):
          out_s, (ncs_s, nrs_s) = call_gdn(
              qkv_s, b_s, a_s, cw_in, alog_in, dtb_in, cs_in, rs_in, None, "cp", opt_on
          )
          return out_s, ncs_s, nrs_s

      def loss_fn(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in, cs_in, rs_in):
        if seg is not None:
          out, ncs, nrs = cp_step(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in, cs_in, rs_in, seg)
        else:
          out, ncs, nrs = cp_step(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in, cs_in, rs_in)
        return (
            jnp.sum(out.astype(jnp.float32) * do)
            + 0.1 * jnp.sum(ncs.astype(jnp.float32))
            + 0.2 * jnp.sum(nrs.astype(jnp.float32)),
            (out, ncs, nrs),
        )

      return jax.jit(jax.value_and_grad(loss_fn, argnums=tuple(range(8)), has_aux=True))(
          qkv, b, a, cw, a_log, dtb, cs, rs
      )

    def run_single():
      def loss_fn(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in, cs_in, rs_in):
        out, (ncs, nrs) = call_gdn(qkv_in, b_in, a_in, cw_in, alog_in, dtb_in, cs_in, rs_in, seg, None, False)
        return (
            jnp.sum(out.astype(jnp.float32) * do)
            + 0.1 * jnp.sum(ncs.astype(jnp.float32))
            + 0.2 * jnp.sum(nrs.astype(jnp.float32)),
            (out, ncs, nrs),
        )

      return jax.jit(jax.value_and_grad(loss_fn, argnums=tuple(range(8)), has_aux=True))(
          qkv, b, a, cw, a_log, dtb, cs, rs
      )

    (loss_off, aux_off), grads_off = run_cp(False)
    (loss_on, aux_on), grads_on = run_cp(True)
    (_, aux_ref), grads_ref = run_single()

    # 1. opt_on=True vs opt_on=False on 2-rank CP must be bit-identical (catches T-M6, V-M6, T-M7).
    self.assertEqual(_bits(loss_on), _bits(loss_off))
    for x_on, x_off in zip(aux_on, aux_off):
      self.assertEqual(float(jnp.max(jnp.abs(x_on - x_off))), 0.0)
      self.assertEqual(_bits(x_on), _bits(x_off))
    for g_on, g_off in zip(grads_on, grads_off):
      self.assertEqual(float(jnp.max(jnp.abs(g_on - g_off))), 0.0)
      self.assertEqual(_bits(g_on), _bits(g_off))

    # 2. 2-rank CP vs single-rank unsharded reference must match numerically (catches T-M7 even if both branches match).
    for x_cp, x_ref in zip(aux_off, aux_ref):
      np.testing.assert_allclose(np.asarray(x_cp, np.float32), np.asarray(x_ref, np.float32), rtol=1e-2, atol=1e-2)
    for g_cp, g_ref in zip(grads_off, grads_ref):
      np.testing.assert_allclose(np.asarray(g_cp, np.float32), np.asarray(g_ref, np.float32), rtol=2e-2, atol=2e-2)


def _validator_cfg(**overrides):
  fields = {
      "decoder_block": config_types.DecoderBlockType.QWEN3_5,
      "ici_context_parallelism": 1,
      "dcn_context_parallelism": 1,
      "ici_context_usp_ulysses_parallelism": 1,
      "dcn_context_usp_ulysses_parallelism": 1,
      "use_gdn_kernel": True,
      "ici_tensor_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "gdn_chunk_size": 64,
      "gdn_key_head_dim": 128,
      "gdn_value_head_dim": 128,
      "gdn_num_key_heads": 16,
      "gdn_num_value_heads": 64,
      "use_qk_norm_in_gdn": True,
      "max_target_length": 4096,
      "gdn_conv1d_input_in_qkv_dtype": False,
      "gdn_conv1d_dense_token_stack": False,
      "gdn_out_proj_token_minor": False,
      "gdn_gated_norm_layout": "auto",
      "gdn_cp_pass1_states_and_bf16_qkv": False,
      "gdn_cp_mode": "auto",
      "packing": True,
      "dtype": "bfloat16",
      "gdn_in_proj_kernel_order": False,
  }
  fields.update(overrides)
  return types.SimpleNamespace(**fields)


class ValidatorTest(parameterized.TestCase):
  """validate_gdn_config and pyconfig defaults for gdn_out_proj_token_minor and gdn_cp_pass1_states_and_bf16_qkv."""

  def test_defaults_and_pyconfig(self):
    def cfg(**kwargs):
      return pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="gdn_pass1_states_and_out_proj",
          enable_checkpointing=False,
          log_config=False,
          skip_jax_distributed_system=True,
          **kwargs,
      )

    default_cfg = cfg()
    self.assertFalse(default_cfg.gdn_out_proj_token_minor)
    self.assertFalse(default_cfg.gdn_cp_pass1_states_and_bf16_qkv)
    self.assertFalse(config_types.Qwen3Next().gdn_out_proj_token_minor)
    self.assertFalse(config_types.Qwen3Next().gdn_cp_pass1_states_and_bf16_qkv)
    on = cfg(
        model_name="qwen3.5-397b-a17b",
        use_gdn_kernel=True,
        gdn_out_proj_token_minor=True,
        gdn_cp_pass1_states_and_bf16_qkv=True,
    )
    self.assertTrue(on.gdn_out_proj_token_minor)
    self.assertTrue(on.gdn_cp_pass1_states_and_bf16_qkv)

  def test_rejected_configs(self):
    with self.assertRaisesRegex(ValueError, "gdn_out_proj_token_minor=True requires a decoder with Gated DeltaNet"):
      config_types.MaxTextConfig.validate_gdn_config(
          _validator_cfg(decoder_block=config_types.DecoderBlockType.LLAMA2, gdn_out_proj_token_minor=True),
          is_rl_config=False,
      )
    with self.assertRaisesRegex(
        ValueError, "gdn_cp_pass1_states_and_bf16_qkv=True requires a decoder with Gated DeltaNet"
    ):
      config_types.MaxTextConfig.validate_gdn_config(
          _validator_cfg(decoder_block=config_types.DecoderBlockType.LLAMA2, gdn_cp_pass1_states_and_bf16_qkv=True),
          is_rl_config=False,
      )
    with self.assertRaisesRegex(ValueError, "gdn_cp_pass1_states_and_bf16_qkv=True requires use_gdn_kernel=True"):
      config_types.MaxTextConfig.validate_gdn_config(
          _validator_cfg(use_gdn_kernel=False, gdn_cp_pass1_states_and_bf16_qkv=True),
          is_rl_config=False,
      )
    with self.assertRaisesRegex(ValueError, "gdn_cp_pass1_states_and_bf16_qkv=True requires gdn_chunk_size=64"):
      config_types.MaxTextConfig.validate_gdn_config(
          _validator_cfg(gdn_chunk_size=128, gdn_cp_pass1_states_and_bf16_qkv=True),
          is_rl_config=False,
      )
    with self.assertRaisesRegex(ValueError, "gdn_cp_pass1_states_and_bf16_qkv=True requires dtype='bfloat16'"):
      config_types.MaxTextConfig.validate_gdn_config(
          _validator_cfg(dtype="float32", gdn_cp_pass1_states_and_bf16_qkv=True),
          is_rl_config=False,
      )


@pytest.mark.cpu_only
def test_tpu_programs_in_a_child_process():
  """Runs `main` below in a fresh CPU process (the program checks trace TPU7x kernels on this host)."""
  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


# ----------------------------------------------------------------------------------------------------------------------
# TPU7x program checks (main).


def _tpu_target(shape=(), axes=()):
  device = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
  mesh = jax.sharding.AbstractMesh(
      shape, axes, axis_types=(jax.sharding.AxisType.Auto,) * len(axes), abstract_device=device
  )
  return jax.sharding.use_abstract_mesh(mesh), mesh


def _forward_kernel_calls(closed_jaxpr):
  """List of (kernel_name, qkv_in_aval_dtype, qkv_in_aval_shape) for every fused_conv1d_gdn* pallas_call."""
  found = []

  def visit(jaxpr):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "pallas_call" and str(eqn.params.get("name", "")).startswith("fused_conv1d_gdn"):
        # PerSeqMetadata has 4 leaves, so invars[4] is qkv (shape (..., 12288)).
        qkv_var = [v for v in eqn.invars if len(v.aval.shape) == 3 and v.aval.shape[-1] == 12288][0]
        found.append((str(eqn.params["name"]), qkv_var.aval.dtype, tuple(qkv_var.aval.shape)))
        continue
      for sub in jax_core.jaxprs_in_params(eqn.params):
        visit(sub)

  visit(closed_jaxpr.jaxpr)
  return found


class _Value:

  def __init__(self, value):
    self.value = value

  def __getitem__(self, idx):
    return self.value[idx]


def _trace_model_runner(mode, nk, nv, *, opt_on, remat=True, seg=True, seq=512, grad=True):
  cp = {"none": 1, "seq4": 4, "head2": 2}[mode]
  if mode == "none":
    ctx, mesh = _tpu_target()
    layer_mesh = None
  else:
    ctx, mesh = _tpu_target((cp,), ("context",))
    layer_mesh = mesh
  cfg = types.SimpleNamespace(
      ici_context_parallelism=cp,
      dcn_context_parallelism=1,
      ici_context_usp_ulysses_parallelism=1,
      dcn_context_usp_ulysses_parallelism=1,
      gdn_cp_mode={"head2": "head", "seq4": "seq"}.get(mode, "auto"),
      gdn_cp_matmul_precision="highest",
      shard_mode=ShardMode.AUTO,
      logical_axis_rules=[],
      gdn_conv_kernel_dim=_KERNEL,
      gdn_chunk_size=_CHUNK,
      use_qk_norm_in_gdn=True,
      context_parallel_load_balance=False,
      gdn_conv1d_input_in_qkv_dtype=True,
      gdn_conv1d_dense_token_stack=True,
      gdn_cp_pass1_states_and_bf16_qkv=opt_on,
      gdn_bwd_natural_layout="off",
      gdn_conv1d_natural_layout=False,
      gdn_fwd_kernel_head_group_size=0,
  )
  key_dim, conv_dim = nk * _HEAD_DIM, (2 * nk + nv) * _HEAD_DIM

  def loss(params, x, aux):
    def body(params, x):
      layer = types.SimpleNamespace(
          config=cfg,
          mesh=layer_mesh,
          num_k_heads=nk,
          num_v_heads=nv,
          head_k_dim=_HEAD_DIM,
          head_v_dim=_HEAD_DIM,
          key_dim=key_dim,
          v_heads_per_k_head=nv // nk,
          conv1d=types.SimpleNamespace(kernel=_Value(params["conv_w"]), bias=None),
          A_log=_Value(params["a_log"]),
          dt_bias=_Value(params["dt_bias"]),
      )
      out, next_cs, next_rs = model_runner.run_gdn_kernel_layer(
          layer=layer,
          query=x["query"],
          key=x["key"],
          value=x["value_raw"].reshape(1, seq, nv, _HEAD_DIM),
          value_raw=x["value_raw"],
          b=x["b"],
          a=x["a"],
          active_cache=None,
          decoder_segment_ids=aux["seg"] if seg else None,
          model_mode="train",
          state_dtype=jnp.float32,
      )
      return (
          jnp.sum(out.astype(jnp.float32) * aux["do"])
          + 0.5 * jnp.sum(next_cs.astype(jnp.float32))
          + 0.25 * jnp.sum(next_rs.astype(jnp.float32))
      )

    if remat:
      body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
    return body(params, x)

  a = jax.ShapeDtypeStruct
  params = {
      "conv_w": a((_KERNEL, 1, conv_dim), jnp.float32),
      "a_log": a((nv,), jnp.float32),
      "dt_bias": a((nv,), jnp.float32),
  }
  x = {
      "query": a((1, seq, nk, _HEAD_DIM), jnp.bfloat16),
      "key": a((1, seq, nk, _HEAD_DIM), jnp.bfloat16),
      "value_raw": a((1, seq, nk, (nv // nk) * _HEAD_DIM), jnp.bfloat16),
      "b": a((1, seq, nv), jnp.bfloat16),
      "a": a((1, seq, nv), jnp.bfloat16),
  }
  aux = {"seg": a((1, seq), jnp.int32), "do": a((1, seq, nv, _HEAD_DIM), jnp.float32)}
  fn = jax.value_and_grad(loss, argnums=(0, 1)) if grad else loss
  with ctx:
    return jax.make_jaxpr(fn)(params, x, aux)


def _check(label, ok, detail=""):
  print(f"{label}: [{'ok' if ok else 'FAIL'}] {detail}", flush=True)
  return ok


def _check_seq_cp_kernel_calls(opt_on, grad=True):
  jaxpr = _trace_model_runner("seq4", 16, 64, opt_on=opt_on, seq=256, grad=grad)
  calls = _forward_kernel_calls(jaxpr)
  names = [n for n, _, _ in calls]
  dtypes = [dt for _, dt, _ in calls]
  shapes = [sh for _, _, sh in calls]
  repeats = 2 if grad else 1
  if opt_on:
    expected_names = ["fused_conv1d_gdn_per_seq_states_b1_c64", "fused_conv1d_gdn_per_seq_b1_c64"] * repeats
    expected_dtypes = [jnp.dtype(jnp.bfloat16)] * (2 * repeats)
    expected_shapes = [(1, 64, 12288)] * (2 * repeats)
  else:
    expected_names = ["fused_conv1d_gdn_per_seq_b1_c64", "fused_conv1d_gdn_per_seq_b1_c64"] * repeats
    expected_dtypes = [jnp.dtype(jnp.float32)] * (2 * repeats)
    expected_shapes = [(64, 1, 12288)] * (2 * repeats)
  good = names == expected_names and dtypes == expected_dtypes and shapes == expected_shapes
  return good, f"names={names} dtypes={dtypes} shapes={shapes}"


def _lowers_states_and_per_seq_to_mosaic():
  ok = True
  for nk, nv in ((2, 4), (16, 64)):
    dim = (2 * nk + nv) * _HEAD_DIM
    a = jax.ShapeDtypeStruct
    args = (
        a((128, dim), jnp.bfloat16),
        a((128, nv), jnp.bfloat16),
        a((128, nv), jnp.bfloat16),
        a((2, _KERNEL - 1, dim), jnp.bfloat16),
        a((2, nv, _HEAD_DIM, _HEAD_DIM), jnp.float32),
        a((dim, 1, _KERNEL), jnp.float32),
        None,
        a((nv,), jnp.float32),
        a((nv,), jnp.float32),
        a((2,), jnp.int32),
        a((1,), jnp.int32),
        a((3,), jnp.int32),
        a((1,), jnp.int32),
    )
    ctx, _ = _tpu_target()
    fn_states = functools.partial(
        wrapper.fused_conv1d_gdn_states,
        n_kq=nk,
        n_v=nv,
        d_k=_HEAD_DIM,
        d_v=_HEAD_DIM,
        kernel_size=_KERNEL,
        compute_precision=jnp.dtype(jnp.float32),
        mixed_tile_size=_CHUNK,
        conv_dense_token_stack=True,
        bf16_qkv_dma=True,
    )
    fn_full = functools.partial(
        wrapper.fused_conv1d_gdn,
        n_kq=nk,
        n_v=nv,
        d_k=_HEAD_DIM,
        d_v=_HEAD_DIM,
        kernel_size=_KERNEL,
        compute_precision=jnp.dtype(jnp.float32),
        mixed_tile_size=_CHUNK,
        is_prefill_only=True,
        conv_dense_token_stack=True,
        bf16_qkv_dma=True,
    )
    with ctx:
      text_states = jax.jit(fn_states).trace(*args).lower(lowering_platforms=("tpu",)).as_text(debug_info=False)
      text_full = jax.jit(fn_full).trace(*args).lower(lowering_platforms=("tpu",)).as_text(debug_info=False)
    names_states = re.findall(r'kernel_name = "([^"]+)"', text_states)
    names_full = re.findall(r'kernel_name = "([^"]+)"', text_full)
    good = (
        names_states == ["fused_conv1d_gdn_per_seq_states_b1_c64"]
        and names_full == ["fused_conv1d_gdn_per_seq_b1_c64"]
    )
    ok &= _check(f"mosaic lowering width {dim}", good, f"states={names_states} full={names_full}")
  return ok


def _backward_rule_positional_forms():
  nk, nv, seq = 2, 4, 128
  dim = (2 * nk + nv) * _HEAD_DIM
  nondiff = (nk, nv, _HEAD_DIM, _HEAD_DIM, _KERNEL, _CHUNK, True, jnp.float32)
  prec = cp_gdn.DEFAULT_PRECISION
  forms = {
      10: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, res, cot),
      11: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, res, cot),
      12: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, res, cot),
      13: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, False, res, cot),
      14: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, False, True, res, cot),
      15: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, False, True, True, res, cot),
      16: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, False, True, True, False, res, cot),
      17: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(
          *nondiff, None, prec, False, True, True, False, True, res, cot
      ),
      18: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(
          *nondiff, None, prec, False, True, True, False, True, False, res, cot
      ),
      19: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(
          *nondiff, None, prec, False, True, True, False, True, False, False, res, cot
      ),
  }

  def program(form):
    def f(qkv, b, a, cw, a_log, dtb, do):
      _, res = gdn_api._gdn_decoupled_conv1d_fwd(qkv, b, a, cw, None, a_log, dtb, None, None, *nondiff)
      return forms[form](res, (do, (None, None)))

    a = jax.ShapeDtypeStruct
    args = (
        a((1, seq, dim), jnp.bfloat16),
        a((1, seq, nv), jnp.bfloat16),
        a((1, seq, nv), jnp.bfloat16),
        a((_KERNEL, 1, dim), jnp.float32),
        a((nv,), jnp.float32),
        a((nv,), jnp.float32),
        a((1, seq, nv, _HEAD_DIM), jnp.float32),
    )
    ctx, _ = _tpu_target()
    with ctx:
      return str(jax.make_jaxpr(f)(*args))

  ref = program(19)
  ok = True
  for form in (10, 11, 12, 13, 14, 15, 16, 17, 18):
    same = program(form) == ref
    ok &= _check(f"backward rule, {form} positional arguments", same)
  return ok


def _pass1_arg_index():
  """Position of pass1_states_and_bf16_qkv among gdn_decoupled_conv1d's arguments.

  The negative controls below overwrite that one argument. Later keys append arguments after it (for example
  cp_fast_state_passing), so it is found by name, not taken to be the last.
  """
  return list(inspect.signature(gdn_api.gdn_decoupled_conv1d.fun).parameters).index("pass1_states_and_bf16_qkv")


def main():
  ok = True
  for opt_on in (False, True):
    for grad in (False, True):
      good, detail = _check_seq_cp_kernel_calls(opt_on, grad=grad)
      ok &= _check(f"seq_cp4 opt_on={opt_on} grad={grad}", good, detail)
  ok &= _lowers_states_and_per_seq_to_mosaic()
  ok &= _backward_rule_positional_forms()

  # Negative control: dropping pass1_states_and_bf16_qkv in custom_vjp forward rule MUST fail the opt_on=True check.
  real_fwd = gdn_api.gdn_decoupled_conv1d.fwd

  pass1_arg = _pass1_arg_index()

  def drop_fwd(*args):
    args = list(args)
    args[pass1_arg] = False  # pass1_states_and_bf16_qkv
    return real_fwd(*args)

  jax.clear_caches()
  gdn_api.gdn_decoupled_conv1d.fwd = drop_fwd
  try:
    good_dropped, detail_dropped = _check_seq_cp_kernel_calls(True, grad=True)
  finally:
    gdn_api.gdn_decoupled_conv1d.fwd = real_fwd
    jax.clear_caches()
  ok &= _check("control_dropped_by_forward_rule", not good_dropped, detail_dropped)

  # Negative control: dropping pass1_states_and_bf16_qkv in custom_vjp primal MUST fail the forward-only check.
  real_fun = gdn_api.gdn_decoupled_conv1d.fun

  @functools.wraps(real_fun)
  def drop_fun(*args):
    args = list(args)
    args[pass1_arg] = False  # pass1_states_and_bf16_qkv
    return real_fun(*args)

  jax.clear_caches()
  gdn_api.gdn_decoupled_conv1d.fun = drop_fun
  try:
    good_primal_dropped, detail_primal_dropped = _check_seq_cp_kernel_calls(True, grad=False)
  finally:
    gdn_api.gdn_decoupled_conv1d.fun = real_fun
    jax.clear_caches()
  ok &= _check("control_dropped_by_primal", not good_primal_dropped, detail_primal_dropped)

  if not ok:
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
