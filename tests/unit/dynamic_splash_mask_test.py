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

"""Tests for the COMPRESSED / LOCAL_SLIDING dynamic-splash boolean masks."""

import math
import types
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import (
    AttentionType,
    DEFAULT_MASK_VALUE,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.layers.attention_op import (
    AttentionOp,
    build_compressed_splash_mask,
    build_local_sliding_splash_mask,
)


def _stub_config(compressed_use_dynamic_splash=False, block=8):
  """A minimal stand-in for the dynamic-splash configuration."""
  kernel_fields = {
      f"sa_block_{n}": block for n in ("q", "kv", "kv_compute", "q_dkv", "kv_dkv", "kv_dkv_compute", "q_dq", "kv_dq")
  }
  kernel_fields.update({f"sa_{n}_layout": "HEAD_DIM_MINOR" for n in ("q", "k", "v")})
  kernel_fields.update(sa_use_fused_bwd_kernel=True, sa_fuse_reciprocal=False, sa_use_base2_exp=False)
  kernel_fields["use_splash_scheduler"] = False
  kernel_fields.update({f"local_{name}": value for name, value in kernel_fields.items()})
  return types.SimpleNamespace(
      compressed_use_dynamic_splash=compressed_use_dynamic_splash,
      context_parallel_load_balance=False,
      context_parallel_strategy="all_gather",
      context_sharding="context",
      **kernel_fields,
  )


def _make_op(attention_type, sliding_window_size, *, config=None, platform="cpu", context_parallel_size=1):
  """Builds an AttentionOp whose only job is mask generation / dispatch."""
  device = types.SimpleNamespace(platform=platform)
  mesh = types.SimpleNamespace(
      devices=np.asarray([device], dtype=object),
      shape={"context": context_parallel_size},
  )
  return AttentionOp(
      config=config if config is not None else _stub_config(),
      num_query_heads=1,
      num_kv_heads=1,
      max_target_length=32,
      mesh=mesh,
      attention_kernel="dot_product",
      attention_type=attention_type,
      sliding_window_size=sliding_window_size,
  )


def _dense_keep_set(additive_mask):
  """Converts a dense additive mask to its keep set."""
  return np.asarray(additive_mask)[:, 0, 0, :, :] >= DEFAULT_MASK_VALUE * 0.5


def _packing(seq_len, batch, packed):
  """Returns segment ids and per-segment positions, which reset under packing."""
  if not packed:
    return None, jnp.asarray(np.arange(seq_len)[None, :]).repeat(batch, axis=0)
  segment_len = seq_len // 2
  segment_ids = jnp.asarray(np.arange(seq_len)[None, :] // segment_len + 1).repeat(batch, axis=0)
  reset_positions = jnp.asarray((np.arange(seq_len) % segment_len)[None, :]).repeat(batch, axis=0)
  return segment_ids, reset_positions


def _additive_lattice(rng, shape):
  """Builds a compressed-block mask, including the keep predicate's boundary."""
  choices = np.asarray([0.0, DEFAULT_MASK_VALUE * 0.5, DEFAULT_MASK_VALUE, -np.inf], dtype=np.float32)
  return jnp.asarray(choices[rng.integers(0, len(choices), size=shape)])


class DynamicSplashMaskParityTest(parameterized.TestCase):
  """The boolean splash mask must keep exactly what the dense additive mask keeps."""

  @pytest.mark.cpu_only
  @parameterized.product(
      seq_len=(16, 32),
      sliding_window_size=(4, 16),
      packed=(False, True),
  )
  def test_local_sliding_mask_matches_dense(self, seq_len, sliding_window_size, packed):
    batch = 2
    segment_ids, _ = _packing(seq_len, batch, packed)
    op = _make_op(AttentionType.LOCAL_SLIDING, sliding_window_size)
    dense = op.generate_attention_mask(
        jnp.zeros((batch, seq_len, 1, 1)),
        jnp.zeros((batch, seq_len, 1, 1)),
        segment_ids,
        MODEL_MODE_TRAIN,
    )
    expected = np.broadcast_to(_dense_keep_set(dense), (batch, seq_len, seq_len))
    actual = build_local_sliding_splash_mask(batch, segment_ids, None, seq_len, seq_len, sliding_window_size)
    np.testing.assert_array_equal(np.asarray(actual), expected)

  @pytest.mark.cpu_only
  @parameterized.product(
      seq_len=(16, 32),
      sliding_window_size=(4, 16, None),
      packed=(False, True),
      compress_ratio=(4, 8),
      positions=("none", "physical", "reset"),
  )
  def test_compressed_mask_matches_dense(self, seq_len, sliding_window_size, packed, compress_ratio, positions):
    batch = 2
    compressed_len = seq_len // compress_ratio
    kv_seq_len = seq_len + compressed_len
    segment_ids, reset_positions = _packing(seq_len, batch, packed)
    segment_positions = {
        "none": None,
        "physical": jnp.asarray(np.arange(seq_len)[None, :]).repeat(batch, axis=0),
        "reset": reset_positions,
    }[positions]
    compressed_mask = _additive_lattice(np.random.default_rng(0), (batch, 1, 1, seq_len, compressed_len))
    op = _make_op(AttentionType.COMPRESSED, sliding_window_size)
    dense = op.generate_attention_mask(
        jnp.zeros((batch, seq_len, 1, 1)),
        jnp.zeros((batch, kv_seq_len, 1, 1)),
        segment_ids,
        MODEL_MODE_TRAIN,
        compressed_mask=compressed_mask,
        segment_positions=segment_positions,
    )
    expected = np.broadcast_to(_dense_keep_set(dense), (batch, seq_len, kv_seq_len))
    actual = build_compressed_splash_mask(
        compressed_mask, segment_ids, segment_positions, seq_len, kv_seq_len, sliding_window_size
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)

  @pytest.mark.cpu_only
  def test_packed_reset_positions_retain_no_uncompressed_columns(self):
    """Pins the dense path's reset-query/physical-key asymmetry under packing."""
    batch, seq_len, window, compressed_len = 1, 16, 4, 4
    kv_seq_len = seq_len + compressed_len
    segment_ids, reset_positions = _packing(seq_len, batch, packed=True)
    compressed_mask = jnp.zeros((batch, 1, 1, seq_len, compressed_len), dtype=jnp.float32)
    op = _make_op(AttentionType.COMPRESSED, window)
    dense = op.generate_attention_mask(
        jnp.zeros((batch, seq_len, 1, 1)),
        jnp.zeros((batch, kv_seq_len, 1, 1)),
        segment_ids,
        MODEL_MODE_TRAIN,
        compressed_mask=compressed_mask,
        segment_positions=reset_positions,
    )
    keep = np.broadcast_to(_dense_keep_set(dense), (batch, seq_len, kv_seq_len))
    self.assertEqual(keep[:, seq_len // 2 :, :seq_len].sum(), 0)
    self.assertEqual(keep[:, : seq_len // 2, :seq_len].sum(), 26)
    actual = build_compressed_splash_mask(compressed_mask, segment_ids, reset_positions, seq_len, kv_seq_len, window)
    np.testing.assert_array_equal(np.asarray(actual), keep)


class DynamicSplashAttentionTest(unittest.TestCase):
  """Checks what dynamic_splash_attention hands to the kernel, without running the kernel."""

  def _capture(self, op, batch, q_seq_len, kv_seq_len, compressed_mask, decoder_segment_ids=None, segment_positions=None):
    """Runs dynamic_splash_attention with a stubbed kernel and returns the captured call kwargs."""
    captured = {}

    def fake_tpu_flash_attention(query, key, value, **kwargs):
      captured.update(kwargs, key=key, value=value)
      return jnp.zeros_like(query), None

    with mock.patch.object(op, "tpu_flash_attention", fake_tpu_flash_attention):
      out, exp_max, exp_sum = op.dynamic_splash_attention(
          jnp.zeros((batch, q_seq_len, 1, 4)),
          jnp.zeros((batch, kv_seq_len, 1, 4)),
          jnp.zeros((batch, kv_seq_len, 1, 4)),
          decoder_segment_ids,
          segment_positions,
          compressed_mask,
          None,
      )
    self.assertIsNone(exp_max)
    self.assertIsNone(exp_sum)
    self.assertEqual(out.shape, (batch, q_seq_len, 1, 4))
    return captured

  @pytest.mark.cpu_only
  def test_kernel_call_pads_kv_folds_packing_into_the_mask_and_guards_its_assumptions(self):
    batch, seq_len, compressed_len, block = 2, 32, 8, 16
    kv_seq_len = seq_len + compressed_len
    segment_ids, reset_positions = _packing(seq_len, batch, packed=True)
    op = _make_op(AttentionType.COMPRESSED, 4, config=_stub_config(True, block=block))
    compressed_mask = _additive_lattice(np.random.default_rng(1), (batch, 1, 1, seq_len, compressed_len))
    captured = self._capture(op, batch, seq_len, kv_seq_len, compressed_mask, segment_ids, reset_positions)

    self.assertIsNone(captured["decoder_segment_ids"])
    padded_kv_len = math.ceil(kv_seq_len / block) * block
    self.assertGreater(padded_kv_len, kv_seq_len)
    mask = np.asarray(captured["indexer_mask"])
    self.assertEqual(mask.dtype, np.bool_)
    self.assertEqual(mask.shape, (batch, seq_len, padded_kv_len))
    self.assertEqual(captured["key"].shape[1], padded_kv_len)
    self.assertEqual(captured["value"].shape[1], padded_kv_len)
    self.assertFalse(mask[:, :, kv_seq_len:].any())
    unpadded = build_compressed_splash_mask(compressed_mask, segment_ids, reset_positions, seq_len, kv_seq_len, 4)
    np.testing.assert_array_equal(mask[:, :, :kv_seq_len], np.asarray(unpadded))

    small = _additive_lattice(np.random.default_rng(2), (1, 1, 1, 24, compressed_len))
    with self.assertRaisesRegex(ValueError, "must be a multiple of"):
      self._capture(op, 1, 24, 24 + compressed_len, small)
    cp_op = _make_op(AttentionType.COMPRESSED, 4, config=_stub_config(True, block=block), context_parallel_size=2)
    with self.assertRaisesRegex(NotImplementedError, "context parallelism"):
      self._capture(cp_op, batch, seq_len, kv_seq_len, compressed_mask)


class DynamicSplashRoutingTest(parameterized.TestCase):
  """Tests dynamic, static, and dense routing."""

  def _dispatch(
      self,
      *,
      flag,
      model_mode,
      platform,
      attention_type,
      compressed_mask,
      previous_chunk=None,
      bidirectional_mask=None,
  ):
    """Returns 'dynamic' (dynamic-mask splash), 'static' (static splash) or 'dot' (dense path)."""
    op = _make_op(attention_type, 4, config=_stub_config(flag), platform=platform)
    with (
        mock.patch.object(op, "dynamic_splash_attention", return_value=(None, None, None)) as dynamic,
        mock.patch.object(op, "tpu_flash_attention", return_value=(None, None)) as static,
        mock.patch.object(op, "apply_attention_dot", return_value=(None, None, None)) as dot,
    ):
      op.apply_attention(
          jnp.zeros((1, 32, 1, 4)),
          jnp.zeros((1, 32, 1, 4)),
          jnp.zeros((1, 32, 1, 4)),
          None,
          None,
          None,
          model_mode,
          previous_chunk=previous_chunk,
          bidirectional_mask=bidirectional_mask,
          compressed_mask=compressed_mask,
          qk_product_einsum=jnp.einsum,
          wv_product_einsum=jnp.einsum,
      )
    self.assertEqual(dynamic.called + static.called + dot.called, 1)
    if dynamic.called:
      return "dynamic"
    return "static" if static.called else "dot"

  @pytest.mark.cpu_only
  @parameterized.named_parameters(
      ("compressed", AttentionType.COMPRESSED, "dynamic"),
      ("local_sliding", AttentionType.LOCAL_SLIDING, "static"),
  )
  def test_gate(self, attention_type, enabled_target):
    compressed_mask = jnp.zeros((1, 1, 1, 32, 8)) if attention_type == AttentionType.COMPRESSED else None
    kwargs = {"attention_type": attention_type, "compressed_mask": compressed_mask}
    on_tpu_train = {"flag": True, "model_mode": MODEL_MODE_TRAIN, "platform": "tpu", **kwargs}
    self.assertEqual(self._dispatch(**on_tpu_train), enabled_target)
    self.assertEqual(self._dispatch(flag=False, model_mode=MODEL_MODE_TRAIN, platform="tpu", **kwargs), "dot")
    self.assertEqual(self._dispatch(flag=True, model_mode=MODEL_MODE_PREFILL, platform="tpu", **kwargs), "dot")
    self.assertEqual(self._dispatch(flag=True, model_mode=MODEL_MODE_AUTOREGRESSIVE, platform="tpu", **kwargs), "dot")
    self.assertEqual(self._dispatch(flag=True, model_mode=MODEL_MODE_TRAIN, platform="cpu", **kwargs), "dot")
    self.assertEqual(self._dispatch(previous_chunk=object(), **on_tpu_train), "dot")
    self.assertEqual(self._dispatch(bidirectional_mask=jnp.ones((1, 32), dtype=bool), **on_tpu_train), "dot")


def _v4_test_config(compressed_use_dynamic_splash, *extra_overrides):
  """A deepseek4 config small enough to instantiate in a test, with the flag set either way."""
  from maxtext.configs.pyconfig import initialize  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  overrides = (
      "model_name=deepseek4-284b attention=dot_product qk_rope_head_dim=16 v_head_dim=16 qk_nope_head_dim=16 "
      "use_tokamax_splash=True override_model_config=True "
      f"compressed_use_dynamic_splash={compressed_use_dynamic_splash}"
  ).split()
  return initialize([None, get_test_config_path(), *overrides, *extra_overrides])


class DynamicSplashEndToEndTest(parameterized.TestCase):
  """Dense-vs-splash parity for a whole compressed attention layer (needs the Pallas TPU kernel)."""

  def _layer_output(self, compressed_use_dynamic_splash, compress_ratio, packed):
    """Runs one CompressedAttention layer with the flag set either way and returns its output."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.layers.attention_compressed import CompressedAttention  # pylint: disable=import-outside-toplevel

    seq_len = 128
    layer = CompressedAttention(
        config=_v4_test_config(compressed_use_dynamic_splash),
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=seq_len,
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        attention_kernel="dot_product",
        inputs_q_shape=(1, seq_len, 4096),
        inputs_kv_shape=(1, seq_len, 4096),
        compress_ratio=compress_ratio,
        sliding_window_size=seq_len // 2,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )
    inputs = jax.random.normal(jax.random.PRNGKey(1), (1, seq_len, 4096), dtype=jnp.float32)
    segment_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len)[None, :]
    if packed:
      segment_ids = segment_ids.at[:, seq_len // 2 :].set(2)
      positions = positions % (seq_len // 2)
    return layer(inputs, inputs, segment_ids, positions, deterministic=True)[0]

  @pytest.mark.tpu_only
  @parameterized.product(
      compress_ratio=(0, 4, 128),
      packed=(False, True),
  )
  def test_dense_and_dynamic_splash_agree(self, compress_ratio, packed):
    dense = self._layer_output(False, compress_ratio, packed)
    splash = self._layer_output(True, compress_ratio, packed)
    np.testing.assert_allclose(np.asarray(splash), np.asarray(dense), rtol=2e-2, atol=2e-2)


class DynamicSplashBackwardParityTest(unittest.TestCase):
  """Tests COMPRESSED GQA gradients on TPU.

  Pallas interpret mode does not propagate the kernel's aliased gradient accumulators.
  """

  def _forward_and_grads(self, compressed_use_dynamic_splash, q, k, v, compressed_mask, decoder_segment_ids, cotangent):
    """Returns (out, dq, dk, dv) for one COMPRESSED AttentionOp with the flag set either way."""
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel

    blocks = [f"sa_block_{n}=128" for n in ("q", "kv", "kv_compute", "q_dkv", "kv_dkv", "kv_dkv_compute")]
    op = AttentionOp(
        config=_v4_test_config(compressed_use_dynamic_splash, *blocks),
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        attention_kernel="dot_product",
        max_target_length=q.shape[1],
        num_query_heads=q.shape[2],
        num_kv_heads=k.shape[2],
        attention_type=AttentionType.COMPRESSED,
        sliding_window_size=64,
    )

    def loss_fn(q, k, v):
      out, _, exp_sum = op.apply_attention(
          q,
          k,
          v,
          decoder_segment_ids,
          None,
          None,
          MODEL_MODE_TRAIN,
          compressed_mask=compressed_mask,
          qk_product_einsum=jnp.einsum,
          wv_product_einsum=jnp.einsum,
      )
      if exp_sum is not None:
        out = out / exp_sum
      return jnp.sum(out * cotangent), out

    (_, out), grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(q, k, v)
    return (out, *grads)

  @pytest.mark.tpu_only
  def test_gradients_match_the_dense_path(self):
    batch, q_seq_len, compressed_len, heads, kv_heads, head_dim = 1, 128, 32, 4, 1, 128
    kv_seq_len = q_seq_len + compressed_len
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(keys[0], (batch, q_seq_len, heads, head_dim), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (batch, kv_seq_len, kv_heads, head_dim), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (batch, kv_seq_len, kv_heads, head_dim), dtype=jnp.float32)
    cotangent = jax.random.normal(keys[3], (batch, q_seq_len, heads, head_dim), dtype=jnp.float32)
    block_of = jnp.arange(compressed_len) * (q_seq_len // compressed_len)
    keep = block_of[None, :] <= jnp.arange(q_seq_len)[:, None]
    compressed_mask = jnp.where(keep, 0.0, DEFAULT_MASK_VALUE)[None, None, None, :, :]
    segment_ids = jnp.ones((batch, q_seq_len), dtype=jnp.int32)

    args = (q, k, v, compressed_mask, segment_ids, cotangent)
    dense = self._forward_and_grads(False, *args)
    splash = self._forward_and_grads(True, *args)
    for name, got, want in zip(("out", "dq", "dk", "dv"), splash, dense):
      got = np.asarray(got, dtype=np.float64).ravel()
      want = np.asarray(want, dtype=np.float64).ravel()
      rel_l2 = np.linalg.norm(got - want) / np.linalg.norm(want)
      cosine = np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want))
      self.assertLess(rel_l2, 5e-2, f"{name}: relative L2 distance {rel_l2:.3e}")
      self.assertGreater(cosine, 0.999, f"{name}: cosine similarity {cosine:.6f}")


class DynamicSplashConfigValidatorTest(unittest.TestCase):
  """compressed_use_dynamic_splash without use_tokamax_splash must be rejected at config time."""

  @pytest.mark.cpu_only
  def test_flag_requires_tokamax_splash(self):
    from maxtext.configs.pyconfig import initialize  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    with self.assertRaisesRegex(Exception, "use_tokamax_splash"):
      initialize(
          [None, get_test_config_path()],
          enable_checkpointing=False,
          compressed_use_dynamic_splash=True,
          use_tokamax_splash=False,
      )


if __name__ == "__main__":
  unittest.main()
