# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Wan 2.1 DiT transformer (`transformer_wan.py`)."""

import functools
import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.m3.models.wan import transformer_wan
from maxtext.m3.models.wan import utils_wan
from maxtext.m3.models.wan.transformer_wan import (
    WanAttention,
    WanModel,
    WanRotaryPosEmbed,
    WanTransformerBlock,
    get_sinusoidal_embeddings,
)

# pylint: enable=wrong-import-position

# Tiny Wan 2.1 transformer (14B: 40 layers, 40 heads x 128, ffn 13824, text_dim 4096, 16 latent channels).
_CONFIG = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 16,
    "freq_dim": 16,
    "ffn_dim": 32,
    "num_layers": 2,
    "rope_max_seq_len": 32,
}
_BATCH, _FRAMES, _HEIGHT, _WIDTH, _TEXT_LEN = 2, 3, 8, 8, 7
# Splash blocks for the TPU flash tests (Tokamax `SplashConfig`; fused backward kernel).
_BLOCK_SIZES_128 = utils_wan.BlockSizes(
    block_q=128, block_kv_compute=128, block_kv=128, block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128
)


def _inputs():
  latents = jax.random.normal(jax.random.key(0), (_BATCH, _CONFIG["in_channels"], _FRAMES, _HEIGHT, _WIDTH))
  timestep = jnp.array([10.0, 500.0])
  text = jax.random.normal(jax.random.key(1), (_BATCH, _TEXT_LEN, _CONFIG["text_dim"]))
  return latents, timestep, text


def _single_device_mesh():
  """The blocks constrain activations to the Wan mesh axes; a 1-device mesh makes those constraints no-ops."""
  axes = ("data", "fsdp", "context", "tensor")  # `mesh_axes` of wan2.1-t2v-14b.yml
  return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(axes)), axes)


class WanModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jax.set_mesh(_single_device_mesh()))

  @parameterized.named_parameters(("scanned", True), ("unscanned", False))
  def test_output_has_latent_shape(self, scan_layers):
    model = WanModel(rngs=nnx.Rngs(0), scan_layers=scan_layers, **_CONFIG)
    latents, timestep, text = _inputs()
    out = model(latents, timestep, text)
    self.assertEqual(out.shape, latents.shape)
    self.assertTrue(np.all(np.isfinite(np.asarray(out))))

  def test_output_depends_on_timestep_and_text(self):
    model = WanModel(rngs=nnx.Rngs(0), **_CONFIG)
    latents, timestep, text = _inputs()
    out = np.asarray(model(latents, timestep, text))
    np.testing.assert_allclose(np.asarray(model(latents, timestep, text)), out)  # deterministic
    self.assertFalse(np.allclose(np.asarray(model(latents, timestep + 100.0, text)), out))
    self.assertFalse(np.allclose(np.asarray(model(latents, timestep, text + 1.0)), out))


class EmbeddingTest(unittest.TestCase):

  def test_rotary_pos_embed_has_one_row_per_patch(self):
    rope = WanRotaryPosEmbed(attention_head_dim=8, patch_size=(1, 2, 2), max_seq_len=32)
    freqs = rope(jnp.zeros((1, _FRAMES, _HEIGHT, _WIDTH, 4)))  # channels-last video
    num_patches = _FRAMES * (_HEIGHT // 2) * (_WIDTH // 2)
    self.assertEqual(freqs.shape[:3], (1, 1, num_patches))

  def test_sinusoidal_embeddings(self):
    emb = get_sinusoidal_embeddings(jnp.array([0.0, 1.0, 999.0]), embedding_dim=16)
    self.assertEqual(emb.shape, (3, 16))
    self.assertTrue(np.all(np.abs(np.asarray(emb)) <= 1.0 + 1e-6))
    self.assertFalse(np.allclose(np.asarray(emb[0]), np.asarray(emb[1])))


def _reference_wan_rope(head_dim, max_seq_len, frames, height, width, theta=10000.0):
  """NumPy port of diffusers' `WanRotaryPosEmbed`: complex freqs, (t, h, w) channel split, (f, h, w) token order."""
  h_dim = w_dim = 2 * (head_dim // 6)
  t_dim = head_dim - h_dim - w_dim
  per_axis = []
  for dim in (t_dim, h_dim, w_dim):
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64)[: dim // 2] / dim))
    per_axis.append(np.exp(1j * np.outer(np.arange(max_seq_len), inv_freq)))
  f = np.broadcast_to(per_axis[0][:frames, None, None, :], (frames, height, width, t_dim // 2))
  h = np.broadcast_to(per_axis[1][None, :height, None, :], (frames, height, width, h_dim // 2))
  w = np.broadcast_to(per_axis[2][None, None, :width, :], (frames, height, width, w_dim // 2))
  return np.concatenate([f, h, w], axis=-1).reshape((1, 1, frames * height * width, head_dim // 2))


def _reference_apply_rotary_emb(x, freqs):
  """diffusers' `apply_rotary_emb`: consecutive (even, odd) channel pairs are complex numbers rotated by `freqs`."""
  x_complex = x[..., 0::2] + 1j * x[..., 1::2]
  rotated = x_complex * freqs
  return np.stack([rotated.real, rotated.imag], axis=-1).reshape(x.shape)


class RopeLayoutTest(unittest.TestCase):
  """RoPE must match diffusers' layout; a wrong (t, h, w) split, token order or channel pairing fails here."""

  _HEAD_DIM, _MAX_SEQ_LEN = 24, 16  # t / h / w get 8 / 8 / 8 channels
  _FRAMES, _HEIGHT, _WIDTH = 3, 8, 12  # after a (1, 2, 2) patch: 3 x 4 x 6 tokens

  def test_rotary_pos_embed_matches_diffusers_layout(self):
    rope = WanRotaryPosEmbed(attention_head_dim=self._HEAD_DIM, patch_size=(1, 2, 2), max_seq_len=self._MAX_SEQ_LEN)
    freqs = np.asarray(rope(jnp.zeros((1, self._FRAMES, self._HEIGHT, self._WIDTH, 4))))
    expected = _reference_wan_rope(self._HEAD_DIM, self._MAX_SEQ_LEN, self._FRAMES, self._HEIGHT // 2, self._WIDTH // 2)
    np.testing.assert_allclose(freqs, expected, rtol=1e-5, atol=1e-6)

  def test_attention_rope_rotates_channel_pairs(self):
    attn = WanAttention(rngs=nnx.Rngs(0), query_dim=48, heads=2, dim_head=self._HEAD_DIM, attention_kernel="dot_product")
    num_tokens = self._FRAMES * (self._HEIGHT // 2) * (self._WIDTH // 2)
    freqs = _reference_wan_rope(self._HEAD_DIM, self._MAX_SEQ_LEN, self._FRAMES, self._HEIGHT // 2, self._WIDTH // 2)
    rng = np.random.default_rng(0)
    xq = rng.standard_normal((2, 2, num_tokens, self._HEAD_DIM)).astype(np.float32)  # (batch, heads, length, head_dim)
    xk = rng.standard_normal((2, 2, num_tokens, self._HEAD_DIM)).astype(np.float32)
    q_out, k_out = attn._apply_rope(jnp.asarray(xq), jnp.asarray(xk), jnp.asarray(freqs, jnp.complex64))  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(q_out), _reference_apply_rotary_emb(xq, freqs), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(k_out), _reference_apply_rotary_emb(xk, freqs), rtol=1e-5, atol=1e-5)


class _Identity(nnx.Module):

  def __call__(self, hidden_states, *args, **kwargs):
    return hidden_states


class _Zeros(nnx.Module):

  def __call__(self, hidden_states, *args, **kwargs):
    return jnp.zeros_like(hidden_states)


def _layer_norm(x, eps=1e-6):
  mean = x.mean(axis=-1, keepdims=True)
  var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
  return (x - mean) / np.sqrt(var + eps)


class ModulationOrderTest(unittest.TestCase):
  """The six AdaLN chunks must be (shift, scale, gate) for self-attention, then (shift, scale, gate) for the FFN.

  Self-attention and the FFN are replaced by identities and cross-attention by zeros, so the block output is a
  closed-form function of the six chunks; distinct per-chunk values make any reordering change the result.
  """

  def test_chunk_order_matches_diffusers(self):
    dim, length = 16, 5
    with jax.set_mesh(_single_device_mesh()):
      block = WanTransformerBlock(rngs=nnx.Rngs(0), dim=dim, ffn_dim=32, num_heads=2, cross_attn_norm=True)
      block.attn1, block.attn2, block.ffn = _Identity(), _Zeros(), _Identity()
      rng = np.random.default_rng(0)
      table = rng.standard_normal((1, 6, dim)).astype(np.float32)
      block.adaln_scale_shift_table.value = jnp.asarray(table)
      temb = (0.1 * rng.standard_normal((1, 6, dim))).astype(np.float32)
      x = rng.standard_normal((1, length, dim)).astype(np.float32)
      out = block(jnp.asarray(x), jnp.zeros((1, 3, dim)), jnp.asarray(temb), rotary_emb=None)

    shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = np.split(  # pylint: disable=unbalanced-tuple-unpacking
        table + temb, 6, axis=1
    )
    h = x + gate_msa * (_layer_norm(x) * (1 + scale_msa) + shift_msa)  # self-attention = identity
    h = h + 0.0  # cross-attention = zeros
    h = h + c_gate * (_layer_norm(h) * (1 + c_scale) + c_shift)  # FFN = identity
    np.testing.assert_allclose(np.asarray(out), h, rtol=1e-5, atol=1e-5)


@pytest.mark.tpu_only
class FlashAttentionEquivalenceTest(unittest.TestCase):
  """`attention="flash"` (the yml's setting; Splash kernel, TPU only) matches `dot_product` on the same weights."""

  def test_flash_matches_dot_product(self):
    # Splash needs 128-aligned blocks: 2 x 8 x 8 = 128 video tokens and 128 text tokens.
    latents = jax.random.normal(jax.random.key(0), (1, _CONFIG["in_channels"], 2, 16, 16))
    timestep = jnp.array([500.0])
    text = jax.random.normal(jax.random.key(1), (1, 128, _CONFIG["text_dim"]))
    block_sizes = _BLOCK_SIZES_128
    # Full-precision matmuls: TPU's default float32 matmul precision (bf16 passes) alone differs by ~1e-2.
    with jax.set_mesh(_single_device_mesh()), jax.default_matmul_precision("float32"):
      dot = WanModel(rngs=nnx.Rngs(0), attention="dot_product", scan_layers=False, **_CONFIG)
      flash = WanModel(
          rngs=nnx.Rngs(0),
          attention="flash",
          flash_min_seq_length=0,  # use the kernel for the short test sequences too
          flash_block_sizes=block_sizes,
          sharding=utils_wan.WanSharding.from_rules(utils_wan.yml_rules("logical_axis_rules")),
          scan_layers=False,
          **_CONFIG,
      )
      nnx.update(flash, nnx.state(dot, nnx.Param))
      expected = np.asarray(dot(latents, timestep, text))
      # Under jit, as in the pipeline: eager `with_sharding_constraint` rejects single-device inputs.
      actual = np.asarray(nnx.jit(lambda model, *args: model(*args))(flash, latents, timestep, text))
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


# --- Attention kernels (`_apply_attention`): Tokamax dot-product and Splash flash attention ---

_MESH_AXES = utils_wan.WAN_MESH_AXES
# Resolved specs as the kernels get them: q sequence-sharded over `context`, KV whole on every shard (flash layout).
_Q_SPEC = P("data", "tensor", "context", None)
_KV_FULL = P("data", "tensor", None, None)
_OUT_SPEC = P("data", "context", "tensor")
_ATTN_BATCH, _ATTN_HEADS = 2, 4


def _attn_qkv(seq_len, head_dim=128, kv_len=None):
  """Random [B, H, S, D] float32 q, k, v (k unscaled: `_apply_attention` applies `scale`)."""
  rng = np.random.default_rng(0)
  q = rng.standard_normal((_ATTN_BATCH, _ATTN_HEADS, seq_len, head_dim), dtype=np.float32)
  k, v = (rng.standard_normal((_ATTN_BATCH, _ATTN_HEADS, kv_len or seq_len, head_dim), dtype=np.float32) for _ in "kv")
  return q, k, v


def _attn_reference(q, k, v, keep=None):
  """NumPy softmax(q k^T / sqrt(D)) v for [B, H, S, D] inputs and a [B, K] keep mask, as [B, S, H * D]."""
  logits = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), k.astype(np.float64)) * q.shape[-1] ** -0.5
  if keep is not None:
    logits = np.where(keep[:, None, None, :], logits, -np.inf)
  probs = np.exp(logits - logits.max(-1, keepdims=True))
  probs /= probs.sum(-1, keepdims=True)
  out = np.einsum("bhqk,bhkd->bhqd", probs, v.astype(np.float64))
  b, h, s, d = out.shape
  return out.transpose(0, 2, 1, 3).reshape(b, s, h * d)


def _attend(q, k, v, attention_kernel, block_sizes=None, attention_mask=None, use_memory_efficient_attention=False, **kw):
  return transformer_wan._apply_attention(  # pylint: disable=protected-access
      jnp.asarray(q),
      jnp.asarray(k),
      jnp.asarray(v),
      heads=_ATTN_HEADS,
      attention_kernel=attention_kernel,
      flash_min_seq_length=0,
      use_memory_efficient_attention=use_memory_efficient_attention,
      scale=(q.shape[-1] if q.ndim == 4 else q.shape[-1] // _ATTN_HEADS) ** -0.5,
      dtype=jnp.float32,
      q_spec=_Q_SPEC,
      kv_spec=_KV_FULL,
      flash_block_sizes=block_sizes,
      attention_mask=attention_mask,
      **kw,
  )


class SplashHelpersTest(unittest.TestCase):
  """Splash block-size selection, padding and the segment ids that mask it."""

  def test_self_attention_returns_configured_blocks(self):
    blocks = transformer_wan._splash_block_sizes(_BLOCK_SIZES_128, 256, 256, jnp.float32)  # pylint: disable=protected-access
    self.assertIs(blocks, _BLOCK_SIZES_128)

  def test_cross_attention_kv_blocks_span_the_text(self):
    blocks = transformer_wan._splash_block_sizes(_BLOCK_SIZES_128, 512, 77, jnp.float32)  # pylint: disable=protected-access
    self.assertEqual((blocks.block_q, blocks.block_kv, blocks.block_kv_compute), (128, 128, 128))
    self.assertTrue(blocks.use_fused_bwd_kernel)

  def test_pad_axis(self):
    x = jnp.ones((1, 2, 100, 64))
    self.assertEqual(transformer_wan._pad_axis(x, 2, 128).shape, (1, 2, 128, 64))  # pylint: disable=protected-access
    self.assertIs(transformer_wan._pad_axis(x, 3, 64), x)  # pylint: disable=protected-access

  def test_padding_segment_ids(self):
    segment_ids = transformer_wan._padding_segment_ids(3, 4, 2, 4, attention_mask=None)  # pylint: disable=protected-access
    np.testing.assert_array_equal(segment_ids.q, [1, 1, 1, 0])
    np.testing.assert_array_equal(segment_ids.kv, [1, 1, 0, 0])
    keep = jnp.asarray([[True, False], [True, True]])
    segment_ids = transformer_wan._padding_segment_ids(3, 4, 2, 4, attention_mask=keep)  # pylint: disable=protected-access
    np.testing.assert_array_equal(segment_ids.q, [[1, 1, 1, 0]] * 2)
    np.testing.assert_array_equal(segment_ids.kv, [[1, 0, 0, 0], [1, 1, 0, 0]])

  def test_unknown_kernel_raises(self):
    q, k, v = _attn_qkv(8, head_dim=8)
    with self.assertRaisesRegex(ValueError, "Unexpected attention kernel"):
      _attend(q, k, v, "ulysses")


class DotProductAttentionTest(parameterized.TestCase):
  """The `dot_product` kernel (`tokamax.dot_product_attention`, runs on CPU) matches the NumPy reference."""

  @parameterized.named_parameters(
      ("bhsd", False, False),
      ("flat", True, False),
      ("memory_efficient", False, True),
  )
  def test_matches_reference(self, flat, use_memory_efficient_attention):
    q, k, v = _attn_qkv(64, head_dim=16, kv_len=48)
    if flat:  # [B, H, S, D] -> [B, S, H * D], the cross-attention layout
      q, k, v = (x.transpose(0, 2, 1, 3).reshape(_ATTN_BATCH, x.shape[2], -1) for x in (q, k, v))
    out = _attend(q, k, v, "dot_product", use_memory_efficient_attention=use_memory_efficient_attention)
    if flat:
      q, k, v = (x.reshape(_ATTN_BATCH, x.shape[1], _ATTN_HEADS, -1).transpose(0, 2, 1, 3) for x in (q, k, v))
    np.testing.assert_allclose(np.asarray(out), _attn_reference(q, k, v), rtol=1e-4, atol=1e-4)

  def test_keep_mask(self):
    q, k, v = _attn_qkv(16, head_dim=8, kv_len=12)
    keep = np.ones((_ATTN_BATCH, 12), bool)
    keep[0, 5:] = False
    out = np.asarray(_attend(q, k, v, "dot_product", attention_mask=jnp.asarray(keep)))
    np.testing.assert_allclose(out, _attn_reference(q, k, v, keep), rtol=1e-4, atol=1e-4)
    # A batch row with no valid key attends to nothing.
    keep[1] = False
    out = np.asarray(_attend(q, k, v, "dot_product", attention_mask=jnp.asarray(keep)))
    np.testing.assert_array_equal(out[1], 0)


@pytest.mark.tpu_only
class TpuFlashAttentionTest(parameterized.TestCase):
  """The Splash `flash` kernel (TPU only) matches the reference on a sequence-sharded `context` mesh."""

  def test_flash(self):
    """Under `jax.set_mesh`, as in `WanModel`; 500 tokens exercise the padding."""
    context = min(4, len(jax.devices()))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:context]).reshape((1, 1, context, 1)), _MESH_AXES)
    q, k, v = _attn_qkv(500)
    run = functools.partial(_attend, attention_kernel="flash", block_sizes=_BLOCK_SIZES_128, out_spec=_OUT_SPEC)
    with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
      out = jax.jit(run)(q, k, v)
    np.testing.assert_allclose(np.asarray(out), _attn_reference(q, k, v), rtol=2e-3, atol=2e-3)

  def test_cross_attention_with_keep_mask(self):
    context = min(4, len(jax.devices()))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:context]).reshape((1, 1, context, 1)), _MESH_AXES)
    q, k, v = _attn_qkv(512, kv_len=77)
    keep = np.ones((_ATTN_BATCH, 77), bool)
    keep[0, 40:] = False
    run = functools.partial(_attend, attention_kernel="flash", block_sizes=_BLOCK_SIZES_128)
    with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
      out = jax.jit(run)(q, k, v, attention_mask=jnp.asarray(keep))
    np.testing.assert_allclose(np.asarray(out), _attn_reference(q, k, v, keep), rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
  unittest.main()
