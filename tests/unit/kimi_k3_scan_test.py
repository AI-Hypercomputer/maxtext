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

"""Scanned Kimi-K3 (`scan_layers=True`) tests.

1. The static-padded AttnRes (`AttnResBuffer` + masked softmax) equals the dynamic
   concat path, including its block-commit semantics.
2. `kimi_scan_layout` splits the stack into unrolled prefix / scanned cycles / unrolled
   suffix, and rejects non-periodic layer patterns.
3. Scanned vs unscanned logits (and gradients) parity on a toy 13-layer model, with
   AttnRes block commits both outside and inside the scanned region.
"""

# pylint: disable=protected-access  # exercises the private AttnRes pooling primitives

import sys
import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attn_res
from maxtext.layers import moe
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_LAYERS = 13
# Prefix 0..3 (layer 0 dense), scanned cycles [KDA,KDA,KDA,MLA] at 4..7 and 8..11, suffix [12].
_FULL_ATTN_LAYERS = [3, 7, 11, 12]
_SEQ_LEN = 8
_EMB_DIM = 64


def _toy_config(**overrides):
  """A 13-layer kimi-k3 at toy scale (fp32 so parity is tight)."""
  kwargs = {
      "override_model_config": True,
      "base_emb_dim": _EMB_DIM,
      "base_num_decoder_layers": _NUM_LAYERS,
      "full_attn_layers": _FULL_ATTN_LAYERS,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 16,
      "vocab_size": 128,
      "q_lora_rank": 32,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 8,
      "v_head_dim": 16,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 32,
      "moe_intermediate_size": 32,
      "shared_intermediate_size": 32,
      "shared_experts": 1,
      "num_shared_experts": 1,
      "routed_expert_hidden_size": 32,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "moe_expert_input_dim": 32,
      "moe_router_input_dim": _EMB_DIM,
      "megablox": False,
      "max_target_length": _SEQ_LEN,
      "max_position_embeddings": _SEQ_LEN,
      "original_max_position_embeddings": _SEQ_LEN,
      "max_prefill_predict_length": _SEQ_LEN // 2,
      "per_device_batch_size": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "packing": False,
      "enable_checkpointing": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], model_name="kimi-k3", **kwargs)


def _build_model(cfg):
  return model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))


def _inputs(batch):
  tokens = (jnp.arange(batch * _SEQ_LEN, dtype=jnp.int32).reshape(batch, _SEQ_LEN) * 5 % 97 + 1).astype(jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(_SEQ_LEN), tokens.shape)
  return tokens, positions


@nnx.jit
def _forward(model, tokens, positions):
  return model(tokens, positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)


def _loss(model, tokens, positions):
  logits = model(tokens, positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)
  return jnp.mean(jax.nn.log_softmax(logits.astype(jnp.float32))[..., 0])


def copy_unscanned_params_to_scanned(unscanned_model, scanned_model, scanned_cfg):
  """Copies the per-layer params of an unscanned Kimi-K3 into the stacked scanned layout.

  Unrolled prefix/suffix layers share names (`decoder/layers_{i}`); a scanned-block variable
  `decoder/scanned_blocks/layers_{j}/...` is the stack of the unscanned
  `decoder/layers_{prefix + b * cycle + j}/...` over blocks b. Every `nnx.Variable` type is
  copied (e.g. the `moe.MoEBiasVar` router bias, not just `nnx.Param`), except RNG state:
  `nnx.Param` is stacked on `param_scan_axis`, other variables on axis 0 (as `nnx.scan` does).
  """
  num_prefix, num_blocks, _, pattern = kimi_k3.kimi_scan_layout(scanned_cfg)
  weights = nnx.All(nnx.Variable, nnx.Not(nnx.RngState))
  src = dict(nnx.to_flat_state(nnx.state(unscanned_model, weights)))
  dst_state = nnx.state(scanned_model, weights)
  new_flat = []
  for path, var in nnx.to_flat_state(dst_state):
    axis = scanned_cfg.param_scan_axis if isinstance(var, nnx.Param) else 0
    if "scanned_blocks" in path:
      i = path.index("scanned_blocks")
      j = int(path[i + 1].split("_")[1])
      per_block = [
          src[path[:i] + (f"layers_{num_prefix + b * len(pattern) + j}",) + path[i + 2 :]].get_value()
          for b in range(num_blocks)
      ]
      value = jnp.stack(per_block, axis=axis)
    else:
      value = src[path].get_value()
    assert value.shape == var.get_value().shape, (path, value.shape, var.get_value().shape)
    new_flat.append((path, var.replace(value)))
  nnx.update(scanned_model, nnx.from_flat_state(new_flat))


class AttnResMaskedTest(parameterized.TestCase):
  """Static-padded AttnRes vs the dynamic concat path."""

  def _weights(self, d, seed):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.normal(k1, (d, 1)), 1.0 + 0.1 * jax.random.normal(k2, (d,))

  @parameterized.parameters(0, 1, 3, 5)
  def test_masked_equals_dynamic(self, num_committed):
    b, s, d, slots = 2, 3, 16, 6
    keys = jax.random.split(jax.random.PRNGKey(num_committed), 3)
    prefix = jax.random.normal(keys[0], (b, s, d))
    # Junk (but finite) values in the unused slots must not leak into the result.
    buffer = jax.random.normal(keys[1], (b, s, slots, d)) * 100.0
    proj, norm = self._weights(d, 7)
    dynamic = attn_res._apply_attn_res(prefix, buffer[..., :num_committed, :], proj, norm, 1e-5)
    masked = attn_res._apply_attn_res(
        prefix, buffer, proj, norm, 1e-5, num_blocks=slots, num_committed=jnp.int32(num_committed)
    )
    np.testing.assert_allclose(np.asarray(masked), np.asarray(dynamic), rtol=1e-6, atol=1e-6)
    # Traced count under jit gives the same answer.
    jitted = jax.jit(attn_res._apply_attn_res_masked)(prefix, buffer, jnp.int32(num_committed), proj, norm)
    # jit fuses the reductions differently; equal up to fp32 rounding (values here are O(100)).
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(dynamic), rtol=2e-5, atol=1e-4)

  def test_masked_grad_equals_dynamic(self):
    b, s, d, slots, n = 1, 2, 8, 4, 2
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    prefix = jax.random.normal(keys[0], (b, s, d))
    buffer = jax.random.normal(keys[1], (b, s, slots, d))
    proj, norm = self._weights(d, 3)

    def f_dyn(p, blk, w, g):
      return jnp.sum(jnp.sin(attn_res._apply_attn_res(p, blk[..., :n, :], w, g, 1e-5)))

    def f_msk(p, blk, w, g):
      return jnp.sum(jnp.sin(attn_res._apply_attn_res_masked(p, blk, jnp.int32(n), w, g, epsilon=1e-5)))

    gd = jax.grad(f_dyn, argnums=(0, 1, 2, 3))(prefix, buffer, proj, norm)
    gm = jax.grad(f_msk, argnums=(0, 1, 2, 3))(prefix, buffer, proj, norm)
    for a, e in zip(gm, gd):
      self.assertTrue(bool(jnp.all(jnp.isfinite(a))))
      np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=1e-5, atol=1e-6)
    # Unused slots get exactly zero gradient.
    np.testing.assert_array_equal(np.asarray(gm[1][..., n:, :]), 0.0)

  def test_bf16_buffer_matches_dynamic(self):
    b, s, d, slots, n = 1, 4, 32, 5, 3
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    prefix = jax.random.normal(keys[0], (b, s, d)).astype(jnp.bfloat16)
    buffer = jax.random.normal(keys[1], (b, s, slots, d)).astype(jnp.bfloat16)
    proj, norm = self._weights(d, 2)
    dynamic = attn_res._apply_attn_res(prefix, buffer[..., :n, :], proj, norm, 1e-5)
    masked = attn_res._apply_attn_res_masked(prefix, buffer, jnp.int32(n), proj, norm, epsilon=1e-5)
    self.assertEqual(masked.dtype, jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(masked, np.float32), np.asarray(dynamic, np.float32), rtol=1e-2, atol=1e-2)

  def test_commit_sequence_matches_concat(self):
    """Speculative writes + count advance reproduce the concatenated block stack."""
    num_layers, block_size, d = 10, 3, 4
    slots = attn_res.attn_res_num_buffer_slots(num_layers, block_size)
    self.assertEqual(slots, 5)  # commits at 0, 3, 6, 9 + 1 spare slot
    xs = [jnp.full((1, 1, d), float(i + 1)) for i in range(num_layers)]
    buf = attn_res.init_attn_res_buffer(xs[0], slots)

    def step(buf, x, idx):
      return attn_res.commit_attn_res_block(buf, x, idx % block_size == 0)

    jitted_step = jax.jit(step)
    concat = []
    for i, x in enumerate(xs):
      # Alternate a static and a traced commit decision.
      buf = step(buf, x, i) if i % 2 else jitted_step(buf, x, jnp.int32(i))
      if i % block_size == 0:
        concat.append(x)
      n = int(buf.count)
      self.assertEqual(n, len(concat))
      np.testing.assert_array_equal(np.asarray(buf.blocks[..., :n, :]), np.asarray(jnp.stack(concat, axis=-2)))


class KimiScanLayoutTest(unittest.TestCase):
  """Prefix / scanned / suffix split."""

  def test_released_layout(self):
    cfg = _toy_config(
        base_num_decoder_layers=93,
        full_attn_layers=[3 + 4 * c for c in range(23)] + [92],
    )
    self.assertEqual(kimi_k3.kimi_scan_layout(cfg), (4, 22, 92, ((True, True),) * 3 + ((False, True),)))
    self.assertEqual(attn_res.attn_res_num_buffer_slots(93, 12), 9)

  def test_toy_layout(self):
    cfg = _toy_config()
    self.assertEqual(kimi_k3.kimi_scan_layout(cfg), (4, 2, 12, ((True, True),) * 3 + ((False, True),)))

  def test_non_periodic_is_rejected(self):
    cfg = _toy_config(full_attn_layers=[3, 6, 11, 12])
    with self.assertRaisesRegex(ValueError, "periodic"):
      kimi_k3.kimi_scan_layout(cfg)


class KimiK3ScannedParityTest(parameterized.TestCase):
  """Scanned vs unscanned Kimi-K3 logits on the same weights."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._unscanned = {}

  def _unscanned_model(self, block_size, remat_policy):
    """Cached unscanned toy model with a randomized (nonzero) MoE router bias."""
    key = (block_size, remat_policy)
    if key not in self._unscanned:
      cfg = _toy_config(scan_layers=False, attn_res_block_size=block_size, remat_policy=remat_policy)
      model = _build_model(cfg)
      # The router bias (`moe.MoEBiasVar`) initializes to zeros; randomize it so the parity
      # tests catch a scanned copy / stacking that drops or misplaces it.
      biases = nnx.state(model, moe.MoEBiasVar)
      leaves = jax.tree.leaves(biases)
      self.assertNotEmpty(leaves)
      keys = jax.random.split(jax.random.PRNGKey(42), len(leaves))
      nnx.update(
          model,
          jax.tree.unflatten(
              jax.tree.structure(biases), [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(keys, leaves)]
          ),
      )
      self._unscanned[key] = model
    return self._unscanned[key]

  def _scanned_model(self, block_size, remat_policy):
    cfg = _toy_config(scan_layers=True, attn_res_block_size=block_size, remat_policy=remat_policy)
    model = _build_model(cfg)
    copy_unscanned_params_to_scanned(self._unscanned_model(block_size, remat_policy), model, cfg)
    return model

  @parameterized.named_parameters(
      # Commits at 0 and 12: prefix and suffix only.
      ("commits_outside_scan", 12, "full"),
      # Commits at 0, 5, 10: two inside the scanned cycles, at different sub-layer positions.
      ("commits_inside_scan", 5, "full"),
      # Commit every other layer, no remat.
      ("every_other_layer_no_remat", 2, "none"),
  )
  def test_logits_parity(self, block_size, remat_policy):
    unscanned = self._unscanned_model(block_size, remat_policy)
    scanned = self._scanned_model(block_size, remat_policy)
    self.assertTrue(hasattr(scanned.decoder, "scanned_blocks"))
    self.assertFalse(hasattr(scanned.decoder, "layers_4"))
    tokens, positions = _inputs(batch=4)
    ref = _forward(unscanned, tokens, positions)
    out = _forward(scanned, tokens, positions)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-5, atol=2e-5)

  def test_get_layers_order(self):
    """get_layers() returns prefix, scanned cycles and the unrolled suffix in forward order."""
    block_size = 5
    unscanned = self._unscanned_model(block_size, "full")
    scanned = self._scanned_model(block_size, "full")
    decoder = scanned.decoder
    num_prefix, _, suffix_start, _ = kimi_k3.kimi_scan_layout(decoder.config)
    expected = [getattr(decoder, f"layers_{i}") for i in range(num_prefix)] + [decoder.scanned_blocks]
    expected += [getattr(decoder, f"layers_{i}") for i in range(suffix_start, _NUM_LAYERS)]
    got = decoder.get_layers()
    self.assertEqual([id(m) for m in got], [id(m) for m in expected])
    self.assertIsInstance(got[num_prefix], kimi_k3.KimiK3ScannableBlock)
    self.assertEqual(got[-1].layer_idx, _NUM_LAYERS - 1)
    # Unscanned: all layers, in order.
    self.assertEqual([m.layer_idx for m in unscanned.decoder.get_layers()], list(range(_NUM_LAYERS)))

  def test_grad_parity(self):
    block_size = 5
    unscanned = self._unscanned_model(block_size, "full")
    scanned = self._scanned_model(block_size, "full")
    tokens, positions = _inputs(batch=4)
    g_ref = nnx.jit(nnx.grad(_loss))(unscanned, tokens, positions)
    g_scan = nnx.jit(nnx.grad(_loss))(scanned, tokens, positions)
    # Map the scanned grads back onto the unscanned layout and compare leaf by leaf.
    cfg = _toy_config(scan_layers=True, attn_res_block_size=block_size)
    num_prefix, num_blocks, _, pattern = kimi_k3.kimi_scan_layout(cfg)
    ref_flat = {p: v.get_value() for p, v in nnx.to_flat_state(g_ref)}
    checked = 0

    def assert_close(got, want):
      # fp32 summation order differs between the scanned and unrolled programs; compare
      # against each leaf's own scale.
      want = np.asarray(want)
      np.testing.assert_allclose(np.asarray(got), want, rtol=1e-3, atol=1e-3 * float(np.max(np.abs(want))) + 1e-8)

    for path, var in nnx.to_flat_state(g_scan):
      if "scanned_blocks" in path:
        i = path.index("scanned_blocks")
        j = int(path[i + 1].split("_")[1])
        for b in range(num_blocks):
          ref_path = path[:i] + (f"layers_{num_prefix + b * len(pattern) + j}",) + path[i + 2 :]
          assert_close(jnp.take(var.get_value(), b, axis=cfg.param_scan_axis), ref_flat[ref_path])
          checked += 1
      else:
        assert_close(var.get_value(), ref_flat[path])
        checked += 1
    self.assertEqual(checked, len(ref_flat))


if __name__ == "__main__":
  unittest.main()
