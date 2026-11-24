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

"""DeepSeek-V4.1 tests that need neither PyTorch nor the reference implementation.

The parity tests against the plain-PyTorch reference are in deepseek_v41_vs_reference_test.py.
"""

import sys
import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.layers import attention_deepseek41
from maxtext.layers import engram_deepseek41
from maxtext.layers import moe
from maxtext.models import deepseek41

_FP4_E2M1 = np.array(param_mapping._FP4_E2M1_VALUES, np.float32)  # pylint: disable=protected-access


def _scales(rng, shape):
  return np.exp2(rng.integers(-8, 4, shape)).astype(ml_dtypes.float8_e8m0fnu)


def _expand(scale, rows, cols):
  s = scale.astype(np.float32)
  return np.repeat(np.repeat(s, -(-rows // s.shape[0]), axis=0)[:rows], -(-cols // s.shape[1]), axis=1)[:, :cols]


def _fp8_case(rng, rows, cols, scale_shape):
  weight = (rng.standard_normal((rows, cols)) * 4).astype(ml_dtypes.float8_e4m3fn)
  scale = _scales(rng, scale_shape)
  return weight, scale, weight.astype(np.float32) * _expand(scale, rows, cols)


def _fp4_case(rng, rows, cols, scale_shape):
  codes = rng.integers(0, 16, (rows, cols))
  packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).view(np.int8)
  scale = _scales(rng, scale_shape)
  return packed, scale, _FP4_E2M1[codes] * _expand(scale, rows, cols)


class DequantizeTest(parameterized.TestCase):
  """`param_mapping.dequantize_block_scaled` on numpy, single-device and row-split inputs, against numpy."""

  def _check(self, weight, scale, expected):
    """Checks the numpy, single-device and row-split device paths against `expected`."""
    np.testing.assert_array_equal(
        np.asarray(param_mapping.dequantize_block_scaled((weight, scale)), np.float32), expected
    )
    got = param_mapping.dequantize_block_scaled((jnp.asarray(weight), jnp.asarray(scale)))
    self.assertEqual(got.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got, np.float32), expected)
    # As the streaming loader hands them over: the weight split by rows over the chips, the scales either a full copy
    # or (with one scale row per weight row) split the same way.
    devices = jax.devices()
    for n in sorted({d for d in (2, 4, 8, len(devices)) if 1 < d <= len(devices) and weight.shape[0] % d == 0}):
      w = jax.device_put(weight, hf_streaming_load.choose_load_sharding(weight.shape, weight.dtype, devices[:n], 0))
      for split_scale in (False, True):
        if split_scale and scale.shape[0] != weight.shape[0]:
          continue
        min_bytes = 0 if split_scale else 1 << 62
        s = jax.device_put(
            scale, hf_streaming_load.choose_load_sharding(scale.shape, scale.dtype, devices[:n], min_bytes)
        )
        got = param_mapping.dequantize_block_scaled((w, s))
        self.assertEqual(got.sharding, w.sharding)
        np.testing.assert_array_equal(
            np.asarray(got, np.float32), expected, err_msg=f"{n} chips, split scales {split_scale}"
        )

  @parameterized.named_parameters(
      ("fp8_32x32_blocks", 96, 160, (3, 5)),
      # 256 rows over 2 / 4 / 8 chips: 128 / 64 / 32 rows each, whole scale blocks (the 64-chip v5p `wq_b` case).
      ("fp8_shards_aligned_to_blocks", 256, 96, (8, 3)),
      # 8x32 blocks: over 8 chips each holds 8 weight rows, as many as the scale has rows (the replicated scale must
      # still be indexed by block, e.g. 32x32 blocks over 32 chips).
      ("fp8_rows_per_chip_equal_scale_rows", 64, 64, (8, 2)),
      ("fp8_1x32_rows", 200, 64, (200, 2)),
  )
  def test_fp8(self, rows, cols, scale_shape):
    self._check(*_fp8_case(np.random.default_rng(rows), rows, cols, scale_shape))

  def test_fp4_packed(self):
    self._check(*_fp4_case(np.random.default_rng(0), 64, 128, (64, 4)))


def _reference_hashes(compressed, pad, multipliers, primes, offsets, max_ngram_size, n_heads):
  """The upstream n-gram hash in plain Python integers (int64 semantics; every product fits)."""
  b, s = compressed.shape
  out = np.zeros((b, s, (max_ngram_size - 1) * n_heads), np.int64)
  for i in range(b):
    for t in range(s):
      tokens = [int(compressed[i, t - k]) if t - k >= 0 else pad for k in range(max_ngram_size)]
      rolling = tokens[0] * int(multipliers[0])
      for k in range(1, max_ngram_size):
        rolling ^= tokens[k] * int(multipliers[k])
        for h in range(n_heads):
          col = (k - 1) * n_heads + h
          out[i, t, col] = rolling % int(primes[col]) + int(offsets[col])
  return out


class EngramHashTest(parameterized.TestCase):
  """The uint32-limb engram hash equals the upstream int64 hash, at the real (~16M) bucket sizes."""

  def test_matches_int64_hash(self):
    rng = np.random.default_rng(0)
    vocab, compressed_vocab, layer_ids, max_ngram_size, n_heads, pad_id = 1000, 99092, (1, 14), 4, 8, 2
    token_map = rng.integers(0, compressed_vocab, vocab)
    ids = rng.integers(0, vocab, (2, 12))
    multipliers = engram_deepseek41.compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab)
    primes = engram_deepseek41.engram_primes(layer_ids, max_ngram_size, n_heads, 16_000_000)
    for index, layer_id in enumerate(layer_ids):
      hasher = engram_deepseek41.DeepSeek41NgramHasher(
          token_map, compressed_vocab, pad_id, layer_ids, layer_id, max_ngram_size, n_heads, 16_000_000
      )
      flat = np.array([p for per_ngram in primes[index] for p in per_ngram], np.int64)
      offsets = np.concatenate([[0], np.cumsum(flat)[:-1]])
      want = _reference_hashes(
          token_map[ids], int(token_map[pad_id]), multipliers[index], flat, offsets, max_ngram_size, n_heads
      )
      np.testing.assert_array_equal(np.asarray(hasher(jnp.asarray(ids))), want)


class LayerStructureTest(parameterized.TestCase):
  """The released config's layer roles and the scanned segments built from them."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="deepseek41_structure_test",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        model_name="deepseek4.1-flash",
    )

  def test_layer_specs(self):
    specs = deepseek41.build_layer_specs(self.config)
    self.assertLen(specs, 40)
    self.assertEqual([s.layer_id for s in specs if s.is_kv_source], [2, 8, 14, 20])
    self.assertEqual([s.layer_id for s in specs if s.is_index_source], [2, 8, 14, 20, 24, 28, 32, 36])
    self.assertEqual([s.layer_id for s in specs if s.is_candidate_source], [20])
    # Every layer after the candidate source; the index sources among them restrict their top-k to the candidates.
    self.assertEqual([s.layer_id for s in specs if s.uses_candidates], list(range(21, 40)))
    self.assertEqual([s.layer_id for s in specs if s.has_engram], [1, 14])

  def test_segments(self):
    segments = deepseek41.build_segments(self.config)
    layout = [(seg.groups[0][0], seg.groups[0][-1], seg.repeats) for seg in segments]
    # [0] [1] [2-7]x2 [14-19] [20-23] [24-27]x4: every layer exactly once, in order.
    self.assertEqual(layout, [(0, 0, 1), (1, 1, 1), (2, 7, 2), (14, 19, 1), (20, 23, 1), (24, 27, 4)])
    self.assertEqual([i for seg in segments for group in seg.groups for i in group], list(range(40)))

  def test_segment_carry_plans(self):
    segments = deepseek41.build_segments(self.config)
    plans = deepseek41.build_segment_carry_plans(self.config, segments)
    # Segment [20-23] (KV / index / candidate source 20) feeds every later ratio-1 segment.
    self.assertEqual(
        [(p.add_before, p.keep_after) for p in plans],
        [
            ((), ()),
            ((), ()),
            ((), ()),
            ((), ()),
            (("compress_kv", "index_k", "candidates"), ("compress_kv", "index_k", "candidates")),
            ((), ()),
        ],
    )
    self.assertEqual([p.compress_ratio for p in plans], [0, 0, 2, 2, 1, 1])


def _flash_config(model_name="deepseek4.1-flash", **overrides):
  return pyconfig.initialize(
      [sys.argv[0], "src/maxtext/configs/base.yml"],
      run_name="deepseek41_config_test",
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
      model_name=model_name,
      **overrides,
  )


class ConfigValidationTest(parameterized.TestCase):
  """`validate_deepseek41` rejects layouts the shared attention state cannot express."""

  @parameterized.named_parameters(
      ("wrong_length", {"compress_ratios": [0, 0, 2]}, "entries for 40 decoder layers"),
      # Layer 20 switches to ratio 1 but is no longer a KV source: nothing produces ratio-1 compressed KV.
      ("ratio_change_without_kv_source", {"ds41_kv_source_layers": [2, 8, 14]}, "reuses compressed KV"),
      ("candidates_smaller_than_topk", {"ds41_candidate_topk_blocks": 4, "indexer_topk": 64}, "must be >= indexer_topk"),
      ("candidate_source_not_index_source", {"ds41_candidate_source_layer": 21}, "must be one of"),
      ("sliding_window_kv_source", {"ds41_kv_source_layers": [0, 2, 8, 14, 20]}, "compress_ratio > 0"),
  )
  def test_rejects(self, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      _flash_config(override_model_config=True, **overrides)

  def test_rejects_packed_data(self):
    with self.assertRaisesRegex(ValueError, "deepseek41 does not support packed sequences yet: set packing=False"):
      _flash_config(packing=True, override_model_config=True)
    with self.assertRaisesRegex(ValueError, "deepseek41 does not support packed sequences yet: set packing=False"):
      _flash_config(dataset_type="hf", hf_path="dummy", packing=True, override_model_config=True)

  def test_rejects_engram_without_ngrams(self):
    with self.assertRaisesRegex(ValueError, "ds41_engram_max_ngram_size must be >= 2"):
      _flash_config(ds41_engram_max_ngram_size=1, override_model_config=True)

  def test_training_moe_balancing(self):
    from flax import nnx  # pylint: disable=import-outside-toplevel

    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg = _flash_config(model_name)
      self.assertEqual(cfg.routed_bias_update_rate, 0.001)
      self.assertEqual(cfg.load_balance_loss_weight, 0.0001)
    with self.assertRaisesRegex(ValueError, "routed_bias_update_rate must be >= 0"):
      _flash_config(routed_bias_update_rate=-0.001, override_model_config=True)
    with self.assertRaises(ValueError):
      _flash_config(load_balance_loss_weight=-0.0001, override_model_config=True)
    tokens = jnp.ones((1, _TINY_SEQ), jnp.int32)
    for scan_layers in (False, True):
      _, model = _tiny_model(scan_layers=scan_layers)
      _logits(model, tokens)
      intermediates = nnx.traversals.flatten_mapping(nnx.state(model, nnx.Intermediate).to_pure_dict())
      lb_keys = [k for k in intermediates if "moe_lb_loss" in k]
      bias_keys = [k for k in intermediates if "moe_bias_updates" in k]
      self.assertTrue(lb_keys, f"missing moe_lb_loss for scan_layers={scan_layers}")
      self.assertTrue(bias_keys, f"missing moe_bias_updates for scan_layers={scan_layers}")
      for k in lb_keys:
        self.assertTrue(np.isfinite(np.asarray(intermediates[k])).all(), f"non-finite {k}")
        self.assertGreater(float(np.sum(np.asarray(intermediates[k]))), 0.0, f"non-positive {k}")
      for k in bias_keys:
        self.assertTrue(np.isfinite(np.asarray(intermediates[k])).all(), f"non-finite {k}")

  def test_gate_matmul_precision(self):
    self.assertEqual(_flash_config().gate_matmul_precision, "highest")
    self.assertEqual(_flash_config(model_name="default").gate_matmul_precision, "")

  def test_muon_allowed(self):
    self.assertEqual(_flash_config(opt_type="muon").opt_type, "muon")

  def test_rejects_invalid_gate_matmul_precision_and_prefetch_fsdp_weights(self):
    with self.assertRaises(ValueError):
      _flash_config(gate_matmul_precision="not_a_real_precision", override_model_config=True)
    with self.assertRaises(ValueError):
      types.RLConfig(model_name="deepseek4.1-flash-mini", gate_matmul_precision="not_a_real_precision")
    self.assertEqual(
        _flash_config(gate_matmul_precision="high", override_model_config=True).gate_matmul_precision,
        types.MatmulPrecision.HIGH,
    )
    with self.assertRaisesRegex(ValueError, "prefetch_fsdp_weights"):
      _flash_config(prefetch_fsdp_weights=True, override_model_config=True)

  def test_megatron_seq_aux_loss_enabled_for_ds41_configs(self):
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      with self.subTest(model_name=model_name):
        cfg = _flash_config(model_name)
        self.assertTrue(cfg.moe_use_megatron_seq_aux_loss)
        self.assertTrue(moe.uses_megatron_seq_aux_loss(cfg))
        cfg_off = _flash_config(model_name, override_model_config=True, moe_use_megatron_seq_aux_loss=False)
        self.assertFalse(cfg_off.moe_use_megatron_seq_aux_loss)
        self.assertFalse(moe.uses_megatron_seq_aux_loss(cfg_off))


class CheckpointHookTest(parameterized.TestCase):
  """Engram-table row handling and quantized-save rejection in the DeepSeek-V4.1 hooks."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = _flash_config("deepseek4.1-flash-mini")  # ~250k-row engram tables
    cls.rows = engram_deepseek41.engram_num_embeddings(cls.config)
    cls.key = "params-decoder-layers_1-engram-embed"

  def _hook(self, saving_to_hf, quantized=False):
    hf_config = {"quantization_config": {"quant_method": "fp8"}} if quantized else {}
    hooks = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, self.config, saving_to_hf=saving_to_hf)
    return hooks[self.key]

  def test_engram_rows_round_trip(self):
    padded = engram_deepseek41.padded_engram_rows(self.rows[0])
    hf_table = np.ones((self.rows[0], 2), np.float32)
    loaded = self._hook(saving_to_hf=False)(hf_table, (padded, 2))
    self.assertEqual(loaded.shape, (padded, 2))
    np.testing.assert_array_equal(loaded[self.rows[0] :], 0)
    saved = self._hook(saving_to_hf=True)(loaded, (self.rows[0], 2))
    np.testing.assert_array_equal(saved, hf_table)

  def test_engram_rows_mismatch(self):
    padded = engram_deepseek41.padded_engram_rows(self.rows[0])
    with self.assertRaisesRegex(ValueError, "engram table in the checkpoint has"):
      self._hook(saving_to_hf=False)(np.ones((self.rows[0] - 1, 2), np.float32), (padded, 2))
    with self.assertRaisesRegex(ValueError, "engram table in the checkpoint has"):
      self._hook(saving_to_hf=False)(np.ones((padded + 1, 2), np.float32), (padded, 2))

  def test_quantized_save_rejected(self):
    with self.assertRaisesRegex(ValueError, "quantized"):
      self._hook(saving_to_hf=True, quantized=True)

  def test_quantized_load_dequantizes_then_pads(self):
    hooks = self._hook(saving_to_hf=False, quantized=True)
    self.assertIs(hooks[0], param_mapping.dequantize_block_scaled)
    self.assertTrue(getattr(hooks[1], "pads_rows_to_target", False))


class TokenMapTest(unittest.TestCase):

  def test_pads_to_vocab_with_pad_token(self):
    token_map = np.array([5, 3, 7, 1], np.int32)
    padded = engram_deepseek41.pad_token_map(token_map, 7, pad_id=2)
    np.testing.assert_array_equal(padded, [5, 3, 7, 1, 7, 7, 7])
    self.assertIs(engram_deepseek41.pad_token_map(token_map, 4, pad_id=2), token_map)


class MuonTest(unittest.TestCase):
  """Muon dimension numbers of the DeepSeek-V4.1 parameters."""

  def test_engram_tables_and_scales_excluded_only_under_engram(self):
    from maxtext.utils import muon_utils  # pylint: disable=import-outside-toplevel

    for name in ("embed", "q_weight", "k_weight"):
      self.assertIsNone(muon_utils.transform_logic(("decoder", "layers_1", "engram", name), (1024, 256)))
    self.assertIsNotNone(muon_utils.transform_logic(("decoder", "layers_1", "mlp", "embed"), (1024, 256)))

  def test_scanned_single_head_wkv_keeps_layer_axis_out(self):
    from maxtext.utils import muon_utils  # pylint: disable=import-outside-toplevel

    for path in (
        ("decoder", "segments_2", "layers_0", "self_attention", "wkv", "kernel"),
        ("decoder", "segments_2", "layers_0", "self_attention", "compressor", "wkv", "kernel"),
    ):
      dims = muon_utils.transform_logic(path, (5120, 2, 512), scan_layers=True)
      self.assertEqual((dims.reduction_axis, dims.output_axis), ((0,), (-1,)))
      dims = muon_utils.transform_logic(path[:1] + ("layers_2",) + path[3:], (5120, 512), scan_layers=False)
      self.assertEqual((dims.reduction_axis, dims.output_axis), ((0,), (-1,)))
    # Head-expanded projections keep their head axis in the output, scanned or not.
    q = ("decoder", "segments_2", "layers_0", "self_attention", "wq_b", "kernel")
    dims = muon_utils.transform_logic(q, (1280, 2, 64, 512), scan_layers=True)
    self.assertEqual((dims.reduction_axis, dims.output_axis), ((0,), (-2, -1)))

  def test_layers_all_stacked(self):
    from maxtext.utils import muon_utils  # pylint: disable=import-outside-toplevel

    self.assertTrue(muon_utils.layers_all_stacked(_flash_config(scan_layers=True)))
    self.assertFalse(muon_utils.layers_all_stacked(_flash_config(scan_layers=False)))
    self.assertIsNone(muon_utils.layers_all_stacked(None))


class RotaryTest(unittest.TestCase):

  def test_inv_freq_computed_on_host_in_float64(self):
    rope = attention_deepseek41.DeepSeek41RotaryEmbedding(
        head_dim=512,
        partial_rotary_factor=64 / 512,
        rope_theta=160000.0,
        rope_type="yarn",
        original_max_position_embeddings=65536,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
    )
    dim = 64
    freqs = 1.0 / (160000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    low = max(int(np.floor(dim * np.log(65536 / (32 * 2 * np.pi)) / (2 * np.log(160000.0)))), 0)
    high = min(int(np.ceil(dim * np.log(65536 / (1 * 2 * np.pi)) / (2 * np.log(160000.0)))), dim - 1)
    smooth = 1 - np.clip((np.arange(dim // 2) - low) / max(high - low, 1e-3), 0, 1)
    want = (freqs / 16 * (1 - smooth) + freqs * smooth).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(rope.inv_freq), want)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda: rope.inv_freq)()), want)


# A tiny torch-free model with the mini layout's sharing structure (no engram, so no tokenizer is needed): segments
# [0] [1] [2-3]x2 [6-7] [8-9]x2, with compressed KV, the indexer and candidate blocks all active.
_TINY_SEQ = 32
_TINY = {
    "override_model_config": True,
    "per_device_batch_size": 1.0,
    "max_target_length": _TINY_SEQ,
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "base_emb_dim": 64,
    "base_num_query_heads": 4,
    "head_dim": 32,
    "qk_rope_head_dim": 8,
    "q_lora_rank": 16,
    "o_groups": 2,
    "o_lora_rank": 8,
    "base_mlp_dim": 32,
    "base_moe_mlp_dim": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "sliding_window_size": 8,
    "base_num_decoder_layers": 12,
    "compress_ratios": [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    "ds41_kv_source_layers": [2, 4, 6],
    "ds41_index_source_layers": [2, 4, 6, 8, 10],
    "ds41_candidate_source_layer": 6,
    "ds41_candidate_topk_blocks": 2,
    "ds41_candidate_block_size": 4,
    "indexer_n_heads": 4,
    "indexer_head_dim": 16,
    "indexer_topk": 3,
    "ds41_engram_layers": [],
}


def _tiny_model(scan_layers):
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel

  config = _flash_config("deepseek4.1-flash-mini", scan_layers=scan_layers, **_TINY)
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
  return config, model_creation_utils.from_config(config, mesh=mesh, rngs=nnx.Rngs(0))


def _logits(model, tokens):
  return np.asarray(
      model(
          decoder_input_tokens=tokens,
          decoder_positions=jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape),
          decoder_segment_ids=jnp.ones_like(tokens),
          enable_dropout=False,
      )
  )


class TinyForwardTest(unittest.TestCase):
  """Forward passes of a tiny random DeepSeek-V4.1 model (runs in PR CI: needs neither PyTorch nor a tokenizer)."""

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.tokens = jnp.asarray(rng.integers(3, 1000, (1, _TINY_SEQ)), jnp.int32)

  def test_causal(self):
    """Changing token t leaves every earlier position's logits unchanged (window, compressed-KV, indexer and
    candidate masks)."""
    _, model = _tiny_model(scan_layers=False)
    base = _logits(model, self.tokens)
    self.assertTrue(np.isfinite(base).all())
    for t in (5, 17, _TINY_SEQ - 1):
      changed = _logits(model, self.tokens.at[:, t].set(self.tokens[0, t] + 1))
      np.testing.assert_allclose(changed[:, :t], base[:, :t], rtol=0, atol=1e-6, err_msg=f"position {t}")
      self.assertFalse(np.allclose(changed[:, t], base[:, t]), f"position {t} ignores its own token")

  def test_scanned_matches_unscanned(self):
    """The scanned segments (stacks, carry plans and placeholders) compute what the unscanned layers do."""
    from flax import nnx  # pylint: disable=import-outside-toplevel

    config, unscanned = _tiny_model(scan_layers=False)
    _, scanned = _tiny_model(scan_layers=True)
    weights = (nnx.Param, moe.MoEBiasVar)
    flat_u = nnx.traversals.flatten_mapping(nnx.state(unscanned, weights).to_pure_dict())
    scanned_state = nnx.state(scanned, weights)
    flat_s = dict(nnx.traversals.flatten_mapping(scanned_state.to_pure_dict()))
    # (segment, block layer, repeat) of every layer, as stacked by `build_segments`.
    where = {
        layer: (k, j, r)
        for k, segment in enumerate(deepseek41.build_segments(config))
        for r, group in enumerate(segment.groups)
        for j, layer in enumerate(group)
    }
    stacked = {}
    for path, value in flat_u.items():
      i = path.index("decoder") + 1 if "decoder" in path else -1
      if i < 0 or not path[i].startswith("layers_"):
        flat_s[path] = value  # embeddings, final norm, logits
        continue
      k, j, r = where[int(path[i][len("layers_") :])]
      key = path[:i] + (f"segments_{k}", f"layers_{j}") + path[i + 1 :]
      stacked.setdefault(key, {})[r] = np.asarray(value)
    for key, per_repeat in stacked.items():
      target = np.asarray(flat_s[key])
      parts = [per_repeat[r] for r in range(len(per_repeat))]
      # Layers stack on `param_scan_axis` (1); vectors (e.g. the routing bias) may keep the layer axis in front.
      axes = (1,) if parts[0].ndim > 1 else (1, 0)
      matching = [a for a in axes if np.stack(parts, axis=a).shape == target.shape]
      self.assertTrue(matching, key)
      flat_s[key] = np.stack(parts, axis=matching[0])
    nnx.replace_by_pure_dict(scanned_state, nnx.traversals.unflatten_mapping(flat_s))
    nnx.update(scanned, scanned_state)
    np.testing.assert_allclose(_logits(scanned, self.tokens), _logits(unscanned, self.tokens), rtol=1e-5, atol=1e-5)


class DecodeGuardTest(unittest.TestCase):

  def test_autoregressive_mode_rejected(self):
    from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE  # pylint: disable=import-outside-toplevel

    _, model = _tiny_model(scan_layers=False)
    tokens = jnp.ones((1, 1), jnp.int32)
    with self.assertRaisesRegex(NotImplementedError, "no KV cache yet"):
      model(
          decoder_input_tokens=tokens,
          decoder_positions=jnp.zeros_like(tokens),
          enable_dropout=False,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )


class RotaryAndCandidateBlockTest(unittest.TestCase):
  """Torch-free unit tests for DeepSeek-V4.1 rotary invertibility and candidate selection."""

  def test_rotary_invertibility_and_candidate_blocks(self):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 32, 3, 32), dtype=np.float32))
    positions = jnp.broadcast_to(jnp.arange(32, dtype=jnp.int32), (2, 32))
    for orig_len, theta in ((65536, 160000.0), (0, 10000.0)):
      rope = attention_deepseek41.DeepSeek41RotaryEmbedding(
          head_dim=32,
          partial_rotary_factor=8 / 32,
          rope_theta=theta,
          rope_type="yarn" if orig_len > 0 else "default",
          original_max_position_embeddings=orig_len if orig_len > 0 else 65536,
          rope_factor=16,
          beta_fast=32,
          beta_slow=1,
      )
      rotated = rope(x, positions, unsqueeze_dim=-2)
      np.testing.assert_array_equal(np.asarray(rotated[..., :-8]), np.asarray(x[..., :-8]))
      recovered = rope(rotated, positions, unsqueeze_dim=-2, reverse=True)
      np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), rtol=1e-5, atol=1e-5)

    raw_logits = rng.standard_normal((2, 16, 16), dtype=np.float32)
    lens_np = np.broadcast_to(np.arange(1, 17, dtype=np.int32)[:, None], (2, 16, 1))
    masked_logits = jnp.asarray(np.where(np.arange(16)[None, None, :] >= lens_np, -np.inf, raw_logits))
    lens = jnp.broadcast_to(jnp.arange(1, 17, dtype=jnp.int32), (2, 16))
    selected = attention_deepseek41.select_candidate_blocks(masked_logits, lens, topk_blocks=2, block_size=4)
    self.assertEqual(selected.shape, (2, 16, 16))
    self.assertEqual(selected.dtype, jnp.bool_)
    np.testing.assert_array_equal(np.asarray(selected[:, -1]).sum(axis=-1), np.array([8, 8]))


if __name__ == "__main__":
  unittest.main()
