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

# pylint: disable=protected-access
"""Mini-verification suite for DeepSeek-V4.1-Flash LLM backbone on CPU.

Covers:
1. Out-of-the-box config initialization for deepseek4.1-flash and deepseek4.1-flash-mini.
2. Activation quantization (ds41_activation_quant) FP8/FP4 forward values, STE gradients, and MoE path guard.
3. Non-multiple sequence length boundary handling (S % compress_ratio != 0) against the PyTorch reference.
4. Right-padded sequence isolation (decoder_segment_ids == 0 pad tokens do not affect valid token logits).
5. Packed multi-document isolation (decoder_segment_ids == 1, 2 match standalone document runs across all layer types).
6. Individual loss terms (lm_loss, moe_lb_loss), MoEBiasVar updates, scan_layers={False,True} train_step equivalence,
   and multi-step monotonic loss decrease via real train.py functions.
"""

import shutil
import sys
import tempfile
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import deepseek41_quant
from maxtext.models import deepseek41
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
import numpy as np
import optax
import pytest
from typing import Any
from tests.unit import deepseek_v41_vs_reference_test as base_test

torch: Any = base_test.torch
ref_model: Any = base_test.ref_model
ref_runner: Any = base_test.ref_runner


def setUpModule():
  jax.config.update("jax_default_matmul_precision", "highest")


@pytest.mark.cpu_only
class DeepSeek41ConfigAndQuantTest(unittest.TestCase):
  """Tests default YAML initialization and ds41_activation_quant STE/guards."""

  def test_default_yaml_initialization_without_overrides(self):
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              f"model_name={model_name}",
              "run_name=cfg_init_test",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertEqual(cfg.decoder_block.value, "deepseek41")
      self.assertFalse(cfg.sparse_matmul)

  def test_activation_quant_ste_and_guard(self):
    x = jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).reshape(2, 32)
    q8 = deepseek41_quant.fp8_qdq(x, block_size=32)
    q4 = deepseek41_quant.fp4_qdq(x, block_size=32, e4m3_scale=False)
    q4_e4m3 = deepseek41_quant.fp4_qdq(x, block_size=16, e4m3_scale=True)
    self.assertEqual(q8.shape, x.shape)
    self.assertEqual(q4.shape, x.shape)
    self.assertEqual(q4_e4m3.shape, x.shape)

    # Straight-through estimator passes exact unit gradients.
    g8 = jax.grad(lambda t: jnp.sum(deepseek41_quant.fp8_qdq(t, block_size=32)))(x)
    g4 = jax.grad(lambda t: jnp.sum(deepseek41_quant.fp4_qdq(t, block_size=32)))(x)
    np.testing.assert_allclose(np.asarray(g8), np.ones_like(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(g4), np.ones_like(x), rtol=1e-6, atol=1e-6)

    with self.assertRaisesRegex(ValueError, "ds41_activation_quant"):
      base_test._maxtext_config(ds41_activation_quant=True, sparse_matmul=True)


@pytest.mark.cpu_only
@unittest.skipUnless(base_test.HAS_TORCH, "PyTorch not installed")
class DeepSeek41OddSeqLenReferenceTest(unittest.TestCase):
  """Tests non-multiple sequence length boundary handling against the PyTorch reference."""

  @classmethod
  def setUpClass(cls):
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    torch.manual_seed(7)
    cls.tokenizer = AutoTokenizer.from_pretrained(base_test.TOKENIZER)
    cls.ref = ref_runner.build_model(ref_model.ModelArgs(**base_test.REF_ARGS), cls.tokenizer)
    base_test._init_reference(cls.ref, seed=7)

  def test_non_multiple_sequence_length_matches_reference(self):
    # S = 35 is odd, so ratio-2 compressor has a trailing 1-token partial group that must be dropped.
    seq_odd = 35
    config = base_test._maxtext_config(max_target_length=seq_odd)
    ids = torch.randint(0, base_test.REF_ARGS["vocab_size"], (2, seq_odd))
    with torch.inference_mode():
      hashes = self.ref.engram_hash(ids, 0)
      h = self.ref.embed(ids).unsqueeze(2).repeat(1, 1, base_test.REF_ARGS["hc_mult"], 1)
      pre_mix = ref_model.make_identity_pre_mix(h, base_test.REF_ARGS["hc_mult"])
      for layer in self.ref.layers:
        if layer.engram is not None:
          h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
        h, pre_mix = layer(h, 0, pre_mix, None)
      ref_collapsed = self.ref.layers[-1].hc_pre(h, pre_mix).numpy()

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    with mesh:
      specs = deepseek41.build_layer_specs(config)
      layers = []
      for i, spec in enumerate(specs):
        layer = deepseek41.DeepSeek41DecoderLayer(config, mesh, "train", spec, rngs=nnx.Rngs(i))
        base_test.copy_layer(layer, self.ref.layers[i])
        layers.append(layer)
      tokens = jnp.asarray(ids.numpy(), jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_odd, dtype=jnp.int32), (2, seq_odd))
      seg_ids = jnp.ones((2, seq_odd), dtype=jnp.int32)
      carry = deepseek41.initial_carry(base_test._t2j(self.ref.embed(ids)), base_test.REF_ARGS["hc_mult"])
      for layer in layers:
        carry = layer(carry, positions, tokens, decoder_segment_ids=seg_ids)
      mt_collapsed = base_test._j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry["h"], carry["pre_mix"]))
    np.testing.assert_allclose(mt_collapsed, ref_collapsed, rtol=2e-3, atol=2e-3)


@pytest.mark.cpu_only
class DeepSeek41ChallengingInputsTest(unittest.TestCase):
  """Torch-free tests for odd sequence lengths, right-padding isolation, and packed document isolation."""

  @classmethod
  def setUpClass(cls):
    rng = np.random.default_rng(7)
    cls.embed_table = rng.standard_normal((base_test.MINI_VOCAB_SIZE, base_test.REF_ARGS["dim"])).astype(np.float32)

  def _cfg(self, max_target_length: int, **overrides):
    return base_test._maxtext_config(
        vocab_size=base_test.MINI_VOCAB_SIZE,
        max_target_length=max_target_length,
        tokenizer_path=base_test.OFFLINE_TOKENIZER,
        ds41_engram_compressed_vocab_size=0,
        **overrides,
    )

  def _build_unscanned_layers(self, config, mesh):
    """Builds unscanned DeepSeek-V4.1 decoder layers with non-zero Engram embeddings."""
    specs = deepseek41.build_layer_specs(config)
    layers = []
    for i, spec in enumerate(specs):
      layer = deepseek41.DeepSeek41DecoderLayer(config, mesh, "train", spec, rngs=nnx.Rngs(100 + i))
      if spec.has_engram:
        engram_rng = np.random.default_rng(200 + i)
        layer.engram.embed[...] = jnp.asarray(
            engram_rng.normal(0.0, 0.2, size=layer.engram.embed[...].shape).astype(np.float32)
        )
      layers.append(layer)
    return layers

  def _embed(self, ids_np: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(self.embed_table[ids_np])

  def test_non_multiple_sequence_length_scanned_vs_unscanned(self):
    ckpt_dir = tempfile.mkdtemp(prefix="dsv41_odd_seq_")
    self.addCleanup(shutil.rmtree, ckpt_dir, True)
    base_test.write_numpy_mini_checkpoint(ckpt_dir, seed=5)

    seq_odd = 35
    rng = np.random.default_rng(17)
    tokens = jnp.asarray(rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(2, seq_odd), dtype=np.int32))
    positions = jnp.broadcast_to(jnp.arange(seq_odd, dtype=jnp.int32), tokens.shape)
    seg_ids = jnp.ones_like(tokens)
    outs = []
    for scan_layers in (False, True):
      cfg = self._cfg(
          seq_odd,
          base_num_decoder_layers=base_test.MINI_LAYOUT["n_layers"],
          compress_ratios=base_test.MINI_COMPRESS_RATIOS,
          kv_source_layers=list(base_test.MINI_LAYOUT["kv_source_layers"]),
          index_source_layers=list(base_test.MINI_LAYOUT["index_source_layers"]),
          candidate_source_layer=base_test.MINI_LAYOUT["candidate_source_layer"],
          ds41_engram_layers=list(base_test.MINI_LAYOUT["engram_layer_ids"]),
          scan_layers=scan_layers,
          load_parameters_path=ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
      )
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      logits = model(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg_ids,
          enable_dropout=False,
      )
      outs.append(base_test._j2np(logits))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-5, atol=1e-5)

  def test_right_padding_isolation(self):
    valid_len = 24
    padded_len = 40
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    cfg_valid = self._cfg(valid_len)
    cfg_padded = self._cfg(padded_len)

    rng = np.random.default_rng(21)
    ids_valid = rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(1, valid_len), dtype=np.int32)
    pad_tokens_a = rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(1, padded_len - valid_len), dtype=np.int32)
    pad_tokens_b = rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(1, padded_len - valid_len), dtype=np.int32)
    ids_pad_a = np.concatenate([ids_valid, pad_tokens_a], axis=1)
    ids_pad_b = np.concatenate([ids_valid, pad_tokens_b], axis=1)

    with mesh:
      layers_valid = self._build_unscanned_layers(cfg_valid, mesh)
      layers_padded = self._build_unscanned_layers(cfg_padded, mesh)

      # 1. Unpadded forward
      tok_v = jnp.asarray(ids_valid, jnp.int32)
      pos_v = jnp.arange(valid_len, dtype=jnp.int32)[None, :]
      seg_v = jnp.ones((1, valid_len), dtype=jnp.int32)
      carry_v = deepseek41.initial_carry(self._embed(ids_valid), base_test.REF_ARGS["hc_mult"])
      for layer in layers_valid:
        carry_v = layer(carry_v, pos_v, tok_v, decoder_segment_ids=seg_v)
      out_v = base_test._j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry_v["h"], carry_v["pre_mix"]))

      # 2. Padded forward with two different random pad tails
      pos_p = jnp.concatenate(
          [jnp.arange(valid_len, dtype=jnp.int32), jnp.zeros(padded_len - valid_len, dtype=jnp.int32)]
      )[None, :]
      seg_p = jnp.concatenate([jnp.ones(valid_len, dtype=jnp.int32), jnp.zeros(padded_len - valid_len, dtype=jnp.int32)])[
          None, :
      ]
      outs_p = []
      for ids_p in (ids_pad_a, ids_pad_b):
        tok_p = jnp.asarray(ids_p, jnp.int32)
        carry_p = deepseek41.initial_carry(self._embed(ids_p), base_test.REF_ARGS["hc_mult"])
        for layer in layers_padded:
          carry_p = layer(carry_p, pos_p, tok_p, decoder_segment_ids=seg_p)
        outs_p.append(base_test._j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry_p["h"], carry_p["pre_mix"])))

    np.testing.assert_allclose(outs_p[0][:, :valid_len], out_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outs_p[0][:, :valid_len], outs_p[1][:, :valid_len], rtol=1e-6, atol=1e-6)

  def test_packed_multi_document_isolation(self):
    # Doc 1: length 16 (aligned to compress_ratio=2 and candidate_block_size*ratio = 4*2 = 8).
    # Doc 2: length 24. Packed length = 40.
    len1, len2 = 16, 24
    packed_len = len1 + len2
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    cfg1 = self._cfg(len1)
    cfg2 = self._cfg(len2)
    cfg_packed = self._cfg(packed_len)

    rng = np.random.default_rng(29)
    ids1 = rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(1, len1), dtype=np.int32)
    ids1_alt = rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(1, len1), dtype=np.int32)
    ids2 = rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(1, len2), dtype=np.int32)
    ids_packed = np.concatenate([ids1, ids2], axis=1)
    ids_packed_alt = np.concatenate([ids1_alt, ids2], axis=1)

    with mesh:
      layers1 = self._build_unscanned_layers(cfg1, mesh)
      layers2 = self._build_unscanned_layers(cfg2, mesh)
      layers_packed = self._build_unscanned_layers(cfg_packed, mesh)

      def run_standalone(layers, ids_np, length):
        tok = jnp.asarray(ids_np, jnp.int32)
        pos = jnp.arange(length, dtype=jnp.int32)[None, :]
        seg = jnp.ones((1, length), dtype=jnp.int32)
        carry = deepseek41.initial_carry(self._embed(ids_np), base_test.REF_ARGS["hc_mult"])
        for layer in layers:
          carry = layer(carry, pos, tok, decoder_segment_ids=seg)
        return base_test._j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry["h"], carry["pre_mix"]))

      out1 = run_standalone(layers1, ids1, len1)
      out2 = run_standalone(layers2, ids2, len2)

      pos_packed = jnp.concatenate([jnp.arange(len1, dtype=jnp.int32), jnp.arange(len2, dtype=jnp.int32)])[None, :]
      seg_packed = jnp.concatenate([jnp.ones(len1, dtype=jnp.int32), jnp.full(len2, 2, dtype=jnp.int32)])[None, :]

      def run_packed(ids_p):
        tok = jnp.asarray(ids_p, jnp.int32)
        carry = deepseek41.initial_carry(self._embed(ids_p), base_test.REF_ARGS["hc_mult"])
        for layer in layers_packed:
          carry = layer(carry, pos_packed, tok, decoder_segment_ids=seg_packed)
        return base_test._j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry["h"], carry["pre_mix"]))

      out_packed = run_packed(ids_packed)
      out_packed_alt = run_packed(ids_packed_alt)

    np.testing.assert_allclose(out_packed[:, :len1], out1, rtol=1e-4, atol=5e-5)
    np.testing.assert_allclose(out_packed[:, len1:], out2, rtol=1e-4, atol=5e-5)
    # Changing Doc 1 tokens does not perturb Doc 2 outputs at all.
    np.testing.assert_allclose(out_packed[:, len1:], out_packed_alt[:, len1:], rtol=1e-6, atol=1e-6)


@pytest.mark.cpu_only
class DeepSeek41TrainingAndMiniVerificationTest(unittest.TestCase):
  """Tests real train.py loss_fn, moe_lb_loss, MoEBiasVar updates, scan_layers parity, and multi-step training."""

  @classmethod
  def setUpClass(cls):
    cls.ckpt_dir = tempfile.mkdtemp(prefix="dsv41_train_verify_")
    _, cls.init_gate_biases = base_test.write_numpy_mini_checkpoint(cls.ckpt_dir, seed=11)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.ckpt_dir, ignore_errors=True)
    super().tearDownClass()

  def _make_config(self, scan_layers: bool, **overrides):
    """Constructs a MaxTextConfig for DeepSeek-V4.1 mini training."""
    cfg = base_test._maxtext_config(
        vocab_size=base_test.MINI_VOCAB_SIZE,
        base_num_decoder_layers=base_test.MINI_LAYOUT["n_layers"],
        compress_ratios=base_test.MINI_COMPRESS_RATIOS,
        kv_source_layers=list(base_test.MINI_LAYOUT["kv_source_layers"]),
        index_source_layers=list(base_test.MINI_LAYOUT["index_source_layers"]),
        candidate_source_layer=base_test.MINI_LAYOUT["candidate_source_layer"],
        ds41_engram_layers=list(base_test.MINI_LAYOUT["engram_layer_ids"]),
        ds41_engram_compressed_vocab_size=0,
        tokenizer_path=base_test.OFFLINE_TOKENIZER,
        scan_layers=scan_layers,
        load_parameters_path=self.ckpt_dir,
        source_checkpoint_layout="safetensors_dynamic",
        load_balance_loss_weight=0.0001,
        routed_bias=True,
        routed_bias_update_rate=0.001,
        learning_rate=1e-3,
        gradient_clipping_threshold=1.0,
        **overrides,
    )
    object.__setattr__(cfg, "model_name", "deepseek4.1-flash-mini")
    return cfg

  def _build_train_state(self, config):
    """Builds TrainStateNNX and shardings on the target mesh."""
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    tx = optax.adamw(learning_rate=config.learning_rate)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)
    graphdef, pure_state = nnx.split(state)
    state_shardings = jax.tree.map(lambda x: None, pure_state)
    params_shardings = jax.tree.map(lambda x: None, nnx.state(model, nnx.Param))
    return graphdef, pure_state, state_shardings, params_shardings, mesh

  def test_scan_vs_unscanned_train_step_and_aux_losses(self):
    rng = np.random.default_rng(42)
    tokens = jnp.asarray(rng.integers(0, base_test.MINI_VOCAB_SIZE, size=(2, base_test.SEQ), dtype=np.int32))
    targets = jnp.roll(tokens, -1, axis=1)
    positions = jnp.broadcast_to(jnp.arange(base_test.SEQ, dtype=jnp.int32), tokens.shape)
    seg_ids = jnp.ones_like(tokens)
    batch = {
        "inputs": tokens,
        "inputs_position": positions,
        "inputs_segmentation": seg_ids,
        "targets": targets,
        "targets_position": positions,
        "targets_segmentation": seg_ids,
    }

    cfg_unscanned = self._make_config(scan_layers=False)
    cfg_scanned = self._make_config(scan_layers=True)

    g_un, s_un, ss_un, ps_un, mesh_un = self._build_train_state(cfg_unscanned)
    g_sc, s_sc, ss_sc, ps_sc, mesh_sc = self._build_train_state(cfg_scanned)

    with mesh_un:
      s_un_next, metrics_un = train.train_step(g_un, cfg_unscanned, ss_un, ps_un, s_un, batch)
    with mesh_sc:
      s_sc_next, metrics_sc = train.train_step(g_sc, cfg_scanned, ss_sc, ps_sc, s_sc, batch)

    loss_un = float(metrics_un["scalar"]["learning/loss"])
    loss_sc = float(metrics_sc["scalar"]["learning/loss"])
    lb_un = float(metrics_un["scalar"]["learning/moe_lb_loss"])
    lb_sc = float(metrics_sc["scalar"]["learning/moe_lb_loss"])
    gnorm_un = float(metrics_un["scalar"]["learning/grad_norm"])
    gnorm_sc = float(metrics_sc["scalar"]["learning/grad_norm"])

    # 1. Individual loss terms must be finite and non-zero.
    self.assertTrue(np.isfinite(loss_un) and loss_un > 0.0)
    self.assertTrue(np.isfinite(lb_un) and lb_un > 0.0, f"Expected non-zero moe_lb_loss, got {lb_un}")
    self.assertTrue(np.isfinite(gnorm_un) and gnorm_un > 0.0)

    # 2. Scanned and unscanned loss, moe_lb_loss, and grad_norm must match closely.
    np.testing.assert_allclose(loss_sc, loss_un, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lb_sc, lb_un, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gnorm_sc, gnorm_un, rtol=1e-4, atol=1e-4)

    # 3. Verify MoEBiasVar was actually updated and matches layer-by-layer between unscanned and scanned.
    m_un = nnx.merge(g_un, s_un_next).model
    m_sc = nnx.merge(g_sc, s_sc_next).model
    unscanned_biases = [
        np.asarray(getattr(m_un.decoder, f"layers_{i}").mlp.MoeBlock_0.gate.bias.value)
        for i in range(cfg_unscanned.num_decoder_layers)
    ]
    scanned_biases = []
    for k, seg in enumerate(deepseek41.build_segments(cfg_scanned)):
      block = getattr(m_sc.decoder, f"segments_{k}")
      for r in range(seg.repeats):
        for j in range(seg.block_size):
          b_val = np.asarray(getattr(block, f"layers_{j}").mlp.MoeBlock_0.gate.bias.value)
          scanned_biases.append(b_val[r])

    self.assertEqual(len(unscanned_biases), len(scanned_biases))
    for i, (b_u, b_s) in enumerate(zip(unscanned_biases, scanned_biases)):
      np.testing.assert_allclose(b_s, b_u, rtol=1e-6, atol=1e-6, err_msg=f"MoEBiasVar mismatch at layer {i}")
      orig_b = self.init_gate_biases[i]
      self.assertGreater(np.max(np.abs(b_u - orig_b)), 0.0, f"MoEBiasVar did not update at layer {i}")

    # 4. Run 2 more steps on the scanned model and verify monotonic loss reduction on the fixed batch.
    with mesh_sc:
      s_sc_2, m2 = train.train_step(g_sc, cfg_scanned, ss_sc, ps_sc, s_sc_next, batch)
      _, m3 = train.train_step(g_sc, cfg_scanned, ss_sc, ps_sc, s_sc_2, batch)
    loss_2 = float(m2["scalar"]["learning/loss"])
    loss_3 = float(m3["scalar"]["learning/loss"])
    self.assertLess(loss_2, loss_sc)
    self.assertLess(loss_3, loss_2)


if __name__ == "__main__":
  unittest.main()
