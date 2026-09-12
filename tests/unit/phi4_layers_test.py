# Copyright 2026 Google LLC
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

"""Phi-4-mini reference parity and bidirectional fused-weight conversion."""

import unittest
from types import SimpleNamespace
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from transformers import Phi3Config, Phi3ForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding, apply_rotary_pos_emb
from maxtext.configs import pyconfig
from maxtext.layers.embeddings import LongRoPERotaryEmbedding
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.checkpoint_conversion.utils.param_mapping import (
    PHI4_MAXTEXT_TO_HF_PARAM_MAPPING,
    PHI4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
)


def hf_config():
  # Transformers generates this constructor dynamically.
  return Phi3Config(  # pylint: disable=unexpected-keyword-arg
      hidden_size=128,
      intermediate_size=256,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      vocab_size=256,
      pad_token_id=0,
      eos_token_id=255,
      bos_token_id=1,
      max_position_embeddings=131072,
      original_max_position_embeddings=4096,
      partial_rotary_factor=0.75,
      rms_norm_eps=1e-5,
      tie_word_embeddings=True,
      rope_scaling={"type": "longrope", "short_factor": [1.0] * 12, "long_factor": [2.0] * 12},
      attention_dropout=0.0,
      attn_implementation="eager",
      resid_pdrop=0.0,
      embd_pdrop=0.0,
  )


class Phi4LayersTest(unittest.TestCase):

  def test_partial_longrope_matches_hf(self):
    config = hf_config()
    reference = Phi3RotaryEmbedding(config)
    rope = LongRoPERotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=Mesh(jax.devices(), "data"),
        embedding_dims=32,
        partial_rotary_factor=0.75,
        short_factor=[1.0] * 12,
        long_factor=[2.0] * 12,
        original_max_position_embeddings=4096,
        max_position_embeddings=131072,
        fprop_dtype=jnp.float32,
    )
    x = np.random.default_rng(42).normal(size=(2, 4, 4, 32)).astype(np.float32)
    # Short, exact boundary, long, single-token decode, then reset to short.
    for positions in ([0, 1, 2, 3], [4092, 4093, 4094, 4095], [0, 1, 4095, 4096], [131071], [0, 1]):
      with self.subTest(positions=positions):
        pos = np.tile(positions, (2, 1))
        inputs = x[:, : len(positions)]
        tx = torch.tensor(inputs).transpose(1, 2)
        cos, sin = reference(tx, torch.tensor(pos))
        expected, _ = apply_rotary_pos_emb(tx, tx, cos, sin)
        actual = jax.jit(rope)(jnp.asarray(inputs), jnp.asarray(pos))
        np.testing.assert_allclose(
            actual, expected.transpose(1, 2).numpy(), atol=1e-3 if max(positions) > 100000 else 1e-4, rtol=1e-4
        )
        np.testing.assert_array_equal(actual[..., 24:], inputs[..., 24:])

  def test_longrope_ignores_padding_for_regime_selection(self):
    rope = LongRoPERotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=Mesh(jax.devices(), "data"),
        embedding_dims=32,
        partial_rotary_factor=0.75,
        short_factor=[1.0] * 12,
        long_factor=[2.0] * 12,
        original_max_position_embeddings=4096,
        max_position_embeddings=131072,
        fprop_dtype=jnp.float32,
    )
    inputs = jnp.ones((1, 4, 2, 32))
    positions = jnp.array([[0, 1, 4096, 4097]])
    padded = rope(inputs, positions, max_position=jnp.array(1))
    unpadded = rope(inputs[:, :2], positions[:, :2])
    np.testing.assert_array_equal(padded[:, :2], unpadded)

  def test_fused_projection_roundtrip(self):
    config = hf_config().to_dict()
    mt = SimpleNamespace(num_decoder_layers=2, head_dim=32, num_query_heads=4, num_kv_heads=2, mlp_dim=256)
    rng = np.random.default_rng(0)
    for scan in (False, True):
      mapping = PHI4_MAXTEXT_TO_HF_PARAM_MAPPING(config, mt, scan)
      imports = PHI4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, mt, scan)
      exports = PHI4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, mt, scan, saving_to_hf=True)
      self.assertNotIn("params-decoder-logits_dense-kernel", mapping)
      for keys, export in exports.items():
        if not isinstance(keys, tuple):
          continue
        qkv = "self_attention" in keys[0]
        shape = (256, 128) if qkv else (512, 128)
        original = rng.normal(size=shape).astype(np.float32)
        shapes = [(128, 4, 32), (128, 2, 32), (128, 2, 32)] if qkv else [(128, 256)] * 2
        converted = [imports[key](original, target) for key, target in zip(keys, shapes)]
        np.testing.assert_allclose(export(converted, shape), original, rtol=2e-7, atol=2e-7)
        for key in keys:
          self.assertEqual(isinstance(mapping[key], list), scan)
          self.assertEqual(mapping[key], mapping[keys[0]])

  def test_decoder_logits_match_hf(self):
    """Copied HF weights exercise norms, residuals, GQA, SwiGLU and tied logits."""
    torch.manual_seed(7)
    reference = Phi3ForCausalLM(hf_config()).eval()
    state = {key: value.detach().numpy() for key, value in reference.state_dict().items()}
    for scan in (False, True):
      with self.subTest(scan_layers=scan):
        cfg = pyconfig.initialize(
            ["", "src/maxtext/configs/base.yml"],
            model_name="phi4-mini-instruct",
            override_model_config=True,
            hardware="cpu",
            skip_jax_distributed_system=True,
            run_name="phi4-test",
            enable_checkpointing=False,
            base_num_decoder_layers=2,
            base_emb_dim=128,
            base_num_query_heads=4,
            base_num_kv_heads=2,
            head_dim=32,
            base_mlp_dim=256,
            vocab_size=256,
            longrope_short_factor=[1.0] * 12,
            longrope_long_factor=[2.0] * 12,
            max_target_length=32,
            max_prefill_predict_length=4,
            per_device_batch_size=1.0,
            attention="dot_product",
            dtype="float32",
            weight_dtype="float32",
            matmul_precision="highest",
            scan_layers=scan,
            float32_logits=True,
            float32_qk_product=True,
            activations_in_float32=True,
        )
        mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
        model = models.transformer_as_linen(cfg, mesh, None)
        ids = jnp.array([[1, 7, 19, 3]])
        pos = jnp.arange(4)[None, :]
        with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
          variables = model.init(jax.random.key(0), ids, pos, jnp.ones_like(ids), enable_dropout=False)
          flat = flatten_dict(variables)
          mapping = PHI4_MAXTEXT_TO_HF_PARAM_MAPPING(reference.config.to_dict(), cfg, scan)
          hooks = PHI4_MAXTEXT_TO_HF_PARAM_HOOK_FN(reference.config.to_dict(), cfg, scan)
          for path, leaf in flat.items():
            if path[0] != "params":
              continue
            key = "-".join(path)
            value = leaf.unbox() if hasattr(leaf, "unbox") else leaf
            sources = mapping[key]
            hook = hooks.get(key, lambda x, shape: x)
            if isinstance(sources, list):
              shape = list(value.shape)
              shape.pop(cfg.param_scan_axis)
              converted = np.stack([hook(state[source], tuple(shape)) for source in sources], axis=cfg.param_scan_axis)
            else:
              converted = hook(state[sources], value.shape)
            flat[path] = (
                leaf.replace_boxed(jnp.asarray(converted)) if hasattr(leaf, "replace_boxed") else jnp.asarray(converted)
            )
          variables = unflatten_dict(flat)
          for offset in (0, 4094):
            positions = pos + offset
            actual = model.apply(variables, ids, positions, jnp.ones_like(ids), enable_dropout=False)
            with torch.no_grad():
              expected = reference(torch.tensor(np.array(ids)), position_ids=torch.tensor(np.array(positions))).logits
            np.testing.assert_allclose(actual, expected.numpy(), atol=2e-4, rtol=2e-4)
          # A padded prefill must not switch the real short prompt to long factors.
          positions = jnp.array([[0, 1, 4096, 4097]])
          segments = jnp.array([[1, 1, 0, 0]])
          actual = model.apply(variables, ids, positions, segments, enable_dropout=False)
          with torch.no_grad():
            expected = reference(torch.tensor(np.array(ids[:, :2]))).logits
          np.testing.assert_allclose(actual[:, :2], expected.numpy(), atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
  unittest.main()
