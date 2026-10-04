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

"""Unit tests comparing MaxText Laya JAX/NNX layers against HuggingFace ModernBERT and Laya PyTorch references."""

import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from laya.common import DecisionModel
import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoModel

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import nnx_decoders
from maxtext.layers.embeddings import Embed
from maxtext.models import laya
from maxtext.utils import maxtext_utils


MINI_CKPT_DIR = "/dev/shm/hf_mini/laya_4layers"


def _get_config_and_mesh(extra_args=None):
  """Initializes MaxText config and device mesh for Laya unit tests."""
  args = [
      None,
      "src/maxtext/configs/base.yml",
      "model_name=laya",
      "override_model_config=True",
      "base_num_decoder_layers=4",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "float32_qk_product=True",
      "float32_logits=True",
      "scan_layers=False",
  ]
  if extra_args:
    args.extend(extra_args)
  cfg = pyconfig.initialize(args)
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  return cfg, mesh


class LayaLayersTest(unittest.TestCase):
  """Layer-by-layer numerical equivalence tests for Laya in MaxText."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.cfg, cls.mesh = _get_config_and_mesh()
    cls.has_mini_ckpt = os.path.exists(os.path.join(MINI_CKPT_DIR, "model.safetensors"))
    if cls.has_mini_ckpt:
      cls.st_weights = {}
      with safe_open(os.path.join(MINI_CKPT_DIR, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
          cls.st_weights[k] = f.get_tensor(k).float()
      cls.hf_config = AutoConfig.from_pretrained(MINI_CKPT_DIR)
      cls.hf_config._attn_implementation = "eager"  # pylint: disable=protected-access
      cls.hf_encoder = AutoModel.from_config(cls.hf_config).eval().float()
      enc_sd = {k[len("encoder.") :]: v for k, v in cls.st_weights.items() if k.startswith("encoder.")}
      cls.hf_encoder.load_state_dict(enc_sd, strict=True)

  def test_layer_norm(self):
    """Tests LayaLayerNorm with and without bias against torch.nn.LayerNorm."""
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 16, 1024)).astype(np.float32)
    w_np = rng.standard_normal((1024,)).astype(np.float32)
    b_np = rng.standard_normal((1024,)).astype(np.float32)

    # Without bias
    pt_ln_nobias = torch.nn.LayerNorm(1024, eps=1e-5, bias=False).eval()
    with torch.no_grad():
      pt_ln_nobias.weight.copy_(torch.from_numpy(w_np))
      pt_out_nobias = pt_ln_nobias(torch.from_numpy(x_np)).numpy()

    jax_ln_nobias = laya.LayaLayerNorm(
        num_features=1024,
        epsilon=1e-5,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        use_bias=False,
        rngs=nnx.Rngs(params=0),
    )
    jax_ln_nobias.scale[...] = jnp.asarray(w_np)
    jax_out_nobias = np.asarray(jax_ln_nobias(jnp.asarray(x_np)))
    np.testing.assert_allclose(jax_out_nobias, pt_out_nobias, rtol=1e-5, atol=1e-5)

    # With bias
    pt_ln_bias = torch.nn.LayerNorm(1024, eps=1e-5, bias=True).eval()
    with torch.no_grad():
      pt_ln_bias.weight.copy_(torch.from_numpy(w_np))
      pt_ln_bias.bias.copy_(torch.from_numpy(b_np))
      pt_out_bias = pt_ln_bias(torch.from_numpy(x_np)).numpy()

    jax_ln_bias = laya.LayaLayerNorm(
        num_features=1024,
        epsilon=1e-5,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        use_bias=True,
        rngs=nnx.Rngs(params=0),
    )
    jax_ln_bias.scale[...] = jnp.asarray(w_np)
    jax_ln_bias.bias[...] = jnp.asarray(b_np)
    jax_out_bias = np.asarray(jax_ln_bias(jnp.asarray(x_np)))
    np.testing.assert_allclose(jax_out_bias, pt_out_bias, rtol=1e-5, atol=1e-5)

  def test_mlp_block(self):
    """Tests LayaMLP (GeGLU) against HuggingFace ModernBertMLP."""
    if not self.has_mini_ckpt:
      self.skipTest("Mini checkpoint not found")
    rng = np.random.default_rng(123)
    x_np = rng.standard_normal((2, 12, 1024)).astype(np.float32)

    pt_mlp = self.hf_encoder.layers[0].mlp
    with torch.no_grad():
      pt_out = pt_mlp(torch.from_numpy(x_np)).numpy()

    jax_mlp = laya.LayaMLP(config=self.cfg, rngs=nnx.Rngs(params=0))
    jax_mlp.Wi.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.0.mlp.Wi.weight"].numpy().T)
    jax_mlp.Wo.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.0.mlp.Wo.weight"].numpy().T)

    jax_out = np.asarray(jax_mlp(jnp.asarray(x_np), deterministic=True))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-4, atol=1e-4)

  def test_decoder_layers_full_and_sliding(self):
    """Tests LayaDecoderLayer for layer 0 (full attention, no attn_norm) and layer 1 (sliding attention, with attn_norm)."""
    if not self.has_mini_ckpt:
      self.skipTest("Mini checkpoint not found")
    rng = np.random.default_rng(7)
    # Use seq_len=160 > sliding_window (128) so sliding window masking is actively exercised!
    batch_size, seq_len = 2, 160
    input_ids_np = rng.integers(10, 5000, size=(batch_size, seq_len), dtype=np.int32)
    attn_mask_np = np.ones((batch_size, seq_len), dtype=np.int32)
    attn_mask_np[1, 140:] = 0  # Pad last 20 tokens of batch item 1
    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (batch_size, seq_len)).copy()

    captured = {}

    def make_hook(idx):
      def _hook(_module, args, output):
        inp = args[0].detach().cpu().numpy()
        out = (output[0] if isinstance(output, tuple) else output).detach().cpu().numpy()
        captured[idx] = (inp, out)

      return _hook

    h0 = self.hf_encoder.layers[0].register_forward_hook(make_hook(0))
    h1 = self.hf_encoder.layers[1].register_forward_hook(make_hook(1))
    try:
      with torch.no_grad():
        _ = self.hf_encoder(
            input_ids=torch.from_numpy(input_ids_np).long(),
            attention_mask=torch.from_numpy(attn_mask_np).long(),
        )
    finally:
      h0.remove()
      h1.remove()

    in0_np, pt_out0 = captured[0]
    in1_np, pt_out1 = captured[1]

    # JAX Layer 0
    jax_layer0 = laya.LayaDecoderLayer(config=self.cfg, mesh=self.mesh, layer_idx=0, rngs=nnx.Rngs(params=0))
    self.assertIsNone(jax_layer0.attn_norm)
    jax_layer0.attn.Wqkv.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.0.attn.Wqkv.weight"].numpy().T)
    jax_layer0.attn.Wo.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.0.attn.Wo.weight"].numpy().T)
    jax_layer0.mlp_norm.scale[...] = jnp.asarray(self.st_weights["encoder.layers.0.mlp_norm.weight"].numpy())
    jax_layer0.mlp.Wi.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.0.mlp.Wi.weight"].numpy().T)
    jax_layer0.mlp.Wo.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.0.mlp.Wo.weight"].numpy().T)

    jax_out0, _ = jax_layer0(
        jnp.asarray(in0_np),
        decoder_segment_ids=jnp.asarray(attn_mask_np),
        decoder_positions=jnp.asarray(pos_np),
        deterministic=True,
    )
    for b in range(batch_size):
      valid_len = int(attn_mask_np[b].sum())
      np.testing.assert_allclose(np.asarray(jax_out0[b, :valid_len]), pt_out0[b, :valid_len], rtol=1e-4, atol=1e-4)

    # JAX Layer 1
    jax_layer1 = laya.LayaDecoderLayer(config=self.cfg, mesh=self.mesh, layer_idx=1, rngs=nnx.Rngs(params=0))
    self.assertIsNotNone(jax_layer1.attn_norm)
    jax_layer1.attn_norm.scale[...] = jnp.asarray(self.st_weights["encoder.layers.1.attn_norm.weight"].numpy())
    jax_layer1.attn.Wqkv.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.1.attn.Wqkv.weight"].numpy().T)
    jax_layer1.attn.Wo.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.1.attn.Wo.weight"].numpy().T)
    jax_layer1.mlp_norm.scale[...] = jnp.asarray(self.st_weights["encoder.layers.1.mlp_norm.weight"].numpy())
    jax_layer1.mlp.Wi.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.1.mlp.Wi.weight"].numpy().T)
    jax_layer1.mlp.Wo.kernel[...] = jnp.asarray(self.st_weights["encoder.layers.1.mlp.Wo.weight"].numpy().T)

    jax_out1, _ = jax_layer1(
        jnp.asarray(in1_np),
        decoder_segment_ids=jnp.asarray(attn_mask_np),
        decoder_positions=jnp.asarray(pos_np),
        deterministic=True,
    )
    for b in range(batch_size):
      valid_len = int(attn_mask_np[b].sum())
      np.testing.assert_allclose(np.asarray(jax_out1[b, :valid_len]), pt_out1[b, :valid_len], rtol=1e-4, atol=1e-4)

  def test_4layer_encoder_and_decision_head_e2e(self):
    """Tests the 4-layer NNXDecoder + LayaDecisionHead against PyTorch DecisionModel on the 4-layer mini checkpoint."""
    if not self.has_mini_ckpt:
      self.skipTest("Mini checkpoint not found")

    pt_dm = DecisionModel(self.hf_encoder).eval().float()
    pt_dm.load_state_dict(self.st_weights, strict=True)

    batch_size, seq_len = 2, 32
    rng = np.random.default_rng(99)
    input_ids_np = rng.integers(10, 5000, size=(batch_size, seq_len), dtype=np.int32)
    attn_mask_np = np.ones((batch_size, seq_len), dtype=np.int32)
    attn_mask_np[1, 24:] = 0
    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (batch_size, seq_len)).copy()

    qtype_np = np.array([0, 2], dtype=np.int32)
    marker_pos_np = np.array([[5, 10, 15], [4, 8, 0]], dtype=np.int32)
    marker_mask_np = np.array([[True, True, True], [True, True, False]], dtype=bool)

    with torch.no_grad():
      pt_enc_out = pt_dm.encoder(
          input_ids=torch.from_numpy(input_ids_np).long(),
          attention_mask=torch.from_numpy(attn_mask_np).long(),
      ).last_hidden_state.numpy()
      pt_opt_logits, pt_act_logits = pt_dm(
          input_ids=torch.from_numpy(input_ids_np).long(),
          attention_mask=torch.from_numpy(attn_mask_np).long(),
          marker_pos=torch.from_numpy(marker_pos_np).long(),
          marker_mask=torch.from_numpy(marker_mask_np).bool(),
          qtype=torch.from_numpy(qtype_np).long(),
      )
      pt_opt_logits = pt_opt_logits.numpy()
      pt_act_logits = pt_act_logits.numpy()

    # Build JAX shared_embedding and NNXDecoder
    shared_emb = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        attend_dtype=jnp.float32,
        embedding_init=jax.nn.initializers.normal(stddev=0.02),
        config=self.cfg,
        mesh=self.mesh,
        rngs=nnx.Rngs(params=0),
    )
    decoder = nnx_decoders.NNXDecoder(
        config=self.cfg,
        mesh=self.mesh,
        quant=None,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0),
    )

    # Load weights from st_weights into shared_emb and decoder
    shared_emb.embedding[...] = jnp.asarray(self.st_weights["encoder.embeddings.tok_embeddings.weight"].numpy())
    decoder.embedding_norm.scale[...] = jnp.asarray(self.st_weights["encoder.embeddings.norm.weight"].numpy())
    decoder.decoder_norm.scale[...] = jnp.asarray(self.st_weights["encoder.final_norm.weight"].numpy())

    for i in range(4):
      lyr = getattr(decoder, f"layers_{i}")
      if i > 0:
        lyr.attn_norm.scale[...] = jnp.asarray(self.st_weights[f"encoder.layers.{i}.attn_norm.weight"].numpy())
      lyr.attn.Wqkv.kernel[...] = jnp.asarray(self.st_weights[f"encoder.layers.{i}.attn.Wqkv.weight"].numpy().T)
      lyr.attn.Wo.kernel[...] = jnp.asarray(self.st_weights[f"encoder.layers.{i}.attn.Wo.weight"].numpy().T)
      lyr.mlp_norm.scale[...] = jnp.asarray(self.st_weights[f"encoder.layers.{i}.mlp_norm.weight"].numpy())
      lyr.mlp.Wi.kernel[...] = jnp.asarray(self.st_weights[f"encoder.layers.{i}.mlp.Wi.weight"].numpy().T)
      lyr.mlp.Wo.kernel[...] = jnp.asarray(self.st_weights[f"encoder.layers.{i}.mlp.Wo.weight"].numpy().T)

    dh = decoder.decision_head
    dh.type_emb.embedding[...] = jnp.asarray(self.st_weights["type_emb.weight"].numpy())
    for hi in range(2):
      hl = getattr(dh, f"head_layers_{hi}")
      hl.norm1.scale[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.norm1.weight"].numpy())
      hl.norm1.bias[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.norm1.bias"].numpy())
      hl.in_proj.kernel[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.self_attn.in_proj_weight"].numpy().T)
      hl.in_proj.bias[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.self_attn.in_proj_bias"].numpy())
      hl.out_proj.kernel[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.self_attn.out_proj.weight"].numpy().T)
      hl.out_proj.bias[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.self_attn.out_proj.bias"].numpy())
      hl.norm2.scale[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.norm2.weight"].numpy())
      hl.norm2.bias[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.norm2.bias"].numpy())
      hl.linear1.kernel[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.linear1.weight"].numpy().T)
      hl.linear1.bias[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.linear1.bias"].numpy())
      hl.linear2.kernel[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.linear2.weight"].numpy().T)
      hl.linear2.bias[...] = jnp.asarray(self.st_weights[f"head.layers.{hi}.linear2.bias"].numpy())

    dh.scorer_0.scale[...] = jnp.asarray(self.st_weights["scorer.0.weight"].numpy())
    dh.scorer_0.bias[...] = jnp.asarray(self.st_weights["scorer.0.bias"].numpy())
    dh.scorer_1.kernel[...] = jnp.asarray(self.st_weights["scorer.1.weight"].numpy().T)
    dh.scorer_1.bias[...] = jnp.asarray(self.st_weights["scorer.1.bias"].numpy())
    dh.scorer_3.kernel[...] = jnp.asarray(self.st_weights["scorer.3.weight"].numpy().T)
    dh.scorer_3.bias[...] = jnp.asarray(self.st_weights["scorer.3.bias"].numpy())
    dh.act_head_0.kernel[...] = jnp.asarray(self.st_weights["act_head.0.weight"].numpy().T)
    dh.act_head_0.bias[...] = jnp.asarray(self.st_weights["act_head.0.bias"].numpy())
    dh.act_head_2.kernel[...] = jnp.asarray(self.st_weights["act_head.2.weight"].numpy().T)
    dh.act_head_2.bias[...] = jnp.asarray(self.st_weights["act_head.2.bias"].numpy())
    dh.temperature[...] = jnp.asarray(self.st_weights["temperature"].numpy())

    decoder_out = decoder(
        shared_emb,
        jnp.asarray(input_ids_np),
        jnp.asarray(pos_np),
        decoder_segment_ids=jnp.asarray(attn_mask_np),
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    raw_hidden_state = decoder_out[1]
    jax_enc_out = decoder.decoder_norm(raw_hidden_state)
    # Compare valid (non-padded) token hidden states
    for b in range(batch_size):
      valid_len = int(attn_mask_np[b].sum())
      np.testing.assert_allclose(
          np.asarray(jax_enc_out[b, :valid_len]),
          pt_enc_out[b, :valid_len],
          rtol=1e-4,
          atol=1e-4,
      )

    jax_opt_logits, jax_act_logits = decoder.decision_head(
        jax_enc_out,
        jnp.asarray(attn_mask_np),
        jnp.asarray(qtype_np),
        jnp.asarray(marker_pos_np),
        jnp.asarray(marker_mask_np),
    )
    np.testing.assert_allclose(np.asarray(jax_opt_logits), pt_opt_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(jax_act_logits), pt_act_logits, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
