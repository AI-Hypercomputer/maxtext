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

"""Tests for Phi4 layers comparing MaxText implementation against PyTorch reference."""

import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common import common_types
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.models.phi4 import Phi4DecoderLayer as JaxPhi4DecoderLayer
from tests.utils.multimodal_test_utils import (
    assert_all_close_jax_torch,
    copy_linear_weights,
    copy_rmsnorm_weights,
    create_random_jax_torch,
)
import numpy as np
import torch

# PyTorch/transformers imports
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer as TorchPhi3DecoderLayer

# Initialize config once for all tests
base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="phi4",
    attention="dot_product",
    attention_type="full",
    matmul_precision="highest",
    dropout_rate=0.0,
    dtype="float32",
    weight_dtype="float32",
    float32_logits=True,
    float32_qk_product=True,
)

torch_config = Phi3Config(
    vocab_size=jax_config.vocab_size,
    hidden_size=jax_config.emb_dim,
    intermediate_size=jax_config.mlp_dim,
    num_hidden_layers=jax_config.num_decoder_layers,
    num_attention_heads=jax_config.num_query_heads,
    num_key_value_heads=jax_config.num_kv_heads,
    max_position_embeddings=jax_config.max_position_embeddings,
    original_max_position_embeddings=jax_config.original_max_position_embeddings,
    rms_norm_eps=jax_config.normalization_layer_epsilon,
    hidden_act="silu",
    tie_word_embeddings=True,
    rope_parameters={
        "rope_type": "longrope",
        "rope_theta": jax_config.rope_max_timescale,
        "partial_rotary_factor": jax_config.partial_rotary_factor,
        "original_max_position_embeddings": jax_config.original_max_position_embeddings,
        "short_factor": [1.0]*48,
        "long_factor": [1.0]*48,
    }
)
torch_config._attn_implementation = "eager"

torch.set_grad_enabled(False)


def setup_test_seeds():
  """Set random seeds for reproducibility."""
  np.random.seed(42)
  torch.manual_seed(42)


def copy_phi4_attention_weights(torch_attn, jax_attn):
  """Copy weights from PyTorch Phi3Attention to JAX Attention."""
  num_heads = jax_attn.num_query_heads
  head_dim = jax_attn.head_dim
  hidden_size = num_heads * head_dim
  output_dim = hidden_size

  qkv_weight = torch_attn.qkv_proj.weight.detach().cpu().numpy()  # Shape: (op_size, hidden_size)
  out_weight = torch_attn.o_proj.weight.detach().cpu().numpy()

  query_pos = num_heads * head_dim
  kv_states_dim = jax_attn.num_kv_heads * head_dim

  # Split fused qkv_proj
  q_weight = qkv_weight[:query_pos, :]
  k_weight = qkv_weight[query_pos:query_pos + kv_states_dim, :]
  v_weight = qkv_weight[query_pos + kv_states_dim:, :]

  # JAX general kernels shape: (in_features, num_heads, head_dim)
  depth_scaling = np.sqrt(head_dim)
  jax_attn.query.kernel.value = jnp.array(q_weight.T.reshape(hidden_size, num_heads, head_dim)) / depth_scaling
  jax_attn.key.kernel.value = jnp.array(k_weight.T.reshape(hidden_size, jax_attn.num_kv_heads, head_dim))
  jax_attn.value.kernel.value = jnp.array(v_weight.T.reshape(hidden_size, jax_attn.num_kv_heads, head_dim))

  # JAX out kernel shape: (num_heads, head_dim, output_dim)
  jax_attn.out.kernel.value = jnp.array(out_weight.T.reshape(num_heads, head_dim, output_dim))


def copy_phi4_mlp_weights(torch_mlp, jax_mlp):
  """Copy weights from PyTorch Phi3MLP to JAX MlpBlock."""
  gate_up_weight = torch_mlp.gate_up_proj.weight.detach().cpu().numpy()  # Shape: (2 * intermediate_size, hidden_size)
  down_weight = torch_mlp.down_proj.weight.detach().cpu().numpy()

  intermediate_size = jax_mlp.intermediate_dim
  hidden_size = jax_mlp.in_features

  # Split fused gate_up_proj
  gate_weight = gate_up_weight[:intermediate_size, :]
  up_weight = gate_up_weight[intermediate_size:, :]

  # Copy to wi_0 (gate) and wi_1 (up)
  jax_mlp.wi_0.kernel.value = jnp.array(gate_weight.T.reshape(hidden_size, intermediate_size))
  jax_mlp.wi_1.kernel.value = jnp.array(up_weight.T.reshape(hidden_size, intermediate_size))

  # Copy to wo (down)
  jax_mlp.wo.kernel.value = jnp.array(down_weight.T.reshape(intermediate_size, hidden_size))


def copy_phi4_decoder_layer_weights(torch_layer, jax_layer):
  """Copy all weights from PyTorch Phi3DecoderLayer to JAX Phi4DecoderLayer."""
  copy_phi4_attention_weights(torch_layer.self_attn, jax_layer.self_attention)
  copy_phi4_mlp_weights(torch_layer.mlp, jax_layer.mlp)
  copy_rmsnorm_weights(torch_layer.input_layernorm, jax_layer.pre_self_attention_layer_norm)
  copy_rmsnorm_weights(torch_layer.post_attention_layernorm, jax_layer.post_self_attention_layer_norm)


class Phi4LayersTest(unittest.TestCase):
  """Unit tests comparing Phi4 JAX layers against PyTorch references."""

  def setUp(self):
    self.config = jax_config
    setup_test_seeds()
    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

  def test_decoder_layer_step_by_step(self):
    batch, seq_len, hidden_dim = 2, 4, jax_config.emb_dim
    setup_test_seeds()
    torch_head_dim = torch_config.hidden_size // torch_config.num_attention_heads

    # 1. Setup inputs
    jax_inputs, torch_inputs = create_random_jax_torch(batch, seq_len, hidden_dim)

    # 2. Initialize PyTorch Layer
    torch_layer = TorchPhi3DecoderLayer(torch_config, layer_idx=0)

    # 3. Initialize JAX Layer
    rngs = nnx.Rngs(0)
    jax_layer = JaxPhi4DecoderLayer(
        config=self.config,
        model_mode=common_types.MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=rngs,
    )

    # 4. Copy Weights
    copy_phi4_decoder_layer_weights(torch_layer, jax_layer)

    # 5. Generate Cos/Sin position embeddings for eager attention forward in PyTorch
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
    rotary_emb = Phi3RotaryEmbedding(torch_config)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary_emb(torch_inputs, position_ids=position_ids)

    # ----------------------------------------------------
    # Step 1: Pre-Attention LayerNorm
    # ----------------------------------------------------
    torch_norm1 = torch_layer.input_layernorm(torch_inputs)
    jax_norm1 = jax_layer.pre_self_attention_layer_norm(jax_inputs)
    print("\n=== STEP 1: Pre-LN ===")
    assert_all_close_jax_torch(jax_norm1, torch_norm1, rtol=1e-3, atol=1e-3)
    print("Pre-LN matches!")

    # ----------------------------------------------------
    # Step 2: QKV Projections (unfused)
    # ----------------------------------------------------
    # PyTorch:
    qkv = torch_layer.self_attn.qkv_proj(torch_norm1)
    query_pos = torch_config.num_attention_heads * torch_head_dim
    q_torch = qkv[..., :query_pos]
    k_torch = qkv[..., query_pos : query_pos + torch_config.num_key_value_heads * torch_head_dim]
    v_torch = qkv[..., query_pos + torch_config.num_key_value_heads * torch_head_dim :]

    # JAX:
    q_jax = jax_layer.self_attention.query(jax_norm1)
    k_jax = jax_layer.self_attention.key(jax_norm1)
    v_jax = jax_layer.self_attention.value(jax_norm1)

    # Compare fused shapes.
    q_jax_flat = q_jax.reshape(batch, seq_len, -1)
    k_jax_flat = k_jax.reshape(batch, seq_len, -1)
    v_jax_flat = v_jax.reshape(batch, seq_len, -1)

    print("\n=== STEP 2: QKV Projections ===")
    assert_all_close_jax_torch(q_jax_flat, q_torch / np.sqrt(torch_head_dim), rtol=1e-3, atol=1e-3)
    print("Query matches (scaled by 1/sqrt(head_dim))!")
    assert_all_close_jax_torch(k_jax_flat, k_torch, rtol=1e-3, atol=1e-3)
    print("Key matches!")
    assert_all_close_jax_torch(v_jax_flat, v_torch, rtol=1e-3, atol=1e-3)
    print("Value matches!")

    # ----------------------------------------------------
    # Step 3: Rotary Embedding Application
    # ----------------------------------------------------
    # PyTorch:
    hidden_shape = (batch, seq_len, torch_config.num_attention_heads, torch_head_dim)
    q_torch_reshaped = q_torch.view(hidden_shape).transpose(1, 2)
    kv_shape = (batch, seq_len, torch_config.num_key_value_heads, torch_head_dim)
    k_torch_reshaped = k_torch.view(kv_shape).transpose(1, 2)
    v_torch_reshaped = v_torch.view(kv_shape).transpose(1, 2)

    cos, sin = position_embeddings
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb
    q_torch_rot, k_torch_rot = apply_rotary_pos_emb(q_torch_reshaped, k_torch_reshaped, cos, sin)

    # JAX:
    rotary_embedding = jax_layer.self_attention.rotary_embedding
    decoder_positions = jnp.arange(seq_len)[jnp.newaxis, :].repeat(batch, axis=0)
    decoder_segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)

    print(f"\nJAX rotary_embedding dims: {rotary_embedding.embedding_dims}, partial_factor: {getattr(rotary_embedding, 'partial_rotary_factor', None)}, scaling: {getattr(rotary_embedding, 'rope_attention_scaling', None)}")
    q_jax_rot = rotary_embedding(q_jax, decoder_positions)
    k_jax_rot = rotary_embedding(k_jax, decoder_positions)

    q_jax_rot_t = q_jax_rot.transpose(0, 2, 1, 3)
    k_jax_rot_t = k_jax_rot.transpose(0, 2, 1, 3)

    print("\n=== STEP 3: RoPE Projections ===")
    assert_all_close_jax_torch(q_jax_rot_t, q_torch_rot / np.sqrt(torch_head_dim), rtol=1e-3, atol=1e-3)
    print("RoPE Query matches (scaled by 1/sqrt(head_dim))!")
    assert_all_close_jax_torch(k_jax_rot_t, k_torch_rot, rtol=1e-3, atol=1e-3)
    print("RoPE Key matches!")

    # ----------------------------------------------------
    # Step 4: Attention Output (after O projection)
    # ----------------------------------------------------
    print(f"\nDEBUG JAX attention_type: {jax_layer.self_attention.attention_type} (type: {type(jax_layer.self_attention.attention_type)})")
    print(f"DEBUG JAX attention_op attention_type: {jax_layer.self_attention.attention_op.attention_type} (type: {type(jax_layer.self_attention.attention_op.attention_type)})")
    # PyTorch:
    from transformers.models.phi3.modeling_phi3 import create_causal_mask
    causal_mask = create_causal_mask(
        config=torch_config,
        inputs_embeds=torch_norm1,
        attention_mask=None,
        past_key_values=None,
    )
    torch_attn_out, _ = torch_layer.self_attn(
        torch_norm1,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
    )

    # JAX:
    jax_attn_out, _ = jax_layer.self_attention(
        jax_norm1,
        jax_norm1,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
    )

    print("\n=== STEP 4: Attention Block Output ===")
    assert_all_close_jax_torch(jax_attn_out, torch_attn_out, rtol=1e-3, atol=1e-3)
    print("Attention Block Output matches!")

    # ----------------------------------------------------
    # Step 5: Post-Attention LayerNorm
    # ----------------------------------------------------
    # PyTorch:
    torch_residual1 = torch_inputs + torch_attn_out
    torch_norm2 = torch_layer.post_attention_layernorm(torch_residual1)

    # JAX:
    jax_residual1 = jax_inputs + jax_attn_out
    jax_norm2 = jax_layer.post_self_attention_layer_norm(jax_residual1)

    print("\n=== STEP 5: Post-Attention LN ===")
    assert_all_close_jax_torch(jax_norm2, torch_norm2, rtol=1e-3, atol=1e-3)
    print("Post-Attention LN matches!")

    # ----------------------------------------------------
    # Step 6: MLP Output
    # ----------------------------------------------------
    # PyTorch:
    torch_mlp_out = torch_layer.mlp(torch_norm2)

    # JAX:
    jax_mlp_out = jax_layer.mlp(
        jax_norm2,
        deterministic=True,
    )

    print("\n=== STEP 6: MLP Block Output ===")
    assert_all_close_jax_torch(jax_mlp_out, torch_mlp_out, rtol=1e-3, atol=1e-3)
    print("MLP Block Output matches!")

    # ----------------------------------------------------
    # Step 7: Final Block Output
    # ----------------------------------------------------
    torch_final = torch_residual1 + torch_mlp_out
    jax_final = jax_residual1 + jax_mlp_out

    print("\n=== STEP 7: Final Layer Output ===")
    assert_all_close_jax_torch(jax_final, torch_final, rtol=1e-3, atol=1e-3)
    print("Final Layer Output matches!")

  def test_decoder_layer_forward(self):
    batch, seq_len, hidden_dim = 2, 4, jax_config.emb_dim
    setup_test_seeds()

    # 1. Setup inputs
    jax_inputs, torch_inputs = create_random_jax_torch(batch, seq_len, hidden_dim)

    # 2. Initialize PyTorch Layer
    torch_layer = TorchPhi3DecoderLayer(torch_config, layer_idx=0)

    # 3. Initialize JAX Layer
    rngs = nnx.Rngs(0)
    jax_layer = JaxPhi4DecoderLayer(
        config=self.config,
        model_mode=common_types.MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=rngs,
    )

    # 4. Copy Weights
    copy_phi4_decoder_layer_weights(torch_layer, jax_layer)

    # 5. Generate Cos/Sin position embeddings for eager attention forward in PyTorch
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
    rotary_emb = Phi3RotaryEmbedding(torch_config)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary_emb(torch_inputs, position_ids=position_ids)

    # 6. Run PyTorch forward pass
    from transformers.models.phi3.modeling_phi3 import create_causal_mask
    torch_norm1 = torch_layer.input_layernorm(torch_inputs)
    causal_mask = create_causal_mask(
        config=torch_config,
        inputs_embeds=torch_norm1,
        attention_mask=None,
        past_key_values=None,
    )
    torch_output = torch_layer(
        torch_inputs,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
    )

    # 7. Run JAX forward pass
    decoder_positions = jnp.arange(seq_len)[jnp.newaxis, :].repeat(batch, axis=0)
    decoder_segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)

    jax_output, _ = jax_layer(
        jax_inputs,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
    )

    # 8. Compare outputs
    # Using rtol=1e-3 and atol=1e-3 for exact mathematical validation
    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-3,
        atol=1e-3,
        error_msg="Phi4 JAX decoder layer output differs from PyTorch golden reference!",
    )


if __name__ == "__main__":
  unittest.main()
