# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Qwen3.5 Moe Vision subclasses comparing JAX implementations against PyTorch references."""

import os
import unittest
import numpy as np
import torch
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

# Explicit, differentiated imports matching Gemma4 layer-wise testing style
from maxtext.models.qwen3_5_vision import (
    Qwen3_5MoeVisionEncoder as JaxQwen3_5MoeVisionEncoder,
    Qwen3_5MoeVisionProjector as JaxQwen3_5MoeVisionProjector,
)

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionModel as TorchQwen3_5MoeVisionModel,
)

from tests.utils.multimodal_test_utils import (
    assert_all_close_jax_torch,
    copy_patch_embed_weights,
    copy_layernorm_weights,
    copy_attention_weights_to_maxtext,
    copy_linear_weights,
    create_random_jax_torch,
)

# Initialize JAX config cleanly using Qwen3.5-397B VL model registered config
base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen3.5-397b-a17b",
    attention="dot_product",
    attention_type="full",
    matmul_precision="highest",
    dropout_rate=0.0,
    dtype="float32",
    dtype_mm="float32",
    weight_dtype="float32",
    float32_logits=True,
    float32_qk_product=True,
)

# PyTorch Vision Config: all fields match Qwen3_5MoeVisionConfig defaults for the 397B-A17B vision
# tower except out_hidden_size, which must match the text decoder's base_emb_dim (4096 vs default 2048 of 35B-A3B).
torch_vision_config = Qwen3_5MoeVisionConfig(
    out_hidden_size=jax_config.out_hidden_size_for_vit,
    attn_implementation="eager",
)

torch.set_grad_enabled(False)


def create_torch_vision_encoder():
  """Create and configure PyTorch Qwen3.5 vision model."""
  encoder = TorchQwen3_5MoeVisionModel(torch_vision_config)
  encoder.eval()
  return encoder


def copy_qwen3_5_vision_encoder_weights(torch_encoder, jax_encoder):
  """Copy weights from PyTorch Qwen3.5 vision encoder to JAX subclassed encoder."""
  # Copy patch embedding
  copy_patch_embed_weights(torch_encoder.patch_embed, jax_encoder.patch_embed)

  # Copy positional embedding weights
  torch_pos_embed = torch_encoder.pos_embed.weight.detach().cpu().numpy()
  jax_encoder.pos_embed_interpolate.pos_embed.value = jnp.array(torch_pos_embed)

  # Copy encoder blocks
  for i, torch_block in enumerate(torch_encoder.blocks):
    jax_block = getattr(jax_encoder, f"blocks_{i}")
    copy_layernorm_weights(torch_block.norm1, jax_block.ln1)
    copy_layernorm_weights(torch_block.norm2, jax_block.ln2)
    copy_attention_weights_to_maxtext(torch_block.attn, jax_block.attn.attn, fused_qkv=True)
    copy_linear_weights(torch_block.mlp.linear_fc1, jax_block.mlp)
    copy_linear_weights(torch_block.mlp.linear_fc2, jax_block.mlp_out)


def copy_qwen3_5_patch_merger_weights(torch_merger, jax_merger):
  """Copy patch merger weights from PyTorch Qwen3.5 to JAX subclassed merger."""
  copy_layernorm_weights(torch_merger.norm, jax_merger.ln_q)
  copy_linear_weights(torch_merger.linear_fc1, jax_merger.mlp_0)
  copy_linear_weights(torch_merger.linear_fc2, jax_merger.mlp_2)


class TestQwen3_5MoeVisionEncoderEndToEnd(unittest.TestCase):
  """End-to-end equivalence test for Qwen3.5 Moe Vision Encoder + Projector JAX subclasses."""

  def setUp(self):
    np.random.seed(42)
    torch.manual_seed(42)
    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

  def test_vision_encoder_subclasses_match_torch(self):
    """Test full JAX vision subclassed tower matches PyTorch Qwen3.5 vision tower."""
    torch_encoder = create_torch_vision_encoder()

    jax_encoder = JaxQwen3_5MoeVisionEncoder(config=jax_config, mesh=self.mesh, rngs=nnx.Rngs(42))
    jax_projector = JaxQwen3_5MoeVisionProjector(config=jax_config, rngs=nnx.Rngs(43))

    # Copy weights
    copy_qwen3_5_vision_encoder_weights(torch_encoder, jax_encoder)
    copy_qwen3_5_patch_merger_weights(torch_encoder.merger, jax_projector.merger)

    patch_size = jax_config.patch_size_for_vit
    temporal_patch_size = jax_config.temporal_patch_size_for_vit
    in_channels = jax_config.num_channels_for_vit
    h, w = 8, 8  # 8x8 patches

    n_patches = h * w
    total_elements = n_patches * in_channels * temporal_patch_size * patch_size * patch_size
    flat_data, _ = create_random_jax_torch(total_elements)

    # Reshape inputs
    jax_hidden_states = flat_data.reshape(1, in_channels, temporal_patch_size, h * patch_size, w * patch_size)
    torch_hidden_states = torch.from_numpy(
        np.array(flat_data).reshape((n_patches, in_channels, temporal_patch_size, patch_size, patch_size))
    )

    grid_thw = np.array([[1, h, w]], dtype=np.int64)
    grid_thw_torch = torch.from_numpy(grid_thw)

    # PyTorch forward
    torch_out = torch_encoder(torch_hidden_states, grid_thw_torch)
    torch_output = torch_out.pooler_output  # after merger

    # JAX forward
    jax_encoder_output, _ = jax_encoder(jax_hidden_states)
    jax_output = jax_projector(jax_encoder_output)
    jax_output = jax_output[0]

    # Compare final projected outputs. Use 2e-2 (vs 1e-2 for omni) because out_hidden_size=4096
    # (vs 2048 for omni) doubles the projector MLP output, accumulating slightly more float32 error.
    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-2,
        atol=2e-2,
        error_msg="Qwen3.5 JAX subclassed vision tower final output differs from PyTorch reference",
    )


if __name__ == "__main__":
  unittest.main()
