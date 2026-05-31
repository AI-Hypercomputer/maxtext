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

"""Tests for Gemma 4 vision layers comparing MaxText implementation against PyTorch reference."""

import os
import unittest

import pytest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common import common_types
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.models.gemma4_vision import (
    VisionEntry as JaxVisionEntry,
    Gemma4VisionRotaryEmbedding as JaxGemma4VisionRotaryEmbedding,
    Gemma4Attention as JaxGemma4VisionAttention,
    Gemma4EncoderBlock as JaxGemma4EncoderBlock,
    VisionExit as JaxVisionExit,
    Gemma4VisionEncoderLayer as JaxGemma4VisionEncoderLayer,
    Gemma4VisionProjector as JaxGemma4VisionProjector,
    patchify,
)
from tests.utils.multimodal_test_utils import (
    assert_all_close_jax_torch,
    assert_stats_close,
    copy_linear_weights,
    copy_rmsnorm_weights,
    create_random_jax_torch,
    get_array_stats,
)
import numpy as np

try:
  import torch
  import torch.nn.functional as F
  from transformers.models.gemma4.configuration_gemma4 import (
      Gemma4VisionConfig,
      Gemma4TextConfig,
  )
  from transformers.models.gemma4.modeling_gemma4 import (
      Gemma4VisionPatchEmbedder as TorchGemma4VisionPatchEmbedder,
      Gemma4VisionRotaryEmbedding as TorchGemma4VisionRotaryEmbedding,
      Gemma4VisionAttention as TorchGemma4VisionAttention,
      Gemma4VisionEncoderLayer as TorchGemma4VisionEncoderLayer,
      Gemma4VisionPooler as TorchGemma4VisionPooler,
      Gemma4VisionModel as TorchGemma4VisionModel,
      Gemma4MultimodalEmbedder as TorchGemma4MultimodalEmbedder,
      apply_multidimensional_rope,
  )
  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False
  torch = None


# Initialize config once for all tests
base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="gemma4-26b",
    use_multimodal=True,
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

if HAS_TORCH:
  # PyTorch vision encoder config
  torch_vision_config = Gemma4VisionConfig(
      hidden_size=jax_config.hidden_size_for_vit,
      intermediate_size=jax_config.intermediate_size_for_vit,
      num_hidden_layers=jax_config.num_hidden_layers_for_vit,
      num_attention_heads=jax_config.num_attention_heads_for_vit,
      num_key_value_heads=jax_config.num_attention_heads_for_vit,
      head_dim=jax_config.hidden_size_for_vit // jax_config.num_attention_heads_for_vit,
      patch_size=jax_config.patch_size_for_vit,
      position_embedding_size=jax_config.num_position_embeddings_for_vit,
      rope_parameters={"rope_type": "default", "rope_theta": jax_config.rope_theta_for_vit},
      pooling_kernel_size=3,
      standardize=True,
      rms_norm_eps=jax_config.normalization_layer_epsilon,
  )
  torch_vision_config._attn_implementation = "eager"  # pylint: disable=protected-access

  # PyTorch text config for multimodal embedder
  torch_text_config = Gemma4TextConfig(hidden_size=jax_config.emb_dim)

  torch.set_grad_enabled(False)
else:
  torch_vision_config = None
  torch_text_config = None



def setup_test_seeds():
  """Set random seeds for reproducibility."""
  np.random.seed(42)
  if HAS_TORCH:
    torch.manual_seed(42)



# =============================================================================
# Weight Copying Helpers
# =============================================================================


def copy_vision_entry_weights(torch_embed, jax_embed):
  """Copy weights from PyTorch Gemma4VisionPatchEmbedder to JAX VisionEntry."""
  copy_linear_weights(torch_embed.input_proj, jax_embed.input_projection)
  # PyTorch: [2, max_pos, hidden_size] -> JAX: [max_pos, 2, hidden_size]
  torch_pos = torch_embed.position_embedding_table.detach().cpu().numpy()
  jax_embed.pos_emb_param.value = jnp.array(np.transpose(torch_pos, (1, 0, 2)))


def copy_gemma4_attention_weights(torch_attn, jax_attn):
  """Copy weights from PyTorch Gemma4VisionAttention to JAX Gemma4Attention."""
  num_heads = jax_attn.num_query_heads
  head_dim = jax_attn.head_dim
  hidden_size = num_heads * head_dim
  output_dim = hidden_size

  # PyTorch weights from Gemma4ClippableLinear (.linear.weight)
  q_weight = torch_attn.q_proj.linear.weight.detach().cpu().numpy()
  k_weight = torch_attn.k_proj.linear.weight.detach().cpu().numpy()
  v_weight = torch_attn.v_proj.linear.weight.detach().cpu().numpy()
  out_weight = torch_attn.o_proj.linear.weight.detach().cpu().numpy()

  # JAX DenseGeneral kernel shape: (in_features, num_heads, head_dim)
  jax_attn.query.kernel.value = jnp.array(q_weight.T.reshape(hidden_size, num_heads, head_dim))
  jax_attn.key.kernel.value = jnp.array(k_weight.T.reshape(hidden_size, num_heads, head_dim))
  jax_attn.value.kernel.value = jnp.array(v_weight.T.reshape(hidden_size, num_heads, head_dim))

  # JAX out kernel shape: (num_heads, head_dim, output_dim)
  jax_attn.out.kernel.value = jnp.array(out_weight.T.reshape(num_heads, head_dim, output_dim))

  # Copy norms
  copy_rmsnorm_weights(torch_attn.q_norm, jax_attn.query_norm)
  copy_rmsnorm_weights(torch_attn.k_norm, jax_attn.key_norm)
  copy_rmsnorm_weights(torch_attn.v_norm, jax_attn.value_norm)


def copy_gemma4_mlp_weights(torch_mlp, jax_mlp):
  """Copy weights from PyTorch Gemma4VisionMLP to JAX MlpBlock."""
  copy_linear_weights(torch_mlp.gate_proj.linear, jax_mlp.wi_0)
  copy_linear_weights(torch_mlp.up_proj.linear, jax_mlp.wi_1)
  copy_linear_weights(torch_mlp.down_proj.linear, jax_mlp.wo)


def copy_gemma4_vision_encoder_weights(torch_model, jax_model):
  """Copy all weights from PyTorch Gemma4VisionModel to JAX Gemma4VisionEncoderLayer."""
  # 1. Copy patch embedder (VisionEntry)
  copy_vision_entry_weights(torch_model.patch_embedder, jax_model.vision_entry)

  # 2. Copy encoder blocks
  for i, torch_layer in enumerate(torch_model.encoder.layers):
    jax_layer = getattr(jax_model, f"layer_{i}")
    copy_gemma4_attention_weights(torch_layer.self_attn, jax_layer.attention)
    copy_gemma4_mlp_weights(torch_layer.mlp, jax_layer.mlp)
    copy_rmsnorm_weights(torch_layer.input_layernorm, jax_layer.pre_attention_norm)
    copy_rmsnorm_weights(torch_layer.post_attention_layernorm, jax_layer.post_attention_norm)
    copy_rmsnorm_weights(torch_layer.pre_feedforward_layernorm, jax_layer.pre_ffw_norm)
    copy_rmsnorm_weights(torch_layer.post_feedforward_layernorm, jax_layer.post_ffw_norm)

  # 3. Copy std_bias and std_scale
  if hasattr(torch_model, "std_bias") and hasattr(jax_model, "std_bias"):
    jax_model.std_bias.value = jnp.array(torch_model.std_bias.detach().cpu().numpy())
    jax_model.std_scale.value = jnp.array(torch_model.std_scale.detach().cpu().numpy())


def copy_gemma4_vision_projector_weights(torch_model, jax_model):
  """Copy weights from PyTorch Gemma4MultimodalEmbedder to JAX Gemma4VisionProjector."""
  copy_linear_weights(torch_model.embedding_projection, jax_model.projection)


# =============================================================================
# Unit Tests
# =============================================================================


class BaseGemma4VisionTestCase(unittest.TestCase):
  """Base class for Gemma 4 vision tests with common setup."""

  pytestmark = pytest.mark.tpu_only

  def setUp(self):
    self.config = jax_config
    setup_test_seeds()


class BaseGemma4VisionTestCaseWithMesh(BaseGemma4VisionTestCase):
  """Base class for Gemma 4 vision tests that require mesh setup."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))


class TestGemma4VisionEntry(BaseGemma4VisionTestCase):
  """Test cases for VisionEntry layer."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_vision_entry_matches_torch(self):
    torch_model = TorchGemma4VisionPatchEmbedder(torch_vision_config)
    torch_model.eval()

    jax_model = JaxVisionEntry(
        d_model=self.config.hidden_size_for_vit,
        patch_size=self.config.patch_size_for_vit,
        pos_emb_shape_yx=(self.config.num_position_embeddings_for_vit, 2),
        normalize_input_range=True,
        rngs=nnx.Rngs(42),
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        matmul_precision=self.config.matmul_precision,
    )

    copy_vision_entry_weights(torch_model, jax_model)

    batch_size = 2
    h, w = 672, 960
    c = self.config.num_channels_for_vit
    patch_size = self.config.patch_size_for_vit

    # Create random image
    # PyTorch expects NCHW for patchification, but here we pass pre-patchified input to torch_model
    # MaxText patchify expects NHWC
    jax_images, _ = create_random_jax_torch(batch_size, h, w, c)

    # Get patches and positions using MaxText patchify
    jax_patches, jax_positions = patchify(jax_images, patch_size)
    torch_patches = torch.from_numpy(np.array(jax_patches))
    torch_positions = torch.from_numpy(np.array(jax_positions)).long()
    padding_positions = torch.zeros((batch_size, jax_patches.shape[1]), dtype=torch.bool)

    torch_output = torch_model(torch_patches, torch_positions, padding_positions)
    jax_output, _ = jax_model(jax_patches, jax_positions)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-3,
        atol=1e-3,
        error_msg="VisionEntry outputs differ",
    )

  def test_vision_entry_jax_only(self):
    jax_model = JaxVisionEntry(
        d_model=self.config.hidden_size_for_vit,
        patch_size=self.config.patch_size_for_vit,
        pos_emb_shape_yx=(self.config.num_position_embeddings_for_vit, 2),
        normalize_input_range=True,
        rngs=nnx.Rngs(42),
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        matmul_precision=self.config.matmul_precision,
    )

    batch_size = 2
    h, w = 672, 960
    c = self.config.num_channels_for_vit
    patch_size = self.config.patch_size_for_vit

    jax_images, _ = create_random_jax_torch(batch_size, h, w, c)

    jax_patches, jax_positions = patchify(jax_images, patch_size)
    jax_output, _ = jax_model(jax_patches, jax_positions)

    actual_stats = get_array_stats(jax_output)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -11.684295654296875,
        "max": 11.459238052368164,
        "mean": 0.02052328735589981,
        "std": 2.2361695766448975,
        "first_5": [-1.7952595949172974, -0.9984737634658813, -1.6569695472717285, -0.9252973198890686, 2.6700427532196045],
        "last_5": [-0.8076034784317017, -4.024001121520996, 2.3631980419158936, -0.6509848237037659, -0.17981889843940735],
    }

    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)



class TestGemma4VisionRotaryEmbedding(BaseGemma4VisionTestCase):
  """Test cases for Gemma4VisionRotaryEmbedding layer."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_rotary_embedding_matches_torch(self):
    torch_model = TorchGemma4VisionRotaryEmbedding(torch_vision_config)
    torch_model.eval()

    jax_model = JaxGemma4VisionRotaryEmbedding(
        base_frequency=self.config.rope_theta_for_vit,
        rotary_fraction=None,
    )

    batch_size = 2
    seq_len = 42 * 60
    num_heads = self.config.num_attention_heads_for_vit
    head_dim = self.config.hidden_size_for_vit // num_heads

    jax_inputs, torch_inputs = create_random_jax_torch(batch_size, seq_len, num_heads, head_dim)
    
    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)
    torch_positions = torch.from_numpy(positions_xy).long()

    cos, sin = torch_model(torch_inputs, torch_positions)
    torch_output = apply_multidimensional_rope(torch_inputs, cos, sin, torch_positions, unsqueeze_dim=2)

    jax_output = jax_model(jax_inputs, jax_positions)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-3,
        atol=1e-3,
        error_msg="Gemma4VisionRotaryEmbedding outputs differ",
    )

  def test_rotary_embedding_jax_only(self):
    jax_model = JaxGemma4VisionRotaryEmbedding(
        base_frequency=self.config.rope_theta_for_vit,
        rotary_fraction=None,
    )

    batch_size = 2
    seq_len = 42 * 60
    num_heads = self.config.num_attention_heads_for_vit
    head_dim = self.config.hidden_size_for_vit // num_heads

    jax_inputs, _ = create_random_jax_torch(batch_size, seq_len, num_heads, head_dim)
    
    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)

    jax_output = jax_model(jax_inputs, jax_positions)

    actual_stats = get_array_stats(jax_output)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -5.0020833015441895,
        "max": 5.087890625,
        "mean": -0.0001776936842361465,
        "std": 1.000144600868225,
        "first_5": [0.49671414494514465, -0.13826429843902588, 0.6476885676383972, 1.5230298042297363, -0.2341533750295639],
        "last_5": [1.732391357421875, -0.21856474876403809, 1.5401558876037598, 0.06624335050582886, -1.2687402963638306],
    }
    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)


class TestGemma4VisionAttention(BaseGemma4VisionTestCaseWithMesh):
  """Test cases for Gemma4Attention layer."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_attention_matches_torch(self):
    torch_model = TorchGemma4VisionAttention(torch_vision_config, layer_idx=0)
    torch_model.eval()

    batch_size = 2
    seq_len = 42 * 60
    dummy_shape = (batch_size, seq_len, self.config.hidden_size_for_vit)

    jax_model = JaxGemma4VisionAttention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit,
        max_target_length=seq_len,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=dummy_shape,
        inputs_kv_shape=dummy_shape,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dropout_rate=self.config.dropout_rate,
        attention_type=common_types.AttentionType.FULL,
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=1.0,
        is_vision=True,
        rngs=nnx.Rngs(42),
    )

    copy_gemma4_attention_weights(torch_model, jax_model)

    jax_inputs, torch_inputs = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)
    torch_positions = torch.from_numpy(positions_xy).long()

    # Get rotary embeddings for PyTorch
    torch_rotary = TorchGemma4VisionRotaryEmbedding(torch_vision_config)
    cos, sin = torch_rotary(torch_inputs, torch_positions)

    torch_output, _ = torch_model(torch_inputs, position_embeddings=(cos, sin), position_ids=torch_positions)
    jax_output, _ = jax_model(jax_inputs, jax_inputs, inputs_positions=jax_positions, deterministic=True)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-2,
        atol=1e-2,
        error_msg="Gemma4Attention outputs differ",
    )

  def test_attention_jax_only(self):
    batch_size = 2
    seq_len = 42 * 60
    dummy_shape = (batch_size, seq_len, self.config.hidden_size_for_vit)

    jax_model = JaxGemma4VisionAttention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit,
        max_target_length=seq_len,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=dummy_shape,
        inputs_kv_shape=dummy_shape,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dropout_rate=self.config.dropout_rate,
        attention_type=common_types.AttentionType.FULL,
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=1.0,
        is_vision=True,
        rngs=nnx.Rngs(42),
    )

    jax_inputs, _ = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)

    jax_output, _ = jax_model(jax_inputs, jax_inputs, inputs_positions=jax_positions, deterministic=True)

    actual_stats = get_array_stats(jax_output)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -4.0221757888793945,
        "max": 4.046579360961914,
        "mean": -0.0002807701821438968,
        "std": 0.7455713152885437,
        "first_5": [0.3752165138721466, 0.9831300973892212, 0.4212400019168854, 0.46032220125198364, 0.15025049448013306],
        "last_5": [0.1874169111251831, -0.09554912149906158, -0.3366227447986603, 1.2252843379974365, -0.5885721445083618],
    }
    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)



class TestGemma4VisionEncoderBlock(BaseGemma4VisionTestCaseWithMesh):
  """Test cases for Gemma4EncoderBlock layer."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_encoder_block_matches_torch(self):
    torch_model = TorchGemma4VisionEncoderLayer(torch_vision_config, layer_idx=0)
    torch_model.eval()

    jax_model = JaxGemma4EncoderBlock(self.config, self.mesh, rngs=nnx.Rngs(42))

    # Copy weights
    copy_gemma4_attention_weights(torch_model.self_attn, jax_model.attention)
    copy_gemma4_mlp_weights(torch_model.mlp, jax_model.mlp)
    copy_rmsnorm_weights(torch_model.input_layernorm, jax_model.pre_attention_norm)
    copy_rmsnorm_weights(torch_model.post_attention_layernorm, jax_model.post_attention_norm)
    copy_rmsnorm_weights(torch_model.pre_feedforward_layernorm, jax_model.pre_ffw_norm)
    copy_rmsnorm_weights(torch_model.post_feedforward_layernorm, jax_model.post_ffw_norm)

    batch_size = 2
    seq_len = 42 * 60
    jax_inputs, torch_inputs = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)
    torch_positions = torch.from_numpy(positions_xy).long()

    # Get rotary embeddings for PyTorch
    torch_rotary = TorchGemma4VisionRotaryEmbedding(torch_vision_config)
    cos, sin = torch_rotary(torch_inputs, torch_positions)

    torch_output = torch_model(torch_inputs, position_embeddings=(cos, sin), position_ids=torch_positions)
    jax_output = jax_model(jax_inputs, positions=jax_positions, deterministic=True)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-2,
        atol=1e-2,
        error_msg="Gemma4EncoderBlock outputs differ",
    )

  def test_encoder_block_jax_only(self):
    jax_model = JaxGemma4EncoderBlock(self.config, self.mesh, rngs=nnx.Rngs(42))

    batch_size = 2
    seq_len = 42 * 60
    jax_inputs, _ = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)

    jax_output = jax_model(jax_inputs, positions=jax_positions, deterministic=True)

    actual_stats = get_array_stats(jax_output)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -8.562429428100586,
        "max": 9.196337699890137,
        "mean": -0.0017009616130962968,
        "std": 1.7320671081542969,
        "first_5": [0.64936363697052, 0.276985764503479, 2.0699658393859863, 1.582777976989746, -1.8568949699401855],
        "last_5": [0.4208720922470093, -0.3171665668487549, 0.08753800392150879, -1.1161333322525024, -3.1165666580200195],
    }
    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)


class TestGemma4VisionExit(BaseGemma4VisionTestCase):
  """Test cases for VisionExit layer."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_vision_exit_matches_torch(self):
    torch_model = TorchGemma4VisionPooler(torch_vision_config)
    torch_model.eval()

    jax_model = JaxVisionExit(
        d_model=self.config.hidden_size_for_vit,
        output_length=self.config.vision_output_length,
        rngs=nnx.Rngs(42),
        precision=self.config.matmul_precision,
    )

    batch_size = 2
    seq_len = 42 * 60
    jax_inputs, torch_inputs = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)
    torch_positions = torch.from_numpy(positions_xy).long()
    padding_positions = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    torch_output, _ = torch_model(
        torch_inputs, torch_positions, padding_positions, output_length=self.config.vision_output_length
    )
    jax_results = jax_model(jax_inputs, positions_xy=jax_positions)
    jax_output = jax_results[0][0]

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-3,
        atol=1e-3,
        error_msg="VisionExit outputs differ",
    )

  def test_vision_exit_jax_only(self):
    jax_model = JaxVisionExit(
        d_model=self.config.hidden_size_for_vit,
        output_length=self.config.vision_output_length,
        rngs=nnx.Rngs(42),
        precision=self.config.matmul_precision,
    )

    batch_size = 2
    seq_len = 42 * 60
    jax_inputs, _ = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    # Create grid positions
    h_patches, w_patches = 42, 60
    xy = np.meshgrid(np.arange(w_patches), np.arange(h_patches))
    positions_xy = np.stack(xy, axis=-1).reshape(-1, 2)
    positions_xy = np.broadcast_to(positions_xy, (batch_size, seq_len, 2))
    jax_positions = jnp.array(positions_xy)

    jax_results = jax_model(jax_inputs, positions_xy=jax_positions)
    jax_output = jax_results[0][0]

    actual_stats = get_array_stats(jax_output)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -54.34581756591797,
        "max": 49.49297332763672,
        "mean": -0.006532127968966961,
        "std": 11.311936378479004,
        "first_5": [-1.9137297868728638, 6.875588417053223, 15.854601860046387, 8.226688385009766, 8.342312812805176],
        "last_5": [21.86972999572754, -0.21706300973892212, 7.504371643066406, -4.01204776763916, -6.540284633636475],
    }
    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)


class TestGemma4VisionEncoderEndToEnd(BaseGemma4VisionTestCaseWithMesh):
  """End-to-end test for the full vision encoder."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_vision_encoder_matches_torch(self):
    torch_model = TorchGemma4VisionModel(torch_vision_config)
    torch_model.eval()

    jax_model = JaxGemma4VisionEncoderLayer(self.config, self.mesh, rngs=nnx.Rngs(42))

    copy_gemma4_vision_encoder_weights(torch_model, jax_model)

    batch_size = 2
    h, w = 672, 960
    c = self.config.num_channels_for_vit
    patch_size = self.config.patch_size_for_vit

    # Create random image (NHWC for MaxText)
    jax_images, _ = create_random_jax_torch(batch_size, h, w, c)
    
    # Get patches and positions for PyTorch using MaxText patchify
    jax_patches, jax_positions = patchify(jax_images, patch_size)
    torch_patches = torch.from_numpy(np.array(jax_patches))
    torch_positions = torch.from_numpy(np.array(jax_positions)).long()

    torch_output = torch_model(torch_patches, torch_positions)
    torch_lhs = torch_output.last_hidden_state.view(batch_size, self.config.vision_output_length, self.config.hidden_size_for_vit)
    
    # MaxText Gemma4VisionEncoderLayer expects 4D (NHWC) or 5D (N, num_images, H, W, C)
    jax_output = jax_model(jax_images, deterministic=True)
    
    # jax_output has shape [batch, num_images=1, length, dim], squeeze num_images for comparison
    jax_output_squeezed = jax_output.squeeze(1)

    assert_all_close_jax_torch(
        jax_output_squeezed,
        torch_lhs,
        rtol=5e-2,
        atol=5e-2,
        error_msg="Gemma4VisionEncoderLayer end-to-end outputs differ",
    )

  def test_vision_encoder_jax_only(self):
    jax_model = JaxGemma4VisionEncoderLayer(self.config, self.mesh, rngs=nnx.Rngs(42))

    batch_size = 2
    h, w = 672, 960
    c = self.config.num_channels_for_vit

    jax_images, _ = create_random_jax_torch(batch_size, h, w, c)
    
    jax_output = jax_model(jax_images, deterministic=True)
    jax_output_squeezed = jax_output.squeeze(1)

    actual_stats = get_array_stats(jax_output_squeezed)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -675.96240234375,
        "max": 626.8897094726562,
        "mean": 5.652062892913818,
        "std": 134.99948120117188,
        "first_5": [-140.56704711914062, 108.94792175292969, -68.02703857421875, -13.687520980834961, -24.473756790161133],
        "last_5": [-249.64027404785156, 180.6945343017578, 77.77986907958984, 89.37599182128906, 26.383209228515625],
    }
    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)


class TestGemma4VisionProjector(BaseGemma4VisionTestCaseWithMesh):
  """Test cases for Gemma4VisionProjector layer."""

  @unittest.skipIf(not HAS_TORCH, "Torch not available")
  def test_vision_projector_matches_torch(self):
    torch_model = TorchGemma4MultimodalEmbedder(torch_vision_config, torch_text_config)
    torch_model.eval()

    jax_model = JaxGemma4VisionProjector(self.config, self.mesh, rngs=nnx.Rngs(42))

    copy_gemma4_vision_projector_weights(torch_model, jax_model)

    batch_size = 2
    seq_len = self.config.vision_output_length
    jax_inputs, torch_inputs = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    torch_output = torch_model(torch_inputs)
    jax_output = jax_model(jax_inputs)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-3,
        atol=1e-3,
        error_msg="Gemma4VisionProjector outputs differ",
    )

  def test_vision_projector_jax_only(self):
    jax_model = JaxGemma4VisionProjector(self.config, self.mesh, rngs=nnx.Rngs(42))

    batch_size = 2
    seq_len = self.config.vision_output_length
    jax_inputs, _ = create_random_jax_torch(batch_size, seq_len, self.config.hidden_size_for_vit)

    jax_output = jax_model(jax_inputs)

    actual_stats = get_array_stats(jax_output)
    print(f"\nSTATS for {self._testMethodName}: {actual_stats}")

    expected_stats = {
        "min": -5.151237964630127,
        "max": 4.770052433013916,
        "mean": -0.0003473749675322324,
        "std": 1.0001885890960693,
        "first_5": [-0.015990108251571655, 1.83672297000885, -1.8761849403381348, -0.19131013751029968, 1.2963128089904785],
        "last_5": [0.16569757461547852, 0.6256201267242432, 0.2053520828485489, -0.25832629203796387, 2.071470260620117],
    }
    assert_stats_close(self, actual_stats, expected_stats, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  unittest.main()
