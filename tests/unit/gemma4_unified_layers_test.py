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

"""Tests for Gemma 4 Unified (12B) layers against the PyTorch reference.

Gemma 4 12B ships as the `gemma4_unified` HF architecture. Its text tower is the
dense Gemma 4 tower, but the vision side has no ViT trunk: raw 3x3-merged pixel
patches are projected straight into the language-model space. The MaxText equivalent
is ``Gemma4UnifiedVisionEmbedder`` followed by the shared ``Gemma4VisionProjector``,
which together mirror HF's ``Gemma4UnifiedVisionEmbedder``.

The text-tower test doubles as a numerical check of the HF -> MaxText checkpoint
conversion: it moves weights across with the real ``PARAM_MAPPING`` / ``HOOK_FNS``
entries rather than a test-local copy helper.
"""

import os
import unittest
import pytest

try:
  import torch
  from transformers.models.gemma4_unified.configuration_gemma4_unified import (
      Gemma4UnifiedVisionConfig,
      Gemma4UnifiedTextConfig,
  )
  from transformers.models.gemma4_unified.modeling_gemma4_unified import (
      Gemma4UnifiedForCausalLM as TorchGemma4UnifiedForCausalLM,
      Gemma4UnifiedVisionEmbedder as TorchGemma4UnifiedVisionEmbedder,
  )
  from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
      convert_image_to_patches,
      patches_merge,
  )
  from tests.utils.multimodal_test_utils import (
      assert_all_close_jax_torch,
      copy_layernorm_weights,
      copy_linear_weights,
  )

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers with gemma4_unified not available"),
]

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext import get_maxtext_model_info
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.models.gemma4_vision import (
    Gemma4UnifiedVisionEmbedder as JaxGemma4UnifiedVisionEmbedder,
    Gemma4VisionProjector as JaxGemma4VisionProjector,
    patchify,
)
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

# Shrunken dimensions keep the test light. The patch geometry is kept at the real
# values (16px teacher patches merged 3x3 into 48px model patches) since that is what
# the patchify/merge equivalence depends on.
_PATCH_SIZE = 16
_POOLING_KERNEL = 3
_MODEL_PATCH_SIZE = _PATCH_SIZE * _POOLING_KERNEL
_IMAGE_HEIGHT = 3 * _MODEL_PATCH_SIZE
_IMAGE_WIDTH = 4 * _MODEL_PATCH_SIZE
_NUM_SOFT_TOKENS = 12
_MM_EMBED_DIM = 128
_TEXT_HIDDEN = 64
_POSEMB_SIZE = 32

base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="gemma4-12b",
    use_multimodal=True,
    override_model_config=True,
    matmul_precision="highest",
    dropout_rate=0.0,
    dtype="float32",
    dtype_mm="float32",
    weight_dtype="float32",
    base_emb_dim=_TEXT_HIDDEN,
    hidden_size_for_vit=_MM_EMBED_DIM,
    num_position_embeddings_for_vit=_POSEMB_SIZE,
    patch_size_for_vit=_MODEL_PATCH_SIZE,
    image_size_for_vit=[_IMAGE_HEIGHT, _IMAGE_WIDTH],
    vision_output_length=_NUM_SOFT_TOKENS,
)

if HAS_TORCH:
  torch_vision_config = Gemma4UnifiedVisionConfig(
      mm_embed_dim=_MM_EMBED_DIM,
      mm_posemb_size=_POSEMB_SIZE,
      model_patch_size=_MODEL_PATCH_SIZE,
      patch_size=_PATCH_SIZE,
      pooling_kernel_size=_POOLING_KERNEL,
      num_soft_tokens=_NUM_SOFT_TOKENS,
      output_proj_dims=_MM_EMBED_DIM,
      rms_norm_eps=jax_config.normalization_layer_epsilon,
  )
  torch_text_config = Gemma4UnifiedTextConfig(hidden_size=_TEXT_HIDDEN)
  torch.set_grad_enabled(False)
else:
  torch_vision_config = None
  torch_text_config = None


def setup_test_seeds():
  """Set random seeds for reproducibility."""
  np.random.seed(42)
  if HAS_TORCH:
    torch.manual_seed(42)


def make_mesh():
  devices = np.array(jax.devices()).reshape([1] * len(jax_config.mesh_axes))
  return jax.sharding.Mesh(devices, jax_config.mesh_axes)


def copy_vision_embedder_weights(torch_embedder, jax_embedder, jax_projector):
  """Copy weights from the PyTorch unified embedder to the JAX embedder + projector."""
  copy_layernorm_weights(torch_embedder.patch_ln1, jax_embedder.patch_ln1)
  copy_linear_weights(torch_embedder.patch_dense, jax_embedder.patch_dense)
  copy_layernorm_weights(torch_embedder.patch_ln2, jax_embedder.patch_ln2)
  copy_layernorm_weights(torch_embedder.pos_norm, jax_embedder.pos_norm)
  # HF and MaxText agree on the (num_positions, 2, dim) layout for this table.
  jax_embedder.pos_emb_param.value = jnp.array(torch_embedder.pos_embedding.detach().cpu().numpy())
  copy_linear_weights(torch_embedder.multimodal_embedder.embedding_projection, jax_projector.projection)


class TestGemma4UnifiedPatchMerge(unittest.TestCase):
  """MaxText patchifies at the model patch size; HF patchifies at 16px and merges 3x3."""

  def test_patchify_matches_hf_teacher_patch_merge(self):
    setup_test_seeds()
    image = np.random.randn(3, _IMAGE_HEIGHT, _IMAGE_WIDTH).astype(np.float32)

    # HF path: (C, H, W) -> 16px patches -> merge 3x3.
    torch_image = torch.from_numpy(image)
    teacher_patches = convert_image_to_patches(torch_image, _PATCH_SIZE)
    patch_grid = torch.meshgrid(
        torch.arange(_IMAGE_WIDTH // _PATCH_SIZE),
        torch.arange(_IMAGE_HEIGHT // _PATCH_SIZE),
        indexing="xy",
    )
    teacher_positions = torch.stack(patch_grid, dim=-1).reshape(teacher_patches.shape[0], 2)
    torch_patches, torch_positions = patches_merge(
        teacher_patches.unsqueeze(0), teacher_positions.unsqueeze(0), _NUM_SOFT_TOKENS
    )

    # MaxText path: (H, W, C) -> 48px patches directly.
    jax_patches, jax_positions = patchify(jnp.asarray(np.transpose(image, (1, 2, 0))[None]), _MODEL_PATCH_SIZE)

    assert_all_close_jax_torch(jax_patches, torch_patches, rtol=0, atol=0, error_msg="merged patches differ")
    np.testing.assert_array_equal(np.asarray(jax_positions), torch_positions.numpy())


class TestGemma4UnifiedVisionEmbedder(unittest.TestCase):
  """End-to-end parity of the encoder-free embedder plus the multimodal projection."""

  def _build(self):
    """Builds the torch reference and the JAX pair, sharing one set of weights."""
    torch_embedder = TorchGemma4UnifiedVisionEmbedder(torch_vision_config, torch_text_config).eval()
    # HF zero-initializes pos_embedding; randomize so the position path is exercised.
    torch_embedder.pos_embedding.copy_(torch.randn_like(torch_embedder.pos_embedding))

    mesh = make_mesh()
    rngs = nnx.Rngs(params=0)
    jax_embedder = JaxGemma4UnifiedVisionEmbedder(config=jax_config, mesh=mesh, rngs=rngs)
    jax_projector = JaxGemma4VisionProjector(config=jax_config, mesh=mesh, rngs=rngs)
    copy_vision_embedder_weights(torch_embedder, jax_embedder, jax_projector)
    return torch_embedder, jax_embedder, jax_projector

  def test_matches_torch_reference(self):
    setup_test_seeds()
    torch_embedder, jax_embedder, jax_projector = self._build()

    images = np.random.rand(2, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3).astype(np.float32)
    jax_patches, jax_positions = patchify(jnp.asarray(images), _MODEL_PATCH_SIZE)

    torch_output = torch_embedder(
        torch.from_numpy(np.asarray(jax_patches)),
        torch.from_numpy(np.asarray(jax_positions)).long(),
    )
    jax_output = jax_projector(jax_embedder(jnp.asarray(images))[:, 0])

    assert_all_close_jax_torch(
        jax_output, torch_output, rtol=1e-4, atol=1e-4, error_msg="unified vision embedder outputs differ"
    )

  def test_padded_patches_only_change_their_own_rows(self):
    # HF pads short images with position (-1, -1); those rows must not leak into the
    # embeddings of real patches.
    setup_test_seeds()
    torch_embedder, _, _ = self._build()

    patch_dim = _MODEL_PATCH_SIZE * _MODEL_PATCH_SIZE * 3
    patches = torch.from_numpy(np.random.rand(1, _NUM_SOFT_TOKENS, patch_dim).astype(np.float32))
    positions = torch.from_numpy(
        np.stack(np.meshgrid(np.arange(4), np.arange(3)), axis=-1).reshape(1, -1, 2).astype(np.int64)
    )
    padded_positions = positions.clone()
    padded_positions[:, -2:] = -1

    full = torch_embedder(patches, positions)
    padded = torch_embedder(patches, padded_positions)

    torch.testing.assert_close(full[:, :-2], padded[:, :-2])
    self.assertFalse(torch.allclose(full[:, -2:], padded[:, -2:]))


class TestGemma4UnifiedTextTower(unittest.TestCase):
  """Whole-tower logit parity, with weights moved across by the real param mapping.

  The tower is shrunk but keeps every structural property the mapping depends on:
  two full period-6 blocks (so both the sliding and the global layer types appear
  more than once), a wider global head dim, a single global KV head, shared global
  K/V projections, per-layer scalars and final logit softcapping.
  """

  NUM_LAYERS = 12
  EMB_DIM = 128
  NUM_QUERY_HEADS = 4
  NUM_KV_HEADS = 2
  HEAD_DIM = 16
  GLOBAL_HEAD_DIM = 32
  MLP_DIM = 256
  VOCAB_SIZE = 256

  def _torch_text_config(self):
    """Builds the PyTorch text config for the shrunken tower."""
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    config = Gemma4UnifiedTextConfig(
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        enable_moe_block=False,
        final_logit_softcapping=30.0,
        global_head_dim=self.GLOBAL_HEAD_DIM,
        head_dim=self.HEAD_DIM,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=self.EMB_DIM,
        hidden_size_per_layer_input=0,
        intermediate_size=self.MLP_DIM,
        layer_types=layer_types * (self.NUM_LAYERS // len(layer_types)),
        max_position_embeddings=4096,
        num_attention_heads=self.NUM_QUERY_HEADS,
        num_experts=None,
        num_global_key_value_heads=1,
        num_hidden_layers=self.NUM_LAYERS,
        num_key_value_heads=self.NUM_KV_HEADS,
        num_kv_shared_layers=0,
        rms_norm_eps=1e-6,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
        sliding_window=1024,
        tie_word_embeddings=True,
        top_k_experts=None,
        use_cache=False,
        use_double_wide_mlp=False,
        vocab_size=self.VOCAB_SIZE,
        vocab_size_per_layer_input=self.VOCAB_SIZE,
    )
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    return config

  def _maxtext_config(self):
    """Builds the matching MaxText config for the shrunken tower."""
    return pyconfig.initialize(
        ["", base_config_path],
        model_name="gemma4-12b",
        use_multimodal=False,
        scan_layers=False,
        override_model_config=True,
        skip_jax_distributed_system=True,
        enable_checkpointing=False,
        run_name="gemma4_unified_text_parity",
        per_device_batch_size=1,
        max_target_length=32,
        max_prefill_predict_length=16,
        base_num_decoder_layers=self.NUM_LAYERS,
        base_emb_dim=self.EMB_DIM,
        base_num_query_heads=self.NUM_QUERY_HEADS,
        base_num_kv_heads=self.NUM_KV_HEADS,
        head_dim=self.HEAD_DIM,
        global_head_dim=self.GLOBAL_HEAD_DIM,
        global_num_kv_heads=1,
        base_mlp_dim=self.MLP_DIM,
        vocab_size=self.VOCAB_SIZE,
        attention="dot_product",
        matmul_precision="highest",
        dtype="float32",
        weight_dtype="float32",
        float32_logits=True,
        float32_qk_product=True,
        logits_dot_in_fp32=True,
    )

  def test_logits_match_torch_reference(self):
    setup_test_seeds()
    torch_config = self._torch_text_config()
    torch_model = TorchGemma4UnifiedForCausalLM(torch_config).eval()

    # Randomize every float tensor, including the layer_scalar buffers, so no part of
    # the mapping can pass by accidentally being left at its initial value.
    state_dict = torch_model.state_dict()
    for name, tensor in state_dict.items():
      if tensor.dtype.is_floating_point:
        offset = 1.0 if "layer_scalar" in name else 0.0
        state_dict[name] = torch.randn_like(tensor) * (0.05 if tensor.ndim > 1 else 1.0) + offset
    torch_model.load_state_dict(state_dict)

    maxtext_config = self._maxtext_config()
    hf_config = {"text_config": torch_config.to_dict()}
    mapping = PARAM_MAPPING["gemma4-12b"](hf_config, maxtext_config, False)
    hooks = HOOK_FNS["gemma4-12b"](hf_config, maxtext_config, False, saving_to_hf=False)
    abstract_params, treedef = get_maxtext_model_info(maxtext_config)

    torch_weights = {k: v.numpy().astype(np.float32) for k, v in torch_model.state_dict().items()}
    leaves = [None] * len(abstract_params)
    for maxtext_key, (index, target_shape) in abstract_params.items():
      weight = torch_weights[mapping[maxtext_key]]
      hook = hooks.get(maxtext_key)
      weight = hook(weight, target_shape) if hook else weight.reshape(target_shape)
      self.assertEqual(tuple(weight.shape), tuple(target_shape), msg=maxtext_key)
      leaves[index] = jnp.asarray(weight)
    params = jax.tree_util.tree_unflatten(treedef, leaves)

    devices = np.array(jax.devices()).reshape([1] * len(maxtext_config.mesh_axes))
    mesh = jax.sharding.Mesh(devices, maxtext_config.mesh_axes)
    maxtext_model = models.transformer_as_linen(maxtext_config, mesh, quant=None, model_mode=MODEL_MODE_TRAIN)

    ids = np.array([[3, 17, 42, 8, 91, 5, 60, 12]], dtype=np.int32)
    positions = np.arange(ids.shape[1], dtype=np.int32)[None]
    maxtext_logits = maxtext_model.apply(
        {"params": params},
        jnp.asarray(ids),
        jnp.asarray(positions),
        decoder_segment_ids=jnp.ones_like(jnp.asarray(ids)),
        enable_dropout=False,
    )
    torch_logits = torch_model(input_ids=torch.from_numpy(ids.astype(np.int64))).logits

    assert_all_close_jax_torch(
        maxtext_logits, torch_logits, rtol=1e-2, atol=2e-2, error_msg="gemma4-12b text tower logits differ"
    )
    np.testing.assert_array_equal(np.asarray(maxtext_logits).argmax(-1), torch_logits.numpy().argmax(-1))


if __name__ == "__main__":
  unittest.main()
