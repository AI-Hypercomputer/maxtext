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

"""Unit tests for Cosmos 3 PR 2.1 primitives with direct Diffusers reference parity."""

import unittest

from diffusers.models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
    Cosmos3VLTextRotaryEmbedding,
    TimestepEmbedding,
    Timesteps,
    _rotate_half as _torch_rotate_half,
)
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from maxtext.layers.embeddings import (
    Cosmos3RotaryEmbedding,
    Cosmos3TimeEmbedder,
    Cosmos3Timesteps,
    apply_cosmos3_rope,
    apply_timestep_embeds_to_noisy_tokens,
    build_cosmos3_3d_mrope_position_ids,
    get_3d_mrope_ids_vae_tokens,
    get_cosmos3_timestep_embedding,
    patchify_and_pack_latents,
    patchify_latents,
    unpatchify_and_unpack_latents,
    unpatchify_latents,
)


class Cosmos3EmbeddingsTest(unittest.TestCase):
  """Verifies numerical parity against official Diffusers reference modules on CPU and TPU."""

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.manual_seed(42)
    np.random.seed(42)

  def test_sinusoidal_timestep_embedding_parity(self):
    timesteps_np = np.array([0.0, 1.0, 12.5, 500.0, 999.9], dtype=np.float32) * 0.001
    pt_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0.0)
    pt_out = pt_proj(torch.from_numpy(timesteps_np)).numpy()

    jax_out = np.asarray(
        get_cosmos3_timestep_embedding(
            jnp.asarray(timesteps_np),
            embedding_dim=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
    )
    np.testing.assert_allclose(jax_out, pt_out, atol=1e-6, rtol=1e-6)

    # Batched 2D timesteps [B, T] for training
    batched_t = jnp.asarray(timesteps_np.reshape(1, 5))
    batched_out = np.asarray(Cosmos3Timesteps(num_channels=256)(batched_t))
    np.testing.assert_allclose(batched_out[0], pt_out, atol=1e-6, rtol=1e-6)

  def test_cosmos3_time_embedder_weight_copy_parity_and_grads(self):
    torch.manual_seed(42)
    in_channels = 256
    embed_dim = 512
    pt_embedder = TimestepEmbedding(in_channels=in_channels, time_embed_dim=embed_dim).eval()

    jax_embedder = Cosmos3TimeEmbedder(
        in_channels=in_channels,
        time_embed_dim=embed_dim,
        timestep_scale=0.001,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    jax_embedder.load_from_torch(
        pt_embedder.linear_1.weight.detach().numpy(),
        pt_embedder.linear_1.bias.detach().numpy(),
        pt_embedder.linear_2.weight.detach().numpy(),
        pt_embedder.linear_2.bias.detach().numpy(),
    )

    raw_timesteps = np.array([10.0, 250.0, 500.0, 875.5], dtype=np.float32)
    with torch.no_grad():
      pt_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0.0)
      pt_sin = pt_proj(torch.from_numpy(raw_timesteps * 0.001))
      pt_expected = pt_embedder(pt_sin).numpy()

    # 1. Direct raw timesteps call
    jax_actual = np.asarray(jax_embedder(jnp.asarray(raw_timesteps)))
    np.testing.assert_allclose(jax_actual, pt_expected, atol=2e-3, rtol=2e-3)

    # 2. Pre-projected sinusoidal input call
    jax_from_sin = np.asarray(jax_embedder(jnp.asarray(pt_sin.numpy())))
    np.testing.assert_allclose(jax_from_sin, pt_expected, atol=2e-3, rtol=2e-3)

    # 3. JIT & gradient flow check for training
    @jax.jit
    def loss_fn(model, t):
      out = model(t, target_dtype=jnp.bfloat16)
      return jnp.mean(out.astype(jnp.float32) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(jax_embedder, jnp.asarray(raw_timesteps))
    self.assertTrue(np.isfinite(float(loss)))
    self.assertGreater(float(jnp.linalg.norm(grads.linear_1.kernel[...])), 0.0)
    self.assertGreater(float(jnp.linalg.norm(grads.linear_2.kernel[...])), 0.0)

  def test_apply_timestep_embeds_to_noisy_tokens_serving_and_training(self):
    # 1 frame conditioned (frame 0 clean), 2 frames noisy (frames 1, 2)
    # token_shape = (T=3, H_p=2, W_p=2) -> 4 tokens per frame, 12 tokens total, dim=8
    np.random.seed(7)
    tokens_np = np.random.randn(12, 8).astype(np.float32)
    noisy_t_embeds_np = np.random.randn(8, 8).astype(np.float32)  # 2 noisy frames * 4 spatial tokens

    token_shapes = [(3, 2, 2)]

    # Expected: frames 1 and 2 (token indices 4..11) get noisy_t_embeds added, tokens 0..3 untouched
    expected = tokens_np.copy()
    expected[4:12] += noisy_t_embeds_np

    # Serving / packed index mode
    jax_indexed = np.asarray(
        apply_timestep_embeds_to_noisy_tokens(
            jnp.asarray(tokens_np),
            jnp.asarray(noisy_t_embeds_np),
            noisy_frame_indexes=[jnp.array([1, 2])],
            token_shapes=token_shapes,
        )
    )
    np.testing.assert_allclose(jax_indexed, expected, atol=1e-6)

    # Training / static-shape masked mode ([B=1, T*S=12, D=8] with per-frame embed [B=1, T=3, D=8] and mask [1, 3])
    per_frame_t = np.random.randn(1, 3, 8).astype(np.float32)
    mask = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)
    masked_out = np.asarray(
        apply_timestep_embeds_to_noisy_tokens(
            jnp.asarray(tokens_np[None, ...]),
            jnp.asarray(per_frame_t),
            noisy_frame_mask=jnp.asarray(mask),
        )
    )
    np.testing.assert_allclose(masked_out[0, :4], tokens_np[:4], atol=1e-6)
    np.testing.assert_allclose(masked_out[0, 4:8], tokens_np[4:8] + per_frame_t[0, 1], atol=1e-6)
    np.testing.assert_allclose(masked_out[0, 8:12], tokens_np[8:12] + per_frame_t[0, 2], atol=1e-6)

    # Single timestep embedding shared across all frames: shape [B=1, 1, D=8] with packed tokens [B=1, T*S=12, D=8]
    shared_t = np.random.randn(1, 1, 8).astype(np.float32)
    shared_out = np.asarray(
        apply_timestep_embeds_to_noisy_tokens(
            jnp.asarray(tokens_np[None, ...]),
            jnp.asarray(shared_t),
            noisy_frame_mask=jnp.asarray(mask),
        )
    )
    np.testing.assert_allclose(shared_out[0, :4], tokens_np[:4], atol=1e-6)
    np.testing.assert_allclose(shared_out[0, 4:], tokens_np[4:] + shared_t[0, 0], atol=1e-6)

  def test_3d_mrope_position_ids_and_rotary_parity(self):
    # Build joint 3D mRoPE position IDs (text + video + sound + action)
    pos_ids, vision_offset = build_cosmos3_3d_mrope_position_ids(
        text_len=6,
        vision_grid_thw=(3, 4, 4),
        temporal_modality_margin=15000,
        reset_spatial_indices=True,
        enable_fps_modulation=True,
        fps=16.0,
        base_fps=24.0,
        temporal_compression_factor=4,
        sound_len=5,
        sound_fps=25.0,
        action_len=4,
    )
    self.assertEqual(vision_offset, 15006)
    # Total tokens = 6 (text) + 48 (vision) + 5 (sound) + 4 (action) = 63
    self.assertEqual(pos_ids.shape, (3, 63))

    # Test JIT-friendliness of get_3d_mrope_ids_vae_tokens
    @jax.jit
    def jit_vae_ids(offset):
      ids, _ = get_3d_mrope_ids_vae_tokens(
          grid_t=3, grid_h=4, grid_w=4, temporal_offset=offset, fps=16.0, base_fps=24.0
      )
      return ids

    jit_ids = jit_vae_ids(100)
    self.assertEqual(jit_ids.shape, (3, 3 * 4 * 4))

    # Compare rotary cos/sin matrices against PyTorch Cosmos3VLTextRotaryEmbedding
    head_dim = 128
    rope_theta = 5000000.0
    rope_axes_dim = (24, 20, 20)

    pt_rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes_dim)
    jax_rope = Cosmos3RotaryEmbedding(
        head_dim=head_dim,
        rope_theta=rope_theta,
        rope_axes_dim=rope_axes_dim,
        cast_as_fprop_dtype=False,
        fprop_dtype=jnp.float32,
    )

    pt_pos = torch.from_numpy(np.asarray(pos_ids).copy()).unsqueeze(1)  # [3, 1, 63]
    pt_cos, pt_sin = pt_rope(pt_pos, device=torch.device("cpu"), dtype=torch.float32)
    pt_cos_np = pt_cos.squeeze(0).numpy()  # [63, 128]
    pt_sin_np = pt_sin.squeeze(0).numpy()  # [63, 128]

    # Test [3, N], [N, 3], [3, B, N], and [B, 3, N] layouts
    # Tolerances account for TPU transcendentals unit polynomial precision vs CPU float32 at position 15000+
    cos_3n, sin_3n = jax_rope(pos_ids, dtype=jnp.float32)
    cos_n3, sin_n3 = jax_rope(pos_ids.T, dtype=jnp.float32)
    cos_3bn, sin_3bn = jax_rope(pos_ids[:, None, :], dtype=jnp.float32)
    cos_b3n, sin_b3n = jax_rope(pos_ids[None, :, :], dtype=jnp.float32)

    np.testing.assert_allclose(np.asarray(cos_3n), pt_cos_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(sin_3n), pt_sin_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(cos_n3), pt_cos_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(sin_n3), pt_sin_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(cos_3bn[0]), pt_cos_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(sin_3bn[0]), pt_sin_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(cos_b3n[0]), pt_cos_np, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(sin_b3n[0]), pt_sin_np, atol=2e-2, rtol=2e-2)

    # Verify Q/K rotation parity against PyTorch _rotate_half
    q_np = np.random.randn(63, 8, head_dim).astype(np.float32)
    pt_q = torch.from_numpy(q_np)
    pt_q_rot = (pt_q * pt_cos.squeeze(0).unsqueeze(1) + _torch_rotate_half(pt_q) * pt_sin.squeeze(0).unsqueeze(1)).numpy()

    jax_q_rot = np.asarray(apply_cosmos3_rope(jnp.asarray(q_np), cos_3n, sin_3n))
    np.testing.assert_allclose(jax_q_rot, pt_q_rot, atol=0.05, rtol=0.05)

    # Verify 4D batched [B, S, H, D] Attention-style call `jax_rope(inputs, position)`
    jax_q_rot_4d = np.asarray(jax_rope(jnp.asarray(q_np[None, ...]), position=pos_ids[None, :, :]))
    np.testing.assert_allclose(jax_q_rot_4d[0], pt_q_rot, atol=0.05, rtol=0.05)

  def test_patchify_and_unpatchify_latents_parity(self):
    np.random.seed(123)
    # Test both divisible (H=4, W=6) and non-divisible (H=5, W=7) spatial shapes with C=48, patch_size=2
    latent_1_np = np.random.randn(1, 48, 3, 4, 6).astype(np.float32)
    latent_2_np = np.random.randn(1, 48, 2, 5, 7).astype(np.float32)

    pt_model = Cosmos3OmniTransformer(num_hidden_layers=1)
    pt_packed, pt_orig_shapes = pt_model._patchify_and_pack_latents(
        [torch.from_numpy(latent_1_np), torch.from_numpy(latent_2_np)]
    )
    jax_packed, jax_orig_shapes = patchify_and_pack_latents(
        [jnp.asarray(latent_1_np), jnp.asarray(latent_2_np)],
        patch_size=2,
    )

    self.assertEqual(jax_orig_shapes, pt_orig_shapes)
    self.assertEqual(jax_packed.shape[1], 192)
    np.testing.assert_allclose(np.asarray(jax_packed), pt_packed.numpy(), atol=0.0, rtol=0.0)

    # Test partial noisy-frame unpatchify (e.g. I2V where frame 0 is clean, remaining frames noisy)
    token_shapes = [(3, 2, 3), (2, 3, 4)]
    pt_noisy_idxs = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
    jax_noisy_idxs = [jnp.array([1, 2]), jnp.array([1])]

    # Noisy tokens count = (2 * 2 * 3) + (1 * 3 * 4) = 12 + 12 = 24
    preds_np = np.random.randn(24, 192).astype(np.float32)
    pt_unpacked = pt_model._unpatchify_and_unpack_latents(
        torch.from_numpy(preds_np),
        token_shapes_vision=token_shapes,
        noisy_frame_indexes_vision=pt_noisy_idxs,
        original_latent_shapes=pt_orig_shapes,
    )
    jax_unpacked = unpatchify_and_unpack_latents(
        jnp.asarray(preds_np),
        token_shapes_vision=token_shapes,
        noisy_frame_indexes_vision=jax_noisy_idxs,
        original_latent_shapes=jax_orig_shapes,
        patch_size=2,
        latent_channel=48,
    )

    for jax_item, pt_item in zip(jax_unpacked, pt_unpacked):
      np.testing.assert_allclose(np.asarray(jax_item), pt_item.numpy(), atol=0.0, rtol=0.0)

    # Test 5D batched [B, C=48, T, H, W] roundtrip for training
    batched_latents = jnp.asarray(np.random.randn(2, 48, 3, 6, 8).astype(np.float32))
    patches, grid_thw, orig_thw = patchify_latents(batched_latents, patch_size=2)
    self.assertEqual(patches.shape, (2, 3 * 3 * 4, 192))
    self.assertEqual(grid_thw, (3, 3, 4))
    reconstructed = unpatchify_latents(patches, latent_shape=orig_thw, patch_size=2, latent_channel=48)
    np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(batched_latents), atol=0.0, rtol=0.0)


if __name__ == "__main__":
  unittest.main()
