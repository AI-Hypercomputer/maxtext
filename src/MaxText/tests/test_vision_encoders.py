import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
import os

from flax import nnx
from jax.sharding import Mesh
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionPatchMerger as TorchQwen3OmniMoeVisionPatchMerger,
    Qwen3OmniMoeVisionMLP as TorchQwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed as TorchQwen3OmniMoeVisionPatchEmbed,
    Qwen3OmniMoeVisionEncoder as TorchQwen3OmniMoeVisionEncoder,
)

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    apply_rotary_pos_emb_vision,
)

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)
from MaxText.layers.vision_encoders import (
    Qwen3OmniMoeVisionAttention as JaxQwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionPatchMerger as JaxQwen3OmniMoeVisionPatchMerger,
    Qwen3OmniMoeVisionMLP as JaxQwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed as JaxQwen3OmniMoeVisionPatchEmbed,
)
from MaxText.layers.embeddings import (
    Qwen3OmniMoeVisionRotaryEmbedding as JaxQwen3OmniMoeVisionRotaryEmbedding,
    Qwen3OmniMoeVisionPosEmbedInterpolate as JaxQwen3OmniMoeVisionPosEmbedInterpolate,
)
from MaxText import pyconfig
from MaxText.tests.test_utils import (
    copy_attention_weights_to_maxtext,
    copy_patch_embed_weights,
    copy_mlp_weights,
    copy_patch_merger_weights,
    copy_vision_encoder_weights,
    create_random_jax_torch,
    assert_all_close_jax_torch,
)

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    apply_rotary_pos_emb_vision,
)

base_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yml")
jax_vision_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen3-omni-30b-a3b",
    attention="dot_product",
    attention_type="full",
)

torch.set_grad_enabled(False)


def print_configs():
    print("\n" + "=" * 80)
    print("VISION ENCODER TEST CONFIGS")
    print("=" * 80)
    print("\nJAX/MaxText Vision Config:")
    print(repr(jax_vision_config))
    print("=" * 80 + "\n")


class TestQwen3OmniMoeVisionAttention(unittest.TestCase):
    def setUp(self):
        self.config = jax_vision_config
        self.seq_length = 16
        self.hidden_size = self.config.hidden_size_for_vit
        self.num_heads = self.config.num_attention_heads_for_vit
        np.random.seed(42)
        torch.manual_seed(42)

        # Create mesh following audio encoder pattern
        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def test_attention_output_matches_torch(self):
        """Test that JAX vision attention output matches PyTorch implementation."""
        # Create PyTorch encoder (need full encoder to access rotary embeddings)
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_heads=self.config.num_attention_heads_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            depth=self.config.num_hidden_layers_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            patch_size=self.config.patch_size_for_vit,
            temporal_patch_size=self.config.temporal_patch_size_for_vit,
            in_channels=self.config.num_channels_for_vit,
        )
        torch_config._attn_implementation = "eager"

        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)
        torch_encoder.eval()
        torch_model = torch_encoder.blocks[0].attn

        # Create JAX model
        jax_model = JaxQwen3OmniMoeVisionAttention(
            config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42)
        )

        # Copy weights (vision uses fused QKV projection)
        copy_attention_weights_to_maxtext(torch_model, jax_model.attn, fused_qkv=True)

        # Create test input: 16 tokens = 1 frame, 4x4 patches
        jax_hidden_states_2d, torch_hidden_states = create_random_jax_torch(
            self.seq_length, self.hidden_size
        )
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32)

        cu_seqlens = torch.tensor([0, self.seq_length], dtype=torch.int32)

        # Compute rotary position embeddings for PyTorch
        rotary_pos_emb = torch_encoder.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(self.seq_length, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Debug: Show PyTorch Q/K after RoPE
        qkv = torch_model.qkv(torch_hidden_states)
        q_torch, k_torch, v_torch = (
            qkv.reshape(self.seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q_rope_torch, k_rope_torch = apply_rotary_pos_emb_vision(
            q_torch, k_torch, cos, sin
        )
        # Forward pass - manually compute to get intermediate values
        torch_output = torch_model(
            torch_hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

        # JAX forward pass - VisionAttention expects (seq_len, hidden_size)
        # and internally adds batch=1 dimension
        jax_output = jax_model(
            jax_hidden_states_2d,  # Shape: (seq_len, hidden_size)
            grid_thw=jnp.array(grid_thw.numpy()),
            deterministic=True,
        )

        assert_all_close_jax_torch(
            jax_output,
            torch_output,
            rtol=1e-2,
            atol=1e-2,
            error_msg="Vision attention outputs differ",
        )

    def test_attention_is_jittable(self):
        """Test that attention is JIT-compilable."""
        model = JaxQwen3OmniMoeVisionAttention(
            config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42)
        )
        hidden_states = jnp.ones((1, 16, self.hidden_size))
        grid_thw = jnp.array([[1, 4, 4]], dtype=jnp.int32)

        @nnx.jit
        def forward(model, hidden_states, grid_thw):
            return model(hidden_states, grid_thw=grid_thw, deterministic=True)

        output = forward(model, hidden_states, grid_thw)


class TestQwen3OmniMoeVisionPatchMerger(unittest.TestCase):
    def setUp(self):
        self.config = jax_vision_config
        np.random.seed(42)
        torch.manual_seed(42)

    def test_patch_merger_output_matches_torch_without_postshuffle(self):
        """Test patch merger without postshuffle_norm matches PyTorch."""
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            out_hidden_size=self.config.out_hidden_size_for_vit,
        )
        torch_model = TorchQwen3OmniMoeVisionPatchMerger(
            torch_config, use_postshuffle_norm=False
        )
        torch_model.eval()

        jax_model = JaxQwen3OmniMoeVisionPatchMerger(
            config=self.config, use_postshuffle_norm=False, rngs=nnx.Rngs(42)
        )

        # Copy weights
        copy_patch_merger_weights(torch_model, jax_model)

        seq_len = 64  # Will become 16 after merging (64 / 4)
        jax_hidden_states, torch_hidden_states = create_random_jax_torch(
            seq_len, self.config.hidden_size_for_vit
        )

        # Forward pass
        torch_output = torch_model(torch_hidden_states)
        jax_output = jax_model(jax_hidden_states)

        # Compare outputs
        assert_all_close_jax_torch(jax_output, torch_output, rtol=1e-3, atol=3e-3)

    def test_patch_merger_output_matches_torch_with_postshuffle(self):
        """Test patch merger with postshuffle_norm matches PyTorch."""
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            out_hidden_size=self.config.out_hidden_size_for_vit,
        )
        torch_model = TorchQwen3OmniMoeVisionPatchMerger(
            torch_config, use_postshuffle_norm=True
        )
        torch_model.eval()

        jax_model = JaxQwen3OmniMoeVisionPatchMerger(
            config=self.config, use_postshuffle_norm=True, rngs=nnx.Rngs(42)
        )

        copy_patch_merger_weights(torch_model, jax_model)

        # Create test input
        seq_len = 64  # Will become 16 after merging (64 / 4)
        jax_hidden_states, torch_hidden_states = create_random_jax_torch(
            seq_len, self.config.hidden_size_for_vit
        )

        # Forward pass
        torch_output = torch_model(torch_hidden_states)
        jax_output = jax_model(jax_hidden_states)

        # Compare outputs
        assert_all_close_jax_torch(jax_output, torch_output, rtol=1e-3, atol=3e-3)

    def test_patch_merger_is_jittable(self):
        """Test that patch merger is JIT-compilable."""
        model = JaxQwen3OmniMoeVisionPatchMerger(
            config=self.config, use_postshuffle_norm=False, rngs=nnx.Rngs(42)
        )

        @nnx.jit
        def forward(model, hidden_states):
            return model(hidden_states)

        seq_len = 64
        hidden_states = jnp.ones((seq_len, self.config.hidden_size_for_vit))
        output = forward(model, hidden_states)


class TestQwen3OmniMoeVisionMLP(unittest.TestCase):
    def setUp(self):
        self.config = jax_vision_config
        np.random.seed(42)
        torch.manual_seed(42)

    def test_mlp_output_matches_torch(self):
        """Test that JAX MLP output matches PyTorch implementation."""
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            intermediate_size=self.config.intermediate_size_for_vit,
            hidden_act=self.config.hidden_act_for_vit,
        )
        torch_model = TorchQwen3OmniMoeVisionMLP(torch_config)
        torch_model.eval()

        jax_model = JaxQwen3OmniMoeVisionMLP(config=self.config, rngs=nnx.Rngs(42))

        copy_mlp_weights(torch_model, jax_model)

        # Create test input
        seq_len = 16
        jax_hidden_states, torch_hidden_states = create_random_jax_torch(
            seq_len, self.config.hidden_size_for_vit
        )

        # Forward pass
        torch_output = torch_model(torch_hidden_states)
        jax_output = jax_model(jax_hidden_states)

        # Compare outputs
        assert_all_close_jax_torch(jax_output, torch_output, rtol=1e-4, atol=3e-3)

    def test_mlp_is_jittable(self):
        """Test that MLP is JIT-compilable."""
        model = JaxQwen3OmniMoeVisionMLP(config=self.config, rngs=nnx.Rngs(42))

        @nnx.jit
        def forward(model, hidden_states):
            return model(hidden_states)

        hidden_states = jnp.ones((16, self.config.hidden_size_for_vit))
        output = forward(model, hidden_states)

        self.assertEqual(output.shape, (16, self.config.hidden_size_for_vit))


class TestQwen3OmniMoeVisionPatchEmbed(unittest.TestCase):
    def setUp(self):
        self.config = jax_vision_config
        np.random.seed(42)
        torch.manual_seed(42)

    def test_patch_embed_output_matches_torch(self):
        """Test that JAX patch embed output matches PyTorch implementation."""
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            patch_size=self.config.patch_size_for_vit,
            temporal_patch_size=self.config.temporal_patch_size_for_vit,
            in_channels=self.config.num_channels_for_vit,
        )
        torch_model = TorchQwen3OmniMoeVisionPatchEmbed(torch_config)
        torch_model.eval()

        jax_model = JaxQwen3OmniMoeVisionPatchEmbed(
            config=self.config, rngs=nnx.Rngs(42)
        )

        copy_patch_embed_weights(torch_model, jax_model)

        batch_size = 2
        total_elements = (
            batch_size
            * self.config.num_channels_for_vit
            * self.config.temporal_patch_size_for_vit
            * self.config.patch_size_for_vit
            * self.config.patch_size_for_vit
        )
        jax_hidden_states, torch_hidden_states = create_random_jax_torch(total_elements)

        # Forward pass
        torch_output = torch_model(torch_hidden_states)
        jax_output = jax_model(jax_hidden_states)

        # Compare outputs
        assert_all_close_jax_torch(jax_output, torch_output, rtol=1e-3, atol=5e-3)

    def test_patch_embed_is_jittable(self):
        """Test that patch embed is JIT-compilable."""
        model = JaxQwen3OmniMoeVisionPatchEmbed(config=self.config, rngs=nnx.Rngs(42))

        @nnx.jit
        def forward(model, hidden_states):
            return model(hidden_states)

        batch_size = 2
        total_elements = (
            batch_size
            * self.config.num_channels_for_vit
            * self.config.temporal_patch_size_for_vit
            * self.config.patch_size_for_vit
            * self.config.patch_size_for_vit
        )
        hidden_states = jnp.ones(total_elements)
        output = forward(model, hidden_states)


class TestQwen3OmniMoeVisionRotaryEmbedding(unittest.TestCase):
    """Test the grid-based rotary embedding from embeddings.py against PyTorch."""

    def setUp(self):
        self.config = jax_vision_config
        np.random.seed(42)
        torch.manual_seed(42)

    def test_grid_based_embedding_matches_torch(self):
        """Test that JAX grid-based rotary embedding matches PyTorch implementation."""
        head_dim = (
            self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
        )

        jax_model = JaxQwen3OmniMoeVisionRotaryEmbedding(
            head_dim=head_dim,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            cast_as_fprop_dtype=False,
            fprop_dtype=jnp.float32,
            rngs=nnx.Rngs(42),
        )

        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_heads=self.config.num_attention_heads_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            depth=self.config.num_hidden_layers_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
        )
        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)
        torch_encoder.eval()

        grid_thw_np = np.array([[1, 8, 8]], dtype=np.int64)
        grid_thw_jax = jnp.array(grid_thw_np)
        grid_thw_torch = torch.from_numpy(grid_thw_np)

        cos_emb_jax, sin_emb_jax = jax_model.compute_cos_sin(grid_thw_jax)

        rotary_pos_emb = torch_encoder.rot_pos_emb(grid_thw_torch)
        embeddings = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
        cos_emb_torch = embeddings.cos()
        sin_emb_torch = embeddings.sin()

        assert_all_close_jax_torch(cos_emb_jax, cos_emb_torch, rtol=1e-5, atol=1e-5)
        assert_all_close_jax_torch(sin_emb_jax, sin_emb_torch, rtol=1e-5, atol=1e-5)

    def test_grid_based_embedding_shapes(self):
        """Test that grid-based rotary embedding produces correct shapes."""
        head_dim = (
            self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
        )

        model = JaxQwen3OmniMoeVisionRotaryEmbedding(
            head_dim=head_dim,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            rngs=nnx.Rngs(42),
        )

        grid_thw = jnp.array([[1, 16, 16]])
        cos_emb, sin_emb = model.compute_cos_sin(grid_thw)
        self.assertEqual(
            cos_emb.shape, (256, head_dim)
        )  # Returns head_dim after doubling
        self.assertEqual(sin_emb.shape, (256, head_dim))

        # Multiple images/videos: 16x16 + 4*(8x8) = 256 + 256 = 512 tokens
        grid_thw = jnp.array([[1, 16, 16], [4, 8, 8]])
        cos_emb, sin_emb = model.compute_cos_sin(grid_thw)
        self.assertEqual(cos_emb.shape, (512, head_dim))

    def test_rotation_application_matches_torch(self):
        """Test that applying rotary embedding to Q/K tensors matches PyTorch."""
        head_dim = (
            self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
        )

        jax_model = JaxQwen3OmniMoeVisionRotaryEmbedding(
            head_dim=head_dim,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            cast_as_fprop_dtype=False,
            fprop_dtype=jnp.float32,
            rngs=nnx.Rngs(42),
        )

        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_heads=self.config.num_attention_heads_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            depth=self.config.num_hidden_layers_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
        )
        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)
        torch_encoder.eval()

        # Test with single image: 1 frame, 8x8 patches
        grid_thw_np = np.array([[1, 8, 8]], dtype=np.int64)
        grid_thw_jax = jnp.array(grid_thw_np)
        grid_thw_torch = torch.from_numpy(grid_thw_np)

        seq_len = 64  # 8x8
        q_jax, q_torch = create_random_jax_torch(
            seq_len, self.config.num_attention_heads_for_vit, head_dim
        )
        k_jax, k_torch = create_random_jax_torch(
            seq_len, self.config.num_attention_heads_for_vit, head_dim
        )

        q_rotated_jax = jax_model(q_jax, grid_thw_jax)
        k_rotated_jax = jax_model(k_jax, grid_thw_jax)

        rotary_pos_emb = torch_encoder.rot_pos_emb(grid_thw_torch)
        embeddings = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
        cos = embeddings.cos()  # [seq_len, head_dim]
        sin = embeddings.sin()  # [seq_len, head_dim]

        q_rotated_torch, k_rotated_torch = apply_rotary_pos_emb_vision(
            q_torch, k_torch, cos, sin
        )

        assert_all_close_jax_torch(
            q_rotated_jax,
            q_rotated_torch,
            rtol=1e-3,
            atol=1e-4,
            error_msg="Q rotation mismatch",
        )
        assert_all_close_jax_torch(
            k_rotated_jax,
            k_rotated_torch,
            rtol=1e-3,
            atol=1e-4,
            error_msg="K rotation mismatch",
        )


class TestQwen3OmniMoeVisionPosEmbedInterpolate(unittest.TestCase):
    """Test bilinear position embedding interpolation from embeddings.py."""

    def setUp(self):
        self.config = jax_vision_config
        np.random.seed(42)
        torch.manual_seed(42)

    def test_pos_embed_interpolate_matches_torch(self):
        """Test that JAX position embedding interpolation matches PyTorch encoder."""
        jax_model = JaxQwen3OmniMoeVisionPosEmbedInterpolate(
            num_position_embeddings=self.config.num_position_embeddings_for_vit,
            hidden_size=self.config.hidden_size_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            dtype=jnp.float32,
            rngs=nnx.Rngs(42),
        )

        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_position_embeddings=self.config.num_position_embeddings_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
        )
        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)

        torch_pos_embed_weight = torch_encoder.pos_embed.weight.detach().cpu().numpy()
        jax_model.pos_embed.value = jnp.array(torch_pos_embed_weight)

        # Test with single 16x16 image
        grid_thw_np = np.array([[1, 16, 16]], dtype=np.int64)
        grid_thw_jax = jnp.array(grid_thw_np)
        grid_thw_torch = torch.from_numpy(grid_thw_np)

        pos_embed_jax = jax_model(grid_thw_jax)
        pos_embed_torch = torch_encoder.fast_pos_embed_interpolate(grid_thw_torch)

        assert_all_close_jax_torch(
            pos_embed_jax, pos_embed_torch, rtol=1e-4, atol=5e-5
        )

    def test_pos_embed_interpolate_multiple_images(self):
        """Test position embedding interpolation with multiple images/videos."""
        jax_model = JaxQwen3OmniMoeVisionPosEmbedInterpolate(
            num_position_embeddings=self.config.num_position_embeddings_for_vit,
            hidden_size=self.config.hidden_size_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            dtype=jnp.float32,
            rngs=nnx.Rngs(42),
        )

        # Create PyTorch encoder
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_position_embeddings=self.config.num_position_embeddings_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
        )
        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)

        # Copy weights
        torch_pos_embed_weight = torch_encoder.pos_embed.weight.detach().cpu().numpy()
        jax_model.pos_embed.value = jnp.array(torch_pos_embed_weight)

        # Test with multiple images: 1 image (8x8) + 1 video (2 frames, 16x16)
        grid_thw_np = np.array([[1, 8, 8], [2, 16, 16]], dtype=np.int64)
        grid_thw_jax = jnp.array(grid_thw_np)
        grid_thw_torch = torch.from_numpy(grid_thw_np)

        pos_embed_jax = jax_model(grid_thw_jax)

        pos_embed_torch = torch_encoder.fast_pos_embed_interpolate(grid_thw_torch)

        # Compare outputs
        assert_all_close_jax_torch(
            pos_embed_jax, pos_embed_torch, rtol=1e-4, atol=5e-5
        )


class TestQwen3OmniMoeVisionEncoderEndToEnd(unittest.TestCase):
    """End-to-end test for the full vision encoder."""

    def setUp(self):
        self.config = jax_vision_config
        np.random.seed(42)
        torch.manual_seed(42)

        # Create mesh for attention
        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def test_vision_encoder_single_image(self):
        """Test full vision encoder with single image matches PyTorch."""
        # Create PyTorch encoder
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_heads=self.config.num_attention_heads_for_vit,
            intermediate_size=self.config.intermediate_size_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            depth=self.config.num_hidden_layers_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            patch_size=self.config.patch_size_for_vit,
            temporal_patch_size=self.config.temporal_patch_size_for_vit,
            in_channels=self.config.num_channels_for_vit,
            num_position_embeddings=self.config.num_position_embeddings_for_vit,
            out_hidden_size=self.config.out_hidden_size_for_vit,
            deepstack_visual_indexes=list(self.config.deepstack_visual_indexes_for_vit),
            hidden_act=self.config.hidden_act_for_vit,
        )
        torch_config._attn_implementation = "eager"
        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)
        torch_encoder.eval()

        # Create JAX encoder
        from MaxText.layers.vision_encoders import Qwen3OmniMoeVisionEncoder as JaxQwen3OmniMoeVisionEncoder
        jax_encoder = JaxQwen3OmniMoeVisionEncoder(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42))

        # Copy weights from PyTorch to JAX
        copy_vision_encoder_weights(torch_encoder, jax_encoder)

        # Create test input: single 8x8 image = 64 tokens after patch embed
        # Input before patch embed: (batch * temporal * patch_h * patch_w * channels)
        patch_size = self.config.patch_size_for_vit
        temporal_patch_size = self.config.temporal_patch_size_for_vit
        in_channels = self.config.num_channels_for_vit
        h, w = 8, 8  # 8x8 patches

        total_elements = 1 * in_channels * temporal_patch_size * (h * patch_size) * (w * patch_size)
        jax_hidden_states, torch_hidden_states = create_random_jax_torch(total_elements)

        grid_thw = np.array([[1, h, w]], dtype=np.int64)
        grid_thw_jax = jnp.array(grid_thw)
        grid_thw_torch = torch.from_numpy(grid_thw)

        # Forward pass
        torch_output, torch_deep_feats = torch_encoder(torch_hidden_states, grid_thw_torch)
        jax_output, jax_deep_feats = jax_encoder(jax_hidden_states, grid_thw_jax)

        # Compare final output
        assert_all_close_jax_torch(
            jax_output,
            torch_output,
            rtol=1e-2,
            atol=1e-2,
            error_msg="Vision encoder final output differs"
        )

        # Compare deep features
        self.assertEqual(len(jax_deep_feats), len(torch_deep_feats),
                        "Number of deep features should match")
        for i, (jax_feat, torch_feat) in enumerate(zip(jax_deep_feats, torch_deep_feats)):
            assert_all_close_jax_torch(
                jax_feat,
                torch_feat,
                rtol=1e-2,
                atol=1e-2,
                error_msg=f"Deep feature {i} differs"
            )

    def test_vision_encoder_multiple_images(self):
        """Test full vision encoder with multiple images of different sizes."""
        # Create PyTorch encoder
        torch_config = Qwen3OmniMoeVisionEncoderConfig(
            hidden_size=self.config.hidden_size_for_vit,
            num_heads=self.config.num_attention_heads_for_vit,
            intermediate_size=self.config.intermediate_size_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            depth=self.config.num_hidden_layers_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            patch_size=self.config.patch_size_for_vit,
            temporal_patch_size=self.config.temporal_patch_size_for_vit,
            in_channels=self.config.num_channels_for_vit,
            num_position_embeddings=self.config.num_position_embeddings_for_vit,
            out_hidden_size=self.config.out_hidden_size_for_vit,
            deepstack_visual_indexes=list(self.config.deepstack_visual_indexes_for_vit),
            hidden_act=self.config.hidden_act_for_vit,
        )
        torch_config._attn_implementation = "eager"
        torch_encoder = TorchQwen3OmniMoeVisionEncoder(torch_config)
        torch_encoder.eval()

        # Create JAX encoder
        from MaxText.layers.vision_encoders import Qwen3OmniMoeVisionEncoder as JaxQwen3OmniMoeVisionEncoder
        jax_encoder = JaxQwen3OmniMoeVisionEncoder(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42))

        # Copy weights
        copy_vision_encoder_weights(torch_encoder, jax_encoder)

        # Create test input: 2 images of sizes 4x4 and 8x8
        patch_size = self.config.patch_size_for_vit
        temporal_patch_size = self.config.temporal_patch_size_for_vit
        in_channels = self.config.num_channels_for_vit

        # First image: 4x4 patches
        h1, w1 = 4, 4
        elements1 = 1 * in_channels * temporal_patch_size * (h1 * patch_size) * (w1 * patch_size)

        # Second image: 8x8 patches
        h2, w2 = 8, 8
        elements2 = 1 * in_channels * temporal_patch_size * (h2 * patch_size) * (w2 * patch_size)

        total_elements = elements1 + elements2
        jax_hidden_states, torch_hidden_states = create_random_jax_torch(total_elements)

        grid_thw = np.array([[1, h1, w1], [1, h2, w2]], dtype=np.int64)
        grid_thw_jax = jnp.array(grid_thw)
        grid_thw_torch = torch.from_numpy(grid_thw)

        # Forward pass
        torch_output, torch_deep_feats = torch_encoder(torch_hidden_states, grid_thw_torch)
        jax_output, jax_deep_feats = jax_encoder(jax_hidden_states, grid_thw_jax)

        # Compare outputs
        assert_all_close_jax_torch(
            jax_output,
            torch_output,
            rtol=1e-2,
            atol=1e-2,
            error_msg="Vision encoder output differs for multiple images"
        )

        # Compare deep features
        for i, (jax_feat, torch_feat) in enumerate(zip(jax_deep_feats, torch_deep_feats)):
            assert_all_close_jax_torch(
                jax_feat,
                torch_feat,
                rtol=1e-2,
                atol=1e-2,
                error_msg=f"Deep feature {i} differs for multiple images"
            )


if __name__ == "__main__":
    print_configs()
    unittest.main()
