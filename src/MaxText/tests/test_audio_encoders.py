import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from MaxText.tests.test_utils import (
    copy_attention_weights_to_maxtext,
    copy_audio_model,
    copy_maxtext_encoder_layer_weights,
    copy_maxtext_encoder_weights,
    create_block_diagonal_attention_mask,
    create_random_jax_torch,
    assert_all_close_jax_torch,
)
import torch
import os

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention as TorchQwen3OmniMoeAudioAttention,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderLayer as TorchQwen3OmniMoeAudioEncoderLayer,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    SinusoidsPositionEmbedding as TorchSinusoidsPositionEmbedding,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as TorchQwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from MaxText.layers.audio_encoders import (
    AudioEncoderLayer,
    AudioEncoder,
    AudioModel,
    audio_encoder_get_feat_extract_output_lengths,
    compute_chunk_lengths,
    prepare_audio_chunks,
    generate_segment_ids,
)
from MaxText.layers.embeddings import SinusoidsPositionEmbedding
from MaxText.layers.attentions import Attention
from MaxText import common_types
from MaxText.layers.audio_encoders import (
    AudioModel,
    audio_encoder_get_feat_extract_output_lengths,
)
from MaxText import pyconfig
from jax.sharding import Mesh
import math


base_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yml")
jax_audio_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen3-omni-30b-a3b",
    attention="dot_product",
    attention_type="full",
    dropout_rate=0.0,
)

torch_audio_encoder_config = Qwen3OmniMoeAudioEncoderConfig(
    d_model=256,
    encoder_attention_heads=4,
    encoder_ffn_dim=512,
    encoder_layers=2,
    attention_dropout=0.0,
    dropout=0.0,
    activation_dropout=0.0,
    activation_function="gelu",
    num_mel_bins=128,
    max_source_positions=1500,
    scale_embedding=True,
    n_window=50,
    n_window_infer=800,
    conv_chunksize=500,
    downsample_hidden_size=256,
    output_dim=512,
)
torch_audio_encoder_config._attn_implementation = "eager"

torch.set_grad_enabled(False)


def print_configs():
    print("\n" + "=" * 80)
    print("AUDIO ENCODER TEST CONFIGS")
    print("=" * 80)
    print("\nJAX/MaxText Audio Config:")
    print(repr(jax_audio_config))
    print("\nPyTorch Reference Audio Config:")
    print(repr(torch_audio_encoder_config))
    print("=" * 80 + "\n")


class TestMaxTextAttentionVsPyTorch(unittest.TestCase):
    """Test that MaxText's Attention module matches PyTorch's audio attention implementation."""

    def setUp(self):
        self.batch_size = 1
        self.seq_length = 16
        self.config = jax_audio_config
        self.embed_dim = self.config.d_model_for_audio
        self.num_heads = self.config.encoder_attention_heads_for_audio
        self.head_dim = self.embed_dim // self.num_heads
        np.random.seed(42)
        torch.manual_seed(42)
        self.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))

    def test_attention_output_matches_torch(self):
        """Test that MaxText Attention produces same output as PyTorch attention."""
        torch_config = torch_audio_encoder_config
        torch_model = TorchQwen3OmniMoeAudioAttention(torch_config)
        torch_model.eval()

        # Create input - PyTorch expects (seq_length, channels), MaxText expects (batch, seq, channels)
        jax_hidden_states_2d, torch_hidden_states = create_random_jax_torch(
            self.seq_length, self.embed_dim
        )
        jax_hidden_states = jax_hidden_states_2d[
            jnp.newaxis, :, :
        ]  # Add batch dimension for MaxText

        # Create cu_seqlens for PyTorch (cumulative sequence lengths)
        cu_seqlens = torch.tensor([0, self.seq_length], dtype=torch.long)

        jax_attn = Attention(
            config=self.config,
            num_query_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            max_target_length=self.config.max_source_positions_for_audio,
            attention_kernel="dot_product",
            inputs_q_shape=(
                self.config.per_device_batch_size,
                self.seq_length,
                self.embed_dim,
            ),
            inputs_kv_shape=(
                self.config.per_device_batch_size,
                self.seq_length,
                self.embed_dim,
            ),
            float32_qk_product=self.config.float32_qk_product,
            float32_logits=self.config.float32_logits,
            dtype=self.config.dtype_mm,
            weight_dtype=self.config.weight_dtype,
            mesh=self.mesh,
            dropout_rate=0.0,
            name="test_attention",
            attention_type=common_types.AttentionType.FULL,
            is_nope_layer=True,
            use_bias_in_projections=True,
            use_qk_norm=False,
            query_pre_attn_scalar=1 / math.sqrt(self.head_dim),
            model_mode=common_types.MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(42),
        )

        copy_attention_weights_to_maxtext(torch_model, jax_attn)
        torch_output = torch_model(torch_hidden_states, cu_seqlens=cu_seqlens)

        jax_output = jax_attn(
            inputs_q=jax_hidden_states, inputs_kv=jax_hidden_states, deterministic=True
        )

        assert_all_close_jax_torch(
            jax_output[0],  # Remove batch dimension
            torch_output,
            rtol=1e-5,
            atol=5e-3,
            error_msg="Attention outputs differ",
        )


class TestAudioEncoderLayer(unittest.TestCase):
    """Test MaxText AudioEncoderLayer against PyTorch implementation."""

    def setUp(self):
        self.config = jax_audio_config
        self.torch_config = torch_audio_encoder_config
        np.random.seed(42)
        torch.manual_seed(42)

        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def _test_encoder_layer_with_batch_size(self, batch_size):
        """Helper function to test encoder layer with a given batch size."""

        torch_layer = TorchQwen3OmniMoeAudioEncoderLayer(self.torch_config)
        torch_layer.eval()

        maxtext_layer = AudioEncoderLayer(
            config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0)
        )

        # Copy weights from PyTorch to MaxText
        copy_maxtext_encoder_layer_weights(torch_layer, maxtext_layer)

        # Create test input
        seq_len = 12  # After conv layers
        hidden_size = self.config.d_model_for_audio

        jax_input, torch_input_3d = create_random_jax_torch(
            batch_size, seq_len, hidden_size
        )

        # PyTorch forward pass - expects 2D input (total_seq_len, hidden_dim) with cu_seqlens
        torch_input_2d = torch_input_3d.reshape(-1, hidden_size)

        # Create cu_seqlens for PyTorch (cumulative sequence lengths for each batch)
        # For batch_size=2, seq_len=12: [0, 12, 24] indicates two sequences of length 12 each
        cu_seqlens = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32
        )

        attention_mask = create_block_diagonal_attention_mask(
            cu_seqlens, torch_input_2d.dtype
        )

        torch_output_1d = torch_layer(
            torch_input_2d, cu_seqlens=cu_seqlens, attention_mask=attention_mask
        )[0]
        torch_output = torch_output_1d.reshape(batch_size, seq_len, hidden_size)

        jax_output = maxtext_layer(jax_input, deterministic=True)

        assert_all_close_jax_torch(
            jax_output,
            torch_output,
            rtol=1e-5,
            atol=5e-3,
            error_msg="AudioEncoderLayer outputs differ",
        )

    def test_encoder_layer_matches_torch_batch_1(self):
        """Test that MaxText AudioEncoderLayer matches PyTorch with batch_size=1."""
        self._test_encoder_layer_with_batch_size(batch_size=1)

    def test_encoder_layer_matches_torch_batch_2(self):
        """Test that MaxText AudioEncoderLayer matches PyTorch with batch_size=2."""
        self._test_encoder_layer_with_batch_size(batch_size=2)

    def test_encoder_layer_is_jittable(self):
        """Test that encoder layer can be JIT compiled."""
        with self.mesh:
            jax_layer = AudioEncoderLayer(
                config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0)
            )

        @nnx.jit
        def forward(layer, x):
            return layer(x, deterministic=True)

        batch_size = 2
        seq_len = 12
        hidden_size = self.config.d_model_for_audio

        hidden_states = jnp.ones((batch_size, seq_len, hidden_size))
        output = forward(jax_layer, hidden_states)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))


class TestSinusoidsPositionEmbedding(unittest.TestCase):
    def setUp(self):
        self.length = 100
        self.channels = 512
        self.max_timescale = 10000.0
        np.random.seed(42)
        torch.manual_seed(42)

    def test_positional_embedding_matches_torch(self):
        torch_model = TorchSinusoidsPositionEmbedding(
            self.length, self.channels, self.max_timescale
        )
        jax_model = SinusoidsPositionEmbedding(
            self.length, self.channels, self.max_timescale
        )

        # Test full sequence
        torch_output = torch_model(self.length)
        jax_output = jax_model(self.length)

        assert_all_close_jax_torch(
            jax_output,
            torch_output,
            rtol=1e-5,
            atol=3e-4,
            error_msg="Positional embedding outputs differ",
        )

    def test_positional_embedding_is_jittable(self):
        model = SinusoidsPositionEmbedding(
            self.length, self.channels, self.max_timescale
        )

        # Note: seqlen must be static for JIT compilation
        @nnx.jit(static_argnames=["seqlen"])
        def forward(model, seqlen):
            return model(seqlen)

        output = forward(model, seqlen=self.length)
        self.assertEqual(output.shape, (self.length, self.channels))


class TestChunkComputation(unittest.TestCase):
    def test_compute_chunk_lengths_matches_torch(self):
        feature_lens_np = np.array([3000, 1500, 4000], dtype=np.int64)
        n_window = 50

        torch_feature_lens = torch.from_numpy(feature_lens_np)
        chunk_window_size = n_window * 2

        torch_chunk_num = torch.ceil(
            torch_feature_lens.float() / chunk_window_size
        ).long()
        torch_chunk_lengths = torch.tensor(
            [chunk_window_size] * torch_chunk_num.sum(), dtype=torch.long
        )
        tail_chunk_index = torch.nn.functional.pad(
            torch_chunk_num, (1, 0), value=-1
        ).cumsum(0)[1:]
        torch_chunk_lengths[tail_chunk_index] = (
            torch_feature_lens % chunk_window_size
        ).long()
        torch_chunk_lengths[torch_chunk_lengths == 0] = chunk_window_size

        jax_chunk_lengths, jax_chunk_num = compute_chunk_lengths(
            jnp.array(feature_lens_np), n_window
        )

        np.testing.assert_array_equal(jax_chunk_num, torch_chunk_num.numpy())
        np.testing.assert_array_equal(jax_chunk_lengths, torch_chunk_lengths.numpy())


class TestPrepareAudioChunks(unittest.TestCase):
    def test_prepare_audio_chunks_shape(self):
        input_features = jnp.ones((2, 128, 300), dtype=jnp.float32)
        feature_lens = jnp.array([300, 250])
        n_window = 50

        chunk_lengths, chunk_num = compute_chunk_lengths(feature_lens, n_window)
        padded_feature, padded_mask_after_cnn = prepare_audio_chunks(
            input_features, feature_lens, chunk_lengths, chunk_num, n_window
        )

        total_chunks = int(jnp.sum(chunk_num))
        max_chunk_len = int(jnp.max(chunk_lengths))

        self.assertEqual(padded_feature.shape[0], total_chunks)
        self.assertEqual(padded_feature.shape[1], 128)
        self.assertEqual(padded_feature.shape[2], max_chunk_len)

        expected_len_after_cnn = audio_encoder_get_feat_extract_output_lengths(
            chunk_lengths
        )
        max_len_after_cnn = int(jnp.max(expected_len_after_cnn))
        self.assertEqual(padded_mask_after_cnn.shape, (total_chunks, max_len_after_cnn))

    def test_prepare_audio_chunks_mask_validity(self):
        input_features = jnp.ones((1, 128, 100), dtype=jnp.float32)
        feature_lens = jnp.array([100])
        n_window = 50

        chunk_lengths, chunk_num = compute_chunk_lengths(feature_lens, n_window)
        padded_feature, padded_mask_after_cnn = prepare_audio_chunks(
            input_features, feature_lens, chunk_lengths, chunk_num, n_window
        )

        self.assertTrue(jnp.all(padded_mask_after_cnn.sum(axis=1) > 0))


class TestAudioEncoder(unittest.TestCase):
    """Test full AudioEncoder (stack of AudioEncoderLayers) against PyTorch implementation."""

    def setUp(self):
        self.config = jax_audio_config
        self.torch_config = torch_audio_encoder_config
        np.random.seed(42)
        torch.manual_seed(42)

        # Create a dummy mesh for MaxText
        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def test_audio_encoder_matches_torch(self):
        """Test that MaxText AudioEncoder matches PyTorch full encoder."""
        torch_encoder = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
        torch_encoder.eval()

        maxtext_encoder = AudioEncoder(
            config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0)
        )

        copy_maxtext_encoder_weights(torch_encoder, maxtext_encoder)

        batch_size = 1
        seq_len = 12
        hidden_size = self.config.d_model_for_audio

        jax_input, torch_input_3d = create_random_jax_torch(
            batch_size, seq_len, hidden_size
        )

        # PyTorch forward pass - 2D input with cu_seqlens
        torch_input = torch_input_3d[0]  # (seq_len, hidden_size)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        attention_mask = create_block_diagonal_attention_mask(
            cu_seqlens, torch_input.dtype
        )

        # Run through encoder layers
        hidden_state = torch_input
        for layer in torch_encoder.layers:
            hidden_state = layer(
                hidden_state, cu_seqlens=cu_seqlens, attention_mask=attention_mask
            )[0]
        torch_output = hidden_state

        # MaxText forward pass - 3D batched input
        jax_output = maxtext_encoder(jax_input, deterministic=True)

        assert_all_close_jax_torch(
            jax_output[0],  # Remove batch dimension
            torch_output,
            rtol=1e-5,
            atol=1e-2,
            error_msg="AudioEncoder outputs differ",
        )


class TestAudioModel(unittest.TestCase):
    """Test full AudioModel end-to-end against PyTorch implementation."""

    def setUp(self):
        self.config = jax_audio_config
        self.torch_config = torch_audio_encoder_config
        np.random.seed(42)
        torch.manual_seed(42)

        # Create a dummy mesh for MaxText
        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def test_audio_model_end_to_end(self):
        """Test full AudioModel pipeline against PyTorch including conv + encoder + projections."""

        # Create PyTorch model
        torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
        torch_model.eval()

        maxtext_model = AudioModel(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))

        copy_audio_model(torch_model, maxtext_model, self.config)

        batch_size = 1
        num_mel_bins = self.config.num_mel_bins_for_audio
        audio_length = 300  # Should result in multiple chunks

        jax_audio_features, torch_audio_features_3d = create_random_jax_torch(
            batch_size, num_mel_bins, audio_length
        )
        audio_lengths_np = np.array([audio_length], dtype=np.int64)

        # PyTorch forward pass - expects 2D input (num_mel_bins, audio_length) for single sample
        torch_audio_features = torch_audio_features_3d[0]  # Remove batch dimension
        torch_audio_lengths = torch.from_numpy(audio_lengths_np)

        torch_output = torch_model(
            input_features=torch_audio_features, feature_lens=torch_audio_lengths
        )
        torch_output_tensor = torch_output.last_hidden_state

        jax_audio_lengths = jnp.array(audio_lengths_np)

        jax_output = maxtext_model(
            audio_features=jax_audio_features,
            audio_lengths=jax_audio_lengths,
            deterministic=True,
        )

        chunk_lengths_jax, _ = compute_chunk_lengths(
            jax_audio_lengths, self.config.n_window_for_audio
        )

        padded_mask_after_cnn_jax = (
            jnp.arange(jax_output.shape[1])[None, :]
            < audio_encoder_get_feat_extract_output_lengths(
                chunk_lengths_jax, self.config.n_window_for_audio
            )[:, None]
        )

        # Extract valid positions from MaxText output (matching PyTorch's behavior)
        jax_output_flat = jax_output[padded_mask_after_cnn_jax]

        assert_all_close_jax_torch(
            jax_output_flat,
            torch_output_tensor,
            rtol=1e-5,
            atol=5e-2,
            error_msg="AudioModel outputs differ",
        )


class TestHelperFunctions(unittest.TestCase):
    def test_audio_encoder_get_feat_extract_output_lengths_is_jittable(self):
        """Test that audio_encoder_get_feat_extract_output_lengths is JIT-compilable."""

        @jax.jit
        def jitted_fn(lengths):
            return audio_encoder_get_feat_extract_output_lengths(lengths)

        input_lengths = jnp.array([3000, 1500])
        # Should not raise any JIT compilation errors
        output_lengths = jitted_fn(input_lengths)

        # Verify output shape is correct
        self.assertEqual(output_lengths.shape, input_lengths.shape)

    def test_prepare_audio_chunks_is_jittable(self):
        """Test that prepare_audio_chunks is JIT-compilable."""
        n_window = 50

        @jax.jit
        def jitted_fn(input_features, feature_lengths, chunk_lengths, chunk_num):
            return prepare_audio_chunks(
                input_features, feature_lengths, chunk_lengths, chunk_num, n_window
            )

        batch_size = 2
        num_mel_bins = 128
        max_len = 300

        input_features = jnp.ones((batch_size, num_mel_bins, max_len))
        feature_lengths = jnp.array([300, 250])
        chunk_lengths = jnp.array([100, 100, 100, 100, 50])
        chunk_num = jnp.array([3, 2])

        padded_feature, padded_mask_after_cnn = jitted_fn(
            input_features, feature_lengths, chunk_lengths, chunk_num
        )


if __name__ == "__main__":
    print_configs()
    unittest.main()
