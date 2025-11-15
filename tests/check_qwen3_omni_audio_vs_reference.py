# Copyright 2025 Google LLC
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

import unittest
import os
import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx
from jax.sharding import Mesh

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention as TorchQwen3OmniMoeAudioAttention,
    Qwen3OmniMoeAudioEncoderLayer as TorchQwen3OmniMoeAudioEncoderLayer,
    SinusoidsPositionEmbedding as TorchSinusoidsPositionEmbedding,
    Qwen3OmniMoeAudioEncoder as TorchQwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

from MaxText.layers.qwen3 import (
    Qwen3OmniAudioEncoderLayer,
    Qwen3OmniAudioEncoder,
    Qwen3OmniAudioModel,
)
from MaxText.layers.embeddings import SinusoidsPositionEmbedding
from MaxText.layers.attentions import Attention
from MaxText import common_types
from MaxText import pyconfig

from multimodal_test_utils import (
    copy_attention_weights_to_maxtext,
    copy_audio_model,
    copy_maxtext_encoder_layer_weights,
    create_block_diagonal_attention_mask,
    create_random_jax_torch,
    assert_all_close_jax_torch,
)


base_config_path = os.path.join(os.path.dirname(__file__), "..", "src", "MaxText", "configs", "base.yml")
jax_audio_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen3-omni-30b-a3b",
    attention="dot_product",
    attention_type="full",
    dropout_rate=0.0,
    dtype="float32",
    weight_dtype="float32",
    float32_logits=True,
)

torch_audio_encoder_config = Qwen3OmniMoeAudioEncoderConfig(
    d_model=jax_audio_config.d_model_for_audio,
    encoder_attention_heads=jax_audio_config.encoder_attention_heads_for_audio,
    encoder_ffn_dim=jax_audio_config.encoder_ffn_dim_for_audio,
    encoder_layers=jax_audio_config.encoder_layers_for_audio,
    attention_dropout=jax_audio_config.attention_dropout_for_audio,
    dropout=0.0,
    activation_dropout=0.0,
    activation_function="gelu",
    num_mel_bins=jax_audio_config.num_mel_bins_for_audio,
    max_source_positions=jax_audio_config.max_source_positions_for_audio,
    scale_embedding=True,
    n_window=jax_audio_config.n_window_for_audio,
    n_window_infer=jax_audio_config.n_window_infer_for_audio,
    conv_chunksize=jax_audio_config.conv_chunksize_for_audio,
    downsample_hidden_size=jax_audio_config.downsample_hidden_size_for_audio,
    output_dim=jax_audio_config.output_dim_for_audio,
    torch_dtype=torch.float32,
    weight_dtype=torch.float32,
)
torch_audio_encoder_config._attn_implementation = "eager"

torch.set_grad_enabled(False)


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

        maxtext_layer = Qwen3OmniAudioEncoderLayer(
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
            jax_layer = Qwen3OmniAudioEncoderLayer(
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
            self.length, self.channels, self.max_timescale, cast_as_fprop_dtype=False
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

        @nnx.jit(static_argnames=["seqlen"])
        def forward(model, seqlen):
            return model(seqlen)

        output = forward(model, seqlen=self.length)
        self.assertEqual(output.shape, (self.length, self.channels))


class TestAudioEncoder(unittest.TestCase):
    """Test AudioEncoder (convs + transformer, no projector) against PyTorch implementation."""

    def setUp(self):
        self.config = jax_audio_config
        self.torch_config = torch_audio_encoder_config
        np.random.seed(42)
        torch.manual_seed(42)

        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def test_audio_encoder_matches_torch(self):
        """Test that MaxText AudioEncoder matches PyTorch encoder (convs + transformer + layernorm, before projector)."""
        torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
        torch_model.eval()

        maxtext_encoder = Qwen3OmniAudioEncoder(
            config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0)
        )

        from multimodal_test_utils import copy_maxtext_audio_encoder_weights
        copy_maxtext_audio_encoder_weights(torch_model, maxtext_encoder, self.config)

        batch_size = 1
        num_mel_bins = self.config.num_mel_bins_for_audio
        audio_length = 200  # n_window=50, chunk_size=100, gives 2 chunks

        jax_audio_features, torch_audio_features_3d = create_random_jax_torch(
            batch_size, num_mel_bins, audio_length
        )

        # PyTorch forward (manually run convs + transformer encoder without projector)
        torch_audio_features = torch_audio_features_3d[0]
        audio_lengths_np = np.array([audio_length], dtype=np.int64)
        torch_audio_lengths = torch.from_numpy(audio_lengths_np)

        # Run through PyTorch convs + positional + encoder
        chunk_size = self.torch_config.n_window * 2
        num_chunks = audio_length // chunk_size
        chunk_lengths = torch.tensor([chunk_size] * num_chunks, dtype=torch.long)
        chunk_list = torch_audio_features.T.split(chunk_lengths.tolist(), dim=0)
        torch_padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        torch_padded_feature = torch_padded_feature.unsqueeze(1)

        torch_conv1 = F.gelu(torch_model.conv2d1(torch_padded_feature))
        torch_conv2 = F.gelu(torch_model.conv2d2(torch_conv1))
        torch_conv3 = F.gelu(torch_model.conv2d3(torch_conv2))

        b, c, f, t = torch_conv3.size()
        torch_conv_out = torch_model.conv_out(torch_conv3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        torch_pos_emb = torch_model.positional_embedding.positional_embedding[:torch_conv_out.shape[1], :].unsqueeze(0).to(torch_conv_out.dtype)
        torch_after_pos = torch_conv_out + torch_pos_emb

        # Run through encoder layers + layernorm (but not projector)
        # Process all chunks together
        seq_len_per_chunk = torch_after_pos.shape[1]
        cu_seqlens = torch.tensor([i * seq_len_per_chunk for i in range(num_chunks + 1)], dtype=torch.int32)
        attention_mask = create_block_diagonal_attention_mask(cu_seqlens, torch_after_pos.dtype)

        # Flatten: (num_chunks, seq_len_per_chunk, hidden) -> (num_chunks*seq_len_per_chunk, hidden)
        hidden_state = torch_after_pos.reshape(-1, torch_after_pos.shape[-1])
        for layer in torch_model.layers:
            hidden_state = layer(hidden_state, cu_seqlens=cu_seqlens, attention_mask=attention_mask)[0]
        hidden_state = torch_model.ln_post(hidden_state)

        # Reshape back: (num_chunks*seq_len_per_chunk, hidden) -> (batch=1, num_chunks*seq_len_per_chunk, hidden)
        torch_output = hidden_state.reshape(1, num_chunks * seq_len_per_chunk, -1)

        # MaxText forward
        jax_output = maxtext_encoder(jax_audio_features, deterministic=True)

        assert_all_close_jax_torch(
            jax_output,
            torch_output,
            rtol=1e-3,
            atol=0.1,
            error_msg="AudioEncoder outputs differ",
        )


class TestAudioModel(unittest.TestCase):
    """Test full AudioModel end-to-end against PyTorch implementation."""

    def setUp(self):
        self.config = jax_audio_config
        self.torch_config = torch_audio_encoder_config
        np.random.seed(42)
        torch.manual_seed(42)

        devices = jax.devices()
        self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

    def test_audio_model_end_to_end(self):
        """Test full AudioModel pipeline against PyTorch."""
        torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
        torch_model.eval()

        maxtext_model = Qwen3OmniAudioModel(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))
        copy_audio_model(torch_model, maxtext_model, self.config)

        batch_size = 1
        num_mel_bins = self.config.num_mel_bins_for_audio
        audio_length = 200  # With n_window=50, chunk_size=100, gives 2 chunks

        jax_audio_features, torch_audio_features_3d = create_random_jax_torch(
            batch_size, num_mel_bins, audio_length
        )
        audio_lengths_np = np.array([audio_length], dtype=np.int64)

        torch_audio_features = torch_audio_features_3d[0]
        torch_audio_lengths = torch.from_numpy(audio_lengths_np)

        torch_output = torch_model(
            input_features=torch_audio_features, feature_lens=torch_audio_lengths
        )
        torch_output_tensor = torch_output.last_hidden_state

        jax_output = maxtext_model(
            audio_features=jax_audio_features,
            deterministic=True,
        )

        assert_all_close_jax_torch(
            jax_output[0],
            torch_output_tensor,
            rtol=1e-3,
            atol=0.02,
            error_msg="AudioModel outputs differ",
        )

    def test_audio_model_intermediates(self):
        """Debug intermediate outputs to verify each stage matches PyTorch."""
        torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
        torch_model.eval()

        maxtext_model = Qwen3OmniAudioModel(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))
        copy_audio_model(torch_model, maxtext_model, self.config)

        batch_size = 1
        num_mel_bins = self.config.num_mel_bins_for_audio
        audio_length = 100

        jax_audio_features, torch_audio_features_3d = create_random_jax_torch(
            batch_size, num_mel_bins, audio_length
        )
        torch_audio_features = torch_audio_features_3d[0]

        # PyTorch forward
        chunk_size = self.torch_config.n_window * 2
        num_chunks = audio_length // chunk_size
        chunk_lengths = torch.tensor([chunk_size] * num_chunks, dtype=torch.long)
        chunk_list = torch_audio_features.T.split(chunk_lengths.tolist(), dim=0)
        torch_padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        torch_padded_feature = torch_padded_feature.unsqueeze(1)

        torch_conv1 = F.gelu(torch_model.conv2d1(torch_padded_feature))
        torch_conv2 = F.gelu(torch_model.conv2d2(torch_conv1))
        torch_conv3 = F.gelu(torch_model.conv2d3(torch_conv2))

        b, c, f, t = torch_conv3.size()
        torch_conv_out = torch_model.conv_out(torch_conv3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        torch_pos_emb = torch_model.positional_embedding.positional_embedding[:torch_conv_out.shape[1], :].unsqueeze(0).to(torch_conv_out.dtype)
        torch_after_pos = torch_conv_out + torch_pos_emb

        # JAX forward
        jax_audio_chunks = jax_audio_features.reshape(batch_size, num_mel_bins, num_chunks, chunk_size)
        jax_audio_chunks = jax_audio_chunks.transpose(0, 2, 1, 3).reshape(batch_size * num_chunks, num_mel_bins, chunk_size)
        jax_hidden = jax_audio_chunks[:, :, :, jnp.newaxis]

        jax_conv1 = jax.nn.gelu(maxtext_model.conv2d1(jax_hidden))
        jax_conv2 = jax.nn.gelu(maxtext_model.conv2d2(jax_conv1))
        jax_conv3 = jax.nn.gelu(maxtext_model.conv2d3(jax_conv2))

        bc, f_jax, t_jax, c_jax = jax_conv3.shape
        jax_conv_out = maxtext_model.conv_out(jax_conv3.transpose(0, 2, 3, 1).reshape(bc, t_jax, c_jax * f_jax))

        seq_len_per_chunk = jax_conv_out.shape[1]
        jax_pos_emb = maxtext_model.positional_embedding(seq_len_per_chunk)
        jax_pos_emb = jnp.broadcast_to(jax_pos_emb[None, :, :], (batch_size * num_chunks, seq_len_per_chunk, self.config.d_model_for_audio))
        jax_after_pos = jax_conv_out + jax_pos_emb

        # Verify all stages match
        assert_all_close_jax_torch(jax_conv1[0], torch_conv1.permute(0, 2, 3, 1)[0], rtol=1e-4, atol=1e-3, error_msg="Conv1 differs")
        assert_all_close_jax_torch(jax_conv2[0], torch_conv2.permute(0, 2, 3, 1)[0], rtol=1e-4, atol=1e-3, error_msg="Conv2 differs")
        assert_all_close_jax_torch(jax_conv3[0], torch_conv3.permute(0, 2, 3, 1)[0], rtol=1e-4, atol=1e-3, error_msg="Conv3 differs")
        assert_all_close_jax_torch(jax_conv_out[0], torch_conv_out[0], rtol=1e-4, atol=1e-3, error_msg="Conv out differs")
        assert_all_close_jax_torch(jax_after_pos[0], torch_after_pos[0], rtol=1e-4, atol=1e-3, error_msg="After pos emb differs")

if __name__ == "__main__":
    unittest.main()
