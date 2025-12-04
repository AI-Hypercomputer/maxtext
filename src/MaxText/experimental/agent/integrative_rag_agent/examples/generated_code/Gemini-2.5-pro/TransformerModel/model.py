
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence

# Reused modules from MaxText
# src.MaxText.layers.embeddings.embed_as_linen
from MaxText.layers.embeddings import embed_as_linen
# src.MaxText.layers.attentions.attention_as_linen
from MaxText.layers.attentions import attention_as_linen
# src.MaxText.layers.linears.mlp_block
from MaxText.layers.linears import mlp_block
# src.MaxText.layers.normalizations.rms_norm
from MaxText.layers.normalizations import rms_norm
# src.MaxText.layers.linears.dense_general
from MaxText.layers.linears import dense_general

from MaxText.common_types import Array, Config, Mesh


class TransformerEncoderLayer(nn.Module):
  """A single Transformer Encoder layer, mirroring PyTorch's nn.TransformerEncoderLayer."""
  ninp: int
  nhead: int
  nhid: int
  dropout: float
  config: Config
  mesh: Mesh

  def setup(self):
    """Initializes the components of the Transformer encoder layer."""
    # Dummy shapes for initializing MaxText attention and MLP modules
    dummy_inputs_shape = (1, self.config.max_target_length, self.ninp)

    self.self_attn = attention_as_linen(
        config=self.config,
        num_query_heads=self.nhead,
        num_kv_heads=self.nhead,
        head_dim=self.ninp // self.nhead,
        max_target_length=self.config.max_target_length,
        mesh=self.mesh,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.dropout,
        name='self_attention')
    self.mlp = mlp_block(
        config=self.config,
        mesh=self.mesh,
        in_features=self.ninp,
        intermediate_dim=self.nhid,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.dropout,
        name='mlp')
    self.norm1 = rms_norm(
        num_features=self.ninp,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name='norm1')
    self.norm2 = rms_norm(
        num_features=self.ninp,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name='norm2')
    self.dropout1 = nn.Dropout(rate=self.dropout)
    self.dropout2 = nn.Dropout(rate=self.dropout)

  def __call__(self, src: Array, src_mask: Optional[Array] = None,
               deterministic: bool = True):
    """Applies the Transformer encoder layer to the input sequence."""
    # Self-attention block (pre-norm)
    norm_src = self.norm1(src)
    batch_size, seq_len, _ = src.shape
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    positions = jnp.broadcast_to(positions, (batch_size, seq_len))

    attn_output = self.self_attn(
        inputs_q=norm_src,
        inputs_kv=norm_src,
        inputs_positions=positions,
        bidirectional_mask=src_mask,
        deterministic=deterministic)
    src = src + self.dropout1(attn_output, deterministic=deterministic)

    # Feed-forward block (pre-norm)
    norm_src = self.norm2(src)
    mlp_output = self.mlp(norm_src, deterministic=deterministic)
    src = src + self.dropout2(mlp_output, deterministic=deterministic)
    return src


class TransformerModel(nn.Module):
  """A Transformer model implementation in Flax."""
  ntoken: int
  ninp: int
  nhead: int
  nhid: int
  nlayers: int
  dropout: float = 0.5
  mesh: Optional[Mesh] = None

  def setup(self):
    """Initializes the transformer model components."""
    self.model_type = 'Transformer'

    # Create a minimal config object for sub-modules
    class MockConfig:
      pass

    config = MockConfig()
    config.dtype = jnp.float32
    config.weight_dtype = jnp.float32
    config.max_target_length = 2048  # A reasonable default
    config.attention = 'dot_product'
    config.float32_qk_product = False
    config.float32_logits = False
    config.mlp_activations = ('relu',)
    config.mlp_bias = True
    config.use_bias_in_projections = True
    config.normalization_layer_epsilon = 1e-5
    config.dropout_rate = self.dropout
    config.fused_mlp = False
    config.quantization = None
    config.quantize_kvcache = False
    config.shard_mode = 'AUTO'
    config.use_pre_norm = False  # For mlp_block
    config.decoder_block = 'default'  # For mlp_block norm

    self.encoder = embed_as_linen(
        num_embeddings=self.ntoken,
        features=self.ninp,
        dtype=config.dtype,
        name='token_embedder')

    self.transformer_encoder = [
        TransformerEncoderLayer(
            ninp=self.ninp,
            nhead=self.nhead,
            nhid=self.nhid,
            dropout=self.dropout,
            config=config,
            mesh=self.mesh,
            name=f'encoder_layer_{i}') for i in range(self.nlayers)
    ]

    self.decoder = dense_general(
        features=self.ntoken,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        use_bias=True,
        name='output_projection')

  def __call__(self, src: Array, src_mask: Optional[Array] = None,
               deterministic: bool = True):
    """Forward pass of the transformer model."""
    output = self.encoder(src) * jnp.sqrt(self.ninp)

    for layer in self.transformer_encoder:
      output = layer(output, src_mask=src_mask, deterministic=deterministic)

    output = self.decoder(output)
    return output
