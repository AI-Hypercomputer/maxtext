"""DeepSeek-V4 Attention Compressed Components.

For architectural details, refer to the official DeepSeek-V4 paper:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
"""

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import normalizations


class Compressor(nnx.Module):
  """Token compressor for DeepSeek-V4 HCA and CSA layers."""

  def __init__(
      self,
      config,
      compress_ratio: int,
      rngs: nnx.Rngs,
  ):
    """Initialize the projection layers and normalization for the compressor."""
    super().__init__()
    self.config = config
    self.compress_ratio = compress_ratio
    self.overlap = compress_ratio == 4

    # Project to 2 * compressed_dim for CSA to allow splitting Ca/Cb, and compressed_dim for HCA
    proj_dim = 2 * config.compressed_dim if self.overlap else config.compressed_dim

    # W^{KV}: Projection layer mapping input to the Key/Value representation space
    self.wkv = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=proj_dim,
        axis=-1,
        kernel_axes=("embed", "kv"),
        use_bias=False,
        dtype=config.dtype,
        rngs=rngs,
    )

    # W^Z: Projection layer for the latent compression representation
    self.wgate = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=proj_dim,
        axis=-1,
        kernel_axes=("embed", "kv"),
        use_bias=False,
        dtype=config.dtype,
        rngs=rngs,
    )

    # RMSNorm for normalizing the compressed representations
    self.norm = normalizations.RMSNorm(
        num_features=config.compressed_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        rngs=rngs,
    )

    # Absolute Positional Embedding (APE)
    # self.ape acts as a learnable, static positional bias added to the gate logits before the
    # Softmax (B_p in Equation 11/22 in the paper: alpha_j = softmax_j(Z_j + B_p)).
    # This localized bias allows the model to learn a static positional priority
    # (e.g., prioritizing the most recent tokens) to preserve relative temporal
    # ordering inside the pooling window, regardless of dynamic token content.
    # Combined window size is 2 * compress_ratio for CSA, and compress_ratio for HCA.
    ape_ratio = 2 * compress_ratio if self.overlap else compress_ratio
    self.ape = nnx.Param(initializers.default_bias_init(rngs.params(), (ape_ratio, config.compressed_dim), config.dtype))

  def __call__(self, x, return_intermediates: bool = False):
    """
    Forward pass for the token compressor.

    Args:
        x: Input sequence tensor.
           Shape: [Batch, SeqLen, Dim]

    Returns:
        compressed_x: The compressed representation.
                      Shape for HCA: [Batch, SeqLen // compress_ratio, CompressedDim]
                      Shape for CSA: [Batch, SeqLen // compress_ratio, CompressedDim]
    """
    if self.overlap:
      # ---------------------------------------------------------------------
      # Overlapping CSA compression logic (m = 4)
      # ---------------------------------------------------------------------

      # 1. Initial Input Shape & Sequence Truncation
      # x shape: [Batch, SeqLen, Dim]
      # Safely truncate the sequence length to the nearest multiple of compress_ratio (R)
      # to ensure shape-divisibility during chunk-wise reshape operations and prevent Tracer crashes
      B, S, _ = x.shape
      R = self.compress_ratio  # R = 4 for CSA
      usable_S = (S // R) * R
      x_usable = x[:, :usable_S, :]
      C = usable_S // R

      # 2. Linear Projections
      # Apply W^{KV} and W^Z projections mapping to 2 * compressed_dim
      # Shape: [Batch, SeqLen, Dim] -> [Batch, SeqLen, 2 * compressed_dim]
      x_kv = self.wkv(x_usable)
      x_gate = self.wgate(x_usable)

      # 3. Splitting along Features (Axis -1)
      # Split the feature dimension in half to separate Ca and Cb components.
      # Shape of each: [Batch, SeqLen, compressed_dim]
      kv_Ca, kv_Cb = jnp.split(x_kv, 2, axis=-1)
      gate_Ca, gate_Cb = jnp.split(x_gate, 2, axis=-1)

      # 4. Reshape to Chunks
      # Reshape into blocks of size R
      # Shape of each: [Batch, Chunks, Ratio, compressed_dim]
      kv_Ca_chunked = jnp.reshape(kv_Ca, (B, C, R, -1))
      kv_Cb_chunked = jnp.reshape(kv_Cb, (B, C, R, -1))
      gate_Ca_chunked = jnp.reshape(gate_Ca, (B, C, R, -1))
      gate_Cb_chunked = jnp.reshape(gate_Cb, (B, C, R, -1))

      # 5. Prior Ca Shift (JAX-friendly padding)
      # Shift Ca chunks to the right by 1 to form the prior sequence.
      # Pad the first chunk's prior KV Ca with zeros.
      # Shape: [Batch, 1, Ratio, compressed_dim]
      zero_kv = jnp.zeros_like(kv_Ca_chunked[:, :1, :, :])
      # Shape: [Batch, Chunks, Ratio, compressed_dim]
      prior_kv_Ca = jnp.concatenate([zero_kv, kv_Ca_chunked[:, :-1, :, :]], axis=1)

      # Pad the first chunk's prior gate Ca with -1e4 so it gets softmax weight 0 safely in any dtype.
      # Shape: [Batch, 1, Ratio, compressed_dim]
      zero_gate = jnp.ones_like(gate_Ca_chunked[:, :1, :, :]) * -1e4
      # Shape: [Batch, Chunks, Ratio, compressed_dim]
      prior_gate_Ca = jnp.concatenate([zero_gate, gate_Ca_chunked[:, :-1, :, :]], axis=1)

      # 6. Combine
      # Concatenate prior Ca and current Cb along the window axis (axis=2)
      # Shape: [Batch, Chunks, 2 * Ratio, compressed_dim]
      kv_overlapped = jnp.concatenate([prior_kv_Ca, kv_Cb_chunked], axis=2)
      gate_overlapped = jnp.concatenate([prior_gate_Ca, gate_Cb_chunked], axis=2)

      # 7. Softmax Gating
      # Upcast to float32 before adding APE bias and applying softmax for absolute stability
      gate_overlapped_fp32 = gate_overlapped.astype(jnp.float32) + self.ape[...].astype(jnp.float32)
      weights = jax.nn.softmax(gate_overlapped_fp32, axis=2).astype(gate_overlapped.dtype)

      # 8. Weighted Reduction with FP32 Accumulation
      # Multiply weights by the overlapped KV and reduce in float32 precision across the '2 * Ratio'
      # dimension to prevent severe BF16 precision loss or summation truncation
      # Shape before sum: [Batch, Chunks, 2 * Ratio, compressed_dim]
      # Shape after sum: [Batch, Chunks, compressed_dim]
      compressed_x_prenorm = jnp.sum((weights * kv_overlapped).astype(jnp.float32), axis=2).astype(gate_overlapped.dtype)

      # 9. RMSNorm
      # Apply normalization over the feature dimension.
      # Shape: [Batch, Chunks, compressed_dim] -> [Batch, Chunks, compressed_dim]
      compressed_x = self.norm(compressed_x_prenorm)
      if return_intermediates:
        return compressed_x, {"kv": x_kv, "gate": x_gate, "weights": weights, "prenorm": compressed_x_prenorm}
      return compressed_x

    # ---------------------------------------------------------------------
    # Non-overlapping HCA compression logic (m' = 128)
    # ---------------------------------------------------------------------

    # 1. Initial Input Shape & Sequence Truncation
    # x shape: [Batch, SeqLen, Dim]
    # Safely truncate the sequence length to the nearest multiple of compress_ratio (R)
    # to ensure shape-divisibility during chunk-wise reshape operations and prevent Tracer crashes
    B, S, _ = x.shape
    R = self.compress_ratio
    usable_S = (S // R) * R
    x_usable = x[:, :usable_S, :]
    C = usable_S // R

    # 2. Linear Projections
    # Apply W^{KV} projection mapping to Key/Value space
    # Shape: [Batch, SeqLen, Dim] -> [Batch, SeqLen, compressed_dim]
    x_kv = self.wkv(x_usable)

    # Apply W^Z projection for the latent compression gate
    # Shape: [Batch, SeqLen, Dim] -> [Batch, SeqLen, compressed_dim]
    x_gate = self.wgate(x_usable)

    # 3. Reshaping into Blocks/Chunks
    # Reshape to isolate the compression ratio dimension
    # Shape: [Batch, SeqLen, compressed_dim] -> [Batch, Chunks, Ratio, compressed_dim]
    x_kv_chunked = jnp.reshape(x_kv, (B, C, R, -1))
    x_gate_chunked = jnp.reshape(x_gate, (B, C, R, -1))

    # 4. Softmax Gating
    # Upcast to float32 before adding APE bias and applying softmax for absolute stability
    x_gate_chunked_fp32 = x_gate_chunked.astype(jnp.float32) + self.ape[...].astype(jnp.float32)
    weights = jax.nn.softmax(x_gate_chunked_fp32, axis=2).astype(x_gate_chunked.dtype)

    # 5. Reduction (Weighted Sum) with FP32 Accumulation
    # Multiply the weights by the KV representation and sum in float32 precision across the 'Ratio'
    # dimension to prevent severe BF16 precision loss or summation truncation
    # Shape before sum: [Batch, Chunks, Ratio, compressed_dim]
    # Shape after sum: [Batch, Chunks, compressed_dim]
    compressed_x_prenorm = jnp.sum((weights * x_kv_chunked).astype(jnp.float32), axis=2).astype(x_gate_chunked.dtype)

    # 6. Normalization
    # Apply RMSNorm over the last dimension (compressed_dim)
    # Shape: [Batch, Chunks, compressed_dim] -> [Batch, Chunks, compressed_dim]
    compressed_x = self.norm(compressed_x_prenorm)

    if return_intermediates:
      return compressed_x, {"kv": x_kv, "gate": x_gate, "weights": weights, "prenorm": compressed_x_prenorm}
    return compressed_x
