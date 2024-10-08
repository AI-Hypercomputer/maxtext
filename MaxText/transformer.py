"""A simple transformer model."""

import dataclasses
import functools
from typing import Any, Mapping, Sequence, cast
from absl import logging
import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class Einsum(nn.Module):
  """Flax implementation of the Einsum layer."""

  shape: Sequence[int]  # the shape on the right hand side
  einsum_str: str

  @nn.compact
  def __call__(self, inputs):
    w_init = jax.nn.initializers.truncated_normal()
    w = self.param('w', w_init, self.shape)
    out = jnp.einsum(self.einsum_str, inputs, w)
    return out


@dataclasses.dataclass(slots=True)
class TransformerConfig:
  """Transformer config."""

  # Vocab size
  vocab_size: int = 30000
  # Primary width, the number of channels on the carrier path.
  d_model: int = 128
  # Depth, or num transformer blocks. One 'layer' is attn + ffw.
  num_layers: int = 8
  # Number of heads for self-attention.
  num_heads: int = 16

  sequence_length: int = 32

  # Whether to use a jax.lax.scan style loop around the transformer layers
  use_layer_stack: bool = True

  def __post_init__(self):
    if self.d_model % self.num_heads != 0:
      raise ValueError('num_heads has to divide d_model exactly')

  def compute_static_step_metrics(
      self,
      data: Mapping[str, Any] | None = None,
  ) -> Mapping[str, chex.Array]:
    """Computes metrics that can be inferred without any runtime information.

    Args:
      data: one batch of inputs data or None.

    Returns:
      Metrics for one step or struct stub with zeros if data is None.
    """

    if data is None:
      logging.info('Initializing static step metrics.')
      return dict(
          sequences=np.array(0, dtype=np.int64),
          tokens=np.array(0, dtype=np.int64),
          flops=np.array(0, dtype=np.float64),
      )

    batch_size, seq_len, *_ = jax.tree_util.tree_leaves(data)[0].shape
    unmasked_tokens = batch_size * seq_len

    return dict(
        sequences=batch_size,
        tokens=unmasked_tokens,
        flops=6 * unmasked_tokens * self.count_params_wo_embedding(),
    )

  def count_params_wo_embedding(self) -> int:
    """Number of parameters in the model, ignoring embeddings."""
    d_model = self.d_model
    num_layers = self.num_layers

    attention_params = 4 * d_model * d_model
    ffw_params = 4 * d_model * d_model * 2

    return num_layers * (attention_params + ffw_params)


class FFWBlock(nn.Module):
  """A simple FFW block."""

  d_model: int

  def setup(self):
    self.ffw_up = Einsum(
        (self.d_model, 4 * self.d_model),
        'btd,df->btf',
    )
    self.ffw_down = Einsum(
        (self.d_model * 4, self.d_model),
        'btf,fd->btd',
    )

  def __call__(self, x):
    x = jax.nn.relu(self.ffw_up(x))
    result = self.ffw_down(x)
    return result


def add_positional_encoding(embeddings, max_timescale: int = 10_000):
  """Fixed sinusoidal position encoding."""
  _, sequence_length, d_model = embeddings.shape
  freqs = np.arange(0, d_model, 2)
  inv_freq = max_timescale ** (-freqs / d_model)
  pos_seq = np.arange(sequence_length, 0, -1.0)
  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings + pos_emb


class MHABlock(nn.Module):
  """A simple MHA block."""

  d_model: int
  num_heads: int

  def setup(self):
    self.qkv_size, r = divmod(self.d_model, self.num_heads)
    assert r == 0

    # Linear goes to d_model, then we split by num_heads
    self.mha_proj_q = Einsum(
        (self.d_model, self.d_model),
        'btf,fd->btd',
    )
    self.mha_proj_k = Einsum(
        (self.d_model, self.d_model),
        'btf,fd->btd',
    )
    self.mha_proj_v = Einsum(
        (self.d_model, self.d_model),
        'btf,fd->btd',
    )
    self.mha_final = Einsum(
        (self.d_model, self.d_model),
        'btf,fd->btd',
    )

  def __call__(self, x, mask=None):
    q = jnp.reshape(
        self.mha_proj_q(x), x.shape[:-1] + (self.num_heads, self.qkv_size)
    )
    k = jnp.reshape(
        self.mha_proj_k(x), x.shape[:-1] + (self.num_heads, self.qkv_size)
    )
    v = jnp.reshape(
        self.mha_proj_v(x), x.shape[:-1] + (self.num_heads, self.qkv_size)
    )

    attn_logits = jnp.einsum('...thd,...Thd->...htT', q, k)
    attn_logits = attn_logits / np.sqrt(self.qkv_size).astype(attn_logits.dtype)

    if mask is not None:
      attn_logits = jnp.where(mask[:, None, None, :], attn_logits, -1e30)

    attn_softmax = jax.nn.softmax(attn_logits)
    x = jnp.einsum('...htT,...Thd->...thd', attn_softmax, v)
    x = jnp.reshape(x, x.shape[:-2] + (self.d_model,))
    result = self.mha_final(x)
    return result


class Block(nn.Module):
  """A simple transformer block."""

  d_model: int
  num_heads: int

  def setup(self):
    self.mha = MHABlock(
        d_model=self.d_model,
        num_heads=self.num_heads,
    )
    self.ffw = FFWBlock(d_model=self.d_model)
    self.ln = nn.LayerNorm(reduction_axes=-1, use_scale=False, use_bias=False)

  def __call__(self, x, mask):
    y = self.ln(x)
    y = self.mha(y, mask=mask)
    y = self.ffw(y)
    return x + y, None


class Transformer(nn.Module):
  """A simple transformer Flax module."""

  cfg: TransformerConfig

  def setup(self):
    emb_init = jax.nn.initializers.truncated_normal()
    self.embeddings = cast(
        jax.Array,
        self.param(
            'embeddings',
            emb_init,
            (self.cfg.vocab_size, self.cfg.d_model),
        ),
    )

    if self.cfg.use_layer_stack:
      self.stacked_blocks = nn.scan(
          functools.partial(Block, name='block'),
          length=self.cfg.num_layers,
          variable_axes={'params': 0},
          in_axes=(nn.broadcast,),
          split_rngs={'params': True},
      )(self.cfg.d_model, self.cfg.num_heads)
    else:

      def make_block(i):
        return Block(
            d_model=self.cfg.d_model,
            num_heads=self.cfg.num_heads,
            name=f'block_{i}',
        )

      self.blocks = [make_block(i) for i in range(self.cfg.num_layers)]

  def __call__(self, tokens, input_mask):
    chex.assert_rank(tokens, 2)
    embedded_tokens = self.embeddings[(tokens,)]
    x = add_positional_encoding(embedded_tokens)
    if self.cfg.use_layer_stack:
      x, _ = self.stacked_blocks(x, input_mask)
    else:
      for block in self.blocks:
        x, _ = block(x, input_mask)

    return x @ self.embeddings.T


class FakeData:

  def __init__(self, seed: int, batch_size: int, cfg: TransformerConfig):
    self._key = jax.random.PRNGKey(seed)
    self._batch_size = batch_size
    self._cfg = cfg

  def __next__(self):
    def random_sequence():
      self._key, k = jax.random.split(self._key)
      return jax.random.randint(
          k,
          (self._batch_size, self._cfg.sequence_length),
          minval=0,
          maxval=self._cfg.vocab_size,
      )

    return {
        'observation': random_sequence(),
        'input_mask': jnp.ones(
            (self._batch_size, self._cfg.sequence_length), dtype=jnp.int32
        ),
        'target': random_sequence(),
    }


def loss_fn(model):
  def f(params, data):
    logits = model.apply(params, data['observation'], data['input_mask'])
    logits = logits.astype(jnp.float32)
    logits = jax.nn.log_softmax(logits)
    targets = jax.nn.one_hot(data['target'], logits.shape[-1])
    return -jnp.sum(targets * logits)

  return f


SMALL_TRANSFORMER_CONFIG = TransformerConfig(
    d_model=128, num_heads=16, num_layers=8, sequence_length=32
)
BIGGER_TRANSFORMER_CONFIG = TransformerConfig(
    d_model=2_048, num_heads=16, num_layers=4, sequence_length=32
)