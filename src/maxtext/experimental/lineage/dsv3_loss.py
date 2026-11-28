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

"""Loss functions of the DSv3 training objective.

* `cross_entropy_with_logits`: the T5X one-hot cross entropy with z-loss and a
  stable custom gradient, verbatim from `maxtext/utils/max_utils.py`.
* `cross_entropy_with_integer_labels`: the plain-autodiff cross entropy of the
  MTP depths (`multi_token_prediction.py`).
* `masked_token_loss`: the main-objective sums MaxText's `train.loss_fn`
  derives from the logits.
* MoE router auxiliaries: Megatron-LM's sigmoid seq_aux_loss, the Switch-style
  load balance loss and the loss-free routed-bias updates, computed from the
  Lineage `DSv3RouterAux`.
"""

import dataclasses
import functools
import math
import operator
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_types

T = TypeVar(
    "T",
    dsv3_types.ArrayType,
    dsv3_types.ShardingType,
    dsv3_types.AxisNameType,
)

# `train.py` EPS: denominators of the per-token averages.
EPS = 1e-8


# Cross entropy implementation is taken from original T5X codebase:
# https://github.com/google-research/t5x/blob/ace831eea1e2742b4299cd1a9af7e4f302038351/t5x/losses.py#L25-L101
@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.

  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxiliary z-loss loss term.

  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[Any, ...]]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (
      logits,
      targets,
      z_loss,
      exp_shifted,
      sum_exp,
      log_z,
  )


def _cross_entropy_with_logits_bwd(
    res: tuple[Any, ...], cotangents: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[jnp.ndarray, None, None]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = cotangents[0]  # Ignore z_loss component; it is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv

  return (
      jnp.asarray(g_logits, logits.dtype),
      None,  # we don't need gradients on targets
      None,  # we don't need gradients on z_loss
  )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def cross_entropy_with_integer_labels_and_z_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Integer-label equivalent of `cross_entropy_with_logits` (zero fwd one-hot)."""
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  target_logit = jnp.take_along_axis(logits, targets[..., None], axis=-1)
  loss = jnp.squeeze(logits_sum - target_logit, axis=-1)
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss = loss + total_z_loss
  return loss, total_z_loss


def _xent_int_z_fwd(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[Any, ...]]:
  """Forward pass avoiding `[B, T, V]` float32 one-hot allocation."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_sum_exp = jnp.log(sum_exp)
  target_shifted = jnp.take_along_axis(shifted, targets[..., None], axis=-1)
  loss = jnp.squeeze(log_sum_exp - target_shifted, axis=-1)
  log_z = jnp.squeeze(log_sum_exp + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss = loss + total_z_loss
  logits_meta = jnp.empty((0, logits.shape[-1]), dtype=logits.dtype)
  return (loss, total_z_loss), (logits_meta, targets, exp_shifted, sum_exp, log_z)


def _xent_int_z_bwd(
    z_loss: float,
    res: tuple[Any, ...],
    cotangents: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, None]:
  """Stabilized custom VJP matching `cross_entropy_with_logits`."""
  g = cotangents[0]
  logits_meta, targets, exp_shifted, sum_exp, log_z = res
  one_hot = jax.nn.one_hot(targets, logits_meta.shape[1], dtype=exp_shifted.dtype)
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - one_hot
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  return jnp.asarray(g_logits, logits_meta.dtype), None


cross_entropy_with_integer_labels_and_z_loss.defvjp(_xent_int_z_fwd, _xent_int_z_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def cross_entropy_with_integer_labels(
    logits: jt.Float[jax.Array, "*batch V"],
    labels: jt.Int[jax.Array, "*batch"],
) -> jt.Float[jax.Array, "*batch"]:
  """Per-token cross entropy against integer labels (plain autodiff).

  Mirrors `multi_token_prediction._cross_entropy_with_integer_labels`.

  Args:
    logits: Logits `[*batch, vocab]`.
    labels: Integer labels `[*batch]`.

  Returns:
    The per-token losses `[*batch]`.
  """
  lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  target = jnp.take_along_axis(logits, labels[..., None], axis=-1)
  return jnp.squeeze(lse - target, axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3TokenLossSums(Generic[T]):
  """Sums of the main objective over the valid tokens of a batch.

  Attributes:
    xent: Sum of the per-token cross entropy (including the z-loss term).
    z_loss: Sum of the per-token z-loss.
    weights: Number of valid (non-padding) target tokens.
  """

  xent: T | None = None
  z_loss: T | None = None
  weights: T | None = None


def _constrain(x: jax.Array, sharding: jax.sharding.NamedSharding) -> jax.Array:
  """Places `x` on `sharding`: `reshard` under Explicit axes, else a constraint."""
  if sharding.mesh.abstract_mesh.are_all_axes_explicit:
    return jax.reshard(x, sharding)
  return jax.lax.with_sharding_constraint(x, sharding)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def masked_token_loss(
    logits: jt.Float[jax.Array, "B T V"],
    targets: jt.Int[jax.Array, "B T"],
    target_segmentation: jt.Int[jax.Array, "B T"],
    *,
    z_loss_multiplier: float = 0.0,
    token_sharding: jax.sharding.NamedSharding | None = None,
) -> DSv3TokenLossSums[jax.Array]:
  """Computes the main-objective sums of MaxText's `train.loss_fn`.

  Args:
    logits: Logits `[batch, seq, vocab]`.
    targets: Target token ids `[batch, seq]`.
    target_segmentation: Target segmentation `[batch, seq]`; zero marks padding.
    z_loss_multiplier: z-loss coefficient.
    token_sharding: Optional sharding constraint of the per-token losses.

  Returns:
    The summed cross entropy, z-loss and valid-token count.
  """
  one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
  xent, z_loss = cross_entropy_with_logits(logits, one_hot_targets, z_loss=z_loss_multiplier)
  if token_sharding is not None:
    xent = _constrain(xent, token_sharding)
    z_loss = _constrain(z_loss, token_sharding)
  valid = target_segmentation != 0
  xent = xent * valid
  z_loss = z_loss * valid
  return DSv3TokenLossSums[jax.Array](
      xent=jnp.sum(xent),
      z_loss=jnp.sum(z_loss),
      weights=jnp.sum(valid),
  )


def _abstract_mesh(mesh: Any) -> jax.sharding.AbstractMesh | None:
  if isinstance(mesh, jax.sharding.AbstractMesh):
    return mesh
  if isinstance(mesh, jax.sharding.Mesh):
    return mesh.abstract_mesh
  return None


def _axis_size(amesh: jax.sharding.AbstractMesh | None, ax: Any) -> int:
  if amesh is None or ax is None:
    return 1
  if isinstance(ax, str):
    return int(amesh.shape[ax])
  return math.prod(int(amesh.shape[a]) for a in ax)


def _split_token_chunks(
    normed: jax.Array,
    targets: jax.Array,
    target_mask: jax.Array,
    loss_chunk_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
  """Splits `(B, T, D)` and `(B, T)` arrays locally along `T` into `num_chunks`."""
  bsz, seq_len, emb_dim = normed.shape
  sharding = jax.typeof(normed).sharding
  mesh = getattr(sharding, "mesh", None)
  amesh = _abstract_mesh(mesh)
  spec = tuple(getattr(sharding, "spec", ()))
  tok0 = spec[0] if len(spec) > 0 else None
  tok1 = spec[1] if len(spec) > 1 else None
  b_local = bsz // _axis_size(amesh, tok0)
  t_local = seq_len // _axis_size(amesh, tok1)
  num_chunks = t_local
  if loss_chunk_tokens > 0:
    for d in range(1, t_local + 1):
      if t_local % d == 0 and b_local * (t_local // d) <= loss_chunk_tokens:
        num_chunks = d
        break
  if num_chunks <= 1:
    return normed[None], targets[None], target_mask[None], 1

  chunk_t = seq_len // num_chunks
  pspec = jax.sharding.PartitionSpec

  def _reshape_move(arr: jax.Array, new_shape: tuple[int, ...], p: jax.sharding.PartitionSpec) -> jax.Array:
    if amesh is not None and not amesh.empty:
      ns = jax.sharding.NamedSharding(mesh, p)
      if amesh.are_all_axes_explicit:
        reshaped = jax.lax.reshape(arr, new_shape, out_sharding=ns)
      else:
        reshaped = jax.lax.with_sharding_constraint(jnp.reshape(arr, new_shape), ns)
    else:
      reshaped = jnp.reshape(arr, new_shape)
    return jnp.moveaxis(reshaped, 2, 0)

  x_chunks = _reshape_move(normed, (bsz, chunk_t, num_chunks, emb_dim), pspec(tok0, tok1, None, None))
  t_chunks = _reshape_move(targets, (bsz, chunk_t, num_chunks), pspec(tok0, tok1, None))
  m_chunks = _reshape_move(target_mask, (bsz, chunk_t, num_chunks), pspec(tok0, tok1, None))
  return x_chunks, t_chunks, m_chunks, num_chunks


@jax.named_call
def chunked_masked_token_loss(
    normed: jt.Float[jax.Array, "B T D"],
    targets: jt.Int[jax.Array, "B T"],
    target_segmentation: jt.Int[jax.Array, "B T"],
    *,
    head_fn: Callable[[jax.Array], jax.Array],
    loss_chunk_tokens: int,
    z_loss_multiplier: float = 0.0,
    token_sharding: jax.sharding.NamedSharding | None = None,
) -> DSv3TokenLossSums[jax.Array]:
  """Computes LM head + `masked_token_loss` via `@jax.checkpoint` token chunks."""
  x_chunks, t_chunks, m_chunks, num_chunks = _split_token_chunks(normed, targets, target_segmentation, loss_chunk_tokens)

  @jax.checkpoint
  def _chunk_fn(carry, xs):
    del carry
    x_c, t_c, m_c = xs
    logits_c = head_fn(x_c)
    xent_c, z_c = cross_entropy_with_integer_labels_and_z_loss(logits_c, t_c, z_loss_multiplier)
    if token_sharding is not None:
      xent_c = _constrain(xent_c, token_sharding)
      z_c = _constrain(z_c, token_sharding)
    valid_c = m_c != 0
    return None, (
        jnp.sum(xent_c * valid_c),
        jnp.sum(z_c * valid_c),
        jnp.sum(valid_c),
    )

  if num_chunks == 1:
    _, (xent_s, z_s, w_s) = _chunk_fn(None, (x_chunks[0], t_chunks[0], m_chunks[0]))
    return DSv3TokenLossSums[jax.Array](xent=xent_s, z_loss=z_s, weights=w_s)

  _, (xent_ys, z_ys, w_ys) = jax.lax.scan(_chunk_fn, None, (x_chunks, t_chunks, m_chunks))
  return DSv3TokenLossSums[jax.Array](
      xent=jnp.sum(xent_ys),
      z_loss=jnp.sum(z_ys),
      weights=jnp.sum(w_ys),
  )


@jax.named_call
def chunked_integer_token_loss(
    normed: jt.Float[jax.Array, "B T D"],
    targets: jt.Int[jax.Array, "B T"],
    target_mask: jax.Array,
    *,
    head_fn: Callable[[jax.Array], jax.Array],
    loss_chunk_tokens: int,
) -> tuple[jax.Array, jax.Array]:
  """Computes MTP LM head + integer cross-entropy sum via `@jax.checkpoint` chunks."""
  x_chunks, t_chunks, m_chunks, num_chunks = _split_token_chunks(normed, targets, target_mask, loss_chunk_tokens)

  @jax.checkpoint
  def _chunk_fn(carry, xs):
    del carry
    x_c, t_c, m_c = xs
    logits_c = head_fn(x_c)
    xent_c = cross_entropy_with_integer_labels(logits_c, t_c)
    valid_c = m_c != 0
    return None, (jnp.sum(xent_c * valid_c), jnp.sum(valid_c).astype(jnp.float32))

  if num_chunks == 1:
    _, (loss_s, w_s) = _chunk_fn(None, (x_chunks[0], t_chunks[0], m_chunks[0]))
    return loss_s, w_s

  _, (loss_ys, w_ys) = jax.lax.scan(_chunk_fn, None, (x_chunks, t_chunks, m_chunks))
  return jnp.sum(loss_ys), jnp.sum(w_ys)


def _num_flat_chunks(n_local: int, loss_chunk_tokens: int) -> int:
  if loss_chunk_tokens <= 0 or n_local <= loss_chunk_tokens:
    return 1
  for d in range(1, n_local + 1):
    if n_local % d == 0 and (n_local // d) <= loss_chunk_tokens:
      return d
  return n_local


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def chunked_sharded_head_xent(
    normed: jt.Float[jax.Array, "B T D"],
    sharded_kernel: jt.Float[jax.Array, "D V"],
    targets: jt.Int[jax.Array, "B T"],
    target_mask: jax.Array,
    loss_chunk_tokens: int,
    z_loss_multiplier: float,
    dot_dtype: jax.typing.DTypeLike,
    cast_logits_to_fp32: bool,
    precision: str,
    token_sharding: jax.sharding.NamedSharding | None,
    with_z_loss: bool,
    weights_in_fp32: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes LM head + token cross-entropy with only sharded head saved in `res`."""
  sums, _ = _chunked_sharded_head_xent_fwd(
      normed,
      sharded_kernel,
      targets,
      target_mask,
      loss_chunk_tokens,
      z_loss_multiplier,
      dot_dtype,
      cast_logits_to_fp32,
      precision,
      token_sharding,
      with_z_loss,
      weights_in_fp32,
  )
  return sums


def _axis_tuple(spec_entry: Any) -> tuple[str, ...]:
  if spec_entry is None:
    return ()
  if isinstance(spec_entry, str):
    return (spec_entry,)
  return tuple(spec_entry)


def _sharded_head_axes(
    normed: jax.Array,
    sharded_kernel: jax.Array,
    token_sharding: jax.sharding.NamedSharding | None,
):
  k_sharding = jax.typeof(sharded_kernel).sharding
  mesh = getattr(k_sharding, "mesh", None)
  amesh = _abstract_mesh(mesh)
  k_spec = tuple(getattr(k_sharding, "spec", ()))
  if len(k_spec) > 1 and k_spec[1] is not None:
    raise ValueError(f"vocab dim of sharded_kernel must be replicated: {k_spec}")
  emb_axes = _axis_tuple(k_spec[0] if k_spec else None)
  if token_sharding is not None:
    t_spec = tuple(token_sharding.spec)
  else:
    n_spec = tuple(getattr(jax.typeof(normed).sharding, "spec", ()))
    t_spec = n_spec[:2]
  tok0 = t_spec[0] if len(t_spec) > 0 else None
  tok1 = t_spec[1] if len(t_spec) > 1 else None
  tok_axes = (
      tuple(dict.fromkeys((*_axis_tuple(tok0), *_axis_tuple(tok1)))) if (amesh is not None and not amesh.empty) else ()
  )
  tok_only_axes = tuple(ax for ax in tok_axes if ax not in emb_axes)
  return mesh, amesh, jax.sharding.PartitionSpec(*k_spec), emb_axes, tok_axes, tok_only_axes, tok0, tok1


def _num_vocab_tiles(vocab_size: int) -> int:
  for t in (4, 2):
    if vocab_size >= 64 and vocab_size % t == 0:
      return t
  return 1


def _chunked_sharded_head_xent_fwd(
    normed: jax.Array,
    sharded_kernel: jax.Array,
    targets: jax.Array,
    target_mask: jax.Array,
    loss_chunk_tokens: int,
    z_loss_multiplier: float,
    dot_dtype: jax.typing.DTypeLike,
    cast_logits_to_fp32: bool,
    precision: str,
    token_sharding: jax.sharding.NamedSharding | None,
    with_z_loss: bool,
    weights_in_fp32: bool,
):
  mesh, amesh, k_pspec, emb_axes, tok_axes, _, tok0, tok1 = _sharded_head_axes(normed, sharded_kernel, token_sharding)
  pspec = jax.sharding.PartitionSpec
  if amesh is not None and not amesh.empty:
    normed = _constrain(normed, jax.sharding.NamedSharding(mesh, pspec(tok0, tok1, None)))
    tok_ns = jax.sharding.NamedSharding(mesh, pspec(tok0, tok1))
    targets = _constrain(targets, tok_ns)
    target_mask = _constrain(target_mask, tok_ns)

  emb_dim, vocab_size = sharded_kernel.shape
  v_tiles = _num_vocab_tiles(vocab_size)
  v_sz = vocab_size // v_tiles
  prec = jax.lax.Precision(precision)

  def _fwd_local(x_loc, k_loc, t_loc, m_loc):
    k_cast = jnp.asarray(k_loc, dot_dtype)
    n_local = x_loc.shape[0] * x_loc.shape[1]
    n_chunks = _num_flat_chunks(n_local, loss_chunk_tokens)
    c_sz = n_local // n_chunks
    x_cs = x_loc.reshape(n_chunks, c_sz, emb_dim)
    t_cs = t_loc.reshape(n_chunks, c_sz)
    m_cs = m_loc.reshape(n_chunks, c_sz)
    w_dtype = jnp.float32 if weights_in_fp32 else jnp.int32

    def _vocab_body(vt, stats):
      max_buf, sum_buf, tgt_buf = stats
      v_start = vt * v_sz
      k_vt = jax.lax.dynamic_slice_in_dim(k_cast, v_start, v_sz, axis=1)
      w_vt = jax.lax.all_gather(k_vt, emb_axes, axis=0, tiled=True) if emb_axes else k_vt

      def _chunk_body(i, inner):
        mb, sb, tb = inner
        logits_cv = jax.lax.dot_general(jnp.asarray(x_cs[i], dot_dtype), w_vt, (((1,), (0,)), ((), ())), precision=prec)
        logits_f = logits_cv.astype(jnp.float32) if (cast_logits_to_fp32 or v_tiles > 1) else logits_cv
        m_old = mb[i]
        m_tile = jnp.max(logits_f, axis=-1).astype(jnp.float32)
        m_new = jnp.maximum(m_old, m_tile)
        s_old = sb[i]
        scale_old = jnp.where(jnp.isfinite(m_old), jnp.exp(m_old - m_new), 0.0)
        s_tile = jnp.sum(jnp.exp(logits_f.astype(jnp.float32) - m_new[:, None]), axis=-1)
        s_new = s_old * scale_old + s_tile
        t_rel = t_cs[i] - v_start
        in_tile = (t_rel >= 0) & (t_rel < v_sz)
        safe_t = jnp.where(in_tile, t_rel, 0)
        picked = jnp.squeeze(jnp.take_along_axis(logits_f.astype(jnp.float32), safe_t[:, None], axis=-1), axis=-1)
        t_new = tb[i] + jnp.where(in_tile, picked, 0.0)
        return (
            jax.lax.dynamic_update_index_in_dim(mb, m_new, i, axis=0),
            jax.lax.dynamic_update_index_in_dim(sb, s_new, i, axis=0),
            jax.lax.dynamic_update_index_in_dim(tb, t_new, i, axis=0),
        )

      inner_state = (max_buf, sum_buf, tgt_buf)
      return jax.lax.fori_loop(0, n_chunks, _chunk_body, inner_state) if n_chunks > 1 else _chunk_body(0, inner_state)

    init_stats = (
        jnp.full((n_chunks, c_sz), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((n_chunks, c_sz), dtype=jnp.float32),
        jnp.zeros((n_chunks, c_sz), dtype=jnp.float32),
    )
    max_buf, sum_buf, tgt_buf = (
        jax.lax.fori_loop(0, v_tiles, _vocab_body, init_stats) if v_tiles > 1 else _vocab_body(0, init_stats)
    )
    lse = jnp.log(sum_buf)
    xent = lse + (max_buf - tgt_buf)
    valid = m_cs != 0
    if with_z_loss:
      log_z = lse + max_buf
      z_arr = z_loss_multiplier * jax.lax.square(log_z)
      xent = xent + z_arr
      z_sum = jnp.sum(jnp.where(valid, z_arr, 0.0)).astype(jnp.float32)
    else:
      z_sum = jnp.zeros((), dtype=jnp.float32)
    totals = (
        jnp.sum(jnp.where(valid, xent, 0.0)).astype(jnp.float32),
        z_sum,
        jnp.sum(valid).astype(w_dtype),
    )
    if tok_axes:
      totals = jax.lax.psum(totals, tok_axes)
    tok_shape = x_loc.shape[:2]
    return totals, (max_buf.reshape(tok_shape), sum_buf.reshape(tok_shape))

  if amesh is not None and not amesh.empty:
    sums, (saved_max, saved_sum) = jax.shard_map(
        _fwd_local,
        mesh=mesh,
        in_specs=(pspec(tok0, tok1, None), k_pspec, pspec(tok0, tok1), pspec(tok0, tok1)),
        out_specs=((pspec(), pspec(), pspec()), (pspec(tok0, tok1), pspec(tok0, tok1))),
        check_vma=False,
    )(normed, sharded_kernel, targets, target_mask)
  else:
    sums, (saved_max, saved_sum) = _fwd_local(normed, sharded_kernel, targets, target_mask)
  return sums, (normed, sharded_kernel, targets, target_mask, saved_max, saved_sum)


def _chunked_sharded_head_xent_bwd(
    loss_chunk_tokens: int,
    z_loss_multiplier: float,
    dot_dtype: jax.typing.DTypeLike,
    cast_logits_to_fp32: bool,
    precision: str,
    token_sharding: jax.sharding.NamedSharding | None,
    with_z_loss: bool,
    weights_in_fp32: bool,
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array, jax.Array],
):
  del weights_in_fp32
  normed, sharded_kernel, targets, target_mask, saved_max, saved_sum = res
  g_xent = cotangents[0]
  mesh, amesh, k_pspec, emb_axes, tok_axes, tok_only_axes, tok0, tok1 = _sharded_head_axes(
      normed, sharded_kernel, token_sharding
  )
  pspec = jax.sharding.PartitionSpec
  emb_dim, vocab_size = sharded_kernel.shape
  v_tiles = _num_vocab_tiles(vocab_size)
  v_sz = vocab_size // v_tiles
  prec = jax.lax.Precision(precision)
  all_emb_in_tok = bool(emb_axes) and all(ax in tok_axes for ax in emb_axes)
  emb_axis_sizes = tuple(int(amesh.shape[ax]) for ax in emb_axes) if (amesh is not None and emb_axes) else ()

  def _bwd_local(x_loc, k_loc, t_loc, m_loc, max_loc, sum_loc, g_loc):
    k_cast = jnp.asarray(k_loc, dot_dtype)
    n_shards = emb_dim // k_loc.shape[0]
    n_local = x_loc.shape[0] * x_loc.shape[1]
    n_chunks = _num_flat_chunks(n_local, loss_chunk_tokens)
    c_sz = n_local // n_chunks
    x_cs = x_loc.reshape(n_chunks, c_sz, emb_dim)
    t_cs = t_loc.reshape(n_chunks, c_sz)
    m_cs = m_loc.reshape(n_chunks, c_sz)
    max_cs = max_loc.reshape(n_chunks, c_sz)
    sum_cs = sum_loc.reshape(n_chunks, c_sz)
    if with_z_loss:
      z_factor_cs = 1.0 + 2.0 * z_loss_multiplier * (jnp.log(sum_cs) + max_cs)
    else:
      z_factor_cs = jnp.ones_like(sum_cs)
    scale_cs = g_loc.astype(jnp.float32) * (m_cs != 0).astype(jnp.float32)

    def _vocab_bwd_body(vt, state):
      dx_f32, dk_buf = state
      v_start = vt * v_sz
      k_vt = jax.lax.dynamic_slice_in_dim(k_cast, v_start, v_sz, axis=1)
      w_vt = jax.lax.all_gather(k_vt, emb_axes, axis=0, tiled=True) if emb_axes else k_vt

      def _chunk_bwd_body(i, inner):
        dx_cur, dw_vt = inner
        x_c_dot = jnp.asarray(x_cs[i], dot_dtype)
        logits_cv = jax.lax.dot_general(x_c_dot, w_vt, (((1,), (0,)), ((), ())), precision=prec)
        logits_f = logits_cv.astype(jnp.float32) if (cast_logits_to_fp32 or v_tiles > 1) else logits_cv
        exp_s = jnp.exp(logits_f.astype(jnp.float32) - max_cs[i][:, None])
        prob_s = z_factor_cs[i][:, None] * exp_s / sum_cs[i][:, None]
        one_hot_v = jax.nn.one_hot(t_cs[i] - v_start, v_sz, dtype=jnp.float32)
        g_logits_v = jnp.asarray(scale_cs[i][:, None] * (prob_s - one_hot_v), dot_dtype)
        dx_add = jax.lax.dot_general(
            g_logits_v,
            w_vt,
            (((1,), (1,)), ((), ())),
            precision=prec,
            preferred_element_type=jnp.float32,
        )
        dw_add = jax.lax.dot_general(
            x_c_dot,
            g_logits_v,
            (((0,), (0,)), ((), ())),
            precision=prec,
            preferred_element_type=jnp.float32,
        )
        return (
            jax.lax.dynamic_update_index_in_dim(dx_cur, dx_cur[i] + dx_add, i, axis=0),
            dw_vt + dw_add,
        )

      init_inner = (dx_f32, jnp.zeros((emb_dim, v_sz), dtype=jnp.float32))
      dx_f32_next, dw_vt_acc = (
          jax.lax.fori_loop(0, n_chunks, _chunk_bwd_body, init_inner) if n_chunks > 1 else _chunk_bwd_body(0, init_inner)
      )
      if not emb_axes:
        dk_vt_cast = jnp.asarray(dw_vt_acc, dot_dtype)
        if tok_only_axes:
          dk_vt_cast = jax.lax.psum(dk_vt_cast, tok_only_axes)
      elif all_emb_in_tok:
        dw_shards = jnp.asarray(dw_vt_acc, dot_dtype).reshape(n_shards, k_loc.shape[0], v_sz)
        if tok_only_axes:
          dw_shards = jax.lax.psum(dw_shards, tok_only_axes)
        dk_vt_cast = jax.lax.psum_scatter(dw_shards, emb_axes, scatter_dimension=0, tiled=True).squeeze(0)
      else:
        dw_sh = jnp.asarray(dw_vt_acc, dot_dtype).reshape(*emb_axis_sizes, k_loc.shape[0], v_sz)
        if tok_only_axes:
          dw_sh = jax.lax.psum(dw_sh, tok_only_axes)
        for ax in emb_axes:
          if ax in tok_axes:
            dw_sh = jax.lax.psum_scatter(dw_sh, ax, scatter_dimension=0, tiled=True).squeeze(0)
          else:
            dw_sh = jax.lax.dynamic_index_in_dim(dw_sh, jax.lax.axis_index(ax), axis=0, keepdims=False)
        dk_vt_cast = dw_sh
      dk_buf_next = jax.lax.dynamic_update_slice_in_dim(dk_buf, dk_vt_cast.astype(k_loc.dtype), v_start, axis=1)
      return dx_f32_next, dk_buf_next

    init_state = (
        jnp.zeros((n_chunks, c_sz, emb_dim), dtype=jnp.float32),
        jnp.zeros_like(k_loc),
    )
    dx_f32_out, dk_out = (
        jax.lax.fori_loop(0, v_tiles, _vocab_bwd_body, init_state) if v_tiles > 1 else _vocab_bwd_body(0, init_state)
    )
    return dx_f32_out.astype(x_loc.dtype).reshape(x_loc.shape), dk_out

  if amesh is not None and not amesh.empty:
    g_rep = _constrain(jnp.asarray(g_xent), jax.sharding.NamedSharding(mesh, pspec()))
    dx, dk = jax.shard_map(
        _bwd_local,
        mesh=mesh,
        in_specs=(
            pspec(tok0, tok1, None),
            k_pspec,
            pspec(tok0, tok1),
            pspec(tok0, tok1),
            pspec(tok0, tok1),
            pspec(tok0, tok1),
            pspec(),
        ),
        out_specs=(pspec(tok0, tok1, None), k_pspec),
        check_vma=False,
    )(normed, sharded_kernel, targets, target_mask, saved_max, saved_sum, g_rep)
  else:
    dx, dk = _bwd_local(normed, sharded_kernel, targets, target_mask, saved_max, saved_sum, jnp.asarray(g_xent))
  return dx, dk, None, None


chunked_sharded_head_xent.defvjp(_chunked_sharded_head_xent_fwd, _chunked_sharded_head_xent_bwd)


def _with_layer_axis(x: jax.Array, ndim_per_layer: int) -> tuple[jax.Array, bool]:
  """Prepends a singleton layer axis to unscanned router aux arrays."""
  scanned = x.ndim == ndim_per_layer + 1
  return (x if scanned else x[None]), scanned


@jax.named_call
def moe_switch_load_balance_loss(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    weight: float,
) -> jax.Array:
  """Switch-style sequence-wise load balance loss, averaged over layers.

  Mirrors MaxText's `RoutedMoE.load_balance_loss` over the softmax-normalized
  pre-bias router scores (`aux.logits`), one layer at a time to avoid a
  `(layers, batch, seq, top_k, experts)` intermediate.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    num_experts_per_tok: Router top-k.
    weight: `load_balance_loss_weight`.

  Returns:
    The scalar loss.
  """
  selected, _ = _with_layer_axis(aux.selected_experts, 3)
  gates, _ = _with_layer_axis(aux.logits, 3)

  def layer_loss(selected_l, gates_l):
    probs = jax.nn.softmax(gates_l.astype(jnp.float32), axis=-1)
    expert_mask = jax.nn.one_hot(selected_l, num_classes=num_experts, dtype=jnp.float32)
    # Fraction of tokens dispatched to each expert, per sequence.
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / num_experts_per_tok
    # Fraction of routing probability allocated to each expert, per sequence.
    density_prob = jnp.mean(probs, axis=1)
    return jnp.mean(density * density_prob)

  num_layers = selected.shape[0]
  total = functools.reduce(
      operator.add,
      (layer_loss(selected[i], gates[i]) for i in range(num_layers)),
  )
  return total / num_layers * (num_experts**2) * weight


@jax.named_call
def moe_megatron_seq_aux_loss(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    weight: float,
) -> jax.Array:
  """Megatron-LM's sigmoid-router seq_aux_loss, one loss per layer.

  Mirrors MaxText's `RoutedMoE.megatron_seq_aux_loss`: the probabilities are
  the pre-bias sigmoid scores (`aux.logits`, upcast to fp32) normalized per
  token, the token fractions come from a plain top-k over those scores, and
  both are per-sequence statistics averaged over the batch and scaled by
  `num_experts**2 * weight`. MaxText sums the per-layer losses into the
  objective.

  The scores are floored at 1e-6 and the normalization uses a 1e-6 (rather
  than 1e-20) epsilon: bf16 sigmoid scores can underflow to 0, and the
  normalization's pullback divides by the squared row sum, which flushes to
  zero on TPU with the smaller epsilon and yields NaN gradients.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    num_experts_per_tok: Router top-k.
    weight: `load_balance_loss_weight`.

  Returns:
    The per-layer losses `(num_layers,)` for a scanned stack, or a scalar for
    a single layer.
  """
  logits, scanned = _with_layer_axis(aux.logits, 3)

  def layer_loss(logits_l):
    scores = jnp.maximum(logits_l.astype(jnp.float32), 1e-6)
    probs = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-6)
    _, top_k_indices = jax.lax.top_k(scores, k=num_experts_per_tok)
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.int32)
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / num_experts_per_tok
    density_prob = jnp.mean(probs, axis=1)
    return jnp.mean(density * density_prob) * (num_experts**2) * weight

  # One batched top-k and reduction over the layer axis rather than a Python
  # loop over layers, which issued a few hundred small ops (one top-k, one-hot
  # and mean per layer) outside the layer scan. The one-hot stays fused into
  # its reduction, so the `(layers, batch, seq, top_k, experts)` mask is never
  # materialized.
  losses = jax.vmap(layer_loss)(logits)
  return losses if scanned else losses[0]


def expert_counts(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int) -> jax.Array:
  """Full-batch int32 expert counts, `(num_layers, E)` if scanned else `(E,)`.

  Prefers the router's per-shard dispatch counts (`aux.group_sizes`), summing
  the shards; otherwise mirrors `moe.calculate_expert_counts` over
  `aux.selected_experts`, one layer at a time.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.

  Returns:
    The expert counts.
  """
  if aux.group_sizes is not None:
    return jnp.sum(aux.group_sizes, axis=-2, dtype=jnp.int32)
  selected, scanned = _with_layer_axis(aux.selected_experts, 3)
  counts = jnp.stack(
      [
          jnp.sum(
              jax.nn.one_hot(selected[i], num_experts, dtype=jnp.int32),
              axis=(0, 1, 2),
          )
          for i in range(selected.shape[0])
      ]
  )
  return counts if scanned else counts[0]


def load_balance_updates_from_counts(counts: jax.Array, *, num_experts: int, rate: float) -> jax.Array:
  """Mirrors `moe.load_balance_updates_from_counts`: rate * sign(mean - load)."""
  total_tokens = jnp.sum(counts, axis=-1, keepdims=True)
  average_load = total_tokens / num_experts
  return jnp.sign(average_load - counts) * rate


@jax.named_call
def routed_bias_updates(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int, rate: float) -> jax.Array:
  """Loss-free load balancing updates of the routed gate bias.

  Mirrors `RoutedMoE` under `should_update_load_balance()`:
  `rate * sign(mean_load - load)` over the global batch's expert counts.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    rate: `routed_bias_update_rate`.

  Returns:
    The updates, `(num_layers, E)` for a scanned stack or `(E,)` for a single
    layer.
  """
  counts = expert_counts(aux, num_experts=num_experts)
  return load_balance_updates_from_counts(counts, num_experts=num_experts, rate=rate)


def moe_router_aux_outputs(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    load_balance_loss_weight: float,
    megatron_seq_aux_loss: bool,
    routed_bias_update_rate: float,
) -> tuple[jax.Array | None, jax.Array | None]:
  """Returns `(load balance loss, routed bias updates)` of a router aux.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    num_experts_per_tok: Router top-k.
    load_balance_loss_weight: Weight of the load balance loss; 0 disables it.
    megatron_seq_aux_loss: Use `moe_megatron_seq_aux_loss` (per layer) rather
      than `moe_switch_load_balance_loss` (averaged over layers).
    routed_bias_update_rate: Rate of the routed-bias updates; 0 disables them.

  Returns:
    The loss and the updates; either is None when disabled.
  """
  lb_loss = None
  if load_balance_loss_weight > 0.0:
    loss_fn = moe_megatron_seq_aux_loss if megatron_seq_aux_loss else moe_switch_load_balance_loss
    lb_loss = loss_fn(
        aux,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        weight=load_balance_loss_weight,
    )
  updates = None
  if routed_bias_update_rate > 0.0:
    updates = routed_bias_updates(aux, num_experts=num_experts, rate=routed_bias_update_rate)
  return lb_loss, updates
