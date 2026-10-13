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

"""DeepSeek MoE experts implementation."""

import functools
from typing import Protocol

import jax
from jax.experimental import layout as jax_layout
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage import quantization


class GmmFn(Protocol):
  """Per-expert grouped matmuls on the 3D `[BT, D0, D1]` token layout.

  The optional `[1, 1]` f32 scales select per-tensor fp8: operands that are
  already fp8 are consumed as is, `x_scale` / `x_qtype` quantize a bf16 x
  in-kernel, and the f32 accumulator is multiplied by the scales of the fp8
  operands (`out_scale`, `dx_scale`, `dw_scale`) before the bf16 output.
  """

  def gate(
      self,
      x: jt.Num[jax.Array, "BT D0 D1"],
      w: jt.Num[jax.Array, "E D F"],
      group_sizes: jt.Num[jax.Array, "E"],
      out: jt.Num[jax.Array, "R F"],
      out_offset: jt.Int[jax.Array, ""] | int,
      out_scale: jt.Float[jax.Array, "1 1"] | None = None,
  ) -> jt.Num[jax.Array, "R F"]:
    """Writes the gating GMM of x to rows [out_offset, ...) of out (fwd only)."""
    ...

  def gate_bwd(
      self,
      x: jt.Num[jax.Array, "BT D0 D1"],
      w: jt.Num[jax.Array, "E D F"],
      group_sizes: jt.Num[jax.Array, "E"],
      dgate: jt.Num[jax.Array, "BT F"],
      dw_acc: jt.Num[jax.Array, "E D F"] | None = None,
      *,
      w_q: jt.Num[jax.Array, "E D F"] | None = None,
      dx_scale: jt.Float[jax.Array, "1 1"] | None = None,
      dw_scale: jt.Float[jax.Array, "1 1"] | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jt.Num[jax.Array, "BT D0 D1"], jt.Num[jax.Array, "E D F"]]:
    """Returns (dx, dw) of the gating GMM, accumulating into dw_acc in place.

    With `w_q` (the fp8 copy of w), dx is computed from w_q; w then only gives
    the type of dw. dw is in `dw_dtype`, if given, else in w's dtype.
    """
    ...

  def linear(
      self,
      x: jt.Num[jax.Array, "BT F"],
      w: jt.Num[jax.Array, "E F D"],
      group_sizes: jt.Num[jax.Array, "E"],
      *,
      x_scale: jt.Float[jax.Array, "1 1"] | None = None,
      x_qtype: jnp.dtype | None = None,
      out_scale: jt.Float[jax.Array, "1 1"] | None = None,
  ) -> jt.Num[jax.Array, "BT D0 D1"]:
    ...

  def linear_bwd(
      self,
      x: jt.Num[jax.Array, "BT F"],
      w: jt.Num[jax.Array, "E F D"],
      group_sizes: jt.Num[jax.Array, "E"],
      dout: jt.Num[jax.Array, "BT D0 D1"],
      dw_acc: jt.Num[jax.Array, "E F D"] | None = None,
      *,
      w_q: jt.Num[jax.Array, "E F D"] | None = None,
      x_scale: jt.Float[jax.Array, "1 1"] | None = None,
      x_qtype: jnp.dtype | None = None,
      dx_scale: jt.Float[jax.Array, "1 1"] | None = None,
      dw_scale: jt.Float[jax.Array, "1 1"] | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jt.Num[jax.Array, "BT F"], jt.Num[jax.Array, "E F D"]]:
    """Returns (dx, dw) of `linear`, accumulating into dw_acc in place.

    dw is in `dw_dtype`, if given, else in w's dtype.
    """
    ...


# Row alignment of each chunk's gating activations in the activation bank. The
# gating GMM writes whole sublane tiles of its x: 16 rows for bf16, 32 for fp8.
_BANK_ROW_ALIGN = 32


def bank_aligned(num_rows: jax.Array | int) -> jax.Array | int:
  """Rounds `num_rows` up to the bank row alignment."""
  return -(-num_rows // _BANK_ROW_ALIGN) * _BANK_ROW_ALIGN


def bank_chunk_row(
    bank: jt.Num[jax.Array, "bank_rows D_gate"],
    capacity: int,
    offset: jt.Int[jax.Array, ""],
    num_tokens: jt.Int[jax.Array, ""],
) -> tuple[jt.Int[jax.Array, ""], jt.Bool[jax.Array, ""]]:
  """Returns the bank row holding a chunk's gating activations, and overflow.

  The bank holds its checkpoint rows followed by `bank_aligned(capacity)`
  scratch rows. A chunk pushed at `offset` that does not fit in the checkpoint
  rows lives in the scratch rows instead, only until it is consumed.

  Args:
    bank: Cross-layer gating activation bank on this device.
    capacity: Maximum number of tokens in a chunk.
    offset: Bank offset at which the chunk was pushed.
    num_tokens: Number of tokens in the chunk.

  Returns:
    Tuple of (first bank row of the chunk, whether it overflowed).
  """
  checkpoint_rows = bank.shape[0] - bank_aligned(capacity)
  is_overflow = (offset < 0) | (offset + num_tokens > checkpoint_rows)
  return jnp.where(is_overflow, checkpoint_rows, offset), is_overflow


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_shared_expert(
    x: jt.Num[jax.Array, "*BT D"],
    w: dsv3_types.DSv3MoESharedExpertWeightsPytree,
) -> jt.Num[jax.Array, "*BT D"]:
  """Performs computation for a shared expert.

  Args:
    x: Input tokens with any number of leading dimensions.
    w: Shared expert weights.

  Returns:
    Output tokens with the same shape as the input.
  """
  assert w.gate_0 is not None
  assert w.gate_1 is not None
  assert w.linear is not None
  dot = functools.partial(jnp.tensordot, axes=1)
  return dot(jax.nn.silu(dot(x, w.gate_0)) * dot(x, w.gate_1), w.linear)


def _scale(value: float) -> jt.Float[jax.Array, "1 1"]:
  return jnp.full((1, 1), value, jnp.float32)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_impl(
    x: jt.Num[jax.Array, "BT D0 D1"],
    w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "bank_rows D_gate"],
    bank_offset: jt.Int[jax.Array, ""],
    *,
    gmm_fn: GmmFn,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    jt.Num[jax.Array, "bank_rows D_gate"],
]:
  """Performs routed expert computation on local tokens, pushing gate_out.

  The gating GMM writes gate_out straight into the bank at `bank_chunk_row`,
  and the SiLU reads it from there.

  Args:
    x: Input tokens, sorted by expert.
    w: Routed expert weights.
    group_sizes: Number of tokens assigned to each expert on the local device.
    sorted_coeffs: Routing coefficients sorted like x, applied to the SiLU
      output before the linear GMM.
    bank: Cross-layer gating activation bank on this device.
    bank_offset: Bank offset at which gate_out is pushed.
    gmm_fn: Function to perform Grouped Matrix Multiplication.
    quant_rule: Optional fp8 quantization. x and w must then already be
      quantized with its static act / weight scales; the SiLU output is
      quantized inside the linear GMM. gate_out and the output stay bf16.

  Returns:
    Tuple of (output tokens sorted by expert, updated bank).
  """
  assert w.gate is not None
  assert w.linear is not None
  if sorted_coeffs.ndim == 1:
    sorted_coeffs = sorted_coeffs[:, None]
  gate_scale = act_scale = w_scale = act_qtype = None
  if quant_rule is not None:
    sa, sw = quant_rule.act_scale, quant_rule.weight_scale
    gate_scale, act_scale, w_scale = _scale(sa * sw), _scale(sa), _scale(sw)
    act_qtype = quant_rule.act_qtype

  num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  row, _ = bank_chunk_row(bank, x.shape[0], bank_offset, num_tokens)
  if quant_rule is not None and getattr(gmm_fn, "supports_fused_glu", False):
    with jax.named_scope("gate"):
      bank, act = getattr(gmm_fn, "gate")(
          x,
          w.gate,
          group_sizes,
          bank,
          row,
          out_scale=gate_scale,
          glu_coeffs=sorted_coeffs,
          glu_out_dtype=act_qtype,
          glu_out_scale=act_scale,
      )
  else:
    with jax.named_scope("gate"):
      bank = gmm_fn.gate(x, w.gate, group_sizes, bank, row, out_scale=gate_scale)
    with jax.named_scope("silu"):
      act = ops.ragged_silu_mul_fwd(
          bank,
          row,
          sorted_coeffs,
          num_tokens,
          block_size_tokens=1024,
          block_size_hidden=2048,
      )
  with jax.named_scope("linear"):
    routed_out = gmm_fn.linear(
        act,
        w.linear,
        group_sizes,
        x_scale=act_scale,
        x_qtype=act_qtype,
        out_scale=w_scale,
    )
  return routed_out, bank


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts(
    x: jt.Num[jax.Array, "BT D0 D1"],
    w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "B_cap D_gate"],
    bank_offset: jt.Num[jax.Array, "num_token_shards"],
    *,
    gmm_fn: GmmFn,
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    jt.Num[jax.Array, "B_cap D_gate"],
    jt.Num[jax.Array, "num_token_shards"],
]:
  """Sharded wrapper for routed expert computation, pushing gate_out to the bank."""
  bank_pspec = jax.typeof(bank).sharding.spec
  offset_pspec = jax.typeof(bank_offset).sharding.spec

  def _fwd_push(x, w, group_sizes, sorted_coeffs, bank, bank_offset):
    routed_out, bank = dsv3_routed_experts_impl(x, w, group_sizes, sorted_coeffs, bank, bank_offset[0], gmm_fn=gmm_fn)
    num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
    return routed_out, bank, bank_offset + bank_aligned(num_tokens)

  return jax.shard_map(
      _fwd_push,
      mesh=mesh,
      out_specs=(jax.typeof(x).sharding.spec, bank_pspec, offset_pspec),
  )(x, w, group_sizes, sorted_coeffs, bank, bank_offset)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_bwd_impl(
    grad_routed: jt.Num[jax.Array, "BT D0 D1"],
    routed: jt.Num[jax.Array, "BT D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    bank: jt.Num[jax.Array, "bank_rows D_gate"],
    bank_offset: jt.Int[jax.Array, ""],
    grad_w_acc: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None,
    *,
    gmm_fn: GmmFn,
    quant_rule: quantization.GmmQuantRule | None = None,
    w_routed_q: dsv3_types.DSv3MoERoutedExpertWeightsPytree | None = None,
    grad_routed_amax: jt.Float[jax.Array, "1 1"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    jt.Num[jax.Array, "bank_rows D_gate"],
]:
  """Performs routed expert backward pass on local tokens, reading gate_out from the bank.

  gate_out of the `sum(group_sizes)` tokens pushed at `bank_offset` is read
  from the bank in place. If it overflowed the bank's checkpoint rows, it is
  first rematerialized into the scratch rows. The gradients only cover the
  rows of this chunk, never the whole bank.

  Args:
    grad_routed: Cotangent of the routed expert output.
    routed: Input tokens, sorted by expert.
    w_routed: Routed expert weights.
    group_sizes: Number of tokens assigned to each expert on the local device.
    sorted_coeffs: Routing coefficients sorted like x.
    bank: Cross-layer gating activation bank on this device.
    bank_offset: Bank offset at which gate_out was pushed.
    grad_w_acc: Optional routed expert weight gradients (e.g. of other chunks or
      microbatches) to accumulate this chunk's into, in place.
    gmm_fn: Function to perform Grouped Matrix Multiplication.
    quant_rule: Optional fp8 quantization. routed must then already be quantized
      with its static act scale, and `w_routed_q` holds w_routed quantized with
      its static weight scale. The output gradients of both GMMs are quantized
      with one absmax scale per chunk. The weight gradients are in its
      `grad_dtype` and the token gradients in bf16, so w_routed may itself be
      the fp8 weights.
    w_routed_q: fp8 routed expert weights, required with `quant_rule`.
    grad_routed_amax: Optional precomputed `[1, 1]` f32 absmax of
      `grad_routed[:num_tokens]` (e.g. from `permute_chunk(...,
      with_absmax=True)`).

  Returns:
    Tuple of (grad_routed_in, grad_w_routed, grad_sorted_coeffs, bank), with
    grad_w_routed including grad_w_acc.
  """
  assert w_routed.gate is not None
  assert w_routed.linear is not None
  orig_coeffs_shape = sorted_coeffs.shape
  if sorted_coeffs.ndim == 1:
    sorted_coeffs = sorted_coeffs[:, None]
  if grad_w_acc is None:
    grad_w_acc = dsv3_types.DSv3MoERoutedExpertWeightsPytree()
  num_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  row, is_overflow = bank_chunk_row(bank, routed.shape[0], bank_offset, num_tokens)
  w_gate_q = w_linear_q = gate_scale = act_scale = w_scale = act_qtype = dw_dtype = None
  if quant_rule is not None:
    assert w_routed_q is not None
    sa, sw = quant_rule.act_scale, quant_rule.weight_scale
    w_gate_q, w_linear_q = w_routed_q.gate, w_routed_q.linear
    gate_scale, act_scale, w_scale = _scale(sa * sw), _scale(sa), _scale(sw)
    act_qtype = quant_rule.act_qtype
    dw_dtype = quant_rule.grad_dtype

  @jax.named_call
  def _remat_gate_out(bank):
    with jax.named_scope("gate"):
      return gmm_fn.gate(
          routed,
          w_routed.gate if w_gate_q is None else w_gate_q,
          group_sizes,
          bank,
          row,
          out_scale=gate_scale,
      )

  with jax.named_scope("retrieve_or_remat_gate_out"):
    bank = jax.lax.cond(is_overflow, _remat_gate_out, lambda b: b, bank)

  fuse_glu = quant_rule is not None and getattr(gmm_fn, "supports_fused_glu", False)
  dyn_grad_scale = getattr(gmm_fn, "supports_dynamic_grad_scale", False)
  with jax.named_scope("silu"):
    act = ops.ragged_silu_mul_fwd(
        bank,
        row,
        sorted_coeffs,
        num_tokens,
        block_size_tokens=1024,
        block_size_hidden=2048,
        out_dtype=act_qtype if fuse_glu else None,
        out_scale=act_scale if fuse_glu else None,
    )
  with jax.named_scope("linear"):
    if quant_rule is not None and dyn_grad_scale:
      if grad_routed_amax is None:
        grad_routed_amax = quantization.ragged_absmax(grad_routed, num_tokens)
      s_dy = quantization.absmax_to_scale(grad_routed_amax, quant_rule.bwd_qtype)
      grad_act, grad_w_linear = getattr(gmm_fn, "linear_bwd")(
          act,
          w_routed.linear,
          group_sizes,
          grad_routed,
          grad_w_acc.linear,
          w_q=w_linear_q,
          x_scale=act_scale,
          x_qtype=act_qtype,
          dout_scale=s_dy,
          dout_qtype=quant_rule.bwd_qtype,
          dx_scale=w_scale,
          dw_scale=None,
          dw_dtype=dw_dtype,
      )
    else:
      dact_scale = dw_linear_scale = None
      if quant_rule is not None:
        grad_routed, s_dy = quantization.absmax_quantize(grad_routed, quant_rule.bwd_qtype, num_tokens)
        dact_scale, dw_linear_scale = s_dy * quant_rule.weight_scale, s_dy
      grad_act, grad_w_linear = gmm_fn.linear_bwd(
          act,
          w_routed.linear,
          group_sizes,
          grad_routed,
          grad_w_acc.linear,
          w_q=w_linear_q,
          x_scale=act_scale,
          x_qtype=act_qtype,
          dx_scale=dact_scale,
          dw_scale=dw_linear_scale,
          dw_dtype=dw_dtype,
      )
    act_max = None
    if quant_rule is not None:
      # Like qwix's straight-through estimator, don't backprop through the
      # values clipped by the fixed act calibration.
      act_max = quant_rule.act_scale * float(jnp.finfo(act_qtype).max)
      if not fuse_glu:
        grad_act = jnp.where(jnp.abs(act) <= act_max, grad_act, 0).astype(grad_act.dtype)
  with jax.named_scope("silu"):
    grad_gate_amax = None
    if quant_rule is not None and dyn_grad_scale:
      grad_gate_out, grad_sorted_coeffs, grad_gate_amax = ops.ragged_silu_mul_bwd(
          grad_act,
          bank,
          row,
          sorted_coeffs,
          num_tokens,
          block_size_tokens=1024,
          block_size_hidden=1024,
          g_mask_max=act_max if fuse_glu else None,
          with_absmax=True,
      )
    else:
      grad_gate_out, grad_sorted_coeffs = ops.ragged_silu_mul_bwd(
          grad_act,
          bank,
          row,
          sorted_coeffs,
          num_tokens,
          block_size_tokens=1024,
          block_size_hidden=1024,
          g_mask_max=act_max if fuse_glu else None,
      )
  with jax.named_scope("gate"):
    dx_scale = dw_gate_scale = None
    if quant_rule is not None:
      if grad_gate_amax is not None:
        s_dg = quantization.absmax_to_scale(grad_gate_amax, quant_rule.bwd_qtype)
        grad_gate_out = quantization.quantize_with_scale(grad_gate_out, s_dg, quant_rule.bwd_qtype, num_tokens)
      else:
        grad_gate_out, s_dg = quantization.absmax_quantize(grad_gate_out, quant_rule.bwd_qtype, num_tokens)
      dx_scale = s_dg * quant_rule.weight_scale
      dw_gate_scale = s_dg * quant_rule.act_scale
    grad_routed_in, grad_w_gate = gmm_fn.gate_bwd(
        routed,
        w_routed.gate,
        group_sizes,
        grad_gate_out,
        grad_w_acc.gate,
        w_q=w_gate_q,
        dx_scale=dx_scale,
        dw_scale=dw_gate_scale,
        dw_dtype=dw_dtype,
    )
  grad_w_routed = dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=grad_w_gate, linear=grad_w_linear)
  grad_sorted_coeffs = jnp.reshape(grad_sorted_coeffs, orig_coeffs_shape)
  return grad_routed_in, grad_w_routed, grad_sorted_coeffs, bank


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_bwd(
    grad_routed: jt.Num[jax.Array, "BT D0 D1"],
    routed: jt.Num[jax.Array, "BT D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    sorted_coeffs: jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
    ragged_bank: jt.Num[jax.Array, "B_cap D_gate"],
    bank_offset: jt.Num[jax.Array, "num_token_shards"],
    *,
    gmm_fn: GmmFn,
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    jt.Num[jax.Array, "B_cap D_gate"],
    jt.Num[jax.Array, "num_token_shards"],
    jt.Num[jax.Array, "BT 1"] | jt.Num[jax.Array, "BT"],
]:
  """Sharded wrapper for routed expert backward pass, popping gate_out from the bank."""
  tiling = ((8, 128), (2, 1)) if jnp.dtype(ragged_bank.dtype).itemsize == 2 else ((8, 128),)
  bank_layout = jax_layout.Layout(
      major_to_minor=(0, 1),
      tiling=tiling,
  )
  ragged_bank = jax_layout.with_layout_constraint(ragged_bank, bank_layout)
  x_pspec = jax.typeof(routed).sharding.spec
  offset_pspec = jax.typeof(bank_offset).sharding.spec
  out_specs = (
      x_pspec,
      jax.tree_util.tree_map(lambda g: jax.typeof(g).sharding.spec.to_ct_spec(), w_routed),
      jax.typeof(ragged_bank).sharding.spec,
      offset_pspec,
      jax.typeof(sorted_coeffs).sharding.spec,
  )

  def _bwd_pop(
      grad_routed,
      routed,
      w_routed,
      group_sizes,
      sorted_coeffs,
      ragged_bank,
      bank_offset,
  ):
    new_bank_offset = bank_offset - bank_aligned(jnp.sum(group_sizes, dtype=jnp.int32))
    grad_routed_in, grad_w_routed, grad_sorted_coeffs, ragged_bank = dsv3_routed_experts_bwd_impl(
        grad_routed,
        routed,
        w_routed,
        group_sizes,
        sorted_coeffs,
        ragged_bank,
        new_bank_offset[0],
        gmm_fn=gmm_fn,
    )
    return (
        grad_routed_in,
        grad_w_routed,
        ragged_bank,
        new_bank_offset,
        grad_sorted_coeffs,
    )

  return jax.shard_map(
      _bwd_pop,
      mesh=mesh,
      out_specs=out_specs,
  )(
      grad_routed,
      routed,
      w_routed,
      group_sizes,
      sorted_coeffs,
      ragged_bank,
      bank_offset,
  )
