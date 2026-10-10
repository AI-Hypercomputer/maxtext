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

"""DSv3 MLA implementation."""

from collections.abc import Mapping
import dataclasses
import functools
import math
from typing import Any, Protocol

import jax
import jax.experimental.compute_on
import jax.numpy as jnp
import jaxtyping as jt
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask
import typeguard

from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage import quantization
from maxtext.experimental.lineage import tokamax_splash_attention_kernel

compute_on = jax.experimental.compute_on.compute_on


class NormFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*input_dims"],
      scale: jt.Num[jax.Array, "..."],
  ) -> jt.Num[jax.Array, "*input_dims"]:
    ...


class RopeFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*BT N D"],
      freqs: tuple[
          jt.Num[jax.Array, "*BT 1 D"],
          jt.Num[jax.Array, "*BT 1 D"],
      ],
  ) -> jt.Num[jax.Array, "*BT N D"]:
    ...


def _quantize_weight(w: jax.Array, quant_rule: quantization.GmmQuantRule) -> jax.Array:
  """Returns w quantized with the rule's static weight scale, unless it is."""
  if w.dtype == quant_rule.weight_qtype:
    return w
  return quantization.static_quantize(w, quant_rule.weight_qtype, quant_rule.weight_scale)


def quantize_mla_weights(
    w: dsv3_types.DSv3MLAWeightsPytree,
    quant_rule: quantization.GmmQuantRule,
) -> dsv3_types.DSv3MLAWeightsPytree:
  """Returns w with its projection weights quantized, e.g. before collection.

  Weights already in the rule's weight dtype are returned as is, and the norm
  scales stay unquantized.

  Args:
    w: MLA weights, possibly stacked for all layers.
    quant_rule: Quantization rule of the MLA projections.

  Returns:
    w with the q down, q up, kv down, v up and out projection weights
    quantized with the rule's static weight scale. k up stays unquantized: an
    fp8 k up dot outputs k head-dim-minor, which splash's SEQ_MINOR k layout
    would have to transpose.
  """

  def q(w: jax.Array | None) -> jax.Array | None:
    return None if w is None else _quantize_weight(w, quant_rule)

  return dataclasses.replace(
      w,
      q_down=q(w.q_down),
      q_up=q(w.q_up),
      kv_down=q(w.kv_down),
      v_up=q(w.v_up),
      out=q(w.out),
  )


def _quantize_act(x: jax.Array, quant_rule: quantization.GmmQuantRule) -> jax.Array:
  """Returns x quantized with the rule's static act scale, unless it is."""
  if x.dtype == quant_rule.act_qtype:
    return x
  return quantization.static_quantize(x, quant_rule.act_qtype, quant_rule.act_scale)


def _fp8_dot(
    x: jax.Array,
    y: jax.Array,
    axes: int | tuple[Any, Any],
    out_dtype: Any = jnp.float32,
) -> jax.Array:
  """Returns `jnp.tensordot(x, y, axes)` of fp8 operands, accumulated in f32.

  Unlike `jnp.tensordot`, never promotes mixed fp8 operands (e.g. e5m2 and
  e4m3fn) to a wider dtype before the dot.

  Args:
    x: Left operand.
    y: Right operand.
    axes: `jnp.tensordot` axes.
    out_dtype: Dtype the f32 accumulator is rounded to.

  Returns:
    The product in `out_dtype`, with x's free dimensions followed by y's.
  """
  if isinstance(axes, int):
    axes = (tuple(range(x.ndim - axes, x.ndim)), tuple(range(axes)))
  return jax.lax.dot_general(x, y, (axes, ((), ())), preferred_element_type=out_dtype)


def _qdot(
    x: jax.Array,
    w: jax.Array,
    quant_rule: quantization.GmmQuantRule,
    axes: int = 1,
) -> jax.Array:
  """Returns the bf16 `dot(x, w, axes)` of static-quantized x and w."""
  out = _fp8_dot(_quantize_act(x, quant_rule), _quantize_weight(w, quant_rule), axes)
  scale = quant_rule.act_scale * quant_rule.weight_scale
  return (out * scale).astype(jnp.bfloat16)


def _ste(grad_x: jax.Array, x: jax.Array, quant_rule: quantization.GmmQuantRule) -> jax.Array:
  """Zeroes grad_x where the static act calibration clipped x.

  Like qwix's straight-through estimator for `fixed` calibrations.

  Args:
    grad_x: Gradient of x.
    x: Unquantized activations.
    quant_rule: Quantization rule x was quantized with.

  Returns:
    grad_x, zeroed where |x| exceeds the calibration range.
  """
  x_max = quant_rule.act_scale * float(jnp.finfo(quant_rule.act_qtype).max)
  # Compared in f32 (exactly): a bf16 compare can crash the TPU compiler.
  mask = jnp.abs(x.astype(jnp.float32)) <= x_max
  return jnp.where(mask, grad_x, 0).astype(grad_x.dtype)


def _match_spec(x: jax.Array, spec: jax.sharding.PartitionSpec) -> jax.Array:
  """Inside a shard_map, makes the local partial x match out spec `spec`.

  Axes that x varies along but `spec` neither shards nor leaves unreduced are
  summed over, and the ones `spec` leaves unreduced are cast to unreduced.
  Unreduced axes that x does not vary along must have size 1 and are cast too.

  Args:
    x: Local value, possibly a partial sum along some mesh axes.
    spec: Output PartitionSpec of x.

  Returns:
    x, reduced or cast to match `spec`.
  """
  sharded = set(spec.unreduced)
  for part in spec.partitions:
    if part is not None:
      sharded.update((part,) if isinstance(part, str) else part)
  vma = jax.typeof(x).manual_axis_type.varying
  if missing := tuple(sorted(vma - sharded)):
    x = jax.lax.psum(x, missing)
  if invarying := tuple(sorted(spec.unreduced - vma)):
    mesh = jax.sharding.get_abstract_mesh()
    assert all(mesh.shape[a] == 1 for a in invarying), invarying
    x = jax.lax.pcast(x, invarying, to="varying")
  if spec.unreduced:
    x = jax.lax.pcast(x, tuple(sorted(spec.unreduced)), to="unreduced")
  return x


def _qdot_bwd(
    grad_out: jax.Array,
    x: jax.Array,
    w: jax.Array,
    *,
    quant_rule: quantization.GmmQuantRule,
    dgrad_axes: tuple[Any, Any],
    wgrad_axes: tuple[Any, Any],
    dx_spec: jax.sharding.PartitionSpec,
    dw_spec: jax.sharding.PartitionSpec,
    ste_x: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass of `_qdot(x, w)`, like the fp8 routed expert GMMs.

  On each device, grad_out is quantized to the rule's bwd dtype with one absmax
  scale of its local shard. dx = grad_out_q . w_q is output in bf16 and dw =
  x_q . grad_out_q in the rule's grad_dtype, where x_q and w_q are x and w
  quantized with their static scales.

  Args:
    grad_out: Cotangent of the `_qdot` output.
    x: Activations, unquantized or already quantized with the act scale.
    w: Weights, unquantized or already quantized with the weight scale.
    quant_rule: Quantization rule of the dot.
    dgrad_axes: `jnp.tensordot` axes of `dot(grad_out, w)` giving dx.
    wgrad_axes: `jnp.tensordot` axes of `dot(x, grad_out)` giving dw.
    dx_spec: PartitionSpec of dx, unreduced along any axes its local results are
      partial sums along.
    dw_spec: PartitionSpec of dw, likewise.
    ste_x: Optional unquantized x. If given, the local dx is masked with its
      `_ste` before being reduced, which is exact because the mask does not vary
      along the axes dx is reduced over.

  Returns:
    Tuple of (dx, dw).
  """

  def _local_bwd(grad_out, x_q, w_q, ste_x):
    grad_out_q, s_g = quantization.absmax_quantize(grad_out, quant_rule.bwd_qtype)
    s_g = s_g.reshape(())
    dx = _fp8_dot(grad_out_q, w_q, dgrad_axes) * (s_g * quant_rule.weight_scale)
    if ste_x is not None:
      dx = _ste(dx, ste_x, quant_rule)
    dw = _fp8_dot(x_q, grad_out_q, wgrad_axes) * (s_g * quant_rule.act_scale)
    return (
        _match_spec(dx.astype(jnp.bfloat16), dx_spec),
        _match_spec(dw.astype(quant_rule.grad_dtype), dw_spec),
    )

  return jax.shard_map(
      _local_bwd,
      mesh=jax.typeof(grad_out).sharding.mesh,
      out_specs=(dx_spec, dw_spec),
      check_vma=True,
  )(
      grad_out,
      _quantize_act(x, quant_rule),
      _quantize_weight(w, quant_rule),
      ste_x,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_down_projection(
    x: jt.Num[jax.Array, "B T D"],
    wq_down: jt.Num[jax.Array, "D Cq"],
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    quant_rule: quantization.GmmQuantRule | None = None,
) -> jt.Num[jax.Array, "B T Cq"]:
  """Performs query down projection, in fp8 with `quant_rule`."""
  x_sharding = getattr(jax.typeof(x), "sharding", None)
  assert isinstance(x_sharding, jax.sharding.NamedSharding)
  physical_attention_axis = ops.physical_pspec(jax.sharding.PartitionSpec("attention"), axis_mapping)[0]
  out_spec = jax.sharding.PartitionSpec(x_sharding.spec[0], None, None)

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _q_down_all_gather(x: jax.Array) -> jax.Array:
    return jax.lax.all_gather(
        x,
        axis_name=physical_attention_axis,
        axis=1,
        tiled=True,
        to="invarying",
    )

  def _q_down(x: jax.Array, wq_down: jax.Array) -> jax.Array:
    with jax.named_scope("q_down_projection"):
      if quant_rule is None:
        cq = dot(x, wq_down)
      else:
        cq = _qdot(x, wq_down, quant_rule)
    return _q_down_all_gather(cq)

  return jax.shard_map(
      _q_down,
      mesh=mesh,
      out_specs=out_spec,
      check_vma=True,
  )(x, wq_down)


def _mesh_axis_names(axis: Any) -> frozenset[str]:
  """Returns the mesh axis names of a single PartitionSpec entry as a set."""
  return frozenset((axis,) if isinstance(axis, str) else axis)


def _cotangent_sharding(x: jax.Array, *, unreduced: frozenset[str] = frozenset()) -> jax.sharding.NamedSharding:
  """Returns the sharding that the cotangent of `x` must have.

  The cotangent of a value that is reduced along a mesh axis is unreduced along
  that axis, and vice versa.

  Args:
    x: The primal value.
    unreduced: Additional mesh axes along which the cotangent is left as a
      pending sum.
  """
  sharding = jax.typeof(x).sharding
  assert isinstance(sharding, jax.sharding.NamedSharding)
  spec = sharding.spec.to_ct_spec()
  return sharding.update(spec=spec.update(unreduced=spec.unreduced | unreduced))


@jax.named_call
def down_projection_bwd(
    grad_down: jt.Num[jax.Array, "B T C"],
    x: jt.Num[jax.Array, "B T D"],
    w_down: jt.Num[jax.Array, "D C"],
    *,
    quant_rule: quantization.GmmQuantRule,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "D C"]]:
  """fp8 backward pass for `q_down_projection` / `kv_down_projection`.

  The forward pass all-gathered the projection along the sequence, so each
  device keeps the sequence shard of grad_down matching its x, then runs both
  backward dots on it like `_qdot_bwd`.

  Args:
    grad_down: Cotangent of the down projection output.
    x: Input activations, unquantized or already quantized.
    w_down: Down projection weights, unquantized or already quantized.
    quant_rule: Quantization rule of the projection.

  Returns:
    The gradients with respect to x and w_down.
  """
  x_sharding = jax.typeof(x).sharding
  grad_down = jax.reshard(
      grad_down,
      x_sharding.update(spec=jax.sharding.PartitionSpec(*x_sharding.spec[:2])),
  )
  return _qdot_bwd(
      grad_down,
      x,
      w_down,
      quant_rule=quant_rule,
      dgrad_axes=((2,), (1,)),
      wgrad_axes=((0, 1), (0, 1)),
      dx_spec=_cotangent_sharding(x).spec,
      dw_spec=_cotangent_sharding(w_down).spec,
  )


def _softmax_scale(
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
) -> float:
  """Computes the YaRN-adjusted scale applied to the query."""
  softmax_scale = (qk_head_dim + rope_head_dim) ** -0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m
  return softmax_scale


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_up_from_q_down(
    q_down: jt.Num[jax.Array, "B T Cq"],
    wq_up: jt.Num[jax.Array, "Cq N QK_plus_R"],
    wq_norm_scale: jt.Num[jax.Array, "Cq"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T N QK_plus_R"],
    tuple[
        jt.Num[jax.Array, "B T Cq"],
        jt.Num[jax.Array, "B T N R"],
    ],
]:
  """Computes full query from compressed query, in fp8 with `quant_rule`.

  Returns:
    A tuple of the full query and the residuals required by
    `q_up_from_q_down_bwd`, namely the normalized compressed query and the
    pre-RoPE query.
  """
  softmax_scale = _softmax_scale(
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
  )
  cq = norm_fn(q_down, wq_norm_scale)
  dequant_scale = 1.0
  with jax.named_scope("q_up_projection"):
    if quant_rule is None:
      q = dot(cq, wq_up)
    else:
      # Output in bf16 and dequantized with the softmax scale below, saving a
      # pass over q. Exact for power of 2 dequant scales, like the default.
      q = _fp8_dot(
          _quantize_act(cq, quant_rule),
          _quantize_weight(wq_up, quant_rule),
          1,
          jnp.bfloat16,
      )
      dequant_scale = quant_rule.act_scale * quant_rule.weight_scale
  q_pe = q[..., qk_head_dim:]
  scale = dequant_scale * softmax_scale
  if rope_fn is yarn:
    # Same as the concatenation below, but fused by XLA into one pass over q.
    q = _yarn_tail(q, yarn_freqs, scale)
  else:
    q = jnp.concatenate([q[..., :qk_head_dim], rope_fn(q_pe, yarn_freqs)], axis=-1) * scale
  return q, (cq, q_pe * dequant_scale)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_up_from_q_down_bwd(
    grad_q: jt.Num[jax.Array, "B T N QK_plus_R"],
    q_down: jt.Num[jax.Array, "B T Cq"],
    residuals: tuple[
        jt.Num[jax.Array, "B T Cq"],
        jt.Num[jax.Array, "B T N R"],
    ],
    wq_up: jt.Num[jax.Array, "Cq N QK_plus_R"],
    wq_norm_scale: jt.Num[jax.Array, "Cq"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T Cq"],
    jt.Num[jax.Array, "Cq N QK_plus_R"],
    jt.Num[jax.Array, "Cq"],
    tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
]:
  """Backward pass for `q_up_from_q_down`.

  The `grad_cq` matmul contracts the head dimension, which is sharded along the
  attention axis, so its result is left unreduced and the resulting all-reduce
  is issued explicitly on the device core instead of being scheduled by XLA.

  Args:
    grad_q: Cotangent of the full query.
    q_down: Compressed query, i.e. the primal input.
    residuals: Residuals returned by `q_up_from_q_down`.
    wq_up: Query up projection weights.
    wq_norm_scale: Query normalization scale.
    yarn_freqs: YaRN cos and sin frequencies.
    qk_head_dim: Non-RoPE query/key head dimension.
    rope_head_dim: RoPE head dimension.
    max_position_embeddings: The maximum position indices in the input.
    original_max_position_embeddings: Original maximum position embeddings.
    rope_factor: RoPE factor, used for scaling.
    mscale: YaRN magnitude scale.
    norm_fn: Normalization function.
    rope_fn: RoPE function.
    quant_rule: Optional fp8 quantization of the q up projection; see
      `_qdot_bwd`.

  Returns:
    The gradients with respect to `q_down`, `wq_up`, `wq_norm_scale` and
    `yarn_freqs`.
  """
  cq, q_pe = residuals
  grad_q_sharding = jax.typeof(grad_q).sharding
  assert isinstance(grad_q_sharding, jax.sharding.NamedSharding)
  head_axis = grad_q_sharding.spec[2]

  softmax_scale = _softmax_scale(
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
  )
  grad_q_nope, grad_q_roped = jnp.split(grad_q * softmax_scale, [qk_head_dim], axis=-1)
  _, rope_vjp = jax.vjp(rope_fn, q_pe, yarn_freqs)
  grad_q_pe, grad_yarn_freqs = rope_vjp(grad_q_roped)
  if rope_fn is yarn:
    # Same as the concatenation below, but fused by XLA into one pass.
    grad_q_up = _yarn_tail(grad_q, yarn_freqs, softmax_scale, transpose=True)
  else:
    grad_q_up = jnp.concatenate([grad_q_nope, grad_q_pe], axis=-1)

  unreduced_cq_sharding = _cotangent_sharding(cq, unreduced=_mesh_axis_names(head_axis))
  if quant_rule is None:
    with jax.named_scope("q_up_projection_wgrad"):
      grad_wq_up = dot(
          cq,
          grad_q_up,
          axes=((0, 1), (0, 1)),
          out_sharding=_cotangent_sharding(wq_up),
      )
    with jax.named_scope("q_up_projection_dgrad"):
      grad_cq = dot(
          grad_q_up,
          wq_up,
          axes=((2, 3), (1, 2)),
          out_sharding=unreduced_cq_sharding,
      )
  else:
    with jax.named_scope("q_up_projection_bwd"):
      grad_cq, grad_wq_up = _qdot_bwd(
          grad_q_up,
          cq,
          wq_up,
          quant_rule=quant_rule,
          dgrad_axes=((2, 3), (1, 2)),
          wgrad_axes=((0, 1), (0, 1)),
          dx_spec=unreduced_cq_sharding.spec,
          dw_spec=_cotangent_sharding(wq_up).spec,
      )

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _grad_cq_all_reduce(grad_cq: jax.Array) -> jax.Array:
    return jax.lax.psum(grad_cq, axis_name=head_axis)

  grad_cq = jax.shard_map(
      _grad_cq_all_reduce,
      mesh=grad_q_sharding.mesh,
      out_specs=_cotangent_sharding(cq).spec,
      check_vma=True,
  )(grad_cq)
  if quant_rule is not None:
    grad_cq = _ste(grad_cq, cq, quant_rule)

  _, norm_vjp = jax.vjp(norm_fn, q_down, wq_norm_scale)
  grad_q_down, grad_wq_norm_scale = norm_vjp(grad_cq)
  return grad_q_down, grad_wq_up, grad_wq_norm_scale, grad_yarn_freqs


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_projection(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    wq_down: jt.Num[jax.Array, "D Cq"],
    wq_up: jt.Num[jax.Array, "Cq N QK_plus_R"],
    wq_norm_scale: jt.Num[jax.Array, "Cq"],
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "B T N QK_plus_R"]:
  """Performs query projection."""
  q_down = q_down_projection(
      x,
      wq_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  q, _ = q_up_from_q_down(
      q_down,
      wq_up,
      wq_norm_scale,
      yarn_freqs,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
  )
  return q


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_down_projection(
    x: jt.Num[jax.Array, "B T D"],
    wkv_down: jt.Num[jax.Array, "D Ckv_plus_R"],
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    quant_rule: quantization.GmmQuantRule | None = None,
) -> jt.Num[jax.Array, "B T Ckv_plus_R"]:
  """Performs key/value down projection, in fp8 with `quant_rule`."""
  x_sharding = getattr(jax.typeof(x), "sharding", None)
  assert isinstance(x_sharding, jax.sharding.NamedSharding)
  physical_attention_axis = ops.physical_pspec(jax.sharding.PartitionSpec("attention"), axis_mapping)[0]
  out_spec = jax.sharding.PartitionSpec(x_sharding.spec[0], None, None)

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _kv_down_all_gather(x: jax.Array) -> jax.Array:
    return jax.lax.all_gather(
        x,
        axis_name=physical_attention_axis,
        axis=1,
        tiled=True,
        to="invarying",
    )

  def _kv_down(x: jax.Array, wkv_down: jax.Array) -> jax.Array:
    with jax.named_scope("kv_down_projection"):
      if quant_rule is None:
        ckv_and_k_pe = dot(x, wkv_down)
      else:
        ckv_and_k_pe = _qdot(x, wkv_down, quant_rule)
    return _kv_down_all_gather(ckv_and_k_pe)

  return jax.shard_map(
      _kv_down,
      mesh=mesh,
      out_specs=out_spec,
      check_vma=True,
  )(x, wkv_down)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_up_from_kv_down(
    kv_down: jt.Num[jax.Array, "B T Ckv_plus_R"],
    wk_up: jt.Num[jax.Array, "Ckv N QK"],
    wv_up: jt.Num[jax.Array, "Ckv N V"],
    wkv_norm_scale: jt.Num[jax.Array, "Ckv"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    *,
    kv_lora_rank: int,
    num_query_heads: int,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    tuple[
        jt.Num[jax.Array, "B T N QK_plus_R"],
        jt.Num[jax.Array, "B T N V"],
    ],
    jt.Num[jax.Array, "B T Ckv"],
]:
  """Computes key and value from compressed key/value, in fp8 with `quant_rule`.

  Returns:
    A tuple of the (key, value) pair and the residual required by
    `kv_up_from_kv_down_bwd`, namely the normalized compressed key/value.
  """
  ckv, k_pe = jnp.split(kv_down, [kv_lora_rank], axis=-1)
  ckv = norm_fn(ckv, wkv_norm_scale)
  ckv_q = ckv if quant_rule is None else _quantize_act(ckv, quant_rule)

  def _up(w: jax.Array) -> jax.Array:
    return dot(ckv, w) if quant_rule is None else _qdot(ckv_q, w, quant_rule)

  with jax.named_scope("k_up_projection"):
    k_nope = dot(ckv, wk_up)  # Unquantized; see `quantize_mla_weights`.
  with jax.named_scope("v_up_projection"):
    v = _up(wv_up)

  # Add head dimension of size 1 to k_pe.
  k_pe = jnp.expand_dims(k_pe, axis=2)
  k_pe = rope_fn(k_pe, yarn_freqs)
  sharding = getattr(jax.typeof(k_nope), "sharding", None)
  # Broadcast k_pe to have num_query_heads heads.
  k_pe = jnp.broadcast_to(
      k_pe,
      k_pe.shape[:2] + (num_query_heads, k_pe.shape[-1]),
      out_sharding=sharding,
  )
  return (jnp.concatenate([k_nope, k_pe], axis=-1), v), ckv


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_up_from_kv_down_bwd(
    grad_k: jt.Num[jax.Array, "B T N QK_plus_R"],
    grad_v: jt.Num[jax.Array, "B T N V"],
    kv_down: jt.Num[jax.Array, "B T Ckv_plus_R"],
    ckv: jt.Num[jax.Array, "B T Ckv"],
    wk_up: jt.Num[jax.Array, "Ckv N QK"],
    wv_up: jt.Num[jax.Array, "Ckv N V"],
    wkv_norm_scale: jt.Num[jax.Array, "Ckv"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    *,
    kv_lora_rank: int,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T Ckv_plus_R"],
    jt.Num[jax.Array, "Ckv N QK"],
    jt.Num[jax.Array, "Ckv N V"],
    jt.Num[jax.Array, "Ckv"],
    tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
]:
  """Backward pass for `kv_up_from_kv_down`.

  Both the `grad_ckv` matmuls and the sum over the heads that `k_pe` was
  broadcast across reduce along the attention axis. Their results are left
  unreduced and combined into a single all-reduce that is issued explicitly on
  the device core instead of being scheduled by XLA.

  Args:
    grad_k: Cotangent of the key.
    grad_v: Cotangent of the value.
    kv_down: Compressed key/value, i.e. the primal input.
    ckv: Residual returned by `kv_up_from_kv_down`.
    wk_up: Key up projection weights.
    wv_up: Value up projection weights.
    wkv_norm_scale: Key/value normalization scale.
    yarn_freqs: YaRN cos and sin frequencies.
    kv_lora_rank: Compressed key/value dimension.
    norm_fn: Normalization function.
    rope_fn: RoPE function.
    quant_rule: Optional fp8 quantization of the v up projection; see
      `_qdot_bwd`.

  Returns:
    The gradients with respect to `kv_down`, `wk_up`, `wv_up`,
    `wkv_norm_scale` and `yarn_freqs`.
  """
  grad_k_sharding = jax.typeof(grad_k).sharding
  assert isinstance(grad_k_sharding, jax.sharding.NamedSharding)
  head_axis = grad_k_sharding.spec[2]
  unreduced_sharding = _cotangent_sharding(ckv, unreduced=_mesh_axis_names(head_axis))

  grad_k_nope, grad_k_pe = jnp.split(grad_k, [wk_up.shape[-1]], axis=-1)

  # k up is unquantized; see `quantize_mla_weights`.
  with jax.named_scope("k_up_projection_wgrad"):
    grad_wk_up = dot(
        ckv,
        grad_k_nope,
        axes=((0, 1), (0, 1)),
        out_sharding=_cotangent_sharding(wk_up),
    )
  with jax.named_scope("k_up_projection_dgrad"):
    grad_ckv_k = dot(
        grad_k_nope,
        wk_up,
        axes=((2, 3), (1, 2)),
        out_sharding=unreduced_sharding,
    )
  if quant_rule is None:
    with jax.named_scope("v_up_projection_wgrad"):
      grad_wv_up = dot(
          ckv,
          grad_v,
          axes=((0, 1), (0, 1)),
          out_sharding=_cotangent_sharding(wv_up),
      )
    with jax.named_scope("v_up_projection_dgrad"):
      grad_ckv_v = dot(grad_v, wv_up, axes=((2, 3), (1, 2)), out_sharding=unreduced_sharding)
  else:
    with jax.named_scope("v_up_projection_bwd"):
      grad_ckv_v, grad_wv_up = _qdot_bwd(
          grad_v,
          ckv,
          wv_up,
          quant_rule=quant_rule,
          dgrad_axes=((2, 3), (1, 2)),
          wgrad_axes=((0, 1), (0, 1)),
          dx_spec=unreduced_sharding.spec,
          dw_spec=_cotangent_sharding(wv_up).spec,
          ste_x=ckv,
      )

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _grad_kv_down_all_reduce(grad_ckv_k: jax.Array, grad_ckv_v: jax.Array, grad_k_pe: jax.Array) -> jax.Array:
    # The heads are manual here, so this sum is local and its result is a
    # partial sum that is pending an all-reduce, just like the dgrad matmuls.
    grad_k_pe = jax.lax.pcast(jnp.sum(grad_k_pe, axis=2), head_axis, to="unreduced")
    return jax.lax.psum(
        jnp.concatenate([grad_ckv_k + grad_ckv_v, grad_k_pe], axis=-1),
        axis_name=head_axis,
    )

  grad_ckv, grad_k_pe = jnp.split(
      jax.shard_map(
          _grad_kv_down_all_reduce,
          mesh=grad_k_sharding.mesh,
          out_specs=_cotangent_sharding(kv_down).spec,
          check_vma=True,
      )(grad_ckv_k, grad_ckv_v, grad_k_pe),
      [kv_lora_rank],
      axis=-1,
  )

  _, norm_vjp = jax.vjp(norm_fn, kv_down[..., :kv_lora_rank], wkv_norm_scale)
  grad_ckv, grad_wkv_norm_scale = norm_vjp(grad_ckv)

  k_pe = jnp.expand_dims(kv_down[..., kv_lora_rank:], axis=2)
  _, rope_vjp = jax.vjp(rope_fn, k_pe, yarn_freqs)
  grad_k_pe, grad_yarn_freqs = rope_vjp(jnp.expand_dims(grad_k_pe, axis=2))

  grad_kv_down = jnp.concatenate([grad_ckv, jnp.squeeze(grad_k_pe, axis=2)], axis=-1)
  return (
      grad_kv_down,
      grad_wk_up,
      grad_wv_up,
      grad_wkv_norm_scale,
      grad_yarn_freqs,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_projection(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    wkv_down: jt.Num[jax.Array, "D Ckv_plus_R"],
    wk_up: jt.Num[jax.Array, "Ckv N QK"],
    wv_up: jt.Num[jax.Array, "Ckv N V"],
    wkv_norm_scale: jt.Num[jax.Array, "Ckv"],
    *,
    kv_lora_rank: int,
    num_query_heads: int,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N V"],
]:
  """Performs key/value projection."""
  kv_down = kv_down_projection(
      x,
      wkv_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  kv, _ = kv_up_from_kv_down(
      kv_down,
      wk_up,
      wv_up,
      wkv_norm_scale,
      yarn_freqs,
      kv_lora_rank=kv_lora_rank,
      num_query_heads=num_query_heads,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
  )
  return kv


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def out_projection(
    splash_out: jt.Num[jax.Array, "B T N V"],
    w_out: jt.Num[jax.Array, "N V D"],
    quant_rule: quantization.GmmQuantRule | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Performs output projection.

  Args:
    splash_out: Splash attention output activations.
    w_out: Output projection weights.
    quant_rule: Optional fp8 quantization of the projection.

  Returns:
    The output activations.
  """
  attn_sharding = jax.typeof(splash_out).sharding
  assert isinstance(attn_sharding, jax.sharding.NamedSharding)
  head_axis = attn_sharding.spec[2]
  out_spec = jax.sharding.PartitionSpec(attn_sharding.spec[0], head_axis, None)

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _out_reduce_scatter(x: jax.Array) -> jax.Array:
    return jax.lax.psum_scatter(
        x,
        axis_name=head_axis,
        scatter_dimension=1,
        tiled=True,
    )

  def _out(splash_out: jax.Array, w_out: jax.Array) -> jax.Array:
    with jax.named_scope("out_projection"):
      if quant_rule is None:
        out = dot(splash_out, w_out, axes=2)
      else:
        out = _qdot(splash_out, w_out, quant_rule, axes=2)
    return _out_reduce_scatter(out)

  return jax.shard_map(
      _out,
      mesh=attn_sharding.mesh,
      out_specs=out_spec,
      check_vma=True,
  )(splash_out, w_out)


@jax.named_call
def out_projection_bwd(
    grad_out: jt.Num[jax.Array, "B T D"],
    splash_out: jt.Num[jax.Array, "B T N V"],
    w_out: jt.Num[jax.Array, "N V D"],
    *,
    quant_rule: quantization.GmmQuantRule,
) -> tuple[jt.Num[jax.Array, "B T N V"], jt.Num[jax.Array, "N V D"]]:
  """fp8 backward pass for `out_projection`.

  grad_out is all-gathered along the sequence in bf16, undoing the forward
  reduce-scatter, and then quantized on each device by `_qdot_bwd`.

  Args:
    grad_out: Cotangent of the output projection.
    splash_out: Splash attention output activations.
    w_out: Output projection weights, unquantized or already quantized.
    quant_rule: Quantization rule of the projection.

  Returns:
    The gradients with respect to splash_out and w_out.
  """
  attn_sharding = jax.typeof(splash_out).sharding
  assert isinstance(attn_sharding, jax.sharding.NamedSharding)
  head_axis = attn_sharding.spec[2]

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _grad_out_all_gather(x: jax.Array) -> jax.Array:
    return jax.lax.all_gather(x, axis_name=head_axis, axis=1, tiled=True, to="invarying")

  grad_out = jax.shard_map(
      _grad_out_all_gather,
      mesh=attn_sharding.mesh,
      out_specs=jax.sharding.PartitionSpec(attn_sharding.spec[0], None, None),
      check_vma=True,
  )(
      jax.reshard(
          grad_out,
          attn_sharding.update(spec=jax.sharding.PartitionSpec(attn_sharding.spec[0], head_axis, None)),
      )
  )
  grad_splash_out, grad_w_out = _qdot_bwd(
      grad_out,
      splash_out,
      w_out,
      quant_rule=quant_rule,
      dgrad_axes=((2,), (2,)),
      wgrad_axes=((0, 1), (0, 1)),
      dx_spec=_cotangent_sharding(splash_out).spec,
      dw_spec=_cotangent_sharding(w_out).spec,
  )
  return _ste(grad_splash_out, splash_out, quant_rule), grad_w_out


@jt.jaxtyped(typechecker=typeguard.typechecked)
def rotate_half(
    x: jt.Num[jax.Array, "... D"],
) -> jt.Num[jax.Array, "... D"]:
  """Rotates half the hidden dimensions of the input.

  Args:
    x: Input array with shape [..., embedding_dim].

  Returns:
    The array with half dimensions rotated, i.e.
    [x1, x2] -> [-x2, x1].
  """
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([-x2, x1], axis=-1)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def yarn(
    x: jt.Num[jax.Array, "B T N D"],
    freqs: tuple[
        jt.Num[jax.Array, "B T 1 D"],
        jt.Num[jax.Array, "B T 1 D"],
    ],
) -> jt.Num[jax.Array, "B T N D"]:
  """Performs YaRN rotary embedding."""
  cos, sin = freqs
  return x * cos + rotate_half(x) * sin


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _yarn_tail(
    x: jt.Num[jax.Array, "B T N D"],
    freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    scale: float,
    *,
    transpose: bool = False,
) -> jt.Num[jax.Array, "B T N D"]:
  """Applies `yarn`, or its transpose, to the last R dims of x, and scales it.

  Same as `jnp.concatenate([x[..., :-R], yarn(x[..., -R:], freqs)], -1) *
  scale`, but written as a sum of padded terms with the scale folded into the
  frequencies, which XLA fuses into a single pass over x instead of
  materializing the concatenated slices or the scaled x.

  Args:
    x: Input whose last R dims are rotated, e.g. the query.
    freqs: YaRN cos and sin frequencies.
    scale: Scale of the result, e.g. the softmax scale.
    transpose: Whether to apply the transposed rotation, e.g. to a cotangent.

  Returns:
    x with its last R dims rotated, scaled.
  """
  cos, sin = (f * scale for f in freqs)
  d, r = x.shape[-1], cos.shape[-1]
  h = r // 2
  x1, x2 = x[..., d - r : d - h], x[..., d - h :]
  # With sin = [sin1, sin2], yarn([x1, x2]) is [x1, x2] * cos + [-x2 * sin1,
  # x1 * sin2], and its transpose is [x1, x2] * cos + [x2 * sin2, -x1 * sin1].
  if transpose:
    rot1, rot2 = x2 * sin[..., h:], -x1 * sin[..., :h]
  else:
    rot1, rot2 = -x2 * sin[..., :h], x1 * sin[..., h:]

  def pad(y, lo, hi, value=0):
    pads = [(0, 0)] * (y.ndim - 1) + [(lo, hi)]
    return jnp.pad(y, pads, constant_values=value)

  return x * pad(cos, d - r, 0, scale) + pad(rot1, d - r, h) + pad(rot2, d - h, 0)


def dot(x, y, axes=1, out_sharding=None):
  """Helper function for jnp.tensordot with default axes=1."""
  return jnp.tensordot(x, y, axes=axes, out_sharding=out_sharding)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def init_splash_kernel(
    sa_block_q: int,
    sa_block_kv: int,
    sa_block_kv_compute: int,
    sa_block_q_dkv: int,
    sa_block_kv_dkv: int,
    sa_block_kv_dkv_compute: int,
    sa_q_layout: Any,
    sa_k_layout: Any,
    sa_v_layout: Any,
    max_target_length: int,
    num_query_heads: int,
    kernel_out_spec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    qk_diag_skip: bool = True,
    qk_diag_grid: int = 8,
    sv_diag_skip: bool = True,
) -> Any:
  """Initializes the Splash kernel with the given block sizes."""
  del num_query_heads
  physical_kernel_out_spec = ops.physical_pspec(kernel_out_spec, axis_mapping)
  kernel_out_sharding = jax.sharding.NamedSharding(mesh, physical_kernel_out_spec)
  seq_spec_dim = physical_kernel_out_spec[1]

  def _get_num_shards(spec_dim: Any, mesh: jax.sharding.AbstractMesh | jax.sharding.Mesh) -> int:
    if spec_dim is None:
      return 1
    if isinstance(spec_dim, str):
      return mesh.shape[spec_dim]
    return math.prod(mesh.shape[name] for name in spec_dim)

  q_seq_shards = _get_num_shards(seq_spec_dim, mesh)
  if q_seq_shards > 1:
    raise NotImplementedError(
        "Sequence sharding is not supported for" " tokamax_splash_attention_kernel. Only head sharding is supported."
    )

  q_seq_len_per_shard = max_target_length // q_seq_shards

  block_q = min(sa_block_q, q_seq_len_per_shard)
  block_kv = min(sa_block_kv, q_seq_len_per_shard)
  block_kv_compute = min(sa_block_kv_compute, q_seq_len_per_shard)
  effective_qk_diag_skip = qk_diag_skip and (block_q == block_kv == block_kv_compute)
  effective_sv_diag_skip = sv_diag_skip and (block_q == block_kv == block_kv_compute)
  sa_config = tokamax_splash_attention_kernel.SplashConfig(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=min(sa_block_q_dkv, q_seq_len_per_shard),
      block_kv_dkv=min(sa_block_kv_dkv, q_seq_len_per_shard),
      block_kv_dkv_compute=min(sa_block_kv_dkv_compute, q_seq_len_per_shard),
      block_q_dq=None,
      block_kv_dq=None,
      use_fused_bwd_kernel=True,
      q_layout=tokamax_splash_attention_kernel.QKVLayout[sa_q_layout] if isinstance(sa_q_layout, str) else sa_q_layout,
      k_layout=tokamax_splash_attention_kernel.QKVLayout[sa_k_layout] if isinstance(sa_k_layout, str) else sa_k_layout,
      v_layout=tokamax_splash_attention_kernel.QKVLayout[sa_v_layout] if isinstance(sa_v_layout, str) else sa_v_layout,
      qk_diag_skip=effective_qk_diag_skip,
      qk_diag_grid=qk_diag_grid,
      sv_diag_skip=effective_sv_diag_skip,
  )
  mask = tokamax_splash_mask.CausalMask(shape=(max_target_length, max_target_length))
  splash_kernel = tokamax_splash_attention_kernel.make_splash_mha(
      mask=mask,
      q_seq_shards=q_seq_shards,
      config=sa_config,
  )
  kernel_pspec = splash_kernel.manual_sharding_spec(kernel_out_sharding)

  def _reshard_leaf(arr: Any, spec: Any) -> Any:
    if arr is None:
      return None
    sharding = jax.sharding.NamedSharding(mesh, ops.physical_pspec(spec, axis_mapping))
    return jax.reshard(arr, sharding)

  splash_kernel = jax.tree.map(
      _reshard_leaf,
      splash_kernel,
      kernel_pspec,
      is_leaf=lambda x: x is None,
  )
  return splash_kernel


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def splash_attention(
    q: jt.Num[jax.Array, "B T N QK_plus_R"],
    k: jt.Num[jax.Array, "B T N QK_plus_R"],
    v: jt.Num[jax.Array, "B T N V"],
    splash_kernel: Any,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T N V"],
    jt.Num[jax.Array, "B N T"],
]:
  """Performs splash attention returning output and context."""
  query = jnp.transpose(q, axes=(0, 2, 1, 3))
  key = jnp.transpose(k, axes=(0, 2, 1, 3))
  value = jnp.transpose(v, axes=(0, 2, 1, 3))

  mesh = jax.typeof(query).sharding.mesh
  query_spec = jax.typeof(query).sharding.spec
  context_spec = jax.sharding.PartitionSpec(*query_spec[:3])

  kernel_pspec = jax.tree.map(
      lambda arr: None if arr is None else jax.typeof(arr).sharding.spec,
      splash_kernel,
      is_leaf=lambda x: x is None,
  )

  if segment_ids is not None:
    segment_spec = jax.sharding.PartitionSpec(query_spec[0], None)
    segment_ids_tuple = tokamax_splash_attention_kernel.SegmentIds(q=segment_ids, kv=segment_ids)
    segment_ids_in_spec = tokamax_splash_attention_kernel.SegmentIds(
        q=segment_spec,  # pyrefly: ignore[bad-argument-type]
        kv=segment_spec,  # pyrefly: ignore[bad-argument-type]
    )
  else:
    segment_ids_tuple = None
    segment_ids_in_spec = None

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          query_spec,
          query_spec,
          query_spec,
          kernel_pspec,
          segment_ids_in_spec,
      ),
      out_specs=(
          query_spec,
          context_spec,
      ),
      check_vma=False,
  )
  def wrap_splash_attention(query, key, value, splash_kernel, segment_ids):
    attention_output, context = jax.vmap(
        splash_kernel.manual_fwd,
        in_axes=(0, 0, 0, 0 if segment_ids is not None else None),
        out_axes=(0, 0),
    )(query, key, value, segment_ids)
    return attention_output, context

  attention_output, context = wrap_splash_attention(
      query,
      key,
      value,
      splash_kernel,
      segment_ids_tuple,
  )
  splash_out = jnp.transpose(attention_output, axes=(0, 2, 1, 3))
  return splash_out, context


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def splash_attention_bwd(
    grad_splash_out: jt.Num[jax.Array, "B T N V"],
    splash_out: jt.Num[jax.Array, "B T N V"],
    context: jt.Num[jax.Array, "B N T"],
    q: jt.Num[jax.Array, "B T N QK_plus_R"],
    k: jt.Num[jax.Array, "B T N QK_plus_R"],
    v: jt.Num[jax.Array, "B T N V"],
    splash_kernel: Any,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N V"],
]:
  """Performs backward splash attention."""
  do = jnp.transpose(grad_splash_out, axes=(0, 2, 1, 3))
  o = jnp.transpose(splash_out, axes=(0, 2, 1, 3))
  query = jnp.transpose(q, axes=(0, 2, 1, 3))
  key = jnp.transpose(k, axes=(0, 2, 1, 3))
  value = jnp.transpose(v, axes=(0, 2, 1, 3))

  mesh = jax.typeof(query).sharding.mesh
  query_spec = jax.typeof(query).sharding.spec
  context_spec = jax.sharding.PartitionSpec(*query_spec[:3])

  kernel_pspec = jax.tree.map(
      lambda arr: None if arr is None else jax.typeof(arr).sharding.spec,
      splash_kernel,
      is_leaf=lambda x: x is None,
  )

  if segment_ids is not None:
    segment_spec = jax.sharding.PartitionSpec(query_spec[0], None)
    segment_ids_tuple = tokamax_splash_attention_kernel.SegmentIds(q=segment_ids, kv=segment_ids)
    segment_ids_in_spec = tokamax_splash_attention_kernel.SegmentIds(
        q=segment_spec,  # pyrefly: ignore[bad-argument-type]
        kv=segment_spec,  # pyrefly: ignore[bad-argument-type]
    )
  else:
    segment_ids_tuple = None
    segment_ids_in_spec = None

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          query_spec,
          query_spec,
          context_spec,
          query_spec,
          query_spec,
          query_spec,
          kernel_pspec,
          segment_ids_in_spec,
      ),
      out_specs=(
          query_spec,
          query_spec,
          query_spec,
      ),
      check_vma=False,
  )
  def wrap_splash_attention_bwd(do, o, context, query, key, value, splash_kernel, segment_ids):
    dq, dk, dv = jax.vmap(
        splash_kernel.manual_bwd,
        in_axes=(0, 0, 0, 0, 0, 0, 0 if segment_ids is not None else None),
        out_axes=(0, 0, 0),
    )(query, key, value, o, context, do, segment_ids)
    return dq, dk, dv

  dq, dk, dv = wrap_splash_attention_bwd(
      do,
      o,
      context,
      query,
      key,
      value,
      splash_kernel,
      segment_ids_tuple,
  )

  dq = jnp.transpose(dq, axes=(0, 2, 1, 3))
  dk = jnp.transpose(dk, axes=(0, 2, 1, 3))
  dv = jnp.transpose(dv, axes=(0, 2, 1, 3))
  return dq, dk, dv


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mla_fwd(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    w: dsv3_types.DSv3MLAWeightsPytree,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    tuple[
        jt.Num[jax.Array, "B T Cq"],
        jt.Num[jax.Array, "B T Ckv_plus_R"],
        jt.Num[jax.Array, "B N T"],
        jt.Num[jax.Array, "B T N V"],
    ],
]:
  """Forward pass for MLA returning output activations and residuals.

  With `quant_rule`, all projections but k up run in fp8 (see
  `quantize_mla_weights`); weights not already quantized (e.g. before
  collection) are quantized here.
  """
  if quant_rule is not None:
    w = quantize_mla_weights(w, quant_rule)
    x = _quantize_act(x, quant_rule)
  assert w.q_down is not None
  assert w.q_up is not None
  assert w.q_norm_scale is not None
  assert w.kv_down is not None
  assert w.k_up is not None
  assert w.v_up is not None
  assert w.kv_norm_scale is not None
  assert w.out is not None
  q_down = q_down_projection(
      x,
      w.q_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
      quant_rule=quant_rule,
  )
  q, _ = q_up_from_q_down(
      q_down,
      w.q_up,
      w.q_norm_scale,
      yarn_freqs,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      quant_rule=quant_rule,
  )
  kv_down = kv_down_projection(
      x,
      w.kv_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
      quant_rule=quant_rule,
  )
  (k, v), _ = kv_up_from_kv_down(
      kv_down,
      w.k_up,
      w.v_up,
      w.kv_norm_scale,
      yarn_freqs,
      kv_lora_rank=kv_lora_rank,
      num_query_heads=num_query_heads,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      quant_rule=quant_rule,
  )
  splash_out, context = splash_attention(
      q,
      k,
      v,
      splash_kernel,
      segment_ids=segment_ids,
  )
  out = out_projection(
      splash_out,
      w.out,
      quant_rule=quant_rule,
  )
  residuals = (q_down, kv_down, context, splash_out)
  return out, residuals


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mla_bwd(
    grad_out: jt.Num[jax.Array, "B T D"],
    norm_x: jt.Num[jax.Array, "B T D"],
    q_down: jt.Num[jax.Array, "B T Cq"],
    kv_down: jt.Num[jax.Array, "B T Ckv_plus_R"],
    context: jt.Num[jax.Array, "B N T"],
    splash_out: jt.Num[jax.Array, "B T N V"],
    w: dsv3_types.DSv3MLAWeightsPytree,
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    dsv3_types.DSv3MLAWeightsPytree,
    tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
]:
  """Backward pass for MLA using saved residuals and rematerialization.

  With `quant_rule`, all projections but k up are backpropagated through like
  the fp8 routed expert GMMs; see `_qdot_bwd`.
  """
  x_q = norm_x
  if quant_rule is not None:
    w = quantize_mla_weights(w, quant_rule)
    x_q = _quantize_act(norm_x, quant_rule)
  assert w.q_down is not None
  assert w.q_up is not None
  assert w.q_norm_scale is not None
  assert w.kv_down is not None
  assert w.k_up is not None
  assert w.v_up is not None
  assert w.kv_norm_scale is not None
  assert w.out is not None

  # 1. Backprop through out_projection
  if quant_rule is None:

    def _out_proj_fwd(splash_out, w_out):
      return out_projection(
          splash_out,
          w_out,
      )

    _, out_proj_vjp = jax.vjp(_out_proj_fwd, splash_out, w.out)
    grad_splash_out, grad_w_out = out_proj_vjp(grad_out)
  else:
    grad_splash_out, grad_w_out = out_projection_bwd(grad_out, splash_out, w.out, quant_rule=quant_rule)

  # 2. Remat q from q_down
  q, q_residuals = q_up_from_q_down(
      q_down,
      w.q_up,
      w.q_norm_scale,
      yarn_freqs,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      quant_rule=quant_rule,
  )

  # 3. Remat k, v from kv_down
  (k, v), ckv = kv_up_from_kv_down(
      kv_down,
      w.k_up,
      w.v_up,
      w.kv_norm_scale,
      yarn_freqs,
      kv_lora_rank=kv_lora_rank,
      num_query_heads=num_query_heads,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      quant_rule=quant_rule,
  )

  # 4. Splash attention backward
  dq, dk, dv = splash_attention_bwd(
      grad_splash_out,
      splash_out,
      context,
      q,
      k,
      v,
      splash_kernel,
      segment_ids=segment_ids,
  )

  # 5. Backprop through Q projection
  grad_q_down, grad_w_q_up, grad_w_q_norm_scale, grad_yarn_freqs_q = q_up_from_q_down_bwd(
      dq,
      q_down,
      q_residuals,
      w.q_up,
      w.q_norm_scale,
      yarn_freqs,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      quant_rule=quant_rule,
  )

  if quant_rule is None:

    def _q_down_fwd(norm_x, wq_down):
      return q_down_projection(
          norm_x,
          wq_down,
          mesh=mesh,
          axis_mapping=axis_mapping,
      )

    _, vjp_q_down = jax.vjp(_q_down_fwd, norm_x, w.q_down)
    grad_norm_x_q, grad_w_q_down = vjp_q_down(grad_q_down)
  else:
    grad_norm_x_q, grad_w_q_down = down_projection_bwd(grad_q_down, x_q, w.q_down, quant_rule=quant_rule)

  # 6. Backprop through KV projection
  (
      grad_kv_down,
      grad_w_k_up,
      grad_w_v_up,
      grad_w_kv_norm_scale,
      grad_yarn_freqs_kv,
  ) = kv_up_from_kv_down_bwd(
      dk,
      dv,
      kv_down,
      ckv,
      w.k_up,
      w.v_up,
      w.kv_norm_scale,
      yarn_freqs,
      kv_lora_rank=kv_lora_rank,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      quant_rule=quant_rule,
  )

  if quant_rule is None:

    def _kv_down_fwd(norm_x, wkv_down):
      return kv_down_projection(
          norm_x,
          wkv_down,
          mesh=mesh,
          axis_mapping=axis_mapping,
      )

    _, vjp_kv_down = jax.vjp(_kv_down_fwd, norm_x, w.kv_down)
    grad_norm_x_kv, grad_w_kv_down = vjp_kv_down(grad_kv_down)
  else:
    grad_norm_x_kv, grad_w_kv_down = down_projection_bwd(grad_kv_down, x_q, w.kv_down, quant_rule=quant_rule)

  # 7. Accumulate gradients
  grad_norm_x = grad_norm_x_q + grad_norm_x_kv
  if quant_rule is not None:
    grad_norm_x = _ste(grad_norm_x, norm_x, quant_rule)
  grad_yarn_freqs = jax.tree.map(lambda g1, g2: g1 + g2, grad_yarn_freqs_q, grad_yarn_freqs_kv)
  grad_w = dsv3_types.DSv3MLAWeightsPytree(
      q_down=grad_w_q_down,
      q_up=grad_w_q_up,
      q_norm_scale=grad_w_q_norm_scale,
      kv_down=grad_w_kv_down,
      k_up=grad_w_k_up,
      v_up=grad_w_v_up,
      kv_norm_scale=grad_w_kv_norm_scale,
      out=grad_w_out,
  )
  return grad_norm_x, grad_w, grad_yarn_freqs


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(5, 18)))
def _dsv3_mla_vjp(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    w: dsv3_types.DSv3MLAWeightsPytree,
    segment_ids: jt.Num[jax.Array, "B T"] | None,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    quant_rule: quantization.GmmQuantRule | None,
) -> jt.Num[jax.Array, "B T D"]:
  """Custom VJP wrapper for MLA."""
  return _dsv3_mla_custom_fwd(
      x,
      yarn_freqs,
      splash_kernel,
      w,
      segment_ids,
      kv_lora_rank,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mscale,
      norm_fn,
      rope_fn,
      mesh,
      axis_mapping,
      quant_rule,
  )[0]


def _dsv3_mla_custom_fwd(
    x,
    yarn_freqs,
    splash_kernel,
    w,
    segment_ids,
    kv_lora_rank,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    norm_fn,
    rope_fn,
    mesh,
    axis_mapping,
    quant_rule,
):
  out, residuals = dsv3_mla_fwd(
      x,
      yarn_freqs,
      splash_kernel,
      w,
      kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      mesh=mesh,
      axis_mapping=axis_mapping,
      segment_ids=segment_ids,
      quant_rule=quant_rule,
  )
  q_down, kv_down, context, splash_out = residuals
  res = (
      x,
      w,
      yarn_freqs,
      splash_kernel,
      segment_ids,
      q_down,
      kv_down,
      context,
      splash_out,
  )
  return out, res


def _dsv3_mla_custom_bwd(
    kv_lora_rank,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    norm_fn,
    rope_fn,
    mesh,
    axis_mapping,
    quant_rule,
    res,
    grad_out,
):
  (
      x,
      w,
      yarn_freqs,
      splash_kernel,
      segment_ids,
      q_down,
      kv_down,
      context,
      splash_out,
  ) = res
  grad_x, grad_w, grad_yarn_freqs = dsv3_mla_bwd(
      grad_out,
      x,
      q_down,
      kv_down,
      context,
      splash_out,
      w,
      yarn_freqs,
      splash_kernel,
      kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      mesh=mesh,
      axis_mapping=axis_mapping,
      segment_ids=segment_ids,
      quant_rule=quant_rule,
  )
  return grad_x, grad_yarn_freqs, None, grad_w, None


_dsv3_mla_vjp.defvjp(_dsv3_mla_custom_fwd, _dsv3_mla_custom_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mla(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    w: dsv3_types.DSv3MLAWeightsPytree,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    quant_rule: quantization.GmmQuantRule | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Performs MLA for DSv3, in fp8 with `quant_rule`."""
  return _dsv3_mla_vjp(
      x,
      yarn_freqs,
      splash_kernel,
      w,
      segment_ids,
      kv_lora_rank,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mscale,
      norm_fn,
      rope_fn,
      mesh,
      axis_mapping,
      quant_rule,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def get_yarn_freqs(
    positions: jt.Num[jax.Array, "B T"],
    rope_head_dim: int,
    rope_theta: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    beta_fast: int,
    beta_slow: int,
    rope_factor: int,
    out_pspec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    dtype: jt.DTypeLike = jnp.bfloat16,
) -> tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]]:
  """Initializes YaRN cos and sin frequencies.

  These frequencies are used to rotate pairs of dimensions in the query and key
  vectors. Base frequencies are first computed for each pair of dimensions and
  then piecewise scaled and smoothed across the positions. Up to
  max_position_embeddings frequencies are precomputed and then looked up using
  the input positions, with cos and sin precomputed in the specified dtype.

  Args:
    positions: Position indices corresponding to the positions of the input
      tokens.
    rope_head_dim: RoPE dimension per head.
    rope_theta: RoPE theta, used to determine base frequencies.
    max_position_embeddings: The maximum position indices in the input.
    original_max_position_embeddings: Original maximum position embeddings, used
      for scaling.
    beta_fast: Beta fast, used for scaling.
    beta_slow: Beta slow, used for scaling.
    rope_factor: RoPE factor, used for scaling.
    out_pspec: Logical PartitionSpec for output sharding of frequencies.
    mesh: Physical mesh.
    axis_mapping: Mapping from logical to physical axes.
    dtype: Data type for cos and sin frequencies. Defaults to bfloat16.

  Returns:
    A tuple of (cos, sin) YaRN frequencies each with shape [B, T, 1, R] and the
    specified dtype.
  """
  out_sharding = jax.sharding.NamedSharding(mesh, ops.physical_pspec(out_pspec, axis_mapping))
  half_dim = rope_head_dim // 2
  # Compute base frequencies for each (even-indexed) dimension.
  # (Note: We use jnp.arange with float32 for precision.)
  freqs = 1.0 / (rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / rope_head_dim))

  low = (
      rope_head_dim * math.log(original_max_position_embeddings / (beta_fast * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  high = (
      rope_head_dim * math.log(original_max_position_embeddings / (beta_slow * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  low = max(math.floor(low), 0)
  high = min(math.ceil(high), rope_head_dim - 1)
  diff = high - low if high > low else 0.001
  linear_func = (jnp.arange(half_dim, dtype=jnp.float32) - low) / diff
  smooth = 1 - jnp.clip(linear_func, 0, 1)
  # The corrected frequency is a weighted mix of the scaled and base values.
  freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

  # Precompute frequencies for all positions by taking the outer product.
  t = jnp.arange(max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
  # [max_position_embeddings, half_dim] tensor with rows as time steps.
  freqs = jnp.outer(t, freqs)
  # Lookup the precomputed frequencies using the position indices.
  # freqs has shape [max_position_embeddings, half_dim].
  # After indexing, shape becomes [B, T, half_dim]; we then add the head axis.
  freqs = freqs.at[positions].get(out_sharding=out_sharding)
  freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, T, 1, half_dim]
  freqs = jnp.concatenate([freqs, freqs], axis=-1)  # shape: [B, T, 1, rope_head_dim]
  cos = jnp.cos(freqs).astype(dtype)
  sin = jnp.sin(freqs).astype(dtype)
  return cos, sin
