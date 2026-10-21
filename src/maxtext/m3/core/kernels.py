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

"""Fused-kernel adapters: the one place m3 picks and shards custom kernels.

A model calls two arithmetic entry points, `core.attention.dot_product_attention`
and `core.moe.routed_experts`. Both compute in plain `jnp` unless the kernel
object they are handed says otherwise, in which case the arithmetic is
delegated here. Models therefore contain no kernel names, no `shard_map`, no
block sizes and no platform checks -- `m3/README.md` rule 4. What a model does
contain is one attribute, resolved once at construction::

    self.kernel = kernels.attention_from_config(config, sharding)
    ...
    out = dot_product_attention(q, k, v, positions=..., segment_ids=..., kernel=self.kernel)

The resolved object carries the backend name and the `PartitionSpec`s the
kernel needs, looked up from the model's `Sharding` with the shared logical
vocabulary (`"batch"`, `"heads"`, `"kv_heads"`, `"length"`). It is a frozen
dataclass, so Flax treats it as static module state: it survives
`nnx.split`/`merge`, `jit` and the scan transform, and two layers with the
same kernel choice have equal graph definitions.

Why Pallas kernels need the specs at all: Mosaic kernels cannot be partitioned
by XLA, so each one runs per device inside a `jax.shard_map`, and `shard_map`
wants explicit `in_specs`. The specs are activation rules, so a different
`Sharding` strategy moves the kernels with it; the mesh itself is the one in
scope (`jax.set_mesh`), read the same way `Sharding.constrain` reads it.

Why the specs come from `Sharding` and not from the inputs: the prototype
read `jax.typeof(query).sharding` at call time. That only reports the real
spec under a mesh whose axes are `Explicit`; on m3's mesh (`Auto` axes,
`core.sharding.create_mesh`) a traced input sharded `P("fsdp", None)` reports
`P(None, None)` inside `jit`, with or without a `with_sharding_constraint`
before it, and the kernel would be given replicated specs without any error.
Checked on a v6e-8 with jax 0.11.1. If m3 moves to explicit sharding, the
lookup from inputs becomes an option again.

Backends, all from upstream packages (JAX's Pallas ops and `tokamax`), none
from legacy MaxText's forked copies under `src/maxtext/kernels/`:

  * attention: `"xla"` is the reference in `core/attention.py`; `"splash"` is
    JAX's TPU splash attention; `"tokamax"` is `tokamax.dot_product_attention`,
    which picks its own implementation per platform (Mosaic on TPU).
  * grouped matmul (MoE): `"ragged_dot"` is `jax.lax.ragged_dot`, an XLA op
    that shards without help; `"megablox"` is JAX's Pallas `gmm`; `"tokamax"`
    is `tokamax.ragged_dot`.

Legacy config names are translated in the two `*_from_config` functions and
nowhere else.
"""

import dataclasses
import functools
import math
from typing import Literal

import jax
from jax.experimental.pallas.ops.tpu import megablox
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.m3.core.sharding import MeshAxes, Sharding, TensorType

AttentionBackend = Literal["xla", "splash", "tokamax"]
GroupedMatmulBackend = Literal["ragged_dot", "megablox", "tokamax"]


def mesh_axis_names(axes: MeshAxes) -> tuple[str, ...]:
  """Normalizes one `PartitionSpec` entry to a tuple of mesh axis names."""
  if axes is None:
    return ()
  return (axes,) if isinstance(axes, str) else tuple(axes)


def mesh_axis_size(mesh: jax.sharding.AbstractMesh, axes: MeshAxes) -> int:
  """Number of shards a `PartitionSpec` entry produces on `mesh`."""
  return math.prod(mesh.shape[name] for name in mesh_axis_names(axes))


def mesh_in_scope() -> jax.sharding.AbstractMesh:
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    raise ValueError("Fused kernels run under jax.shard_map and need a mesh in scope; call inside `jax.set_mesh(mesh)`.")
  return mesh


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class SplashBlocks:
  """Splash attention tile sizes, in legacy's `sa_block_*` vocabulary.

  Each is clamped to the sequence length at call time, so one setting serves
  short test sequences and long training ones.
  """

  block_q: int = 512
  block_kv: int = 512
  block_kv_compute: int = 512
  block_q_dkv: int = 512
  block_kv_dkv: int = 512
  block_kv_dkv_compute: int = 512
  block_q_dq: int = 512
  block_kv_dq: int = 512
  use_fused_bwd_kernel: bool = False

  @classmethod
  def from_config(cls, config) -> "SplashBlocks":
    return cls(
        block_q=config.sa_block_q,
        block_kv=config.sa_block_kv,
        block_kv_compute=config.sa_block_kv_compute,
        block_q_dkv=config.sa_block_q_dkv,
        block_kv_dkv=config.sa_block_kv_dkv,
        block_kv_dkv_compute=config.sa_block_kv_dkv_compute,
        block_q_dq=config.sa_block_q_dq,
        block_kv_dq=config.sa_block_kv_dq,
        use_fused_bwd_kernel=config.sa_use_fused_bwd_kernel,
    )

  def resolve(self, q_length: int, kv_length: int) -> splash_attention_kernel.BlockSizes:
    """Clamps to the actual lengths; the fused backward kernel takes no dq blocks."""
    return splash_attention_kernel.BlockSizes(
        block_q=min(self.block_q, q_length),
        block_kv=min(self.block_kv, kv_length),
        block_kv_compute=min(self.block_kv_compute, kv_length),
        block_q_dkv=min(self.block_q_dkv, q_length),
        block_kv_dkv=min(self.block_kv_dkv, kv_length),
        block_kv_dkv_compute=min(self.block_kv_dkv_compute, kv_length),
        block_q_dq=None if self.use_fused_bwd_kernel else min(self.block_q_dq, q_length),
        block_kv_dq=None if self.use_fused_bwd_kernel else min(self.block_kv_dq, kv_length),
        use_fused_bwd_kernel=self.use_fused_bwd_kernel,
    )


@dataclasses.dataclass(frozen=True)
class AttentionKernel:
  """A resolved attention backend and the mesh axes its inputs are sharded over.

  `batch`, `heads`, `kv_heads` and `length` are the mesh axes the model's
  `Sharding` assigns to those logical activation axes; `head_dim` is never
  sharded. The specs a kernel needs are assembled from them in whichever
  layout that kernel wants (splash takes `(batch, heads, length, head_dim)`,
  tokamax `(batch, length, heads, head_dim)`).
  """

  backend: AttentionBackend = "xla"
  batch: MeshAxes = None
  heads: MeshAxes = None
  kv_heads: MeshAxes = None
  length: MeshAxes = None
  blocks: SplashBlocks = SplashBlocks()


XLA_ATTENTION = AttentionKernel("xla")


def attention_from_config(config, sharding: Sharding) -> AttentionKernel:
  """Translates legacy's `attention` / `use_tokamax_splash` / `sa_block_*` fields.

  `dot_product` is the reference; `flash` is splash, or tokamax's splash under
  `use_tokamax_splash`; `autoselected` is `flash` on TPU and the reference
  elsewhere. Legacy also drops `autoselected` to the reference below 128
  tokens at call time; m3 does not, so a short-sequence test that wants the
  reference must say `attention=dot_product`. Attention variants other than
  full causal (`global`, `mla`) are refused: sliding windows and chunking are
  mask shapes these kernels can take but no m3 model asks for yet.
  """
  if config.attention_type not in ("global", "mla"):
    raise ValueError(
        f"m3 attention kernels implement full causal attention; got attention_type={config.attention_type!r}."
    )
  match config.attention:
    case "dot_product":
      backend = "xla"
    case "flash":
      backend = "tokamax" if config.use_tokamax_splash else "splash"
    case "autoselected":
      on_tpu = jax.default_backend() == "tpu"
      backend = ("tokamax" if config.use_tokamax_splash else "splash") if on_tpu else "xla"
    case other:
      raise ValueError(f"m3 supports attention in ('dot_product', 'flash', 'autoselected'); got {other!r}.")
  if backend == "xla":
    return XLA_ATTENTION
  q = sharding("attention_kernel_q", ("batch", "heads", "length", "head_dim"), TensorType.ACTIVATION)
  kv = sharding("attention_kernel_kv", ("batch", "kv_heads", "length", "head_dim"), TensorType.ACTIVATION)
  return AttentionKernel(
      backend=backend, batch=q[0], heads=q[1], kv_heads=kv[1], length=q[2], blocks=SplashBlocks.from_config(config)
  )


@functools.partial(jax.jit, static_argnames=("mask", "head_shards", "block_sizes"))
def _splash_kernel(mask, head_shards: int, block_sizes: splash_attention_kernel.BlockSizes):
  """Builds a splash kernel; `jit` with static arguments is the cache.

  `make_splash_mha` does its block-mask bookkeeping in numpy and then turns
  the tables into `jnp` arrays. Under `jit` that Python work runs once per
  distinct `(mask, block_sizes)`, while every trace that calls this still
  gets arrays of its own rather than another trace's tracers -- the same
  arrangement legacy's `wrap_jax_splash_kernel` uses.
  """
  return splash_attention_kernel.make_splash_mha(
      mask,
      head_shards=head_shards,
      q_seq_shards=1,
      block_sizes=block_sizes,
      # Lets a remat policy that saves `"context"` keep the kernel's residuals,
      # as legacy's `minimal_with_context` policy does.
      residual_checkpoint_name="context",
  )


def _splash_attention(query, key, value, *, segment_ids, kernel: AttentionKernel):
  """JAX's TPU splash attention, per device under `shard_map`, `vmap`ped over the local batch."""
  mesh = mesh_in_scope()
  # Splash wants (heads, length, head_dim) per batch element.
  query, key, value = (jnp.swapaxes(x, 1, 2) for x in (query, key, value))
  _, num_heads, q_length, _ = query.shape
  kv_length = key.shape[2]
  mask = splash_attention_mask.MultiHeadMask((splash_attention_mask.CausalMask((q_length, kv_length)),) * num_heads)
  splash = _splash_kernel(mask, mesh_axis_size(mesh, kernel.heads), kernel.blocks.resolve(q_length, kv_length))

  q_spec = P(kernel.batch, kernel.heads, kernel.length, None)
  kv_spec = P(kernel.batch, kernel.kv_heads, kernel.length, None)
  segment_spec = P(kernel.batch, kernel.length)
  # The kernel object is a pytree of block-mask tables laid out (heads, q blocks).
  splash_spec = splash.manual_sharding_spec(NamedSharding(mesh, P(kernel.heads, kernel.length)))

  def per_device(q, k, v, segments, splash):
    if segments is None:
      return jax.vmap(splash)(q, k, v)
    return jax.vmap(lambda q, k, v, s: splash(q, k, v, splash_attention_kernel.SegmentIds(s, s)))(q, k, v, segments)

  sharded = jax.shard_map(
      per_device,
      mesh=mesh,
      in_specs=(q_spec, kv_spec, kv_spec, None if segment_ids is None else segment_spec, splash_spec),
      out_specs=q_spec,
      check_vma=False,
  )
  return jnp.swapaxes(sharded(query, key, value, segment_ids, splash), 1, 2)


def _tokamax_attention(query, key, value, *, segment_ids, kernel: AttentionKernel):
  """`tokamax.dot_product_attention`, which shards itself from `q_sharding`.

  Packed segments become a dense boolean mask, the only form the public API
  takes; its footprint is `batch x length^2` bytes, which is why the mask is
  omitted when there are no segments.
  """
  import tokamax  # pylint: disable=import-outside-toplevel

  mesh = mesh_in_scope()
  mask = None
  if segment_ids is not None:
    mask = (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None, :, :]
  return tokamax.dot_product_attention(
      query,
      key,
      value,
      mask=mask,
      is_causal=True,
      # m3 models scale the query themselves (`head_dim**-0.5`, or MLA's mscale).
      scale=1.0,
      q_sharding=NamedSharding(mesh, P(kernel.batch, kernel.length, kernel.heads, None)),
  )


def fused_attention(query, key, value, *, segment_ids, kernel: AttentionKernel) -> jax.Array:
  """Causal attention through the kernel `kernel` names; layouts as in `core.attention.dot_product_attention`.

  The query must already be scaled. Logits are accumulated in float32 inside
  every backend, so there is no `float32_logits` switch here.
  """
  match kernel.backend:
    case "splash":
      return _splash_attention(query, key, value, segment_ids=segment_ids, kernel=kernel)
    case "tokamax":
      return _tokamax_attention(query, key, value, segment_ids=segment_ids, kernel=kernel)
    case other:
      raise ValueError(f"No fused attention backend {other!r}; 'xla' is handled by core.attention.")


# --------------------------------------------------------------------------- #
# Grouped matmul (routed experts)
# --------------------------------------------------------------------------- #


def _tile(size: int, candidates: tuple[int, ...], what: str) -> int:
  """Largest candidate tile that divides `size`; megablox requires exact tiling."""
  for tile in candidates:
    if size % tile == 0:
      return tile
  raise ValueError(f"megablox needs the {what} dimension ({size}) to be a multiple of {candidates[-1]}.")


def _megablox_tiling(m: int, k: int, n: int) -> tuple[int, int, int]:
  """Legacy's default tiles (512, 1024, 1024), stepped down to divide the problem."""
  return (
      _tile(m, (512, 256, 128, 64, 32, 16, 8), "token-row"),
      _tile(k, (1024, 512, 256, 128), "contracting"),
      _tile(n, (1024, 512, 256, 128), "output"),
  )


def grouped_matmul(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, backend: GroupedMatmulBackend) -> jax.Array:
  """`out[i] = lhs[i] @ rhs[g(i)]` where rows are grouped contiguously by expert; float32 accumulation.

  Args:
    lhs: `(rows, k)`, rows sorted by expert.
    rhs: `(num_experts, k, n)`.
    group_sizes: `(num_experts,)` int32 row counts, summing to `rows`.
    backend: See the module docstring. The Pallas backends must be called
      per device, i.e. inside a `shard_map`; `core.moe` does that.
  """
  match backend:
    case "ragged_dot":
      return jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32)
    case "megablox":
      return megablox.gmm(
          lhs,
          rhs,
          group_sizes,
          preferred_element_type=jnp.float32,
          tiling=_megablox_tiling(lhs.shape[0], lhs.shape[1], rhs.shape[2]),
          # Mosaic lowers only for TPU; elsewhere the kernel runs in the Pallas interpreter.
          interpret=jax.default_backend() != "tpu",
      )
    case "tokamax":
      import tokamax  # pylint: disable=import-outside-toplevel

      return tokamax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32)
    case other:
      raise ValueError(f"Unknown grouped matmul backend {other!r}; expected 'ragged_dot', 'megablox' or 'tokamax'.")
