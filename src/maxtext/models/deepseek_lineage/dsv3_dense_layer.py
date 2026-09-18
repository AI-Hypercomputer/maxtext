# Copyright 2023–2026 Google LLC
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

"""Single DeepSeekV3 dense layer."""

from collections.abc import Mapping
import functools
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import jaxtyping as jt
from maxtext.models.deepseek_lineage import dsv3_mla
from maxtext.models.deepseek_lineage import dsv3_types
import typeguard


class NormFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*input_dims"],
      scale: jt.Num[jax.Array, "..."] | None = None,
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


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mlp(
    x: jt.Num[jax.Array, "*BT D"],
    w: dsv3_types.DSv3MLPWeightsPytree,
) -> jt.Num[jax.Array, "*BT D"]:
  """Performs computation for an MLP.

  Args:
    x: Input tokens with any number of leading dimensions.
    w: MLP weights.

  Returns:
    Output tokens with the same shape as the input.
  """
  dot = functools.partial(jnp.tensordot, axes=1)
  with jax.named_scope("gate_0"):
    g0 = dot(x, w.gate_0)
  with jax.named_scope("gate_1"):
    g1 = dot(x, w.gate_1)
  with jax.named_scope("silu"):
    act = jax.nn.silu(g0) * g1
  with jax.named_scope("linear"):
    return dot(act, w.linear)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dense_layer(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3DenseLayerWeightsPytree,
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    mscale: float,
    kv_lora_rank: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mesh: jax.sharding.Mesh,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Executes a single DSv3 dense layer.

  Args:
    x: Input tokens.
    w: DSv3 weights.
    yarn_freqs: YaRN frequencies.
    splash_kernel: Initialized Splash kernel.
    qk_head_dim: Non-positional key/query dimension per head.
    rope_head_dim: RoPE dimension per head.
    num_query_heads: Number of query heads.
    mscale: mscale, used for scaling.
    kv_lora_rank: LoRA rank of the key/value projection.
    max_position_embeddings: Maximum position embeddings, used for scaling.
    original_max_position_embeddings: Original maximum position embeddings, used
      for scaling.
    rope_factor: RoPE factor, used for scaling.
    mesh: JAX mesh over which the data and model are sharded.
    norm_fn: Normalization function.
    rope_fn: RoPE function.
    axis_mapping: Mapping from logical to physical mesh axes.
    segment_ids: Optional segment IDs for sequence packing.

  Returns:
    Output tokens with the same shape and sharding as the input tokens.
  """
  orig_x = x
  with jax.named_scope("pre_attn_norm"):
    x = norm_fn(x, w.pre_attn_norm_scale)
  x = dsv3_mla.dsv3_mla(
      x,
      yarn_freqs,
      splash_kernel,
      w.mla,
      segment_ids=segment_ids,
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
  )
  attn_out = x + orig_x
  with jax.named_scope("post_attn_norm"):
    x = norm_fn(attn_out, w.post_attn_norm_scale)
  mlp_out = dsv3_mlp(x, w.mlp)
  return attn_out + mlp_out
