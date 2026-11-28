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

"""DeepSeekV3 scan over multiple dense layers."""

from collections.abc import Mapping
import functools
from typing import Any, Protocol

import jax
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_dense_layer
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage import quantization


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


@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dense_layers(
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
    quant: quantization.QuantConfig | None = None,
    remat_layers: bool = False,
    compute_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Executes multiple DSv3 dense layers.

  Args:
    x: Input tokens.
    w: DSv3 dense layer weights, stacked for all layers on the first dimension.
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
    quant: Optional quantization config.

  Returns:
    Output tokens with the same shape and sharding as the input tokens.
  """

  def _collect_w_dcn_layer(w_layer):
    if compute_dtype is not None:
      w_layer = jax.tree.map(
          lambda a: a.astype(compute_dtype) if a is not None else None,
          w_layer,
          is_leaf=lambda a: a is None,
      )
    return dsv3_dense_layer.collect_w_dcn(w_layer, axis_mapping=axis_mapping)

  _collect_w_ici = functools.partial(
      dsv3_dense_layer.collect_w_ici,
      axis_mapping=axis_mapping,
  )
  _reduce_w_ici = functools.partial(
      dsv3_dense_layer.reduce_w_ici,
      axis_mapping=axis_mapping,
  )

  _dsv3_dense_layer = functools.partial(
      dsv3_dense_layer.dsv3_dense_layer,
      yarn_freqs=yarn_freqs,
      splash_kernel=splash_kernel,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      mscale=mscale,
      kv_lora_rank=kv_lora_rank,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mesh=mesh,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      axis_mapping=axis_mapping,
      segment_ids=segment_ids,
      quant=quant,
  )

  if remat_layers:

    @jax.checkpoint
    def _remat_step(carry_x, w_layer_sharded):
      w_layer = _collect_w_ici(_collect_w_dcn_layer(w_layer_sharded))
      return _dsv3_dense_layer(carry_x, w_layer), None

    x, _ = jax.lax.scan(_remat_step, x, w)
    return x

  def _scan_fn(x, w):
    return _dsv3_dense_layer(x, w), ()

  x, _ = ops.w_collect_pipelined_scan(_scan_fn, x, w, _collect_w_dcn_layer, _collect_w_ici, _reduce_w_ici)
  return x
