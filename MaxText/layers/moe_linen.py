#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""MoE related Linen Layers."""

import enum
import functools
import math
from typing import Iterable, Optional, Tuple, Union

from aqt.jax.v2 import aqt_tensor as aqt
import flax.linen as nn
from flax import nnx
import jax
from jax import ad_checkpoint as adc
from jax.experimental import shard_map
from jax.experimental import xla_metadata
import jax.numpy as jnp
from MaxText import common_types as ctypes
from MaxText import max_logging
from MaxText import max_utils
from MaxText.kernels import megablox as mblx
from MaxText.layers import moe
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import quantizations
import numpy as np
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import NdInitializer, nd_dense_init, default_bias_init, variable_to_logically_partitioned


def gate_logit_module(
    inputs_shape: tuple[int, ...],
    out_features_shape: Union[Iterable[int], int],
    model_name: str,
    axis: Union[Iterable[int], int] = -1,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: Tuple[Optional[str], ...] = (),
    use_bias: bool = False,
    score_func: str = "",
    quant: Optional[quantizations.AqtQuantization] = None,
    matmul_precision: str = "default",
    name: Optional[str] = None,
):
  """Creates a GateLogit Linen module."""

  axis = linears._canonicalize_tuple(axis)
  in_features_shape = tuple(inputs_shape[ax] for ax in linears._normalize_axes(axis, len(inputs_shape)))

  module = nnx_wrappers.to_linen(
      moe.GateLogit,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      model_name=model_name,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      use_bias=use_bias,
      score_func=score_func,
      quant=quant,
      matmul_precision=matmul_precision,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module