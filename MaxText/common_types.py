#  Copyright 2023 Google LLC
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

"""Common types."""

from typing import Any, Sequence

from flax.linen import partitioning
import jax
import jax.numpy as jnp

Config = Any

Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = Sequence[int]

Mesh = jax.sharding.Mesh
ScanIn = partitioning.ScanIn

AxisNames = tuple[str, ...]
AxisIdxes = tuple[int, ...]

BATCH = "activation_batch"
LENGTH = "activation_length"
HEAD = "activation_heads"
KV_BATCH = "activation_kv_batch"
KV_HEAD = "activation_kv_heads"
KV_HEAD_DIM = "activation_kv_head_dim"
D_KV = "activation_kv"
CACHE_BATCH = "cache_batch"
CACHE_SEQUENCE = "cache_sequence"
CACHE_HEADS = "cache_heads"
CACHE_KV = "cache_kv"
CACHE_SCALE_BATCH = "cache_scale_batch"
CACHE_SCALE_SEQUENCE = "cache_scale_sequence"
CACHE_SCALE_HEADS = "cache_scale_heads"
CACHE_SCALE_KV = "cache_scale_kv"

MODEL_MODE_AUTOREGRESSIVE = "autoregressive"
MODEL_MODE_PREFILL = "prefill"
MODEL_MODE_TRAIN = "train"

DECODING_ACTIVE_SEQUENCE_INDICATOR = 1
