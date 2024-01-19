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

BATCH = 'activation_batch'
LENGTH = 'activation_length'
HEAD = 'activation_heads'
D_KV = 'activation_kv'

MODEL_MODE_AUTOREGRESSIVE = 'autoregressive'
MODEL_MODE_PREFILL = 'prefill'
MODEL_MODE_TRAIN = 'train'

DECODING_ACTIVE_SEQUENCE_INDICATOR = 1
