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

NORM = "norm"
VOCAB = "vocab"
MLP = "mlp"
LAYERS = "layers"

EMBED = "embed"
HEADS = "heads"
KV_HEADS = "kv_heads"
KV = "kv"
QKV = "qkv"

ACTIVATION_EMBED = "activation_embed"
ACTIVATION_EMBED_AND_LOGITS_BATCH = "activation_embed_and_logits_batch"
ACTIVATION_BATCH = "activation_batch"
ACTIVATION_LENGTH_NO_HEADS = "activation_length_no_heads"
ACTIVATION_LENGTH = "activation_length"
ACTIVATION_HEADS = "activation_heads"
ACTIVATION_KV_HEADS = "activation_kv_heads"
ACTIVATION_KV = "activation_kv"
ACTIVATION_MLP = "activation_mlp"
ACTIVATION_VOCAB = "activation_vocab"

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
