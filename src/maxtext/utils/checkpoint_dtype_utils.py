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

"""Canonical full-precision (float32) parameter retention rules for bfloat16 checkpoints."""

from typing import Any, Iterable

from flax import nnx
from flax.nnx import variablelib

# Router state bf16 cannot hold exactly: the fp32 routing bias and integer
# token-to-expert ids (bf16 rounds integers above 256).
FULL_PRECISION_PREFIXES: tuple[str, ...] = ("MoEBiasVar-", "Tid2EidVar-")

FULL_PRECISION_SUBSTRINGS: tuple[str, ...] = (
    "-attn_hc-",
    "-ffn_hc-",
    "-hc_head-",
    "-self_attention-sinks",
    "-self_attention-indexer-k_norm-",
    "-self_attention-indexer-weights_proj-",
)


def is_full_precision_param_key(mt_param_key: str) -> bool:
  """Returns True when a `-`-delimited MaxText parameter key must remain float32."""
  return mt_param_key.startswith(FULL_PRECISION_PREFIXES) or any(sub in mt_param_key for sub in FULL_PRECISION_SUBSTRINGS)


def param_save_dtype(mt_param_key_or_keys: str | tuple[str, ...], save_dtype: Any) -> Any:
  """Returns `'float32'` for full-precision router/HC/sink/indexer state when saving `'bfloat16'`."""
  if getattr(save_dtype, "value", save_dtype) != "bfloat16":
    return save_dtype
  keys = mt_param_key_or_keys if isinstance(mt_param_key_or_keys, tuple) else (mt_param_key_or_keys,)
  if any(is_full_precision_param_key(k) for k in keys):
    return "float32"
  return save_dtype


def format_nnx_path_as_param_key(path: Iterable[Any], var: Any = None) -> str:
  """Formats an NNX flat-state `(path, var)` pair into a `-`-delimited MaxText parameter key."""
  collection = (
      variablelib.variable_name_from_type(type(var), allow_register=True) if isinstance(var, nnx.Variable) else "params"
  )
  parts: list[str] = []
  for entry in path:
    seg = getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry)))
    is_index = (isinstance(seg, int) and not isinstance(seg, bool)) or (isinstance(seg, str) and seg.isdigit())
    if is_index and parts:
      parts[-1] = f"{parts[-1]}_{seg}"
    else:
      parts.append(str(seg))
  if parts and parts[-1] == "value":
    parts.pop()
  return f"{collection}-{'-'.join(parts)}" if parts else collection


def should_keep_full_precision_nnx(path: Iterable[Any], var: Any = None) -> bool:
  """Returns True if an NNX `(path, var)` leaf must stay float32 when saving bfloat16."""
  return is_full_precision_param_key(format_nnx_path_as_param_key(path, var))
