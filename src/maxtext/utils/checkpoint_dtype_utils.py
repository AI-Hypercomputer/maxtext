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
import jax.numpy as jnp

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

# Per-model MaxText key suffixes kept in float32: tensors the HF release itself stores in F32.
# Kimi-K3: the KDA decay/conv/norm parameters (`self_attn.{A_log,dt_bias,{q,k,v}_conv1d.weight,
# o_norm.weight}`, F32 in the moonshotai/Kimi-K3 safetensors); bf16 would shift exp(A_log).
FULL_PRECISION_SUFFIXES_BY_MODEL: dict[str, tuple[str, ...]] = {
    "kimi-k3": (
        "-self_attention-A_log",
        "-self_attention-dt_bias",
        "-self_attention-q_conv1d-kernel",
        "-self_attention-k_conv1d-kernel",
        "-self_attention-v_conv1d-kernel",
        "-self_attention-o_norm-scale",
    ),
    # `MuseSparkPreTrainedModel._keep_in_fp32_modules_strict` keeps router kernels/biases and residual gates in fp32.
    "muse-spark-816b-a42b": (
        "-moe_block-gate-kernel",
        "-moe_block-gate-bias",
        "-post_attention_residual_gate-gate",
        "-post_feedforward_residual_gate-gate",
    ),
}


def is_full_precision_param_key(mt_param_key: str, model_name: str | None = None) -> bool:
  """Returns True when a `-`-delimited MaxText parameter key must remain float32."""
  if mt_param_key.startswith(FULL_PRECISION_PREFIXES) or any(sub in mt_param_key for sub in FULL_PRECISION_SUBSTRINGS):
    return True
  suffixes = FULL_PRECISION_SUFFIXES_BY_MODEL.get(model_name, ()) if model_name is not None else ()
  return bool(suffixes and mt_param_key.endswith(suffixes))


def param_save_dtype(mt_param_key_or_keys: str | tuple[str, ...], save_dtype: Any, model_name: str | None = None) -> Any:
  """Returns `'float32'` for full-precision router/HC/sink/indexer or per-model F32 state when saving `'bfloat16'`."""
  raw_dtype = getattr(save_dtype, "value", save_dtype)
  try:
    is_bf16 = raw_dtype == "bfloat16" or jnp.dtype(raw_dtype) == jnp.bfloat16
  except TypeError:
    is_bf16 = False
  if not is_bf16:
    return save_dtype
  keys = mt_param_key_or_keys if isinstance(mt_param_key_or_keys, tuple) else (mt_param_key_or_keys,)
  if any(is_full_precision_param_key(k, model_name=model_name) for k in keys):
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


def should_keep_full_precision_nnx(path: Iterable[Any], var: Any = None, model_name: str | None = None) -> bool:
  """Returns True if an NNX `(path, var)` leaf must stay float32 when saving bfloat16."""
  return is_full_precision_param_key(format_nnx_path_as_param_key(path, var), model_name=model_name)
