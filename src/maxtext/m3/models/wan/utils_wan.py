# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan model utilities: sharding helpers, activations, hardware, precision, GCS and profiler helpers."""

from collections.abc import Sequence
import dataclasses
from enum import Enum
import json
import os
from typing import Any

import jax
from jax.sharding import PartitionSpec as P
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.core.sharding import TensorType, create_mesh
from maxtext.utils import max_logging
import yaml

# Tokamax's Splash attention kernel (TPU flash attention), under either of its two module layouts.
try:
  from tokamax._src.ops.splash_attention import splash_attention_kernel as tokamax_splash
except ImportError:
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash

# Splash block sizes and kernel flags (`flash_block_sizes` in the yml), as Tokamax's `SplashConfig`.
BlockSizes = tokamax_splash.SplashConfig

# Default sequence-parallel mesh axis.
CONTEXT = "context"

# ==============================================================================
# Sharding: `WanSharding` serves the transformer, the VAE and the UMT5 text encoder. Wan's models refer only to logical
# axes and take a `maxtext.m3.core.sharding.Sharding`; each is built from its yml rule table (`logical_axis_rules` /
# `vae_logical_axis_rules` / `text_encoder_logical_axis_rules` in wan2.1-t2v-14b.yml, overridable on the command line;
# checked against Flax's resolution of the same tables by tests/unit/m3/wan/wan_sharding_test.py).
#
# `sharding=None` (the models' default) means "no sharding at all": parameters carry no `out_sharding` metadata and
# activation constraints are skipped. It exists for mesh-free uses only -- checkpoint conversion and abstract-shape
# inspection (`nnx.eval_shape`) -- and the drivers always pass a real `Sharding`.
# ==============================================================================

# Logical axis names of the transformer's attention q / kv ([batch, heads, length, head_dim], as the kernels see them).
SELF_ATTN_HEAD = "activation_self_attn_heads"
SELF_ATTN_Q_LENGTH = "activation_self_attn_q_length"
SELF_ATTN_KV_LENGTH = "activation_self_attn_kv_length"
CROSS_ATTN_HEAD = "activation_cross_attn_heads"
CROSS_ATTN_Q_LENGTH = "activation_cross_attn_q_length"
CROSS_ATTN_KV_LENGTH = "activation_cross_attn_kv_length"


# Mesh axes of the transformer / text-encoder mesh (`mesh_axes` in wan2.1-t2v-14b.yml), in mesh order, and of the VAE
# mesh (a reshape of the same devices).
WAN_MESH_AXES = ("data", "fsdp", "context", "tensor")
VAE_MESH_AXES = ("redundant", "vae_spatial")


@dataclasses.dataclass(frozen=True)
class WanSharding(m3_sharding.Sharding):
  """A `Sharding` that looks logical axes up in a yml rule table, over the mesh axes `mesh_axes`.

  One class serves the transformer, the VAE and the UMT5 text encoder; only the table and the mesh differ::

      WanSharding.from_rules(config.logical_axis_rules, WAN_MESH_AXES)               # transformer
      WanSharding.from_rules(config.vae_logical_axis_rules, VAE_MESH_AXES)           # VAE
      WanSharding.from_rules(config.text_encoder_logical_axis_rules, WAN_MESH_AXES)  # UMT5 text encoder

  `rules` is `((logical_axis, mesh_axes), ...)`, with `mesh_axes` None (replicated), one mesh axis or a tuple of them
  (their product shards the dimension). Stricter than Flax's rule tables so a typo fails loudly instead of silently
  replicating: every logical axis a model uses must be listed (`KeyError` otherwise, at model construction), a logical
  axis may be listed only once (no fallback), and two dims of one tensor claiming the same mesh axis raise.
  `context_axis` is the sequence-parallel mesh axis the attention kernels shard_map / ring over.

  Overrides `Sharding.__call__` only because core validates rule outputs against `MESH_AXIS_NAMES`, which has no
  `context` / `redundant` / `vae_spatial`. Frozen dataclass: the models keep their `sharding` as a static attribute,
  so it must be hashable and compare by value (equal tables give the same graphdef).
  """

  rules: tuple[tuple[str, m3_sharding.MeshAxes], ...] = ()
  mesh_axes: tuple[str, ...] = WAN_MESH_AXES
  context_axis: str = CONTEXT

  @classmethod
  def from_rules(cls, rules: Sequence[Sequence[Any]], mesh_axes: Sequence[str] = WAN_MESH_AXES, **kwargs):
    """Builds the strategy from a yml rule table: `[[logical_axis, mesh_axis | [mesh_axis, ...] | null], ...]`."""
    normalized = []
    for logical_axis, mapped in rules:
      if mapped is not None and not isinstance(mapped, str):
        mapped = tuple(mapped)
      normalized.append((logical_axis, mapped))
    names = [name for name, _ in normalized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
      raise ValueError(f"{cls.__name__}: logical axes listed more than once in the rule table: {duplicates}.")
    return cls(rules=tuple(normalized), mesh_axes=tuple(mesh_axes), **kwargs)

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> m3_sharding.MeshAxes:
    for logical_axis, mapped in self.rules:
      if logical_axis == axis:
        return mapped
    raise KeyError(
        f"{type(self).__name__}: no rule for logical axis {axis!r} ({tensor_type.value}) on tensor {tensor_name!r}; "
        "add it to the yml rule table."
    )

  def __call__(
      self,
      tensor_name: str,
      axes: Sequence[str | None],
      tensor_type: TensorType = TensorType.WEIGHT,
  ) -> P:
    resolved = []
    claimed: dict[str, str] = {}
    for axis in axes:
      if axis is None:
        resolved.append(None)
        continue
      mapped = self.map_axis(axis, tensor_name, tensor_type)
      names = () if mapped is None else ((mapped,) if isinstance(mapped, str) else tuple(mapped))
      unknown = [name for name in names if name not in self.mesh_axes]
      if unknown:
        raise ValueError(
            f"Sharding rule for logical axis {axis!r} of tensor {tensor_name!r} returned unknown mesh axes "
            f"{unknown}; expected names from {self.mesh_axes}."
        )
      for mesh_axis in names:
        if mesh_axis in claimed:
          raise ValueError(
              f"Tensor {tensor_name!r} maps both {claimed[mesh_axis]!r} and {axis!r} onto mesh axis "
              f"{mesh_axis!r}; a mesh axis can shard only one dimension of a tensor."
          )
        claimed[mesh_axis] = axis
      resolved.append(mapped)
    return P(*resolved)


def param_metadata(sharding: m3_sharding.Sharding | None, tensor_name: str, axes: Sequence[str | None]) -> dict:
  """`{"out_sharding": spec}` for a parameter's `*_metadata=`, or `{}` when `sharding` is None (mesh-free use)."""
  return {} if sharding is None else {"out_sharding": sharding(tensor_name, axes)}


def constrain(sharding: m3_sharding.Sharding | None, x: Any, tensor_name: str, axes: Sequence[str | None]) -> Any:
  """`sharding.constrain(x, ...)`, or `x` unchanged when `sharding` is None (mesh-free use)."""
  return x if sharding is None else sharding.constrain(x, tensor_name, axes)


# The bundled Wan 2.1 config, the home of the default rule tables.
WAN21_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "wan2.1-t2v-14b.yml")


def yml_rules(key: str, path: str = WAN21_CONFIG_PATH) -> tuple:
  """The `key` rule table (e.g. `"logical_axis_rules"`) of the yml at `path`, for `<Strategy>.from_rules`.

  The drivers pass `config.<key>` (which honours command-line overrides); this reads the file directly, for tests and
  tools that build models without a full config.
  """
  with open(path, "rt", encoding="utf-8") as f:
    return tuple(yaml.safe_load(f)[key])


_ACTIVATIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "gelu_tanh": jax.nn.gelu,
    "mish": jax.nn.mish,
}


def get_activation(name: str):
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


# ==============================================================================
# Wan Model Hardware, Mesh, GCS, Profiler, and Configuration Utilities
# ==============================================================================


# --- Hardware & TPU Generation Detection ---
class TpuType(Enum):
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6_LITE = "v6e"
  TPU_7X = "v7x"
  UNKNOWN = "unknown"


def get_tpu_type() -> TpuType:
  """Detects the current TPU hardware generation."""
  try:
    device_kind = jax.devices()[0].device_kind
    if "7x" in device_kind:
      return TpuType.TPU_7X
    elif "v6 lite" in device_kind:
      return TpuType.TPU_V6_LITE
    else:
      return TpuType.UNKNOWN
  except Exception:  # pylint: disable=broad-exception-caught
    # Any failure to query devices means "unknown TPU type".
    return TpuType.UNKNOWN


def safe_getattr(obj, name, default=None):
  try:
    return getattr(obj, name)
  except (AttributeError, KeyError):
    return default


# --- Mesh & Precision Utilities ---
def get_precision(config):
  if not hasattr(config, "precision") or not config.precision:
    return None
  return jax.lax.Precision[config.precision]


def get_flash_block_sizes(config):
  """Build Tokamax Splash `BlockSizes` (`SplashConfig`) from config.flash_block_sizes (dict or JSON string).

  Tokamax's Splash kernel only has the fused backward kernel (dq computed inside the dkv kernel), so the yml's
  `block_q_dq` / `block_kv_dq` / `use_fused_bwd_kernel` keys are accepted but ignored.
  """
  if not hasattr(config, "flash_block_sizes") or not config.flash_block_sizes:
    return None
  if isinstance(config.flash_block_sizes, str):
    flash_block_sizes = json.loads(config.flash_block_sizes)
  else:
    flash_block_sizes = config.flash_block_sizes
  if flash_block_sizes:
    return BlockSizes(
        block_q=flash_block_sizes["block_q"],
        block_kv_compute=flash_block_sizes["block_kv_compute"],
        block_kv=flash_block_sizes["block_kv"],
        block_q_dkv=flash_block_sizes["block_q_dkv"],
        block_kv_dkv=flash_block_sizes["block_kv_dkv"],
        block_kv_dkv_compute=flash_block_sizes["block_kv_dkv_compute"],
        use_fused_bwd_kernel=True,
    )
  return None


def device_put_replicated(x, sharding):
  """Although the name indicates replication, this function can be used

  to also shard an array based on sharding.
  """
  from jax.sharding import NamedSharding  # pylint: disable=import-outside-toplevel

  arr = getattr(x, "value", x)
  shd = getattr(sharding, "value", sharding)
  if hasattr(shd, "devices") and not hasattr(shd, "device_set"):
    shd = NamedSharding(shd, P())
  res = jax.make_array_from_callback(arr.shape, shd, lambda index: arr[index])
  if hasattr(x, "set_value"):
    x.set_value(res)
    return x
  return res


def create_wan_mesh(config) -> jax.sharding.Mesh:
  """The device mesh over `config.mesh_axes`, sized by `config.{ici,dcn}_<axis>_parallelism` (`create_mesh`)."""
  if getattr(config, "allow_split_physical_axes", False):
    raise NotImplementedError("allow_split_physical_axes=True is not supported by m3's create_mesh.")
  ici = {axis: getattr(config, f"ici_{axis}_parallelism", 1) for axis in config.mesh_axes}
  dcn = {axis: getattr(config, f"dcn_{axis}_parallelism", 1) for axis in config.mesh_axes}
  return create_mesh(axis_names=tuple(config.mesh_axes), ici=ici, dcn=dcn)


# --- File, GCS, and Prompt Utilities ---
def delete_file(file_path: str):
  """Removes a local file, e.g. after it has been uploaded to GCS."""
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def get_gcs_output_path(config):
  """Return the gs:// output path for this run, or None if output_dir is not on GCS."""
  output_dir = getattr(config, "output_dir", "")
  if not output_dir or not output_dir.startswith("gs://"):
    return None
  run_name = getattr(config, "run_name", "")
  return os.path.join(output_dir, run_name) if run_name else output_dir


def load_prompts(prompt_file_path: str, default_prompt: str = "") -> list:
  """Load non-empty, non-comment prompts from a local file or http(s) URL."""
  if not prompt_file_path:
    if not default_prompt:
      raise ValueError("Neither prompt_file nor prompt was specified.")
    return [default_prompt]
  if prompt_file_path.startswith("http://") or prompt_file_path.startswith("https://"):
    import requests  # pylint: disable=import-outside-toplevel

    response = requests.get(prompt_file_path, timeout=60)
    response.raise_for_status()
    raw_lines = response.text.splitlines()
  else:
    if not os.path.isfile(prompt_file_path):
      raise FileNotFoundError(f"Prompt file not found at local path: {prompt_file_path}")
    with open(prompt_file_path, "r", encoding="utf-8") as f:
      raw_lines = f.readlines()
  prompts = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
  if not prompts:
    if default_prompt:
      return [default_prompt]
    raise ValueError(f"Prompt file '{prompt_file_path}' contains no valid non-empty prompts.")
  return prompts


def chunk_and_pad(items: list, batch_size: int):
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}")
  for i in range(0, len(items), batch_size):
    chunk = items[i : i + batch_size]
    actual_len = len(chunk)
    padded_chunk = chunk + [chunk[-1]] * (batch_size - actual_len) if actual_len < batch_size else chunk
    yield i, padded_chunk, actual_len
