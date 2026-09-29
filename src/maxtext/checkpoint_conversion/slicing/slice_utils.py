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

"""Shared MaxText / JAX / checkpoint plumbing for the slicing tools.

How to ask existing MaxText infrastructure to compile, inspect, or save a model.
Architecture and reduction policy live in `reducer.py`, not here.
"""

from __future__ import annotations

import enum
import math
import os
from typing import Any

import jax
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from maxtext.common import train_state_nnx
from maxtext.trainers.pre_train import train
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding

GiB = 1024**3

# ----------------------------------------------------------------------------- config / math helpers


def get_config_value(config: Any, name: str, default: Any = 0) -> Any:
  """`config.<name>`, with `default` for missing or None fields."""
  value = getattr(config, name, default)
  return default if value is None else value


def decoder_block(config: Any) -> str:
  block = get_config_value(config, "decoder_block", "default")
  return str(block.value if isinstance(block, enum.Enum) else block).lower()


def lcm(*values: int) -> int:
  """Least common multiple of the values > 1 (1 if there are none)."""
  result = 1
  for v in values:
    if v > 1:
      result = result * v // math.gcd(result, v)
  return result


def align_down(value: int, quantum: int) -> int:
  return (value // quantum) * quantum


def align_up(value: int, quantum: int) -> int:
  return -(-value // quantum) * quantum


# ----------------------------------------------------------------------------- compile + HBM

# Same markers `train_compile.is_oom` uses, plus the generic XLA wording.
_OOM_MARKERS = ("resource_exhausted", "resource exhausted", "out of memory", "hbm")


class CompileFailedError(RuntimeError):
  """Compilation failed for a reason other than running out of memory."""


def is_oom_error(exc: BaseException) -> bool:
  message = str(exc).lower()
  return any(marker in message for marker in _OOM_MARKERS)


def peak_bytes(memory_analysis: Any) -> int:
  """Per-device peak bytes: MaxText's `output + temp + argument - alias`, or the compiler peak if larger."""

  def field(name: str) -> int:
    return int(getattr(memory_analysis, name, 0) or 0)

  total = (
      field("output_size_in_bytes")
      + field("temp_size_in_bytes")
      + field("argument_size_in_bytes")
      - field("alias_size_in_bytes")
  )
  return max(total, field("peak_memory_in_bytes"))


def target_mesh_shape(config: Any) -> dict[str, int]:
  """Axis sizes of the target-topology mesh `train_compile` builds for `config`."""
  return {str(k): int(v) for k, v in train_compile.get_topology_mesh(config).shape.items()}


def compile_train_step(config: Any) -> Any:
  """AoT-compile the MaxText train step for `config.compile_topology` (mirrors `train_compile.is_oom`)."""
  train_compile.validate_config(config)
  if get_config_value(config, "enable_diloco", False):
    raise CompileFailedError("enable_diloco=true is not supported by the slicer.")

  mesh = train_compile.get_topology_mesh(config)
  shaped_args, shaped_kwargs, state_mesh_shardings, _, model = train_compile.get_shaped_inputs(mesh, config)
  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  input_state_mesh_shardings = sharding.build_zero1_input_state_mesh_shardings(
      config, state_mesh_shardings, params_shardings
  )
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  func, in_shard, out_shard, static_argnums, donate_argnums = maxtext_utils.get_functional_train_with_signature(
      train.train_step, data_sharding, input_state_mesh_shardings, model, config, params_shardings
  )
  return train_compile.jit_and_compile(
      func,
      shaped_args,
      shaped_kwargs,
      mesh,
      in_shard,
      out_shard,
      static_argnums,
      donate_argnums,
      config,
      nn_partitioning.axis_rules(config.logical_axis_rules),
  )


def measure_peak_hbm(config: Any) -> int | None:
  """Per-device peak HBM bytes of the train step, or None if the compiler reports OOM.

  Raises:
    CompileFailedError: compilation failed for any other reason.
  """
  try:
    compiled = compile_train_step(config)
  except CompileFailedError:
    raise
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if is_oom_error(exc):
      return None
    raise CompileFailedError(f"{type(exc).__name__}: {exc}") from exc
  memory_analysis = compiled.memory_analysis()
  if memory_analysis is None:
    raise CompileFailedError("The compiler did not return a memory analysis for the target topology.")
  return peak_bytes(memory_analysis)


# ----------------------------------------------------------------------------- checkpoint


def write_random_checkpoint(config: Any, checkpoint_dir: str, seed: int = 0) -> str:
  """Initialize `config`'s parameters on local devices and save them; returns `load_parameters_path`.

  Parameters come from MaxText's own model construction and initializers (the
  `jit(nnx.state(create_model()))` path `setup_initial_state` uses) and are written
  with the same writer as `to_maxtext.py`. The weights are random: the checkpoint
  is for bring-up and verification, not pretrained behavior.
  """
  mesh = maxtext_utils.get_mesh_from_config(config)
  create_model_fn = model_creation_utils.get_nnx_create_model_fn(config, mesh, rng_key=jax.random.PRNGKey(seed))
  _, _, state_shardings = maxtext_utils.get_abstract_state(config, mesh, create_model_fn, is_training=False)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state = jax.jit(lambda: nnx.state(create_model_fn()), out_shardings=state_shardings)()

  params, _ = nnx.split_state(state, nnx.Param, ...)
  # Linen on-disk layout ({"params": {"params": ...}}), identical to converted checkpoints.
  linen_params = train_state_nnx.to_linen_checkpoint_dict({"model": params.to_pure_dict()})["params"]
  save_weights_to_checkpoint(
      checkpoint_dir,
      linen_params,
      1,  # Arrays are already sharded jax.Arrays; skip simulated-device resharding.
      config.checkpoint_storage_use_ocdbt,
      config.checkpoint_storage_use_zarr3,
      config=config,
  )
  return os.path.join(checkpoint_dir, "0", "items")
