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

"""m3 model garden: one self-contained directory per model family.

Each family owns its architecture end to end:

    models/
        qwen3/
            modeling_qwen3.py  # attention, MLP, decoder layer, model

This module holds the single dispatch point (create_model) and the model
registry. There is no shared Decoder, no shared attention class, and no
model-name branching.
"""

from collections.abc import Callable

from flax import nnx
import jax
from maxtext.m3.models.qwen3 import modeling_qwen3  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import qwen3_8_flops  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next.modeling_qwen3_8_flash_next import create_qwen3_8_flash_next_model  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_vl import qwen3_vl_flops  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3 import qwen3_flops  # pylint: disable=no-name-in-module


def create_glm5_3_flash_model(config, mesh=None, *, rngs, **kwargs):
  """Lazy factory for GLM-5.3-Flash to avoid eager tokamax import on CI."""
  # pylint: disable=import-outside-toplevel
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash  # pylint: disable=g-import-not-at-top

  return modeling_glm5_3_flash.create_glm5_3_flash_model(config, mesh=mesh, rngs=rngs, **kwargs)


def create_granitemoe_model(config, mesh=None, *, rngs, **kwargs):
  """Lazy factory for GraniteMoe (`granite-3.0-1b-a400m` / `granitemoe`)."""
  # pylint: disable=import-outside-toplevel
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  from maxtext.m3.models.granitemoe import modeling_granitemoe  # pylint: disable=g-import-not-at-top

  return modeling_granitemoe.create_granitemoe_model(config, mesh=mesh, rngs=rngs, **kwargs)


MODEL_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": create_glm5_3_flash_model,
    "granite-3.0-1b-a400m": create_granitemoe_model,
    "granitemoe": create_granitemoe_model,
    "qwen3-0.6b": create_qwen3_model,
    "qwen3-vl-2b": create_qwen3_vl_model,
    "qwen3.8-flash-next": create_qwen3_8_flash_next_model,
    # Debug-only mini variant of qwen3.8-flash-next: same architecture and constructor, shrunk to
    # 4 layers / 1 cycle (configs/models/qwen3.8-flash-next-mini.yml). Used only by CPU/CI unit tests.
    "qwen3.8-flash-next-mini": create_qwen3_8_flash_next_model,
}

_FOUR_AXIS_MESH = ("data", "fsdp", "tensor", "expert")
MODEL_SUPPORTED_MESH_AXES: dict[str, tuple[str, ...]] = {
    "glm5.3-flash": _FOUR_AXIS_MESH,
    "granite-3.0-1b-a400m": _FOUR_AXIS_MESH,
    "granitemoe": _FOUR_AXIS_MESH,
    "qwen3-0.6b": modeling_qwen3.m3_sharding.DEFAULT_SHARDING_AXES,
    "qwen3-vl-2b": _FOUR_AXIS_MESH,
    "qwen3.8-flash-next": ("data", "fsdp", "fsdp_transpose", "tensor", "tensor_sequence", "expert"),
    "qwen3.8-flash-next-mini": ("data", "fsdp", "fsdp_transpose", "tensor", "tensor_sequence", "expert"),
}
MODEL_EXPLICIT_SHARDING_SUPPORT: frozenset[str] = frozenset({"qwen3-0.6b"})


def _unsupported_model_mesh_settings(config) -> dict[str, object]:
  """Flags sharding modes or parallelism axes that `config.model_name` does not implement."""
  model_name = getattr(config, "model_name", None)
  if model_name not in MODEL_SUPPORTED_MESH_AXES:
    return {}
  unsupported: dict[str, object] = {}
  allowed = set(MODEL_SUPPORTED_MESH_AXES[model_name])
  for network in ("ici", "dcn"):
    for axis in modeling_qwen3.m3_sharding.DEFAULT_SHARDING_AXES:
      if axis in allowed:
        continue
      field = f"{network}_{axis}_parallelism"
      val = getattr(config, field, 1)
      if val != 1:
        unsupported[field] = val
  shard_mode = getattr(config, "shard_mode", "auto")
  shard_mode_val = getattr(shard_mode, "value", shard_mode)
  if shard_mode_val != "auto" and model_name not in MODEL_EXPLICIT_SHARDING_SUPPORT:
    unsupported["shard_mode"] = shard_mode_val
  return unsupported


modeling_qwen3.m3_sharding.register_model_mesh_validator(_unsupported_model_mesh_settings)
_end_eager_placement = modeling_qwen3.end_eager_placement


def create_model(config, mesh, **kwargs):
  """Dispatches to a registered m3 constructor using driver configuration.

  Parameters are placed from their sharding metadata as they are created,
  which needs a mesh in scope (README rule 3). If none is in scope, the mesh
  passed is entered for construction. If one is, it must be the one passed: a
  strategy resolved against another mesh is silently replaced by full
  replication downstream, so that case is an error here. Placement ends with
  construction (`_end_eager_placement`), so what is built from the model
  afterwards does not need the mesh.
  """
  if config.model_name not in MODEL_REGISTRY:
    raise ValueError(
        f"Model {config.model_name!r} does not support the m3 backend (use_m3_model=True). "
        f"Registered models: {', '.join(sorted(MODEL_REGISTRY))}."
    )
  if mesh is not None:
    if kwargs.get("sharding") is None:
      modeling_qwen3.check_mesh(mesh, MODEL_SUPPORTED_MESH_AXES[config.model_name])
    axis_types = tuple(getattr(mesh, "axis_types", ()))
    if (
        axis_types
        and any(t == jax.sharding.AxisType.Explicit for t in axis_types)
        and config.model_name not in MODEL_EXPLICIT_SHARDING_SUPPORT
    ):
      raise ValueError(
          f"Model {config.model_name!r} does not implement shard_mode='explicit' yet "
          f"(supported models: {sorted(MODEL_EXPLICIT_SHARDING_SUPPORT)}). Use shard_mode='auto'."
      )
    if dict(mesh.shape).get("context", 1) > 1:
      cp_strategy = getattr(config, "context_parallel_strategy", "all_gather")
      cp_strategy_val = str(getattr(cp_strategy, "value", cp_strategy)).lower()
      if cp_strategy_val != "all_gather":
        raise ValueError(
            f"m3 context parallelism only implements context_parallel_strategy='all_gather', got {cp_strategy!r}."
        )
      if bool(getattr(config, "context_parallel_load_balance", False)):
        raise ValueError(
            "m3 context parallelism requires context_parallel_load_balance=False (natural contiguous token order)."
        )
  constructor = MODEL_REGISTRY[config.model_name]
  with modeling_qwen3.enter_mesh_scope(mesh):
    return _end_eager_placement(constructor(config, mesh, **kwargs))


def calculate_glm5_3_flash_tflops_per_device(config, log: bool = True):
  """Lazy wrapper for GLM-5.3-Flash TFLOPs calculation."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.m3.models.glm5_3_flash import glm5_flops

  return glm5_flops.calculate_glm5_3_flash_tflops_per_device(config, log=log)


def calculate_granitemoe_tflops_per_device(config, log: bool = True):
  """Lazy wrapper for GraniteMoe TFLOPs calculation."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.m3.models.granitemoe import utils_granitemoe

  return utils_granitemoe.calculate_granitemoe_tflops_per_device(config, log=log)


MODEL_FLOPS_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": calculate_glm5_3_flash_tflops_per_device,
    "granite-3.0-1b-a400m": calculate_granitemoe_tflops_per_device,
    "granitemoe": calculate_granitemoe_tflops_per_device,
    "qwen3-0.6b": qwen3_flops.calculate_qwen3_tflops_per_device,
    "qwen3-vl-2b": qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device,
    "qwen3.8-flash-next": qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device,
    "qwen3.8-flash-next-mini": qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device,
}


def calculate_tflops_per_device(config, log: bool = True):
  """Calculates per-device training TFLOPs for registered m3 models."""
  if config.model_name not in MODEL_FLOPS_REGISTRY:
    raise ValueError(
        "TFLOPs calculation not implemented for m3 model"
        f" {config.model_name!r}. Registered models:"
        f" {', '.join(sorted(MODEL_FLOPS_REGISTRY))}."
    )
  return MODEL_FLOPS_REGISTRY[config.model_name](config, log=log)


def configure_glm5_3_flash_engine(engine):
  """Lazy wrapper for GLM-5.3-Flash training engine configuration."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.m3.models.glm5_3_flash import glm5_train

  return glm5_train.configure_glm5_engine(engine)


def configure_granitemoe_engine(engine):
  """Lazy wrapper for GraniteMoe training engine configuration."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.m3.models.granitemoe import utils_granitemoe

  return utils_granitemoe.configure_granitemoe_engine(engine)


MODEL_ENGINE_CONFIG_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": configure_glm5_3_flash_engine,
    "granite-3.0-1b-a400m": configure_granitemoe_engine,
    "granitemoe": configure_granitemoe_engine,
}


def configure_engine(engine):
  """Applies model-specific MaxTextTrainingEngine configuration for registered m3 models."""
  model_name = engine._config.model_name  # pylint: disable=protected-access
  if model_name in MODEL_ENGINE_CONFIG_REGISTRY:
    return MODEL_ENGINE_CONFIG_REGISTRY[model_name](engine)
  return engine
