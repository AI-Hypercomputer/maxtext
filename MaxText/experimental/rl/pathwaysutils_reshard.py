"""Experimental resharding API for elastic device sets."""

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
from typing import Any, Dict, Sequence

import jax
from pathwaysutils import plugin_executable

from MaxText import max_logging

class ReshardingPlanWrapper:
  """Wrapper around PluginProgram(reshard_request)."""

  _plugin_executable: plugin_executable.PluginExecutable
  _avals: tuple[jax.core.ShapedArray, ...]
  _out_shardings: tuple[jax.sharding.Sharding, ...]

  def __init__(
      self,
      avals: Sequence[jax.core.ShapedArray],
      source_shardings: Sequence[jax.sharding.Sharding],
      destination_shardings: Sequence[jax.sharding.Sharding],
      donate: bool,
  ):
    def ifrt_hlo_sharding(
        aval: jax.core.ShapedArray, sharding: jax.sharding.Sharding
    ) -> Dict[str, Any]:
      result = {
          "devices": {
              "device_ids": [
                  device.id for device in sharding._addressable_device_assignment  # pylint: disable=protected-access
              ]
          },
          "xla_hlo_sharding": (
              base64.b64encode(
                  sharding._to_xla_hlo_sharding(aval.ndim)  # pylint: disable=protected-access
                  .to_proto()
                  .SerializeToString()
              ).decode("utf-8")
          ),
      }
      if sharding.memory_kind is not None:
        result["memory_kind"] = sharding.memory_kind
      return result

    request = {
        "reshardRequest": {
            "donateInput": donate,
            "inSharding": [
                ifrt_hlo_sharding(aval, old_sharding)
                for aval, old_sharding in zip(avals, source_shardings)
            ],
            "outSharding": [
                ifrt_hlo_sharding(aval, new_sharding)
                for aval, new_sharding in zip(avals, destination_shardings)
            ],
        }
    }

    self._plugin_executable = plugin_executable.PluginExecutable(
        json.dumps(request)
    )
    self._avals = avals
    self._out_shardings = destination_shardings

  def execute(self, inp_arrays: tuple[jax.Array, ...]) -> Sequence[jax.Array]:
    out_arrays, fut = self._plugin_executable.call(
        inp_arrays, self._out_shardings, self._avals
    )
    fut.result()
    return out_arrays


def _get_resharding_plan(
    avals: tuple[jax.core.ShapedArray, ...],
    old_shardings: tuple[jax.sharding.Sharding, ...],
    new_shardings: tuple[jax.sharding.Sharding, ...],
    donate: bool,
) -> ReshardingPlanWrapper:
  """Returns a resharding plan for the given sharding task."""
  return ReshardingPlanWrapper(
      avals, old_shardings, new_shardings, donate
  )


_get_resharding_plan_cached = jax.util.cache()(_get_resharding_plan)


def reshard(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate: bool = False,
    may_alias: bool | None = None,  # pylint: disable=unused-argument
    cache_resharding_plans: bool = False,
) -> Any:
  """Reshards `x` to `sharding`.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    sharding: A `Sharding` or a (nested) `Sharding` in standard Python container
      (must be a tree prefix of `x`), representing the device(s) and sharding to
      which `x` should be sharded to. The result will be committed to the
      device(s) of the sharding.
    donate: If `True`, donate all input arrays, which may reduce the
      amount memory needed for resharding. Buffers donated to resharding should
      not be reused.
    may_alias: If `True`, may alias the input array with the output array.
      May reduce the amount of memory needed for resharding. Not used at the
      moment.
    cache_resharding_plans: If `True`, uses a resharding plan cache to avoid
      recreating plans for the same resharding operation. May improve
      performance for use cases where the same resharding operation is done many
      times. May degrade performance if most reshardings operations are
      different, since the cache will cause Pathways Components to remain loaded
      for each cached plan. `False` by default.

  Returns:
    A copy of `x` whose sharding is `sharding`.
  """
  flat_x, tree_def = jax.tree_util.tree_flatten(x)
  flat_sharding = jax.api_util.flatten_axes(
      "reshard sharding", tree_def, sharding
  )

  jax_arrays = []
  jax_array_dst_shardings = []
  non_jax_arrays = []
  non_jax_array_dst_shardings = []
  for arr, dst_sharding in zip(flat_x, flat_sharding):
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `jax.sharding.Sharding`")
    if isinstance(arr, jax.Array):
      max_logging.log(f"Debug resharding {arr.shape=} {arr.sharding} {dst_sharding}")
      jax_arrays.append(arr)
      jax_array_dst_shardings.append(dst_sharding)
    else:
      non_jax_arrays.append(arr)
      non_jax_array_dst_shardings.append(dst_sharding)

  if non_jax_arrays:
    non_jax_arrays = jax.device_put(non_jax_arrays, non_jax_array_dst_shardings)

  if jax_arrays:
    get_resharding_plan_func = (
        _get_resharding_plan_cached
        if cache_resharding_plans
        else _get_resharding_plan
    )
    jax_arrays = get_resharding_plan_func(
        tuple(arr.aval for arr in jax_arrays),
        tuple(arr.sharding for arr in jax_arrays),
        tuple(jax_array_dst_shardings),
        donate,
    ).execute(tuple(jax_arrays))

  result = []
  jax_iter = iter(jax_arrays)
  non_jax_iter = iter(non_jax_arrays)

  for arr in flat_x:
    if isinstance(arr, jax.Array):
      result.append(next(jax_iter))
    else:
      result.append(next(non_jax_iter))
  return jax.tree_util.tree_unflatten(tree_def, result)
