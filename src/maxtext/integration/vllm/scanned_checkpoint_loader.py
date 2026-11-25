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

"""Loads scanned MaxText Orbax checkpoints into in-process vLLM rollout weights.

`load_and_convert_scanned_checkpoint` provides the caller boundary for loading a
scanned MaxText checkpoint (`scan_layers=True`) into an initialized unscanned
in-process vLLM rollout target (`ScannedCheckpointTarget`) while freeing scanned
source buffers once their last conversion group completes, keeping peak HBM near
one copy of the model weights.

TODO: Replace the `_build_plan`/`_execute_group` hook orchestration around
`MaxTextToMaxTextConverter` once MaxText's consolidated weight conversion API
lands with native source-buffer freeing and pre-fused MoE `wi` re-layout
support, without modifying legacy `MaxTextToMaxTextConverter`.
"""

from collections.abc import Callable, Mapping
import functools
import gc
import logging
import os
from typing import Any, Protocol, runtime_checkable

from flax import nnx
import jax
import jaxtyping
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.vllm.weight_converter import MaxTextToMaxTextConverter
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


@runtime_checkable
class ScannedCheckpointTarget(Protocol):
  """In-process vLLM weights that a scanned MaxText checkpoint is loaded into.

  `tunix.generate.vllm_sampler.VllmSampler` satisfies this protocol, as does an
  initialized `InprocessVllmSampler` (`tunix.dist_rl.rollout`).
  `update_params` re-allocates the KV cache that `delete_cache` freed, and
  reshards in chunks when configured by the caller.
  """

  @property
  def transformer_state(self) -> nnx.State:
    ...

  def update_params(self, updated_weights: jaxtyping.PyTree) -> None:
    ...

  def delete_cache(self) -> None:
    ...


def _delete_arrays(tree: jaxtyping.PyTree, keep: jaxtyping.PyTree | None = None) -> None:
  """Frees the live `jax.Array` leaves of `tree` that are not in `keep`."""
  keep_ids = {id(leaf) for leaf in jax.tree_util.tree_leaves(keep)}
  for leaf in jax.tree_util.tree_leaves(tree):
    if isinstance(leaf, jax.Array) and id(leaf) not in keep_ids and not leaf.is_deleted():
      leaf.delete()


def _place_on(out: jax.Array, sharding: jax.sharding.Sharding) -> jax.Array:
  """Moves `out` onto `sharding`, freeing its old buffer unless it is reused."""
  if out.sharding == sharding:
    return out
  placed = jax.device_put(out, sharding)
  jax.block_until_ready(placed)
  # An equivalent sharding can hand back the same buffer.
  if not out.sharding.is_equivalent_to(sharding, out.ndim):
    _delete_arrays(out, keep=placed)
  return placed


def _split_prefused_wi_then_build_plan(
    build_plan: Callable[..., Any],
    prefused_wi: dict[tuple[Any, ...], jax.Array],
    remaining_reads: dict[tuple[Any, ...], int],
    split_prefused_wi: bool,
    src_flat: dict[tuple[Any, ...], jax.Array],
    tgt_flat: Mapping[tuple[Any, ...], jax.Array],
    skip_paths: frozenset[tuple[Any, ...]] = frozenset(),
) -> Any:
  """Plans conversion, splitting pre-fused `wi` if needed, and counts reads.

  When `split_prefused_wi` is true, exposes each scanned pre-fused MoE `wi` as
  `wi_0`/`wi_1` so the converter re-fuses `[gate | up]` into the rollout's
  per-shard interleaved layout. `_execute_group_and_free_sources` slices the
  halves lazily and frees each source only after its last group finishes.

  Args:
    build_plan: The converter's `_build_plan`.
    prefused_wi: Filled with the pre-fused `wi` arrays, keyed by their parent
      path, when `split_prefused_wi` is true.
    remaining_reads: Filled with the number of distinct groups reading each
      source key.
    split_prefused_wi: Whether to split pre-fused `wi` keys into `wi_0`/`wi_1`.
    src_flat: Flattened source state; `wi` keys are replaced in place when
      `split_prefused_wi` is true.
    tgt_flat: Flattened target state.
    skip_paths: Target paths the converter skips.

  Returns:
    The plan from `build_plan`.
  """
  if split_prefused_wi:
    for key in [k for k in src_flat if "layers" in k and k[-1] == "wi"]:
      wi = src_flat.pop(key)
      prefused_wi[key[:-1]] = wi
      src_flat[key[:-1] + ("wi_0",)] = wi
      src_flat[key[:-1] + ("wi_1",)] = wi
  plan = build_plan(src_flat, tgt_flat, skip_paths)
  remaining_reads.clear()
  distinct_groups = dict.fromkeys((entry.source_keys, entry.op, getattr(entry, "local_index", None)) for entry in plan)
  for source_keys, _, _ in distinct_groups:
    for key in dict.fromkeys(source_keys):
      remaining_reads[key] = remaining_reads.get(key, 0) + 1
  return plan


def _execute_group_and_free_sources(
    execute_group: Callable[..., list[tuple[tuple[Any, ...], jax.Array]]],
    prefused_wi: dict[tuple[Any, ...], jax.Array],
    sliced_wi: dict[tuple[Any, ...], jax.Array],
    remaining_reads: dict[tuple[Any, ...], int],
    group: Any,
    src_flat: Mapping[tuple[Any, ...], jax.Array],
    tgt_flat: Mapping[tuple[Any, ...], jax.Array],
) -> list[tuple[tuple[Any, ...], jax.Array]]:
  """Runs one conversion group, frees exhausted sources, and places its outputs.

  Freeing each scanned source once its last group finishes keeps peak HBM near
  one copy of the weights instead of two, while keeping shared `local_layers`
  sources alive across all cycle slots.

  Args:
    execute_group: The converter's `_execute_group`.
    prefused_wi: Pre-fused `wi` arrays, filled by
      `_split_prefused_wi_then_build_plan`.
    sliced_wi: Cached `wi_0`/`wi_1` slices for active pre-fused `wi` sources.
    remaining_reads: Remaining group reads per source key.
    group: The converter's plan group (`op`, `source_keys`).
    src_flat: Flattened source state.
    tgt_flat: Flattened target state.

  Returns:
    `(target_key, array)` pairs, each on its target's sharding.
  """
  parent = group.source_keys[0][:-1]
  if group.op == "fuse_moe" and parent in prefused_wi:
    k0, k1 = group.source_keys
    if k0 not in sliced_wi or k1 not in sliced_wi:
      prefused = prefused_wi[parent]
      half = prefused.shape[-1] // 2
      sliced_wi[k0] = prefused[..., :half]
      sliced_wi[k1] = prefused[..., half:]
    src_flat = {**src_flat, k0: sliced_wi[k0], k1: sliced_wi[k1]}
  sources = {key: src_flat[key] for key in group.source_keys}
  outs = execute_group(group, src_flat, tgt_flat)
  out_arrays = [out for _, out in outs]
  jax.block_until_ready(out_arrays)

  exhausted: list[jax.Array] = []
  for key, source in sources.items():
    remaining = remaining_reads.get(key, 1) - 1
    remaining_reads[key] = remaining
    if remaining <= 0:
      exhausted.append(source)
      sliced = sliced_wi.pop(key, None)
      if sliced is not None:
        exhausted.append(sliced)
  if (
      group.op == "fuse_moe"
      and parent in prefused_wi
      and remaining_reads.get(parent + ("wi_0",), 0) <= 0
      and remaining_reads.get(parent + ("wi_1",), 0) <= 0
  ):
    exhausted.append(prefused_wi.pop(parent))

  # An identity group's output can be its source buffer.
  if group.op != "identity" and exhausted:
    _delete_arrays(exhausted, keep=out_arrays)
  return [(tgt_key, _place_on(out, tgt_flat[tgt_key].sharding)) for tgt_key, out in outs]


def load_and_convert_scanned_checkpoint(
    path: str | os.PathLike[str],
    target: ScannedCheckpointTarget,
    *,
    mesh_tp: int,
    ckpt_prefuse_moe: bool,
    fp8_moe: bool,
    maxtext_config_overrides: Mapping[str, object],
    base_config_path: str | os.PathLike[str] | None = None,
) -> None:
  """Loads a scanned MaxText checkpoint into `target`'s unscanned vLLM weights.

  Restores the checkpoint with `scan_layers=True`, converts it in memory with
  `MaxTextToMaxTextConverter`, and passes the result to `target.update_params`.
  The KV cache and `target`'s placeholder parameters are freed first, and each
  scanned source is freed once converted, so the checkpoint fits next to the
  rollout on large MoE models. A failure leaves `target` without weights; the
  caller should not serve from it.

  Args:
    path: Orbax checkpoint directory of the scanned parameters.
    target: Initialized in-process vLLM weights to load into.
    mesh_tp: Tensor-parallel degree of the rollout.
    ckpt_prefuse_moe: Whether the checkpoint stores MoE gate/up as one `wi`.
    fp8_moe: Whether to quantize the MoE expert weights to FP8.
    maxtext_config_overrides: MaxText config of the model (`model_name`,
      checkpoint storage settings, ...). Scan, attention, and checkpoint path
      settings are overridden for the restore.
    base_config_path: MaxText base config; defaults to MaxText's `base.yml`.

  Raises:
    RuntimeError: If a weight `target` serves was freed during the load.
  """
  logging.info("Loading scanned MaxText checkpoint %s into vLLM.", path)
  target.delete_cache()
  _delete_arrays(target.transformer_state.filter(nnx.Param))

  if base_config_path is None:
    base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  scanned_cfg = pyconfig.initialize(
      ["", str(base_config_path)],
      **{
          **maxtext_config_overrides,
          "load_parameters_path": str(path),
          "scan_layers": True,
          "prefuse_moe_weights": ckpt_prefuse_moe,
          "attention": "dot_product",
          "model_call_mode": "",
      },
  )
  scanned_model, _ = model_creation_utils.from_pretrained(
      scanned_cfg, devices=jax.devices(), model_mode=MODEL_MODE_AUTOREGRESSIVE
  )
  scanned_state = nnx.state(scanned_model, nnx.Param)
  del scanned_model
  gc.collect()

  converter = MaxTextToMaxTextConverter(
      config=scanned_cfg,
      tp=mesh_tp,
      prefuse_moe_weights=True,
      target_dtype=None,
      fp8_moe=fp8_moe,
  )
  # TODO: Remove the `_build_plan`/`_execute_group` hook bridge once MaxText's
  # consolidated weight conversion API lands with native source freeing and
  # pre-fused `wi` re-layout support.
  prefused_wi: dict[tuple[Any, ...], jax.Array] = {}
  sliced_wi: dict[tuple[Any, ...], jax.Array] = {}
  remaining_reads: dict[tuple[Any, ...], int] = {}
  # pylint: disable=protected-access
  build_plan, execute_group = converter._build_plan, converter._execute_group
  converter._build_plan = functools.partial(
      _split_prefused_wi_then_build_plan,
      build_plan,
      prefused_wi,
      remaining_reads,
      ckpt_prefuse_moe and mesh_tp > 1,
  )
  converter._execute_group = functools.partial(
      _execute_group_and_free_sources,
      execute_group,
      prefused_wi,
      sliced_wi,
      remaining_reads,
  )
  try:
    converted = converter.convert(scanned_state, target_state=target.transformer_state)
  finally:
    converter._build_plan, converter._execute_group = build_plan, execute_group
  # pylint: enable=protected-access
  del scanned_state
  gc.collect()

  while isinstance(converted, dict) and "model" in converted:
    converted = converted["model"]
  target.update_params(converted)
  del converted
  gc.collect()
  jax.effects_barrier()

  freed = [
      leaf.shape
      for leaf in jax.tree_util.tree_leaves(target.transformer_state)
      if isinstance(leaf, jax.Array) and leaf.is_deleted()
  ]
  if freed:
    raise RuntimeError(f"{len(freed)} vLLM weight(s) were freed while loading {path}; first shapes: {freed[:10]}")
  logging.info("Loaded scanned MaxText checkpoint %s into vLLM.", path)
