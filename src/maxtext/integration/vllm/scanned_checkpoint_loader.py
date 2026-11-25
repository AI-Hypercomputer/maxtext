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

`load_scanned_checkpoint` loads a scanned MaxText checkpoint (`scan_layers=True`)
into an initialized unscanned in-process vLLM rollout target
(`ScannedCheckpointTarget`) while freeing scanned source buffers once their last
conversion group completes, keeping peak HBM near one copy of the model weights.
It detects the checkpoint's param item dir, storage format and MoE `wi` layout
itself. `resolve_scanned_checkpoint` is the file-only part of that detection,
for callers that want a bad path to fail before starting vLLM.

TODO: Replace the `_build_plan`/`_execute_group` hook orchestration around
`MaxTextToMaxTextConverter` once MaxText's consolidated weight conversion API
lands with native source-buffer freeing and pre-fused MoE `wi` re-layout
support, without modifying legacy `MaxTextToMaxTextConverter`.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import functools
import gc
import json
import logging
import os
import re
import threading
from typing import Any, Protocol, runtime_checkable
import warnings

from etils import epath
from flax import nnx
import jax
import jaxtyping
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.vllm.convert_utils import _get_n_shards
from maxtext.integration.vllm.weight_converter import MaxTextToMaxTextConverter
from maxtext.layers import fsdp_prefetch_inhomogeneous
from maxtext.layers import moe
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from orbax import checkpoint as ocp

# Orbax PyTree item dirs a scanned checkpoint step dir may hold its params in,
# tried in order after `path` itself. These are checkpoint item names, not
# parameter names; `from_pretrained` needs the item dir itself.
_PARAM_ITEM_DIRS = (
    # MaxText `CheckpointManager`, `to_maxtext` and `save_params_to_path`.
    "items",
    # Orbax `CheckpointManager` with a single unnamed item.
    "default",
    # Tunix `sft/checkpoint_manager.py`.
    "model_params",
)


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


@dataclasses.dataclass(frozen=True)
class ScannedCheckpointStorage:
  """Where and how a scanned checkpoint stores its parameters.

  Attributes:
    params_dir: Orbax PyTree item dir of the parameters, as passed to
      `from_pretrained` in `load_parameters_path`.
    use_ocdbt: Whether the item is stored with OCDBT.
    use_zarr3: Whether the arrays are stored as Zarr v3.
  """

  params_dir: str
  use_ocdbt: bool
  use_zarr3: bool


def resolve_scanned_checkpoint(path: str | os.PathLike[str]) -> ScannedCheckpointStorage:
  """Finds the param item dir of a scanned checkpoint and its storage format.

  Only reads files: it neither builds a MaxText config nor initializes JAX, so
  callers can run it before starting vLLM to fail fast on a bad path.
  `load_scanned_checkpoint` calls it too.

  Args:
    path: A checkpoint step dir, or the param item dir itself.

  Returns:
    The param item dir and its storage format.

  Raises:
    FileNotFoundError: If neither `path` nor any known item dir under it holds
      Orbax `_METADATA`.
    ValueError: If `_METADATA` has no Orbax tree metadata.
  """
  root = epath.Path(path)
  candidates = [root] + [root / name for name in _PARAM_ITEM_DIRS]
  params_dir = next((c for c in candidates if (c / "_METADATA").exists()), None)
  if params_dir is None:
    raise FileNotFoundError(
        f"No Orbax checkpoint `_METADATA` found for scanned checkpoint {path}; tried: "
        + ", ".join(str(c) for c in candidates)
    )
  metadata_file = params_dir / "_METADATA"
  try:
    metadata = json.loads(metadata_file.read_text())
  except json.JSONDecodeError as e:
    raise ValueError(f"{metadata_file} is not valid JSON.") from e
  if not isinstance(metadata, dict) or not isinstance(metadata.get("tree_metadata"), dict):
    raise ValueError(f"{metadata_file} has no Orbax `tree_metadata` mapping.")
  # `PyTreeCheckpointHandler().metadata()` would give `use_zarr3` too, but
  # constructing the handler initializes the JAX backends. Orbax reads the same
  # key, defaulting to False (`InternalTreeMetadata.from_json`).
  return ScannedCheckpointStorage(
      params_dir=str(params_dir),
      use_ocdbt=ocp.type_handlers.is_ocdbt_checkpoint(params_dir),
      use_zarr3=bool(metadata.get("use_zarr3", False)),
  )


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
    split_keys: Callable[..., Iterable[tuple[Any, ...]]],
    src_flat: dict[tuple[Any, ...], jax.Array],
    tgt_flat: Mapping[tuple[Any, ...], jax.Array],
    skip_paths: frozenset[tuple[Any, ...]] = frozenset(),
) -> Any:
  """Plans conversion, splitting pre-fused `wi` if needed, and counts reads.

  Exposes each scanned pre-fused MoE `wi` key returned by `split_keys` as
  `wi_0`/`wi_1` so the converter re-fuses `[gate | up]` into the rollout's
  per-shard interleaved layout. `_execute_group_and_free_sources` slices the
  halves lazily and frees each source only after its last group finishes.

  Args:
    build_plan: The converter's `_build_plan`.
    prefused_wi: Filled with the pre-fused `wi` arrays that are split, keyed by
      their parent path.
    remaining_reads: Filled with the number of distinct groups reading each
      source key.
    split_keys: Called as `split_keys(src_flat, tgt_flat, plan_fn)`, where
      `plan_fn(src_flat, tgt_flat)` builds an unsplit plan; returns the
      pre-fused `wi` source keys to split.
    src_flat: Flattened source state; split `wi` keys are replaced in place.
    tgt_flat: Flattened target state.
    skip_paths: Target paths the converter skips.

  Returns:
    The plan from `build_plan`.
  """
  for key in list(split_keys(src_flat, tgt_flat, lambda src, tgt: build_plan(src, tgt, skip_paths))):
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


def _prefused_wi_keys_to_split(
    prefused_keys: Sequence[tuple[Any, ...]],
    src_flat: Mapping[tuple[Any, ...], jax.Array],
    tgt_flat: Mapping[tuple[Any, ...], jax.Array],
    plan_fn: Callable[..., Any],
) -> tuple[tuple[Any, ...], ...]:
  """Returns `prefused_keys` if a rollout `wi` they feed is sharded on its fused axis.

  The converter interleaves `[gate | up]` per shard of the target's fused (last)
  axis, counting the shards from the target's sharding (`_fuse_moe_bulk`). With
  one shard that layout equals the pre-fused one, so the scanned `wi` is sliced
  as is, without an extra copy.

  Args:
    prefused_keys: The checkpoint's pre-fused `wi` source keys.
    src_flat: Flattened source state.
    tgt_flat: Flattened target state.
    plan_fn: Builds an unsplit plan from `(src_flat, tgt_flat)`.

  Returns:
    The keys to split into `wi_0`/`wi_1`, or `()`.

  Raises:
    RuntimeError: If a pre-fused key is not in `src_flat` or feeds no rollout
      parameter, rather than silently copying it in the wrong layout.
  """
  if not prefused_keys:
    return ()
  missing = [key for key in prefused_keys if key not in src_flat]
  if missing:
    raise RuntimeError(
        f"Pre-fused MoE weight {'.'.join(map(str, missing[0]))} detected in the checkpoint is not in the restored "
        "scanned state."
    )
  plan = plan_fn(src_flat, tgt_flat)
  unused = [key for key in prefused_keys if not any(key in entry.source_keys for entry in plan)]
  if unused:
    raise RuntimeError(
        f"Pre-fused MoE weight {'.'.join(map(str, unused[0]))} detected in the checkpoint feeds no rollout parameter."
    )
  keys = set(prefused_keys)
  targets = [tgt_flat[entry.target_key] for entry in plan if keys.intersection(entry.source_keys)]
  n_shards = max(_get_n_shards(t, t.ndim - 1) for t in targets)
  return tuple(prefused_keys) if n_shards > 1 else ()


def _checkpoint_param_tree(tree: Mapping[str, Any]) -> Mapping[str, Any]:
  """Returns the model params of an Orbax metadata tree.

  Locates them as `model_creation_utils.from_pretrained` does: Linen
  `{"params": {"params": ...}}`, or NNX with an optional `"base"` root.

  Args:
    tree: The checkpoint's Orbax metadata tree.

  Returns:
    The subtree keyed like the model's NNX graph.
  """
  params = tree.get("params")
  if isinstance(params, Mapping) and "params" in params:
    return params["params"]
  return tree.get("base", tree)


_UNSTACK_LOCK = threading.Lock()


class _FlatCycleLayers(nnx.Module):
  """Holds unstacked `layer_0..layer_{n-1}` sub-layers of a scanned inhomogeneous cycle."""

  def __init__(self, stages: Sequence[tuple[Any, Any, Any]]):
    for i, (graphdef, params, state) in enumerate(stages):
      setattr(self, f"layer_{i}", nnx.merge(graphdef, params, state))


def _unstack_decoder_local_layers(model: nnx.Module) -> nnx.Module:
  """Replaces a nested `(local_layers, global_layer)` decoder block stack with flat `layer_<i>` attributes."""
  decoder = getattr(model, "decoder", None)
  layers = getattr(decoder, "layers", None)
  if layers is None or getattr(layers, "local_layers", None) is None or hasattr(decoder, "layers_remainder"):
    return model
  stages = fsdp_prefetch_inhomogeneous.sublayer_stages(layers)
  if stages is not None:
    decoder.layers = _FlatCycleLayers(stages)
  return model


@contextlib.contextmanager
def _unstacked_local_layers(enabled: bool) -> Iterator[None]:
  """Temporarily wraps `model_creation_utils.get_nnx_create_model_fn` to unstack nested cycle layers."""
  if not enabled:
    yield
    return
  with _UNSTACK_LOCK:
    orig_get_create_fn = model_creation_utils.get_nnx_create_model_fn

    def _wrapped_get_create_fn(*args: Any, **kwargs: Any) -> Callable[[], nnx.Module]:
      create_fn = orig_get_create_fn(*args, **kwargs)
      return lambda: _unstack_decoder_local_layers(create_fn())

    model_creation_utils.get_nnx_create_model_fn = _wrapped_get_create_fn
    try:
      yield
    finally:
      model_creation_utils.get_nnx_create_model_fn = orig_get_create_fn


def _nested_cycle_sublayer_count(abstract_model: nnx.Module) -> int:
  """Returns the sub-layer count of a nested `(local_layers, global_layer)` decoder stack, or 0."""
  decoder = getattr(abstract_model, "decoder", None)
  layers = getattr(decoder, "layers", None)
  if layers is None or getattr(layers, "local_layers", None) is None or hasattr(decoder, "layers_remainder"):
    return 0
  count = int(getattr(layers, "num_local", 0)) + int(getattr(layers, "num_global", 0))
  return count if count >= 2 else 0


def _uses_flat_cycle_layers(tree: Mapping[str, Any], num_sublayers: int, params_dir: str) -> bool:
  """Returns whether `tree` stores a `num_sublayers`-layer cycle as flat `decoder.layers.layer_<i>` modules.

  Checkpoints written before Qwen3-Next/Qwen3.5 adopted nested `local_layers` + `global_layer`
  blocks keep one module per cycle slot (`layer_0..layer_{num_sublayers - 1}`).

  Raises:
    ValueError: If `decoder.layers` mixes flat `layer_<i>` modules with `local_layers`/`global_layer`,
      or holds a set of `layer_<i>` modules other than `layer_0..layer_{num_sublayers - 1}`.
  """
  if num_sublayers == 0:
    return False
  decoder = tree.get("decoder")
  ckpt_layers = decoder.get("layers") if isinstance(decoder, Mapping) else None
  if not isinstance(ckpt_layers, Mapping):
    return False
  slot_keys = {k for k in ckpt_layers if isinstance(k, str) and re.fullmatch(r"layer_\d+", k)}
  if not slot_keys:
    return False

  def _slot_index(key: str) -> int:
    return int(key.split("_", 1)[1])

  nested_present = sorted({"local_layers", "global_layer"} & set(ckpt_layers))
  if nested_present:
    raise ValueError(
        f"Scanned checkpoint {params_dir} mixes nested {nested_present} and flat "
        f"{sorted(slot_keys, key=_slot_index)} under decoder/layers."
    )
  expected = {f"layer_{i}" for i in range(num_sublayers)}
  if slot_keys != expected:
    raise ValueError(
        f"Scanned checkpoint {params_dir} has flat cycle layers {sorted(slot_keys, key=_slot_index)} "
        f"under decoder/layers; expected {sorted(expected, key=_slot_index)}."
    )
  return True


def _detect_checkpoint_layout(
    probe_cfg: pyconfig.HyperParameters, params_dir: str
) -> tuple[bool, tuple[tuple[Any, ...], ...]]:
  """Detects flat-cycle layer layout and pre-fused routed MoE `wi` keys from checkpoint metadata.

  Inspects the configured scanned model's NNX graph against the checkpoint's
  Orbax tree metadata:
  - If the model nests cycle sub-layers under `decoder.layers.{local_layers, global_layer}`
    while the checkpoint stores `decoder.layers.layer_0..layer_{n-1}`, restores
    into the flat-cycle structure directly without stacking `local_layers`.
  - Looks up each `moe.RoutedMoE` module in the matching graph: `wi` means
    pre-fused `[gate | up]`, `wi_0` and `wi_1` mean unfused. Other parameters,
    such as a dense MLP `wi`, are never looked at.

  Args:
    probe_cfg: Scanned MaxText config of the model.
    params_dir: Orbax item dir of the checkpoint's parameters.

  Returns:
    `(unstack_local_layers, prefused_keys)`: whether to unstack nested cycle
    sub-layers for restore, and the `wi` source keys of every routed MoE when
    the checkpoint is pre-fused (`()` when unfused or when the model has no
    routed MoE).

  Raises:
    ValueError: If flat-cycle layers are malformed, a routed MoE has neither
      layout, MoE layouts are mixed, or `num_experts > 1` has no `RoutedMoE`.
  """
  _, abstract_model = model_creation_utils.create_nnx_abstract_model(
      probe_cfg, devices=jax.devices(), model_mode=MODEL_MODE_AUTOREGRESSIVE
  )
  num_sublayers = _nested_cycle_sublayer_count(abstract_model)
  moe_paths = [tuple(path) for path, node in nnx.iter_graph(abstract_model) if isinstance(node, moe.RoutedMoE)]
  if not moe_paths:
    del abstract_model
    if probe_cfg.num_experts > 1:
      raise ValueError(
          f"Cannot locate routed MoE weights: {probe_cfg.model_name} has num_experts={probe_cfg.num_experts} "
          "but no `RoutedMoE` module in its scanned NNX graph."
      )
    if num_sublayers == 0:
      return False, ()

  # The checkpoint is read after vLLM started JAX, so the handler may initialize it.
  tree = _checkpoint_param_tree(ocp.PyTreeCheckpointHandler().metadata(epath.Path(params_dir)).tree)
  unstack_local_layers = _uses_flat_cycle_layers(tree, num_sublayers, params_dir)
  if moe_paths:
    if unstack_local_layers:
      flat_model = nnx.eval_shape(_unstack_decoder_local_layers, abstract_model)
      moe_paths = [tuple(path) for path, node in nnx.iter_graph(flat_model) if isinstance(node, moe.RoutedMoE)]
      del flat_model
    del abstract_model
  if not moe_paths:
    return unstack_local_layers, ()

  prefused: list[tuple[Any, ...]] = []
  unfused: list[tuple[Any, ...]] = []
  for path in moe_paths:
    node = tree
    for name in path:
      node = node.get(str(name)) if isinstance(node, Mapping) else None
    names = set(node) if isinstance(node, Mapping) else set()
    if "wi" in names and not names & {"wi_0", "wi_1"}:
      prefused.append(path)
    elif {"wi_0", "wi_1"} <= names and "wi" not in names:
      unfused.append(path)
    else:
      raise ValueError(
          f"Scanned checkpoint {params_dir} has neither a pre-fused `wi` nor `wi_0`/`wi_1` under routed MoE "
          f"{'/'.join(map(str, path))}; found {sorted(names)}."
      )
  if prefused and unfused:
    raise ValueError(
        f"Scanned checkpoint {params_dir} mixes pre-fused MoE `wi` ({'/'.join(map(str, prefused[0]))}) and "
        f"unfused `wi_0`/`wi_1` ({'/'.join(map(str, unfused[0]))})."
    )
  return unstack_local_layers, tuple(path + ("wi",) for path in prefused)


def _load(
    path: str | os.PathLike[str],
    target: ScannedCheckpointTarget,
    *,
    config_argv: list[str],
    restore_overrides: Mapping[str, object],
    split_keys: Callable[..., Iterable[tuple[Any, ...]]],
    tp: int,
    fp8_moe: bool,
    unstack_local_layers: bool = False,
) -> None:
  """Restores the scanned checkpoint and converts it into `target`.

  Args:
    path: The checkpoint, for logs and errors.
    target: Initialized in-process vLLM weights to load into.
    config_argv: `pyconfig.initialize` argv naming the MaxText base config.
    restore_overrides: MaxText config overrides for the scanned restore.
    split_keys: Picks the pre-fused `wi` source keys to split; see
      `_split_prefused_wi_then_build_plan`.
    tp: The converter's `tp`.
    fp8_moe: Whether to quantize the MoE expert weights to FP8.
    unstack_local_layers: Whether to unstack nested `(local_layers, global_layer)`
      blocks into flat `layer_<i>` modules for restore.
  """
  logging.info("Loading scanned MaxText checkpoint %s into vLLM.", path)
  target.delete_cache()
  _delete_arrays(target.transformer_state.filter(nnx.Param))

  scanned_cfg = pyconfig.initialize(config_argv, **restore_overrides)
  with _unstacked_local_layers(unstack_local_layers):
    scanned_model, _ = model_creation_utils.from_pretrained(
        scanned_cfg,
        devices=jax.devices(),
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )
  scanned_state = nnx.state(scanned_model, nnx.Param)
  del scanned_model
  gc.collect()

  converter = MaxTextToMaxTextConverter(
      config=scanned_cfg,
      tp=tp,
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
      split_keys,
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

  # `_place_on` already placed every converted weight on `target.transformer_state`'s
  # sharding. When `target` exposes `reinitialize_cache` (`VllmSampler` /
  # `InprocessVllmSampler`), update `transformer_state` in place and drop `converted`
  # before reallocating the KV cache: `update_params` runs a non-donating Pathways
  # reshard copy and reallocates the KV cache while `converted` is still live.
  reinit_cache = getattr(target, "reinitialize_cache", None)
  if callable(reinit_cache) and isinstance(target.transformer_state, nnx.State):
    while isinstance(converted, dict) and "model" in converted and "model" not in target.transformer_state:
      converted = converted["model"]
    nnx.update(target.transformer_state, converted)
    del converted
    gc.collect()
    jax.effects_barrier()
    reinit_cache()
  else:
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


def _config_argv(base_config_path: str | os.PathLike[str] | None) -> list[str]:
  """Returns `pyconfig.initialize` argv for `base_config_path`, defaulting to `base.yml`."""
  if base_config_path is None:
    base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  return ["", str(base_config_path)]


def load_scanned_checkpoint(
    path: str | os.PathLike[str],
    target: ScannedCheckpointTarget,
    maxtext_config: Mapping[str, object],
    *,
    base_config_path: str | os.PathLike[str] | None = None,
) -> None:
  """Loads a scanned MaxText checkpoint into `target`'s unscanned vLLM weights.

  Detects the checkpoint's layout itself:
  - the param item dir and storage format (`resolve_scanned_checkpoint`);
  - whether a hybrid cycle is stored as flat `decoder.layers.layer_<i>` modules
    rather than nested `decoder.layers.{local_layers, global_layer}`;
  - whether routed MoE gate/up are pre-fused into one `wi`, from the
    checkpoint's tree metadata at the configured model's `RoutedMoE` paths;
  - whether a pre-fused `wi` must be re-laid out per shard, from the sharding
    of the rollout `wi` it feeds.

  Restores the checkpoint with `scan_layers=True` in its stored layout and
  precision, converts it in memory with `MaxTextToMaxTextConverter`, and passes
  the result to `target.update_params`. The KV cache and `target`'s placeholder
  parameters are freed first, and each scanned source is freed once converted,
  so the checkpoint fits next to the rollout on large MoE models. A failure
  after detection leaves `target` without weights; the caller should not serve
  from it.

  Args:
    path: A scanned checkpoint step dir, or its param item dir.
    target: Initialized in-process vLLM weights to load into.
    maxtext_config: The rollout model's MaxText config overrides, as given to
      vLLM in `additional_config["maxtext_config"]` (`model_name`,
      `prefuse_moe_weights`, `fp8_moe`, ...). Scan, attention, checkpoint path
      and storage settings are overridden for the restore.
    base_config_path: MaxText base config; defaults to MaxText's `base.yml`.

  Raises:
    FileNotFoundError: If no Orbax checkpoint is found at `path`.
    ValueError: If the checkpoint's MoE layout can't be determined, or it is
      pre-fused while the rollout's `prefuse_moe_weights` is false.
    RuntimeError: If a weight `target` serves was freed during the load.
  """
  storage = resolve_scanned_checkpoint(path)
  config_argv = _config_argv(base_config_path)
  overrides = {
      **maxtext_config,
      "scan_layers": True,
      "attention": "dot_product",
      "model_call_mode": "",
      "skip_jax_distributed_system": True,
  }
  probe_cfg = pyconfig.initialize(config_argv, **overrides)
  if probe_cfg.use_m3_model:
    raise ValueError(
        "load_scanned_checkpoint does not support use_m3_model: M3 renames layers on restore, "
        "which MoE layout detection does not apply."
    )
  unstack_local_layers, prefused_keys = _detect_checkpoint_layout(probe_cfg, storage.params_dir)
  if prefused_keys and not probe_cfg.prefuse_moe_weights:
    raise ValueError(
        f"Scanned checkpoint {storage.params_dir} stores routed MoE gate/up pre-fused as `wi`, but the rollout "
        "has prefuse_moe_weights=False; loading a pre-fused checkpoint into an unfused rollout is not supported."
    )
  _load(
      storage.params_dir,
      target,
      config_argv=config_argv,
      restore_overrides={
          **overrides,
          "load_parameters_path": storage.params_dir,
          # Restore in the stored layout: `from_pretrained` would otherwise fuse
          # `wi_0`/`wi_1` itself, an extra copy of every MoE gate/up.
          "prefuse_moe_weights": bool(prefused_keys),
          # Restore in the stored precision; the converter quantizes.
          "fp8_moe": False,
          "checkpoint_storage_use_ocdbt": storage.use_ocdbt,
          "checkpoint_storage_use_zarr3": storage.use_zarr3,
      },
      split_keys=functools.partial(_prefused_wi_keys_to_split, prefused_keys),
      # With `NamedSharding` targets the converter takes the `wi` shard count
      # from the target; otherwise `_get_n_shards` gives 1 and the converter
      # resolves `tp` from the config or environment as for `tp=1`.
      tp=1,
      fp8_moe=probe_cfg.fp8_moe,
      unstack_local_layers=unstack_local_layers,
  )


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
  """Deprecated: use `load_scanned_checkpoint`, which detects the layout itself.

  Loads a scanned MaxText checkpoint into `target`'s unscanned vLLM weights with
  a caller-provided layout. Restores the checkpoint with `scan_layers=True`,
  converts it in memory with `MaxTextToMaxTextConverter`, and passes the result
  to `target.update_params`. The KV cache and `target`'s placeholder parameters
  are freed first, and each scanned source is freed once converted, so the
  checkpoint fits next to the rollout on large MoE models. A failure leaves
  `target` without weights; the caller should not serve from it.

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
  warnings.warn(
      "load_and_convert_scanned_checkpoint is deprecated; use load_scanned_checkpoint, which detects the "
      "checkpoint layout itself.",
      DeprecationWarning,
      stacklevel=2,
  )
  split_prefused_wi = ckpt_prefuse_moe and mesh_tp > 1
  _load(
      path,
      target,
      config_argv=_config_argv(base_config_path),
      restore_overrides={
          **maxtext_config_overrides,
          "load_parameters_path": str(path),
          "scan_layers": True,
          "prefuse_moe_weights": ckpt_prefuse_moe,
          "attention": "dot_product",
          "model_call_mode": "",
      },
      split_keys=lambda src_flat, tgt_flat, plan_fn: (
          [k for k in src_flat if "layers" in k and k[-1] == "wi"] if split_prefused_wi else []
      ),
      tp=mesh_tp,
      fp8_moe=fp8_moe,
  )
