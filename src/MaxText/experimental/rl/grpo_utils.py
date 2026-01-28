# Copyright 2023–2025 Google LLC
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

"""Utility functions for GRPO (Generative Rejection-based Policy Optimization)."""

import math
import numpy as np
import jax
import jax.numpy as jnp
import jaxtyping
from typing import Any, Callable

from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import DecoderBlockType
from maxtext.inference.offline_engine import InputData

from pathwaysutils.experimental import reshard as experimental_reshard
from pathwaysutils.experimental import split_by_mesh_axis


def _identity(x):
  return x


INTERMEDIATE_SPLIT_SUFFIX = "_intermediate_split"
INTERMEDIATE_REPLICA_SUFFIX = "_intermediate_replica"


def compute_log_probs(
    model,
    params,
    inputs,
    inputs_position,
    inputs_segmentation,
    completion_segmentation,
    config,
    is_train=False,
    rngs=None,
):
  """Computes per-token log-probabilities for a sequence of tokens.

  This helper calls model.apply (with dropout enabled if is_train) to obtain
  logits and then computes per-token log-probabilities.

  Note: We assume that tokens have been already appropriately padded.

  Args:
    model: The transformer model.
    params: Model parameters.
    inputs: A [B, L] array of input token IDs.
    inputs_position: A [B, L] array of token positions.
    inputs_segmentation: A [B, L] array of segment IDs.
    completion_segmentation: A [B, L] array that masks the completion part of
      the sequence.
    config: The configuration object.
    is_train: Whether to run in training mode (e.g., with dropout).
    rngs: JAX PRNG keys for dropout.

  Returns:
    A tuple containing:
      - token_log_probs: A [B, L-1] array of log-probabilities for each token
        in the completion.
      - intermediate_outputs: A dictionary of intermediate activations from the
        model.
  """
  if not is_train:
    params = jax.lax.stop_gradient(params)
  logits, intermediate_outputs = model.apply(
      params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=(config.enable_dropout if is_train else False),
      rngs=rngs,
      mutable="intermediates",
  )  # [B, S, E] - [batch, sequence, embedding/vocab]
  logits = logits / config.decode_sampling_temperature
  # Remove last time step since there is no target for the final position.
  targets = inputs[:, 1:]
  # Shift left using dynamic slice (skip first column)
  shifted_completion_segmentation = jax.lax.dynamic_slice(
      completion_segmentation, (0, 1), (completion_segmentation.shape[0], completion_segmentation.shape[1] - 1)
  )
  # Pad with 0 at the end to maintain the original shape
  shifted_completion_segmentation = jnp.pad(
      shifted_completion_segmentation, ((0, 0), (0, 1)), mode="constant", constant_values=0
  )

  mask = shifted_completion_segmentation[..., None]
  mask = jnp.broadcast_to(mask, logits.shape)

  masked_logits = jnp.where(mask, logits, -jnp.inf)
  log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
  log_probs = jnp.where(mask, log_probs, -0.0)
  log_probs = log_probs[:, :-1, :]
  # Gather the log probabilities corresponding to each target token.
  token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  token_log_probs = token_log_probs * shifted_completion_segmentation[:, :-1]

  return token_log_probs, intermediate_outputs


def generate_offline_completions(config, tokenizer_model, inference_engine, data):
  """Generates completions for a batch of prompts using an offline engine.

  Args:
    config: The configuration object.
    tokenizer_model: The tokenizer model.
    inference_engine: The offline inference engine for generation.
    data: A dictionary containing the input prompts and their true lengths.

  Returns:
    The input `data` dictionary updated with the generated completions,
    segmentations, positions, and log-probabilities.
  """
  data[config.train_data_columns] = np.asarray(
      jnp.repeat(data[config.train_data_columns], config.num_generations, axis=0)
  )
  data[f"{config.train_data_columns}_true_length"] = np.asarray(
      jnp.repeat(data[f"{config.train_data_columns}_true_length"], config.num_generations, axis=0)
  )
  input_data = []
  for i, d in enumerate(data[config.train_data_columns]):
    input_data.append(
        InputData(
            id=f"input_{i}",
            tokens=np.array(d),
            true_length=np.array(data[f"{config.train_data_columns}_true_length"][i])[0],
        )
    )

  results = inference_engine.batch_inference(input_data)
  prompt_completions_segmentation = []
  completion_segmentation = []
  prompt_completions = []
  prompt_completions_logprobs = []
  for i, r in enumerate(results):
    indices = np.arange(r.token_ids.shape[0])
    completion_mask = (indices >= np.array(data[f"{config.train_data_columns}_true_length"][i])[0]).astype(jnp.int32)
    completion_segmentation.append(completion_mask)
    prompt_completions.append(r.token_ids)
    prompt_completions_segmentation.append(np.full((r.token_ids.shape[0],), 1))
    prompt_completions_logprobs.append(r.logprobs)

  prompt_completions = pad_or_trim(prompt_completions, config.max_target_length, 0)  # assume 0 for pad_token_id
  completion_segmentation = pad_or_trim(completion_segmentation, config.max_target_length, 0)
  prompt_completions_segmentation = pad_or_trim(prompt_completions_segmentation, config.max_target_length, 0)
  prompt_completions_logprobs = pad_or_trim(prompt_completions_logprobs, config.max_target_length, -np.inf)

  data[f"{config.train_data_columns}_completions"] = prompt_completions
  data[f"{config.train_data_columns}_completions_segmentation"] = prompt_completions_segmentation
  data[f"{config.train_data_columns}_completions_position"] = np.where(
      data[f"{config.train_data_columns}_completions_segmentation"],
      np.arange(data[f"{config.train_data_columns}_completions"].shape[1]),
      0,
  )
  data["ar_completions_segmentation"] = completion_segmentation
  # off-policy
  if config.inference_rollouts > 1:
    data["completions_logprobs"] = prompt_completions_logprobs
  else:
    data["completions_logprobs"] = None
  return data


def pathways_reshard(config, inference_engine, params, source_shardings, source_mesh, destination_shardings):
  """Reshards model parameters from training to inference sharding.

  This function handles the resharding of parameters between different device
  meshes and sharding specifications, which is often necessary when moving from
  a training setup to an inference setup. It also handles unscanning of
  parameters if `config.scan_layers` is False.

  Args:
    config: The configuration object.
    inference_engine: The inference engine whose parameters will be updated.
    params: The model parameters to be resharded.
    source_shardings: The sharding specification of the source parameters.
    source_mesh: The source device mesh.
    destination_shardings: The sharding specification for the destination.
  """
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    layer_groups = [
        ("dense_layers", config.first_num_dense_layers),
        ("moe_layers", config.base_num_decoder_layers - config.first_num_dense_layers),
    ]
  else:
    layer_groups = [("layers", config.base_num_decoder_layers)]
  if not config.scan_layers:
    params_to_reshard = jax.tree_util.tree_map(lambda x: x, params)
    max_utils.unscan_train_state_params(
        params_to_reshard,
        source_shardings,
        source_mesh,
        scan_axis=config.param_scan_axis,
        layer_groups=layer_groups,
    )
  else:
    params_to_reshard = params

  with (
      jax.transfer_guard_device_to_host("disallow_explicit"),
      jax.transfer_guard_host_to_device("disallow_explicit"),
  ):
    resharded_params = reshard_pytree(params_to_reshard, destination_shardings)
  inference_engine.update_params(resharded_params)


def dummy_reward_len(valid_seq_mask):
  """Calculates a dummy reward based on the length of the valid sequence.

  Args:
    valid_seq_mask: A [BxG] mask indicating the valid (non-padded) tokens in
      the sequence.

  Returns:
    A [BxG] array of rewards, where the reward is the negative absolute
    difference between the sequence length and 20.
  """
  # adding a 1 because valid_seq_mask is actually one less than the number of valid tokens
  reward = -abs(20 - (1 + jnp.sum(valid_seq_mask, axis=-1)))  # [BxG]
  return reward


def concatenate_prompt_with_completions(config, tokenizer_model, data, completions):
  """Concatenates prompts with their generated completions.

  This function takes a batch of prompts and a corresponding batch of
  completions, concatenates them, and then truncates each sequence at the
  first end-of-sequence (EOS) token. It updates the data dictionary with the
  new combined sequences and their corresponding segmentation and position
  arrays.

  Args:
    config: Configuration object containing generation settings such as max
      sequence length or EOS token ID.
    tokenizer_model: Tokenizer used to decode or manipulate tokens (e.g.,
      identifying special tokens like EOS).
    data: Input batch containing prompt tokens, segmentation and position.
    completions: Generated token sequences to be appended to the corresponding
      prompts.

  Returns:
    The `data` dictionary updated with the concatenated sequences and new
    segmentation and position information.
  """

  def _concat_and_find_eos(prompt, true_len, completion):
    total_len = prompt.shape[0] + completion.shape[0]
    prompt_mask = jnp.arange(prompt.shape[0]) < true_len[0]
    trimmed_prompt = jnp.where(prompt_mask, prompt, 0)

    # Initialize with padded prompt
    full_seq = jnp.zeros((total_len,), dtype=prompt.dtype)
    full_seq = full_seq.at[: prompt.shape[0]].set(trimmed_prompt)

    # Dynamically insert completion at true_len position
    full_seq = jax.lax.dynamic_update_slice(full_seq, completion, (true_len[0],))

    # Find EOS index
    eos_mask = full_seq == tokenizer_model.eos_token_id
    eos_indices = jnp.where(eos_mask, jnp.arange(total_len), total_len)
    eos_index = jnp.min(eos_indices)

    return full_seq, eos_index

  batched_concat_and_eos = jax.vmap(_concat_and_find_eos, in_axes=(0, 0, 0))
  prompts = data[config.train_data_columns]
  true_length = data[f"{config.train_data_columns}_true_length"]
  prompt_completions, eos_positions = batched_concat_and_eos(prompts, true_length, completions)
  data[f"{config.train_data_columns}_completions"] = prompt_completions
  data[f"{config.train_data_columns}_completions_segmentation"] = (
      jnp.arange(data[f"{config.train_data_columns}_completions"].shape[1])[None, :] < eos_positions[:, None]
  ).astype(jnp.int32)
  data[f"{config.train_data_columns}_completions_position"] = jnp.where(
      data[f"{config.train_data_columns}_completions_segmentation"],
      jnp.arange(data[f"{config.train_data_columns}_completions"].shape[1]),
      0,
  )
  completion_mask = data[f"{config.train_data_columns}_completions_position"] >= true_length - 1
  data["ar_completions_segmentation"] = data[
      f"{config.train_data_columns}_completions_segmentation"
  ] * completion_mask.astype(jnp.int32)
  return data


def pad_or_trim(arr, max_target_length, pad_token):
  """Pads or trims a list of sequences to a maximum target length.

  Args:
    arr: A list of 1D numpy arrays (sequences).
    max_target_length: The desired length for all sequences.
    pad_token: The token ID to use for padding.

  Returns:
    A 2D numpy array of shape `(len(arr), max_target_length)`.
  """
  padded = np.array(
      [
          np.pad(seq[:max_target_length], (0, max(0, max_target_length - len(seq))), constant_values=pad_token)
          for seq in arr
      ]
  )
  return padded


def filter_and_split(config, example_batch, num_groups, global_batch_size_per_group):
  """Splits an example_batch into a list of smaller batches.

  Samples are taken from the beginning of the input batch, and extras are
  dropped if there are not enough to form the requested number of groups.

  Args:
    config: The configuration object.
    example_batch: A dictionary where keys are feature names (e.g., 'inputs')
      and values are arrays. The first dimension of each array is the batch
      size.
    num_groups: The number of smaller batches to create.
    global_batch_size_per_group: The size of each smaller batch.

  Returns:
    A list of dictionaries. Each dictionary has the same keys as
    `example_batch` but with values being arrays sliced for that group.
    Returns an empty list if not enough samples to form the required groups.
  """
  if not example_batch:  # Handles None or empty dict
    return []

  if num_groups <= 0 or global_batch_size_per_group <= 0:
    max_logging.log(
        f"Warning: config_inference.inference_replicas ({num_groups}) or config_inference.per_device_batch_size "
        f"({global_batch_size_per_group}) is not positive. Cannot split batch."
    )
    return []

  total_samples_needed = num_groups * global_batch_size_per_group
  total_samples_available = example_batch[config.train_data_columns].shape[0]
  if total_samples_available < total_samples_needed:
    max_logging.log(
        f"Warning: Not enough samples ({total_samples_available}) in batch to create {num_groups} groups of size"
        f" {global_batch_size_per_group} (needed {total_samples_needed}). Dropping batch."
    )
    return []

  # Slice the required number of samples
  sliced_batch = jax.tree_util.tree_map(lambda arr: arr[:total_samples_needed], example_batch)

  list_of_output_batches = []
  for i in range(num_groups):
    current_group_dict = {}
    start_index = i * global_batch_size_per_group
    end_index = start_index + global_batch_size_per_group
    for key, sliced_array in sliced_batch.items():
      # Slice each group
      current_group_dict[key] = sliced_array[start_index:end_index]
    list_of_output_batches.append(current_group_dict)

  return list_of_output_batches


def _maybe_find_intermediate_sharding(source_sharding, target_sharding):
  """Maybe finds an intermediate sharding to reshard to before target sharding."""
  if not isinstance(source_sharding, jax.sharding.NamedSharding) or not isinstance(
      target_sharding, jax.sharding.NamedSharding
  ):
    max_logging.log(
        "None-NamedSharding does not need intermediate sharding." f" {source_sharding=}, {target_sharding=}",
    )
    return None
  src_mesh = source_sharding.mesh
  dst_mesh = target_sharding.mesh

  def _get_sharding_dims(sharding, mesh):
    sharding_dims = {}
    used_mesh_axis_names = set()
    for i, axis_name in enumerate(sharding.spec):
      if axis_name is None:
        sharding_dims[(i, None)] = 1
      else:
        if isinstance(axis_name, tuple):
          used_mesh_axis_names |= set(axis_name)
          shard_size = math.prod([mesh.shape[name] for name in axis_name])
          first_axis_name = axis_name[0]
          sharding_dims[(i, mesh.axis_names.index(first_axis_name))] = shard_size
        else:
          assert isinstance(axis_name, str), "axis_name expected to be a string or a tuple of strings."
          used_mesh_axis_names.add(axis_name)
          sharding_dims[(i, mesh.axis_names.index(axis_name))] = mesh.shape[axis_name]
    largest_shards = max(sharding_dims.values()) if len(sharding_dims) else 1
    if len(sharding_dims) < len(mesh.shape):
      for mi, mesh_axis in enumerate(mesh.axis_names):
        if mesh_axis not in used_mesh_axis_names:
          sharding_dims[(None, mi)] = 1
    return sharding_dims, largest_shards

  src_sharding_dims, src_largest_shards = _get_sharding_dims(source_sharding, src_mesh)
  dst_sharding_dims, dst_largest_shards = _get_sharding_dims(target_sharding, dst_mesh)
  # Not able to handle resharding with undividable shardings.
  if src_largest_shards % dst_largest_shards != 0:
    return None

  total_source_sharding_dims = math.prod(list(src_sharding_dims.values()))
  total_dst_sharding_dims = math.prod(list(dst_sharding_dims.values()))

  if total_source_sharding_dims <= total_dst_sharding_dims or total_source_sharding_dims % total_dst_sharding_dims != 0:
    return None

  new_split_dim_shards = None
  new_split_axis = None
  replicas = src_largest_shards // dst_largest_shards

  # Find gcd(src_dim_shards, dst_dim_shards),
  # If all of them are 1s, an all-gather is needed as the single replica of
  # the source cannot be presented by any sharded form on the target devices.
  gcd_shards = []
  for (sharding_mesh_axis_idx, src_dim_shards), (_, dst_dim_shards) in zip(
      src_sharding_dims.items(), dst_sharding_dims.items()
  ):
    gcd_dim_shards = math.gcd(src_dim_shards, dst_dim_shards)
    if gcd_dim_shards == 1:
      if src_dim_shards > dst_dim_shards and src_dim_shards == src_largest_shards:
        new_split_axis = sharding_mesh_axis_idx
        new_split_dim_shards = (src_dim_shards // replicas, replicas)
    gcd_shards.append(gcd_dim_shards)
  if math.prod(gcd_shards) != 1 or new_split_axis is None:
    return None

  # Generate the intermediate sharding.
  new_split_mesh_axis_name = src_mesh.axis_names[new_split_axis[1]] + INTERMEDIATE_SPLIT_SUFFIX
  new_split_mesh_replica_axis_name = src_mesh.axis_names[new_split_axis[1]] + INTERMEDIATE_REPLICA_SUFFIX
  intermediate_mesh = jax.sharding.Mesh(
      src_mesh.devices.reshape(
          tuple(
              list(src_mesh.devices.shape[: new_split_axis[1]])
              + [new_split_dim_shards[0], new_split_dim_shards[1]]
              + list(src_mesh.devices.shape[new_split_axis[1] + 1 :])
          )
      ),
      axis_names=tuple(
          list(src_mesh.axis_names[: new_split_axis[1]])
          + [new_split_mesh_axis_name, new_split_mesh_replica_axis_name]
          + list(src_mesh.axis_names[new_split_axis[1] + 1 :])
      ),
  )

  intermediate_spec = tuple(
      list(source_sharding.spec[: new_split_axis[0]])
      + [new_split_mesh_axis_name]
      + list(source_sharding.spec[new_split_axis[0] + 1 :])
  )
  intermediate_sharding = jax.sharding.NamedSharding(
      intermediate_mesh,
      jax.sharding.PartitionSpec(*intermediate_spec),
      memory_kind=source_sharding.memory_kind,
  )

  return intermediate_sharding


def _experimental_pre_reshard(splitfn, src_pytree, target_shardings):
  """Simple heuristic to determine if resharding with replicated all-gather is needed."""
  src_shardings = jax.tree_util.tree_map(
      lambda x: x.sharding,
      src_pytree,
  )
  intermediate_shardings = jax.tree_util.tree_map(
      _maybe_find_intermediate_sharding,
      src_shardings,
      target_shardings,
  )

  src_leaves_with_path, src_treedef = jax.tree_util.tree_flatten_with_path(src_pytree)
  intermediate_sharding_leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(intermediate_shardings)
  intermediate_sharding_leaves_with_path = dict(intermediate_sharding_leaves_with_path)

  to_split_src_pytree_leaves = []
  to_split_src_pytree_leaves_indexes = []
  to_split_intermediate_sharding_leaves = []

  intermediate_mesh = None
  to_update_src_pytree_leaves = []

  for i, (path, src) in enumerate(src_leaves_with_path):
    to_update_src_pytree_leaves.append(src)
    if intermediate_sharding := intermediate_sharding_leaves_with_path.get(path, None):
      if intermediate_mesh is None:
        intermediate_mesh = intermediate_sharding.mesh
      to_split_src_pytree_leaves.append(src)
      to_split_src_pytree_leaves_indexes.append(i)
      to_split_intermediate_sharding_leaves.append(intermediate_sharding)

  if intermediate_mesh is None:
    return src_pytree

  to_split_axis = None
  for axis_name in intermediate_mesh.axis_names:
    if axis_name.endswith(INTERMEDIATE_REPLICA_SUFFIX):
      to_split_axis = axis_name
      break
  assert to_split_axis is not None, f"No replica axis found in the intermediate mesh {intermediate_mesh}."

  temp_source = jax.jit(
      _identity,
      out_shardings=to_split_intermediate_sharding_leaves,
  )(to_split_src_pytree_leaves)

  to_split_src_pytree_leaves, *_ = splitfn(temp_source, to_split_axis)

  for i, src_pytree_leaf in enumerate(to_split_src_pytree_leaves):
    to_update_src_pytree_leaves[to_split_src_pytree_leaves_indexes[i]] = src_pytree_leaf
  updated_src_pytree = jax.tree_util.tree_unflatten(src_treedef, to_update_src_pytree_leaves)
  return updated_src_pytree


def _get_reshard_fn_pathwaysutils(
    *,
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
):
  """Returns a reshard function using pathwaysutils."""

  def reshard_fn(
      x: Any,
      sharding: jax.sharding.Sharding | Any,
  ):
    if use_experimental_pre_reshard:
      x = _experimental_pre_reshard(split_by_mesh_axis.split_by_mesh_axis, x, sharding)

    return experimental_reshard.reshard(
        x,
        sharding,
        donate=donate,
        cache_resharding_plans=cache_resharding_plans,
    )

  return reshard_fn


def _get_reshard_fn(
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
    get_reshard_fns: list[Callable[..., Any]],
):
  """Returns a reshard function."""
  for get_reshard_fn in get_reshard_fns:
    try:
      reshard_fn = get_reshard_fn(
          cache_resharding_plans=cache_resharding_plans,
          donate=donate,
          use_experimental_pre_reshard=use_experimental_pre_reshard,
      )
    except (ImportError, EnvironmentError):
      max_logging.log(f"Could not support {get_reshard_fn=}.")
    else:
      return reshard_fn

  raise ValueError(f"Could not find a reshard function from {get_reshard_fns=}.")


def reshard_pytree(
    source: jaxtyping.PyTree,
    target: jaxtyping.PyTree,
    cache_plan: bool = True,
    donate_input: bool = False,
    use_experimental_pre_reshard: bool = True,
) -> jaxtyping.PyTree:
  """Reshard input pytree from source sharding and mesh to target sharding and mesh."""

  def _get_dst_sharding(x):
    if isinstance(x, jax.sharding.NamedSharding | jax.sharding.SingleDeviceSharding):
      return x
    else:
      return jax.sharding.NamedSharding(
          x.sharding.mesh,
          x.sharding.spec,
          memory_kind=x.sharding.memory_kind,
      )

  dst_shardings = jax.tree_util.tree_map(
      _get_dst_sharding,
      target,
  )
  reshard_fn = _get_reshard_fn(
      cache_resharding_plans=cache_plan,
      donate=donate_input,
      use_experimental_pre_reshard=use_experimental_pre_reshard,
      get_reshard_fns=[
          _get_reshard_fn_pathwaysutils,
      ],
  )

  resharded_array = reshard_fn(source, dst_shardings)
  resharded_array = jax.block_until_ready(resharded_array)
  return resharded_array
