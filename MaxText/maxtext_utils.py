"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=bare-except, consider-using-generator
""" Utils that are only interesting to MaxText. """

from typing import Optional
import functools
import pickle

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state

import numpy as np

from jax.experimental import mesh_utils
from jax.experimental.serialize_executable import deserialize_and_load
from jax.sharding import PartitionSpec as P
import jax
import jax.numpy as jnp

import optax

import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import DecoderBlockType, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText.inference.page_manager import PageState

OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"

# Multimodal constants
NUM_IMAGES_PER_SEQUENCE = 1
NUM_IMAGE_CHANNELS = 3


def get_input_data_sharding(config, mesh):
  """Get the input data sharding for the model"""
  return nn.logical_to_mesh_sharding(P(*config.input_data_sharding_logical_axes), mesh, config.logical_axis_rules)


def get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for `train_step`."""
  functional_train = get_functional_train_step(train_step, model, config, state_mesh_shardings)
  functional_train.__name__ = "train_step"
  data_sharding = get_input_data_sharding(config, mesh)
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = (state_mesh_shardings, None)  # State, metrics
  static_argnums = ()  # We partial out the static argnums of model and config
  donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_train_step(train_step, model, config, state_mesh_shardings):
  return functools.partial(train_step, model, config, state_mesh_shardings)


def get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for eval_step"""
  functional_eval = get_functional_eval_step(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  data_sharding = get_input_data_sharding(config, mesh)
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = None  # metrics
  static_argnums = ()  # We partial out the static argnums of model, config
  donate_argnums = ()  # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_eval_step(eval_step, model, config):
  return functools.partial(eval_step, model, config)


def get_shaped_batch(config):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator, but eval_shape doesn't work, see b/306901078."""
  batch_shape = (config.global_batch_size_to_load, config.max_target_length)
  shaped_batch = {}
  shaped_batch["inputs"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  return shaped_batch


def load_compiled(config, partial_train, state):
  """# Loading a serialized compiled train step function."""

  # Currently partial_train and state  are needed to reconstruct
  # input/output shapes to construct the in_trees and out_trees for load API
  # Parker is working on a serializing these
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as f:
      serialized_compiled = pickle.load(f)
    return serialized_compiled

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree)
  return p_train_step


def calculate_tokens_training_per_device(config):
  """Calculate training Tokens per device"""
  return config.max_target_length * config.per_device_batch_size * config.gradient_accumulation_steps


def calculate_gemma2_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
  """
  Calculate training TFLOP for Gemma2 as in Gemma2 we combine [local_attention, global_attention] into one decoder
  layer and we use sliding window attention in local_attention
  """
  attention_flops = (
      # global attention
      4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
      +
      # local attention
      4
      * config.per_device_batch_size
      * config.max_target_length
      * min(config.sliding_window_size, config.max_target_length)
      * config.num_query_heads
      * config.head_dim
  )
  attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12

  # multiply num_decoder_layers by 2 because we combine [local_attention, global_attention] into one decoder layer
  learnable_weight_tflops = (
      ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers * 2 + embedding_flops) * 3 / 10**12
  )

  return attention_tflops, learnable_weight_tflops


def calculate_mla_tflops_per_device(config):
  """Calculate Multi-Head Latent Attention TFLOP"""
  batch_len = config.per_device_batch_size * config.max_target_length
  qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim
  # calculate mla query projection
  if config.q_lora_rank == 0:
    q_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * qk_head_dim_sum
  else:
    # calculate query down and up flops
    q_flops = (
        2 * batch_len * (config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum)
    )
  # calculate mla kv projection with down and up flops
  kv_flops = (
      2
      * batch_len
      * (
          config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
          + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim)
      )
  )
  qkv_flops = q_flops + kv_flops

  attention_flops = 2 * batch_len * config.max_target_length * config.num_query_heads * (qk_head_dim_sum + config.v_head_dim)
  projection_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * config.v_head_dim
  return qkv_flops, attention_flops, projection_flops


def calculate_ffn_mamtul_tflops_per_device(config, mlp_dim):
  """Helper function to calculate matmul TFLOP in ffn based on MLP dimension.

  Applies to:
    - Dense FFN layers (mlp_dim = config.mlp_dim).
    - MoE FFN layers (mlp_dim = config.moe_mlp_dim),
      need to scale by shared_experts or num_experts_per_tok.
  """
  ffn1_flops = (
      2 * config.per_device_batch_size * config.max_target_length * mlp_dim * config.emb_dim * len(config.mlp_activations)
  )
  ffn2_flops = 2 * config.per_device_batch_size * config.max_target_length * mlp_dim * config.emb_dim
  return ffn1_flops + ffn2_flops


def calculate_routed_and_shared_ffn_tflops_per_device(config):
  """Helper function to calculate DeepSeek-style ffn TFLOP"""
  gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
  # Due to the mixed decoder layers, the flops is multiplied by num of layers for both dense and moe
  num_dense_layers, num_moe_layers = get_dense_moe_layers(config)
  dense_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * num_dense_layers
  shared_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.shared_experts
  routed_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.num_experts_per_tok
  moe_ffn_flops = (gate_flops + shared_experts_flops + routed_experts_flops) * num_moe_layers
  total_ffn_flops = dense_ffn_flops + moe_ffn_flops
  return total_ffn_flops


def get_dense_moe_layers(config):
  """Helper function to calculate number of dense and moe layers"""
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    num_dense_layers = config.first_num_dense_layers
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    return num_dense_layers, num_moe_layers
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    num_moe_layers = config.num_decoder_layers // config.interleave_moe_layer_step
    num_dense_layers = config.num_decoder_layers - num_moe_layers
  else:
    raise ValueError("Currently we only support DeepSeek and Llama4 calculation.")

  return num_dense_layers, num_moe_layers


def calculate_tflops_training_per_device(config, log=True):
  """Calculate training TFLOP"""
  # MLP flops
  if config.num_experts > 1:
    # calculation based on dropless implementation
    if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LLAMA4):
      total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)
    else:
      gate_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_experts
      total_ffn_flops = (
          gate_flops + calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) * config.num_experts_per_tok
      )
  else:
    total_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)

  # Attention flops
  if config.attention_type == "mla":
    qkv_flops, attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
  else:
    qkv_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * (config.num_query_heads + 2 * config.num_kv_heads)
        * config.head_dim
    )
    attention_flops = (
        4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
    )
    projection_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * config.num_query_heads
        * config.head_dim
    )

  # Embedding flops
  embedding_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.vocab_size

  # Combine flops with number of decoder layers
  if config.decoder_block == DecoderBlockType.GEMMA2:
    attention_tflops, learnable_weight_tflops = calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LLAMA4):
    learnable_weight_tflops = (
        (total_ffn_flops + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12
  else:
    # multiply by 3 for both feed forward and back propagation flops
    learnable_weight_tflops = (
        ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    # megatron tflops calculation does not account for causality in attention
    attention_tflops = attention_flops * config.num_decoder_layers * 3 / 10**12

  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps

  # DPO includes one additional forward pass per gradient accumulation step
  if config.use_dpo:
    reference_model_tflops = learnable_weight_tflops / 3  # additional forward pass
    reference_model_attention_tflops = attention_tflops / 3
    attention_tflops = attention_tflops + reference_model_attention_tflops
  else:
    reference_model_tflops = 0

  total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops

  if log:
    print(
        "Per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops/total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops


# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_prefill_tflops_per_device(num_model_parameters, prefill_length, config, log=True):
  """Calculate training TFLOP"""
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count() / 1e12
  noncasual_attention_flops = (
      4
      * config.num_query_heads
      * config.num_decoder_layers
      * config.head_dim
      * prefill_length**2
      / jax.device_count()
      / 1e12
  )
  causal_attention_tflops = noncasual_attention_flops / 2  # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print(
        "Per prefill step per device: \n",
        f"\tTotal TFLOPs: {total_tflops:.2f} \n",
        f"\t\tLearnable weight TFLOPs: {learnable_weight_tflops:.2f} ",
        f"({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n",
        f"\t\tCausal attention TFLOPs: {causal_attention_tflops:.2f} ",
        f"({100 * causal_attention_tflops/total_tflops:.2f})% of Total",
    )
  return total_tflops, learnable_weight_tflops, causal_attention_tflops


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  """Checks whether most params are sharded across sharding axis.

  This function determines whether the majority of parameters  are distributed
  across a specified sharding axes with an acceptable tolerance. It compares the
  current distribution to a scenario where all parameters are fully sharded
  across the 'fsdp', 'fsdp_transpose', 'sequence', and 'tensor' axes.

  Args:
    params: params of the model state
    mesh: mesh constructed from config
    tolerance: float between 0.0 and 1.0 representing the allowed percentage of
    non-sharded parameters.
  Returns:
    bool: True if the majority of parameters are sufficiently sharded
  """
  total_num_params = max_utils.calculate_num_params_from_pytree(params)
  product_num_devices_for_weight_sharding = 1
  for axis in [
      "fsdp",
      "fsdp_transpose",
      "sequence",
      "context",
      "context_autoregressive",
      "tensor",
      "tensor_transpose",
      "tensor_sequence",
      "stage",
      "expert",
  ]:
    product_num_devices_for_weight_sharding *= mesh.shape[axis]
  total_num_params_per_chip = max_utils.calculate_total_params_per_chip(params)
  perfectly_sharded_params_per_chip = total_num_params / product_num_devices_for_weight_sharding
  assert total_num_params_per_chip >= perfectly_sharded_params_per_chip, (
      "Number of parameters per chip must not be less than in the ideal sharded "
      "scenario across `fsdp`, `fsdp_transpose`, `context`, `sequence`, `tensor`, `tensor_transpose`, "
      "`tensor_sequence`, `stage`, `expert` axes."
  )
  unsharded_param_perc = total_num_params_per_chip / perfectly_sharded_params_per_chip - 1
  assert unsharded_param_perc < tolerance, (
      f"Number of unsharded parameters exceeds tolerance {tolerance * 100}% "
      f"of total parameters with a value of {unsharded_param_perc * 100}%."
  )


def apply_gradient_clipping(raw_grads, state, clipping_threshold):
  """Applies gradient clipping to raw gradients, with special handing for FLAX fp8 stats.

  Args:
    raw_grads: A pytree of raw gradients.
    state: The current optimizer state.
    clipping_threshold: The gradient clipping threshold.

  Returns:
    A pytree of clipped gradients.
  """
  gradient_clip_transformation = optax.clip_by_global_norm(clipping_threshold)
  if OVERWRITE_WITH_GRADIENT in raw_grads:
    # Scales + Amax History for Delayed Tensor Scaling SHOULD NOT be clipped or affect clipping
    fp8_stats = raw_grads.pop(OVERWRITE_WITH_GRADIENT)
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)
    grads[OVERWRITE_WITH_GRADIENT] = fp8_stats  # pytype: disable=unsupported-operands
    raw_grads[OVERWRITE_WITH_GRADIENT] = fp8_stats  # pytype: disable=unsupported-operands
  else:
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)

  return grads


def get_nested_value(dictionary, nested_key, default=None):
  """
  Retrieves a value from a nested key in a dictionary.

  Args:
      dictionary: The dictionary to search in.
      nested_key: A tuple representing the nested key, e.g., ('level1', 'level2', 'key').
      default: The value to return if the nested key is not found.

  Returns:
      The value associated with the nested key, or the default value if not found.
  """
  current_level = dictionary

  for key in nested_key:
    if not isinstance(current_level, dict) or key not in current_level:
      return default
    current_level = current_level[key]
  return current_level


def init_decode_state(apply_fn, params) -> train_state.TrainState:
  """Init train state with null opt state for decode."""
  state = train_state.TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})  # type: ignore
  return state


def init_training_state(apply_fn, params, tx):
  """Init train state with null opt state for decode."""
  state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
  return state


def init_initial_state(model, tx, config, is_training, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, is_training, key
  """
  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  image_shape = (
      config.micro_batch_size_to_train_on,
      NUM_IMAGES_PER_SEQUENCE,
      config.image_size_for_vit,
      config.image_size_for_vit,
      NUM_IMAGE_CHANNELS,
  )
  model_vars = model.init(
      {"params": key, "dropout": key, "aqt": key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32),
  )
  if is_training:
    return init_training_state(model.apply, model_vars, tx)
  return init_decode_state(model.apply, model_vars)


def setup_decode_state(model, config, rng, mesh, checkpoint_manager):
  """Setup decode state by loading params from a checkpoint.
  Args:
    model: the flax model to initialize
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: Checkpoint manager

  Returns:
    state: state with decode params loaded from the checkpoint
    state_mesh_annotations: the mesh annotations for the state
  """
  if not config.load_parameters_path:
    # generate random params
    max_logging.log("No decode checkpoint specified - generating random weights.")
    state, state_mesh_annotations, _, _ = setup_initial_state(
        model, None, None, config, rng, mesh, checkpoint_manager, False
    )
  else:
    # Load params from checkpoint
    max_logging.log(f"Loading decode params from {config.load_parameters_path}")
    unboxed_abstract_state, state_mesh_annotations, _ = get_abstract_state(model, None, config, rng, mesh, False)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      params = checkpointing.load_params_from_path(
          config.load_parameters_path,
          unboxed_abstract_state.params,
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )
    state = init_decode_state(None, params)

  state = max_utils.unbox_logicallypartioned(state)
  return state, state_mesh_annotations


def setup_training_state(model, data_iterator, tx, config, rng, mesh, checkpoint_manager):
  is_training = True
  return setup_initial_state(
      model,
      data_iterator,
      tx,
      config,
      rng,
      mesh,
      checkpoint_manager,
      is_training,
  )


def setup_initial_state(
    model,
    data_iterator,
    tx,
    config,
    rng,
    mesh,
    checkpoint_manager,
    is_training=True,
):
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    is_training: True to initialize training state, False for decode state

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
      model, tx, config, rng, mesh, is_training
  )

  # Initialization
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    restored, raw_params = checkpointing.load_state_if_possible(
        checkpoint_manager,
        data_iterator,
        config.load_parameters_path,
        config.load_full_state_path,
        config.checkpoint_storage_concurrent_gb,
        unboxed_abstract_state,
        config.enable_single_replica_ckpt_restoring,
        config.dataset_type,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )

    if restored:
      if isinstance(
          checkpoint_manager,
          (
              emergency_checkpoint_manager.CheckpointManager,
              emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager,
          ),
      ):
        state = restored
      else:
        if "iter" in restored and restored["iter"] is not None:
          data_iterator.local_iterator = restored["iter"]
        state = restored["items"]
    else:
      init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training)
      init_state_partial.__name__ = "initialize_state"
      # pylint: disable=not-callable
      state = jax.jit(
          init_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings,
      )(rng)
      if raw_params:  # If we loaded a partial state, we need to merge it.
        state = state.replace(params=raw_params)

  state = max_utils.unbox_logicallypartioned(state)

  return state, state_mesh_annotations, state_mesh_shardings, data_iterator


def get_abstract_state(model, tx, config, rng, mesh, is_training=True):
  """Get a shaped abstraction of the state (including optimizer)"""
  init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training, rng)

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_partial)

  state_logical_annotations = nn.get_partition_spec(abstract_state)

  state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, config.logical_axis_rules)
  if is_training and config.optimizer_memory_host_offload:
    opt_state = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.opt_state)
    state_mesh_shardings = state_mesh_shardings.replace(opt_state=opt_state)
  if is_training and config.parameter_memory_host_offload:
    assert config.param_scan_axis == 0, "You must set the scan axis 0 to enable parameter offloading."

    def move(path, x):
      max_logging.log(f"max_utils.py: Moving {path} to host")
      return x.with_memory_kind(kind="pinned_host")

    params = jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params)
    state_mesh_shardings = state_mesh_shardings.replace(params=params)

  abstract_sharded_state = jax.jit(init_state_partial, in_shardings=None, out_shardings=state_mesh_shardings).eval_shape()

  unboxed_abstract_sharded_state = max_utils.unbox_logicallypartioned(abstract_sharded_state)
  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return (
      unboxed_abstract_sharded_state,
      state_mesh_annotations,
      state_mesh_shardings,
  )


def get_prefill_kv_cache_annotations(model, config, rng, mesh, page_state: Optional[PageState] = None):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
        config.global_batch_size_to_load,
        config.max_prefill_predict_length,
    )
    image_shape = (
        config.global_batch_size_to_load,
        NUM_IMAGES_PER_SEQUENCE,
        config.image_size_for_vit,
        config.image_size_for_vit,
        NUM_IMAGE_CHANNELS,
    )

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        encoder_images=jnp.ones(image_shape) if config.use_multimodal else None,
        model_mode=MODEL_MODE_PREFILL,
        slot=0,
        page_state=page_state,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def get_kv_cache_annotations(model, config, rng, mesh, page_state: Optional[PageState] = None):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
        config.global_batch_size_to_load,
        1,
    )
    image_shape = (
        config.global_batch_size_to_load,
        NUM_IMAGES_PER_SEQUENCE,
        config.image_size_for_vit,
        config.image_size_for_vit,
        NUM_IMAGE_CHANNELS,
    )

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        encoder_images=jnp.ones(image_shape) if config.use_multimodal else None,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        slot=0,
        page_state=page_state,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def save_quantized_checkpoint_if_configured(config, params):
  """Save quantized checkpoint if configured"""
  assert config.quantization, "quantization must be configured"
  if config.save_quantized_params_path:
    checkpointing.save_params_to_path(
        checkpoint_dir=config.save_quantized_params_path,
        params=params,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
  else:
    max_logging.log("Skipping saving quantized checkpoint as save_quantized_params_path is null.")


def add_config_to_summary_writer(config, summary_writer):
  """Writes config params to tensorboard"""
  if jax.process_index() == 0:
    for key, value in config.get_keys().items():
      max_utils.add_text_to_summary_writer(key, str(value), summary_writer)


def logical_axis_rules_pp_act_as_dp(logical_rules):
  """Add stage as a physical axes before data for each rule, so stage acts just like data instead of PP.
  This is used when we want to pipeline only a subset of layers, and leave the rest like DP.
  """
  new_rules = []
  for key, physical_axes in logical_rules:
    if isinstance(physical_axes, str):
      physical_axes = (physical_axes,)
    else:
      physical_axes = tuple(physical_axes)
    new_physical_axes = tuple(axis for axis in physical_axes if axis != "stage")
    if "data" in new_physical_axes:
      data_idx = new_physical_axes.index("data")
      new_physical_axes = new_physical_axes[0:data_idx] + ("stage",) + new_physical_axes[data_idx:]
    new_rules.append((key, new_physical_axes))
  return tuple(new_rules)


def create_device_mesh(config, devices=None):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas"""
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  num_slices = 1 if config.inference_benchmark_test else config.num_slices
  num_devices_per_slice = num_devices // num_slices

  multi_slice_env = num_slices > 1

  # Find possible unspecified parallelisms
  ici_parallelism = max_utils.fill_unspecified_mesh_axes(config.ici_parallelism.copy(), num_devices_per_slice, "ICI")

  allow_split_physical_axes = config.allow_split_physical_axes if config.allow_split_physical_axes else False

  if multi_slice_env:
    dcn_parallelism = max_utils.fill_unspecified_mesh_axes(config.dcn_parallelism.copy(), num_slices, "DCN")
    if max_utils.is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = max_utils.create_custom_device_mesh(ici_parallelism, dcn_parallelism, devices, config.custom_mesh)
    else:
      mesh = mesh_utils.create_hybrid_device_mesh(
          ici_parallelism,
          dcn_parallelism,
          devices,
          allow_split_physical_axes=allow_split_physical_axes,
      )
  else:
    if allow_split_physical_axes:
      if max_utils.is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
        mesh = mesh_utils.create_device_mesh(
            [16, 16],
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=False,
        )
        mesh = max_utils.reshape_mesh_to_rings(mesh, config.custom_mesh)
        mesh = np.reshape(mesh, ici_parallelism)
      else:
        mesh = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
      mesh = mesh_utils.create_device_mesh(
          ici_parallelism,
          devices,
      )
      if config.optimize_mesh_for_tpu_v6e:
        mesh = max_utils.optimize_mesh_for_tpu_v6e(mesh, devices)

  max_logging.log(f"Num_devices: {num_devices}, shape {mesh.shape}")

  return mesh


# Learning Rate Schedule
# -----------------------------------------------------------------------------


def create_learning_rate_schedule(config):
  """Creates a warmup and cosine decay learning rate schedule:
  We take inspiration from Llama2's learning rate (LR) schedule, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  Learning rate schedule has either two or three parts:
  1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
  2) Cosine from [learning_rate] to [learning_rate * cosine_learning_rate_final_fraction] until learning_rate_schedule_steps
  3) Constant learning rate of 0 from learning_rate_schedule_steps to steps.
  The zero learning rate section can be used to more accurately measure the fully trained model's performance.
  """

  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = (step) / len_steps
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr

    return schedule

  lr = config.learning_rate
  cos_final_lr = lr * config.cosine_learning_rate_final_fraction

  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  cos_steps = config.learning_rate_schedule_steps - warmup_steps
  constant_zero_steps = config.steps - config.learning_rate_schedule_steps

  warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
  cos_schedule = make_cos_schedule(lr, cos_final_lr, cos_steps)
  constant_schedule = optax.constant_schedule(0.0)

  pieces = [warmup_schedule, cos_schedule]
  boundaries = [
      warmup_steps,
      warmup_steps + cos_steps,
  ]

  if constant_zero_steps > 0:
    pieces.append(constant_schedule)
    boundaries.append(warmup_steps + cos_steps + constant_zero_steps)

  return optax.join_schedules(pieces, boundaries)
