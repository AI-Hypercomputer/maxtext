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

# pylint: disable=line-too-long, disable=bare-except, consider-using-generator
""" Utils that are only interesting to MaxText. """

from typing import Optional
import functools
import pickle

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state

import numpy as np

from collections.abc import Iterable
from jax.experimental import mesh_utils
from jax.experimental.serialize_executable import deserialize_and_load
from jax.sharding import PartitionSpec as P

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

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
NUM_TILES_PER_IMAGE = 5  # Fake number of tiles for llama4, init purpose


def get_input_data_sharding(config, mesh):
  """Get the input data sharding for the model"""
  return nn.logical_to_mesh_sharding(P(*config.input_data_sharding_logical_axes), mesh, config.logical_axis_rules)

def get_functional_train_with_signature(train_step,mesh, data_sharding, state_mesh_shardings, model, config,params_shardings):
  """Get the shardings (both state and data) for `train_step`."""
  functional_train = functools.partial(train_step, model, config,mesh, state_mesh_shardings,params_shardings)
  functional_train.__name__ = "train_step"
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  
  # Ensure out_shardings are NamedSharding objects
  out_shardings = (
      state_mesh_shardings,
      jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), 
                             {'scalar': 1, 'scalars': 1}) # Create dummy structure for metrics
  )

  static_argnums = ()  # We partial out the static argnums of model and config
  donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for `eval_step`."""
  functional_eval = functools.partial(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = None  # metrics
  static_argnums = ()  # We partial out the static argnums of model, config
  donate_argnums = ()  # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums


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
  if config.use_multimodal:
    image_shape = get_dummy_image_shape_for_init(config)
    shaped_batch["images"] = jax.ShapeDtypeStruct(image_shape, jnp.int32)
  return shaped_batch


def get_dummy_image_shape_for_init(config):
  """Return the shape of the dummy image for specific model's initialization."""
  image_shape = ()
  if config.model_name.startswith("gemma3"):
    image_shape = (
        config.micro_batch_size_to_train_on,
        NUM_IMAGES_PER_SEQUENCE,
        config.image_size_for_vit,
        config.image_size_for_vit,
        NUM_IMAGE_CHANNELS,
    )
  elif config.model_name.startswith("llama4"):
    image_shape = (
        config.micro_batch_size_to_train_on,
        NUM_TILES_PER_IMAGE,
        NUM_IMAGE_CHANNELS,
        config.tile_size_for_vit,
        config.tile_size_for_vit,
    )
  return image_shape


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
  noncausal_attention_flops = (
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
  causal_attention_flops = noncausal_attention_flops / 2
  attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12

  # multiply num_decoder_layers by 2 because we combine [local_attention, global_attention] into one decoder layer
  learnable_weight_tflops = (
      ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers * 2 + embedding_flops) * 3 / 10**12
  )

  return attention_tflops, learnable_weight_tflops


def calculate_gemma3_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
  """
  Calculate training TFLOPs for Gemma3, which has an alternating pattern of
  5 local attention layers and 1 global attention layer.
  """
  num_layers = config.num_decoder_layers

  num_global_layers = num_layers // 6
  num_local_layers = num_layers - num_global_layers

  # FLOPs for a single global attention layer (full attention)
  # Formula: 4 * batch_size * seq_len^2 * num_heads * head_dim
  global_attention_flops_per_layer = (
      4 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
  )

  # FLOPs for a single local attention layer (sliding window)
  # Formula: 4 * batch_size * seq_len * window_size * num_heads * head_dim
  local_attention_flops_per_layer = (
      4
      * config.per_device_batch_size
      * config.max_target_length
      * min(config.sliding_window_size, config.max_target_length)
      * config.num_query_heads
      * config.head_dim
  )

  # Total attention FLOPs = (num_global_layers * FLOPs_per_global) + (num_local_layers * FLOPs_per_local)
  noncausal_attention_flops = (
      num_global_layers * global_attention_flops_per_layer + num_local_layers * local_attention_flops_per_layer
  )
  causal_attention_flops = noncausal_attention_flops / 2

  # Convert to TFLOPs and multiply by 3 for fwd/bwd pass
  attention_tflops = causal_attention_flops * 3 / 10**12

  # Learnable weights (FFN, QKV, Projections) are present in every layer.
  learnable_weight_tflops = ((total_ffn_flops + qkv_flops + projection_flops) * num_layers + embedding_flops) * 3 / 10**12

  return attention_tflops, learnable_weight_tflops


def _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size):
  """Calculates the non-causal FLOPs for a single layer of chunked attention."""
  num_chunks = seq_len // chunk_size
  rem_chunk_size = seq_len % chunk_size
  # The complexity of chunked attention is the sum of squares of chunk lengths.
  chunked_complexity = (num_chunks * chunk_size**2) + (rem_chunk_size**2)
  # The formula for non-causal attention FLOPs is 4 * B * complexity * H * D,
  # where B=batch_size, H=num_heads, D=head_dim.
  return 4 * config.per_device_batch_size * chunked_complexity * config.num_query_heads * config.head_dim


def calculate_llama4_attention_tflops(config):
  """
  Calculates attention-only training TFLOPs for Llama4's specific architecture,
  which has an alternating pattern of global and chunked attention layers.
  """
  num_layers = config.num_decoder_layers
  seq_len = config.max_target_length
  chunk_size = config.chunk_attn_window_size

  # Determine number of global vs. chunked layers based on the NoPE interval.
  # A "NoPE" layer uses global attention.
  num_global_layers = num_layers // config.nope_layer_interval
  num_chunked_layers = num_layers - num_global_layers

  # FLOPs for a single global attention layer (full attention, non-causal)
  global_attention_flops_per_layer = 4 * config.per_device_batch_size * seq_len**2 * config.num_query_heads * config.head_dim

  # FLOPs for a single chunked attention layer (non-causal)
  chunked_attention_flops_per_layer = _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size)

  # Total non-causal attention FLOPs is the sum of all global and all chunked layers
  noncausal_attention_flops = (num_global_layers * global_attention_flops_per_layer) + (
      num_chunked_layers * chunked_attention_flops_per_layer
  )

  # Apply causal mask and convert to TFLOPs (multiply by 3 for fwd/bwd pass)
  causal_attention_flops = noncausal_attention_flops / 2
  attention_tflops = causal_attention_flops * 3 / 10**12

  return attention_tflops


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
    qkv_flops, noncausal_attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
  else:
    qkv_flops = (
        2
        * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * (config.num_query_heads + 2 * config.num_kv_heads)
        * config.head_dim
    )
    noncausal_attention_flops = (
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

  # Divide attantion flops by 2 due to causal mask
  # References:
  # NVIDIA/Megatron-LM (2025 March): https://github.com/NVIDIA/Megatron-LM/blob/250b79415dcc4b660521273c87f15334c804eeae/megatron/training/training.py#L361-L362
  # NVIDIA/NeMo (2025 April): https://github.com/NVIDIA/NeMo/blob/ba4d6d116463de512ff0cfc14641aa6cf4577a42/nemo/utils/flops_formulas.py#L259-L272
  causal_attention_flops = noncausal_attention_flops / 2

  # Embedding flops
  embedding_flops = 2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.vocab_size

  # Combine flops with number of decoder layers
  if config.decoder_block == DecoderBlockType.GEMMA2:
    attention_tflops, learnable_weight_tflops = calculate_gemma2_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.GEMMA3:
    attention_tflops, learnable_weight_tflops = calculate_gemma3_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops
    )
  elif config.decoder_block == DecoderBlockType.LLAMA4:
    # Use the new helper to calculate attention TFLOPs correctly.
    attention_tflops = calculate_llama4_attention_tflops(config)
    # The learnable weight calculation remains the same as it correctly handles Llama4's MoE structure.
    learnable_weight_tflops = (
        (total_ffn_flops + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
  elif config.decoder_block == DecoderBlockType.DEEPSEEK:
    learnable_weight_tflops = (
        (total_ffn_flops + (qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12
  else:
    # multiply by 3 for both feed forward and back propagation flops
    learnable_weight_tflops = (
        ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers + embedding_flops) * 3 / 10**12
    )
    attention_tflops = causal_attention_flops * config.num_decoder_layers * 3 / 10**12

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
  noncausal_attention_flops = (
      4
      * config.num_query_heads
      * config.num_decoder_layers
      * config.head_dim
      * prefill_length**2
      / jax.device_count()
      / 1e12
  )
  causal_attention_tflops = noncausal_attention_flops / 2  # due to causality in attention
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


def get_mesh_axes_used_by_tensor_spec(tensor_sharding_spec):
  """
  Extracts the set of mesh axis names that a tensor's PartitionSpec uses.

  This function inspects a tensor's sharding specification (PartitionSpec) and
  identifies which mesh axes are actively used for sharding. If a tensor is not
  sharded (i.e., fully replicated), the resulting set will be empty.

  Args:
    tensor_sharding_spec: The PartitionSpec of a tensor, which defines how it's partitioned across the mesh.
    It can be None or contain strings and iterables representing the mesh axes.
    all_mesh_axis_names: A collection of all available mesh axis names in the current device mesh.

  Returns:
    A set of strings, where each string is a mesh axis name used by the
    tensor's sharding spec. Returns an empty set for unsharded tensors.
  """
  # Flatten the sharding spec, as it can contain nested iterables (e.g., ('data', 'mdl')).
  tensor_sharding_spec = sum(
      [
          [axis] if isinstance(axis, str) else list(axis) if isinstance(axis, Iterable) else []
          for axis in tensor_sharding_spec
      ],
      [],
  )
  return tensor_sharding_spec


def _get_nontrival_mesh_axes(mesh):
  """
  Returns mesh axes from config that are valid and have more than one shard.

  This function identifies which of the predefined potential sharding axes are
  actually present in the current device mesh and are configured with a size
  greater than one (i.e., are actually sharded).

  Args:
    mesh: The device mesh object, which contains information about the mesh topology, including axis names and their sizes.

  Returns:
    A set of strings, where each string is a mesh axis name that is both
    pre-configured as a target for sharding and has more than one shard in the mesh.
  """

  target_sharding_axes_config = [
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
  ]

  # Filter the target axes to find those that exist in the current mesh
  # and have a size greater than 1, meaning they are actually used for sharding.
  return {axis for axis in target_sharding_axes_config if axis in mesh.axis_names and mesh.shape[axis] > 1}


def _analyze_sharding(params, mesh, valid_target_mesh_axes):
  """
  Analyzes parameters to find which are unsharded on any valid mesh axis.

  This function iterates through all parameters in a model, checking their
  sharding specifications. It identifies parameters that are not sharded along any
  of the provided valid target axes (i.e., they are fully replicated across these axes).

  Args:
    params: A PyTree of model parameters.
    mesh: The device mesh object.
    valid_target_mesh_axes: A set of mesh axis names that are considered valid targets for sharding.

  Returns:
    A tuple containing:
      - unsharded_params_total_size (int): The total size (number of elements) of all parameters found to be
        unsharded on the target axes.
      - problematic_tensors_details (list): A list of dictionaries, where each
        dictionary contains details about a tensor that is not sharded on any of the target axes.
  """
  unsharded_params_total_size = 0  # Initialize a counter for the size of unsharded parameters.
  problematic_tensors_details = []  # Initialize a list to store details of problematic tensors.

  # Get a flattened list of all parameters (leaves) in the PyTree, along with their paths.
  all_params_leaves = jtu.tree_leaves_with_path(params)

  for path, p_leaf in all_params_leaves:  # Iterate over each parameter leaf
    param_name_str = jtu.keystr(path)  # Convert the tree path to a readable string

    # Check that sharding and spec exist and are valid
    sharding = getattr(p_leaf, "sharding", None)
    spec = getattr(sharding, "spec", None)
    assert sharding is not None and spec is not None and isinstance(spec, P), (
        f"Parameter '{param_name_str}' is missing a valid '.sharding.spec'."
        "Expected 'p_leaf.sharding.spec' to be a non-null 'partitionspec'."
    )

    current_sharding_spec = p_leaf.sharding.spec  # Extract the current tensor's sharding spec
    # Identify axes used for sharding
    mesh_axes_used = get_mesh_axes_used_by_tensor_spec(current_sharding_spec)
    # Check if the parameter is sharded on all the valid target axes.
    is_sharded_on_all_target_axis = all(axis in mesh_axes_used for axis in valid_target_mesh_axes)

    # If the parameter is not sharded on all of the target axes, it's considered "problematic."
    if not is_sharded_on_all_target_axis:
      unsharded_params_total_size += p_leaf.size  # Add to total unsharded parameter size
      unsharded_axes = set(valid_target_mesh_axes) - set(mesh_axes_used)
      # Add detailed info to list of problematic tensors
      problematic_tensors_details.append(
          {
              "name": param_name_str,  # Tensor name
              "size": p_leaf.size,  # tensor size
              "shape": p_leaf.shape,  # tensor shape
              "spec": str(current_sharding_spec),  # Tensor sharding spec as string
              "available_axes": sorted(list(valid_target_mesh_axes)),  # Axes that could be used for sharding
              "unsharded_axes": sorted(list(unsharded_axes)),  # Unsharded axes
          }
      )
  # Return the total size of unsharded parameters and the list of problematic tensors.
  return unsharded_params_total_size, problematic_tensors_details  # Return results


def _raise_if_unsharded_exceeds_tolerance(unsharded_size, total_size, tolerance, problematic_tensors_details):
  """
  Raises an AssertionError if the percentage of unsharded parameters exceeds the given tolerance.

  This function calculates the proportion of model parameters that are unsharded
  and compares it against a specified tolerance. If the tolerance is exceeded,
  it constructs and raises a detailed error message.

  Args:
    unsharded_size: The total size of parameters not sharded on target axes.
    total_size: The total size of all parameters in the model.
    tolerance: A float (e.g., 0.05 for 5%) representing the maximum allowed percentage of unsharded parameters.
    problematic_tensors_details: A list of details about the unsharded tensors,
    used to generate an informative error message.

  Raises:
    AssertionError: If the percentage of unsharded parameters is greater than the tolerance.
  """
  if total_size <= 0:
    raise ValueError("Total size must be greater than zero.")

  # Calculate the percentage of unsharded parameters.
  unsharded_param_perc = unsharded_size / total_size

  # If the percentage is over the tolerance, prepare and raise an error.
  if unsharded_param_perc > tolerance:
    # Sort the problematic tensors by size to show the largest ones first.
    problematic_tensors_details.sort(key=lambda x: x["size"], reverse=True)

    # Begin constructing the error message.
    error_msg_lines = [
        f"Unsharded parameter percentage ({unsharded_param_perc:.2%})" f"exceeds tolerance ({tolerance:.2%})."
    ]
    # Add a header explaining the issue.
    error_msg_lines.append(
        "The following large tensors are replicated (unsharded) but could be sharded on at "
        "least one of the available axes:"
    )
    # Add details for the top 5 largest problematic tensors.
    for detail in problematic_tensors_details[:5]:  # Show top 5 largest problematic tensors
      error_msg_lines.append(
          f" - Name: {detail['name']}(Size: {detail['size']}, Shape: {detail['spec']}, Spec: {detail['spec']}) "
          f" is unsharded on axis: {detail['unsharded_axes']}"
          f" could be sharded on: {detail['available_axes']}"
      )

    # Raise the assertion error with the combined, formatted message.
    raise AssertionError("\n".join(error_msg_lines))


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  """
  Asserts that the total size of replicated parameters is within a given tolerance.

  This is the main function that orchestrates the sharding analysis. It determines
  the total number of parameters, identifies valid sharding axes, analyzes the
  sharding of all parameters, and then raises an error if the amount of
  unsharded parameters exceeds the specified tolerance.

  Args:
    params: A PyTree of model parameters.
    mesh: The device mesh object.
    tolerance: A float representing the maximum allowed percentage of unsharded parameters.
  """
  # Calculate the total size of all parameters in the model.
  total_num_params = max_utils.calculate_bytes_from_pytree(params)

  # Get the set of nontrival mesh axes that can be used for sharding.
  valid_target_mesh_axes = _get_nontrival_mesh_axes(mesh)
  # If there are no valid axes to shard along, there's nothing to check, so we can exit.
  if not valid_target_mesh_axes:
    return  # Exit early

  # Analyze the parameters to find the total size of unsharded parameters
  # and get details on which tensors are problematic.
  unsharded_params_total_size, problematic_tensors_details = _analyze_sharding(params, mesh, valid_target_mesh_axes)

  # Check if the amount of unsharded parameters is within the tolerance and
  # raise an exception if it is not.
  _raise_if_unsharded_exceeds_tolerance(
      unsharded_params_total_size, total_num_params, tolerance, problematic_tensors_details
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
  image_shape = get_dummy_image_shape_for_init(config)
  model_vars = model.init(
      {"params": key, "dropout": key, "aqt": key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
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

  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings,params_shardings = get_abstract_state(
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
        enable_orbax_v1=config.enable_orbax_v1,
        checkpoint_conversion_fn=config.checkpoint_conversion_fn,
        source_checkpoint_layout=config.source_checkpoint_layout,
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

  return state, state_mesh_annotations, state_mesh_shardings, data_iterator,params_shardings



def add_data_to_sharding(mesh, path, aval, sharding):
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise AssertionError(
        f"Expected NamedSharding, found {sharding} of {type(sharding)=} at {jax.tree_util.keystr(path)}")
    try:
      sharded_shape = sharding.shard_shape(aval.shape)
    except Exception as e:
      raise AssertionError(
        f"Could not shard value {jax.tree_util.keystr(path)} of shape={aval.shape} with {sharding=}") from e
    pspec = sharding.spec

    if 'data' in jax.tree.leaves(pspec):
      return sharding

    for idx, (size, partition) in enumerate(zip(sharded_shape, pspec)):
      if partition is None:
        partition = ()

      if isinstance(partition, str):
        partition = (partition,)

      if size % mesh.shape['data'] == 0 and (partition is None or 'tensor' not in partition):
        added_component = ('data',) + partition
        new_pspec = jax.sharding.PartitionSpec(*(pspec[:idx] + (added_component,) + pspec[idx + 1:]))
        new_sharding = jax.sharding.NamedSharding(sharding.mesh, new_pspec)
        # return sharding.with_spec(new_pspec)
        return new_sharding
    return sharding

def maybe_update_params_sharding_with_opt(state_mesh_shardings):
  prev_params_shardings = state_mesh_shardings.params
  if isinstance(state_mesh_shardings.opt_state, optax.ScaleByAdamState):
    sharded_fp32_params = state_mesh_shardings.opt_state.mu
  elif isinstance(state_mesh_shardings.opt_state, tuple) and isinstance(state_mesh_shardings.opt_state[0],
                                                                        optax.ScaleByAdamState):
    sharded_fp32_params = state_mesh_shardings.opt_state[0].mu
  else:
    raise NotImplementedError(
      f"Could not find optimizer state shardings from optimizer of type {type(state_mesh_shardings.opt_state)}")
  if "params" not in sharded_fp32_params.keys():
    # When quantization=fp8 is enabled the sharded_fp32_params
    # are not wrapped in `params`. Here we wrap them back.
    sharded_fp32_params = {"params": sharded_fp32_params}
  state_mesh_shardings = state_mesh_shardings.replace(params=dict(prev_params_shardings, **sharded_fp32_params))
  return prev_params_shardings, state_mesh_shardings

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

  if is_training and config.optimizer_zero1:
    opt_state = jax.tree.map_with_path(
      functools.partial(add_data_to_sharding, mesh),
      max_utils.unbox_logicallypartioned(abstract_state).opt_state,
      state_mesh_shardings.opt_state
    )
    state_mesh_shardings = state_mesh_shardings.replace(opt_state=opt_state)
    # Shard params to be the same as the opt_state, keep the orginal params shardings in params_shardings
    params_shardings, state_mesh_shardings = maybe_update_params_sharding_with_opt(state_mesh_shardings)
  else:
    params_shardings=state_mesh_shardings.params

  abstract_sharded_state = jax.jit(init_state_partial, in_shardings=None, out_shardings=state_mesh_shardings).eval_shape()
  unboxed_abstract_sharded_state = max_utils.unbox_logicallypartioned(abstract_sharded_state)
  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return (
      unboxed_abstract_sharded_state,
      state_mesh_annotations,
      state_mesh_shardings,
      params_shardings,
  )



def get_prefill_kv_cache_annotations(model, config, rng, mesh, page_state: Optional[PageState] = None):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
        config.global_batch_size_to_load,
        config.max_prefill_predict_length,
    )
    image_shape = get_dummy_image_shape_for_init(config)

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
    image_shape = get_dummy_image_shape_for_init(config)

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
  if config.subslice_shape and config.enable_single_controller and config.num_slices == 1:
    max_logging.log(f"Trying to create a subslice with shape: {config.subslice_shape}")
    subslice_shape = tuple(int(x) for x in config.subslice_shape.split(","))
    device_coords = [device.coords for device in devices]
    device_coords_np = np.array(device_coords)

    # Find the minimum coordinates to start the subslice
    min_coords = device_coords_np.min(axis=0)

    subslice_devices = []
    for device in devices:
      coords = device.coords
      if all(min_coords[i] <= coords[i] < min_coords[i] + subslice_shape[i] for i in range(len(subslice_shape))):
        subslice_devices.append(device)
    devices = subslice_devices

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


def get_formatted_sharding_annotations(params, mesh=None):
  """
  Generates a readable string report of sharding annotations for all parameters.

  This function iterates through a PyTree of model parameters and inspects the
  sharding information attached to each parameter (leaf). It creates a
  human-readable summary that is useful for debugging sharding configurations.

  Args:
    params: The PyTree of model parameters to inspect.
    mesh: (Optional) The device mesh. If provided, its axis names and shape
          are included in the report for additional context.

  Returns:
    A single string containing the formatted report of sharding annotations
    for every parameter, with each entry on a new line.
  """
  # Initialize a list to hold the lines of the report, starting with a title.
  annotation_lines = ["Comprehensice Weight Sharding Annotations:"]

  # If a mesh object is provided, add its details to the report header.
  if mesh:
    annotation_lines.append(f"Mesh axes: {mesh.axis_names}, Mesh shape: {mesh.shape}")
    annotation_lines.append("-" * 30)

  # Get a flattened list of all parameters (leaves) and their corresponding paths in the PyTree.
  all_params_leaves = jtu.tree_leaves_with_path(params)

  # Loop through each parameter leaf in the flattened list.
  for path, p_leaf in all_params_leaves:
    # Convert the parameter's path (a sequence of keys) into a readable string name.
    param_name_str = jtu.keystr(path)
    # Get the shape of the parameter as a string.
    shape_str = str(p_leaf.shape)
    # Set a default description for sharding, in case none is found.
    sharding_desc = "N/A"

    # Check if the parameter leaf has a 'sharding' attribute.
    if hasattr(p_leaf, "sharding"):
      # Case 1: Standard JAX sharding with a PartitionSpec.
      if hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is not None:
        # The spec is a tuple (PartitionSpec), format it for readability.
        spec_parts = []
        for item in p_leaf.sharding.spec:
          # Represent None as "Replicated" to make it explicit.
          spec_parts.append(str(item) if item is not None else "Replicated")
        sharding_desc = f"PartitionSpec({', '.join(spec_parts)})"
      # Case 2: The parameter is explicitly marked as fully replicated.
      elif hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is None:
        sharding_desc = "Fully Replicated (spec is None)"
      # Case 3: A generic fallback if a sharding object exists but has no recognized spec attribute.
      else:
        # Print the string representation of the sharding object itself.
        sharding_desc = str(p_leaf.sharding)
    # Case 4: The parameter has no .sharding attribute at all.
    else:
      sharding_desc = "No .sharding attribute found"

    # Append the formatted details for the current parameter to our list of lines.
    annotation_lines.append(f" - Param: {param_name_str}\n" f"   Shape: {shape_str}\n" f"   Sharding: {sharding_desc}")
  # Join all the collected lines into a single string, separated by newlines.
  return "\n".join(annotation_lines)


def get_physical_spec_no_fsdp(full_logical, mesh, logical_axis_rules):
  """
  Generates a physical sharding spec for fully replicated weights.

  This function computes a target sharding layout where model parameters are fully
  replicated across the 'fsdp' mesh axis. It starts with the original logical
  sharding and removes any rules that shard along the 'fsdp' or
  'fsdp_transpose' axes.

  Replacing a sharding axis with `None` in a PartitionSpec instructs JAX to
  replicate the array data along that physical mesh dimension. The resulting
  specification is used as a target layout for an all-gather operation.

  Args:
    full_logical: A PyTree of logical PartitionSpecs for the model parameters.
    mesh: The JAX device mesh.
    logical_axis_rules: Rules for converting logical axes to physical mesh axes.

  Returns:
    A PyTree of physical `jax.sharding.NamedSharding` objects that describe a
    layout where parameters are fully gathered (replicated) across the 'fsdp'
    mesh axis.
  """

  def remove_fsdp_sharding(sharding_tree):
    """Recursively traverses the sharding tree to remove fsdp axes."""

    def _remove_fsdp_from_partition_spec(named_sharding):
      """Removes 'fsdp' and 'fsdp_transpose' from a PartitionSpec."""
      if isinstance(named_sharding, jax.sharding.NamedSharding):
        new_spec = []
        # Iterate through each axis in the original PartitionSpec.
        for axis in named_sharding.spec:
          if axis is None:
            new_spec.append(None)
          elif isinstance(axis, str):
            # If the axis is 'fsdp', replace it with None to signify replication.
            if axis not in ("fsdp", "fsdp_transpose"):
              new_spec.append(axis)
            else:
              new_spec.append(None)
          elif isinstance(axis, (list, tuple)):
            # If the axis is a collection, filter out 'fsdp'.
            new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
            new_spec.append(tuple(new_axis))
          else:
            raise ValueError(f"Unsupported_axis_type: {type(axis)}")
          # Return a new sharding object with the modified spec.
        return jax.sharding.NamedSharding(named_sharding.mesh, jax.sharding.PartitionSpec(*new_spec))
      return named_sharding

    return jax.tree.map(_remove_fsdp_from_partition_spec, sharding_tree)

  # Convert the high-level logical spec to a physical one using default rules.
  physical = nn.logical_to_mesh_sharding(full_logical, mesh=mesh, rules=logical_axis_rules)
  # Apply the function to remove the FSDP sharding, defining our target layout.
  physical_no_fsdp = remove_fsdp_sharding(physical)
  return physical_no_fsdp


def all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules):
  """Performs an all-gather on FSDP-sharded variables via a sharding constraint.
  This function triggers an all-gather operation on the model's parameters.
  It does so by applying a sharding constraint that specifies a fully
  replicated layout.

  The JAX compiler satisfies this constraint by automatically inserting the
  necessary `all-gather` collective communication operations into the
  computation graph, effectively gathering the sharded weights.

  Args:
    variables: The PyTree of model parameters, currently sharded across devices.
    sharding_info: The logical partition spec of the currently sharded `variables`.
    mesh: The JAX device mesh.
    logical_axis_rules: Rules for converting logical axes to physical mesh axes.

  Returns:
    The model's variables with the all-gather operation applied, resulting
    in the weights being fully replicated on all devices in the 'fsdp' mesh.
  """
  # Get the target physical layout (weights fully replicated).
  physical_constraint_no_fsdp = get_physical_spec_no_fsdp(sharding_info, mesh, logical_axis_rules)
  # Apply the constraint to the model's current variables. This tells JAX to
  # gather the weights into this layout.
  return jax.lax.with_sharding_constraint(variables, physical_constraint_no_fsdp)

def named_sharding_to_partition_spec(sharding):
    """Convert NamedSharding to PartitionSpec for shard_map"""
    if isinstance(sharding, jax.sharding.NamedSharding):
      return sharding.spec
    elif isinstance(sharding, dict):
      return jax.tree_util.tree_map(named_sharding_to_partition_spec, sharding)
    else:
      return sharding