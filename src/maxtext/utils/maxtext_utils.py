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

# pylint: disable=line-too-long, disable=bare-except, consider-using-generator
""" Utils that are only interesting to MaxText. """

import functools
import pickle
import os
from typing import Sequence
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.serialize_executable import deserialize_and_load

from flax import nnx, linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training.train_state import TrainState

import optax
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

from maxtext.configs import pyconfig
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE, ShardMode
from maxtext.configs import types
from maxtext.inference.page_manager import PageState
from maxtext.common import checkpointing
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import sharding
from maxtext.utils import maxtext_utils_nnx

OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"


def get_input_data_sharding(config, mesh):
  max_logging.log(
      "WARNING: Function maxtext_utils.get_input_data_sharding is deprecated. Please use sharding.get_input_data_sharding."
  )
  return sharding.get_input_data_sharding(config, mesh)


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  max_logging.log(
      "WARNING: Function maxtext_utils.assert_params_sufficiently_sharded is deprecated."
      "Please use sharding.assert_params_sufficiently_sharded."
  )
  return sharding.assert_params_sufficiently_sharded(params, mesh, tolerance)


def add_data_to_sharding(mesh, path, aval, shardings):
  max_logging.log(
      "WARNING: Function maxtext_utils.add_data_to_sharding is deprecated. Please use sharding.add_data_to_sharding."
  )
  return sharding.add_data_to_sharding(mesh, path, aval, shardings)


def maybe_update_params_sharding_with_opt(config, state_mesh_shardings):
  max_logging.log(
      "WARNING: Function maxtext_utils.maybe_update_params_sharding_with_opt is deprecated."
      "Please use sharding.maybe_update_params_sharding_with_opt."
  )
  return sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)


def all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode):
  max_logging.log(
      "WARNING: Function maxtext_utils.all_gather_over_fsdp is deprecated. Please use sharding.all_gather_over_fsdp."
  )
  return sharding.all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode)


def get_functional_train_with_signature(
    train_step, data_sharding, state_mesh_shardings, model, config, params_shardings=None
):
  """Get the shardings (both state and data) for `train_step`."""
  functional_train = functools.partial(train_step, model, config, state_mesh_shardings, params_shardings)
  functional_train.__name__ = "train_step"
  if config.pure_nnx:
    in_shardings = (state_mesh_shardings, data_sharding)  # State, batch
  else:
    in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = (state_mesh_shardings, None)  # State, metrics
  static_argnums = ()  # We partial out the static argnums of model and config
  donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for `eval_step`."""
  functional_eval = functools.partial(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  if config.pure_nnx:
    in_shardings = (state_mesh_shardings, data_sharding)  # State, batch
  else:
    in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = None  # metrics
  static_argnums = ()  # We partial out the static argnums of model, config
  donate_argnums = ()  # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums


def shard_reorder_causal_load_balanced(batch, cp_size, shard_mode):
  """Shard the output of the reordered sequence."""
  reordered = max_utils.reorder_causal_load_balanced(batch, cp_size)
  for _, v in batch.items():
    if isinstance(v, jax.Array):
      reordered = sharding.maybe_shard_with_name(reordered, v.sharding, shard_mode)
      break
  return reordered


def get_reorder_callable(cp_size, shard_mode):
  """Creates a callable that can be used with map() to reorder batches."""
  return functools.partial(shard_reorder_causal_load_balanced, cp_size=cp_size, shard_mode=shard_mode)


def get_shaped_batch(config):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator, but eval_shape doesn't work, see b/306901078."""
  if config.enable_diloco:
    batch_shape = (
        config.num_diloco_replicas,
        config.global_batch_size_to_load // config.num_diloco_replicas,
        config.max_target_length,
    )
  else:
    batch_shape = (config.global_batch_size_to_load, config.max_target_length)
  shaped_batch = {}
  shaped_batch["inputs"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  if config.use_multimodal:
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.model_name, batch_size=config.micro_batch_size_to_train_on
    )
    shaped_batch["images"] = jax.ShapeDtypeStruct(image_shape, jnp.int32)
    shaped_batch["image_masks"] = jax.ShapeDtypeStruct(image_shape[:2], jnp.int32)
  if config.use_audio:
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
    shaped_batch["audios"] = jax.ShapeDtypeStruct(audio_shape, jnp.float32)
  return shaped_batch


def should_prevent_cse_in_remat(config):
  """Determines whether to prevent common subexpression elimination (CSE) in remat.

  CSE should not be prevented when:
  1. Layers are being scanned (scan_layers=True), OR
  2. Gradient accumulation is enabled (gradient_accumulation_steps > 1) on GPU hardware

  Args:
    config: Configuration object with scan_layers, gradient_accumulation_steps, and hardware

  Returns:
    bool: True if CSE should be prevented, False otherwise
  """
  if config.scan_layers:
    return False

  if config.gradient_accumulation_steps > 1 and config.hardware in ("gpu", "gpu_multiprocess"):
    return False

  return True


def load_compiled(config, partial_train, state, execution_devices):
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
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree, execution_devices=execution_devices)
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


def calculate_mixed_attention_model_tflops_training_per_device(
    config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length
):
  """
  Calculate training TFLOPs for models with a mixed attention pattern of local
  and global attention layers, like Gemma3 and GPT-OSS.
  """
  num_layers = config.num_decoder_layers

  num_global_layers = num_layers // attention_pattern_length
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
  global_attention_flops_per_layer = (
      4 * config.per_device_batch_size * seq_len**2 * config.num_query_heads * config.head_dim
  )

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


def calculate_indexer_mask_ratio(index_topk, max_target_length):
  """
  Calculates the sparse-to-dense ratio for Indexer TFLOPs.

  The indexer evaluates all previous tokens in a causal manner until it hits
  the Top-K limit.

  Visual Representation (T=8, K=4):
  Key (S) ->
  Q1 [X . . . . . . .]  <- 1 token scored
  Q2 [X X . . . . . .]  <- 2 tokens scored
  Q3 [X X X . . . . .]  <- 3 tokens scored
  Q4 [X X X X . . . .]  <- 4 tokens scored (K limit reached)
  Q5 [X X X . X . . .]  <- 4 tokens scored
  Q6 [X X . X . X . .]  <- 4 tokens scored
  Q7 [X . X X . . X .]  <- 4 tokens scored
  Q8 [X X . X . . . X]  <- 4 tokens scored

  For MFU calculation:

  Visual Representation (T=8, K=4):
  Key (S) ->
  Q1 [X . . . . . . .]  <- 1 token scored
  Q2 [X X . . . . . .]  <- 2 tokens scored
  Q3 [X X X . . . . .]  <- 3 tokens scored
  Q4 [X X X X . . . .]  <- 4 tokens scored (K limit reached)
  Q5 [X X X X . . . .]  <- 4 tokens scored
  Q6 [X X X X . . . .]  <- 4 tokens scored
  Q7 [X X X X . . . .]  <- 4 tokens scored
  Q8 [X X X X . . . .]  <- 4 tokens scored

  Mathematical Calculation:
  - Triangle (Phase 1: 1 to K): K^2 / 2
  - Rectangle (Phase 2: K+1 to T): (T - K) * K
  - Total Active Area = TK - K^2 / 2
  - Dense Area = T^2

  Ratio = (TK - 0.5*K^2) / T^2  =>  (K/T) - 0.5*(K/T)^2
  """

  T = float(max_target_length)
  K = float(index_topk)

  ratio = K / T
  mask_multiplier = ratio - (0.5 * ratio**2)
  return mask_multiplier


def calculate_indexer_tflops_per_device(config):
  """Calculates TFLOPs for the DeepSeek Lightning Indexer (handles causal reduction)."""
  batch_len = config.per_device_batch_size * config.max_target_length

  # 1. Calculate projections flops
  # Query: [batch, seq, q_lora_rank] @ [q_lora_rank, index_n_heads, index_head_dim]
  q_flops = 2 * batch_len * config.q_lora_rank * config.index_n_heads * config.index_head_dim
  # Key: [batch, seq, emb_dim] @ [emb_dim, index_head_dim]
  k_flops = 2 * batch_len * config.emb_dim * config.index_head_dim
  # Head weight: [batch, seq, emb_dim] @ [emb_dim, index_n_heads]
  head_weight_flops = 2 * batch_len * config.emb_dim * config.index_n_heads
  proj_flops = q_flops + k_flops + head_weight_flops

  # 2. Calculate index score flops
  # QK product [batch, seq, index_n_heads, index_head_dim] @ [batch, seq, index_head_dim]
  # --> [batch, seq, seq, index_n_heads]
  qk_product_flops = 2 * batch_len * config.max_target_length * config.index_n_heads * config.index_head_dim
  # Aggregate heads [batch, seq, seq, index_n_heads] @ [batch, seq, index_n_heads]
  head_reduction_flops = 2 * batch_len * config.max_target_length * config.index_n_heads
  # Apply causal mask: Divide by 2 to account for triangular interactions
  # The mask restricts the indexer's search space prior to Top-K filtering
  scoring_flops = (qk_product_flops + head_reduction_flops) / 2

  return proj_flops, scoring_flops


def calculate_mla_tflops_per_device(config):
  """Calculate Multi-Head Latent Attention TFLOP (handles causal reduction)"""
  batch_len = config.per_device_batch_size * config.max_target_length
  qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim

  # 1. calculate mla query projection
  if config.q_lora_rank == 0:
    q_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * qk_head_dim_sum
  else:
    # calculate query down and up flops
    q_flops = (
        2
        * batch_len
        * (config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum)
    )

  # 2. calculate mla kv projection
  kv_flops = (
      2
      * batch_len
      * (
          config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
          + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim)
      )
  )
  qkv_flops = q_flops + kv_flops

  # 3. calculate attention
  if config.use_sparse_indexer and config.max_target_length > config.index_topk:
    # get indexer flops
    indexer_proj_flops, indexer_scoring_flops = calculate_indexer_tflops_per_device(config)
    qkv_flops += indexer_proj_flops

    # calculate the proportion of the T x T causal matrix that the Indexer actually explores
    # this follows the area: (TK - 0.5*K^2) / T^2 (T: max_target_length, K: index_topk)
    multiplier = calculate_indexer_mask_ratio(config.index_topk, config.max_target_length)
    attention_flops = (
        2
        * batch_len
        * config.max_target_length
        * config.num_query_heads
        * (qk_head_dim_sum + config.v_head_dim)
        * multiplier
    )
    attention_flops += indexer_scoring_flops
  else:
    # standard MLA & max_target_length <= index_topk in sparse indexer
    # in both cases, the indexer is bypassed as the causal mask remains the efficient representation
    attention_flops = (
        2 * batch_len * config.max_target_length * config.num_query_heads * (qk_head_dim_sum + config.v_head_dim)
    )
    attention_flops = attention_flops / 2
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
  elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
    num_moe_layers = config.num_decoder_layers
    num_dense_layers = 0
  else:
    raise ValueError("Currently we only support DeepSeek, Llama4, and Qwen3-Next calculation.")

  return num_dense_layers, num_moe_layers


def calculate_gated_delta_net_flops_per_device(config):
  """
  - Calculates the FLOPs for a single Gated Delta Net (Linear Attention) layer.
  - Ref: Megatron calculation for the gated delta net:
    - https://github.com/NVIDIA/Megatron-LM/blob/8f1c2f8ae53b4e3f32c0ae7f397d8b38a675eaa2/megatron/training/training.py#L513
  - Core complexity is based on the recurrent state update view (4 ops * 2 FLOPs = 8).
  """
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim

  H_k = config.gdn_num_key_heads
  H_v = config.gdn_num_value_heads
  D_k = config.gdn_key_head_dim
  D_v = config.gdn_value_head_dim
  K_conv = config.gdn_conv_kernel_dim

  K_dim = H_k * D_k
  V_dim = H_v * D_v

  # 1. Projections (Learnable Weights)
  # Represents: in_proj_qkvz (2*K + 2*V) + in_proj_ba (2*H_v)
  # We multiply by 2 for FMA (Multiply + Add)
  flops_qkvz_ba = 2 * B * S * E * (2 * K_dim + 2 * V_dim + 2 * H_v)

  # Represents: out_proj
  flops_out = 2 * B * S * E * V_dim

  flops_projections = flops_qkvz_ba + flops_out

  # 2. Convolution (Learnable Weights)
  # We multiply by 2 for FMA
  flops_conv = 2 * B * S * K_conv * (2 * K_dim + V_dim)

  # 3. Core Gated Delta Net
  # This counts 4 distinct O(D^2) operations in the recurrent update:
  #   KK^T, VK^T, S(a(I-bKK^T)), and SQ.
  # We multiply by 2 for FMA.
  # Total Core FLOPs = 2 (FMA) * 4 (Ops) * H * D^2 = 8 * H * D^2 per token.
  # We use D_k * D_v to generalize D^2 for potentially differing head dimensions.
  flops_core_per_token = H_v * (D_k * D_v) * 8
  flops_core = B * S * flops_core_per_token

  # Weights part: Projections + Conv
  gdn_weight_flops = flops_projections + flops_conv
  # Attention part: Core
  gdn_attn_flops = flops_core

  return gdn_weight_flops, gdn_attn_flops


def calculate_gemma3_vision_layers_tflops_per_device(config):
  """
  Estimate TFLOPs for Gemma3 vision encoder (ViT-style).
  Returns:
      total_tflops: Total TFLOPs (counts for fwd + bwd + optimizer)
      learnable_weight_tflops: TFLOPs from learnable weights (patch embedding, qkv, MLP, projections)
      attention_tflops: TFLOPs from attention multiplications
  """
  # Config values
  B = config.per_device_batch_size
  C = config.num_channels_for_vit
  H = W = config.image_size_for_vit  # Gemma3 default 896
  embed_dim = config.emb_dim  # text embedding dim after projection
  # Values below are hardcoded in Gemma3VisionEncoderLayer
  patch_size = 14
  hidden_dim = 1152
  intermediate_dim = 4304
  num_layers = 27
  vision_exit_pooling_window = 4

  # 1. Patch embedding (Conv2D)
  num_patches_h = H // patch_size
  num_patches_w = W // patch_size
  seq_len = num_patches_h * num_patches_w  # 64*64=4096
  patch_embed_flops = 2 * B * seq_len * (C * patch_size * patch_size) * hidden_dim

  # 2. gemma3.Encoder: num_layers * gemma3.Encoder1DBlock
  qkv_flops_per_layer = 3 * (2 * B * seq_len * hidden_dim * hidden_dim)
  attn_flops_per_layer = 4 * B * seq_len * seq_len * hidden_dim
  projection_flops_per_layer = 2 * B * seq_len * hidden_dim * hidden_dim  # projection after attention multiplication
  mlp_flops_per_layer = 2 * (2 * B * seq_len * hidden_dim * intermediate_dim)  # two fc layers
  total_attn_flops = attn_flops_per_layer * num_layers
  encoder_flops = (+qkv_flops_per_layer + projection_flops_per_layer + mlp_flops_per_layer) * num_layers

  # 4. VisionEmbedder
  seq_len_after_pooling = (num_patches_h // vision_exit_pooling_window) * (num_patches_w // vision_exit_pooling_window)
  vision_embedder_flops = 2 * B * seq_len_after_pooling * hidden_dim * embed_dim  # One linear projection

  # Learnable weights summation
  learnable_weight_flops = patch_embed_flops + encoder_flops + vision_embedder_flops

  if config.freeze_vision_encoder_params:
    learnable_weight_flops += 2 * vision_embedder_flops  # only projector is learnable, add fwd+optimizer
  else:
    learnable_weight_flops *= 3  # multiply by 3 for fwd + bwd + optimizer

  # Convert to TFLOPs
  learnable_weight_tflops = learnable_weight_flops / 1e12
  total_attn_tflops = total_attn_flops / 1e12
  total_tflops = learnable_weight_tflops + total_attn_tflops

  return total_tflops, learnable_weight_tflops, total_attn_tflops


def calculate_llama4_vision_layers_tflops_per_device(config):
  """
  Estimate TFLOPs for Llama4 vision encoder (ViT-style).
  Returns:
      total_tflops: Total TFLOPs (counts for fwd + bwd + optimizer)
      learnable_weight_tflops: TFLOPs from learnable weights (patch embedding, qkv, MLP, projections)
      attention_tflops: TFLOPs from attention multiplications
  """
  # Config values
  B = config.per_device_batch_size
  C = config.num_channels_for_vit
  H = W = config.tile_size_for_vit
  patch_size = config.patch_size_for_vit
  hidden_dim = config.hidden_size_for_vit
  intermediate_dim = config.intermediate_size_for_vit
  num_layers = config.num_hidden_layers_for_vit
  pixel_shuffle_fc1_out_dim = config.projector_input_dim_for_vit  # 4096
  pixel_shuffle_fc2_out_dim = config.projector_output_dim_for_vit  # 4096
  base_emb_dim = config.base_emb_dim
  pixel_shuffle_ratio = config.pixel_shuffle_ratio_for_vit  # 0.5
  num_patches = (H // patch_size) * (W // patch_size)  # 24*24 = 576
  pixel_shuffle_tokens = num_patches * pixel_shuffle_ratio**2  # 144

  # 1. Llama4UnfoldConvolution (flops by linear projection)
  # lax.conv_general_dilated_patches extracts patches through reshaping/indexing without flops
  # Each patch: C * patch_size * patch_size -> hidden_dim
  patch_embed_flops = 2 * B * num_patches * (C * patch_size * patch_size) * hidden_dim

  # 2. Llama4VisionEncoder: num_layers * (qkv + att_projection + mlp)
  seq_len = num_patches + 1  # +1 for class token, so 577
  qkv_flops_per_layer = 3 * (2 * B * seq_len * hidden_dim * hidden_dim)  # Q, K, V projections
  attn_flops_per_layer = 4 * B * seq_len * seq_len * hidden_dim  # Attention scores and weighted sum
  projection_flops_per_layer = 2 * B * seq_len * hidden_dim * hidden_dim  # projection after attention multiplication
  mlp_flops_per_layer = 2 * (2 * B * seq_len * hidden_dim * intermediate_dim)  # two fc layers
  total_attn_flops = attn_flops_per_layer * num_layers
  vision_encoder_flops = (+qkv_flops_per_layer + projection_flops_per_layer + mlp_flops_per_layer) * num_layers

  # 3. Llama4VisionPixelShuffleMLP
  # (B, 144, 5632) -> (B, 144, 4096) -> (B, 144, 4096)
  pixel_shuffle_fc1_flops = 2 * B * pixel_shuffle_tokens * intermediate_dim * pixel_shuffle_fc1_out_dim
  pixel_shuffle_fc2_flops = 2 * B * pixel_shuffle_tokens * pixel_shuffle_fc1_out_dim * pixel_shuffle_fc2_out_dim
  pixel_shuffle_total_flops = pixel_shuffle_fc1_flops + pixel_shuffle_fc2_flops

  # 4. Llama4MultiModalProjector: (B, 144, 5120) x (5120, base_emb_dim)
  projector_flops = 2 * B * pixel_shuffle_tokens * pixel_shuffle_fc1_out_dim * base_emb_dim

  # Learnable weights: all matmuls above
  learnable_weight_flops = patch_embed_flops + vision_encoder_flops + pixel_shuffle_total_flops + projector_flops

  if config.freeze_vision_encoder_params:
    learnable_weight_flops += 2 * projector_flops  # only projector is learnable, add fwd+optimizer
  else:
    learnable_weight_flops *= 3  # multiply by 3 for fwd + bwd + optimizer

  # Convert to TFLOPs
  learnable_weight_tflops = learnable_weight_flops / 1e12
  total_attn_tflops = total_attn_flops / 1e12
  total_tflops = learnable_weight_tflops + total_attn_tflops

  return total_tflops, learnable_weight_tflops, total_attn_tflops


def calculate_vision_encoder_tflops(config):
  """Calculate vision encoder TFLOPs per prefill step per device."""
  if config.model_name.startswith("gemma3"):
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_gemma3_vision_layers_tflops_per_device(
        config
    )
  elif config.model_name.startswith("llama4"):
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_llama4_vision_layers_tflops_per_device(
        config
    )
  else:
    max_logging.log(
        f"Vision encoder TFLOPs calculation not implemented for model {config.model_name}, counting as 0 for now."
    )
    mm_total_tflops = mm_learnable_weight_tflops = mm_attention_tflops = 0

  return mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops


def calculate_tflops_training_per_device(config, log=True):
  """Calculate training TFLOP"""
  # MLP flops
  if config.num_experts > 1:
    # calculation based on dropless implementation
    if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LLAMA4, DecoderBlockType.QWEN3_NEXT):
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
    qkv_flops, causal_attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
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

    # Divide attention flops by 2 due to causal mask
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
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=6
    )
  elif config.decoder_block == DecoderBlockType.GPT_OSS:
    attention_tflops, learnable_weight_tflops = calculate_mixed_attention_model_tflops_training_per_device(
        config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops, attention_pattern_length=2
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
  elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
    gdn_weight_flops_per_layer, gdn_attn_flops_per_layer = calculate_gated_delta_net_flops_per_device(config)
    cycle_interval = config.inhomogeneous_layer_cycle_interval
    num_full_attn_layers = config.num_decoder_layers // cycle_interval
    num_linear_attn_layers = config.num_decoder_layers - num_full_attn_layers

    # Weights TFLOPs:
    total_weights = (
        total_ffn_flops
        + embedding_flops
        + (qkv_flops + projection_flops) * num_full_attn_layers
        + gdn_weight_flops_per_layer * num_linear_attn_layers
    )
    learnable_weight_tflops = total_weights * 3 / 10**12

    # Attention TFLOPs:
    total_attn = (causal_attention_flops * num_full_attn_layers) + (gdn_attn_flops_per_layer * num_linear_attn_layers)
    attention_tflops = total_attn * 3 / 10**12
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

  if config.use_multimodal:
    # Add vision layers TFLOPs for multimodal models
    mm_total_tflops, mm_learnable_weight_tflops, mm_attention_tflops = calculate_vision_encoder_tflops(config)
    if log:
      print(
          f"{config.model_name} vision layers per train step:\n",
          f"Total TFLOPs: {mm_total_tflops:.2f} \n",
          f"split as {100 * mm_learnable_weight_tflops/mm_total_tflops:.2f}% learnable weight flops",
          f"and {100 * mm_attention_tflops/mm_total_tflops:.2f}% attention flops;\n",
          f"learnable weight {mm_learnable_weight_tflops:.2f} TFLOPs, attention {mm_attention_tflops:.2f} TFLOPs",
      )
    total_tflops += mm_total_tflops
    learnable_weight_tflops += mm_learnable_weight_tflops
    attention_tflops += mm_attention_tflops

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


def get_intermediate_value(model, nested_key, default=None, clear=False):
  """
  Retrieves an intermediate value from an NNX model. This functions has context about
  where the intermediate value is located.

  Args:
    model: The NNX model.
    nested_key: A string representing the nested key, e.g., hidden_states_norm_out
    default: The value to return if the nested key is not found.
    clear: Clears the intermediate value from the model.

  Returns:
    The value associated with the nested key, or the default value if not found.
  """
  intermediate_value = default
  match nested_key:
    case "out_projection_activations":
      if nested_key in model.decoder.layers["self_attention"]:
        intermediate_value = model.decoder.layers["self_attention"][nested_key].get_value()[-1]
        if clear:
          del model.decoder.layers["self_attention"][nested_key]
    case _:
      # Default case to handle any unknown nested keys
      raise ValueError(f"Incorrect nested_key: {nested_key}")

  return intermediate_value


def update_state_param(state, target_path, value):
  """
  Updates a specific parameter in state.params at the given path.

  Args:
      state: The current TrainState.
      target_path: A tuple of keys matching the structure inside state.params.
      value: The value to apply.
  """

  def create_jax_path(target_path):
    path = []
    for k in target_path:
      path.append(jax.tree_util.DictKey(key=k))
    return tuple(path)

  def _apply_update(path, param):
    if path == updated_target_path:
      return param + value
    return param

  updated_target_path = create_jax_path(target_path)
  new_params = jax.tree_util.tree_map_with_path(_apply_update, state.params)
  return state.replace(params=new_params)


def init_decode_state(apply_fn, params) -> TrainState:
  """Init train state with null opt state for decode."""
  state = TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})  # type: ignore
  return state


def init_training_state(apply_fn, params, tx):
  """Init train state with null opt state for decode."""
  state = TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
  return state


def init_initial_state(model, tx, config, is_training, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, is_training, key
  """
  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  image_shape = mm_processor.get_dummy_image_shape_for_init(
      config.model_name, batch_size=config.micro_batch_size_to_train_on
  )
  audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
  # Split the master key into independent keys for each RNG collection
  # Reference: https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/rng_guide.html
  params_key, dropout_key, aqt_key = jax.random.split(key, 3)

  model_vars = model.init(
      {"params": params_key, "dropout": dropout_key, "aqt": aqt_key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
      encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
      # nnx_method="no_op",
  )
  if is_training:
    return init_training_state(model.apply, model_vars, tx)
  return init_decode_state(model.apply, model_vars)


def get_abstract_param(model, config):
  """Get abstract model structure (name, shape) without materializing the weights to save memory"""
  with model.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    key = jax.random.PRNGKey(0)
    input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.model_name, batch_size=config.micro_batch_size_to_train_on
    )
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
  abstract_vars = jax.eval_shape(
      model.init,
      {"params": key, "dropout": key, "aqt": key},
      jnp.ones(input_shape, dtype=jnp.int32),
      jnp.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
      encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
  )
  return abstract_vars


def setup_decode_state(config, mesh, checkpoint_manager, init_state_fn):
  """Setup decode state by loading params from a checkpoint.
  Args:
    config: config object
    mesh: jax.devices() mesh
    checkpoint_manager: Checkpoint manager
    init_state_fn: function to initialize the model state

  Returns:
    state: state with decode params loaded from the checkpoint
    state_mesh_annotations: the mesh annotations for the state
  """
  if not config.load_parameters_path:
    # generate random params
    max_logging.log("No decode checkpoint specified - generating random weights.")
    state, state_mesh_annotations, _, _ = setup_initial_state(
        None, config, mesh, checkpoint_manager, init_state_fn, False
    )
  else:
    # Load params from checkpoint
    max_logging.log(f"Loading decode params from {config.load_parameters_path}")
    unboxed_abstract_state, state_mesh_annotations, _ = get_abstract_state(config, mesh, init_state_fn, False)
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


def setup_training_state(data_iterator, config, mesh, checkpoint_manager, init_state_fn):
  is_training = True
  return setup_initial_state(
      data_iterator,
      config,
      mesh,
      checkpoint_manager,
      init_state_fn,
      is_training,
  )


def setup_initial_state(
    data_iterator,
    config,
    mesh,
    checkpoint_manager,
    init_state_fn,
    is_training=True,
):
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    data_iterator: data iterator
    config: config object
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    init_state_fn: function to initialize the training state
    is_training: True to initialize training state, False for decode state

  Returns:
    train_state: the initialized train state. For NNX, this is a TrainStateNNX instance
    state_mesh_annotations: the mesh annotations for the train state
  """

  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
      config, mesh, init_state_fn, is_training
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
        expansion_factor_real_data=config.expansion_factor_real_data,
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
        # The update of data_iterator state happens in place, no need to assign explicitly
        state = restored["items"]

      # For NNX, convert the pure dict to nnx.State using the abstract state as template
      if config.pure_nnx:
        nnx.replace_by_pure_dict(unboxed_abstract_state, state)
        state = unboxed_abstract_state
    else:
      init_state_partial = init_state_fn
      init_state_partial.__name__ = "initialize_state"
      if config.pure_nnx:
        state = jax.jit(
            lambda: nnx.state(init_state_partial()),  # Get state only, mapping to out_sharding structure
            in_shardings=None,
            out_shardings=state_mesh_shardings,
        )()
      else:
        # pylint: disable=not-callable
        state = jax.jit(
            init_state_partial,
            in_shardings=None,
            out_shardings=state_mesh_shardings,
        )()
      if raw_params:  # If we loaded a partial state, we need to merge it.
        if config.pure_nnx:
          # raw_params should have the same sharding info as in the model
          nnx.update(state.model, raw_params)
        else:
          state = state.replace(params=raw_params)
  if not config.pure_nnx:
    state = max_utils.unbox_logicallypartioned(state)

  return state, state_mesh_annotations, state_mesh_shardings, data_iterator


def get_logical_annotations(config, mesh, init_state_fn):
  init_state_partial = init_state_fn

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_partial)
    logical_annotations = nn.get_partition_spec(abstract_state)
  return logical_annotations


def get_abstract_state(config, mesh, init_state_fn, is_training=True):
  """Get a shaped abstraction of the state (including optimizer)"""
  if config.pure_nnx:
    return get_abstract_state_nnx(config, mesh, init_state_fn, is_training)

  init_state_partial = init_state_fn

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_partial)

  state_logical_annotations = nn.get_partition_spec(abstract_state)

  state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, config.logical_axis_rules)
  if is_training and config.shard_optimizer_over_data:
    # Add data to sharding for optimizer state
    state_mesh_shardings = state_mesh_shardings.replace(
        opt_state=jax.tree.map_with_path(
            functools.partial(sharding.add_data_to_sharding, mesh),
            max_utils.unbox_logicallypartioned(abstract_state).opt_state,
            state_mesh_shardings.opt_state,
        )
    )
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
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return (
      unboxed_abstract_sharded_state,
      state_mesh_annotations,
      state_mesh_shardings,
  )


def get_abstract_state_nnx(config, mesh, nnx_init_trainstate_fn, is_training=True):
  """Calculates the abstract sharded state and memory placement for an NNX TrainState.

  This function performs an abstract trace of the NNX model and optimizer using
  `nnx.get_abstract_model`. It resolves logical sharding annotations into physical
  JAX shardings and applies memory placement optimizations such as optimizer
  sharding and host memory offloading (pinning to CPU RAM).

  Args:
    config: Configuration object containing sharding and offloading hyperparameters
      (e.g., shard_optimizer_over_data, optimizer_memory_host_offload).
    mesh: JAX physical mesh used to resolve logical axis names to physical devices.
    nnx_init_trainstate_fn: A zero-argument factory function that produces a
      TrainStateNNX instance during the abstract trace.
    is_training: Boolean indicating if the state is for training. If True,
      optimizer state is processed and memory offloading strategies are applied.

  Returns:
    A tuple containing (abstract_sharded_state, None, state_mesh_shardings):
      abstract_sharded_state: An nnx.State containing ShapeDtypeStructs with
        fully resolved physical sharding and memory_kind metadata.
      state_mesh_annotations: An nnx.State tree consisting of the raw PartitionSpec
        objects corresponding to each parameter/variable.
      state_mesh_shardings: An nnx.State tree consisting of the raw JAX
        Sharding objects corresponding to each parameter/variable.
  """
  assert nnx_init_trainstate_fn is not None, "get_abstract_state_nnx: init function must be given."

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    # Use nnx.get_abstract_model to get the abstract_state with NamedSharding info
    _, abstract_state = nnx.get_abstract_model(nnx_init_trainstate_fn, mesh)

  state_mesh_shardings = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)

  if is_training and config.shard_optimizer_over_data:
    # Add data to sharding for optimizer state
    optimizer_sharding = jax.tree_util.tree_map_with_path(
        functools.partial(sharding.add_data_to_sharding, mesh),
        abstract_state.optimizer,
        state_mesh_shardings.optimizer,
    )
    state_mesh_shardings.optimizer = optimizer_sharding
  if is_training and config.optimizer_memory_host_offload:
    optimizer_sharding = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_host,
        state_mesh_shardings.optimizer,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )
    state_mesh_shardings.optimizer = optimizer_sharding
  if is_training and config.parameter_memory_host_offload:
    assert config.param_scan_axis == 0, "You must set the scan axis 0 to enable parameter offloading."
    _, state_params, _ = nnx.split(state_mesh_shardings, nnx.Param, ...)
    state_params = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_host,
        state_params,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )
    nnx.update(state_mesh_shardings, state_params)

  abstract_sharded_state = maxtext_utils_nnx.set_named_sharding_nnx(abstract_state, state_mesh_shardings)
  state_mesh_annotations = maxtext_utils_nnx.get_partition_spec_nnx(state_mesh_shardings)
  return (
      abstract_sharded_state,
      state_mesh_annotations,
      state_mesh_shardings,
  )


def get_prefill_kv_cache_annotations(model, config, rng, mesh, page_state: None | PageState = None):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
        config.micro_batch_size_to_train_on,
        config.max_prefill_predict_length,
    )
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.model_name, batch_size=config.micro_batch_size_to_train_on
    )
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        encoder_images=jnp.ones(image_shape) if config.use_multimodal else None,
        encoder_audios=jnp.ones(audio_shape) if config.use_audio else None,
        model_mode=MODEL_MODE_PREFILL,
        slot=0,
        page_state=page_state,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def get_kv_cache_annotations(model, config, rng, mesh, page_state: None | PageState = None):
  """Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (config.micro_batch_size_to_train_on, 1)
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.model_name, batch_size=config.micro_batch_size_to_train_on
    )
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        encoder_images=jnp.ones(image_shape) if config.use_multimodal else None,
        encoder_audios=jnp.ones(audio_shape) if config.use_audio else None,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        slot=0,
        page_state=page_state,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
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
  """Creates a learning rate schedule with warmup and decay.

  Supports two schedule types:
  - Cosine: Inspired by Llama2's learning rate schedule, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  - WSD (Warmup-Stable-Decay): Maintains constant learning rate for most of training before final decay

  Schedule structure:
  1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
  2) Decay from [learning_rate] to a final value until learning_rate_schedule_steps
     - Cosine: decays to [learning_rate * learning_rate_final_fraction]
     - WSD: maintains [learning_rate] for a stable phase, then decays to [learning_rate * learning_rate_final_fraction]
       using either linear or cosine decay based on wsd_decay_style
  3) Constant learning rate of 0 from learning_rate_schedule_steps to steps.
  The zero learning rate section can be used to more accurately measure the fully trained model's performance.
  """

  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = step / (len_steps - 1) if len_steps > 1 else 1.0
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr

    return schedule

  lr = config.learning_rate
  final_lr = lr * config.learning_rate_final_fraction
  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  constant_zero_steps = config.steps - config.learning_rate_schedule_steps

  pieces = []
  boundaries = []

  if warmup_steps > 0:
    warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps - 1)
    pieces.append(warmup_schedule)
    boundaries.append(warmup_steps)

  if config.lr_schedule_type == types.LearningRateScheduleType.COSINE:
    cos_steps = config.learning_rate_schedule_steps - warmup_steps
    if cos_steps > 0:
      cos_schedule = make_cos_schedule(lr, final_lr, cos_steps)
      pieces.append(cos_schedule)
      boundaries.append(warmup_steps + cos_steps)

  else:  # WSD
    decay_steps = int(config.learning_rate_schedule_steps * config.wsd_decay_steps_fraction)
    stable_steps = config.learning_rate_schedule_steps - warmup_steps - decay_steps

    if stable_steps > 0:
      stable_schedule = optax.constant_schedule(lr)
      pieces.append(stable_schedule)
      boundaries.append(warmup_steps + stable_steps)
    if decay_steps > 0:
      # Create decay schedule based on wsd_decay_style
      if config.wsd_decay_style == types.WsdDecayStyle.LINEAR:
        decay_schedule = optax.linear_schedule(init_value=lr, end_value=final_lr, transition_steps=decay_steps - 1)
      else:  # COSINE
        decay_schedule = make_cos_schedule(lr, final_lr, decay_steps)
      pieces.append(decay_schedule)
      boundaries.append(warmup_steps + stable_steps + decay_steps)

  if constant_zero_steps > 0:
    constant_schedule = optax.constant_schedule(0.0)
    pieces.append(constant_schedule)
    boundaries.append(config.learning_rate_schedule_steps)

  return optax.join_schedules(pieces, boundaries)


def print_shardings_params(params, params_sharding, mesh, logical_annotations=None):
  """
  Print state shardings comparing Logical Definition vs Physical Result.
  """
  if not hasattr(params, "params"):
    params = {"params": params}
  if not hasattr(params_sharding, "params"):
    params_sharding = {"params": params_sharding}
  if logical_annotations and not hasattr(logical_annotations, "params"):
    logical_annotations = {"params": logical_annotations}

  leaves_params, _ = jax.tree_util.tree_flatten_with_path(params)
  leaves_sharding, _ = jax.tree_util.tree_flatten_with_path(params_sharding)
  leaves_logical, _ = jax.tree_util.tree_flatten_with_path(logical_annotations)

  for (path, leaf_val), (_, leaf_sharding), (_, leaf_logical_val) in zip(leaves_params, leaves_sharding, leaves_logical):
    path_str = "/".join(str(p.key if hasattr(p, "key") else p.name) for p in path)
    shape = jax.typeof(leaf_val)
    pspec = sharding.remove_size_one_mesh_axis(leaf_sharding.spec, mesh)
    pspec_str = str(tuple(pspec))
    logical_str = str(leaf_logical_val)

    message = f" {path_str}\n" f"    Shape:     {shape}\n" f"    Logical:   {logical_str}\n" f"    Physical:  {pspec_str}"
    max_logging.info(message)

  print(flush=True)


def maybe_dump_jaxpr(config, p_train_step, train_step_inputs):
  """Dump jaxpr to local then upload to GCS."""
  if not config.dump_jaxpr:
    return
  max_logging.log("Tracing train_step to jaxpr...")

  # We use the p_train_step (the JIT-decorated function)
  p_train_jaxpr = jax.make_jaxpr(p_train_step)(*train_step_inputs)

  local_filename = "train_step.jaxpr"
  local_path = os.path.join(config.dump_jaxpr_local_dir, local_filename)

  os.makedirs(config.dump_jaxpr_local_dir, exist_ok=True)

  # pylint: disable=unspecified-encoding
  with open(local_path, "w") as f:
    f.write(str(p_train_jaxpr))

  max_logging.log(f"Jaxpr dumped locally to {local_path}")

  if config.dump_jaxpr_gcs_dir:
    gcs_utils.upload_dump(
        config.dump_jaxpr_local_dir,
        config.dump_jaxpr_gcs_dir,
        module_name=local_filename,
        delete_local_after=config.dump_jaxpr_delete_local_after,  # Keeping local for debugging
        all_host_upload=False,  # Only upload from lead host (Host 0)
    )


def get_mesh_from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
) -> Mesh:
  """
  Geh mesh from the configuration.

  Args:
    config: the configuration
    devices: the devices

  Returns:
    the device mesh
  """
  devices_array = create_device_mesh(config, devices)

  if config.shard_mode == ShardMode.EXPLICIT:
    axis_types = tuple([AxisType.Explicit] * len(config.mesh_axes))
  else:
    axis_types = tuple([AxisType.Auto] * len(config.mesh_axes))

  return Mesh(devices_array, config.mesh_axes, axis_types=axis_types)
