# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in Llama/DeepSeek (V3, V4) 1F1B accumulation for the normal MaxText train step.

The decoder's backward/next-forward layer pair shares one scan body. Embedding
and loss/head differentiation happen at its boundaries, with fixed parameters
throughout the accumulated step. There is no optimizer or host dispatch here.
"""

import re

from absl import logging
from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp

from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from maxtext.experimental.dualpipe_schedule import PER_LAYER_KEY, make_training_schedule
from maxtext.utils.globals import EPS
from maxtext.utils.max_utils import model_loop_unroll_context


def _routed_bias_updates(config):
  """Whether the loss-free-balancing router bias is updated each step."""
  return bool(getattr(config, "routed_bias", False)) and getattr(config, "routed_bias_update_rate", 0.0) > 0.0


def _lb_loss_enabled(config):
  return getattr(config, "load_balance_loss_weight", 0.0) > 0.0


def validate_training_config(config):
  """Require deterministic layers and stateless TE quantization recipes."""
  is_deepseek = config.decoder_block == DecoderBlockType.DEEPSEEK
  is_deepseek4 = _is_deepseek4(config)
  if config.decoder_block not in (DecoderBlockType.LLAMA2, DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4):
    raise ValueError("dual_pipe currently supports decoder_block=llama2, deepseek or deepseek4")
  if not config.scan_layers or config.inhomogeneous_layer_cycle_interval != 1:
    raise ValueError("dual_pipe requires scan_layers=true and inhomogeneous_layer_cycle_interval=1")
  if config.num_decoder_layers < 1:
    raise ValueError("At least one decoder layer is required")
  if config.shard_mode != ShardMode.AUTO:
    raise ValueError("dual_pipe currently requires shard_mode=auto")
  if config.dropout_rate != 0:
    raise ValueError("dual_pipe requires dropout_rate=0; layer state is read-only")
  if config.mtp_num_layers != 0:
    raise ValueError("dual_pipe requires mtp_num_layers=0")
  if is_deepseek4:
    # The hash-routed prefix and the scanned blocks follow the decoder's own layout (see _deepseek4_layout).
    if _indexer_loss_enabled(config) and not getattr(config, "indexer_sparse_training", False):
      raise ValueError(
          "dual_pipe does not support the indexer dense warm-up (use_indexer with indexer_loss_scaling_factor>0 and "
          "indexer_sparse_training=false): it trains only the indexer and skips the LM loss"
      )
    if not _layer_groups(config):
      raise ValueError("DeepSeek V4 dual_pipe needs at least one prefix layer or one two-layer block")
  elif not is_deepseek:
    if config.quantization not in ("", "te_no_quant"):
      raise ValueError("Llama dual_pipe requires quantization='' or te_no_quant")
    if config.num_experts != 1 or getattr(config, "te_moe_block", False):
      raise ValueError("Llama dual_pipe currently supports dense layers only")
  if getattr(config, "training_objective", "causal_lm") != "causal_lm":
    raise ValueError("dual_pipe currently supports training_objective=causal_lm only")
  if getattr(config, "attention_type", "global") == "block_diffusion":
    raise ValueError("dual_pipe does not support block-diffusion attention with causal-LM targets")
  if config.num_vocab_tiling != 1:
    raise ValueError("dual_pipe currently requires num_vocab_tiling=1")
  unsupported = (
      "use_qwix_quantization",
      "use_manual_quantization",
      "quantize_kvcache",
      "record_internal_nn_metrics",
      "parameter_memory_host_offload",
      "using_pipeline_parallelism",
      "use_batch_split_schedule",
      "use_multimodal",
      "use_audio",
      "learn_to_init_mode",
      "use_tunix_gradient_accumulation",
      "use_dpo",
      "shard_optimizer_over_data",
      "optimizer_memory_host_offload",
      "use_indexer",
      "enable_diloco",
      "retry_when_tokens_dropped",
      "use_qk_clip",
      "engram_layers",
  )
  # DeepSeek V4's CSA indexer is supported; the V3.2 (MLA) indexer is not.
  enabled = [
      name for name in unsupported if getattr(config, name, False) and not (is_deepseek4 and name == "use_indexer")
  ]
  if not (is_deepseek or is_deepseek4) and getattr(config, "routed_bias", False):
    enabled.append("routed_bias")
  if getattr(getattr(config, "lora", None), "enable_lora", False):
    enabled.append("lora.enable_lora")
  # The fallback modes need MoE overflow/fallback aux that this schedule does not produce: "layer" would fail in
  # train_step, and "step" would never roll back because has_moe_overflow is a constant False.
  if getattr(config, "moe_dropless_fallback", None):
    enabled.append(f"moe_dropless_fallback={config.moe_dropless_fallback}")
  if enabled:
    raise ValueError(f"Unsupported with gradient_accumulation_schedule=dual_pipe: {', '.join(enabled)}")
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("dual_pipe requires a JAX build providing jax.fwd_and_bwd")


def _remat_enabled(config):
  """Whether decoder layers are checkpointed; pyconfig turns remat_policy=none into None."""
  return bool(config.remat_policy) and config.remat_policy.lower() != "none"


def _is_deepseek4(config):
  return config.decoder_block == DecoderBlockType.DEEPSEEK4


def _is_unscanned_name(name):
  """DeepSeek V4's hash-routed prefix layers (layers_0, layers_1, ...) are separate, unscanned modules."""
  return re.fullmatch(r"layers_\d+", name) is not None


def _deepseek4_layout(config):
  """(prefix layer count, scanned two-layer block count) exactly as the decoder builds them.

  The decoder unrolls `first_num_hash_layers` prefix layers, then scans `(num_decoder_layers - prefix) // 2`
  two-layer blocks (HCA + CSA). The floor division means an odd remainder leaves one trailing layer that the
  decoder never creates or applies, and a model with fewer layers than prefix layers has no blocks at all.
  This mirrors that rounding instead of re-deriving the layer count, so dual_pipe runs the same layers as serial.
  """
  prefix = config.first_num_hash_layers
  remaining = config.num_decoder_layers - prefix
  blocks = max(remaining // 2, 0)
  if remaining % 2 or remaining < 0:
    logging.warning(
        "dual_pipe: DeepSeek V4 has %d layers after the %d hash-routed prefix layers; the decoder scans floor(%d / 2) = "
        "%d two-layer blocks and ignores the remainder, and so does dual_pipe.",
        remaining,
        prefix,
        remaining,
        blocks,
    )
  return prefix, blocks


def _layer_groups(config):
  """Names and lengths of contiguous homogeneous stacks, in forward order."""
  if _is_deepseek4(config):
    prefix, blocks = _deepseek4_layout(config)
    groups = tuple((f"layers_{i}", 1) for i in range(prefix)) + (("scanned_blocks", blocks),)
    return tuple((name, count) for name, count in groups if count)
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    groups = (
        ("dense_layers", config.first_num_dense_layers),
        ("moe_layers", config.num_decoder_layers - config.first_num_dense_layers),
    )
    return tuple((name, count) for name, count in groups if count)
  return (("layers", config.num_decoder_layers),)


LB_LOSS_KEY = "moe_lb_loss"
INDEXER_LOSS_KEY = "indexer_loss"
BIAS_UPDATES_KEY = "moe_bias_updates"
# Differentiable per-layer objective term: the layer's weighted load-balance and indexer losses.
AUX_OBJECTIVE_KEY = "aux_objective"
V4_BLOCK_LAYERS = ("layers_0", "layers_1")


def _indexer_loss_enabled(config):
  return bool(getattr(config, "use_indexer", False)) and getattr(config, "indexer_loss_scaling_factor", 0.0) > 0.0


def _flatten_sown(state):
  """{path tuple: value} of everything sown into an NNX State (nested modules give longer paths)."""
  return traverse_util.flatten_dict(state.to_pure_dict())


def _sown_by_suffix(flat, key):
  """[(path to the module that sowed it, array)] for every sown value named `key` (sow stores tuples)."""
  return [(path[:-1], leaf) for path, value in flat.items() if path[-1] == key for leaf in jax.tree.leaves(value)]


def _bias_metric_key(module_path):
  return BIAS_UPDATES_KEY if not module_path else f"{BIAS_UPDATES_KEY}:{'/'.join(module_path)}"


def _bias_metric_keys(config):
  """Per-layer metric keys under which the routed-bias updates travel through the schedule."""
  if _is_deepseek4(config):
    return tuple(_bias_metric_key((name,)) for name in V4_BLOCK_LAYERS)
  return (BIAS_UPDATES_KEY,)


def _lb_layer_scale(config, groups):
  """Weight of each MoE layer's load-balance loss in the objective: 1/#MoE layers (the mean serial GA takes
  over layers), or 1 for Megatron's seq_aux_loss, which sums over layers.

  DeepSeek V4's hash-routed prefix layers have no load-balance loss, so only the scanned blocks' two layers count.
  """
  from maxtext.layers.moe import uses_megatron_seq_aux_loss  # pylint: disable=import-outside-toplevel

  counts = dict(groups)
  num_moe_layers = (
      len(V4_BLOCK_LAYERS) * counts.get("scanned_blocks", 0) if _is_deepseek4(config) else counts.get("moe_layers", 0)
  )
  if num_moe_layers < 1:
    if _is_deepseek4(config):  # only hash-routed prefix layers: nothing sows a load-balance loss, as in serial
      return 1.0
    raise ValueError("load_balance_loss_weight>0 requires at least one MoE layer")
  return 1.0 if uses_megatron_seq_aux_loss(config) else 1.0 / num_moe_layers


def _indexer_layer_scale(config, groups):
  """Weight of each CSA layer's indexer loss: serial GA averages it over the layers that sow one.

  Those are the compress_ratio=4 layers: the prefix layers whose ratio is 4 and the second layer of every block.
  """
  counts = dict(groups)
  ratios = list(getattr(config, "compress_ratios", ()) or ())
  prefix_csa = sum(1 for i in range(config.first_num_hash_layers) if i < len(ratios) and ratios[i] == 4)
  csa_layers = prefix_csa + counts.get("scanned_blocks", 0)
  # With no CSA layer nothing sows an indexer loss, as in serial; the scale is then unused.
  return 1.0 / max(csa_layers, 1)


def _token_count_cotangent(microbatch):
  """Per-microbatch weight of the auxiliary losses: the microbatch's token count, as in serial GA (whose summed
  objective is divided by the step's token count). The per-layer scales are applied inside each layer."""
  return jnp.sum(microbatch["targets_segmentation"] != 0).astype(jnp.float32)


def _finalize_bias_updates(per_layer, layer_names, config):
  """The `moe_bias_updates` aux train_step expects, from per-layer updates summed over microbatches.

  Like serial GA, summed integer expert counts become one full-batch sign update; float updates (already
  per-microbatch steps, e.g. from TE) stay summed.
  """
  return (_ga().finalize_bias_update(per_layer[layer_names.index("moe_layers")][BIAS_UPDATES_KEY], config),)


def _deepseek4_bias_intermediates(per_layer, layer_names, config):
  """DeepSeek V4's routed-bias updates in the nested layout train_step reads from `intermediate_outputs`.

  Each scanned block's two layers have their own router bias, stacked over blocks: [blocks, experts] at
  decoder/scanned_blocks/layers_{0,1}/moe_bias_updates, the paths serial training sows them at.
  """
  if "scanned_blocks" not in layer_names:  # only hash-routed prefix layers: no routed bias to update
    return {}
  group = per_layer[layer_names.index("scanned_blocks")]
  return {
      "decoder": {
          "scanned_blocks": {
              name: {BIAS_UPDATES_KEY: _ga().finalize_bias_update(group[_bias_metric_key((name,))], config)}
              for name in V4_BLOCK_LAYERS
          }
      }
  }


def _te_layer_metrics(intermediates, required):
  """Read the real TE counters sown by DeepSeekGenericLayer.post_process, wherever they were sown.

  A plain layer sows them at the top level. A DeepSeek V4 scanned block holds two layers, so its counters sit under
  `layers_0` and `layers_1`: they are combined here (any overflow, the largest receive count, the smallest capacity)
  exactly as the schedule combines them across layers and microbatches. Accepts an NNX State or a (nested) dict.
  """
  keys = ("te_moe_capacity_overflow", "te_moe_total_recv_tokens", "te_moe_recv_capacity_per_rank")
  if isinstance(intermediates, nnx.State):
    flat = _flatten_sown(intermediates)
  else:
    flat = traverse_util.flatten_dict(intermediates)
  found = [[leaf for _, leaf in _sown_by_suffix(flat, key)] for key in keys]
  if not all(found):
    if required:
      raise ValueError("TE MoE layer did not produce receive-capacity intermediates")
    return {
        "te_moe_capacity_overflow": jnp.bool_(False),
        "te_moe_max_total_recv_tokens": jnp.int32(0),
        "te_moe_recv_capacity_per_rank": jnp.int32(jnp.iinfo(jnp.int32).max),
    }
  values = [jnp.concatenate([jnp.ravel(x) for x in leaves]) for leaves in found]
  return {
      "te_moe_capacity_overflow": jnp.any(values[0]),
      "te_moe_max_total_recv_tokens": jnp.max(values[1]),
      "te_moe_recv_capacity_per_rank": jnp.min(values[2]),
  }


def _layer_stats(layer, config, layer_name, lb_scale, indexer_scale):
  """The layer's metrics: TE capacity counters, load-balance / indexer losses, routed-bias updates.

  AUX_OBJECTIVE_KEY is the one differentiable output: the layer's loss terms with their per-layer weights, which
  the schedule adds to the objective with a per-microbatch cotangent. Everything else is a plain metric. A layer
  that does not sow a term (dense layers, DeepSeek V4's hash-routed prefix layers) contributes 0 to it.
  """
  lb_on, indexer_on, bias_on = _lb_loss_enabled(config), _indexer_loss_enabled(config), _routed_bias_updates(config)
  deepseek4 = _is_deepseek4(config)
  # Layers that must sow their MoE terms: DeepSeek's MoE stack, and V4's scanned blocks (whose two layers are both
  # MoE; the hash-routed prefix layers sow only what they have, so they contribute neutral values otherwise).
  is_moe = layer_name == "moe_layers" or (deepseek4 and layer_name == "scanned_blocks")
  stats, flat, intermediates = {}, {}, None
  if config.te_moe_block or lb_on or bias_on:
    intermediates = nnx.pop(layer, nnx.Intermediate)
    flat = _flatten_sown(intermediates)
  if config.te_moe_block:
    stats.update(_te_layer_metrics(intermediates, required=is_moe))
  objective = jnp.float32(0.0)
  if lb_on:
    found = _sown_by_suffix(flat, LB_LOSS_KEY)
    if not found and is_moe:
      raise ValueError(f"MoE layer did not sow {LB_LOSS_KEY}")
    total = sum((jnp.sum(x).astype(jnp.float32) for _, x in found), jnp.float32(0.0))
    stats[LB_LOSS_KEY] = total
    objective += lb_scale * total
  if indexer_on:
    from maxtext.layers.attention_mla import indexer_losses  # pylint: disable=import-outside-toplevel

    found = _sown_by_suffix(_flatten_sown(nnx.pop(layer, indexer_losses)), INDEXER_LOSS_KEY)
    total = sum((jnp.sum(x).astype(jnp.float32) for _, x in found), jnp.float32(0.0))
    stats[INDEXER_LOSS_KEY] = total
    objective += indexer_scale * total
  if lb_on or indexer_on:
    stats[AUX_OBJECTIVE_KEY] = objective
  if bias_on and is_moe:
    found = _sown_by_suffix(flat, BIAS_UPDATES_KEY)
    keys = {_bias_metric_key(path): value for path, value in found}
    if not found or not set(keys) <= set(_bias_metric_keys(config)):
      raise ValueError(f"{layer_name} sowed routed-bias updates at {sorted(keys)}; expected {_bias_metric_keys(config)}")
    stats.update(keys)
  return stats


def _make_layer_adapter(layers, config, layer_name, layer_count, remat_policy=None):
  """Expose one homogeneous stack as a pure, single-layer function.

  The schedule uses a leading layer axis for params/state; _restore_gradients
  restores the original parameter axis afterward. Sharding metadata stays with
  each variable, with the sliced layer axis removed inside layer_apply.

  DeepSeek V4's hash-routed prefix layers are separate, unscanned modules: they get a length-1 leading axis here
  (and no scan axis to remove from their metadata), and its scanned two-layer blocks need each layer's input
  tokens, which the schedule passes in the `segments` argument.
  """
  # Keep heavy model dependencies out of the lightweight NNX bookkeeping tests.
  # pylint: disable=import-outside-toplevel
  from maxtext.models.llama2 import LlamaDecoderLayer
  from maxtext.utils import maxtext_utils, maxtext_utils_nnx

  deepseek4 = _is_deepseek4(config)
  unscanned = deepseek4 and _is_unscanned_name(layer_name)
  if deepseek4:
    from maxtext.models import deepseek4 as deepseek4_models

    layer_class = deepseek4_models.DeepSeek4DecoderLayer if unscanned else deepseek4_models.DeepSeek4ScannableBlock
  elif config.decoder_block == DecoderBlockType.DEEPSEEK:
    from maxtext.models.deepseek import DeepSeekDenseLayer, DeepSeekMoELayer

    layer_class = DeepSeekDenseLayer if layer_name == "dense_layers" else DeepSeekMoELayer
  else:
    layer_class = LlamaDecoderLayer
  if not isinstance(layers, layer_class):
    raise TypeError(f"Expected a {layer_class.__name__} for {layer_name}")
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  if unscanned:
    params = jax.tree.map(lambda x: jnp.asarray(x)[None], params)
    state = jax.tree.map(lambda x: jnp.asarray(x)[None], state)
  else:
    if config.param_scan_axis != 0:
      params = jax.tree.map(
          lambda x: jnp.moveaxis(x, config.param_scan_axis, 0) if x.ndim > config.param_scan_axis else x, params
      )
    params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, layer_count)
    state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, layer_count)

  groups = _layer_groups(config)
  lb_scale = _lb_layer_scale(config, groups) if _lb_loss_enabled(config) else 1.0
  indexer_scale = _indexer_layer_scale(config, groups) if _indexer_loss_enabled(config) else 1.0

  def layer_apply(weights, hidden, layer_state, positions, segments):
    if not unscanned:
      weights, layer_state = maxtext_utils_nnx.nnx_remove_scan_axis((weights, layer_state), layer_name)
    layer = nnx.merge(graphdef, weights, layer_state, copy=True)
    extra = {}
    if deepseek4:  # `segments` is {"segments": ..., "inputs": token ids}: hash routing needs the tokens
      segments, extra["decoder_input_tokens"] = segments["segments"], segments["inputs"]
    hidden, _ = layer(
        hidden,
        decoder_segment_ids=segments,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        **extra,
    )
    return hidden, _layer_stats(layer, config, layer_name, lb_scale, indexer_scale)

  if _remat_enabled(config):
    # Same checkpointing as the regular decoder's scanned layers: any remat
    # policy (full, custom with device/offload/remat per tensor, presets). The
    # schedule keeps only the residuals this policy saves. DeepSeek V4's scanned
    # blocks also checkpoint each layer internally; this outer checkpoint keeps
    # the schedule from saving that inner scan's copies of the layer weights.
    layer_apply = jax.checkpoint(
        layer_apply,
        policy=remat_policy if remat_policy is not None else jax.checkpoint_policies.nothing_saveable,
        prevent_cse=maxtext_utils.should_prevent_cse_in_remat(config),
    )
  return layer_apply, params, state


def _make_boundaries(model, config, loss_from_logits, layer_names=("layers",)):
  """Reuse MaxText's actual embedding/head methods without carrying layer weights.

  The copy changes only Python graph structure, not the caller's model. Both
  boundaries share the same parameter tree, so tied embedding gradients add.
  Zero dropout and the allowed stateless recipes do not advance mutable state.
  """
  boundary_model = nnx.clone(model)
  for name in layer_names:
    delattr(boundary_model.decoder, name)
  graphdef, params, state = nnx.split(boundary_model, nnx.Param, ...)

  def prefix_apply(weights, batch):
    local_model = nnx.merge(graphdef, weights, state, copy=True)
    hidden = local_model.decoder._apply_embedding(  # pylint: disable=protected-access
        local_model.token_embedder,
        batch["inputs"],
        batch["inputs_position"],
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    if getattr(config, "mhc_expansion_rate", 1) > 1:
      # The decoder expands the embedding into the hyper-connection streams after _apply_embedding, not inside it:
      # [batch, seq, emb] -> [batch, seq, mhc_expansion_rate, emb].
      from maxtext.layers import mhc  # pylint: disable=import-outside-toplevel

      hidden = mhc.get_functions(config.mhc_expansion_rate)[0](hidden)
    return hidden

  def loss_apply(weights, hidden, batch):
    local_model = nnx.merge(graphdef, weights, state, copy=True)
    logits = local_model.decoder.apply_output_head(
        local_model.token_embedder, hidden, deterministic=True, model_mode=MODEL_MODE_TRAIN
    )
    xent_sum, z_loss_sum, total_weights = loss_from_logits(logits, batch, config, local_model.mesh)
    # Normal GA sums per-microbatch normalized z-loss metrics. The objective
    # already includes z-loss regularization; do not add this metric to it.
    z_loss = z_loss_sum / (total_weights + EPS)
    return xent_sum, {"xent_sum": xent_sum, "z_loss": z_loss, "total_weights": total_weights}

  return prefix_apply, loss_apply, params


def _ga():
  """The shared gradient-accumulation helpers, imported on use (the module pulls in the MoE kernels)."""
  from maxtext.utils import gradient_accumulation  # pylint: disable=import-outside-toplevel

  return gradient_accumulation


def _microbatches(data, count, micro_batch_size):
  """Gradient accumulation's microbatch layout, keeping the first `micro_batch_size` rows of each microbatch."""
  required = {"inputs", "inputs_position", "inputs_segmentation", "targets", "targets_segmentation"}
  if not required.issubset(data):
    raise ValueError(f"dual_pipe requires batch keys: {sorted(required - data.keys())}")
  if "forced_routed_experts" in data:
    raise ValueError("dual_pipe does not support forced expert routing")
  microbatches = _ga().reshape_to_microbatches(data, count)
  batch_size = jax.tree.leaves(microbatches)[0].shape[1]
  if not 0 < micro_batch_size <= batch_size:
    raise ValueError("Invalid micro_batch_size_to_train_on for dual_pipe")
  return jax.tree.map(lambda x: x[:, :micro_batch_size], microbatches)


def _make_residual_constraint(mesh, shard_mode, batch_seq, debug_sharding=False):
  """Sharding constraint for saved activations stored in the dual_pipe residual pools.

  Pool buffers have no sharding of their own, so without a constraint the partitioner may give a
  pool a feature-sharded layout and reshard every stored activation. Only activations whose leading
  dims are the hidden state's [batch, seq] are known to be batch-major: their batch dim is pinned to
  the activation_batch mesh axes and every other dim is left UNCONSTRAINED, so tensor/context-parallel
  sharding from the forward carries through. Anything else (expert-sorted MoE rows, routing metadata,
  expert-major tensors, scalars) is returned unchanged.

  TODO: record the logical axes per checkpoint_name in the model code and constrain each saved
  activation with its own axes, instead of matching shapes against [batch, seq].
  """
  # pylint: disable-next=import-outside-toplevel
  from maxtext.utils.sharding import create_sharding, maybe_shard_with_pspec

  batch_axes = create_sharding(mesh, ("activation_batch",)).spec
  batch_axes = batch_axes[0] if len(batch_axes) else None
  batch_seq = tuple(batch_seq)

  def constrain(x):
    if x.ndim < 2 or tuple(x.shape[:2]) != batch_seq:
      return x
    pspec = jax.sharding.PartitionSpec(batch_axes, *([jax.sharding.PartitionSpec.UNCONSTRAINED] * (x.ndim - 1)))
    return maybe_shard_with_pspec(x, pspec, mesh, shard_mode, debug_sharding=debug_sharding)

  return constrain


def _restore_gradients(layer_grads, boundary_grads, param_scan_axis, layer_names=("layers",)):
  """Restore the full model parameter tree and its original layer axis."""
  groups = layer_grads
  if len(groups) != len(layer_names):
    raise ValueError("Layer gradient groups do not match the decoder stack names")
  restored = []
  for name, group in zip(layer_names, groups):
    if _is_unscanned_name(name):  # a single unscanned layer: drop its length-1 leading axis
      group = jax.tree.map(lambda x: x[0], group)
    elif param_scan_axis != 0:
      group = jax.tree.map(lambda x: jnp.moveaxis(x, 0, param_scan_axis) if x.ndim > param_scan_axis else x, group)
    restored.append(group)
  return nnx.merge_state(boundary_grads, nnx.State({"decoder": dict(zip(layer_names, restored))}))


def dualpipe_loss_and_grad(config, model, params_shardings, data, loss_from_logits):
  """Drop-in loss/aux/gradient result for train_step, not a separate executable."""
  # pylint: disable-next=import-outside-toplevel
  from maxtext.utils.sharding import maybe_shard_with_name

  validate_training_config(config)
  if not isinstance(model, nnx.Module):
    raise TypeError("dual_pipe requires an NNX model")

  # Use the same input and output parameter sharding constraints as normal GA.
  # Work on a private graph so the optimizer still sees its original state.
  local_model = nnx.clone(model)
  nnx.pop(local_model, nnx.Intermediate)
  params = nnx.state(local_model, nnx.Param)
  supported_dtypes = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
  if any(p.dtype not in supported_dtypes for p in jax.tree.leaves(params)):
    raise ValueError("dual_pipe currently supports float32/bfloat16/float16 parameters, not quantized weights")

  def shard(value, spec):
    return maybe_shard_with_name(value, spec, config.shard_mode, debug_sharding=config.debug_sharding)

  params = jax.tree.map(shard, params, params_shardings)
  nnx.update(local_model, params)
  groups = _layer_groups(config)
  layer_names = tuple(name for name, _ in groups)
  deepseek4 = _is_deepseek4(config)
  if deepseek4:
    # Same policy as the decoder uses for V4's layers (offloaded residuals cannot cross its per-layer scans), and
    # the same override propagation onto the scanned blocks.
    remat_policy = local_model.decoder._get_scan_safe_remat_policy()  # pylint: disable=protected-access
    if "scanned_blocks" in layer_names:
      decoder = local_model.decoder
      decoder._sync_remat_policy_override(decoder.scanned_blocks)  # pylint: disable=protected-access
  else:
    remat_policy = local_model.decoder.get_remat_policy()
  adapters = tuple(
      _make_layer_adapter(getattr(local_model.decoder, name), config, name, count, remat_policy) for name, count in groups
  )
  layer_apply, layer_params, layer_state = map(tuple, zip(*adapters))
  prefix_apply, loss_apply, boundary_params = _make_boundaries(local_model, config, loss_from_logits, layer_names)
  batch = _microbatches(data, config.gradient_accumulation_steps, config.micro_batch_size_to_train_on)

  constrain_residual = _make_residual_constraint(
      local_model.mesh, config.shard_mode, batch["inputs"].shape[1:3], debug_sharding=config.debug_sharding
  )

  # Objective terms beyond the cross-entropy. Serial GA adds each microbatch's MoE load-balance loss and CSA indexer
  # loss, each averaged over the layers that have one (the load-balance loss summed instead for Megatron's
  # seq_aux_loss) and scaled by that microbatch's token count, to its summed cross-entropy, and divides the gradient
  # by the step's token count. Each layer returns its terms already scaled per layer as one differentiable value
  # (AUX_OBJECTIVE_KEY); the schedule seeds its backward with the microbatch's token count.
  lb_enabled = _lb_loss_enabled(config)
  indexer_enabled = _indexer_loss_enabled(config)
  aux_enabled = lb_enabled or indexer_enabled
  lb_layer_scale = _lb_layer_scale(config, groups) if lb_enabled else 1.0
  indexer_layer_scale = _indexer_layer_scale(config, groups) if indexer_enabled else 1.0

  bias_enabled = _routed_bias_updates(config)
  # Serial GA differentiates the objective scaled by 1/token_capacity (see gradient_loss_scale); do the same so the
  # backward cotangents have the same magnitude, and fold the scale back into the final division.
  loss_scale = _ga().gradient_loss_scale(config)
  schedule = make_training_schedule(
      prefix_apply,
      layer_apply,
      loss_apply,
      aux_loss_key=AUX_OBJECTIVE_KEY if aux_enabled else None,
      aux_loss_cotangent=_token_count_cotangent if aux_enabled else None,
      per_layer_keys=_bias_metric_keys(config) if bias_enabled else (),
      layer_data_keys=("inputs",) if deepseek4 else (),
      loss_scale=loss_scale,
      grad_dtype=config.grad_dtype,
      constrain_residual=constrain_residual,
      layer_loop_context=(lambda: model_loop_unroll_context(config))
      if config.dual_pipe_double_buffer_layer_loops
      else None,
  )
  with jax.named_scope("dual_pipe"):
    loss_sum, aux, layer_grads, boundary_grads = schedule(layer_params, layer_state, boundary_params, batch)

  raw_grads = _restore_gradients(layer_grads, boundary_grads, config.param_scan_axis, layer_names)
  if jax.tree.structure(raw_grads) != jax.tree.structure(params):
    raise ValueError("dual_pipe gradient tree does not match the full model parameter tree")
  raw_grads = jax.tree.map(shard, raw_grads, params_shardings)

  # Match GA's valid-token normalization, including completely padded batches, and its reported losses: each
  # auxiliary loss is the mean over microbatches of its per-microbatch layer mean (or sum), and is added to the loss.
  # The optimizer receives every parameter gradient, including embeddings/head.
  aux.pop(AUX_OBJECTIVE_KEY, None)
  aux_totals = {}
  if lb_enabled:
    aux_totals["moe_lb_loss"] = aux.pop(LB_LOSS_KEY) * lb_layer_scale
  if indexer_enabled:
    aux_totals["indexer_loss"] = aux.pop(INDEXER_LOSS_KEY) * indexer_layer_scale
  loss, aux_means, has_weights, denominator = _ga().finalize_accumulated_loss(
      loss_sum, aux["total_weights"], aux_totals, config.gradient_accumulation_steps
  )
  moe_lb_loss = aux_means.get("moe_lb_loss", 0.0)
  indexer_loss = aux_means.get("indexer_loss", 0.0)
  divisor = denominator * loss_scale
  raw_grads = jax.tree.map(lambda x: jnp.where(has_weights, (x / divisor).astype(x.dtype), jnp.zeros_like(x)), raw_grads)
  # Routed-bias updates were summed per layer over microbatches. Like serial GA, turn summed integer expert counts
  # into one full-batch sign update; float updates (already per-microbatch steps) stay summed.
  moe_bias_updates, intermediate_outputs = None, {}
  per_layer = aux.pop(PER_LAYER_KEY, None)
  if bias_enabled and deepseek4:
    # train_step applies V4's bias updates by the module path they were sown at.
    intermediate_outputs = _deepseek4_bias_intermediates(per_layer, layer_names, config)
  elif bias_enabled:
    moe_bias_updates = _finalize_bias_updates(per_layer, layer_names, config)
  aux.update(
      intermediate_outputs=intermediate_outputs,
      moe_lb_loss=moe_lb_loss,
      indexer_loss=indexer_loss,
      mtp_loss=0.0,
      moe_bias_updates=moe_bias_updates,
      mtp_moe_bias_updates=None,
      batch_stats=None,
      has_moe_overflow=jnp.bool_(False),
  )
  return loss, aux, raw_grads
