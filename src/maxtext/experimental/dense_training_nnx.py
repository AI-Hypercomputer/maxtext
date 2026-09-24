"""Opt-in Llama/DeepSeek 1F1B accumulation for the normal MaxText train step.

The decoder's backward/next-forward layer pair shares one scan body. Embedding
and loss/head differentiation happen at its boundaries, with fixed parameters
throughout the accumulated step. There is no optimizer or host dispatch here.
"""

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from maxtext.experimental.dense_training_schedule import make_training_schedule
from maxtext.utils.globals import EPS


def validate_training_config(config):
  """Require deterministic layers and stateless TE quantization recipes."""
  is_deepseek = config.decoder_block == DecoderBlockType.DEEPSEEK
  if config.decoder_block not in (DecoderBlockType.LLAMA2, DecoderBlockType.DEEPSEEK):
    raise ValueError("dual_pipe currently supports decoder_block=llama2 or deepseek")
  if not config.scan_layers or config.inhomogeneous_layer_cycle_interval != 1:
    raise ValueError("dual_pipe requires scan_layers=true and inhomogeneous_layer_cycle_interval=1")
  if config.num_decoder_layers < 1:
    raise ValueError("At least one decoder layer is required")
  if config.shard_mode != ShardMode.AUTO:
    raise ValueError("dual_pipe currently requires shard_mode=auto")
  if config.remat_policy not in ("none", "full"):
    raise ValueError("dual_pipe requires remat_policy=none or remat_policy=full")
  if config.dropout_rate != 0:
    raise ValueError("dual_pipe requires dropout_rate=0; layer state is read-only")
  if config.mtp_num_layers != 0:
    raise ValueError("dual_pipe requires mtp_num_layers=0")
  if is_deepseek:
    if not 0 <= config.first_num_dense_layers < config.num_decoder_layers:
      raise ValueError("DeepSeek dual_pipe requires at least one MoE layer and a valid dense prefix")
    if config.num_experts <= 1 or not config.te_moe_block:
      raise ValueError("DeepSeek dual_pipe currently requires num_experts>1 and te_moe_block=true")
    # Current scaling and MXFP8 derive scales from each invocation's tensors;
    # unlike delayed scaling, neither needs persistent amax-history updates.
    if config.quantization not in ("te_no_quant", "te_fp8_currentscaling"):
      raise ValueError("DeepSeek dual_pipe requires quantization=te_no_quant or te_fp8_currentscaling")
    if config.te_gmm_quantization not in ("te_no_quant", "te_mxfp8"):
      raise ValueError("DeepSeek dual_pipe requires te_gmm_quantization=te_no_quant or te_mxfp8")
    # DeepSeek's frozen MoEBiasVar is already carried as read-only layer state.
    # Bias updates and auxiliary load-balancing loss are not implemented here.
    if config.load_balance_loss_weight != 0 or config.routed_bias_update_rate != 0:
      raise ValueError("DeepSeek dual_pipe requires load_balance_loss_weight=0 and routed_bias_update_rate=0 (frozen bias)")
  else:
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
  if getattr(config, "mhc_expansion_rate", 1) != 1:
    raise ValueError("dual_pipe currently requires mhc_expansion_rate=1")
  unsupported = (
      "use_qwix_quantization", "use_manual_quantization", "quantize_kvcache",
      "record_internal_nn_metrics", "parameter_memory_host_offload",
      "using_pipeline_parallelism", "use_batch_split_schedule",
      "use_multimodal", "use_audio", "learn_to_init_mode",
      "use_tunix_gradient_accumulation", "shard_optimizer_over_data",
      "optimizer_memory_host_offload", "use_indexer", "enable_diloco",
      "retry_when_tokens_dropped", "use_qk_clip", "engram_layers",
  )
  enabled = [name for name in unsupported if getattr(config, name, False)]
  if not is_deepseek and getattr(config, "routed_bias", False):
    enabled.append("routed_bias")
  if getattr(getattr(config, "lora", None), "enable_lora", False):
    enabled.append("lora.enable_lora")
  if enabled:
    raise ValueError(f"Unsupported with gradient_accumulation_schedule=dual_pipe: {', '.join(enabled)}")
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("dual_pipe requires a JAX build providing jax.fwd_and_bwd")


def _layer_groups(config):
  """Names and lengths of contiguous homogeneous stacks, in forward order."""
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    groups = (("dense_layers", config.first_num_dense_layers),
              ("moe_layers", config.num_decoder_layers - config.first_num_dense_layers))
    return tuple((name, count) for name, count in groups if count)
  return (("layers", config.num_decoder_layers),)


def _te_layer_metrics(intermediates, required):
  """Read the real TE counters sown by DeepSeekGenericLayer.post_process."""
  keys = ("te_moe_capacity_overflow", "te_moe_total_recv_tokens", "te_moe_recv_capacity_per_rank")
  if not all(key in intermediates for key in keys):
    if required:
      raise ValueError("TE MoE layer did not produce receive-capacity intermediates")
    return {
        "te_moe_capacity_overflow": jnp.bool_(False),
        "te_moe_max_total_recv_tokens": jnp.int32(0),
        "te_moe_recv_capacity_per_rank": jnp.int32(jnp.iinfo(jnp.int32).max),
    }
  values = [jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(intermediates[key])]) for key in keys]
  return {
      "te_moe_capacity_overflow": jnp.any(values[0]),
      "te_moe_max_total_recv_tokens": jnp.max(values[1]),
      "te_moe_recv_capacity_per_rank": jnp.min(values[2]),
  }


def _reduce_aux(stacked_aux):
  """Match normal GA: sum loss metrics, but OR/max/min TE capacity metrics."""
  reducers = {
      "te_moe_capacity_overflow": jnp.any,
      "te_moe_max_total_recv_tokens": jnp.max,
      "te_moe_recv_capacity_per_rank": jnp.min,
  }
  return {key: reducers.get(key, jnp.sum)(value, axis=0) for key, value in stacked_aux.items()}


def _make_layer_adapter(layers, config, layer_name, layer_count):
  """Expose one homogeneous scanned stack as a pure, single-layer function.

  The schedule uses a leading layer axis for params/state; _restore_gradients
  restores the original parameter axis afterward. Sharding metadata stays with
  each variable, with the sliced layer axis removed inside layer_apply.
  """
  # Keep heavy model dependencies out of the lightweight NNX bookkeeping tests.
  from maxtext.models.llama2 import LlamaDecoderLayer
  from maxtext.utils import maxtext_utils, maxtext_utils_nnx

  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    from maxtext.models.deepseek import DeepSeekDenseLayer, DeepSeekMoELayer

    layer_class = DeepSeekDenseLayer if layer_name == "dense_layers" else DeepSeekMoELayer
  else:
    layer_class = LlamaDecoderLayer
  if not isinstance(layers, layer_class):
    raise TypeError(f"Expected a scanned {layer_class.__name__} for {layer_name}")
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  if config.param_scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, config.param_scan_axis, 0), params)
  params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, layer_count)
  state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, layer_count)

  def layer_apply(weights, hidden, layer_state, positions, segments):
    weights, layer_state = maxtext_utils_nnx.nnx_remove_scan_axis((weights, layer_state), layer_name)
    layer = nnx.merge(graphdef, weights, layer_state, copy=True)
    hidden, _ = layer(
        hidden,
        decoder_segment_ids=segments,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    if config.te_moe_block:
      stats = _te_layer_metrics(nnx.pop(layer, nnx.Intermediate), required=layer_name == "moe_layers")
      return hidden, stats
    return hidden

  if config.remat_policy == "full":
    layer_apply = jax.checkpoint(
        layer_apply,
        policy=jax.checkpoint_policies.nothing_saveable,
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
    return local_model.decoder._apply_embedding(
        local_model.token_embedder,
        batch["inputs"],
        batch["inputs_position"],
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

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


def _microbatches(data, count, micro_batch_size):
  """Keep the exact interleaved batch layout and per-MB slicing used by GA."""
  required = {"inputs", "inputs_position", "inputs_segmentation", "targets", "targets_segmentation"}
  if not required.issubset(data):
    raise ValueError(f"dual_pipe requires batch keys: {sorted(required - data.keys())}")
  if "forced_routed_experts" in data:
    raise ValueError("dual_pipe does not support forced expert routing")

  def reshape(value):
    if value.ndim < 1 or value.shape[0] % count:
      raise ValueError("The leading data dimension must be divisible by gradient_accumulation_steps")
    batch_size = value.shape[0] // count
    if not 0 < micro_batch_size <= batch_size:
      raise ValueError("Invalid micro_batch_size_to_train_on for dual_pipe")
    value = value.reshape((batch_size, count) + value.shape[1:])
    return jnp.swapaxes(value, 0, 1)[:, :micro_batch_size]

  return jax.tree.map(reshape, data)


def _restore_gradients(layer_grads, boundary_grads, param_scan_axis, layer_names=("layers",)):
  """Restore the full model parameter tree and its original layer axis."""
  groups = layer_grads if isinstance(layer_grads, tuple) else (layer_grads,)
  if len(groups) != len(layer_names):
    raise ValueError("Layer gradient groups do not match the decoder stack names")
  if param_scan_axis != 0:
    groups = jax.tree.map(lambda x: jnp.moveaxis(x, 0, param_scan_axis), groups)
  return nnx.merge_state(boundary_grads, nnx.State({"decoder": dict(zip(layer_names, groups))}))


def dualpipe_loss_and_grad(config, model, params_shardings, data, loss_from_logits):
  """Drop-in loss/aux/gradient result for train_step, not a separate executable."""
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
  adapters = tuple(_make_layer_adapter(getattr(local_model.decoder, name), config, name, count) for name, count in groups)
  layer_apply, layer_params, layer_state = map(tuple, zip(*adapters))
  prefix_apply, loss_apply, boundary_params = _make_boundaries(local_model, config, loss_from_logits, layer_names)
  batch = _microbatches(data, config.gradient_accumulation_steps, config.micro_batch_size_to_train_on)
  schedule = make_training_schedule(
      prefix_apply, layer_apply, loss_apply, grad_dtype=config.grad_dtype,
      layer_has_aux=config.te_moe_block, reduce_aux=_reduce_aux,
  )
  with jax.named_scope("dual_pipe"):
    loss_sum, aux, layer_grads, boundary_grads = schedule(layer_params, layer_state, boundary_params, batch)

  raw_grads = _restore_gradients(layer_grads, boundary_grads, config.param_scan_axis, layer_names)
  if jax.tree.structure(raw_grads) != jax.tree.structure(params):
    raise ValueError("dual_pipe gradient tree does not match the full model parameter tree")
  raw_grads = jax.tree.map(shard, raw_grads, params_shardings)

  # Match GA's valid-token normalization, including completely padded batches.
  # The optimizer receives every parameter gradient, including embeddings/head.
  has_weights = aux["total_weights"] > 0
  denominator = jnp.maximum(aux["total_weights"], 1)
  loss = jnp.where(has_weights, loss_sum / denominator, 0.0)
  raw_grads = jax.tree.map(
      lambda x: jnp.where(has_weights, x / denominator, jnp.zeros_like(x)), raw_grads
  )
  aux.update(
      intermediate_outputs={}, moe_lb_loss=0.0, indexer_loss=0.0, mtp_loss=0.0,
      moe_bias_updates=None, mtp_moe_bias_updates=None, batch_stats=None, has_moe_overflow=jnp.bool_(False),
  )
  return loss, aux, raw_grads
