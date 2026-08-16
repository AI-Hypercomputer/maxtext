# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxText concrete implementation of AbstractTrainer for RL post-training.

Adapts MaxText's single-step compilation and execution primitives to implement
the MaxRL AbstractTrainer interface without running an outer loop.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train as maxtext_train
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import checkpointing
from maxtext.training_engine import inflight_throttler
from maxtext.training_engine import metrics as metrics_module
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils


# `id` of the empty buffer `get_metrics` returns when no metrics have been recorded.
# Mirrors Tunix's `PeftTrainer.get_metrics`, which returns `MetricsBuffer(id=-1)` in the
# same situation. Real buffers are identified by their train step, so this cannot collide.
EMPTY_METRICS_BUFFER_ID = -1


def _is_jax_dynamic(value: Any) -> bool:
  """Returns True if `value` can cross a `jax.jit` boundary as a traced argument.

  A `gen_model_input_fn` returns the loss function's keyword arguments, and only some of
  them are arrays. Tunix's GRPO adapter, for instance, returns a `TrainExample` alongside
  an `algo_config` object and integer `pad_id`/`eos_id`. The arrays must be traced; the
  rest must be closed over, or `jax.jit` rejects the call outright.
  """
  leaves = jax.tree.leaves(value)
  if not leaves:
    # An all-`None` subtree (e.g. an unset `ref_per_token_logps`) flattens to nothing. It
    # carries no data either way, so tracing it is harmless and keeps the treedef intact.
    return True
  return all(isinstance(leaf, (float, int, jax.Array, np.generic, np.ndarray)) for leaf in leaves)


def _split_static_and_dynamic(batch: Any) -> tuple[Any, dict[str, Any]]:
  """Splits a loss-input batch into traced arrays and closed-over constants.

  Only dict batches can be split -- those are the ones produced by a `gen_model_input_fn`,
  whose entries are named keyword arguments. Anything else is MaxText's positional `data`
  and is passed through untouched.

  Returns:
    `(dynamic, static)`. `static` is empty for non-dict batches.
  """
  if not isinstance(batch, dict):
    return batch, {}
  dynamic: dict[str, Any] = {}
  static: dict[str, Any] = {}
  for key, value in batch.items():
    # Python scalars are constants in practice (`pad_id`, `eos_id`) and are better baked
    # into the executable than traced, which would defeat constant folding in the masks
    # they build.
    if isinstance(value, (int, float, bool)) or not _is_jax_dynamic(value):
      static[key] = value
    else:
      dynamic[key] = value
  return dynamic, static


def _batch_signature(dynamic_batch: Any) -> Any:
  """Returns a hashable key identifying a batch's jit-relevant structure.

  `in_shardings` is baked into the compiled callable, so a batch whose treedef or shapes
  differ needs a fresh one. Without this the second structure raises a confusing
  `in_shardings` prefix-mismatch instead of simply recompiling.
  """
  leaves, treedef = jax.tree.flatten(dynamic_batch)
  return (treedef, tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves))


class MaxTextTrainingEngine(abstract_engine.AbstractTrainingEngine):
  """Concrete trainer wrapping MaxText single-step SPMD execution for NNX models."""

  def __init__(
      self,
      training_config: pyconfig.HyperParameters,
      mesh: jax.sharding.Mesh | None = None,
      wrap_with_tunix_adapter: bool = False,
      tokenizer_pad_id: int | None = None,
  ) -> None:
    """Initializes the MaxText trainer state and sharded model.

    Args:
      training_config: MaxText HyperParameters configuration instance.
      mesh: Optional SPMD device mesh.
      wrap_with_tunix_adapter: If True, wraps the model in `TunixMaxTextAdapter` so it accepts Tunix's
        `model(input_tokens, positions=..., attention_mask=..., cache=...)` call signature and returns
        `(logits, None)`. Required when driving this engine from a Tunix loss function.
      tokenizer_pad_id: Tokenizer pad token id, forwarded to the adapter. Required when
        `wrap_with_tunix_adapter` is True: without it the adapter passes `decoder_segment_ids=None`, MaxText
        falls back to causal-only masking, and pad positions are attended to -- silently corrupting trainer
        log-probs on every batch.

    Raises:
      TypeError: If training_config is not a pyconfig.HyperParameters instance.
      ValueError: If training_config.model_name is not specified or empty, or if `wrap_with_tunix_adapter`
        is requested without `tokenizer_pad_id` or without a `mesh`.
    """
    if not isinstance(training_config, pyconfig.HyperParameters):
      raise TypeError(
          "MaxTextTrainingEngine requires a pyconfig.HyperParameters instance," f" got {type(training_config).__name__}"
      )
    if wrap_with_tunix_adapter:
      if tokenizer_pad_id is None:
        raise ValueError(
            "wrap_with_tunix_adapter=True requires tokenizer_pad_id. Without it the adapter cannot build "
            "decoder_segment_ids, so pad positions are attended to and trainer log-probs are silently wrong."
        )
      if mesh is None:
        raise ValueError("wrap_with_tunix_adapter=True requires a mesh; the adapter is built under it.")
    self._config = training_config
    self._mesh = mesh
    self._init_rng = jax.random.PRNGKey(training_config.init_weights_seed)
    self._loss_fn: Callable[..., Any] | None = None
    # Defaults to True so the built-in `maxtext_train.loss_fn` -- which returns
    # `(loss, aux)` with real aux and is used whenever `with_loss_fn` is never called --
    # keeps recording its aux metrics. `with_loss_fn` overrides this per its own default.
    self._has_aux: bool = True
    self._gen_model_input_fn: Callable[[Any], dict[str, Any]] | None = None
    # Tracked per instance rather than via logging.log_first_n, which is process-wide and
    # would make the warning depend on whether some earlier engine already triggered it.
    self._eval_step_warned: bool = False
    self._compiled = False
    # Set by `compile()`, including when it defers for want of a dummy payload. `_compiled`
    # alone cannot express "wanted, not yet built", and conflating them would either make
    # every eager caller compile or make a deferred compile never happen.
    self._compile_requested = False
    self._compiled_signature: Any = None
    if not training_config.model_name:
      raise ValueError("training_config.model_name must be specified")
    model_or_model_mesh_pair = model_creation_utils.from_pretrained(
        config=self._config,
        mesh=self._mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=self._init_rng,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )
    # `from_pretrained` returns `(model, mesh)` when it had to derive the mesh itself, and just the model
    # when one was supplied. Adopt the derived mesh so `self._model` is always a module and `compile()` can
    # still build shardings.
    if self._mesh is None:
      self._model, self._mesh = model_or_model_mesh_pair
    else:
      self._model = model_or_model_mesh_pair
    self._state: Any = None
    self._accumulated_grads: Any = None
    self._micro_step_count = 0
    self._cached_losses: list[abstract_engine.WeightedMetric | jax.Array] = []
    # `create_training_optimizer` returns a raw optax GradientTransformation. `TrainStateNNX.apply_gradients`
    # calls `optimizer.update(model, grads)`, which is the nnx.Optimizer signature, and
    # `checkpointing.CheckpointState` expects an nnx.Optimizer too, so wrap it here. Mirrors the `wrt`
    # selection in `create_train_state_fn` inside `train_utils.setup_train_loop`.
    self._learning_rate_schedule, tx = train_utils.create_training_optimizer(self._config, self._model)
    wrt = (
        getattr(nnx, "LoRAParam", nnx.Param)
        if getattr(getattr(self._config, "lora", None), "enable_lora", False)
        else nnx.Param
    )
    self._optimizer = nnx.Optimizer(self._model, tx, wrt=wrt)
    self._train_step: int = 0

    self._checkpoint_manager = checkpointing.CheckpointManager(
        checkpoint_dir=self._config.checkpoint_dir,
        config=self._config,
    )
    self._metrics_recorder = metrics_module.MetricsRecorder()
    self._throttler = inflight_throttler.InflightThrottler(config=self._config)

  @property
  def model(self) -> Any:
    """Returns the NNX model instance."""
    return self._model

  @model.setter
  def model(self, new_model: Any) -> None:
    """Sets the NNX model instance."""
    self._model = new_model
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_update = None
    self._model_graphdef = None

  @property
  def optimizer(self) -> Any:
    """Returns the NNX optimizer instance."""
    return self._optimizer

  @optimizer.setter
  def optimizer(self, new_optimizer: Any) -> None:
    """Sets the NNX optimizer instance."""
    self._optimizer = new_optimizer
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_update = None
    self._state_graphdef = None

  @property
  def train_step(self) -> int:
    """Returns the current step integer."""
    return self._train_step

  @train_step.setter
  def train_step(self, step: int) -> None:
    """Sets the current step integer."""
    self._train_step = step

  @property
  def state(self) -> Any:
    """Returns the current train state, initializing it if necessary."""
    if self._state is None and self._model is not None and self._optimizer is not None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    return self._state

  @state.setter
  def state(self, new_state: Any) -> None:
    """Sets the current train state."""
    self._state = new_state
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_update = None
    self._state_graphdef = None

  @property
  def micro_step_count(self) -> int:
    """Returns the current micro-batch count in gradient accumulation."""
    return self._micro_step_count

  @property
  def has_accumulated_grads(self) -> bool:
    """Returns True if accumulated gradients are present."""
    return self._accumulated_grads is not None

  def with_loss_fn(self, customized_fn: Callable[..., Any], has_aux: bool = False) -> "MaxTextTrainingEngine":
    """Overrides the default autoregressive loss function with a custom RL loss.

    Args:
      customized_fn: Custom loss callable matching the MaxText loss signature.
      has_aux: Whether `customized_fn` returns auxiliary output alongside the loss, i.e.
        the `(loss, aux)` tuple form. When False, an aux returned that way is not recorded
        as metrics -- though it is still read when the primary loss must be derived from
        its `xent_sum`/`total_weights`. Structured returns (`LossOutput`, `WeightedMetric`)
        carry their aux intrinsically rather than alongside, so they are unaffected.

    Returns:
      self, for chaining.
    """
    self._loss_fn = customized_fn
    self._has_aux = has_aux
    self._compiled = False
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> "MaxTextTrainingEngine":
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs.

    Setting one also selects how the loss is invoked. With an adapter, the loss is called
    as `loss_fn(model, **gen_model_input_fn(payload))` -- Tunix's convention, and what this
    adapter's return value has always been documented to be. Without one, it is called
    MaxText's way, `loss_fn(model, config, data, dropout_rng, params, is_train=True)`,
    which is what `maxtext_train.loss_fn` expects.

    Args:
      gen_model_input_fn: Maps a payload to a dict of loss-fn keyword arguments.

    Returns:
      self, for chaining.
    """
    self._gen_model_input_fn = gen_model_input_fn
    # The adapter decides which batch entries are traced and which are baked into the
    # executable, so a compiled kernel built against the previous one is stale.
    self._compiled = False
    return self

  def _fwd_bwd_kernel(self, params, rest, batch):
    """Executes a single forward and backward pass to compute gradients."""
    loss_callable = self._loss_fn if self._loss_fn is not None else maxtext_train.loss_fn

    def diff_wrapper(p, r, b):
      mdl = nnx.merge(self._model_graphdef, p, r, copy=True)
      if self._gen_model_input_fn is not None:
        # A gen_model_input_fn maps a payload to the loss fn's *keyword arguments* -- see
        # `with_gen_model_input_fn` -- so its output is unpacked rather than passed as the
        # positional `data`. This is how Tunix invokes losses
        # (`loss_fn(model, **gen_model_input_fn(payload))`), which lets a Tunix loss such
        # as `algo_core.grpo_loss_fn` be used with no adapter.
        if not isinstance(b, dict):
          raise TypeError(
              "gen_model_input_fn must return a dict of loss-fn keyword arguments, got "
              f"{type(b).__name__}."
          )
        out = loss_callable(mdl, **b)
      else:
        # No adapter set: the loss is a MaxText one, called MaxText's way.
        out = loss_callable(mdl, self._config, b, None, None, is_train=True)
      _, _, new_r = nnx.split(mdl, nnx.Param, ...)

      if isinstance(out, abstract_engine.LossOutput):
        return out.primary_loss.unreduced_sum, (out, new_r)
      elif isinstance(out, abstract_engine.WeightedMetric):
        loss_out = abstract_engine.LossOutput(
            primary_loss=out,
            aux_metrics={},
        )
        return out.unreduced_sum, (loss_out, new_r)
      elif isinstance(out, (tuple, list)) and len(out) == 2:
        loss_val, aux = out
        if isinstance(loss_val, abstract_engine.WeightedMetric):
          primary_loss = loss_val
        elif isinstance(aux, dict) and "xent_sum" in aux and "total_weights" in aux:
          primary_loss = abstract_engine.WeightedMetric(
              unreduced_sum=aux["xent_sum"],
              denominator=aux["total_weights"],
          )
        else:
          raise TypeError(
              f"Cannot construct WeightedMetric from 2-tuple loss return with elements "
              f"of type ({type(loss_val).__name__}, {type(aux).__name__}). Expected first element to be a "
              "WeightedMetric, or second element to be a dict containing 'xent_sum' and 'total_weights'."
          )

        # `has_aux=False` means the caller does not consider the second element to be
        # auxiliary output, so it is not recorded -- even though it may have been read
        # above to build `primary_loss`.
        loss_out = abstract_engine.LossOutput(
            primary_loss=primary_loss,
            aux_metrics=aux if (self._has_aux and isinstance(aux, dict)) else {},
        )
        return primary_loss.unreduced_sum, (loss_out, new_r)
      else:
        raise TypeError(
            f"Unsupported return type from loss function: {type(out)}. "
            "Expected abstract_engine.LossOutput, abstract_engine.WeightedMetric, "
            "or a 2-element tuple/list: (loss, aux_metrics)."
        )

    grad_func = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
    # Every non-raising branch of `diff_wrapper` builds a LossOutput, so `loss_out` is
    # always one and the gradient scaling below is unconditional. The value returned by
    # `value_and_grad` is the unreduced sum that was differentiated, which
    # `loss_out.primary_loss` already carries, so it is discarded here.
    (_, (loss_out, new_rest)), micro_grads = grad_func(params, rest, batch)
    scale = loss_out.primary_loss.compute_scale()
    micro_grads = jax.tree.map(lambda g: g * scale, micro_grads)

    micro_grads = jax.tree.map(
        lambda x: (x.astype(self._config.grad_dtype) if hasattr(x, "dtype") and x.dtype == jnp.float32 else x),
        micro_grads,
    )

    return loss_out.primary_loss, loss_out.aux_metrics, new_rest, micro_grads

  def _update_kernel(self, state_pure, accumulated_grads, micro_step_count, mean_loss):
    """Applies accumulated gradients to update the NNX model state."""
    grad_norm = None
    is_skipped_val = None
    if state_pure is not None:
      if micro_step_count <= 1:
        grads = accumulated_grads
      else:
        grads = jax.tree.map(
            lambda g: g / micro_step_count,
            accumulated_grads,
        )
      if self._config.gradient_clipping_threshold > 0:
        grads = maxtext_utils.apply_gradient_clipping(grads, None, self._config.gradient_clipping_threshold)
      local_state = nnx.merge(self._state_graphdef, state_pure, copy=True)
      if hasattr(local_state, "apply_gradients"):
        if self._config.skip_step_on_spikes:
          grad_norm = max_utils.l2norm_pytree(grads)
          local_state.apply_gradients(grads, loss=mean_loss, grad_norm=grad_norm)
          opt_obj = getattr(local_state, "optimizer", self._optimizer)
          if opt_obj is not None:
            opt_state = nnx.to_pure_dict(nnx.state(opt_obj)).get("opt_state", {})
            is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
            if is_skipped is not None:
              is_skipped_val = is_skipped.astype(jnp.float32)
        else:
          local_state.apply_gradients(grads)
      _, new_state_pure = nnx.split(local_state)
      return new_state_pure, grad_norm, is_skipped_val
    return state_pure, grad_norm, is_skipped_val

  def _prepare_batch(self, payload: Any) -> Any:
    """Maps a payload to the inputs the loss function is called with."""
    if self._gen_model_input_fn is not None:
      return self._gen_model_input_fn(payload)
    if dataclasses.is_dataclass(payload):
      return {k: getattr(payload, k) for k in payload.__dataclass_fields__ if getattr(payload, k) is not None}
    return payload

  def _mesh_sharding(self, leaf: Any) -> jax.sharding.Sharding:
    """Returns `leaf`'s own sharding when it lives on this mesh, else a replicated one.

    Reading `.sharding` unconditionally mixes device sets inside one `jax.jit`: model
    parameters come from `from_pretrained` as NamedShardings spanning the whole mesh,
    while scalars the optimizer allocates eagerly (adamw's `count`) carry a
    SingleDeviceSharding on device 0. jit rejects that combination with "Received
    incompatible devices for jitted computation", naming two unrelated arrays and no
    cause. Normalizing the strays to a replicated NamedSharding puts every argument on the
    same device set, while leaving genuinely sharded parameters untouched.
    """
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == self._mesh:
      return leaf_sharding
    return jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())

  def _batch_data_shardings(self, dynamic_batch: Any) -> Any:
    """Builds a per-leaf sharding tree for the traced part of a batch.

    `get_input_data_sharding` returns one sharding describing a rank-2 `[batch, sequence]`
    input. It cannot be used as a pytree prefix for a `gen_model_input_fn` batch, whose
    leaves have mixed rank -- a rank-2 spec applied to a rank-1 array is an error. Each
    leaf instead takes the leading entries of that spec that its own rank can absorb, so
    the batch dimension stays sharded and everything below it is replicated.
    """
    data_sharding = sharding.get_input_data_sharding(self._config, self._mesh)
    data_spec = tuple(data_sharding.spec)

    def leaf_sharding(leaf):
      rank = jnp.ndim(leaf)
      spec = jax.sharding.PartitionSpec(*data_spec[:rank])
      return jax.sharding.NamedSharding(self._mesh, spec)

    return jax.tree.map(leaf_sharding, dynamic_batch)

  def _compile_for_batch(self, dynamic_batch: Any, static_batch: dict[str, Any]) -> None:
    """JIT-compiles the fwd/bwd and update kernels for one batch structure.

    `static_batch` is closed over rather than passed, so non-array loss arguments (Tunix's
    `algo_config`, `pad_id`, `eos_id`) never reach the jit boundary.
    """
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)

    self._state_graphdef, state_pure = nnx.split(self._state)
    self._model_graphdef, params_pure, rest_pure = nnx.split(self._model, nnx.Param, ...)

    def kernel(params, rest, dynamic):
      batch = {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic
      return self._fwd_bwd_kernel(params, rest, batch)

    if self._mesh is not None:
      state_mesh_shardings = jax.tree.map(self._mesh_sharding, state_pure)
      params_shardings = jax.tree.map(self._mesh_sharding, params_pure)
      rest_shardings = jax.tree.map(self._mesh_sharding, rest_pure)
      fwd_bwd_in_shardings = (params_shardings, rest_shardings, self._batch_data_shardings(dynamic_batch))
      fwd_bwd_out_shardings = (None, None, rest_shardings, params_shardings)
      update_in_shardings = (state_mesh_shardings, params_shardings, None)
      update_out_shardings = (state_mesh_shardings, None, None)
    else:
      fwd_bwd_in_shardings = None
      fwd_bwd_out_shardings = None
      update_in_shardings = None
      update_out_shardings = None

    # 1. JIT Compile Micro FWD/BWD Pass
    self._compiled_fwd_bwd = jax.jit(
        kernel,
        in_shardings=fwd_bwd_in_shardings,
        out_shardings=fwd_bwd_out_shardings,
    )

    # 2. JIT Compile Optimizer Update Pass
    self._compiled_update = jax.jit(
        self._update_kernel,
        in_shardings=update_in_shardings,
        out_shardings=update_out_shardings,
        static_argnums=(2,),
    )
    self._compiled_signature = _batch_signature(dynamic_batch)
    self._compiled = True

  def compile(self, dummy_data: abstract_engine.TrainerPayload) -> None:
    """Triggers SPMD JIT compilation of fwd_bwd and update steps.

    Args:
      dummy_data: Sample TrainerPayload providing representative tensor shapes. Its shapes
        must match the real batches, or the first `fwd_bwd` simply recompiles. When it is
        `None` the engine cannot know the input shapes, so it stays on the eager path and
        compiles lazily on the first `fwd_bwd` instead.
    """
    # Recorded even when compilation is deferred: it is what tells `fwd_bwd` the caller
    # wants the compiled path at all. Engines that never call `compile` stay eager.
    self._compile_requested = True
    if self._compiled:
      return

    if dummy_data is None:
      # Callers driving a generic worker lifecycle (Tunix's `TrainerWorker.compile`) pass
      # nothing to compile against. Deferring costs nothing measurable: `jax.jit` is lazy,
      # so even with a payload this method only stages the wrappers and XLA still runs on
      # the first `fwd_bwd`. Logged at info, not warning -- the first `fwd_bwd` compiles
      # against the real batch, whose shapes are right by construction.
      logging.info(
          "MaxTextTrainingEngine.compile() was called without dummy_data; compiling "
          "against the first fwd_bwd payload instead."
      )
      return

    dynamic_batch, static_batch = _split_static_and_dynamic(self._prepare_batch(dummy_data))
    self._compile_for_batch(dynamic_batch, static_batch)

  def fwd_bwd(self, payload: abstract_engine.TrainerPayload, **kwargs: Any) -> None:
    """Executes a micro-batch forward-backward pass and accumulates gradients.

    Args:
      payload: Packed micro-batch training input.
      **kwargs: Implementation-specific options, accepted for interface compatibility
        and ignored by this engine.
    """
    batch = self._prepare_batch(payload)

    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")

    # Wait for previous computations to finish before dispatching the next one to TPU.
    self._throttler.wait_for_next()

    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    model = getattr(self._state, "model", self._model)
    self._model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    if self._compile_requested:
      dynamic_batch, static_batch = _split_static_and_dynamic(batch)
      # `in_shardings` is baked into the compiled callable, so a batch whose structure or
      # shapes changed needs a new one. Recompiling is the correct response; reusing it
      # would raise an in_shardings mismatch instead.
      if not self._compiled or self._compiled_signature != _batch_signature(dynamic_batch):
        self._compile_for_batch(dynamic_batch, static_batch)
      loss, aux, new_rest, micro_grads = self._compiled_fwd_bwd(params, rest, dynamic_batch)
    else:
      loss, aux, new_rest, micro_grads = self._fwd_bwd_kernel(params, rest, batch)
    nnx.update(model, new_rest)

    # Don't add metrics to the throttler queue because metrics are logged after
    # the update step.
    self._throttler.add_computation(computation=loss, metrics=None)

    if isinstance(loss, abstract_engine.WeightedMetric):
      self.record_metrics("loss", loss)

    # Record auxiliary metrics.
    if isinstance(aux, dict):
      for key, value in aux.items():
        if value is not None:
          self.record_metrics(key, value)

    self._cached_losses.append(loss)
    if self._accumulated_grads is None:
      self._accumulated_grads = micro_grads
    else:
      self._accumulated_grads = jax.tree.map(jnp.add, self._accumulated_grads, micro_grads)
    self._micro_step_count += 1

  def update(self, **kwargs: Any) -> int:
    """Applies accumulated gradients to update NNX model weights in HBM.

    Reuses NNX optimizer step from train.py (lines 511-535).

    Args:
      **kwargs: Implementation-specific options, accepted for interface compatibility
        and ignored by this engine.

    Returns:
      The train step count after this update. Unchanged when there is nothing to apply.
    """
    if self._accumulated_grads is None:
      return self.train_step

    if self._learning_rate_schedule is not None:
      lr = self._learning_rate_schedule(self.train_step)
      self.record_metrics("learning_rate", lr)

    # Wait for previous computations to finish before dispatching the update step to TPU.
    self._throttler.wait_for_next()

    # TODO(mazumdera): The logic below should be pre-compiled.
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    self._state_graphdef, state_pure = nnx.split(self._state)

    if self._cached_losses:
      loss_values = [l.compute() if isinstance(l, abstract_engine.WeightedMetric) else l for l in self._cached_losses]
      mean_loss = jnp.mean(jnp.stack(loss_values)) if len(loss_values) > 1 else loss_values[0]
    else:
      mean_loss = jnp.array(0.0)
    if self._compiled and hasattr(self, "_compiled_update"):
      new_state_pure, grad_norm, is_skipped = self._compiled_update(
          state_pure, self._accumulated_grads, self._micro_step_count, mean_loss
      )
    else:
      new_state_pure, grad_norm, is_skipped = self._update_kernel(
          state_pure, self._accumulated_grads, self._micro_step_count, mean_loss
      )
    nnx.update(self._state, new_state_pure)

    if grad_norm is not None:
      self.record_metrics("gradient_norm", grad_norm)
    if is_skipped is not None:
      self.record_metrics("step_skipped", is_skipped)

    # Add the state to the throttler queue so jax.block_until_ready() waits
    # for the optimizer update to complete before logging the metrics.
    self._throttler.add_computation(
        self._state if self._state is not None else self._model,
        self._metrics_recorder.get_step_metrics(self.train_step),
    )

    self._cached_losses.clear()
    self._accumulated_grads = None
    self._micro_step_count = 0
    self._train_step += 1
    return self.train_step

  def eval_step(self, payload: abstract_engine.TrainerPayload, **kwargs: Any) -> None:
    """Warns once that evaluation is not implemented, then does nothing.

    A silent no-op lets `TrainerWorker.run_eval` report success having evaluated nothing,
    so any eval metrics for the run are meaningless rather than absent. Warning makes that
    audible; warning only once keeps a loop that evaluates every step from flooding the
    log. Implementing this properly means a forward-only pass plus deciding how eval
    metrics bucket via `MetricsBuffer.mode`, which is tracked separately.

    Mutates no trainer state -- in particular not `_accumulated_grads` or
    `_micro_step_count` -- as `AbstractTrainer.eval_step` requires.

    Args:
      payload: Packed micro-batch evaluation input. Currently unused.
      **kwargs: Additional keyword arguments for evaluation. Currently unused.
    """
    if not self._eval_step_warned:
      self._eval_step_warned = True
      logging.warning(
          "MaxTextTrainingEngine.eval_step is not implemented: it evaluates nothing and "
          "records no metrics, so any eval result reported for this run is meaningless. "
          "Logged once per engine instance."
      )

  def save_checkpoint(self, metadata: Any, **kwargs: Any) -> None:
    """Forces asynchronous Orbax checkpoint serialization.

    Args:
      metadata: Checkpoint metadata payload from Orchestrator.
      **kwargs: Additional checkpoint saving options.
    """
    # Drain all inflight computations and log pending metrics before checkpointing.
    self._throttler.wait_for_all()

    step = kwargs.get("step", self.train_step)
    force_ckpt_save = kwargs.get("force", False)

    custom_metadata = {}
    if self._micro_step_count > 0:
      logging.info(
          "Saving intra-step checkpoint at step %d (micro_step_count=%d).",
          step,
          self._micro_step_count,
      )
      force_ckpt_save = True
      custom_metadata["micro_step_count"] = self._micro_step_count
    else:
      logging.info("Saving checkpoint at step %d.", step)

    if metadata:
      # Metadata from Orchestrator
      custom_metadata["additional_metadata"] = metadata

    ckpt_saved = self._checkpoint_manager.save_checkpoint(
        step=step,
        checkpoint_state=checkpointing.CheckpointState(
            model=self.model,
            optimizer=self.optimizer,
            # The full history, not `get_metrics()`: CheckpointState.accumulated_metrics is
            # a list, and restore_checkpoint iterates it back into the recorder's buffer.
            accumulated_metrics=self._metrics_recorder.get_metrics_history(clear_cache=False),
            accumulated_grads=self._accumulated_grads,
        ),
        custom_metadata=custom_metadata,
        force_ckpt_save=force_ckpt_save,
    )
    if ckpt_saved:
      logging.info("Checkpoint saved at step %d.", step)

  def restore_checkpoint(self, **kwargs: Any) -> Any:
    """Restores the latest Multi-Tier Checkpoint and returns its metadata.

    Args:
      **kwargs: Additional checkpoint restoration options.

    Returns:
      The metadata PyTree of the restored checkpoint.
    """
    step = kwargs.get("step", None)
    checkpoint_state = checkpointing.CheckpointState(
        model=self.model,
        optimizer=self.optimizer,
        accumulated_grads=self._accumulated_grads,
    )

    restored_step, restored_checkpoint_state, restored_metadata = self._checkpoint_manager.restore_checkpoint(
        checkpoint_state=checkpoint_state,
        step=step,
    )
    if restored_step is None:
      return None

    logging.info("Checkpoint restored from step %d.", restored_step)
    self.train_step = restored_step

    if restored_checkpoint_state.accumulated_metrics:
      buffers = []
      for b in restored_checkpoint_state.accumulated_metrics:
        if isinstance(b, dict):
          wms = {}
          for k, wm in b.get("weighted_metrics", {}).items():
            if isinstance(wm, dict):
              wms[k] = abstract_engine.WeightedMetric(**wm)
            else:
              wms[k] = wm
          buffers.append(
              abstract_engine.MetricsBuffer(
                  id=b.get("id", 0),
                  mode=b.get("mode", "train"),
                  weighted_metrics=wms,
                  scalar_metrics=b.get("scalar_metrics", {}),
                  aggregation_fns=b.get("aggregation_fns", {}),
              )
          )
        else:
          buffers.append(b)
      # pylint: disable-next=protected-access
      self._metrics_recorder._metrics_buffer = buffers

    restored_additional_metadata = None
    if restored_metadata:
      self._micro_step_count = restored_metadata.get("micro_step_count", 0)
      restored_additional_metadata = restored_metadata.get("additional_metadata", None)

    # Restore intra-step state if it exists.
    if restored_checkpoint_state.accumulated_grads:
      self._accumulated_grads = restored_checkpoint_state.accumulated_grads

      if self._micro_step_count > 0 and self._metrics_recorder._metrics_buffer:  # pylint: disable=protected-access
        active_buf = self._metrics_recorder.get_step_metrics(restored_step)
        if active_buf and "loss" in active_buf.weighted_metrics:
          wm = active_buf.weighted_metrics["loss"]
          if wm.unreduced_sum.ndim > 0:
            self._cached_losses = [
                abstract_engine.WeightedMetric(
                    unreduced_sum=wm.unreduced_sum[i],
                    denominator=wm.denominator[i],
                    eps=wm.eps,
                    min_denom=wm.min_denom,
                )
                for i in range(wm.unreduced_sum.shape[0])
            ]
          else:
            self._cached_losses = [wm]

    return restored_additional_metadata

  def record_metrics(
      self,
      name: str,
      metric: abstract_engine.WeightedMetric | jax.Array | float | int | dict[str, Any],
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ) -> None:
    """Records a metric into the buffer, appending to JAX arrays.

    Args:
      name: The name of the metric.
      metric: The metric to record.
      aggregation_fn: The aggregation function to apply to the metric.
    """
    if metric is None:
      return
    if isinstance(metric, dict):
      for sub_k, sub_v in metric.items():
        if sub_v is not None:
          self.record_metrics(
              f"{name}/{sub_k}" if name else sub_k,
              sub_v,
              aggregation_fn=aggregation_fn,
          )
    else:
      self._metrics_recorder.buffer_metrics(
          train_step=self.train_step,
          name=name,
          metric=metric,
          aggregation_fn=aggregation_fn,
      )

  def get_metrics(self, clear_cache: bool = True) -> abstract_engine.MetricsBuffer:
    """Returns the most recent step's metrics as an on-device MetricsBuffer.

    One buffer accumulates per train step. Both this engine's own abstract interface and
    Tunix's `AbstractTrainer` declare a single buffer, so callers that want every step
    must reach for `MetricsRecorder.get_metrics_history` instead.

    Args:
      clear_cache: Whether to reset cached metrics after retrieval.

    Returns:
      The newest on-device MetricsBuffer. When nothing has been recorded this is an empty
      buffer with `id` set to EMPTY_METRICS_BUFFER_ID, not None, matching Tunix's
      `PeftTrainer.get_metrics`. Callers detect that case with
      `buffer.id == EMPTY_METRICS_BUFFER_ID`; a real buffer's id is its train step, which
      is never negative.
    """
    history = self._metrics_recorder.get_metrics_history(clear_cache=clear_cache)
    if not history:
      return abstract_engine.MetricsBuffer(id=EMPTY_METRICS_BUFFER_ID)
    if len(history) > 1:
      # Returning only the newest would otherwise drop the rest without a word, which is
      # the silent-loss pattern that produced the fabricated 0.0 in the parity harness.
      logging.warning(
          "get_metrics() is returning the buffer for step %s and dropping %d older buffer(s); "
          "call get_metrics() once per update, or use MetricsRecorder.get_metrics_history() "
          "to read every step.",
          history[-1].id,
          len(history) - 1,
      )
    return history[-1]

  def prepare_weight_sync(self, **kwargs: Any) -> Any:
    """Stages weights for transfer and returns access coordinates.

    Args:
      **kwargs: Weight staging parameters.

    Returns:
      Synchronization endpoints or coordinates for rollout actors.
    """
    return {}

  def close(self) -> None:
    """Closes the trainer and its associated resources."""
    self._throttler.cleanup()
    self._metrics_recorder.cleanup()
    self._checkpoint_manager.close()
