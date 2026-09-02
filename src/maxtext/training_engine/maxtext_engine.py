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

from collections.abc import Callable, Mapping
import contextlib
import dataclasses
import gc
import os
from typing import Any

from absl import logging
from flax import nnx
from flax import struct
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.integration.tunix.weight_mapping import raiden_unscan
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
import numpy as np

# `id` of the empty buffer `get_metrics` returns when no metrics have been recorded.
# Mirrors Tunix's `PeftTrainer.get_metrics`, which returns `MetricsBuffer(id=-1)` in the
# same situation. Real buffers are identified by their train step, so this cannot collide.
EMPTY_METRICS_BUFFER_ID = -1

# Where `nnx.split(TrainStateNNX(...))` puts the model's own state; verified, not assumed,
# by `_check_pure_state_reusable`.
_MODEL_STATE_KEY = "model"

_PURE_STATE_FALLBACK_WARNING = (
    "Cannot keep the train state as a pure pytree across steps (%s), so every fwd_bwd and "
    "update will re-walk the NNX module graph. That is correct but slow -- the two "
    "`nnx.split` calls cost ~92 ms per step on an unrolled 28-layer qwen3-0.6b, against "
    "~2 ms for the pure-state equivalent. Logged once per engine instance."
)

_RAIDEN_WORKER_INDEX_STRIDE = 10_000  # >> any plausible piece count (dozens-to-low-hundreds of groups)


def _malloc_trim() -> None:
  try:
    import ctypes  # pylint: disable=g-import-not-at-top
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except Exception:
    pass


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
  return any(isinstance(leaf, (jax.Array, np.ndarray, np.generic)) for leaf in leaves)


def _split_static_and_dynamic(batch: Any) -> tuple[Any, dict[str, Any]]:
  """Splits a loss-input batch into traced arrays and closed-over constants.

  Only dict batches can be split -- those are the ones produced by a `gen_model_input_fn`,
  whose entries are named keyword arguments. Anything else is MaxText's positional `data`
  and is passed through untouched.

  The split has a consequence worth stating plainly: whatever lands in `static` is
  **closed over** by the compiled kernel, not passed to it. If a `gen_model_input_fn`
  returns a different static value on a later call -- a swapped `algo_config`, a pad id
  that varies by step -- the kernel keeps using the one captured at compile time. That
  would be a wrong answer with no error and no log line, so `_batch_signature` includes
  the static half by value and a change forces a recompile. Do not drop it from the
  signature as a cost saving; the comparison is over a handful of small objects, while the
  failure it prevents is silent.

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


def _batch_signature(dynamic_batch: Any, static_batch: dict[str, Any]) -> Any:
  """Returns a key identifying everything a compiled kernel was built against.

  Two halves, for two different reasons:

  - the traced half by treedef/shape/dtype, because `in_shardings` is baked into the
    compiled callable, so a batch whose structure differs needs a fresh one. Without this
    the second structure raises a confusing `in_shardings` prefix-mismatch instead of
    simply recompiling.
  - the static half by value, because those are *closed over* by the compiled kernel. A
    caller that changes one -- a different `algo_config`, a scheduled `pad_id` -- would
    otherwise keep silently computing against the value captured at compile time.

  Compared with `!=` rather than hashed: the static half holds arbitrary caller objects,
  and common ones (`types.SimpleNamespace`) define `__eq__` and are therefore unhashable.
  """
  leaves, treedef = jax.tree.flatten(dynamic_batch)
  shapes = tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves)
  return (treedef, shapes, static_batch)


_REPLICATED_BATCH_DIM_WARNING = (
    "Loss input with batch dim %d does not divide mesh axis %r (size %d), so that "
    "dimension is replicated instead of sharded: every device along the axis holds and "
    "computes the whole micro-batch, %dx the work a sharded one would do there. Results "
    "stay correct. If it was not deliberate -- a sequence-packed micro-batch is always "
    "size 1 and has no alternative -- make the micro-batch a multiple of the axis size."
)

_UNCOMPARABLE_SIGNATURE_WARNING = (
    "Could not compare %s between fwd_bwd calls (%s), so the engine cannot tell whether "
    "the compiled kernel is still valid and will recompile on EVERY fwd_bwd from now on. %s"
)

_UNCOMPARABLE_STATIC_HINT = (
    "This happens when a `gen_model_input_fn` returns a fresh object each call that holds "
    "an array -- a dataclass or SimpleNamespace whose `__eq__` then compares elementwise, "
    "where `bool()` is ambiguous. (Reusing one instance is fine: comparison short-circuits "
    "on identity.) Fix by returning arrays as top-level entries of the batch dict, so they "
    "are traced rather than closed over, or by holding the config object fixed across calls."
)

_UNCOMPARABLE_STRUCTURE_HINT = (
    "This is unexpected: the structural half is built from pytree treedefs, shapes and "
    "dtypes, which compare cleanly. Recompilation stays correct, but the cost is now paid "
    "every step, so this is worth reporting rather than living with."
)


@struct.dataclass(frozen=True, kw_only=True)
class RouterReplayTrainerPayload(abstract_engine.TrainerPayload):
  """A TrainerPayload extension carrying forced router-replay expert decisions.

  Pairs with `router_replay_gen_model_input_fn` (via `with_gen_model_input_fn`)
  and `make_router_replay_loss_fn` (via `with_loss_fn`) below to let a caller
  (e.g. an RL rollout that already computed expert routing decisions) replay
  them during the forward pass instead of letting the model's gate re-route.

  Attributes:
    token_ids: Inherited from TrainerPayload; redeclared here (rather than
      relying only on inheritance) so static-analysis tools that can't
      introspect tunix's `TrainerPayload` dataclass still resolve these as valid
      constructor keyword arguments.
    token_mask: See TrainerPayload.
    segment_ids: See TrainerPayload.
    forced_routed_experts: Optional `[batch, seq, top_k]` (or `[batch, seq,
      num_layers, top_k]` for a distinct routing per layer) array of expert
      indices to replay, overriding the model's normal top-k routing. `-1` marks
      a padded/unused slot. See `check_forced_routing_support` in
      `maxtext.configs.types` for which decoder_blocks accept this.
  """

  token_ids: ArrayLike
  token_mask: ArrayLike
  segment_ids: ArrayLike | None = None
  forced_routed_experts: ArrayLike | None = None


def router_replay_gen_model_input_fn(
    payload: RouterReplayTrainerPayload,
) -> dict[str, Any]:
  """Adapts a RouterReplayTrainerPayload into router_replay_loss_fn's kwargs.

  Output keys are unpacked directly as `loss_fn(model, **kwargs)` by
  `_fwd_bwd_kernel`, so they must match the loss function's parameter names --
  they are not nested inside a `data` dict.

  Args:
    payload: A RouterReplayTrainerPayload (or subclass).

  Returns:
    `inputs`, `inputs_position`, `inputs_segmentation`, `targets`,
    `targets_segmentation`, and (when present) `forced_routed_experts`.
  """
  token_ids = jnp.asarray(payload.token_ids)
  token_mask = jnp.asarray(payload.token_mask) if payload.token_mask is not None else jnp.ones_like(token_ids)
  segment_ids = jnp.asarray(payload.segment_ids) if payload.segment_ids is not None else token_mask

  # TrainerPayload rows are left-padded prompt + right-padded completion, so a
  # plain arange would give the first real token a nonzero RoPE position and
  # shift every token relative to the rollout that produced the routing.
  positions = jnp.maximum(jnp.cumsum(token_mask != 0, axis=-1) - 1, 0).astype(jnp.int32)

  # roll(-1) wraps token 0 into the last position, which is not its real next
  # token; mask that position out instead of training on the wrap-around. Do
  # the same wherever a packed segment ends, since the next row belongs to a
  # different sequence.
  targets_segmentation = token_mask.at[:, -1].set(0)
  same_segment = segment_ids[:, :-1] == segment_ids[:, 1:]
  targets_segmentation = targets_segmentation.at[:, :-1].multiply(same_segment.astype(token_mask.dtype))

  kwargs = {
      "inputs": token_ids,
      "inputs_position": positions,
      "inputs_segmentation": segment_ids,
      "targets": jnp.roll(token_ids, -1, axis=-1),
      "targets_segmentation": targets_segmentation,
  }
  forced_routed_experts = getattr(payload, "forced_routed_experts", None)
  if forced_routed_experts is not None:
    kwargs["forced_routed_experts"] = jnp.asarray(forced_routed_experts)
  return kwargs


def make_router_replay_loss_fn(
    config: pyconfig.HyperParameters, dropout_rng: jax.Array | None = None
) -> Callable[..., Any]:
  """Builds a loss fn with the flat kwarg names router_replay_gen_model_input_fn

  produces, wrapping train.loss_fn's `(model, config, data, ...)` calling
  convention so the pair can be used together via `with_gen_model_input_fn` +
  `with_loss_fn` (see `_fwd_bwd_kernel`'s `loss_callable(mdl, **b)` call).

  Args:
    config: The MaxText config to pass through to `train.loss_fn`.
    dropout_rng: Required when `config.enable_dropout` is set.

  Returns:
    A callable `(model, inputs, inputs_position, inputs_segmentation,
    targets, targets_segmentation, forced_routed_experts=None)` suitable for
    `MaxTextTrainingEngine.with_loss_fn`.

  Raises:
    ValueError: If dropout is enabled but no `dropout_rng` was supplied.
  """
  if config.enable_dropout and dropout_rng is None:
    raise ValueError(
        "make_router_replay_loss_fn requires a dropout_rng when"
        " config.enable_dropout is set; pass one, or set enable_dropout=False"
        " for router replay."
    )

  def router_replay_loss_fn(
      model,
      inputs,
      inputs_position,
      inputs_segmentation,
      targets,
      targets_segmentation,
      forced_routed_experts=None,
  ):
    data = {
        "inputs": inputs,
        "inputs_position": inputs_position,
        "inputs_segmentation": inputs_segmentation,
        "targets": targets,
        "targets_segmentation": targets_segmentation,
    }
    if forced_routed_experts is not None:
      data["forced_routed_experts"] = forced_routed_experts
    return maxtext_train.loss_fn(model, config, data, dropout_rng=dropout_rng, params=None, is_train=True)

  return router_replay_loss_fn


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
      NotImplementedError: If `training_config.lora.enable_lora` is True. This engine has no LoRA
        path and would otherwise full-finetune the base model while the config claims LoRA.
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

    # This engine has no LoRA path
    if getattr(getattr(training_config, "lora", None), "enable_lora", False):
      raise NotImplementedError(
          "MaxTextTrainingEngine does not support LoRA, but lora.enable_lora=True was set. "
          "This engine trains all parameters, set lora.enable_lora=False to train with this engine."
      )
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
    self._signature_compare_warned: bool = False
    self._replicated_batch_warned: bool = False
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
    # Pure-pytree mirror of the model and train state, carried across steps so the step path
    # never re-walks the module graph. `None` means "not cached".
    self._params_pure: Any = None
    self._rest_pure: Any = None
    self._state_pure: Any = None
    self._pure_state_warned: bool = False
    self._accumulated_grads: Any = None
    # Summed loss denominators behind `_accumulated_grads`, which are unreduced: this is the
    # divisor `update()` applies once.
    self._accumulated_denominator: Any = None
    self._micro_step_count = 0
    # Set when this run resumed from an intra-step checkpoint, cleared once the step it
    # resumed into completes and its finished state has been checkpointed.
    self._resumed_mid_step = False
    self._cached_losses: list[abstract_engine.WeightedMetric | jax.Array] = []
    # `create_training_optimizer` returns a raw optax GradientTransformation. `TrainStateNNX.apply_gradients`
    # calls `optimizer.update(model, grads)`, which is the nnx.Optimizer signature, and
    # `checkpointing.CheckpointState` expects an nnx.Optimizer too, so wrap it here. `wrt=nnx.Param`
    # covers every parameter, which is correct only because LoRA is rejected above.
    self._learning_rate_schedule, tx = train_utils.create_training_optimizer(self._config, self._model)
    self._optimizer = nnx.Optimizer(self._model, tx, wrt=nnx.Param)
    self._train_step: int = 0

    self._checkpoint_manager = checkpointing.CheckpointManager(
        checkpoint_dir=self._config.checkpoint_dir,
        config=self._config,
    )
    self._metrics_recorder = metrics_module.MetricsRecorder()
    self._throttler = inflight_throttler.InflightThrottler(config=self._config)
    self._raiden_syncs: Any = None
    self._last_staged_step: Optional[int] = None
    self._staged_metadata: Any = None
    self._warned_raiden_sync_chunks: bool = False
    vllm_cfg = getattr(self._config, "vllm", {})
    if isinstance(vllm_cfg, dict):
      vllm_use_wc = vllm_cfg.get("use_weight_converter", False)
      vllm_backend = vllm_cfg.get("rollout_backend", "maxtext")
    else:
      vllm_use_wc = getattr(vllm_cfg, "use_weight_converter", False)
      vllm_backend = getattr(vllm_cfg, "rollout_backend", "maxtext")

    self._use_weight_converter = bool(
        getattr(self._config, "use_weight_converter", False)
        or vllm_use_wc
        or os.environ.get("USE_WEIGHT_CONVERTER", "0").lower() in ("1", "true", "yes")
    )
    self._rollout_backend = (
        getattr(self._config, "rollout_backend", None)
        or vllm_backend
        or os.environ.get("ROLLOUT_BACKEND", "maxtext")
    )
    if self._use_weight_converter:
      from maxtext.integration.vllm.weight_converter import WeightConverter  # pylint: disable=g-import-not-at-top,import-outside-toplevel
      self._weight_converter = WeightConverter(
          config=self._config,
          rollout_backend=self._rollout_backend,
          debug=getattr(self._config, "weight_sync_debug", False),
      )
    else:
      self._weight_converter = None

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
    self._compiled_fwd_bwd_accum = None
    self._compiled_update = None
    self._model_graphdef = None
    self._invalidate_pure_state()

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
    self._compiled_fwd_bwd_accum = None
    self._compiled_update = None
    self._state_graphdef = None
    self._invalidate_pure_state()

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
    self._compiled_fwd_bwd_accum = None
    self._compiled_update = None
    self._state_graphdef = None
    self._invalidate_pure_state()

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

  @contextlib.contextmanager
  def _sharding_ctx(self):
    """Activates the mesh and logical axis rules the MaxText layers are written against.

    The rules live in a context variable, so a kernel traced outside this context sees an
    empty rule set, every `maybe_shard_with_logical` becomes a no-op and XLA guesses the
    partitioning -- badly: 1012 ms against 581 ms for the same fwd/bwd on llama3.1-8b/fsdp=8.
    `train.py` wraps its own `jax.jit` the same way. Entered around the *call*, since jit is
    lazy and the rules must be live when tracing happens. Note a batch that data x fsdp
    cannot divide gives NaN gradients once the constraints are real.
    """
    if self._mesh is None:
      yield
      return
    with jax.set_mesh(self._mesh), nn_partitioning.axis_rules(self._config.logical_axis_rules):
      yield

  def _invalidate_pure_state(self) -> None:
    """Forgets the cached pure state, so the next step re-reads it from the NNX objects.

    For the three ways the live NNX variables get replaced behind the engine's back: the
    `model`/`optimizer`/`state` setters, and a checkpoint restore.
    """
    self._params_pure = None
    self._rest_pure = None
    self._state_pure = None

  def _disable_pure_state(self, reason: str) -> None:
    """Falls back to re-splitting the module graph on every step, saying so once."""
    self._invalidate_pure_state()
    if not self._pure_state_warned:
      self._pure_state_warned = True
      logging.warning(_PURE_STATE_FALLBACK_WARNING, reason)

  @staticmethod
  def _with_model_state(state_pure: Any, model_pure: Any) -> Any:
    """Returns `state_pure` with its model subtree replaced by `model_pure`.

    Through `raw_mapping` rather than `{**state_pure}`: `nnx.State` wraps children in a
    `State` on `__getitem__`, and rebuilding from those views gives an equal-valued but
    deeper pytree that `jax.jit` rejects as an `in_shardings` prefix mismatch.
    """
    return nnx.State({**state_pure.raw_mapping, _MODEL_STATE_KEY: model_pure.raw_mapping})

  def _check_pure_state_reusable(self, state_pure: Any, params_pure: Any, rest_pure: Any) -> str | None:
    """Returns why the pure state cannot be carried across steps, or None if it can.

    The step path rebuilds the kernel's `state_pure` by dropping the model's state back into
    `state_pure["model"]`, which only reproduces `nnx.split` if NNX put it there exactly
    once. `TrainStateNNX` does, but `engine.state` is a public setter that accepts anything,
    so this runs the real reconstruction and compares treedefs -- once per compile, against
    a `jax.jit` in_shardings mismatch that is hard to read back to its cause.

    Returns:
      `None` when the fast path is safe, else a short phrase naming what did not line up.
    """
    if not isinstance(state_pure, nnx.State) or _MODEL_STATE_KEY not in state_pure:
      return f"the train state's pure form has no {_MODEL_STATE_KEY!r} entry"
    if not hasattr(state_pure, "raw_mapping"):
      return "this version of flax.nnx.State does not expose raw_mapping"
    rebuilt = self._with_model_state(state_pure, nnx.merge_state(params_pure, rest_pure))
    if jax.tree.structure(rebuilt) != jax.tree.structure(state_pure):
      return f"state[{_MODEL_STATE_KEY!r}] is not the model's own state"
    return None

  def _refresh_pure_state(self) -> None:
    """Re-reads the model and train state as pure `nnx.State`, and caches both.

    Once per compile rather than once per step, which is the point: the two `nnx.split`
    calls the step path used to make walked 1756 graph nodes for 92 ms of a 283 ms step on
    an unrolled qwen3-0.6b, against 0.84 ms for `nnx.split_state` over the flat state.
    Publication is unchanged -- `fwd_bwd` and `update` still `nnx.update` the live objects
    where they always did, so `self.model` and `self.state` are never stale.
    """
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    model = getattr(self._state, _MODEL_STATE_KEY, self._model)
    self._state_graphdef, state_pure = nnx.split(self._state)
    self._model_graphdef, params_pure, rest_pure = nnx.split(model, nnx.Param, ...)

    reason = self._check_pure_state_reusable(state_pure, params_pure, rest_pure)
    if reason is not None:
      self._disable_pure_state(reason)
      return
    self._params_pure, self._rest_pure, self._state_pure = params_pure, rest_pure, state_pure

  def _read_model_pure(self, model: Any) -> tuple[Any, Any]:
    """Returns the model's `(params, rest)` pure state, from the cache when it is live."""
    if self._params_pure is not None:
      return self._params_pure, self._rest_pure
    self._model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    return params, rest

  def _read_state_pure(self) -> Any:
    """Returns the train state's pure form, from the cache when it is live."""
    if self._state_pure is not None:
      return self._state_pure
    self._state_graphdef, state_pure = nnx.split(self._state)
    return state_pure

  def _publish_model_rest(self, new_rest: Any) -> None:
    """Folds a fwd/bwd's updated non-parameter state into the cached train state.

    Required, not an optimization: without it `update()` would see the *previous*
    micro-batch's RNG counters and batch statistics. The structure is checked because
    anything the forward pass `sow`s (`record_max_logits`, `distill_beta`, MTP) widens
    `new_rest`, which would disagree with the shardings the kernel was compiled against.
    """
    if self._params_pure is None:
      return
    if jax.tree.structure(new_rest) != jax.tree.structure(self._rest_pure):
      self._disable_pure_state("fwd_bwd returned a wider non-parameter state than the model was split into")
      return
    self._rest_pure = new_rest
    self._state_pure = self._with_model_state(self._state_pure, nnx.merge_state(self._params_pure, new_rest))

  def _publish_state(self, new_state_pure: Any) -> None:
    """Adopts the update kernel's output as the cached state, re-deriving `(params, rest)`.

    The re-derivation reads `.type` off leaves that survive `jax.jit` in the treedef; the
    treedef comparison costs ~1 ms and turns a future NNX flattening change into a fallback
    plus a warning rather than a pytree error inside a traced kernel. Correctness does not
    hinge on it: `update()` has already written the state into the live NNX objects.
    """
    if self._params_pure is None:
      return
    if not isinstance(new_state_pure, nnx.State) or _MODEL_STATE_KEY not in new_state_pure:
      self._disable_pure_state("the update kernel returned a state with no model entry")
      return
    params_pure, rest_pure = nnx.split_state(new_state_pure[_MODEL_STATE_KEY], nnx.Param, ...)
    if jax.tree.structure(params_pure) != jax.tree.structure(self._params_pure):
      self._disable_pure_state("the update kernel's output does not partition into the same parameters")
      return
    self._params_pure, self._rest_pure, self._state_pure = params_pure, rest_pure, new_state_pure

  def _fwd_bwd_kernel(self, params, rest, batch, acc_grads=None, acc_denom=None):
    """Executes a single forward and backward pass and folds the result into the accumulator.

    Args:
      params: Pure `nnx.Param` state to differentiate against.
      rest: The model's remaining (non-parameter) pure state.
      batch: Loss-function inputs for this micro-batch.
      acc_grads: Gradients accumulated over earlier micro-batches of this update, or None on
        the first, which is what lets it skip allocating a parameter-sized buffer.
      acc_denom: Denominator accumulated alongside `acc_grads`, or None with it.

    Returns:
      `(primary_loss, aux_metrics, new_rest, acc_grads, acc_denom)`, where the last two are
      this micro-batch folded into the running totals.
    """
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
              "gen_model_input_fn must return a dict of loss-fn keyword arguments, got " f"{type(b).__name__}."
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
    # Every non-raising branch of `diff_wrapper` builds a LossOutput, so `loss_out` is always
    # one. The value returned by `value_and_grad` is the unreduced sum that was
    # differentiated, which `loss_out.primary_loss` already carries, so it is discarded here.
    (_, (loss_out, new_rest)), micro_grads = grad_func(params, rest, batch)

    micro_grads = jax.tree.map(
        lambda x: (
            x.astype(self._config.grad_dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != self._config.grad_dtype
            else x
        ),
        micro_grads,
    )

    # Accumulated UNREDUCED, with no `1/denominator` applied: `_update_kernel` divides once
    # by the total, so the optimizer sees `sum(grads)/sum(denom)` rather than a mean of
    # per-micro-batch means, which overweights short micro-batches. Same as
    # `gradient_accumulation.py` in the pre-train path.
    denominator = loss_out.primary_loss.denominator.astype(jnp.float32)
    if acc_grads is None:
      return loss_out.primary_loss, loss_out.aux_metrics, new_rest, micro_grads, denominator
    acc_grads = jax.tree.map(jnp.add, acc_grads, micro_grads)
    return loss_out.primary_loss, loss_out.aux_metrics, new_rest, acc_grads, acc_denom + denominator

  def _update_kernel(self, state_pure, accumulated_grads, accumulated_denominator, mean_loss):
    """Applies accumulated gradients to update the NNX model state.

    Returns:
      `(new_state_pure, grad_norm, is_skipped)`. `grad_norm` doubles as the throttler's
      handle on this update; see the `add_computation` call in `update()`.
    """
    grad_norm = None
    is_skipped_val = None
    if state_pure is not None:
      # This one division is the whole normalization. A zero total means every micro-batch
      # was empty; yield zeros rather than a NaN, as `gradient_accumulation.py` does.
      has_weights = accumulated_denominator > 0
      safe_denominator = jnp.where(has_weights, accumulated_denominator, 1.0)
      grads = jax.tree.map(
          lambda g: jnp.where(has_weights, g / safe_denominator.astype(g.dtype), jnp.zeros_like(g)),
          accumulated_grads,
      )
      # Before clipping, where Tunix's `optax.global_norm` also sits -- `train.py` would call
      # this `raw_grad_norm`. In float32 whatever `grad_dtype` is: a sum of squares over bf16
      # overflows on production-size models.
      grad_norm = max_utils.l2norm_pytree(jax.tree.map(lambda g: g.astype(jnp.float32), grads))
      if self._config.gradient_clipping_threshold > 0:
        grads = maxtext_utils.apply_gradient_clipping(grads, None, self._config.gradient_clipping_threshold)
      local_state = nnx.merge(self._state_graphdef, state_pure, copy=True)
      if hasattr(local_state, "apply_gradients"):
        if self._config.skip_step_on_spikes:
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

  def _warn_uncomparable(self, what: str, hint: str, exc: Exception) -> None:
    """Warns once per instance that a signature half could not be compared.

    Once per instance rather than `logging.log_first_n`, which is keyed per call site and
    process-wide: a second engine would never warn, and the warning would become
    execution-order dependent in tests.
    """
    if self._signature_compare_warned:
      return
    self._signature_compare_warned = True
    logging.warning(_UNCOMPARABLE_SIGNATURE_WARNING, what, exc, hint)

  def _needs_recompile(self, signature: Any) -> bool:
    """Returns whether the compiled kernel is stale for `signature`.

    An unanswerable comparison counts as stale. That is the right direction for
    correctness -- reusing a kernel whose closed-over static arguments changed is a silently wrong
    answer -- but it is not free: if the comparison always raises, this recompiles on
    every call, forever. XLA's cache can mask that as nothing worse than a mysteriously
    slow run, so the exception path says so out loud, once.

    The two halves are compared separately so that report names the right culprit.
    Comparing the signature as a whole would route a badly-behaved treedef or shape entry
    into a message blaming the caller's static loss arguments.
    """
    previous = self._compiled_signature
    if previous is None:
      return True

    try:
      if bool(previous[:2] != signature[:2]):
        return True
    except Exception as exc:  # pylint: disable=broad-except
      self._warn_uncomparable("batch structure", _UNCOMPARABLE_STRUCTURE_HINT, exc)
      return True

    try:
      return bool(previous[2] != signature[2])
    except Exception as exc:  # pylint: disable=broad-except
      self._warn_uncomparable("static loss arguments", _UNCOMPARABLE_STATIC_HINT, exc)
      return True

  def _prepare_batch(self, payload: Any) -> Any:
    """Maps a payload to the inputs the loss function is called with."""
    if self._gen_model_input_fn is not None:
      return self._gen_model_input_fn(payload)
    if dataclasses.is_dataclass(payload):
      return {k: getattr(payload, k) for k in payload.__dataclass_fields__ if getattr(payload, k) is not None}
    return payload

  def _mesh_sharding(self, leaf: Any) -> jax.sharding.Sharding | None:
    """Returns `leaf`'s own sharding when it lives on this mesh, else a replicated one.

    Reading `.sharding` unconditionally mixes device sets inside one `jax.jit`: model
    parameters come from `from_pretrained` as NamedShardings spanning the whole mesh,
    while scalars the optimizer allocates eagerly (adamw's `count`) carry a
    SingleDeviceSharding on device 0. jit rejects that combination with "Received
    incompatible devices for jitted computation", naming two unrelated arrays and no
    cause. Normalizing the strays to a replicated NamedSharding puts every argument on the
    same device set, while leaving genuinely sharded parameters untouched.
    """
    if leaf is None:
      return None
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == self._mesh:
      return leaf_sharding
    return jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())

  def _batch_axis_size(self, axis_names: Any) -> int:
    """Product of mesh dims backing one PartitionSpec entry (None -> 1)."""
    if axis_names is None:
      return 1
    names = axis_names if isinstance(axis_names, tuple) else (axis_names,)
    size = 1
    for name in names:
      size *= self._mesh.shape[name]
    return size

  def _batch_data_shardings(self, dynamic_batch: Any) -> Any:
    """Builds a per-leaf sharding tree for the traced part of a batch.

    `get_input_data_sharding` returns one sharding describing a rank-2 `[batch, sequence]`
    input. It cannot be used as a pytree prefix for a `gen_model_input_fn` batch, whose
    leaves have mixed rank -- a rank-2 spec applied to a rank-1 array is an error. Each
    leaf instead takes the leading entries of that spec that its own rank can absorb, so
    the batch dimension stays sharded and everything below it is replicated.

    A leaf whose batch dim doesn't evenly divide the batch axis's mesh size (e.g. a
    sequence-packed micro-batch, always size 1) replicates that dim instead of sharding
    it -- every device holds and computes on the same data with no cross-device split,
    which is correct (there's nothing to reduce back together afterwards) but wastes
    compute across the axis for that micro-batch. That is an N-fold cost, so it warns
    once per instance rather than living only in this docstring.
    """
    data_sharding = sharding.get_input_data_sharding(self._config, self._mesh)
    data_spec = tuple(data_sharding.spec)

    def leaf_sharding(leaf):
      if leaf is None:
        return None
      rank = jnp.ndim(leaf)
      spec = list(data_spec[:rank])
      if spec and spec[0] is not None:
        axis_size = self._batch_axis_size(spec[0])
        if leaf.shape[0] % axis_size:
          # Warn once per instance, not per leaf: this runs under a tree_map over every
          # loss input, and they normally share a batch dim. Silence here would leave an
          # N-fold compute cliff visible only in a docstring.
          if not self._replicated_batch_warned:
            self._replicated_batch_warned = True
            logging.warning(_REPLICATED_BATCH_DIM_WARNING, leaf.shape[0], spec[0], axis_size, axis_size)
          spec[0] = None
      return jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(*spec))

    return jax.tree.map(leaf_sharding, dynamic_batch)

  def _compile_for_batch(self, dynamic_batch: Any, static_batch: dict[str, Any]) -> None:
    """JIT-compiles the fwd/bwd and update kernels for one batch structure.

    `static_batch` is closed over rather than passed, so non-array loss arguments (Tunix's
    `algo_config`, `pad_id`, `eos_id`) never reach the jit boundary.
    """
    # The only place the graphs are walked: a recompile is when they may legitimately have
    # changed shape, and everything after is maintained as plain pytrees.
    self._refresh_pure_state()
    state_pure = self._read_state_pure()
    params_pure, rest_pure = self._read_model_pure(getattr(self._state, _MODEL_STATE_KEY, self._model))

    def first_kernel(params, rest, dynamic):
      batch = {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic
      return self._fwd_bwd_kernel(params, rest, batch)

    def accum_kernel(params, rest, dynamic, acc_grads, acc_denom):
      batch = {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic
      return self._fwd_bwd_kernel(params, rest, batch, acc_grads, acc_denom)

    if self._mesh is not None:
      replicated = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
      state_mesh_shardings = jax.tree.map(self._mesh_sharding, state_pure)
      params_shardings = jax.tree.map(self._mesh_sharding, params_pure)
      rest_shardings = jax.tree.map(self._mesh_sharding, rest_pure)
      batch_shardings = self._batch_data_shardings(dynamic_batch)
      first_in_shardings = (params_shardings, rest_shardings, batch_shardings)
      accum_in_shardings = first_in_shardings + (params_shardings, replicated)
      fwd_bwd_out_shardings = (None, None, rest_shardings, params_shardings, replicated)
      update_in_shardings = (state_mesh_shardings, params_shardings, replicated, None)
      update_out_shardings = (state_mesh_shardings, None, None)
    else:
      first_in_shardings = None
      accum_in_shardings = None
      fwd_bwd_out_shardings = None
      update_in_shardings = None
      update_out_shardings = None

    # 1. JIT Compile Micro FWD/BWD Pass.
    #
    # Two kernels: the first micro-batch has no accumulator to add to and allocates none,
    # later ones fold in and donate it, so the sum is written back in place instead of
    # materializing the micro gradients plus a fresh sum. `jax.jit` is lazy, so the second
    # costs nothing when every update takes one micro-batch. `params` is deliberately NOT
    # donated: JAX matches donations by shard-shape, not position, so it would alias the
    # weights into the gradient output.
    self._compiled_fwd_bwd = jax.jit(
        first_kernel,
        in_shardings=first_in_shardings,
        out_shardings=fwd_bwd_out_shardings,
    )
    self._compiled_fwd_bwd_accum = jax.jit(
        accum_kernel,
        in_shardings=accum_in_shardings,
        out_shardings=fwd_bwd_out_shardings,
        donate_argnums=(3, 4),
    )

    # 2. JIT Compile Optimizer Update Pass.
    #
    # `state_pure` is donated, as `get_functional_train_with_signature` does for the
    # standalone trainer: the engine rebinds the state from the output, so it is dead on
    # return. The gradients are not -- every parameter-shaped output is already claimed by
    # the incoming state, so JAX would only warn.
    self._compiled_update = jax.jit(
        self._update_kernel,
        in_shardings=update_in_shardings,
        out_shardings=update_out_shardings,
        donate_argnums=(0,),
    )
    self._compiled_signature = _batch_signature(dynamic_batch, static_batch)
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
    model = getattr(self._state, _MODEL_STATE_KEY, self._model)

    if self._compile_requested:
      dynamic_batch, static_batch = _split_static_and_dynamic(batch)
      # A compiled kernel is valid only for the batch it was built against: the traced
      # half is baked into `in_shardings`, and the static half is closed over. Either
      # changing needs a fresh kernel -- reusing it would raise an in_shardings mismatch
      # for the first, and silently use stale values for the second.
      signature = _batch_signature(dynamic_batch, static_batch)
      if not self._compiled or self._needs_recompile(signature):
        self._compile_for_batch(dynamic_batch, static_batch)
      # After any recompile, not before: reading first would hand the new kernel a pure
      # state split against the old graph.
      params, rest = self._read_model_pure(model)
      with self._sharding_ctx():
        if self._accumulated_grads is None:
          loss, aux, new_rest, acc_grads, acc_denom = self._compiled_fwd_bwd(params, rest, dynamic_batch)
        else:
          # Both accumulators are donated here, so they are rebound from the outputs below.
          loss, aux, new_rest, acc_grads, acc_denom = self._compiled_fwd_bwd_accum(
              params, rest, dynamic_batch, self._accumulated_grads, self._accumulated_denominator
          )
    else:
      params, rest = self._read_model_pure(model)
      with self._sharding_ctx():
        loss, aux, new_rest, acc_grads, acc_denom = self._fwd_bwd_kernel(
            params, rest, batch, self._accumulated_grads, self._accumulated_denominator
        )
    nnx.update(model, new_rest)
    self._publish_model_rest(new_rest)

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
    self._accumulated_grads = acc_grads
    self._accumulated_denominator = acc_denom
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

    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    state_pure = self._read_state_pure()

    # `_update_kernel` reads `mean_loss` only under `skip_step_on_spikes`, which is traced
    # off `self._config`, so otherwise this was seven eager launches per step
    # (`WeightedMetric.compute()`) feeding an argument the executable does not contain.
    if not self._config.skip_step_on_spikes:
      mean_loss = None
    elif self._cached_losses:
      loss_values = [l.compute() if isinstance(l, abstract_engine.WeightedMetric) else l for l in self._cached_losses]
      mean_loss = jnp.mean(jnp.stack(loss_values)) if len(loss_values) > 1 else loss_values[0]
    else:
      mean_loss = jnp.array(0.0)
    # `state_pure` is donated, so between this call and the `nnx.update` below `self._state`
    # is torn -- reading one of its arrays raises "Array has been deleted". Keep them adjacent.
    with self._sharding_ctx():
      if self._compiled and hasattr(self, "_compiled_update"):
        new_state_pure, grad_norm, is_skipped = self._compiled_update(
            state_pure, self._accumulated_grads, self._accumulated_denominator, mean_loss
        )
      else:
        new_state_pure, grad_norm, is_skipped = self._update_kernel(
            state_pure, self._accumulated_grads, self._accumulated_denominator, mean_loss
        )
    nnx.update(self._state, new_state_pure)
    self._publish_state(new_state_pure)

    if grad_norm is not None:
      self.record_metrics("gradient_norm", grad_norm)
    if is_skipped is not None:
      self.record_metrics("step_skipped", is_skipped)

    # Queue something the update produced, so `jax.block_until_ready` waits for it before the
    # metrics are logged. The gradient norm rather than the state: the throttler holds queued
    # entries until it pops them, which pinned three parameter trees, and once the state is
    # donated a late pop raises "Array has been deleted". The norm comes out of the same
    # executable, so its readiness still means the update landed. Tunix v2 does the same.
    self._throttler.add_computation(
        grad_norm if grad_norm is not None else (self._state if self._state is not None else self._model),
        self._metrics_recorder.get_step_metrics(self.train_step),
    )

    self._cached_losses.clear()
    self._accumulated_grads = None
    self._accumulated_denominator = None
    self._micro_step_count = 0
    self._train_step += 1

    if self._resumed_mid_step:
      # This is the step the run resumed into, and it just finished. The checkpoint on disk
      # for it is still the partial one. Orbax's save-interval policy will not save this step
      # again.
      self._resumed_mid_step = False
      self.save_checkpoint(metadata={"step": self.train_step}, force=True)

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

    step = kwargs.pop("step", None)
    if step is None and isinstance(metadata, Mapping):
      step = metadata.get("step")
    if step is None:
      # checkpoint for incomplete train step is saved for self.train_step+1
      # because update() is not called yet to increment train_step;
      # checkpoint for completed step is saved for self.train_step because update() increments train_step
      step = self.train_step + 1 if self._micro_step_count > 0 else self.train_step

    if self._micro_step_count > 0:
      logging.info(
          "Saving intra-step checkpoint at step %d (micro_step_count=%d).",
          step,
          self._micro_step_count,
      )
    else:
      logging.info("Saving checkpoint at step %d.", step)

    custom_metadata = {}
    if metadata:
      # Metadata from Orchestrator
      custom_metadata["additional_metadata"] = metadata
    # The gradients are stored unreduced, so their divisor has to survive the round-trip too.
    if self._micro_step_count > 0 and self._accumulated_denominator is not None:
      custom_metadata["accumulated_denominator"] = float(self._accumulated_denominator)

    ckpt_saved = self._checkpoint_manager.save_checkpoint(
        step=step,
        checkpoint_state=checkpointing.CheckpointState(
            model=self.model,
            optimizer=self.optimizer,
            # The full history, not `get_metrics()`: CheckpointState.accumulated_metrics is
            # a list, and restore_checkpoint iterates it back into the recorder's buffer.
            accumulated_metrics=self._metrics_recorder.get_metrics_history(clear_cache=False),
            accumulated_grads=self._accumulated_grads,
            # Recorded by the CheckpointManager into custom_metadata, so that a later save
            # at this same step can tell it supersedes this one.
            micro_step_count=self._micro_step_count,
        ),
        custom_metadata=custom_metadata,
        **kwargs,
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
    # Orbax has just written new arrays into the live NNX variables, so the cache is wrong
    # rather than merely old.
    self._invalidate_pure_state()

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
    # Checkpoint with no metadata says nothing about how far into its step it
    # got, and must not inherit the count from whatever this engine was doing before.
    self._micro_step_count = 0
    restored_denominator = None
    if restored_metadata:
      self._micro_step_count = restored_metadata.get("micro_step_count", 0)
      restored_denominator = restored_metadata.get("accumulated_denominator", None)
      restored_additional_metadata = restored_metadata.get("additional_metadata", None)

    if self._micro_step_count > 0:
      logging.info(
          "Restored intra-step checkpoint at step %d (micro_step_count=%d).",
          restored_step,
          self._micro_step_count,
      )
      # update() will increment the step after applying the accumulated gradients
      self.train_step = restored_step - 1
      # The checkpoint at `restored_step` holds a partially accumulated step. Once that step
      # completes, `update` replaces it with the finished state.
      self._resumed_mid_step = True
    else:
      self.train_step = restored_step
      self._resumed_mid_step = False

    # Restore intra-step state if it exists. Gated on the count because for a complete step
    # `restored_checkpoint_state.accumulated_grads` is just the value this engine passed in
    # above, which the branch above has already discarded.
    if self._micro_step_count > 0 and restored_checkpoint_state.accumulated_grads:
      self._accumulated_grads = restored_checkpoint_state.accumulated_grads
      self._accumulated_denominator = jnp.float32(restored_denominator if restored_denominator else 0.0)

      rebuilt_losses = None
      if self._metrics_recorder._metrics_buffer:  # pylint: disable=protected-access
        active_buf = self._metrics_recorder.get_step_metrics(restored_step)
        if active_buf and "loss" in active_buf.weighted_metrics:
          wm = active_buf.weighted_metrics["loss"]
          if wm.unreduced_sum.ndim > 0:
            rebuilt_losses = [
                abstract_engine.WeightedMetric(
                    unreduced_sum=wm.unreduced_sum[i],
                    denominator=wm.denominator[i],
                    eps=wm.eps,
                    min_denom=wm.min_denom,
                )
                for i in range(wm.unreduced_sum.shape[0])
            ]
          else:
            rebuilt_losses = [wm]
          self._cached_losses = rebuilt_losses

      # Checkpoints predating the denominator carry no value for it, but the losses rebuilt
      # above carry the very denominators that went into the saved gradients. Only those:
      # any `_cached_losses` from before the restore belong to a different run.
      if not restored_denominator and rebuilt_losses:
        denominator = jnp.float32(0.0)
        for cached_loss in rebuilt_losses:
          denominator = denominator + jnp.sum(cached_loss.denominator).astype(jnp.float32)
        self._accumulated_denominator = denominator

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

  def _get_trainable_params_state(self) -> Any:
    """Extracts pure parameter weights from the model, excluding optimizer and RNG state."""
    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if isinstance(model, nnx.Module):
      return nnx.state(model, nnx.Param)
    return self.model

  def _raiden_worker_index(self, piece_idx: int) -> int:
    return jax.process_index() * _RAIDEN_WORKER_INDEX_STRIDE + piece_idx + 1

  def prepare_weight_sync(
      self,
      staging_transport: str = "raiden",
      **kwargs: Any,
  ) -> Any:
    """Stages weights for transfer and returns access coordinates.

    Args:
      staging_transport: Weight staging transport ('raiden' or custom).
      **kwargs: Weight staging parameters.

    Returns:
      Sequence of WorkUnitMetadata or synchronization endpoints.
    """
    if staging_transport == "raiden":
      try:
        from tunix.experimental.weight_sync import raiden_synchronizer  # pylint: disable=g-import-not-at-top,import-outside-toplevel
      except ImportError as exc:
        # Fatal, not a warning: Raiden staging was explicitly requested and cannot be
        # provided. Returning empty metadata instead defers the failure to the caller --
        # `WeightSyncCoordinator` eventually raises "metadata collection returned an empty
        # side", which reports a count from another process and never mentions the missing
        # module, leaving the real cause in this worker's log on another host.
        raise RuntimeError(
            "staging_transport='raiden' requires tunix.experimental.weight_sync."
            "raiden_synchronizer, which the installed tunix does not provide. Install a"
            " tunix build that ships it, or select a different staging_transport."
        ) from exc

      if (
          self._raiden_syncs is not None
          and self._last_staged_step == self.train_step
          and self._staged_metadata is not None
      ):
        logging.info(
            "Trainer re-using staged weight sync for step %d (%d variables)",
            self.train_step,
            sum(len(m.variables) for m in self._staged_metadata),
        )
        return self._staged_metadata

      if self._raiden_syncs is not None:
        for sync in self._raiden_syncs:
          sync.release_host_arrays()
        gc.collect()
        _malloc_trim()

      # 1. Drain all in-flight TPU computations to ensure weights are fully updated
      self._throttler.wait_for_all()
      gc.collect()

      # 2. Extract clean trainable parameters
      params_state = self._get_trainable_params_state()
      piece_batch = max(1, int(os.environ.get("RAIDEN_STREAM_PIECE_BATCH", "1")))
      if "RAIDEN_WEIGHT_SYNC_CHUNKS" in os.environ and not getattr(self, "_warned_raiden_sync_chunks", False):
        logging.warning(
            "RAIDEN_WEIGHT_SYNC_CHUNKS is deprecated and no longer affects Raiden staging; "
            "use RAIDEN_STREAM_PIECE_BATCH instead."
        )
        self._warned_raiden_sync_chunks = True

      if self._use_weight_converter:
        if self._weight_converter is None:
          from maxtext.integration.vllm.weight_converter import WeightConverter  # pylint: disable=g-import-not-at-top,import-outside-toplevel
          self._weight_converter = WeightConverter(
              config=self._config,
              rollout_backend=self._rollout_backend,
              debug=getattr(self._config, "weight_sync_debug", False),
          )
        piece_iter = self._weight_converter.convert_streaming(params_state, groups_per_piece=piece_batch)
      else:
        # UNCHANGED, deliberately out of scope: this fp32->bf16 cast is an
        # on-device (HBM, not host RAM) full materialization -- a different
        # memory pool than the host OOM this plan addresses. Candidate
        # fast-follow: fold into unscan_layers_streaming's per-piece slicing.
        params_state = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
            params_state,
        )
        if self._config.scan_layers:
          piece_iter = raiden_unscan.unscan_layers_streaming(
              params_state,
              num_layers=self._config.num_decoder_layers,
              scan_axis=self._config.param_scan_axis,
              keys_per_piece=piece_batch,
          )
        else:
          piece_iter = iter([params_state])

      del params_state
      gc.collect()

      if self._raiden_syncs is None:
        self._raiden_syncs = []

      expected_num_pieces = len(self._raiden_syncs) if self._raiden_syncs else None
      is_pathways = bool("proxy" in os.environ.get("JAX_PLATFORMS", "") and os.environ.get("JAX_BACKEND_TARGET"))
      verify_weights = os.environ.get("VERIFY_WEIGHTS", "").lower() == "true"
      all_metadata = []
      total_variables = 0
      piece_idx = -1

      for piece_idx, piece in enumerate(piece_iter):
        if expected_num_pieces is None and piece_idx >= len(self._raiden_syncs):
          self._raiden_syncs.append(
              raiden_synchronizer.RaidenSynchronizer(
                  job_name="trainer",
                  worker_index=self._raiden_worker_index(piece_idx),
                  auto_h2d=False,
                  host_stage=is_pathways,
                  parallelism=4,
              )
          )
        elif piece_idx >= len(self._raiden_syncs):
          del piece
          break

        sync = self._raiden_syncs[piece_idx]
        sync.bind(piece)
        del piece
        gc.collect()

        # 4. Initiate Device-to-Host transfer to stage this piece for network
        # transfer before moving on to the next piece.
        if sync.active:
          sync.d2h()

        if verify_weights:
          logging.info("Source weights checksums (piece %d): %s", piece_idx, sync.checksums())

        metadata = sync.work_unit_metadata()
        total_variables += len(metadata.variables)
        all_metadata.append(metadata)
        sync.release_host_arrays()

      if expected_num_pieces is not None:
        remaining = 0
        for _ in piece_iter:
          remaining += 1
        num_pieces = (piece_idx + 1) + remaining
        if num_pieces != expected_num_pieces:
          raise RuntimeError(
              f"weight-sync piece count changed from {expected_num_pieces} to {num_pieces} "
              "between rounds; the cached conversion plan should make this impossible "
              "unless the model/config changed mid-run."
          )
      else:
        num_pieces = piece_idx + 1

      gc.collect()
      _malloc_trim()

      logging.info(
          "Trainer prepared weight sync for step %d: registered %d variables across %d piece(s) on mesh %s",
          self.train_step,
          total_variables,
          num_pieces,
          all_metadata[0].mesh_axes if all_metadata else None,
      )
      self._last_staged_step = self.train_step
      self._staged_metadata = all_metadata
      return all_metadata

    # Unknown transport: raise rather than return empty metadata. A typo would otherwise
    # surface only as the coordinator's "empty side" error, with nothing logged anywhere
    # naming the transport that was actually asked for.
    raise ValueError(f"unknown staging_transport {staging_transport!r}; expected 'raiden'.")

  def release_weight_sync(self, **kwargs: Any) -> Any:
    """Releases staged weight buffers after transfer completion."""
    self._last_staged_step = None
    self._staged_metadata = None
    if self._raiden_syncs:
      for sync in self._raiden_syncs:
        logging.vlog(1, "Trainer Raiden metrics: %s", sync.metrics())
        sync.release_host_arrays()
    gc.collect()
    _malloc_trim()
    return True

  def close(self) -> None:
    """Closes the trainer, writes buffered metrics and final checkpoint."""
    if self._raiden_syncs:
      for sync in self._raiden_syncs:
        if hasattr(sync, "close"):
          sync.close()
      self._raiden_syncs = None
    self._last_staged_step = None
    self._staged_metadata = None

    self.save_checkpoint(metadata=None, force=True)
    self._checkpoint_manager.close()

    # Write the metrics and cleanup metrics logger resources
    self._throttler.cleanup()

    # Cleanup metrics recorder resources after saving the checkpoint, ensuring all buffered metrics are saved properly
    self._metrics_recorder.cleanup()
