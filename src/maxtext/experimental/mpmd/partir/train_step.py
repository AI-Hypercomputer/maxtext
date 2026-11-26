# Copyright 2026 Google LLC
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
"""The training step, shared by every execution mode.

There is exactly one implementation of the loss and of the microbatched
optimizer step here.  The MPMD (pipelined) and SPMD (single-program) modes
differ only in two substitutions:

  ==================  ==========================  ========================
  mode                stage wrapper               microbatch wrapper
  ==================  ==========================  ========================
  ``MPMD``            ``mpmd.named_computation``  ``mpmd.call``
  ``SPMD`` reference  identity                    identity
  ==================  ==========================  ========================

Because the arithmetic is literally the same code, any numerical difference
observed between the two modes is attributable to the MPMD partitioning
itself, which is exactly the property ``parity.py`` asserts. The trainer and
the lowering check share it too, so they cannot optimise different objectives.

The optimizer is a constant-learning-rate ``optax.adamw``
(:func:`make_optimizer`), not MaxText's optimizer and schedule.
"""

from __future__ import annotations

from collections.abc import Callable
import enum
from typing import Any

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from maxtext.experimental.mpmd.partir import mesh_utils
from maxtext.experimental.mpmd.partir import models

# The batch keys the model and loss consume.
BATCH_KEYS = (
    "inputs",
    "inputs_position",
    "inputs_segmentation",
    "targets",
    "targets_position",
    "targets_segmentation",
)


class ExecutionMode(enum.Enum):
  """How the training step should be traced."""

  # Pipelined across meshes by PartIR MPMD.
  MPMD = "mpmd"
  # A single program; the numerical reference.
  SPMD = "spmd"

  @property
  def stage_wrapper(self) -> models.Wrapper:
    return models.NAMED_COMPUTATION if self is ExecutionMode.MPMD else models.UNWRAPPED


def microbatch_call(fn: Callable[..., Any], call_counter: int) -> Callable[..., Any]:
  """Wraps one microbatch's step in ``mpmd.call``, the pipeline slot `call_counter`."""
  # Imported here, so that the SPMD step (and its tests) runs without Shardy
  # MPMD installed.
  from maxtext.experimental.mpmd.partir.sdy_mpmd_compat import mpmd  # pylint: disable=import-outside-toplevel

  return mpmd.call(fn, call_counter=call_counter)


@jax.custom_vjp
def _sparse_cross_entropy_with_logits(
    logits: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Computes per-token sparse cross-entropy without a full-vocab one_hot."""
  logits_f32 = logits.astype(jnp.float32)
  max_logit = jax.lax.stop_gradient(jnp.max(logits_f32, axis=-1, keepdims=True))
  shifted = logits_f32 - max_logit
  log_z = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1))
  target_logit = jnp.squeeze(jnp.take_along_axis(shifted, targets[..., None], axis=-1), axis=-1)
  return log_z - target_logit


def _sparse_ce_fwd(
    logits: jax.Array,
    targets: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
  """Forward pass for custom-VJP sparse cross-entropy."""
  logits_f32 = logits.astype(jnp.float32)
  max_logit = jax.lax.stop_gradient(jnp.max(logits_f32, axis=-1, keepdims=True))
  shifted = logits_f32 - max_logit
  sum_exp = jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)
  log_z = jnp.squeeze(jnp.log(sum_exp), axis=-1)
  target_logit = jnp.squeeze(jnp.take_along_axis(shifted, targets[..., None], axis=-1), axis=-1)
  return log_z - target_logit, (logits, targets, max_logit, sum_exp)


def _sparse_ce_bwd(
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, None]:
  """Backward pass for custom-VJP sparse cross-entropy."""
  logits, targets, max_logit, sum_exp = res
  vocab_size = logits.shape[-1]
  probs = jnp.exp(logits.astype(jnp.float32) - max_logit) / sum_exp
  one_hot = jax.nn.one_hot(targets, vocab_size, dtype=jnp.float32)
  d_logits = (probs - one_hot) * g[..., None].astype(jnp.float32)
  return d_logits.astype(logits.dtype), None


_sparse_cross_entropy_with_logits.defvjp(_sparse_ce_fwd, _sparse_ce_bwd)


def compute_loss(
    graphdef: models.PipelineGraphDef,
    params: dict[str, Any],
    batch: dict[str, Any],
    *,
    vocab_size: int,
    wrap: models.Wrapper,
    pregathered_params: dict[str, Any] | None = None,
) -> jax.Array:
  """Returns the mean cross-entropy over the unpadded tokens of `batch`.

  Args:
    graphdef: Static model structure.
    params: Per-unit parameters.
    batch: A single microbatch.
    vocab_size: Number of classes for the one-hot targets.
    wrap: Stage wrapping strategy, see :mod:`models`.
    pregathered_params: Optional pre-cast, FSDP-gathered layer parameters.

  Returns:
    A scalar loss.
  """
  del vocab_size
  logits = models.forward(
      graphdef,
      params,
      batch,
      wrap=wrap,
      pregathered_params=pregathered_params,
  )
  targets = batch["targets"]
  segmentation = batch.get("targets_segmentation")

  @wrap(mesh_utils.UNEMBED_NAME)
  def _loss_head(logits_arr, targets_arr, seg_arr):
    with models._logical_axis_context(graphdef, logits_arr, wrap):  # pylint: disable=protected-access
      token_loss = _sparse_cross_entropy_with_logits(logits_arr, targets_arr)
      if seg_arr is None:
        return jnp.mean(token_loss)
      # Padding is encoded as segment id 0; excluding it keeps the loss (and
      # hence the gradients) independent of how the batch happens to be padded.
      mask = (seg_arr != 0).astype(token_loss.dtype)
      num_tokens = jnp.sum(mask)
      return jnp.where(num_tokens > 0, jnp.sum(token_loss * mask) / num_tokens, 0.0)

  return _loss_head(logits, targets, segmentation)


def make_train_step(
    graphdef: models.PipelineGraphDef,
    *,
    vocab_size: int,
    num_microbatches: int,
    mode: ExecutionMode,
) -> Callable[[train_state.TrainState, dict[str, Any]], Any]:
  """Builds a microbatched training step.

  The global batch (sharded along axis 0 over ``fsdp``) is reshaped to
  ``[micro_size, num_microbatches, ...]`` and sliced along axis 1 so that every
  microbatch keeps its leading ``micro_size`` dimension evenly sharded across
  all FSDP devices in the stage without cross-device resharding. In MPMD mode
  every microbatch is wrapped in ``mpmd.call`` with a distinct
  ``call_counter``; that is what lets PartIR identify the individual pipeline
  "slots" and interleave them according to the configured schedule.

  Args:
    graphdef: Static model structure.
    vocab_size: Number of classes for the one-hot targets.
    num_microbatches: Number of microbatches per optimizer step.
    mode: MPMD (pipelined) or SPMD (reference).

  Returns:
    A function ``(state, batch) -> (new_state, loss)``.
  """
  wrap = mode.stage_wrapper

  def train_step(state: train_state.TrainState, batch: dict[str, Any]) -> tuple[train_state.TrainState, jax.Array]:
    def to_microbatches(array):
      micro_size = array.shape[0] // num_microbatches
      return array.reshape((micro_size, num_microbatches) + array.shape[1:])

    def loss_fn(params, pregathered, microbatch):
      return compute_loss(
          graphdef,
          params,
          microbatch,
          vocab_size=vocab_size,
          wrap=wrap,
          pregathered_params=pregathered,
      )

    def fwd_with_carry_passthrough(params, pregathered, microbatch):
      return loss_fn(params, pregathered, microbatch), params

    def accumulate(carry, params, pregathered, microbatch):
      loss, grads = jax.value_and_grad(loss_fn)(params, pregathered, microbatch)
      return jax.tree.map(jax.lax.add, carry, (loss, grads))

    microbatched = jax.tree.map(to_microbatches, batch)
    params = state.params
    pregathered_params = None
    if mode is ExecutionMode.MPMD and graphdef.data_parallel_size > 1:
      pregathered_params = models.pregather_params(
          graphdef,
          params,
          microbatch_rows=microbatched["inputs"].shape[0],
          wrap=wrap,
      )

    with models.defer_fsdp_shard_map_psum(graphdef, pregathered_params, params):
      if mode is ExecutionMode.MPMD:
        # Trace all forward microbatches before backward microbatches so that in
        # the unscheduled MPMD block every forward computation F_s(m+1) precedes
        # B_s(m). In 1F1B wavefront passes, mlir::sortTopologically then
        # schedules each consumer F_{s+1}(m) and B_s(m) immediately after its
        # incoming stage transfer before the next microbatch's transfer is
        # emitted, preventing IfrtMergeCopiesAndReshardsPass from fusing
        # transfers across consecutive microbatches. Passing `grad_carry`
        # through `fwd_with_carry_passthrough` on microbatches 1..N-1 causes
        # JAX's VJP transpose to accumulate `total_grads + mb_grads` inside
        # each `bwd_call` fragment without allocating a zero-init accumulator on
        # microbatch 0.
        grad_carry = models.build_grad_carry(graphdef, params, pregathered_params)
        vjp_steps = []
        for index in range(num_microbatches):
          microbatch = jax.tree.map(lambda x: x[:, index], microbatched)  # pylint: disable=cell-var-from-loop
          if index == 0:
            step_fn = microbatch_call(loss_fn, call_counter=index)
            mb_loss, vjp_fn = jax.vjp(step_fn, grad_carry, pregathered_params, microbatch)
          else:
            step_fn = microbatch_call(fwd_with_carry_passthrough, call_counter=index)
            (mb_loss, _), vjp_fn = jax.vjp(step_fn, grad_carry, pregathered_params, microbatch)
          vjp_steps.append((index, mb_loss, vjp_fn))

        total_loss = jnp.zeros(())
        total_grads = None
        for index, mb_loss, vjp_fn in vjp_steps:
          total_loss = total_loss + mb_loss
          if index == 0:
            total_grads, _, _ = vjp_fn(jnp.ones_like(mb_loss))
          else:
            total_grads, _, _ = vjp_fn((jnp.ones_like(mb_loss), total_grads))
      else:
        carry = (jnp.zeros(()), jax.tree.map(jnp.zeros_like, params))
        for index in range(num_microbatches):
          microbatch = jax.tree.map(lambda x: x[:, index], microbatched)  # pylint: disable=cell-var-from-loop
          carry = accumulate(carry, params, pregathered_params, microbatch)
        total_loss, total_grads = carry

    grads = models.reduce_scatter_deferred_grads(
        graphdef,
        params,
        pregathered_params,
        total_grads,
        num_microbatches=num_microbatches,
        wrap=wrap,
    )
    loss = total_loss / num_microbatches
    return state.apply_gradients(grads=grads), loss

  return train_step


def make_optimizer(learning_rate: float) -> optax.GradientTransformation:
  """Returns the optimizer used by every mode.

  Kept here so the MPMD run and its SPMD reference cannot accidentally use
  different optimizers or hyperparameters.

  Args:
    learning_rate: Peak (constant) learning rate.
  """
  return optax.adamw(learning_rate=learning_rate)


def create_train_state(params: dict[str, Any], learning_rate: float) -> train_state.TrainState:
  """Creates the optimizer state for `params`.

  Args:
    params: The model parameters.
    learning_rate: Peak (constant) learning rate.

  Returns:
    A :class:`flax.training.train_state.TrainState` whose every leaf is an
    array.
  """
  state = train_state.TrainState.create(
      # This model is applied through `models.forward`, not through
      # `state.apply_fn`; TrainState nonetheless requires the field.
      apply_fn=lambda *args, **kwargs: None,
      params=params,
      tx=make_optimizer(learning_rate),
  )
  # `TrainState.create` sets `step` to the Python `int` 0, but
  # `apply_gradients` returns `step + 1` as a traced array. The state therefore
  # changes type after one step, which breaks both PartIR's input/output
  # constraint checking ("'int' object has no attribute 'shape'") and the
  # requirement that a compiled step accept its own output. Pinning the dtype
  # up front keeps the state a fixed point of the step function.
  return state.replace(step=jnp.asarray(state.step, jnp.int32))
