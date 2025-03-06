"""
Copyright 2024 Google LLC

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


from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
from flax.linen import partitioning as nn_partitioning
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from maxtext import common_types
from maxtext import optimizers
from maxtext.layers import quantizations

from maxtext.mpmd.maxtext_pp import llama2
from maxtext.mpmd.maxtext_pp import max_utils
from maxtext.mpmd.maxtext_pp import models

EPS = 1e-8
Transformer = models.Transformer


class PipelineStage:
  """Class for a pipeline stage in a MPMD pipeline parallelism setup.

  Prototype with stage:Transformer:Decoder:layer = 1:1:1:1 mapping.

  Maxtext decode step function has pre-process and post-process logics that are
  only executed on Stage 0 and Stage 1, respectively.
  So we need to have the Decoder be aware of the stage index.
  TODO(linchai): Should we extract the pre-process and post-process logics out of Decoder.

  """

  def _setup_mesh_and_model(self, config):
    """Set up the mesh and the model for training.

    Args:
      config:

    Returns:
      init_rng: RNG key
      writer: Summary writer for tensorboard
      checkpoint_manager: Orbax checkpointer
      state_mesh_annotations: the mesh annotations for the train state
      model:
      mesh:
      tx:
    TODO(linchai): setup model should be logic outside of Stage initialization;
    otherwise, the stage implementation will be coupled with model
    implementation.
    """

    self.init_rng = random.PRNGKey(config.init_weights_seed)

    # Mesh definition
    devices_array = max_utils.create_device_mesh(config, jax.local_devices())
    self.mesh = Mesh(devices_array, config.mesh_axes)

    # Model and Optimizer definition
    quant = quantizations.configure_quantization(config)

    if self.stage_index == 0:
      self.model = Transformer(
          self.config,
          self.mesh,
          quant=quant,
          stage_index=self.stage_index,
          num_stages=self._num_stages,
      )
      self._is_first = True
    elif self.stage_index == self._num_stages - 1:
      self.model = Transformer(
          self.config,
          self.mesh,
          quant=quant,
          stage_index=self.stage_index,
          num_stages=self._num_stages,
      )
      self._is_last = True
    else:
      self.model = llama2.LlamaDecoderLayerStage(
          self.config,
          self.mesh,
          quant=quant,
          stage_index=self.stage_index,
      )
    learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
    self.tx = optimizers.get_optimizer(config, learning_rate_schedule)

  def __init__(self, config, stage_index, num_stages, is_train=True):
    self.config = config
    self.stage_index = stage_index
    self._num_stages = num_stages
    self._has_backward = self._is_train = is_train
    self._is_first = self.stage_index == 0
    self._is_last = self.stage_index == self._num_stages - 1

    self._setup_mesh_and_model(self.config)

    self.train_state, _, self.state_mesh_shardings = (
        max_utils.setup_training_state(
            self.model,
            self.tx,
            self.config,
            self.stage_index,
            self.init_rng,
            self.mesh,
        )
    )

    # TODO(linchai): the two caches below are limited to the prototype before
    # the remote-transfer is ready to use.
    # map microbatch ID to the stage forward output and input values
    self.fwd_cache: Dict[int, Tuple[Any, Any]] = {}
    # map microbatch ID to the stage backward grads of the inputs.
    self.bwd_cache: Dict[int, Any] = {}
    # Since the grad_func is input dependent, we need to cache it for each
    # microbatch. The logic in this file only prototypes with remat enabled.
    self.grad_func:  Any = None
    self.grad_and_update_func: Any = None
    self.input_caches: Dict[int, Any] = {}
    self._set_jitted_forward_func()
    self._set_jit_update_train_state()
    self._set_jitted_loss_func()

  def _set_jitted_forward_func(self):
    if self.stage_index == 0 or self.stage_index == self._num_stages - 1:
      # Forward computation with logits and intermediate outputs
      def vjp_fwd(model, enable_dropout, state, args, nextrng):
        def forward_func(model, enable_dropout, state, args, dropout_rng):
          rng1, aqt_rng = jax.random.split(dropout_rng)
          outputs = model.apply(
              state.params,
              args["inputs"],
              args["inputs_position"],
              decoder_segment_ids=args["inputs_segmentation"],
              enable_dropout=enable_dropout,
              rngs={"dropout": rng1, "params": aqt_rng},
              )
          return outputs
        partial_forward_func = partial(forward_func, model, enable_dropout)
        outputs, grad_func = jax.vjp(partial_forward_func, state, args, nextrng)
        return outputs, grad_func

      with self.mesh, nn_partitioning.axis_rules(
          self.config.logical_axis_rules
      ):
        partial_vjp_fwd = partial(
            vjp_fwd,
            self.model,
            self.config.enable_dropout,
        )
        partial_vjp_fwd.__name__ = "vjp_fwd"
        fwd_jitted = jax.jit(
            partial_vjp_fwd,
            in_shardings=(
                self.state_mesh_shardings,
                self.input_sharding(),
                None,
            ),
            out_shardings=None,
            static_argnums=(),
        )
    else:
      # Forward computation with logits and intermediate outputs
      def vjp_fwd(model, enable_dropout, state, args, nextrng):
        # Compute forward
        def forward_func(model, enable_dropout, state, args, dropout_rng):
          rng1, aqt_rng = jax.random.split(dropout_rng)
          outputs = model.apply(
              state.params,
              args["inputs"],
              args["inputs_position"],
              decoder_segment_ids=args["inputs_segmentation"],
              deterministic=not enable_dropout,
              rngs={"dropout": rng1, "params": aqt_rng},
              )
          return outputs
        partial_forward_func = partial(forward_func, model, enable_dropout)
        outputs, grad_func = jax.vjp(partial_forward_func, state, args, nextrng)
        return outputs, grad_func

      with self.mesh, nn_partitioning.axis_rules(
          self.config.logical_axis_rules
      ):
        partial_vjp_fwd = partial(
            vjp_fwd,
            self.model,
            self.config.enable_dropout,
        )
        partial_vjp_fwd.__name__ = "vjp_fwd"
        fwd_jitted = jax.jit(
            partial_vjp_fwd,
            in_shardings=(
                self.state_mesh_shardings,
                self.input_sharding(),
                None,
            ),
            out_shardings=None,
            static_argnums=(),
        )

    self.fwd_jitted = fwd_jitted

  def get_jitted_forward_func(self):
    return self.fwd_jitted

  def _set_jitted_loss_func(self):
    # loss_func is only used by the last stage to compute the loss for training.
    # It's not the only necessacity for gradients computation on inputs.
    # If we have the outputs and output grads, we should still be able to
    # compute the grads on inputs using jax.vjp.
    def loss_func(logits, targets, targets_segmentation):
      assert logits is not None
      one_hot_targets = jax.nn.one_hot(targets, self.config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(
          logits, one_hot_targets, 0.0
      )
      xent = nn.with_logical_constraint(
          xent, ("activation_embed_and_logits_batch", "activation_length")
      )
      # Mask out paddings at the end of each example.
      xent = xent * (targets_segmentation != 0)
      total_loss = jnp.sum(xent)
      total_weights = jnp.sum(targets_segmentation != 0)
      loss = total_loss / (total_weights + EPS)
      return loss
    loss_func.__name__ = "loss_func"
    self.loss_func = jax.jit(loss_func)

  def get_loss_func(self):
    return self.loss_func

  @property
  def has_backward(self) -> bool:
    """Returns true if this stage has a backward pass."""
    return self._has_backward

  @has_backward.setter
  def has_backward(self, has_backward: bool):
    self._has_backward = has_backward

  @property
  def is_first(self):
    """Returns true if this stage is the first stage in the pipeline."""
    return self._is_first

  @is_first.setter
  def is_first(self, is_first: bool):
    """Returns true if this stage is the first stage in the pipeline."""
    self._is_first = is_first

  @property
  def is_last(self):
    """Returns true if this stage is the last stage in the pipeline."""
    return self._is_last

  @is_last.setter
  def is_last(self, is_last: bool):
    """Returns true if this stage is the last stage in the pipeline."""
    self._is_last = is_last

  def _set_jit_update_train_state(self):
    """Update the train state with the grads."""
    def update_state(state, grads):
      return state.apply_gradients(grads=grads)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      self.update_train_state_jitted = jax.jit(
          update_state,
          donate_argnums=0,
          in_shardings=(self.state_mesh_shardings, None),
          out_shardings=self.state_mesh_shardings,
      )
    return self.update_train_state_jitted

  def input_sharding(self):
    """Return the sharding of the inputs of this stage."""
    data_pspec = P(*self.config.data_sharding)
    data_sharding = jax.tree_util.tree_map(
        lambda p: jax.sharding.NamedSharding(self.mesh, p), data_pspec
    )
    return data_sharding

  def output_sharding(self):
    return None

  def reset_runtime_states(self) -> None:
    """Reset runtime states of the stage for the next execution."""
    self.fwd_cache = {}
    self.bwd_cache = {}

  def block_until_ready_train_state(self) -> None:
    """Block until the train state is ready."""
    jax.block_until_ready(self.train_state)

  def forward_step(
      self,
      fwd_jitted,
      fwd_microbatch_id: int,
      args,
  ):
    """Perform forward pass on the stage with one microbatch.

    Args:
      fwd_microbatch_id: the chunk ID of the microbatch
      args: the inputs from *external* to this stage. The prototype has done the
        data transfer in scheduler before entering the stage.

    Returns:
      The output of the stage.
    """
    # TODO(linchai): should we use the same rng for all microbatches and all stages.
    nextrng = jax.jit(jax.random.fold_in)(self.init_rng, fwd_microbatch_id)
    self.input_caches[fwd_microbatch_id] = args
    if not self.grad_func:
      outputs, *_, grad_func = fwd_jitted(
          self.train_state, args, nextrng
          )
      grad_func.__name__ = "grad_func"
      # self.grad_func[0] = jax.jit(grad_func)
      self.grad_func = grad_func

      def grad_and_update(state, args, output_grads):
        _, grad_func = self.fwd_jitted(state, args, nextrng)
        raw_grads = grad_func(output_grads)
        state = state.apply_gradients(grads=raw_grads[0].params)
        return state, raw_grads[1]["inputs"]
      grad_and_update.__name__ = "grad_and_update"
      self.grad_and_update_func = jax.jit(
          grad_and_update,
          donate_argnums=0,
          in_shardings=(self.state_mesh_shardings, self.input_sharding()),
          out_shardings=(self.state_mesh_shardings, self.input_sharding()),
      )
    else:
      outputs, *_ = fwd_jitted(self.train_state, args, nextrng)

    # Save activations and inputs for backward
    self.fwd_cache[fwd_microbatch_id] = (
        outputs,  # stage_output
        args,  # input_values
    )

    # We return the original user-provied output, not normalized to tuple.
    # See [Note: pipeline model output type]
    return outputs

  def backward_step(
      self,
      bwd_microbatch_id: int,
      loss=None,
      output_grads: Any = None,
  ):
    """Perform backward pass on the module.

    This should only be called once per microbatch.

    Args:
      bwd_microbatch_id: the chunk ID of the microbatch
      loss: the loss from the last stage
      output_grads: the gradients of the outputs from the last stage
      last_backward: a flag to indicate if it is the last backward
    """

    with self.mesh, nn_partitioning.axis_rules(
        self.config.logical_axis_rules
    ):
      if self.is_last:
        # The last stage doesn't have the gradients of its outputs.
        # It computes the gradients from the loss.
        assert loss is not None
        # reshape the loss to the shape of the last stage output.
        loss = jnp.broadcast_to(
            loss, self.fwd_cache[bwd_microbatch_id][0].shape
        )
        output_grads = jax.device_put(loss, self.input_sharding())
      else:
        assert output_grads is not None

      self.train_state, self.bwd_cache[bwd_microbatch_id] = (
          self.grad_and_update_func(self.train_state, self.input_caches[bwd_microbatch_id], output_grads)
      )
    return self.bwd_cache[bwd_microbatch_id]
