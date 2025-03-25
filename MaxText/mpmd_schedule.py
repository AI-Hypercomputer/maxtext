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

from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import jax
from jax import numpy as jnp
from jax.experimental import transfer
from jax.sharding import Mesh
from maxtext import common_types
from maxtext import profiler
from maxtext.mpmd.maxtext_pp import max_utils
from maxtext.mpmd.maxtext_pp.stage import PipelineStage
import numpy as np
import functools


class PipelineGPipeCircular(ABC):
  """Pipeline for GPipe."""

  def __init__(self, config: common_types.Config, stage_index: int, local_transfer_server: transfer.TransferServer, transfer_connection: transfer.TransferConnection,):
    # Initialize the pipeline hyperparameters
    self.stage_index = stage_index
    self.config = config
    self.local_transfer_server = local_transfer_server
    self.transfer_connection = transfer_connection

    self.num_stages = (
        self.config.ici_pipeline_parallelism
        * self.config.dcn_pipeline_parallelism
    )

    self.pipeline_microbatch_size = (
        self.config.micro_batch_size_to_train_on
        // self.config.num_pipeline_microbatches
    )
    # TODO(linchai): hard coded for now.
    # The batch size needs to be sharded on fsdp dimension.
    self.microbatches_per_stage = 8
    self.num_microbatches = (
        self.config.micro_batch_size_to_train_on // self.microbatches_per_stage
    )

    self.num_pipeline_repeats = self.config.num_pipeline_repeats
    self.num_stage_layers = self.num_stages * self.num_pipeline_repeats
    self.stage = PipelineStage(config, stage_index, self.num_stage_layers)
    # Holds the losses for each microbatch.
    self._internal_losses: List[jax.Array] = []
    self._targets_cache = None

  def targets_cache(self):
    return self._targets_cache

  def step_forward(self, data_iterator, step_index):
    """Run one iteration of the pipeline schedule with *whole-batch* input.

    Will chunk the input into microbatches automatically, and go through the
    microbatches according to the schedule implementation.

    Args:
      data_iterator: only used for the first stage to load the whole batch.
      pipelines_indexed: a dictionary of all the pipeline stages, used to
        facilitate/emulate communication between stages.
    """
    # Clean per iteration runtime states
    self.stage.reset_runtime_states()
    self._internal_losses = []

    example_batch = None
    if self.stage.is_first:
      example_batch = max_utils.load_next_batch(
          data_iterator, example_batch, self.config
      )
      uuid = max_utils.generate_transfer_uuid(
          self.stage_index,
          self.num_stage_layers - 1,
          step_index,
          step_index,
          is_fwd=True,
      )
      # Example_batch has the targets to be transferred to the last stage.
      self.local_transfer_server.await_pull(
          uuid,
          [
              example_batch["targets"],
              example_batch["targets_position"],
              example_batch["targets_segmentation"],
          ],
      )
    elif self.stage.is_last:
      self._recv_targets_from_first_stage(step_index)
    elif self.stage.is_last:
      self._targets_cache = pipelines_indexed[0].targets_cache()
    # Run microbatches
    self._step_microbatches_fwd(step_index, example_batch, self._targets_cache)

  def step_backward(self, step_index):
    """Run one iteration of the pipeline schedule with *whole-batch* input.

    Will chunk the input into microbatches automatically, and go through the
    microbatches according to the schedule implementation.

    Args:
      step_index: the current step index used to generate transfer uuid.
    """

    # Run microbatches
    self._step_microbatches_bwd(step_index)

  def _fwd_send(
      self,
      outputs_mb,
      inputs_position_mb,
      inputs_segmentation_mb,
      microbatch_id,
      step_index,
  ):
    if self.stage.is_last:
      return

    uuid = max_utils.generate_transfer_uuid(
        self.stage_index, self._fwd_next_stage(), microbatch_id, step_index, is_fwd=True
        )
    self.local_transfer_server.await_pull(
        uuid,
        [outputs_mb, inputs_position_mb, inputs_segmentation_mb],
    )

  def _fwd_recv(
      self,
      inputs_wb: Any = None,
      microbatch_id: int = 0,
      step_index: int = 0,
  ):
    """Prepare inputs for a microbatch.

    Args:
      inputs_wb: whole batch of inputs; first dimension is batch size.
      microbatch_id: which microbatch to prepare
      microbatch_size: size of a microbatch
      pipeline_indexed: a dictionary of pipeline stages keyed by stage index.

    Returns:
      inputs for a microbatch
    """
    # If this is the first stage, the inputs come from the data iterator.
    # Otherwise, the inputs com from the previous stage.
    if self.stage.is_first:
      assert inputs_wb is not None
      return {
          "inputs": inputs_wb["inputs"][microbatch_id],
          "inputs_position": inputs_wb["inputs_position"][microbatch_id],
          "inputs_segmentation": inputs_wb["inputs_segmentation"][
              microbatch_id
          ],
      }
    else:
      inputs_mb_spec = jax.ShapeDtypeStruct(
          [
              self.microbatches_per_stage,
              self.config.max_target_length,
              self.config.emb_dim,
          ],
          jnp.bfloat16,
          sharding=self.stage.input_sharding(),
      )
      inputs_pos_seg_spec = jax.ShapeDtypeStruct(
          [self.microbatches_per_stage, self.config.max_target_length],
          jnp.int32,
          sharding=self.stage.input_sharding(),
      )
      uuid = max_utils.generate_transfer_uuid(
          self._fwd_prev_stage(), self.stage_index, microbatch_id, step_index, is_fwd=True
      )
      inputs_mb_hdl, inputs_position_mb_hdl, inputs_segmentation_mb_hdl = (
          self.transfer_connection.pull(
              uuid,
              [
                  inputs_mb_spec,
                  inputs_pos_seg_spec,
                  inputs_pos_seg_spec,
              ],
          )
      )
      return {
          "inputs": inputs_mb_hdl,
          "inputs_position": inputs_position_mb_hdl,
          "inputs_segmentation": inputs_segmentation_mb_hdl,
      }

  def _fwd_next_stage(self):
    return (self.stage_index + 1) % self.num_stage_layers

  def _fwd_prev_stage(self):
    return (self.stage_index - 1) % self.num_stage_layers

  def _bwd_next_stage(self):
    return (self.stage_index - 1) % self.num_stage_layers

  def _bwd_prev_stage(self):
    return (self.stage_index + 1) % self.num_stage_layers

  def _bwd_recv(
      self,
      microbatch_id: int,
      step_index: int,
  ):
    """Receive backward inputs(aka current stage outputs grads) from dependencies."""
    if self.stage.is_last:
      # last stage doesn't have backward inputs, use loss instead.
      return None
    else:
      input_grads_spec = jax.ShapeDtypeStruct(
          [
              self.microbatches_per_stage,
              self.config.max_target_length,
              self.config.emb_dim,
          ],
          jnp.bfloat16,
          sharding=self.stage.input_sharding(),
      )
      uuid = max_utils.generate_transfer_uuid(
          self._bwd_prev_stage(), self.stage_index, microbatch_id, step_index, is_fwd=False
      )
      input_grads = self.transfer_connection.pull(
          uuid,
          [input_grads_spec],
      )[0]
      return input_grads

  def _bwd_send(
      self,
      input_grads,
      microbatch_id,
      step_index,
  ):
    if self.stage.is_first:
      return
    uuid = max_utils.generate_transfer_uuid(
        self.stage_index, self._bwd_next_stage(), microbatch_id, step_index, is_fwd=False
    )
    logging.info(
        "#### input grads shape: %s, input grads dtype: %s, input grads"
        " sharding: %s",
        input_grads.shape,
        input_grads.dtype,
        input_grads.sharding,
    )
    logging.info(
        "### uuid for inputs grads bwd: %s, sender stage: %s, recv stage: %s",
        uuid,
        self.stage_index,
        self._bwd_next_stage(),
    )
    self.local_transfer_server.await_pull(
        uuid,
        [input_grads],
    )

  def _recv_targets_from_first_stage(self, wb_index):
    assert self.stage.is_last
    # receive 'targets'
    targets_pos_seg_spec = jax.ShapeDtypeStruct(
        [
            self.config.global_batch_size_to_load,
            self.config.max_target_length,
        ],
        jnp.int32,
        sharding=self.stage.target_sharding(),
    )
    uuid = max_utils.generate_transfer_uuid(
        0, self.num_stage_layers - 1, wb_index, wb_index, is_fwd=True
    )
    targets, targets_position, targets_segmentation = (
        self.transfer_connection.pull(
            uuid,
            [targets_pos_seg_spec, targets_pos_seg_spec, targets_pos_seg_spec],
        )
    )
    self._targets_cache = {
        "targets": targets,
        "targets_position": targets_position,
        "targets_segmentation": targets_segmentation,
        }

  def _maybe_compute_loss(self, logits, target_wb, mb_index, loss_func):
    if self.stage.is_last and self.stage.has_backward:
      assert target_wb is not None
      # split the targets into microbatches
      targets_microbatch_size = np.prod([
          self.microbatches_per_stage,
          self.config.max_target_length,
      ])
      num_microbatches = (
          targets_microbatch_size // target_wb["targets"].shape[1]
      )
      targets_mb = target_wb["targets"][
          mb_index * num_microbatches : (mb_index + 1) * num_microbatches
      ]
      targets_mb_segmentation = target_wb["targets_segmentation"][
          mb_index * num_microbatches : (mb_index + 1) * num_microbatches
      ]
      loss_broadcasted = loss_func(
          logits,
          targets_mb,
          targets_mb_segmentation,
      )
      self._internal_losses.append(loss_broadcasted)

  def _maybe_get_loss(self, mb_index):
    valid_index = 0 <= mb_index < len(self._internal_losses)
    if self.stage.is_last and self.stage.has_backward and valid_index:
      return self._internal_losses[mb_index]
    elif self._internal_losses and not valid_index:
      raise RuntimeError(
          f"Loss for microbatch {mb_index} is not available. "
          f"Available losses for microbatches: {self._internal_losses}"
      )
    else:
      return None

  def _step_microbatches_fwd(
      self,
      step_index: int,
      example_batch: Optional[Any] = None,
      targets: Optional[Any] = None,
  ):
    """Run one iteration of the pipeline schedule with list of microbatches.

    Will go through all the microbatches according to the GPipe schedule.

    Args:

    microbatches: list of microbatch args.
    """
    # Run microbatches
    inputs_wb = None
    if example_batch is not None:
      inputs_wb = example_batch
    nextrng = jax.jit(jax.random.fold_in)(self.stage.init_rng, step_index)
    for i in range(self.num_microbatches):
      # receive inputs for forward pass.
      input_mb = self._fwd_recv(
          inputs_wb, i, step_index
      )
      with jax.named_scope(f"Forward_stage_{self.stage.stage_index}_mb_{i}"):
        outputs = self.stage.forward_step(self.stage.get_jitted_forward_func(), i, input_mb, nextrng)  # type: ignore[index]
        self._fwd_send(
            outputs,
            input_mb["inputs_position"],
            input_mb["inputs_segmentation"],
            i,
            step_index,
        )
        self._maybe_compute_loss(
            outputs, targets, i, self.stage.get_loss_func()
        )

  def _step_microbatches_bwd(
      self,
      step_index: int,
      ):
    """Run one iteration of the pipeline schedule with list of microbatches.

    Will go through all the microbatches according to the GPipe schedule.

    Args:
    targets:
    pipeline_indexed: a dictionary of pipeline stages keyed by stage index.
    """
    # Run microbatches
    for i in range(self.num_microbatches):
      with jax.named_scope(f"Backward_stage_{self.stage.stage_index}_mb_{i}"):
        # receive inputs for backward pass.
        output_grads = self._bwd_recv(i, step_index)
        loss = self._maybe_get_loss(i)
        input_grads = self.stage.backward_step(
            i, loss=loss, output_grads=output_grads
        )
        self._bwd_send(input_grads, i, step_index)

        # wait until the whole batch is done.
        if (
            self.stage.is_first or self.stage_index == 1
        ) and i == self.num_microbatches - 1:
          self.stage.block_until_ready_train_state()
