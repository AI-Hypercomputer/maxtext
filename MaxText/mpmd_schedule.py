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
from jax.sharding import Mesh
from maxtext import common_types
from maxtext import profiler
from maxtext.mpmd.maxtext_pp import max_utils
from maxtext.mpmd.maxtext_pp.stage import PipelineStage
import numpy as np


class PipelineGPipeCircular(ABC):
  """Pipeline for GPipe."""

  def __init__(self, config: common_types.Config, stage_index: int):
    # Initialize the pipeline hyperparameters
    self.stage_index = stage_index
    self.config = config

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
    self.microbatches_per_stage = 4
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

  def step_forward(self, data_iterator, pipelines_indexed, step):
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

    example_batch = None
    if self.stage.is_first:
      example_batch = max_utils.load_next_batch(
          data_iterator, example_batch, self.config
      )
      # Example_batch has the targets to be transferred to the last stage.
      self._targets_cache = {
          "targets": example_batch["targets"],
          "targets_position": example_batch["targets_position"],
          "targets_segmentation": example_batch["targets_segmentation"],
      }
    elif self.stage.is_last:
      self._targets_cache = pipelines_indexed[0].targets_cache()
    # Run microbatches
    self._step_microbatches_fwd(
        example_batch, self._targets_cache, pipelines_indexed
    )

  def step_backward(self, pipelines_indexed):
    """Run one iteration of the pipeline schedule with *whole-batch* input.

    Will chunk the input into microbatches automatically, and go through the
    microbatches according to the schedule implementation.

    Args:
      pipelines_indexed: a dictionary of all the pipeline stages, used to
        facilitate/emulate communication between stages.
    """

    # Run microbatches
    self._step_microbatches_bwd(pipelines_indexed)

  def _fwd_recv(
      self,
      inputs_wb: Any = None,
      microbatch_id: int = 0,
      microbatch_size: int = 0,
      pipeline_indexed: Optional[Dict[int, Any]] = None,
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
    # TODO(linchai): need to optimize.
    # If this is the first stage, the inputs come from the data iterator.
    # Otherwise, the inputs com from the previous stage.
    if self.stage.is_first:
      assert inputs_wb is not None
      inputs_wb_inputs = inputs_wb["inputs"]
      num_microbatches = microbatch_size // inputs_wb_inputs.shape[1]
      inputs_mb_inputs = inputs_wb_inputs[
          microbatch_id
          * num_microbatches : (microbatch_id + 1)
          * num_microbatches
      ]
      inputs_mb_inputs_hdl = jax.device_put(
          inputs_mb_inputs,
          self.stage.input_sharding(),
      )

      inputs_wb_inputs_position = inputs_wb["inputs_position"]
      num_microbatches = microbatch_size // inputs_wb_inputs_position.shape[1]
      inputs_position_mb = inputs_wb_inputs_position[
          microbatch_id
          * num_microbatches : (microbatch_id + 1)
          * num_microbatches
      ]
      inputs_position_mb_hdl = jax.device_put(
          inputs_position_mb,
          self.stage.input_sharding(),
      )

      inputs_wb_inputs_segmentation = inputs_wb["inputs_segmentation"]
      num_microbatches = (
          microbatch_size // inputs_wb_inputs_segmentation.shape[1]
      )
      inputs_segmentation_mb = inputs_wb_inputs_segmentation[
          microbatch_id
          * num_microbatches : (microbatch_id + 1)
          * num_microbatches
      ]
      inputs_segmentation_mb_hdl = jax.device_put(
          inputs_segmentation_mb,
          self.stage.input_sharding(),
      )
      return {
          "inputs": inputs_mb_inputs_hdl,
          "inputs_position": inputs_position_mb_hdl,
          "inputs_segmentation": inputs_segmentation_mb_hdl,
      }
    else:
      fwd_previous_stage_index = (self.stage_index - 1) % self.num_stage_layers
      (inputs_mb, previous_inputs_mb) = (
          pipeline_indexed[fwd_previous_stage_index].stage.fwd_cache[
              microbatch_id
          ]
      )
      inputs_mb_hdl = jax.device_put(inputs_mb, self.stage.input_sharding())
      inputs_position_mb_hdl = jax.device_put(
          previous_inputs_mb["inputs_position"], self.stage.input_sharding()
      )
      inputs_segmentation_mb_hdl = jax.device_put(
          previous_inputs_mb["inputs_segmentation"], self.stage.input_sharding()
      )
      return {
          "inputs": inputs_mb_hdl,
          "inputs_position": inputs_position_mb_hdl,
          "inputs_segmentation": inputs_segmentation_mb_hdl,
      }

  def _bwd_recv(self, microbatch_id: int, pipeline_indexed: Dict[int, Any]):
    """Receive backward inputs(aka current stage outputs grads) from dependencies."""
    if self.stage.is_last:
      # last stage doesn't have backward inputs, use loss instead.
      return None
    else:
      bwd_previous_stage_index = (self.stage_index + 1) % self.num_stage_layers
      received_from_previous_stage = pipeline_indexed[
          bwd_previous_stage_index
      ].stage.bwd_cache[microbatch_id]
      return jax.device_put(
          received_from_previous_stage, self.stage.input_sharding()
      )

  def _maybe_compute_loss(self, logits, target_mbs, mb_index, loss_func):
    if self.stage.is_last and self.stage.has_backward:
      assert target_mbs is not None
      # split the targets into microbatches
      targets_microbatch_size = np.prod([
          self.microbatches_per_stage,
          self.config.max_target_length,
      ])
      num_microbatches = (
          targets_microbatch_size // target_mbs["targets"].shape[1]
      )
      targets_mb = target_mbs["targets"][
          mb_index * num_microbatches : (mb_index + 1) * num_microbatches
      ]
      targets_mb_hdl = jax.device_put(
          targets_mb,
          self.stage.input_sharding(),
      )
      targets_mb_segmentation = target_mbs["targets_segmentation"][
          mb_index * num_microbatches : (mb_index + 1) * num_microbatches
      ]
      targets_mb_segmentation_hdl = jax.device_put(
          targets_mb_segmentation,
          self.stage.input_sharding(),
      )
      loss = loss_func(
          logits,
          targets_mb_hdl,
          targets_mb_segmentation_hdl,
      )
      self._internal_losses.append(loss)

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
      example_batch: Optional[Any] = None,
      targets: Optional[Any] = None,
      pipeline_indexed: Optional[Dict[int, Any]] = None,
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
    for i in range(self.num_microbatches):
      # receive inputs for forward pass.
      if self.stage_index == 0:
        input_microbatch_size = np.prod([
            self.microbatches_per_stage,
            self.config.max_target_length,
        ])
      else:
        input_microbatch_size = np.prod([
            self.microbatches_per_stage,
            self.config.max_target_length,
            self.config.emb_dim,
        ])
      input_mb = self._fwd_recv(
          inputs_wb, i, input_microbatch_size, pipeline_indexed
      )
      with jax.named_scope(f"Forward_stage_{self.stage.stage_index}_mb_{i}"):
        outputs = self.stage.forward_step(self.stage.get_jitted_forward_func(), i, input_mb)  # type: ignore[index]
        self._maybe_compute_loss(
            outputs, targets, i, self.stage.get_loss_func()
        )

  def _step_microbatches_bwd(
      self,
      pipeline_indexed: Optional[Dict[int, Any]] = None,
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
        output_grads = self._bwd_recv(i, pipeline_indexed)
        loss = self._maybe_get_loss(i)
        self.stage.backward_step(i, loss=loss, output_grads=output_grads)
