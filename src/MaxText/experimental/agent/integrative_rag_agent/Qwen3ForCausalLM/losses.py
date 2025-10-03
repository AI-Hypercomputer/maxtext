
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp


def load_balancing_loss_func(
    gate_logits: Union[jnp.ndarray, Tuple[jnp.ndarray, ...], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[jnp.ndarray] = None,
) -> Union[jnp.ndarray, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in JAX.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`jnp.ndarray`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        concatenated_gate_logits = jnp.concatenate(gate_logits, axis=0)

    routing_weights = jax.nn.softmax(concatenated_gate_logits, axis=-1)

    _, selected_experts = jax.lax.top_k(routing_weights, top_k)

    expert_mask = jax.nn.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = jnp.mean(expert_mask.astype(jnp.float32), axis=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = jnp.mean(routing_weights, axis=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = jnp.broadcast_to(
            attention_mask[jnp.newaxis, :, :, jnp.newaxis, jnp.newaxis],
            (num_hidden_layers, batch_size, sequence_length, top_k, num_experts),
        )
        expert_attention_mask = expert_attention_mask.reshape(-1, top_k, num_experts)

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = jnp.sum(expert_mask.astype(jnp.float32) * expert_attention_mask, axis=0) / jnp.sum(
            expert_attention_mask, axis=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = jnp.broadcast_to(
            attention_mask[jnp.newaxis, :, :, jnp.newaxis],
            (num_hidden_layers, batch_size, sequence_length, num_experts),
        )
        router_per_expert_attention_mask = router_per_expert_attention_mask.reshape(-1, num_experts)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = jnp.sum(routing_weights * router_per_expert_attention_mask, axis=0) / jnp.sum(
            router_per_expert_attention_mask, axis=0
        )

    overall_loss = jnp.sum(tokens_per_expert * jnp.expand_dims(router_prob_per_expert, axis=0))
    return overall_loss * num_experts

# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

# Assuming RTDetrLoss is a Flax module defined in the project's `layers` directory
# and RTDetrConfig is defined in the project's `config` directory.
# These are not part of the provided JAX modules and would need to be converted separately.
# from ...configs.rtdetr_config import RTDetrConfig
# from ...layers.loss import RTDetrLoss

# Dummy placeholders for type hinting if the actual modules are not available.
RTDetrConfig = Any
RTDetrLoss = Any

Array = jnp.ndarray


def _set_aux_loss(outputs_class: Array, outputs_coord: Array) -> List[Dict[str, Array]]:
  """Helper function to set auxiliary loss outputs."""
  # this is a workaround to make torchscript happy, as torchscript
  # doesn't support dictionary with non-homogeneous values, such
  # as a dict having both a Tensor and a list.
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


def RTDetrForObjectDetectionLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    config: RTDetrConfig,
    outputs_class: Optional[Array] = None,
    outputs_coord: Optional[Array] = None,
    enc_topk_logits: Optional[Array] = None,
    enc_topk_bboxes: Optional[Array] = None,
    denoising_meta_values: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Array, Dict[str, Array], List[Dict[str, Array]]]:
  """
  Computes the loss for RT-DETR object detection.

  Args:
      logits: Logits from the model's final layer.
      labels: A list of dictionaries, each containing ground-truth 'class_labels' and 'boxes'.
      pred_boxes: Predicted boxes from the model's final layer.
      config: The model configuration object.
      outputs_class: Class logits from auxiliary decoder layers.
      outputs_coord: Box predictions from auxiliary decoder layers.
      enc_topk_logits: Top-K logits from the encoder.
      enc_topk_bboxes: Top-K bounding boxes from the encoder.
      denoising_meta_values: Metadata for denoising task.
      **kwargs: Additional arguments.

  Returns:
      A tuple containing:
      - The total loss.
      - A dictionary of individual loss components.
      - A list of auxiliary outputs.
  """
  criterion = RTDetrLoss(config=config)

  # Second: compute the losses, based on outputs and labels
  outputs_loss = {}
  outputs_loss["logits"] = logits
  outputs_loss["pred_boxes"] = pred_boxes

  # Initialize to fix potential UnboundLocalError from original code
  auxiliary_outputs = []

  if config.auxiliary_loss:
    if denoising_meta_values is not None:
      split_index = denoising_meta_values["dn_num_split"]
      dn_out_coord, outputs_coord = jnp.split(outputs_coord, [split_index], axis=2)
      dn_out_class, outputs_class = jnp.split(outputs_class, [split_index], axis=2)

    transposed_class = jnp.transpose(outputs_class[:, :-1], (1, 0) + tuple(range(2, outputs_class.ndim)))
    transposed_coord = jnp.transpose(outputs_coord[:, :-1], (1, 0) + tuple(range(2, outputs_coord.ndim)))
    auxiliary_outputs = _set_aux_loss(transposed_class, transposed_coord)

    outputs_loss["auxiliary_outputs"] = auxiliary_outputs
    outputs_loss["auxiliary_outputs"].extend(_set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

    if denoising_meta_values is not None:
      transposed_dn_class = jnp.transpose(dn_out_class, (1, 0) + tuple(range(2, dn_out_class.ndim)))
      transposed_dn_coord = jnp.transpose(dn_out_coord, (1, 0) + tuple(range(2, dn_out_coord.ndim)))
      outputs_loss["dn_auxiliary_outputs"] = _set_aux_loss(transposed_dn_class, transposed_dn_coord)
      outputs_loss["denoising_meta_values"] = denoising_meta_values

  loss_dict = criterion(outputs_loss, labels)

  loss = sum(loss_dict.values())
  return loss, loss_dict, auxiliary_outputs

from typing import Optional

import jax.numpy as jnp
from jax import Array
import optax


def ForMaskedLMLoss(
    logits: Array,
    labels: Array,
    vocab_size: int,
    num_items_in_batch: Optional[Array] = None,
    ignore_index: int = -100,
    **kwargs,
) -> Array:
  """
  Computes the masked language modeling loss.

  Args:
    logits: The model's output logits, shape [batch_size, seq_len, vocab_size].
    labels: The ground truth labels, shape [batch_size, seq_len].
    vocab_size: The size of the vocabulary.
    num_items_in_batch: Optional tensor for normalizing the loss.
    ignore_index: The index to ignore in the loss calculation.
    **kwargs: Additional arguments (not used in this implementation).

  Returns:
    A scalar tensor representing the final loss.
  """
  # Upcast to float to compute the loss and avoid potential precision issues
  logits = logits.astype(jnp.float32)

  # Flatten the tokens
  logits = logits.reshape(-1, vocab_size)
  labels = labels.reshape(-1)

  # This logic is a JAX-native equivalent of the `fixed_cross_entropy`
  # function from the original PyTorch file.
  # Calculate per-token loss
  per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)

  # Create a mask for valid (non-ignored) tokens
  mask = (labels != ignore_index).astype(per_token_loss.dtype)

  # Apply the mask to the per-token loss
  masked_loss = per_token_loss * mask

  if num_items_in_batch is not None:
    # Sum the loss and divide by the number of items in the batch
    total_loss = jnp.sum(masked_loss)
    loss = total_loss / num_items_in_batch
  else:
    # Mean loss over valid tokens
    total_loss = jnp.sum(masked_loss)
    num_valid_tokens = jnp.sum(mask)
    # Avoid division by zero if there are no valid tokens
    loss = jnp.where(num_valid_tokens > 0, total_loss / num_valid_tokens, 0.0)

  return loss

from typing import Any

import jax.numpy as jnp
from maxtext.common_types import Array, Config

# Reused from Qwen3ForCausalLM.losses.ForMaskedLMLoss
from Qwen3ForCausalLM.losses import ForMaskedLMLoss


def ForTokenClassification(logits: Array, labels: Array, config: Config, **kwargs: Any) -> Array:
  """Computes the token classification loss.

  This function reshapes logits and labels and computes the cross-entropy loss,
  which is functionally equivalent to the masked language modeling loss with
  the vocabulary size set to the number of labels.

  Args:
    logits: The raw, unscaled output from the model.
    labels: The ground truth labels.
    config: The model's configuration object, used to get the number of labels.
    **kwargs: Additional arguments passed to the loss function (e.g.,
      `ignore_index`, `num_items_in_batch`).

  Returns:
    The computed scalar loss.
  """
  # The logic of reshaping, casting, and computing cross-entropy is encapsulated
  # in the reused ForMaskedLMLoss function.
  return ForMaskedLMLoss(logits=logits, labels=labels, vocab_size=config.num_labels, **kwargs)

from typing import Optional, Sequence, List, Dict

from maxtext.common_types import Array


def _set_aux_loss2(
    outputs_class: Sequence[Array],
    outputs_coord: Sequence[Array],
    outputs_corners: Sequence[Array],
    outputs_ref: Sequence[Array],
    teacher_corners: Optional[Array] = None,
    teacher_logits: Optional[Array] = None,
) -> List[Dict[str, Optional[Array]]]:
  """A helper function to structure auxiliary loss outputs.

  This is a workaround for TorchScript limitations with non-homogeneous dictionaries,
  a detail retained in this JAX implementation for structural consistency.

  Args:
    outputs_class: A sequence of class logits from intermediate decoder layers.
    outputs_coord: A sequence of coordinate predictions from intermediate decoder
      layers.
    outputs_corners: A sequence of corner predictions from intermediate decoder
      layers.
    outputs_ref: A sequence of reference points from intermediate decoder
      layers.
    teacher_corners: Optional teacher corner predictions.
    teacher_logits: Optional teacher logits.

  Returns:
    A list of dictionaries, each containing the outputs for a single auxiliary
    layer.
  """
  return [
      {
          "logits": a,
          "pred_boxes": b,
          "pred_corners": c,
          "ref_points": d,
          "teacher_corners": teacher_corners,
          "teacher_logits": teacher_logits,
      }
      for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
  ]

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

import jax.numpy as jnp
import optax


def fixed_cross_entropy(
    source: jnp.ndarray,
    target: jnp.ndarray,
    num_items_in_batch: Optional[jnp.ndarray] = None,
    ignore_index: int = -100,
    **kwargs,
) -> jnp.ndarray:
  """
  Computes cross-entropy loss with optional manual normalization.

  This function mirrors the behavior of PyTorch's functional.cross_entropy,
  particularly how it handles reduction and normalization when `num_items_in_batch`
  is provided.

  Args:
    source: Logits, a float array of shape [batch..., num_classes].
    target: Labels, an integer array of shape [batch...].
    num_items_in_batch: An optional scalar for normalizing the summed loss.
      If provided, the loss is summed and then divided by this value.
      If not provided, the loss is averaged over valid (non-ignored) tokens.
    ignore_index: The label value to ignore in the loss calculation.
    **kwargs: Additional arguments, not used.

  Returns:
    A scalar float array representing the calculated loss.
  """
  # Calculate per-token cross-entropy loss.
  per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits=source, labels=target)

  # Create a mask for valid (non-ignored) tokens.
  weights = jnp.where(target != ignore_index, 1.0, 0.0)

  # Apply the mask and sum the losses over all valid tokens.
  total_loss = jnp.sum(per_token_loss * weights)

  if num_items_in_batch is not None:
    # Corresponds to PyTorch's reduction='sum' followed by manual division.
    loss = total_loss / num_items_in_batch
  else:
    # Corresponds to PyTorch's reduction='mean'.
    num_valid_tokens = jnp.sum(weights)
    # Avoid division by zero if there are no valid tokens.
    loss = total_loss / jnp.maximum(num_valid_tokens, 1.0)

  return loss

from typing import Optional

import jax.numpy as jnp
from maxtext.common_types import Array


def ForCausalLMLoss(
    logits: Array,
    labels: Array,
    vocab_size: int,
    num_items_in_batch: Optional[Array] = None,
    ignore_index: int = -100,
    shift_labels: Optional[Array] = None,
    **kwargs,
) -> Array:
  """Computes causal language modeling loss."""
  # Upcast to float if we need to compute the loss to avoid potential precision issues
  logits = logits.astype(jnp.float32)

  if shift_labels is None:
    # Shift so that tokens < n predict n
    pad_widths = [(0, 0)] * (labels.ndim - 1) + [(0, 1)]
    labels = jnp.pad(labels, pad_widths, constant_values=ignore_index)
    shift_labels = labels[..., 1:]

  # Flatten the tokens
  logits = logits.reshape(-1, vocab_size)
  shift_labels = shift_labels.reshape(-1)
  # In JAX, device placement is handled by pjit, so we don't need to move tensors manually.

  # Reused from Qwen3ForCausalLM.losses.fixed_cross_entropy
  loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
  return loss

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
Loss functions for various models.
"""

from typing import Optional

import jax.numpy as jnp

from maxtext.common_types import Array
# from Qwen3ForCausalLM.losses.fixed_cross_entropy import fixed_cross_entropy
from ..Qwen3ForCausalLM.losses import fixed_cross_entropy


def ForQuestionAnsweringLoss(
    start_logits: Array,
    end_logits: Array,
    start_positions: Optional[Array],
    end_positions: Optional[Array],
    **kwargs,
) -> Optional[Array]:
  """Computes the loss for question answering tasks.

  Args:
    start_logits: Logits for the start position of the answer.
    end_logits: Logits for the end position of the answer.
    start_positions: Ground truth start positions.
    end_positions: Ground truth end positions.
    **kwargs: Additional arguments for the loss function.

  Returns:
    The total loss as a scalar array, or None if positions are not provided.
  """
  total_loss = None
  if start_positions is not None and end_positions is not None:
    # If we are on multi-GPU, split add a dimension
    if start_positions.ndim > 1:
      start_positions = jnp.squeeze(start_positions, axis=-1)
    if end_positions.ndim > 1:
      end_positions = jnp.squeeze(end_positions, axis=-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.shape[1]
    start_positions = jnp.clip(start_positions, 0, ignored_index)
    end_positions = jnp.clip(end_positions, 0, ignored_index)

    # Reused from Qwen3ForCausalLM.losses.fixed_cross_entropy
    start_loss = fixed_cross_entropy(start_logits, start_positions, ignore_index=ignored_index, **kwargs)
    end_loss = fixed_cross_entropy(end_logits, end_positions, ignore_index=ignored_index, **kwargs)
    total_loss = (start_loss + end_loss) / 2
  return total_loss

import jax.numpy as jnp
import optax
from maxtext.common_types import Array

# Reused from Qwen3ForCausalLM.losses.fixed_cross_entropy
from Qwen3ForCausalLM.losses import fixed_cross_entropy


def ForSequenceClassificationLoss(labels: Array, pooled_logits: Array, config, **kwargs) -> Array:
  """Computes the loss for sequence classification tasks."""
  num_labels = config.num_labels
  # Determine problem type if not specified
  if config.problem_type is None:
    if num_labels == 1:
      config.problem_type = "regression"
    elif num_labels > 1 and jnp.issubdtype(labels.dtype, jnp.integer):
      config.problem_type = "single_label_classification"
    else:
      config.problem_type = "multi_label_classification"

  if config.problem_type == "regression":
    # MSELoss equivalent
    if num_labels == 1:
      loss = jnp.mean(jnp.square(jnp.squeeze(pooled_logits) - jnp.squeeze(labels)))
    else:
      loss = jnp.mean(jnp.square(pooled_logits - labels))
    return loss

  if config.problem_type == "single_label_classification":
    return fixed_cross_entropy(pooled_logits.reshape((-1, num_labels)), labels.reshape((-1,)), **kwargs)

  if config.problem_type == "multi_label_classification":
    # BCEWithLogitsLoss equivalent with mean reduction
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pooled_logits, labels))
    return loss

  raise RuntimeError(f"Invalid problem type: {config.problem_type}")

from jax import Array
import jax.numpy as jnp
import jax.nn
import optax


def sigmoid_focal_loss(
    inputs: Array, targets: Array, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0
) -> Array:
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.

    Args:
        inputs (`jax.Array` of arbitrary shape):
            The predictions for each example.
        targets (`jax.Array` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`float`):
            The number of boxes to normalize the loss by.
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`float`, *optional*, defaults to `2.0`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = jax.nn.sigmoid(inputs)
    ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return jnp.mean(loss, axis=1).sum() / num_boxes

from jax import nn
import jax.numpy as jnp

from maxtext.common_types import Array


def dice_loss(inputs: Array, targets: Array, num_boxes: float) -> Array:
  """
  Computes the DICE loss, similar to generalized IOU for masks.

  Args:
    inputs: A float array of arbitrary shape. The predictions for each example.
    targets: A float array with the same shape as inputs. Stores the binary
      classification label for each element in inputs (0 for the negative class
      and 1 for the positive class).
    num_boxes: The number of boxes to normalize the loss by.

  Returns:
    The total DICE loss normalized by the number of boxes.
  """
  inputs = nn.sigmoid(inputs)
  inputs = inputs.reshape((inputs.shape[0], -1))
  numerator = 2 * (inputs * targets).sum(axis=1)
  denominator = inputs.sum(axis=-1) + targets.sum(axis=-1)
  loss = 1 - (numerator + 1) / (denominator + 1)
  return loss.sum() / num_boxes

import jax
import jax.numpy as jnp
import optax
from jax import Array


def sigmoid_focal_loss(
    inputs: Array,
    targets: Array,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Array:
  """
  Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.

  Args:
      inputs: A float tensor of arbitrary shape.
          The predictions for each example.
      targets: A float tensor with the same shape as `inputs`.
          A tensor storing the binary classification label for each element in the `inputs`
          (0 for the negative class and 1 for the positive class).
      num_boxes: The total number of boxes in the batch.
      alpha: Optional weighting factor in the range (0,1) to balance
          positive vs. negative examples.
      gamma: Exponent of the modulating factor (1 - p_t) to balance
          easy vs hard examples.

  Returns:
      Loss tensor
  """
  prob = jax.nn.sigmoid(inputs)
  ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)
  # add modulating factor
  p_t = prob * targets + (1 - prob) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  return jnp.sum(loss) / num_boxes

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax.experimental.pallas.ops.tpu.splash_attention import pallas as tpu_pallas

# Reused from /.../Qwen3ForCausalLM/bbox_utils.py
from Qwen3ForCausalLM.bbox_utils import _center_to_corners_format_jax as center_to_corners_format
# Reused from /.../Qwen3ForCausalLM/structures.py
from Qwen3ForCausalLM.structures import NestedTensor
# Reused from /.../Qwen3ForCausalLM/box_utils.py
from Qwen3ForCausalLM.box_utils import generalized_box_iou
# Reused from /.../Qwen3ForCausalLM/losses.py
from Qwen3ForCausalLM.losses import dice_loss, sigmoid_focal_loss
# Reused from /.../Qwen3ForCausalLM/utils.py
from Qwen3ForCausalLM.utils import nested_tensor_from_tensor_list

from maxtext.common_types import Array, DType


class ImageLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Attributes:
        matcher (`nn.Module`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`list[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    matcher: nn.Module
    num_classes: int
    eos_coef: float
    losses: List[str]

    def setup(self):
        empty_weight = jnp.ones(self.num_classes + 1)
        self.empty_weight = empty_weight.at[-1].set(self.eos_coef)

    def loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float) -> Dict:
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = jnp.full(source_logits.shape[:2], self.num_classes, dtype=jnp.int32)
        target_classes = target_classes.at[idx].set(target_classes_o)

        # In PyTorch, nn.functional.cross_entropy combines log_softmax and nll_loss.
        # It also supports class weighting. We can implement this manually in JAX.
        log_probs = jax.nn.log_softmax(source_logits, axis=-1)
        one_hot_labels = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)
        per_token_loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)

        class_weights = self.empty_weight[target_classes]
        weighted_loss = per_token_loss * class_weights

        loss_ce = jnp.mean(weighted_loss)
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_cardinality(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float) -> Dict:
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """

        def compute_cardinality_error(logits, targets):
            target_lengths = jnp.array([len(v["class_labels"]) for v in targets])
            # Count the number of predictions that are NOT "no-object" (which is the last class)
            card_pred = (jnp.argmax(logits, axis=-1) != logits.shape[-1] - 1).sum(axis=1)
            card_err = jnp.mean(jnp.abs(card_pred.astype(jnp.float32) - target_lengths.astype(jnp.float32)))
            return {"cardinality_error": card_err}

        return jax.lax.stop_gradient(compute_cardinality_error(outputs["logits"], targets))

    def loss_boxes(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float) -> Dict:
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = jnp.abs(source_boxes - target_boxes)

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - jnp.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float) -> Dict:
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.astype(source_masks.dtype)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = jax.image.resize(
            source_masks[:, None],
            shape=(source_masks.shape[0], 1) + target_masks.shape[-2:],
            method="bilinear",
        )
        source_masks = source_masks[:, 0].reshape(source_masks.shape[0], -1)

        target_masks = target_masks.reshape(target_masks.shape[0], -1)
        target_masks = target_masks.reshape(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices: List[Tuple]) -> Tuple[Array, Array]:
        # permute predictions following indices
        batch_idx = jnp.concatenate([jnp.full_like(source, i, dtype=jnp.int32) for i, (source, _) in enumerate(indices)])
        source_idx = jnp.concatenate([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices: List[Tuple]) -> Tuple[Array, Array]:
        # permute targets following indices
        batch_idx = jnp.concatenate([jnp.full_like(target, i, dtype=jnp.int32) for i, (_, target) in enumerate(indices)])
        target_idx = jnp.concatenate([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(
        self, loss: str, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float
    ) -> Dict:
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def __call__(self, outputs: Dict, targets: List[Dict]) -> Dict:
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`list[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = jnp.asarray([num_boxes], dtype=jnp.float32)

        # In JAX, this is done using psum over the data parallel axis
        num_boxes = jax.lax.psum(num_boxes, axis_name="data")
        world_size = jax.lax.psum(jnp.array([1.0]), axis_name="data")

        num_boxes = jnp.clip(num_boxes / world_size, a_min=1.0)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple

# Reused from Qwen3ForCausalLM.losses.ImageLoss
from Qwen3ForCausalLM.losses import ImageLoss
# Reused from Qwen3ForCausalLM.losses.sigmoid_focal_loss
from Qwen3ForCausalLM.losses import sigmoid_focal_loss


class DeformableDetrImageLoss(ImageLoss):
  """
  Computes the loss for Deformable DETR.
  The process happens in two steps:
  1) we compute hungarian assignment between ground truth boxes and the outputs of the model
  2) we supervise each pair of matched ground-truth / prediction
  """

  focal_alpha: float

  # removed logging parameter, which was part of the original implementation
  def loss_labels(
      self,
      outputs: Dict[str, jnp.ndarray],
      targets: List[Dict[str, jnp.ndarray]],
      indices: List[Tuple[jnp.ndarray, jnp.ndarray]],
      num_boxes: int,
      **kwargs,
  ) -> Dict[str, jnp.ndarray]:
    """
    Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
    of dim [nb_target_boxes]
    """
    if "logits" not in outputs:
      raise KeyError("No logits were found in the outputs")
    source_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = jnp.full(source_logits.shape[:2], self.num_classes, dtype=jnp.int32)
    target_classes = target_classes.at[idx].set(target_classes_o)

    target_classes_onehot = jax.nn.one_hot(
        target_classes, num_classes=source_logits.shape[2] + 1, dtype=source_logits.dtype
    )

    target_classes_onehot = target_classes_onehot[..., :-1]
    loss_ce = sigmoid_focal_loss(
        source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2
    )
    losses = {"loss_ce": loss_ce}

    return losses

from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from Qwen3ForCausalLM.losses import sigmoid_focal_loss


# Re-implementation of src.MaxText.layers.object_detection.ImageLoss
class ImageLoss(nn.Module):
  """Base class for image losses."""

  def _get_source_permutation_idx(
      self, indices: List[Tuple[jnp.ndarray, jnp.ndarray]]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Permute predictions following indices.
    """
    batch_idx = jnp.concatenate([jnp.full_like(src, i, dtype=src.dtype) for i, (src, _) in enumerate(indices)])
    src_idx = jnp.concatenate([src for (src, _) in indices])
    return batch_idx, src_idx


class GroundingDinoImageLoss(ImageLoss):
  """
  This class computes the losses for `GroundingDinoForObjectDetection`. The process happens in two steps: 1) we
  compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
  matched ground-truth / prediction (supervise class and box).

  Attributes:
      matcher (`GroundingDinoHungarianMatcher`):
          Module able to compute a matching between targets and proposals.
      focal_alpha (`float`):
          Alpha parameter in focal loss.
      losses (`list[str]`):
          List of all the losses to be applied. See `get_loss` for a list of all available losses.
  """

  matcher: Any
  focal_alpha: float
  losses: List[str]

  def _get_target_classes_one_hot(
      self,
      outputs: Dict[str, Any],
      targets: List[Dict[str, jnp.ndarray]],
      indices: List[Tuple[jnp.ndarray, jnp.ndarray]],
  ) -> jnp.ndarray:
    """
    Create one_hot based on the matching indices
    """
    logits = outputs["logits"]
    # Add offsets to class_labels to select the correct label map
    label_map_lengths = [label_map.shape[0] for label_map in outputs["label_maps"]]
    offsets = jnp.cumsum(jnp.array([0] + label_map_lengths[:-1]))

    class_labels_list = []
    for i, (target, (_, J)) in enumerate(zip(targets, indices)):
      labels = target["class_labels"][J] + offsets[i]
      class_labels_list.append(labels)
    class_labels = jnp.concatenate(class_labels_list)

    label_maps = jnp.concatenate(outputs["label_maps"], axis=0)

    idx = self._get_source_permutation_idx(indices)
    target_classes_onehot = jnp.zeros_like(logits, dtype=jnp.int32)
    target_classes_onehot = target_classes_onehot.at[idx].set(label_maps[class_labels].astype(jnp.int32))

    return target_classes_onehot

  def loss_labels(
      self,
      outputs: Dict[str, Any],
      targets: List[Dict[str, jnp.ndarray]],
      indices: List[Tuple[jnp.ndarray, jnp.ndarray]],
      num_boxes: int,
  ) -> Dict[str, jnp.ndarray]:
    """
    Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
    of dim [nb_target_boxes]
    """
    if "logits" not in outputs:
      raise KeyError("No logits were found in the outputs")
    if "text_mask" not in outputs:
      raise KeyError("No text_mask were found in the outputs")

    target_classes_onehot = self._get_target_classes_one_hot(outputs, targets, indices)
    source_logits = outputs["logits"]
    text_mask = outputs["text_mask"]

    # Select only valid logits
    source_logits = source_logits[text_mask]
    target_classes_onehot = target_classes_onehot[text_mask]

    target_classes_onehot = target_classes_onehot.astype(jnp.float32)
    loss_ce = sigmoid_focal_loss(
        inputs=source_logits,
        targets=target_classes_onehot,
        num_boxes=num_boxes,
        alpha=self.focal_alpha,
        gamma=2,
    )

    losses = {"loss_ce": loss_ce}

    return losses

from typing import Dict, List, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

# Re-used modules
from Qwen3ForCausalLM.matchers import RTDetrHungarianMatcher  # path: Qwen3ForCausalLM.matchers.RTDetrHungarianMatcher
from Qwen3ForCausalLM.bbox_utils import center_to_corners_format  # path: Qwen3ForCausalLM.bbox_utils.center_to_corners_format
from Qwen3ForCausalLM.box_utils import box_iou, generalized_box_iou  # path: Qwen3ForCausalLM.box_utils.box_iou, Qwen3ForCausalLM.box_utils.generalized_box_iou
from Qwen3ForCausalLM.losses import dice_loss, sigmoid_focal_loss  # path: Qwen3ForCausalLM.losses.dice_loss, Qwen3ForCausalLM.losses.sigmoid_focal_loss
from Qwen3ForCausalLM.modeling_utils import nested_tensor_from_tensor_list  # path: Qwen3ForCausalLM.modeling_utils.nested_tensor_from_tensor_list


def binary_cross_entropy_with_logits(logits, labels, weight=None, reduction="none"):
  """
    Computes weighted binary cross-entropy with logits.
    """
  loss = optax.sigmoid_binary_cross_entropy(logits, labels)
  if weight is not None:
    loss = loss * weight

  if reduction == "none":
    return loss
  elif reduction == "mean":
    return jnp.mean(loss)
  elif reduction == "sum":
    return jnp.sum(loss)
  else:
    raise ValueError(f"Unsupported reduction: {reduction}")


def cross_entropy(logits, labels, weight=None):
  """
    Computes weighted cross-entropy loss.
    """
  one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
  loss = optax.softmax_cross_entropy(logits, one_hot_labels)
  if weight is not None:
    sample_weights = jnp.take(weight, labels)
    loss = loss * sample_weights
  return jnp.mean(loss)


class RTDetrLoss(nn.Module):
  """
    This class computes the losses for RTDetr. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).

    Attributes:
        config: Configuration object with model and loss parameters.
    """

  config: object

  def setup(self):
    self.matcher = RTDetrHungarianMatcher(self.config)
    self.num_classes = self.config.num_labels
    self.weight_dict = {
        "loss_vfl": self.config.weight_loss_vfl,
        "loss_bbox": self.config.weight_loss_bbox,
        "loss_giou": self.config.weight_loss_giou,
    }
    self.losses = ["vfl", "boxes"]
    self.eos_coef = self.config.eos_coefficient
    empty_weight = jnp.ones(self.config.num_labels + 1)
    self.empty_weight = empty_weight.at[-1].set(self.eos_coef)
    self.alpha = self.config.focal_loss_alpha
    self.gamma = self.config.focal_loss_gamma

  def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
    if "pred_boxes" not in outputs:
      raise KeyError("No predicted boxes found in outputs")
    if "logits" not in outputs:
      raise KeyError("No predicted logits found in outputs")
    idx = self._get_source_permutation_idx(indices)

    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = jnp.concatenate([_target["boxes"][i] for _target, (_, i) in zip(targets, indices)], axis=0)
    ious, _ = box_iou(
        center_to_corners_format(jax.lax.stop_gradient(src_boxes)), center_to_corners_format(target_boxes)
    )
    ious = jnp.diag(ious)

    src_logits = outputs["logits"]
    target_classes_original = jnp.concatenate([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
    target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int32)
    target_classes = target_classes.at[idx].set(target_classes_original)
    target = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

    target_score_original = jnp.zeros_like(target_classes, dtype=src_logits.dtype)
    target_score_original = target_score_original.at[idx].set(ious.astype(target_score_original.dtype))
    target_score = jnp.expand_dims(target_score_original, -1) * target

    pred_score = jax.nn.sigmoid(jax.lax.stop_gradient(src_logits))
    weight = self.alpha * (pred_score**self.gamma) * (1 - target) + target_score

    loss = binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
    loss = loss.mean(axis=1).sum() * src_logits.shape[1] / num_boxes
    return {"loss_vfl": loss}

  def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    """Classification loss (NLL)
        targets dicts must contain the key "class_labels" containing a tensor of dim [nb_target_boxes]
        """
    if "logits" not in outputs:
      raise KeyError("No logits were found in the outputs")

    src_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_original = jnp.concatenate([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
    target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
    target_classes = target_classes.at[idx].set(target_classes_original)

    # The original code uses `self.class_weight`, which is not defined in the original `__init__`.
    # Using `self.empty_weight` as it is the only weight defined.
    loss_ce = cross_entropy(src_logits, target_classes, self.empty_weight)
    losses = {"loss_ce": loss_ce}
    return losses

  def loss_cardinality(self, outputs, targets, indices, num_boxes):
    """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes. This is not
        really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
    logits = outputs["logits"]
    target_lengths = jnp.array([len(v["class_labels"]) for v in targets])
    # Count the number of predictions that are NOT "no-object" (which is the last class)
    card_pred = (jnp.argmax(logits, axis=-1) != logits.shape[-1] - 1).sum(1)
    card_err = jnp.mean(jnp.abs(card_pred.astype(jnp.float32) - target_lengths.astype(jnp.float32)))
    losses = {"cardinality_error": jax.lax.stop_gradient(card_err)}
    return losses

  def loss_boxes(self, outputs, targets, indices, num_boxes):
    """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss. Targets dicts must
        contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
        format (center_x, center_y, w, h), normalized by the image size.
        """
    if "pred_boxes" not in outputs:
      raise KeyError("No predicted boxes found in outputs")
    idx = self._get_source_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

    losses = {}

    loss_bbox = jnp.abs(src_boxes - target_boxes)
    losses["loss_bbox"] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - jnp.diag(
        generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
    )
    losses["loss_giou"] = loss_giou.sum() / num_boxes
    return losses

  def loss_masks(self, outputs, targets, indices, num_boxes):
    """
        Compute the losses related to the masks: the focal loss and the dice loss. Targets dicts must contain the key
        "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
    if "pred_masks" not in outputs:
      raise KeyError("No predicted masks found in outputs")

    source_idx = self._get_source_permutation_idx(indices)
    target_idx = self._get_target_permutation_idx(indices)
    source_masks = outputs["pred_masks"]
    source_masks = source_masks[source_idx]
    masks = [t["masks"] for t in targets]
    target_masks, _ = nested_tensor_from_tensor_list(masks)
    target_masks = target_masks[target_idx]

    # upsample predictions to the target size
    target_shape = target_masks.shape[-2:]
    # JAX resize expects NHWC, PyTorch interpolate expects NCHW
    source_masks_nhwc = jnp.expand_dims(source_masks, axis=-1)
    resized_masks = jax.image.resize(
        source_masks_nhwc, (source_masks.shape[0], *target_shape, 1), method="bilinear"
    )
    source_masks = jnp.squeeze(resized_masks, axis=-1).reshape(resized_masks.shape[0], -1)

    target_masks = target_masks.reshape(target_masks.shape[0], -1)
    losses = {
        "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
        "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
    }
    return losses

  def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
    src_logits = outputs["logits"]
    idx = self._get_source_permutation_idx(indices)
    target_classes_original = jnp.concatenate([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
    target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
    target_classes = target_classes.at[idx].set(target_classes_original)

    target = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
    loss = binary_cross_entropy_with_logits(src_logits, target * 1.0, reduction="none")
    loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
    return {"loss_bce": loss}

  def _get_source_permutation_idx(self, indices: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # permute predictions following indices
    batch_idx = jnp.concatenate([jnp.full_like(source, i, dtype=jnp.int32) for i, (source, _) in enumerate(indices)])
    source_idx = jnp.concatenate([source for (source, _) in indices])
    return batch_idx, source_idx

  def _get_target_permutation_idx(self, indices: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # permute targets following indices
    batch_idx = jnp.concatenate([jnp.full_like(target, i, dtype=jnp.int32) for i, (_, target) in enumerate(indices)])
    target_idx = jnp.concatenate([target for (_, target) in indices])
    return batch_idx, target_idx

  def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
    if "logits" not in outputs:
      raise KeyError("No logits found in outputs")

    src_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_original = jnp.concatenate([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
    target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
    target_classes = target_classes.at[idx].set(target_classes_original)

    target = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
    loss = sigmoid_focal_loss(src_logits, target, alpha=self.alpha, gamma=self.gamma, reduction="none")
    loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
    return {"loss_focal": loss}

  def get_loss(self, loss: str, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: jnp.ndarray) -> Dict:
    loss_map = {
        "labels": self.loss_labels,
        "cardinality": self.loss_cardinality,
        "boxes": self.loss_boxes,
        "masks": self.loss_masks,
        "bce": self.loss_labels_bce,
        "focal": self.loss_labels_focal,
        "vfl": self.loss_labels_vfl,
    }
    if loss not in loss_map:
      raise ValueError(f"Loss {loss} not supported")
    return loss_map[loss](outputs, targets, indices, num_boxes)

  @staticmethod
  def get_cdn_matched_indices(dn_meta: Dict, targets: List[Dict]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
    num_gts = [len(t["class_labels"]) for t in targets]

    dn_match_indices = []
    for i, num_gt in enumerate(num_gts):
      if num_gt > 0:
        gt_idx = jnp.arange(num_gt, dtype=jnp.int64)
        gt_idx = jnp.tile(gt_idx, dn_num_group)
        assert len(dn_positive_idx[i]) == len(gt_idx)
        dn_match_indices.append((dn_positive_idx[i], gt_idx))
      else:
        dn_match_indices.append(
            (
                jnp.zeros(0, dtype=jnp.int64),
                jnp.zeros(0, dtype=jnp.int64),
            )
        )

    return dn_match_indices

  def __call__(self, outputs: Dict, targets: List[Dict]) -> Dict:
    """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`list[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
    outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}

    # Retrieve the matching between the outputs of the last layer and the targets
    indices = self.matcher(outputs_without_aux, targets)

    # Compute the average number of target boxes across all nodes, for normalization purposes
    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = jnp.array([num_boxes], dtype=jnp.float32)
    num_boxes = jnp.clip(num_boxes, a_min=1)

    # Compute all the requested losses
    losses = {}
    for loss in self.losses:
      l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
      l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
      losses.update(l_dict)

    # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    if "auxiliary_outputs" in outputs:
      for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
        indices = self.matcher(auxiliary_outputs, targets)
        for loss in self.losses:
          if loss == "masks":
            # Intermediate masks losses are too costly to compute, we ignore them.
            continue
          l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
          l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
          l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
          losses.update(l_dict)

    # In case of cdn auxiliary losses. For rtdetr
    if "dn_auxiliary_outputs" in outputs:
      if "denoising_meta_values" not in outputs:
        raise ValueError(
            "The output must have the 'denoising_meta_values` key. Please, ensure that 'outputs' includes a 'denoising_meta_values' entry."
        )
      indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
      num_boxes_dn = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]

      for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
        for loss in self.losses:
          if loss == "masks":
            # Intermediate masks losses are too costly to compute, we ignore them.
            continue
          l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes_dn)
          l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
          l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
          losses.update(l_dict)

    return losses

from functools import partial
from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import Array

from ..modeling_utils import _get_source_permutation_idx
from .loss_for_object_detection import box_iou
from .loss_rt_detr import RTDetrHungarianMatcher, RTDetrLoss
from .bbox_utils import bbox2distance, center_to_corners_format

# Assuming RTDetrLoss is a Flax Module. If it's not, the inheritance needs to be adjusted.
# Reused: Qwen3ForCausalLM.losses.RTDetrLoss
# Reused: Qwen3ForCausalLM.matchers.RTDetrHungarianMatcher
# Reused: Qwen3ForCausalLM.box_utils.box_iou
# Reused: Qwen3ForCausalLM.bbox_utils.center_to_corners_format
# Reused: Qwen3ForCausalLM.bbox_utils.bbox2distance


class DFineLoss(RTDetrLoss):
    """
    This class computes the losses for D-FINE. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).

    Attributes:
        config: Configuration object containing model and loss parameters.
    """

    config: Any

    def setup(self):
        """Initializes the DFineLoss module."""
        super().setup()

        self.matcher = RTDetrHungarianMatcher(config=self.config)
        self.max_num_bins = self.config.max_num_bins
        self.weight_dict = {
            "loss_vfl": self.config.weight_loss_vfl,
            "loss_bbox": self.config.weight_loss_bbox,
            "loss_giou": self.config.weight_loss_giou,
            "loss_fgl": self.config.weight_loss_fgl,
            "loss_ddf": self.config.weight_loss_ddf,
        }
        self.losses = ["vfl", "boxes", "local"]
        self.reg_scale = self.config.reg_scale
        self.up = jnp.array([self.config.up])

    def unimodal_distribution_focal_loss(
        self,
        pred: Array,
        label: Array,
        weight_right: Array,
        weight_left: Array,
        weight: Array = None,
        reduction: str = "sum",
        avg_factor: float = None,
    ) -> Array:
        """Computes the unimodal distribution focal loss."""
        dis_left = label.astype(jnp.int32)
        dis_right = dis_left + 1

        loss = optax.softmax_cross_entropy_with_integer_labels(
            pred, dis_left
        ) * weight_left.reshape(-1) + optax.softmax_cross_entropy_with_integer_labels(pred, dis_right) * weight_right.reshape(
            -1
        )

        if weight is not None:
            weight = weight.astype(jnp.float32)
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def loss_local(
        self, outputs: Dict[str, Array], targets: List[Dict[str, Array]], indices: List[Tuple], num_boxes: int, T: int = 5
    ) -> Dict[str, Array]:
        """Compute Fine-Grained Localization (FGL) Loss and Decoupled Distillation Focal (DDF) Loss."""
        losses = {}
        if "pred_corners" in outputs:
            idx = _get_source_permutation_idx(indices)
            target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.max_num_bins + 1))
            ref_points = jax.lax.stop_gradient(outputs["ref_points"][idx])

            target_corners, weight_right, weight_left = bbox2distance(
                ref_points,
                center_to_corners_format(target_boxes),
                self.max_num_bins,
                self.reg_scale,
                self.up,
            )

            ious, _ = box_iou(
                center_to_corners_format(outputs["pred_boxes"][idx]), center_to_corners_format(target_boxes)
            )
            ious = jnp.diag(ious)
            weight_targets = jax.lax.stop_gradient(
                jnp.expand_dims(ious, axis=-1).repeat(4, axis=-1).reshape(-1)
            )

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            pred_corners = outputs["pred_corners"].reshape(-1, (self.max_num_bins + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (self.max_num_bins + 1))
            if jnp.array_equal(pred_corners, target_corners):
                losses["loss_ddf"] = pred_corners.sum() * 0.0
            else:
                weight_targets_local = jax.nn.sigmoid(outputs["teacher_logits"]).max(axis=-1)
                mask = jnp.zeros_like(weight_targets_local, dtype=jnp.bool_)
                mask = mask.at[idx].set(True)
                mask = mask.reshape(mask.shape[0], -1)
                mask = jnp.expand_dims(mask, axis=-1).repeat(4, axis=-1).reshape(-1)

                weight_targets_local = weight_targets_local.at[idx].set(
                    ious.reshape(weight_targets_local[idx].shape).astype(weight_targets_local.dtype)
                )
                weight_targets_local = jax.lax.stop_gradient(
                    jnp.expand_dims(weight_targets_local.reshape(weight_targets_local.shape[0], -1), axis=-1)
                    .repeat(4, axis=-1)
                    .reshape(-1)
                )

                log_q = jax.nn.log_softmax(pred_corners / T, axis=1)
                log_p = jax.nn.log_softmax(jax.lax.stop_gradient(target_corners) / T, axis=1)
                kl_div = optax.kl_divergence(log_p, log_q).sum(axis=-1)
                loss_match_local = weight_targets_local * (T**2) * kl_div

                batch_scale = 1 / outputs["pred_boxes"].shape[0]
                num_pos = jnp.sqrt(mask.sum() * batch_scale)
                num_neg = jnp.sqrt((~mask).sum() * batch_scale)

                masked_sum1 = jnp.sum(loss_match_local * mask)
                num_masked1 = jnp.sum(mask)
                loss_match_local1 = jnp.divide(masked_sum1, num_masked1, where=(num_masked1 > 0))

                masked_sum2 = jnp.sum(loss_match_local * (~mask))
                num_masked2 = jnp.sum(~mask)
                loss_match_local2 = jnp.divide(masked_sum2, num_masked2, where=(num_masked2 > 0))

                losses["loss_ddf"] = (loss_match_local1 * num_pos + loss_match_local2 * num_neg) / (num_pos + num_neg)

        return losses

    def get_loss(
        self, loss: str, outputs: Dict, targets: List, indices: List, num_boxes: int
    ) -> Dict[str, Array]:
        """Dispatcher for loss functions."""
        loss_map = {
            "cardinality": self.loss_cardinality,
            "local": self.loss_local,
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp

# Assuming DFineLoss is a Flax Module that has been converted from the PyTorch original.
# from path.to.d_fine_loss import DFineLoss
# Using a placeholder for demonstration.
from .loss_rt_detr import RTDetrLoss as DFineLoss


Array = jnp.ndarray


def _set_aux_loss(outputs_class: Sequence[Array], outputs_coord: Sequence[Array]) -> List[Dict[str, Array]]:
  """
  This is a helper function that combines sequences of class predictions and coordinate predictions
  into a list of dictionaries, formatted for auxiliary loss calculation.
  """
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


def _set_aux_loss2(
    outputs_class: Sequence[Array],
    outputs_coord: Sequence[Array],
    outputs_corners: Sequence[Array],
    outputs_ref: Sequence[Array],
    teacher_corners: Optional[Array] = None,
    teacher_logits: Optional[Array] = None,
) -> List[Dict[str, Array]]:
  """
  A helper function to structure auxiliary loss outputs by combining sequences of predictions
  into a list of dictionaries.
  """
  return [
      {
          "logits": a,
          "pred_boxes": b,
          "pred_corners": c,
          "ref_points": d,
          "teacher_corners": teacher_corners,
          "teacher_logits": teacher_logits,
      }
      for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
  ]


def DFineForObjectDetectionLoss(
    logits: Array,
    labels: Any,
    pred_boxes: Array,
    config: Any,
    outputs_class: Optional[Array] = None,
    outputs_coord: Optional[Array] = None,
    enc_topk_logits: Optional[Array] = None,
    enc_topk_bboxes: Optional[Array] = None,
    denoising_meta_values: Optional[Dict[str, Any]] = None,
    predicted_corners: Optional[Array] = None,
    initial_reference_points: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Optional[List[Dict[str, Array]]]]:
  """
  Computes the total loss for the D-FINE object detection model.

  Args:
      logits: A float array of shape (batch_size, num_queries, num_classes + 1) containing the classification logits.
      labels: A list of dictionaries, where each dictionary contains the ground truth information for a single image.
      pred_boxes: A float array of shape (batch_size, num_queries, 4) containing the predicted bounding boxes.
      config: A configuration object with model and loss hyperparameters.
      outputs_class: Optional float array from intermediate decoder layers for auxiliary loss.
      outputs_coord: Optional float array from intermediate decoder layers for auxiliary loss.
      enc_topk_logits: Optional float array from the encoder for auxiliary loss.
      enc_topk_bboxes: Optional float array from the encoder for auxiliary loss.
      denoising_meta_values: Optional dictionary with metadata for the denoising task.
      predicted_corners: Optional float array of predicted corner distributions.
      initial_reference_points: Optional float array of initial reference points.
      **kwargs: Additional keyword arguments.

  Returns:
      A tuple containing:
      - The total loss as a scalar float array.
      - A dictionary of individual loss components.
      - A list of formatted auxiliary outputs, or None.
  """
  criterion = DFineLoss(config)
  # Second: compute the losses, based on outputs and labels
  outputs_loss = {}
  outputs_loss["logits"] = logits
  outputs_loss["pred_boxes"] = jnp.clip(pred_boxes, a_min=0, a_max=1)
  auxiliary_outputs = None
  if config.auxiliary_loss:
    if denoising_meta_values is not None:
      dn_num_split = denoising_meta_values["dn_num_split"]

      clamped_outputs_coord = jnp.clip(outputs_coord, a_min=0, a_max=1)
      dn_out_coord, outputs_coord = jnp.split(clamped_outputs_coord, [dn_num_split], axis=2)

      dn_out_class, outputs_class = jnp.split(outputs_class, [dn_num_split], axis=2)
      dn_out_corners, out_corners = jnp.split(predicted_corners, [dn_num_split], axis=2)
      dn_out_refs, out_refs = jnp.split(initial_reference_points, [dn_num_split], axis=2)

      auxiliary_outputs = _set_aux_loss2(
          jnp.transpose(outputs_class[:, :-1], (1, 0, 2, 3)),
          jnp.transpose(outputs_coord[:, :-1], (1, 0, 2, 3)),
          jnp.transpose(out_corners[:, :-1], (1, 0, 2, 3)),
          jnp.transpose(out_refs[:, :-1], (1, 0, 2, 3)),
          out_corners[:, -1],
          outputs_class[:, -1],
      )

      outputs_loss["auxiliary_outputs"] = auxiliary_outputs
      outputs_loss["auxiliary_outputs"].extend(
          _set_aux_loss([enc_topk_logits], [jnp.clip(enc_topk_bboxes, a_min=0, a_max=1)])
      )

      dn_auxiliary_outputs = _set_aux_loss2(
          jnp.transpose(dn_out_class, (1, 0, 2, 3)),
          jnp.transpose(dn_out_coord, (1, 0, 2, 3)),
          jnp.transpose(dn_out_corners, (1, 0, 2, 3)),
          jnp.transpose(dn_out_refs, (1, 0, 2, 3)),
          dn_out_corners[:, -1],
          dn_out_class[:, -1],
      )
      outputs_loss["dn_auxiliary_outputs"] = dn_auxiliary_outputs
      outputs_loss["denoising_meta_values"] = denoising_meta_values

  loss_dict = criterion(outputs_loss, labels)

  loss = sum(loss_dict.values())
  return loss, loss_dict, auxiliary_outputs

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

# Reused modules from JAX_MODULES_DICT
# Qwen3ForCausalLM.modeling_utils.DeformableDetrHungarianMatcher was used for DeformableDetrHungarianMatcher
from ..matchers.deformable_detr_hungarian_matcher import DeformableDetrHungarianMatcher
# Qwen3ForCausalLM.losses.DeformableDetrImageLoss was used for DeformableDetrImageLoss
from .deformable_detr_image_loss import DeformableDetrImageLoss
# Qwen3ForCausalLM.modeling_utils._set_aux_loss was used for _set_aux_loss
from ..modeling_utils import _set_aux_loss


def DeformableDetrForObjectDetectionLoss(
    logits: jnp.ndarray,
    labels: List[Dict[str, jnp.ndarray]],
    pred_boxes: jnp.ndarray,
    config: Any,
    outputs_class: Optional[jnp.ndarray] = None,
    outputs_coord: Optional[jnp.ndarray] = None,
    **kwargs,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Optional[List[Dict[str, jnp.ndarray]]]]:
    """
    Computes the total loss for Deformable DETR object detection.

    This function orchestrates the loss computation by:
    1. Creating a Hungarian matcher to assign predictions to ground truth.
    2. Creating a criterion (loss function) that uses the matcher.
    3. Computing a dictionary of individual losses (classification, bbox, etc.).
    4. Calculating a final weighted total loss.

    Args:
        logits: A tensor of shape (batch_size, num_queries, num_classes) containing the classification logits.
        labels: A list of dictionaries, one per image, containing the ground truth labels.
        pred_boxes: A tensor of shape (batch_size, num_queries, 4) containing the predicted bounding boxes.
        config: A configuration object with loss coefficients and model parameters.
        outputs_class: Optional intermediate decoder layer logits for auxiliary loss.
        outputs_coord: Optional intermediate decoder layer boxes for auxiliary loss.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing:
        - The total loss.
        - A dictionary of individual loss components.
        - The formatted auxiliary outputs, if any.
    """
    # First: create the matcher
    matcher = DeformableDetrHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = DeformableDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=losses,
    )
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    if config.auxiliary_loss:
        # Reused Qwen3ForCausalLM.modeling_utils._set_aux_loss
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = config.giou_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from scipy.optimize import linear_sum_assignment
import optax
from typing import List, Dict, Tuple, Any, Optional

from MaxText.common_types import Array, DType, PyTree, Config
# New generated code.
# Based on huggingface/transformers/src/transformers/models/deformable_detr/modeling_deformable_detr.py
# and src/transformers/models/detr/modeling_detr.py

def _set_aux_loss(
    outputs_class: Array, outputs_coord: Array
) -> List[Dict[str, Array]]:
    """
    A helper function that formats the auxiliary branch outputs to the format expected by the criterion.
    """
    return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
def center_to_corners_format(x: Array) -> Array:
    """
    Converts bounding box coordinates from (center_x, center_y, width, height) to (x1, y1, x2, y2) format.
    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x: jnp.ndarray of shape (..., 4)
    Returns:
        jnp.ndarray of shape (..., 4)
    """
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [
        (x_c - 0.5 * w),
        (y_c - 0.5 * h),
        (x_c + 0.5 * w),
        (y_c + 0.5 * h),
    ]
    return jnp.stack(b, axis=-1)


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
def box_area(boxes: Array) -> Array:
    """
    Computes the area of a set of bounding boxes, which are specified by their (x1, y1, x2, y2) coordinates.

    Args:
        boxes: boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Returns:
        A 1-D tensor holding the area of each box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
def box_iou(boxes1: Array, boxes2: Array) -> Tuple[Array, Array]:
    """
    Computes the intersection over union (IoU) of two sets of bounding boxes.

    Args:
        boxes1: first set of boxes of shape (N, 4)
        boxes2: second set of boxes of shape (M, 4)
    Returns:
        A tuple containing the IoU and the union of the two sets of boxes
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
    """
    Computes the generalized box IoU (GIoU) of two sets of bounding boxes.

    Args:
        boxes1: first set of boxes of shape (N, 4)
        boxes2: second set of boxes of shape (M, 4)
    Returns:
        A [N, M] matrix containing the GIoU values.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
def sigmoid_focal_loss(
    inputs: Array,
    targets: Array,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2,
) -> Array:
    """
    Computes the sigmoid focal loss.
    """
    prob = jax.nn.sigmoid(inputs)
    ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
def dice_loss(inputs: Array, targets: Array, num_boxes: int) -> Array:
    """
    Computes the DICE loss, similar to generalized IOU for masks.
    """
    inputs = jax.nn.sigmoid(inputs)
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    """
    class_cost: float = 1.0
    bbox_cost: float = 1.0
    giou_cost: float = 1.0

    @nn.compact
    def __call__(self, outputs: Dict[str, Array], targets: List[Dict[str, Array]]) -> List[Tuple[Array, Array]]:
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = jax.nn.softmax(outputs["logits"], axis=-1).reshape(
            -1, outputs["logits"].shape[-1]
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = jnp.concatenate([jnp.asarray(v["class_labels"]) for v in targets])
        target_bbox = jnp.concatenate([jnp.asarray(v["boxes"]) for v in targets])

        # Compute the classification cost.
        cost_class = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        cost_bbox = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(
            center_to_corners_format(out_bbox), center_to_corners_format(target_bbox)
        )

        # Final cost matrix
        cost_matrix = self.bbox_cost * cost_bbox + self.class_cost * cost_class + self.giou_cost * cost_giou
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

        cost_matrix_np = jax.device_get(cost_matrix)
        sizes = [len(v["boxes"]) for v in targets]
        
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(np.split(cost_matrix_np, np.cumsum(sizes)[:-1], axis=-1))
        ]

        return [
            (jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64))
            for i, j in indices
        ]


# New generated code.
# Based on src/transformers/models/detr/modeling_detr.py
class ImageLoss(nn.Module):
    """
    This class computes the losses for DETR.
    """
    matcher: nn.Module
    num_classes: int
    eos_coef: float
    losses: List[str]

    def setup(self):
        empty_weight = jnp.ones(self.num_classes + 1)
        self.empty_weight = empty_weight.at[-1].set(self.eos_coef)

    def loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        raise NotImplementedError("Base ImageLoss.loss_labels is not implemented as it's overridden.")

    def loss_cardinality(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        logits = outputs["logits"]
        target_lengths = jnp.array([len(v["class_labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = jnp.abs(card_pred.astype(jnp.float32) - target_lengths.astype(jnp.float32)).mean()
        losses = {"cardinality_error": jax.lax.stop_gradient(card_err)}
        return losses

    def loss_boxes(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        idx = self._get_source_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = jnp.abs(src_boxes - target_boxes).sum() / num_boxes

        loss_giou = 1 - jnp.diag(
            generalized_box_iou(
                center_to_corners_format(src_boxes), center_to_corners_format(target_boxes)
            )
        )
        loss_giou = loss_giou.sum() / num_boxes

        losses = {}
        losses["loss_bbox"] = loss_bbox
        losses["loss_giou"] = loss_giou
        return losses

    def loss_masks(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        src_idx = self._get_source_permutation_idx(indices)
        tgt_idx = self._get_target_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"][src_idx]
        
        masks = [t["masks"] for t in targets]
        target_masks = jnp.concatenate([t["masks"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        # upsample predictions to the target size
        src_masks = jax.image.resize(src_masks[:, None], 
                                     (src_masks.shape[0], 1) + target_masks.shape[1:], 
                                     method='bilinear')
        src_masks = src_masks[:, 0]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices: List[Tuple]) -> Tuple[Array, Array]:
        batch_idx = jnp.concatenate([jnp.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jnp.concatenate([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _get_target_permutation_idx(self, indices: List[Tuple]) -> Tuple[Array, Array]:
        batch_idx = jnp.concatenate([jnp.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jnp.concatenate([tgt for (_, tgt) in indices])
        return (batch_idx, tgt_idx)

    def get_loss(self, loss: str, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def __call__(self, outputs: Dict, targets: List[Dict]) -> Dict:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = jnp.asarray([num_boxes], dtype=jnp.float32)
        num_boxes = jnp.clip(num_boxes[0], a_min=1)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "auxiliary_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# New generated code.
# Based on huggingface/transformers/src/transformers/models/deformable_detr/modeling_deformable_detr.py
class DeformableDetrImageLoss(ImageLoss):
    """
    This class computes the losses for Deformable DETR.
    """
    focal_alpha: float

    def loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        """
        Classification loss (Binary focal loss)
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        src_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
        target_classes = target_classes.at[idx].set(target_classes_o)

        target_classes_onehot = jax.nn.one_hot(
            target_classes, num_classes=src_logits.shape[2] + 1, dtype=src_logits.dtype
        )
        target_classes_onehot = target_classes_onehot[..., :-1]
        
        loss_ce = (
            sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses


def DeformableDetrForSegmentationLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    pred_masks: Array,
    config: Any,
    outputs_class: Optional[Array] = None,
    outputs_coord: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Optional[List[Dict[str, Array]]]]:
    """
    Computes the loss for DeformableDetrForSegmentation.

    Args:
        logits: Classification logits for all queries.
        labels: A list of dictionaries, one per image, containing the ground truth annotations.
        pred_boxes: Normalized bounding box predictions for all queries.
        pred_masks: Predicted segmentation masks for all queries.
        config: A config object with loss coefficients and model parameters.
        outputs_class: A stack of intermediate decoder layer logits for auxiliary loss computation.
        outputs_coord: A stack of intermediate decoder layer box predictions for auxiliary loss computation.

    Returns:
        A tuple of (loss, loss_dict, auxiliary_outputs).
    """
    # First: create the matcher
    matcher = HungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality", "masks"]
    criterion = DeformableDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=losses,
        eos_coef=getattr(config, "eos_coefficient", 0.1),
    )
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["pred_masks"] = pred_masks

    auxiliary_outputs = None
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = config.giou_loss_coefficient
    weight_dict["loss_mask"] = config.mask_loss_coefficient
    weight_dict["loss_dice"] = config.dice_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
This file contains the JAX implementation of the loss function for DETR-like object detection models.
It includes the HungarianMatcher for bipartite matching and the ImageLoss criterion.
"""

from functools import partial
from typing import List, Dict, Tuple, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import linear_sum_assignment

from maxtext.common_types import Array, DType


# Utility functions for bounding box operations
# Equivalent to functions from transformers.image_transforms and detr.util.box_ops
def center_to_corners_format(x: Array) -> Array:
  """Converts bounding box coordinates from [center, size] to [x1, y1, x2, y2] format."""
  x_center, y_center, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
  b = [
      (x_center - 0.5 * w),
      (y_center - 0.5 * h),
      (x_center + 0.5 * w),
      (y_center + 0.5 * h),
  ]
  return jnp.stack(b, axis=-1)


def _upcast(t: Array) -> Array:
  # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t.astype(jnp.float32) if t.dtype not in (jnp.float32, jnp.float64) else t
  else:
    return t.astype(jnp.int32) if t.dtype not in (jnp.int32, jnp.int64) else t


def box_area(boxes: Array) -> Array:
  """
  Computes the area of a set of bounding boxes, which are specified by their (x1, y1, x2, y2) coordinates.
  """
  boxes = _upcast(boxes)
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Array, boxes2: Array) -> Tuple[Array, Array]:
  """
  Computes the pairwise intersection over union (IoU) and union area between two sets of bounding boxes.
  """
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  left_top = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])
  right_bottom = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

  width_height = jnp.maximum(0, right_bottom - left_top)
  inter = width_height[:, :, 0] * width_height[:, :, 1]

  union = area1[:, None] + area2 - inter

  iou = inter / (union + 1e-6)
  return iou, union


def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
  """
  Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.
  """
  if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
    raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
  if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
    raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
  iou, union = box_iou(boxes1, boxes2)

  top_left = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
  bottom_right = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

  width_height = jnp.maximum(0, bottom_right - top_left)
  area = width_height[:, :, 0] * width_height[:, :, 1]

  return iou - (area - union) / (area + 1e-6)


# Helper for auxiliary losses, equivalent to _set_aux_loss from DETR
def _set_aux_loss(outputs_class: Sequence[Array], outputs_coord: Sequence[Array]) -> List[Dict[str, Array]]:
  """
  Formats auxiliary outputs from intermediate decoder layers into a list of dictionaries.
  """
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# Converted from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
class HungarianMatcher(nn.Module):
  """
  This class computes an assignment between the targets and the predictions of the network.
  """

  class_cost: float = 1.0
  bbox_cost: float = 1.0
  giou_cost: float = 1.0

  def setup(self):
    if self.class_cost == 0 and self.bbox_cost == 0 and self.giou_cost == 0:
      raise ValueError("All costs of the Matcher can't be 0")

  def __call__(self, outputs: Dict[str, Array], targets: List[Dict[str, Array]]) -> List[Tuple[Array, Array]]:
    """
    Performs the matching.
    """
    batch_size, num_queries = outputs["logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_prob = jax.nn.softmax(outputs["logits"].reshape(batch_size * num_queries, -1))
    out_bbox = outputs["pred_boxes"].reshape(batch_size * num_queries, -1)

    # Also concat the target labels and boxes
    target_ids = jnp.concatenate([v["class_labels"] for v in targets])
    target_bbox = jnp.concatenate([v["boxes"] for v in targets])

    # Compute the classification cost.
    class_cost = -out_prob[:, target_ids]

    # Compute the L1 cost between boxes
    bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

    # Compute the giou cost between boxes
    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    # Final cost matrix
    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

    # Move to CPU for scipy which runs on CPU
    cost_matrix_cpu = np.array(jax.device_get(cost_matrix))

    sizes = [len(v["boxes"]) for v in targets]
    indices = [
        linear_sum_assignment(c[i]) for i, c in enumerate(np.split(cost_matrix_cpu, np.cumsum(sizes)[:-1], axis=-1))
    ]
    return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]


# Converted from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class ImageLoss(nn.Module):
  """
  This class computes the losses for DetrForObjectDetection.
  """

  matcher: nn.Module
  num_classes: int
  eos_coef: float
  losses: List[str]

  def setup(self):
    empty_weight = jnp.ones(self.num_classes + 1)
    self.empty_weight = empty_weight.at[-1].set(self.eos_coef)

  def _get_source_permutation_idx(self, indices: List[Tuple[Array, Array]]) -> Tuple[Array, Array]:
    batch_idx = jnp.concatenate([jnp.full_like(src, i) for i, (src, _) in enumerate(indices)])
    source_idx = jnp.concatenate([src for (src, _) in indices])
    return batch_idx, source_idx

  def _get_target_permutation_idx(self, indices: List[Tuple[Array, Array]]) -> Tuple[Array, Array]:
    batch_idx = jnp.concatenate([jnp.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    target_idx = jnp.concatenate([tgt for (_, tgt) in indices])
    return batch_idx, target_idx

  def loss_labels(
      self, outputs: Dict[str, Array], targets: List[Dict[str, Array]], indices: List[Tuple[Array, Array]], num_boxes: int
  ) -> Dict[str, Array]:
    if "logits" not in outputs:
      raise KeyError("No logits were found in the outputs")
    source_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = jnp.full(source_logits.shape[:2], self.num_classes, dtype=jnp.int64)
    target_classes = target_classes.at[idx].set(target_classes_o)

    # Transpose to [batch, num_classes, num_queries] for cross_entropy
    loss_ce = optax.softmax_cross_entropy(
        source_logits.transpose((0, 2, 1)), jax.nn.one_hot(target_classes, self.num_classes + 1)
    )
    # Apply class weights
    weights = self.empty_weight[target_classes]
    loss_ce = (loss_ce * weights).sum() / weights.sum()

    losses = {"loss_ce": loss_ce}
    return losses

  def loss_cardinality(
      self, outputs: Dict[str, Array], targets: List[Dict[str, Array]], indices: List[Tuple[Array, Array]], num_boxes: int
  ) -> Dict[str, Array]:
    logits = outputs["logits"]
    target_lengths = jnp.asarray([len(v["class_labels"]) for v in targets])
    # Count the number of predictions that are NOT "no-object"
    card_pred = (jnp.argmax(logits, axis=-1) != logits.shape[-1] - 1).sum(axis=1)
    card_err = jnp.mean(jnp.abs(card_pred.astype(jnp.float32) - target_lengths.astype(jnp.float32)))
    losses = {"cardinality_error": jax.lax.stop_gradient(card_err)}
    return losses

  def loss_boxes(
      self, outputs: Dict[str, Array], targets: List[Dict[str, Array]], indices: List[Tuple[Array, Array]], num_boxes: int
  ) -> Dict[str, Array]:
    if "pred_boxes" not in outputs:
      raise KeyError("No predicted boxes found in outputs")
    idx = self._get_source_permutation_idx(indices)
    source_boxes = outputs["pred_boxes"][idx]
    target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

    loss_bbox = jnp.sum(jnp.abs(source_boxes - target_boxes)) / num_boxes

    losses = {"loss_bbox": loss_bbox}

    loss_giou = 1 - jnp.diag(
        generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
    )
    losses["loss_giou"] = loss_giou.sum() / num_boxes
    return losses

  def get_loss(
      self,
      loss: str,
      outputs: Dict[str, Array],
      targets: List[Dict[str, Array]],
      indices: List[Tuple[Array, Array]],
      num_boxes: int,
  ) -> Dict[str, Array]:
    loss_map = {
        "labels": self.loss_labels,
        "cardinality": self.loss_cardinality,
        "boxes": self.loss_boxes,
    }
    if loss not in loss_map:
      raise ValueError(f"Loss {loss} not supported")
    return loss_map[loss](outputs, targets, indices, num_boxes)

  def __call__(self, outputs: Dict[str, Array], targets: List[Dict[str, Array]]) -> Dict[str, Array]:
    outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

    indices = self.matcher(outputs_without_aux, targets)

    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = jnp.asarray(num_boxes, dtype=jnp.float32)
    # This assumes being run in a pjit context with a 'batch' axis
    # It computes the average number of boxes across all devices
    num_boxes = jax.lax.pmean(num_boxes, axis_name="batch")
    num_boxes = jnp.maximum(num_boxes, 1.0)

    losses = {}
    for loss in self.losses:
      losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

    if "auxiliary_outputs" in outputs:
      for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
        indices = self.matcher(auxiliary_outputs, targets)
        for loss in self.losses:
          l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
          l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
          losses.update(l_dict)

    return losses


def ForObjectDetectionLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    config: object,
    outputs_class: Optional[Sequence[Array]] = None,
    outputs_coord: Optional[Sequence[Array]] = None,
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Optional[List[Dict[str, Array]]]]:
  """
  Computes the DETR loss for object detection.

  Args:
      logits: A tensor of shape [batch_size, num_queries, num_classes + 1] with the classification logits.
      labels: A list of dictionaries, one per image in the batch. Each dictionary contains 'class_labels' and 'boxes'.
      pred_boxes: A tensor of shape [batch_size, num_queries, 4] with the predicted box coordinates.
      config: A configuration object with loss coefficients and model parameters.
      outputs_class: Optional auxiliary class logits from intermediate decoder layers.
      outputs_coord: Optional auxiliary box coordinates from intermediate decoder layers.
      **kwargs: Additional keyword arguments.

  Returns:
      A tuple containing:
      - The total weighted loss.
      - A dictionary of individual loss components.
      - The formatted auxiliary outputs, if any.
  """
  # First: create the matcher
  matcher = HungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
  # Second: create the criterion
  losses = ["labels", "boxes", "cardinality"]
  criterion = ImageLoss(
      matcher=matcher,
      num_classes=config.num_labels,
      eos_coef=config.eos_coefficient,
      losses=losses,
  )
  # Third: compute the losses, based on outputs and labels
  outputs_loss = {}
  auxiliary_outputs = None
  outputs_loss["logits"] = logits
  outputs_loss["pred_boxes"] = pred_boxes
  if config.auxiliary_loss:
    auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
    outputs_loss["auxiliary_outputs"] = auxiliary_outputs

  loss_dict = criterion(outputs_loss, labels)
  # Fourth: compute total loss, as a weighted sum of the various losses
  weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
  weight_dict["loss_giou"] = config.giou_loss_coefficient
  if config.auxiliary_loss:
    aux_weight_dict = {}
    for i in range(config.decoder_layers - 1):
      aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
  loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
  return loss, loss_dict, auxiliary_outputs

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

# Reused from Qwen3ForCausalLM.losses.ImageLoss
from Qwen3ForCausalLM.losses import ImageLoss
# Reused from Qwen3ForCausalLM.matchers.HungarianMatcher
from Qwen3ForCausalLM.matchers import HungarianMatcher
# Reused from Qwen3ForCausalLM.modeling_utils._set_aux_loss
from Qwen3ForCausalLM.modeling_utils import _set_aux_loss

Array = jnp.ndarray


def ForSegmentationLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    pred_masks: Array,
    config: Any,
    outputs_class: Optional[Array] = None,
    outputs_coord: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Optional[List[Dict[str, Array]]]]:
  """
  Computes the segmentation loss for a DETR-like model.

  This function orchestrates the loss computation by:
  1. Creating a HungarianMatcher to find the optimal assignment between predictions and targets.
  2. Creating an ImageLoss criterion to compute various loss components (classification, bbox, masks).
  3. Preparing model outputs, including auxiliary outputs from intermediate layers if enabled.
  4. Calling the criterion to get a dictionary of individual losses.
  5. Calculating a final weighted sum of all loss components.

  Args:
    logits: The final classification logits from the model.
      Shape: (batch_size, num_queries, num_classes + 1).
    labels: A list of dictionaries, one per image, containing the ground truth labels.
      Each dictionary has keys like 'class_labels' and 'boxes'.
    pred_boxes: The final predicted bounding boxes from the model.
      Shape: (batch_size, num_queries, 4).
    pred_masks: The final predicted segmentation masks from the model.
      Shape: (batch_size, num_queries, height, width).
    config: A configuration object with hyperparameters for the loss calculation.
    outputs_class: Logits from intermediate decoder layers for auxiliary loss.
    outputs_coord: Bounding box predictions from intermediate decoder layers for auxiliary loss.
    **kwargs: Additional keyword arguments.

  Returns:
    A tuple containing:
      - The total scalar loss.
      - A dictionary of individual loss components.
      - The formatted auxiliary outputs, if any.
  """
  # First: create the matcher
  matcher = HungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
  # Second: create the criterion
  losses = ["labels", "boxes", "cardinality", "masks"]
  criterion = ImageLoss(
      matcher=matcher,
      num_classes=config.num_labels,
      eos_coef=config.eos_coefficient,
      losses=losses,
  )
  # Third: compute the losses, based on outputs and labels
  outputs_loss = {}
  outputs_loss["logits"] = logits
  outputs_loss["pred_boxes"] = pred_boxes
  outputs_loss["pred_masks"] = pred_masks

  auxiliary_outputs = None
  if config.auxiliary_loss:
    auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
    outputs_loss["auxiliary_outputs"] = auxiliary_outputs

  loss_dict = criterion(outputs_loss, labels)
  # Fourth: compute total loss, as a weighted sum of the various losses
  weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
  weight_dict["loss_giou"] = config.giou_loss_coefficient
  weight_dict["loss_mask"] = config.mask_loss_coefficient
  weight_dict["loss_dice"] = config.dice_loss_coefficient
  if config.auxiliary_loss:
    aux_weight_dict = {}
    for i in range(config.decoder_layers - 1):
      aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
  loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
  return loss, loss_dict, auxiliary_outputs

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

# Reused from Qwen3ForCausalLM.matcher.GroundingDinoHungarianMatcher
from Qwen3ForCausalLM.matcher import GroundingDinoHungarianMatcher
# Reused from Qwen3ForCausalLM.losses.GroundingDinoImageLoss
from Qwen3ForCausalLM.losses import GroundingDinoImageLoss
# Reused from Qwen3ForCausalLM.modeling_utils._set_aux_loss
from Qwen3ForCausalLM.modeling_utils import _set_aux_loss


def GroundingDinoForObjectDetectionLoss(
    logits: jnp.ndarray,
    labels: List[Dict[str, jnp.ndarray]],
    pred_boxes: jnp.ndarray,
    config: Any,
    label_maps: Tuple[jnp.ndarray, ...],
    text_mask: jnp.ndarray,
    outputs_class: Optional[jnp.ndarray] = None,
    outputs_coord: Optional[jnp.ndarray] = None,
    encoder_logits: Optional[jnp.ndarray] = None,
    encoder_pred_boxes: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Optional[List[Dict[str, jnp.ndarray]]]]:
    """
    Computes the Grounding DINO loss for object detection.

    Args:
        logits (`jnp.ndarray`):
            Classification logits from the model.
        labels (`List[Dict[str, jnp.ndarray]]`):
            A list of ground truth labels for each image in the batch.
        pred_boxes (`jnp.ndarray`):
            Predicted bounding boxes from the model.
        config (`Any`):
            Configuration object with model and loss hyperparameters.
        label_maps (`Tuple[jnp.ndarray, ...]`):
            Label maps for text-based class representations.
        text_mask (`jnp.ndarray`):
            Mask to indicate valid text tokens in the logits.
        outputs_class (`Optional[jnp.ndarray]`, *optional*):
            Classification logits from intermediate decoder layers for auxiliary loss.
        outputs_coord (`Optional[jnp.ndarray]`, *optional*):
            Predicted bounding boxes from intermediate decoder layers for auxiliary loss.
        encoder_logits (`Optional[jnp.ndarray]`, *optional*):
            Classification logits from the encoder for two-stage loss.
        encoder_pred_boxes (`Optional[jnp.ndarray]`, *optional*):
            Predicted bounding boxes from the encoder for two-stage loss.

    Returns:
        A tuple containing:
        - The total computed loss as a scalar `jnp.ndarray`.
        - A dictionary of individual loss components.
        - A list of formatted auxiliary outputs, if applicable.
    """
    # First: create the matcher
    matcher = GroundingDinoHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    # Second: create the criterion
    losses = ["labels", "boxes", "cardinality"]
    criterion = GroundingDinoImageLoss(
        matcher=matcher,
        focal_alpha=config.focal_alpha,
        losses=losses,
    )
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    outputs_loss["label_maps"] = label_maps
    outputs_loss["text_mask"] = text_mask

    auxiliary_outputs = None
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        for aux_output in auxiliary_outputs:
            aux_output["label_maps"] = label_maps
            aux_output["text_mask"] = text_mask
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)

    if config.two_stage:
        encoder_outputs_loss = {
            "logits": encoder_logits,
            "pred_boxes": encoder_pred_boxes,
            "label_maps": label_maps,
            "text_mask": text_mask,
        }
        encoder_loss_dict = criterion(encoder_outputs_loss, labels)
        encoder_loss_dict = {k + "_enc": v for k, v in encoder_loss_dict.items()}
        loss_dict.update(encoder_loss_dict)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": config.bbox_loss_coefficient,
        "loss_giou": config.giou_loss_coefficient,
    }

    if config.two_stage:
        enc_weight_dict = {k + "_enc": v for k, v in weight_dict.items()}
        weight_dict.update(enc_weight_dict)

    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs
from MaxText.losses import (
    ForCausalLMLoss,
    ForMaskedLMLoss,
    ForQuestionAnsweringLoss,
    ForSequenceClassificationLoss,
    ForTokenClassification,
)
from MaxText.losses.loss_d_fine import DFineForObjectDetectionLoss
from MaxText.losses.loss_deformable_detr import (
    DeformableDetrForObjectDetectionLoss,
    DeformableDetrForSegmentationLoss,
)
from MaxText.losses.loss_for_object_detection import (
    ForObjectDetectionLoss,
    ForSegmentationLoss,
)
from MaxText.losses.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
from MaxText.losses.loss_rt_detr import RTDetrForObjectDetectionLoss


LOSS_MAPPING = {
    "ForCausalLM": ForCausalLMLoss,
    "ForMaskedLM": ForMaskedLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForImageClassification": ForSequenceClassificationLoss,
    "ForVideoClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
    "ForSegmentation": ForSegmentationLoss,
    "ForObjectDetection": ForObjectDetectionLoss,
    "ForConditionalGeneration": ForCausalLMLoss,
    "DeformableDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "ConditionalDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "DabDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "GroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "MMGroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "ConditionalDetrForSegmentation": DeformableDetrForSegmentationLoss,
    "RTDetrForObjectDetection": RTDetrForObjectDetectionLoss,
    "RTDetrV2ForObjectDetection": RTDetrForObjectDetectionLoss,
    "DFineForObjectDetection": DFineForObjectDetectionLoss,
    "CsmForConditionalGeneration": ForCausalLMLoss,
}