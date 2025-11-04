
from __future__ import annotations
from typing import Union, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


def load_balancing_loss_func(
    gate_logits: Union[Array, Tuple[Array, ...], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[Array] = None,
) -> Union[Array, int]:
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
        attention_mask (`jax.numpy.ndarray`, *optional*):
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

  _, selected_experts = lax.top_k(routing_weights, top_k)

  expert_mask = jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32)

  if attention_mask is None:
    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = jnp.mean(expert_mask, axis=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = jnp.mean(routing_weights, axis=0)
  else:
    batch_size, sequence_length = attention_mask.shape
    num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

    # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
    expert_attention_mask = jnp.expand_dims(attention_mask, axis=(0, 3, 4))
    expert_attention_mask = jnp.broadcast_to(
        expert_attention_mask, (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
    )
    expert_attention_mask = expert_attention_mask.reshape(-1, top_k, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = jnp.sum(expert_mask * expert_attention_mask, axis=0) / jnp.sum(expert_attention_mask, axis=0)

    # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
    router_per_expert_attention_mask = jnp.expand_dims(attention_mask, axis=(0, 3))
    router_per_expert_attention_mask = jnp.broadcast_to(
        router_per_expert_attention_mask, (num_hidden_layers, batch_size, sequence_length, num_experts)
    )
    router_per_expert_attention_mask = router_per_expert_attention_mask.reshape(-1, num_experts)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = jnp.sum(routing_weights * router_per_expert_attention_mask, axis=0) / jnp.sum(
        router_per_expert_attention_mask, axis=0
    )

  overall_loss = jnp.sum(tokens_per_expert * jnp.expand_dims(router_prob_per_expert, axis=0))
  return overall_loss * num_experts

from typing import Dict, List, Sequence

from jax import Array


def _set_aux_loss(outputs_class: Sequence[Array], outputs_coord: Sequence[Array]) -> List[Dict[str, Array]]:
  # this is a workaround to make torchscript happy, as torchscript
  # doesn't support dictionary with non-homogeneous values, such
  # as a dict having both a Tensor and a list.
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

from typing import List, Dict, Optional, Sequence
from jax import Array


def _set_aux_loss2(
    outputs_class: Sequence[Array],
    outputs_coord: Sequence[Array],
    outputs_corners: Sequence[Array],
    outputs_ref: Sequence[Array],
    teacher_corners: Optional[Array] = None,
    teacher_logits: Optional[Array] = None,
) -> List[Dict[str, Optional[Array]]]:
  """Creates a list of dictionaries for auxiliary losses.

  This is a workaround to make torchscript happy, as torchscript
  doesn't support dictionary with non-homogeneous values, such
  as a dict having both a Tensor and a list.

  Args:
    outputs_class: A sequence of class logits arrays.
    outputs_coord: A sequence of coordinate prediction arrays.
    outputs_corners: A sequence of corner prediction arrays.
    outputs_ref: A sequence of reference point arrays.
    teacher_corners: Optional teacher corner predictions.
    teacher_logits: Optional teacher logits.

  Returns:
    A list of dictionaries, each containing predictions for a single layer.
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
      for a, b, c, d in zip(
          outputs_class, outputs_coord, outputs_corners, outputs_ref
      )
  ]

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
Flax DETR model.
"""
from typing import Any, Dict, List, Sequence

import jax.numpy as jnp


def _set_aux_loss(
    outputs_class: Sequence[jnp.ndarray], outputs_coord: Sequence[jnp.ndarray]
) -> List[Dict[str, Any]]:
  """
  This is a workaround to make torchscript happy, as torchscript
  doesn't support dictionary with non-homogeneous values, such
  as a dict having both a Tensor and a list.
  """
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

from typing import Dict, List, Sequence
from jax import Array


def _set_aux_loss(
    outputs_class: Sequence[Array], outputs_coord: Sequence[Array]
) -> List[Dict[str, Array]]:
  """Formats auxiliary layer outputs into a list of dictionaries.

  This is a helper function to structure the outputs from intermediate decoder
  layers for auxiliary loss calculation. It pairs up class logits and predicted
  boxes for each layer.

  Args:
    outputs_class: A sequence of class prediction tensors from intermediate
      layers.
    outputs_coord: A sequence of bounding box coordinate prediction tensors from
      intermediate layers.

  Returns:
    A list of dictionaries, where each dictionary contains the 'logits' and
    'pred_boxes' for one auxiliary layer.
  """
  # this is a workaround to make torchscript happy, as torchscript
  # doesn't support dictionary with non-homogeneous values, such
  # as a dict having both a Tensor and a list.
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

from typing import Optional

from jax import Array
import jax.numpy as jnp

# From src.MaxText.maxtext_utils.cross_entropy_with_logits
from MaxText import maxtext_utils


def fixed_cross_entropy(
    source: Array,
    target: Array,
    num_items_in_batch: Optional[Array] = None,
    ignore_index: int = -100,
    **kwargs,
) -> Array:
  """Computes cross-entropy loss with optional manual normalization.

  This function calculates cross-entropy loss, mirroring the logic of summing
  the loss and then dividing by a specified number of items, or taking a
  standard mean over valid (non-ignored) tokens.

  Args:
    source: Logits tensor of shape [..., vocab_size].
    target: Target labels of shape [...].
    num_items_in_batch: Optional denominator for normalization. If provided,
      the total loss is divided by this value.
    ignore_index: The label index to be ignored in the loss calculation.
    **kwargs: Additional arguments, ignored.

  Returns:
    The calculated cross-entropy loss as a scalar array.
  """
  # cross_entropy_with_logits returns the total loss and the number of non-padded tokens
  total_loss, num_valid_tokens = maxtext_utils.cross_entropy_with_logits(
      logits=source,
      targets=target,
      label_smoothing=0.0,
      z_loss=0.0,
      ignore_index=ignore_index,
  )

  if num_items_in_batch is not None:
    # Corresponds to PyTorch's reduction='sum' followed by manual division.
    loss = total_loss / num_items_in_batch
  else:
    # Corresponds to PyTorch's reduction='mean'.
    # Use jnp.maximum to prevent division by zero if there are no valid tokens.
    loss = total_loss / jnp.maximum(1.0, num_valid_tokens)

  return loss

from typing import Optional

import jax
import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.losses.fixed_cross_entropy
from .loss import fixed_cross_entropy


def ForCausalLMLoss(
    logits: jax.Array,
    labels: jax.Array,
    vocab_size: int,
    num_items_in_batch: Optional[jax.Array] = None,
    ignore_index: int = -100,
    shift_labels: Optional[jax.Array] = None,
    **kwargs,
) -> jax.Array:
  """Computes causal language modeling loss."""
  # Upcast to float if we need to compute the loss to avoid potential precision issues
  logits = logits.astype(jnp.float32)

  if shift_labels is None:
    # Shift so that tokens < n predict n
    pad_width = [(0, 0)] * (labels.ndim - 1) + [(0, 1)]
    labels = jnp.pad(labels, pad_width, constant_values=ignore_index)
    shift_labels = labels[..., 1:]

  # Flatten the tokens
  logits = logits.reshape(-1, vocab_size)
  shift_labels = shift_labels.reshape(-1)
  # Enable model parallelism (handled by JAX sharding)

  loss = fixed_cross_entropy(
      source=logits, target=shift_labels, num_items_in_batch=num_items_in_batch, ignore_index=ignore_index, **kwargs
  )
  return loss

from typing import Optional

import jax.numpy as jnp
from jax import Array

# Reused from generated_code.Qwen3MoeForCausalLM.losses.fixed_cross_entropy
from MaxText.losses import fixed_cross_entropy


def ForMaskedLMLoss(
    logits: Array,
    labels: Array,
    vocab_size: int,
    num_items_in_batch: Optional[Array] = None,
    ignore_index: int = -100,
    **kwargs,
) -> Array:
  """Computes the masked language modeling loss."""
  # Upcast to float if we need to compute the loss to avoid potential precision issues
  logits = logits.astype(jnp.float32)

  # Flatten the tokens
  logits = logits.reshape(-1, vocab_size)
  labels = labels.reshape(-1)
  # Enable model parallelism

  # In JAX, device placement is handled by the context (e.g., pjit), so `labels.to(logits.device)` is not needed.
  loss = fixed_cross_entropy(
      source=logits,
      target=labels,
      num_items_in_batch=num_items_in_batch,
      ignore_index=ignore_index,
      **kwargs,
  )
  return loss

from typing import Optional

import jax.numpy as jnp
from jax import Array

# Assuming fixed_cross_entropy is defined in the same file, as in the PyTorch source.
# A JAX implementation of fixed_cross_entropy was found at generated_code.Qwen3MoeForCausalLM.losses.fixed_cross_entropy
from .losses import fixed_cross_entropy


def ForQuestionAnsweringLoss(
    start_logits: Array,
    end_logits: Array,
    start_positions: Optional[Array],
    end_positions: Optional[Array],
    **kwargs,
) -> Optional[Array]:
  """
  Computes the loss for question answering tasks.
  """
  total_loss: Optional[Array] = None
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

    start_loss = fixed_cross_entropy(start_logits, start_positions, ignore_index=ignored_index, **kwargs)
    end_loss = fixed_cross_entropy(end_logits, end_positions, ignore_index=ignored_index, **kwargs)
    total_loss = (start_loss + end_loss) / 2
  return total_loss

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
Loss functions for MaxText.
"""

from typing import Any
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax

from MaxText.common_types import Config
# Reused from generated_code.Qwen3MoeForCausalLM.losses.fixed_cross_entropy
from MaxText.losses import fixed_cross_entropy


def ForSequenceClassificationLoss(labels: ArrayLike, pooled_logits: ArrayLike, config: Config, **kwargs: Any) -> ArrayLike:
  """Computes loss for sequence classification tasks."""
  num_labels = config.num_labels
  problem_type = config.problem_type
  if problem_type is None:
    if num_labels == 1:
      problem_type = "regression"
    elif num_labels > 1 and jnp.issubdtype(labels.dtype, jnp.integer):
      problem_type = "single_label_classification"
    else:
      problem_type = "multi_label_classification"

  if problem_type == "regression":
    if num_labels == 1:
      loss = optax.squared_error(jnp.squeeze(pooled_logits), jnp.squeeze(labels))
    else:
      loss = optax.squared_error(pooled_logits, labels)
    return jnp.mean(loss)

  if problem_type == "single_label_classification":
    return fixed_cross_entropy(pooled_logits.reshape(-1, num_labels), labels.reshape(-1), **kwargs)

  if problem_type == "multi_label_classification":
    loss = optax.sigmoid_binary_cross_entropy(pooled_logits, labels)
    return jnp.mean(loss)

  raise RuntimeError(f"Invalid problem type: {problem_type}")

from typing import Any
from jax import Array
import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.losses.fixed_cross_entropy
from generated_code.Qwen3MoeForCausalLM.losses import fixed_cross_entropy


def ForTokenClassification(logits: Array, labels: Array, config: Any, **kwargs) -> Array:
  """Computes the loss for token classification tasks.

  Args:
    logits: The predicted logits from the model, with shape [batch, seq_len,
      num_labels].
    labels: The ground truth labels, with shape [batch, seq_len].
    config: The model configuration object, containing `num_labels`.
    **kwargs: Additional arguments to be passed to the loss function.

  Returns:
    The computed cross-entropy loss as a scalar array.
  """
  # Upcast to float if we need to compute the loss to avoid potential precision
  # issues
  logits = logits.reshape(-1, config.num_labels)
  labels = labels.reshape(-1)
  logits = logits.astype(jnp.float32)
  # Flatten the tokens
  return fixed_cross_entropy(logits, labels, **kwargs)

import jax
import jax.numpy as jnp
import optax
from jax import Array


def sigmoid_focal_loss(
    inputs: Array, targets: Array, num_boxes: int, alpha: float = 0.25, gamma: float = 2
) -> Array:
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.

    Args:
        inputs (`jnp.ndarray` of arbitrary shape):
            The predictions for each example.
        targets (`jnp.ndarray` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`int`):
            The number of boxes.
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
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

    return loss.mean(1).sum() / num_boxes

import jax
import jax.numpy as jnp
from jax import Array


def dice_loss(inputs: Array, targets: Array, num_boxes: float) -> Array:
  """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float array of arbitrary shape.
                The predictions for each example.
        targets: A float array with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        num_boxes: The number of boxes to normalize the loss by.
    """
  inputs = jax.nn.sigmoid(inputs)
  inputs = inputs.reshape((inputs.shape[0], -1))
  numerator = 2 * (inputs * targets).sum(axis=1)
  denominator = inputs.sum(axis=-1) + targets.sum(axis=-1)
  loss = 1 - (numerator + 1) / (denominator + 1)
  return loss.sum() / num_boxes

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float


# Reused from generated_code.Qwen3MoeForCausalLM.losses.sigmoid_focal_loss
def sigmoid_focal_loss(
    inputs: Float[Array, "..."],
    targets: Float[Array, "..."],
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Float[Array, ""]:
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.

    Args:
        inputs (`jax.Array` of arbitrary shape):
            The predictions for each example.
        targets (`jax.Array` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`int`):
            The total number of boxes in the batch.
        alpha (`float`, *optional*, defaults to 0.25):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`float`, *optional*, defaults to 2.0):
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
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from flax import linen as nn
from jax import Array


# Bounding box utilities converted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def _upcast(t: Array) -> Array:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if jnp.issubdtype(t.dtype, jnp.floating):
        return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
    else:
        return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)


def box_area(boxes: Array) -> Array:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Array, boxes2: Array) -> Tuple[Array, Array]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = jnp.clip(right_bottom - left_top, a_min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.
    """
    # degenerate boxes gives inf / nan results so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = jnp.clip(bottom_right - top_left, a_min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# Converted from transformers.image_transforms
def center_to_corners_format(bboxes_center: Array) -> Array:
    center_x, center_y, width, height = jnp.moveaxis(bboxes_center, -1, 0)
    x1 = center_x - 0.5 * width
    y1 = center_y - 0.5 * height
    x2 = center_x + 0.5 * width
    y2 = center_y + 0.5 * height
    return jnp.stack([x1, y1, x2, y2], axis=-1)


# Loss functions converted from the source file
def dice_loss(inputs: Array, targets: Array, num_boxes: float) -> Array:
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """
    inputs = jax.nn.sigmoid(inputs)
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    numerator = 2 * (inputs * targets).sum(axis=1)
    denominator = inputs.sum(axis=-1) + targets.sum(axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs: Array, targets: Array, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0
) -> Array:
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.
    """
    prob = jax.nn.sigmoid(inputs)
    ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(axis=1).sum() / num_boxes


# Misc utilities converted from https://github.com/facebookresearch/detr/blob/master/util/misc.py
def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    if not the_list:
        return []
    return jnp.array(the_list).max(axis=0).tolist()


@dataclass
class NestedTensor:
    tensors: Array
    mask: Optional[Array]

    def decompose(self) -> Tuple[Array, Optional[Array]]:
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Array]) -> NestedTensor:
    if not tensor_list:
        return NestedTensor(tensors=jnp.array([]), mask=jnp.array([]))
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        
        padded_tensors = []
        padded_masks = []
        for img in tensor_list:
            pad_shape = [
                (0, max_size[0] - img.shape[0]),
                (0, max_size[1] - img.shape[1]),
                (0, max_size[2] - img.shape[2]),
            ]
            padded_tensors.append(jnp.pad(img, pad_shape))
            
            mask_pad_shape = [
                (0, max_size[1] - img.shape[1]),
                (0, max_size[2] - img.shape[2]),
            ]
            img_mask = jnp.zeros(img.shape[1:], dtype=jnp.bool_)
            padded_masks.append(jnp.pad(img_mask, mask_pad_shape, constant_values=True))

        tensor = jnp.stack(padded_tensors)
        mask = jnp.stack(padded_masks)
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


class ImageLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).
    """

    matcher: nn.Module
    num_classes: int
    eos_coef: float
    losses: List[str]

    def setup(self):
        empty_weight = jnp.ones(self.num_classes + 1)
        self.empty_weight = empty_weight.at[-1].set(self.eos_coef)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = jnp.full(source_logits.shape[:2], self.num_classes, dtype=jnp.int64)
        target_classes = target_classes.at[idx].set(target_classes_o)

        # Transpose source_logits to [batch, num_classes, num_queries] to match PyTorch's cross_entropy
        source_logits_transposed = jnp.transpose(source_logits, (0, 2, 1))
        log_softmax = jax.nn.log_softmax(source_logits_transposed, axis=1)
        one_hot_labels = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1, axis=1)
        loss_per_token = -jnp.sum(one_hot_labels * log_softmax, axis=1)
        weights_per_token = self.empty_weight[target_classes]
        weighted_loss_per_token = loss_per_token * weights_per_token
        loss_ce = jnp.mean(weighted_loss_per_token)

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        target_lengths = jnp.array([len(v["class_labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (jnp.argmax(logits, axis=-1) != logits.shape[-1] - 1).sum(axis=1)
        card_err = optax.l1_loss(card_pred.astype(jnp.float32), target_lengths.astype(jnp.float32))
        losses = {"cardinality_error": jax.lax.stop_gradient(card_err)}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
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

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.astype(source_masks.dtype)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks_chw = source_masks[:, None, :, :]
        source_masks_hwc = jnp.transpose(source_masks_chw, (0, 2, 3, 1))
        new_shape = (
            source_masks.shape[0],
            target_masks.shape[-2],
            target_masks.shape[-1],
            1,
        )
        resized_hwc = jax.image.resize(source_masks_hwc, new_shape, method="bilinear")
        resized_chw = jnp.transpose(resized_hwc, (0, 3, 1, 2))
        source_masks = resized_chw.squeeze(axis=1)
        source_masks = source_masks.reshape((source_masks.shape[0], -1))

        target_masks = target_masks.reshape((target_masks.shape[0], -1))
        target_masks = target_masks.reshape(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = jnp.concatenate([jnp.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = jnp.concatenate([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = jnp.concatenate([jnp.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = jnp.concatenate([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def __call__(self, outputs, targets):
        """
        This performs the loss computation.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes_local = sum(len(t["class_labels"]) for t in targets)
        # Assumes pmap/pjit context with a 'data' axis for multi-device computation
        num_boxes_global_average = jax.lax.pmean(num_boxes_local, axis_name="data")
        num_boxes = jnp.clip(num_boxes_global_average, a_min=1)

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

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.matchers.HungarianMatcher
from generated_code.Qwen3MoeForCausalLM.matchers import HungarianMatcher
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ImageLoss
from generated_code.Qwen3MoeForCausalLM.losses import ImageLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss
from generated_code.Qwen3MoeForCausalLM.losses import _set_aux_loss


def ForObjectDetectionLoss(
    logits: jnp.ndarray,
    labels: List[Dict[str, jnp.ndarray]],
    pred_boxes: jnp.ndarray,
    config: Any,
    outputs_class: Optional[jnp.ndarray] = None,
    outputs_coord: Optional[jnp.ndarray] = None,
    **kwargs,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Optional[List[Dict[str, jnp.ndarray]]]]:
  """
  Computes the loss for object detection, typically for a DETR-like model.

  This function orchestrates the four main steps of DETR loss computation:
  1. Matching predictions to ground truth targets using the Hungarian algorithm.
  2. Defining the loss criteria (classification, bounding box, GIoU).
  3. Computing the individual losses for the final and auxiliary decoder outputs.
  4. Calculating the total loss as a weighted sum of the individual losses.

  Args:
    logits: Class logits from the model's output. Shape: (batch_size, num_queries, num_classes + 1).
    labels: A list of dictionaries, one per image, containing ground truth 'class_labels' and 'boxes'.
    pred_boxes: Predicted bounding boxes from the model's output. Shape: (batch_size, num_queries, 4).
    config: A configuration object with parameters like costs and loss coefficients.
    outputs_class: Optional class logits from intermediate decoder layers for auxiliary loss.
    outputs_coord: Optional bounding box predictions from intermediate decoder layers for auxiliary loss.
    **kwargs: Additional keyword arguments (not used).

  Returns:
    A tuple containing:
      - The total weighted loss as a scalar JAX array.
      - A dictionary of all computed individual losses.
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

from functools import partial
from typing import Dict, List, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import numpy
import optax
from flax import linen as nn
from jax import Array
from scipy.optimize import linear_sum_assignment

# MaxText matched dependencies:
# generated_code.Qwen3MoeForCausalLM.bbox_utils.center_to_corners_format
# generated_code.Qwen3MoeForCausalLM.bbox_utils.generalized_box_iou
# generated_code.Qwen3MoeForCausalLM.bbox_utils.box_iou
# generated_code.Qwen3MoeForCausalLM.bbox_utils.box_area
# generated_code.Qwen3MoeForCausalLM.tensor_utils._upcast
# generated_code.Qwen3MoeForCausalLM.losses.dice_loss
# generated_code.Qwen3MoeForCausalLM.losses.sigmoid_focal_loss
# generated_code.Qwen3MoeForCausalLM.losses.ImageLoss
# generated_code.Qwen3MoeForCausalLM.matchers.HungarianMatcher
# generated_code.Qwen3MoeForCausalLM.tensor_utils.NestedTensor
# generated_code.Qwen3MoeForCausalLM.tensor_utils.nested_tensor_from_tensor_list
# generated_code.Qwen3MoeForCausalLM.tensor_utils._max_by_axis
# generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss


def _upcast(t: Array) -> Array:
  # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
  else:
    return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)


def box_area(boxes: Array) -> Array:
  """
  Computes the area of a set of bounding boxes, which are specified by their (x1, y1, x2, y2) coordinates.
  """
  boxes = _upcast(boxes)
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  left_top = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])
  right_bottom = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

  width_height = jnp.maximum(0.0, right_bottom - left_top)
  inter = width_height[:, :, 0] * width_height[:, :, 1]

  union = area1[:, None] + area2 - inter

  iou = inter / union
  return iou, union


def generalized_box_iou(boxes1, boxes2):
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

  width_height = jnp.maximum(0.0, bottom_right - top_left)
  area = width_height[:, :, 0] * width_height[:, :, 1]

  return iou - (area - union) / (area + 1e-6)


def center_to_corners_format(bboxes_center: Array) -> Array:
  center_x, center_y, width, height = jnp.moveaxis(bboxes_center, -1, 0)
  top_left_x = center_x - 0.5 * width
  top_left_y = center_y - 0.5 * height
  bottom_right_x = center_x + 0.5 * width
  bottom_right_y = center_y + 0.5 * height
  return jnp.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)


def dice_loss(inputs, targets, num_boxes):
  """
  Compute the DICE loss, similar to generalized IOU for masks
  """
  inputs = jax.nn.sigmoid(inputs)
  inputs = inputs.reshape(inputs.shape[0], -1)
  numerator = 2 * (inputs * targets).sum(1)
  denominator = inputs.sum(-1) + targets.sum(-1)
  loss = 1 - (numerator + 1) / (denominator + 1)
  return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0):
  """
  Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.
  """
  prob = jax.nn.sigmoid(inputs)
  ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)

  p_t = prob * targets + (1 - prob) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  return loss.mean(1).sum() / num_boxes


@flax.struct.dataclass
class NestedTensor:
  tensors: Array
  mask: Optional[Array]

  def decompose(self):
    return self.tensors, self.mask

  def __repr__(self):
    return str(self.tensors)


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
  maxes = the_list[0]
  for sublist in the_list[1:]:
    for index, item in enumerate(sublist):
      maxes[index] = max(maxes[index], item)
  return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Array]):
  if tensor_list[0].ndim == 3:
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    batch_size, num_channels, height, width = batch_shape
    dtype = tensor_list[0].dtype
    tensor = jnp.zeros(batch_shape, dtype=dtype)
    mask = jnp.ones((batch_size, height, width), dtype=jnp.bool_)
    for i, img in enumerate(tensor_list):
      tensor = tensor.at[i, : img.shape[0], : img.shape[1], : img.shape[2]].set(img)
      mask = mask.at[i, : img.shape[1], : img.shape[2]].set(False)
  else:
    raise ValueError("Only 3-dimensional tensors are supported")
  return NestedTensor(tensor, mask)


class HungarianMatcher(nn.Module):
  class_cost: float = 1.0
  bbox_cost: float = 1.0
  giou_cost: float = 1.0

  def setup(self):
    if self.class_cost == 0 and self.bbox_cost == 0 and self.giou_cost == 0:
      raise ValueError("All costs of the Matcher can't be 0")

  def __call__(self, outputs, targets):
    batch_size, num_queries = outputs["logits"].shape[:2]

    out_prob = jax.nn.softmax(outputs["logits"].reshape(-1, outputs["logits"].shape[-1]), axis=-1)
    out_bbox = outputs["pred_boxes"].reshape(-1, 4)

    target_ids = jnp.concatenate([v["class_labels"] for v in targets])
    target_bbox = jnp.concatenate([v["boxes"] for v in targets])

    class_cost = -out_prob[:, target_ids]

    bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

    cost_matrix_np = numpy.asarray(cost_matrix)
    sizes = [len(v["boxes"]) for v in targets]
    indices = []
    offset = 0
    for i, size in enumerate(sizes):
      if size == 0:
        indices.append((numpy.array([], dtype=numpy.int64), numpy.array([], dtype=numpy.int64)))
        continue
      cost_slice = cost_matrix_np[i, :, offset : offset + size]
      indices.append(linear_sum_assignment(cost_slice))
      offset += size

    return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]


class ImageLoss(nn.Module):
  matcher: nn.Module
  num_classes: int
  eos_coef: float
  losses: List[str]

  def setup(self):
    empty_weight = jnp.ones(self.num_classes + 1)
    self.empty_weight = empty_weight.at[-1].set(self.eos_coef)

  def loss_labels(self, outputs, targets, indices, num_boxes):
    if "logits" not in outputs:
      raise KeyError("No logits were found in the outputs")
    source_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_o = jnp.concatenate([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = jnp.full(source_logits.shape[:2], self.num_classes, dtype=jnp.int64)
    target_classes = target_classes.at[idx].set(target_classes_o)

    # Manual cross-entropy with class weights
    one_hot_labels = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)
    log_probs = jax.nn.log_softmax(source_logits, axis=-1)
    loss_per_token = -jnp.sum(one_hot_labels * log_probs, axis=-1)
    class_weights = jnp.take(self.empty_weight, target_classes)
    weighted_loss = loss_per_token * class_weights
    loss_ce = weighted_loss.mean()

    losses = {"loss_ce": loss_ce}
    return losses

  def loss_cardinality(self, outputs, targets, indices, num_boxes):
    logits = outputs["logits"]
    target_lengths = jnp.asarray([len(v["class_labels"]) for v in targets])
    card_pred = (jnp.argmax(logits, axis=-1) != logits.shape[-1] - 1).sum(1)
    card_err = optax.l1_loss(card_pred.astype(jnp.float32), target_lengths.astype(jnp.float32)).mean()
    losses = {"cardinality_error": jax.lax.stop_gradient(card_err)}
    return losses

  def loss_boxes(self, outputs, targets, indices, num_boxes):
    if "pred_boxes" not in outputs:
      raise KeyError("No predicted boxes found in outputs")
    idx = self._get_source_permutation_idx(indices)
    source_boxes = outputs["pred_boxes"][idx]
    target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

    loss_bbox = optax.l1_loss(source_boxes, target_boxes).sum() / num_boxes

    losses = {"loss_bbox": loss_bbox}

    loss_giou = (
        1
        - jnp.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
    ).sum() / num_boxes
    losses["loss_giou"] = loss_giou
    return losses

  def loss_masks(self, outputs, targets, indices, num_boxes):
    if "pred_masks" not in outputs:
      raise KeyError("No predicted masks found in outputs")

    source_idx = self._get_source_permutation_idx(indices)
    target_idx = self._get_target_permutation_idx(indices)
    source_masks = outputs["pred_masks"]
    source_masks = source_masks[source_idx]
    masks = [t["masks"] for t in targets]

    target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
    target_masks = target_masks[target_idx]

    source_masks = jax.image.resize(
        source_masks[:, None],
        (source_masks.shape[0], 1) + target_masks.shape[-2:],
        method="bilinear",
    )
    source_masks = source_masks[:, 0].reshape(source_masks.shape[0], -1)

    target_masks = target_masks.reshape(target_masks.shape[0], -1)
    losses = {
        "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
        "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
    }
    return losses

  def _get_source_permutation_idx(self, indices):
    batch_idx = jnp.concatenate([jnp.full_like(source, i) for i, (source, _) in enumerate(indices)])
    source_idx = jnp.concatenate([source for (source, _) in indices])
    return batch_idx, source_idx

  def _get_target_permutation_idx(self, indices):
    batch_idx = jnp.concatenate([jnp.full_like(target, i) for i, (_, target) in enumerate(indices)])
    target_idx = jnp.concatenate([target for (_, target) in indices])
    return batch_idx, target_idx

  def get_loss(self, loss, outputs, targets, indices, num_boxes):
    loss_map = {
        "labels": self.loss_labels,
        "cardinality": self.loss_cardinality,
        "boxes": self.loss_boxes,
        "masks": self.loss_masks,
    }
    if loss not in loss_map:
      raise ValueError(f"Loss {loss} not supported")
    return loss_map[loss](outputs, targets, indices, num_boxes)

  def __call__(self, outputs, targets):
    outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

    indices = self.matcher(outputs_without_aux, targets)

    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = jnp.asarray([num_boxes], dtype=jnp.float32)
    num_boxes = jax.lax.pmean(num_boxes, axis_name="batch").sum()
    num_boxes = jnp.maximum(num_boxes, 1.0)

    losses = {}
    for loss in self.losses:
      losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

    if "auxiliary_outputs" in outputs:
      for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
        indices = self.matcher(auxiliary_outputs, targets)
        for loss in self.losses:
          if loss == "masks":
            continue
          l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
          l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
          losses.update(l_dict)

    return losses


def _set_aux_loss(outputs_class, outputs_coord):
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def ForSegmentationLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    pred_masks: Array,
    config,
    outputs_class: Optional[Array] = None,
    outputs_coord: Optional[Array] = None,
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Optional[List[Dict[str, Array]]]]:
  """
  Computes the loss for DETR segmentation.
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

  # The device argument is not needed in JAX.
  # We assume the computation is run under a pjit context.
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

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any, Dict, List, Tuple

from transformers.modeling_flax_utils import FlaxPreTrainedModel


# The following `sigmoid_focal_loss` function is a JAX conversion of the
# PyTorch version from the original file, as it is a dependency for `loss_labels`.
def sigmoid_focal_loss(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.
    Args:
        inputs (`jnp.ndarray` of arbitrary shape):
            The predictions for each example.
        targets (`jnp.ndarray` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`int`):
            The total number of boxes in the batch.
        alpha (`float`, *optional*, defaults to 0.25):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to 2):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = jax.nn.sigmoid(inputs)
    ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * jnp.power(1 - p_t, gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return jnp.sum(loss) / num_boxes


class GroundingDinoImageLoss(nn.Module):
    """
    This class computes the losses for `GroundingDinoForObjectDetection`. The process happens in two steps: 1) we
    compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
    matched ground-truth / prediction (supervise class and box).
    This is the JAX/Flax equivalent of the PyTorch GroundingDinoImageLoss class.
    The original class inherits from `ImageLoss`, which is not provided. This implementation inherits from `nn.Module`
    and includes necessary helper methods.

    Attributes:
        matcher (`nn.Module`):
            Module able to compute a matching between targets and proposals.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    matcher: Any
    focal_alpha: float
    losses: List[str]

    # The _get_source_permutation_idx method is part of the base ImageLoss class in the original
    # implementation. It is included here as a helper method.
    def _get_source_permutation_idx(
        self, indices: List[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Gets the source permutation index for a batch of indices.
        Args:
            indices (`List[Tuple[jnp.ndarray, jnp.ndarray]]`):
                A list of tuples of matched indices for each batch element.
        Returns:
            A tuple of two arrays: the batch indices and the source indices.
        """
        batch_idx = jnp.concatenate([jnp.full_like(src, i, dtype=jnp.int32) for i, (src, _) in enumerate(indices)])
        src_idx = jnp.concatenate([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def _get_target_classes_one_hot(
        self, outputs: Dict, targets: List[Dict], indices: List[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> jnp.ndarray:
        """
        Create one_hot based on the matching indices
        """
        logits = outputs["logits"]
        # Add offsets to class_labels to select the correct label map
        label_map_lengths = [label_map.shape[0] for label_map in outputs["label_maps"]]
        offsets = np.cumsum([0] + label_map_lengths[:-1])

        class_labels_list = [
            targets[i]["class_labels"][J] + offsets[i] for i, (target, (_, J)) in enumerate(zip(targets, indices))
        ]
        class_labels = jnp.concatenate(class_labels_list)

        label_maps = jnp.concatenate(outputs["label_maps"], axis=0)

        idx = self._get_source_permutation_idx(indices)
        target_classes_onehot = jnp.zeros_like(logits, dtype=jnp.int32)
        selected_labels = label_maps[class_labels].astype(jnp.int32)
        target_classes_onehot = target_classes_onehot.at[idx].set(selected_labels)

        return target_classes_onehot

    def loss_labels(
        self,
        outputs: Dict,
        targets: List[Dict],
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
            gamma=2.0,
        )

        losses = {"loss_ce": loss_ce}

        return losses

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.linen import Module
from scipy.optimize import linear_sum_assignment

# Reused from generated_code.Qwen3MoeForCausalLM.matchers.HungarianMatcher
from generated_code.Qwen3MoeForCausalLM.matchers import HungarianMatcher
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ImageLoss
from generated_code.Qwen3MoeForCausalLM.losses import ImageLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.sigmoid_focal_loss
from generated_code.Qwen3MoeForCausalLM.losses import sigmoid_focal_loss
# Reused from generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss
from generated_code.Qwen3MoeForCausalLM.losses import _set_aux_loss
# Reused from generated_code.Qwen3MoeForCausalLM.bbox_utils.generalized_box_iou
from generated_code.Qwen3MoeForCausalLM.bbox_utils import generalized_box_iou
# Reused from generated_code.Qwen3MoeForCausalLM.bbox_utils.center_to_corners_format
from generated_code.Qwen3MoeForCausalLM.bbox_utils import center_to_corners_format


class GroundingDinoHungarianMatcher(HungarianMatcher):
  """
  Computes an assignment between targets and predictions of the model.
  For efficiency reasons, the targets don't include the no_object. Because of this, in general,
  there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
  while the others are un-matched (and thus treated as non-objects).
  """

  def __call__(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Args:
        outputs (`dict`):
            A dictionary that contains at least these entries:
            * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            * "label_maps": Tuple of tensors of dim [num_classes, hidden_dim].
        targets (`list[dict]`):
            A list of targets (len(targets) = batch_size), where each target is a dict containing:
            * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
              ground-truth
             objects in the target) containing the class labels
            * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.
    Returns:
        `list[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
        - index_i is the indices of the selected predictions (in order)
        - index_j is the indices of the corresponding selected targets (in order)
        For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    """
    batch_size, num_queries = outputs["logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_prob = jax.nn.sigmoid(outputs["logits"].reshape(batch_size * num_queries, -1))
    out_bbox = outputs["pred_boxes"].reshape(batch_size * num_queries, -1)
    label_maps = outputs["label_maps"]

    # First take the label map for each class in each batch and then concatenate them
    label_maps_cat = jnp.concatenate([label_map[target["class_labels"]] for label_map, target in zip(label_maps, targets)])
    # Normalize label maps based on number of tokens per class
    label_maps_cat = label_maps_cat / jnp.sum(label_maps_cat, axis=-1, keepdims=True)

    # Also concat the target labels and boxes
    target_bbox = jnp.concatenate([v["boxes"] for v in targets])

    # Compute the classification cost.
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-jnp.log(1 - out_prob + 1e-8))
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-jnp.log(out_prob + 1e-8))
    # Compute the classification cost by taking pos and neg cost in the appropriate index
    class_cost = (pos_cost_class - neg_cost_class) @ label_maps_cat.T

    # Compute the L1 cost between boxes
    bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

    # Compute the giou cost between boxes
    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    # Final cost matrix
    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

    sizes = [len(v["boxes"]) for v in targets]
    # Move to CPU and convert to numpy for scipy
    cost_matrix_np = np.array(cost_matrix)
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(np.split(cost_matrix_np, np.cumsum(sizes)[:-1], axis=-1))]
    return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]


class GroundingDinoImageLoss(ImageLoss):
  """
  This class computes the losses for `GroundingDinoForObjectDetection`. The process happens in two steps: 1) we
  compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
  matched ground-truth / prediction (supervise class and box).
  Args:
      matcher (`GroundingDinoHungarianMatcher`):
          Module able to compute a matching between targets and proposals.
      focal_alpha (`float`):
          Alpha parameter in focal loss.
      losses (`list[str]`):
          List of all the losses to be applied. See `get_loss` for a list of all available losses.
  """

  focal_alpha: float

  def _get_target_classes_one_hot(self, outputs, targets, indices):
    """
    Create one_hot based on the matching indices
    """
    logits = outputs["logits"]
    # Add offsets to class_labels to select the correct label map
    class_labels_list = []
    offset = 0
    for i, (target, (_, J)) in enumerate(zip(targets, indices)):
      class_labels_list.append(target["class_labels"][J] + offset)
      offset += outputs["label_maps"][i].shape[0]

    class_labels = jnp.concatenate(class_labels_list)
    label_maps = jnp.concatenate(outputs["label_maps"], axis=0)

    idx = self._get_source_permutation_idx(indices)
    target_classes_onehot = jnp.zeros_like(logits, dtype=jnp.int32)
    target_classes_onehot = target_classes_onehot.at[idx].set(label_maps[class_labels].astype(jnp.int32))

    return target_classes_onehot

  def loss_labels(self, outputs, targets, indices, num_boxes):
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
        gamma=2.0,
    )

    losses = {"loss_ce": loss_ce}

    return losses


def GroundingDinoForObjectDetectionLoss(
    logits: jnp.ndarray,
    labels: List[Dict[str, jnp.ndarray]],
    pred_boxes: jnp.ndarray,
    config: FrozenDict,
    label_maps: Tuple[jnp.ndarray, ...],
    text_mask: jnp.ndarray,
    outputs_class: Optional[jnp.ndarray] = None,
    outputs_coord: Optional[jnp.ndarray] = None,
    encoder_logits: Optional[jnp.ndarray] = None,
    encoder_pred_boxes: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Optional[List[Dict[str, jnp.ndarray]]]]:
  # First: create the matcher
  matcher = GroundingDinoHungarianMatcher(
      class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
  )
  # Second: create the criterion
  losses = ["labels", "boxes", "cardinality"]
  criterion = GroundingDinoImageLoss(
      matcher=matcher,
      num_classes=config.num_labels,
      eos_coef=config.eos_coefficient,
      losses=losses,
      focal_alpha=config.focal_alpha,
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

import jax
import jax.numpy as jnp
from flax.linen import nn
from typing import Dict, List, Tuple

# Re-used modules from MaxText
# Path: generated_code/Qwen3MoeForCausalLM/losses/ImageLoss
from generated_code.Qwen3MoeForCausalLM.losses import ImageLoss
# Path: generated_code/Qwen3MoeForCausalLM/losses/sigmoid_focal_loss
from generated_code.Qwen3MoeForCausalLM.losses import sigmoid_focal_loss


class DeformableDetrImageLoss(ImageLoss):
  """
  Computes the loss for Deformable DETR. The process is as follows:
  1) for each image, find the best match between predicted and ground truth objects
  2) compute all the requested losses
  """

  focal_alpha: float

  def loss_labels(
      self,
      outputs: Dict[str, jnp.ndarray],
      targets: List[Dict[str, jnp.ndarray]],
      indices: List[Tuple[jnp.ndarray, jnp.ndarray]],
      num_boxes: int,
  ) -> Dict[str, jnp.ndarray]:
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

    target_classes_onehot = jax.nn.one_hot(
        target_classes, num_classes=source_logits.shape[2] + 1, dtype=source_logits.dtype
    )

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = (
        sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
        * source_logits.shape[1]
    )
    losses = {"loss_ce": loss_ce}

    return losses

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss
from generated_code.Qwen3MoeForCausalLM.losses import _set_aux_loss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.DeformableDetrImageLoss
from generated_code.Qwen3MoeForCausalLM.losses import DeformableDetrImageLoss
# Reused from generated_code.Qwen3MoeForCausalLM.matchers.DeformableDetrHungarianMatcher
from generated_code.Qwen3MoeForCausalLM.matchers import DeformableDetrHungarianMatcher


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
  Computes the loss for Deformable DETR object detection.

  Args:
    logits: The model's predicted logits.
    labels: A list of dictionaries, each containing the ground truth labels for an image.
    pred_boxes: The model's predicted bounding boxes.
    config: The model configuration object.
    outputs_class: Intermediate decoder layer class logits for auxiliary loss.
    outputs_coord: Intermediate decoder layer coordinates for auxiliary loss.
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
    auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
    outputs_loss["auxiliary_outputs"] = auxiliary_outputs

  loss_dict = criterion(outputs=outputs_loss, targets=labels)
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

from maxtext.common_types import Array

# Reused from generated_code.Qwen3MoeForCausalLM.matchers.HungarianMatcher
from ..matchers import HungarianMatcher
# Reused from generated_code.Qwen3MoeForCausalLM.losses.DeformableDetrImageLoss
from . import DeformableDetrImageLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss
from .loss_for_object_detection import _set_aux_loss


def DeformableDetrForSegmentationLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    pred_masks: Array,
    config: Any,
    outputs_class: Optional[Array] = None,
    outputs_coord: Optional[Array] = None,
    **kwargs: Any,
) -> Tuple[Array, Dict[str, Array], Optional[List[Dict[str, Array]]]]:
    """
    Computes the loss for Deformable DETR for segmentation.
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

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import Array
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional, Tuple

import optax

# The following imports are assumed to be available in a MaxText environment.
# They are stubbed here for completeness.
# from maxtext.layers.bbox_utils import center_to_corners_format
# from maxtext.layers.losses import (
#     box_iou,
#     dice_loss,
#     generalized_box_iou,
#     nested_tensor_from_tensor_list,
#     sigmoid_focal_loss,
# )

# --- Stubs for assumed available functions ---
def center_to_corners_format(x: Array) -> Array:
  """Placeholder for center_to_corners_format utility."""
  return x

def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
  """Placeholder for generalized_box_iou utility."""
  # Returns a matrix of shape [N, M]
  return jnp.zeros((boxes1.shape[0], boxes2.shape[0]))

def box_iou(boxes1: Array, boxes2: Array) -> Tuple[Array, Array]:
  """Placeholder for box_iou utility."""
  n, m = boxes1.shape[0], boxes2.shape[0]
  return jnp.zeros((n, m)), jnp.zeros((n, m))

def sigmoid_focal_loss(
    inputs: Array, targets: Array, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0
) -> Array:
  """Placeholder for sigmoid_focal_loss utility."""
  return jnp.array(0.0)

def dice_loss(inputs: Array, targets: Array, num_boxes: float) -> Array:
  """Placeholder for dice_loss utility."""
  return jnp.array(0.0)

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask
    def decompose(self):
        return self.tensors, self.mask

def nested_tensor_from_tensor_list(tensor_list: List[Array]) -> NestedTensor:
    """Placeholder for nested_tensor_from_tensor_list utility."""
    # This is a complex function. A simple mock is provided.
    max_shape = tuple(max(s) for s in zip(*[t.shape for t in tensor_list]))
    padded_tensors = jnp.stack([jnp.pad(t, [(0, max_s - s) for s, max_s in zip(t.shape, max_shape)]) for t in tensor_list])
    mask = jnp.ones(padded_tensors.shape, dtype=jnp.bool_)
    return NestedTensor(padded_tensors, mask)
# --- End Stubs ---


def _set_aux_loss(outputs_class: Array, outputs_coord: Array) -> List[Dict[str, Array]]:
  """
  Different for RT-DETR: not slicing the last element like in DETR one.
  This is a workaround to make torchscript happy, as torchscript
  doesn't support dictionary with non-homogeneous values, such
  as a dict having both a Tensor and a list.
  """
  return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


class RTDetrHungarianMatcher(nn.Module):
  """
  This class computes an assignment between the targets and the predictions of the network.
  For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
  predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
  un-matched (and thus treated as non-objects).
  """

  config: Any

  @nn.compact
  def __call__(self, outputs: Dict[str, Array], targets: List[Dict[str, Array]]) -> List[Tuple[Array, Array]]:
    """
    Performs the matching.
    """
    batch_size, num_queries = outputs["logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]
    # Also concat the target labels and boxes
    target_ids = jnp.concatenate([v["class_labels"] for v in targets])
    target_bbox = jnp.concatenate([v["boxes"] for v in targets])

    # Compute the classification cost.
    if self.config.use_focal_loss:
      out_prob = jax.nn.sigmoid(outputs["logits"].reshape(-1, outputs["logits"].shape[-1]))
      out_prob = out_prob[:, target_ids]
      neg_cost_class = (1 - self.config.matcher_alpha) * (out_prob**self.config.matcher_gamma) * (
          -jnp.log(1 - out_prob + 1e-8)
      )
      pos_cost_class = self.config.matcher_alpha * ((1 - out_prob) ** self.config.matcher_gamma) * (
          -jnp.log(out_prob + 1e-8)
      )
      class_cost = pos_cost_class - neg_cost_class
    else:
      out_prob = jax.nn.softmax(outputs["logits"].reshape(-1, outputs["logits"].shape[-1]), axis=-1)
      class_cost = -out_prob[:, target_ids]

    # Compute the L1 cost between boxes
    bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

    # Compute the giou cost between boxes
    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    # Compute the final cost matrix
    cost_matrix = (
        self.config.matcher_bbox_cost * bbox_cost
        + self.config.matcher_class_cost * class_cost
        + self.config.matcher_giou_cost * giou_cost
    )
    cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

    sizes = [len(v["boxes"]) for v in targets]
    # Move to CPU for scipy
    cost_matrix_cpu = jax.device_get(cost_matrix)
    indices = [
        linear_sum_assignment(c[i]) for i, c in enumerate(np.split(cost_matrix_cpu, np.cumsum(sizes[:-1]), axis=-1))
    ]

    return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]


class RTDetrLoss(nn.Module):
  """
  This class computes the losses for RTDetr.
  """

  config: Any

  def setup(self):
    self.matcher = RTDetrHungarianMatcher(config=self.config)
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

  def _get_source_permutation_idx(self, indices: List[Tuple[Array, Array]]) -> Tuple[Array, Array]:
    batch_idx = jnp.concatenate([jnp.full_like(src, i, dtype=jnp.int32) for i, (src, _) in enumerate(indices)])
    source_idx = jnp.concatenate([src for (src, _) in indices])
    return batch_idx, source_idx

  def loss_labels_vfl(
      self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float, log: bool = True
  ) -> Dict[str, Array]:
    idx = self._get_source_permutation_idx(indices)

    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)
    ious, _ = box_iou(center_to_corners_format(jax.lax.stop_gradient(src_boxes)), center_to_corners_format(target_boxes))
    ious = jnp.diag(ious)

    src_logits = outputs["logits"]
    target_classes_original = jnp.concatenate([t["class_labels"][i] for t, (_, i) in zip(targets, indices)])
    target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
    target_classes = target_classes.at[idx].set(target_classes_original)
    target = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

    target_score_original = jnp.zeros_like(target_classes, dtype=src_logits.dtype)
    target_score_original = target_score_original.at[idx].set(ious.astype(target_score_original.dtype))
    target_score = target_score_original[..., None] * target

    pred_score = jax.nn.sigmoid(jax.lax.stop_gradient(src_logits))
    weight = self.alpha * pred_score**self.gamma * (1 - target) + target_score

    # Manual implementation of binary_cross_entropy_with_logits with weights and reduction='none'
    bce_loss = jax.nn.sigmoid_binary_cross_entropy(src_logits, target_score)
    loss = weight * bce_loss
    loss = loss.mean(axis=1).sum() * src_logits.shape[1] / num_boxes
    return {"loss_vfl": loss}

  def loss_boxes(
      self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float
  ) -> Dict[str, Array]:
    idx = self._get_source_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

    loss_bbox = jnp.sum(jnp.abs(src_boxes - target_boxes)) / num_boxes
    loss_giou = (
        1 - jnp.diag(generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes)))
    ).sum() / num_boxes

    return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

  def get_loss(
      self, loss: str, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: float
  ) -> Dict[str, Array]:
    loss_map = {
        "vfl": self.loss_labels_vfl,
        "boxes": self.loss_boxes,
    }
    if loss not in loss_map:
      raise ValueError(f"Loss {loss} not supported")
    return loss_map[loss](outputs, targets, indices, num_boxes)

  @staticmethod
  def get_cdn_matched_indices(dn_meta: Dict, targets: List[Dict]) -> List[Tuple[Array, Array]]:
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
        dn_match_indices.append((jnp.zeros(0, dtype=jnp.int64), jnp.zeros(0, dtype=jnp.int64)))

    return dn_match_indices

  def __call__(self, outputs: Dict, targets: List[Dict]) -> Dict[str, Array]:
    outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}

    indices = self.matcher(outputs_without_aux, targets)

    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = jnp.asarray([num_boxes], dtype=jnp.float32)
    num_boxes = jnp.clip(num_boxes, a_min=1).item()

    losses = {}
    for loss in self.losses:
      l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
      l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
      losses.update(l_dict)

    if "auxiliary_outputs" in outputs:
      for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
        indices = self.matcher(auxiliary_outputs, targets)
        for loss in self.losses:
          l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
          l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
          l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
          losses.update(l_dict)

    if "dn_auxiliary_outputs" in outputs:
      if "denoising_meta_values" not in outputs:
        raise ValueError("The output must have the 'denoising_meta_values` key.")
      indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
      num_boxes_dn = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]

      for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
        for loss in self.losses:
          l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes_dn)
          l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
          l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
          losses.update(l_dict)

    return losses


def RTDetrForObjectDetectionLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
    pred_boxes: Array,
    config: Any,
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
      labels: A list of dictionaries, one per image, containing ground truth information.
      pred_boxes: Predicted bounding boxes from the model's final layer.
      config: Configuration object with model and loss hyperparameters.
      outputs_class: Class predictions from intermediate decoder layers.
      outputs_coord: Coordinate predictions from intermediate decoder layers.
      enc_topk_logits: Top-k class predictions from the encoder.
      enc_topk_bboxes: Top-k bounding box predictions from the encoder.
      denoising_meta_values: Metadata for denoising task.
      **kwargs: Additional keyword arguments.

  Returns:
      A tuple containing the total loss, a dictionary of individual losses, and auxiliary outputs.
  """
  criterion = RTDetrLoss(config=config)

  # Second: compute the losses, based on outputs and labels
  outputs_loss = {}
  outputs_loss["logits"] = logits
  outputs_loss["pred_boxes"] = pred_boxes
  auxiliary_outputs = []
  if config.auxiliary_loss:
    if denoising_meta_values is not None:
      dn_split_point = denoising_meta_values["dn_num_split"]
      dn_out_coord, outputs_coord = jnp.split(outputs_coord, [dn_split_point], axis=2)
      dn_out_class, outputs_class = jnp.split(outputs_class, [dn_split_point], axis=2)

    # In PyTorch, outputs_class is [batch, num_layers, ...], transpose(0,1) makes it [num_layers, batch, ...]
    # Assuming outputs_class is [batch, num_layers-1, num_queries, num_classes]
    auxiliary_outputs = _set_aux_loss(
        jnp.transpose(outputs_class, (1, 0, 2, 3)), jnp.transpose(outputs_coord, (1, 0, 2, 3))
    )
    outputs_loss["auxiliary_outputs"] = auxiliary_outputs
    outputs_loss["auxiliary_outputs"] += _set_aux_loss(jnp.expand_dims(enc_topk_logits, 0), jnp.expand_dims(enc_topk_bboxes, 0))
    if denoising_meta_values is not None:
      outputs_loss["dn_auxiliary_outputs"] = _set_aux_loss(
          jnp.transpose(dn_out_class, (1, 0, 2, 3)), jnp.transpose(dn_out_coord, (1, 0, 2, 3))
      )
      outputs_loss["denoising_meta_values"] = denoising_meta_values

  loss_dict = criterion(outputs_loss, labels)

  loss = sum(loss for loss in loss_dict.values())
  return loss, loss_dict, auxiliary_outputs

from typing import Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import Array

# From generated_code.Qwen3MoeForCausalLM.bbox_utils import RTDetrLoss
from generated_code.Qwen3MoeForCausalLM.bbox_utils import RTDetrLoss
# From generated_code.Qwen3MoeForCausalLM.matchers import RTDetrHungarianMatcher
from generated_code.Qwen3MoeForCausalLM.matchers import RTDetrHungarianMatcher
# From generated_code.Qwen3MoeForCausalLM.bbox_utils import bbox2distance
from generated_code.Qwen3MoeForCausalLM.bbox_utils import bbox2distance
# From generated_code.Qwen3MoeForCausalLM.bbox_utils import center_to_corners_format
from generated_code.Qwen3MoeForCausalLM.bbox_utils import center_to_corners_format
# From generated_code.Qwen3MoeForCausalLM.bbox_utils import box_iou
from generated_code.Qwen3MoeForCausalLM.bbox_utils import box_iou


class DFineLoss(RTDetrLoss):
    """
    This class computes the losses for D-FINE. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).

    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        weight_dict (`Dict`):
            Dictionary relating each loss with its weights. These losses are configured in DFineConf as
            `weight_loss_vfl`, `weight_loss_bbox`, `weight_loss_giou`, `weight_loss_fgl`, `weight_loss_ddf`
        losses (`list[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
        alpha (`float`):
            Parameter alpha used to compute the focal loss.
        gamma (`float`):
            Parameter gamma used to compute the focal loss.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
    """

    def __init__(self, config):
        super().__init__(config)

        self.matcher = RTDetrHungarianMatcher(config)
        self.max_num_bins = config.max_num_bins
        self.weight_dict = {
            "loss_vfl": config.weight_loss_vfl,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
            "loss_fgl": config.weight_loss_fgl,
            "loss_ddf": config.weight_loss_ddf,
        }
        self.losses = ["vfl", "boxes", "local"]
        self.reg_scale = config.reg_scale
        self.up = jnp.array([config.up])

    def unimodal_distribution_focal_loss(
        self,
        pred: Array,
        label: Array,
        weight_right: Array,
        weight_left: Array,
        weight: Optional[Array] = None,
        reduction: str = "sum",
        avg_factor: Optional[float] = None,
    ) -> Array:
        dis_left = label.astype(jnp.int32)
        dis_right = dis_left + 1

        loss = optax.softmax_cross_entropy_with_integer_labels(
            pred, dis_left
        ) * weight_left.reshape(-1) + optax.softmax_cross_entropy_with_integer_labels(
            pred, dis_right
        ) * weight_right.reshape(
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
        self, outputs: Dict, targets: List[Dict], indices: Tuple, num_boxes: int, T: int = 5
    ) -> Dict[str, Array]:
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_source_permutation_idx(indices)
            target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

            pred_corners_idx = outputs["pred_corners"][idx].reshape(-1, (self.max_num_bins + 1))
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
                jnp.tile(ious[:, None, None], (1, 1, 4)).reshape(-1)
            )

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners_idx,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            pred_corners = outputs["pred_corners"].reshape(-1, (self.max_num_bins + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (self.max_num_bins + 1))
            if jnp.array_equal(pred_corners, target_corners):
                losses["loss_ddf"] = pred_corners.sum() * 0
            else:
                weight_targets_local = jax.nn.sigmoid(outputs["teacher_logits"]).max(axis=-1)
                mask = jnp.zeros_like(weight_targets_local, dtype=jnp.bool_)
                mask = mask.at[idx].set(True)
                mask = jnp.tile(mask[:, :, None], (1, 1, 4)).reshape(-1)

                weight_targets_local = weight_targets_local.at[idx].set(
                    ious.reshape(weight_targets_local[idx].shape).astype(weight_targets_local.dtype)
                )
                weight_targets_local = jax.lax.stop_gradient(
                    jnp.tile(weight_targets_local[:, :, None], (1, 1, 4)).reshape(-1)
                )

                log_pred_corners = jax.nn.log_softmax(pred_corners / T, axis=1)
                softmax_target_corners = jax.nn.softmax(jax.lax.stop_gradient(target_corners) / T, axis=1)
                kl_div_loss = optax.kl_divergence(log_pred_corners, softmax_target_corners)

                loss_match_local = weight_targets_local * (T**2) * kl_div_loss

                batch_scale = 1 / outputs["pred_boxes"].shape[0]
                num_pos, num_neg = (
                    (mask.sum() * batch_scale) ** 0.5,
                    ((~mask).sum() * batch_scale) ** 0.5,
                )
                loss_match_local1 = jnp.mean(loss_match_local[mask]) if jnp.any(mask) else 0.0
                loss_match_local2 = jnp.mean(loss_match_local[~mask]) if jnp.any(~mask) else 0.0
                losses["loss_ddf"] = (loss_match_local1 * num_pos + loss_match_local2 * num_neg) / (
                    num_pos + num_neg
                )

        return losses

    def get_loss(
        self, loss: str, outputs: Dict, targets: List[Dict], indices: Tuple, num_boxes: int
    ) -> Dict[str, Array]:
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

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

# From MaxText matched dependencies
from generated_code.Qwen3MoeForCausalLM.losses import (  # generated_code.Qwen3MoeForCausalLM.losses.DFineLoss
    DFineLoss,
)
from generated_code.Qwen3MoeForCausalLM.losses import (  # generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss
    _set_aux_loss,
)
from generated_code.Qwen3MoeForCausalLM.losses import (  # generated_code.Qwen3MoeForCausalLM.losses._set_aux_loss2
    _set_aux_loss2,
)


def DFineForObjectDetectionLoss(
    logits: Array,
    labels: List[Dict[str, Array]],
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
    Computes the loss for D-FINE object detection.
    """
    criterion = DFineLoss(config)
    # Second: compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = jnp.clip(pred_boxes, a_min=0, a_max=1)
    auxiliary_outputs = None
    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            dn_out_coord, outputs_coord = jnp.split(
                jnp.clip(outputs_coord, a_min=0, a_max=1), [denoising_meta_values["dn_num_split"]], axis=2
            )
            dn_out_class, outputs_class = jnp.split(outputs_class, [denoising_meta_values["dn_num_split"]], axis=2)
            dn_out_corners, out_corners = jnp.split(
                predicted_corners, [denoising_meta_values["dn_num_split"]], axis=2
            )
            dn_out_refs, out_refs = jnp.split(
                initial_reference_points, [denoising_meta_values["dn_num_split"]], axis=2
            )

            # Note: Original PyTorch code had a `.transpose(0, 1)` here, which seems incorrect for creating
            # auxiliary outputs per layer. The logic is implemented to iterate over the layer dimension (axis 0),
            # which is the standard for DETR-like models.
            auxiliary_outputs = _set_aux_loss2(
                outputs_class[:-1],
                outputs_coord[:-1],
                out_corners[:-1],
                out_refs[:-1],
                teacher_corners=out_corners[-1],
                teacher_logits=outputs_class[-1],
            )

            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            outputs_loss["auxiliary_outputs"].extend(
                _set_aux_loss([enc_topk_logits], [jnp.clip(enc_topk_bboxes, a_min=0, a_max=1)])
            )

            # Note: Original PyTorch code had a `.transpose(0, 1)` here as well.
            dn_auxiliary_outputs = _set_aux_loss2(
                dn_out_class,
                dn_out_coord,
                dn_out_corners,
                dn_out_refs,
                teacher_corners=dn_out_corners[-1],
                teacher_logits=dn_out_class[-1],
            )
            outputs_loss["dn_auxiliary_outputs"] = dn_auxiliary_outputs
            outputs_loss["denoising_meta_values"] = denoising_meta_values

    loss_dict = criterion(outputs_loss, labels)

    loss = sum(loss_dict.values())
    return loss, loss_dict, auxiliary_outputs

# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForCausalLMLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForMaskedLMLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForQuestionAnsweringLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForSequenceClassificationLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForTokenClassification
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForSegmentationLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.ForObjectDetectionLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.DeformableDetrForObjectDetectionLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.DeformableDetrForSegmentationLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.GroundingDinoForObjectDetectionLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.RTDetrForObjectDetectionLoss
# Reused from generated_code.Qwen3MoeForCausalLM.losses.DFineForObjectDetectionLoss
from .losses import (
    DFineForObjectDetectionLoss,
    DeformableDetrForObjectDetectionLoss,
    DeformableDetrForSegmentationLoss,
    ForCausalLMLoss,
    ForMaskedLMLoss,
    ForObjectDetectionLoss,
    ForQuestionAnsweringLoss,
    ForSegmentationLoss,
    ForSequenceClassificationLoss,
    ForTokenClassification,
    GroundingDinoForObjectDetectionLoss,
    RTDetrForObjectDetectionLoss,
)

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
