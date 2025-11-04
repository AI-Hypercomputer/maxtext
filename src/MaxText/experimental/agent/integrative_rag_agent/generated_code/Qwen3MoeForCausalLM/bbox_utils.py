
import jax.numpy as jnp


def _center_to_corners_format_jax(bboxes_center: jnp.ndarray) -> jnp.ndarray:
  center_x, center_y, width, height = bboxes_center.T
  bboxes_corners = jnp.stack(
      # top left x, top left y, bottom right x, bottom right y
      [
          center_x - 0.5 * width,
          center_y - 0.5 * height,
          center_x + 0.5 * width,
          center_y + 0.5 * height,
      ],
      axis=-1,
  )
  return bboxes_corners

import jax.numpy as jnp
from jax import Array


def _center_to_corners_format_jax(bboxes_center: Array) -> Array:
    center_x, center_y, width, height = jnp.moveaxis(bboxes_center, -1, 0)
    bbox_corners = jnp.stack(
        # top left x, top left y, bottom right x, bottom right y
        [
            (center_x - 0.5 * width),
            (center_y - 0.5 * height),
            (center_x + 0.5 * width),
            (center_y + 0.5 * height),
        ],
        axis=-1,
    )
    return bbox_corners

# Copyright 2024 Google LLC
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

"""Image processing utilities."""

from jax import Array
import jax.numpy as jnp


# Reused from generated_code.Qwen3MoeForCausalLM.bbox_utils._center_to_corners_format_jax
def center_to_corners_format(bboxes_center: Array) -> Array:
  """Converts bounding boxes from center format to corners format.

  center format: contains the coordinate for the center of the box and its
    width, height dimensions (center_x, center_y, width, height)
  corners format: contains the coordinates for the top-left and bottom-right
    corners of the box (top_left_x, top_left_y, bottom_right_x,
    bottom_right_y)

  Args:
    bboxes_center: Bounding boxes in center format.

  Returns:
    Bounding boxes in corners format.
  """
  center_x, center_y, width, height = jnp.moveaxis(bboxes_center, -1, 0)
  bboxes_corners = jnp.stack(
      [
          center_x - 0.5 * width,
          center_y - 0.5 * height,
          center_x + 0.5 * width,
          center_y + 0.5 * height,
      ],
      axis=-1,
  )
  return bboxes_corners

import jax
import jax.numpy as jnp


def weighting_function(max_num_bins: int, up: jax.Array, reg_scale: int) -> jax.Array:
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        max_num_bins (int): Max number of the discrete bins.
        up (jax.Array): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(max_num_bins/2)=0
                           and steeper weights at both ends.
    Returns:
        jax.Array: Sequence of Weighting Function.
    """
    upper_bound1 = jnp.abs(up[0]) * jnp.abs(reg_scale)
    upper_bound2 = jnp.abs(up[0]) * jnp.abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))

    # Vectorized creation of left and right side values
    left_indices = jnp.arange(max_num_bins // 2 - 1, 0, -1, dtype=jnp.float32)
    left_values = -(step**left_indices) + 1

    right_indices = jnp.arange(1, max_num_bins // 2, dtype=jnp.float32)
    right_values = step**right_indices - 1

    # The original PyTorch code builds a list of mixed scalars and tensors,
    # then unsqueezes scalars to 1D tensors before concatenating.
    # In JAX, we can directly create 1D arrays and concatenate them.
    values = jnp.concatenate(
        [
            -jnp.expand_dims(upper_bound2, 0),
            left_values,
            jnp.zeros(1, dtype=up.dtype),
            right_values,
            jnp.expand_dims(upper_bound2, 0),
        ]
    )
    return values

from typing import Tuple
from jax import Array
import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.bbox_utils.weighting_function
from generated_code.Qwen3MoeForCausalLM.bbox_utils import weighting_function


def translate_gt(gt: Array, max_num_bins: int, reg_scale: int, up: Array) -> Tuple[Array, Array, Array]:
  """
  Decodes bounding box ground truth (GT) values into distribution-based GT representations.

  This function maps continuous GT values into discrete distribution bins, which can be used
  for regression tasks in object detection models. It calculates the indices of the closest
  bins to each GT value and assigns interpolation weights to these bins based on their proximity
  to the GT value.

  Args:
      gt (Array): Ground truth bounding box values, shape (N, ).
      max_num_bins (int): Maximum number of discrete bins for the distribution.
      reg_scale (float): Controls the curvature of the Weighting Function.
      up (Array): Controls the upper bounds of the Weighting Function.

  Returns:
      tuple[Array, Array, Array]:
          - indices (Array): Index of the left bin closest to each GT value, shape (N, ).
          - weight_right (Array): Weight assigned to the right bin, shape (N, ).
          - weight_left (Array): Weight assigned to the left bin, shape (N, ).
  """
  gt = gt.reshape(-1)
  function_values = weighting_function(max_num_bins, up, reg_scale)

  # Find the closest left-side indices for each value
  diffs = function_values[jnp.newaxis, :] - gt[:, jnp.newaxis]
  mask = diffs <= 0
  closest_left_indices = jnp.sum(mask, axis=1) - 1
  indices = closest_left_indices.astype(jnp.float32)

  # Define masks for different index ranges.
  # The positive invalid mask is corrected to `max_num_bins - 1` to prevent an
  # out-of-bounds access that exists in the original PyTorch code.
  invalid_idx_mask_neg = indices < 0
  invalid_idx_mask_pos_for_indices = indices >= max_num_bins
  invalid_idx_mask_pos_for_weights = indices >= max_num_bins - 1
  valid_idx_mask = ~invalid_idx_mask_neg & ~invalid_idx_mask_pos_for_weights

  # Vectorized calculation for interpolation weights.
  # We clip indices to prevent out-of-bounds access during gathering.
  safe_indices = jnp.clip(indices, 0, max_num_bins - 2).astype(jnp.int32)

  left_values = function_values[safe_indices]
  right_values = function_values[safe_indices + 1]

  left_diffs = jnp.abs(gt - left_values)
  right_diffs = jnp.abs(right_values - gt)

  denominator = left_diffs + right_diffs
  # If denominator is 0, it means gt is on a bin value, and left_diffs is also 0.
  # The result should be 0. jnp.where handles this safely.
  interp_weight_right = jnp.where(denominator > 0, left_diffs / denominator, 0.0)

  # Combine results for weight_right using the masks
  weight_right = jnp.where(valid_idx_mask, interp_weight_right, 0.0)
  weight_right = jnp.where(invalid_idx_mask_pos_for_weights, 1.0, weight_right)

  # Calculate weight_left based on weight_right
  weight_left = 1.0 - weight_right

  # Set final indices based on original logic
  final_indices = jnp.where(invalid_idx_mask_neg, 0.0, indices)
  final_indices = jnp.where(invalid_idx_mask_pos_for_indices, max_num_bins - 0.1, final_indices)

  return final_indices, weight_right, weight_left

import jax
import jax.numpy as jnp
from jax import Array

# No matching module found for `weighting_function` in JAX_MODULES_DICT.
# This is a new JAX implementation based on the original PyTorch code.
def weighting_function(max_num_bins: int, up: Array, reg_scale: int) -> Array:
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        max_num_bins (int): Max number of the discrete bins.
        up (Array): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(max_num_bins/2)=0
                           and steeper weights at both ends.
    Returns:
        Array: Sequence of Weighting Function.
    """
    reg_scale = jnp.abs(reg_scale)
    upper_bound1 = jnp.abs(up[0]) * reg_scale
    upper_bound2 = jnp.abs(up[0]) * reg_scale * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))

    left_indices = jnp.arange(max_num_bins // 2 - 1, 0, -1, dtype=jnp.float32)
    left_values = -jnp.power(step, left_indices) + 1

    right_indices = jnp.arange(1, max_num_bins // 2, dtype=jnp.float32)
    right_values = jnp.power(step, right_indices) - 1

    values = jnp.concatenate(
        [
            jnp.array([-upper_bound2]),
            left_values,
            jnp.zeros_like(up[0][None]),
            right_values,
            jnp.array([upper_bound2]),
        ]
    )
    return values


# No matching module found for `translate_gt` in JAX_MODULES_DICT.
# This is a new JAX implementation based on the original PyTorch code.
def translate_gt(gt: Array, max_num_bins: int, reg_scale: int, up: Array) -> tuple[Array, Array, Array]:
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Array): Ground truth bounding box values, shape (N, ).
        max_num_bins (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Array): Controls the upper bounds of the Weighting Function.

    Returns:
        tuple[Array, Array, Array]:
            - indices (Array): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Array): Weight assigned to the right bin, shape (N, ).
            - weight_left (Array): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(max_num_bins, up, reg_scale)

    # Find the closest left-side indices for each value using searchsorted for efficiency
    indices = jnp.searchsorted(function_values, gt, side="right") - 1
    indices = indices.astype(jnp.float32)

    # Define masks for different conditions
    invalid_idx_mask_neg = indices < 0
    # The right value is at index `i+1`, so the last valid index is `max_num_bins - 2`
    invalid_idx_mask_pos = indices >= (max_num_bins - 1)
    valid_idx_mask = ~invalid_idx_mask_neg & ~invalid_idx_mask_pos

    # Clip indices to a safe range to perform gather operations without error
    safe_indices = jnp.clip(indices, 0, max_num_bins - 2).astype(jnp.int32)

    # Obtain distances for all elements; we'll mask the results later
    left_values = function_values[safe_indices]
    right_values = function_values[safe_indices + 1]

    left_diffs = jnp.abs(gt - left_values)
    right_diffs = jnp.abs(right_values - gt)

    # Calculate weights for the valid case, adding an epsilon for numerical stability
    denominator = left_diffs + right_diffs
    # Handle cases where gt is exactly on a bin value, preventing division by zero
    safe_denominator = jnp.where(denominator == 0, 1.0, denominator)
    all_weight_right = left_diffs / safe_denominator

    # Use jnp.where to construct the final weights based on the masks
    weight_right = jnp.where(
        valid_idx_mask,
        all_weight_right,
        jnp.where(invalid_idx_mask_pos, 1.0, 0.0),  # if pos, 1.0; else (must be neg), 0.0
    )
    weight_left = 1.0 - weight_right

    # Use jnp.where to construct the final indices based on the masks
    # This logic matches the original PyTorch implementation's handling of out-of-bounds values.
    final_indices = jnp.where(
        invalid_idx_mask_neg,
        0.0,
        jnp.where(invalid_idx_mask_pos, max_num_bins - 0.1, indices),
    )

    return final_indices, weight_right, weight_left


# No matching module found for `bbox2distance` in JAX_MODULES_DICT.
# This is a new JAX implementation based on the original PyTorch code.
def bbox2distance(points: Array, bbox: Array, max_num_bins: float, reg_scale: float, up: Array, eps: float = 0.1) -> tuple[Array, Array, Array]:
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Array): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Array): (n, 4) bounding boxes in "xyxy" format.
        max_num_bins (float): Maximum bin value.
        reg_scale (float): Controlling curvarture of W(n).
        up (Array): Controlling upper bounds of W(n).
        eps (float): Small value to ensure target < max_num_bins.

    Returns:
        tuple[Array, Array, Array]: Decoded distances and weights.
    """

    reg_scale = jnp.abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = jnp.stack([left, top, right, bottom], axis=-1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, int(max_num_bins), reg_scale, up)
    if max_num_bins is not None:
        four_lens = jnp.clip(four_lens, a_min=0, a_max=max_num_bins - eps)
    return (
        jax.lax.stop_gradient(four_lens.reshape(-1)),
        jax.lax.stop_gradient(weight_right),
        jax.lax.stop_gradient(weight_left),
    )

import jax.numpy as jnp
from jax import Array


def _upcast(t: Array) -> Array:
  # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
  else:
    return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)


def box_area(boxes: Array) -> Array:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`jax.Array` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `jax.Array`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

from typing import Tuple

import jax.numpy as jnp
from jax import Array


def box_iou(boxes1: Array, boxes2: Array) -> Tuple[Array, Array]:
  """
  Computes the intersection over union (IoU) between two sets of boxes.
  Also returns the union. Modified from torchvision to also return the union.

  Args:
    boxes1 (`Array` of shape `(N, 4)`): Boxes in (x1, y1, x2, y2) format.
    boxes2 (`Array` of shape `(M, 4)`): Boxes in (x1, y1, x2, y2) format.

  Returns:
    `Tuple[Array, Array]`: A tuple containing the IoU matrix of shape `(N, M)`
    and the union matrix of shape `(N, M)`.
  """
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  left_top = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
  right_bottom = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

  width_height = jnp.clip(right_bottom - left_top, a_min=0)  # [N,M,2]
  inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

  union = area1[:, None] + area2 - inter

  iou = inter / union
  return iou, union

import jax.numpy as jnp
from jax import Array

# box_iou is a dependency from the same file in the original PyTorch code.
# Assuming it is converted and available in the same JAX module.
from . import box_iou


def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
  """
  Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

  Returns:
      `jnp.ndarray`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
  """
  # degenerate boxes gives inf / nan results
  # so do an early check
  if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
    raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
  if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
    raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
  iou, union = box_iou(boxes1, boxes2)

  top_left = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
  bottom_right = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

  width_height = jnp.maximum(bottom_right - top_left, 0)  # [N,M,2]
  area = width_height[:, :, 0] * width_height[:, :, 1]

  return iou - (area - union) / (area + 1e-6)

from __future__ import annotations

import flax.struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional, Tuple

# The following helper functions are new JAX implementations equivalent to
# the PyTorch versions in `transformers.models.detr.modeling_detr`.

def center_to_corners_format(bboxes_center: jax.Array) -> jax.Array:
    """
    Converts bounding boxes from (center_x, center_y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        bboxes_center (`jax.Array` of shape `(batch_size, num_queries, 4)`):
            Bounding boxes in (center_x, center_y, width, height) format.

    Returns:
        `jax.Array` of shape `(batch_size, num_queries, 4)`: Bounding boxes in (x1, y1, x2, y2) format.
    """
    center_x, center_y, width, height = jnp.moveaxis(bboxes_center, -1, 0)
    x1 = center_x - 0.5 * width
    y1 = center_y - 0.5 * height
    x2 = center_x + 0.5 * width
    y2 = center_y + 0.5 * height
    return jnp.stack([x1, y1, x2, y2], axis=-1)


def box_area(boxes: jax.Array) -> jax.Array:
    """
    Computes the area of a set of bounding boxes, which are specified by their (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`jax.Array` of shape `(N, 4)`):
            Boxes for which the area will be computed. They are specified as (x1, y1, x2, y2).

    Returns:
        `jax.Array` of shape `(N,)`: The area for each box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: jax.Array, boxes2: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the intersection over union (IoU) of two sets of bounding boxes.

    Args:
        boxes1 (`jax.Array` of shape `(N, 4)`):
            Boxes for which the area will be computed. They are specified as (x1, y1, x2, y2).
        boxes2 (`jax.Array` of shape `(M, 4)`):
            Boxes for which the area will be computed. They are specified as (x1, y1, x2, y2).

    Returns:
        `Tuple[jax.Array, jax.Array]`: A tuple containing the IoU matrix of shape `(N, M)` and the union matrix of shape `(N, M)`.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clip(min=0)
    inter = width_height[:, :, 0] * width_height[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    return iou, union


def generalized_box_iou(boxes1: jax.Array, boxes2: jax.Array) -> jax.Array:
    """
    Computes the generalized intersection over union (GIoU) of two sets of bounding boxes.

    Args:
        boxes1 (`jax.Array` of shape `(N, 4)`):
            Boxes for which the area will be computed. They are specified as (x1, y1, x2, y2).
        boxes2 (`jax.Array` of shape `(M, 4)`):
            Boxes for which the area will be computed. They are specified as (x1, y1, x2, y2).

    Returns:
        `jax.Array`: A matrix of shape `(N, M)` containing the GIoU values.
    """
    iou, union = box_iou(boxes1, boxes2)

    top_left = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clip(min=0)
    enclosing_area = width_height[:, :, 0] * width_height[:, :, 1]

    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = jax.nn.sigmoid(inputs)
    ce_loss = optax.sigmoid_binary_cross_entropy(inputs, targets)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs, targets, num_boxes):
    inputs = jax.nn.sigmoid(inputs)
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


@flax.struct.dataclass
class NestedTensor:
    tensors: jax.Array
    mask: Optional[jax.Array]

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[jax.Array]):
    if tensor_list[0].ndim == 3:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = (len(tensor_list),) + max_size
        dtype = tensor_list[0].dtype
        
        padded_tensors = jnp.zeros(batch_shape, dtype=dtype)
        mask = jnp.ones(batch_shape, dtype=jnp.bool_)

        for i, tensor in enumerate(tensor_list):
            padded_tensors = padded_tensors.at[i, : tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].set(tensor)
            mask = mask.at[i, : tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].set(False)
    else:
        raise ValueError("Only 3D tensors are supported")
    return NestedTensor(tensors=padded_tensors, mask=mask)


class RTDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    """
    config: Any

    @nn.compact
    def __call__(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[jax.Array, jax.Array]]:
        batch_size, num_queries = outputs["logits"].shape[:2]

        out_prob = outputs["logits"].reshape(batch_size * num_queries, -1)
        out_bbox = outputs["pred_boxes"].reshape(batch_size * num_queries, -1)

        target_ids = jnp.concatenate([v["class_labels"] for v in targets])
        target_bbox = jnp.concatenate([v["boxes"] for v in targets])

        if self.config.use_focal_loss:
            out_prob_sigmoid = jax.nn.sigmoid(out_prob)
            # The following is the cost computation for focal loss
            # We select the out_prob for the target classes
            out_prob = out_prob_sigmoid[:, target_ids]
            neg_cost_class = (1 - self.config.matcher_alpha) * (out_prob**self.config.matcher_gamma) * (
                -jnp.log(1 - out_prob + 1e-8)
            )
            pos_cost_class = self.config.matcher_alpha * ((1 - out_prob) ** self.config.matcher_gamma) * (
                -jnp.log(out_prob + 1e-8)
            )
            class_cost = pos_cost_class - neg_cost_class
        else:
            out_prob_softmax = jax.nn.softmax(out_prob, axis=-1)
            class_cost = -out_prob_softmax[:, target_ids]

        bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

        giou_cost = -generalized_box_iou(
            center_to_corners_format(out_bbox), center_to_corners_format(target_bbox)
        )

        cost_matrix = (
            self.config.matcher_bbox_cost * bbox_cost
            + self.config.matcher_class_cost * class_cost
            + self.config.matcher_giou_cost * giou_cost
        )
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

        cost_matrix_np = jax.device_get(cost_matrix)
        sizes = [len(v["boxes"]) for v in targets]
        
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(np.split(cost_matrix_np, np.cumsum(sizes)[:-1], axis=-1))
        ]

        return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]


class RTDetrLoss(nn.Module):
    """
    This class computes the losses for RTDetr. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).
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
        self.empty_weight = jnp.ones(self.config.num_labels + 1).at[-1].set(self.eos_coef)
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
        target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
        target_classes = target_classes.at[idx].set(target_classes_original)
        target = jax.nn.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_original = jnp.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_original = target_score_original.at[idx].set(ious.astype(target_score_original.dtype))
        target_score = jnp.expand_dims(target_score_original, -1) * target

        pred_score = jax.nn.sigmoid(jax.lax.stop_gradient(src_logits))
        weight = self.alpha * jnp.power(pred_score, self.gamma) * (1 - target) + target_score

        loss = weight * optax.sigmoid_binary_cross_entropy(src_logits, target_score)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)"""
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")

        src_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_original = jnp.concatenate([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = jnp.full(src_logits.shape[:2], self.num_classes, dtype=jnp.int64)
        target_classes = target_classes.at[idx].set(target_classes_original)

        loss_ce = optax.softmax_cross_entropy_with_integer_labels(
            jnp.transpose(src_logits, (0, 2, 1)), target_classes
        )
        # In PyTorch, a class_weight can be passed. We can replicate this by weighting the loss.
        # Assuming self.class_weight is available and has shape (num_classes + 1,)
        # This part is not fully convertible without knowing self.class_weight
        # For now, we compute unweighted loss.
        losses = {"loss_ce": loss_ce.mean()}
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error"""
        logits = outputs["logits"]
        target_lengths = jnp.array([len(v["class_labels"]) for v in targets])
        card_pred = (jnp.argmax(logits, axis=-1) != logits.shape[-1] - 1).sum(1)
        card_err = jnp.mean(jnp.abs(card_pred.astype(jnp.float32) - target_lengths.astype(jnp.float32)))
        losses = {"cardinality_error": jax.lax.stop_gradient(card_err)}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes"""
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = jnp.concatenate([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        losses = {}

        loss_bbox = jnp.sum(jnp.abs(src_boxes - target_boxes))
        losses["loss_bbox"] = loss_bbox / num_boxes

        loss_giou = 1 - jnp.diag(
            generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks"""
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = jax.image.resize(
            jnp.expand_dims(source_masks, -1), 
            (source_masks.shape[0],) + target_masks.shape[-2:] + (1,), 
            method="bilinear"
        )
        source_masks = jnp.squeeze(source_masks, -1).reshape(source_masks.shape[0], -1)

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
        loss = optax.sigmoid_binary_cross_entropy(src_logits, target * 1.0)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_bce": loss}

    def _get_source_permutation_idx(self, indices):
        batch_idx = jnp.concatenate([jnp.full_like(src, i, dtype=jnp.int32) for i, (src, _) in enumerate(indices)])
        source_idx = jnp.concatenate([src for (src, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        batch_idx = jnp.concatenate([jnp.full_like(tgt, i, dtype=jnp.int32) for i, (_, tgt) in enumerate(indices)])
        target_idx = jnp.concatenate([tgt for (_, tgt) in indices])
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
        loss = sigmoid_focal_loss(src_logits, target, num_boxes, self.alpha, self.gamma)
        # The reduction in the original is different from the helper
        # Re-implementing element-wise focal loss here
        prob = jax.nn.sigmoid(src_logits)
        ce_loss = optax.sigmoid_binary_cross_entropy(src_logits, target)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        
        loss = focal_loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_focal": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
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
    def get_cdn_matched_indices(dn_meta, targets):
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

    def __call__(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = jnp.clip(jnp.array([num_boxes], dtype=jnp.float32), a_min=1).item()

        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_auxiliary_outputs" in outputs:
            if "denoising_meta_values" not in outputs:
                raise ValueError("Missing 'denoising_meta_values' in outputs.")
            indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
            num_boxes = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]

            for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
