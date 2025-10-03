
from __future__ import annotations

from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment


# Note: The following helper functions and base class are implemented here as they
# were dependencies in the original file but do not have equivalents in the provided JAX modules.
# In a real-world scenario, these would likely be in their own utility modules.


def _center_to_corners_format_jax(x: jnp.ndarray) -> jnp.ndarray:
  """
  Converts bounding box coordinates from (center_x, center_y, width, height)
  to (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
  """
  center_x, center_y, width, height = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
  top_left_x = center_x - 0.5 * width
  top_left_y = center_y - 0.5 * height
  bottom_right_x = center_x + 0.5 * width
  bottom_right_y = center_y + 0.5 * height
  return jnp.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)


def _box_area(boxes: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the area of a set of bounding boxes.
  """
  return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def _generalized_box_iou_jax(boxes1: jnp.ndarray, boxes2: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the generalized box IoU between two sets of boxes.
  """
  # Intersection area
  top_left = jnp.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
  bottom_right = jnp.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
  wh = (bottom_right - top_left).clip(0)
  intersection = wh[..., 0] * wh[..., 1]

  # Union area
  area1 = _box_area(boxes1)
  area2 = _box_area(boxes2)
  union = area1[:, None] + area2[None, :] - intersection

  iou = intersection / (union + 1e-6)

  # Enclosing box area
  enclose_top_left = jnp.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
  enclose_bottom_right = jnp.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
  enclose_wh = (enclose_bottom_right - enclose_top_left).clip(0)
  enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

  giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
  return giou


class HungarianMatcher(nn.Module):
  """
  Base class for Hungarian Matcher. This is a simple container for cost coefficients.
  """

  class_cost: float
  bbox_cost: float
  giou_cost: float


class GroundingDinoHungarianMatcher(HungarianMatcher):
  """
  This class computes an assignment between the targets and the predictions of the network.
  """

  @nn.compact
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
    out_prob = jax.nn.sigmoid(
        outputs["logits"].reshape(-1, outputs["logits"].shape[-1])
    )  # [batch_size * num_queries, hidden_dim]
    out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]
    label_maps = outputs["label_maps"]

    # First take the label map for each class in each batch and then concatenate them
    concatenated_label_maps = jnp.concatenate(
        [label_map[target["class_labels"]] for label_map, target in zip(label_maps, targets)]
    )
    # Normalize label maps based on number of tokens per class
    label_maps_normalized = concatenated_label_maps / (concatenated_label_maps.sum(axis=-1, keepdims=True) + 1e-8)

    # Also concat the target labels and boxes
    target_bbox = jnp.concatenate([v["boxes"] for v in targets])

    # Compute the classification cost.
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-jnp.log(1 - out_prob + 1e-8))
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-jnp.log(out_prob + 1e-8))
    # Compute the classification cost by taking pos and neg cost in the appropriate index
    class_cost = (pos_cost_class - neg_cost_class) @ label_maps_normalized.T

    # Compute the L1 cost between boxes
    bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

    # Compute the giou cost between boxes
    giou_cost = -_generalized_box_iou_jax(
        _center_to_corners_format_jax(out_bbox), _center_to_corners_format_jax(target_bbox)
    )

    # Final cost matrix
    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)
    cost_matrix_np = np.array(cost_matrix)

    sizes = [len(v["boxes"]) for v in targets]
    split_indices = np.cumsum(sizes)[:-1]
    cost_matrix_splits = np.split(cost_matrix_np, split_indices, axis=-1)

    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix_splits)]
    return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]
