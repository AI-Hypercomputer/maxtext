
import jax.numpy as jnp
from jax import Array


def _center_to_corners_format_jax(bboxes_center: Array) -> Array:
  """Converts bounding boxes from center format to corners format using JAX.

  Args:
    bboxes_center: A JAX array of bounding boxes in the format
      (center_x, center_y, width, height).

  Returns:
    A JAX array of bounding boxes in the format
      (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
  """
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

from typing import Tuple
import jax.numpy as jnp
from jaxtyping import Array, Float


# Note: The following helper functions `_upcast` and `box_area` are also converted
# from the original file as they are direct dependencies of `box_iou` and no
# equivalent JAX module was found.

def _upcast(t: Array) -> Array:
  """Protects from numerical overflows by upcasting to a higher precision type."""
  if jnp.issubdtype(t.dtype, jnp.floating):
    return t if t.dtype in (jnp.float32, jnp.float64) else t.astype(jnp.float32)
  else:
    return t if t.dtype in (jnp.int32, jnp.int64) else t.astype(jnp.int32)


def box_area(boxes: Float[Array, "... 4"]) -> Float[Array, "..."]:
  """
  Computes the area of a set of bounding boxes.

  Args:
    boxes: Boxes for which the area will be computed. They are expected to be in
      (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.

  Returns:
    A tensor containing the area for each box.
  """
  boxes = _upcast(boxes)
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(
    boxes1: Float[Array, "N 4"], boxes2: Float[Array, "M 4"]
) -> Tuple[Float[Array, "N M"], Float[Array, "N M"]]:
  """
  Computes the intersection over union (IoU) of two sets of bounding boxes.

  Also returns the union for use in other metrics like GIoU.

  Args:
      boxes1: A tensor of shape (N, 4) representing N bounding boxes in
          (x1, y1, x2, y2) format.
      boxes2: A tensor of shape (M, 4) representing M bounding boxes in
          (x1, y1, x2, y2) format.

  Returns:
      A tuple of (iou, union):
      - iou: A tensor of shape (N, M) representing the pairwise IoU values.
      - union: A tensor of shape (N, M) representing the pairwise union areas.
  """
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  left_top = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
  right_bottom = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

  width_height = jnp.maximum(0, right_bottom - left_top)  # [N,M,2]
  inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

  union = area1[:, None] + area2 - inter

  iou = inter / union
  return iou, union

import jax.numpy as jnp
from jax import Array

# The following functions are utilities for bounding box operations,
# converted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py

def box_area(boxes: Array) -> Array:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`Array` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `Array`: a tensor containing the area for each box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1: Array, boxes2: Array) -> tuple[Array, Array]:
    """
    Computes the pairwise intersection over union (IoU) and union area for two sets of bounding boxes.

    Args:
        boxes1 (`Array` of shape `(N, 4)`): A set of N bounding boxes.
        boxes2 (`Array` of shape `(M, 4)`): A set of M bounding boxes.

    Returns:
        A tuple of two `Array`s:
        - The first `Array` is the pairwise IoU matrix of shape `(N, M)`.
        - The second `Array` is the pairwise union area matrix of shape `(N, M)`.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = jnp.maximum(0, right_bottom - left_top)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: Array, boxes2: Array) -> Array:
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Args:
        boxes1 (`Array` of shape `(N, 4)`): A set of N bounding boxes.
        boxes2 (`Array` of shape `(M, 4)`): A set of M bounding boxes.

    Returns:
        `Array`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not jnp.all(boxes1[:, 2:] >= boxes1[:, :2]):
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not jnp.all(boxes2[:, 2:] >= boxes2[:, :2]):
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = jnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = jnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = jnp.maximum(0, bottom_right - top_left)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area
