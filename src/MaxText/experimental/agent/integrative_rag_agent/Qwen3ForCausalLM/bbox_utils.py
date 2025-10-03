
import jax.numpy as jnp


def _center_to_corners_format_jax(bboxes_center: jnp.ndarray) -> jnp.ndarray:
  """Converts bounding boxes from center format to corners format for JAX arrays."""
  center_x, center_y, width, height = bboxes_center.T
  bboxes_corners = jnp.stack(
      # top left x, top left y, bottom right x, bottom right y
      [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
      axis=-1,
  )
  return bboxes_corners

import jax
import jax.numpy as jnp
from jax import Array

# The functionality of this function is already available in Qwen3ForCausalLM.bbox_utils._center_to_corners_format_jax.
# The implementation below is a robust JAX conversion of the PyTorch source.
def _center_to_corners_format_jax(bboxes_center: Array) -> Array:
  """Converts bounding box coordinates from center-based to corner-based format.

  Args:
    bboxes_center: A JAX array of shape `[..., 4]` representing bounding boxes
      in `[center_x, center_y, width, height]` format.

  Returns:
    A JAX array of the same shape, converted to
    `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` format.
  """
  center_x = bboxes_center[..., 0]
  center_y = bboxes_center[..., 1]
  width = bboxes_center[..., 2]
  height = bboxes_center[..., 3]

  bbox_corners = jnp.stack(
      [
          # top left x, top left y, bottom right x, bottom right y
          (center_x - 0.5 * width),
          (center_y - 0.5 * height),
          (center_x + 0.5 * width),
          (center_y + 0.5 * height),
      ],
      axis=-1,
  )
  return bbox_corners

import jax.numpy as jnp
from transformers.utils.generic import TensorType


def _center_to_corners_format_jax(bboxes_center: jnp.ndarray) -> jnp.ndarray:
  """Helper function to convert bounding boxes from center format to corners format for JAX arrays."""
  # A matching module was found in src.MaxText.layers.Qwen3ForCausalLM.bbox_utils._center_to_corners_format_jax
  center_x = bboxes_center[..., 0]
  center_y = bboxes_center[..., 1]
  width = bboxes_center[..., 2]
  height = bboxes_center[..., 3]

  bboxes_corners = jnp.stack(
      [
          # top left x, top left y, bottom right x, bottom right y
          center_x - 0.5 * width,
          center_y - 0.5 * height,
          center_x + 0.5 * width,
          center_y + 0.5 * height,
      ],
      axis=-1,
  )
  return bboxes_corners


def center_to_corners_format(bboxes_center: TensorType) -> TensorType:
  """
  Converts bounding boxes from center format to corners format.

  center format: contains the coordinate for the center of the box and its width, height dimensions
      (center_x, center_y, width, height)
  corners format: contains the coordinates for the top-left and bottom-right corners of the box
      (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
  """
  # In JAX, we don't need a dispatcher for different tensor types like torch, tf, numpy.
  # We write one version that works with JAX arrays (and numpy arrays implicitly).
  return _center_to_corners_format_jax(bboxes_center)

import jax
import jax.numpy as jnp
from maxtext.common_types import Array

def weighting_function(max_num_bins: int, up: Array, reg_scale: float) -> Array:
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
    upper_bound1 = jnp.abs(up[0]) * jnp.abs(reg_scale)
    upper_bound2 = jnp.abs(up[0]) * jnp.abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))

    i_left = jnp.arange(max_num_bins // 2 - 1, 0, -1)
    left_values = -((step) ** i_left) + 1

    i_right = jnp.arange(1, max_num_bins // 2)
    right_values = (step) ** i_right - 1

    values = jnp.concatenate([
        jnp.array([-upper_bound2]),
        left_values,
        jnp.zeros_like(up[0][None]),
        right_values,
        jnp.array([upper_bound2])
    ])
    return values


def translate_gt(gt: Array, max_num_bins: int, reg_scale: float, up: Array) -> tuple[Array, Array, Array]:
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
    diffs = function_values[None, :] - gt[:, None]
    mask = diffs <= 0
    closest_left_indices = jnp.sum(mask, axis=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.astype(jnp.float32)

    # Vectorized computation of weights
    indices_int = closest_left_indices
    # Clamp indices to be valid for gathering from function_values
    safe_indices_int = jnp.clip(indices_int, 0, max_num_bins - 2)

    left_values = function_values[safe_indices_int]
    right_values = function_values[safe_indices_int + 1]

    left_diffs = jnp.abs(gt - left_values)
    right_diffs = jnp.abs(right_values - gt)

    total_diff = left_diffs + right_diffs
    # Handle potential division by zero
    wr = jnp.where(total_diff > 1e-9, left_diffs / total_diff, 0.0)
    wl = 1.0 - wr

    # Apply conditions for valid and invalid indices
    valid_idx_mask = (indices >= 0) & (indices < max_num_bins)
    invalid_idx_mask_neg = indices < 0
    invalid_idx_mask_pos = indices >= max_num_bins

    weight_right = jnp.where(valid_idx_mask, wr, 0.0)
    weight_right = jnp.where(invalid_idx_mask_pos, 1.0, weight_right)

    weight_left = jnp.where(valid_idx_mask, wl, 0.0)
    weight_left = jnp.where(invalid_idx_mask_neg, 1.0, weight_left)

    final_indices = jnp.where(invalid_idx_mask_neg, 0.0, indices)
    final_indices = jnp.where(invalid_idx_mask_pos, max_num_bins - 0.1, final_indices)

    return final_indices, weight_right, weight_left


def bbox2distance(
    points: Array,
    bbox: Array,
    max_num_bins: float,
    reg_scale: float,
    up: Array,
    eps: float = 0.1
) -> tuple[Array, Array, Array]:
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
        tuple[Array, Array, Array]: Decoded distances, right weights, and left weights.
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

    four_lens = jax.lax.stop_gradient(four_lens.reshape(-1))
    weight_right = jax.lax.stop_gradient(weight_right)
    weight_left = jax.lax.stop_gradient(weight_left)

    return four_lens, weight_right, weight_left
