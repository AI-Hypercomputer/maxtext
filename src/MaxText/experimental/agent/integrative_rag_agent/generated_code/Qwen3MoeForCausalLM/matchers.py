
from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

# The following functions are referenced from the JAX_MODULES_DICT and are assumed to be available in the scope.
# generated_code.Qwen3MoeForCausalLM.bbox_utils.center_to_corners_format
# generated_code.Qwen3MoeForCausalLM.bbox_utils.generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Attributes:
        class_cost: The relative weight of the classification error in the matching cost.
        bbox_cost: The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost: The relative weight of the giou loss of the bounding box in the matching cost.
    """

    class_cost: float = 1.0
    bbox_cost: float = 1.0
    giou_cost: float = 1.0

    def setup(self):
        if self.class_cost == 0 and self.bbox_cost == 0 and self.giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    def __call__(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
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
        out_prob = jax.nn.softmax(outputs["logits"].reshape(batch_size * num_queries, -1))
        out_bbox = outputs["pred_boxes"].reshape(batch_size * num_queries, -1)

        # Also concat the target labels and boxes
        target_ids = jnp.concatenate([v["class_labels"] for v in targets])
        target_bbox = jnp.concatenate([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

        # Compute the giou cost between boxes
        # Re-used from generated_code.Qwen3MoeForCausalLM.bbox_utils.center_to_corners_format
        # Re-used from generated_code.Qwen3MoeForCausalLM.bbox_utils.generalized_box_iou
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

        # Move to CPU for scipy
        cost_matrix_np = np.asarray(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        start_idx = 0
        for i in range(batch_size):
            num_targets = sizes[i]
            end_idx = start_idx + num_targets
            c_i = cost_matrix_np[i, :, start_idx:end_idx]
            row_ind, col_ind = linear_sum_assignment(c_i)
            indices.append((row_ind, col_ind))
            start_idx = end_idx

        return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]

from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

# Re-used from generated_code.Qwen3MoeForCausalLM.bbox_utils.center_to_corners_format
from ....models.qwen3_moe.bbox_utils import center_to_corners_format

# Re-used from generated_code.Qwen3MoeForCausalLM.bbox_utils.generalized_box_iou
from ....models.qwen3_moe.bbox_utils import generalized_box_iou

# Re-used from generated_code.Qwen3MoeForCausalLM.matchers.HungarianMatcher
from ....models.qwen3_moe.matchers import HungarianMatcher


class GroundingDinoHungarianMatcher(HungarianMatcher):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __call__(self, outputs: Dict[str, Any], targets: List[Dict[str, jnp.ndarray]]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Performs the matching.

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
            outputs["logits"].reshape(batch_size * num_queries, -1)
        )  # [batch_size * num_queries, hidden_dim]
        out_bbox = outputs["pred_boxes"].reshape(batch_size * num_queries, 4)  # [batch_size * num_queries, 4]
        label_maps = outputs["label_maps"]

        # First take the label map for each class in each batch and then concatenate them
        label_maps = jnp.concatenate(
            [label_map[target["class_labels"]] for label_map, target in zip(label_maps, targets)]
        )
        # Normalize label maps based on number of tokens per class
        label_maps = label_maps / label_maps.sum(axis=-1, keepdims=True)

        # Also concat the target labels and boxes
        target_bbox = jnp.concatenate([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(jnp.log(1 - out_prob + 1e-8)))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(jnp.log(out_prob + 1e-8)))
        # Compute the classification cost by taking pos and neg cost in the appropriate index
        class_cost = (pos_cost_class - neg_cost_class) @ label_maps.T

        # Compute the L1 cost between boxes
        bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)
        cost_matrix_cpu = np.asarray(jax.device_get(cost_matrix))

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(np.split(cost_matrix_cpu, np.cumsum(sizes)[:-1], axis=-1))
        ]
        return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]

from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import linear_sum_assignment

# from transformers.models.rt_detr.modeling_rt_detr import generalized_box_iou, center_to_corners_format
# reused from generated_code/Qwen3MoeForCausalLM/bbox_utils.py
from generated_code.Qwen3MoeForCausalLM.bbox_utils import (
    center_to_corners_format,
    generalized_box_iou,
)


class RTDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Attributes:
        class_cost (float): The relative weight of the classification cost in the matching cost.
        bbox_cost (float): The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost (float): The relative weight of the generalized IoU loss of the bounding box in the matching cost.
        use_focal_loss (bool): Whether to use focal loss for the classification cost.
        alpha (float): Alpha parameter for focal loss.
        gamma (float): Gamma parameter for focal loss.
    """

    class_cost: float
    bbox_cost: float
    giou_cost: float
    use_focal_loss: bool
    alpha: float
    gamma: float

    def setup(self):
        if self.class_cost == 0 and self.bbox_cost == 0 and self.giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    def __call__(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[Array, Array]]:
        """
        Performs the matching.

        Args:
            outputs (Dict):
                This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets (List[Dict]):
                This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = jnp.concatenate([v["class_labels"] for v in targets])
        target_bbox = jnp.concatenate([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        if self.use_focal_loss:
            out_prob = jax.nn.sigmoid(outputs["logits"].reshape(batch_size * num_queries, -1))
            out_prob = out_prob[:, target_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-jnp.log(1 - out_prob + 1e-8))
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-jnp.log(out_prob + 1e-8))
            class_cost = pos_cost_class - neg_cost_class
        else:
            out_prob = jax.nn.softmax(
                outputs["logits"].reshape(batch_size * num_queries, -1), axis=-1
            )  # [batch_size * num_queries, num_classes]
            class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Compute the final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

        # Move to CPU for scipy
        cost_matrix_cpu = jax.device_get(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        start_idx = 0
        for i in range(batch_size):
            num_targets = sizes[i]
            end_idx = start_idx + num_targets
            cost_matrix_i = cost_matrix_cpu[i, :, start_idx:end_idx]
            row_ind, col_ind = linear_sum_assignment(cost_matrix_i)
            indices.append((row_ind, col_ind))
            start_idx = end_idx

        return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import Array
from scipy.optimize import linear_sum_assignment

# From generated_code.Qwen3MoeForCausalLM.matchers.HungarianMatcher
from generated_code.Qwen3MoeForCausalLM.matchers import HungarianMatcher
# From generated_code.Qwen3MoeForCausalLM.bbox_utils.center_to_corners_format
from generated_code.Qwen3MoeForCausalLM.bbox_utils import center_to_corners_format
# From generated_code.Qwen3MoeForCausalLM.bbox_utils.generalized_box_iou
from generated_code.Qwen3MoeForCausalLM.bbox_utils import generalized_box_iou


class DeformableDetrHungarianMatcher(HungarianMatcher):
    """
    Computes an assignment between targets and predictions of the deformable DETR model.
    """

    def __call__(self, outputs: Dict[str, Array], targets: List[FrozenDict]) -> List[Tuple[Array, Array]]:
        """
        Performs the matching
        Args:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                 objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds the indices of the matched queries and targets.
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = jax.nn.sigmoid(
            outputs["logits"].reshape(-1, outputs["logits"].shape[-1])
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = jnp.concatenate([v["class_labels"] for v in targets])
        target_bbox = jnp.concatenate([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(jnp.log(1 - out_prob + 1e-8)))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(jnp.log(out_prob + 1e-8)))
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = jnp.sum(jnp.abs(out_bbox[:, None, :] - target_bbox[None, :, :]), axis=-1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(
            center_to_corners_format(out_bbox), center_to_corners_format(target_bbox)
        )

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)
        cost_matrix_cpu = jax.device_get(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(np.split(cost_matrix_cpu, np.cumsum(sizes)[:-1], axis=-1))
        ]
        return [
            (jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices
        ]
