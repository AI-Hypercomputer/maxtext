
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Tuple

# Re-used from Qwen3ForCausalLM.bbox_utils.center_to_corners_format
from Qwen3ForCausalLM.bbox_utils import center_to_corners_format
# Re-used from Qwen3ForCausalLM.box_utils.generalized_box_iou
from Qwen3ForCausalLM.box_utils import generalized_box_iou


class RTDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Attributes:
        config: RTDetrConfig
    """

    config: Any

    @nn.compact
    def __call__(
        self, outputs: Dict[str, jnp.ndarray], targets: List[Dict[str, jnp.ndarray]]
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Performs the matching.

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
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if (
            self.config.matcher_class_cost == self.config.matcher_bbox_cost == self.config.matcher_giou_cost == 0
        ):
            raise ValueError("All costs of the Matcher can't be 0")

        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].reshape(-1, 4)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        target_ids = jnp.concatenate([v["class_labels"] for v in targets])
        target_bbox = jnp.concatenate([v["boxes"] for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
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

        # Move to CPU and convert to numpy for scipy
        cost_matrix_np = np.array(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]

        # This part is not JIT-able as it uses scipy and a python loop
        indices = []
        start_indices = [0] + list(np.cumsum(sizes[:-1]))
        for i in range(batch_size):
            start = start_indices[i]
            end = start + sizes[i]
            cost_matrix_i = cost_matrix_np[i, :, start:end]
            row_ind, col_ind = linear_sum_assignment(cost_matrix_i)
            indices.append((row_ind, col_ind))

        return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]

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
import numpy as np
from scipy.optimize import linear_sum_assignment

# Re-used from Qwen3ForCausalLM.bbox_utils.center_to_corners_format
from Qwen3ForCausalLM.bbox_utils import center_to_corners_format
# Re-used from Qwen3ForCausalLM.box_utils.generalized_box_iou
from Qwen3ForCausalLM.box_utils import generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Attributes:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
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
        out_prob = jax.nn.softmax(outputs["logits"].reshape(batch_size * num_queries, -1), axis=-1)
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
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.reshape(batch_size, num_queries, -1)

        # Move to CPU for scipy
        cost_matrix_cpu = jax.device_get(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]

        # The following part is not JIT-compatible as it uses scipy.optimize.linear_sum_assignment
        # which runs on CPU.
        indices = []
        if sum(sizes) > 0:
            split_indices = np.cumsum(sizes[:-1])
            cost_matrix_splits = np.split(cost_matrix_cpu, split_indices, axis=-1)
            for i, c in enumerate(cost_matrix_splits):
                # c has shape [batch_size, num_queries, size_i]
                # We need the cost matrix for the i-th batch item, which is c[i]
                cost_matrix_i = c[i]
                row_ind, col_ind = linear_sum_assignment(cost_matrix_i)
                indices.append((row_ind, col_ind))
        else:
            indices = [(np.array([], dtype=np.int64), np.array([], dtype=np.int64)) for _ in sizes]

        return [(jnp.asarray(i, dtype=jnp.int64), jnp.asarray(j, dtype=jnp.int64)) for i, j in indices]
