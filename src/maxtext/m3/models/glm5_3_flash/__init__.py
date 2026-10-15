# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.3-Flash training bringup package for MaxText."""

from maxtext.m3.models.glm5_3_flash.create_mini_slice import extract_mini_slice_keys
from maxtext.m3.models.glm5_3_flash.create_mini_slice import write_mini_slice_checkpoint
from maxtext.m3.models.glm5_3_flash.glm5_ckpt import convert_hf_checkpoint_to_orbax
from maxtext.m3.models.glm5_3_flash.glm5_ckpt import dequantize_fp8_block128
from maxtext.m3.models.glm5_3_flash.glm5_ckpt import load_hf_state_into_model
from maxtext.m3.models.glm5_3_flash.glm5_ckpt import restore_orbax_checkpoint
from maxtext.m3.models.glm5_3_flash.glm5_ckpt import save_orbax_checkpoint
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import MTP_LAYER_INDEX
from maxtext.m3.models.glm5_3_flash.glm5_config import validate_no_packing
from maxtext.m3.models.glm5_3_flash.glm5_dsa_mla import Glm5DsaMlaAttention
from maxtext.m3.models.glm5_3_flash.glm5_dsa_mla import Glm5Indexer
from maxtext.m3.models.glm5_3_flash.glm5_dsa_mla import Glm5LayerNorm
from maxtext.m3.models.glm5_3_flash.glm5_dsa_mla import Glm5RMSNorm
from maxtext.m3.models.glm5_3_flash.glm5_flops import calculate_glm5_3_flash_tflops_per_device
from maxtext.m3.models.glm5_3_flash.glm5_kda import Glm5KdaAttention
from maxtext.m3.models.glm5_3_flash.glm5_kda import Glm5RMSNormGated
from maxtext.m3.models.glm5_3_flash.glm5_kda import Glm5ShortConv1d
from maxtext.m3.models.glm5_3_flash.glm5_kda import l2norm
from maxtext.m3.models.glm5_3_flash.glm5_loss import glm5_loss_fn
from maxtext.m3.models.glm5_3_flash.glm5_loss import trainable_mask
from maxtext.m3.models.glm5_3_flash.glm5_mhc import Glm5DecoderLayer
from maxtext.m3.models.glm5_3_flash.glm5_mhc import Glm5HyperConnection
from maxtext.m3.models.glm5_3_flash.glm5_mhc import Glm5HyperHead
from maxtext.m3.models.glm5_3_flash.glm5_mhc import Glm5UnweightedRMSNorm
from maxtext.m3.models.glm5_3_flash.glm5_moe import compute_moe_load_balancing_loss
from maxtext.m3.models.glm5_3_flash.glm5_moe import Glm5Experts
from maxtext.m3.models.glm5_3_flash.glm5_moe import Glm5MLP
from maxtext.m3.models.glm5_3_flash.glm5_moe import Glm5MoE
from maxtext.m3.models.glm5_3_flash.glm5_moe import Glm5RouterBias
from maxtext.m3.models.glm5_3_flash.glm5_moe import Glm5TopkRouter
from maxtext.m3.models.glm5_3_flash.glm5_mtp import Glm5Mtp
from maxtext.m3.models.glm5_3_flash.glm5_scan import apply_glm5_remat
from maxtext.m3.models.glm5_3_flash.glm5_scan import create_stacked_super_block
from maxtext.m3.models.glm5_3_flash.glm5_scan import Glm5SuperBlock
from maxtext.m3.models.glm5_3_flash.glm5_scan import layer_to_block_slot
from maxtext.m3.models.glm5_3_flash.glm5_scan import scan_super_blocks
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import create_glm5_3_flash_model
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5Embedding
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5Model
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5Output

__all__ = [
    "Glm5Config",
    "Glm5DecoderLayer",
    "Glm5DsaMlaAttention",
    "Glm5Embedding",
    "Glm5Experts",
    "Glm5HyperConnection",
    "Glm5HyperHead",
    "Glm5Indexer",
    "Glm5KdaAttention",
    "Glm5LayerNorm",
    "Glm5MLP",
    "Glm5MoE",
    "Glm5Model",
    "Glm5Mtp",
    "Glm5Output",
    "Glm5RMSNorm",
    "Glm5RMSNormGated",
    "Glm5RouterBias",
    "Glm5Sharding",
    "Glm5ShortConv1d",
    "Glm5SuperBlock",
    "Glm5TopkRouter",
    "Glm5UnweightedRMSNorm",
    "MTP_LAYER_INDEX",
    "apply_glm5_remat",
    "calculate_glm5_3_flash_tflops_per_device",
    "compute_moe_load_balancing_loss",
    "convert_hf_checkpoint_to_orbax",
    "create_glm5_3_flash_model",
    "create_stacked_super_block",
    "dequantize_fp8_block128",
    "extract_mini_slice_keys",
    "glm5_loss_fn",
    "l2norm",
    "layer_to_block_slot",
    "load_hf_state_into_model",
    "restore_orbax_checkpoint",
    "save_orbax_checkpoint",
    "scan_super_blocks",
    "trainable_mask",
    "validate_no_packing",
    "write_mini_slice_checkpoint",
]
