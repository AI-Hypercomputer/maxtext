
from typing import TypeVar

SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")

from MaxText import max_logging as logging

logger = logging.getLogger(__name__)

_torch_available = False

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from flax.configurations import PretrainedConfig

# from transformers.modeling_flax_utils import FlaxPreTrainedModel
# from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPast
# from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
# from transformers.models.qwen2.modeling_flax_qwen2 import FlaxQwen2RMSNorm, FlaxQwen2RotaryEmbedding, ACT2FN
# from ...modeling_flax_outputs import FlaxCausalLMOutputWithPast
# from ...modeling_flax_utils import ACT2FN
# from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# from flax.linen import Dense, RMSNorm
# from flax.linen.attention import dot_product_attention_weights
# from flax.traverse_util import flatten_dict, unflatten_dict
# from jax.sharding import PartitionSpec
# from jax import lax
# from flax import linen as nn
# from ..flax_utils import FlaxAttentionModule, FlaxDynamicCache, get_dot_general_by_env
# from ...generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
# from ..qwen2.modeling_flax_qwen2 import FlaxQwen2Attention, FlaxQwen2DecoderLayer, FlaxQwen2ForCausalLMModule, FlaxQwen2MLP, FlaxQwen2Model, FlaxQwen2PreTrainedModel
# from .configuration_qwen2_moe import Qwen2MoeConfig
# from ...utils import logging
# from .moe import FlaxQwen2MoeBlock
# from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
# from jax.experimental.shard_map import shard_map
# from functools import partial
# from flax.linen.partitioning import remat
# from jax.experimental import pallas as pl
# from jax.experimental.pallas import tpu as pltpu
# from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes as FlashBlockSizes
# from jax.experimental.pallas.ops.tpu.flash_attention import PallasFlashAttentionFwd
# from jax.experimental.pallas.ops.tpu.flash_attention import mha_forward_kernel as tpu_flash_attention_fwd
# from jax.experimental.pallas.ops.tpu.flash_attention import mha_bwd_kernel as tpu_flash_attention_bwd
# from jax.experimental.pallas.ops.tpu.flash_attention import MHA BwdPrefetch
# from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds
# from jax.experimental.pallas.ops.tpu.flash_attention import mha_prepare_bwd_kernel as tpu_flash_attention_prepare_bwd
# from jax.experimental.pallas.ops.tpu.flash_attention import BwdDqDkDv
# from jax.experimental.pallas.ops.tpu.flash_attention import BwdPass as FlashBwdPass
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_preprocess_dk_dv_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_metadata_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_q_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_dq_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_dk_dv_reduce_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_dq_reduce_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_dq_accum_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_dk_dv_accum_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_q_dot_grad_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_main_loop
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_accum_dk_dv_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_accum_dq_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_launch_dk_dv_reduce
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_launch_dq_reduce
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_slice
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_q_dot_grad_reduce_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_launch_q_dot_grad_reduce
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_accum_q_dot_grad_kernel
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_k_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_v_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_o_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_do_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_l_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_m_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dk_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dv_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_dq_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_d_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_scratch_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_accum_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_dot_grad_tile
# from jax.experimental.pallas.ops.tpu.flash_attention import _bwd_get_q_tile
from MaxText import max_logging as logging

logger = logging.get_logger(__name__)
pip install -U scikit-learn

pip install tokenizers

from typing import List
from .configuration_qwen3_moe import Qwen3MoeConfig
# Re-used from 'Qwen3ForCausalLM.modeling.PreTrainedModel'
from ..modeling import PreTrainedModel


class Qwen3MoePreTrainedModel(PreTrainedModel):
    config: Qwen3MoeConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    # The following attributes are specific to the Hugging Face PyTorch implementation
    # and do not have a direct functional equivalent in a JAX/Flax environment.
    # Their functionality is achieved through different mechanisms like JAX sharding
    # annotations. They are kept here for structural consistency.
    _no_split_modules: List[str] = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement: List[str] = ["past_key_values"]

    # Flags indicating support for various attention backends. In MaxText, the
    # attention kernel is usually selected via a config string.
    _supports_flash_attn: bool = True
    _supports_sdpa: bool = True
    _supports_flex_attn: bool = True
    _supports_attention_backend: bool = True

    # JAX uses `jit` compilation, which has different characteristics than `torch.compile`.
    _can_compile_fullgraph: bool = False

    # In Flax, intermediate outputs like router_logits, hidden_states, and attentions
    # are captured using the `sow` mechanism within the respective layers.
    # The `_can_record_outputs` dictionary is a Hugging Face specific mechanism for hooks
    # and is therefore omitted here.

from typing import Optional
import flax.linen as nn
import jax.numpy as jnp

from maxtext.layers.embeddings import Embed
from maxtext.layers.qwen3 import Qwen3MoeDecoderLayer, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding
from maxtext.common_types import Array, DType
from maxtext.layers.models import MoeModelOutputWithPast
from ..configs.qwen3_moe import Qwen3MoeConfig


class Qwen3MoeModel(nn.Module):
  """A JAX/Flax implementation of Qwen3MoeModel."""

  config: Qwen3MoeConfig
  dtype: DType = jnp.float32

  def setup(self):
    self.vocab_size = self.config.vocab_size

    # Reused MaxText module
    # Path: src/MaxText/layers/embeddings.py
    self.embed_tokens = Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.hidden_size,
        dtype=self.dtype,
        name="embed_tokens",
    )
    # Reused MaxText module
    # Path: src/MaxText/layers/qwen3.py
    self.layers = [
        Qwen3MoeDecoderLayer(
            config=self.config, layer_idx=i, name=f"layers_{i}", dtype=self.dtype
        )
        for i in range(self.config.num_hidden_layers)
    ]
    # Reused MaxText module
    # Path: Qwen3ForCausalLM/layers/Qwen3MoeRMSNorm.py
    self.norm = Qwen3MoeRMSNorm(
        self.config.hidden_size,
        eps=self.config.rms_norm_eps,
        name="norm",
        dtype=self.dtype,
    )
    # Reused MaxText module
    # Path: Qwen3ForCausalLM/layers/Qwen3MoeRotaryEmbedding.py
    self.rotary_emb = Qwen3MoeRotaryEmbedding(
        config=self.config, name="rotary_emb", dtype=self.dtype
    )
    # gradient_checkpointing is handled by nn.remat in Flax, not as a module attribute.

  def __call__(
      self,
      input_ids: Optional[Array] = None,
      attention_mask: Optional[Array] = None,
      position_ids: Optional[Array] = None,
      inputs_embeds: Optional[Array] = None,
      # Flax specific arguments
      deterministic: bool = True,
      init_cache: bool = False,
      return_dict: bool = True,
  ):
    if (input_ids is None) == (inputs_embeds is None):
      raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
      inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

    # In JAX/Flax, KV-cache handling, position_ids generation for decoding,
    # and attention mask creation are typically handled outside or within
    # lower-level layers (like Attention), controlled by `model_mode` and
    # input `decoder_segment_ids`. The explicit mask creation from PyTorch
    # is omitted here to follow the MaxText pattern.
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer in self.layers:
      # The JAX Qwen3MoeDecoderLayer is expected to handle mask creation and KV caching internally
      hidden_states = layer(
          hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          position_ids=position_ids,
          deterministic=deterministic,
          init_cache=init_cache,
      )

    hidden_states = self.norm(hidden_states)

    if not return_dict:
      return (hidden_states,)

    # In JAX, past_key_values are not returned directly from the __call__ method.
    # They are part of the mutable state managed by the linen Module.
    # The caller will have access to the updated cache. Here we return None
    # for past_key_values to match the dataclass structure.
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
    )

from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...modeling_flax_outputs import FlaxCausalLMOutputWithPast
from .configuration_qwen3_moe import Qwen3MoeConfig
# Reused from Qwen3ForCausalLM.modeling import Qwen3MoePreTrainedModel, Qwen3MoeModel
from .modeling import Qwen3MoeModel, Qwen3MoePreTrainedModel
# Reused from Qwen3ForCausalLM.model_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from .model_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
# Reused from Qwen3ForCausalLM.losses import load_balancing_loss_func
from .losses import load_balancing_loss_func
# Reused from Qwen3ForCausalLM.cache import Cache
from .cache import Cache


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel):
    """
    The Qwen3Moe model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """

    config: Qwen3MoeConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

    def setup(self):
        self.model = Qwen3MoeModel(
            self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            name="lm_head",
        )
        self.router_aux_loss_coef = self.config.router_aux_loss_coef
        self.num_experts = self.config.num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            deterministic=deterministic,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # In JAX/Flax, loss computation is typically handled outside the model's __call__ function,
        # often in a separate train_step or eval_step function.
        loss = None

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            # The addition of aux_loss to the main loss is handled in the training step.

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
