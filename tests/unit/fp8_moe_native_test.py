# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""fp8_moe_native_fwd / fp8_moe_native_bwd_dlhs on CPU: config validation, the outside-layer-scan guard, the
dequantization contract the native path relies on, and the fixed activation-quantization / accumulator settings.
"""

import sys
import types
from unittest import mock

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

_EMB = 256
_MOE_MLP = 128
_SEQ = 64
_VOCAB = 512
_EXPERTS = 8
_TOP_K = 2

# A shrunk Qwen3.5 MoE model on one device, with the fake-QAT gather8 path the native flags build on.
_MODEL = {
    "model_name": "qwen3.5-397b-a17b",
    "override_model_config": True,
    "base_num_decoder_layers": 4,
    "base_emb_dim": _EMB,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 64,
    "vocab_size": _VOCAB,
    "base_mlp_dim": _MOE_MLP,
    "base_moe_mlp_dim": _MOE_MLP,
    "num_experts": _EXPERTS,
    "num_experts_per_tok": _TOP_K,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 32,
    "gdn_value_head_dim": 32,
    "gdn_chunk_size": 16,
    "use_gdn_kernel": False,
    "use_mrope": False,
    "max_target_length": _SEQ,
    "max_prefill_predict_length": _SEQ,
    "per_device_batch_size": 2,
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "attention": "dot_product",
    "use_tokamax_splash": False,
    "prefuse_moe_weights": True,
    "moe_unsplit_prefused_wi": True,
    "megablox": True,
    "sparse_matmul": True,
    "fp8_moe_fake_quant": True,
    "fp8_fake_quant_hoist": "gather8",
    "ragged_buffer_factor": -1.0,
    "scan_layers": True,
    "enable_dropout": False,
}

# What fp8_moe_native_fwd needs on top of the fake-QAT gather8 model.
_NATIVE = {
    "fp8_moe_native_fwd": True,
    "use_tokamax_gmm": True,
    "use_gmm_v2": True,
    "moe_wi_gmm_fused_swiglu": True,
    "moe_wo_gmm_fused_dswiglu": True,
    "wi_tile_fwd_embed_dim": _EMB,
    "wo_tile_fwd_mlp_dim": _MOE_MLP,
    "wo_tile_dlhs_mlp_dim": _MOE_MLP,
}

_NATIVE_BWD_DLHS = {"fp8_moe_native_bwd_dlhs": True, "moe_gmm_v2_dlhs_transpose_rhs": True}


def _cfg(**overrides):
  kwargs = {
      "run_name": "fp8_moe_native",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      **_MODEL,
      **overrides,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def test_flags_are_defined_and_off_by_default():
  for key in ("fp8_moe_native_fwd", "fp8_moe_native_bwd_dlhs"):
    assert key in set(config_types.MoEGeneral.model_fields), key
  cfg = _cfg()
  assert not cfg.fp8_moe_native_fwd and not cfg.fp8_moe_native_bwd_dlhs


def test_native_config_is_accepted():
  cfg = _cfg(**_NATIVE, **_NATIVE_BWD_DLHS)
  assert cfg.fp8_moe_native_fwd and cfg.fp8_moe_native_bwd_dlhs
  eval_cfg = _cfg(
      **_NATIVE,
      custom_mesh_and_rule_for_eval="pure-fsdp",
      eval_wi_tile_fwd_embed_dim=_EMB,
      eval_wo_tile_fwd_mlp_dim=_MOE_MLP,
  )
  assert eval_cfg.fp8_moe_native_fwd


@pytest.mark.parametrize(
    "override,requirement",
    [
        ({"fp8_moe_fake_quant": False}, "fp8_moe_fake_quant=True"),
        ({"fp8_fake_quant_hoist": "absmax"}, "fp8_fake_quant_hoist=gather8"),
        ({"use_gmm_v2": False}, "use_tokamax_gmm=True and use_gmm_v2=True"),
        # The heuristic tiling budgets a bf16 accumulator; the native forward accumulates in f32.
        ({"use_gmm_v2_heuristic_tiling": True}, "use_gmm_v2_heuristic_tiling=False"),
        # Full-K dynamic lhs quantization requires one K tile (tile_k == size_k) in wi, wo and separate-rule eval.
        ({"base_emb_dim": 4096, "wi_tile_fwd_embed_dim": 1024}, "wi_tile_fwd_embed_dim=4096"),
        ({"wi_tile_fwd_embed_dim": 2 * _EMB}, f"wi_tile_fwd_embed_dim={_EMB}"),
        ({"wo_tile_fwd_mlp_dim": _MOE_MLP // 2}, f"wo_tile_fwd_mlp_dim={_MOE_MLP}"),
        (
            {"ici_expert_parallelism": 2, "custom_mesh_and_rule_for_eval": "ep-as-cp"},
            f"eval_wi_tile_fwd_embed_dim={_EMB}",
        ),
        (
            {
                "ici_expert_parallelism": 2,
                "custom_mesh_and_rule_for_eval": "ep-as-cp",
                "eval_wi_tile_fwd_embed_dim": _EMB,
            },
            f"eval_wo_tile_fwd_mlp_dim={_MOE_MLP}",
        ),
    ],
)
def test_native_fwd_requirement_is_enforced(override, requirement):
  with pytest.raises(Exception, match=requirement):
    _cfg(**{**_NATIVE, **override})


@pytest.mark.parametrize(
    "override,requirement",
    [
        ({"fp8_moe_native_fwd": False}, "fp8_moe_native_fwd=True"),
        ({"moe_gmm_v2_dlhs_transpose_rhs": False}, "moe_gmm_v2_dlhs_transpose_rhs=True"),
        ({"use_gmm_v2_heuristic_tiling": True}, "use_gmm_v2_heuristic_tiling=False"),
    ],
)
def test_native_bwd_dlhs_requirement_is_enforced(override, requirement):
  with pytest.raises(Exception, match=requirement):
    _cfg(**{**_NATIVE, **_NATIVE_BWD_DLHS, **override})


def _lower_forward(cfg):
  """Builds the model and lowers its forward on CPU (no compile)."""
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(cfg, mesh=mesh)
  model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_fn, mesh=mesh)
  batch = int(cfg.per_device_batch_size * jax.device_count())
  rng = np.random.default_rng(0)
  tokens = jnp.asarray(rng.integers(1, _VOCAB, size=(batch, _SEQ), dtype=np.int32))
  positions = jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32), (batch, _SEQ))
  segments = jnp.ones((batch, _SEQ), jnp.int32)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def forward(p):
      return nnx.merge(graphdef, p, rest, copy=True)(
          tokens, positions, decoder_segment_ids=segments, enable_dropout=False
      )

    jax.jit(forward).lower(params)


def test_native_fwd_outside_the_layer_scan_raises():
  # Unscanned layers get no gather8 absmax; without the guard the layer would quietly run fake QAT.
  with pytest.raises(ValueError, match="needs the gather8 absmax"):
    _lower_forward(_cfg(**_NATIVE, scan_layers=False))


def test_fake_quant_gather8_outside_the_layer_scan_still_lowers():
  # Control: the same unscanned model without fp8_moe_native_fwd lowers (fake QAT rounds the weights in the layer).
  _lower_forward(_cfg(scan_layers=False))


def test_dequantize_fp8_codes_matches_the_gather8_dequantization():
  # The native backward multiplies by dequantize_fp8_codes; fake QAT's gather8 path by dequantize_weight_codes_ste.
  # The native path claims bit-identical gradients, so the two expressions must agree bit for bit.
  rng = np.random.default_rng(0)
  weight = jnp.asarray(rng.standard_normal((4, 64, 32)) * 0.02, jnp.bfloat16)
  codes, scale = quantizations.quantize_weight_codes_per_channel(weight)
  native = megablox_ops.dequantize_fp8_codes(codes, scale, jnp.bfloat16)
  gather8 = quantizations.dequantize_weight_codes_ste(weight, codes, scale)
  assert native.dtype == gather8.dtype == jnp.bfloat16
  np.testing.assert_array_equal(np.asarray(native).view(np.uint16), np.asarray(gather8).view(np.uint16))


def test_native_fwd_uses_full_k_f32_activation_quant_and_f32_accumulation():
  # One float32 absmax scale per row over the full K dim and a float32 accumulator, with no config knob: the plain
  # fp8_moe_native_fwd=True must get the numerics closest to the rollout's FP8 MoE kernel.
  kwargs = moe.fp8_native_fwd_kwargs(_cfg(**_NATIVE))
  assert kwargs["lhs_quant_mode"] == "fullk_f32"
  assert kwargs["lhs_quant_mode"] in megablox_ops._LHS_QUANT_MODES  # pylint: disable=protected-access
  assert kwargs["acc_dtype"] == jnp.float32
  assert not kwargs["native_bwd_dlhs"]


def test_gmm_v2_full_k_lhs_quant_requires_single_k_tile():
  # lhs_quant_block_size=0 means one dynamic scale per row over all size_k columns; inner_kernel slices one tile_k at a
  # time, so make_gmm_configs must reject tile_k != size_k instead of computing per-tile scales.
  lhs = jnp.zeros((512, 4096), jnp.bfloat16)
  rhs = jnp.zeros((4, 4096, 2048), jnp.float8_e4m3fn)
  rhs_scale = jnp.ones((4, 1, 1, 2048), jnp.float32)
  group_sizes = jnp.array([128, 128, 128, 128], jnp.int32)
  group_offset = jnp.array([0], jnp.int32)
  fake_tpu = types.SimpleNamespace(
      fp8_ops_per_second=1, int8_ops_per_second=1, num_lanes=128, mxu_column_size=256, get_sublane_tiling=lambda _: 8
  )

  def make(tile_k):
    return gmm_v2.make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        None,
        None,
        group_sizes,
        group_offset,
        tile_info=gmm_v2.TileSizes(tile_m=512, tile_k=tile_k, tile_n=1024),
        vmem_limit_bytes=None,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        lhs_quant_block_size=0,
        lhs_f32_scale=True,
    )

  with mock.patch.object(gmm_v2.pltpu, "get_tpu_info", return_value=fake_tpu):
    assert make(4096).lhs_cfgs.quant_block_size == 4096
    for bad_tile_k in (1024, 8192):
      with pytest.raises(NotImplementedError, match="tile_k == size_k"):
        make(bad_tile_k)
