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
"""fp8_moe_native_fwd / fp8_moe_native_bwd_dlhs on CPU: config validation, the outside-layer-scan guard, and the
dequantization and lhs-quant-mode contracts the native path relies on.
"""

import sys
import typing

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
from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

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
    "base_emb_dim": 256,
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
  for key in (
      "fp8_moe_native_fwd",
      "fp8_moe_native_fwd_acc_dtype",
      "fp8_moe_native_fwd_lhs_quant",
      "fp8_moe_native_bwd_dlhs",
  ):
    assert key in set(config_types.MoEGeneral.model_fields), key
  cfg = _cfg()
  assert not cfg.fp8_moe_native_fwd and not cfg.fp8_moe_native_bwd_dlhs
  assert cfg.fp8_moe_native_fwd_acc_dtype == "" and cfg.fp8_moe_native_fwd_lhs_quant == ""


def test_native_config_is_accepted():
  cfg = _cfg(
      **_NATIVE, **_NATIVE_BWD_DLHS, fp8_moe_native_fwd_lhs_quant="fullk_f32", fp8_moe_native_fwd_acc_dtype="float32"
  )
  assert cfg.fp8_moe_native_fwd and cfg.fp8_moe_native_bwd_dlhs


@pytest.mark.parametrize(
    "override,requirement",
    [
        ({"fp8_moe_fake_quant": False}, "fp8_moe_fake_quant=True"),
        ({"fp8_fake_quant_hoist": "absmax"}, "fp8_fake_quant_hoist=gather8"),
        ({"use_gmm_v2": False}, "use_tokamax_gmm=True and use_gmm_v2=True"),
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


def test_lhs_quant_modes_match_the_config_values():
  # A value the config accepts but ops does not know would fail only at trace time.
  field = dict(config_types.MoEGeneral.model_fields)["fp8_moe_native_fwd_lhs_quant"]
  config_values = set(typing.get_args(field.annotation))
  assert config_values == set(megablox_ops._LHS_QUANT_MODES)  # pylint: disable=protected-access
