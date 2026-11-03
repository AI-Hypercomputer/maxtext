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

import re
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
from maxtext.kernels.ragged import ragged_gather
from maxtext.kernels.ragged import ragged_sort
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
        # One activation scale per row over the whole K: no mesh axis or emb chunking may split wi's or wo's K.
        # (moe_unsplit_prefused_wi rejects a sharded MLP dim and emb chunks on its own; off, so only this check fires.)
        (
            {"ici_tensor_sequence_parallelism": 2, "moe_unsplit_prefused_wi": False},
            "an unsplit MoE MLP dim, wo's K",
        ),
        ({"ici_tensor_parallelism": 2, "moe_unsplit_prefused_wi": False}, "an unsplit embed dim, wi's K"),
        ({"num_moe_emb_chunks": 2, "moe_unsplit_prefused_wi": False}, "num_moe_emb_chunks=0"),
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


def test_native_fwd_k_splits_reads_the_weight_k_dim():
  # Dim 1 of the [experts, K, N] weights is K (wi: embed, wo: MoE MLP); size-1 mesh axes split nothing.
  spec = jax.sharding.PartitionSpec
  mesh_shape = {"expert": 4, "fsdp": 2, "tensor": 2, "tensor_sequence": 1}
  wi, wo = spec("expert", None, "tensor"), spec("expert", "tensor", None)
  assert moe.fp8_native_fwd_k_splits(mesh_shape, wi, wo, 0) == ["wo's K (MoE MLP dim) sharded 2-way over ('tensor',)"]
  assert moe.fp8_native_fwd_k_splits(
      mesh_shape, spec("expert", ("tensor_sequence", "tensor", "fsdp"), None), spec("expert", "tensor_sequence", None), 2
  ) == [
      "wi's K (embed dim) sharded 4-way over ('tensor', 'fsdp')",
      "wi's K in num_moe_emb_chunks=2 chunks",
  ]
  assert not moe.fp8_native_fwd_k_splits({**mesh_shape, "tensor": 1}, wi, wo, 0)


def test_native_fwd_k_split_guard_runs_before_the_gmms():
  # custom_mesh_and_rule and parallelism -1 are only resolved in the layer, so RoutedMoE checks the traced specs.
  with mock.patch.object(moe, "fp8_native_fwd_k_splits", return_value=["wo's K split for the test"]) as k_splits:
    with pytest.raises(ValueError, match="one scale over the whole K dim.*wo's K split for the test"):
      _lower_forward(_cfg(**_NATIVE))
  assert k_splits.called


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


def test_deferred_fp8_token_dispatch_config_allows_multi_k_tile_wi():
  # With moe_fp8_token_all_gather=True on the ring-of-experts SparseCore ragged-sort path, wi consumes pre-quantized
  # FP8 tokens + precomputed full-K f32 row scales, so wi_tile_fwd_embed_dim may split K into multiple 128-aligned tiles.
  deferred_overrides = {
      **_NATIVE,
      "ici_expert_parallelism": 2,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "moe_fp8_token_all_gather": True,
  }
  cfg = _cfg(**{**deferred_overrides, "wi_tile_fwd_embed_dim": _EMB // 2})
  assert cfg.wi_tile_fwd_embed_dim == _EMB // 2
  with pytest.raises(Exception, match="wi_tile_fwd_embed_dim dividing"):
    _cfg(**{**deferred_overrides, "wi_tile_fwd_embed_dim": 192})
  with pytest.raises(Exception, match=f"wo_tile_fwd_mlp_dim={_MOE_MLP}"):
    _cfg(**{**deferred_overrides, "wo_tile_fwd_mlp_dim": _MOE_MLP // 2})


@pytest.mark.parametrize("index_math", ["sort", "count"])
@pytest.mark.parametrize("buffer_size", [None, 20])
def test_ring_ragged_sort_fp8_ste_tokens_matches_dequant_path(index_math, buffer_size):
  # Deferred FP8 token dispatch gathers 1-byte codes + 1D scales through ring_ragged_sort and carries bf16 cotangents
  # back via x_ste; forward dequantization and backward cotangents must match sorting bf16 tokens bit for bit.
  rng = np.random.default_rng(7)
  tokens = jnp.asarray(rng.standard_normal((16, 128)) * 0.5, jnp.bfloat16)
  topk_ids = jnp.asarray(rng.integers(0, 4, size=(16, 2), dtype=np.int32))
  upstream = jnp.asarray(rng.standard_normal((buffer_size or 32, 128)) * 0.25, jnp.bfloat16)
  qcodes, qscale = moe.quantize_rows_fp8(tokens, jnp.float8_e4m3fn)
  tokens_deq = moe.dequantize_rows_fp8(qcodes, qscale, jnp.bfloat16)
  mesh = jax.make_mesh((1,), ("expert",))

  @jax.jit
  def run_ste(x):
    q, s = moe.quantize_rows_fp8(x, jnp.float8_e4m3fn)
    ste = quantizations.Fp8SteTokens(x_ste=x, codes=q, scale=s)

    def body(tok_in):
      sorted_ste, *_ = ragged_sort.ring_ragged_sort(
          tok_in,
          topk_ids,
          num_experts=4,
          topk=2,
          ep_name="expert",
          ep_size=1,
          buffer_size=buffer_size,
          enforce_gather_fallback=True,
          enforce_gather_reduce_fallback=True,
          index_math=index_math,
      )
      val = sorted_ste.dequantize(jnp.bfloat16)
      # Route backward cotangent through sorted_ste.x_ste, matching gmm_silu_fp8_bwd.
      return jax.lax.stop_gradient(val) + (sorted_ste.x_ste - jax.lax.stop_gradient(sorted_ste.x_ste))

    out = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(
            quantizations.Fp8SteTokens(
                jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()
            ),
        ),
        out_specs=jax.sharding.PartitionSpec(),
        check_vma=False,
    )(ste)
    return jnp.sum(out.astype(jnp.float32) * upstream.astype(jnp.float32)), out

  @jax.jit
  def run_ref(x):
    def body(tok_in):
      sorted_bf16, *_ = ragged_sort.ring_ragged_sort(
          tok_in,
          topk_ids,
          num_experts=4,
          topk=2,
          ep_name="expert",
          ep_size=1,
          buffer_size=buffer_size,
          enforce_gather_fallback=True,
          enforce_gather_reduce_fallback=True,
          index_math=index_math,
      )
      return sorted_bf16

    out = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(jax.sharding.PartitionSpec(),),
        out_specs=jax.sharding.PartitionSpec(),
        check_vma=False,
    )(x)
    return jnp.sum(out.astype(jnp.float32) * upstream.astype(jnp.float32)), out

  (_, out_ste_fwd), grad_ste = jax.value_and_grad(run_ste, has_aux=True)(tokens)
  (_, out_ref_fwd), grad_ref = jax.value_and_grad(run_ref, has_aux=True)(tokens_deq)
  np.testing.assert_array_equal(np.asarray(out_ste_fwd).view(np.uint16), np.asarray(out_ref_fwd).view(np.uint16))
  np.testing.assert_array_equal(np.asarray(grad_ste).view(np.uint16), np.asarray(grad_ref).view(np.uint16))


def test_gmm_v2_prequantized_row_scale_and_subtile_m_configs():
  lhs = jnp.zeros((512, 4096), jnp.float8_e4m3fn)
  lhs_scale = jnp.ones((512, 1), jnp.float32)
  rhs = jnp.zeros((4, 4096, 2048), jnp.float8_e4m3fn)
  rhs_scale = jnp.ones((4, 1, 1, 2048), jnp.float32)
  group_sizes = jnp.array([128, 128, 128, 128], jnp.int32)
  group_offset = jnp.array([0], jnp.int32)
  fake_tpu = types.SimpleNamespace(
      fp8_ops_per_second=1, int8_ops_per_second=1, num_lanes=128, mxu_column_size=256, get_sublane_tiling=lambda _: 8
  )
  with mock.patch.object(gmm_v2.pltpu, "get_tpu_info", return_value=fake_tpu):
    cfgs = gmm_v2.make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        None,
        None,
        group_sizes,
        group_offset,
        tile_info=gmm_v2.TileSizes(tile_m=512, tile_k=2048, tile_n=1024),
        vmem_limit_bytes=None,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        maybe_quantize_lhs=False,
        zero_initialize=True,
        lhs_scale=lhs_scale,
        enable_subtile_m=True,
    )
    meta = gmm_v2.get_metadata(cfgs)
  assert cfgs.lhs_cfgs.has_scale and cfgs.lhs_cfgs.is_row_scale and cfgs.lhs_cfgs.quant_dtype is None
  assert cfgs.enable_subtile_m
  assert meta["enable_subtile_m"] == 1
  assert meta["lhs_cfgs.is_row_scale"] == 1


def test_transposed_wo_drhs_ste_vjp_and_fp8_native_weight_pspec():
  rng = np.random.default_rng(3)
  w = jnp.asarray(rng.standard_normal((4, 64, 32)) * 0.02, jnp.bfloat16)
  codes, scale = quantizations.quantize_weight_codes_per_channel(w)
  cotangent_t = jnp.asarray(rng.standard_normal((4, 32, 64)), jnp.bfloat16)

  anchor_t, vjp_fn = jax.vjp(
      lambda weight: quantizations.dequantize_weight_codes_ste(
          weight, codes, scale, transposed_anchor=True, transposed_grad_transform=lambda g: g * 2.0
      ),
      w,
  )
  assert anchor_t.shape == (4, 32, 64)
  (grad_w,) = vjp_fn(cotangent_t)
  expected = jnp.swapaxes(cotangent_t * 2.0, 1, 2)
  np.testing.assert_array_equal(np.asarray(grad_w).view(np.uint16), np.asarray(expected).view(np.uint16))

  spec = jax.sharding.PartitionSpec("expert", None, "fsdp")
  pspec_t = moe.Fp8NativeWeight.pspec(spec, transpose_anchor=True)
  assert pspec_t.transpose_anchor is True
  assert pspec_t.anchor == jax.sharding.PartitionSpec("expert", "fsdp", None)
  assert pspec_t.codes == spec


def test_tpu7x_lowering_for_deferred_fp8_subtile_m_and_transposed_wo_drhs():
  tpu_dev = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
  single_mesh = jax.sharding.AbstractMesh((), (), abstract_device=tpu_dev)
  with jax.sharding.use_abstract_mesh(single_mesh):
    x_fp8 = jax.ShapeDtypeStruct((256, 256), jnp.float8_e4m3fn)
    idx = jax.ShapeDtypeStruct((512,), jnp.int32)
    rg_mlir = (
        jax.jit(lambda x, i: ragged_gather.ragged_gather(x, i, jnp.array(0), jnp.array(256)))
        .trace(x_fp8, idx)
        .lower(lowering_platforms=("tpu",))
        .as_text()
    )
    assert "sc_ragged_gather" in rg_mlir and "ui8" in rg_mlir

    ste_lhs = quantizations.Fp8SteTokens(
        x_ste=jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
        codes=jax.ShapeDtypeStruct((512, 256), jnp.float8_e4m3fn),
        scale=jax.ShapeDtypeStruct((512, 1), jnp.float32),
    )
    wi_anchor = jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16)
    wi_codes = jax.ShapeDtypeStruct((4, 256, 512), jnp.float8_e4m3fn)
    wi_scale = jax.ShapeDtypeStruct((4, 1, 512), jnp.float32)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    wi_tiling = (256, 128, 256, 256, 256, 256, 256, 256, 256)

    def wi_loss(ste, anchor, c, s, g):
      act, gate_up = megablox_ops.gmm_silu_fp8(
          ste,
          anchor,
          c,
          s,
          g,
          jnp.bfloat16,
          wi_tiling,
          use_dlhs_transpose_rhs=True,
          acc_dtype=jnp.float32,
          lhs_quant_mode="fullk_f32",
          native_bwd_dlhs=True,
      )
      return jnp.sum(jax.lax.stop_gradient(act).astype(jnp.float32)) + jnp.sum(gate_up.astype(jnp.float32))

    wi_mlir = (
        jax.jit(jax.value_and_grad(wi_loss, argnums=(0, 1)))
        .trace(ste_lhs, wi_anchor, wi_codes, wi_scale, gs)
        .lower(lowering_platforms=("tpu",))
        .as_text()
    )
    wi_kernels = re.findall(r'kernel_name = "([^"]+)"', wi_mlir)
    assert any("tk_128" in k and "pre_act_concat" in k for k in wi_kernels), wi_kernels
    assert any("rhs_cdq" in k for k in wi_kernels), wi_kernels

    lhs_act = jax.ShapeDtypeStruct((512, 128), jnp.bfloat16)
    gate_up = jax.ShapeDtypeStruct((512, 256), jnp.bfloat16)
    wo_anc_t = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    wo_codes = jax.ShapeDtypeStruct((4, 128, 256), jnp.float8_e4m3fn)
    wo_scale = jax.ShapeDtypeStruct((4, 1, 256), jnp.float32)
    wo_tiling = (256, 128, 256, 256, 256, 128, 256, 128, 256)

    def wo_loss(h, gu, anc_t, c, s, g):
      out = megablox_ops.gmm_dswiglu_fp8(
          h,
          gu,
          anc_t,
          c,
          s,
          g,
          jnp.bfloat16,
          wo_tiling,
          use_dlhs_transpose_rhs=True,
          dact_pre_act_tile_n=128,
          acc_dtype=jnp.float32,
          lhs_quant_mode="fullk_f32",
          native_bwd_dlhs=True,
          transpose_drhs=True,
      )
      return jnp.sum(out.astype(jnp.float32))

    wo_traced = jax.jit(jax.value_and_grad(wo_loss, argnums=(1, 2))).trace(
        lhs_act, gate_up, wo_anc_t, wo_codes, wo_scale, gs
    )
    assert [x.shape for x in wo_traced.out_info[1]] == [(512, 256), (4, 256, 128)]
    wo_mlir = wo_traced.lower(lowering_platforms=("tpu",)).as_text()
    wo_kernels = re.findall(r'kernel_name = "([^"]+)"', wo_mlir)
    assert any("tgmm_v2-g_4-m_512-k_256-act_None-n_128" in k for k in wo_kernels), wo_kernels
