# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for OLMoE3 paged-state KDA (vLLM) and RL/Tunix integration."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.integration.vllm.hybrid_cache_utils import build_qwen_gdn_cache_layout
from maxtext.models import olmoe3
from maxtext.utils.globals import MAXTEXT_PKG_DIR


@dataclass
class _FakeAttentionMetadata:
  input_positions: jax.Array
  seq_lens: jax.Array
  query_start_loc: jax.Array
  request_distribution: jax.Array
  mamba_state_indices: jax.Array | None = None
  block_tables: jax.Array | None = None
  padded_num_reqs: int = -1


def _make_config(overrides: dict | None = None):
  argv = [
      "",
      f"{MAXTEXT_PKG_DIR}/configs/base.yml",
      "run_name=olmoe3_vllm_kda_test",
      "model_name=olmoe3-30m",
      "override_model_config=True",
      "decoder_block=olmoe3",
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=16",
      "base_num_decoder_layers=2",
      "first_num_dense_layers=1",
      "base_mlp_dim=128",
      "base_moe_mlp_dim=64",
      "base_shared_expert_mlp_dim=64",
      "num_experts=8",
      "num_experts_per_tok=2",
      "moe_expert_input_dim=32",
      "inhomogeneous_layer_cycle_interval=2",
      "gdn_num_key_heads=4",
      "gdn_num_value_heads=4",
      "gdn_key_head_dim=16",
      "gdn_value_head_dim=32",
      "gdn_conv_kernel_dim=4",
      "gdn_chunk_size=16",
      "gdn_state_dtype=float32",
      "kda_matmul_precision=highest",
      "kda_inverse_precision=highest",
      "kda_pallas_fused=False",
      "kda_flat_head_layout=False",
      "kda_fused_input_proj=False",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "scan_layers=False",
      "per_device_batch_size=1",
      "max_target_length=256",
      "enable_checkpointing=False",
  ]
  if overrides:
    for k, v in overrides.items():
      argv.append(f"{k}={v}")
  return pyconfig.initialize(argv, log_config=False)


class OLMoE3VllmKdaTest(unittest.TestCase):

  def _make_kda_and_caches(self, cfg, num_blocks: int = 8):
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    rngs = nnx.Rngs(params=0)
    kda = olmoe3.OLMoE3KimiDeltaAttention(cfg, mesh, quant=None, rngs=rngs)
    key_dim = cfg.gdn_num_key_heads * cfg.gdn_key_head_dim
    val_dim = cfg.gdn_num_value_heads * cfg.gdn_value_head_dim
    conv_dim = 2 * key_dim + val_dim
    conv_state = jnp.zeros((num_blocks, cfg.gdn_conv_kernel_dim - 1, conv_dim), dtype=jnp.float32)
    rec_state = jnp.zeros(
        (num_blocks, cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim), dtype=jnp.float32
    )
    return kda, (conv_state, rec_state)

  def test_prefill_then_decode_and_mixed_batch_exact_parity(self):
    """Multi-length prefill + multi-step decode + chunked/mixed batch matches full-seq KDA."""
    cfg = _make_config()
    kda, kv_cache = self._make_kda_and_caches(cfg, num_blocks=10)

    key = jax.random.PRNGKey(123)
    # 3 independent sequences of varying lengths spanning <1, >1, and >2 chunks (C=64).
    seq_lens_full = [23, 75, 138]
    seqs = []
    refs = []
    for i, length in enumerate(seq_lens_full):
      key, sub = jax.random.split(key)
      x_i = jax.random.normal(sub, (1, length, cfg.emb_dim), dtype=jnp.float32) * 0.5
      seqs.append(x_i)
      refs.append(kda(x_i, decoder_segment_ids=None)[0])  # [length, emb_dim]

    # Put canary non-zero garbage in unused slots (0 = null_block, 9 = sentinel) to verify no clobber.
    conv_state, rec_state = kv_cache
    canary_c = jnp.full_like(conv_state[9], 7.25)
    canary_r = jnp.full_like(rec_state[9], -3.5)
    conv_state = conv_state.at[0].set(1.5).at[9].set(canary_c)
    rec_state = rec_state.at[0].set(2.5).at[9].set(canary_r)
    kv_cache = (conv_state, rec_state)

    # Step 1: Prefill req 0 (first 15 tokens) and req 1 (first 70 tokens) in one ragged prefill batch.
    # Also prefill req 2 in two chunks (first 67 tokens here, next 65 tokens in Step 2 mixed batch).
    p0, p1, p2_a = 15, 70, 67
    x_prefill = jnp.concatenate([seqs[0][0, :p0], seqs[1][0, :p1], seqs[2][0, :p2_a]], axis=0)
    # Pad total tokens slightly to simulate vLLM token bucketing.
    pad_toks = 5
    x_prefill_pad = jnp.pad(x_prefill, ((0, pad_toks), (0, 0)))[:, None, :]  # [T, 1, D]
    md_prefill = _FakeAttentionMetadata(
        input_positions=jnp.zeros((x_prefill_pad.shape[0],), dtype=jnp.int32),
        seq_lens=jnp.array([p0, p1, p2_a, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, p0, p0 + p1, p0 + p1 + p2_a, p0 + p1 + p2_a], dtype=jnp.int32),
        request_distribution=jnp.array([0, 3, 3], dtype=jnp.int32),
        mamba_state_indices=jnp.array([3, 5, 2, 0], dtype=jnp.int32),
        padded_num_reqs=4,
    )
    out_p, kv_cache = kda(x_prefill_pad, kv_cache=kv_cache, attention_metadata=md_prefill)
    out_p = out_p[:, 0, :]

    np.testing.assert_allclose(out_p[:p0], refs[0][:p0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_p[p0 : p0 + p1], refs[1][:p1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_p[p0 + p1 : p0 + p1 + p2_a], refs[2][:p2_a], rtol=1e-4, atol=1e-4)
    # Canary slot untouched!
    np.testing.assert_array_equal(kv_cache[0][9], canary_c)
    np.testing.assert_array_equal(kv_cache[1][9], canary_r)

    # Step 2: Mixed batch!
    #   r=0: 1-token decode for req 0 (token p0)
    #   r=1: 1-token decode for req 1 (token p1)
    #   r=2: chunked prefill continuation for req 2 (tokens p2_a .. p2_a + 65)
    #   r=3: inactive padded slot
    p2_b = 65
    x_mixed = jnp.concatenate(
        [seqs[0][0, p0 : p0 + 1], seqs[1][0, p1 : p1 + 1], seqs[2][0, p2_a : p2_a + p2_b]], axis=0
    )
    x_mixed_pad = jnp.pad(x_mixed, ((0, 3), (0, 0)))[:, None, :]
    md_mixed = _FakeAttentionMetadata(
        input_positions=jnp.zeros((x_mixed_pad.shape[0],), dtype=jnp.int32),
        seq_lens=jnp.array([p0 + 1, p1 + 1, p2_a + p2_b, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2, 2 + p2_b, 2 + p2_b], dtype=jnp.int32),
        request_distribution=jnp.array([2, 2, 3], dtype=jnp.int32),
        mamba_state_indices=jnp.array([3, 5, 2, 0], dtype=jnp.int32),
        padded_num_reqs=4,
    )
    out_m, kv_cache = kda(x_mixed_pad, kv_cache=kv_cache, attention_metadata=md_mixed)
    out_m = out_m[:, 0, :]

    np.testing.assert_allclose(out_m[0], refs[0][p0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_m[1], refs[1][p1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_m[2 : 2 + p2_b], refs[2][p2_a : p2_a + p2_b], rtol=1e-4, atol=1e-4)

    # Step 3: Pure autoregressive decode steps for remaining tokens of req 0, req 1, req 2!
    cur = [p0 + 1, p1 + 1, p2_a + p2_b]
    slots = [3, 5, 2]
    while any(cur[i] < seq_lens_full[i] for i in range(3)):
      active = [i for i in range(3) if cur[i] < seq_lens_full[i]]
      toks = [seqs[i][0, cur[i] : cur[i] + 1] for i in active]
      x_dec = jnp.concatenate(toks, axis=0)
      n_act = len(active)
      # Pad to 4 slots to test masked scatter.
      x_dec_pad = jnp.pad(x_dec, ((0, 4 - n_act), (0, 0)))[:, None, :]
      qsl = list(range(n_act + 1)) + [n_act] * (4 - n_act)
      sl = [cur[i] + 1 for i in active] + [0] * (4 - n_act)
      si = [slots[i] for i in active] + [0] * (4 - n_act)
      md_dec = _FakeAttentionMetadata(
          input_positions=jnp.zeros((4,), dtype=jnp.int32),
          seq_lens=jnp.array(sl, dtype=jnp.int32),
          query_start_loc=jnp.array(qsl, dtype=jnp.int32),
          request_distribution=jnp.array([n_act, n_act, n_act], dtype=jnp.int32),
          mamba_state_indices=jnp.array(si, dtype=jnp.int32),
          padded_num_reqs=4,
      )
      out_d, kv_cache = kda(x_dec_pad, kv_cache=kv_cache, attention_metadata=md_dec)
      out_d = out_d[:, 0, :]
      for idx_in_batch, req_i in enumerate(active):
        np.testing.assert_allclose(out_d[idx_in_batch], refs[req_i][cur[req_i]], rtol=1e-4, atol=1e-4)
        cur[req_i] += 1

    # Canary slot STILL untouched after full prefill + mixed + autoregressive decode!
    np.testing.assert_array_equal(kv_cache[0][9], canary_c)
    np.testing.assert_array_equal(kv_cache[1][9], canary_r)

  def test_decoder_layer_inference_skips_emo_and_aux_loss(self):
    """OLMoE3DecoderLayer in inference mode routes kv_cache through KDA and does not sow aux losses."""
    cfg = _make_config({
        "model_call_mode": "inference",
        "emo_enabled": True,
        "emo_min_document_expert_pool": 2,
        "emo_max_document_expert_pool": 8,
        "emo_eval_document_expert_pool": 8,
        "load_balance_loss_weight": 0.01,
        "first_num_dense_layers": 0,  # Make layer 0 MoE + KDA.
    })
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    rngs = nnx.Rngs(params=0)
    layer = olmoe3.OLMoE3DecoderLayer(cfg, mesh, model_mode="autoregressive", layer_idx=0, rngs=rngs)

    key_dim = cfg.gdn_num_key_heads * cfg.gdn_key_head_dim
    val_dim = cfg.gdn_num_value_heads * cfg.gdn_value_head_dim
    kv_cache = (
        jnp.zeros((4, cfg.gdn_conv_kernel_dim - 1, 2 * key_dim + val_dim), dtype=jnp.float32),
        jnp.zeros((4, cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim), dtype=jnp.float32),
    )
    x = jax.random.normal(jax.random.PRNGKey(7), (4, 1, cfg.emb_dim), dtype=jnp.float32)
    md = _FakeAttentionMetadata(
        input_positions=jnp.arange(4, dtype=jnp.int32),
        seq_lens=jnp.array([4, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4, 4], dtype=jnp.int32),
        request_distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        mamba_state_indices=jnp.array([1, 0], dtype=jnp.int32),
        padded_num_reqs=2,
    )
    out, new_kv = layer(
        x,
        decoder_segment_ids=None,
        decoder_positions=jnp.arange(4, dtype=jnp.int32)[:, None],
        deterministic=True,
        model_mode="autoregressive",
        kv_cache=kv_cache,
        attention_metadata=md,
    )
    self.assertEqual(out.shape, x.shape)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    self.assertIsInstance(new_kv, tuple)
    self.assertEqual(len(new_kv), 2)
    # Slot 1 was updated from zeros.
    self.assertGreater(float(jnp.max(jnp.abs(new_kv[1][1]))), 0.0)
    # No nnx.Intermediate aux loss sown during inference.
    intermediates = nnx.state(layer, nnx.Intermediate)
    self.assertEqual(len(jax.tree_util.tree_leaves(intermediates)), 0)

  def test_tunix_adapter_noop_mappings_without_hf_config(self):
    """TunixMaxTextAdapter(use_no_op_mappings=True) succeeds for OLMo 3.5 models absent from HF_MODEL_CONFIGS."""
    fake_base = SimpleNamespace(config=SimpleNamespace(model_name="olmo35-tiny-e256k8-sh2x"))
    adapter = TunixMaxTextAdapter(base_model=fake_base, use_no_op_mappings=True)
    self.assertEqual(adapter.to_hf_mappings(), {})
    self.assertEqual(adapter.to_hf_transpose_keys(), {})
    self.assertEqual(adapter.to_hf_hook_fns(), {})
    self.assertEqual(adapter.lora_to_hf_mappings(), {})

  def test_hybrid_cache_layout_for_olmo35(self):
    """build_qwen_gdn_cache_layout matches OLMo 3.5 KDA state geometry."""
    cfg = _make_config({
        "gdn_num_key_heads": 8,
        "gdn_num_value_heads": 8,
        "gdn_key_head_dim": 128,
        "gdn_value_head_dim": 256,
        "gdn_conv_kernel_dim": 4,
    })

    class _FakeDtype:
      def __init__(self, sz):
        self._sz = sz

      def element_size(self):
        return self._sz

    class _FakeTorch:
      bfloat16 = _FakeDtype(2)
      float32 = _FakeDtype(4)

      @staticmethod
      def empty(_shape, dtype):
        return dtype

    shapes, dtypes, page_bytes = build_qwen_gdn_cache_layout(cfg, _FakeTorch)
    self.assertEqual(shapes, ((3, 2 * 8 * 128 + 8 * 256), (8, 128, 256)))
    self.assertEqual(dtypes, (_FakeTorch.bfloat16, _FakeTorch.float32))
    self.assertEqual(page_bytes, 3 * 4096 * 2 + 8 * 128 * 256 * 4)

  def test_latent_moe_fused_matmul_restores_topk_weight_scale(self):
    """OLMoE3LatentRoutedMoE.fused_moe_matmul restores the * num_experts_per_tok scale."""
    from unittest import mock
    from maxtext.layers import moe

    cfg = _make_config({
        "model_call_mode": "inference",
        "num_experts": 8,
        "num_experts_per_tok": 4,
        "emo_enabled": True,
        "emo_min_document_expert_pool": 4,
        "emo_max_document_expert_pool": 8,
        "emo_eval_document_expert_pool": 8,
    })
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    block = olmoe3.OLMoE3LatentRoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=lambda _rng, shape, dtype, *_: jnp.zeros(shape, dtype=dtype),
        kernel_axes=(None, None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(params=0),
        gate_in_features=cfg.emb_dim,
    )
    raw_out = jnp.ones((2, 3, cfg.moe_expert_input_dim), dtype=jnp.bfloat16) * 0.25
    with mock.patch.object(moe.RoutedMoE, "fused_moe_matmul", return_value=(raw_out, None, None)):
      out, lb, bias_upd = block.fused_moe_matmul(None, None, None)
    self.assertIsNone(lb)
    self.assertIsNone(bias_upd)
    self.assertEqual(out.dtype, jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(out, dtype=np.float32), 1.0, atol=1e-3)

    # Guard when EMo eval pool is strictly smaller than num_experts.
    cfg_emo_sub = _make_config({
        "model_call_mode": "inference",
        "num_experts": 8,
        "num_experts_per_tok": 4,
        "emo_enabled": True,
        "emo_min_document_expert_pool": 4,
        "emo_max_document_expert_pool": 8,
        "emo_eval_document_expert_pool": 6,
    })
    block_sub = olmoe3.OLMoE3LatentRoutedMoE(
        config=cfg_emo_sub,
        num_experts=cfg_emo_sub.num_experts,
        num_experts_per_tok=cfg_emo_sub.num_experts_per_tok,
        mesh=mesh,
        kernel_init=lambda _rng, shape, dtype, *_: jnp.zeros(shape, dtype=dtype),
        kernel_axes=(None, None),
        intermediate_dim=cfg_emo_sub.moe_mlp_dim,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(params=0),
        gate_in_features=cfg_emo_sub.emb_dim,
    )
    with self.assertRaises(NotImplementedError):
      block_sub.fused_moe_matmul(None, None, None)

  def test_create_models_and_meshes_clone_detaches_buffers(self):
    """load_checkpoint_only_once=True detaches actor jax.Array buffers from reference_model."""
    from unittest import mock
    from maxtext.utils import model_creation_utils

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))

    class _TinyBase(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((4, 4), dtype=jnp.bfloat16))
        self.scale = nnx.Param(jnp.ones((4,), dtype=jnp.float32))
        self.config = SimpleNamespace(model_name="olmo35-tiny-e256k8-sh2x")

    ref_adapter = TunixMaxTextAdapter(base_model=_TinyBase(), use_no_op_mappings=True)
    trainer_cfg = SimpleNamespace(
        load_checkpoint_only_once=True,
        vllm_additional_config='{"maxtext_config": {}}',
    )
    sampler_cfg = SimpleNamespace(mesh_axes=("data",))

    with (
        mock.patch.object(model_creation_utils, "from_pretrained", return_value=(ref_adapter, mesh)),
        mock.patch.object(model_creation_utils.maxtext_utils, "create_device_mesh", return_value=np.array(jax.devices()[:1])),
    ):
      ref_model, _, actor_model, _, _ = model_creation_utils.create_models_and_meshes(
          trainer_cfg, sampler_cfg, jax.devices()[:1], jax.devices()[:1]
      )

    # Buffers must be distinct objects so deleting actor/rollout buffers never deletes reference_model.
    self.assertIsNot(ref_model.base.w[...], actor_model.base.w[...])
    self.assertIsNot(ref_model.base.scale[...], actor_model.base.scale[...])
    actor_model.base.scale[...].delete()
    self.assertFalse(ref_model.base.scale[...].is_deleted())
    np.testing.assert_array_equal(np.asarray(ref_model.base.scale[...]), np.ones((4,), dtype=np.float32))


if __name__ == "__main__":
  unittest.main()
