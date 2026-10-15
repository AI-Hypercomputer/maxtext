# Copyright 2023–2026 Google LLC
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

"""Tests the MaxText DeepSeek-V4.1 modules against the plain-PyTorch reference in
tests/utils/deepseek_v41_reference (adapted from the upstream DeepSeek-V4.1-Flash inference code).

The engram tests need the DeepSeek-V4.1 tokenizer: set DEEPSEEK_V41_TOKENIZER to a local directory holding
tokenizer.json / tokenizer_config.json (defaults to the HuggingFace repo id).

  python -m pytest tests/unit/deepseek_v41_vs_reference_test.py
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.configs import pyconfig
from maxtext.layers import attention_deepseek41
from maxtext.layers import engram_deepseek41
from maxtext.models import deepseek41
from tests.utils.deepseek_v41_reference import model as ref_model
from tests.utils.deepseek_v41_reference import engram as ref_engram
from tests.utils.deepseek_v41_reference import runner as ref_runner

TOKENIZER = os.environ.get("DEEPSEEK_V41_TOKENIZER", "deepseek-ai/DeepSeek-V4.1-Flash")

# A tiny model with every DeepSeek-V4.1 layer type:
#   0: sliding window      1: sliding window + engram
#   2: r2 KV/index source + engram   3: r2 consumer
#   4: r1 KV/index/candidate source  5: r1 consumer
#   6: r1 index source (uses candidates)   7: r1 consumer
SEQ = 40
COMPRESS_RATIOS = [0, 0, 2, 2, 1, 1, 1, 1]
KV_SOURCES = [2, 4]
INDEX_SOURCES = [2, 4, 6]
CANDIDATE_SOURCE = 4
ENGRAM_LAYERS = [1, 2]

REF_ARGS = dict(
    dtype="fp32",
    max_batch_size=2,
    max_seq_len=64,
    vocab_size=129280,
    dim=64,
    moe_inter_dim=32,
    n_layers=len(COMPRESS_RATIOS),
    n_mtp_layers=0,
    n_heads=4,
    n_routed_experts=4,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=16,
    head_dim=32,
    rope_head_dim=8,
    norm_eps=1e-20,
    o_groups=2,
    o_lora_rank=8,
    window_size=8,
    compress_ratios=tuple(COMPRESS_RATIOS),
    kv_source_layers=tuple(KV_SOURCES),
    index_source_layers=tuple(INDEX_SOURCES),
    compress_rope_theta=160000.0,
    original_seq_len=65536,
    rope_theta=10000.0,
    rope_factor=16,
    beta_fast=32,
    beta_slow=1,
    index_n_heads=16,
    index_head_dim=16,
    index_topk=3,
    candidate_source_layer=CANDIDATE_SOURCE,
    candidate_topk_blocks=2,
    candidate_block_size=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=tuple(ENGRAM_LAYERS),
    engram_max_ngram_size=3,
    engram_vocab_size=50,
    engram_n_heads=2,
    engram_head_dim=8,
    engram_pad_id=2,
    engram_compressed_vocab_size=99092,
    dspark_block_size=0,
)
REF_ARGS["engram_num_embeddings"] = tuple(
    sum(sum(per_ngram) for per_ngram in layer)
    for layer in engram_deepseek41.engram_primes(
        ENGRAM_LAYERS, REF_ARGS["engram_max_ngram_size"], REF_ARGS["engram_n_heads"], REF_ARGS["engram_vocab_size"]
    )
)


def setUpModule():
  jax.config.update("jax_default_matmul_precision", "highest")


def _maxtext_config(**overrides):
  args = dict(
      run_name="deepseek41_test",
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
      model_name="deepseek4.1-flash-mini",
      override_model_config=True,
      per_device_batch_size=1.0,
      max_target_length=SEQ,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      base_emb_dim=REF_ARGS["dim"],
      base_num_query_heads=REF_ARGS["n_heads"],
      head_dim=REF_ARGS["head_dim"],
      qk_rope_head_dim=REF_ARGS["rope_head_dim"],
      q_lora_rank=REF_ARGS["q_lora_rank"],
      o_groups=REF_ARGS["o_groups"],
      o_lora_rank=REF_ARGS["o_lora_rank"],
      base_mlp_dim=REF_ARGS["moe_inter_dim"],
      base_moe_mlp_dim=REF_ARGS["moe_inter_dim"],
      num_experts=REF_ARGS["n_routed_experts"],
      num_experts_per_tok=REF_ARGS["n_activated_experts"],
      sparse_matmul=False,
      capacity_factor=-1,
      sliding_window_size=REF_ARGS["window_size"],
      base_num_decoder_layers=len(COMPRESS_RATIOS),
      compress_ratios=COMPRESS_RATIOS,
      kv_source_layers=KV_SOURCES,
      index_source_layers=INDEX_SOURCES,
      indexer_n_heads=REF_ARGS["index_n_heads"],
      indexer_head_dim=REF_ARGS["index_head_dim"],
      indexer_topk=REF_ARGS["index_topk"],
      candidate_source_layer=CANDIDATE_SOURCE,
      candidate_topk_blocks=REF_ARGS["candidate_topk_blocks"],
      candidate_block_size=REF_ARGS["candidate_block_size"],
      ds41_engram_layers=ENGRAM_LAYERS,
      ds41_engram_vocab_size=REF_ARGS["engram_vocab_size"],
      ds41_engram_max_ngram_size=REF_ARGS["engram_max_ngram_size"],
      ds41_engram_n_heads=REF_ARGS["engram_n_heads"],
      ds41_engram_head_dim=REF_ARGS["engram_head_dim"],
      tokenizer_path=TOKENIZER,
  )
  args.update(overrides)
  return pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml"], **args)


def _t2j(t: torch.Tensor) -> jnp.ndarray:
  return jnp.asarray(t.detach().float().numpy())


def _j2np(x) -> np.ndarray:
  return np.asarray(jax.device_get(x), dtype=np.float32)


def _set(param, value):
  assert param[...].shape == value.shape, (param[...].shape, value.shape)
  param[...] = value


def _copy_linear(mt_dense, pt_linear, out_shape=None):
  """torch [out, in] -> DenseGeneral kernel [in, *out_shape]."""
  w = _t2j(pt_linear.weight).T
  if out_shape is not None:
    w = w.reshape((w.shape[0],) + tuple(out_shape))
  _set(mt_dense.kernel, w)


def _copy_norm(mt_norm, pt_norm):
  _set(mt_norm.scale, _t2j(pt_norm.weight))


def copy_layer(mt: deepseek41.DeepSeek41DecoderLayer, ref: ref_model.Block):
  """Copies a reference Block's weights into a MaxText DeepSeek41DecoderLayer."""
  _copy_norm(mt.pre_self_attention_layer_norm, ref.attn_norm)
  _copy_norm(mt.post_self_attention_layer_norm, ref.ffn_norm)
  for name in ("hc_attn", "hc_ffn"):
    _set(getattr(mt, f"{name}_fn"), _t2j(getattr(ref, f"{name}_fn")).T)
    _set(getattr(mt, f"{name}_base"), _t2j(getattr(ref, f"{name}_base")))
    _set(getattr(mt, f"{name}_scale"), _t2j(getattr(ref, f"{name}_scale")))

  a, ra = mt.self_attention, ref.attn
  _copy_linear(a.wq_a, ra.wq_a)
  _copy_norm(a.q_norm, ra.q_norm)
  _copy_linear(a.wq_b, ra.wq_b, (ra.n_heads, ra.head_dim))
  _copy_linear(a.wkv, ra.wkv)
  _copy_norm(a.kv_norm, ra.kv_norm)
  g = ra.n_groups
  _set(a.o_a_proj.kernel, _t2j(ra.wo_a.weight).reshape(g, ra.o_lora_rank, -1).transpose(0, 2, 1))
  _copy_linear(a.o_b_proj, ra.wo_b)
  _set(a.sinks, _t2j(ra.attn_sink))
  if ra.compressor is not None:
    _copy_linear(a.compressor.wkv, ra.compressor.wkv)
    if ra.compressor.compress_ratio > 1:
      _copy_linear(a.compressor.wgate, ra.compressor.wgate)
    _copy_norm(a.compressor.norm, ra.compressor.norm)
  if ra.indexer is not None:
    _copy_linear(a.indexer.wq_b, ra.indexer.wq_b, (ra.indexer.n_heads, ra.indexer.index_head_dim))
    _copy_linear(a.indexer.weights_proj, ra.indexer.weights_proj)
    if ra.indexer.owns_k:
      _copy_linear(a.indexer.wk, ra.indexer.wk)
      _copy_norm(a.indexer.k_norm, ra.indexer.k_norm)

  moe_block, rf = mt.mlp.MoeBlock_0, ref.ffn
  _set(moe_block.gate.kernel, _t2j(rf.gate.weight).T)
  moe_block.gate.bias[...] = _t2j(rf.gate.bias)
  _set(moe_block.wi_0, jnp.stack([_t2j(e.w1.weight).T for e in rf.experts]))
  _set(moe_block.wi_1, jnp.stack([_t2j(e.w3.weight).T for e in rf.experts]))
  _set(moe_block.wo, jnp.stack([_t2j(e.w2.weight).T for e in rf.experts]))
  _copy_linear(mt.mlp.shared_experts.wi_0, rf.shared_experts.w1)
  _copy_linear(mt.mlp.shared_experts.wi_1, rf.shared_experts.w3)
  _copy_linear(mt.mlp.shared_experts.wo, rf.shared_experts.w2)

  if ref.engram is not None:
    table = _t2j(ref.engram.embed.weight)
    rows = mt.engram.embed[...].shape[0]
    _set(mt.engram.embed, jnp.pad(table, ((0, rows - table.shape[0]), (0, 0))))
    _copy_linear(mt.engram.wkv, ref.engram.wkv)
    _set(mt.engram.q_weight, _t2j(ref.engram.q_weight))
    _set(mt.engram.k_weight, _t2j(ref.engram.k_weight))


def _init_reference(model: torch.nn.Module, seed: int = 0):
  """Random weights in a realistic range (norm scales around 1)."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      r = torch.randn(p.shape, generator=g)
      if name.endswith("norm.weight") or name.endswith(("q_weight", "k_weight")):
        p.copy_(1.0 + 0.1 * r)
      elif name.endswith("attn_sink"):
        p.copy_(0.5 * r)
      elif name.endswith(("_base",)):
        p.copy_(0.1 * r)
      elif name.endswith(("_scale",)) and "hc_" in name:
        p.copy_(0.5 + 0.1 * r)
      elif name.endswith("gate.bias"):
        p.copy_(0.1 * r)
      elif "embed" in name:
        p.copy_(r)
      else:
        p.copy_(r / p.shape[-1] ** 0.5)


@pytest.mark.cpu_only
class DeepSeek41RotaryTest(unittest.TestCase):
  """YaRN / plain interleaved partial RoPE against `precompute_freqs_cis` + `apply_rotary_emb`."""

  def _check(self, original_seq_len, theta):
    head_dim, rd, seq = 32, 8, 50
    freqs_cis = ref_model.precompute_freqs_cis(rd, seq, original_seq_len, theta, 16, 32, 1)
    x = torch.randn(2, seq, 3, head_dim)
    ref = x.clone()
    ref_model.apply_rotary_emb(ref[..., -rd:], freqs_cis)
    rope = attention_deepseek41.DeepSeek41RotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=rd / head_dim,
        rope_theta=theta,
        original_seq_len=original_seq_len,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
    )
    positions = jnp.broadcast_to(jnp.arange(seq), (2, seq))
    out = rope(_t2j(x), positions, unsqueeze_dim=-2)
    np.testing.assert_allclose(_j2np(out), ref.numpy(), rtol=1e-5, atol=1e-5)
    back = rope(out, positions, unsqueeze_dim=-2, reverse=True)
    np.testing.assert_allclose(_j2np(back), x.numpy(), rtol=1e-5, atol=1e-5)

  def test_yarn(self):
    self._check(65536, 160000.0)

  def test_plain(self):
    self._check(0, 10000.0)


@pytest.mark.cpu_only
class DeepSeek41CandidateBlocksTest(unittest.TestCase):

  def test_select_candidate_blocks(self):
    torch.manual_seed(0)
    s, c = 30, 30
    logits = torch.randn(2, s, c)
    lens = (torch.arange(1, s + 1)).unsqueeze(-1)
    logits = logits.masked_fill(torch.arange(c) >= lens, float("-inf"))
    ref = ref_model.select_candidate_blocks(logits, lens, 3, 4)
    out = attention_deepseek41.select_candidate_blocks(_t2j(logits), jnp.broadcast_to(jnp.arange(1, s + 1), (2, s)), 3, 4)
    np.testing.assert_array_equal(np.asarray(out), ref.numpy())


@pytest.mark.cpu_only
class DeepSeek41EngramHashTest(unittest.TestCase):

  def test_hash_ids_match_reference(self):
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    args = ref_model.ModelArgs(**{**REF_ARGS, "engram_vocab_size": 16_000_000})  # real bucket sizes (< 2**24)
    layout = ref_engram.EngramLayout.from_args(args)
    state = ref_engram.NgramHashState(args, layout, tokenizer)
    ids = torch.randint(0, 129280, (2, SEQ))
    ref = state(ids, 0)  # [B, S, n_layers, cols]

    token_map, vocab = engram_deepseek41.build_compressed_token_map(tokenizer)
    self.assertEqual(vocab, args.engram_compressed_vocab_size)
    for li, layer_id in enumerate(args.engram_layer_ids):
      hasher = engram_deepseek41.DeepSeek41NgramHasher(
          token_map,
          vocab,
          args.engram_pad_id,
          args.engram_layer_ids,
          layer_id,
          args.engram_max_ngram_size,
          args.engram_n_heads,
          args.engram_vocab_size,
      )
      out = hasher(jnp.asarray(ids.numpy()))
      np.testing.assert_array_equal(np.asarray(out), ref[:, :, li].numpy())


@pytest.mark.cpu_only
class DeepSeek41DequantTest(unittest.TestCase):
  """The MaxText loader hook and the reference loader dequantize the released FP8 / FP4 formats identically."""

  def _check(self, weight, scale, expected):
    from maxtext.checkpoint_conversion.utils import param_mapping  # pylint: disable=import-outside-toplevel

    out = torch.empty(expected.shape, dtype=torch.bfloat16)
    ref_runner.dequantize_into(out, weight, scale, rows_per_chunk=64)
    np.testing.assert_array_equal(out.float().numpy(), expected.numpy())
    w_j = jnp.asarray(weight.view(torch.int8).numpy()) if weight.dtype == torch.int8 else None
    if w_j is None:
      w_j = jnp.asarray(weight.float().numpy()).astype(jnp.float8_e4m3fn)
    s_j = jnp.asarray(scale.float().numpy()).astype(jnp.float8_e8m0fnu)
    got = param_mapping.dequantize_block_scaled((w_j, s_j))
    self.assertEqual(got.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(_j2np(got), expected.numpy())
    # As the streaming loader hands them over: the weight split by rows over the chips (including splits whose
    # pieces line up with the scale blocks), the scales either a full copy or split the same way.
    devices = jax.devices()
    for n in sorted({d for d in (2, len(devices) // 2, len(devices)) if d > 1 and w_j.shape[0] % d == 0}):
      w_s = jax.device_put(w_j, hf_streaming_load.choose_load_sharding(w_j.shape, w_j.dtype, devices[:n], 0))
      for split_scale in (False, True):
        if split_scale and s_j.shape[0] != w_j.shape[0]:
          continue
        s_s = jax.device_put(
            s_j,
            hf_streaming_load.choose_load_sharding(s_j.shape, s_j.dtype, devices[:n], 0 if split_scale else 1 << 62),
        )
        got = param_mapping.dequantize_block_scaled((w_s, s_s))
        self.assertEqual(got.sharding, w_s.sharding)
        np.testing.assert_array_equal(_j2np(got), expected.numpy(), err_msg=f"{n} chips, split scales {split_scale}")

  def test_fp8_block_32x32(self):
    torch.manual_seed(0)
    weight = (torch.randn(96, 160) * 4).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-8, 4, (3, 5)).float()).to(torch.float8_e8m0fnu)
    expected = weight.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    self._check(weight, scale, expected)

  def test_fp8_block_32x32_shards_aligned_to_blocks(self):
    # 256 rows over 2 / 4 / 8 chips: 128 / 64 / 32 rows each, whole scale blocks (the 64-chip v5p `wq_b` case).
    torch.manual_seed(3)
    weight = (torch.randn(256, 96) * 4).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-8, 4, (8, 3)).float()).to(torch.float8_e8m0fnu)
    expected = weight.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    self._check(weight, scale, expected)

  def test_fp8_rows_1x32(self):
    torch.manual_seed(1)
    weight = torch.randn(200, 64).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-8, 4, (200, 2)).float()).to(torch.float8_e8m0fnu)
    expected = weight.float() * scale.float().repeat_interleave(32, 1)
    self._check(weight, scale, expected)

  def test_fp4_packed(self):
    torch.manual_seed(2)
    codes = torch.randint(0, 16, (64, 128))
    table = ref_runner._FP4_E2M1_VALUES  # pylint: disable=protected-access
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    scale = torch.exp2(torch.randint(-8, 4, (64, 4)).float()).to(torch.float8_e8m0fnu)
    expected = table[codes] * scale.float().repeat_interleave(32, 1)
    self._check(packed, scale, expected)


@pytest.mark.cpu_only
class DeepSeek41LayersTest(unittest.TestCase):
  """Runs every layer type of a tiny model, layer by layer, against the reference blocks."""

  @classmethod
  def setUpClass(cls):
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    torch.manual_seed(0)
    cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    cls.ref = ref_runner.build_model(ref_model.ModelArgs(**REF_ARGS), cls.tokenizer)
    _init_reference(cls.ref)
    cls.config = _maxtext_config()

  def test_layers_match_reference(self):
    ref = self.ref
    ids = torch.randint(0, REF_ARGS["vocab_size"], (2, SEQ))
    with torch.inference_mode():
      hashes = ref.engram_hash(ids, 0)
      h = ref.embed(ids).unsqueeze(2).repeat(1, 1, REF_ARGS["hc_mult"], 1)
      pre_mix = ref_model.make_identity_pre_mix(h, REF_ARGS["hc_mult"])
      ref_outs = []
      for layer in ref.layers:
        if layer.engram is not None:
          h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
        h, pre_mix = layer(h, 0, pre_mix, None)
        ref_outs.append((h.clone(), pre_mix.clone()))

    specs = deepseek41.build_layer_specs(self.config)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    tokens = jnp.asarray(ids.numpy(), jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(SEQ), (2, SEQ))
    carry = deepseek41.initial_carry(_t2j(ref.embed(ids)), REF_ARGS["hc_mult"])
    with mesh:
      for i, spec in enumerate(specs):
        layer = deepseek41.DeepSeek41DecoderLayer(self.config, mesh, "train", spec, rngs=nnx.Rngs(i))
        copy_layer(layer, ref.layers[i])
        carry = layer(carry, positions, tokens)
        ref_h, ref_pre = ref_outs[i]
        np.testing.assert_allclose(
            _j2np(carry["h"]), ref_h.numpy(), rtol=2e-3, atol=2e-3, err_msg=f"hidden mismatch at layer {i}"
        )
        np.testing.assert_allclose(
            _j2np(carry["pre_mix"]), ref_pre.numpy(), rtol=2e-3, atol=2e-3, err_msg=f"pre_mix mismatch at layer {i}"
        )


# The mini layout (14 layers, each repeating group twice) at tiny widths, so scanned stacks really stack.
MINI_COMPRESS_RATIOS = [0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
MINI_LAYOUT = dict(
    n_layers=len(MINI_COMPRESS_RATIOS),
    compress_ratios=tuple(MINI_COMPRESS_RATIOS),
    kv_source_layers=(2, 4, 6, 8),
    index_source_layers=(2, 4, 6, 8, 10, 12),
    candidate_source_layer=8,
    engram_layer_ids=(1, 6),
)


@pytest.mark.cpu_only
class DeepSeek41CheckpointTest(unittest.TestCase):
  """Saves a tiny random checkpoint in the upstream layout, streams it into MaxText through the param mapping
  (`safetensors_dynamic`), and compares the logits with the reference for scanned and unscanned layers."""

  @classmethod
  def setUpClass(cls):
    import tempfile  # pylint: disable=import-outside-toplevel
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    args = {**REF_ARGS, **MINI_LAYOUT}
    args["engram_num_embeddings"] = tuple(
        sum(sum(per_ngram) for per_ngram in layer)
        for layer in engram_deepseek41.engram_primes(
            args["engram_layer_ids"], args["engram_max_ngram_size"], args["engram_n_heads"], args["engram_vocab_size"]
        )
    )
    cls.ref = ref_runner.build_model(ref_model.ModelArgs(**args), tokenizer)
    _init_reference(cls.ref, seed=1)
    cls.ckpt_dir = tempfile.mkdtemp(prefix="dsv41_tiny_")
    state = {k: v.contiguous() for k, v in cls.ref.state_dict().items()}
    save_file(state, os.path.join(cls.ckpt_dir, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})

  def _maxtext_logits(self, ids, scan_layers):
    from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel

    config = _maxtext_config(
        base_num_decoder_layers=MINI_LAYOUT["n_layers"],
        compress_ratios=MINI_COMPRESS_RATIOS,
        kv_source_layers=list(MINI_LAYOUT["kv_source_layers"]),
        index_source_layers=list(MINI_LAYOUT["index_source_layers"]),
        candidate_source_layer=MINI_LAYOUT["candidate_source_layer"],
        ds41_engram_layers=list(MINI_LAYOUT["engram_layer_ids"]),
        scan_layers=scan_layers,
        load_parameters_path=self.ckpt_dir,
        source_checkpoint_layout="safetensors_dynamic",
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    tokens = jnp.asarray(ids.numpy(), jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(SEQ), tokens.shape)
    logits = model(
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=jnp.ones_like(tokens),
        enable_dropout=False,
    )
    return _j2np(logits)

  def _check(self, scan_layers):
    ids = torch.randint(0, REF_ARGS["vocab_size"], (1, SEQ))
    ref_logits = self.ref(ids, full_logits=True).numpy()
    logits = self._maxtext_logits(ids, scan_layers)[:1]
    np.testing.assert_allclose(logits, ref_logits, rtol=2e-3, atol=2e-3)

  def test_unscanned(self):
    self._check(scan_layers=False)

  def test_scanned(self):
    self._check(scan_layers=True)


if __name__ == "__main__":
  unittest.main()
