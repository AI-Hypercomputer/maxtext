# pylint: disable=missing-function-docstring,use-dict-literal,chained-comparison,unnecessary-lambda-assignment,missing-module-docstring,import-outside-toplevel
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

"""Forward + backward parity of the full MaxText DeepSeek-V4.1 model against the plain-PyTorch reference.

The reference (tests/utils/deepseek_v41_reference, adapted from the upstream DeepSeek-V4.1-Flash inference code)
is differentiated with torch autograd; MaxText with `jax.value_and_grad` through the whole Transformer (embedding,
hyper-connected layers with engram / sliding-window / compressed KV / indexer / candidate blocks, MoE, final
hc collapse + norm + head). Both compute the mean next-token cross entropy of the same random tokens with the same
weights in float32, and every parameter gradient is compared after mapping it into the reference layout.

The upstream code is inference only, so the reference gradient is the exact gradient of the reference forward: the
indexer top-k / candidate selections are discrete, so the indexer projections get no gradient on either side.

  DEEPSEEK_V41_TOKENIZER=<dir with tokenizer.json> python -m pytest tests/unit/deepseek_v41_grad_vs_reference_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import deepseek41
from maxtext.utils import maxtext_utils, model_creation_utils

try:
  import torch
  from tests.utils.deepseek_v41_reference import model as ref_model
  from tests.utils.deepseek_v41_reference import runner as ref_runner
except ImportError:
  torch = None
  ref_model = None
  ref_runner = None

TOKENIZER = os.environ.get("DEEPSEEK_V41_TOKENIZER", "deepseek-ai/DeepSeek-V4.1-Flash")
SEQ = 40
BATCH = 2

# The mini layout (every DeepSeek-V4.1 layer type, repeating groups twice so scanned stacks really stack):
#   0: sliding window   1: sliding window + engram   2-7: r2 (KV/index sources 2, 4; engram on 6)
#   8-13: r1 (KV + index + candidate source 8; index sources 10, 12 restricted to the candidates)
COMPRESS_RATIOS = [0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
KV_SOURCES = [2, 4, 6, 8]
INDEX_SOURCES = [2, 4, 6, 8, 10, 12]
CANDIDATE_SOURCE = 8
ENGRAM_LAYERS = [1, 6]

REF_ARGS = dict(
    dtype="fp32",
    max_batch_size=BATCH,
    max_seq_len=64,
    vocab_size=129280,
    dim=64,
    moe_inter_dim=32,
    n_layers=len(COMPRESS_RATIOS),
    n_mtp_layers=0,
    n_heads=4,
    n_routed_experts=8,
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


def _engram_rows():
  from maxtext.layers import engram_deepseek41  # pylint: disable=import-outside-toplevel

  return tuple(
      sum(sum(per_ngram) for per_ngram in layer)
      for layer in engram_deepseek41.engram_primes(
          ENGRAM_LAYERS,
          REF_ARGS["engram_max_ngram_size"],
          REF_ARGS["engram_n_heads"],
          REF_ARGS["engram_vocab_size"],
      )
  )


def _init_reference(model: torch.nn.Module, seed: int):
  """Random weights in a realistic range (norm scales around 1), as in deepseek_v41_vs_reference_test."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      r = torch.randn(p.shape, generator=g)
      if name.endswith("norm.weight") or name.endswith(("q_weight", "k_weight")):
        p.copy_(1.0 + 0.1 * r)
      elif name.endswith("attn_sink"):
        p.copy_(0.5 * r)
      elif name.endswith("_base"):
        p.copy_(0.1 * r)
      elif name.endswith("_scale") and "hc_" in name:
        p.copy_(0.5 + 0.1 * r)
      elif name.endswith("gate.bias"):
        p.copy_(0.1 * r)
      elif "embed" in name:
        p.copy_(r)
      else:
        p.copy_(r / p.shape[-1] ** 0.5)


def _reference_logits_with_grad(ref: ref_model.Transformer, ids: torch.Tensor) -> torch.Tensor:
  """`Transformer.forward(ids, full_logits=True)` without its `@torch.inference_mode()`, so autograd records it.

  The engram hash ids are integers computed under inference mode upstream; they are cloned into a normal tensor
  because `F.embedding` saves its indices for the backward pass.
  """
  hashes = ref.engram_hash(ids, 0).clone() if ref.engram_hash is not None else None
  h = ref.embed(ids)
  h = h.unsqueeze(2).repeat(1, 1, ref.hc_mult, 1)
  pre_mix = ref_model.make_identity_pre_mix(h, ref.hc_mult)
  layer = None
  for layer in ref.layers:
    if layer.engram is not None:
      h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
    h, pre_mix = layer(h, 0, pre_mix, None)
  h = layer.hc_pre(h, pre_mix)
  return ref.head(ref.norm(h), full_logits=True)


def _maxtext_config(ckpt_dir, scan_layers, **overrides):
  args = dict(
      run_name="deepseek41_grad_test",
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
      scan_layers=scan_layers,
      load_parameters_path=ckpt_dir,
      source_checkpoint_layout="safetensors_dynamic",
  )
  args.update(overrides)
  return pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml"], **args)


def _np(x) -> np.ndarray:
  return np.asarray(jax.device_get(x), dtype=np.float32)


def _xent(logits, targets):
  """Mean next-token cross entropy (float32), identical formula on both sides."""
  logz = jax.nn.logsumexp(logits, axis=-1)
  picked = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
  return jnp.mean(logz - picked)


def _layer_param_views(spec, get):
  """(reference name, MaxText array -> reference-layout array) for one layer; `get(path)` reads a MaxText leaf."""
  i = spec.layer_id
  ref = f"layers.{i}"

  def t(path):  # torch Linear [out, in] <- kernel [in, *out]
    w = get(path)
    return w.reshape(w.shape[0], -1).T

  out = {
      f"{ref}.attn_norm.weight": get("pre_self_attention_layer_norm/scale"),
      f"{ref}.ffn_norm.weight": get("post_self_attention_layer_norm/scale"),
      f"{ref}.attn.wq_a.weight": t("self_attention/wq_a/kernel"),
      f"{ref}.attn.q_norm.weight": get("self_attention/q_norm/scale"),
      f"{ref}.attn.wq_b.weight": t("self_attention/wq_b/kernel"),
      f"{ref}.attn.wkv.weight": t("self_attention/wkv/kernel"),
      f"{ref}.attn.kv_norm.weight": get("self_attention/kv_norm/scale"),
      # [g, d, r] -> [g * r, d]
      f"{ref}.attn.wo_a.weight": np.transpose(get("self_attention/o_a_proj/kernel"), (0, 2, 1)).reshape(
          -1, get("self_attention/o_a_proj/kernel").shape[1]
      ),
      f"{ref}.attn.wo_b.weight": t("self_attention/o_b_proj/kernel"),
      f"{ref}.attn.attn_sink": get("self_attention/sinks"),
      f"{ref}.ffn.gate.weight": t("mlp/MoeBlock_0/gate/kernel"),
      f"{ref}.ffn.shared_experts.w1.weight": t("mlp/shared_experts/wi_0/kernel"),
      f"{ref}.ffn.shared_experts.w3.weight": t("mlp/shared_experts/wi_1/kernel"),
      f"{ref}.ffn.shared_experts.w2.weight": t("mlp/shared_experts/wo/kernel"),
  }
  for name in ("hc_attn", "hc_ffn"):
    out[f"{ref}.{name}_fn"] = get(f"{name}_fn").T
    out[f"{ref}.{name}_base"] = get(f"{name}_base")
    out[f"{ref}.{name}_scale"] = get(f"{name}_scale")
  wi_0, wi_1, wo = (get(f"mlp/MoeBlock_0/{n}") for n in ("wi_0", "wi_1", "wo"))
  for e in range(wi_0.shape[0]):
    out[f"{ref}.ffn.experts.{e}.w1.weight"] = wi_0[e].T
    out[f"{ref}.ffn.experts.{e}.w3.weight"] = wi_1[e].T
    out[f"{ref}.ffn.experts.{e}.w2.weight"] = wo[e].T
  if spec.is_kv_source:
    out[f"{ref}.attn.compressor.wkv.weight"] = t("self_attention/compressor/wkv/kernel")
    if spec.compress_ratio > 1:
      out[f"{ref}.attn.compressor.wgate.weight"] = t("self_attention/compressor/wgate/kernel")
    out[f"{ref}.attn.compressor.norm.weight"] = get("self_attention/compressor/norm/scale")
  if spec.is_index_source:
    out[f"{ref}.attn.indexer.wq_b.weight"] = t("self_attention/indexer/wq_b/kernel")
    out[f"{ref}.attn.indexer.weights_proj.weight"] = t("self_attention/indexer/weights_proj/kernel")
    if spec.is_kv_source:
      out[f"{ref}.attn.indexer.wk.weight"] = t("self_attention/indexer/wk/kernel")
      out[f"{ref}.attn.indexer.k_norm.weight"] = get("self_attention/indexer/k_norm/scale")
  if spec.has_engram:
    out[f"{ref}.engram.embed.weight"] = get("engram/embed")  # padded rows, trimmed by the caller
    out[f"{ref}.engram.wkv.weight"] = t("engram/wkv/kernel")
    out[f"{ref}.engram.q_weight"] = get("engram/q_weight")
    out[f"{ref}.engram.k_weight"] = get("engram/k_weight")
  return out


def maxtext_grads_in_reference_layout(config, flat: dict) -> dict:
  """Maps a flat {'a/b/c': array} MaxText gradient tree onto the reference parameter names and layouts."""
  specs = deepseek41.build_layer_specs(config)
  out = {
      "embed.weight": flat["token_embedder/embedding"],
      "head.weight": flat["decoder/logits_dense/kernel"].T,
      "norm.weight": flat["decoder/decoder_norm/scale"],
  }
  if config.scan_layers:
    for k, segment in enumerate(deepseek41.build_segments(config)):
      for r, group in enumerate(segment.groups):
        for j, layer_id in enumerate(group):
          prefix = f"decoder/segments_{k}/layers_{j}/"
          # Stacked leaves carry the scan axis at param_scan_axis (= 1 for every leaf here).
          get = lambda p, prefix=prefix, r=r: np.take(flat[prefix + p], r, axis=config.param_scan_axis)
          out.update(_layer_param_views(specs[layer_id], get))
  else:
    for spec in specs:
      prefix = f"decoder/layers_{spec.layer_id}/"
      out.update(_layer_param_views(spec, lambda p, prefix=prefix: flat[prefix + p]))
  return out


@pytest.mark.cpu_only
@unittest.skipIf(torch is None, "PyTorch required for upstream reference parity tests")
class DeepSeek41GradParityTest(unittest.TestCase):
  """Loss and every parameter gradient of the full MaxText model vs torch autograd through the reference."""

  @classmethod
  def setUpClass(cls):
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    jax.config.update("jax_default_matmul_precision", "highest")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    args = {**REF_ARGS, "engram_num_embeddings": _engram_rows()}
    cls.ref = ref_runner.build_model(ref_model.ModelArgs(**args), tokenizer)
    _init_reference(cls.ref, seed=7)
    cls.ckpt_dir = tempfile.mkdtemp(prefix="dsv41_grad_")
    state = {k: v.contiguous() for k, v in cls.ref.state_dict().items()}
    save_file(
        state,
        os.path.join(cls.ckpt_dir, "model-00001-of-00001.safetensors"),
        metadata={"format": "pt"},
    )

    # Tokens with lots of repeated n-grams so the engram rows actually collide / get reused.
    g = torch.Generator().manual_seed(11)
    base = torch.randint(0, REF_ARGS["vocab_size"], (BATCH, SEQ // 4), generator=g)
    cls.ids = torch.cat(
        [
            base,
            base,
            torch.randint(0, REF_ARGS["vocab_size"], (BATCH, SEQ // 2), generator=g),
        ],
        dim=1,
    )

    # Reference forward + backward (float32, autograd).
    cls.ref.zero_grad(set_to_none=True)
    with torch.inference_mode():
      inference_logits = cls.ref(cls.ids, full_logits=True)
    logits = _reference_logits_with_grad(cls.ref, cls.ids)  # [B, S, V] float32
    # The grad-enabled replica of the forward must reproduce the upstream (inference-mode) forward exactly.
    np.testing.assert_array_equal(logits.detach().numpy(), inference_logits.numpy())
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        cls.ids[:, 1:].reshape(-1),
    )
    loss.backward()
    cls.ref_loss = float(loss.detach())
    cls.ref_logits = logits.detach().numpy()
    cls.ref_grads = {
        n: (p.grad.detach().float().numpy() if p.grad is not None else np.zeros(p.shape, np.float32))
        for n, p in cls.ref.named_parameters()
    }

  def _maxtext_loss_and_grads(self, scan_layers, **overrides):
    config = _maxtext_config(self.ckpt_dir, scan_layers, **overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    tokens = jnp.asarray(self.ids.numpy(), jnp.int32)
    # The test runs a global batch of per_device_batch_size * num_devices rows; only the first BATCH are scored.
    gbs = config.global_batch_size_to_train_on
    tokens = jnp.concatenate([tokens, jnp.zeros((gbs - BATCH, SEQ), jnp.int32)], axis=0) if gbs > BATCH else tokens
    positions = jnp.broadcast_to(jnp.arange(SEQ), tokens.shape)
    targets = tokens[:BATCH, 1:]
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss_fn(params, rest):
      m = nnx.merge(graphdef, params, rest, copy=True)
      logits = m(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=jnp.ones_like(tokens),
          enable_dropout=False,
      )
      return (
          _xent(logits[:BATCH, :-1].astype(jnp.float32), targets),
          logits[:BATCH],
      )

    with mesh:
      (loss, logits), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params, rest)
    flat = {"/".join(str(k) for k in path): _np(v) for path, v in nnx.to_flat_state(grads)}
    return config, float(loss), _np(logits), flat

  def _check(self, scan_layers, rtol=2e-3, **overrides):
    config, loss, logits, flat = self._maxtext_loss_and_grads(scan_layers, **overrides)
    np.testing.assert_allclose(logits, self.ref_logits, rtol=2e-3, atol=2e-3)
    self.assertAlmostEqual(loss, self.ref_loss, delta=1e-4 * max(1.0, abs(self.ref_loss)))

    mt = maxtext_grads_in_reference_layout(config, flat)
    rows = {}
    failures = []
    for name, g_ref in self.ref_grads.items():
      if name.endswith("gate.bias"):
        continue  # selection-only bias: no gradient upstream (a MoEBiasVar, not a Param, in MaxText)
      self.assertIn(name, mt, f"no MaxText gradient mapped for {name}")
      g = mt[name]
      if name.endswith("engram.embed.weight"):
        self.assertTrue(
            np.all(g[g_ref.shape[0] :] == 0),
            f"{name}: padding rows got a gradient",
        )
        g = g[: g_ref.shape[0]]
      self.assertEqual(g.shape, g_ref.shape, name)
      ref_norm = float(np.linalg.norm(g_ref))
      err = float(np.linalg.norm(g - g_ref))
      rel = err / ref_norm if ref_norm > 0 else err
      rows[name] = (ref_norm, rel)
      if rel > rtol and err > 1e-6:
        failures.append(f"{name}: |g_ref|={ref_norm:.3e} rel_err={rel:.3e}")
    worst = sorted(rows.items(), key=lambda kv: -kv[1][1])[:8]
    print(
        f"\n[scan_layers={scan_layers} {overrides}] loss mt={loss:.6f} ref={self.ref_loss:.6f}; "
        f"{len(rows)} grads compared; worst rel errors:"
    )
    for name, (n, rel) in worst:
      print(f"   {name:60s} |g_ref|={n:.3e} rel={rel:.2e}")
    zero_both = [n for n, (norm, _) in rows.items() if norm == 0.0]
    print(
        f"   params with zero gradient in the reference ({len(zero_both)}): "
        f"{sorted({n.split('.', 2)[-1] for n in zero_both})}"
    )
    self.assertFalse(failures, "gradient mismatches:\n" + "\n".join(failures))

  def test_unscanned_dense_moe(self):
    self._check(scan_layers=False)

  def test_scanned_dense_moe(self):
    self._check(scan_layers=True)

  def test_scanned_sparse_matmul_moe(self):
    # The training path: dropless ragged-dot MoE (megablox is a TPU kernel; ragged_dot runs anywhere).
    self._check(scan_layers=True, sparse_matmul=True, megablox=False)

  def test_scanned_chunked_attention(self):
    # Query-chunked attention (5 chunks of 8 positions, window 8): same softmax over fewer keys, per-chunk remat.
    self._check(scan_layers=True, ds41_attention_chunk=8)

  def test_unscanned_chunked_attention_ragged_moe(self):
    self._check(
        scan_layers=False,
        ds41_attention_chunk=8,
        sparse_matmul=True,
        megablox=False,
    )


class PureJaxTrainingFixesTest(unittest.TestCase):
  """Pure-JAX verification of PR #5812 training fixes (runs on all CI environments without PyTorch)."""

  def test_scatter_bool_matches_3d_scatter_max_and_vjp(self):
    from maxtext.layers import attention_deepseek41  # pylint: disable=import-outside-toplevel

    scores = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 12))
    top_vals, top_idx = jax.lax.top_k(scores, 4)
    vals = top_vals > 0.0

    # Reference 3D advanced scatter-max implementation from base PR #5802:
    bi = jnp.arange(2)[:, None, None]
    si = jnp.arange(16)[None, :, None]
    expected = jnp.zeros((2, 16, 12), dtype=jnp.bool_).at[bi, si, top_idx].max(vals)

    got, vjp_fn = jax.vjp(
        lambda idx, v: attention_deepseek41._scatter_bool(idx, v, 12), top_idx, vals  # pylint: disable=protected-access
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    d_idx, d_vals = vjp_fn(jnp.zeros_like(got))
    self.assertEqual(d_idx.dtype, jax.dtypes.float0)
    self.assertEqual(d_vals.dtype, jax.dtypes.float0)

  def test_sharded_row_lookup_forward_and_grads_match_take(self):
    from maxtext.layers import engram_deepseek41  # pylint: disable=import-outside-toplevel

    devs = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devs).reshape(len(devs), 1), ("fsdp", "expert"))
    rules = (("embed", "fsdp"), ("activation_batch", "fsdp"))

    num_real_rows, padded_rows, head_dim = 40, 48, 8
    table_init = jax.random.normal(jax.random.PRNGKey(1), (padded_rows, head_dim), dtype=jnp.float32)
    # Include deliberate duplicate n-gram IDs across batch/seq/cols within [0, num_real_rows):
    raw_ids = jax.random.randint(jax.random.PRNGKey(2), (len(devs), 6, 4), 0, num_real_rows // 2)
    cotangent = jax.random.normal(jax.random.PRNGKey(3), (len(devs), 6, 4, head_dim), dtype=jnp.float32)

    ref_out, ref_grad = jax.value_and_grad(lambda t: jnp.sum(jnp.take(t, raw_ids, axis=0) * cotangent))(table_init)
    with mesh:
      for method in ("all_to_all", "psum_scatter", "gather"):
        got_val, got_grad = jax.value_and_grad(
            lambda t, m=method: jnp.sum(
                engram_deepseek41.sharded_row_lookup(
                    t,
                    raw_ids,
                    mesh=mesh,
                    rules=rules,
                    out_dtype=jnp.float32,
                    method=m,
                    capacity_factor=4.0,
                    seq_chunk=3,
                    num_rows=num_real_rows,
                )
                * cotangent
            )
        )(table_init)
        self.assertAlmostEqual(float(got_val), float(ref_out), delta=1e-4)
        np.testing.assert_allclose(np.asarray(got_grad), np.asarray(ref_grad), rtol=1e-5, atol=1e-5)

      # Verify non-divisible batch fallback (e.g. single-prompt inference/eval) when multi-device mesh is active:
      single_ids = raw_ids[:1]
      single_cot = cotangent[:1]
      ref_single_val, ref_single_grad = jax.value_and_grad(
          lambda t: jnp.sum(jnp.take(t, single_ids, axis=0) * single_cot)
      )(table_init)
      got_single_val, got_single_grad = jax.value_and_grad(
          lambda t: jnp.sum(
              engram_deepseek41.sharded_row_lookup(
                  t,
                  single_ids,
                  mesh=mesh,
                  rules=rules,
                  out_dtype=jnp.float32,
                  method="all_to_all",
                  capacity_factor=4.0,
                  num_rows=num_real_rows,
              )
              * single_cot
          )
      )(table_init)
      self.assertAlmostEqual(float(got_single_val), float(ref_single_val), delta=1e-4)
      np.testing.assert_allclose(np.asarray(got_single_grad), np.asarray(ref_single_grad), rtol=1e-5, atol=1e-5)

  def test_hc_pre_unrolled_sum_matches_reduce_and_grads(self):
    h_f32 = jax.random.normal(jax.random.PRNGKey(4), (2, 12, 4, 16), dtype=jnp.float32)
    mix = jax.random.normal(jax.random.PRNGKey(5), (2, 12, 4), dtype=jnp.float32)
    cot = jax.random.normal(jax.random.PRNGKey(6), (2, 12, 16), dtype=jnp.float32)

    for dtype in (jnp.float32, jnp.bfloat16):
      h = h_f32.astype(dtype)
      ref_val, (ref_dh, ref_dmix) = jax.value_and_grad(
          lambda h_in, m_in: jnp.sum(
              jnp.sum(m_in[..., None] * h_in.astype(jnp.float32), axis=2).astype(h_in.dtype) * cot.astype(h_in.dtype)
          ),
          argnums=(0, 1),
      )(h, mix)
      got_val, (got_dh, got_dmix) = jax.value_and_grad(
          lambda h_in, m_in: jnp.sum(deepseek41.DeepSeek41DecoderLayer.hc_pre(h_in, m_in) * cot.astype(h_in.dtype)),
          argnums=(0, 1),
      )(h, mix)
      np.testing.assert_allclose(np.asarray(got_val), np.asarray(ref_val), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(np.asarray(got_dh), np.asarray(ref_dh), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(np.asarray(got_dmix), np.asarray(ref_dmix), rtol=1e-5, atol=1e-5)

  def test_chunked_attention_matches_dense_sink_attention_and_grads(self):
    import functools  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attention_deepseek41  # pylint: disable=import-outside-toplevel

    b, s, h, d, window_size, chunk, c_len = 2, 16, 4, 8, 4, 4, 6
    q = jax.random.normal(jax.random.PRNGKey(7), (b, s, h, d), dtype=jnp.float32)
    kv = jax.random.normal(jax.random.PRNGKey(8), (b, s, d), dtype=jnp.float32)
    compress_kv = jax.random.normal(jax.random.PRNGKey(9), (b, c_len, d), dtype=jnp.float32)
    sinks = jax.random.normal(jax.random.PRNGKey(10), (h,), dtype=jnp.float32)
    cot = jax.random.normal(jax.random.PRNGKey(11), (b, s, h, d), dtype=jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32), (b, s))
    topk_mask = jax.random.uniform(jax.random.PRNGKey(12), (b, s, c_len)) > 0.4

    attend = functools.partial(
        attention_deepseek41._sink_attention,  # pylint: disable=protected-access
        softmax_scale=d**-0.5,
        precision=jax.lax.Precision.HIGHEST,
        out_dtype=jnp.float32,
    )
    dummy = type("_DummyAttn", (), {"window_size": window_size})()

    for use_compressed in (False, True):
      c_kv = compress_kv if use_compressed else None
      t_mask = topk_mask if use_compressed else None

      def dense_loss(q_in, kv_in, ckv_in, s_in):
        delta = positions[:, :, None] - positions[:, None, :]
        mask = (delta >= 0) & (delta < window_size)
        keys = kv_in
        if ckv_in is not None:
          keys = jnp.concatenate([kv_in, ckv_in], axis=1)
          mask = jnp.concatenate([mask, t_mask], axis=-1)
        return jnp.sum(attend(q_in, keys, mask, s_in) * cot)

      def chunked_loss(q_in, kv_in, ckv_in, s_in):
        out = attention_deepseek41.DeepSeek41Attention._chunked_attention(  # pylint: disable=protected-access
            dummy, q_in, kv_in, ckv_in, t_mask, positions, s_in, attend, chunk
        )
        return jnp.sum(out * cot)

      ref_val, ref_grads = jax.value_and_grad(dense_loss, argnums=(0, 1, 3))(q, kv, c_kv, sinks)
      got_val, got_grads = jax.value_and_grad(chunked_loss, argnums=(0, 1, 3))(q, kv, c_kv, sinks)
      self.assertAlmostEqual(float(got_val), float(ref_val), delta=1e-4)
      for g_got, g_ref in zip(got_grads, ref_grads):
        np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
