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

"""Multi-document sequence packing isolation suite for DeepSeek-V4.1-Flash (text & multimodal).

Verifies forward and backward multi-document packing invariants on a self-contained CPU
DeepSeek-V4.1 multimodal configuration where every architectural mechanism is simultaneously
active:
  1. Local sliding-window attention (`sliding_window_size=6 < doc_len=16`).
  2. Two-stage CSA2 compressed KV (`DeepSeek41Compressor`, `compress_ratio in {1, 2}`) and
     two-level `DeepSeek41Indexer` (`select_candidate_blocks` on `candidate_source_layer` +
     downstream `uses_candidates` top-k selection).
  3. `DeepSeek41Engram` N-gram hashing (`DEAD` segment-boundary and image-token lookback barriers
     plus gate zeroing on `segment 0` padding and image positions).
  4. `hc=4` manifold-constrained hyper-connections (`mHC` Sinkhorn mixes).
  5. `RoutedAndSharedMoE` with distinct non-zero `gate.bias` and `gate.bias_vl` (`bias_vl != bias`
     so image tokens route differently from text tokens).
  6. `DeepSeek41VisionEncoder` + 3x3 `DeepSeek41VisionProjector` (`w1`, `w2`, `image_start`,
     `image_newline`, `image_end`) + `cumsum_mask` multimodal embedding merge, with
     `freeze_vision_encoder_params=False` so all vision/projector/delimiter leaves receive
     gradients.
  7. Both `scan_layers=False` and `scan_layers=True` (`prefix + >=2 scanned blocks + suffix`).

Gradient superposition tolerance note:
  For unnormalized token-sum cross-entropy `L_packed = L_A + L_B` on aligned boundaries,
  `grad_params(L_packed)` matches `grad_params(L_A_standalone) + grad_params(L_B_standalone)`
  within `frob_rel <= 1e-5` (`atol = 1e-5 * max(1, ||ref||_inf), rtol = 1e-5`) in IEEE-754
  `float32` (`matmul_precision="highest"`), and within `frob_rel <= 1e-10` (~`1e-15` measured)
  when upcast to `float64`. The `~1e-6` `float32` residual arises solely from floating-point
  non-associativity in GEMM/einsum contractions over `S_packed=40` versus `S_standalone=16`.
"""

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
import json
import os
import shutil
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.common.common_types import MultimodalInput
from maxtext.configs import pyconfig
from maxtext.models import deepseek41
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import model_creation_utils
from tests.utils import deepseek_v41_mm_testing as mm_testing

# Vocabulary and special-token geometry.
_VOCAB_SIZE = 512
_IMAGE_TOKEN_ID = 500
_ENGRAM_PAD_ID = 2
_MIN_TEXT_TOKEN_ID = 10
_MAX_TEXT_TOKEN_ID = 450

# Sequence and DS4.1 mechanism geometry:
# 7 layers -> prefix [0+engram], scanned ((1,2),(3,4)) repeats=2, suffix [5(cand_src), 6(uses_cand+engram)].
_SEQ_PACKED = 40
_SLIDING_WINDOW_SIZE = 6
_COMPRESS_RATIOS = (0, 2, 2, 2, 2, 1, 1)
_KV_SOURCE_LAYERS = (1, 3, 5)
_INDEX_SOURCE_LAYERS = (1, 3, 5, 6)
_CANDIDATE_SOURCE_LAYER = 5
_CANDIDATE_BLOCK_SIZE = 4
# Keep `(topk_blocks - 1) * block_size >= indexer_topk` (as in production `4 * 64 >> 64`) so `uses_candidates`
# (`attention_deepseek41.py:359-368`) never hits `-inf` tie-breaking while `2 * 4 = 8 < _DOC_B_LEN = 16` filters actively.
_CANDIDATE_TOPK_BLOCKS = 2
_INDEXER_TOPK = 3
_ENGRAM_LAYERS = (0, 6)
_ENGRAM_MAX_NGRAM_SIZE = 4
_MHC_EXPANSION_RATE = 4
_SINKHORN_ITERS = 3
_EXPECTED_UNSCANNED_LEAVES = 212

# Vision geometry: 4x5 ViT grid -> 20 patches; r=3 pads to 6x6 -> 2x2 LLM grid -> 8-token span
# `[START, IMG, IMG, NEWLINE, IMG, IMG, NEWLINE, END]`.
_VISION_GRID = (4, 5)
_VISION_DOWNSAMPLE_RATIO = 3
_VISION_PATCH_SIZE = 2
_VISION_CHANNELS = 3
_VISION_NUM_PATCHES = _VISION_GRID[0] * _VISION_GRID[1]
_VISION_PATCH_DIM = _VISION_CHANNELS * (_VISION_PATCH_SIZE**2)
_IMAGE_SPAN_LEN = 8

# Alignment stride = max(compress_ratios) * candidate_block_size = 2 * 4 = 8.
_DOC_A_LEN = 16
_DOC_B_LEN = 16
_PAD_LEN = _SEQ_PACKED - _DOC_A_LEN - _DOC_B_LEN

# Numerical parity & negative-control thresholds (`jax_ml_parity.md` §5).
_FP32_ATOL_FLOOR = 1e-5
_FP32_RTOL = 1e-5
_FP32_FROB_REL_CEILING = 1e-5
_FP64_ATOL_FLOOR = 1e-10
_FP64_RTOL = 1e-10
_FP64_FROB_REL_CEILING = 1e-10
_MM_BATCH_REASSOC_TOL = 1e-6
_MISALIGNED_MIN_DIFF = 1e-2
_NEGCTL_LOGIT_MIN_DIFF = 1e-2
_NEGCTL_EMBED_GRAD_MIN_NORM = 1e-2
_NEGCTL_PARAM_GRAD_MIN_REL = 1e-3

_TOKENIZER_DIR: str = ""


def setUpModule() -> None:
  """Configures highest IEEE-754 matmul precision and builds a local 512-token HF tokenizer."""
  global _TOKENIZER_DIR  # pylint: disable=global-statement
  jax.config.update("jax_default_matmul_precision", "highest")
  _TOKENIZER_DIR = tempfile.mkdtemp(prefix="ds41_packing_tok_")
  vocab = {f"tok_{idx}": idx for idx in range(_VOCAB_SIZE)}
  tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="tok_0"))
  tok_backend.pre_tokenizer = Whitespace()
  tok_backend.save(os.path.join(_TOKENIZER_DIR, "tokenizer.json"))
  with open(os.path.join(_TOKENIZER_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "tok_0"}, fh)


def tearDownModule() -> None:
  """Removes the temporary tokenizer directory."""
  if _TOKENIZER_DIR and os.path.isdir(_TOKENIZER_DIR):
    shutil.rmtree(_TOKENIZER_DIR, ignore_errors=True)


def _build_packing_config(*, scan_layers: bool) -> Any:
  """Builds the tiny DeepSeek-V4.1 multimodal config with all mechanisms active."""
  return pyconfig.initialize(
      [sys.argv[0], "src/maxtext/configs/base.yml"],
      run_name="ds41_packing_isolation_test",
      base_output_directory=os.path.join(tempfile.gettempdir(), "ds41_packing_test_out"),
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
      model_name="deepseek4.1-flash-mini",
      override_model_config=True,
      use_multimodal=True,
      vision_encoder_block="deepseek41",
      freeze_vision_encoder_params=False,
      scan_layers=scan_layers,
      remat_policy="none",
      per_device_batch_size=1.0,
      max_target_length=_SEQ_PACKED,
      dtype="float32",
      weight_dtype="float32",
      dtype_mm="float32",
      float32_gate_logits=True,
      matmul_precision="highest",
      vocab_size=_VOCAB_SIZE,
      ds41_image_token_id=_IMAGE_TOKEN_ID,
      ds41_engram_pad_id=_ENGRAM_PAD_ID,
      ds41_engram_compressed_vocab_size=0,
      mhc_expansion_rate=_MHC_EXPANSION_RATE,
      ds41_hc_sinkhorn_iters=_SINKHORN_ITERS,
      ds41_hc_eps=1.0e-6,
      base_emb_dim=32,
      base_num_query_heads=4,
      head_dim=16,
      qk_rope_head_dim=8,
      q_lora_rank=16,
      o_groups=2,
      o_lora_rank=8,
      base_mlp_dim=32,
      base_moe_mlp_dim=32,
      num_experts=4,
      num_experts_per_tok=2,
      shared_experts=1,
      routed_bias=True,
      sparse_matmul=False,
      capacity_factor=-1,
      sliding_window_size=_SLIDING_WINDOW_SIZE,
      base_num_decoder_layers=len(_COMPRESS_RATIOS),
      compress_ratios=list(_COMPRESS_RATIOS),
      kv_source_layers=list(_KV_SOURCE_LAYERS),
      index_source_layers=list(_INDEX_SOURCE_LAYERS),
      indexer_n_heads=4,
      indexer_head_dim=8,
      indexer_topk=_INDEXER_TOPK,
      candidate_source_layer=_CANDIDATE_SOURCE_LAYER,
      candidate_topk_blocks=_CANDIDATE_TOPK_BLOCKS,
      candidate_block_size=_CANDIDATE_BLOCK_SIZE,
      ds41_engram_layers=list(_ENGRAM_LAYERS),
      ds41_engram_vocab_size=32,
      ds41_engram_max_ngram_size=_ENGRAM_MAX_NGRAM_SIZE,
      ds41_engram_n_heads=2,
      ds41_engram_head_dim=8,
      num_hidden_layers_for_vit=1,
      hidden_size_for_vit=32,
      num_attention_heads_for_vit=2,
      intermediate_size_for_vit=64,
      patch_size_for_vit=_VISION_PATCH_SIZE,
      rope_theta_for_vit=10000,
      num_channels_for_vit=_VISION_CHANNELS,
      ds41_vision_downsample_ratio=_VISION_DOWNSAMPLE_RATIO,
      ds41_vision_grid=list(_VISION_GRID),
      tokenizer_path=_TOKENIZER_DIR,
  )


def _iter_decoder_layer_modules(model: Any, cfg: Any) -> list[tuple[bool, Any]]:
  """Returns `(has_engram, layer_module)` pairs across unscanned or scanned decoder stacks."""
  engram_set = set(cfg.ds41_engram_layers)
  if not cfg.scan_layers:
    return [(idx in engram_set, getattr(model.decoder, f"layers_{idx}")) for idx in range(cfg.num_decoder_layers)]
  pairs: list[tuple[bool, Any]] = []
  for seg_idx, segment in enumerate(deepseek41.build_segments(cfg)):
    block = getattr(model.decoder, f"segments_{seg_idx}")
    for sub_idx, layer_id in enumerate(segment.groups[0]):
      pairs.append((layer_id in engram_set, getattr(block, f"layers_{sub_idx}")))
  return pairs


def _randomize_ds41_state(model: Any, cfg: Any, *, seed: int) -> None:
  """Populates zero-initialized DS4.1 tables/biases with well-conditioned `init_mini_reference` scales."""
  rng_key = jax.random.PRNGKey(seed)
  # Ensure text vs image routing biases favor opposite expert pairs (`bias_vl != bias`).
  text_profile = jnp.asarray([0.45, 0.35, -0.35, -0.45], dtype=jnp.float32)
  vl_profile = jnp.asarray([-0.45, -0.35, 0.40, 0.50], dtype=jnp.float32)
  for has_engram, layer_mod in _iter_decoder_layer_modules(model, cfg):
    rng_key, k_sink, k_hca, k_hcf, k_sa, k_sf, k_bt, k_bvl = jax.random.split(rng_key, 8)
    sinks = layer_mod.self_attention.sinks
    sinks[...] = jax.random.normal(k_sink, sinks[...].shape, dtype=sinks[...].dtype) * 0.25
    hc_ab = layer_mod.hc_attn_base
    hc_ab[...] = jax.random.normal(k_hca, hc_ab[...].shape, dtype=hc_ab[...].dtype) * 0.1
    hc_fb = layer_mod.hc_ffn_base
    hc_fb[...] = jax.random.normal(k_hcf, hc_fb[...].shape, dtype=hc_fb[...].dtype) * 0.1
    hc_as = layer_mod.hc_attn_scale
    hc_as[...] = 0.5 + jax.random.normal(k_sa, hc_as[...].shape, dtype=hc_as[...].dtype) * 0.08
    hc_fs = layer_mod.hc_ffn_scale
    hc_fs[...] = 0.5 + jax.random.normal(k_sf, hc_fs[...].shape, dtype=hc_fs[...].dtype) * 0.08
    if has_engram:
      rng_key, k_eng, k_eq, k_ek = jax.random.split(rng_key, 4)
      emb = layer_mod.engram.embed
      emb[...] = jax.random.normal(k_eng, emb[...].shape, dtype=emb[...].dtype)
      qw = layer_mod.engram.q_weight
      qw[...] = 1.0 + jax.random.normal(k_eq, qw[...].shape, dtype=qw[...].dtype) * 0.1
      kw = layer_mod.engram.k_weight
      kw[...] = 1.0 + jax.random.normal(k_ek, kw[...].shape, dtype=kw[...].dtype) * 0.1
    gate = layer_mod.mlp.MoeBlock_0.gate
    b_shape = gate.bias[...].shape
    b_dtype = gate.bias[...].dtype
    gate.bias[...] = jnp.broadcast_to(text_profile.astype(b_dtype), b_shape) + (
        jax.random.normal(k_bt, b_shape, dtype=b_dtype) * 0.05
    )
    gate.bias_vl[...] = jnp.broadcast_to(vl_profile.astype(b_dtype), b_shape) + (
        jax.random.normal(k_bvl, b_shape, dtype=b_dtype) * 0.05
    )


def _ce_token_sum(logits: jax.Array, targets: jax.Array, weights: jax.Array) -> jax.Array:
  """Unnormalized weighted next-token cross-entropy loss `sum_{b, s} NLL[b, s] * weights[b, s]`."""
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  token_nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
  return jnp.sum(token_nll * weights)


@dataclasses.dataclass(frozen=True)
class _ModelBundle:
  """Cached initialized DS4.1 model state and compiled forward/backward entrypoints."""

  cfg: Any
  mesh: jax.sharding.Mesh
  graphdef: nnx.GraphDef
  param_state: nnx.State
  pure_params: dict[str, Any]
  rest_state: nnx.State
  fwd_fn: Callable[..., jax.Array]
  fwd_with_rest_fn: Callable[..., jax.Array]
  fwd_and_grad_fn: Callable[..., tuple[jax.Array, dict[str, Any]]]
  embed_fn: Callable[..., jax.Array]
  embed_grad_fn: Callable[..., tuple[jax.Array, jax.Array]]


def _build_model_bundle(*, scan_layers: bool, seed: int) -> _ModelBundle:
  """Initializes a DS4.1 multimodal model on a 1-device mesh and compiles JIT entrypoints."""
  cfg = _build_packing_config(scan_layers=scan_layers)
  mesh_devices = np.array(jax.devices()[:1]).reshape((1,) * len(cfg.mesh_axes))
  mesh = jax.sharding.Mesh(mesh_devices, cfg.mesh_axes)
  with mesh:
    model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  _randomize_ds41_state(model, cfg, seed=seed)
  graphdef, param_state, pure_params, rest_state = mm_testing.extract_pure_params(model)

  @jax.jit
  def fwd_with_rest_fn(
      params: dict[str, Any],
      rest: nnx.State,
      tokens: jax.Array,
      positions: jax.Array,
      *,
      segment_ids: jax.Array,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      input_embeddings: jax.Array | None = None,
  ) -> jax.Array:
    mod = mm_testing.apply_with_pure_params(graphdef, param_state, params, rest)
    return mod(
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        encoder_images=images,
        encoder_image_masks=image_masks,
        enable_dropout=False,
        decoder_input_embeddings=input_embeddings,
    )

  def fwd_fn(
      params: dict[str, Any],
      tokens: jax.Array,
      positions: jax.Array,
      segment_ids: jax.Array,
      *,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      input_embeddings: jax.Array | None = None,
  ) -> jax.Array:
    return fwd_with_rest_fn(
        params,
        rest_state,
        tokens,
        positions,
        segment_ids=segment_ids,
        images=images,
        image_masks=image_masks,
        input_embeddings=input_embeddings,
    )

  @jax.jit
  def fwd_and_grad_fn(
      params: dict[str, Any],
      tokens: jax.Array,
      positions: jax.Array,
      segment_ids: jax.Array,
      *,
      targets: jax.Array,
      weights: jax.Array,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, Any]]:
    def loss_fn(p: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
      mod = mm_testing.apply_with_pure_params(graphdef, param_state, p, rest_state)
      logits = mod(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=segment_ids,
          encoder_images=images,
          encoder_image_masks=image_masks,
          enable_dropout=False,
      )
      return _ce_token_sum(logits, targets, weights), logits

    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return logits, grads

  @jax.jit
  def embed_fn(
      params: dict[str, Any],
      tokens: jax.Array,
      positions: jax.Array,
      *,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
  ) -> jax.Array:
    mod = mm_testing.apply_with_pure_params(graphdef, param_state, params, rest_state)
    multimodal_input = None
    if images is not None:
      image_embeddings, _ = mod.vision_encoder(input_images=images, deterministic=True)
      bidir_mask = mm_processor.get_bidirectional_mask_vision(cfg, tokens, is_video=False)
      multimodal_input = MultimodalInput(
          image_embeddings=image_embeddings,
          image_masks=image_masks,
          bidirectional_mask=bidir_mask,
      )
    return mod.decoder._apply_embedding(  # pylint: disable=protected-access
        mod.token_embedder,
        tokens,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        multimodal_input=multimodal_input,
        decoder_input_embeddings=None,
    )

  @jax.jit
  def embed_grad_fn(
      params: dict[str, Any],
      embed_in: jax.Array,
      tokens: jax.Array,
      positions: jax.Array,
      *,
      segment_ids: jax.Array,
      targets: jax.Array,
      weights: jax.Array,
      images: jax.Array | None = None,
      image_masks: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    def loss_wrt_embed(emb: jax.Array) -> tuple[jax.Array, jax.Array]:
      mod = mm_testing.apply_with_pure_params(graphdef, param_state, params, rest_state)
      logits = mod(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=segment_ids,
          encoder_images=images,
          encoder_image_masks=image_masks,
          enable_dropout=False,
          decoder_input_embeddings=emb,
      )
      return _ce_token_sum(logits, targets, weights), logits

    (_, logits), grad_emb = jax.value_and_grad(loss_wrt_embed, has_aux=True)(embed_in)
    return logits, grad_emb

  return _ModelBundle(
      cfg=cfg,
      mesh=mesh,
      graphdef=graphdef,
      param_state=param_state,
      pure_params=pure_params,
      rest_state=rest_state,
      fwd_fn=fwd_fn,
      fwd_with_rest_fn=fwd_with_rest_fn,
      fwd_and_grad_fn=fwd_and_grad_fn,
      embed_fn=embed_fn,
      embed_grad_fn=embed_grad_fn,
  )


def _compute_merged_input_embeddings(
    bundle: _ModelBundle,
    tokens: jax.Array,
    positions: jax.Array,
    *,
    images: jax.Array | None = None,
    image_masks: jax.Array | None = None,
) -> jax.Array:
  """Computes exact `[B, S, D]` post-merge input embeddings entering the DS4.1 decoder layers."""
  with bundle.mesh:
    return bundle.embed_fn(bundle.pure_params, tokens, positions, images=images, image_masks=image_masks)


@contextlib.contextmanager
def _fp64_upcast_context(cfg: Any) -> Iterator[None]:
  """Test-scoped `float64` upcast context without any production dtype knob."""
  with (
      jax.enable_x64(True),
      mock.patch.dict(
          cfg._flat_config,  # pylint: disable=protected-access
          {
              "dtype": jnp.float64,
              "weight_dtype": jnp.float64,
              "dtype_mm": jnp.float64,
              "float32_gate_logits": False,
          },
      ),
      mock.patch.object(jnp, "float32", jnp.float64),
  ):
    yield


def _assert_tensor_parity(
    tc: unittest.TestCase,
    cand: Any,
    ref: Any,
    *,
    atol_floor: float,
    rtol: float,
    frob_rel_ceiling: float,
    label: str,
) -> None:
  """Asserts elementwise (`atol_floor * max(1, ||ref||_inf) + rtol * |ref|`) and Frobenius parity."""
  cand_64 = np.asarray(cand, dtype=np.float64)
  ref_64 = np.asarray(ref, dtype=np.float64)
  tc.assertEqual(cand_64.shape, ref_64.shape, msg=f"{label}: shape mismatch {cand_64.shape} vs {ref_64.shape}")
  tc.assertTrue(np.all(np.isfinite(cand_64)), msg=f"{label}: candidate has non-finite values")
  tc.assertTrue(np.all(np.isfinite(ref_64)), msg=f"{label}: reference has non-finite values")
  diff = np.abs(cand_64 - ref_64)
  max_abs = float(np.max(diff))
  ref_inf = float(np.max(np.abs(ref_64)))
  elem_bound = atol_floor * max(1.0, ref_inf) + rtol * np.abs(ref_64)
  tc.assertTrue(
      bool(np.all(diff <= elem_bound)),
      msg=f"{label}: elementwise failure max_abs={max_abs:.3e}, ref_inf={ref_inf:.3e}",
  )
  delta_frob = float(np.linalg.norm(diff.ravel()))
  ref_frob = float(np.linalg.norm(ref_64.ravel()))
  frob_rel = delta_frob / max(1.0, ref_frob)
  tc.assertLessEqual(
      frob_rel,
      frob_rel_ceiling,
      msg=f"{label}: frob_rel={frob_rel:.3e} > ceiling {frob_rel_ceiling:.3e} (max_abs={max_abs:.3e})",
  )


def _assert_all_param_leaves_parity(
    tc: unittest.TestCase,
    cand_tree: dict[str, Any],
    ref_tree: dict[str, Any],
    *,
    atol_floor: float,
    rtol: float,
    frob_rel_ceiling: float,
    label: str,
) -> tuple[int, float, float]:
  """Compares EVERY parameter leaf in `cand_tree` and `ref_tree` (`jax_ml_parity.md` §4 & §5)."""
  cand_host, ref_host = jax.device_get((cand_tree, ref_tree))
  cand_leaves = jax.tree_util.tree_leaves_with_path(cand_host)
  ref_leaves = jax.tree_util.tree_leaves_with_path(ref_host)
  tc.assertEqual(len(cand_leaves), len(ref_leaves), msg=f"{label}: leaf count mismatch")
  tc.assertGreater(len(cand_leaves), 0, msg=f"{label}: empty parameter tree")
  failures: list[str] = []
  worst_max_abs = 0.0
  worst_frob_rel = 0.0
  for (c_path, c_val), (r_path, r_val) in zip(cand_leaves, ref_leaves):
    path_str = "/".join(str(entry.key) for entry in c_path)
    r_path_str = "/".join(str(entry.key) for entry in r_path)
    tc.assertEqual(path_str, r_path_str, msg=f"{label}: tree path mismatch {path_str} vs {r_path_str}")
    c_64 = np.asarray(c_val, dtype=np.float64)
    r_64 = np.asarray(r_val, dtype=np.float64)
    if not (np.all(np.isfinite(c_64)) and np.all(np.isfinite(r_64))):
      failures.append(f"{path_str}: non-finite values")
      continue
    diff = np.abs(c_64 - r_64)
    max_abs = float(np.max(diff))
    ref_inf = float(np.max(np.abs(r_64)))
    elem_bound = atol_floor * max(1.0, ref_inf) + rtol * np.abs(r_64)
    delta_frob = float(np.linalg.norm(diff.ravel()))
    ref_frob = float(np.linalg.norm(r_64.ravel()))
    frob_rel = delta_frob / max(1.0, ref_frob)
    worst_max_abs = max(worst_max_abs, max_abs)
    worst_frob_rel = max(worst_frob_rel, frob_rel)
    if not bool(np.all(diff <= elem_bound)) or frob_rel > frob_rel_ceiling:
      failures.append(f"{path_str}: max_abs={max_abs:.3e} (ref_inf={ref_inf:.3e}), frob_rel={frob_rel:.3e}")
  tc.assertEqual(failures, [], msg=f"{label}: {len(failures)}/{len(cand_leaves)} leaves failed: {failures[:8]}")
  return len(cand_leaves), worst_max_abs, worst_frob_rel


def _assert_all_param_leaves_exact_zero_diff(
    tc: unittest.TestCase,
    tree_a: dict[str, Any],
    tree_b: dict[str, Any],
    *,
    label: str,
) -> None:
  """Asserts bit-exact `0.0` difference across EVERY parameter leaf."""
  host_a, host_b = jax.device_get((tree_a, tree_b))
  leaves_a = jax.tree_util.tree_leaves_with_path(host_a)
  leaves_b = jax.tree_util.tree_leaves_with_path(host_b)
  tc.assertEqual(len(leaves_a), len(leaves_b))
  nonzero: list[str] = []
  for (path, val_a), (_, val_b) in zip(leaves_a, leaves_b):
    path_str = "/".join(str(entry.key) for entry in path)
    max_diff = float(np.max(np.abs(np.asarray(val_a, dtype=np.float64) - np.asarray(val_b, dtype=np.float64))))
    if max_diff != 0.0:
      nonzero.append(f"{path_str}={max_diff:.3e}")
  tc.assertEqual(nonzero, [], msg=f"{label}: expected bit-exact 0.0 across all leaves, got {nonzero[:8]}")


def _tree_relative_frob_diff(tree_a: dict[str, Any], tree_b: dict[str, Any]) -> float:
  """Computes global `||tree_a - tree_b||_2 / max(1e-7, ||tree_a||_2)` across all leaves."""
  host_a, host_b = jax.device_get((tree_a, tree_b))
  sq_diff = 0.0
  sq_ref = 0.0
  for leaf_a, leaf_b in zip(jax.tree_util.tree_leaves(host_a), jax.tree_util.tree_leaves(host_b)):
    a_64 = np.asarray(leaf_a, dtype=np.float64).ravel()
    b_64 = np.asarray(leaf_b, dtype=np.float64).ravel()
    sq_diff += float(np.dot(a_64 - b_64, a_64 - b_64))
    sq_ref += float(np.dot(a_64, a_64))
  return float(np.sqrt(sq_diff)) / max(1e-7, float(np.sqrt(sq_ref)))


def _make_aligned_pack_inputs(
    *,
    seed: int,
    multimodal_mode: bool,
    doc_a_span_start: int = 4,
    doc_b_span_start: int = 4,
) -> dict[str, Any]:
  """Constructs `[Doc A (16, seg=1) | Doc B (16, seg=2) | Pad (8, seg=0)]` and standalone twins."""
  rng = np.random.default_rng(seed)
  tok_a = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_A_LEN), dtype=np.int32)
  tok_a_alt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_A_LEN), dtype=np.int32)
  tok_b = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_B_LEN), dtype=np.int32)
  tgt_a = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_A_LEN), dtype=np.int32)
  tgt_b = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_B_LEN), dtype=np.int32)
  pad_a = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _PAD_LEN), dtype=np.int32)
  pad_b = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _PAD_LEN), dtype=np.int32)

  if multimodal_mode:
    tok_a[:, doc_a_span_start : doc_a_span_start + _IMAGE_SPAN_LEN] = _IMAGE_TOKEN_ID
    tok_a_alt[:, doc_a_span_start : doc_a_span_start + _IMAGE_SPAN_LEN] = _IMAGE_TOKEN_ID
    tok_b[:, doc_b_span_start : doc_b_span_start + _IMAGE_SPAN_LEN] = _IMAGE_TOKEN_ID

  raw_mask_a = (tok_a != _IMAGE_TOKEN_ID).astype(np.float32)
  raw_mask_b = (tok_b != _IMAGE_TOKEN_ID).astype(np.float32)
  n_valid_total = float(np.sum(raw_mask_a) + np.sum(raw_mask_b))
  mask_a = raw_mask_a / n_valid_total
  mask_b = raw_mask_b / n_valid_total

  pos_packed = jnp.asarray(
      np.concatenate(
          [np.arange(_DOC_A_LEN), np.arange(_DOC_B_LEN), np.zeros(_PAD_LEN, dtype=np.int32)],
          axis=0,
      )[None, :],
      dtype=jnp.int32,
  )
  seg_packed = jnp.asarray(
      np.concatenate(
          [
              np.ones(_DOC_A_LEN, dtype=np.int32),
              np.full(_DOC_B_LEN, 2, dtype=np.int32),
              np.zeros(_PAD_LEN, dtype=np.int32),
          ],
          axis=0,
      )[None, :],
      dtype=jnp.int32,
  )
  tgt_packed = jnp.asarray(
      np.concatenate([tgt_a, tgt_b, np.zeros((1, _PAD_LEN), dtype=np.int32)], axis=1),
      dtype=jnp.int32,
  )
  w_a_packed = jnp.asarray(
      np.concatenate([mask_a, np.zeros((1, _DOC_B_LEN + _PAD_LEN), dtype=np.float32)], axis=1),
      dtype=jnp.float32,
  )
  w_b_packed = jnp.asarray(
      np.concatenate(
          [np.zeros((1, _DOC_A_LEN), dtype=np.float32), mask_b, np.zeros((1, _PAD_LEN), dtype=np.float32)],
          axis=1,
      ),
      dtype=jnp.float32,
  )

  # Padded single-document twins of identical length `_SEQ_PACKED = 40` (`[Doc | Pad]`).
  tail_for_a = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _SEQ_PACKED - _DOC_A_LEN), dtype=np.int32)
  tail_for_b = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _SEQ_PACKED - _DOC_B_LEN), dtype=np.int32)
  tok_a_padded = jnp.asarray(np.concatenate([tok_a, tail_for_a], axis=1), dtype=jnp.int32)
  pos_a_padded = jnp.asarray(
      np.concatenate([np.arange(_DOC_A_LEN), np.zeros(_SEQ_PACKED - _DOC_A_LEN, dtype=np.int32)])[None, :],
      dtype=jnp.int32,
  )
  seg_a_padded = jnp.asarray(
      np.concatenate([np.ones(_DOC_A_LEN, dtype=np.int32), np.zeros(_SEQ_PACKED - _DOC_A_LEN, dtype=np.int32)])[None, :],
      dtype=jnp.int32,
  )
  tgt_a_padded = jnp.asarray(
      np.concatenate([tgt_a, np.zeros((1, _SEQ_PACKED - _DOC_A_LEN), dtype=np.int32)], axis=1),
      dtype=jnp.int32,
  )
  w_a_padded = jnp.asarray(
      np.concatenate([mask_a, np.zeros((1, _SEQ_PACKED - _DOC_A_LEN), dtype=np.float32)], axis=1),
      dtype=jnp.float32,
  )

  tok_b_padded = jnp.asarray(np.concatenate([tok_b, tail_for_b], axis=1), dtype=jnp.int32)
  pos_b_padded = jnp.asarray(
      np.concatenate([np.arange(_DOC_B_LEN), np.zeros(_SEQ_PACKED - _DOC_B_LEN, dtype=np.int32)])[None, :],
      dtype=jnp.int32,
  )
  seg_b_padded = jnp.asarray(
      np.concatenate([np.ones(_DOC_B_LEN, dtype=np.int32), np.zeros(_SEQ_PACKED - _DOC_B_LEN, dtype=np.int32)])[None, :],
      dtype=jnp.int32,
  )
  tgt_b_padded = jnp.asarray(
      np.concatenate([tgt_b, np.zeros((1, _SEQ_PACKED - _DOC_B_LEN), dtype=np.int32)], axis=1),
      dtype=jnp.int32,
  )
  w_b_padded = jnp.asarray(
      np.concatenate([mask_b, np.zeros((1, _SEQ_PACKED - _DOC_B_LEN), dtype=np.float32)], axis=1),
      dtype=jnp.float32,
  )

  img_0 = rng.standard_normal((1, 1, _VISION_NUM_PATCHES, _VISION_PATCH_DIM)).astype(np.float32)
  img_0_alt = rng.standard_normal((1, 1, _VISION_NUM_PATCHES, _VISION_PATCH_DIM)).astype(np.float32)
  img_1 = rng.standard_normal((1, 1, _VISION_NUM_PATCHES, _VISION_PATCH_DIM)).astype(np.float32)

  return {
      "tok_a": jnp.asarray(tok_a, dtype=jnp.int32),
      "tok_b": jnp.asarray(tok_b, dtype=jnp.int32),
      "tgt_a": jnp.asarray(tgt_a, dtype=jnp.int32),
      "tgt_b": jnp.asarray(tgt_b, dtype=jnp.int32),
      "mask_a": jnp.asarray(mask_a, dtype=jnp.float32),
      "mask_b": jnp.asarray(mask_b, dtype=jnp.float32),
      "tok_packed": jnp.asarray(np.concatenate([tok_a, tok_b, pad_a], axis=1), dtype=jnp.int32),
      "tok_packed_alt_a": jnp.asarray(np.concatenate([tok_a_alt, tok_b, pad_a], axis=1), dtype=jnp.int32),
      "tok_packed_alt_pad": jnp.asarray(np.concatenate([tok_a, tok_b, pad_b], axis=1), dtype=jnp.int32),
      "pos_packed": pos_packed,
      "seg_packed": seg_packed,
      "tgt_packed": tgt_packed,
      "w_a_packed": w_a_packed,
      "w_b_packed": w_b_packed,
      "tok_a_padded": tok_a_padded,
      "pos_a_padded": pos_a_padded,
      "seg_a_padded": seg_a_padded,
      "tgt_a_padded": tgt_a_padded,
      "w_a_padded": w_a_padded,
      "tok_b_padded": tok_b_padded,
      "pos_b_padded": pos_b_padded,
      "seg_b_padded": seg_b_padded,
      "tgt_b_padded": tgt_b_padded,
      "w_b_padded": w_b_padded,
      "imgs_packed": jnp.asarray(np.concatenate([img_0, img_1], axis=1), dtype=jnp.float32),
      "imgs_packed_alt_0": jnp.asarray(np.concatenate([img_0_alt, img_1], axis=1), dtype=jnp.float32),
      "img_0": jnp.asarray(img_0, dtype=jnp.float32),
      "img_1": jnp.asarray(img_1, dtype=jnp.float32),
      "imask_2": jnp.ones((1, 2), dtype=jnp.int32),
      "imask_10": jnp.asarray([[1, 0]], dtype=jnp.int32),
      "imask_01": jnp.asarray([[0, 1]], dtype=jnp.int32),
      "imask_1": jnp.ones((1, 1), dtype=jnp.int32),
  }


@pytest.mark.cpu_only
class DeepSeek41PackingIsolationTest(unittest.TestCase):
  """End-to-end forward/backward packing isolation tests for DeepSeek-V4.1 text and multimodal."""

  unscanned: _ModelBundle
  scanned: _ModelBundle

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.unscanned = _build_model_bundle(scan_layers=False, seed=2026)
    cls.scanned = _build_model_bundle(scan_layers=True, seed=2026)

  def test_regime_activity_and_scan_geometry(self) -> None:
    """Verifies scan geometry (`prefix + >=2 scanned blocks + suffix`) and non-vacuous mechanism activity."""
    cfg = self.unscanned.cfg
    segments = deepseek41.build_segments(cfg)
    # Prefix layer [0+engram], scanned segment ((1,2), (3,4)) with repeats=2, suffix [5(cand_src)], [6(uses_cand+engram)].
    self.assertEqual([seg.groups for seg in segments], [((0,),), ((1, 2), (3, 4)), ((5,),), ((6,),)])
    self.assertEqual(segments[1].repeats, 2)
    self.assertGreater(_DOC_B_LEN, _SLIDING_WINDOW_SIZE)
    self.assertGreater(_DOC_B_LEN, _CANDIDATE_TOPK_BLOCKS * _CANDIDATE_BLOCK_SIZE)
    self.assertGreaterEqual(_DOC_A_LEN, 2 * cfg.num_experts_per_tok)
    self.assertGreaterEqual(_DOC_A_LEN, 2 * _INDEXER_TOPK)

    # Verify the local 512-token tokenizer maps all 512 tokens distinctly in Engram.
    _, compressed_vocab_size = deepseek41.compressed_token_map(cfg)
    self.assertEqual(compressed_vocab_size, _VOCAB_SIZE)

    mm_data = _make_aligned_pack_inputs(seed=101, multimodal_mode=True)
    with self.unscanned.mesh:
      _, grads = self.unscanned.fwd_and_grad_fn(
          self.unscanned.pure_params,
          mm_data["tok_packed"],
          mm_data["pos_packed"],
          mm_data["seg_packed"],
          targets=mm_data["tgt_packed"],
          weights=mm_data["w_a_packed"] + mm_data["w_b_packed"],
          images=mm_data["imgs_packed"],
          image_masks=mm_data["imask_2"],
      )

    grad_norms: dict[str, float] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
      path_str = "/".join(str(entry.key) for entry in path)
      grad_norms[path_str] = float(np.linalg.norm(np.asarray(leaf, dtype=np.float64).ravel()))

    # 1. CSA2 compressor gradients (`wkv` + `wgate`) must be strictly positive on all KV source layers.
    for kv_layer in _KV_SOURCE_LAYERS:
      self.assertGreater(
          grad_norms[f"decoder/layers_{kv_layer}/self_attention/compressor/wkv/kernel"],
          0.0,
          msg=f"Expected active compressor.wkv gradient on layer {kv_layer}",
      )
    self.assertGreater(grad_norms["decoder/layers_1/self_attention/compressor/wgate/kernel"], 0.0)

    # 2. Engram `embed` and `wkv` gradients must be strictly positive on all Engram layers.
    for eng_layer in _ENGRAM_LAYERS:
      self.assertGreater(grad_norms[f"decoder/layers_{eng_layer}/engram/embed"], 0.0)
      self.assertGreater(grad_norms[f"decoder/layers_{eng_layer}/engram/wkv/kernel"], 0.0)

    # 3. `hc=4` hyper-connection parameters must receive strictly positive gradients.
    for hc_leaf in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
      self.assertGreater(grad_norms[f"decoder/layers_0/{hc_leaf}"], 0.0, msg=f"Zero gradient on {hc_leaf}")

    # 4. Vision encoder, 3x3 aligner (`w1`, `w2`), and all 3 image delimiters must receive positive gradients.
    for mm_leaf in (
        "vision_encoder/DeepSeek41VisionEncoder_0/patch_embed/kernel",
        "vision_encoder/DeepSeek41VisionEncoder_0/blocks_0/attn/wqkv/kernel",
        "vision_encoder/DeepSeek41VisionEncoder_0/blocks_0/mlp/w2/kernel",
        "vision_encoder/DeepSeek41VisionProjector_0/w1/kernel",
        "vision_encoder/DeepSeek41VisionProjector_0/w2/kernel",
        "vision_encoder/DeepSeek41VisionProjector_0/image_start",
        "vision_encoder/DeepSeek41VisionProjector_0/image_newline",
        "vision_encoder/DeepSeek41VisionProjector_0/image_end",
    ):
      self.assertGreater(grad_norms[mm_leaf], 0.0, msg=f"Zero gradient on multimodal leaf {mm_leaf}")

    # 5. Every differentiable leaf in the parameter tree must be non-zero (only discrete top-k indexer leaves are 0).
    self.assertEqual(len(grad_norms), _EXPECTED_UNSCANNED_LEAVES)
    zero_non_indexer = [p for p, norm in grad_norms.items() if norm == 0.0 and "/self_attention/indexer/" not in p]
    self.assertEqual(zero_non_indexer, [], msg=f"Unexpected zero-gradient leaves: {zero_non_indexer}")

    # 6. Verify `gate.bias_vl != gate.bias` actively alters expert routing on image tokens while leaving text untouched.
    txt_data = _make_aligned_pack_inputs(seed=102, multimodal_mode=False)
    rest_copy = jax.tree.map(lambda x: x, self.unscanned.rest_state)
    mod_swapped = mm_testing.apply_with_pure_params(
        self.unscanned.graphdef, self.unscanned.param_state, self.unscanned.pure_params, rest_copy
    )
    for _, layer_mod in _iter_decoder_layer_modules(mod_swapped, cfg):
      gate = layer_mod.mlp.MoeBlock_0.gate
      gate.bias_vl[...] = gate.bias[...]
    _, _, _, rest_swapped = mm_testing.extract_pure_params(mod_swapped)

    with self.unscanned.mesh:
      mm_logits_vl = np.asarray(
          self.unscanned.fwd_with_rest_fn(
              self.unscanned.pure_params,
              self.unscanned.rest_state,
              mm_data["tok_packed"],
              mm_data["pos_packed"],
              segment_ids=mm_data["seg_packed"],
              images=mm_data["imgs_packed"],
              image_masks=mm_data["imask_2"],
          ),
          dtype=np.float64,
      )
      mm_logits_no_vl = np.asarray(
          self.unscanned.fwd_with_rest_fn(
              self.unscanned.pure_params,
              rest_swapped,
              mm_data["tok_packed"],
              mm_data["pos_packed"],
              segment_ids=mm_data["seg_packed"],
              images=mm_data["imgs_packed"],
              image_masks=mm_data["imask_2"],
          ),
          dtype=np.float64,
      )
      txt_logits_vl = np.asarray(
          self.unscanned.fwd_with_rest_fn(
              self.unscanned.pure_params,
              self.unscanned.rest_state,
              txt_data["tok_packed"],
              txt_data["pos_packed"],
              segment_ids=txt_data["seg_packed"],
          ),
          dtype=np.float64,
      )
      txt_logits_no_vl = np.asarray(
          self.unscanned.fwd_with_rest_fn(
              self.unscanned.pure_params,
              rest_swapped,
              txt_data["tok_packed"],
              txt_data["pos_packed"],
              segment_ids=txt_data["seg_packed"],
          ),
          dtype=np.float64,
      )
    self.assertGreater(
        float(np.max(np.abs(mm_logits_vl[:, : _DOC_A_LEN + _DOC_B_LEN] - mm_logits_no_vl[:, : _DOC_A_LEN + _DOC_B_LEN]))),
        1e-2,
        msg="Expected bias_vl != bias to change routing and logits on multimodal spans",
    )
    self.assertEqual(float(np.max(np.abs(txt_logits_vl - txt_logits_no_vl))), 0.0)

  def test_forward_isolation_text_and_multimodal(self) -> None:
    """Group 1: Packed `[Doc A | Doc B | Pad]` forward standalone parity and perturbation invariance."""
    bundle = self.unscanned
    pos_a_unpadded = jnp.arange(_DOC_A_LEN, dtype=jnp.int32)[None, :]
    seg_a_unpadded = jnp.ones((1, _DOC_A_LEN), dtype=jnp.int32)
    pos_b_unpadded = jnp.arange(_DOC_B_LEN, dtype=jnp.int32)[None, :]
    seg_b_unpadded = jnp.ones((1, _DOC_B_LEN), dtype=jnp.int32)

    # 1A. Text-only forward isolation.
    txt = _make_aligned_pack_inputs(seed=201, multimodal_mode=False)
    with bundle.mesh:
      logits_p = bundle.fwd_fn(bundle.pure_params, txt["tok_packed"], txt["pos_packed"], txt["seg_packed"])
      logits_p_alt_a = bundle.fwd_fn(bundle.pure_params, txt["tok_packed_alt_a"], txt["pos_packed"], txt["seg_packed"])
      logits_p_alt_pad = bundle.fwd_fn(
          bundle.pure_params, txt["tok_packed_alt_pad"], txt["pos_packed"], txt["seg_packed"]
      )
      logits_a_padded = bundle.fwd_fn(bundle.pure_params, txt["tok_a_padded"], txt["pos_a_padded"], txt["seg_a_padded"])
      logits_b_padded = bundle.fwd_fn(bundle.pure_params, txt["tok_b_padded"], txt["pos_b_padded"], txt["seg_b_padded"])
      logits_a_solo = bundle.fwd_fn(bundle.pure_params, txt["tok_a"], pos_a_unpadded, seg_a_unpadded)
      logits_b_solo = bundle.fwd_fn(bundle.pure_params, txt["tok_b"], pos_b_unpadded, seg_b_unpadded)

    doc_b_slice = slice(_DOC_A_LEN, _DOC_A_LEN + _DOC_B_LEN)
    valid_slice = slice(0, _DOC_A_LEN + _DOC_B_LEN)
    _assert_tensor_parity(
        self,
        logits_p[:, :_DOC_A_LEN],
        logits_a_padded[:, :_DOC_A_LEN],
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="text Doc A packed vs [Doc A | Pad]",
    )
    _assert_tensor_parity(
        self,
        logits_p[:, doc_b_slice],
        logits_b_padded[:, :_DOC_B_LEN],
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="text Doc B packed vs [Doc B | Pad]",
    )
    _assert_tensor_parity(
        self,
        logits_p[:, :_DOC_A_LEN],
        logits_a_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="text Doc A packed vs standalone",
    )
    _assert_tensor_parity(
        self,
        logits_p[:, doc_b_slice],
        logits_b_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="text Doc B packed vs standalone",
    )
    # Bit-exact 0.0 perturbation invariance across documents and trailing padding.
    self.assertEqual(
        float(np.max(np.abs(np.asarray(logits_p[:, doc_b_slice]) - np.asarray(logits_p_alt_a[:, doc_b_slice])))),
        0.0,
    )
    self.assertEqual(
        float(np.max(np.abs(np.asarray(logits_p[:, valid_slice]) - np.asarray(logits_p_alt_pad[:, valid_slice])))),
        0.0,
    )

    # 1B. Multimodal forward isolation (Image 0 in Doc A, Image 1 in Doc B).
    mm = _make_aligned_pack_inputs(seed=202, multimodal_mode=True)
    with bundle.mesh:
      mm_p = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_packed"],
          mm["pos_packed"],
          mm["seg_packed"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_2"],
      )
      mm_p_alt_a_tok = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_packed_alt_a"],
          mm["pos_packed"],
          mm["seg_packed"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_2"],
      )
      mm_p_alt_a_both = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_packed_alt_a"],
          mm["pos_packed"],
          mm["seg_packed"],
          images=mm["imgs_packed_alt_0"],
          image_masks=mm["imask_2"],
      )
      mm_p_alt_pad = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_packed_alt_pad"],
          mm["pos_packed"],
          mm["seg_packed"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_2"],
      )
      mm_b_padded = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_b_padded"],
          mm["pos_b_padded"],
          mm["seg_b_padded"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_01"],
      )
      mm_a_solo = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_a"],
          pos_a_unpadded,
          seg_a_unpadded,
          images=mm["img_0"],
          image_masks=mm["imask_1"],
      )
      mm_b_solo = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_b"],
          pos_b_unpadded,
          seg_b_unpadded,
          images=mm["img_1"],
          image_masks=mm["imask_1"],
      )

    _assert_tensor_parity(
        self,
        mm_p[:, doc_b_slice],
        mm_b_padded[:, :_DOC_B_LEN],
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="multimodal Doc B packed vs [Doc B | Pad]",
    )
    _assert_tensor_parity(
        self,
        mm_p[:, :_DOC_A_LEN],
        mm_a_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="multimodal Doc A packed vs standalone",
    )
    _assert_tensor_parity(
        self,
        mm_p[:, doc_b_slice],
        mm_b_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="multimodal Doc B packed vs standalone",
    )
    # Perturbing Doc A text tokens or trailing pad tokens changes Doc B logits by bit-exact 0.0.
    self.assertEqual(
        float(np.max(np.abs(np.asarray(mm_p[:, doc_b_slice]) - np.asarray(mm_p_alt_a_tok[:, doc_b_slice])))),
        0.0,
    )
    self.assertEqual(
        float(np.max(np.abs(np.asarray(mm_p[:, valid_slice]) - np.asarray(mm_p_alt_pad[:, valid_slice])))),
        0.0,
    )
    # Perturbing both Doc A text tokens AND Image 0 pixels leaves Doc B invariant (within batched ViT GEMM 1e-6).
    self.assertLessEqual(
        float(
            np.max(
                np.abs(
                    np.asarray(mm_p[:, doc_b_slice], dtype=np.float64)
                    - np.asarray(mm_p_alt_a_both[:, doc_b_slice], dtype=np.float64)
                )
            )
        ),
        _MM_BATCH_REASSOC_TOL,
    )

  def test_backward_parameter_gradients_and_superposition_fp32(self) -> None:
    """Group 2 (fp32): All-parameter-leaf gradient isolation, standalone parity, and total-loss superposition."""
    bundle = self.unscanned
    pos_a_unpadded = jnp.arange(_DOC_A_LEN, dtype=jnp.int32)[None, :]
    seg_a_unpadded = jnp.ones((1, _DOC_A_LEN), dtype=jnp.int32)
    pos_b_unpadded = jnp.arange(_DOC_B_LEN, dtype=jnp.int32)[None, :]
    seg_b_unpadded = jnp.ones((1, _DOC_B_LEN), dtype=jnp.int32)

    for mode_name, is_mm, seed in (("text", False, 301), ("multimodal", True, 302)):
      with self.subTest(mode=mode_name):
        data = _make_aligned_pack_inputs(seed=seed, multimodal_mode=is_mm)
        imgs_p = data["imgs_packed"] if is_mm else None
        imgs_p_alt = data["imgs_packed_alt_0"] if is_mm else None
        imask_p = data["imask_2"] if is_mm else None
        img_a = data["img_0"] if is_mm else None
        img_b = data["img_1"] if is_mm else None
        imask_1 = data["imask_1"] if is_mm else None

        with bundle.mesh:
          _, grads_total = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_packed"],
              data["pos_packed"],
              data["seg_packed"],
              targets=data["tgt_packed"],
              weights=data["w_a_packed"] + data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          _, grads_b_packed = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_packed"],
              data["pos_packed"],
              data["seg_packed"],
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          _, grads_b_alt_a_tok = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_packed_alt_a"],
              data["pos_packed"],
              data["seg_packed"],
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          _, grads_b_alt_pad = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_packed_alt_pad"],
              data["pos_packed"],
              data["seg_packed"],
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          _, grads_a_solo = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_a"],
              pos_a_unpadded,
              seg_a_unpadded,
              targets=data["tgt_a"],
              weights=data["mask_a"],
              images=img_a,
              image_masks=imask_1,
          )
          _, grads_b_solo = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_b"],
              pos_b_unpadded,
              seg_b_unpadded,
              targets=data["tgt_b"],
              weights=data["mask_b"],
              images=img_b,
              image_masks=imask_1,
          )
          if is_mm:
            _, grads_b_alt_img0 = bundle.fwd_and_grad_fn(
                bundle.pure_params,
                data["tok_packed_alt_a"],
                data["pos_packed"],
                data["seg_packed"],
                targets=data["tgt_packed"],
                weights=data["w_b_packed"],
                images=imgs_p_alt,
                image_masks=imask_p,
            )

        # Perturbing Doc A text tokens or trailing pad tokens changes Doc B parameter gradients by bit-exact 0.0.
        _assert_all_param_leaves_exact_zero_diff(
            self, grads_b_packed, grads_b_alt_a_tok, label=f"{mode_name} grad_B vs perturbed Doc A tokens"
        )
        _assert_all_param_leaves_exact_zero_diff(
            self, grads_b_packed, grads_b_alt_pad, label=f"{mode_name} grad_B vs perturbed Pad tokens"
        )
        if is_mm:
          _assert_all_param_leaves_parity(
              self,
              grads_b_packed,
              grads_b_alt_img0,
              atol_floor=_MM_BATCH_REASSOC_TOL,
              rtol=_MM_BATCH_REASSOC_TOL,
              frob_rel_ceiling=_MM_BATCH_REASSOC_TOL,
              label="multimodal grad_B vs perturbed Doc A tokens + Image 0",
          )

        # Per-document standalone gradient parity across EVERY parameter leaf (`frob_rel <= 1e-5`).
        num_leaves, _, _ = _assert_all_param_leaves_parity(
            self,
            grads_b_packed,
            grads_b_solo,
            atol_floor=_FP32_ATOL_FLOOR,
            rtol=_FP32_RTOL,
            frob_rel_ceiling=_FP32_FROB_REL_CEILING,
            label=f"{mode_name} grad_B packed vs standalone",
        )
        self.assertEqual(num_leaves, _EXPECTED_UNSCANNED_LEAVES)

        # Total-loss gradient superposition: `grad(L_packed) == grad(L_A_standalone) + grad(L_B_standalone)`
        # (`L_packed = w_A * mean_A + w_B * mean_B`, equivalent up to constant `N_total` to unnormalized token-sum).
        # In fp32 (`matmul_precision="highest"`), GEMM/einsum contractions over `S_packed=40` vs `S_standalone=16`
        # introduce IEEE-754 non-associativity bounded by `1e-5` (`~1e-6` measured), which vanishes to
        # `~1e-15 <= 1e-10` in fp64 (`test_fp64_all_leaf_superposition_and_standalone_parity`).
        grads_sum_solo = jax.tree.map(lambda ga, gb: ga + gb, grads_a_solo, grads_b_solo)
        _assert_all_param_leaves_parity(
            self,
            grads_total,
            grads_sum_solo,
            atol_floor=_FP32_ATOL_FLOOR,
            rtol=_FP32_RTOL,
            frob_rel_ceiling=_FP32_FROB_REL_CEILING,
            label=f"{mode_name} fp32 total-loss gradient superposition",
        )

  def test_fp64_all_leaf_superposition_and_standalone_parity(self) -> None:
    """Group 2 (fp64): Upcast-only `float64` check proving `frob_rel <= 1e-10` across all parameter leaves."""
    src_bundle = self.unscanned
    mm = _make_aligned_pack_inputs(seed=303, multimodal_mode=True)
    with _fp64_upcast_context(src_bundle.cfg):
      with src_bundle.mesh:
        model64 = model_creation_utils.from_pretrained(src_bundle.cfg, mesh=src_bundle.mesh, model_mode=MODEL_MODE_TRAIN)
      # Promote exact randomized MoEBiasVar buffers (`bias` and `bias_vl`) from the fp32 model.
      src_mod = mm_testing.apply_with_pure_params(
          src_bundle.graphdef, src_bundle.param_state, src_bundle.pure_params, src_bundle.rest_state
      )
      for (_, lyr_src), (_, lyr_dst) in zip(
          _iter_decoder_layer_modules(src_mod, src_bundle.cfg),
          _iter_decoder_layer_modules(model64, src_bundle.cfg),
      ):
        lyr_dst.mlp.MoeBlock_0.gate.bias[...] = lyr_src.mlp.MoeBlock_0.gate.bias[...].astype(jnp.float64)
        lyr_dst.mlp.MoeBlock_0.gate.bias_vl[...] = lyr_src.mlp.MoeBlock_0.gate.bias_vl[...].astype(jnp.float64)
      gdef64, pstate64, _, rest64 = mm_testing.extract_pure_params(model64)
      pure64 = jax.tree.map(lambda x: x.astype(jnp.float64), src_bundle.pure_params)

      @jax.jit
      def fg64(
          params: dict[str, Any],
          tokens: jax.Array,
          positions: jax.Array,
          segment_ids: jax.Array,
          *,
          targets: jax.Array,
          weights: jax.Array,
          images: jax.Array,
          image_masks: jax.Array,
      ) -> tuple[jax.Array, dict[str, Any]]:
        def loss_fn(p: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
          mod = mm_testing.apply_with_pure_params(gdef64, pstate64, p, rest64)
          logits = mod(
              decoder_input_tokens=tokens,
              decoder_positions=positions,
              decoder_segment_ids=segment_ids,
              encoder_images=images,
              encoder_image_masks=image_masks,
              enable_dropout=False,
          )
          return _ce_token_sum(logits, targets, weights), logits

        (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return logits, grads

      imgs_p64 = mm["imgs_packed"].astype(jnp.float64)
      img_a64 = mm["img_0"].astype(jnp.float64)
      img_b64 = mm["img_1"].astype(jnp.float64)
      wa_p64 = mm["w_a_packed"].astype(jnp.float64)
      wb_p64 = mm["w_b_packed"].astype(jnp.float64)
      mask_a64 = mm["mask_a"].astype(jnp.float64)
      mask_b64 = mm["mask_b"].astype(jnp.float64)

      with src_bundle.mesh:
        logits_p64, grads_tot64 = fg64(
            pure64,
            mm["tok_packed"],
            mm["pos_packed"],
            mm["seg_packed"],
            targets=mm["tgt_packed"],
            weights=wa_p64 + wb_p64,
            images=imgs_p64,
            image_masks=mm["imask_2"],
        )
        _, grads_b_p64 = fg64(
            pure64,
            mm["tok_packed"],
            mm["pos_packed"],
            mm["seg_packed"],
            targets=mm["tgt_packed"],
            weights=wb_p64,
            images=imgs_p64,
            image_masks=mm["imask_2"],
        )
        logits_a64, grads_a64 = fg64(
            pure64,
            mm["tok_a"],
            jnp.arange(_DOC_A_LEN, dtype=jnp.int32)[None, :],
            jnp.ones((1, _DOC_A_LEN), dtype=jnp.int32),
            targets=mm["tgt_a"],
            weights=mask_a64,
            images=img_a64,
            image_masks=mm["imask_1"],
        )
        logits_b64, grads_b64 = fg64(
            pure64,
            mm["tok_b"],
            jnp.arange(_DOC_B_LEN, dtype=jnp.int32)[None, :],
            jnp.ones((1, _DOC_B_LEN), dtype=jnp.int32),
            targets=mm["tgt_b"],
            weights=mask_b64,
            images=img_b64,
            image_masks=mm["imask_1"],
        )

    self.assertEqual(logits_p64.dtype, jnp.float64)
    _assert_tensor_parity(
        self,
        logits_p64[:, :_DOC_A_LEN],
        logits_a64,
        atol_floor=_FP64_ATOL_FLOOR,
        rtol=_FP64_RTOL,
        frob_rel_ceiling=_FP64_FROB_REL_CEILING,
        label="fp64 Doc A logits packed vs standalone",
    )
    _assert_tensor_parity(
        self,
        logits_p64[:, _DOC_A_LEN : _DOC_A_LEN + _DOC_B_LEN],
        logits_b64,
        atol_floor=_FP64_ATOL_FLOOR,
        rtol=_FP64_RTOL,
        frob_rel_ceiling=_FP64_FROB_REL_CEILING,
        label="fp64 Doc B logits packed vs standalone",
    )
    _assert_all_param_leaves_parity(
        self,
        grads_b_p64,
        grads_b64,
        atol_floor=_FP64_ATOL_FLOOR,
        rtol=_FP64_RTOL,
        frob_rel_ceiling=_FP64_FROB_REL_CEILING,
        label="fp64 Doc B all-leaf gradients packed vs standalone",
    )
    grads_sum64 = jax.tree.map(lambda ga, gb: ga + gb, grads_a64, grads_b64)
    num_leaves64, _, _ = _assert_all_param_leaves_parity(
        self,
        grads_tot64,
        grads_sum64,
        atol_floor=_FP64_ATOL_FLOOR,
        rtol=_FP64_RTOL,
        frob_rel_ceiling=_FP64_FROB_REL_CEILING,
        label="fp64 total-loss gradient superposition",
    )
    self.assertEqual(num_leaves64, _EXPECTED_UNSCANNED_LEAVES)

  def test_backward_input_embedding_zero_cross_doc_gradient(self) -> None:
    """Group 2: Differentiates `loss_B` and `loss_A` w.r.t. input embeddings (`embed_A`, `embed_B`, `embed_pad`)."""
    bundle = self.unscanned
    doc_b_slice = slice(_DOC_A_LEN, _DOC_A_LEN + _DOC_B_LEN)
    pad_slice = slice(_DOC_A_LEN + _DOC_B_LEN, _SEQ_PACKED)

    for mode_name, is_mm, seed in (("text", False, 401), ("multimodal", True, 402)):
      with self.subTest(mode=mode_name):
        data = _make_aligned_pack_inputs(seed=seed, multimodal_mode=is_mm)
        imgs_p = data["imgs_packed"] if is_mm else None
        imask_p = data["imask_2"] if is_mm else None
        embed_in = _compute_merged_input_embeddings(
            bundle,
            data["tok_packed"],
            data["pos_packed"],
            images=imgs_p,
            image_masks=imask_p,
        )
        with bundle.mesh:
          logits_direct = bundle.fwd_fn(
              bundle.pure_params,
              data["tok_packed"],
              data["pos_packed"],
              data["seg_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          logits_via_emb, grad_emb_from_b = bundle.embed_grad_fn(
              bundle.pure_params,
              embed_in,
              data["tok_packed"],
              data["pos_packed"],
              segment_ids=data["seg_packed"],
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          _, grad_emb_from_a = bundle.embed_grad_fn(
              bundle.pure_params,
              embed_in,
              data["tok_packed"],
              data["pos_packed"],
              segment_ids=data["seg_packed"],
              targets=data["tgt_packed"],
              weights=data["w_a_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )

        # Confirm `decoder_input_embeddings=embed_in` matches the fused production forward path within fp32 precision.
        _assert_tensor_parity(
            self,
            logits_via_emb[:, : _DOC_A_LEN + _DOC_B_LEN],
            logits_direct[:, : _DOC_A_LEN + _DOC_B_LEN],
            atol_floor=_FP32_ATOL_FLOOR,
            rtol=_FP32_RTOL,
            frob_rel_ceiling=_FP32_FROB_REL_CEILING,
            label=f"{mode_name} decoder_input_embeddings vs direct forward",
        )

        gb_np = np.asarray(grad_emb_from_b, dtype=np.float64)
        self.assertEqual(float(np.max(np.abs(gb_np[:, :_DOC_A_LEN, :]))), 0.0)
        self.assertEqual(float(np.max(np.abs(gb_np[:, pad_slice, :]))), 0.0)
        self.assertGreater(float(np.linalg.norm(gb_np[:, doc_b_slice, :].ravel())), 0.0)

        ga_np = np.asarray(grad_emb_from_a, dtype=np.float64)
        self.assertEqual(float(np.max(np.abs(ga_np[:, doc_b_slice, :]))), 0.0)
        self.assertEqual(float(np.max(np.abs(ga_np[:, pad_slice, :]))), 0.0)
        self.assertGreater(float(np.linalg.norm(ga_np[:, :_DOC_A_LEN, :].ravel())), 0.0)

  def test_scanned_layers_multimodal_forward_and_backward_isolation(self) -> None:
    """Covers `scan_layers=True` (`prefix + >=2 scanned blocks + suffix`) packed forward + all-leaf backward."""
    bundle = self.scanned
    mm = _make_aligned_pack_inputs(seed=501, multimodal_mode=True)
    doc_b_slice = slice(_DOC_A_LEN, _DOC_A_LEN + _DOC_B_LEN)
    pos_b_unpadded = jnp.arange(_DOC_B_LEN, dtype=jnp.int32)[None, :]
    seg_b_unpadded = jnp.ones((1, _DOC_B_LEN), dtype=jnp.int32)

    with bundle.mesh:
      logits_p, grads_tot = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm["tok_packed"],
          mm["pos_packed"],
          mm["seg_packed"],
          targets=mm["tgt_packed"],
          weights=mm["w_a_packed"] + mm["w_b_packed"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_2"],
      )
      logits_p_alt_a, grads_b_alt_a = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm["tok_packed_alt_a"],
          mm["pos_packed"],
          mm["seg_packed"],
          targets=mm["tgt_packed"],
          weights=mm["w_b_packed"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_2"],
      )
      _, grads_b_packed = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm["tok_packed"],
          mm["pos_packed"],
          mm["seg_packed"],
          targets=mm["tgt_packed"],
          weights=mm["w_b_packed"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_2"],
      )
      logits_a_padded, grads_a_padded = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm["tok_a_padded"],
          mm["pos_a_padded"],
          mm["seg_a_padded"],
          targets=mm["tgt_a_padded"],
          weights=mm["w_a_padded"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_10"],
      )
      logits_b_padded, grads_b_padded = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm["tok_b_padded"],
          mm["pos_b_padded"],
          mm["seg_b_padded"],
          targets=mm["tgt_b_padded"],
          weights=mm["w_b_padded"],
          images=mm["imgs_packed"],
          image_masks=mm["imask_01"],
      )
      logits_b_solo = bundle.fwd_fn(
          bundle.pure_params,
          mm["tok_b"],
          pos_b_unpadded,
          seg_b_unpadded,
          images=mm["img_1"],
          image_masks=mm["imask_1"],
      )

    self.assertEqual(
        float(np.max(np.abs(np.asarray(logits_p[:, doc_b_slice]) - np.asarray(logits_p_alt_a[:, doc_b_slice])))),
        0.0,
    )
    _assert_all_param_leaves_exact_zero_diff(
        self, grads_b_packed, grads_b_alt_a, label="scan_layers=True grad_B vs perturbed Doc A tokens"
    )
    _assert_tensor_parity(
        self,
        logits_p[:, :_DOC_A_LEN],
        logits_a_padded[:, :_DOC_A_LEN],
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="scan_layers=True Doc A logits packed vs [Doc A | Pad]",
    )
    _assert_tensor_parity(
        self,
        logits_p[:, doc_b_slice],
        logits_b_padded[:, :_DOC_B_LEN],
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="scan_layers=True Doc B logits packed vs [Doc B | Pad]",
    )
    _assert_tensor_parity(
        self,
        logits_p[:, doc_b_slice],
        logits_b_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="scan_layers=True Doc B logits packed vs standalone",
    )
    num_scanned_leaves, _, _ = _assert_all_param_leaves_parity(
        self,
        grads_b_packed,
        grads_b_padded,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="scan_layers=True grad_B packed vs [Doc B | Pad]",
    )
    self.assertGreater(num_scanned_leaves, 150)
    grads_sum_padded = jax.tree.map(lambda ga, gb: ga + gb, grads_a_padded, grads_b_padded)
    _assert_all_param_leaves_parity(
        self,
        grads_tot,
        grads_sum_padded,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="scan_layers=True total-loss gradient superposition",
    )

  def test_misaligned_boundaries_and_segment_zero_padding(self) -> None:
    """Group 3: Misaligned `len(A) in {11, 10}` CSA2 behavior and non-zero `segment 0` padding isolation."""
    bundle = self.unscanned
    rng = np.random.default_rng(601)
    tok_b = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_B_LEN), dtype=np.int32)
    tgt_b = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_B_LEN), dtype=np.int32)

    with bundle.mesh:
      logits_b_offset0 = np.asarray(
          bundle.fwd_fn(
              bundle.pure_params,
              jnp.asarray(tok_b, dtype=jnp.int32),
              jnp.arange(_DOC_B_LEN, dtype=jnp.int32)[None, :],
              jnp.ones((1, _DOC_B_LEN), dtype=jnp.int32),
          ),
          dtype=np.float64,
      )

    # Test both `len(A) = 11` (odd: `% compress_ratio != 0`) and `len(A) = 10` (even: `% compress_ratio == 0`,
    # `% candidate_block_size != 0` and `% (compress_ratio * candidate_block_size) != 0`).
    # Mathematical truth (`attention_deepseek41.py:125-134,226,338-350`):
    #  - Cross-document isolation is strictly bit-exact (`0.0`) for BOTH `len(A) = 11` and `len(A) = 10`.
    #  - Same-offset `[Pad (len_A, seg=0) | Doc B (seg=2)]` matches `[Doc A (len_A, seg=1) | Doc B (seg=2)]`
    #    bit-for-bit (`0.0`) in both logits and all-leaf gradients because `DeepSeek41Indexer` rejects any block
    #    where `(block_start_seg != query_seg) | (block_end_seg != query_seg)` (`attention_deepseek41.py:338-350`).
    #  - Compared against standalone `[Doc B at offset 0]`:
    #    * `len(A) = 11` differs (`O(1)`) because `DeepSeek41Compressor` (`attention_deepseek41.py:226`) pools fixed
    #      physical pairs `(10, 11), (12, 13), ...`, dropping straddling block `(10, 11)` via the segment filter
    #      (`attention_deepseek41.py:338-350`) and pairing Doc B tokens as `(1, 2), (3, 4), ...`.
    #    * `len(A) = 10` ALSO differs (`O(1)`) when `len(B) > candidate_topk_blocks * candidate_block_size` because
    #      `select_candidate_blocks` (`attention_deepseek41.py:125-134`) reshapes compressed scores on the global
    #      physical buffer into `candidate_block_size=4` chunks (`[8..11]` holds only Doc B's first 2 tokens
    #      `[10, 11]`), altering candidate-block top-k selection on `candidate_source_layer` (`5`) and `layer 6`.
    for len_a in (11, 10):
      with self.subTest(len_a=len_a):
        pad_tail_len = _SEQ_PACKED - len_a - _DOC_B_LEN
        tok_a1 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len_a), dtype=np.int32)
        tok_a2 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len_a), dtype=np.int32)
        tok_pad1 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, pad_tail_len), dtype=np.int32)
        tok_pad2 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, pad_tail_len), dtype=np.int32)

        tok_p1 = jnp.asarray(np.concatenate([tok_a1, tok_b, tok_pad1], axis=1), dtype=jnp.int32)
        tok_p2 = jnp.asarray(np.concatenate([tok_a2, tok_b, tok_pad2], axis=1), dtype=jnp.int32)
        pos_p = jnp.asarray(
            np.concatenate([np.arange(len_a), np.arange(_DOC_B_LEN), np.zeros(pad_tail_len, dtype=np.int32)])[None, :],
            dtype=jnp.int32,
        )
        seg_doc_a = jnp.asarray(
            np.concatenate(
                [
                    np.ones(len_a, dtype=np.int32),
                    np.full(_DOC_B_LEN, 2, dtype=np.int32),
                    np.zeros(pad_tail_len, dtype=np.int32),
                ]
            )[None, :],
            dtype=jnp.int32,
        )
        seg_pad_prefix = jnp.asarray(
            np.concatenate(
                [
                    np.zeros(len_a, dtype=np.int32),
                    np.full(_DOC_B_LEN, 2, dtype=np.int32),
                    np.zeros(pad_tail_len, dtype=np.int32),
                ]
            )[None, :],
            dtype=jnp.int32,
        )
        tgt_p = jnp.asarray(
            np.concatenate(
                [np.zeros((1, len_a), dtype=np.int32), tgt_b, np.zeros((1, pad_tail_len), dtype=np.int32)],
                axis=1,
            ),
            dtype=jnp.int32,
        )
        wb_p = jnp.asarray(
            np.concatenate(
                [
                    np.zeros((1, len_a), dtype=np.float32),
                    np.full((1, _DOC_B_LEN), 1.0 / _DOC_B_LEN, dtype=np.float32),
                    np.zeros((1, pad_tail_len), dtype=np.float32),
                ],
                axis=1,
            ),
            dtype=jnp.float32,
        )

        embed_in = _compute_merged_input_embeddings(bundle, tok_p1, pos_p)
        with bundle.mesh:
          logits_1, grads_1 = bundle.fwd_and_grad_fn(
              bundle.pure_params, tok_p1, pos_p, seg_doc_a, targets=tgt_p, weights=wb_p
          )
          logits_2, grads_2 = bundle.fwd_and_grad_fn(
              bundle.pure_params, tok_p2, pos_p, seg_doc_a, targets=tgt_p, weights=wb_p
          )
          logits_pad_pref, grads_pad_pref = bundle.fwd_and_grad_fn(
              bundle.pure_params, tok_p2, pos_p, seg_pad_prefix, targets=tgt_p, weights=wb_p
          )
          _, grad_emb = bundle.embed_grad_fn(
              bundle.pure_params, embed_in, tok_p1, pos_p, segment_ids=seg_doc_a, targets=tgt_p, weights=wb_p
          )

        b_slice = slice(len_a, len_a + _DOC_B_LEN)
        l1_b = np.asarray(logits_1[:, b_slice], dtype=np.float64)
        l2_b = np.asarray(logits_2[:, b_slice], dtype=np.float64)
        lpad_b = np.asarray(logits_pad_pref[:, b_slice], dtype=np.float64)

        # 1. Bit-exact forward and backward perturbation invariance (`Doc A` + non-zero `segment 0` pad).
        self.assertEqual(float(np.max(np.abs(l1_b - l2_b))), 0.0)
        _assert_all_param_leaves_exact_zero_diff(self, grads_1, grads_2, label=f"len_a={len_a} grad_B invariance")

        # 2. Bit-exact zero input-embedding gradient on `embed_A` and `embed_pad`.
        grad_emb_np = np.asarray(grad_emb, dtype=np.float64)
        self.assertEqual(float(np.max(np.abs(grad_emb_np[:, :len_a, :]))), 0.0)
        self.assertEqual(float(np.max(np.abs(grad_emb_np[:, len_a + _DOC_B_LEN :, :]))), 0.0)
        self.assertGreater(float(np.linalg.norm(grad_emb_np[:, b_slice, :].ravel())), 0.0)

        # 3. Same-offset `[Pad(len_A, seg=0) | Doc B(seg=2)]` matches `[Doc A(len_A, seg=1) | Doc B(seg=2)]` (<= 1e-6).
        self.assertLessEqual(float(np.max(np.abs(l1_b - lpad_b))), 1e-6)
        _assert_all_param_leaves_parity(
            self,
            grads_1,
            grads_pad_pref,
            atol_floor=1e-6,
            rtol=1e-6,
            frob_rel_ceiling=1e-6,
            label=f"len_a={len_a} Doc A vs same-offset Pad prefix",
        )

        # 4. Offset-0 standalone `[Doc B]` differs measurably due to physical compressor / candidate-block grid.
        self.assertGreater(
            float(np.max(np.abs(l1_b - logits_b_offset0))),
            _MISALIGNED_MIN_DIFF,
            msg=f"len_a={len_a}: expected physical grid shift vs offset-0 standalone Doc B",
        )

  def test_multimodal_images_adjacent_to_document_boundary(self) -> None:
    """Group 3: Multimodal images flush against `A|B` boundary (Case 3a: trailing A image; Case 3b: leading B image)."""
    bundle = self.unscanned
    doc_b_slice = slice(_DOC_A_LEN, _DOC_A_LEN + _DOC_B_LEN)
    pad_slice = slice(_DOC_A_LEN + _DOC_B_LEN, _SEQ_PACKED)
    pos_b_unpadded = jnp.arange(_DOC_B_LEN, dtype=jnp.int32)[None, :]
    seg_b_unpadded = jnp.ones((1, _DOC_B_LEN), dtype=jnp.int32)

    # Case 3a: Doc A (`len=16`) ends with an 8-token image span at `[8:16]` (`len(A) - 1 = 15`), immediately before
    # Doc B's text token `0` at index `16`. Exercises Engram `DEAD`/`image_mask` shift barrier, SWA/CSA2 boundary,
    # and `bias_vl` pointwise non-bleeding (`moe.py:770-781`).
    rng = np.random.default_rng(701)
    tok_a_3a = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_A_LEN), dtype=np.int32)
    tok_a_3a_alt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_A_LEN), dtype=np.int32)
    tok_a_3a[:, _DOC_A_LEN - _IMAGE_SPAN_LEN : _DOC_A_LEN] = _IMAGE_TOKEN_ID
    tok_a_3a_alt[:, _DOC_A_LEN - _IMAGE_SPAN_LEN : _DOC_A_LEN] = _IMAGE_TOKEN_ID
    tok_b_txt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_B_LEN), dtype=np.int32)
    tgt_b_txt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _DOC_B_LEN), dtype=np.int32)
    pad_3a = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _PAD_LEN), dtype=np.int32)
    img0_3a = jnp.asarray(
        rng.standard_normal((1, 1, _VISION_NUM_PATCHES, _VISION_PATCH_DIM)).astype(np.float32), dtype=jnp.float32
    )
    img0_3a_alt = jnp.asarray(
        rng.standard_normal((1, 1, _VISION_NUM_PATCHES, _VISION_PATCH_DIM)).astype(np.float32), dtype=jnp.float32
    )

    tok_p_3a = jnp.asarray(np.concatenate([tok_a_3a, tok_b_txt, pad_3a], axis=1), dtype=jnp.int32)
    tok_p_3a_alt = jnp.asarray(np.concatenate([tok_a_3a_alt, tok_b_txt, pad_3a], axis=1), dtype=jnp.int32)
    pos_p = jnp.asarray(
        np.concatenate([np.arange(_DOC_A_LEN), np.arange(_DOC_B_LEN), np.zeros(_PAD_LEN, dtype=np.int32)])[None, :],
        dtype=jnp.int32,
    )
    seg_p = jnp.asarray(
        np.concatenate(
            [
                np.ones(_DOC_A_LEN, dtype=np.int32),
                np.full(_DOC_B_LEN, 2, dtype=np.int32),
                np.zeros(_PAD_LEN, dtype=np.int32),
            ]
        )[None, :],
        dtype=jnp.int32,
    )
    tgt_p_3a = jnp.asarray(
        np.concatenate(
            [np.zeros((1, _DOC_A_LEN), dtype=np.int32), tgt_b_txt, np.zeros((1, _PAD_LEN), dtype=np.int32)],
            axis=1,
        ),
        dtype=jnp.int32,
    )
    wb_3a = jnp.asarray(
        np.concatenate(
            [
                np.zeros((1, _DOC_A_LEN), dtype=np.float32),
                np.full((1, _DOC_B_LEN), 1.0 / _DOC_B_LEN, dtype=np.float32),
                np.zeros((1, _PAD_LEN), dtype=np.float32),
            ],
            axis=1,
        ),
        dtype=jnp.float32,
    )
    imask_10 = jnp.asarray([[1, 0]], dtype=jnp.int32)
    imgs_3a = jnp.concatenate([img0_3a, jnp.zeros_like(img0_3a)], axis=1)
    imgs_3a_alt = jnp.concatenate([img0_3a_alt, jnp.zeros_like(img0_3a)], axis=1)
    emb_3a = _compute_merged_input_embeddings(bundle, tok_p_3a, pos_p, images=imgs_3a, image_masks=imask_10)

    with bundle.mesh:
      logits_3a, grads_3a = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          tok_p_3a,
          pos_p,
          seg_p,
          targets=tgt_p_3a,
          weights=wb_3a,
          images=imgs_3a,
          image_masks=imask_10,
      )
      logits_3a_alt, grads_3a_alt = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          tok_p_3a_alt,
          pos_p,
          seg_p,
          targets=tgt_p_3a,
          weights=wb_3a,
          images=imgs_3a_alt,
          image_masks=imask_10,
      )
      logits_b_txt_solo, grads_b_txt_solo = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          jnp.asarray(tok_b_txt, dtype=jnp.int32),
          pos_b_unpadded,
          seg_b_unpadded,
          targets=jnp.asarray(tgt_b_txt, dtype=jnp.int32),
          weights=jnp.full((1, _DOC_B_LEN), 1.0 / _DOC_B_LEN, dtype=jnp.float32),
      )
      _, grad_emb_3a = bundle.embed_grad_fn(
          bundle.pure_params,
          emb_3a,
          tok_p_3a,
          pos_p,
          segment_ids=seg_p,
          targets=tgt_p_3a,
          weights=wb_3a,
          images=imgs_3a,
          image_masks=imask_10,
      )

    # Doc B is text-only and segment-isolated from Doc A, so perturbing Image 0 + Doc A text is bit-exact 0.0 on Doc B!
    self.assertEqual(
        float(np.max(np.abs(np.asarray(logits_3a[:, doc_b_slice]) - np.asarray(logits_3a_alt[:, doc_b_slice])))),
        0.0,
    )
    _assert_all_param_leaves_exact_zero_diff(self, grads_3a, grads_3a_alt, label="Case 3a grad_B invariance")
    _assert_tensor_parity(
        self,
        logits_3a[:, doc_b_slice],
        logits_b_txt_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="Case 3a Doc B logits vs text-only standalone",
    )
    # Every shared decoder leaf of `grads_3a` matches text-only standalone `grads_b_txt_solo` (vision leaves are 0).
    _assert_all_param_leaves_parity(
        self,
        grads_3a,
        grads_b_txt_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="Case 3a Doc B all-leaf gradients vs text-only standalone",
    )
    grad_emb_3a_np = np.asarray(grad_emb_3a, dtype=np.float64)
    self.assertEqual(float(np.max(np.abs(grad_emb_3a_np[:, :_DOC_A_LEN, :]))), 0.0)
    self.assertEqual(float(np.max(np.abs(grad_emb_3a_np[:, pad_slice, :]))), 0.0)
    self.assertGreater(float(np.linalg.norm(grad_emb_3a_np[:, doc_b_slice, :].ravel())), 0.0)

    # Case 3b: Doc A contains Image 0 at `[2:10]` and Doc B starts with Image 1 at index `len(A) = 16` (`doc_b[0:8]`)
    # immediately after Doc A's last text token `15`. Verifies `cumsum_mask` pairs Image 1 to Doc B, `bias_vl` routes
    # Doc B's boundary image identically to standalone Doc B, and perturbing Image 0 + Doc A leaves Doc B invariant.
    mm_3b = _make_aligned_pack_inputs(seed=702, multimodal_mode=True, doc_a_span_start=2, doc_b_span_start=0)
    emb_3b = _compute_merged_input_embeddings(
        bundle, mm_3b["tok_packed"], mm_3b["pos_packed"], images=mm_3b["imgs_packed"], image_masks=mm_3b["imask_2"]
    )
    with bundle.mesh:
      logits_3b, grads_3b = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm_3b["tok_packed"],
          mm_3b["pos_packed"],
          mm_3b["seg_packed"],
          targets=mm_3b["tgt_packed"],
          weights=mm_3b["w_b_packed"],
          images=mm_3b["imgs_packed"],
          image_masks=mm_3b["imask_2"],
      )
      logits_3b_alt, grads_3b_alt = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm_3b["tok_packed_alt_a"],
          mm_3b["pos_packed"],
          mm_3b["seg_packed"],
          targets=mm_3b["tgt_packed"],
          weights=mm_3b["w_b_packed"],
          images=mm_3b["imgs_packed_alt_0"],
          image_masks=mm_3b["imask_2"],
      )
      logits_3b_solo, grads_3b_solo = bundle.fwd_and_grad_fn(
          bundle.pure_params,
          mm_3b["tok_b"],
          pos_b_unpadded,
          seg_b_unpadded,
          targets=mm_3b["tgt_b"],
          weights=mm_3b["mask_b"],
          images=mm_3b["img_1"],
          image_masks=mm_3b["imask_1"],
      )
      logits_3b_wrong_img = bundle.fwd_fn(
          bundle.pure_params,
          mm_3b["tok_b"],
          pos_b_unpadded,
          seg_b_unpadded,
          images=mm_3b["img_0"],
          image_masks=mm_3b["imask_1"],
      )
      _, grad_emb_3b = bundle.embed_grad_fn(
          bundle.pure_params,
          emb_3b,
          mm_3b["tok_packed"],
          mm_3b["pos_packed"],
          segment_ids=mm_3b["seg_packed"],
          targets=mm_3b["tgt_packed"],
          weights=mm_3b["w_b_packed"],
          images=mm_3b["imgs_packed"],
          image_masks=mm_3b["imask_2"],
      )

    _assert_tensor_parity(
        self,
        logits_3b[:, doc_b_slice],
        logits_3b_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="Case 3b Doc B logits vs standalone Image 1",
    )
    # Prove `cumsum_mask` specifically paired Image 1 (not Image 0) with Doc B.
    self.assertGreater(
        float(np.max(np.abs(np.asarray(logits_3b[:, doc_b_slice]) - np.asarray(logits_3b_wrong_img)))),
        1e-2,
    )
    _assert_all_param_leaves_parity(
        self,
        grads_3b,
        grads_3b_solo,
        atol_floor=_FP32_ATOL_FLOOR,
        rtol=_FP32_RTOL,
        frob_rel_ceiling=_FP32_FROB_REL_CEILING,
        label="Case 3b Doc B all-leaf gradients vs standalone Image 1",
    )
    self.assertLessEqual(
        float(
            np.max(
                np.abs(
                    np.asarray(logits_3b[:, doc_b_slice], dtype=np.float64)
                    - np.asarray(logits_3b_alt[:, doc_b_slice], dtype=np.float64)
                )
            )
        ),
        _MM_BATCH_REASSOC_TOL,
    )
    _assert_all_param_leaves_parity(
        self,
        grads_3b,
        grads_3b_alt,
        atol_floor=_MM_BATCH_REASSOC_TOL,
        rtol=_MM_BATCH_REASSOC_TOL,
        frob_rel_ceiling=_MM_BATCH_REASSOC_TOL,
        label="Case 3b grad_B vs perturbed Image 0 + Doc A",
    )
    grad_emb_3b_np = np.asarray(grad_emb_3b, dtype=np.float64)
    self.assertEqual(float(np.max(np.abs(grad_emb_3b_np[:, :_DOC_A_LEN, :]))), 0.0)
    self.assertEqual(float(np.max(np.abs(grad_emb_3b_np[:, pad_slice, :]))), 0.0)
    self.assertGreater(float(np.linalg.norm(grad_emb_3b_np[:, doc_b_slice, :].ravel())), 0.0)

  def test_three_packed_documents_misaligned_pairwise_isolation(self) -> None:
    """Group 3: 3 misaligned packed documents `[9 | 15 | 9 | Pad(7)]` forward, backward, and 3x4 Jacobian blocks."""
    bundle = self.unscanned
    len1, len2, len3 = 9, 15, 9
    pad_len = _SEQ_PACKED - len1 - len2 - len3
    self.assertEqual(pad_len, 7)
    slices = (
        slice(0, len1),
        slice(len1, len1 + len2),
        slice(len1 + len2, len1 + len2 + len3),
        slice(len1 + len2 + len3, _SEQ_PACKED),
    )

    rng = np.random.default_rng(801)
    d1 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len1), dtype=np.int32)
    d1_alt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len1), dtype=np.int32)
    d2 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len2), dtype=np.int32)
    d2_alt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len2), dtype=np.int32)
    d3 = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len3), dtype=np.int32)
    d3_alt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, len3), dtype=np.int32)
    pad = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, pad_len), dtype=np.int32)
    pad_alt = rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, pad_len), dtype=np.int32)
    tgts = jnp.asarray(
        rng.integers(_MIN_TEXT_TOKEN_ID, _MAX_TEXT_TOKEN_ID, size=(1, _SEQ_PACKED), dtype=np.int32),
        dtype=jnp.int32,
    )

    tok_base = jnp.asarray(np.concatenate([d1, d2, d3, pad], axis=1), dtype=jnp.int32)
    tok_pert = (
        jnp.asarray(np.concatenate([d1_alt, d2, d3, pad], axis=1), dtype=jnp.int32),
        jnp.asarray(np.concatenate([d1, d2_alt, d3, pad], axis=1), dtype=jnp.int32),
        jnp.asarray(np.concatenate([d1, d2, d3_alt, pad], axis=1), dtype=jnp.int32),
        jnp.asarray(np.concatenate([d1, d2, d3, pad_alt], axis=1), dtype=jnp.int32),
    )
    pos_3 = jnp.asarray(
        np.concatenate([np.arange(len1), np.arange(len2), np.arange(len3), np.zeros(pad_len, dtype=np.int32)])[None, :],
        dtype=jnp.int32,
    )
    seg_3 = jnp.asarray(
        np.concatenate(
            [
                np.ones(len1, dtype=np.int32),
                np.full(len2, 2, dtype=np.int32),
                np.full(len3, 3, dtype=np.int32),
                np.zeros(pad_len, dtype=np.int32),
            ]
        )[None, :],
        dtype=jnp.int32,
    )
    doc_weights = tuple(
        jnp.asarray((np.asarray(seg_3) == doc_id).astype(np.float32), dtype=jnp.float32) for doc_id in (1, 2, 3)
    )

    embed_in = _compute_merged_input_embeddings(bundle, tok_base, pos_3)
    with bundle.mesh:
      logits_base = np.asarray(bundle.fwd_fn(bundle.pure_params, tok_base, pos_3, seg_3), dtype=np.float64)
      logits_pert = [np.asarray(bundle.fwd_fn(bundle.pure_params, tp, pos_3, seg_3), dtype=np.float64) for tp in tok_pert]
      base_grads = [
          bundle.fwd_and_grad_fn(bundle.pure_params, tok_base, pos_3, seg_3, targets=tgts, weights=doc_weights[doc_idx])[
              1
          ]
          for doc_idx in range(3)
      ]
      for pert_idx, tp in enumerate(tok_pert):
        for doc_idx in range(3):
          if doc_idx == pert_idx:
            continue
          self.assertEqual(
              float(np.max(np.abs(logits_base[:, slices[doc_idx]] - logits_pert[pert_idx][:, slices[doc_idx]]))),
              0.0,
              msg=f"Forward leakage into Doc {doc_idx + 1} when perturbing slot {pert_idx + 1}",
          )
          _, pert_grad = bundle.fwd_and_grad_fn(
              bundle.pure_params, tp, pos_3, seg_3, targets=tgts, weights=doc_weights[doc_idx]
          )
          _assert_all_param_leaves_exact_zero_diff(
              self,
              base_grads[doc_idx],
              pert_grad,
              label=f"3-doc grad_{doc_idx + 1} vs perturbed slot {pert_idx + 1}",
          )

      # Full 3x4 input-embedding gradient block matrix (`doc_idx in {0,1,2}` x `slot_idx in {0,1,2,3(pad)}`).
      for doc_idx in range(3):
        _, grad_emb = bundle.embed_grad_fn(
            bundle.pure_params,
            embed_in,
            tok_base,
            pos_3,
            segment_ids=seg_3,
            targets=tgts,
            weights=doc_weights[doc_idx],
        )
        grad_emb_np = np.asarray(grad_emb, dtype=np.float64)
        for slot_idx, sl in enumerate(slices):
          if slot_idx == doc_idx:
            self.assertGreater(float(np.linalg.norm(grad_emb_np[:, sl, :].ravel())), 0.0)
          else:
            self.assertEqual(
                float(np.max(np.abs(grad_emb_np[:, sl, :]))),
                0.0,
                msg=f"Non-zero off-diagonal input-embedding gradient d(L_{doc_idx + 1})/d(embed_{slot_idx + 1})",
            )

  def test_negative_controls_all_ones_segment_ids_break_isolation(self) -> None:
    """Group 4 (Mandatory): `decoder_segment_ids = ones_like(...)` MUST break forward and backward isolation."""
    bundle = self.unscanned
    doc_b_slice = slice(_DOC_A_LEN, _DOC_A_LEN + _DOC_B_LEN)
    pad_slice = slice(_DOC_A_LEN + _DOC_B_LEN, _SEQ_PACKED)

    for mode_name, is_mm, seed in (("text", False, 901), ("multimodal", True, 902)):
      with self.subTest(mode=mode_name):
        data = _make_aligned_pack_inputs(seed=seed, multimodal_mode=is_mm)
        seg_unmasked = jnp.ones_like(data["seg_packed"])
        imgs_p = data["imgs_packed"] if is_mm else None
        imgs_p_alt = data["imgs_packed_alt_0"] if is_mm else None
        imask_p = data["imask_2"] if is_mm else None

        embed_in = _compute_merged_input_embeddings(
            bundle, data["tok_packed"], data["pos_packed"], images=imgs_p, image_masks=imask_p
        )
        with bundle.mesh:
          logits_1, grads_b_1 = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_packed"],
              data["pos_packed"],
              seg_unmasked,
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )
          logits_2, grads_b_2 = bundle.fwd_and_grad_fn(
              bundle.pure_params,
              data["tok_packed_alt_a"],
              data["pos_packed"],
              seg_unmasked,
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p_alt,
              image_masks=imask_p,
          )
          _, grad_emb_unmasked = bundle.embed_grad_fn(
              bundle.pure_params,
              embed_in,
              data["tok_packed"],
              data["pos_packed"],
              segment_ids=seg_unmasked,
              targets=data["tgt_packed"],
              weights=data["w_b_packed"],
              images=imgs_p,
              image_masks=imask_p,
          )

        # (a) Forward isolation MUST break (`max |logits_B(A) - logits_B(A')| > 1e-2`).
        fwd_leak = float(
            np.max(
                np.abs(
                    np.asarray(logits_1[:, doc_b_slice], dtype=np.float64)
                    - np.asarray(logits_2[:, doc_b_slice], dtype=np.float64)
                )
            )
        )
        self.assertGreater(
            fwd_leak,
            _NEGCTL_LOGIT_MIN_DIFF,
            msg=f"{mode_name} negative control failed: forward isolation did not break ({fwd_leak:.3e})",
        )

        # (b) Backward input-embedding isolation MUST break (`||d(loss_B)/d(embed_A)||_2 > 1e-2`).
        grad_emb_np = np.asarray(grad_emb_unmasked, dtype=np.float64)
        embed_a_leak_norm = float(np.linalg.norm(grad_emb_np[:, :_DOC_A_LEN, :].ravel()))
        embed_pad_leak_norm = float(np.linalg.norm(grad_emb_np[:, pad_slice, :].ravel()))
        self.assertGreater(
            embed_a_leak_norm,
            _NEGCTL_EMBED_GRAD_MIN_NORM,
            msg=f"{mode_name} negative control failed: d(loss_B)/d(embed_A) stayed small ({embed_a_leak_norm:.3e})",
        )
        self.assertGreater(
            embed_pad_leak_norm,
            _NEGCTL_EMBED_GRAD_MIN_NORM,
            msg=f"{mode_name} negative control failed: d(loss_B)/d(embed_pad) stayed small ({embed_pad_leak_norm:.3e})",
        )

        # (c) Backward parameter-gradient isolation MUST break (`||grad_B(A) - grad_B(A')||_2 / ||grad_B(A)||_2 > 1e-3`).
        param_grad_rel_leak = _tree_relative_frob_diff(grads_b_1, grads_b_2)
        self.assertGreater(
            param_grad_rel_leak,
            _NEGCTL_PARAM_GRAD_MIN_REL,
            msg=f"{mode_name} negative control failed: parameter gradients stayed invariant ({param_grad_rel_leak:.3e})",
        )


if __name__ == "__main__":
  unittest.main()
