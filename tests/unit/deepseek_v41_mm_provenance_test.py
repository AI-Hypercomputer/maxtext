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

"""Provenance and self-check tests (T16) for the DeepSeek-V4.1-Flash multimodal reference oracle."""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import pathlib
from typing import Any
import unittest

import numpy as np
from PIL import Image
import pytest

from tests.utils.parity import layer_parity

torch: Any
AutoTokenizer: Any
image_processor: Any
ref_model: Any
multimodal: Any
vision: Any
try:
  import torch  # pylint: disable=g-import-not-at-top
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import image_processor  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import model as ref_model  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import vision  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  AutoTokenizer = None
  image_processor = None
  ref_model = None
  multimodal = None
  vision = None
  HAS_TORCH = False

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REF_DIR = _REPO_ROOT / "tests" / "utils" / "deepseek_v41_reference"
_PROVENANCE_PATH = _REF_DIR / "provenance.json"
_TOKENIZER_PATH = (
    os.environ["DEEPSEEK_V41_TOKENIZER"] if "DEEPSEEK_V41_TOKENIZER" in os.environ else "deepseek-ai/DeepSeek-V4.1-Flash"
)
_SKIP_AST_FIELDS = frozenset({"ctx", "type_comment"})
_R8_FP64_TOL = layer_parity.Tolerance(atol=1e-12, rtol=1e-12, frob_rel=1e-12)
_EXPECTED_MM_GRAD_NONE_COUNT = 48
_EXPECTED_MM_NON_GRAD_COUNT = 28
_EXPECTED_MM_DISCRETE_ONLY_COUNT = 20
_EXPECTED_DEVIATION_IDS = (
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
)


def _norm_docstrings(tree: ast.AST) -> ast.AST:
  """Normalizes docstrings with inspect.cleandoc and per-line rstrip."""
  stack = [tree]
  while stack:
    node = stack.pop()
    stack.extend(ast.iter_child_nodes(node))
    body = node.body if hasattr(node, "body") else None
    if (
        isinstance(body, list)
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
      doc = inspect.cleandoc(body[0].value.value)
      body[0].value.value = "\n".join(line.rstrip() for line in doc.splitlines())
  return tree


def _canon_ast(node: object) -> str:
  """Serializes an AST node into a Python-version-stable canonical string."""
  if isinstance(node, ast.AST):
    parts: list[str] = []
    for field in node._fields:
      if field in _SKIP_AST_FIELDS:
        continue
      value = getattr(node, field)
      if value is None or (isinstance(value, list) and not value):
        continue
      parts.append(f"{field}={_canon_ast(value)}")
    return f"{type(node).__name__}({','.join(parts)})"
  if isinstance(node, list):
    return "[" + ",".join(_canon_ast(v) for v in node) + "]"
  return repr(node)


def _canon_sha256(src: str) -> tuple[str, dict[str, str]]:
  """Returns `(module_canon_sha256, {top_level_symbol: canon_sha256_prefix16})`."""
  tree = _norm_docstrings(ast.parse(src))
  assert isinstance(tree, ast.Module)
  whole = hashlib.sha256(_canon_ast(tree).encode("utf-8")).hexdigest()
  syms = {
      n.name: hashlib.sha256(_canon_ast(n).encode("utf-8")).hexdigest()[:16]
      for n in tree.body
      if isinstance(n, (ast.ClassDef, ast.FunctionDef))
  }
  return whole, syms


def _numpy_vision_cos_sin_fp64(n_h: int, n_w: int, dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
  """Computes 2D vision RoPE cos/sin tables in float64 NumPy from first principles."""
  even_idx = np.arange(0, dim, 2, dtype=np.float64)
  inv_freq = 1.0 / (np.float64(theta) ** (even_idx / np.float64(dim)))
  hpos, wpos = np.meshgrid(
      np.arange(n_h, dtype=np.float64),
      np.arange(n_w, dtype=np.float64),
      indexing="ij",
  )
  freqs = np.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1) * inv_freq
  freqs = freqs.reshape(n_h * n_w, dim)
  return np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]


def _numpy_apply_rotary_fp64(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
  """Applies rotate-half 2D vision RoPE in float64 NumPy."""
  x1, x2 = np.split(x, 2, axis=-1)
  return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _lru_cache_currsize(fn: Any) -> int:
  """Returns the current size of an `@lru_cache`-wrapped function."""
  return int(fn.cache_info().currsize)


@pytest.mark.cpu_only
class DeepSeekV41MultimodalProvenanceTest(unittest.TestCase):
  """Verifies T16 provenance hashes, R8 fp32/fp64 properties, and mini-twin invariants."""

  tokenizer: object
  args: ref_model.ModelArgs
  model: multimodal.MultimodalTransformer
  batch_padded: multimodal.MiniVLBatch
  batch_exact: multimodal.MiniVLBatch

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    if not HAS_TORCH:
      return
    try:
      cls.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, local_files_only=True)
    except OSError:
      cls.tokenizer = AutoTokenizer.from_pretrained(str(_REPO_ROOT / "src/maxtext/assets/tokenizers/qwen3-tokenizer"))
    cls.args = multimodal.mini_model_args()
    cls.model = multimodal.build_multimodal_model(cls.args, tokenizer=cls.tokenizer)
    multimodal.init_mini_reference(cls.model, seed=1)
    cls.batch_padded = multimodal.build_mini_vl_batch(cls.args, grid=multimodal.MINI_GRID_PADDED, seed=0)
    cls.batch_exact = multimodal.build_mini_vl_batch(cls.args, grid=multimodal.MINI_GRID_EXACT, seed=1)

  def test_provenance_manifest_and_vendored_ast_hashes(self) -> None:
    manifest = json.loads(_PROVENANCE_PATH.read_text(encoding="utf-8"))
    self.assertEqual(manifest["repo_id"], "deepseek-ai/DeepSeek-V4.1-Flash")
    self.assertEqual(manifest["revision"], "314113f8a7dadb388215396ba5fbee2c610a04d6")
    deviation_ids = tuple(entry["id"] for entry in manifest["deviations"])
    self.assertEqual(deviation_ids, _EXPECTED_DEVIATION_IDS)
    for entry in manifest["deviations"]:
      for key in ("summary", "release_lines", "reason", "numerics", "how_checked"):
        self.assertTrue(entry[key], f"Empty {key} in deviation {entry['id']}")

    vendored = manifest["vendored_files"]
    self.assertEqual(set(vendored.keys()), {"vision.py", "image_processor.py"})
    for filename, meta in vendored.items():
      path = _REPO_ROOT / meta["vendored_path"]
      self.assertEqual(path, _REF_DIR / filename)
      src = path.read_text(encoding="utf-8")
      whole_sha, sym_shas = _canon_sha256(src)
      self.assertEqual(
          whole_sha,
          meta["canonical_ast_sha256"],
          f"Canonical AST sha256 mismatch for {filename}",
      )
      self.assertEqual(
          sym_shas,
          meta["symbols"],
          f"Per-symbol canonical AST sha256 mismatch for {filename}",
      )
      release_meta = manifest["release_originals"][meta["release_path"]]
      self.assertEqual(meta["release_raw_sha256"], release_meta["raw_sha256"])
      self.assertEqual(meta["canonical_ast_sha256"], release_meta["canonical_ast_sha256"])

    pinned_symbols = manifest["pinned_symbols"]
    self.assertEqual(
        set(pinned_symbols.keys()),
        {
            "Gate",
            "MoE",
            "Block",
            "Engram",
            "Expert",
            "RMSNorm",
            "NgramHashState",
        },
    )
    symbol_cache: dict[str, dict[str, str]] = {}
    for sym_name, sym_meta in pinned_symbols.items():
      rel_file = sym_meta["file"]
      if rel_file not in symbol_cache:
        file_src = (_REPO_ROOT / rel_file).read_text(encoding="utf-8")
        _, symbol_cache[rel_file] = _canon_sha256(file_src)
      self.assertEqual(
          symbol_cache[rel_file][sym_name],
          sym_meta["canonical_ast_sha256_prefix16"],
          f"Pinned symbol {sym_name} in {rel_file} changed",
      )

  def test_negative_control_one_token_mutation_changes_hash(self) -> None:
    manifest = json.loads(_PROVENANCE_PATH.read_text(encoding="utf-8"))
    expected_meta = manifest["vendored_files"]["vision.py"]
    src = (_REF_DIR / "vision.py").read_text(encoding="utf-8")

    # Positive check: unmodified source matches the pinned hashes.
    orig_whole, orig_syms = _canon_sha256(src)
    self.assertEqual(orig_whole, expected_meta["canonical_ast_sha256"])
    self.assertEqual(orig_syms["Aligner"], expected_meta["symbols"]["Aligner"])

    # Negative control: mutate a single token (`stride=r` -> `stride=1`) in memory.
    self.assertEqual(src.count("stride=r"), 1)
    mutated_src = src.replace("stride=r", "stride=1", 1)
    mutated_whole, mutated_syms = _canon_sha256(mutated_src)
    self.assertNotEqual(mutated_whole, expected_meta["canonical_ast_sha256"])
    self.assertNotEqual(mutated_syms["Aligner"], expected_meta["symbols"]["Aligner"])

  def test_numpy_vision_rope_without_torch(self) -> None:
    cos_np, sin_np = _numpy_vision_cos_sin_fp64(7, 5, 32, 10000.0)
    np.testing.assert_allclose(cos_np**2 + sin_np**2, np.ones_like(cos_np), rtol=1e-14, atol=1e-14)
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((35, 4, 64)).astype(np.float64)
    rot_np = _numpy_apply_rotary_fp64(x_np, cos_np, sin_np)
    np.testing.assert_allclose(
        np.linalg.norm(rot_np, axis=-1),
        np.linalg.norm(x_np, axis=-1),
        rtol=1e-14,
        atol=1e-14,
    )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_r8_patched_functions_bitwise_equal_at_fp32(self) -> None:
    g = torch.Generator().manual_seed(42)
    n_h, n_w = multimodal.MINI_GRID_PADDED
    n_patches = n_h * n_w
    rope_dim = self.args.vision_dim // self.args.vision_n_heads // 2
    head_dim = self.args.vision_dim // self.args.vision_n_heads

    q_fp32 = torch.randn(n_patches, self.args.vision_n_heads, head_dim, generator=g, dtype=torch.float32)
    vit_x_fp32 = torch.randn(n_patches, self.args.vision_dim, generator=g, dtype=torch.float32)
    lin_x_fp32 = torch.randn(6, self.args.dim, generator=g, dtype=torch.float32)
    lin_w_fp32 = torch.randn(self.args.moe_inter_dim, self.args.dim, generator=g, dtype=torch.float32)
    engram_layer = self.model.layers[1].engram
    assert engram_layer is not None
    engram_x_fp32 = torch.randn(
        self.args.max_batch_size,
        multimodal.MINI_SEQ_LEN,
        self.args.hc_mult,
        self.args.dim,
        generator=g,
        dtype=torch.float32,
    )
    engram_mask = ~self.batch_padded.image_mask
    assert self.model.engram_hash is not None
    engram_hashes = self.model.engram_hash(self.batch_padded.input_ids, 0, engram_mask).clone()
    layer_hashes = engram_hashes[:, :, engram_layer.layer_hash_index, :]
    gate_mod = self.model.layers[0].ffn.gate
    gate_x_fp32 = torch.randn(
        self.batch_padded.image_mask.numel(),
        self.args.dim,
        generator=g,
        dtype=torch.float32,
    )
    flat_image_mask = self.batch_padded.image_mask.reshape(-1)
    patches_fp32 = self.batch_padded.images[0][0].patches

    with torch.no_grad():
      orig_cos_sin_fn = vision.get_vision_cos_sin
      cos_orig, sin_orig = orig_cos_sin_fn(n_h, n_w, rope_dim, self.args.vision_rope_theta)
      self.assertGreater(_lru_cache_currsize(orig_cos_sin_fn), 0)
      rot_orig = vision.apply_rotary(q_fp32, cos_orig, sin_orig)
      assert self.model.vision is not None
      rmsnorm_orig = self.model.vision.norm(vit_x_fp32)
      vit_orig = self.model.vision(patches_fp32, n_h, n_w)
      lin_orig = ref_model.linear(lin_x_fp32, lin_w_fp32)
      engram_orig = engram_layer(engram_x_fp32, layer_hashes, engram_mask)
      gate_w_orig, gate_idx_orig = gate_mod(gate_x_fp32, flat_image_mask)

      with multimodal.fp64_reference_context():
        # Cache on the original function must be cleared upon entering the context.
        self.assertEqual(_lru_cache_currsize(orig_cos_sin_fn), 0)
        cos_patch, sin_patch = vision.get_vision_cos_sin(
            n_h,
            n_w,
            rope_dim,
            self.args.vision_rope_theta,
            dtype=torch.float32,
        )
        rot_patch = vision.apply_rotary(q_fp32, cos_patch, sin_patch)
        rmsnorm_patch = self.model.vision.norm(vit_x_fp32)
        vit_patch = self.model.vision(patches_fp32, n_h, n_w)
        lin_patch = ref_model.linear(lin_x_fp32, lin_w_fp32)
        engram_patch = engram_layer(engram_x_fp32, layer_hashes, engram_mask)
        gate_w_patch, gate_idx_patch = gate_mod(gate_x_fp32, flat_image_mask)
        # Repopulate the original LRU cache inside the context to verify exit clearing.
        orig_cos_sin_fn(n_h, n_w, rope_dim, self.args.vision_rope_theta)
        self.assertGreater(_lru_cache_currsize(orig_cos_sin_fn), 0)

      # Cache on the original function must be cleared upon exiting the context.
      self.assertIs(vision.get_vision_cos_sin, orig_cos_sin_fn)
      self.assertEqual(_lru_cache_currsize(orig_cos_sin_fn), 0)

    self.assertTrue(torch.equal(cos_patch, cos_orig))
    self.assertTrue(torch.equal(sin_patch, sin_orig))
    self.assertTrue(torch.equal(rot_patch, rot_orig))
    self.assertTrue(torch.equal(rmsnorm_patch, rmsnorm_orig))
    self.assertTrue(torch.equal(vit_patch, vit_orig))
    self.assertTrue(torch.equal(lin_patch, lin_orig))
    self.assertTrue(torch.equal(engram_patch, engram_orig))
    self.assertTrue(torch.equal(gate_w_patch, gate_w_orig))
    self.assertTrue(torch.equal(gate_idx_patch, gate_idx_orig))

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_r8_fp64_rope_and_rotary_match_numpy_formula(self) -> None:
    rope_dim = self.args.vision_dim // self.args.vision_n_heads // 2
    head_dim = self.args.vision_dim // self.args.vision_n_heads
    theta = self.args.vision_rope_theta
    rng = np.random.default_rng(123)

    for n_h, n_w in (multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT):
      x_np = rng.standard_normal((n_h * n_w, self.args.vision_n_heads, head_dim)).astype(np.float64)
      ct_np = rng.standard_normal(x_np.shape).astype(np.float64)
      cos_np, sin_np = _numpy_vision_cos_sin_fp64(n_h, n_w, rope_dim, theta)
      rot_np = _numpy_apply_rotary_fp64(x_np, cos_np, sin_np)
      ct1, ct2 = np.split(ct_np, 2, axis=-1)
      grad_np = np.concatenate([ct1 * cos_np + ct2 * sin_np, ct2 * cos_np - ct1 * sin_np], axis=-1)

      with multimodal.fp64_reference_context():
        cos_pt, sin_pt = vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
        self.assertEqual(cos_pt.dtype, torch.float64)
        self.assertEqual(sin_pt.dtype, torch.float64)
        x_pt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        rot_pt = vision.apply_rotary(x_pt, cos_pt, sin_pt)
        self.assertEqual(rot_pt.dtype, torch.float64)
        loss = (rot_pt * torch.tensor(ct_np, dtype=torch.float64)).sum()
        (grad_pt,) = torch.autograd.grad(loss, (x_pt,))

      cos_ok, cos_metrics = layer_parity.compare(cos_pt.detach().cpu().numpy(), cos_np, _R8_FP64_TOL)
      sin_ok, sin_metrics = layer_parity.compare(sin_pt.detach().cpu().numpy(), sin_np, _R8_FP64_TOL)
      rot_ok, rot_metrics = layer_parity.compare(rot_pt.detach().cpu().numpy(), rot_np, _R8_FP64_TOL)
      grad_ok, grad_metrics = layer_parity.compare(grad_pt.detach().cpu().numpy(), grad_np, _R8_FP64_TOL)
      self.assertTrue(cos_ok, f"fp64 RoPE cos mismatch on {(n_h, n_w)}: {cos_metrics}")
      self.assertTrue(sin_ok, f"fp64 RoPE sin mismatch on {(n_h, n_w)}: {sin_metrics}")
      self.assertTrue(rot_ok, f"fp64 apply_rotary mismatch on {(n_h, n_w)}: {rot_metrics}")
      self.assertTrue(
          grad_ok,
          f"fp64 apply_rotary grad mismatch on {(n_h, n_w)}: {grad_metrics}",
      )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_grad_none_census_matches_declared_classes(self) -> None:
    self.model.zero_grad(set_to_none=True)
    for param in self.model.parameters():
      param.requires_grad_(True)

    logits = self.model(
        self.batch_padded.input_ids,
        images=self.batch_padded.images,
        token_types=self.batch_padded.token_types,
        full_logits=True,
    )
    assert isinstance(logits, torch.Tensor)
    cotangent = torch.randn(
        logits.shape,
        generator=torch.Generator().manual_seed(7),
        dtype=logits.dtype,
    )
    (logits * cotangent).sum().backward()

    ref_grads = {name: p.grad for name, p in self.model.named_parameters()}
    grad_classes = multimodal.reference_gradient_classes(self.model, freeze_vision=False)
    layer_parity.validate_gradient_classes(grad_classes, set(ref_grads.keys()), ref_grads)
    self.assertEqual(len(grad_classes.non_gradient), _EXPECTED_MM_NON_GRAD_COUNT)
    self.assertEqual(len(grad_classes.discrete_only), _EXPECTED_MM_DISCRETE_ONLY_COUNT)
    self.assertEqual(len(grad_classes.frozen), 0)
    self.assertEqual(len(grad_classes.expected_grad_none), _EXPECTED_MM_GRAD_NONE_COUNT)

    for name in grad_classes.compared | grad_classes.padded:
      grad = ref_grads[name]
      self.assertIsNotNone(grad, f"Expected non-None gradient for {name}")
      assert grad is not None
      self.assertTrue(bool(torch.isfinite(grad).all()), f"Non-finite gradient for {name}")
      self.assertGreater(
          float(grad.abs().max().item()),
          0.0,
          f"All-zero gradient on active parameter {name}",
      )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_forward_with_images_matches_inference_mode_bitwise(self) -> None:
    buf = io.BytesIO()
    Image.fromarray(np.full((98, 70, 3), 128, dtype=np.uint8)).save(buf, format="PNG")
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor.load_image({"data": buf.getvalue()}, self.args)
    self.assertEqual((n_vit_h, n_vit_w), multimodal.MINI_GRID_PADDED)
    self.assertEqual((n_llm_h, n_llm_w), (3, 2))
    self.assertEqual(patches.shape, (35, 3, 14, 14))

    for batch in (self.batch_padded, self.batch_exact):
      logits_grad = self.model(
          batch.input_ids,
          images=batch.images,
          token_types=batch.token_types,
          full_logits=True,
      )
      logits_inf = self.model.forward_inference_mode(
          batch.input_ids,
          images=batch.images,
          token_types=batch.token_types,
          full_logits=True,
      )
      assert isinstance(logits_grad, torch.Tensor)
      assert isinstance(logits_inf, torch.Tensor)
      self.assertEqual(
          logits_grad.shape,
          (multimodal.MINI_BATCH_SIZE, multimodal.MINI_SEQ_LEN, self.args.vocab_size),
      )
      self.assertTrue(bool(torch.isfinite(logits_grad).all()))
      self.assertTrue(
          torch.equal(logits_grad.detach(), logits_inf),
          f"Forward with autograd differs from inference_mode on grid {(batch.n_vit_h, batch.n_vit_w)}",
      )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_bias_vl_swap_changes_image_token_topk(self) -> None:
    changed = multimodal.count_bias_vl_routing_changes(self.model, self.batch_padded)
    self.assertGreaterEqual(
        changed,
        1,
        "Expected bias_vl swap to change >= 1 image-token top-k expert selection",
    )
    for layer in self.model.layers:
      self.assertIsNotNone(layer.ffn.gate.bias_vl)
      assert layer.ffn.gate.bias_vl is not None
      self.assertEqual(layer.ffn.gate.bias_vl.dtype, torch.float32)
      self.assertEqual(layer.ffn.gate.bias_vl.shape, layer.ffn.gate.bias.shape)

    with torch.no_grad():
      logits_with_vl = self.model(
          self.batch_padded.input_ids,
          images=self.batch_padded.images,
          token_types=self.batch_padded.token_types,
          full_logits=True,
      )
      saved_bias_vl = [layer.ffn.gate.bias_vl.detach().clone() for layer in self.model.layers]
      try:
        for layer in self.model.layers:
          layer.ffn.gate.bias_vl.copy_(layer.ffn.gate.bias)
        logits_with_text_bias = self.model(
            self.batch_padded.input_ids,
            images=self.batch_padded.images,
            token_types=self.batch_padded.token_types,
            full_logits=True,
        )
      finally:
        for layer, saved in zip(self.model.layers, saved_bias_vl):
          layer.ffn.gate.bias_vl.copy_(saved)

    assert isinstance(logits_with_vl, torch.Tensor)
    assert isinstance(logits_with_text_bias, torch.Tensor)
    # Row 1 starts with 4 text tokens before its first image span at index 4:
    # causal attention means positions 0..3 cannot depend on image routing.
    self.assertTrue(torch.equal(logits_with_vl[1, :4], logits_with_text_bias[1, :4]))
    # Image-span positions must differ when bias_vl != bias.
    self.assertFalse(
        torch.equal(
            logits_with_vl[self.batch_padded.image_mask],
            logits_with_text_bias[self.batch_padded.image_mask],
        )
    )


if __name__ == "__main__":
  unittest.main()
