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

"""Unit tests for the DeepSeek-V4.1-Flash multimodal reference loader and golden-logit tool."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer

from tests.utils.parity import negctl

try:
  import torch
  from safetensors.torch import save_file
  from tests.utils.deepseek_v41_mm_golden import generate_mm_golden_logits
  from tests.utils.deepseek_v41_mm_golden import reference_loader
  from tests.utils.deepseek_v41_reference import image_processor
  from tests.utils.deepseek_v41_reference import model as ref_model
  from tests.utils.deepseek_v41_reference import multimodal
  from tests.utils.deepseek_v41_reference import runner

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  save_file = None  # type: ignore[assignment]
  generate_mm_golden_logits = None  # type: ignore[assignment]
  reference_loader = None  # type: ignore[assignment]
  image_processor = None  # type: ignore[assignment]
  ref_model = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  runner = None  # type: ignore[assignment]
  HAS_TORCH = False

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TEST_IMAGE_PATH = _REPO_ROOT / "tests" / "assets" / "test_image.jpg"
_LAZY_MEMORY_RATIO_CEILING = 0.65
_NEGCTL_REL_SCALE = 1e-2


def _sha256_file(path: pathlib.Path) -> str:
  """Computes the hex SHA-256 digest of `path`."""
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _clone_and_mutate_checkpoint(
    src_ckpt_dir: pathlib.Path,
    dst_ckpt_dir: pathlib.Path,
    mutate_fn: Callable[[list[dict[str, torch.Tensor]]], None],
) -> pathlib.Path:
  """Copies `src_ckpt_dir` to `dst_ckpt_dir`, mutates shard dicts, and syncs `index.json`."""
  shutil.copytree(src_ckpt_dir, dst_ckpt_dir)
  shard_paths = sorted(dst_ckpt_dir.glob("*.safetensors"))
  shard_dicts: list[dict[str, torch.Tensor]] = []
  for shard_path in shard_paths:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
      for key in sf.keys():
        tensors[key] = sf.get_tensor(key).clone()
    shard_dicts.append(tensors)

  mutate_fn(shard_dicts)

  weight_map: dict[str, str] = {}
  for shard_path, tensors in zip(shard_paths, shard_dicts, strict=True):
    save_file(tensors, str(shard_path), metadata={"format": "pt"})
    for key in tensors:
      weight_map[key] = shard_path.name

  index_path = dst_ckpt_dir / "model.safetensors.index.json"
  index_payload = {"metadata": {}, "weight_map": weight_map}
  index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")
  return dst_ckpt_dir


def _build_two_image_mini_batch(
    args: ref_model.ModelArgs,
) -> tuple[torch.Tensor, list[image_processor.ImageInput], torch.Tensor]:
  """Builds a deterministic `[1, 40]` 2-image + text batch exercising all multimodal stages."""
  seq_len = 40
  raw_img_bytes = _TEST_IMAGE_PATH.read_bytes()
  with Image.open(io.BytesIO(raw_img_bytes)) as im:
    rgb_a = im.convert("RGB")
  prep_a = generate_mm_golden_logits.prepare_letterboxed_image(
      "asset:test_image.jpg",
      rgb_a,
      raw_img_bytes,
      args,
      target_grid_hw=(6, 6),
  )

  rng = np.random.default_rng(42)
  synth_arr = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
  rgb_b = Image.fromarray(synth_arr, mode="RGB")
  png_buf = io.BytesIO()
  rgb_b.save(png_buf, format="PNG")
  prep_b = generate_mm_golden_logits.prepare_letterboxed_image(
      "synth:42",
      rgb_b,
      png_buf.getvalue(),
      args,
      target_grid_hw=(6, 6),
  )

  gen = torch.Generator().manual_seed(123)
  tokens = torch.randint(10, 5000, (1, seq_len), generator=gen, dtype=torch.int64)
  token_types = torch.full((1, seq_len), image_processor.TEXT, dtype=torch.int64)

  span_len_a = int(prep_a.token_types.numel())
  span_len_b = int(prep_b.token_types.numel())
  start_a, start_b = 0, 18
  tokens[0, start_a : start_a + span_len_a] = args.image_token_id
  token_types[0, start_a : start_a + span_len_a] = prep_a.token_types
  tokens[0, start_b : start_b + span_len_b] = args.image_token_id
  token_types[0, start_b : start_b + span_len_b] = prep_b.token_types

  images = [
      image_processor.ImageInput(
          start=start_a,
          patches=prep_a.patches_bf16,
          n_vit_h=prep_a.n_vit_h,
          n_vit_w=prep_a.n_vit_w,
          types=prep_a.token_types,
      ),
      image_processor.ImageInput(
          start=start_b,
          patches=prep_b.patches_bf16,
          n_vit_h=prep_b.n_vit_h,
          n_vit_w=prep_b.n_vit_w,
          types=prep_b.token_types,
      ),
  ]
  return tokens, images, token_types


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class DeepSeekV41MmGoldenToolTest(unittest.TestCase):
  """Tests `reference_loader` and `generate_mm_golden_logits` on quantized mini checkpoints."""

  _tmp_root: str
  _mini_ckpt_dir: pathlib.Path
  _tokenizer: Any

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls._tmp_root = tempfile.mkdtemp(prefix="ds41_mm_golden_tool_test_")
    cls._mini_ckpt_dir = generate_mm_golden_logits.build_quantized_mini_checkpoint(
        pathlib.Path(cls._tmp_root) / "mini_ckpt",
        seed=1,
    )
    cls._tokenizer = AutoTokenizer.from_pretrained(str(cls._mini_ckpt_dir))

  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(cls._tmp_root, ignore_errors=True)
    super().tearDownClass()

  def test_lazy_dequant_bit_exact_multimodal_and_text_and_memory_savings(self) -> None:
    """Verifies `lazy_dequant=True` matches `lazy_dequant=False` bit-for-bit with <65% resident bytes."""
    model_eager, args_eager = reference_loader.load_multimodal_reference_model(
        str(self._mini_ckpt_dir),
        dtype="bf16",
        max_seq_len=64,
        tokenizer=self._tokenizer,
        fp32_compute=True,
        lazy_dequant=False,
    )
    model_lazy, args_lazy = reference_loader.load_multimodal_reference_model(
        str(self._mini_ckpt_dir),
        dtype="bf16",
        max_seq_len=64,
        tokenizer=self._tokenizer,
        fp32_compute=True,
        lazy_dequant=True,
    )
    self.assertEqual(args_lazy, args_eager)
    self.assertTrue(args_lazy.vision_enabled)

    bytes_eager = reference_loader.resident_model_bytes(model_eager)
    bytes_lazy = reference_loader.resident_model_bytes(model_lazy)
    self.assertGreater(bytes_eager, 0)
    self.assertLess(
        bytes_lazy,
        _LAZY_MEMORY_RATIO_CEILING * bytes_eager,
        msg=f"Expected lazy resident bytes < {_LAZY_MEMORY_RATIO_CEILING} * eager ({bytes_lazy} vs {bytes_eager})",
    )

    mm_tokens, mm_images, mm_types = _build_two_image_mini_batch(args_eager)
    logits_eager_mm = reference_loader.forward_multimodal_logits(
        model_eager,
        mm_tokens,
        images=[mm_images],
        token_types=mm_types,
    )
    logits_lazy_mm = reference_loader.forward_multimodal_logits(
        model_lazy,
        mm_tokens,
        images=[mm_images],
        token_types=mm_types,
    )
    self.assertEqual(logits_lazy_mm.dtype, torch.float32)
    self.assertTrue(torch.isfinite(logits_eager_mm).all())
    self.assertEqual(float((logits_lazy_mm - logits_eager_mm).abs().max().item()), 0.0)
    self.assertTrue(torch.equal(logits_lazy_mm, logits_eager_mm))

    gen_txt = torch.Generator().manual_seed(777)
    text_tokens = torch.randint(10, 5000, (1, 40), generator=gen_txt, dtype=torch.int64)
    logits_eager_txt = reference_loader.forward_multimodal_logits(model_eager, text_tokens)
    logits_lazy_txt = reference_loader.forward_multimodal_logits(model_lazy, text_tokens)
    self.assertEqual(float((logits_lazy_txt - logits_eager_txt).abs().max().item()), 0.0)
    self.assertTrue(torch.equal(logits_lazy_txt, logits_eager_txt))

    # Negative control (U1 negctl.perturb_one_leaf): perturbing one parameter leaf on the
    # compared path by rel_scale in [5e-3, 2e-2] must cause the exact comparison to fail.
    assert model_lazy.aligner is not None
    orig_w1 = model_lazy.aligner.w1.weight.detach().clone()
    perturbed_tree = negctl.perturb_one_leaf(
        {"aligner_w1": jnp.asarray(orig_w1.cpu().numpy())},
        target_path="aligner_w1",
        rel_scale=_NEGCTL_REL_SCALE,
        seed=17,
    )
    try:
      with torch.no_grad():
        perturbed_np = np.array(perturbed_tree["aligner_w1"], copy=True)
        model_lazy.aligner.w1.weight.copy_(torch.from_numpy(perturbed_np).to(orig_w1.dtype))
      logits_perturbed_mm = reference_loader.forward_multimodal_logits(
          model_lazy,
          mm_tokens,
          images=[mm_images],
          token_types=mm_types,
      )
      self.assertFalse(torch.equal(logits_perturbed_mm, logits_eager_mm))
    finally:
      with torch.no_grad():
        model_lazy.aligner.w1.weight.copy_(orig_w1)

  def test_header_dtype_and_key_completeness_negative_controls(self) -> None:
    """Verifies header dtype checks (`BF16`/`F32`) and non-MTP key completeness guards."""
    hf_cfg = runner.load_hf_config(str(self._mini_ckpt_dir))
    margs = reference_loader.multimodal_model_args_from_hf_config(
        hf_cfg,
        dtype="bf16",
        max_seq_len=64,
        max_batch_size=1,
        fp32_compute=True,
    )

    # Positive liveness check outside assertRaisesRegex: unmodified checkpoint loads cleanly.
    clean_model = multimodal.build_multimodal_model(margs, tokenizer=self._tokenizer)
    reference_loader.load_multimodal_checkpoint_into(clean_model, str(self._mini_ckpt_dir), lazy_dequant=True)
    headers = reference_loader.read_safetensors_headers(str(self._mini_ckpt_dir))
    flat_dtypes = {k: meta["dtype"] for shard in headers.values() for k, meta in shard.items()}
    self.assertEqual(flat_dtypes["vision.patch_embed.proj.weight"], "BF16")
    self.assertEqual(flat_dtypes["layers.0.ffn.gate.bias_vl"], "F32")

    with tempfile.TemporaryDirectory(prefix="ds41_mm_neg_") as tmp:
      tmp_path = pathlib.Path(tmp)

      # 1. Saving a vision.* tensor as float32 -> ValueError.
      def _cast_vision_to_fp32(shards: list[dict[str, torch.Tensor]]) -> None:
        for shard in shards:
          if "vision.patch_embed.proj.weight" in shard:
            shard["vision.patch_embed.proj.weight"] = shard["vision.patch_embed.proj.weight"].to(torch.float32)

      bad_vision_dir = _clone_and_mutate_checkpoint(
          self._mini_ckpt_dir,
          tmp_path / "bad_vision_fp32",
          _cast_vision_to_fp32,
      )
      model_a = multimodal.build_multimodal_model(margs, tokenizer=self._tokenizer)
      with self.assertRaisesRegex(ValueError, r"BF16.*vision\.patch_embed\.proj\.weight"):
        reference_loader.load_multimodal_checkpoint_into(model_a, str(bad_vision_dir), lazy_dequant=True)

      # 2. Saving layers.0.ffn.gate.bias_vl as bfloat16 -> ValueError.
      def _cast_bias_vl_to_bf16(shards: list[dict[str, torch.Tensor]]) -> None:
        for shard in shards:
          if "layers.0.ffn.gate.bias_vl" in shard:
            shard["layers.0.ffn.gate.bias_vl"] = shard["layers.0.ffn.gate.bias_vl"].to(torch.bfloat16)

      bad_bias_dir = _clone_and_mutate_checkpoint(
          self._mini_ckpt_dir,
          tmp_path / "bad_bias_bf16",
          _cast_bias_vl_to_bf16,
      )
      model_b = multimodal.build_multimodal_model(margs, tokenizer=self._tokenizer)
      with self.assertRaisesRegex(ValueError, r"F32.*layers\.0\.ffn\.gate\.bias_vl"):
        reference_loader.load_multimodal_checkpoint_into(model_b, str(bad_bias_dir), lazy_dequant=True)

      # 3. Adding a bogus non-mtp.* key bogus.weight -> ValueError.
      def _add_bogus_key(shards: list[dict[str, torch.Tensor]]) -> None:
        shards[0]["bogus.weight"] = torch.ones((4, 4), dtype=torch.bfloat16)

      bogus_key_dir = _clone_and_mutate_checkpoint(
          self._mini_ckpt_dir,
          tmp_path / "bogus_key",
          _add_bogus_key,
      )
      model_c = multimodal.build_multimodal_model(margs, tokenizer=self._tokenizer)
      with self.assertRaisesRegex(ValueError, r"bogus\.weight"):
        reference_loader.load_multimodal_checkpoint_into(model_c, str(bogus_key_dir), lazy_dequant=True)

      # 4. Dropping aligner.w1.weight from the checkpoint -> ValueError.
      def _drop_aligner_w1(shards: list[dict[str, torch.Tensor]]) -> None:
        for shard in shards:
          if "aligner.w1.weight" in shard:
            del shard["aligner.w1.weight"]

      missing_aligner_dir = _clone_and_mutate_checkpoint(
          self._mini_ckpt_dir,
          tmp_path / "missing_aligner",
          _drop_aligner_w1,
      )
      model_d = multimodal.build_multimodal_model(margs, tokenizer=self._tokenizer)
      with self.assertRaisesRegex(ValueError, r"aligner\.w1\.weight"):
        reference_loader.load_multimodal_checkpoint_into(model_d, str(missing_aligner_dir), lazy_dequant=True)

  def test_r9_patch_dtype_cast_and_diagnostic_contexts_restore_state(self) -> None:
    """Verifies R9 bfloat16 patch casting under `fp32_compute=True` and clean context restoration."""
    model_fp32, args_fp32 = reference_loader.load_multimodal_reference_model(
        str(self._mini_ckpt_dir),
        dtype="bf16",
        max_seq_len=64,
        tokenizer=self._tokenizer,
        fp32_compute=True,
        lazy_dequant=True,
    )
    assert model_fp32.vision is not None
    self.assertEqual(model_fp32.vision.patch_embed.proj.weight.dtype, torch.float32)

    mm_tokens, mm_images, mm_types = _build_two_image_mini_batch(args_fp32)
    self.assertEqual(mm_images[0].patches.dtype, torch.bfloat16)

    logits_fp32 = reference_loader.forward_multimodal_logits(
        model_fp32,
        mm_tokens,
        images=[mm_images],
        token_types=mm_types,
    )
    self.assertEqual(logits_fp32.dtype, torch.float32)
    self.assertTrue(torch.isfinite(logits_fp32).all())
    self.assertEqual(mm_images[0].patches.dtype, torch.bfloat16)

    # Verify bf16_compute_context matches a native fp32_compute=False load and restores model_fp32.
    model_native_bf16, _ = reference_loader.load_multimodal_reference_model(
        str(self._mini_ckpt_dir),
        dtype="bf16",
        max_seq_len=64,
        tokenizer=self._tokenizer,
        fp32_compute=False,
        lazy_dequant=True,
    )
    logits_native_bf16 = reference_loader.forward_multimodal_logits(
        model_native_bf16,
        mm_tokens,
        images=[mm_images],
        token_types=mm_types,
    )
    # Reload fp32 compute globals before entering bf16_compute_context on model_fp32.
    ref_model.compute_fp32 = True

    with reference_loader.bf16_compute_context(model_fp32):
      self.assertFalse(ref_model.compute_fp32)
      self.assertEqual(model_fp32.vision.patch_embed.proj.weight.dtype, torch.bfloat16)
      logits_ctx_bf16 = reference_loader.forward_multimodal_logits(
          model_fp32,
          mm_tokens,
          images=[mm_images],
          token_types=mm_types,
      )
    self.assertTrue(ref_model.compute_fp32)
    self.assertEqual(model_fp32.vision.patch_embed.proj.weight.dtype, torch.float32)
    self.assertTrue(torch.equal(logits_ctx_bf16, logits_native_bf16))

    logits_after_bf16 = reference_loader.forward_multimodal_logits(
        model_fp32,
        mm_tokens,
        images=[mm_images],
        token_types=mm_types,
    )
    self.assertTrue(torch.equal(logits_after_bf16, logits_fp32))

    orig_gate_forward = ref_model.Gate.forward
    orig_indexer_forward = ref_model.Indexer.forward
    with reference_loader.one_ulp_routing_context():
      self.assertIsNot(ref_model.Gate.forward, orig_gate_forward)
      self.assertIsNot(ref_model.Indexer.forward, orig_indexer_forward)
      logits_ulp = reference_loader.forward_multimodal_logits(
          model_fp32,
          mm_tokens,
          images=[mm_images],
          token_types=mm_types,
      )
      self.assertTrue(torch.isfinite(logits_ulp).all())
    self.assertIs(ref_model.Gate.forward, orig_gate_forward)
    self.assertIs(ref_model.Indexer.forward, orig_indexer_forward)

    logits_after_ulp = reference_loader.forward_multimodal_logits(
        model_fp32,
        mm_tokens,
        images=[mm_images],
        token_types=mm_types,
    )
    self.assertTrue(torch.equal(logits_after_ulp, logits_fp32))

    # Direct Gate tie-flip + unscaled-weights check under one_ulp_routing_context:
    # tie odd expert 1 and even expert 2 at topk=1 with norm_topk_prob=False.
    tie_gate_args = dataclasses.replace(
        args_fp32,
        n_routed_experts=4,
        n_activated_experts=1,
        norm_topk_prob=False,
        route_scale=2.5,
        score_func="sigmoid",
    )
    tie_gate = ref_model.Gate(1, tie_gate_args)
    with torch.no_grad():
      tie_gate.weight.zero_()
      tie_gate.bias = torch.nn.Parameter(torch.tensor([0.0, 0.25, 0.25, 0.0], dtype=torch.float32), requires_grad=False)
    x_tie = torch.zeros((4, args_fp32.dim), dtype=torch.float32)
    w_base, idx_base = tie_gate(x_tie)
    self.assertEqual(idx_base.tolist(), [[1], [1], [1], [1]])
    self.assertTrue(torch.equal(w_base, torch.full((4, 1), 1.25, dtype=torch.float32)))
    with reference_loader.one_ulp_routing_context():
      w_ulp, idx_ulp = tie_gate(x_tie)
    self.assertEqual(idx_ulp.tolist(), [[2], [2], [2], [2]])
    self.assertTrue(torch.equal(w_ulp, w_base))

  def test_end_to_end_mini_generation_determinism_and_manifest(self) -> None:
    """Runs `generate_mm_golden_logits` twice in `--mini` mode and verifies determinism + manifest."""
    sample_subset = "L1,L2,L4,M1,M8"
    with tempfile.TemporaryDirectory(prefix="ds41_mm_gen_det_") as tmp:
      out_a = pathlib.Path(tmp) / "run_a"
      out_b = pathlib.Path(tmp) / "run_b"

      manifest_a = generate_mm_golden_logits.run_generation(
          output_dir=out_a,
          checkpoint_dir=str(self._mini_ckpt_dir),
          mini=True,
          lazy_dequant=True,
          samples=sample_subset,
          diagnostics=True,
      )
      manifest_b = generate_mm_golden_logits.run_generation(
          output_dir=out_b,
          checkpoint_dir=str(self._mini_ckpt_dir),
          mini=True,
          lazy_dequant=True,
          samples=sample_subset,
          diagnostics=True,
      )

      for sid in ("L1", "L2", "L4", "M1", "M8"):
        npz_a_path = out_a / f"{sid}.npz"
        npz_b_path = out_b / f"{sid}.npz"
        self.assertEqual(_sha256_file(npz_a_path), _sha256_file(npz_b_path))
        self.assertEqual(manifest_a["output_files_sha256"][f"{sid}.npz"], _sha256_file(npz_a_path))
        with np.load(npz_a_path) as arr_a, np.load(npz_b_path) as arr_b:
          self.assertEqual(set(arr_a.files), set(arr_b.files))
          for key in arr_a.files:
            self.assertTrue(np.array_equal(arr_a[key], arr_b[key]), msg=f"Mismatch in {sid}.npz[{key!r}]")

      # Check L2 matches release chat-mode encoding (<｜Assistant｜></think>, no <think>, \n\n around image spans).
      assistant_end_ids = [int(t) for t in self._tokenizer.encode("<｜Assistant｜></think>", add_special_tokens=False)]
      think_open_ids = [int(t) for t in self._tokenizer.encode("<think>", add_special_tokens=False)]
      nl_ids = [int(t) for t in self._tokenizer.encode("\n\n", add_special_tokens=False)]
      self.assertEqual(len(assistant_end_ids), 2)
      self.assertEqual(len(think_open_ids), 1)
      self.assertGreaterEqual(len(nl_ids), 1)
      with np.load(out_a / "L2.npz") as l2:
        l2_tokens = l2["tokens"].tolist()
        l2_starts = l2["span_starts"].tolist()
        self.assertEqual(len(l2_starts), 2)
        self.assertEqual(l2_tokens[-len(assistant_end_ids) :], assistant_end_ids)
        self.assertNotIn(think_open_ids[0], l2_tokens)
        for span_start in l2_starts:
          self.assertEqual(l2_tokens[span_start - len(nl_ids) : span_start], nl_ids)
          self.assertEqual(l2_tokens[span_start + 8 : span_start + 8 + len(nl_ids)], nl_ids)

      # Check M1 starts with IMAGE_START at token index 0 and exact length 48.
      with np.load(out_a / "M1.npz") as m1:
        self.assertEqual(int(m1["seq_len"]), 48)
        self.assertEqual(int(m1["tokens"][0]), 129264)
        self.assertEqual(int(m1["token_types"][0]), image_processor.IMAGE_START)
        self.assertEqual(tuple(m1["patches"].shape), (4, 36, 588))
        self.assertEqual(m1["image_masks"].tolist(), [1, 0, 0, 0])

      # Check M8 has 4 active images starting at token 0 and ending at token 96.
      with np.load(out_a / "M8.npz") as m8:
        self.assertEqual(int(m8["seq_len"]), 96)
        self.assertEqual(m8["image_masks"].tolist(), [1, 1, 1, 1])
        self.assertEqual(int(m8["token_types"][0]), image_processor.IMAGE_START)
        self.assertEqual(int(m8["token_types"][-1]), image_processor.IMAGE_END)

      # Verify mm_logit.pkl determinism and forward_pass_logit_checker schema compatibility.
      pkl_a_path = out_a / "mm_logit.pkl"
      pkl_b_path = out_b / "mm_logit.pkl"
      self.assertEqual(_sha256_file(pkl_a_path), _sha256_file(pkl_b_path))
      self.assertEqual(manifest_a["output_files_sha256"]["mm_logit.pkl"], _sha256_file(pkl_a_path))
      with open(pkl_a_path, "rb") as f:
        pkl_records = pickle.load(f)
      self.assertEqual([rec["sample_id"] for rec in pkl_records], ["L1", "L2", "L4"])
      for rec in pkl_records:
        for req_key in (
            "tokens",
            "logits",
            "prompt",
            "sample_id",
            "token_types",
            "encoder_images",
            "encoder_image_masks",
            "pixel_values",
            "grid_hw",
        ):
          self.assertIn(req_key, rec)
        self.assertTrue(np.array_equal(rec["pixel_values"], rec["encoder_images"]))

      # Verify manifest provenance, tool SHA-256s, R1..R10 deviations, and L4/M8 diagnostics.
      self.assertEqual(manifest_a["release_revision"], "314113f8a7dadb388215396ba5fbee2c610a04d6")
      self.assertEqual(len(manifest_a["u2_commit_sha"]), 40)
      expected_tool_files = (
          "tests/utils/deepseek_v41_mm_golden/__init__.py",
          "tests/utils/deepseek_v41_mm_golden/reference_loader.py",
          "tests/utils/deepseek_v41_mm_golden/generate_mm_golden_logits.py",
      )
      for rel_file in expected_tool_files:
        self.assertEqual(manifest_a["tool_files_sha256"][rel_file], _sha256_file(_REPO_ROOT / rel_file))

      deviation_ids = [entry["id"] for entry in manifest_a["deviations"]]
      self.assertEqual(deviation_ids, ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"])

      for diag_kind in ("bf16_compute", "one_ulp_routing"):
        self.assertEqual(set(manifest_a["diagnostics"][diag_kind]), {"L4", "M8"})
        for sid in ("L4", "M8"):
          stats_a = manifest_a["diagnostics"][diag_kind][sid]
          stats_b = manifest_b["diagnostics"][diag_kind][sid]
          self.assertEqual(stats_a, stats_b)
          self.assertGreaterEqual(stats_a["max_kl"], 0.0)
          self.assertGreaterEqual(stats_a["mean_kl"], 0.0)
          self.assertTrue(np.isfinite(stats_a["max_kl"]))
          self.assertTrue(np.isfinite(stats_a["mean_kl"]))

  def test_cli_subprocess_smoke_and_leakr_clean_manifest(self) -> None:
    """Runs the CLI entrypoint in a fresh child process and verifies flag reachability + clean manifest."""
    with tempfile.TemporaryDirectory(prefix="ds41_mm_cli_") as tmp:
      cli_out = pathlib.Path(tmp) / "cli_out"
      env = os.environ.copy()
      env["PYTHONPATH"] = str(_REPO_ROOT / "src") + os.pathsep + str(_REPO_ROOT)
      subprocess.run(
          [
              sys.executable,
              "-m",
              "tests.utils.deepseek_v41_mm_golden.generate_mm_golden_logits",
              "--mini",
              "--golden-version",
              "mm_v1",
              "--checkpoint-dir",
              str(self._mini_ckpt_dir),
              "--output-dir",
              str(cli_out),
              "--samples",
              "L1",
              "--no-diagnostics",
              "--num-threads",
              "2",
          ],
          cwd=str(_REPO_ROOT),
          env=env,
          check=True,
          capture_output=True,
          text=True,
      )
      self.assertTrue((cli_out / "L1.npz").exists())
      self.assertTrue((cli_out / "mm_logit.pkl").exists())
      manifest_text = (cli_out / "manifest.json").read_text(encoding="utf-8")
      manifest = json.loads(manifest_text)
      self.assertEqual(manifest["cli_args"]["samples"], ["L1"])
      self.assertFalse(manifest["cli_args"]["diagnostics"])
      self.assertEqual(manifest["cli_args"]["num_threads"], 2)
      self.assertNotIn(str(cli_out), manifest_text)
      self.assertNotIn(str(self._mini_ckpt_dir), manifest_text)

  def test_full_config_39x39_grid_planner(self) -> None:
    """Verifies `prepare_letterboxed_image` plans exact `(39, 39)` grids (`1521` patches, `184` tokens)."""
    hf_cfg = runner.load_hf_config(str(self._mini_ckpt_dir))
    hf_cfg["vision_config"] = {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None,
    }
    full_args = reference_loader.multimodal_model_args_from_hf_config(hf_cfg)

    asset_bytes = _TEST_IMAGE_PATH.read_bytes()
    with Image.open(io.BytesIO(asset_bytes)) as im:
      asset_rgb = im.convert("RGB")
    prep_asset = generate_mm_golden_logits.prepare_letterboxed_image(
        "asset:test_image.jpg",
        asset_rgb,
        asset_bytes,
        full_args,
        target_grid_hw=(39, 39),
    )
    self.assertEqual((prep_asset.n_vit_h, prep_asset.n_vit_w), (39, 39))
    self.assertEqual(tuple(prep_asset.patches_bf16.shape), (1521, 3, 14, 14))
    self.assertEqual(tuple(prep_asset.patches_flat_fp32.shape), (1521, 588))
    self.assertEqual(int(prep_asset.token_types.numel()), 184)
    self.assertEqual(int(prep_asset.token_types[0].item()), image_processor.IMAGE_START)
    self.assertEqual(int(prep_asset.token_types[-1].item()), image_processor.IMAGE_END)
    self.assertEqual(int((prep_asset.token_types == image_processor.IMAGE).sum().item()), 169)

    rng = np.random.default_rng(314)
    synth_rgb = Image.fromarray(rng.integers(0, 256, size=(384, 512, 3), dtype=np.uint8), mode="RGB")
    synth_buf = io.BytesIO()
    synth_rgb.save(synth_buf, format="PNG")
    prep_synth = generate_mm_golden_logits.prepare_letterboxed_image(
        "synth:314",
        synth_rgb,
        synth_buf.getvalue(),
        full_args,
        target_grid_hw=(39, 39),
    )
    self.assertEqual((prep_synth.n_vit_h, prep_synth.n_vit_w), (39, 39))
    self.assertEqual(tuple(prep_synth.patches_flat_fp32.shape), (1521, 588))
    self.assertEqual(int(prep_synth.token_types.numel()), 184)

  def test_full_mode_missing_release_assets_and_tokenizer_fail_loudly(self) -> None:
    """Verifies `mini=False` succeeds with valid release assets and raises loudly when any is missing."""
    hf_cfg = runner.load_hf_config(str(self._mini_ckpt_dir))
    hf_cfg["vision_config"] = {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None,
    }
    full_args = reference_loader.multimodal_model_args_from_hf_config(hf_cfg, max_seq_len=2048)
    suite_full = generate_mm_golden_logits.build_sample_suite(mini=False)

    assistant_end_ids = [int(t) for t in self._tokenizer.encode("<｜Assistant｜></think>", add_special_tokens=False)]
    think_open_ids = [int(t) for t in self._tokenizer.encode("<think>", add_special_tokens=False)]
    nl_ids = [int(t) for t in self._tokenizer.encode("\n\n", add_special_tokens=False)]

    with tempfile.TemporaryDirectory(prefix="ds41_mm_full_neg_") as tmp:
      tmp_path = pathlib.Path(tmp)
      staged_dir = tmp_path / "staged_release"
      images_dir = staged_dir / "inference" / "examples" / "images"
      images_dir.mkdir(parents=True, exist_ok=True)
      carrots_path = images_dir / "carrots.jpeg"
      corn_path = images_dir / "corn.jpeg"
      shutil.copy2(_TEST_IMAGE_PATH, carrots_path)
      rng = np.random.default_rng(202)
      corn_img = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), mode="RGB")
      corn_img.save(corn_path, format="JPEG")

      example_path = staged_dir / "inference" / "examples" / "example.txt"
      valid_two_img_prompt = (
          "请按“第一张、第二张”的顺序回答：第一张图<image>examples/images/carrots.jpeg</image>"
          "和第二张图<image>examples/images/corn.jpeg</image>中分别是什么食材？它们通常食用的部位分别是什么？\n"
      )
      example_path.write_text(valid_two_img_prompt, encoding="utf-8")

      # Positive liveness check outside assertRaisesRegex: complete release L2 builds and matches chat encoding.
      built_l2 = generate_mm_golden_logits.build_sample_inputs(
          suite_full["L2"],
          staged_dir,
          self._tokenizer,
          full_args,
          (),
          {},
          mini=False,
      )
      self.assertEqual(built_l2.image_source_ids, ("release:carrots.jpeg", "release:corn.jpeg"))
      self.assertEqual(built_l2.grid_hw, (39, 39))
      self.assertEqual(len(built_l2.span_starts), 2)
      l2_tokens = built_l2.tokens.tolist()
      self.assertEqual(l2_tokens[-len(assistant_end_ids) :], assistant_end_ids)
      self.assertNotIn(think_open_ids[0], l2_tokens)
      for span_start in built_l2.span_starts:
        self.assertEqual(l2_tokens[span_start - len(nl_ids) : span_start], nl_ids)
        self.assertEqual(l2_tokens[span_start + 184 : span_start + 184 + len(nl_ids)], nl_ids)

      # 1. Missing carrots.jpeg in mini=False -> FileNotFoundError (no synthetic fallback).
      carrots_backup = carrots_path.read_bytes()
      carrots_path.unlink()
      with self.assertRaisesRegex(FileNotFoundError, r"Missing required release example image:.*carrots\.jpeg"):
        generate_mm_golden_logits.build_sample_inputs(
            suite_full["L2"],
            staged_dir,
            self._tokenizer,
            full_args,
            (),
            {},
            mini=False,
        )
      carrots_path.write_bytes(carrots_backup)

      # 2. Missing corn.jpeg in mini=False -> FileNotFoundError (no synthetic fallback).
      corn_backup = corn_path.read_bytes()
      corn_path.unlink()
      with self.assertRaisesRegex(FileNotFoundError, r"Missing required release example image:.*corn\.jpeg"):
        generate_mm_golden_logits.build_sample_inputs(
            suite_full["L2"],
            staged_dir,
            self._tokenizer,
            full_args,
            (),
            {},
            mini=False,
        )
      corn_path.write_bytes(corn_backup)

      # 3. example.txt with no two-image prompt in mini=False -> ValueError.
      example_path.write_text("Only one image<image>examples/images/carrots.jpeg</image>here.\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, r"No two-image <image>\.\.\.</image> prompt found"):
        generate_mm_golden_logits.build_sample_inputs(
            suite_full["L2"],
            staged_dir,
            self._tokenizer,
            full_args,
            (),
            {},
            mini=False,
        )

      # 4. Missing example.txt in mini=False -> FileNotFoundError.
      example_path.unlink()
      with self.assertRaisesRegex(FileNotFoundError, r"Missing required release prompt file:.*example\.txt"):
        generate_mm_golden_logits.build_sample_inputs(
            suite_full["L2"],
            staged_dir,
            self._tokenizer,
            full_args,
            (),
            {},
            mini=False,
        )

      # 5. Missing tokenizer.json in checkpoint_dir when mini=False -> FileNotFoundError even if env var is set.
      no_tok_ckpt = tmp_path / "no_tok_ckpt"
      shutil.copytree(self._mini_ckpt_dir, no_tok_ckpt)
      (no_tok_ckpt / "tokenizer.json").unlink()
      with mock.patch.dict(os.environ, {"DEEPSEEK_V41_TOKENIZER": str(self._mini_ckpt_dir)}):
        with self.assertRaisesRegex(FileNotFoundError, r"Missing required tokenizer\.json in checkpoint directory"):
          generate_mm_golden_logits.run_generation(
              output_dir=tmp_path / "no_tok_out",
              checkpoint_dir=str(no_tok_ckpt),
              mini=False,
              samples="L1",
              diagnostics=False,
          )

  def test_m1_to_m8_unique_16gram_text_windows(self) -> None:
    """Verifies zero duplicate 16-token non-image text windows across `M1..M8` (`mini=True` and `mini=False`)."""
    hf_cfg = runner.load_hf_config(str(self._mini_ckpt_dir))
    mini_args = reference_loader.multimodal_model_args_from_hf_config(hf_cfg, max_seq_len=128)
    hf_cfg_full = {**hf_cfg}
    hf_cfg_full["vision_config"] = {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None,
    }
    full_args = reference_loader.multimodal_model_args_from_hf_config(hf_cfg_full, max_seq_len=2048)
    c4_pool = generate_mm_golden_logits.load_c4_token_pool(self._tokenizer, full_args.image_token_id)

    for mini_flag, margs in ((True, mini_args), (False, full_args)):
      suite = generate_mm_golden_logits.build_sample_suite(mini=mini_flag)
      image_cache: dict[str, generate_mm_golden_logits.PreparedImage] = {}
      for sid in generate_mm_golden_logits.PREFILL_SAMPLE_IDS:
        built = generate_mm_golden_logits.build_sample_inputs(
            suite[sid],
            self._mini_ckpt_dir,
            self._tokenizer,
            margs,
            c4_pool,
            image_cache,
            mini=mini_flag,
        )
        text_tokens = [
            int(tok)
            for tok, tok_type in zip(built.tokens.tolist(), built.token_types.tolist(), strict=True)
            if int(tok_type) == image_processor.TEXT
        ]
        self.assertGreaterEqual(len(text_tokens), 16, msg=f"Too few text tokens on {sid} (mini={mini_flag})")
        windows = [tuple(text_tokens[idx : idx + 16]) for idx in range(len(text_tokens) - 15)]
        self.assertEqual(
            len(windows),
            len(set(windows)),
            msg=f"Duplicate 16-token non-image window detected in {sid} (mini={mini_flag})",
        )

    # Negative control: repeated 16-token text block around an image span must fail loudly.
    repeated_block = list(range(500, 516))
    dup_tokens = np.asarray(repeated_block + [129264] * 8 + repeated_block, dtype=np.int32)
    dup_types = np.asarray(
        [image_processor.TEXT] * 16
        + [image_processor.IMAGE_START]
        + [image_processor.IMAGE] * 6
        + [image_processor.IMAGE_END]
        + [image_processor.TEXT] * 16,
        dtype=np.int32,
    )
    with self.assertRaisesRegex(ValueError, r"Duplicate 16-token text window in sample 'M8'"):
      generate_mm_golden_logits.validate_no_duplicate_text_ngrams(dup_tokens, dup_types, sample_id="M8")


if __name__ == "__main__":
  unittest.main()
