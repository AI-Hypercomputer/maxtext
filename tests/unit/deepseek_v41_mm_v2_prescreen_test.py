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

"""Unit tests for Option B-1 `mm_v2` blind pre-screen, `ref_topk_experts`, and fast dequant."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import pathlib
import pickle
import shutil
import tempfile
from typing import Any
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from tests.utils.parity import negctl

try:
  import torch
  from tests.utils.deepseek_v41_mm_golden import generate_mm_golden_logits
  from tests.utils.deepseek_v41_mm_golden import reference_loader
  from tests.utils.deepseek_v41_reference import image_processor
  from tests.utils.deepseek_v41_reference import model as ref_model
  from tests.utils.deepseek_v41_reference import multimodal
  from tests.utils.deepseek_v41_reference import quantize_checkpoint
  from tests.utils.deepseek_v41_reference import runner

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  generate_mm_golden_logits = None  # type: ignore[assignment]
  reference_loader = None  # type: ignore[assignment]
  image_processor = None  # type: ignore[assignment]
  ref_model = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  quantize_checkpoint = None  # type: ignore[assignment]
  runner = None  # type: ignore[assignment]
  HAS_TORCH = False

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TEST_IMAGE_PATH = _REPO_ROOT / "tests" / "assets" / "test_image.jpg"
_TEST_IMAGE_REASONING_PATH = _REPO_ROOT / "tests" / "assets" / "test_image_reasoning.jpg"
_NEGCTL_REL_SCALE = 1e-2


def _make_fixture_release_images(dst_dir: pathlib.Path) -> pathlib.Path:
  """Writes deterministic photographic-like RGB JPEG fixtures (`carrots.jpeg`, `corn.jpeg`)."""
  dst_dir.mkdir(parents=True, exist_ok=True)
  for idx, filename in enumerate(("carrots.jpeg", "corn.jpeg")):
    rng = np.random.default_rng(900 + idx)
    h, w = (360, 480) if idx == 0 else (308, 450)
    ys = np.linspace(15.0, 240.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(20.0, 235.0, w, dtype=np.float32)[None, :]
    r = np.clip(0.55 * ys + 0.35 * xs + 25.0 * np.sin(ys / 13.0) + rng.normal(0.0, 9.0, (h, w)), 0, 255)
    g = np.clip(0.45 * (255.0 - ys) + 0.45 * xs + 25.0 * np.cos(xs / 17.0) + rng.normal(0.0, 9.0, (h, w)), 0, 255)
    b = np.clip(110.0 + 55.0 * np.sin((xs + ys) / 21.0) + rng.normal(0.0, 9.0, (h, w)), 0, 255)
    rgb = Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8), mode="RGB")
    rgb.save(dst_dir / filename, format="JPEG", quality=92)
  return dst_dir


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class DeepSeekV41MmV2PrescreenTest(unittest.TestCase):
  """Tests vectorized `dequantize_into_fast`, `RoutingTrace`, and Option B-1 `mm_v2` generation."""

  _tmp_root: str
  _mini_ckpt_dir: pathlib.Path
  _release_images_dir: pathlib.Path
  _tokenizer: Any
  _model: multimodal.MultimodalTransformer
  _margs: ref_model.ModelArgs
  _c4_pool: list[int]

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls._tmp_root = tempfile.mkdtemp(prefix="ds41_mm_v2_prescreen_test_")
    cls._mini_ckpt_dir = generate_mm_golden_logits.build_quantized_mini_checkpoint(
        pathlib.Path(cls._tmp_root) / "mini_ckpt",
        seed=1,
    )
    cls._release_images_dir = _make_fixture_release_images(pathlib.Path(cls._tmp_root) / "fixture_release_images")
    cls._tokenizer = AutoTokenizer.from_pretrained(str(cls._mini_ckpt_dir))
    cls._model, cls._margs = reference_loader.load_multimodal_reference_model(
        str(cls._mini_ckpt_dir),
        dtype="bf16",
        max_seq_len=128,
        tokenizer=cls._tokenizer,
        fp32_compute=True,
        lazy_dequant=True,
    )
    cls._c4_pool = generate_mm_golden_logits.load_c4_token_pool(
        cls._tokenizer,
        cls._margs.image_token_id,
    )

  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(cls._tmp_root, ignore_errors=True)
    super().tearDownClass()

  def test_vectorized_dequantize_into_fast_bit_identical_and_negative_control(self) -> None:
    """Verifies `dequantize_into_fast` is IEEE-754 bit-identical to `runner.dequantize_into`."""
    reference_loader.verify_dequantize_into_fast()
    gen = torch.Generator().manual_seed(20261026)

    # 1. FP4 E2M1 expert tensor (768x96) + explicit 0x00..0xFF byte sweep (including -0.0 nibble 0x8).
    src_fp4 = torch.randn((12 * 64, 96), generator=gen, dtype=torch.float32) * 1.75
    qw_fp4, qs_fp4 = quantize_checkpoint.quantize_fp4(src_fp4)
    qw_fp4 = qw_fp4.contiguous()
    qs_fp4 = qs_fp4.contiguous()
    qw_fp4.view(-1)[:256] = torch.arange(256, dtype=torch.uint8)

    for out_dtype, view_dtype in ((torch.float32, torch.uint32), (torch.bfloat16, torch.uint16)):
      out_ref = torch.empty((12 * 64, 96), dtype=out_dtype)
      out_fast = torch.empty((12 * 64, 96), dtype=out_dtype)
      runner.dequantize_into(out_ref, qw_fp4, qs_fp4)
      reference_loader.dequantize_into_fast(out_fast, qw_fp4, qs_fp4)
      self.assertTrue(torch.equal(out_fast, out_ref))
      self.assertTrue(torch.equal(out_fast.view(view_dtype), out_ref.view(view_dtype)))

    # 2. FP8 E4M3 32x32 linear and 1x32 engram tables across float32 and bfloat16.
    for shape, bh, bw in (((96, 128), 32, 32), ((65, 96), 1, 32)):
      src_fp8 = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.9
      qw_fp8, qs_fp8 = quantize_checkpoint.quantize_fp8(src_fp8, bh, bw)
      for out_dtype, view_dtype in ((torch.float32, torch.uint32), (torch.bfloat16, torch.uint16)):
        out_ref = torch.empty(shape, dtype=out_dtype)
        out_fast = torch.empty(shape, dtype=out_dtype)
        runner.dequantize_into(out_ref, qw_fp8, qs_fp8)
        reference_loader.dequantize_into_fast(out_fast, qw_fp8, qs_fp8)
        self.assertTrue(torch.equal(out_fast, out_ref))
        self.assertTrue(torch.equal(out_fast.view(view_dtype), out_ref.view(view_dtype)))

    # Negative control via negctl.perturb_one_leaf: perturbing dequantized reference breaks equality.
    base_ref = torch.empty((12 * 64, 96), dtype=torch.float32)
    base_fast = torch.empty((12 * 64, 96), dtype=torch.float32)
    runner.dequantize_into(base_ref, qw_fp4, qs_fp4)
    reference_loader.dequantize_into_fast(base_fast, qw_fp4, qs_fp4)
    perturbed = negctl.perturb_one_leaf(
        {"w": jnp.asarray(base_ref.numpy())},
        target_path="w",
        rel_scale=_NEGCTL_REL_SCALE,
        seed=7,
    )
    perturbed_pt = torch.from_numpy(np.array(perturbed["w"], copy=True))
    self.assertFalse(torch.equal(base_fast, perturbed_pt))

  def test_routing_trace_zero_overhead_capture_and_indexer_nan_safety(self) -> None:
    """Verifies `forward_multimodal_logits_with_routing_trace` matches plain forward and handles Indexer active masks."""
    suite = generate_mm_golden_logits.build_sample_suite(mini=True, golden_version="mm_v1")
    built = generate_mm_golden_logits.build_sample_inputs(
        suite["L4"],
        self._mini_ckpt_dir,
        self._tokenizer,
        self._margs,
        self._c4_pool,
        {},
        mini=True,
    )
    input_ids_pt = torch.from_numpy(built.tokens).to(torch.int64).unsqueeze(0)
    token_types_pt = torch.from_numpy(built.token_types).to(torch.int64).unsqueeze(0)

    logits_plain = reference_loader.forward_multimodal_logits(
        self._model,
        input_ids_pt,
        images=[built.image_inputs],
        token_types=token_types_pt,
    )
    logits_trace, trace = reference_loader.forward_multimodal_logits_with_routing_trace(
        self._model,
        input_ids_pt,
        images=[built.image_inputs],
        token_types=token_types_pt,
    )
    self.assertTrue(torch.equal(logits_trace, logits_plain))
    self.assertEqual(trace.ref_topk_experts.dtype, np.uint16)
    self.assertEqual(
        trace.ref_topk_experts.shape,
        (self._margs.n_layers, len(built.tokens), self._margs.n_activated_experts),
    )
    self.assertTrue(np.all(trace.ref_topk_experts < self._margs.n_routed_experts))
    # Every (layer, pos) top-k row has distinct expert indices.
    self.assertTrue(np.all(trace.ref_topk_experts[:, :, 0] != trace.ref_topk_experts[:, :, 1]))
    self.assertTrue(np.all(np.isfinite(trace.gate_margins)))
    self.assertTrue(np.all(trace.gate_margins >= 0.0))

    # On mini (index_topk=4, seq_len=64 > 4), global sparse layers (2,4,6,8,10,12) have finite
    # positive/non-negative margins on later IMAGE slots, while sliding-window layers are None.
    c1_eval = generate_mm_golden_logits.evaluate_c1_routing_margins(
        trace,
        built,
        (1, 2, 3),
        generate_mm_golden_logits.PrescreenCriteria(),
    )
    idx_by_layer = c1_eval["min_indexer_margin_by_layer_image_tokens"]
    self.assertEqual(len(idx_by_layer), self._margs.n_layers)
    for layer_idx, val in enumerate(idx_by_layer):
      if self._model.layers[layer_idx].attn.indexer is None:
        self.assertIsNone(val)
      else:
        self.assertIsNotNone(val)
        self.assertGreaterEqual(float(val), 0.0)

    # Also verify seq_len <= index_topk (e.g. L1 seq_len=32 <= 512) yields [None] * n_layers without NaN crash.
    built_l1 = generate_mm_golden_logits.build_sample_inputs(
        suite["L1"],
        self._mini_ckpt_dir,
        self._tokenizer,
        self._margs,
        self._c4_pool,
        {},
        mini=True,
    )
    with contextlib.ExitStack() as stack:
      for layer in self._model.layers:
        if layer.attn.indexer is not None:
          stack.enter_context(mock.patch.object(layer.attn.indexer, "index_topk", 512))
      _, trace_short = reference_loader.forward_multimodal_logits_with_routing_trace(
          self._model,
          torch.from_numpy(built_l1.tokens).to(torch.int64).unsqueeze(0),
          images=[built_l1.image_inputs],
          token_types=torch.from_numpy(built_l1.token_types).to(torch.int64).unsqueeze(0),
      )
    c1_short = generate_mm_golden_logits.evaluate_c1_routing_margins(
        trace_short,
        built_l1,
        (0,),
        generate_mm_golden_logits.PrescreenCriteria(),
    )
    self.assertEqual(c1_short["min_indexer_margin_by_layer_image_tokens"], [None] * self._margs.n_layers)

    # Force an exact 0.0 Indexer tie on an active IMAGE position of L4 layer 2 and verify C1 rejects it.
    img_pos_l4 = np.where(built.token_types == image_processor.IMAGE)[0]
    tied_idx_margins = list(trace.indexer_margins)
    assert tied_idx_margins[2] is not None
    tied_layer2 = np.array(tied_idx_margins[2], copy=True)
    tied_layer2[int(img_pos_l4[-1])] = 0.0
    tied_idx_margins[2] = tied_layer2
    tied_trace = dataclasses.replace(trace, indexer_margins=tuple(tied_idx_margins))
    c1_tied = generate_mm_golden_logits.evaluate_c1_routing_margins(
        tied_trace,
        built,
        (1, 2, 3),
        generate_mm_golden_logits.PrescreenCriteria(tau_gate_l0_19_image_tokens=0.0),
    )
    self.assertFalse(c1_tied["passed"])
    self.assertIn("C1_indexer_margin_l0_19_gt0", c1_tied["failed_criteria"])

  def test_candidate_pool_72_crops_interleaving_sha256_and_rank3_deferral(self) -> None:
    """Verifies the 72-crop queue order, 544x544 geometry, SHA-256 guards, and `rank 3` deferral."""
    self.assertEqual(
        hashlib.sha256(_TEST_IMAGE_REASONING_PATH.read_bytes()).hexdigest(),
        generate_mm_golden_logits.B0_SHA256,
    )
    self.assertEqual(
        hashlib.sha256(_TEST_IMAGE_PATH.read_bytes()).hexdigest(),
        generate_mm_golden_logits.B3_SHA256,
    )

    durable_ckpt = pathlib.Path.home() / "ds41_hf_durable"
    if (durable_ckpt / "inference" / "examples" / "images" / "carrots.jpeg").exists():
      real_assets = generate_mm_golden_logits.load_mm_v2_base_assets(
          durable_ckpt,
          verify_release_sha256=True,
      )
      for bid, expected_sha in generate_mm_golden_logits.EXPECTED_BASE_ASSET_SHA256.items():
        self.assertEqual(real_assets[bid].sha256, expected_sha)

    base_assets = generate_mm_golden_logits.load_mm_v2_base_assets(
        self._mini_ckpt_dir,
        release_images_dir=str(self._release_images_dir),
        verify_release_sha256=False,
    )
    queue = generate_mm_golden_logits.build_mm_v2_candidate_queue(base_assets)
    self.assertEqual(len(queue), 72)
    self.assertEqual([c.rank for c in queue], list(range(72)))
    self.assertEqual(len({c.source_id for c in queue}), 72)
    self.assertEqual(
        [c.source_id for c in queue[:4]],
        [
            "crop_544:B0:anchor_0:mirror_0",
            "crop_544:B1:anchor_0:mirror_0",
            "crop_544:B2:anchor_0:mirror_0",
            "crop_544:B3:anchor_0:mirror_0",
        ],
    )
    self.assertEqual(queue[36].source_id, "crop_544:B0:anchor_0:mirror_1")

    crop0_rgb, crop0_png = generate_mm_golden_logits.materialize_candidate_crop_rgb(queue[0], base_assets)
    self.assertEqual(crop0_rgb.size, (544, 544))
    self.assertGreater(len(crop0_png), 0)

    # Verify rank 3 (B3:anchor_0:mirror_0) deferral when asset:test_image.jpg is retained in slot 0,
    # followed by immediate consumption on the next sample without asset:test_image.jpg.
    rem = list(range(72))
    l1_crop = generate_mm_golden_logits._draw_next_candidate_rank(  # pylint: disable=protected-access
        rem, queue, forbid_duplicate_test_image=False, sample_id="L1"
    )
    l4_crops = [
        generate_mm_golden_logits._draw_next_candidate_rank(  # pylint: disable=protected-access
            rem, queue, forbid_duplicate_test_image=True, sample_id="L4"
        )
        for _ in range(3)
    ]
    m1_crop = generate_mm_golden_logits._draw_next_candidate_rank(  # pylint: disable=protected-access
        rem, queue, forbid_duplicate_test_image=False, sample_id="M1"
    )
    self.assertEqual(l1_crop.rank, 0)
    self.assertEqual([c.rank for c in l4_crops], [1, 2, 4])
    self.assertEqual(m1_crop.rank, 3)

    # Negative controls: mismatched release SHA-256 raises ValueError; missing file raises FileNotFoundError.
    with self.assertRaisesRegex(ValueError, r"SHA-256 mismatch for mm_v2 base asset B1"):
      generate_mm_golden_logits.load_mm_v2_base_assets(
          self._mini_ckpt_dir,
          release_images_dir=str(self._release_images_dir),
          verify_release_sha256=True,
      )
    with tempfile.TemporaryDirectory(prefix="ds41_mm_v2_missing_b2_") as tmp_missing:
      shutil.copy2(self._release_images_dir / "carrots.jpeg", pathlib.Path(tmp_missing) / "carrots.jpeg")
      with self.assertRaisesRegex(FileNotFoundError, r"Missing required mm_v2 base asset B2"):
        generate_mm_golden_logits.load_mm_v2_base_assets(
            self._mini_ckpt_dir,
            release_images_dir=tmp_missing,
            verify_release_sha256=False,
        )

  def test_c1_short_circuit_and_c2_perturbation_rejection_criteria(self) -> None:
    """Verifies `C1` short-circuits `C2a/C2b` (`None` metrics, 0 calls) and `C2` enforces all 4 KL/top1 bounds."""
    suite = generate_mm_golden_logits.build_sample_suite(mini=True, golden_version="mm_v1")
    built = generate_mm_golden_logits.build_sample_inputs(
        suite["L3"],
        self._mini_ckpt_dir,
        self._tokenizer,
        self._margs,
        self._c4_pool,
        {},
        mini=True,
    )

    # Positive baseline: passes C1 + C2 cleanly.
    ok_criteria = generate_mm_golden_logits.PrescreenCriteria(tau_gate_l0_19_image_tokens=1e-9)
    accepted_ok, primary_logits, _, ulp_logits, rec_ok = generate_mm_golden_logits.evaluate_sample_prescreen(
        self._model,
        built,
        screened_slots=(0,),
        attempt_index=0,
        queue_ranks=[],
        primary_threads=2,
        perturbation_threads=1,
        criteria=ok_criteria,
    )
    self.assertTrue(accepted_ok)
    self.assertEqual(rec_ok["failed_criteria"], [])
    self.assertIsNotNone(ulp_logits)
    self.assertIsNotNone(rec_ok["perturbation_64t_vs_96t"])
    self.assertIsNotNone(rec_ok["perturbation_one_ulp_vs_96t"])

    # 1. C1 rejection must short-circuit C2a and C2b (only 1 primary forward, 0 one_ulp passes).
    impossible_c1 = generate_mm_golden_logits.PrescreenCriteria(tau_gate_l0_19_image_tokens=10.0)
    with (
        mock.patch.object(
            reference_loader,
            "forward_multimodal_logits",
            wraps=reference_loader.forward_multimodal_logits,
        ) as spy_fwd,
        mock.patch.object(
            reference_loader,
            "one_ulp_routing_context",
            side_effect=AssertionError("C2b one_ulp_routing_context must not run when C1 fails"),
        ) as mock_ulp_ctx,
    ):
      accepted_c1_fail, _, _, ulp_none, rec_c1_fail = generate_mm_golden_logits.evaluate_sample_prescreen(
          self._model,
          built,
          screened_slots=(0,),
          attempt_index=0,
          queue_ranks=[0],
          primary_threads=2,
          perturbation_threads=1,
          criteria=impossible_c1,
      )
      self.assertEqual(spy_fwd.call_count, 1)
      self.assertEqual(mock_ulp_ctx.call_count, 0)
    self.assertFalse(accepted_c1_fail)
    self.assertIsNone(ulp_none)
    self.assertEqual(rec_c1_fail["failed_criteria"], ["C1_gate_margin_l0_19"])
    self.assertIsNone(rec_c1_fail["perturbation_64t_vs_96t"])
    self.assertIsNone(rec_c1_fail["perturbation_one_ulp_vs_96t"])

    # 2. C2 rejection via negctl-perturbed logits triggers max_kl, mean_kl, p95_kl, and top1_agreement failures.
    flipped_logits = -primary_logits.clone()
    kl_bad = generate_mm_golden_logits.compute_prescreen_kl_metrics(primary_logits, flipped_logits)
    c2_tags = generate_mm_golden_logits.evaluate_c2_perturbation_metrics(
        kl_bad,
        tag_prefix="C2a_64t",
        criteria=generate_mm_golden_logits.PrescreenCriteria(),
    )
    self.assertEqual(
        c2_tags,
        ["C2a_64t_max_kl", "C2a_64t_mean_kl", "C2a_64t_p95_kl", "C2a_64t_top1_agreement"],
    )

  def test_initial_real_samples_l2_l3_m2_retention_and_failing_replacement_geometry_lock(self) -> None:
    """Verifies failing initial `L3` (`asset:test_image.jpg`) replaces cleanly, cascades to `L4`, and locks geometry."""
    subset = "L2,L3,L4"
    relax_c1 = generate_mm_golden_logits.PrescreenCriteria(tau_gate_l0_19_image_tokens=1e-9)

    with (
        tempfile.TemporaryDirectory(prefix="ds41_mm_v2_l2l3l4_") as tmp,
        mock.patch.object(generate_mm_golden_logits, "load_c4_token_pool", return_value=self._c4_pool),
        mock.patch.object(AutoTokenizer, "from_pretrained", return_value=self._tokenizer),
        mock.patch.object(
            reference_loader,
            "load_multimodal_reference_model",
            return_value=(self._model, self._margs),
        ),
    ):
      tmp_path = pathlib.Path(tmp)
      v1_out = tmp_path / "v1_out"
      v2_fail_l3_out = tmp_path / "v2_fail_l3_out"

      generate_mm_golden_logits.run_generation(
          output_dir=v1_out,
          checkpoint_dir=str(self._mini_ckpt_dir),
          mini=True,
          samples=subset,
          num_threads=8,
          golden_version="mm_v1",
          diagnostics=False,
      )

      # Force L3's initial asset:test_image.jpg attempt (attempt_index=0) to fail C1 and verify:
      # (1) L2 retains ["release:carrots.jpeg", "release:corn.jpeg"] with queue_ranks=[];
      # (2) L3 replaces slot 0 from the queue at rank 0 while preserving exact mm_v1 geometry;
      # (3) test_image_failed_initial_c1 flips True so L4 replaces ALL 4 slots (0..3) too!
      orig_eval_c1 = generate_mm_golden_logits.evaluate_c1_routing_margins

      def _inject_l3_initial_c1_failure(
          trace: reference_loader.RoutingTrace,
          built: generate_mm_golden_logits.BuiltSampleInputs,
          screened_slots: tuple[int, ...],
          criteria: generate_mm_golden_logits.PrescreenCriteria,
      ) -> dict[str, Any]:
        res = orig_eval_c1(trace, built, screened_slots, criteria)
        if built.sample_id == "L3" and built.image_source_ids == ("asset:test_image.jpg",):
          res = dict(res)
          res["passed"] = False
          res["failed_criteria"] = ["C1_gate_margin_l0_19"]
        return res

      with mock.patch.object(
          generate_mm_golden_logits,
          "evaluate_c1_routing_margins",
          side_effect=_inject_l3_initial_c1_failure,
      ):
        manifest_fail = generate_mm_golden_logits.run_generation(
            output_dir=v2_fail_l3_out,
            checkpoint_dir=str(self._mini_ckpt_dir),
            mini=True,
            samples=subset,
            num_threads=8,
            perturbation_threads=4,
            golden_version="mm_v2",
            release_images_dir=str(self._release_images_dir),
            prescreen_criteria=relax_c1,
            diagnostics=False,
        )

      audit_fail = manifest_fail["prescreen_audit"]
      self.assertTrue(audit_fail["test_image_failed_initial_c1"])
      l2_attempts = [e for e in audit_fail["evaluated_candidates"] if e["sample_id"] == "L2"]
      self.assertEqual(len(l2_attempts), 1)
      self.assertTrue(l2_attempts[0]["accepted"])
      self.assertEqual(l2_attempts[0]["queue_ranks"], [])
      self.assertEqual(l2_attempts[0]["slot_source_ids"], ["release:carrots.jpeg", "release:corn.jpeg"])

      l3_attempts = [e for e in audit_fail["evaluated_candidates"] if e["sample_id"] == "L3"]
      self.assertEqual(len(l3_attempts), 2)
      self.assertFalse(l3_attempts[0]["accepted"])
      self.assertEqual(l3_attempts[0]["queue_ranks"], [])
      self.assertIsNone(l3_attempts[0]["perturbation_64t_vs_96t"])
      self.assertTrue(l3_attempts[1]["accepted"])
      self.assertEqual(l3_attempts[1]["queue_ranks"], [0])
      self.assertEqual(l3_attempts[1]["slot_source_ids"], ["crop_544:B0:anchor_0:mirror_0"])

      l4_accepted = [e for e in audit_fail["evaluated_candidates"] if e["sample_id"] == "L4" and e["accepted"]][0]
      self.assertEqual(l4_accepted["screened_slots"], [0, 1, 2, 3])
      self.assertNotIn("asset:test_image.jpg", l4_accepted["slot_source_ids"])

      # Verify 100% geometry lock (tokens, token_types, positions, span_starts, seq_len, grid_hw) vs mm_v1.
      for sid in ("L2", "L3", "L4"):
        with np.load(v1_out / f"{sid}.npz") as arr_v1, np.load(v2_fail_l3_out / f"{sid}.npz") as arr_v2:
          for key in ("tokens", "token_types", "positions", "span_starts", "seq_len", "num_images", "grid_hw"):
            self.assertTrue(np.array_equal(arr_v1[key], arr_v2[key]), msg=f"Geometry drift on {sid}.{key}")

  def test_full_12_sample_mm_v2_ref_topk_contiguous_ranks_and_cli_smoke(self) -> None:
    """Runs all 12 `mm_v2` `--mini` samples + CLI `main()` smoke check verifying `ref_topk_experts` & audit invariants."""
    relax_c1 = generate_mm_golden_logits.PrescreenCriteria(tau_gate_l0_19_image_tokens=1e-9)
    with tempfile.TemporaryDirectory(prefix="ds41_mm_v2_all12_") as tmp:
      tmp_path = pathlib.Path(tmp)
      all_out = tmp_path / "all_v2"
      cli_out = tmp_path / "cli_v2_out"
      with (
          mock.patch.object(generate_mm_golden_logits, "load_c4_token_pool", return_value=self._c4_pool),
          mock.patch.object(AutoTokenizer, "from_pretrained", return_value=self._tokenizer),
          mock.patch.object(
              reference_loader,
              "load_multimodal_reference_model",
              return_value=(self._model, self._margs),
          ),
      ):
        manifest = generate_mm_golden_logits.run_generation(
            output_dir=all_out,
            checkpoint_dir=str(self._mini_ckpt_dir),
            mini=True,
            samples="all",
            num_threads=8,
            perturbation_threads=4,
            golden_version="mm_v2",
            release_images_dir=str(self._release_images_dir),
            prescreen_criteria=relax_c1,
            diagnostics=False,
        )

      self.assertEqual(manifest["golden_version"], "mm_v2")
      self.assertEqual(
          manifest["prescreen_spec_sha256"],
          "6f70fbe7f5b9696f0160fb2258279647470f1aaa34da78d3ff4b296e12dfeff5",
      )
      self.assertEqual(manifest["ref_topk_experts_dtype"], "uint16")
      self.assertEqual([s["sample_id"] for s in manifest["samples"]], list(generate_mm_golden_logits.ALL_SAMPLE_IDS))

      # Verify L2, L3, M2 are screened FIRST on their original real assets with queue_ranks=[],
      # L4 retains slot 0 asset:test_image.jpg (deferring rank 3 -> [1, 2, 4]), and M1 consumes rank 3.
      audit = manifest["prescreen_audit"]
      evaluated = audit["evaluated_candidates"]
      self.assertEqual([e["sample_id"] for e in evaluated[:3]], ["L2", "L3", "M2"])
      for entry in evaluated[:3]:
        self.assertEqual(entry["queue_ranks"], [])
        self.assertTrue(entry["accepted"])
      self.assertFalse(audit["test_image_failed_initial_c1"])

      by_sid_accepted = {e["sample_id"]: e for e in evaluated if e["accepted"]}
      self.assertEqual(by_sid_accepted["L2"]["slot_source_ids"], ["release:carrots.jpeg", "release:corn.jpeg"])
      self.assertEqual(by_sid_accepted["L3"]["slot_source_ids"], ["asset:test_image.jpg"])
      self.assertEqual(by_sid_accepted["M2"]["slot_source_ids"], ["asset:test_image.jpg"])
      self.assertEqual(by_sid_accepted["L1"]["queue_ranks"], [0])
      self.assertEqual(by_sid_accepted["L4"]["queue_ranks"], [1, 2, 4])
      self.assertEqual(by_sid_accepted["L4"]["slot_source_ids"][0], "asset:test_image.jpg")
      self.assertEqual(by_sid_accepted["M1"]["queue_ranks"], [3])

      # Verify contiguous 0..K-1 queue_ranks with zero gaps and zero duplicates across all 12 samples.
      all_ranks = [r for entry in evaluated for r in entry["queue_ranks"]]
      self.assertGreaterEqual(len(all_ranks), 18)
      self.assertEqual(len(all_ranks), len(set(all_ranks)))
      self.assertEqual(sorted(all_ranks), list(range(len(all_ranks))))

      criteria_dict = audit["criteria"]
      for entry in evaluated:
        if entry["accepted"]:
          self.assertEqual(entry["failed_criteria"], [])
          self.assertGreaterEqual(
              entry["worst_gate_margin_l0_19"]["margin"],
              criteria_dict["tau_gate_l0_19_image_tokens"],
          )
          for pert_key in ("perturbation_64t_vs_96t", "perturbation_one_ulp_vs_96t"):
            pert = entry[pert_key]
            self.assertIsNotNone(pert)
            self.assertLessEqual(pert["max_kl"], criteria_dict["perturbation_max_kl_le"])
            self.assertLessEqual(pert["mean_kl"], criteria_dict["perturbation_mean_kl_le"])
            self.assertLessEqual(pert["p95_kl"], criteria_dict["perturbation_p95_kl_le"])
            self.assertGreaterEqual(pert["top1_agreement"], criteria_dict["perturbation_top1_agreement_ge"])
        else:
          self.assertTrue(len(entry["failed_criteria"]) >= 1)

      # Verify ref_topk_experts in all 12 .npz files and all 4 mm_logit.pkl records.
      for s_meta in manifest["samples"]:
        sid = s_meta["sample_id"]
        seq_len = int(s_meta["seq_len"])
        self.assertEqual(s_meta["ref_topk_experts_shape"], [self._margs.n_layers, seq_len, 2])
        self.assertEqual(s_meta["ref_topk_experts_dtype"], "uint16")
        with np.load(all_out / f"{sid}.npz") as npz_data:
          self.assertIn("ref_topk_experts", npz_data.files)
          self.assertEqual(npz_data["ref_topk_experts"].dtype, np.uint16)
          self.assertEqual(npz_data["ref_topk_experts"].shape, (self._margs.n_layers, seq_len, 2))
          for src_id in npz_data["image_source_ids"].tolist():
            self.assertFalse(str(src_id).startswith("synth:"), msg=f"Residual synth source in {sid}: {src_id}")

      with open(all_out / "mm_logit.pkl", "rb") as f:
        pkl_records = pickle.load(f)
      self.assertEqual([r["sample_id"] for r in pkl_records], ["L1", "L2", "L3", "L4"])
      for rec in pkl_records:
        self.assertIn("ref_topk_experts", rec)
        self.assertEqual(rec["ref_topk_experts"].dtype, np.uint16)
        self.assertEqual(
            rec["ref_topk_experts"].shape,
            (self._margs.n_layers, int(len(rec["tokens"])), 2),
        )

      # CLI main(argv) smoke check (--golden_version mm_v2 + --release-images-dir redaction).
      with (
          mock.patch.object(generate_mm_golden_logits, "load_c4_token_pool", return_value=self._c4_pool),
          mock.patch.object(AutoTokenizer, "from_pretrained", return_value=self._tokenizer),
          mock.patch.object(
              reference_loader,
              "load_multimodal_reference_model",
              return_value=(self._model, self._margs),
          ),
      ):
        generate_mm_golden_logits.main(
            [
                "--mini",
                "--checkpoint-dir",
                str(self._mini_ckpt_dir),
                "--output-dir",
                str(cli_out),
                "--golden_version",
                "mm_v2",
                "--release-images-dir",
                str(self._release_images_dir),
                "--samples",
                "L3",
                "--num-threads",
                "8",
                "--perturbation-threads",
                "4",
                "--no-diagnostics",
            ]
        )
      cli_manifest_text = (cli_out / "manifest.json").read_text(encoding="utf-8")
      cli_manifest = json.loads(cli_manifest_text)
      self.assertEqual(cli_manifest["golden_version"], "mm_v2")
      self.assertNotIn(str(cli_out), cli_manifest_text)
      self.assertNotIn(str(self._mini_ckpt_dir), cli_manifest_text)
      self.assertNotIn(str(self._release_images_dir), cli_manifest_text)

  def test_mm_v3_distinct_crop_assignment_and_routing_margin_ulp_export(self) -> None:
    """Verifies Option C `mm_v3` distinct crop assignment (`0` duplicate SHA-256s) and 7 routing/ULP arrays."""
    base_assets = generate_mm_golden_logits.load_mm_v2_base_assets(
        self._mini_ckpt_dir,
        release_images_dir=str(self._release_images_dir),
        verify_release_sha256=False,
    )
    distinct_queue = generate_mm_golden_logits.build_mm_v3_distinct_crop_queue(base_assets)
    self.assertEqual(len(distinct_queue), 24)
    self.assertEqual(
        tuple(spec.source_id for spec, _ in distinct_queue),
        generate_mm_golden_logits.MM_V3_DISTINCT_CROP_SOURCE_IDS,
    )
    self.assertEqual(
        [spec.rank for spec, _ in distinct_queue],
        list(range(12)) + list(range(36, 48)),
    )
    self.assertEqual(len({sha for _, sha in distinct_queue}), 24)

    durable_ckpt = pathlib.Path.home() / "ds41_hf_durable"
    if (durable_ckpt / "inference" / "examples" / "images" / "carrots.jpeg").exists():
      real_assets = generate_mm_golden_logits.load_mm_v2_base_assets(
          durable_ckpt,
          verify_release_sha256=True,
      )
      real_distinct = generate_mm_golden_logits.build_mm_v3_distinct_crop_queue(real_assets)
      self.assertEqual(len(real_distinct), 24)
      self.assertEqual(real_distinct[0][1], "9071feb68df89aeb94b70fd3cbb6fc8e62abcdce74e5e3821390776c4c30f58d")
      self.assertEqual(real_distinct[23][1], "e27a43c54cceba8c50ddfdd0467c65a46e28a825345806b8b8ac50776170bcab")

    with tempfile.TemporaryDirectory(prefix="ds41_mm_v3_test_") as tmp:
      out_v3 = pathlib.Path(tmp) / "mm_v3_out"
      with (
          mock.patch.object(generate_mm_golden_logits, "load_c4_token_pool", return_value=self._c4_pool),
          mock.patch.object(AutoTokenizer, "from_pretrained", return_value=self._tokenizer),
          mock.patch.object(
              reference_loader,
              "load_multimodal_reference_model",
              return_value=(self._model, self._margs),
          ),
      ):
        manifest = generate_mm_golden_logits.run_generation(
            output_dir=out_v3,
            checkpoint_dir=str(self._mini_ckpt_dir),
            mini=True,
            lazy_dequant=True,
            samples="all",
            num_threads=2,
            golden_version="mm_v3",
            release_images_dir=str(self._release_images_dir),
            diagnostics=True,
        )

      self.assertEqual(manifest["schema_version"], "ds41_mm_golden_v3")
      self.assertEqual(manifest["golden_version"], "mm_v3")
      self.assertEqual(manifest["option_c_spec_version"], generate_mm_golden_logits.OPTION_C_SPEC_VERSION)
      self.assertEqual(manifest["option_c_spec_sha256"], generate_mm_golden_logits.OPTION_C_SPEC_SHA256)
      self.assertEqual(manifest["ref_topk_experts_dtype"], "uint16")
      self.assertEqual(manifest["ref_gate_margins_dtype"], "float32")
      self.assertEqual(manifest["ref_gate_boundary_ulp_dtype"], "float32")
      self.assertEqual(manifest["ref_indexer_topk_dtype"], "int32")
      self.assertEqual(manifest["ref_indexer_margins_dtype"], "float32")
      self.assertEqual(manifest["ref_indexer_boundary_ulp_dtype"], "float32")
      self.assertEqual(manifest["ref_indexer_layer_ids"], [2, 4, 6, 8, 10, 12])
      self.assertEqual(manifest["ref_indexer_topk"], int(self._margs.index_topk))
      dsa = manifest["distinct_slot_image_assignment"]
      self.assertEqual(dsa["total_image_slots"], 26)
      self.assertEqual(dsa["unique_slot_sha256_count"], 26)
      self.assertTrue(dsa["all_slots_distinct"])
      self.assertEqual(len(dsa["slot_records"]), 26)
      self.assertEqual(
          [r["rank"] for r in dsa["slot_records"] if r["sample_id"] != "L2"],
          list(generate_mm_golden_logits.MM_V3_EXPECTED_DISTINCT_NON_L2_RANKS),
      )
      for r in dsa["slot_records"]:
        if r["sample_id"] == "L2":
          self.assertIsNone(r["rank"])
          self.assertIsNone(r["crop_box"])
        else:
          self.assertIsInstance(r["crop_box"], list)
          self.assertEqual(len(r["crop_box"]), 4)
      self.assertEqual(
          generate_mm_golden_logits.parse_args(["--output-dir", "/tmp/out"]).golden_version,
          "mm_v3",
      )
      self.assertEqual(set(manifest["diagnostics"]["bf16_compute"]), {"L4", "M8"})
      self.assertEqual(manifest["diagnostics"]["one_ulp_routing"], {})

      all_slot_shas: list[str] = []
      for s_meta in manifest["samples"]:
        sid = s_meta["sample_id"]
        seq_len = int(s_meta["seq_len"])
        all_slot_shas.extend(s_meta["image_source_sha256s"])
        for src_id in s_meta["image_source_ids"]:
          self.assertFalse(str(src_id).startswith("synth:"))
          self.assertNotEqual(str(src_id), "asset:test_image.jpg")
        with np.load(out_v3 / f"{sid}.npz") as npz_data:
          for req_key in (
              "ref_topk_experts",
              "ref_gate_margins",
              "ref_gate_boundary_ulp",
              "ref_indexer_layer_ids",
              "ref_indexer_topk",
              "ref_indexer_margins",
              "ref_indexer_boundary_ulp",
          ):
            self.assertIn(req_key, npz_data.files)
          self.assertEqual(npz_data["ref_topk_experts"].shape, (self._margs.n_layers, seq_len, 2))
          self.assertEqual(npz_data["ref_gate_margins"].shape, (self._margs.n_layers, seq_len))
          self.assertEqual(npz_data["ref_gate_boundary_ulp"].shape, (self._margs.n_layers, seq_len))
          gate_margins_arr = np.asarray(npz_data["ref_gate_margins"], dtype=np.float32)
          gate_ulp_arr = np.asarray(npz_data["ref_gate_boundary_ulp"], dtype=np.float32)
          idx_margins_arr = np.asarray(npz_data["ref_indexer_margins"], dtype=np.float32)
          idx_ulp_arr = np.asarray(npz_data["ref_indexer_boundary_ulp"], dtype=np.float32)
          self.assertTrue(np.all(np.isfinite(gate_margins_arr)))
          self.assertTrue(np.all(gate_margins_arr >= 0.0))
          self.assertTrue(np.all(np.isfinite(gate_ulp_arr)))
          self.assertTrue(np.all(gate_ulp_arr > 0.0))
          self.assertEqual(npz_data["ref_indexer_layer_ids"].tolist(), [2, 4, 6, 8, 10, 12])
          self.assertEqual(npz_data["ref_indexer_topk"].shape, (6, seq_len, int(self._margs.index_topk)))
          self.assertEqual(idx_margins_arr.shape, (6, seq_len))
          self.assertEqual(idx_ulp_arr.shape, (6, seq_len))
          finite_idx = np.isfinite(idx_margins_arr)
          self.assertTrue(np.array_equal(finite_idx, np.isfinite(idx_ulp_arr)))
          self.assertTrue(np.all(idx_margins_arr[finite_idx] >= 0.0))
          self.assertTrue(np.all(idx_ulp_arr[finite_idx] > 0.0))

      self.assertEqual(len(all_slot_shas), 26)
      self.assertEqual(len(set(all_slot_shas)), 26)

      with open(out_v3 / "mm_logit.pkl", "rb") as f:
        pkl_records = pickle.load(f)
      self.assertEqual([r["sample_id"] for r in pkl_records], ["L1", "L2", "L3", "L4"])
      for rec in pkl_records:
        for req_key in (
            "ref_topk_experts",
            "ref_gate_margins",
            "ref_gate_boundary_ulp",
            "ref_indexer_layer_ids",
            "ref_indexer_topk",
            "ref_indexer_margins",
            "ref_indexer_boundary_ulp",
        ):
          self.assertIn(req_key, rec)

      # Negative control: duplicate slot SHA-256 across samples is rejected fail-closed.
      dup_suite = generate_mm_golden_logits.build_sample_suite(mini=True, golden_version="mm_v3")
      dup_suite["L3"] = dataclasses.replace(dup_suite["L3"], image_sources=dup_suite["L1"].image_sources)
      with (
          mock.patch.object(generate_mm_golden_logits, "build_sample_suite", return_value=dup_suite),
          mock.patch.object(generate_mm_golden_logits, "load_c4_token_pool", return_value=self._c4_pool),
          mock.patch.object(AutoTokenizer, "from_pretrained", return_value=self._tokenizer),
          mock.patch.object(
              reference_loader,
              "load_multimodal_reference_model",
              return_value=(self._model, self._margs),
          ),
          self.assertRaisesRegex(ValueError, "Duplicate image_source_sha256"),
      ):
        generate_mm_golden_logits.run_generation(
            output_dir=pathlib.Path(tmp) / "mm_v3_dup",
            checkpoint_dir=str(self._mini_ckpt_dir),
            mini=True,
            lazy_dequant=True,
            samples="L1,L3",
            num_threads=2,
            golden_version="mm_v3",
            release_images_dir=str(self._release_images_dir),
            diagnostics=False,
        )


if __name__ == "__main__":
  unittest.main()
