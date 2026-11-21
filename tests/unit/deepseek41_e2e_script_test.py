# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for DeepSeek-V4.1 E2E script and forward logit runner contracts."""

import contextlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import jsonlines
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import moe
import numpy as np
import pytest

from tests.utils import deepseek_v41_frozen_topk
from tests.utils import forward_pass_logit_checker
from tests.utils.deepseek_v41 import logit_runner
from tests.utils.deepseek_v41 import masked_prefill_runner

pytestmark = pytest.mark.cpu_only

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/2_test_deepseek41_mm.sh"


class DeepSeek41E2EScriptTest(unittest.TestCase):
  """Unit tests for E2E script startup barrier and forward logit checker contracts."""

  def setUp(self):
    super().setUp()
    self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix="ds41_e2e_test_")  # pylint: disable=consider-using-with
    self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)

  def tearDown(self):
    self.tmp_dir_obj.cleanup()
    super().tearDown()

  def test_script_startup_barrier_and_any_worker_rank0_upload(self):
    """Verifies per-attempt startup barrier blocks stale markers and releases when all workers report."""
    ready_dir = self.tmp_dir / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    now_epoch = int(time.time())

    def run_barrier(  # pylint: disable=too-many-positional-arguments
        num_workers: int = 4,
        worker_id: int = 0,
        run_attempt_id: str = "",
        barrier_start_epoch: int | None = None,
        max_skew_sec: int = 180,
        timeout_sec: int = 2,
        poll_sec: int = 1,
    ) -> subprocess.CompletedProcess[str]:
      if barrier_start_epoch is None:
        barrier_start_epoch = int(time.time())
      env = os.environ.copy()
      env["PYTHON_BIN"] = sys.executable
      return subprocess.run(
          [
              "bash",
              "-c",
              f'source "{SCRIPT_PATH.as_posix()}" && wait_for_workers_ready "$@"',
              "--",
              ready_dir.as_posix(),
              str(num_workers),
              str(worker_id),
              run_attempt_id,
              str(barrier_start_epoch),
              str(max_skew_sec),
              str(timeout_sec),
              str(poll_sec),
          ],
          capture_output=True,
          text=True,
          env=env,
          check=False,
      )

    # 1. Stale attempt markers from attempt_old block attempt_new
    for w in range(4):
      (ready_dir / f"worker_{w}.ready").write_text(
          json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_old", "epoch": now_epoch}),
          encoding="utf-8",
      )
    for w in range(3):
      (ready_dir / f"worker_{w}.ready").write_text(
          json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_new", "epoch": now_epoch}),
          encoding="utf-8",
      )
    res1 = run_barrier(num_workers=4, worker_id=0, run_attempt_id="attempt_new", timeout_sec=2, poll_sec=1)
    self.assertNotEqual(res1.returncode, 0, f"Expected barrier failure with stale marker, got 0. stderr={res1.stderr}")

    # 2. When all workers update to attempt_new, barrier passes
    (ready_dir / "worker_3.ready").write_text(
        json.dumps({"worker_id": 3, "pod": "pod-3", "attempt_id": "attempt_new", "epoch": now_epoch}),
        encoding="utf-8",
    )
    res2 = run_barrier(num_workers=4, worker_id=0, run_attempt_id="attempt_new", timeout_sec=5, poll_sec=1)
    self.assertEqual(res2.returncode, 0, f"Expected barrier success, got {res2.returncode}. stderr={res2.stderr}")

  def test_compile_once_per_shape_and_bit_identical_logits(self):
    """Verifies single JIT compilation per static shape and bit-identical logits."""
    # pylint: disable=protected-access

    class DummyMLP(nnx.Module):
      """Dummy MLP module."""

      def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(d_in, d_out, rngs=rngs)
        self._frozen_topk_indices = nnx.Variable(jnp.array([0.0], dtype=jnp.float32))

      def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
        self.sow(nnx.Intermediate, "moe_lb_loss", jnp.sum(x))
        out = self.dense(x)
        if self._frozen_topk_indices is not None:
          out = out + jnp.sum(self._frozen_topk_indices[...])
        return out

    class DummyDecoderLayer(nnx.Module):
      """Dummy Decoder Layer module."""

      def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.mlp = DummyMLP(d_model, d_model, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
        return self.mlp(x)

    class DummyDecoder(nnx.Module):
      """Dummy Decoder module."""

      def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.layers_0 = DummyDecoderLayer(d_model, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
        return self.layers_0(x)

    class DummyE2EModel(nnx.Module):
      """Dummy E2E Model for unit testing eval_step_nnx."""

      def __init__(self, vocab_size: int, d_model: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.decoder = DummyDecoder(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

      def __call__(  # pylint: disable=arguments-differ,too-many-positional-arguments,too-many-arguments
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          encoder_audios: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, encoder_audios, enable_dropout
        x = self.embed(decoder_input_tokens)
        x = self.decoder(x)
        return self.head(x)

    rngs = nnx.Rngs(0)
    model = DummyE2EModel(vocab_size=32, d_model=16, rngs=rngs)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)), ("data",))

    # (a) make_eval_step_nnx test vs eager model execution
    eval_step, nnx_state = forward_pass_logit_checker.make_eval_step_nnx(model, mesh)

    ids_1 = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    pos_1 = jnp.arange(8, dtype=jnp.int32)[None, :]
    seg_1 = jnp.ones((1, 8), dtype=jnp.int32)

    eager_1 = model(decoder_input_tokens=ids_1, decoder_positions=pos_1, decoder_segment_ids=seg_1, enable_dropout=False)
    jitted_1 = eval_step(nnx_state, ids_1, pos_1, seg_1)
    self.assertTrue(np.array_equal(np.asarray(jitted_1), np.asarray(eager_1)))
    self.assertEqual(float(np.max(np.abs(jitted_1 - eager_1))), 0.0)
    self.assertEqual(eval_step._cache_size(), 1)

    # Distinct tokens, identical shape (1, 8) -> compile count stays 1
    ids_2 = jnp.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=jnp.int32)
    eager_2 = model(decoder_input_tokens=ids_2, decoder_positions=pos_1, decoder_segment_ids=seg_1, enable_dropout=False)
    jitted_2 = eval_step(nnx_state, ids_2, pos_1, seg_1)
    self.assertTrue(np.array_equal(np.asarray(jitted_2), np.asarray(eager_2)))
    self.assertEqual(float(np.max(np.abs(jitted_2 - eager_2))), 0.0)
    self.assertEqual(eval_step._cache_size(), 1)

    # Different shape (1, 12) -> triggers second compilation
    ids_3 = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=jnp.int32)
    pos_3 = jnp.arange(12, dtype=jnp.int32)[None, :]
    seg_3 = jnp.ones((1, 12), dtype=jnp.int32)
    jitted_3 = eval_step(nnx_state, ids_3, pos_3, seg_3)
    self.assertEqual(jitted_3.shape, (1, 12, 32))
    self.assertEqual(eval_step._cache_size(), 2)

    # (b) deepseek_v41_frozen_topk memoized step functions test with nnx.Intermediate sow & update
    deepseek_v41_frozen_topk.clear_frozen_topk_step_caches()
    init_leaf_count = len(jax.tree_util.tree_leaves(nnx.state(model)))

    # Cycle 1: Sample 1 with _frozen_topk_indices = 1.0
    model.decoder.layers_0.mlp._frozen_topk_indices[...] = jnp.array([1.0], dtype=jnp.float32)
    diag_gdef_1, diag_state_1 = nnx.split(model)
    diag_step_1 = deepseek_v41_frozen_topk.get_frozen_topk_step_fn(model, mesh, None, diag_gdef_1)
    out_diag_1, updated_non_params_1 = diag_step_1(diag_state_1, (ids_1, pos_1, seg_1, None, None))
    clean_non_params_1 = nnx.filter_state(updated_non_params_1, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params_1)
    self.assertEqual(len(jax.tree_util.tree_leaves(nnx.state(model))), init_leaf_count)
    self.assertEqual(diag_step_1._cache_size(), 1)

    # Cycle 2: Sample 2 (same shape (1, 8)) with distinct _frozen_topk_indices = 2.0
    model.decoder.layers_0.mlp._frozen_topk_indices[...] = jnp.array([2.0], dtype=jnp.float32)
    diag_gdef_2, diag_state_2 = nnx.split(model)
    diag_step_2 = deepseek_v41_frozen_topk.get_frozen_topk_step_fn(model, mesh, None, diag_gdef_2)
    self.assertIs(diag_step_1, diag_step_2)
    out_diag_2, updated_non_params_2 = diag_step_2(diag_state_2, (ids_2, pos_1, seg_1, None, None))
    clean_non_params_2 = nnx.filter_state(updated_non_params_2, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params_2)
    self.assertEqual(len(jax.tree_util.tree_leaves(nnx.state(model))), init_leaf_count)
    self.assertEqual(diag_step_2._cache_size(), 1)
    diff_val = float(np.max(np.abs(np.asarray(out_diag_2) - np.asarray(out_diag_1))))
    self.assertGreater(diff_val, 0.0)

    # Different shape (1, 12) -> triggers second compilation
    model.decoder.layers_0.mlp._frozen_topk_indices[...] = jnp.array([3.0], dtype=jnp.float32)
    diag_gdef_3, diag_state_3 = nnx.split(model)
    diag_step_3 = deepseek_v41_frozen_topk.get_frozen_topk_step_fn(model, mesh, None, diag_gdef_3)
    out_diag_3, updated_non_params_3 = diag_step_3(diag_state_3, (ids_3, pos_3, seg_3, None, None))
    clean_non_params_3 = nnx.filter_state(updated_non_params_3, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params_3)
    self.assertEqual(out_diag_3.shape, (1, 12, 32))
    self.assertEqual(diag_step_3._cache_size(), 2)

    # Natural routing step cache test
    trace_gdef_1, trace_state_1 = nnx.split(model)
    trace_step_1 = deepseek_v41_frozen_topk.get_natural_routing_step_fn(model, mesh, None, trace_gdef_1)
    out_trace_1, updated_trace_1 = trace_step_1(trace_state_1, (ids_1, pos_1, seg_1, None, None))
    clean_trace_1 = nnx.filter_state(updated_trace_1, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_trace_1)
    self.assertEqual(out_trace_1.shape, (1, 8, 32))
    self.assertEqual(trace_step_1._cache_size(), 1)

    trace_gdef_2, trace_state_2 = nnx.split(model)
    trace_step_2 = deepseek_v41_frozen_topk.get_natural_routing_step_fn(model, mesh, None, trace_gdef_2)
    self.assertIs(trace_step_1, trace_step_2)
    out_trace_2, updated_trace_2 = trace_step_2(trace_state_2, (ids_2, pos_1, seg_1, None, None))
    clean_trace_2 = nnx.filter_state(updated_trace_2, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_trace_2)
    self.assertEqual(out_trace_2.shape, (1, 8, 32))
    self.assertEqual(trace_step_2._cache_size(), 1)

    trace_gdef_3, trace_state_3 = nnx.split(model)
    trace_step_3 = deepseek_v41_frozen_topk.get_natural_routing_step_fn(model, mesh, None, trace_gdef_3)
    out_trace_3, updated_trace_3 = trace_step_3(trace_state_3, (ids_3, pos_3, seg_3, None, None))
    clean_trace_3 = nnx.filter_state(updated_trace_3, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_trace_3)
    self.assertEqual(out_trace_3.shape, (1, 12, 32))
    self.assertEqual(trace_step_3._cache_size(), 2)

  def test_get_data_with_masks_threads_audio_values_without_pixel_values(self):
    """Verifies audio-only multimodal goldens thread tiled audio_values without requiring pixel_values."""
    raw_audio = np.arange(1 * 16 * 80, dtype=np.float32).reshape((1, 16, 80)) / 100.0
    golden_audio_only = {
        "prompt": "Transcribe this clip",
        "tokens": [1, 2, 3, 4, 5],
        "logits": np.zeros((5, 32), dtype=np.float32),
        "audio_values": raw_audio,
    }
    for model_name in ("ming-3.0-audio-tiny", "qwen3-omni-30b-a3b"):
      cfg_audio = types.SimpleNamespace(
          model_name=model_name,
          global_batch_size_to_train_on=2,
          max_target_length=8,
          use_multimodal=True,
          use_audio=True,
          use_mrope=False,
      )
      unpacked = logit_runner.get_data_with_masks(golden_audio_only, cfg_audio)
      self.assertEqual(len(unpacked), 8)
      ids, seg_ids, pos, logits, seq_len, pixel_values, image_masks, audio_values = unpacked
      self.assertEqual(ids.shape, (2, 8))
      self.assertEqual(seg_ids.shape, (2, 8))
      self.assertEqual(pos.shape, (2, 8))
      self.assertEqual(logits.shape, (5, 32))
      self.assertEqual(seq_len, 5)
      self.assertIsNone(pixel_values)
      self.assertIsNone(image_masks)
      self.assertIsNotNone(audio_values)
      assert audio_values is not None
      self.assertEqual(audio_values.shape, (2, 16, 80))
      np.testing.assert_array_equal(audio_values, np.concatenate([raw_audio, raw_audio], axis=0))

      legacy_unpacked = forward_pass_logit_checker.get_data(golden_audio_only, cfg_audio)
      self.assertEqual(len(legacy_unpacked), 7)
      self.assertIsNone(legacy_unpacked[5])
      np.testing.assert_array_equal(legacy_unpacked[6], audio_values)

    cfg_vision_missing_img = types.SimpleNamespace(
        model_name="gemma3-4b",
        global_batch_size_to_train_on=2,
        max_target_length=8,
        use_multimodal=True,
        use_audio=False,
        use_mrope=False,
    )
    with self.assertRaisesRegex(AssertionError, "no image found in golden data while use_multimodal=True"):
      logit_runner.get_data_with_masks(golden_audio_only, cfg_vision_missing_img)

    cfg_text_only = types.SimpleNamespace(
        model_name="llama3.1-8b",
        global_batch_size_to_train_on=2,
        max_target_length=8,
        use_multimodal=False,
        use_audio=False,
        use_mrope=False,
    )
    *_, audio_disabled = logit_runner.get_data_with_masks(golden_audio_only, cfg_text_only)
    self.assertIsNone(audio_disabled)

  def test_make_eval_step_nnx_and_main_thread_encoder_audios(self):
    """Verifies make_eval_step_nnx and main thread encoder_audios end-to-end across pkl and npz goldens."""

    class DummyAudioModel(nnx.Module):
      """Tiny NNX model whose output logits depend strongly on encoder_audios."""

      def __init__(self, vocab_size: int, d_model: int, n_mels: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.audio_proj = nnx.Linear(n_mels, d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

      def __call__(  # pylint: disable=arguments-differ,too-many-positional-arguments,too-many-arguments
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          encoder_audios: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, enable_dropout
        x = self.embed(decoder_input_tokens)
        if encoder_audios is not None:
          pooled_audio = jnp.mean(encoder_audios, axis=1, keepdims=True)
          x = x + 8.0 * self.audio_proj(pooled_audio)
        return self.head(x)

    audio_model = DummyAudioModel(vocab_size=32, d_model=16, n_mels=8, rngs=nnx.Rngs(7))
    single_dev = np.array(jax.devices()[:1]).reshape((1,))
    mesh = jax.sharding.Mesh(single_dev, ("data",))

    ids = jnp.array([[2, 4, 6, 8, 10, 0, 0, 0]], dtype=jnp.int32)
    pos = jnp.arange(8, dtype=jnp.int32)[None, :]
    seg = jnp.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    raw_audio = np.linspace(0.5, 2.5, num=1 * 12 * 8, dtype=np.float32).reshape((1, 12, 8))
    audios_jax = jnp.asarray(raw_audio)

    eager_with_audio = np.asarray(
        audio_model(
            decoder_input_tokens=ids,
            decoder_positions=pos,
            decoder_segment_ids=seg,
            encoder_audios=audios_jax,
            enable_dropout=False,
        ),
        dtype=np.float32,
    )
    eager_without_audio = np.asarray(
        audio_model(
            decoder_input_tokens=ids,
            decoder_positions=pos,
            decoder_segment_ids=seg,
            encoder_audios=None,
            enable_dropout=False,
        ),
        dtype=np.float32,
    )
    self.assertGreater(float(np.max(np.abs(eager_with_audio - eager_without_audio))), 1.0)

    pkl_path = self.tmp_dir / "audio_golden.pkl"
    with open(pkl_path, "wb") as f:
      pickle.dump(
          [
              {
                  "prompt": "audio_sample",
                  "tokens": [2, 4, 6, 8, 10],
                  "logits": eager_with_audio[0, :5, :],
                  "audio_values": raw_audio,
              }
          ],
          f,
      )

    npz_path = self.tmp_dir / "audio_golden.npz"
    np.savez(
        npz_path,
        sample_id=np.asarray("audio_npz"),
        prompt=np.asarray("audio_npz_prompt"),
        tokens=np.asarray([2, 4, 6, 8, 10], dtype=np.int32),
        logits=eager_with_audio[0, :5, :],
        audio_values=raw_audio,
    )

    def _make_cfg(use_multimodal: bool):
      return types.SimpleNamespace(
          init_weights_seed=0,
          mesh_axes=("data",),
          hf_access_token="",
          tokenizer_path="",
          model_name="ming-3.0-audio-tiny",
          load_parameters_path="",
          scan_layers=False,
          use_multimodal=use_multimodal,
          use_audio=True,
          use_mrope=False,
          global_batch_size_to_train_on=1,
          max_target_length=8,
          vocab_size=32,
      )

    def _make_args(golden_path: pathlib.Path):
      return types.SimpleNamespace(
          hf_model_path="",
          trust_remote_code=True,
          run_ds41_mm_audit=False,
          hf_checkpoint_dir_for_audit="",
          export_orbax_dir="",
          mm_golden_logits_path="",
          thresholds_json="",
          max_images_per_sample=4,
          summary_json="",
          enable_long_text_gate=False,
          require_frozen_topk_gate=False,
          golden_logits_path=str(golden_path),
          output_logits_path="",
          gcs_output_logits_path="",
          run_hf_model=False,
          token_size=None,
          skip_first_token=False,
          atol=1e-5,
          rtol=1e-5,
          max_kl_div=1e-6,
          clip_logits_epsilon=None,
      )

    with (
        mock.patch.object(forward_pass_logit_checker.maxtext_utils, "create_device_mesh", return_value=single_dev),
        mock.patch.object(logit_runner.maxtext_utils, "create_device_mesh", return_value=single_dev),
        mock.patch.object(forward_pass_logit_checker.AutoTokenizer, "from_pretrained", return_value=None),
    ):
      # Check both use_multimodal=False (isolates encoder_audios threading) and use_multimodal=True (Ming/Qwen-Omni).
      summary_audio_only = forward_pass_logit_checker.main(
          _make_cfg(use_multimodal=False), _make_args(pkl_path), preloaded_model=audio_model
      )
      self.assertTrue(summary_audio_only["passed"])
      self.assertLess(summary_audio_only["max_kl"], 1e-6)

      summary_mm_audio = forward_pass_logit_checker.main(
          _make_cfg(use_multimodal=True), _make_args(pkl_path), preloaded_model=audio_model
      )
      self.assertTrue(summary_mm_audio["passed"])
      self.assertLess(summary_mm_audio["max_kl"], 1e-6)

      summary_npz_audio = logit_runner.main(
          _make_cfg(use_multimodal=True), _make_args(npz_path), preloaded_model=audio_model
      )
      self.assertTrue(summary_npz_audio["passed"])

      # Negative control: perturbed audio_values fails logit gates when encoder_audios is genuinely wired.
      bad_pkl_path = self.tmp_dir / "audio_golden_perturbed.pkl"
      with open(bad_pkl_path, "wb") as f:
        pickle.dump(
            [
                {
                    "prompt": "audio_sample_bad",
                    "tokens": [2, 4, 6, 8, 10],
                    "logits": eager_with_audio[0, :5, :],
                    "audio_values": -raw_audio,
                }
            ],
            f,
        )
      with self.assertRaisesRegex(AssertionError, "forward logit samples failed gates"):
        forward_pass_logit_checker.main(
            _make_cfg(use_multimodal=True), _make_args(bad_pkl_path), preloaded_model=audio_model
        )

    eval_step, nnx_state = forward_pass_logit_checker.make_eval_step_nnx(audio_model, mesh)
    jitted_with_audio = np.asarray(eval_step(nnx_state, ids, pos, seg, None, None, audios_jax), dtype=np.float32)
    jitted_without_audio = np.asarray(eval_step(nnx_state, ids, pos, seg), dtype=np.float32)
    np.testing.assert_allclose(jitted_with_audio, eager_with_audio, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(jitted_without_audio, eager_without_audio, atol=1e-6, rtol=1e-6)

  def test_main_run_hf_model_initializes_model_and_eval_step_nnx(self):
    """Verifies --run_hf_model=true initializes model and eval_step_nnx without UnboundLocalError (torch-free)."""

    class _NumpyTensorStub:
      """Minimal numpy-backed tensor stub exercising check_kl_divergence without requiring PyTorch."""

      def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr, dtype=np.float32)

      @property
      def shape(self) -> tuple[int, ...]:
        return self._arr.shape

      def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._arr, dtype=dtype)

      def __getitem__(self, idx) -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr[idx])

      def reshape(self, *shape: int) -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr.reshape(shape))

      def unsqueeze(self, dim: int) -> "_NumpyTensorStub":
        return _NumpyTensorStub(np.expand_dims(self._arr, axis=dim))

      def sum(self, dim: int = -1, keepdim: bool = False) -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr.sum(axis=dim, keepdims=keepdim))

      def max(self) -> "_NumpyTensorStub":
        return _NumpyTensorStub(np.asarray(self._arr.max(), dtype=np.float32))

      def item(self) -> float:
        return float(self._arr.item())

      def tolist(self):
        return self._arr.tolist()

      def __truediv__(self, other: "_NumpyTensorStub") -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr / np.asarray(other))

      def __lt__(self, other: float) -> bool:
        return bool(self._arr < other)

    def _fake_softmax(x: _NumpyTensorStub, dim: int = -1) -> _NumpyTensorStub:
      arr = np.asarray(x, dtype=np.float64)
      shifted = arr - np.max(arr, axis=dim, keepdims=True)
      exp = np.exp(shifted)
      return _NumpyTensorStub((exp / np.sum(exp, axis=dim, keepdims=True)).astype(np.float32))

    def _fake_log_softmax(x: _NumpyTensorStub, dim: int = -1) -> _NumpyTensorStub:
      arr = np.asarray(x, dtype=np.float64)
      shifted = arr - np.max(arr, axis=dim, keepdims=True)
      log_z = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
      return _NumpyTensorStub((shifted - log_z).astype(np.float32))

    def _fake_kl_div(
        input: _NumpyTensorStub,  # pylint: disable=redefined-builtin
        target: _NumpyTensorStub,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> _NumpyTensorStub:
      del log_target
      log_q = np.asarray(input, dtype=np.float64)
      p = np.asarray(target, dtype=np.float64)
      elem = np.where(p > 0.0, p * (np.log(np.maximum(p, 1e-30)) - log_q), 0.0)
      if reduction == "none":
        return _NumpyTensorStub(elem.astype(np.float32))
      if reduction == "batchmean":
        return _NumpyTensorStub(np.asarray(elem.sum() / p.shape[0], dtype=np.float32))
      return _NumpyTensorStub(np.asarray(elem.mean(), dtype=np.float32))

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        float16="float16",
        no_grad=contextlib.nullcontext,
        clamp=lambda x, min=None: _NumpyTensorStub(np.clip(np.asarray(x), a_min=min, a_max=None)),
        log=lambda x: _NumpyTensorStub(np.log(np.asarray(x))),
    )
    fake_f = types.SimpleNamespace(
        softmax=_fake_softmax,
        log_softmax=_fake_log_softmax,
        kl_div=_fake_kl_div,
    )

    class DummyRunHFModel(nnx.Module):
      """Tiny NNX model for on-the-fly --run_hf_model parity tests."""

      def __init__(self, vocab_size: int, d_model: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

      def __call__(  # pylint: disable=arguments-differ,too-many-positional-arguments,too-many-arguments
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          encoder_audios: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, encoder_audios, enable_dropout
        return self.head(self.embed(decoder_input_tokens))

    tiny_nnx_model = DummyRunHFModel(vocab_size=32, d_model=16, rngs=nnx.Rngs(11))
    single_dev = np.array(jax.devices()[:1]).reshape((1,))

    class _FakeTokenizer:
      """Deterministic word-length tokenizer stub."""

      eos_token = "<eos>"
      pad_token = "<pad>"

      def __call__(
          self,
          text: str,
          *,
          return_tensors: str = "pt",
          padding: bool = True,
          max_length: int = 16,
          truncation: bool = True,
      ):
        del return_tensors, padding, truncation
        toks = [((ord(ch) % 28) + 2) for ch in text.split()[:max_length] for ch in ch[:1]][:max_length]
        if not toks:
          toks = [2, 3, 4]
        return {"input_ids": np.asarray([toks], dtype=np.int32)}

      def decode(self, tok_id: int) -> str:
        return f"tok_{int(tok_id)}"

    logit_perturbation = {"offset": 0.0}

    class _FakeHFModel:
      """HF model stub emitting exact (or perturbed) logits matching tiny_nnx_model."""

      def __call__(self, **inputs):
        ids_np = np.asarray(inputs["input_ids"], dtype=np.int32)
        logits_jax = tiny_nnx_model(decoder_input_tokens=jnp.asarray(ids_np))
        logits_np = np.asarray(logits_jax, dtype=np.float32)
        if logit_perturbation["offset"] != 0.0:
          logits_np = logits_np[:, :, ::-1] + logit_perturbation["offset"]
        return types.SimpleNamespace(logits=_NumpyTensorStub(logits_np))

    hf_dir = self.tmp_dir / "fake_hf_dir"
    hf_dir.mkdir(parents=True, exist_ok=True)
    out_logits_file = self.tmp_dir / "out" / "run_hf_logits.jsonl"

    cfg = types.SimpleNamespace(
        init_weights_seed=0,
        mesh_axes=("data",),
        hf_access_token="",
        tokenizer_path="",
        model_name="qwen3-0.6b",
        dtype="float32",
        hf_lora_adapter_path="",
        global_batch_size_to_train_on=1,
        max_target_length=16,
        vocab_size=32,
        lora=types.SimpleNamespace(enable_lora=True, lora_restore_path="/tmp/fake_lora"),
    )
    test_args = types.SimpleNamespace(
        hf_model_path=str(hf_dir),
        trust_remote_code=True,
        golden_logits_path="",
        output_logits_path=str(out_logits_file),
        gcs_output_logits_path="",
        run_hf_model=True,
        reference_model="",
        token_size=None,
        skip_first_token=False,
        atol=None,
        rtol=1e-5,
        max_kl_div=1e-5,
        clip_logits_epsilon=None,
    )

    with (
        mock.patch.object(forward_pass_logit_checker.maxtext_utils, "create_device_mesh", return_value=single_dev),
        mock.patch.object(forward_pass_logit_checker, "torch", fake_torch),
        mock.patch.object(forward_pass_logit_checker, "F", fake_f),
        mock.patch.object(
            forward_pass_logit_checker,
            "convert_jax_weight_to_torch",
            side_effect=lambda arr: _NumpyTensorStub(np.asarray(arr, dtype=np.float32)),
        ),
        mock.patch.object(
            forward_pass_logit_checker.AutoTokenizer,
            "from_pretrained",
            return_value=_FakeTokenizer(),
        ),
        mock.patch.object(
            forward_pass_logit_checker,
            "AutoModelForCausalLM",
            types.SimpleNamespace(from_pretrained=mock.Mock(return_value=_FakeHFModel())),
        ),
        mock.patch.object(
            forward_pass_logit_checker.model_creation_utils,
            "from_pretrained",
            return_value=tiny_nnx_model,
        ) as mock_mt_from_pretrained,
        mock.patch.object(
            forward_pass_logit_checker.lora_utils,
            "apply_lora_to_model",
            side_effect=lambda model, mesh, config: model,
        ) as mock_apply_lora,
        mock.patch.object(
            forward_pass_logit_checker.lora_utils,
            "restore_lora_from_path",
        ) as mock_restore_lora,
    ):
      # (0) Missing hf_model_path still fails fast before loading MaxText weights.
      missing_hf_args = types.SimpleNamespace(**{**vars(test_args), "hf_model_path": ""})
      with self.assertRaisesRegex(ValueError, "run_hf_model requires hf_model_path"):
        forward_pass_logit_checker.main(cfg, missing_hf_args, preloaded_model=None)
      mock_mt_from_pretrained.assert_not_called()

      # (i) preloaded_model=None initializes MaxText model (+ LoRA) and runs @jax.jit eval_step_nnx across prompts.
      result_none = forward_pass_logit_checker.main(cfg, test_args, preloaded_model=None)
      self.assertIsNone(result_none)
      mock_mt_from_pretrained.assert_called_once()
      self.assertEqual(mock_mt_from_pretrained.call_args.kwargs["model_mode"], MODEL_MODE_TRAIN)
      mock_apply_lora.assert_called_once()
      mock_restore_lora.assert_called_once()
      self.assertTrue(out_logits_file.is_file())
      with jsonlines.open(out_logits_file, "r") as reader:
        saved_batches = list(reader)
      self.assertEqual(len(saved_batches), 1)
      self.assertEqual(len(saved_batches[0]), 4)

      # (ii) preloaded_model=tiny_nnx_model reuses preloaded model without calling from_pretrained again.
      mock_mt_from_pretrained.reset_mock()
      no_save_args = types.SimpleNamespace(**{**vars(test_args), "output_logits_path": ""})
      result_preloaded = forward_pass_logit_checker.main(cfg, no_save_args, preloaded_model=tiny_nnx_model)
      self.assertIsNone(result_preloaded)
      mock_mt_from_pretrained.assert_not_called()

      # (iii) Negative control: mismatched HF logits fail check_kl_divergence.
      logit_perturbation["offset"] = 3.0
      with self.assertRaisesRegex(AssertionError, "KL divergence values .* exceed the threshold"):
        forward_pass_logit_checker.main(cfg, no_save_args, preloaded_model=tiny_nnx_model)

  def test_get_data_gemma3_gemma4_and_audio_image_coexistence(self):
    """Verifies Gemma3/Gemma4 image loading and audio+image coexistence in forward_pass_logit_checker.get_data."""
    base_yaml = str(REPO_ROOT / "src/maxtext/configs/base.yml")
    for model_name in ("gemma3-4b", "gemma4-26b"):
      cfg = pyconfig.initialize(
          [
              "",
              base_yaml,
              f"model_name={model_name}",
              "use_multimodal=true",
              "global_batch_size_to_train_on=2",
              "max_target_length=64",
              "skip_jax_distributed_system=true",
          ]
      )
      img_sz = cfg.image_size_for_vit
      h, w = (img_sz, img_sz) if isinstance(img_sz, int) else img_sz
      p_sz = cfg.patch_size_for_vit
      c = 3
      golden_3d = {
          "tokens": np.arange(8, dtype=np.int32),
          "logits": np.zeros((8, 16), dtype=np.float32),
          "prompt": "test",
          "pixel_values": np.ones((c, h, w), dtype=np.float32),
      }
      expected_batch = int(cfg.global_batch_size_to_train_on)
      _, _, _, _, _, px3, au3 = forward_pass_logit_checker.get_data(golden_3d, cfg)
      self.assertIsNone(au3)
      self.assertEqual(px3.shape, (expected_batch, h, w, c))

      if model_name == "gemma4-26b":
        golden_2d = {
            "tokens": np.arange(8, dtype=np.int32),
            "logits": np.zeros((8, 16), dtype=np.float32),
            "prompt": "test",
            "pixel_values": np.ones(((h // p_sz) * (w // p_sz), p_sz * p_sz * c), dtype=np.float32),
        }
        _, _, _, _, _, px2, _ = forward_pass_logit_checker.get_data(golden_2d, cfg)
        self.assertEqual(px2.shape, (expected_batch, h, w, c))

    cfg_both = types.SimpleNamespace(
        model_name="deepseek4.1-flash-mini",
        global_batch_size_to_train_on=2,
        max_target_length=16,
        use_audio=True,
        use_multimodal=True,
        use_mrope=False,
    )
    golden_both = {
        "tokens": np.arange(4, dtype=np.int32),
        "logits": np.zeros((4, 8), dtype=np.float32),
        "prompt": "both",
        "audio_values": np.ones((1, 5, 80), dtype=np.float32),
        "pixel_values": np.ones((1, 4, 588), dtype=np.float32),
    }
    _, _, _, _, _, px_both, au_both = forward_pass_logit_checker.get_data(golden_both, cfg_both)
    self.assertEqual(px_both.shape, (2, 1, 4, 588))
    self.assertEqual(au_both.shape, (2, 5, 80))

  def test_run_checker_propagates_transient_errors_and_exits_42_on_kl_failure(self):
    """Verifies run_checker re-raises transient errors and maps KL AssertionErrors to exit 42."""
    cfg = types.SimpleNamespace()
    test_args = types.SimpleNamespace()
    with mock.patch.object(forward_pass_logit_checker, "main", side_effect=OSError("transient GCS timeout")):
      with self.assertRaisesRegex(OSError, "transient GCS timeout"):
        forward_pass_logit_checker.run_checker(cfg, test_args)

    with mock.patch.object(
        forward_pass_logit_checker,
        "main",
        side_effect=AssertionError("KL divergence values [0.5] exceed the threshold 0.01"),
    ):
      with self.assertRaises(SystemExit) as cm:
        forward_pass_logit_checker.run_checker(cfg, test_args)
      self.assertEqual(cm.exception.code, forward_pass_logit_checker.EXIT_DETERMINISTIC_GATE_FAIL)

  def test_masked_prefill_runner_forwards_run_hf_model_to_load_tokenizer(self):
    """Verifies masked_prefill_runner._main_impl forwards run_hf_model=True to load_tokenizer."""
    cfg = types.SimpleNamespace(mesh_axes=("data",))
    test_args = types.SimpleNamespace(
        hf_model_path="org/repo",
        trust_remote_code=False,
        run_hf_model=True,
    )
    with (
        mock.patch.object(
            masked_prefill_runner.maxtext_utils, "create_device_mesh", return_value=np.array([jax.devices()[0]])
        ),
        mock.patch.object(
            masked_prefill_runner.fplc,
            "load_tokenizer",
            side_effect=OSError("tokenizer unavailable"),
        ) as mock_tok,
    ):
      with self.assertRaisesRegex(OSError, "tokenizer unavailable"):
        masked_prefill_runner._main_impl(cfg, test_args)  # pylint: disable=protected-access
      self.assertTrue(mock_tok.call_args.kwargs["run_hf_model"])

  def test_frozen_and_natural_step_fns_exclude_moe_bias_var_sharding(self):
    """Verifies trace/frozen step_fns omit MoEBiasVar from captured_non_params and preserve P(None, None)."""

    class DummyMoEBiasModel(nnx.Module):
      """Minimal NNX module carrying 2D MoEBiasVar leaves and trace buffers."""

      def __init__(self, bias_sharding: jax.sharding.NamedSharding, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(16, 4, rngs=rngs)
        self.bias = moe.MoEBiasVar(jax.device_put(jnp.zeros((2, 4), dtype=jnp.float32), bias_sharding))
        self.bias_vl = moe.MoEBiasVar(jax.device_put(jnp.ones((2, 4), dtype=jnp.float32), bias_sharding))
        self._captured_topk_indices = nnx.Variable(jnp.zeros((1, 2), dtype=jnp.int32))

      def __call__(
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, enable_dropout
        x = self.embed(decoder_input_tokens)
        bias_sum = self.bias.get_value()[0] + self.bias_vl.get_value()[0]
        self._captured_topk_indices.set_value(jnp.ones((1, 2), dtype=jnp.int32))
        return x + bias_sum[None, None, :]

    devs = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devs), ("data",))
    expected_spec = jax.sharding.PartitionSpec(None, None)
    bias_sharding = jax.sharding.NamedSharding(mesh, expected_spec)
    model = DummyMoEBiasModel(bias_sharding, rngs=nnx.Rngs(0))
    ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    step_args = (ids, None, None, None, None)

    deepseek_v41_frozen_topk.clear_frozen_topk_step_caches()
    for step_fn_getter in (
        deepseek_v41_frozen_topk.get_frozen_topk_step_fn,
        deepseek_v41_frozen_topk.get_natural_routing_step_fn,
    ):
      gdef, st = nnx.split(model)
      step_fn = step_fn_getter(model, mesh, None, gdef)
      logits, captured = step_fn(st, step_args)
      self.assertEqual(logits.shape, (1, 4, 4))
      captured_leaves = jax.tree_util.tree_leaves(captured)
      self.assertEqual(len(captured_leaves), 1)
      clean = deepseek_v41_frozen_topk._filter_captured_non_params(captured)  # pylint: disable=protected-access
      nnx.update(model, clean)
      self.assertEqual(model.bias[...].sharding.spec, expected_spec)
      self.assertEqual(model.bias_vl[...].sharding.spec, expected_spec)

  def test_barrier_self_heals_when_clean_stale_ready_deletes_early_worker(self):
    """Verifies wait_for_workers_ready re-publishes deleted worker markers when CLEAN_STALE_READY_ON_START=1."""
    ready_dir = self.tmp_dir / "ready_race"
    ready_dir.mkdir(parents=True, exist_ok=True)
    barrier_epoch = int(time.time())
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable

    w1_cmd = (
        f'source "{SCRIPT_PATH.as_posix()}" && '
        f'wait_for_workers_ready "{ready_dir.as_posix()}" 2 1 att_race {barrier_epoch} 180 6 1'
    )
    with subprocess.Popen(
        ["bash", "-c", w1_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    ) as w1_proc:
      w1_marker = ready_dir / "worker_1.ready"
      deadline = time.time() + 4.0
      while not w1_marker.is_file() and time.time() < deadline:
        time.sleep(0.05)
      self.assertTrue(w1_marker.is_file(), "Worker 1 did not write its initial marker in time")

      # Simulate worker 0 running CLEAN_STALE_READY_ON_START=1 deletion after worker 1 already entered wait loop.
      w0_cmd = (
          f'source "{SCRIPT_PATH.as_posix()}" && '
          f'rm -f "{ready_dir.as_posix()}"/worker_*.ready && '
          f'wait_for_workers_ready "{ready_dir.as_posix()}" 2 0 att_race {barrier_epoch} 180 6 1'
      )
      w0_res = subprocess.run(["bash", "-c", w0_cmd], capture_output=True, text=True, env=env, check=False)
      w1_stdout, w1_stderr = w1_proc.communicate(timeout=10)
      self.assertEqual(w0_res.returncode, 0, msg=f"w0 failed: {w0_res.stdout}\n{w0_res.stderr}")
      self.assertEqual(w1_proc.returncode, 0, msg=f"w1 failed: {w1_stdout}\n{w1_stderr}")

    bench_script = (REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/3_bench_deepseek41_mm.sh").read_text(
        encoding="utf-8"
    )
    self.assertIn("WORKER_ID:-${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}", bench_script)

  def test_thresholds_multimodal_precedence_and_bare_output_logits_path(self):
    """Verifies thresholds.json has no shadowing MM keys, multimodal_cases takes precedence, and bare filenames save."""
    thresholds_path = REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/thresholds.json"
    thresholds_cfg = json.loads(thresholds_path.read_text(encoding="utf-8"))
    for redundant_key in ("short_text_prompts", "long_text_prompts", "mm_short_prompts", "mm_long_prompts"):
      self.assertNotIn(redundant_key, thresholds_cfg)

    flc = thresholds_cfg["forward_logit_checker"]
    mm_cases = flc["multimodal_cases"]
    self.assertIn("text_positions", mm_cases["mm_short_prompts"]["description"])
    self.assertNotIn("and all_positions", mm_cases["mm_short_prompts"]["description"])
    self.assertIn("text_positions", mm_cases["mm_long_prompts"]["description"])
    self.assertNotIn("and all_positions", mm_cases["mm_long_prompts"]["description"])

    # forward_logit_checker.multimodal_cases overrides top-level keys; legacy top-level-only dicts keep fallback.
    override_cfg = {
        "mm_short_prompts": {"max_kl": 0.9, "last_token_top10_matches": 10},
        "forward_logit_checker": {
            "short_text_prompts": {"max_kl": 2e-7, "last_token_top10_matches": 10},
            "long_text_prompts": {"max_kl": 0.0212, "last_token_top10_matches": 10},
            "multimodal_cases": {"L1": {"max_kl": 0.0125, "last_token_top10_matches": 10}},
        },
    }
    gate_id, spec = logit_runner.select_forward_threshold_spec(
        override_cfg, sample_name="L1", seq_len=128, is_mm_sample=True
    )
    self.assertEqual(gate_id, "mm_short_prompts")
    self.assertAlmostEqual(spec["max_kl"], 0.0125)

    legacy_cfg = {"mm_short_prompts": {"max_kl": 0.033, "last_token_top10_matches": 10}}
    legacy_id, legacy_spec = logit_runner.select_forward_threshold_spec(
        legacy_cfg, sample_name="L1", seq_len=128, is_mm_sample=True
    )
    self.assertEqual(legacy_id, "mm_short_prompts")
    self.assertAlmostEqual(legacy_spec["max_kl"], 0.033)

    # Bare filename output_logits_path must not call os.makedirs("").
    lr_src = (REPO_ROOT / "tests/utils/deepseek_v41/logit_runner.py").read_text(encoding="utf-8")
    self.assertNotIn("os.makedirs(os.path.dirname(test_args.output_logits_path)", lr_src)

  def test_masked_prefill_tie_branch_cache_pid_isolated_and_top10_schema(self):
    """Verifies tie-branch cache dir is PID-isolated and torch-free top-10 overlap uses N/M format."""
    cache_dir = masked_prefill_runner._get_tie_branch_cache_dir()  # pylint: disable=protected-access
    self.assertIn(str(os.getpid()), cache_dir.name)
    self.assertNotEqual(str(cache_dir), "/tmp/cached_tie_branches")
    logits = np.arange(16, dtype=np.float32)[None, :]
    with mock.patch.object(masked_prefill_runner, "HAS_TORCH", False):
      summary = masked_prefill_runner._last_token_top10_summary(  # pylint: disable=protected-access
          logits, logits, tokenizer=None
      )
    self.assertEqual(summary["overlap_count"], "10/10")


if __name__ == "__main__":
  unittest.main()
