# Copyright 2026 Google LLC
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

"""Tests for the audio SFT (audio + text -> text) HF input pipeline with the Ming frontend.

The end-to-end pipeline test needs the Ming tokenizer; point `MING_HF_PATH` at a local copy of
inclusionAI/Ming-3.0-audio-tiny (top-level files) to enable it. The featurizer test runs standalone.
"""

import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.multimodal import processor_ming
from tests.utils.test_helpers import get_test_config_path

MING_HF_PATH = os.environ.get("MING_HF_PATH", "")


def _wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
  t = np.arange(int(seconds * sample_rate)) / sample_rate
  wav = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
  buf = io.BytesIO()
  sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
  return buf.getvalue()


class AudioFeaturizerTest(unittest.TestCase):
  """`preprocess_audio_for_training_ming`: static mel shape, true-length placeholder count."""

  def test_static_shape_and_lengths(self):
    class Cfg:  # pylint: disable=too-few-public-methods
      audio_max_mel_frames = 1000  # 10 s

    for seconds, expected_frames in ((3.18, 318), (10.0, 1000), (25.0, 1000)):
      out = processor_ming.preprocess_audio_for_training_ming({"bytes": _wav_bytes(seconds), "path": "x.wav"}, Cfg)
      self.assertEqual(out.audio_values.shape, (1, 1000, processor_ming.NUM_MEL_BINS))
      self.assertEqual(out.audio_values.dtype, np.float32)
      self.assertEqual(int(out.audio_mask.sum()), expected_frames)
      self.assertEqual(int(out.audio_lengths[0]), processor_ming.get_num_audio_tokens(expected_frames))
    # Padded tail is whisper "silence": constant and equal to the clipped floor of the log-mel.
    out = processor_ming.preprocess_audio_for_training_ming(_wav_bytes(2.0), Cfg)
    tail = out.audio_values[0, 500:]
    self.assertAlmostEqual(float(tail.max() - tail.min()), 0.0, places=5)
    # 8 kHz input is resampled to 16 kHz before featurization.
    out_8k = processor_ming.preprocess_audio_for_training_ming({"bytes": _wav_bytes(2.0, 8000), "path": ""}, Cfg)
    self.assertEqual(int(out_8k.audio_mask.sum()), 200)


@unittest.skipUnless(MING_HF_PATH, "set MING_HF_PATH to the Ming HF directory to run the pipeline test")
class AudioSftPipelineTest(unittest.TestCase):
  """End-to-end grain pipeline on a tiny local parquet shard."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    durations = [1.0, 4.5, 7.99, 12.0] * 2
    table = pa.Table.from_pydict(
        {
            "audio": [{"bytes": _wav_bytes(d), "path": f"{i}.wav"} for i, d in enumerate(durations)],
            "text": [f"Transcript number {i}." for i in range(len(durations))],
            "duration": durations,
        },
        schema=pa.schema(
            [
                ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                ("text", pa.string()),
                ("duration", pa.float32()),
            ]
        ),
    )
    cls.parquet = os.path.join(cls.tmp.name, "train-00000-of-00001.parquet")
    pq.write_table(table, cls.parquet)
    cls.config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="ming-3.0-audio-tiny",
        run_name="test",
        per_device_batch_size=2,  # run with JAX_PLATFORMS=cpu (1 device) so the 8-row shard yields a full batch
        max_target_length=256,
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        use_sft=True,
        use_multimodal=True,
        use_audio=True,
        sft_train_on_completion_only=True,
        packing=False,
        audio_max_mel_frames=800,  # 8 s: the 12 s clip is truncated
        dataset_type="hf",
        hf_path="parquet",
        hf_train_files=cls.parquet,
        train_split="train",
        train_data_columns=["prompt", "text"],
        train_audio_column="audio",
        default_prompt="Transcribe the speech into text.",
        tokenizer_type="huggingface",
        tokenizer_path=MING_HF_PATH,
        enable_data_shuffling=False,
        enable_checkpointing=False,
    )
    cls.mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), cls.config.mesh_axes)

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()
    super().tearDownClass()

  def test_batch(self):
    process_indices = input_pipeline_interface.get_process_loading_real_data(
        self.config.data_sharding,
        self.config.global_batch_size_to_load,
        self.config.global_batch_size_to_train_on,
        self.config.max_target_length,
        self.mesh,
    )
    train_iter = hf_data_processing.make_hf_train_iterator(self.config, self.mesh, process_indices)
    batch = {k: np.asarray(v) for k, v in next(train_iter).items()}
    bsz, seq = batch["inputs"].shape
    self.assertEqual(seq, 256)
    self.assertEqual(batch["audios"].shape, (bsz, 800, processor_ming.NUM_MEL_BINS))
    self.assertEqual(batch["audios"].dtype, np.float32)
    self.assertEqual(
        set(batch),
        {
            "inputs",
            "inputs_position",
            "inputs_segmentation",
            "targets",
            "targets_position",
            "targets_segmentation",
            "audios",
        },
    )
    # Placeholder count follows the clip's *true* length (1.0 s -> 100 frames, 4.5 s -> 450 frames).
    for row, frames in zip(range(bsz), (100, 450)):
      num_pads = int((batch["inputs"][row] == processor_ming.AUDIO_PAD_ID).sum())
      self.assertEqual(num_pads, processor_ming.get_num_audio_tokens(frames))
      # Prompt is masked out of the loss; the response (incl. <|role_end|>) is trained on.
      loss_tokens = batch["targets"][row][batch["targets_segmentation"][row] != 0]
      self.assertGreater(len(loss_tokens), 0)
      self.assertNotIn(processor_ming.AUDIO_PAD_ID, loss_tokens)
      self.assertEqual(int(loss_tokens[-1]), 156895)  # <|role_end|>
      # Targets are inputs shifted left by one on the supervised positions.
      pos = np.nonzero(batch["targets_segmentation"][row])[0]
      np.testing.assert_array_equal(batch["targets"][row][pos], batch["inputs"][row][pos + 1])


if __name__ == "__main__":
  unittest.main()
