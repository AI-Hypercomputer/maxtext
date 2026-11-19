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

"""Tests for model-independent audio dispatch, features, and output."""

import tempfile
from types import SimpleNamespace
import unittest
import wave

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common import decoder_features
from maxtext.layers import audio_decoders
from maxtext.inference.component_runtime import ComponentRuntime
from maxtext.multimodal.audio_utils import write_wav


class AudioDispatchTest(unittest.TestCase):
  """Audio selection respects capabilities and existing configuration files."""

  def test_disabled_audio_does_not_select_qwen(self):
    config = SimpleNamespace(enable_audio_output=False, model_name="other")
    self.assertIsNone(audio_decoders.get_audio_output_type(config))
    self.assertEqual(audio_decoders.create_audio_components(config, None, None, model_mode="train", rngs=None), {})

  def test_existing_qwen_config_selects_backend(self):
    config = SimpleNamespace(enable_audio_output=True, model_name="qwen3-omni-30b-a3b", accept_hidden_layer=24)
    self.assertEqual(audio_decoders.get_audio_output_type(config), "qwen3_omni")
    spec = audio_decoders.get_feature_spec(config)
    self.assertTrue(spec.input_embeddings)
    self.assertEqual(spec.hidden_layers, (24,))

  def test_unsupported_audio_is_rejected(self):
    for options in ({"model_name": "other"}, {"model_name": "other", "audio_output_type": "unknown"}):
      with self.subTest(options=options), self.assertRaises(ValueError):
        audio_decoders.get_audio_output_type(SimpleNamespace(enable_audio_output=True, **options))


class DecoderFeaturesTest(unittest.TestCase):
  """Feature extraction uses declared decoder features without model-specific names."""

  def test_extract_multiple_layers_ignores_other_intermediates(self):
    spec = decoder_features.DecoderFeatureSpec(input_embeddings=True, hidden_layers=(1, 3))
    embeddings = jnp.ones((1, 2, 4))
    first = embeddings * 2
    third = embeddings * 4
    intermediates = {
        "decoder": {"input_embeddings": (embeddings,), "hidden_states_1": (first,), "hidden_states_3": (third,)},
        "other": {"input_embeddings": (embeddings * 100,)},
    }
    features = decoder_features.extract_decoder_features(intermediates, spec)
    np.testing.assert_array_equal(features["input_embeddings"], embeddings)
    np.testing.assert_array_equal(features["hidden_states_1"], first)
    np.testing.assert_array_equal(features["hidden_states_3"], third)

  def test_missing_feature_fails_explicitly(self):
    spec = decoder_features.DecoderFeatureSpec(hidden_layers=(2,))
    with self.assertRaisesRegex(ValueError, "hidden_states_2"):
      decoder_features.extract_decoder_features({"decoder": {}}, spec)

  def test_disabled_capture_needs_no_intermediates(self):
    spec = decoder_features.DecoderFeatureSpec()
    self.assertFalse(spec.enabled)
    self.assertEqual(decoder_features.extract_decoder_features({}, spec), {})

  def test_invalid_layer_or_scan_is_rejected(self):
    for layers, scan in (((0,), False), ((4,), False), ((1,), True)):
      with self.subTest(layers=layers, scan=scan), self.assertRaises(ValueError):
        decoder_features.DecoderFeatureSpec(hidden_layers=layers).validate(num_layers=3, scan_layers=scan)


class ComponentRuntimeTest(unittest.TestCase):
  """Component runtimes preserve loaded non-parameter values and independent caches."""

  def test_missing_rng_does_not_replace_loaded_state(self):
    class Component(nnx.Module):

      def __init__(self):
        self.weight = nnx.Param(jnp.ones((2,)))
        self.loaded = nnx.BatchStat(jnp.array([7.0]))
        self.rng = nnx.Rngs(0)
        self.cache = nnx.Cache(jnp.array([9.0]))

    model = nnx.eval_shape(Component)
    params = nnx.state(Component(), nnx.Param)
    loaded = nnx.State({"loaded": nnx.BatchStat(jnp.array([42.0]))})
    runtime = ComponentRuntime.from_model(model, params, loaded)
    restored = runtime.merge()
    np.testing.assert_array_equal(restored.loaded.get_value(), [42.0])
    np.testing.assert_array_equal(restored.cache.get_value(), [0.0])
    self.assertIsInstance(restored.rng.default.key.get_value(), jax.Array)

  def test_missing_non_rng_state_is_rejected(self):
    class Component(nnx.Module):

      def __init__(self):
        self.stat = nnx.BatchStat(jnp.array([7.0]))

    with self.assertRaisesRegex(ValueError, "stat"):
      ComponentRuntime.from_model(nnx.eval_shape(Component), nnx.State({}), nnx.State({}))


class AudioOutputTest(unittest.TestCase):
  """Waveforms carry their own sample rate and are clipped to PCM range."""

  def test_write_wav_preserves_rate_and_clips_samples(self):
    with tempfile.TemporaryDirectory() as directory:
      path = directory + "/audio.wav"
      write_wav(path, np.array([-2.0, 0.0, 2.0]), sample_rate=16000)
      with wave.open(path, "rb") as handle:
        self.assertEqual(handle.getframerate(), 16000)
        self.assertEqual(handle.getnchannels(), 1)
        np.testing.assert_array_equal(np.frombuffer(handle.readframes(3), dtype="<i2"), [-32767, 0, 32767])


if __name__ == "__main__":
  unittest.main()
