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

"""Unit tests for vllm_decode helpers."""

import os
import subprocess
import sys
import types
import unittest

from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import pytest

pytest.importorskip("vllm")
pytest.importorskip("tunix")

pytestmark = pytest.mark.post_training

import numpy as np
from maxtext.inference.vllm_decode import build_chat_messages
from maxtext.integration.vllm.maxtext_vllm_adapter.adapter import MaxTextForCausalLM
from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal import get_multimodal_handler


def _config(prompt: str, system_prompt: str, use_multimodal: bool = False, image_path: str = ""):
  return types.SimpleNamespace(
      prompt=prompt,
      system_prompt=system_prompt,
      use_multimodal=use_multimodal,
      image_path=image_path,
  )


class BuildChatMessagesTest(unittest.TestCase):
  """Chat-message construction for the vllm_decode CLI."""

  def test_user_only_when_no_system_prompt(self):
    messages = build_chat_messages(_config("What is 2+2?", ""))
    self.assertEqual(messages, [{"role": "user", "content": "What is 2+2?"}])

  def test_system_prompt_prepended(self):
    messages = build_chat_messages(_config("Who was Albert Einstein?", "You are a helpful assistant."))
    self.assertEqual(
        messages,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who was Albert Einstein?"},
        ],
    )


class MultimodalHandlerTest(unittest.TestCase):
  """Model-family selection for vLLM multimodal handling."""

  def test_handler_selection(self):
    handler = get_multimodal_handler("qwen3-vl-2b")

    self.assertIsNotNone(handler)
    self.assertEqual(handler.placeholder_token_ids(types.SimpleNamespace(image_token_id=42)), [42])
    self.assertIsNone(get_multimodal_handler("qwen3-30b-a3b"))


class MultimodalChatMessagesTest(unittest.TestCase):
  """Multimodal chat-message construction for the vllm_decode CLI."""

  def test_multimodal_content_contains_one_item_per_image(self):
    messages = build_chat_messages(_config("Compare these.", "", True, "first.jpg,second.jpg"))
    self.assertEqual(
        messages,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Compare these."},
                ],
            }
        ],
    )


class MropeInputPositionsTest(unittest.TestCase):
  """Verify get_mrope_input_positions returns correct NumPy array."""

  def test_mrope_input_positions_returns_numpy_array(self):
    model = object.__new__(MaxTextForCausalLM)
    tokens = [10, 20, 30, 40, 50]
    positions, delta = model.get_mrope_input_positions(tokens)
    self.assertIsInstance(positions, np.ndarray)
    self.assertEqual(delta, 0)
    self.assertEqual(positions.shape, (3, 5))
    self.assertEqual(positions.dtype, np.int32)
    for dim in range(3):
      np.testing.assert_array_equal(positions[dim], np.arange(5, dtype=np.int32))

  def test_mrope_input_positions_varying_lengths(self):
    model = object.__new__(MaxTextForCausalLM)
    for length in [1, 10, 128]:
      positions, delta = model.get_mrope_input_positions(list(range(length)))
      self.assertEqual(positions.shape, (3, length))
      self.assertEqual(delta, 0)
      for dim in range(3):
        np.testing.assert_array_equal(positions[dim], np.arange(length, dtype=np.int32))


_SUBPROC_ENV = "_MAXTEXT_VLLM_DECODE_TEST_SUBPROC"


class RoutedExpertsReplicationTest(unittest.TestCase):
  """Verify MaxTextForCausalLM replicates expert_indices across the mesh."""

  @staticmethod
  def _build_wrapper(mesh: Mesh, decoder: nnx.Module) -> MaxTextForCausalLM:
    cfg = types.SimpleNamespace(dtype=jnp.float32, logical_axis_rules=())
    wrapper = object.__new__(MaxTextForCausalLM)
    object.__setattr__(wrapper, "model", decoder)
    object.__setattr__(wrapper, "mesh", mesh)
    object.__setattr__(wrapper, "maxtext_config", cfg)
    object.__setattr__(wrapper, "model_mode", "autoregressive")
    return wrapper

  def test_expert_indices_are_replicated_across_mesh(self):
    if len(jax.devices()) < 2:
      self.assertNotEqual(
          os.environ.get(_SUBPROC_ENV),
          "1",
          "Expected >= 2 CPU devices in subprocess.",
      )
      env = os.environ.copy()
      env[_SUBPROC_ENV] = "1"
      xla_flag = "--xla_force_host_platform_device_count=2"
      env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} {xla_flag}".strip()
      env["JAX_PLATFORMS"] = "cpu"
      file_dir = os.path.dirname(os.path.abspath(__file__))
      repo_root = os.path.abspath(os.path.join(file_dir, "..", "..", ".."))
      extra_paths = [repo_root, os.path.join(repo_root, "src")]
      if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
      env["PYTHONPATH"] = os.pathsep.join(extra_paths)
      result = subprocess.run(
          [
              sys.executable,
              os.path.abspath(__file__),
              "RoutedExpertsReplicationTest",
          ],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      msg = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
      self.assertEqual(result.returncode, 0, msg)
      return

    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("data",))
    num_tokens = len(devices) * 2

    class _ShardedExpertsDecoder(nnx.Module):
      """Stub decoder returning token-sharded expert_indices."""

      def __call__(self, decoder_input_tokens, **kwargs):
        del kwargs
        tokens = decoder_input_tokens[:, 0]
        hidden = jnp.zeros((tokens.shape[0], 1, 8), dtype=jnp.float32)
        kv_caches = [jnp.zeros((2, 4), dtype=jnp.float32)]
        expert_indices = jnp.stack([tokens, tokens + 100], axis=-1)[None, :, :]
        expert_indices = jax.lax.with_sharding_constraint(
            expert_indices,
            NamedSharding(mesh, PartitionSpec(None, "data", None)),
        )
        return hidden, kv_caches, expert_indices

    wrapper = self._build_wrapper(mesh, _ShardedExpertsDecoder())
    input_ids = jax.device_put(
        jnp.arange(num_tokens, dtype=jnp.int32),
        NamedSharding(mesh, PartitionSpec("data")),
    )
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    kv_caches = [jnp.zeros((2, 4), dtype=jnp.float32)]

    @jax.jit
    def _step(kvc, ids, pos):
      return wrapper(kvc, ids, types.SimpleNamespace(input_positions=pos))

    _, hidden, _, expert_indices = _step(kv_caches, input_ids, positions)

    expected_tokens = np.arange(num_tokens, dtype=np.int32)
    expected_pairs = np.stack([expected_tokens, expected_tokens + 100], axis=-1)
    expected_indices = expected_pairs[None, :, :]
    self.assertEqual(hidden.shape, (num_tokens, 8))
    self.assertIsNotNone(expert_indices)
    self.assertTrue(expert_indices.sharding.is_fully_replicated)
    np.testing.assert_array_equal(np.asarray(expert_indices), expected_indices)

  def test_none_expert_indices_passthrough(self):
    mesh = Mesh(np.array(jax.devices()), ("data",))

    class _NoExpertsDecoder(nnx.Module):
      """Stub decoder returning no expert_indices (2-tuple or None 3-tuple)."""

      def __init__(self, three_tuple: bool):
        self.three_tuple = three_tuple

      def __call__(self, decoder_input_tokens, **kwargs):
        del kwargs
        num_toks = decoder_input_tokens.shape[0]
        hidden = jnp.zeros((num_toks, 1, 8), dtype=jnp.float32)
        kv = [jnp.zeros((2, 4), dtype=jnp.float32)]
        return (hidden, kv, None) if self.three_tuple else (hidden, kv)

    for three_tuple in (False, True):
      wrapper = self._build_wrapper(mesh, _NoExpertsDecoder(three_tuple))
      _, hidden, _, expert_indices = wrapper(
          [jnp.zeros((2, 4), dtype=jnp.float32)],
          jnp.arange(4, dtype=jnp.int32),
          types.SimpleNamespace(input_positions=jnp.arange(4, dtype=jnp.int32)),
      )
      self.assertEqual(hidden.shape, (4, 8))
      self.assertIsNone(expert_indices)


if __name__ == "__main__":
  unittest.main()
