# Copyright 2023-2026 Google LLC
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

"""Greedy decode through MaxEngine must match a teacher-forced forward pass.

Golden-logit tests only cover the forward pass, so nothing catches a broken
prefill -> autoregressive handoff. Here a tiny model is decoded greedily through
prefill/insert/generate and compared against the argmax rollout of a plain forward
pass over the same growing prefix, which is the same sequence by definition. That
holds for any model, so new families can be added to the parameter list.

DeepSeek is the case that motivated this: MaxEngine's NNX path builds the model once
in PREFILL mode and passes the mode per call, so a layer that reads its
construction-time model_mode instead of the argument runs the prefill attention path
during decode and silently produces wrong tokens.
"""

import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import deepseek
from maxtext.utils import maxtext_utils, model_creation_utils

pytest.importorskip("jetstream", reason="jetstream not installed")
from maxtext.inference.maxengine import maxengine
from tests.utils.test_helpers import get_test_config_path

PROMPT = [3, 17, 42, 5, 9]
STEPS = 4

_COMMON = {
    "base_emb_dim": 64,
    "base_mlp_dim": 64,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "base_num_decoder_layers": 3,
    "vocab_size": 64,
    "max_prefill_predict_length": 8,
    "max_target_length": 16,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "sparse_matmul": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
}

_DEEPSEEK = {
    "model_name": "deepseek2-16b",
    "override_model_config": True,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_experts": 1,
    # The naive MLA kv cache sizes its key cache with the value head dim, so prefill
    # writes a wider key than the cache holds. Unrelated to decode consistency and it
    # fails the same way on the Linen path.
    "mla_naive_kvcache": False,
}

# MoE and dense DeepSeek layers take different paths to attention, so cover both.
_DEEPSEEK_MOE = _DEEPSEEK | {"first_num_dense_layers": 1}
_DEEPSEEK_DENSE = _DEEPSEEK | {"first_num_dense_layers": 3}


def _make_config(**overrides):
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **(_COMMON | overrides))


def _mesh(cfg):
  """Builds the mesh over every available device.

  ici_fsdp_parallelism keeps its base.yml default of -1, so the fsdp axis absorbs
  however many devices the host has. Pinning the ICI axes instead would fix the mesh
  at one device and fail everywhere except a single-device runner.

  Args:
    cfg: Model config.

  Returns:
    The device mesh.
  """
  return jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)


class DecodeConsistencyTest(parameterized.TestCase):
  """Decode must agree with the forward pass it is supposed to be replaying."""

  def _forward_rollout(self, cfg, mesh, params_state):
    """Rolls out greedily, recomputing a full forward pass over the prefix each step.

    Args:
      cfg: Model config.
      mesh: Device mesh the model is built on.
      params_state: nnx.Param state to load into the model.

    Returns:
      The generated token ids.
    """
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=0)
      )
    nnx.update(model, params_state)

    tokens = list(PROMPT)
    generated = []
    pad = cfg.max_target_length
    batch = cfg.micro_batch_size_to_train_on
    for _ in range(STEPS + 1):
      ids = jnp.tile(jnp.asarray([tokens + [0] * (pad - len(tokens))], dtype=jnp.int32), (batch, 1))
      positions = jnp.tile(jnp.asarray([list(range(pad))], dtype=jnp.int32), (batch, 1))
      segment_ids = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (pad - len(tokens))], dtype=jnp.int32), (batch, 1))
      with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        logits = model(
            ids,
            positions,
            decoder_segment_ids=segment_ids,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )
      next_token = int(jnp.argmax(logits[0, len(tokens) - 1]))
      generated.append(next_token)
      tokens.append(next_token)
    return generated

  def _engine_rollout(self, engine, params):
    """Rolls out greedily through prefill, insert and generate.

    Args:
      engine: MaxEngine with params already loaded.
      params: Params returned by load_params.

    Returns:
      The generated token ids.
    """
    padded = jnp.asarray(
        PROMPT + [0] * (engine.config.max_prefill_predict_length - len(PROMPT)),
        dtype=jnp.int32,
    )
    prefix, first_token = engine.prefill(params=params, padded_tokens=padded, true_length=len(PROMPT))
    generated = [int(first_token.data[0, 0])]

    decode_state = engine.init_decode_state()
    decode_state = engine.insert(prefix, decode_state, slot=0)
    for _ in range(STEPS):
      decode_state, result = engine.generate(params, decode_state)
      generated.append(int(result.data[0, 0]))
    return generated

  def _build(self, cfg):
    """Builds a freshly initialized model.

    Args:
      cfg: Model config.

    Returns:
      A tuple of (mesh, params_state).
    """
    mesh = _mesh(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
      )
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    return mesh, params_state

  @parameterized.named_parameters(
      ("deepseek_moe", _DEEPSEEK_MOE),
      ("deepseek_dense", _DEEPSEEK_DENSE),
      ("generic", {}),
  )
  def test_greedy_decode_matches_forward_pass(self, overrides):
    cfg = _make_config(**overrides)
    mesh, params_state = self._build(cfg)

    expected = self._forward_rollout(cfg, mesh, params_state)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    actual = self._engine_rollout(engine, params)

    self.assertEqual(
        expected,
        actual,
        f"decode diverged from the forward pass: expected {expected}, got {actual}",
    )


class EngineGraphdefModeTest(unittest.TestCase):
  """MaxEngine must merge the graphdef built for the mode it is running.

  Layers are supposed to honor the call-time model_mode, but the engine should not
  depend on that: merging a prefill graphdef for an autoregressive step leaves every
  layer's `self.model_mode` reading "prefill". This guards the engine side on its own,
  so the two defenses cannot regress together silently.
  """

  def test_generate_merges_autoregressive_graphdef(self):
    cfg = _make_config(**_DEEPSEEK_MOE)
    mesh = _mesh(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
      )
    _, params_state, _ = nnx.split(model, nnx.Param, ...)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)

    # The mode each merged layer was constructed with, recorded from inside the
    # jitted prefill / generate bodies.
    built_modes = []
    original = deepseek.DeepSeekGenericLayer.attention_op

    def recording_attention_op(self, *args, **kwargs):
      built_modes.append(self.model_mode)
      return original(self, *args, **kwargs)

    padded = jnp.asarray(
        PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)),
        dtype=jnp.int32,
    )
    with mock.patch.object(deepseek.DeepSeekGenericLayer, "attention_op", recording_attention_op):
      prefix, _ = engine.prefill(params=params, padded_tokens=padded, true_length=len(PROMPT))
      self.assertTrue(built_modes, "prefill did not reach the DeepSeek attention")
      self.assertEqual(set(built_modes), {MODEL_MODE_PREFILL})

      decode_state = engine.init_decode_state()
      decode_state = engine.insert(prefix, decode_state, slot=0)
      built_modes.clear()
      engine.generate(params, decode_state)

    self.assertTrue(built_modes, "generate did not reach the DeepSeek attention")
    self.assertEqual(
        set(built_modes),
        {MODEL_MODE_AUTOREGRESSIVE},
        "generate merged a graphdef built for another mode",
    )


if __name__ == "__main__":
  unittest.main()
