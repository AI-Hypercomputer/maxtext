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

"""Helpers for m3 tests: reduced configs that keep a model small enough for CPU."""

from flax import nnx
import jax

from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path

# The `base_*` keys are what the model YAMLs set; overriding the derived names
# (`emb_dim`, ...) is silently ignored by pyconfig.
_TINY = {
    "qwen3-0.6b": {
        "base_num_decoder_layers": 2,
        "base_emb_dim": 64,
        "base_mlp_dim": 128,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
    },
}


def tiny_config(model_name: str, use_m3: bool = True, **overrides) -> pyconfig.HyperParameters:
  """A reduced `model_name` config. `overrides` are applied last, as `key=value` strings."""
  args = {
      "model_name": model_name,
      "use_m3_model": use_m3,
      "scan_layers": False,
      "weight_dtype": "bfloat16",
      "per_device_batch_size": 1,
      "max_target_length": 32,
      "max_prefill_predict_length": 16,
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "override_model_config": True,
      **_TINY[model_name],
      **overrides,
  }
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={_fmt(v)}" for k, v in args.items()])


def _fmt(value) -> str:
  return str(value).lower() if isinstance(value, bool) else str(value)


def flat_params(module: nnx.Module) -> dict[tuple, jax.Array]:
  """`{path: value}` for every `nnx.Param` in `module`."""
  return {path: var[...] for path, var in nnx.to_flat_state(nnx.state(module, nnx.Param))}
