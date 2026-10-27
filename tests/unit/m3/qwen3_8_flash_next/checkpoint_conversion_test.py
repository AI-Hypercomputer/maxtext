# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint conversion tests for Qwen3.8-Flash-Next.

Checks the `PARAM_MAPPING`, `HOOK_FNS` and `HF_SHAPE` entries against the
`nnx.eval_shape` parameter tree of `Qwen3_8FlashNextModel`, for both
`scan_layers=False` and `scan_layers=True`:
  * the mapping covers exactly the model's parameter keys, and every HF target
    has a shape entry that the hooks turn into the model's shape;
  * on the mini twin, random HF tensors survive a HF -> MaxText -> HF round trip.
"""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.nnx import variablelib
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import hf_model_configs  # pylint: disable=no-name-in-module
from maxtext.checkpoint_conversion.utils import hf_shape  # pylint: disable=no-name-in-module
from maxtext.checkpoint_conversion.utils import param_mapping  # pylint: disable=no-name-in-module
from maxtext.checkpoint_conversion.utils import tensor_handling  # pylint: disable=no-name-in-module
from maxtext.checkpoint_conversion.utils import utils as conversion_utils  # pylint: disable=no-name-in-module
from maxtext.configs import pyconfig  # pylint: disable=no-name-in-module
from maxtext.m3 import models as m3_models  # pylint: disable=no-name-in-module
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
import numpy as np
import pytest

_FULL = "qwen3.8-flash-next"
_MINI = "qwen3.8-flash-next-mini"
_EMBED_KEY = "params-decoder-token_embedder-embedding"


def _get_config(model_name: str, scan_layers: bool):
  """Builds the MaxText config `to_maxtext` would use for `model_name`."""
  return pyconfig.initialize(
      [None, f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml"],
      model_name=model_name,
      override_model_config=True,
      scan_layers=scan_layers,
      dtype="float32",
      weight_dtype="float32",
      run_name="checkpoint_conversion_test",
  )


def _model_param_shapes(config) -> dict[str, tuple[int, ...]]:
  """Returns {MaxText key: shape} for the abstract model, keyed like `to_maxtext.get_maxtext_model_info`."""
  model = nnx.eval_shape(lambda: m3_models.create_model(config, mesh=None))
  shapes = {}
  for path, variable in nnx.to_flat_state(nnx.state(model)):
    if isinstance(variable, nnx.RngState):
      continue
    collection = variablelib.variable_name_from_type(type(variable), allow_register=True)
    key = "-".join([collection, *conversion_utils.param_key_parts_from_path(path)])
    shapes[key] = tuple(variable.get_value().shape)
  return shapes


def _hf_targets(hf_value) -> list[str]:
  """Flattens a mapping value (key, list of per-layer keys, or tuple of shard keys) into HF keys."""
  if isinstance(hf_value, (list, tuple)):
    return [key for item in hf_value for key in _hf_targets(item)]
  return [hf_value]


def _converted_shape(hf_value, hf_shapes, hook, mt_shape) -> tuple[int, ...]:
  """Returns the shape `to_maxtext` builds for `hf_value` from the HF shapes alone.

  A list stacks scanned layers on `param_scan_axis=0`; a tuple holds the PLE
  n-gram table shards, which the hook concatenates along the vocab rows (summed
  here so the full-size table is never materialized).
  """
  if isinstance(hf_value, list):
    slice_shapes = {_converted_shape(v, hf_shapes, hook, mt_shape[1:]) for v in hf_value}
    if len(slice_shapes) != 1:
      raise ValueError(f"Scanned layers have different shapes: {slice_shapes}")
    return (len(hf_value), *slice_shapes.pop())
  if isinstance(hf_value, tuple):
    shard_shapes = [tuple(hf_shapes[k]) for k in hf_value]
    return (sum(s[0] for s in shard_shapes), *shard_shapes[0][1:])
  hf_struct = jax.ShapeDtypeStruct(tuple(hf_shapes[hf_value]), jnp.float32)
  return tuple(jax.eval_shape(lambda x: tensor_handling.apply_hook_fns(x, mt_shape, hook), hf_struct).shape)


def _to_maxtext(hf_value, hf_tensors, hook, mt_shape) -> np.ndarray:
  """Builds one MaxText tensor from HF tensors the way `to_maxtext` does for this mapping."""
  if isinstance(hf_value, list):
    return np.stack([_to_maxtext(v, hf_tensors, hook, mt_shape[1:]) for v in hf_value], axis=0)
  if isinstance(hf_value, tuple):
    return tensor_handling.apply_hook_fns(tuple(hf_tensors[k] for k in hf_value), mt_shape, hook)
  return tensor_handling.apply_hook_fns(hf_tensors[hf_value], mt_shape, hook)


def _conversion_tables(model_name: str, hf_config: Any, config, saving_to_hf: bool = False):
  """Returns (param mapping, hooks, HF shapes) for `model_name`."""
  scan_layers = config.scan_layers
  return (
      param_mapping.PARAM_MAPPING[model_name](hf_config, config, scan_layers),
      param_mapping.HOOK_FNS[model_name](hf_config, config, scan_layers, saving_to_hf=saving_to_hf),
      hf_shape.HF_SHAPE[model_name](hf_config),
  )


@pytest.mark.cpu_only
class Qwen3_8FlashNextCheckpointConversionTest(parameterized.TestCase):
  """Validates the Qwen3.8-Flash-Next conversion tables against the m3 model."""

  @parameterized.product(model_name=(_FULL, _MINI), scan_layers=(False, True))
  def test_mapping_matches_model_params(self, model_name, scan_layers):
    """Mapping keys equal the model's params, and every HF target converts to the model's shape."""
    config = _get_config(model_name, scan_layers)
    self.assertEqual(config.param_scan_axis, 0)
    hf_config = hf_model_configs.HF_MODEL_CONFIGS[model_name].to_dict()
    mapping, hooks, hf_shapes = _conversion_tables(model_name, hf_config, config)
    mt_shapes = _model_param_shapes(config)

    self.assertEqual(sorted(set(mt_shapes) - set(mapping)), [], "model params missing from the mapping")
    self.assertEqual(sorted(set(mapping) - set(mt_shapes)), [], "mapping keys not in the model")
    self.assertEqual(sorted(set(hooks) - set(mapping)), [], "hooks for keys not in the mapping")
    self.assertEqual(mapping[_EMBED_KEY], "model.language_model.embed_tokens.weight")

    for mt_key, hf_value in mapping.items():
      missing = [k for k in _hf_targets(hf_value) if k not in hf_shapes]
      self.assertEqual(missing, [], f"{mt_key}: HF targets without a shape entry")
      if isinstance(hf_value, tuple):
        # `to_huggingface` looks up the composite shard key as a whole.
        self.assertIn(hf_value, hf_shapes)
      self.assertEqual(
          _converted_shape(hf_value, hf_shapes, hooks.get(mt_key), mt_shapes[mt_key]), mt_shapes[mt_key], mt_key
      )

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_mini_hf_weights_round_trip(self, scan_layers):
    """HF tensors -> MaxText (to_maxtext path) -> HF (`process_maxtext_param`) is lossless."""
    config = _get_config(_MINI, scan_layers)
    hf_config = hf_model_configs.HF_MODEL_CONFIGS[_MINI].to_dict()
    mapping, to_mt_hooks, hf_shapes = _conversion_tables(_MINI, hf_config, config)
    _, to_hf_hooks, _ = _conversion_tables(_MINI, hf_config, config, saving_to_hf=True)
    mt_shapes = _model_param_shapes(config)

    rng = np.random.default_rng(0)
    hf_keys = sorted({k for v in mapping.values() for k in _hf_targets(v)})
    hf_tensors = {k: rng.standard_normal(hf_shapes[k]).astype(np.float32) for k in hf_keys}

    exported = {}
    for mt_key, hf_value in mapping.items():
      mt_weight = _to_maxtext(hf_value, hf_tensors, to_mt_hooks.get(mt_key), mt_shapes[mt_key])
      self.assertEqual(mt_weight.shape, mt_shapes[mt_key], mt_key)
      exported.update(
          conversion_utils.process_maxtext_param(mt_key, jnp.asarray(mt_weight), mapping, to_hf_hooks, hf_shapes, config)
      )

    self.assertEqual(sorted(exported), hf_keys)
    for hf_key in hf_keys:
      np.testing.assert_array_equal(exported[hf_key], hf_tensors[hf_key], err_msg=hf_key)

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_accepts_pretrained_config(self, scan_layers):
    """`HF_MODEL_CONFIGS[...]` passed without `.to_dict()` gives the same tables as the dict."""
    config = _get_config(_FULL, scan_layers)
    hf_config_obj = hf_model_configs.HF_MODEL_CONFIGS[_FULL]
    mapping, hooks, hf_shapes = _conversion_tables(_FULL, hf_config_obj, config)
    dict_mapping, dict_hooks, dict_hf_shapes = _conversion_tables(_FULL, hf_config_obj.to_dict(), config)

    self.assertEqual(mapping, dict_mapping)
    self.assertEqual(set(hooks), set(dict_hooks))
    self.assertEqual(hf_shapes, dict_hf_shapes)
    self.assertEqual(mapping[_EMBED_KEY], "model.language_model.embed_tokens.weight")


if __name__ == "__main__":
  absltest.main()
