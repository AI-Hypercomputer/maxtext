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

"""Direct MaxText-to-vLLM weight sync on real model trees (CPU-only, no weights allocated).

The other weight-sync tests build their trees by hand, so they do not notice when a model's real
parameter tree drifts from what the sync code expects. This test builds, for each model:

* the RL trainer tree (`post_train/rl.yml` with `scan_layers=True`), which is the sync source, and
* the tree `MaxTextForCausalLM` builds, from the adapter's own `generate_maxtext_config`
  (`inference/vllm.yml` plus KV-head replication, GMM_v2 MoE padding and prefused MoE weights),
  which is the sync target.

Both trees are abstract (`jax.eval_shape`), so even 70B models take seconds. The test then runs
`MaxTextVllmSampler.update_params` up to the Tunix transfer, which applies the scanned-weight
unroll and `validate_direct_sync_layer_coverage`. Finally it checks every rollout parameter
against its source with Tunix's own `_unstack_scanned_param` and `_align_per_axis`; the latter raised
the `ShapeMismatchError` behind #5025 and #5216.
"""

import re
import sys
from types import SimpleNamespace
from unittest import mock

from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import pytest

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig, types
from maxtext.integration.vllm import maxtext_vllm_rollout
from maxtext.integration.vllm.maxtext_vllm_adapter import adapter
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path
from tunix.generate import utils as tunix_utils
from tunix.generate.vllm_sampler import VllmSampler

pytestmark = [pytest.mark.post_training]

# Rollout tensor parallelism of the post-training E2E tests. It decides KV-head replication and
# whether the MoE MLP dim is padded for the GMM_v2 kernel.
_ROLLOUT_TP = 4

# `VllmSampler.update_params` calls Tunix's `transfer_state_directly` without a `scan_axis`, so the sync always
# unstacks scanned params on axis 1, whatever the trainer's `param_scan_axis` is.
_TUNIX_SCAN_AXIS = 1

# One model per sync path or tree layout runs on every PR. The rest run on the scheduled CI.
_SUPPORTED = [
    pytest.param("llama3.1-8b", id="llama3.1-8b"),  # stacked scan, sliced by Tunix
    pytest.param("mixtral-8x7b", id="mixtral-8x7b"),  # stacked scan with MoE
    pytest.param("qwen3-30b-a3b", id="qwen3-30b-a3b"),  # Qwen unroll; MoE wi/wo zero-padded to 1024
    pytest.param("qwen3.5-35b-a3b", id="qwen3.5-35b-a3b"),  # nested local/global block scan; KV heads repeated
    pytest.param("gemma3-4b", id="gemma3-4b"),  # Gemma block unroll
    pytest.param("llama3.1-70b", id="llama3.1-70b", marks=pytest.mark.scheduled_only),
    pytest.param("qwen3-4b", id="qwen3-4b", marks=pytest.mark.scheduled_only),
    pytest.param("qwen2.5-7b", id="qwen2.5-7b", marks=pytest.mark.scheduled_only),
    pytest.param("gemma2-2b", id="gemma2-2b", marks=pytest.mark.scheduled_only),
    pytest.param("mistral-7b", id="mistral-7b", marks=pytest.mark.scheduled_only),
]

# Scanned training with direct sync does not work for these models yet: the sync would leave every
# rollout layer at its random initialization, so `validate_direct_sync_layer_coverage` raises at the
# first sync. Move a model to `_SUPPORTED` once its test below starts failing.
_KNOWN_GAPS = [
    # b/584148367: `unroll_gemma_scanned_weights` does not handle the `scanned_blocks.global_layer` /
    # `scanned_blocks.local_layers` block scan, so nothing is unrolled.
    pytest.param("gemma4-26b", id="gemma4-26b"),
    # b/584148367: The `layers.layers_N` block scan is only unrolled for model names that contain "gemma".
    pytest.param("olmo3-7b", id="olmo3-7b", marks=pytest.mark.scheduled_only),
    # b/584148200: Neither unroll handles these block scans. The GPT-OSS E2E RL test uses the HF-mapping sync.
    pytest.param("gpt-oss-20b", id="gpt-oss-20b", marks=pytest.mark.scheduled_only),
    pytest.param("llama4-17b-16e", id="llama4-17b-16e", marks=pytest.mark.scheduled_only),
]

_LAYER_RE = re.compile(r"layers_\d+")


def _trainer_config(model_name):
  return pyconfig.initialize_pydantic(
      [sys.argv[0], get_test_config_path("post_train/rl.yml")],
      config_class=types.RLConfig,
      run_name="weight_sync_structure_test",
      model_name=model_name,
      tokenizer_path="unused",
      scan_layers=True,
      enable_checkpointing=False,
  )


def _rollout_config(trainer_config):
  """Builds the config MaxTextForCausalLM would use, through the adapter's own code path."""
  additional_config = maxtext_vllm_rollout.prepare_direct_sync_additional_config(
      {"maxtext_config": {"model_name": trainer_config.model_name, "log_config": False}},
      direct_maxtext_sync=True,
      num_experts=trainer_config.num_experts,
      tensor_parallel_size=_ROLLOUT_TP,
  )
  hf_config = SimpleNamespace(
      num_key_value_heads=trainer_config.num_kv_heads,
      moe_intermediate_size=trainer_config.base_moe_mlp_dim if trainer_config.num_experts > 1 else None,
  )
  vllm_config = SimpleNamespace(
      additional_config=additional_config,
      load_config=SimpleNamespace(load_format="auto"),
      sharding_config=SimpleNamespace(tp_size=_ROLLOUT_TP, expert_size=1, attn_dp_size=1, attn_dp_expert_size=1),
      model_config=SimpleNamespace(hf_config=hf_config),
      parallel_config=SimpleNamespace(enable_expert_parallel=False),
      cache_config=SimpleNamespace(mamba_cache_mode="none", block_size=16),
  )
  return adapter.generate_maxtext_config(vllm_config)


def _abstract_params(config, **kwargs):
  mesh = maxtext_utils.get_mesh_from_config(config)
  _, model = model_creation_utils.create_nnx_abstract_model(config, mesh, **kwargs)
  return nnx.state(model, nnx.Param).to_pure_dict()


def _synced_source(model_name, source, target):
  """Runs MaxTextVllmSampler.update_params up to the Tunix transfer and returns what it would sync."""
  with mock.patch.object(VllmSampler, "__init__", lambda self, tokenizer, config: None):
    sampler = maxtext_vllm_rollout.MaxTextVllmSampler(
        tokenizer=None,
        config=SimpleNamespace(engine_kwargs={}, model=""),
        direct_maxtext_sync=True,
        model_name=model_name,
    )

  # `MaxTextVllmSampler.update_params` returns what the Tunix transfer (`VllmSampler.update_params`) returns.
  def transfer(self, updated_weights, filter_types=None, **kwargs):
    del self, filter_types, kwargs
    return updated_weights

  with (
      mock.patch.object(VllmSampler, "update_params", transfer),
      mock.patch.object(VllmSampler, "transformer_state", new_callable=mock.PropertyMock, return_value=target),
  ):
    return jax.eval_shape(sampler.update_params, source)


def _source_shape(target_key, target_leaf, source_flat, scan_axis):
  """Shape Tunix's `transfer_state_directly` copies into `target_key`, or None if it has no source.

  Mirrors the lookup that `validate_direct_sync_layer_coverage` also mirrors: the same path, a
  prefused `wi` from split `wi_0`/`wi_1`, or layer N of a scanned `layers` container. Layer N is
  unstacked by Tunix's own `_unstack_scanned_param` with the explicit `scan_axis` that
  `transfer_state_directly` passes, so a source that Tunix cannot unstack raises here as well.
  """

  def lookup(key):
    if key in source_flat:
      return source_flat[key].shape
    if key[-1] == "wi" and key[:-1] + ("wi_0",) in source_flat and key[:-1] + ("wi_1",) in source_flat:
      gate = source_flat[key[:-1] + ("wi_0",)].shape
      return gate[:-1] + (2 * gate[-1],)
    return None

  shape = lookup(target_key)
  if shape is not None:
    return tuple(shape)
  idx = next((i for i, part in enumerate(target_key) if isinstance(part, str) and _LAYER_RE.fullmatch(part)), None)
  if idx is None:
    return None
  stacked = lookup(target_key[:idx] + ("layers",) + target_key[idx + 1 :])
  if stacked is None:
    return None
  unstack = tunix_utils._unstack_scanned_param  # pylint: disable=protected-access
  key_path = ".".join(map(str, target_key))
  layers = jax.eval_shape(
      lambda arr: unstack(arr, target_leaf, key_path, scan_axis=scan_axis),
      jax.ShapeDtypeStruct(stacked, target_leaf.dtype),
  )
  layer = int(target_key[idx].split("_")[1])
  return tuple(layers[layer].shape) if layer < len(layers) else None


def _check_alignable(source_shape, target_leaf, key_path):
  """Raises if Tunix's transfer cannot align a `source_shape` tensor to `target_leaf`.

  The CPU test mesh has a single device, so no sharding is passed; only the shapes are checked.
  """
  align = tunix_utils._align_per_axis  # pylint: disable=protected-access
  target_shape = tuple(target_leaf.shape)
  jax.eval_shape(
      lambda arr: align(arr, target_shape, None, key_path), jax.ShapeDtypeStruct(source_shape, target_leaf.dtype)
  )


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_name", _SUPPORTED)
def test_direct_sync_covers_and_aligns_every_rollout_param(model_name):
  trainer_config = _trainer_config(model_name)
  assert trainer_config.param_scan_axis == _TUNIX_SCAN_AXIS, (
      f"param_scan_axis={trainer_config.param_scan_axis}, but the Tunix sync always unstacks axis {_TUNIX_SCAN_AXIS}, "
      "so the rollout would get corrupted weights."
  )
  target = _abstract_params(_rollout_config(trainer_config), model_mode=MODEL_MODE_AUTOREGRESSIVE)
  source = _synced_source(model_name, _abstract_params(trainer_config), target)

  source_flat, target_flat = flatten_dict(source), flatten_dict(target)
  assert target_flat, "The rollout tree is empty: building its config or abstract model failed."
  unmatched, misaligned = [], []
  for key, leaf in target_flat.items():
    key_path = ".".join(map(str, key))
    try:
      shape = _source_shape(key, leaf, source_flat, _TUNIX_SCAN_AXIS)
      if shape is None:
        unmatched.append(key_path)
        continue
      _check_alignable(shape, leaf, key_path)
    except (tunix_utils.ShapeMismatchError, ValueError, TypeError, IndexError) as e:
      misaligned.append(f"{key_path}: {type(e).__name__}: {e}")

  assert not unmatched, f"{len(unmatched)} rollout params have no sync source and keep random weights: {unmatched[:5]}"
  assert not misaligned, f"{len(misaligned)} rollout params cannot be aligned by Tunix: {misaligned[:3]}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_name", _KNOWN_GAPS)
def test_known_direct_sync_gaps_still_fail(model_name):
  """Fails once the sync works for `model_name`; then move the model to `_SUPPORTED`."""
  trainer_config = _trainer_config(model_name)
  target = _abstract_params(_rollout_config(trainer_config), model_mode=MODEL_MODE_AUTOREGRESSIVE)
  with pytest.raises(ValueError, match="random initialization|failed to unroll"):
    _synced_source(model_name, _abstract_params(trainer_config), target)
