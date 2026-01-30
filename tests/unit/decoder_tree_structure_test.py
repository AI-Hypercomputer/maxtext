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

"""Unit tests for verifying Linen and NNX decoder tree structure parity.

This module tests that all supported models have identical parameter tree
structures between their Linen and NNX implementations.
"""

import logging
import os
import sys
import unittest

# Suppress verbose logging from MaxText modules
logging.getLogger("MaxText").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)

# Suppress TF/XLA C++ warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import linen as nn
from flax import nnx
import pytest

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import models
from MaxText.layers import quantizations

# Get the actual MaxText package directory path by using the models module location
# MAXTEXT_PKG_DIR from globals.py returns a relative path, so we derive the absolute path
_MAXTEXT_PKG_DIR_ABS = os.path.dirname(os.path.abspath(models.__file__))
_MAXTEXT_PKG_DIR_ABS = os.path.dirname(_MAXTEXT_PKG_DIR_ABS)  # Go up from layers/ to MaxText/


# All supported models for tree structure verification
SUPPORTED_MODELS = [
    # LLaMA 2 family
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
    # LLaMA 3 family
    "llama3-8b",
    "llama3-70b",
    # LLaMA 3.1 family
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    # LLaMA 3.3 family
    "llama3.3-70b",
    # Mistral family
    "mistral-7b",
    # Mixtral family
    "mixtral-8x7b",
    "mixtral-8x22b",
    # DeepSeek 2 family
    "deepseek2-16b",
    "deepseek2-236b",
    # DeepSeek 3 family
    "deepseek3-671b",
    "deepseek3-671b-2dfsdp",
    "deepseek3-test",
    "deepseek3-tiny",
    # Kimi family
    "kimi-k2-1t",
    # Gemma family
    "gemma-7b",
    "gemma-2b",
    # Gemma 2 family
    "gemma2-2b",
    "gemma2-9b",
    "gemma2-27b",
    # Gemma 3 family
    "gemma3-4b",
    "gemma3-12b",
    "gemma3-27b",
    # Qwen 3 family
    "qwen3-0.6b",
    "qwen3-4b",
    "qwen3-4b-thinking-2507",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-480b-a35b",
    "qwen3-next-80b-a3b",
    "qwen3-omni-30b-a3b",
    # GPT-3 family
    "gpt3-175b",
    "gpt3-22b",
    "gpt3-6b",
    "gpt3-52k",
    # GPT-OSS family
    "gpt-oss-20b",
    "gpt-oss-120b",
    # LLaMA 4 family
    "llama4-17b-16e",
    "llama4-17b-128e",
]


def is_rng_path(path: str) -> bool:
  """Check if a path is RNG-related."""
  return "/rngs/" in path or path.startswith("rngs/")


def extract_linen_paths(vars_dict):
  """Extract paths from Linen variables dict using JAX tree utilities."""
  paths = []
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(vars_dict)

  for path_parts, leaf in leaves_with_paths:
    path_str = ""
    for part in path_parts:
      if hasattr(part, "key"):
        if path_str:
          path_str += "/" + str(part.key)
        else:
          path_str = str(part.key)
      elif hasattr(part, "idx"):
        path_str += f"[{part.idx}]"
      elif isinstance(part, str):
        if path_str:
          path_str += "/" + part
        else:
          path_str = part
      else:
        if path_str:
          path_str += "/" + str(part)
        else:
          path_str = str(part)

    if hasattr(leaf, "shape"):
      paths.append((path_str, leaf.shape, str(leaf.dtype)))
    else:
      paths.append((path_str, type(leaf).__name__, ""))

  return paths


def extract_nnx_paths(state):
  """Extract paths from NNX state using JAX tree utilities."""
  paths = []
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(state)

  for path_parts, leaf in leaves_with_paths:
    path_str = ""
    for part in path_parts:
      if hasattr(part, "key"):
        if path_str:
          path_str += "/" + str(part.key)
        else:
          path_str = str(part.key)
      elif hasattr(part, "idx"):
        path_str += f"[{part.idx}]"
      elif isinstance(part, str):
        if path_str:
          path_str += "/" + part
        else:
          path_str = part
      else:
        if path_str:
          path_str += "/" + str(part)
        else:
          path_str = str(part)

    if hasattr(leaf, "shape"):
      paths.append((path_str, leaf.shape, str(leaf.dtype)))
    elif hasattr(leaf, "value") and hasattr(leaf.value, "shape"):
      paths.append((path_str, leaf.value.shape, str(leaf.value.dtype)))
    else:
      paths.append((path_str, type(leaf).__name__, ""))

  return paths


def normalize_path(path: str, is_linen: bool = False) -> str:
  """Normalize a path for comparison.

  Linen format: params/params/decoder/layers/0/mlp/wi_0/kernel
  NNX format: decoder/layers/0/mlp/wi_0/kernel

  This removes the double 'params' prefix from Linen paths.
  """
  if is_linen and path.startswith("params/params/"):
    path = path[len("params/params/") :]
  elif is_linen and path.startswith("params/"):
    path = path[len("params/") :]
  return path


def transpose_nnx_shape_for_scanned_layers(path: str, nnx_shape: tuple) -> tuple:
  """Transpose NNX shape for scanned layers to match Linen's axis ordering.

  When scan_layers=True:
  - NNX with nnx.vmap puts the layer dimension at axis 0
  - Linen with nn.scan puts the layer dimension at axis 1

  For paths containing 'layers' with 2+ dimensions, we swap axes 0 and 1.
  Example: NNX (32, 4096) -> (4096, 32) to match Linen

  Args:
      path: The parameter path string
      nnx_shape: The NNX parameter shape tuple

  Returns:
      Transposed shape if applicable, otherwise original shape
  """
  # Only transpose for layer parameters with 2+ dimensions
  if "layers" in path and isinstance(nnx_shape, tuple) and len(nnx_shape) >= 2:
    # Swap axes 0 and 1: (0, 1, 2, ...) -> (1, 0, 2, ...)
    transposed = (nnx_shape[1], nnx_shape[0]) + nnx_shape[2:]
    return transposed
  return nnx_shape


def create_linen_model_abstract(cfg, mesh):
  """Create a Linen model and get its abstract parameter structure."""
  quant = quantizations.configure_quantization(cfg)
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  batch_size = cfg.global_batch_size_to_train_on
  seq_len = cfg.max_target_length

  rng = jax.random.PRNGKey(0)
  dummy_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  dummy_positions = jnp.stack([jnp.arange(seq_len, dtype=jnp.int32) for _ in range(batch_size)])
  dummy_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  def init_fn():
    return model.init(
        {"params": rng, "aqt": rng, "dropout": rng},
        dummy_tokens,
        dummy_positions,
        dummy_segment_ids,
        enable_dropout=False,
    )

  with mesh:
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      abstract_vars = jax.eval_shape(init_fn)

  return abstract_vars


def create_nnx_model_abstract(cfg, mesh):
  """Create an NNX model and get its abstract parameter structure."""
  quant = quantizations.configure_quantization(cfg)

  def create_model():
    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = nnx.Rngs(params=params_rng, dropout=dropout_rng)
    return models.Transformer(cfg, mesh, quant=quant, rngs=rngs, model_mode=MODEL_MODE_TRAIN)

  with mesh:
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      abstract_model = nnx.eval_shape(create_model)

  _, abstract_state = nnx.split(abstract_model)
  return abstract_state


def compare_tree_structures(linen_vars, nnx_state, hide_rngs: bool = True):
  """Compare the tree structures of Linen and NNX models.

  Args:
      linen_vars: Linen model variables (from model.init)
      nnx_state: NNX model state (from nnx.split)
      hide_rngs: If True, filter out RNG-related paths from comparison

  Returns:
      Tuple of (linen_only, nnx_only, shape_mismatches) where:
      - linen_only: Set of paths only in Linen
      - nnx_only: Set of paths only in NNX
      - shape_mismatches: List of (path, linen_shape, nnx_shape) tuples
  """
  linen_paths = extract_linen_paths(linen_vars)
  nnx_paths = extract_nnx_paths(nnx_state)

  if hide_rngs:
    linen_paths = [(p, s, d) for p, s, d in linen_paths if not is_rng_path(p)]
    nnx_paths = [(p, s, d) for p, s, d in nnx_paths if not is_rng_path(p)]

  # Normalize paths for comparison
  linen_normalized = {}
  for path, shape, dtype in linen_paths:
    norm_path = normalize_path(path, is_linen=True)
    linen_normalized[norm_path] = (path, shape, dtype)

  nnx_normalized = {}
  for path, shape, dtype in nnx_paths:
    norm_path = path
    nnx_normalized[norm_path] = (path, shape, dtype)

  # Find matches and mismatches
  linen_only = set(linen_normalized.keys()) - set(nnx_normalized.keys())
  nnx_only = set(nnx_normalized.keys()) - set(linen_normalized.keys())
  common = set(linen_normalized.keys()) & set(nnx_normalized.keys())

  # Check for shape mismatches in common paths
  shape_mismatches = []
  for path in common:
    linen_shape = linen_normalized[path][1]
    nnx_shape = nnx_normalized[path][1]
    # Apply transpose for scanned layers (NNX vmap puts layer dim at axis 0,
    # Linen scan puts it at axis 1)
    nnx_shape_normalized = transpose_nnx_shape_for_scanned_layers(path, nnx_shape)
    if linen_shape != nnx_shape_normalized:
      shape_mismatches.append((path, linen_shape, nnx_shape))

  return linen_only, nnx_only, shape_mismatches


class TestDecoderTreeStructure(unittest.TestCase):
  """Test that Linen and NNX decoders have identical tree structures."""

  def _check_model_config_exists(self, model_name: str) -> bool:
    """Check if a model config file exists."""
    model_config = os.path.join(_MAXTEXT_PKG_DIR_ABS, "configs", "models", f"{model_name}.yml")
    return os.path.exists(model_config)

  def _create_configs_and_mesh(self, model_name: str):
    """Create Linen and NNX configs and mesh for a model."""
    # Create config for Linen model (uses Linen Decoder)
    cfg_linen = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        model_name=model_name,
        per_device_batch_size=1.0,
        run_name="tree_compare_test",
        enable_checkpointing=False,
        max_target_length=32,
        attention="dot_product",
        pure_nnx_decoder=False,
    )

    # Create config for NNX model (uses NNX Decoder)
    cfg_nnx = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        model_name=model_name,
        per_device_batch_size=1.0,
        run_name="tree_compare_test",
        enable_checkpointing=False,
        max_target_length=32,
        attention="dot_product",
        pure_nnx_decoder=True,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg_linen)
    mesh = Mesh(devices_array, cfg_linen.mesh_axes)

    return cfg_linen, cfg_nnx, mesh

  def verify_tree_structure_match(self, model_name: str):
    """Verify that Linen and NNX tree structures match for a model."""
    if not self._check_model_config_exists(model_name):
      self.skipTest(f"Model config not found: {model_name}")

    cfg_linen, cfg_nnx, mesh = self._create_configs_and_mesh(model_name)

    # Create abstract models
    linen_vars = create_linen_model_abstract(cfg_linen, mesh)
    nnx_state = create_nnx_model_abstract(cfg_nnx, mesh)

    # Compare structures (hide RNG paths by default)
    linen_only, nnx_only, shape_mismatches = compare_tree_structures(linen_vars, nnx_state, hide_rngs=True)

    # Build error message if there are differences
    error_messages = []

    if linen_only:
      error_messages.append(f"Paths only in Linen ({len(linen_only)}):")
      for path in sorted(linen_only):
        error_messages.append(f"  {path}")

    if nnx_only:
      error_messages.append(f"Paths only in NNX ({len(nnx_only)}):")
      for path in sorted(nnx_only):
        error_messages.append(f"  {path}")

    if shape_mismatches:
      error_messages.append(f"Shape mismatches ({len(shape_mismatches)}):")
      for path, linen_shape, nnx_shape in shape_mismatches:
        error_messages.append(f"  {path}: Linen={linen_shape}, NNX={nnx_shape}")

    # Assert no differences
    self.assertEqual(
        len(linen_only),
        0,
        f"Model {model_name}: Found paths only in Linen\n" + "\n".join(error_messages),
    )
    self.assertEqual(
        len(nnx_only),
        0,
        f"Model {model_name}: Found paths only in NNX\n" + "\n".join(error_messages),
    )
    self.assertEqual(
        len(shape_mismatches),
        0,
        f"Model {model_name}: Found shape mismatches\n" + "\n".join(error_messages),
    )


# Generate parametrized test methods for each supported model
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_linen_nnx_tree_structure_match(model_name):
  """Test that Linen and NNX tree structures match for a model."""
  test_instance = TestDecoderTreeStructure()
  test_instance.verify_tree_structure_match(model_name)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
