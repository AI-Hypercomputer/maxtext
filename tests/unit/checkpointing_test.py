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

"""Unit tests for the checkpointing components."""

import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import orbax.checkpoint as ocp
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils.tensor_handling import (
    _binary_chunked_stack,
    get_hf_loading_function,
)
from maxtext.common import checkpointing
import numpy as np
import optax
import pytest
import safetensors.numpy

ocp_v0 = ocp

pytestmark = [pytest.mark.decoupled_target]


class BinaryChunkedStackTest(parameterized.TestCase):
  """Tests for the `_binary_chunked_stack` function."""

  def test_binary_chunked_stack(self):
    # Test stacking 1, 2, 3, 5, 8, and 12 tensors
    shapes = [(1,), (2, 3), (4, 5, 6)]
    for shape in shapes:
      for num_tensors in [1, 2, 3, 5, 8, 12]:
        key = jax.random.PRNGKey(0)
        tensors = [jax.random.normal(jax.random.fold_in(key, i), shape) for i in range(num_tensors)]

        # Test along various axes
        for axis in range(-len(shape) - 1, len(shape) + 1):
          expected = jnp.stack(tensors, axis=axis)
          actual = _binary_chunked_stack(tensors, axis)
          np.testing.assert_allclose(actual, expected)


class TensorHandlingTest(parameterized.TestCase):
  """Tests for the tensor handling loader functions."""

  def setUp(self):
    super().setUp()
    self.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("x",))
    self.sharding_rank4 = NamedSharding(self.mesh, PartitionSpec("x", None, None, None))
    self.sharding_rank3 = NamedSharding(self.mesh, PartitionSpec("x", None, None))

  def test_get_hf_loading_function_case_2_3_single_axis(self):
    # Tests Case 2/3 and lines 179 and gets loader for single axis stacked
    class MockConfig:

      def __init__(self):
        self.scan_layers = True
        self.param_scan_axis = 0

    config = MockConfig()

    target_leaf = jax.ShapeDtypeStruct(
        shape=(2, 4, 4),
        dtype=np.float32,
        sharding=self.sharding_rank3,
    )

    hf_keys = ["layer_0.weight", "layer_1.weight"]

    tensors = {
        "layer_0.weight": np.ones((4, 4), dtype=np.float32) * 10,
        "layer_1.weight": np.ones((4, 4), dtype=np.float32) * 20,
    }

    def getter_fn(key):
      return tensors[key]

    hook_fn = None

    loader_fn = get_hf_loading_function(hf_keys, getter_fn, hook_fn, target_leaf, config)

    result = loader_fn()

    self.assertEqual(result.shape, (2, 4, 4))
    np.testing.assert_allclose(result[0], tensors["layer_0.weight"])
    np.testing.assert_allclose(result[1], tensors["layer_1.weight"])

  def test_get_hf_loading_function_case_4_multi_axis(self):
    # Tests Case 4, line 190, 73, and gets loader for multi-axis stacked
    class MockConfig:

      def __init__(self):
        self.scan_layers = True
        self.param_scan_axis = 0

    config = MockConfig()

    target_leaf = jax.ShapeDtypeStruct(
        shape=(2, 2, 4, 4),
        dtype=np.float32,
        sharding=self.sharding_rank4,
    )

    hf_keys = [
        ["expert_0.layer_0.weight", "expert_0.layer_1.weight"],
        ["expert_1.layer_0.weight", "expert_1.layer_1.weight"],
    ]

    tensors = {
        "expert_0.layer_0.weight": np.ones((4, 4), dtype=np.float32) * 11,
        "expert_0.layer_1.weight": np.ones((4, 4), dtype=np.float32) * 12,
        "expert_1.layer_0.weight": np.ones((4, 4), dtype=np.float32) * 21,
        "expert_1.layer_1.weight": np.ones((4, 4), dtype=np.float32) * 22,
    }

    def getter_fn(key):
      return tensors[key]

    hook_fn = None

    loader_fn = get_hf_loading_function(hf_keys, getter_fn, hook_fn, target_leaf, config)

    result = loader_fn()

    self.assertEqual(result.shape, (2, 2, 4, 4))
    np.testing.assert_allclose(result[0, 0], tensors["expert_0.layer_0.weight"])
    np.testing.assert_allclose(result[0, 1], tensors["expert_0.layer_1.weight"])
    np.testing.assert_allclose(result[1, 0], tensors["expert_1.layer_0.weight"])
    np.testing.assert_allclose(result[1, 1], tensors["expert_1.layer_1.weight"])


class LoadDynamicTest(parameterized.TestCase):
  """Tests for cache downloads and dynamic loading of safetensors."""

  @mock.patch("huggingface_hub.HfFileSystem")
  @mock.patch.object(load_dynamic.storage, "Client")
  def test_build_gcs_cache_worker_cache_hit(self, mock_storage_client, mock_hf_fs):
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.exists.return_value = True

    load_dynamic.build_gcs_cache_worker("some_repo/model.safetensors", "gs://my-bucket/cache", "token")
    mock_blob.exists.assert_called_once()
    mock_blob.upload_from_file.assert_not_called()

  @mock.patch("huggingface_hub.HfFileSystem")
  @mock.patch.object(load_dynamic.storage, "Client")
  def test_build_gcs_cache_worker_cache_miss_success(self, mock_storage_client, mock_hf_fs):
    mock_fs_instance = mock_hf_fs.return_value
    mock_remote_file = mock.MagicMock()
    mock_fs_instance.open.return_value.__enter__.return_value = mock_remote_file

    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.exists.return_value = False

    load_dynamic.build_gcs_cache_worker("some_repo/model.safetensors", "gs://my-bucket/cache", "token")
    mock_blob.exists.assert_called_once()
    mock_blob.upload_from_file.assert_called_once_with(mock_remote_file, client=mock_client_instance)

  @mock.patch("huggingface_hub.HfFileSystem")
  @mock.patch.object(load_dynamic.storage, "Client")
  def test_build_gcs_cache_worker_retry_and_fail(self, mock_storage_client, mock_hf_fs):
    mock_fs_instance = mock_hf_fs.return_value
    mock_fs_instance.open.side_effect = Exception("Download failed")

    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.exists.return_value = False

    with mock.patch("time.sleep"):
      with self.assertRaises(Exception):
        load_dynamic.build_gcs_cache_worker("some_repo/model.safetensors", "gs://my-bucket/cache", "token")

  @mock.patch.object(load_dynamic.huggingface_hub, "HfFileSystem")
  @mock.patch.object(load_dynamic.storage, "Client")
  @mock.patch.object(load_dynamic, "load_sharded_hf_state")
  @mock.patch.object(load_dynamic, "transform_hf_state_to_mt_state")
  @mock.patch("jax.process_index", return_value=0)
  @mock.patch("jax.experimental.multihost_utils.sync_global_devices")
  def test_load_safetensors_dynamic_from_hf_hub(
      self,
      mock_sync,
      mock_process_index,
      mock_transform,
      mock_load_sharded,
      mock_storage_client,
      mock_hf_fs,
  ):
    mock_fs_instance = mock_hf_fs.return_value
    mock_fs_instance.glob.return_value = ["repo/meta-llama/model.safetensors"]

    mock_client_instance = mock_storage_client.return_value
    mock_blob = mock.MagicMock()
    mock_blob.name = "hf_cache/repo_meta-llama/model.safetensors"
    mock_client_instance.list_blobs.return_value = [mock_blob]

    mock_load_sharded.return_value = {}
    mock_transform.return_value = {"params": {}}

    class MockConfig:

      def __init__(self):
        self.model_name = "llama3.1-8b"
        self.base_output_directory = "gs://dummy-bucket"
        self.scan_layers = True
        self.param_scan_axis = 0
        self.hf_access_token = "dummy_token"

    config = MockConfig()

    path = "repo/meta-llama"
    dummy_ret_val, loaded_vars = load_dynamic.load_safetensors_dynamic_state(path, {}, config)

    self.assertIsNone(dummy_ret_val)
    self.assertEqual(loaded_vars, {"params": {}})
    mock_hf_fs.assert_called_once_with(token="dummy_token")
    mock_sync.assert_called_once_with("dynamic_hf_download_complete")


class SourceCheckpointLoadingTest(parameterized.TestCase):
  """Tests for the `load_state_if_possible` function with safetensors_dynamic layout."""

  def setUp(self):
    super().setUp()
    self.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("x",))
    self.sharding = NamedSharding(self.mesh, PartitionSpec())

    self.tmp_dir = epath.Path(self.create_tempdir().full_path)
    self.safetensors_ckpt_dir = self.tmp_dir / "hf_safetensors"
    self.safetensors_ckpt_dir.mkdir(parents=True, exist_ok=True)
    self.safetensors_ckpt_path = self.safetensors_ckpt_dir / "model.safetensors"

  def test_load_safetensors_dynamic_single_key(self):
    if os.getenv("JAX_PLATFORMS") == "proxy":
      self.skipTest("SafetensorsLayout is not supported on Pathways backend.")
    # Save a single key (embedding weight) to a safetensors file
    dummy_weight = np.arange(1024, dtype=np.float32).reshape(256, 4)
    safetensors.numpy.save_file({"model.embed_tokens.weight": dummy_weight}, str(self.safetensors_ckpt_path))

    # Setup mock config
    class MockConfig:

      def __init__(self):
        self.model_name = "llama3.1-8b"
        self.base_output_directory = "gs://dummy-bucket"
        self.scan_layers = True
        self.param_scan_axis = 0
        self.hf_access_token = None

    config = MockConfig()

    # Target abstract state matching llama2 embeddings shape
    target_state = {
        "params": {
            "token_embedder": {
                "embedding": jax.ShapeDtypeStruct(shape=(256, 4), dtype=np.float32, sharding=self.sharding)
            }
        }
    }
    abstract_state = train_state.TrainState.create(
        apply_fn=lambda x: x, params=target_state["params"], tx=optax.identity()
    )

    # Load using checkpointing framework dynamically
    loaded_data, loaded_vars = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path=str(self.safetensors_ckpt_dir),
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=1,
        abstract_unboxed_pre_state=abstract_state,
        source_checkpoint_layout="safetensors_dynamic",
        maxtext_config=config,
    )

    self.assertIsNone(loaded_data)
    self.assertIsNotNone(loaded_vars)

    # Assert values match
    loaded_weight = loaded_vars["params"]["token_embedder"]["embedding"]
    np.testing.assert_allclose(loaded_weight, dummy_weight)


class CheckpointMetadataTest(parameterized.TestCase):
  """Tests for loading checkpoint custom metadata."""

  @mock.patch.object(checkpointing.ocp, "checkpointables_metadata")
  def test_load_checkpoint_metadata(self, mock_metadata_fn):
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"lora": {"lora_rank": 8, "lora_alpha": 16.0}}
    mock_metadata_fn.return_value = mock_metadata

    loaded_metadata = checkpointing.load_checkpoint_metadata("dummy/path")
    self.assertEqual(loaded_metadata.get("lora"), {"lora_rank": 8, "lora_alpha": 16.0})
    mock_metadata_fn.assert_called_once()

  @mock.patch.object(checkpointing.ocp, "checkpointables_metadata")
  def test_load_checkpoint_metadata_strips_pytree_suffix(self, mock_metadata_fn):
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"scan_layers": True}
    mock_metadata_fn.return_value = mock_metadata

    loaded_metadata = checkpointing.load_checkpoint_metadata("gs://bucket/ckpt/0/items")
    self.assertEqual(loaded_metadata, {"scan_layers": True})
    (called_path,) = mock_metadata_fn.call_args.args
    self.assertEqual(called_path, epath.Path("gs://bucket/ckpt/0"))

  @mock.patch.object(checkpointing.ocp, "checkpointables_metadata")
  def test_load_checkpoint_metadata_handles_exceptions(self, mock_metadata_fn):
    mock_metadata_fn.side_effect = Exception("Checkpoint read error")

    loaded_metadata = checkpointing.load_checkpoint_metadata("corrupt/path")
    self.assertEqual(loaded_metadata, {})
    mock_metadata_fn.assert_called_once()


class LoadParamsLayoutCompatTest(parameterized.TestCase):
  """load_params_from_path must read every historical params-checkpoint layout."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = epath.Path(self.create_tempdir().full_path)
    self.params = {"dense": {"kernel": jnp.arange(4.0).reshape(2, 2)}}
    self.abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), self.params)

  def test_flat_v0_params_checkpoint(self):
    """v0 save_params_to_path wrote the pytree FLAT at the directory (no items/ subdir)."""
    path = self.tmp_dir / "quantized"
    ocp_v0.PyTreeCheckpointer().save(path, {"params": self.params})

    restored = checkpointing.load_params_from_path(str(path), self.abstract, 8)

    np.testing.assert_allclose(restored["dense"]["kernel"], self.params["dense"]["kernel"])

  def test_step_root_and_items_suffixed_paths(self):
    """v1-written step roots load both as the root and as the v0-documented .../items form."""
    root = self.tmp_dir / "0"
    checkpointing.save_params_to_path(str(root), self.params)

    for path in (str(root), str(root / "items"), str(root / "items") + "/"):
      restored = checkpointing.load_params_from_path(path, self.abstract, 8)
      np.testing.assert_allclose(restored["dense"]["kernel"], self.params["dense"]["kernel"])


class SaveCheckpointStepExistsTest(parameterized.TestCase):
  """v0 parity: saving a step that already exists is silently skipped, not fatal."""

  def test_existing_step_returns_false(self):
    manager = mock.Mock()
    manager.use_async = False
    manager.save_checkpointables.side_effect = FileExistsError("step 5 already exists")

    saved = checkpointing.save_checkpoint(manager, 5, {"w": 1})

    self.assertFalse(saved)

  def test_existing_step_async_returns_false(self):
    manager = mock.Mock()
    manager.use_async = True
    manager.save_checkpointables_async.side_effect = FileExistsError("step 5 already exists")

    saved = checkpointing.save_checkpoint(manager, 5, {"w": 1})

    self.assertFalse(saved)


class SaveCheckpointAsyncTest(parameterized.TestCase):
  """save_checkpoint must honor the manager's use_async flag (v0 async parity)."""

  def test_async_manager_uses_async_save(self):
    manager = mock.Mock()
    manager.use_async = True
    manager.save_checkpointables_async.return_value = mock.Mock()  # AsyncResponse

    saved = checkpointing.save_checkpoint(manager, 5, {"w": 1})

    self.assertTrue(saved)
    manager.save_checkpointables_async.assert_called_once()
    manager.save_checkpointables.assert_not_called()

  def test_async_manager_declined_save_returns_false(self):
    manager = mock.Mock()
    manager.use_async = True
    manager.save_checkpointables_async.return_value = None  # decision policy declined

    saved = checkpointing.save_checkpoint(manager, 5, {"w": 1})

    self.assertFalse(saved)

  def test_sync_manager_uses_blocking_save(self):
    manager = mock.Mock()
    manager.use_async = False
    manager.save_checkpointables.return_value = True

    saved = checkpointing.save_checkpoint(manager, 5, {"w": 1})

    self.assertTrue(saved)
    manager.save_checkpointables.assert_called_once()
    manager.save_checkpointables_async.assert_not_called()


class NormalizeCheckpointRootTest(parameterized.TestCase):
  """Tests for _normalize_checkpoint_root string manipulation function."""

  def test_normalize_checkpoint_root(self):
    normalize = checkpointing._normalize_checkpoint_root  # pylint: disable=protected-access

    cases = [
        ("/foo/bar/items", "/foo/bar"),
        ("/foo/bar/items/", "/foo/bar"),
        ("gs://bucket/0/items", "gs://bucket/0"),
        ("gs://bucket/0/items/", "gs://bucket/0"),
        ("hf://meta-llama/Meta-Llama-3-8B", "hf://meta-llama/Meta-Llama-3-8B"),
        ("hf://meta-llama/Meta-Llama-3-8B/", "hf://meta-llama/Meta-Llama-3-8B"),
        ("items", "."),
        ("items/", "."),
        ("an_item", "an_item"),
        ("", ""),
    ]
    for inp, expected in cases:
      with self.subTest(inp=inp, expected=expected):
        self.assertEqual(normalize(inp), expected)


class CheckpointErrorHandlerTest(parameterized.TestCase):
  """Tests for checkpoint error handling in maybe_save_checkpoint."""

  def setUp(self):
    super().setUp()
    self.mock_manager = mock.MagicMock()
    self.mock_manager.latest_step.return_value = None
    self.mock_manager.reached_preemption.return_value = False
    self.state = mock.Mock()

  def test_error_handler_raises_runtime_error(self):
    """Unexpected checkpointing errors should raise RuntimeError with original error chained."""
    config = mock.Mock()
    config.checkpoint_period = 1
    config.enable_diloco = False
    config.async_checkpointing = False
    config.enable_continuous_checkpointing = False
    config.enable_emergency_checkpoint = False
    config.enable_multi_tier_checkpointing = False
    config.local_checkpoint_period = 0
    config.enable_autocheckpoint = False
    config.elastic_enabled = False

    original_error = RuntimeError("GCS failure")
    with mock.patch.object(checkpointing, "save_checkpoint", side_effect=original_error):
      with self.assertRaises(RuntimeError) as cm:
        checkpointing.maybe_save_checkpoint(self.mock_manager, self.state, config, data_iterator=None, step=1)
      self.assertIn("Checkpointing failed. GCS failure", str(cm.exception))
      self.assertIs(cm.exception.__cause__, original_error)


class FP8DequantizeOnLoadTest(parameterized.TestCase):
  """Tests for dequantize-on-load parameter restoration."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = epath.Path(self.create_tempdir().full_path)

  def _save_and_restore_checkpoint(
      self,
      ckpt_name: str,
      ckpt_weights: dict[str, Any],
      target_abstract: Any,
      is_bare: bool = True,
  ) -> Any:
    """Helper to save weights to an Orbax checkpoint and restore them."""
    path = self.tmp_dir / ckpt_name
    tree = {"params": {"params": ckpt_weights}} if is_bare else {"params": ckpt_weights}
    ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
        path,
        tree,
        force=True,
    )
    restored = checkpointing.load_params_from_path(str(path), target_abstract, 8)
    return restored.to_pure_dict() if isinstance(restored, nnx.State) else restored

  def test_load_fp8_checkpoint_into_bf16_nnx_model(self):
    """Loading an FP8 checkpoint into a BF16 NNX model restores dequantized BF16 weights and drops scale."""

    class BF16Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = BF16Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    fp8_kernel = jnp.array([[0.25, 0.5], [1.0, 1.5], [0.125, 0.75], [2.0, 0.5]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array(2.0, dtype=jnp.float32)
    bias = jnp.array([0.1, -0.2], dtype=jnp.bfloat16)

    ckpt_weights = {
        "linear": {
            "kernel": fp8_kernel,
            "kernel_scale": scale,
            "bias": bias,
        }
    }

    pure = self._save_and_restore_checkpoint("fp8_ckpt", ckpt_weights, params_abstract)
    # Hand-derived expectation: fp8_kernel * scalar scale (2.0):
    # [[0.25, 0.5], [1.0, 1.5], [0.125, 0.75], [2.0, 0.5]] * 2.0
    # = [[0.5, 1.0], [2.0, 3.0], [0.25, 1.5], [4.0, 1.0]]
    expected_kernel = np.array(
        [[0.5, 1.0], [2.0, 3.0], [0.25, 1.5], [4.0, 1.0]],
        dtype=np.float32,
    )

    self.assertNotIn("kernel_scale", pure["linear"])
    self.assertEqual(pure["linear"]["kernel"].dtype, jnp.bfloat16)
    self.assertEqual(pure["linear"]["kernel"].shape, (4, 2))
    np.testing.assert_allclose(
        np.array(pure["linear"]["kernel"]),
        np.array(expected_kernel),
        rtol=1e-3,
        atol=1e-3,
    )
    self.assertIn("bias", pure["linear"])
    self.assertEqual(pure["linear"]["bias"].dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.array(pure["linear"]["bias"]), np.array(bias))

  def test_load_fp8_checkpoint_into_fp8_nnx_model(self):
    """Loading an FP8 checkpoint into an FP8 NNX model preserves FP8 kernel and scale."""

    class FP8Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.float8_e4m3fn)
        self.linear.kernel_scale = nnx.Param(jnp.ones((), dtype=jnp.float32))

    model = FP8Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    fp8_kernel = jnp.array([[0.25, 0.5], [1.0, 1.5], [0.125, 0.75], [2.0, 0.5]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array(3.5, dtype=jnp.float32)
    bias = jnp.zeros((2,), dtype=jnp.float8_e4m3fn)

    ckpt_weights = {
        "linear": {
            "kernel": fp8_kernel,
            "kernel_scale": scale,
            "bias": bias,
        }
    }

    pure = self._save_and_restore_checkpoint("fp8_to_fp8_ckpt", ckpt_weights, params_abstract)

    self.assertIn("kernel_scale", pure["linear"])
    self.assertEqual(pure["linear"]["kernel"].dtype, jnp.float8_e4m3fn)
    self.assertEqual(pure["linear"]["kernel_scale"].dtype, jnp.float32)
    np.testing.assert_array_equal(np.array(pure["linear"]["kernel"]), np.array(fp8_kernel))
    np.testing.assert_array_equal(np.array(pure["linear"]["kernel_scale"]), np.array(scale))

  def test_load_fp8_checkpoint_into_bf16_linen_dict(self):
    """Loading an FP8 checkpoint into a BF16 Linen parameter dict restores dequantized BF16 weights."""
    target_weights = {
        "params": {
            "linear": {
                "kernel": jax.ShapeDtypeStruct(shape=(4, 2), dtype=jnp.bfloat16),
                "bias": jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.bfloat16),
            }
        }
    }

    fp8_kernel = jnp.array([[0.5, 1.0], [0.25, 0.75], [1.5, 0.125], [0.5, 2.0]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array(0.5, dtype=jnp.float32)
    bias = jnp.zeros((2,), dtype=jnp.bfloat16)

    ckpt_weights = {
        "params": {
            "linear": {
                "kernel": fp8_kernel,
                "kernel_scale": scale,
                "bias": bias,
            }
        }
    }

    # Hand-derived expectation: fp8_kernel * scalar scale (0.5):
    # [[0.5, 1.0], [0.25, 0.75], [1.5, 0.125], [0.5, 2.0]] * 0.5
    # = [[0.25, 0.5], [0.125, 0.375], [0.75, 0.0625], [0.25, 1.0]]
    expected_kernel = np.array(
        [[0.25, 0.5], [0.125, 0.375], [0.75, 0.0625], [0.25, 1.0]],
        dtype=np.float32,
    )
    restored = self._save_and_restore_checkpoint("fp8_linen_ckpt", ckpt_weights, target_weights, is_bare=False)

    self.assertNotIsInstance(restored, nnx.State)
    self.assertIn("params", restored)
    self.assertNotIn("kernel_scale", restored["params"]["linear"])
    self.assertEqual(restored["params"]["linear"]["kernel"].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.array(restored["params"]["linear"]["kernel"]),
        np.array(expected_kernel),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_load_fp8_checkpoint_with_per_channel_scale_into_bf16_model(self):
    """Loading an FP8 checkpoint with per-channel scale restores dequantized BF16 weights and drops scale."""

    class BF16Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = BF16Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    fp8_kernel = jnp.array([[0.25, 0.5], [1.0, 1.5], [0.125, 0.75], [2.0, 3.0]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array([2.0, 0.5], dtype=jnp.float32)
    bias = jnp.array([0.1, -0.2], dtype=jnp.bfloat16)

    ckpt_weights = {
        "linear": {
            "kernel": fp8_kernel,
            "kernel_scale": scale,
            "bias": bias,
        }
    }

    pure = self._save_and_restore_checkpoint("fp8_per_channel_ckpt", ckpt_weights, params_abstract)
    # Hand-derived expectation: fp8_kernel (4, 2) * per-channel scale (2,) [2.0, 0.5]
    # Column 0 scaled by 2.0: [0.25, 1.0, 0.125, 2.0] * 2.0 = [0.5, 2.0, 0.25, 4.0]
    # Column 1 scaled by 0.5: [0.5, 1.5, 0.75, 3.0] * 0.5 = [0.25, 0.75, 0.375, 1.5]
    expected_kernel = np.array(
        [[0.5, 0.25], [2.0, 0.75], [0.25, 0.375], [4.0, 1.5]],
        dtype=np.float32,
    )

    self.assertNotIn("kernel_scale", pure["linear"])
    self.assertEqual(pure["linear"]["kernel"].dtype, jnp.bfloat16)
    self.assertEqual(pure["linear"]["kernel"].shape, (4, 2))
    np.testing.assert_allclose(
        np.array(pure["linear"]["kernel"]),
        np.array(expected_kernel),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_load_fp8_checkpoint_with_blockwise_scales_into_bf16_model(self):
    """Loading an FP8 checkpoint with 2D block scales restores dequantized BF16 weights and drops scale."""

    class BF16Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 4, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = BF16Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    fp8_kernel = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5], [0.25, 0.75, 1.25, 1.75], [2.0, 1.0, 0.5, 0.25]],
        dtype=jnp.float8_e4m3fn,
    )
    scale = jnp.array([[0.5, 1.0], [2.0, 0.25]], dtype=jnp.float32)

    ckpt_weights = {
        "linear": {
            "kernel": fp8_kernel,
            "kernel_scale": scale,
        }
    }

    pure = self._save_and_restore_checkpoint("fp8_blockwise_ckpt", ckpt_weights, params_abstract)
    # Hand-derived expectation: fp8_kernel (4, 4) * 2D block scale (2, 2) with 2x2 blocks:
    # Block (0, 0) scale 0.5: [[1.0, 2.0], [0.5, 1.5]] * 0.5 = [[0.5, 1.0], [0.25, 0.75]]
    # Block (0, 1) scale 1.0: [[3.0, 4.0], [2.5, 3.5]] * 1.0 = [[3.0, 4.0], [2.5, 3.5]]
    # Block (1, 0) scale 2.0: [[0.25, 0.75], [2.0, 1.0]] * 2.0 = [[0.5, 1.5], [4.0, 2.0]]
    # Block (1, 1) scale 0.25: [[1.25, 1.75], [0.5, 0.25]] * 0.25 = [[0.3125, 0.4375], [0.125, 0.0625]]
    expected_kernel = np.array(
        [
            [0.5, 1.0, 3.0, 4.0],
            [0.25, 0.75, 2.5, 3.5],
            [0.5, 1.5, 0.3125, 0.4375],
            [4.0, 2.0, 0.125, 0.0625],
        ],
        dtype=np.float32,
    )

    self.assertNotIn("kernel_scale", pure["linear"])
    self.assertEqual(pure["linear"]["kernel"].dtype, jnp.bfloat16)
    self.assertEqual(pure["linear"]["kernel"].shape, (4, 4))
    np.testing.assert_allclose(
        np.array(pure["linear"]["kernel"]),
        np.array(expected_kernel),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_load_fp8_checkpoint_with_moe_scales_into_bf16_model(self):
    """Loading an FP8 checkpoint with 3D block and per-expert MoE scales restores dequantized BF16 weights."""
    target_weights = {
        "params": {
            "moe": {
                "kernel": jax.ShapeDtypeStruct(shape=(2, 4, 4), dtype=jnp.bfloat16),
            }
        }
    }

    # Asymmetric FP8 kernel (2, 4, 4):
    # - Expert 0 differs from Expert 1
    # - Every 2x2 scale block differs from its neighbours
    # - Asymmetric under transpose of axes (1, 2)
    # - All values exactly representable in float8_e4m3fn
    fp8_kernel = jnp.array(
        [
            # Expert 0 (4, 4):
            [
                [0.5, 1.0, 1.5, 2.0],
                [2.5, 3.0, 3.5, 4.0],
                [1.0, 1.5, 2.0, 2.5],
                [3.0, 3.5, 4.0, 0.5],
            ],
            # Expert 1 (4, 4):
            [
                [1.5, 2.0, 2.5, 3.0],
                [0.5, 1.0, 1.5, 2.0],
                [3.5, 4.0, 0.5, 1.0],
                [2.0, 2.5, 3.0, 3.5],
            ],
        ],
        dtype=jnp.float8_e4m3fn,
    )

    # 3D block scale (2, 2, 2) with 2x2 blocks per expert, asymmetric under transpose
    scale_blockwise = jnp.array(
        [
            # Expert 0:
            [[0.5, 1.0], [2.0, 1.5]],
            # Expert 1:
            [[1.0, 2.0], [0.5, 4.0]],
        ],
        dtype=jnp.float32,
    )
    # Per-expert scale (2,)
    scale_per_expert = jnp.array([2.0, 0.5], dtype=jnp.float32)

    # Hand-derived expectation for 3D blockwise:
    # Expert 0:
    # block (0, 0) scale 0.5: [[0.5, 1.0], [2.5, 3.0]] * 0.5 = [[0.25, 0.5], [1.25, 1.5]]
    # block (0, 1) scale 1.0: [[1.5, 2.0], [3.5, 4.0]] * 1.0 = [[1.5, 2.0], [3.5, 4.0]]
    # block (1, 0) scale 2.0: [[1.0, 1.5], [3.0, 3.5]] * 2.0 = [[2.0, 3.0], [6.0, 7.0]]
    # block (1, 1) scale 1.5: [[2.0, 2.5], [4.0, 0.5]] * 1.5 = [[3.0, 3.75], [6.0, 0.75]]
    # Expert 1:
    # block (0, 0) scale 1.0: [[1.5, 2.0], [0.5, 1.0]] * 1.0 = [[1.5, 2.0], [0.5, 1.0]]
    # block (0, 1) scale 2.0: [[2.5, 3.0], [1.5, 2.0]] * 2.0 = [[5.0, 6.0], [3.0, 4.0]]
    # block (1, 0) scale 0.5: [[3.5, 4.0], [2.0, 2.5]] * 0.5 = [[1.75, 2.0], [1.0, 1.25]]
    # block (1, 1) scale 4.0: [[0.5, 1.0], [3.0, 3.5]] * 4.0 = [[2.0, 4.0], [12.0, 14.0]]
    expected_blockwise = np.array(
        [
            [
                [0.25, 0.5, 1.5, 2.0],
                [1.25, 1.5, 3.5, 4.0],
                [2.0, 3.0, 3.0, 3.75],
                [6.0, 7.0, 6.0, 0.75],
            ],
            [
                [1.5, 2.0, 5.0, 6.0],
                [0.5, 1.0, 3.0, 4.0],
                [1.75, 2.0, 2.0, 4.0],
                [1.0, 1.25, 12.0, 14.0],
            ],
        ],
        dtype=np.float32,
    )

    # Hand-derived expectation for per-expert:
    # Expert 0 scaled by 2.0:
    # [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0], [1.0, 1.5, 2.0, 2.5], [3.0, 3.5, 4.0, 0.5]] * 2.0
    # = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 1.0]]
    # Expert 1 scaled by 0.5:
    # [[1.5, 2.0, 2.5, 3.0], [0.5, 1.0, 1.5, 2.0], [3.5, 4.0, 0.5, 1.0], [2.0, 2.5, 3.0, 3.5]] * 0.5
    # = [[0.75, 1.0, 1.25, 1.5], [0.25, 0.5, 0.75, 1.0], [1.75, 2.0, 0.25, 0.5], [1.0, 1.25, 1.5, 1.75]]
    expected_per_expert = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 1.0],
            ],
            [
                [0.75, 1.0, 1.25, 1.5],
                [0.25, 0.5, 0.75, 1.0],
                [1.75, 2.0, 0.25, 0.5],
                [1.0, 1.25, 1.5, 1.75],
            ],
        ],
        dtype=np.float32,
    )

    test_cases = [
        ("blockwise", scale_blockwise, expected_blockwise),
        ("per_expert", scale_per_expert, expected_per_expert),
    ]

    for scale_name, scale, expected_kernel in test_cases:
      ckpt_weights = {
          "params": {
              "moe": {
                  "kernel": fp8_kernel,
                  "kernel_scale": scale,
              }
          }
      }
      restored = self._save_and_restore_checkpoint(
          f"fp8_moe_{scale_name}_ckpt", ckpt_weights, target_weights, is_bare=False
      )

      self.assertNotIn("kernel_scale", restored["params"]["moe"])
      self.assertEqual(restored["params"]["moe"]["kernel"].dtype, jnp.bfloat16)
      self.assertEqual(restored["params"]["moe"]["kernel"].shape, (2, 4, 4))
      np.testing.assert_allclose(
          np.array(restored["params"]["moe"]["kernel"]),
          expected_kernel,
          rtol=1e-3,
          atol=1e-3,
      )

  def test_load_fp8_checkpoint_shape_mismatch_raises(self):
    """Loading an FP8 checkpoint with incompatible weight shape raises a descriptive ValueError."""

    class BF16Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = BF16Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    ckpt_weights = {
        "linear": {
            "kernel": jnp.zeros((4, 3), dtype=jnp.float8_e4m3fn),
            "kernel_scale": jnp.array(1.0, dtype=jnp.float32),
        }
    }

    with self.assertRaisesRegex(ValueError, r"shape \(4, 3\) but the model expects \(4, 2\)"):
      self._save_and_restore_checkpoint("mismatched_ckpt", ckpt_weights, params_abstract)

  def test_augment_want_with_scales_flat_linen_checkpoint(self):
    """Verifies that flat Linen checkpoints without an inner 'params' key correctly traverse line 700."""
    want = {
        "params": {
            "linear": {
                "kernel": jax.ShapeDtypeStruct(shape=(4, 2), dtype=jnp.bfloat16),
            }
        }
    }
    # In a flat Linen checkpoint, stored.get("params") contains layer dicts directly (no inner "params" key).
    stored = {
        "params": {
            "linear": {
                "kernel": jax.ShapeDtypeStruct(shape=(4, 2), dtype=jnp.float8_e4m3fn),
                "kernel_scale": jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            }
        }
    }
    # This call directly exercises checkpointing.py:700:
    # `if not is_nnx and "params" in want and isinstance(meta_weights, dict) and "params" not in meta_weights:`
    augmented = checkpointing._augment_want_with_scales(  # pylint: disable=protected-access
        want, stored, is_nnx=False, restore_key="params"
    )
    self.assertIn("params", augmented)
    self.assertIn("kernel_scale", augmented["params"]["linear"])
    self.assertEqual(augmented["params"]["linear"]["kernel_scale"].shape, ())
    self.assertEqual(augmented["params"]["linear"]["kernel_scale"].dtype, jnp.float32)

  @mock.patch.object(checkpointing.ocp, "load")
  @mock.patch.object(checkpointing.ocp, "metadata")
  def test_load_fp8_flat_linen_checkpoint(self, mock_metadata_fn, mock_load_fn):
    """Loading an FP8 checkpoint with flat Linen weights exercises checkpointing.py:700 in restore flow."""
    want = {
        "params": {
            "linear": {
                "kernel": jax.ShapeDtypeStruct(shape=(4, 2), dtype=jnp.bfloat16),
            }
        }
    }
    fp8_kernel = jnp.array([[0.5, 1.0], [0.25, 0.75], [1.5, 0.125], [0.5, 2.0]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array(0.5, dtype=jnp.float32)

    # Stored metadata has flat Linen layout: stored["params"] has no inner "params" key
    mock_meta = mock.MagicMock()
    mock_meta.metadata = {
        "params": {
            "linear": {
                "kernel": jax.ShapeDtypeStruct(shape=(4, 2), dtype=jnp.float8_e4m3fn),
                "kernel_scale": jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            }
        }
    }
    mock_metadata_fn.return_value = mock_meta

    # ocp.load returns the loaded pytree matching the augmented target
    mock_load_fn.return_value = {
        "params": {
            "params": {
                "linear": {
                    "kernel": fp8_kernel,
                    "kernel_scale": scale,
                }
            }
        }
    }

    restored = checkpointing.load_params_from_path(str(self.tmp_dir / "flat_linen_ckpt"), want, 8)

    # Verify that line 700 augmented want["params"] with kernel_scale
    mock_load_fn.assert_called_once()
    called_target = mock_load_fn.call_args[0][1]
    self.assertIn("kernel_scale", called_target["params"]["params"]["linear"])

    # Hand-derived expectation: fp8_kernel * scalar scale (0.5):
    # [[0.5, 1.0], [0.25, 0.75], [1.5, 0.125], [0.5, 2.0]] * 0.5
    # = [[0.25, 0.5], [0.125, 0.375], [0.75, 0.0625], [0.25, 1.0]]
    expected_kernel = np.array(
        [[0.25, 0.5], [0.125, 0.375], [0.75, 0.0625], [0.25, 1.0]],
        dtype=np.float32,
    )

    self.assertIn("params", restored)
    self.assertNotIn("kernel_scale", restored["params"]["linear"])
    self.assertEqual(restored["params"]["linear"]["kernel"].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.array(restored["params"]["linear"]["kernel"]),
        expected_kernel,
        rtol=1e-3,
        atol=1e-3,
    )

  def test_scale_sharding_derivation(self):
    """Tests that _scale_sharding correctly derives shardings across multi-axis meshes."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("fsdp", "tensor"))

    # 1. Unsharded / None
    self.assertIsNone(checkpointing._scale_sharding(None, (16, 4)))  # pylint: disable=protected-access

    # 2. Divisible block scale: retains divisible axes
    weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", "tensor"))
    scale_sharding = checkpointing._scale_sharding(weight_sharding, (16, 4))  # pylint: disable=protected-access
    self.assertIsInstance(scale_sharding, jax.sharding.NamedSharding)
    self.assertEqual(scale_sharding.mesh, mesh)
    self.assertEqual(scale_sharding.spec, jax.sharding.PartitionSpec("fsdp", "tensor"))

    # 3. Scalar scale (rank 0): falls back to replicated P()
    scalar_sharding = checkpointing._scale_sharding(weight_sharding, ())  # pylint: disable=protected-access
    self.assertIsInstance(scalar_sharding, jax.sharding.NamedSharding)
    self.assertEqual(scalar_sharding.spec, jax.sharding.PartitionSpec())

    # 4. 1D per-channel scale (rank 1): falls back to replicated P()
    channel_sharding = checkpointing._scale_sharding(weight_sharding, (4,))  # pylint: disable=protected-access
    self.assertIsInstance(channel_sharding, jax.sharding.NamedSharding)
    self.assertEqual(channel_sharding.spec, jax.sharding.PartitionSpec())

    # 5. 4D MoE block scale with leading unpartitioned dims
    moe_weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None, "fsdp", "tensor"))
    moe_scale_sharding = checkpointing._scale_sharding(moe_weight_sharding, (16, 10, 16, 4))  # pylint: disable=protected-access
    self.assertIsInstance(moe_scale_sharding, jax.sharding.NamedSharding)
    self.assertEqual(moe_scale_sharding.spec, jax.sharding.PartitionSpec(None, None, "fsdp", "tensor"))


class TrainingEngineCheckpointManagerTest(parameterized.TestCase):
  """Tests for training_engine.checkpointing.CheckpointManager gs:// guard and save_optimizer_state."""

  def _mock_config(self):
    cfg = mock.MagicMock()
    cfg.checkpoint_storage_use_ocdbt = False
    cfg.checkpoint_storage_use_zarr3 = False
    cfg.checkpoint_storage_device_host_concurrent_gb = 8
    cfg.checkpoint_period = 1
    cfg.max_num_checkpoints_to_keep = 2
    cfg.async_checkpointing = True
    cfg.save_optimizer_state = True
    cfg.pathways_checkpointing_impl = "persistence"
    return cfg

  @mock.patch("maxtext.training_engine.checkpointing.ocp.CheckpointManager")
  @mock.patch("maxtext.training_engine.checkpointing.ocp.PyTreeCheckpointHandler")
  def test_pathways_persistence_rejects_non_gs_uri_on_first_and_second_init(
      self, mock_handler, mock_orbax_cm
  ):
    from maxtext.training_engine import checkpointing as engine_ckpt  # pylint: disable=import-outside-toplevel

    cfg = self._mock_config()
    rejected_constructions = 0
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      engine_ckpt._REGISTERED_IMPL = None
      with self.assertRaisesRegex(ValueError, r"checkpoint_dir must be a gs:// URI"):
        engine_ckpt.CheckpointManager("/tmp/local_dir", cfg)
      rejected_constructions += 1

      # Second construction when an impl is already registered:
      engine_ckpt._REGISTERED_IMPL = "persistence"
      with self.assertRaisesRegex(ValueError, r"checkpoint_dir must be a gs:// URI"):
        engine_ckpt.CheckpointManager("/tmp/local_dir", cfg)
      rejected_constructions += 1

    self.assertEqual(rejected_constructions, 2)
    mock_orbax_cm.assert_not_called()

  @mock.patch("maxtext.training_engine.checkpointing._maybe_register_pathways_persistence")
  @mock.patch("maxtext.training_engine.checkpointing.ocp.CheckpointManager")
  @mock.patch("maxtext.training_engine.checkpointing.ocp.PyTreeCheckpointHandler")
  def test_pathways_persistence_accepts_gs_uri_and_registers(
      self, mock_handler, mock_orbax_cm, mock_register
  ):
    from maxtext.training_engine import checkpointing as engine_ckpt  # pylint: disable=import-outside-toplevel

    cfg = self._mock_config()
    gcs_uri = "gs://yixuannwang-maxtext-dataset/trellis/0921"
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      mgr = engine_ckpt.CheckpointManager(gcs_uri, cfg)
    mock_register.assert_called_once_with("persistence")
    mock_orbax_cm.assert_called_once()
    self.assertEqual(mock_orbax_cm.call_args.kwargs["directory"], gcs_uri)
    self.assertIsNotNone(mgr._checkpoint_manager)


def _fake_array_handler(class_name: str, store: Any = object(), has_dispatcher: bool | None = None):
  """Builds a stand-in Orbax array handler for the mode assertions.

  Args:
    class_name: Drives the `persistence` handler-name check.
    store: Stand-in array metadata store.
    has_dispatcher: When not None, the handler exposes `has_dispatcher()` returning
      this value, which is what the `colocated_python` check reads.
  """
  attrs: dict[str, Any] = {"_array_metadata_store": store}
  if has_dispatcher is not None:
    attrs["has_dispatcher"] = lambda self: has_dispatcher
  return type(class_name, (), attrs)()


class _PathwaysRegistrationTestBase(parameterized.TestCase):
  """Isolates the process-global registration latch between tests."""

  def setUp(self):
    super().setUp()
    from maxtext.training_engine import checkpointing as engine_ckpt  # pylint: disable=import-outside-toplevel

    self.engine_ckpt = engine_ckpt
    self._saved_latch = engine_ckpt._REGISTERED_IMPL
    engine_ckpt._REGISTERED_IMPL = None

  def tearDown(self):
    self.engine_ckpt._REGISTERED_IMPL = self._saved_latch
    super().tearDown()


class PathwaysPersistenceFailLoudTest(_PathwaysRegistrationTestBase):
  """ENABLE_PATHWAYS_PERSISTENCE=1 is an explicit request: register it or crash.

  Silent degradation lands on the controller-side handler, which stages every shard
  through proxy-pod host RAM and OOMs at 397B scale ~25 minutes into a run.
  """

  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_registration_failure_raises_every_time(self, mock_register, mock_get_handler):
    """A failed attempt must not latch, or the retry would silently 'succeed'."""
    mock_register.side_effect = ImportError("no pathways backend")

    raises_observed = 0
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      for _ in range(2):
        with self.assertRaisesRegex(
            self.engine_ckpt.PathwaysCheckpointingUnavailableError,
            r"Refusing to fall back to controller-side host-staged checkpointing",
        ):
          self.engine_ckpt._maybe_register_pathways_persistence()
        raises_observed += 1

    self.assertEqual(raises_observed, 2)
    self.assertEqual(mock_register.call_count, 2)
    self.assertIsNone(self.engine_ckpt._REGISTERED_IMPL)
    mock_get_handler.assert_not_called()

  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_non_persistence_handler_raises_and_names_it(self, mock_register, mock_get_handler):
    """Registration 'succeeding' but yielding the controller-side handler is still a failure."""
    mock_get_handler.return_value = _fake_array_handler("ArrayHandler")

    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      with self.assertRaisesRegex(
          self.engine_ckpt.PathwaysCheckpointingUnavailableError,
          r"jax\.Array is handled by ArrayHandler",
      ):
        self.engine_ckpt._maybe_register_pathways_persistence()

    mock_register.assert_called_once()
    self.assertIsNone(self.engine_ckpt._REGISTERED_IMPL)

  @parameterized.named_parameters(
      ("cloud", "CloudPathwaysArrayHandler"),
      ("persistence", "PathwaysPersistenceArrayHandler"),
  )
  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_success_latches_and_registers_exactly_once(self, handler_name, mock_register, mock_get_handler):
    mock_get_handler.return_value = _fake_array_handler(handler_name)

    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      self.engine_ckpt._maybe_register_pathways_persistence()
      self.engine_ckpt._maybe_register_pathways_persistence()

    self.assertEqual(mock_register.call_count, 1)
    self.assertEqual(self.engine_ckpt._REGISTERED_IMPL, "persistence")

  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_env_unset_is_an_unchanged_noop(self, mock_register, mock_get_handler):
    """Back-compat: without the explicit request, behavior is untouched."""
    with mock.patch.dict(os.environ):
      os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)
      self.engine_ckpt._maybe_register_pathways_persistence()

    mock_register.assert_not_called()
    mock_get_handler.assert_not_called()
    self.assertIsNone(self.engine_ckpt._REGISTERED_IMPL)


class PathwaysCheckpointingImplSelectorTest(_PathwaysRegistrationTestBase):
  """`pathways_checkpointing_impl` selects the Orbax impl; the default stays persistence."""

  @parameterized.named_parameters(
      ("persistence", "persistence", "PERSISTENCE", "CloudPathwaysArrayHandler", None),
      ("colocated_python", "colocated_python", "COLOCATED_PYTHON", "ArrayHandler", True),
  )
  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_selector_maps_to_exact_orbax_impl(
      self, impl_name, expected_enum, handler_cls, has_dispatcher, mock_register, mock_get_handler
  ):
    import orbax.checkpoint.pathways as ocp_pathways  # pylint: disable=import-outside-toplevel

    mock_get_handler.return_value = _fake_array_handler(handler_cls, has_dispatcher=has_dispatcher)

    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      self.engine_ckpt._maybe_register_pathways_persistence(impl_name)

    mock_register.assert_called_once()
    kwargs = mock_register.call_args.kwargs
    # Exact enum member, not just "an impl was passed" -- a PERSISTENCE/COLOCATED_PYTHON
    # swap is otherwise completely invisible.
    self.assertIs(kwargs["checkpointing_impl"], getattr(ocp_pathways.CheckpointingImpl, expected_enum))
    # D3: the metadata store is required in BOTH modes for typed PRNG keys.
    self.assertIn("array_metadata_store", kwargs)
    self.assertFalse(kwargs["use_single_replica_array_handler"])
    self.assertEqual(self.engine_ckpt._REGISTERED_IMPL, impl_name)

  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_colocated_without_dispatcher_is_the_oom_path_and_raises(self, mock_register, mock_get_handler):
    """NO_DISPATCHER also yields an ArrayHandler -- only has_dispatcher() separates them."""
    mock_get_handler.return_value = _fake_array_handler("ArrayHandler", has_dispatcher=False)

    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      with self.assertRaisesRegex(
          self.engine_ckpt.PathwaysCheckpointingUnavailableError,
          r"not an ArrayHandler with a dispatcher attached",
      ):
        self.engine_ckpt._maybe_register_pathways_persistence("colocated_python")

    mock_register.assert_called_once()
    self.assertIsNone(self.engine_ckpt._REGISTERED_IMPL)

  @mock.patch("orbax.checkpoint._src.serialization.type_handler_registry.get_type_handler")
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_switching_impl_in_one_process_raises(self, mock_register, mock_get_handler):
    """Orbax registration is process-global, so a second, different impl cannot be honored."""
    mock_get_handler.return_value = _fake_array_handler("CloudPathwaysArrayHandler")

    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      self.engine_ckpt._maybe_register_pathways_persistence("persistence")
      with self.assertRaisesRegex(
          self.engine_ckpt.PathwaysCheckpointingUnavailableError,
          r"already registered 'persistence'; cannot also serve 'colocated_python'",
      ):
        self.engine_ckpt._maybe_register_pathways_persistence("colocated_python")

    self.assertEqual(mock_register.call_count, 1)
    self.assertEqual(self.engine_ckpt._REGISTERED_IMPL, "persistence")

  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_unknown_impl_raises_before_touching_orbax(self, mock_register):
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      with self.assertRaisesRegex(ValueError, r"unknown pathways_checkpointing_impl 'remote_python'"):
        self.engine_ckpt._maybe_register_pathways_persistence("remote_python")

    mock_register.assert_not_called()

  def test_base_yml_default_is_persistence(self):
    """Slice A: the shipped default must not change behavior for existing runs."""
    from maxtext.configs import types as config_types  # pylint: disable=import-outside-toplevel

    field = config_types.Checkpointing.model_fields["pathways_checkpointing_impl"]
    self.assertEqual(field.default, "persistence")


if __name__ == "__main__":
  absltest.main()

