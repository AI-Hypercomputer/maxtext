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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils.tensor_handling import (
    _binary_chunked_stack,
    get_hf_loading_function,
)
from maxtext.common import checkpointing
import orbax.checkpoint as ocp_v0
import numpy as np
import optax
import safetensors.numpy


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
  @mock.patch("google.cloud.storage.Client")
  def test_build_gcs_cache_worker_cache_hit(self, mock_storage_client, mock_hf_fs):
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.exists.return_value = True

    load_dynamic.build_gcs_cache_worker("some_repo/model.safetensors", "gs://my-bucket/cache", "token")
    mock_blob.exists.assert_called_once()
    mock_blob.upload_from_file.assert_not_called()

  @mock.patch("huggingface_hub.HfFileSystem")
  @mock.patch("google.cloud.storage.Client")
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
  @mock.patch("google.cloud.storage.Client")
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

    class DummyAbstractState:

      def __init__(self):
        self.params = {}

    abstract_state = DummyAbstractState()

    path = "repo/meta-llama"
    dummy_ret_val, loaded_vars = load_dynamic.load_safetensors_dynamic_state(path, abstract_state, config)

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


if __name__ == "__main__":
  absltest.main()
