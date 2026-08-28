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

import asyncio
import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from maxtext.layers import linears
import orbax.checkpoint as ocp
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils.tensor_handling import (
    _binary_chunked_stack,
    get_hf_loading_function,
)
from maxtext.common import checkpointing
from maxtext.common import grain_utility
import numpy as np
import optax
import pytest
import safetensors.numpy

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
        enable_orbax_v1=True,
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

  @mock.patch.object(checkpointing.ocp, "StandardCheckpointer")
  def test_load_checkpoint_metadata(self, mock_checkpointer_cls):
    mock_ckptr = mock_checkpointer_cls.return_value
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"lora": {"lora_rank": 8, "lora_alpha": 16.0}}
    mock_ckptr.metadata.return_value = mock_metadata

    loaded_metadata = checkpointing.load_checkpoint_metadata("dummy/path")
    self.assertEqual(loaded_metadata.get("lora"), {"lora_rank": 8, "lora_alpha": 16.0})
    mock_ckptr.metadata.assert_called_once()

  @mock.patch.object(checkpointing.ocp, "StandardCheckpointer")
  def test_load_checkpoint_metadata_handles_exceptions(self, mock_checkpointer_cls):
    mock_ckptr = mock_checkpointer_cls.return_value
    mock_ckptr.metadata.side_effect = Exception("Checkpoint read error")

    loaded_metadata = checkpointing.load_checkpoint_metadata("corrupt/path")
    self.assertEqual(loaded_metadata, {})
    mock_ckptr.metadata.assert_called_once()


class GrainCheckpointableEquivalenceTest(parameterized.TestCase):
  """Tests to ensure GrainCheckpointable is equivalent to GrainCheckpointHandler."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = epath.Path(self.create_tempdir().full_path)

  def test_save_restore_equivalence_single_item(self):
    class FakeIterator:
      """A fake iterator for testing serialization."""

      def __init__(self, state=0):
        self.state = state

      def get_state(self):
        return json.dumps({"state": self.state}).encode()

      def set_state(self, state):
        self.state = json.loads(state.decode())["state"]

      def __next__(self):
        self.state += 1
        return self.state

    iterator_v0 = FakeIterator(10)
    iterator_v1 = FakeIterator(10)

    step = 100
    v0_path = self.tmp_dir / str(step) / "iter_v0"
    v1_path = self.tmp_dir / str(step) / "iter_v1"

    # v0 Save
    handler = grain_utility.GrainCheckpointHandler()
    v0_path.mkdir(parents=True, exist_ok=True)
    handler.save(v0_path, item=iterator_v0)

    # v1 Save
    wrapper = grain_utility.GrainCheckpointable(save_args=grain_utility.GrainCheckpointSave(item=iterator_v1))

    class MockDirectory:
      """Mock directory for testing checkpointing."""

      async def await_creation(self):
        v1_path.mkdir(parents=True, exist_ok=True)
        return v1_path

    commit_func = asyncio.run(wrapper.save(MockDirectory()))
    if commit_func:
      asyncio.run(commit_func)

    # Verify files are identical
    v0_file = v0_path / "process_0-of-1.json"
    v1_file = v1_path / "process_0-of-1.json"

    self.assertTrue(v0_file.exists())
    self.assertTrue(v1_file.exists())
    self.assertEqual(v0_file.read_text(), v1_file.read_text())

    # v0 Restore
    restored_iterator_v0 = FakeIterator(0)
    args_v0 = grain_utility.GrainCheckpointRestore(item=restored_iterator_v0)
    handler.restore(v0_path, args=args_v0)
    self.assertEqual(restored_iterator_v0.state, 10)

    # v1 Restore
    restored_iterator_v1 = FakeIterator(0)
    wrapper_restore = grain_utility.GrainCheckpointable(
        restore_args=grain_utility.GrainCheckpointRestore(item=restored_iterator_v1)
    )

    load_func = asyncio.run(wrapper_restore.load(v1_path))
    asyncio.run(load_func)
    self.assertEqual(restored_iterator_v1.state, 10)

  def test_save_restore_equivalence_list_item(self):
    class FakeIterator:
      """A fake iterator for testing serialization."""

      def __init__(self, state=0):
        self.state = state

      def get_state(self):
        return json.dumps({"state": self.state}).encode()

      def set_state(self, state):
        self.state = json.loads(state.decode())["state"]

    iterator_a = FakeIterator(10)
    iterator_b = FakeIterator(20)

    item_v0 = [(iterator_a, 0, 2), (iterator_b, 1, 2)]
    item_v1 = [(iterator_a, 0, 2), (iterator_b, 1, 2)]

    step = 100
    v0_path = self.tmp_dir / str(step) / "iter_v0"
    v1_path = self.tmp_dir / str(step) / "iter_v1"

    # v0 Save
    handler = grain_utility.GrainCheckpointHandler()
    v0_path.mkdir(parents=True, exist_ok=True)
    handler.save(v0_path, item=item_v0)

    # v1 Save
    wrapper = grain_utility.GrainCheckpointable(save_args=grain_utility.GrainCheckpointSave(item=item_v1))

    class MockDirectory:
      """Mock directory for testing checkpointing."""

      async def await_creation(self):
        v1_path.mkdir(parents=True, exist_ok=True)
        return v1_path

    commit_func = asyncio.run(wrapper.save(MockDirectory()))
    if commit_func:
      asyncio.run(commit_func)

    # Verify files are identical
    v0_file_0 = v0_path / "process_0-of-2.json"
    v1_file_0 = v1_path / "process_0-of-2.json"
    v0_file_1 = v0_path / "process_1-of-2.json"
    v1_file_1 = v1_path / "process_1-of-2.json"

    self.assertTrue(v0_file_0.exists())
    self.assertTrue(v1_file_0.exists())
    self.assertEqual(v0_file_0.read_text(), v1_file_0.read_text())

    self.assertTrue(v0_file_1.exists())
    self.assertTrue(v1_file_1.exists())
    self.assertEqual(v0_file_1.read_text(), v1_file_1.read_text())

    # v0 Restore
    iterators_restore_v0 = [FakeIterator(0), FakeIterator(0)]
    args_v0 = grain_utility.GrainCheckpointRestore(item=iterators_restore_v0, process_index=[0, 1], process_count=2)
    handler.restore(v0_path, args=args_v0)

    self.assertEqual(iterators_restore_v0[0].state, 10)
    self.assertEqual(iterators_restore_v0[1].state, 20)

    # v1 Restore
    iterators_restore_v1 = [FakeIterator(0), FakeIterator(0)]
    wrapper_restore = grain_utility.GrainCheckpointable(
        restore_args=grain_utility.GrainCheckpointRestore(
            item=iterators_restore_v1, process_index=[0, 1], process_count=2
        )
    )
    load_func = asyncio.run(wrapper_restore.load(v1_path))
    asyncio.run(load_func)
    self.assertEqual(iterators_restore_v1[0].state, 10)
    self.assertEqual(iterators_restore_v1[1].state, 20)


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
    config.pure_nnx = True
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
  """Tests for Mode C: Dequantize FP8 on parameter loading."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.tmp_dir.cleanup()
    super().tearDown()

  def test_load_fp8_checkpoint_into_bf16_nnx_model(self):
    """Loading an FP8 checkpoint into a BF16 NNX model restores dequantized BF16 weights and drops scale."""
    class BF16Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = BF16Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    fp8_kernel = jnp.array([[0.25, 0.5], [1.0, 1.5], [0.125, 0.75], [2.0, 0.5]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array(2.0, dtype=jnp.float32)
    bias = jnp.zeros((2,), dtype=jnp.bfloat16)

    ckpt_weights = {
        "linear": {
            "kernel": fp8_kernel,
            "kernel_scale": scale,
            "bias": bias,
        }
    }

    path = os.path.join(self.tmp_dir.name, "fp8_ckpt")
    ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
        epath.Path(path),
        {"params": {"params": ckpt_weights}},
        force=True,
    )

    expected_kernel = linears.dequantize_weight(fp8_kernel, scale, compute_dtype=jnp.bfloat16)

    restored = checkpointing.load_params_from_path(path, params_abstract, 8)
    self.assertIsInstance(restored, nnx.State)
    pure = restored.to_pure_dict()

    self.assertNotIn("kernel_scale", pure["linear"])
    self.assertEqual(pure["linear"]["kernel"].dtype, jnp.bfloat16)
    self.assertEqual(pure["linear"]["kernel"].shape, (4, 2))
    np.testing.assert_allclose(
        np.array(pure["linear"]["kernel"]),
        np.array(expected_kernel),
        rtol=1e-3,
        atol=1e-3,
    )

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

    path = os.path.join(self.tmp_dir.name, "fp8_to_fp8_ckpt")
    ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
        epath.Path(path),
        {"params": {"params": ckpt_weights}},
        force=True,
    )

    restored = checkpointing.load_params_from_path(path, params_abstract, 8)
    self.assertIsInstance(restored, nnx.State)
    pure = restored.to_pure_dict()

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

    path = os.path.join(self.tmp_dir.name, "fp8_linen_ckpt")
    ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
        epath.Path(path),
        {"params": ckpt_weights},
        force=True,
    )

    expected_kernel = linears.dequantize_weight(fp8_kernel, scale, compute_dtype=jnp.bfloat16)

    restored = checkpointing.load_params_from_path(path, target_weights, 8)
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
    """Loading an FP8 checkpoint with per-channel scale dequantizes properly."""
    class BF16Model(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    model = BF16Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)

    fp8_kernel = jnp.array([[0.25, 0.5], [1.0, 1.5], [0.125, 0.75], [2.0, 0.5]], dtype=jnp.float8_e4m3fn)
    scale = jnp.array([2.0, 4.0], dtype=jnp.float32)
    bias = jnp.zeros((2,), dtype=jnp.bfloat16)

    ckpt_weights = {
        "linear": {
            "kernel": fp8_kernel,
            "kernel_scale": scale,
            "bias": bias,
        }
    }

    path = os.path.join(self.tmp_dir.name, "fp8_channel_scale_ckpt")
    ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
        epath.Path(path),
        {"params": {"params": ckpt_weights}},
        force=True,
    )

    expected_kernel = linears.dequantize_weight(fp8_kernel, scale, compute_dtype=jnp.bfloat16)

    restored = checkpointing.load_params_from_path(path, params_abstract, 8)
    pure = restored.to_pure_dict()

    self.assertNotIn("kernel_scale", pure["linear"])
    self.assertEqual(pure["linear"]["kernel"].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.array(pure["linear"]["kernel"]),
        np.array(expected_kernel),
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
  absltest.main()
