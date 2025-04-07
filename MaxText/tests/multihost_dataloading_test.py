"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=missing-module-docstring, missing-function-docstring
import sys
import numpy as np
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec

import tensorflow as tf
import unittest
import pytest

import pyconfig
import multihost_dataloading


class MultihostDataloadingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    batch_size = 8
    self.config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        input_data_sharding_logical_axes=["batch"],
        ici_data_parallelism=-1,
        ici_fsdp_parallelism=1,
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
    )
    global_data_shape = PartitionSpec(batch_size, self.config.max_target_length)
    data_sharding = ("data",)
    mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), self.config.mesh_axes)
    data_axes = PartitionSpec(
        "data",
    )
    # creating 2 batches of data
    global_data = np.arange(np.prod(global_data_shape) * 2).reshape((batch_size * 2, self.config.max_target_length))

    dataset = tf.data.Dataset.from_tensor_slices(global_data)
    dataset = dataset.repeat()
    self.dataset = dataset.batch(batch_size)

  @pytest.mark.tpu_only
  def test_batch_sharded_data_pipeline(self):
    self.multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(self.dataset, self.mesh, self.config)
    first_batch = next(self.multihost_gen)
    sec_batch = next(self.multihost_gen)
    self.assertTrue(not np.array_equal(first_batch, sec_batch, equal_nan=True))

  def test_get_dcn_mesh_axes_prod(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="test",
        mesh_axes=["data", "fsdp", "expert"],
        logical_axis_rules=[["batch", ["data", "fsdp", "expert"]]],
        data_sharding=["data"],
        input_data_sharding_logical_axes=["batch"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
        dcn_data_parallelism=2,
        dcn_fsdp_parallelism=3,
        dcn_expert_parallelism=5,
        disalbe_key_validation=True,  # We don't have that many slices
    )
    default_num = 2
    prefix = "dcn"
    prod = multihost_dataloading._get_mesh_axes_prod(config, default_num, prefix, config.mesh_axes)
    assert prod == 30, f"{prod=} != 30"
    input_data_scale = multihost_dataloading._get_input_data_parallelisms(config, prefix, default_num)
    assert input_data_scale == (30, 1), f"{input_data_scale=} != (30,1)"

  def test_get_dcn_partial_mesh_axes_prod(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="test",
        mesh_axes=["data", "fsdp", "expert"],
        logical_axis_rules=[["batch", ["data", "fsdp", "expert"]]],
        data_sharding=["data"],
        input_data_sharding_logical_axes=["batch"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
        dcn_data_parallelism=2,
        dcn_fsdp_parallelism=3,
        dcn_tensor_parallelism=5,
        disalbe_key_validation=True,  # We don't have that many slices
    )
    default_num = 2
    prefix = "dcn"
    prod = multihost_dataloading._get_mesh_axes_prod(config, default_num, prefix, config.mesh_axes)
    assert prod == 6, f"{prod=} != 6"
    input_data_scale = multihost_dataloading._get_input_data_parallelisms(config, prefix, default_num)
    assert input_data_scale == (6, 1), f"{input_data_scale=} != (6,1)"

  def test_get_2d_dcn_mesh_axes_prod(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="test",
        mesh_axes=["data", "fsdp", "expert", "context", "sequence"],
        logical_axis_rules=[["logical_batch", ["data", "fsdp", "expert"]], ["logical_sequence", ["context", "sequence"]]],
        data_sharding=["data"],
        input_data_sharding_logical_axes=["logical_batch", "logical_sequence"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
        dcn_data_parallelism=2,
        dcn_fsdp_parallelism=3,
        dcn_expert_parallelism=5,
        dcn_context_parallelism=6,
        dcn_sequence_parallelism=7,
        disalbe_key_validation=True,  # We don't have that many slices
    )
    default_num = 2
    prefix = "dcn"
    input_data_scale = multihost_dataloading._get_input_data_parallelisms(config, prefix, default_num)
    assert input_data_scale == (30, 42), f"{input_data_scale=} != (30,42)"

  def test_get_default_ici_mesh_axes_prod(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="test",
        mesh_axes=["data", "fsdp", "expert"],
        logical_axis_rules=[["batch", ["data", "fsdp", "expert"]]],
        data_sharding=["data"],
        input_data_sharding_logical_axes=["batch"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
        ici_data_parallelism=-1,
        ici_fsdp_parallelism=3,
        ici_expert_parallelism=5,
    )
    default_num = 2
    prefix = "ici"
    prod = multihost_dataloading._get_mesh_axes_prod(config, default_num, prefix, config.mesh_axes)
    assert prod == 30
    input_data_scale = multihost_dataloading._get_input_data_parallelisms(config, prefix, default_num)
    assert input_data_scale == (30, 1), f"{input_data_scale=} != (30,1)"

  def test_get_2d_dcn_ici_scale(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        run_name="test",
        mesh_axes=["data", "fsdp", "expert", "context", "sequence"],
        logical_axis_rules=[["logical_batch", ["data", "fsdp", "expert"]], ["logical_sequence", ["context", "sequence"]]],
        data_sharding=["data"],
        input_data_sharding_logical_axes=["logical_batch", "logical_sequence"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
        dcn_data_parallelism=-1,
        dcn_fsdp_parallelism=3,
        dcn_expert_parallelism=5,
        dcn_context_parallelism=6,
        dcn_sequence_parallelism=7,
        ici_data_parallelism=2,
        ici_fsdp_parallelism=3,
        ici_expert_parallelism=5,
        ici_context_parallelism=6,
        ici_sequence_parallelism=-1,
        disalbe_key_validation=True,  # We don't have that many slices
    )
    input_data_dcn_parallelisms = multihost_dataloading._get_input_data_parallelisms(
        config, "dcn", default_mesh_parallelism=2
    )
    input_data_ici_parallelisms = multihost_dataloading._get_input_data_parallelisms(
        config, "ici", default_mesh_parallelism=7
    )
    assert input_data_dcn_parallelisms == (30, 42), f"{input_data_dcn_parallelisms=} != (30,42)"
    assert input_data_ici_parallelisms == (30, 42), f"{input_data_ici_parallelisms=} != (30,42)"

  def test_build_global_shape(self):
    local_shape = (3, 2)
    input_data_scale = (30, 42)
    global_shape = multihost_dataloading._build_global_shape(local_shape, input_data_scale)
    assert global_shape == (90, 84), f"{global_shape=} != (90, 84)"


if __name__ == "__main__":
  unittest.main()
