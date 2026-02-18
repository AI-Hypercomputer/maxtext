# Copyright 2023–2025 Google LLC
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

"""Input pipeline"""
import functools

import jax
from jax.sharding import PartitionSpec as P

from MaxText import pyconfig
from maxtext.input_pipeline.grain_data_processing import make_grain_train_iterator
from maxtext.input_pipeline.grain_data_processing import make_grain_eval_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_train_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_eval_iterator
from maxtext.input_pipeline.tfds_data_processing import make_tfds_train_iterator
from maxtext.input_pipeline.tfds_data_processing import make_tfds_eval_iterator
from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator
from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_eval_iterator
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.utils import max_logging


def get_process_loading_real_data(
    data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh
):
  """Get list of processes loading data from GCS when expansion_factor_real_data != -1"""
  sharding = jax.sharding.NamedSharding(mesh, P(*data_sharding))
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  batch_cutoff = global_batch_size_to_train_on
  process_loading_real_data = set()
  for p, indices in devices_indices_map.items():
    if not indices[0].stop or indices[0].stop <= batch_cutoff:
      process_loading_real_data.add(p.process_index)
  return list(process_loading_real_data)


def create_process_specific_iterator(config: pyconfig.HyperParameters, mesh, process_indices, input_iterator):
  """
  If the current process's index is among the `process_indices`, a real
  data iterator is created. Otherwise, a placeholder iterator is returned.
  """
  if jax.process_index() in process_indices:
    iterator_fn = functools.partial(input_iterator, config, mesh, process_indices)
    output_iterator = iterator_fn()
  else:
    output_iterator = PlaceHolderDataIterator(config, mesh)
  return output_iterator


def create_data_iterator(config: pyconfig.HyperParameters, mesh):
  """Create train and eval data iterators given configs and mesh."""

  # Return synthetic dataset if selected
  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None
  dataset_type_to_train_eval_iterator = {
      "tfds": (make_tfds_train_iterator, make_tfds_eval_iterator),
      "grain": (make_grain_train_iterator, make_grain_eval_iterator),
      "hf": (make_hf_train_iterator, make_hf_eval_iterator),
      "c4_mlperf": (make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator),
  }

  # Collect train and eval iterators
  if config.dataset_type in ["tfds", "grain", "hf", "c4_mlperf"]:
    if config.dataset_type == "c4_mlperf":
      assert config.packing, "c4_mlperf dataloader only works with packing. For padded version, use tfds dataloader"
    train_iterator, eval_iterator = dataset_type_to_train_eval_iterator[config.dataset_type]
  else:
    max_logging.log(
        f"WARNING: '{config.dataset_type}' is not a supported dataset type."
        "Using synthetic data. Please choose from 'tfds', 'grain', 'hf', or 'c4_mlperf'."
    )
    output_train_iterator, output_eval_iterator = SyntheticDataIterator(config, mesh), None
    return output_train_iterator, output_eval_iterator

  # Generate output train iterator
  process_indices_train = get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )
  output_train_iterator = create_process_specific_iterator(config, mesh, process_indices_train, train_iterator)
  if config.expansion_factor_real_data > 1:  # assert number of hosts loading real data
    assert len(process_indices_train) == jax.process_count() // config.expansion_factor_real_data

  # Generate output eval iterator
  output_eval_iterator = None
  if config.eval_interval > 0:
    process_indices_eval = get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load_eval,
        config.global_batch_size_to_eval_on,
        config.max_target_length,
        mesh,
    )

    if config.expansion_factor_real_data > 1:
      assert len(process_indices_eval) == jax.process_count() // config.expansion_factor_real_data
    output_eval_iterator = create_process_specific_iterator(config, mesh, process_indices_eval, eval_iterator)
  return output_train_iterator, output_eval_iterator
