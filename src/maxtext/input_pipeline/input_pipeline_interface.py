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

from maxtext.configs import pyconfig
from maxtext.input_pipeline.grain_data_processing import make_grain_train_iterator
from maxtext.input_pipeline.grain_data_processing import make_grain_eval_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_train_iterator
from maxtext.input_pipeline.hf_data_processing import make_hf_eval_iterator
from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_train_iterator
from maxtext.input_pipeline.olmo_grain_data_processing import make_olmo_grain_eval_iterator
from maxtext.input_pipeline.olmo_mm_data_processing import make_olmo_mm_train_iterator
from maxtext.input_pipeline.olmo_mm_data_processing import make_olmo_mm_eval_iterator
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.utils import max_logging
from maxtext.utils.sharding import remove_size_one_mesh_axis


def get_process_loading_real_data(
    data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh
):
  """Get deterministic sorted list of processes whose P(mesh.axis_names) rows overlap [0, global_batch_size_to_train_on)."""
  # MultiHostDataLoadIterator always forms its global array on P(mesh.axis_names) before
  # DataLoader reshards to data_sharding and train_step slices [:global_batch_size_to_train_on].
  # Selecting hosts on P(mesh.axis_names) prevents -1 PlaceHolderDataIterator rows from landing
  # inside [:global_batch_size_to_train_on] when data_sharding permutes mesh axes (e.g., cp-as-ep).
  del data_sharding
  sharding = jax.sharding.NamedSharding(mesh, P(mesh.axis_names))
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  batch_cutoff = global_batch_size_to_train_on
  process_loading_real_data = set()
  for p, indices in devices_indices_map.items():
    start = indices[0].start or 0
    if not indices[0].stop or start < batch_cutoff:
      process_loading_real_data.add(p.process_index)
  return sorted(process_loading_real_data)


def create_process_specific_iterator(
    config: pyconfig.HyperParameters, mesh, process_indices, input_iterator, is_training: bool = True
):
  """
  If the current process's index is among the `process_indices`, a real
  data iterator is created. Otherwise, a placeholder iterator is returned.
  """
  if jax.process_index() in process_indices:
    iterator_fn = functools.partial(input_iterator, config, mesh, process_indices)
    output_iterator = iterator_fn()
  else:
    output_iterator = PlaceHolderDataIterator(config, mesh, is_training=is_training)
  return output_iterator


def create_data_iterator(config: pyconfig.HyperParameters, mesh):
  """Create train and eval data iterators given configs and mesh."""

  # Return synthetic dataset if selected
  if config.dataset_type == "synthetic":
    eval_iterator = SyntheticDataIterator(config, mesh) if config.eval_interval > 0 else None
    return SyntheticDataIterator(config, mesh), eval_iterator
  dataset_type_to_train_eval_iterator = {
      "grain": (make_grain_train_iterator, make_grain_eval_iterator),
      "hf": (make_hf_train_iterator, make_hf_eval_iterator),
      "olmo_grain": (make_olmo_grain_train_iterator, make_olmo_grain_eval_iterator),
      "olmo_mm": (make_olmo_mm_train_iterator, make_olmo_mm_eval_iterator),
  }
  if config.dataset_type in ("tfds", "c4_mlperf"):
    from maxtext.input_pipeline.tfds_data_processing import make_tfds_train_iterator, make_tfds_eval_iterator  # pylint: disable=import-outside-toplevel
    from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator  # pylint: disable=import-outside-toplevel

    dataset_type_to_train_eval_iterator["tfds"] = (make_tfds_train_iterator, make_tfds_eval_iterator)
    dataset_type_to_train_eval_iterator["c4_mlperf"] = (make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator)

  # Collect train and eval iterators
  if config.dataset_type in ["tfds", "grain", "hf", "c4_mlperf", "olmo_grain", "olmo_mm"]:
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
    output_eval_iterator = create_process_specific_iterator(
        config, mesh, process_indices_eval, eval_iterator, is_training=False
    )
  return output_train_iterator, output_eval_iterator
