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

"""Integration tests for check_vma in megablox MoE training."""

import os

# Must be set before JAX imports so the TPU backend initializes correctly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("TPU_LIBRARY_PATH", os.path.join(_REPO_ROOT, "libtpu.so"))
os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "true")
os.environ.setdefault("ENABLE_PATHWAYS_PERSISTENCE", "1")

import pytest
from absl.testing import absltest
from tempfile import gettempdir

from maxtext.trainers.pre_train.train_compile import main as train_compile_main
from tests.utils.test_helpers import get_test_config_path

_BASE_ARGS = (
    None,
    get_test_config_path(),
    "compile_topology=tpu7x-16",
    "compile_topology_num_slices=1",
    "shard_mode=auto",
    "allow_split_physical_axes=true",
    "model_name=deepseek3-test",
    "num_experts=16",
    "override_model_config=True",
    "sparse_matmul=True",
    "megablox=True",
    "use_ring_of_experts=True",
    "use_random_routing=false",
    "per_device_batch_size=4",
    "max_target_length=128",
    "attention=flash",
    "dtype=bfloat16",
    "use_iota_embed=true",
    "enable_checkpointing=false",
    "async_checkpointing=false",
)


@pytest.mark.tpu_backend
class CheckVmaTest(absltest.TestCase):
  """Tests that megablox MoE compiles under different check_vma / shard_mode settings."""

  @pytest.mark.cpu_only
  def test_check_vma(self):
    """check_vma=True with fsdp and expert parallelism."""
    temp_dir = gettempdir()
    train_compile_main(
        _BASE_ARGS
        + (
            f"compiled_trainstep_file={os.path.join(temp_dir, 'check_vma.pickle')}",
            "check_vma=True",
            "ici_fsdp_parallelism=4",
            "ici_expert_parallelism=-1",
        )
    )

  @pytest.mark.cpu_only
  def test_check_vma_disabled(self):
    """check_vma=False (default) should also compile correctly."""
    temp_dir = gettempdir()
    train_compile_main(
        _BASE_ARGS
        + (
            f"compiled_trainstep_file={os.path.join(temp_dir, 'check_vma_disabled.pickle')}",
            "check_vma=False",
            "ici_fsdp_parallelism=4",
            "ici_expert_parallelism=-1",
        )
    )

  @pytest.mark.cpu_only
  def test_check_vma_pure_fsdp(self):
    """check_vma=True with only FSDP parallelism (no expert parallelism)."""
    temp_dir = gettempdir()
    train_compile_main(
        _BASE_ARGS
        + (
            f"compiled_trainstep_file={os.path.join(temp_dir, 'check_vma_pure_fsdp.pickle')}",
            "check_vma=True",
            "ici_fsdp_parallelism=-1",
            "ici_expert_parallelism=1",
        )
    )

  @pytest.mark.cpu_only
  def test_check_vma_pure_ep(self):
    """check_vma=True with only expert parallelism (no FSDP)."""
    temp_dir = gettempdir()
    train_compile_main(
        _BASE_ARGS
        + (
            f"compiled_trainstep_file={os.path.join(temp_dir, 'check_vma_pure_ep.pickle')}",
            "check_vma=True",
            "ici_fsdp_parallelism=1",
            "ici_expert_parallelism=-1",
        )
    )

  @pytest.mark.cpu_only
  def test_check_vma_error_extra_parallelism(self):
    """check_vma=True must raise ValueError when a non-EP/FSDP ICI axis is enabled."""
    with self.assertRaises(ValueError):
      train_compile_main(
          _BASE_ARGS
          + (
              f"compiled_trainstep_file={os.path.join(gettempdir(), 'check_vma_extra_par.pickle')}",
              "check_vma=True",
              "ici_fsdp_parallelism=4",
              "ici_expert_parallelism=-1",
              "ici_data_parallelism=2",
          )
      )

  @pytest.mark.cpu_only
  def test_check_vma_error_tokamax_gmm(self):
    """check_vma=True must raise ValueError when use_tokamax_gmm=True."""
    with self.assertRaises(ValueError):
      train_compile_main(
          _BASE_ARGS
          + (
              f"compiled_trainstep_file={os.path.join(gettempdir(), 'check_vma_tokamax.pickle')}",
              "check_vma=True",
              "ici_fsdp_parallelism=4",
              "ici_expert_parallelism=-1",
              "use_tokamax_gmm=True",
          )
      )

  @pytest.mark.cpu_only
  def test_explicit_shard_mode(self):
    """shard_mode=explicit should compile correctly without check_vma (check_vma not supported here)."""
    temp_dir = gettempdir()
    train_compile_main(
        _BASE_ARGS
        + (
            f"compiled_trainstep_file={os.path.join(temp_dir, 'explicit_shard.pickle')}",
            "shard_mode=explicit",
            "ici_fsdp_parallelism=4",
            "ici_expert_parallelism=-1",
        )
    )


if __name__ == "__main__":
  absltest.main()
