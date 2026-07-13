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
"""Test for tokamax gmm.

  pytest -v -m integration_test tests/integration/tokamax_test.py
"""

import os
import tempfile
from absl.testing import absltest, parameterized
from maxtext.trainers.pre_train import train
from tests.utils.test_helpers import get_test_config_path
import pytest

train_main = train.main
gettempdir = tempfile.gettempdir


from flax import nnx
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.common import common_types
from maxtext.common.common_types import Config, DType
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import maxtext_utils, sharding
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.layers import linears, moe, nnx_wrappers, decoders, embeddings, quantizations
from maxtext.layers.initializers import NdInitializer, nd_dense_init, variable_to_logically_partitioned
from maxtext.utils import max_logging


@pytest.mark.integration_test
class Train(parameterized.TestCase):
  """Smoke test for tokamax gmm in G3 only.

  Similar to `train_using_ragged_dot_smoke_train.py`
  """

  @parameterized.named_parameters(
      {
          "testcase_name": "megablox bf16",
          "quantization": "",
          "megablox": True,
          "use_tokamax_gmm": False,
          "use_gmm_v2_fwd": False,  # not matter
          "use_gmm_v2_dlhs": False,  # not matter
          "use_gmm_v2_drhs": False,  # not matter
      },
      {
          "testcase_name": "megablox fp8",
          "quantization": "fp8_full",
          "megablox": True,
          "use_tokamax_gmm": False,
          "use_gmm_v2_fwd": False,  # not matter
          "use_gmm_v2_dlhs": False,  # not matter
          "use_gmm_v2_drhs": False,  # not matter
      },
      {
          "testcase_name": "ragged_dot bf16",
          "quantization": "",
          "megablox": False,
          "use_tokamax_gmm": False,
          "use_gmm_v2_fwd": False,  # not matter
          "use_gmm_v2_dlhs": False,  # not matter
          "use_gmm_v2_drhs": False,  # not matter
      },
      # jax_ragged_dot_gmm
      # quantization: tiling = (tiling[0], k, tiling[2])
      # need more vmem
      {
          "testcase_name": "ragged_dot fp8",
          "quantization": "fp8_full",
          "megablox": False,
          "use_tokamax_gmm": False,
          "use_gmm_v2_fwd": False,  # not matter
          "use_gmm_v2_dlhs": False,  # not matter
          "use_gmm_v2_drhs": False,  # not matter
      },
      {
          "testcase_name": "tokamax v1 bf16",
          "quantization": "",
          "megablox": True,  # not matter
          "use_tokamax_gmm": True,
          "use_gmm_v2_fwd": False,
          "use_gmm_v2_dlhs": False,
          "use_gmm_v2_drhs": False,
      },
      {
          "testcase_name": "tokamax v1 fp8",
          "quantization": "fp8_full",
          "megablox": True,  # not matter
          "use_tokamax_gmm": True,
          "use_gmm_v2_fwd": False,
          "use_gmm_v2_dlhs": False,
          "use_gmm_v2_drhs": False,
      },
      {
          "testcase_name": "tokamax v2+v1+v2 bf16",
          "quantization": "",
          "megablox": True,  # not matter
          "use_tokamax_gmm": True,
          "use_gmm_v2_fwd": True,
          "use_gmm_v2_dlhs": False,
          "use_gmm_v2_drhs": True,
      },
      {
          "testcase_name": "tokamax v2+v1+v2 fp8",
          "quantization": "fp8_full",
          "megablox": True,  # not matter
          "use_tokamax_gmm": True,
          "use_gmm_v2_fwd": True,
          "use_gmm_v2_dlhs": False,
          "use_gmm_v2_drhs": True,
      },
      {
          "testcase_name": "tokamax v2+v2+v2 bf16",
          "quantization": "",
          "megablox": True,  # not matter
          "use_tokamax_gmm": True,
          "use_gmm_v2_fwd": True,
          "use_gmm_v2_dlhs": True,
          "use_gmm_v2_drhs": True,
      },
      {
          "testcase_name": "tokamax v2+v2+v2 fp8",
          "quantization": "fp8_full",
          "megablox": True,  # not matter
          "use_tokamax_gmm": True,
          "use_gmm_v2_fwd": True,
          "use_gmm_v2_dlhs": True,
          "use_gmm_v2_drhs": True,
      },
  )
  @pytest.mark.tpu_only
  def test_smoke_train(
      self,
      quantization: str,
      megablox: bool,
      use_tokamax_gmm: bool,
      use_gmm_v2_fwd: bool,
      use_gmm_v2_dlhs: bool,
      use_gmm_v2_drhs: bool,
  ):
    """Smoke train with small config."""
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={test_tmpdir}",
        "run_name=test_smoke_train",
        "base_emb_dim=1024",  # "base_emb_dim=256",
        "base_num_query_heads=1",
        "base_num_kv_heads=1",
        "base_mlp_dim=2048",  # "base_mlp_dim=256",
        "base_moe_mlp_dim=2048",  # "base_moe_mlp_dim=256",
        "base_num_decoder_layers=2",
        "head_dim=64",
        "decoder_block=deepseek",
        "attention_type=mla",
        "num_experts=32",  # "num_experts=2"
        "shared_experts=1",
        # tokamax gmm
        "sparse_matmul=True",
        f"megablox={megablox}",
        f"use_tokamax_gmm={use_tokamax_gmm}",
        f"use_gmm_v2_fwd={use_gmm_v2_fwd}",
        f"use_gmm_v2_dlhs={use_gmm_v2_dlhs}",
        f"use_gmm_v2_drhs={use_gmm_v2_drhs}",
        # tokamax splash
        "max_target_length=1024",
        "attention=flash",
        "use_tokamax_splash=False",
        # quantization
        f"quantization={quantization}",
        f"use_qwix_quantization={quantization != ""}",
        "weight_quantization_calibration_method=fixed,-224,224",
        "act_quantization_calibration_method=fixed,-224,224",
        # train
        "per_device_batch_size=8",
        "dataset_type=synthetic",
        "steps=20",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "enable_checkpoint_cloud_logger=False",
        "monitor_goodput=False",
        f"metrics_file={os.path.join(outputs_dir, 'metrics.json')}",
    ]
    train_main(args)


if __name__ == "__main__":
  absltest.main()
