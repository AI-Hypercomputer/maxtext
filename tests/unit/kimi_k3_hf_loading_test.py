# Copyright 2026 Google LLC
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

"""Unit test for Kimi K3 HuggingFace checkpoint loading and forward pass in MaxText."""

import os
import unittest
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


@pytest.mark.tpu_only
class KimiK3HFLoadingTest(unittest.TestCase):

  """Tests loading a converted Kimi K3 Orbax checkpoint and running a forward pass."""

  @classmethod
  def setUpClass(cls):
    raw_ckpt_dir = os.environ.get(
        "KIMI_K3_CHECKPOINT_DIR",
        "scratch/kimi_k3_orbax_checkpoint",
    )
    if raw_ckpt_dir.startswith("gs://"):
      cls.checkpoint_dir = raw_ckpt_dir
    else:
      cls.checkpoint_dir = os.path.abspath(raw_ckpt_dir)

    cls.config_path = os.environ.get(
        "KIMI_K3_CONFIG",
        "src/maxtext/configs/models/kimi-k3-minimal.yml",
    )
    if not os.path.exists(cls.checkpoint_dir):
      raise unittest.SkipTest(f"Checkpoint directory {cls.checkpoint_dir} does not exist. Run to_maxtext first.")

  def test_load_checkpoint_and_forward_pass(self):
    ckpt_path = (
        self.checkpoint_dir
        if self.checkpoint_dir.endswith("items")
        else os.path.join(self.checkpoint_dir, "0", "items")
    )
    num_devices = jax.device_count()
    expert_parallelism = min(num_devices, 8) if num_devices > 0 else 1
    config = pyconfig.initialize([
        "kimi_k3_hf_loading_test.py",
        self.config_path,
        "model_name=kimi-k3",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "scan_layers=False",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "remat_policy=none",
        f"ici_expert_parallelism={expert_parallelism}",
        "ici_fsdp_parallelism=1",
        f"load_parameters_path={ckpt_path}",
    ])

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Use MaxText's official from_pretrained loader to instantiate and stream checkpoint to TPU
    print(f"Loading Kimi K3 checkpoint from {ckpt_path} onto TPU mesh...")
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    print("Model initialized and checkpoint restored successfully!")

    # Dummy inputs for 2-layer Kimi K3 (1 Dense + 1 MoE/MLA)
    batch_size = 1
    seq_len = 4
    inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Run JIT-compiled NNX forward pass
    @nnx.jit
    def run_forward(m, x, p, s):
      return m(
          decoder_input_tokens=x,
          decoder_positions=p,
          decoder_segment_ids=s,
          enable_dropout=False,
      )

    print("Running JIT-compiled forward pass on TPU...")
    logits = run_forward(model, inputs, positions, segment_ids)
    print("Logits shape:", logits.shape, "dtype:", logits.dtype)

    # Assertions
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    self.assertFalse(jnp.isnan(logits).any(), "Logits contain NaNs!")
    self.assertFalse(jnp.isinf(logits).any(), "Logits contain Infs!")
    print("FORWARD PASS SUCCESSFUL!")




if __name__ == "__main__":
  unittest.main()
