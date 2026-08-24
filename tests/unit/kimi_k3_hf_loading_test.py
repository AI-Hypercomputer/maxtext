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
import orbax.checkpoint as ocp
from maxtext.configs import pyconfig
from maxtext.layers.nnx_wrappers import ToLinen
from maxtext.models.models import Transformer


class KimiK3HFLoadingTest(unittest.TestCase):
  """Tests loading a converted Kimi K3 Orbax checkpoint and running a forward pass."""

  @classmethod
  def setUpClass(cls):
    cls.checkpoint_dir = os.environ.get(
        "KIMI_K3_CHECKPOINT_DIR",
        os.path.abspath("scratch/kimi_k3_orbax_checkpoint"),
    )
    if not os.path.exists(cls.checkpoint_dir):
      raise unittest.SkipTest(f"Checkpoint directory {cls.checkpoint_dir} does not exist. Run to_maxtext first.")


  def test_load_checkpoint_and_forward_pass(self):
    config = pyconfig.initialize([
        "kimi_k3_hf_loading_test.py",
        "src/maxtext/configs/models/kimi-k3-minimal.yml",
        "model_name=kimi-k3",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "skip_jax_distributed_system=True",
        "scan_layers=False",
    ])


    # Initialize model
    model = ToLinen(Transformer, args=(config, None, None))







    
    # Dummy inputs for 2-layer Kimi K3 (1 Dense + 1 MoE/MLA)
    batch_size = 1
    seq_len = 4
    inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Initialize abstract state with zero memory allocation via eval_shape
    rng = jax.random.PRNGKey(0)
    abstract_state = jax.eval_shape(model.init, rng, inputs, positions, segment_ids)

    # Load converted Orbax checkpoint
    mngr = ocp.CheckpointManager(self.checkpoint_dir)
    loaded_state = mngr.restore(0, args=ocp.args.Composite(items=ocp.args.PyTreeRestore(abstract_state)))
    print("Checkpoint restored successfully! Step:", mngr.latest_step())


    # Run forward pass with loaded state
    logits, _ = model.apply(loaded_state["items"], inputs, positions, segment_ids)
    print("Logits shape:", logits.shape, "dtype:", logits.dtype)

    # Assertions
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    self.assertFalse(jnp.isnan(logits).any(), "Logits contain NaNs!")
    self.assertFalse(jnp.isinf(logits).any(), "Logits contain Infs!")

    params = loaded_state["items"]["params"]
    self.assertIn("token_embedder", params)
    self.assertIn("decoder", params)
    self.assertIn("layers_0", params["decoder"])
    self.assertIn("layers_1", params["decoder"])
    print("FORWARD PASS SUCCESSFUL!")



if __name__ == "__main__":
  unittest.main()
