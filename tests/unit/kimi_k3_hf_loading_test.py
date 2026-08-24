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
import flax.linen as nn
import orbax.checkpoint as ocp
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding as sharding_utils




@pytest.mark.tpu_only
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


    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Initialize pure Linen model (zero host RAM footprint)
    quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(config, mesh, quant=quant, model_mode=models.MODEL_MODE_TRAIN)









    
    # Obtain abstract parameters and shardings without materializing weights in host RAM
    abstract_params = maxtext_utils.get_abstract_param(model, config)

    def to_concrete_sharded_leaf(leaf):
      if isinstance(leaf, nn.LogicallyPartitioned):
        shd = sharding_utils.create_sharding(mesh, leaf.names, rules=config.logical_axis_rules)
        val = leaf.value
        return jax.ShapeDtypeStruct(shape=val.shape, dtype=val.dtype, sharding=shd)
      if hasattr(leaf, "value"):
        leaf = leaf.value
      if isinstance(leaf, dict):
        if len(leaf) == 1 and "value" in leaf:
          return to_concrete_sharded_leaf(leaf["value"])
        return {k: to_concrete_sharded_leaf(v) for k, v in leaf.items()}
      if isinstance(leaf, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(shape=leaf.shape, dtype=leaf.dtype, sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
      return leaf

    sharded_target_params = to_concrete_sharded_leaf(abstract_params)

    # Load converted Orbax checkpoint
    mngr = ocp.CheckpointManager(self.checkpoint_dir)
    target_item = {"step": 0, "params": sharded_target_params, "opt_state": {}}
    loaded_state = mngr.restore(0, args=ocp.args.Composite(items=ocp.args.StandardRestore(target_item)))
    print("Checkpoint restored successfully! Step:", mngr.latest_step())

    # Dummy inputs for 2-layer Kimi K3 (1 Dense + 1 MoE/MLA)
    batch_size = 1
    seq_len = 4
    inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)


    params = loaded_state["items"]["params"]["params"]

    # Run forward pass with loaded state
    logits, _ = model.apply({"params": params}, inputs, positions, segment_ids)
    print("Logits shape:", logits.shape, "dtype:", logits.dtype)

    # Assertions
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    self.assertFalse(jnp.isnan(logits).any(), "Logits contain NaNs!")
    self.assertFalse(jnp.isinf(logits).any(), "Logits contain Infs!")

    self.assertIn("token_embedder", params)
    self.assertIn("decoder", params)
    self.assertIn("layers_0", params["decoder"])
    self.assertIn("layers_1", params["decoder"])
    print("FORWARD PASS SUCCESSFUL!")




if __name__ == "__main__":
  unittest.main()
