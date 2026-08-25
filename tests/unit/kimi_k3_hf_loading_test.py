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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
import orbax.checkpoint as ocp
from maxtext.configs import pyconfig
from maxtext.models.models import Transformer
from maxtext.utils import maxtext_utils





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
    config = pyconfig.initialize([
        "kimi_k3_hf_loading_test.py",
        self.config_path,
        "model_name=kimi-k3",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "skip_jax_distributed_system=True",
        "scan_layers=False",
    ])

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Initialize pure NNX abstract model (zero host RAM footprint)
    abstract_model = nnx.eval_shape(
        lambda: Transformer(config, mesh, None, rngs=nnx.Rngs(0))
    )










    
    # Split only nnx.Param
    graphdef, params_state, _ = nnx.split(abstract_model, nnx.Param, ...)
    pure_dict = params_state.to_pure_dict()

    def add_sharding_to_pure_dict(d):
      if isinstance(d, dict):
        return {k: add_sharding_to_pure_dict(v) for k, v in d.items()}
      if isinstance(d, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(shape=d.shape, dtype=d.dtype, sharding=NamedSharding(mesh, P()))
      return d

    sharded_pure_dict = add_sharding_to_pure_dict(pure_dict)

    # Configure memory-bounded PyTreeCheckpointHandler (restore_concurrent_gb=4) to prevent host OOM
    handler = ocp.PyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=True,
        restore_concurrent_gb=4,
    )
    mngr = ocp.CheckpointManager(
        self.checkpoint_dir,
        item_handlers={"items": handler},
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    target_item = {"step": 0, "params": {"params": sharded_pure_dict}, "opt_state": {}}
    loaded_state = mngr.restore(0, args=ocp.args.Composite(items=ocp.args.StandardRestore(target_item)))
    print("Checkpoint restored successfully! Step:", mngr.latest_step())

    params = loaded_state["items"]["params"]["params"]
    del loaded_state
    del target_item
    del sharded_pure_dict
    import gc
    gc.collect()

    # Update NNX abstract model in place with restored parameters
    nnx.update(abstract_model, params_state.from_pure_dict(params))
    del params
    gc.collect()

    # Dummy inputs for 2-layer Kimi K3 (1 Dense + 1 MoE/MLA)
    batch_size = 1
    seq_len = 4
    inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Run JIT-compiled NNX forward pass on TPU
    @nnx.jit
    def run_forward(model, x, pos, seg):
      return model(x, pos, seg)

    logits, _ = run_forward(abstract_model, inputs, positions, segment_ids)
    print("Logits shape:", logits.shape, "dtype:", logits.dtype)

    # Assertions
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    self.assertFalse(jnp.isnan(logits).any(), "Logits contain NaNs!")
    self.assertFalse(jnp.isinf(logits).any(), "Logits contain Infs!")
    print("FORWARD PASS SUCCESSFUL!")




if __name__ == "__main__":
  unittest.main()
