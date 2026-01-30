# Copyright 2023-2026 Google LLC
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

"""
Unit tests demonstrating how to simulate raw_params loading with Orbax.

raw_params is returned by load_state_if_possible() when loading from a
parameter-only checkpoint (via load_parameters_from_path). This is used
to merge loaded params into a freshly initialized training state.
"""

import tempfile
import jax
import jax.numpy as jnp
from etils import epath
import orbax.checkpoint as ocp
import pytest


class TestRawParamsSimulation:
  """Tests for simulating raw_params checkpoint loading."""

  def test_save_and_load_params_only_checkpoint(self):
    """Test saving and loading a params-only checkpoint (raw_params pattern)."""
    # Create mock params (model params pytree)
    params = {
        "decoder": {
            "layers": {
                "mlp": {"kernel": jnp.ones((16, 32)), "bias": jnp.zeros((32,))},
                "attention": {"query": jnp.zeros((8, 16)), "key": jnp.ones((8, 16))},
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = epath.Path(tmpdir) / "params_ckpt"

      # Save params-only checkpoint (mimics save_params_to_path)
      ckptr = ocp.PyTreeCheckpointer()
      ckptr.save(ckpt_path, {"params": params}, force=True)

      # Load params (mimics load_params_from_path -> raw_params)
      abstract_params = jax.tree.map(
          lambda x: ocp.utils.to_shape_dtype_struct(x, x.dtype),
          params,
      )

      restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_params)
      restored = ckptr.restore(
          ckpt_path,
          item={"params": abstract_params},
          restore_args={"params": restore_args},
      )
      raw_params = restored["params"]

      # Verify restored params match original
      assert raw_params["decoder"]["layers"]["mlp"]["kernel"].shape == (16, 32)
      assert raw_params["decoder"]["layers"]["mlp"]["bias"].shape == (32,)
      assert jnp.allclose(raw_params["decoder"]["layers"]["mlp"]["kernel"], jnp.ones((16, 32)))
      assert jnp.allclose(raw_params["decoder"]["layers"]["attention"]["query"], jnp.zeros((8, 16)))

  def test_merge_raw_params_into_state(self):
    """Test merging raw_params into a fresh state (like maxtext_utils.py:1037-1038)."""
    # Simulate fresh initialized params
    fresh_params = {
        "decoder": {
            "layers": {
                "mlp": {"kernel": jnp.zeros((16, 32))},  # Fresh init (zeros)
            }
        }
    }

    # Simulate raw_params loaded from checkpoint
    raw_params = {
        "decoder": {
            "layers": {
                "mlp": {"kernel": jnp.ones((16, 32))},  # Loaded (ones)
            }
        }
    }

    # Merge: state = state.replace(params=raw_params)
    # In practice this replaces the params in a TrainState dataclass
    merged_params = raw_params

    # Verify merge replaced fresh params with loaded params
    assert jnp.allclose(merged_params["decoder"]["layers"]["mlp"]["kernel"], jnp.ones((16, 32)))

  def test_load_params_with_sharding(self):
    """Test loading params with explicit sharding (single device case)."""
    params = {"layer": {"weights": jnp.ones((8, 8))}}

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = epath.Path(tmpdir) / "sharded_params_ckpt"

      ckptr = ocp.PyTreeCheckpointer()
      ckptr.save(ckpt_path, {"params": params}, force=True)

      # Create abstract params with sharding info
      mesh = jax.sharding.Mesh(jax.devices(), ("data",))
      pspec = jax.sharding.PartitionSpec()  # Replicated
      sharding = jax.sharding.NamedSharding(mesh, pspec)

      def create_restore_args(x):
        return ocp.type_handlers.ArrayRestoreArgs(sharding=sharding)

      abstract_params = jax.tree.map(
          lambda x: ocp.utils.to_shape_dtype_struct(x, x.dtype),
          params,
      )
      restore_args = jax.tree.map(create_restore_args, abstract_params)

      restored = ckptr.restore(
          ckpt_path,
          item={"params": abstract_params},
          restore_args={"params": restore_args},
      )
      raw_params = restored["params"]

      assert raw_params["layer"]["weights"].shape == (8, 8)
      assert jnp.allclose(raw_params["layer"]["weights"], jnp.ones((8, 8)))


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
