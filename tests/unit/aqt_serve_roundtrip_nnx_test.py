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

"""Round-trip test for the NNX serve-mode AQT checkpoint path.

Builds a small NNX model in CONVERT mode with int8 quantization, runs a forward
to populate `qrhs.frozen`, saves the serve-mode-shape state to a local orbax
checkpoint, then reloads via `from_pretrained(quant_mode_str="serve")` and
checks that the loaded QTensor leaves match what was saved.

This guards the chain of issues exercised by serve-mode reload (sharding helper
for QTensor, v[...] vs get_value() for composite values, Param-only filter
dropping aqt-typed leaves, Partitioned-unwrap for matching on-disk paths).
"""

import os
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from flax.core.meta import Partitioned
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils, maxtext_utils_nnx
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.utils.layerwise_quantization import LayerwiseQuantization


def _wrap_value(node):
  """Add `{"value": ...}` per-leaf wrap matching `_load_and_quantize_nnx` save format."""
  if isinstance(node, dict):
    return {k: _wrap_value(v) for k, v in node.items()}
  return {"value": node}


def _unbox(x):
  return x.value if isinstance(x, Partitioned) else x


def _walk_qrhs(state):
  """Yield (path_str, variable) pairs for every qrhs.frozen entry in an nnx.State."""
  for path, var in state.flat_state():
    keys = [str(getattr(k, "key", k)) for k in path]
    if "qrhs" in keys and "frozen" in keys:
      yield ".".join(keys), var


class ServeModeRoundTripTest(unittest.TestCase):
  """End-to-end save+reload of a serve-mode NNX AQT checkpoint."""

  def _init_cfg(self, ckpt_path, *, checkpoint_is_quantized):
    """Build a pyconfig for save or reload."""
    # Use base.yml + gpt3-52k. The decoupled test config strips
    # logical_axis_rules (e.g. "norm"), which the AQT serve-mode model
    # construction needs.
    base_yml = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
    args = [
        sys.argv[0],
        base_yml,
        "model_name=gpt3-52k",
        "pure_nnx=true",
        "enable_nnx=true",
        "pure_nnx_decoder=true",
        "max_target_length=64",
        "max_prefill_predict_length=16",
        "per_device_batch_size=1",
        "scan_layers=true",
        "quantization=int8",
        "checkpoint_storage_use_ocdbt=false",
        "checkpoint_storage_use_zarr3=false",
        "skip_jax_distributed_system=true",
    ]
    if checkpoint_is_quantized:
      args += [
          f"load_parameters_path={ckpt_path}",
          "checkpoint_is_quantized=true",
          "enable_checkpointing=true",  # required by config validator when load_parameters_path is set
      ]
    else:
      args += ["enable_checkpointing=false"]
    return pyconfig.initialize(args)

  def test_save_then_reload_preserves_qrhs_frozen(self):
    """Save a serve-mode-shape NNX checkpoint, then reload it and compare qvalue arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = os.path.join(tmpdir, "quantized_ckpt")

      # Step 1: build CONVERT-mode model + run forward to populate qrhs.frozen.
      cfg_save = self._init_cfg(ckpt_path, checkpoint_is_quantized=False)
      mesh = maxtext_utils.get_mesh_from_config(cfg_save)
      rngs = maxtext_utils_nnx.create_nnx_rngs(cfg_save)
      with nn_partitioning.axis_rules(cfg_save.logical_axis_rules):
        convert_model = model_creation_utils.from_config(
            cfg_save,
            mesh=mesh,
            rngs=rngs,
            model_mode="train",
            quant_mode_str="convert",
        )
      L = cfg_save.max_prefill_predict_length
      tokens = jnp.zeros((1, L), dtype=jnp.int32)
      pos = jnp.arange(L, dtype=jnp.int32)[None, :]
      seg = jnp.ones((1, L), dtype=jnp.int32)
      with nn_partitioning.axis_rules(cfg_save.logical_axis_rules):
        _ = convert_model(tokens, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode="train")

      # Step 2: capture the qrhs.frozen leaves we expect to round-trip, then save.
      convert_state = nnx.state(convert_model).to_pure_dict()
      serve_state = LayerwiseQuantization._strip_kernels_at_quantized_paths(convert_state)  # pylint: disable=protected-access
      saved_qrhs = {}
      for path, var in _walk_qrhs(nnx.state(convert_model)):
        qt = var.value if hasattr(var, "value") else var
        saved_qrhs[path] = _unbox(qt.qvalue)

      # Replicate arrays across the mesh; orbax rejects SingleDeviceSharding
      # once another test has initialized JAX-distributed state.
      replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      serve_state = jax.tree.map(
          lambda x: jax.device_put(x, replicated) if isinstance(x, jax.Array) else x,
          serve_state,
      )

      orbax_checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=False, use_zarr3=False)
      orbax_checkpointer.save(ckpt_path, _wrap_value(serve_state), force=True)
      self.assertGreater(len(saved_qrhs), 0, "Test config must produce at least one qrhs.frozen leaf")

      # Step 3: reload via from_pretrained in serve mode.
      cfg_load = self._init_cfg(ckpt_path, checkpoint_is_quantized=True)
      with nn_partitioning.axis_rules(cfg_load.logical_axis_rules):
        loaded_model = model_creation_utils.from_pretrained(
            cfg_load,
            mesh=mesh,
            model_mode="autoregressive",
            quant_mode_str="serve",
        )

      # Step 4: assert every saved qrhs.frozen leaf matches what was persisted.
      loaded_state = nnx.state(loaded_model)
      loaded_qrhs = dict(_walk_qrhs(loaded_state))
      self.assertEqual(set(saved_qrhs.keys()), set(loaded_qrhs.keys()))
      for path, saved_qv in saved_qrhs.items():
        var = loaded_qrhs[path]
        qt = var.value if hasattr(var, "value") else var
        loaded_qv = _unbox(qt.qvalue)
        self.assertEqual(loaded_qv.shape, saved_qv.shape, f"shape mismatch at {path}")
        self.assertEqual(loaded_qv.dtype, saved_qv.dtype, f"dtype mismatch at {path}")
        self.assertTrue(
            jnp.array_equal(loaded_qv.astype(jnp.int32), saved_qv.astype(jnp.int32)),
            f"qvalue not preserved at {path}",
        )


if __name__ == "__main__":
  unittest.main()
