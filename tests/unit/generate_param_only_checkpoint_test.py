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
"""Unit tests for generate_param_only_checkpoint.py."""

import types
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import DecoderBlockType
from maxtext.utils.generate_param_only_checkpoint import (
    _possibly_unroll_params,
    _read_train_checkpoint,
    _save_decode_checkpoint,
)


def _make_mesh(num_axes=1):
  """Create a single-device mesh for testing."""
  devices = jax.local_devices()[:1]
  axis_names = ("data",)[:num_axes]
  return Mesh(np.array(devices).reshape((1,) * num_axes), axis_names=axis_names)


def _make_scanned_state(layer_name, num_layers, hidden, mesh):
  """Build a minimal training_state and annotations for _possibly_unroll_params tests."""
  # Layers are scanned along axis 0: shape (num_layers, hidden)
  with mesh:
    layer_data = jax.device_put(
        jnp.ones((num_layers, hidden)),
        NamedSharding(mesh, PartitionSpec(None, None)),
    )
  # Annotation: one PartitionSpec per tensor leaf
  layer_annotation = PartitionSpec(None, None)

  state = types.SimpleNamespace()
  state.params = {"params": {"decoder": {layer_name: layer_data}}}

  annotations = types.SimpleNamespace()
  annotations.params = {"params": {"decoder": {layer_name: layer_annotation}}}

  return state, annotations


class TestPossiblyUnrollParamsDisabled(unittest.TestCase):
  """Tests for _possibly_unroll_params when unrolling is disabled."""

  def _make_config(self, scan_layers, force_unroll):
    return types.SimpleNamespace(scan_layers=scan_layers, force_unroll=force_unroll)

  def test_no_op_when_scan_layers_false(self):
    """Returns immediately without modifying state when scan_layers=False."""
    config = self._make_config(scan_layers=False, force_unroll=True)
    state = types.SimpleNamespace(params={"params": {"decoder": {"layers": "sentinel"}}})
    annotations = types.SimpleNamespace(params={"params": {"decoder": {"layers": "sentinel"}}})

    _possibly_unroll_params(config, state, annotations, mesh=None)

    # State is unmodified
    self.assertEqual(state.params["params"]["decoder"]["layers"], "sentinel")

  def test_no_op_when_force_unroll_false(self):
    """Returns immediately without modifying state when force_unroll=False."""
    config = self._make_config(scan_layers=True, force_unroll=False)
    state = types.SimpleNamespace(params={"params": {"decoder": {"layers": "sentinel"}}})
    annotations = types.SimpleNamespace(params={"params": {"decoder": {"layers": "sentinel"}}})

    _possibly_unroll_params(config, state, annotations, mesh=None)

    self.assertEqual(state.params["params"]["decoder"]["layers"], "sentinel")


class TestPossiblyUnrollParamsStandardLayers(unittest.TestCase):
  """Tests for _possibly_unroll_params standard (non-DeepSeek) layer unrolling."""

  def setUp(self):
    self.mesh = _make_mesh()
    self.num_layers = 2
    self.hidden = 4
    self.config = types.SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        decoder_block="default",  # not DeepSeek
        param_scan_axis=0,
        num_decoder_layers=self.num_layers,
    )

  def test_unrolls_layers_into_individual_keys(self):
    """Each scanned layer is extracted into a separate key (layers_0, layers_1, ...)."""
    state, annotations = _make_scanned_state("layers", self.num_layers, self.hidden, self.mesh)

    with self.mesh:
      _possibly_unroll_params(self.config, state, annotations, self.mesh)

    decoder = state.params["params"]["decoder"]
    # Original 'layers' key removed
    self.assertNotIn("layers", decoder)
    # Individual keys added
    for i in range(self.num_layers):
      self.assertIn(f"layers_{i}", decoder)
      layer = decoder[f"layers_{i}"]
      # Each unrolled layer has the scan axis removed: shape (hidden,)
      self.assertEqual(layer.shape, (self.hidden,))

  def test_annotations_updated_alongside_state(self):
    """Annotations dict is updated in sync with the state dict."""
    state, annotations = _make_scanned_state("layers", self.num_layers, self.hidden, self.mesh)

    with self.mesh:
      _possibly_unroll_params(self.config, state, annotations, self.mesh)

    ann_decoder = annotations.params["params"]["decoder"]
    self.assertNotIn("layers", ann_decoder)
    for i in range(self.num_layers):
      self.assertIn(f"layers_{i}", ann_decoder)
      # Annotation has scan axis removed: PartitionSpec(None,) instead of PartitionSpec(None, None)
      self.assertEqual(ann_decoder[f"layers_{i}"], PartitionSpec(None))

  def test_raises_value_error_on_missing_layer(self):
    """ValueError raised when the expected layer key is absent from state."""
    config = types.SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        decoder_block="default",
        param_scan_axis=0,
        num_decoder_layers=2,
    )
    state = types.SimpleNamespace(params={"params": {"decoder": {}}})  # no 'layers' key
    annotations = types.SimpleNamespace(params={"params": {"decoder": {}}})

    with self.assertRaises(ValueError, msg="Missing layers in training_state"):
      _possibly_unroll_params(config, state, annotations, self.mesh)


class TestPossiblyUnrollParamsDeepSeek(unittest.TestCase):
  """Tests for _possibly_unroll_params with DeepSeek decoder blocks."""

  def setUp(self):
    self.mesh = _make_mesh()
    self.hidden = 4
    self.first_dense = 1
    self.total_layers = 3
    self.config = types.SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        decoder_block=DecoderBlockType.DEEPSEEK,
        param_scan_axis=0,
        num_decoder_layers=self.total_layers,
        first_num_dense_layers=self.first_dense,
    )

  def _make_deepseek_state(self):
    """Create a DeepSeek-style state for testing."""
    num_moe = self.total_layers - self.first_dense
    with self.mesh:
      dense_data = jax.device_put(
          jnp.ones((self.first_dense, self.hidden)),
          NamedSharding(self.mesh, PartitionSpec(None, None)),
      )
      moe_data = jax.device_put(
          jnp.ones((num_moe, self.hidden)),
          NamedSharding(self.mesh, PartitionSpec(None, None)),
      )

    state = types.SimpleNamespace()
    state.params = {
        "params": {
            "decoder": {
                "dense_layers": dense_data,
                "moe_layers": moe_data,
            }
        }
    }
    annotations = types.SimpleNamespace()
    annotations.params = {
        "params": {
            "decoder": {
                "dense_layers": PartitionSpec(None, None),
                "moe_layers": PartitionSpec(None, None),
            }
        }
    }
    return state, annotations

  def test_unrolls_dense_and_moe_layers_separately(self):
    """DeepSeek blocks unroll dense_layers and moe_layers as distinct groups."""
    state, annotations = self._make_deepseek_state()

    with self.mesh:
      _possibly_unroll_params(self.config, state, annotations, self.mesh)

    decoder = state.params["params"]["decoder"]
    # Original group keys removed
    self.assertNotIn("dense_layers", decoder)
    self.assertNotIn("moe_layers", decoder)

    # Dense layers: 0..first_dense-1
    for i in range(self.first_dense):
      self.assertIn(f"dense_layers_{i}", decoder)
      self.assertEqual(decoder[f"dense_layers_{i}"].shape, (self.hidden,))

    # MoE layers: 0..num_moe-1
    num_moe = self.total_layers - self.first_dense
    for i in range(num_moe):
      self.assertIn(f"moe_layers_{i}", decoder)
      self.assertEqual(decoder[f"moe_layers_{i}"].shape, (self.hidden,))


class TestReadTrainCheckpointPureNNX(unittest.TestCase):
  """Tests for _read_train_checkpoint raising on unsupported pure_nnx path."""

  def test_raises_not_implemented_for_pure_nnx(self):
    """_read_train_checkpoint raises NotImplementedError when pure_nnx=True."""
    config = types.SimpleNamespace(pure_nnx=True)
    with patch("maxtext.utils.generate_param_only_checkpoint.quantizations.configure_quantization", return_value=None):
      with self.assertRaises(NotImplementedError):
        _read_train_checkpoint(config, checkpoint_manager=None, mesh=None)


class TestSaveDecodeCheckpoint(unittest.TestCase):
  """Tests for _save_decode_checkpoint."""

  def setUp(self):
    self.config = types.SimpleNamespace(checkpoint_dir="/tmp/ckpt")
    # A simple state with float32 params
    self.state = types.SimpleNamespace(
        params={"w": jnp.ones((4,), dtype=jnp.float32), "b": jnp.zeros((2,), dtype=jnp.float32)}
    )

  def test_params_cast_to_bfloat16(self):
    """The decode state written to the checkpoint manager contains bfloat16 params."""
    saved_states = []
    cm = MagicMock()
    cm.wait_until_finished.return_value = None

    def capture_save(manager, step, state, **kwargs):
      saved_states.append(state)
      return True

    with patch("maxtext.utils.generate_param_only_checkpoint.checkpointing.save_checkpoint", side_effect=capture_save):
      _save_decode_checkpoint(self.config, self.state, cm)

    self.assertEqual(len(saved_states), 1)
    saved = saved_states[0]
    # params tree should be bfloat16
    leaves = jax.tree.leaves(saved.params)
    for leaf in leaves:
      self.assertEqual(leaf.dtype, jnp.bfloat16)

  def test_checkpoint_manager_wait_always_called(self):
    """wait_until_finished is always called regardless of save_checkpoint outcome."""
    cm = MagicMock()
    cm.wait_until_finished.return_value = None

    with patch("maxtext.utils.generate_param_only_checkpoint.checkpointing.save_checkpoint", return_value=True):
      _save_decode_checkpoint(self.config, self.state, cm)

    cm.wait_until_finished.assert_called_once()

  def test_save_not_called_when_save_checkpoint_returns_false(self):
    """No logging side effect when save_checkpoint returns False, but wait is still called."""
    cm = MagicMock()
    cm.wait_until_finished.return_value = None

    with patch("maxtext.utils.generate_param_only_checkpoint.checkpointing.save_checkpoint", return_value=False):
      with patch("maxtext.utils.generate_param_only_checkpoint.max_logging.log") as mock_log:
        _save_decode_checkpoint(self.config, self.state, cm)

    # The "saved" log message should NOT have been emitted
    for call in mock_log.call_args_list:
      self.assertNotIn("saved an decode checkpoint", str(call))

    cm.wait_until_finished.assert_called_once()

  def test_decode_state_step_is_zero(self):
    """The decode state always has step=0 (no training steps)."""
    saved_states = []
    cm = MagicMock()
    cm.wait_until_finished.return_value = None

    def capture_save(manager, step, state, **kwargs):
      saved_states.append((step, state))
      return True

    with patch("maxtext.utils.generate_param_only_checkpoint.checkpointing.save_checkpoint", side_effect=capture_save):
      _save_decode_checkpoint(self.config, self.state, cm)

    step, _ = saved_states[0]
    self.assertEqual(step, 0)


if __name__ == "__main__":
  unittest.main()
