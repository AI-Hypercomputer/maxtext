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

"""Unit tests for _get_sharding_desc and maybe_shard_with_name in MaxText sharding module."""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.common.common_types import ShardMode
from maxtext.utils.sharding import _get_sharding_desc, maybe_shard_with_name


def _make_single_device_mesh():
  """Create a single-device mesh using one CPU device for testing."""
  devices = np.array(jax.devices()[:1]).reshape((1,))
  return Mesh(devices, axis_names=("data",))


class TestGetShardingDesc(unittest.TestCase):
  """Tests for the _get_sharding_desc function."""

  # The filename prefix expected from the mock
  expected_filename_prefix = os.path.basename(__file__)[:-3] + "/"

  # Each test method now accepts 'mock_pathlib' as an argument.
  # Pytest will run the mock_pathlib fixture before each of these tests.
  def test_direct_call_found(self):
    """Tests when inputs matches a local variable in the direct calling frame."""

    my_test_data = {"key": 123}
    result = _get_sharding_desc(my_test_data, extra_stack_level=0)
    assert result == self.expected_filename_prefix + "my_test_data"

  def test_direct_call_not_found_literal(self):
    """Tests when inputs is a literal, which is not a named local variable."""
    result = _get_sharding_desc({"key": 456}, extra_stack_level=0)
    assert result == "Unknown"

  def test_direct_call_not_found_expression(self):
    """Tests when inputs is a result of an expression, not a variable."""
    data_a = {"a": 1}
    data_b = {"b": 2}
    result = _get_sharding_desc(data_a | data_b, extra_stack_level=0)
    assert result == "Unknown"

  def test_nested_call_found_at_level_1(self):
    """Tests when inputs matches a variable one level up the stack."""
    outer_var = ["a", "b"]

    def inner_func():
      return _get_sharding_desc(outer_var, extra_stack_level=1)

    result = inner_func()
    assert result == self.expected_filename_prefix + "outer_var"

  def test_double_nested_call_found_at_level_2(self):
    """Tests when inputs matches a variable two levels up the stack."""
    deep_var = 12345

    def middle_func():
      def inner_func():
        return _get_sharding_desc(deep_var, extra_stack_level=2)

      return inner_func()

    result = middle_func()
    assert result == self.expected_filename_prefix + "deep_var"

  def test_too_deep_extra_stack_level(self):
    """Tests when extra_stack_level exceeds the actual stack depth."""
    some_inputs = {"c": 3}
    result = _get_sharding_desc(some_inputs, extra_stack_level=100)
    assert result == "Unknown"

  def test_multiple_matches_returns_first(self):
    """Tests that if multiple local vars point to inputs, the first is returned."""
    data = {"test": 1}
    data_alias = data
    result = _get_sharding_desc(data_alias, extra_stack_level=0)
    assert result == self.expected_filename_prefix + "data"

  def test_inputs_is_none_as_variable(self):
    """Tests when inputs is None and assigned to a variable."""
    none_val = None
    result = _get_sharding_desc(none_val, extra_stack_level=0)
    assert result == self.expected_filename_prefix + "none_val"

  def test_inputs_is_none_literal(self):
    """Tests when inputs is the None literal."""
    result = _get_sharding_desc(None, extra_stack_level=0)
    assert result == "Unknown"


class TestMaybeShardWithNameNoneInput(unittest.TestCase):
  """Tests for maybe_shard_with_name when inputs is None."""

  def test_returns_none_auto_mode(self):
    """When inputs is None in AUTO mode, should return None immediately."""
    result = maybe_shard_with_name(None, named_sharding=None, shard_mode=ShardMode.AUTO)
    self.assertIsNone(result)

  def test_returns_none_explicit_mode(self):
    """When inputs is None in EXPLICIT mode, should return None immediately."""
    result = maybe_shard_with_name(None, named_sharding=None, shard_mode=ShardMode.EXPLICIT)
    self.assertIsNone(result)

  def test_returns_none_with_debug_sharding_enabled(self):
    """When inputs is None with debug_sharding=True, should return None."""
    result = maybe_shard_with_name(None, named_sharding=None, shard_mode=ShardMode.AUTO, debug_sharding=True)
    self.assertIsNone(result)

  def test_none_input_does_not_call_with_sharding_constraint(self):
    """When inputs is None, should not call jax.lax.with_sharding_constraint."""
    with mock.patch("jax.lax.with_sharding_constraint") as mock_wsc:
      maybe_shard_with_name(None, named_sharding=None, shard_mode=ShardMode.AUTO)
      mock_wsc.assert_not_called()

  def test_none_input_does_not_call_reshard(self):
    """When inputs is None, should not call reshard even in EXPLICIT mode."""
    with mock.patch("maxtext.utils.sharding.reshard") as mock_reshard:
      maybe_shard_with_name(None, named_sharding=None, shard_mode=ShardMode.EXPLICIT)
      mock_reshard.assert_not_called()


class TestMaybeShardWithNameAutoMode(unittest.TestCase):
  """Tests for maybe_shard_with_name in AUTO shard mode."""

  def setUp(self):
    self.mesh = _make_single_device_mesh()
    self.named_sharding = NamedSharding(self.mesh, P())
    self.inputs = jnp.ones((4, 4))

  @mock.patch("jax.lax.with_sharding_constraint")
  def test_auto_mode_calls_with_sharding_constraint(self, mock_wsc):
    """AUTO mode should delegate to jax.lax.with_sharding_constraint."""
    mock_wsc.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO)
    mock_wsc.assert_called_once_with(self.inputs, self.named_sharding)

  @mock.patch("jax.lax.with_sharding_constraint")
  def test_auto_mode_returns_wsc_result(self, mock_wsc):
    """AUTO mode should return the value produced by with_sharding_constraint."""
    sentinel = object()
    mock_wsc.return_value = sentinel
    result = maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO)
    self.assertIs(result, sentinel)

  @mock.patch("jax.lax.with_sharding_constraint")
  def test_auto_mode_passes_inputs_unchanged(self, mock_wsc):
    """AUTO mode should forward the original inputs to with_sharding_constraint."""
    mock_wsc.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO)
    call_inputs, _ = mock_wsc.call_args[0]
    self.assertIs(call_inputs, self.inputs)

  @mock.patch("jax.lax.with_sharding_constraint")
  def test_auto_mode_passes_sharding_unchanged(self, mock_wsc):
    """AUTO mode should forward the original named_sharding to with_sharding_constraint."""
    mock_wsc.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO)
    _, call_sharding = mock_wsc.call_args[0]
    self.assertIs(call_sharding, self.named_sharding)

  @mock.patch("maxtext.utils.sharding.reshard")
  @mock.patch("jax.lax.with_sharding_constraint")
  def test_auto_mode_does_not_call_reshard(self, mock_wsc, mock_reshard):
    """AUTO mode should NOT call reshard."""
    mock_wsc.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO)
    mock_reshard.assert_not_called()


class TestMaybeShardWithNameExplicitMode(unittest.TestCase):
  """Tests for maybe_shard_with_name in EXPLICIT shard mode."""

  def setUp(self):
    self.mesh = _make_single_device_mesh()
    self.named_sharding = NamedSharding(self.mesh, P())
    self.inputs = jnp.ones((4, 4))

  @mock.patch("maxtext.utils.sharding.reshard")
  def test_explicit_mode_calls_reshard(self, mock_reshard):
    """EXPLICIT mode should delegate to reshard."""
    mock_reshard.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.EXPLICIT)
    mock_reshard.assert_called_once_with(self.inputs, self.named_sharding)

  @mock.patch("maxtext.utils.sharding.reshard")
  def test_explicit_mode_returns_reshard_result(self, mock_reshard):
    """EXPLICIT mode should return the value produced by reshard."""
    sentinel = object()
    mock_reshard.return_value = sentinel
    result = maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.EXPLICIT)
    self.assertIs(result, sentinel)

  @mock.patch("maxtext.utils.sharding.reshard")
  def test_explicit_mode_passes_inputs_unchanged(self, mock_reshard):
    """EXPLICIT mode should forward the original inputs to reshard."""
    mock_reshard.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.EXPLICIT)
    call_inputs, _ = mock_reshard.call_args[0]
    self.assertIs(call_inputs, self.inputs)

  @mock.patch("maxtext.utils.sharding.reshard")
  def test_explicit_mode_passes_sharding_unchanged(self, mock_reshard):
    """EXPLICIT mode should forward the original named_sharding to reshard."""
    mock_reshard.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.EXPLICIT)
    _, call_sharding = mock_reshard.call_args[0]
    self.assertIs(call_sharding, self.named_sharding)

  @mock.patch("maxtext.utils.sharding.reshard")
  @mock.patch("jax.lax.with_sharding_constraint")
  def test_explicit_mode_does_not_call_with_sharding_constraint(self, mock_wsc, mock_reshard):
    """EXPLICIT mode should NOT call with_sharding_constraint."""
    mock_reshard.return_value = self.inputs
    maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.EXPLICIT)
    mock_wsc.assert_not_called()


class TestMaybeShardWithNameDebugSharding(unittest.TestCase):
  """Tests for the debug_sharding logging behavior of maybe_shard_with_name."""

  def setUp(self):
    self.mesh = _make_single_device_mesh()
    self.named_sharding = NamedSharding(self.mesh, P())
    self.inputs = jnp.ones((4, 4))
    # Reset the module-level log-deduplication cache before each test.
    # sharding_module._LOGGED_ACTIVATION_SHARDINGS.clear()

  def tearDown(self):
    # sharding_module._LOGGED_ACTIVATION_SHARDINGS.clear()
    pass

  @mock.patch("jax.lax.with_sharding_constraint")
  def test_no_log_when_debug_sharding_false(self, mock_wsc):
    """When debug_sharding=False, max_logging.info should never be called."""
    mock_wsc.return_value = self.inputs
    with mock.patch("maxtext.utils.sharding.max_logging") as mock_ml:
      maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO, debug_sharding=False)
      mock_ml.info.assert_not_called()

  @mock.patch("jax.lax.with_sharding_constraint")
  def test_no_log_for_non_tracer_input(self, mock_wsc):
    """When debug_sharding=True but input is a concrete array (not a Tracer), should not log."""
    mock_wsc.return_value = self.inputs
    with mock.patch("maxtext.utils.sharding.max_logging") as mock_ml:
      # A jnp array outside of jit is a concrete value, not a Tracer.
      maybe_shard_with_name(self.inputs, self.named_sharding, ShardMode.AUTO, debug_sharding=True)
      mock_ml.info.assert_not_called()

  def test_same_key_logged_only_once(self):
    """The same (type, pspec, stack_level) combination should only produce one log entry."""
    with mock.patch("maxtext.utils.sharding.max_logging") as mock_ml:
      with mock.patch("jax.lax.with_sharding_constraint", side_effect=lambda x, s: x):

        @jax.jit
        def first_fn(x):
          return maybe_shard_with_name(x, self.named_sharding, ShardMode.AUTO, debug_sharding=True)

        first_fn(self.inputs)
        log_count_after_first = mock_ml.info.call_count

        # A distinct jit function forces re-tracing (Python body runs again), but the
        # same log_key is already present in _LOGGED_ACTIVATION_SHARDINGS, so no new log.
        @jax.jit
        def second_fn(x):
          return maybe_shard_with_name(x, self.named_sharding, ShardMode.AUTO, debug_sharding=True)

        second_fn(self.inputs)
        self.assertEqual(mock_ml.info.call_count, log_count_after_first)

  def test_different_stack_levels_produce_separate_log_entries(self):
    """Different extra_stack_level values create different log keys and each logs once."""
    with mock.patch("maxtext.utils.sharding.max_logging") as mock_ml:
      with mock.patch("jax.lax.with_sharding_constraint", side_effect=lambda x, s: x):

        @jax.jit
        def traced_fn(x):
          maybe_shard_with_name(x, self.named_sharding, ShardMode.AUTO, debug_sharding=True, extra_stack_level=0)
          return maybe_shard_with_name(x, self.named_sharding, ShardMode.AUTO, debug_sharding=True, extra_stack_level=1)

        traced_fn(self.inputs)
        self.assertEqual(mock_ml.info.call_count, 4)


if __name__ == "__main__":
  unittest.main()
