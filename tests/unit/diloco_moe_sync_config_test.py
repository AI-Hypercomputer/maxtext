
import unittest
from maxtext.configs.pyconfig import initialize_pydantic
from tests.utils.test_helpers import get_test_config_path


class DiLoCoMoESyncConfigTest(unittest.TestCase):

  def test_default_config(self):
    base_args = ["", get_test_config_path(), "skip_jax_distributed_system=True"]
    config = initialize_pydantic(base_args + ["enable_diloco=True", "diloco_sync_period=36"])
    self.assertEqual(config.moe_router_syncing_period, -1)

  def test_valid_divisible_periods(self):
    base_args = ["", get_test_config_path(), "skip_jax_distributed_system=True"]
    config1 = initialize_pydantic(base_args + ["enable_diloco=True", "diloco_sync_period=36", "moe_router_syncing_period=1"])
    self.assertEqual(config1.moe_router_syncing_period, 1)

    config6 = initialize_pydantic(base_args + ["enable_diloco=True", "diloco_sync_period=36", "moe_router_syncing_period=6"])
    self.assertEqual(config6.moe_router_syncing_period, 6)

  def test_invalid_indivisible_period(self):
    base_args = ["", get_test_config_path(), "skip_jax_distributed_system=True"]
    with self.assertRaises(ValueError) as ctx:
      initialize_pydantic(base_args + ["enable_diloco=True", "diloco_sync_period=36", "moe_router_syncing_period=5"])
    self.assertIn("divisible", str(ctx.exception))

  def test_invalid_negative_period(self):
    base_args = ["", get_test_config_path(), "skip_jax_distributed_system=True"]
    with self.assertRaises(ValueError) as ctx:
      initialize_pydantic(base_args + ["enable_diloco=True", "diloco_sync_period=36", "moe_router_syncing_period=-2"])
    self.assertIn("must be > 0 or -1", str(ctx.exception))

  def test_invalid_greater_than_sync_period(self):
    base_args = ["", get_test_config_path(), "skip_jax_distributed_system=True"]
    with self.assertRaises(ValueError) as ctx:
      initialize_pydantic(base_args + ["enable_diloco=True", "diloco_sync_period=36", "moe_router_syncing_period=40"])
    self.assertIn("cannot be greater than", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()

