"""Tests for WeightConverter."""

import unittest
import jax.numpy as jnp
from flax import traverse_util

from maxtext.integration.vllm.weight_converter import (
    WeightConverter,
    WeightConverterRule,
    Concatenate,
    Transpose,
    UnstackScanned,
)

class WeightConverterTest(unittest.TestCase):
    def test_concatenate(self):
        op = Concatenate(dim=-1)
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[5, 6], [7, 8]])
        res = op([a, b])
        self.assertTrue(jnp.array_equal(res, jnp.concatenate([a, b], axis=-1)))

    def test_transpose(self):
        op = Transpose(axes=(1, 0))
        a = jnp.array([[1, 2], [3, 4]])
        res = op([a])
        self.assertTrue(jnp.array_equal(res, jnp.array([[1, 3], [2, 4]])))

    def test_unstack_scanned(self):
        op = UnstackScanned(scan_axis=0)
        a = jnp.array([
            [[1, 2], [3, 4]], # layer 0
            [[5, 6], [7, 8]], # layer 1
        ])
        res = op([a])
        self.assertTrue(jnp.array_equal(res[0], a[0]))
        self.assertTrue(jnp.array_equal(res[1], a[1]))

    def test_weight_converter_rule(self):
        rule = WeightConverterRule(
            source_patterns=[r"layers\.(\d+)\.attention\.wq\.kernel", r"layers\.(\d+)\.attention\.wk\.kernel"],
            target_pattern=r"layers.{}.attention.qk_proj.weight",
            operations=[Concatenate(dim=-1)]
        )
        # mock unstacked state
        weights = {
            "layers.0.attention.wq.kernel": jnp.array([[1, 2]]),
            "layers.0.attention.wk.kernel": jnp.array([[3, 4]])
        }
        converter = WeightConverter([rule])
        res = converter.convert(weights)
        
        flat_res = traverse_util.flatten_dict(res, sep='.')
        self.assertIn("layers.0.attention.qk_proj.weight", flat_res)
        self.assertTrue(jnp.array_equal(flat_res["layers.0.attention.qk_proj.weight"], jnp.array([[1, 2, 3, 4]])))

if __name__ == "__main__":
    unittest.main()
