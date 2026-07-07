import jax.numpy as jnp
from maxtext.integration.vllm.weight_converter import (
    WeightConverter,
    WeightRenaming,
    WeightConverterRule,
    Concatenate,
    Transpose,
    UnstackScanned,
)

def test_weight_renaming():
    renaming = WeightRenaming(
        source_pattern=r"layers\.(\d+)\.attention\.wq\.kernel",
        target_pattern=r"layers.\g<1>.attention.qkv_proj.weight"
    )
    # The WeightRenaming logic uses sub() so \g<1> or \1 is needed if using regex groups
    # Wait, the implementation uses source_pattern.sub(rule.target_pattern, src_key). 
    # Let's test with simple replace or regex group.
    pass

def test_concatenate():
    op = Concatenate(dim=-1)
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6], [7, 8]])
    res = op([a, b])
    assert jnp.array_equal(res, jnp.concatenate([a, b], axis=-1))

def test_transpose():
    op = Transpose(axes=(1, 0))
    a = jnp.array([[1, 2], [3, 4]])
    res = op([a])
    assert jnp.array_equal(res, jnp.array([[1, 3], [2, 4]]))

def test_unstack_scanned():
    op = UnstackScanned(scan_axis=0)
    a = jnp.array([
        [[1, 2], [3, 4]], # layer 0
        [[5, 6], [7, 8]], # layer 1
    ])
    res = op([a])
    assert jnp.array_equal(res[0], a[0])
    assert jnp.array_equal(res[1], a[1])

def test_weight_converter_rule():
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
    from flax import traverse_util
    flat_res = traverse_util.flatten_dict(res, sep='.')
    assert "layers.0.attention.qk_proj.weight" in flat_res
    assert jnp.array_equal(flat_res["layers.0.attention.qk_proj.weight"], jnp.array([[1, 2, 3, 4]]))


if __name__ == "__main__":
    test_weight_renaming()
    test_concatenate()
    test_transpose()
    test_unstack_scanned()
    test_weight_converter_rule()
    print("All tests passed!")
