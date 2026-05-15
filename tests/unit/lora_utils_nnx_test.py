# Copyright 2023–2026 Google LLC
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

"""Unit tests for the NNX-shaped LoRA helpers in `lora_utils`, plus a small
Linen regression block."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.utils.lora_utils import (
    apply_lora_on_base_params,
    apply_lora_on_base_params_nnx,
    get_lora_abstract_state_nnx,
    unapply_lora_from_base_params,
    unapply_lora_from_base_params_nnx,
)


# ---------------------------------------------------------------------------
# Fake abstract state builders (mirror the NNX vs. Linen tree shapes)
# ---------------------------------------------------------------------------


def _make_nnx_attention_abstract(emb=8, num_heads=2, head_dim=4, dtype=jnp.float32):
  """Tiny NNX-shaped abstract state for one attention block."""

  def _sds(shape):
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=None)

  return {
      "decoder": {
          "layers": {
              "self_attention": {
                  "query": {"kernel": _sds((emb, num_heads, head_dim))},
                  "key": {"kernel": _sds((emb, num_heads, head_dim))},
                  "value": {"kernel": _sds((emb, num_heads, head_dim))},
                  "out": {"kernel": _sds((emb, num_heads, head_dim))},
              },
              "mlp": {"wi": {"kernel": _sds((emb, 4 * emb))}},
          },
          "shared_embedding": {"embedding": _sds((100, emb))},
      },
  }


def _make_linen_attention_abstract(emb=8, num_heads=2, head_dim=4, dtype=jnp.float32):
  """Linen-shaped equivalent (with the `{"params": ...}` outer wrap)."""
  return {"params": _make_nnx_attention_abstract(emb, num_heads, head_dim, dtype)}


def _lora_config(rank=4, alpha=8.0, target_modules=("q_proj", "v_proj")):
  return {
      "r": rank,
      "lora_alpha": alpha,
      "target_modules": list(target_modules),
  }


# ---------------------------------------------------------------------------
# get_lora_abstract_state_nnx
# ---------------------------------------------------------------------------


class TestGetLoraAbstractStateNnx(unittest.TestCase):
  """`get_lora_abstract_state_nnx` shape, sharding, and error-path coverage."""

  def test_lora_shapes_for_query_and_value(self):
    abs_params = _make_nnx_attention_abstract(emb=8, num_heads=2, head_dim=4)
    state, _ = get_lora_abstract_state_nnx(abs_params, _lora_config(rank=4))
    attn = state.params["decoder"]["layers"]["self_attention"]

    a = attn["query"]["lora_a.kernel"]
    b = attn["query"]["lora_b.kernel"]
    self.assertEqual(a.shape, (8, 4))
    self.assertEqual(b.shape, (4, 2, 4))
    self.assertEqual(a.dtype, jnp.float32)
    self.assertEqual(b.dtype, jnp.float32)

    a = attn["value"]["lora_a.kernel"]
    b = attn["value"]["lora_b.kernel"]
    self.assertEqual(a.shape, (8, 4))
    self.assertEqual(b.shape, (4, 2, 4))

  def test_non_target_modules_emit_none_leaves(self):
    abs_params = _make_nnx_attention_abstract()
    state, _ = get_lora_abstract_state_nnx(abs_params, _lora_config(target_modules=("q_proj",)))
    attn = state.params["decoder"]["layers"]["self_attention"]
    self.assertIn("lora_a.kernel", attn["query"])
    self.assertIsNone(attn["key"]["kernel"])
    self.assertIsNone(attn["value"]["kernel"])
    self.assertIsNone(attn["out"]["kernel"])
    self.assertIsNone(state.params["decoder"]["layers"]["mlp"]["wi"]["kernel"])
    self.assertIsNone(state.params["decoder"]["shared_embedding"]["embedding"])

  def test_o_proj_has_distinct_shape(self):
    abs_params = _make_nnx_attention_abstract(emb=8, num_heads=2, head_dim=4)
    state, _ = get_lora_abstract_state_nnx(abs_params, _lora_config(rank=3, target_modules=("o_proj",)))
    out = state.params["decoder"]["layers"]["self_attention"]["out"]
    a = out["lora_a.kernel"]
    b = out["lora_b.kernel"]
    # 3D base (emb, num_heads, head_dim) → lora_a.shape = (..., r), lora_b = (r, last)
    self.assertEqual(a.shape, (8, 2, 3))
    self.assertEqual(b.shape, (3, 4))

  def test_unsupported_leaf_type_raises(self):
    bad = {"decoder": {"layers": {"self_attention": {"query": {"kernel": jnp.zeros((4, 2, 2))}}}}}
    with self.assertRaises(ValueError):
      get_lora_abstract_state_nnx(bad, _lora_config())

  def test_unexpected_leaf_name_raises(self):
    bad = {"decoder": {"layers": {"self_attention": {"query": {"weight": jax.ShapeDtypeStruct((4, 2), jnp.float32)}}}}}
    with self.assertRaises(ValueError):
      get_lora_abstract_state_nnx(bad, _lora_config())

  # Linen-vs-NNX numerical parity is covered by TestApplyLoraNnx.test_numerical_parity_with_linen_apply.


# ---------------------------------------------------------------------------
# apply / unapply on NNX-shape pure dicts
# ---------------------------------------------------------------------------


def _concrete_base(rng=None, emb=4, num_heads=2, head_dim=3):
  """Concrete arrays mirroring the abstract structure used above (NNX-shape)."""
  if rng is None:
    rng = jax.random.key(0)
  k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
  shape_attn = (emb, num_heads, head_dim)
  return {
      "decoder": {
          "layers": {
              "self_attention": {
                  "query": {"kernel": jax.random.normal(k1, shape_attn)},
                  "key": {"kernel": jax.random.normal(k2, shape_attn)},
                  "value": {"kernel": jax.random.normal(k3, shape_attn)},
                  "out": {"kernel": jax.random.normal(k4, shape_attn)},
              },
              "mlp": {"wi": {"kernel": jax.random.normal(k5, (emb, 4 * emb))}},
          },
          "shared_embedding": {"embedding": jax.random.normal(k6, (100, emb))},
      },
  }


def _build_lora_params(base, lora_config_dict, rng):
  """Build a concrete LoRA tree (random arrays) matching `base`."""
  abs_tree = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=None), base)
  lora_state, _ = get_lora_abstract_state_nnx(abs_tree, lora_config_dict)

  def _to_concrete(leaf, rng_key):
    if leaf is None:
      return None
    return jax.random.normal(rng_key, leaf.shape, leaf.dtype)

  leaves, tree = jax.tree_util.tree_flatten(lora_state.params, is_leaf=lambda x: x is None)
  rngs = jax.random.split(rng, max(1, len(leaves)))
  out_leaves = [_to_concrete(l, r) for l, r in zip(leaves, rngs)]
  return jax.tree_util.tree_unflatten(tree, out_leaves)


class TestApplyLoraNnx(unittest.TestCase):
  """`apply_lora_on_base_params_nnx` round-trip and Linen-vs-NNX parity."""

  def test_apply_then_unapply_is_identity(self):
    rng = jax.random.key(42)
    base_orig = _concrete_base(rng)
    base = jax.tree_util.tree_map(jnp.copy, base_orig)
    lora = _build_lora_params(base, _lora_config(rank=2, target_modules=("q_proj", "v_proj")), jax.random.key(7))
    apply_lora_on_base_params_nnx(base, lora, lora_scale_factor=0.5)
    # query/value kernels were modified
    self.assertFalse(
        jnp.allclose(
            base["decoder"]["layers"]["self_attention"]["query"]["kernel"],
            base_orig["decoder"]["layers"]["self_attention"]["query"]["kernel"],
        )
    )
    # key/out are untouched
    np.testing.assert_array_equal(
        np.asarray(base["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(base["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
    )
    unapply_lora_from_base_params_nnx(base, lora, lora_scale_factor=0.5)
    np.testing.assert_allclose(
        np.asarray(base["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(base["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )

  def test_numerical_parity_with_linen_apply(self):
    """Same base+lora numbers → same kernel after apply, on either tree shape."""
    rng = jax.random.key(123)
    base_nnx = _concrete_base(rng)
    base_linen = {"params": jax.tree_util.tree_map(jnp.copy, base_nnx)}
    lora = _build_lora_params(base_nnx, _lora_config(rank=2, target_modules=("q_proj",)), jax.random.key(5))
    apply_lora_on_base_params_nnx(base_nnx, lora, lora_scale_factor=0.7)
    apply_lora_on_base_params(base_linen, {"params": lora}, lora_scale_factor=0.7)
    np.testing.assert_allclose(
        np.asarray(base_nnx["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        np.asarray(base_linen["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        rtol=1e-6,
    )

  def test_apply_with_unexpected_lora_key_raises(self):
    base = _concrete_base()
    bad = {"decoder": {"layers": {"self_attention": {"query": {"unexpected": jnp.zeros((4, 2))}}}}}
    with self.assertRaises(ValueError):
      apply_lora_on_base_params_nnx(base, bad)


class TestLinenLoraRegression(unittest.TestCase):
  """Smoke tests for the Linen apply / unapply helpers (no other unit test exercises them)."""

  def _linen_pair(self, rng=None):
    """Build a Linen-shape (with `{"params": ...}` outer wrapper) base + lora pair."""
    if rng is None:
      rng = jax.random.key(99)
    base_inner = _concrete_base(rng)
    base = {"params": jax.tree_util.tree_map(jnp.copy, base_inner)}
    lora_inner = _build_lora_params(
        base_inner,
        _lora_config(rank=2, target_modules=("q_proj", "v_proj")),
        jax.random.key(7),
    )
    lora = {"params": lora_inner}
    return base, lora

  def test_linen_apply_then_unapply_is_identity(self):
    base, lora = self._linen_pair()
    base_orig = jax.tree_util.tree_map(jnp.copy, base)
    apply_lora_on_base_params(base, lora, lora_scale_factor=0.5)
    unapply_lora_from_base_params(base, lora, lora_scale_factor=0.5)
    np.testing.assert_allclose(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )

  def test_linen_apply_only_modifies_target_modules(self):
    base, lora = self._linen_pair()
    base_orig = jax.tree_util.tree_map(jnp.copy, base)
    apply_lora_on_base_params(base, lora, lora_scale_factor=1.0)
    # query and value are targets — must change.
    self.assertFalse(
        jnp.allclose(
            base["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"],
            base_orig["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"],
        )
    )
    # key and out are non-target — must be untouched.
    np.testing.assert_array_equal(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
    )


if __name__ == "__main__":
  unittest.main()
