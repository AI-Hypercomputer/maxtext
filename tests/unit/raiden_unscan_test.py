# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `raiden_unscan.unscan_layers`.

This transform decides the *names* Raiden binds. The trainer runs scanned
(`scan_layers=True`); the sampler loads its MaxText model unscanned, and Raiden matches
tensors by `jax.tree_util.keystr` path. Nothing downstream cross-checks the two name sets
-- `raiden_handler._validate_metadata` only checks a single manifest's internal
consistency (mesh rank, duplicate variable/layer keys, sharding specs) -- so a naming
error here surfaces as weights that silently never transfer, not as an exception.

The transform is pure pytree manipulation, so all of this runs on CPU in milliseconds.
"""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.integration.tunix.weight_mapping import raiden_unscan
import numpy as np
import pytest

# This transform exists only for the Tunix/Raiden weight-sync path, so it is graded with
# the rest of that work. `tests/unit` is in cpu-post-training-unit's path list, so the
# marker alone routes it there -- no file move needed (unlike tests/ or tests/integration,
# which are not in that list and would be collected by no job at all).
pytestmark = [pytest.mark.post_training]


_NUM_LAYERS = 3
_IN, _OUT, _VOCAB = 4, 8, 10


def _unwrap(leaf):
  """Reads a leaf's array whether or not it is wrapped in an `nnx.Param`."""
  if isinstance(leaf, nnx.Variable):
    return leaf[...]
  return leaf


def _names(tree) -> list[str]:
  """The names Raiden binds: exactly what `raiden_synchronizer.flatten_weights` computes."""
  return sorted(jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(tree))


class ScannedInner(nnx.Module):
  """One scanned param: the layer axis lives at axis 1 of a single array."""

  def __init__(self, num_layers: int = _NUM_LAYERS):
    self.kernel = nnx.Param(jnp.arange(_IN * num_layers * _OUT, dtype=jnp.float32).reshape(_IN, num_layers, _OUT))
    self.scale = nnx.Param(jnp.arange(_OUT * num_layers, dtype=jnp.float32).reshape(_OUT, num_layers))


class ScannedModel(nnx.Module):
  """A trainer-side model: scanned `layers`, plus non-layer params that must pass through."""

  def __init__(self, num_layers: int = _NUM_LAYERS):
    self.layers = ScannedInner(num_layers)
    self.embed = nnx.Param(jnp.zeros((_VOCAB, _IN)))


class UnscannedInner(nnx.Module):

  def __init__(self):
    self.kernel = nnx.Param(jnp.zeros((_IN, _OUT)))
    self.scale = nnx.Param(jnp.zeros((_OUT,)))


class UnscannedModel(nnx.Module):
  """A sampler-side model: one submodule per layer, named `layers_0..N-1`."""

  def __init__(self, num_layers: int = _NUM_LAYERS):
    for i in range(num_layers):
      setattr(self, f"layers_{i}", UnscannedInner())
    self.embed = nnx.Param(jnp.zeros((_VOCAB, _IN)))


class UnscanLayersTest(absltest.TestCase):

  def _scanned_state(self, num_layers: int = _NUM_LAYERS):
    return nnx.state(ScannedModel(num_layers), nnx.Param)

  def test_names_match_an_unscanned_model_exactly(self):
    """The point of the transform: trainer names must equal sampler names.

    Raiden binds by `keystr` path on both sides, and nothing validates that the two sets
    agree, so this is the assertion that a silent no-transfer would violate.
    """
    unscanned = raiden_unscan.unscan_layers(self._scanned_state(), num_layers=_NUM_LAYERS)
    sampler_side = nnx.state(UnscannedModel(), nnx.Param)
    self.assertEqual(_names(unscanned), _names(sampler_side))

  def test_plain_dict_and_nnx_state_produce_identical_names(self):
    """`unscan_layers` returns a plain nested dict, the sampler binds an `nnx.State`.

    `keystr` renders both identically only because the transform rewraps leaves in
    `nnx.Param`. Dropping that rewrap would rename every tensor (`['k']` vs `['k'].value`)
    and break every transfer, so pin it.
    """
    unscanned = raiden_unscan.unscan_layers(self._scanned_state(), num_layers=_NUM_LAYERS)
    self.assertIsInstance(jax.tree_util.tree_leaves(unscanned, is_leaf=lambda x: isinstance(x, nnx.Param))[0], nnx.Param)
    self.assertTrue(all(n.endswith(".value") for n in _names(unscanned)), _names(unscanned))

  def test_slices_carry_the_right_values(self):
    """Layer i must receive index i of the scan axis -- not a transpose or an off-by-one."""
    state = self._scanned_state()
    original = np.asarray(state.to_pure_dict()["layers"]["kernel"])
    unscanned = raiden_unscan.unscan_layers(state, num_layers=_NUM_LAYERS)

    for i in range(_NUM_LAYERS):
      got = unscanned[f"layers_{i}"]["kernel"]
      got = np.asarray(_unwrap(got))
      self.assertEqual(got.shape, (_IN, _OUT))
      np.testing.assert_array_equal(got, original[:, i, :])

  def test_rank_two_param_is_also_unscanned(self):
    """A rank-2 scanned param (e.g. a norm scale) slices down to rank 1."""
    unscanned = raiden_unscan.unscan_layers(self._scanned_state(), num_layers=_NUM_LAYERS)
    for i in range(_NUM_LAYERS):
      scale = unscanned[f"layers_{i}"]["scale"]
      self.assertEqual(np.asarray(_unwrap(scale)).shape, (_OUT,))

  def test_non_layer_entries_pass_through_unchanged(self):
    """Embeddings and final norms have no layer axis and must survive untouched."""
    state = self._scanned_state()
    embed_before = np.asarray(state.to_pure_dict()["embed"])
    unscanned = raiden_unscan.unscan_layers(state, num_layers=_NUM_LAYERS)

    self.assertIn("embed", unscanned)
    embed_after = unscanned["embed"]
    np.testing.assert_array_equal(np.asarray(_unwrap(embed_after)), embed_before)
    self.assertNotIn("layers", unscanned)

  def test_layer_count_mismatch_raises(self):
    """A wrong num_layers must fail loudly rather than bind truncated weights."""
    with self.assertRaisesRegex(ValueError, "expected axis 1 to be num_layers=99"):
      raiden_unscan.unscan_layers(self._scanned_state(), num_layers=99)

  def test_already_unscanned_state_raises(self):
    """The anti-silent-no-op guard.

    Without it an already-unscanned (or wrongly-keyed) state would return unchanged and
    bind under scanned names, transferring nothing with no error anywhere.
    """
    with self.assertRaisesRegex(ValueError, "found no scanned 'layers' entries"):
      raiden_unscan.unscan_layers(nnx.state(UnscannedModel(), nnx.Param), num_layers=_NUM_LAYERS)

  def test_wrong_layer_container_raises(self):
    with self.assertRaisesRegex(ValueError, "found no scanned 'blocks' entries"):
      raiden_unscan.unscan_layers(self._scanned_state(), num_layers=_NUM_LAYERS, layer_container="blocks")

  def test_custom_scan_axis(self):
    """`param_scan_axis` is configurable; axis 0 must slice the leading dim."""
    state = {"layers": {"kernel": jnp.arange(_NUM_LAYERS * _OUT, dtype=jnp.float32).reshape(_NUM_LAYERS, _OUT)}}
    unscanned = raiden_unscan.unscan_layers(state, num_layers=_NUM_LAYERS, scan_axis=0)
    for i in range(_NUM_LAYERS):
      got = unscanned[f"layers_{i}"]["kernel"]
      np.testing.assert_array_equal(np.asarray(_unwrap(got)), np.arange(i * _OUT, (i + 1) * _OUT))

  def test_plain_dict_input_is_accepted(self):
    """The trainer passes an `nnx.State`, but the signature documents plain dicts too."""
    state = {
        "layers": {"kernel": jnp.zeros((_IN, _NUM_LAYERS, _OUT))},
        "embed": jnp.zeros((_VOCAB, _IN)),
    }
    unscanned = raiden_unscan.unscan_layers(state, num_layers=_NUM_LAYERS)
    self.assertEqual(sorted(unscanned.keys()), ["embed"] + [f"layers_{i}" for i in range(_NUM_LAYERS)])

  def test_total_element_count_is_preserved(self):
    """Unscanning reshapes; it must not drop or duplicate any weight."""
    state = self._scanned_state()
    before = sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(state.to_pure_dict()))
    unscanned = raiden_unscan.unscan_layers(state, num_layers=_NUM_LAYERS)
    after = sum(int(np.size(np.asarray(_unwrap(x)))) for x in jax.tree_util.tree_leaves(unscanned))
    self.assertEqual(after, before)


if __name__ == "__main__":
  absltest.main()
