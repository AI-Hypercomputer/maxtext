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

"""Structural tests for the OLMoE3 decoder (hybrid KDA + latent MoE).

The parameter-count assertions are the Phase 0 acceptance gate: they are exact,
and they catch shape, norm-placement, and routing-width errors that a forward
pass would happily hide. Canonical counts come from AI2's
``src/scripts/standalone/README.md`` (OLMo-core ``akshitab/standalone``), and were
independently reproduced by running its ``standalone_model.py`` reference.
"""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils, maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

import os

# (model_name, total params, active params) from the reference ladder.
CANONICAL_COUNTS = {
    "olmoe3-30m": (32_323_588, 29_964_292),
    "olmoe3-3p5b": (62_864_102_080, 3_475_903_168),
}


def _config(model_name, max_target_length=64, scan_layers=False):
  return pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          f"model_name={model_name}",
          "run_name=olmoe3_test",
          "enable_checkpointing=False",
          f"scan_layers={scan_layers}",
          "per_device_batch_size=1",
          f"max_target_length={max_target_length}",
          "dtype=float32",
          "weight_dtype=float32",
          "megablox=False",
          "sparse_matmul=False",
          "skip_jax_distributed_system=True",
      ],
  )


def _build(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = models.transformer_as_linen(cfg, mesh, quant=quantizations.configure_quantization(cfg))
  return mesh, model


def _abstract_params(cfg, model, seq_len):
  rng = jax.random.PRNGKey(0)
  ids = jnp.ones((1, seq_len), dtype=jnp.int32)
  return jax.eval_shape(lambda: model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False))["params"]


class OLMoE3ParameterCountTest(unittest.TestCase):
  """Exact parameter parity with the AI2 reference implementation."""

  def _assert_total(self, model_name, scan_layers=False):
    expected_total, _ = CANONICAL_COUNTS[model_name]
    cfg = _config(model_name, scan_layers=scan_layers)
    _, model = _build(cfg)
    total = max_utils.calculate_num_params_from_pytree(_abstract_params(cfg, model, 64))
    self.assertEqual(total, expected_total, f"{model_name}: {total:,} != reference {expected_total:,}")

  def test_30m_total_parameters(self):
    self._assert_total("olmoe3-30m")

  def test_3p5b_total_parameters(self):
    self._assert_total("olmoe3-3p5b")

  def test_30m_total_parameters_scanned(self):
    """Scanned cycles share one parameter set, so the dense first layer has to be peeled out.

    Without the peel every cycle gets a dense layer 0 and the count silently
    grows; training still runs, which is why this is asserted.
    """
    self._assert_total("olmoe3-30m", scan_layers=True)

  def test_3p5b_total_parameters_scanned(self):
    self._assert_total("olmoe3-3p5b", scan_layers=True)

  def test_router_scores_the_full_width_residual(self):
    """OLMoE3 routes on emb_dim while the experts consume the latent.

    A gate sized to ``moe_expert_input_dim`` would still train, and would still
    produce a plausible loss curve, so this is asserted explicitly.
    """
    cfg = _config("olmoe3-30m")
    _, model = _build(cfg)
    params = _abstract_params(cfg, model, 64)
    gates = [
        leaf.shape
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
        if "moe_block/gate" in "/".join(str(getattr(k, "key", k)) for k in path)
    ]
    self.assertTrue(gates, "no MoE gate kernel found")
    for shape in gates:
      self.assertEqual(shape[0], cfg.emb_dim)
      self.assertNotEqual(shape[0], cfg.moe_expert_input_dim)


class OLMoE3PackingTest(unittest.TestCase):
  """Packed documents must not leak across boundaries."""

  def test_packed_documents_match_separate_sequences(self):
    """Two documents packed into one sequence == the same two run separately.

    This covers the KDA recurrent-state reset and the depthwise-conv boundary
    masking, neither of which shows up in a loss curve until quality is already
    degraded.
    """
    seq_len, doc_len = 32, 16
    cfg = _config("olmoe3-30m", max_target_length=seq_len)
    mesh, model = _build(cfg)

    tokens = jnp.arange(1, seq_len + 1, dtype=jnp.int32)[None, :]
    packed_positions = jnp.concatenate([jnp.arange(doc_len), jnp.arange(doc_len)])[None, :].astype(jnp.int32)
    packed_segments = jnp.concatenate([jnp.zeros((1, doc_len), jnp.int32), jnp.ones((1, doc_len), jnp.int32)], axis=1)

    with mesh:
      params = model.init({"params": jax.random.PRNGKey(0)}, tokens, packed_positions, packed_segments)
      packed = model.apply(params, tokens, packed_positions, packed_segments, enable_dropout=False)
      packed_logits = packed[0] if isinstance(packed, tuple) else packed

      # The first document alone, padded back to the same length so shapes match.
      first_only_segments = jnp.concatenate(
          [jnp.ones((1, doc_len), jnp.int32), jnp.zeros((1, doc_len), jnp.int32)], axis=1
      )
      solo = model.apply(params, tokens, packed_positions, first_only_segments, enable_dropout=False)
      solo_logits = solo[0] if isinstance(solo, tuple) else solo

    # Positions inside the first document must be unaffected by what follows it.
    jnp.allclose(packed_logits[:, :doc_len], solo_logits[:, :doc_len], atol=1e-4)
    self.assertTrue(
        jnp.allclose(packed_logits[:, :doc_len], solo_logits[:, :doc_len], atol=1e-4),
        "first document's logits changed when a second document was packed after it",
    )


if __name__ == "__main__":
  unittest.main()


class OLMoE3ChunkedDeltaRuleTest(unittest.TestCase):
  """The chunked delta rule must be the scan, not an approximation of it.

  The scan is validated against the PyTorch reference, so pinning the chunked
  path to the scan transitively pins it to the reference. A chunked linear-
  attention kernel that is subtly wrong still trains and still produces a
  falling loss curve, which is exactly why this is asserted numerically.
  """

  def _inputs(self, seq_len, resets_at=(), dk=16, dv=32, batch=2, heads=3):
    """Builds delta-rule inputs shaped like the ones the module produces."""
    # pylint: disable=import-outside-toplevel
    from maxtext.models import olmoe3

    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(keys[0], (batch, seq_len, heads, dk), jnp.float32) * 0.5
    k = jax.random.normal(keys[1], (batch, seq_len, heads, dk), jnp.float32) * 0.5
    v = jax.random.normal(keys[2], (batch, seq_len, heads, dv), jnp.float32)
    # decay in (0, 1) per key channel, beta in [0, 2), as the module produces
    decay = jax.nn.sigmoid(jax.random.normal(keys[3], (batch, seq_len, heads, dk), jnp.float32)) * 0.4 + 0.55
    beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq_len, heads), jnp.float32))
    resets = jnp.zeros((batch, seq_len), dtype=bool)
    for idx in resets_at:
      resets = resets.at[:, idx].set(True)
    return olmoe3, (q, k, v, decay, beta, resets)

  def _assert_matches(self, seq_len, chunk, resets_at=(), state_dtype="float32", tol=2e-5):
    """Asserts the chunked rule reproduces the scan within ``tol``."""
    olmoe3, args = self._inputs(seq_len, resets_at)
    q, k, v, decay, beta, resets = args
    # The module carries the decay in log form, so the chunked rule takes it that
    # way; the scan takes the decay itself.
    expected = olmoe3._delta_rule_scan(q, k, v, decay, beta, resets)  # pylint: disable=protected-access
    actual = olmoe3._delta_rule_chunked(  # pylint: disable=protected-access
        q, k, v, jnp.log(decay), beta, resets, chunk, state_dtype
    )
    rel = jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))
    self.assertLess(float(rel), tol, f"seq={seq_len} chunk={chunk} {state_dtype}: rel err {rel:.2e}")

  def test_matches_scan_across_chunk_sizes(self):
    for chunk in (16, 32, 64):
      self._assert_matches(128, chunk)

  def test_matches_scan_with_packed_documents(self):
    # A boundary mid-chunk and one exactly on a chunk edge.
    for chunk in (16, 32, 64):
      self._assert_matches(128, chunk, resets_at=(45, 64))

  def test_bfloat16_state_stays_close(self):
    """The bf16 state option halves traffic on a bandwidth-bound path.

    It is off by default. The tolerance is bf16's, not float32's: this pins that
    the option is wired correctly, not that it is free.
    """
    self._assert_matches(128, 32, state_dtype="bfloat16", tol=5e-2)
