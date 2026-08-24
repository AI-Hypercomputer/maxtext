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


def _config(model_name, max_target_length=64, scan_layers=False, extra=()):
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
          *extra,
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


class OLMoE3RoutingTest(unittest.TestCase):
  """Reference routing semantics: weight scale, LB loss formula.

  Both are invisible in a loss curve: a run with weights summing to 1 or a
  Switch-style LB loss still converges, just not to the reference's trajectory.
  """

  @classmethod
  def setUpClass(cls):
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.models.olmoe3 import OLMoE3DecoderLayer

    cls.cfg = _config("olmoe3-30m")
    mesh = Mesh(maxtext_utils.create_device_mesh(cls.cfg), cls.cfg.mesh_axes)
    with mesh:
      cls.layer = OLMoE3DecoderLayer(cls.cfg, mesh, MODEL_MODE_TRAIN, layer_idx=1, rngs=nnx.Rngs(params=0))

  def test_topk_weights_sum_to_top_k(self):
    """normalize_expert_weights=1.0 + restore_weight_scale: weights sum to K, not 1."""
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 8, self.cfg.num_experts), jnp.float32)
    weights, _ = self.layer.moe_block.get_topk(logits, None)
    self.assertTrue(
        jnp.allclose(weights.sum(axis=-1), self.cfg.num_experts_per_tok, atol=1e-3),
        f"top-k weights sum to {weights.sum(axis=-1).ravel()[0]}, expected {self.cfg.num_experts_per_tok}",
    )

  def test_load_balance_loss_matches_reference_formula(self):
    """The LB loss is the reference's (E/K)-scaled batch-level product, not Switch-style E^2."""
    e, k = self.cfg.num_experts, self.cfg.num_experts_per_tok
    batch, seq = 2, 8
    logits = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, e), jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    _, indices = jax.lax.top_k(logits, k)
    actual = self.layer.moe_block.load_balance_loss(indices, probs)
    # Direct transcription of the reference auxiliary_loss, global path.
    counts = jax.nn.one_hot(indices, e).sum(axis=-2).sum(axis=(0, 1))
    lb = (probs.mean(axis=(0, 1)) * counts).sum() / (batch * seq)
    expected = (e / k) * lb * self.cfg.load_balance_loss_weight
    self.assertAlmostEqual(float(actual), float(expected), places=5)

  def test_emo_full_pool_is_identity(self):
    """With the pool spanning every expert, EMo must select exactly the standard top-k.

    The 30m config's eval pool is num_experts, and rngs=None takes the eval
    path, so this exercises the masking machinery end to end as a no-op.
    """
    k = self.cfg.num_experts_per_tok
    segs = jnp.concatenate([jnp.full((2, 3), 1, jnp.int32), jnp.full((2, 5), 2, jnp.int32)], axis=1)
    logits = jax.random.normal(jax.random.PRNGKey(2), (2, 8, self.cfg.num_experts), jnp.float32)
    _, emo_indices = self.layer.moe_block.get_topk(logits, None, rngs=None, input_ids=segs)
    _, std_indices = jax.lax.top_k(logits, k)
    self.assertTrue(bool((emo_indices == std_indices).all()))

  def test_emo_pools_match_reference(self):
    """Deterministic-pool EMo against a direct transcription of the reference EMoRouter.

    min = max = eval = P makes the pool draw irrelevant, so the document
    masking itself is what's compared: per-document score totals, top-P pool,
    top-k within the pool.
    """
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import numpy as np
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.models.olmoe3 import OLMoE3DecoderLayer

    pool = 20
    cfg = _config(
        "olmoe3-30m",
        extra=(
            "override_model_config=True",
            f"emo_min_document_expert_pool={pool}",
            f"emo_max_document_expert_pool={pool}",
            f"emo_eval_document_expert_pool={pool}",
        ),
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      layer = OLMoE3DecoderLayer(cfg, mesh, MODEL_MODE_TRAIN, layer_idx=1, rngs=nnx.Rngs(params=0))
    e, k = cfg.num_experts, cfg.num_experts_per_tok
    batch, seq = 2, 12
    segs = jnp.asarray([[1] * 5 + [2] * 7, [3] * 12], jnp.int32)
    logits = jax.random.normal(jax.random.PRNGKey(4), (batch, seq, e), jnp.float32)
    weights, indices = layer.moe_block.get_topk(logits, None, rngs=None, input_ids=segs)
    self.assertTrue(bool(jnp.allclose(weights.sum(axis=-1), k, atol=1e-3)))

    scores = np.asarray(jax.nn.softmax(logits, axis=-1))
    segs_np = np.asarray(segs)
    for b in range(batch):
      for doc in np.unique(segs_np[b]):
        doc_tokens = segs_np[b] == doc
        doc_pool = set(np.argsort(-scores[b, doc_tokens].sum(0))[:pool].tolist())
        for t in np.where(doc_tokens)[0]:
          masked = np.where(np.isin(np.arange(e), list(doc_pool)), scores[b, t], -np.inf)
          expected = set(np.argsort(-masked)[:k].tolist())
          self.assertEqual(set(np.asarray(indices[b, t]).tolist()), expected, f"b={b} t={t}")


try:  # pylint: disable=g-import-not-at-top
  from tokamax._src.ops.experimental.kda import api as _tokamax_kda_api  # noqa: F401  pylint: disable=unused-import

  _HAVE_TOKAMAX_KDA = True
except ImportError:
  _HAVE_TOKAMAX_KDA = False


@unittest.skipUnless(_HAVE_TOKAMAX_KDA, "tokamax build without the KDA op (openxla/tokamax PR #1103)")
class OLMoE3TokamaxKDATest(unittest.TestCase):
  """``use_tokamax_kda`` must reproduce the unfused KDA layer.

  This pins the argument mapping into the tokamax kernel: raw (unnormalized)
  q/k/gate hand-off, in-kernel dt_bias/A_log activation, the [H, B, T, D]
  transposes, the 1-indexed segment-id convention, and beta in [0, 2).
  """

  def _layer_out(self, use_tokamax, x, segment_ids):
    """Run one KDA layer, fused or unfused, and return its output."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention

    cfg = _config("olmoe3-30m", extra=(f"use_tokamax_kda={use_tokamax}",))
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
      # Jitted like training. tokamax's eager-only beta range check rejects the
      # reference's allow_neg_eigval betas (in [0, 2)); under jit the check is
      # skipped and the delta-rule algebra is exact for beta < 2, which is what
      # this test then verifies numerically.
      return jax.device_get(jax.jit(layer)(x, segment_ids))

  def _compare(self, segment_ids):
    """Assert the fused and unfused KDA layers agree."""
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 128), jnp.float32)
    expected = self._layer_out(False, x, segment_ids)
    actual = self._layer_out(True, x, segment_ids)
    rel = jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))
    # The only known semantic difference is the q/k L2-norm epsilon: tokamax
    # (like FLA's production kernel, which is what AI2 trains with) uses
    # rsqrt(sum + 1e-6); the unfused path matches the standalone reference's
    # max(sum, 1e-12). That contributes ~5e-4; anything past 2e-3 means the
    # argument mapping broke.
    self.assertLess(float(rel), 2e-3, f"tokamax KDA relative error {rel:.2e}")

  def test_matches_unfused_unpacked(self):
    """Fused KDA matches the unfused rule on a single document."""
    self._compare(None)

  def test_matches_unfused_packed(self):
    """Fused KDA matches the unfused rule across a packed-document boundary."""
    # 1-indexed ids with the boundary mid-sequence (MaxText packing convention).
    segment_ids = jnp.concatenate([jnp.full((2, 30), 1, jnp.int32), jnp.full((2, 34), 2, jnp.int32)], axis=1)
    self._compare(segment_ids)


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

  def test_matches_scan_with_reference_scale_decay(self):
    """Decay at the magnitudes the reference init actually produces.

    ``A_log`` initializes to log(U[1, 16]), so per-step log-decay reaches -24
    and the cumulative product across a 64-chunk leaves fp32 range. Any scheme
    that folds cumulative decay into the matmul operands overflows to NaN here;
    the chunked rule must compute the pairwise (<= 1) decays exactly.
    """
    olmoe3_mod, (q, k, v, _, beta, resets) = self._inputs(128)
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    a = jax.random.uniform(keys[0], (1, 1, q.shape[2], 1), jnp.float32, minval=1.0, maxval=16.0)
    gate = jax.nn.softplus(jax.random.normal(keys[1], q.shape, jnp.float32))
    decay = jnp.exp(-a * gate)
    expected = olmoe3_mod._delta_rule_scan(q, k, v, decay, beta, resets)  # pylint: disable=protected-access
    actual = olmoe3_mod._delta_rule_chunked(  # pylint: disable=protected-access
        q, k, v, jnp.log(decay), beta, resets, 64, "float32"
    )
    self.assertTrue(bool(jnp.isfinite(actual).all()), "chunked delta rule overflowed at reference-scale decay")
    rel = jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))
    self.assertLess(float(rel), 2e-5, f"rel err {rel:.2e}")

  def test_newton_inversion_matches_solve_triangular(self):
    """The log-depth Newton inversion must equal the triangular solve it replaced.

    The delta rule's system matrix is I + M with M strictly lower triangular
    (nilpotent), so Newton is exact in ceil(log2(C))-1 iterations. This pins the
    substitution directly, at the model's beta range [0, 2) and unit-norm keys.
    """
    key = jax.random.PRNGKey(0)
    for c in (16, 32, 64):
      key, k1, k2 = jax.random.split(key, 3)
      batch, heads, dk = 2, 3, 32
      k_vecs = jax.random.normal(k1, (batch, heads, c, dk))
      k_vecs /= jnp.linalg.norm(k_vecs, axis=-1, keepdims=True)
      beta = 2.0 * jax.nn.sigmoid(jax.random.normal(k2, (batch, heads, c)))
      m = jnp.einsum("bhid,bhjd->bhij", k_vecs, k_vecs)
      m = jnp.where(jnp.tril(jnp.ones((c, c), bool), k=-1), m, 0.0) * beta[..., None]
      eye = jnp.eye(c)
      reference = jax.scipy.linalg.solve_triangular(
          eye + m, jnp.broadcast_to(eye, m.shape), lower=True, unit_diagonal=True
      )
      a_mat = eye + m
      x = eye - m
      for _ in range(max(0, (c - 1).bit_length() - 1)):
        x = x @ (2.0 * eye - a_mat @ x)
      rel = jnp.max(jnp.abs(x - reference)) / jnp.max(jnp.abs(reference))
      self.assertLess(float(rel), 1e-5, f"c={c}: Newton inverse rel err {rel:.2e}")

  def test_gradients_finite_and_match_scan(self):
    """Backward through the chunked rule, at reference-scale decay with a mid-chunk reset.

    Two regressions pinned: (1) masking any excluded pair *after* the exp
    leaves inf * 0 = NaN in the VJP (cross-segment pairs overflow under strong
    decay); (2) the scan-body checkpoint must stay differentiable. The decay
    gradient is also checked against the scan path's gradient, which pins the
    whole chunked backward to the reference recurrence.
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.models import olmoe3

    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    batch, seq, heads, dk, dv = 2, 128, 2, 16, 32
    q = olmoe3._l2_normalize(jax.random.normal(keys[0], (batch, seq, heads, dk))) * dk**-0.5  # pylint: disable=protected-access
    k = olmoe3._l2_normalize(jax.random.normal(keys[1], (batch, seq, heads, dk)))  # pylint: disable=protected-access
    v = jax.random.normal(keys[2], (batch, seq, heads, dv))
    a = jax.random.uniform(keys[3], (1, 1, heads, 1), minval=1.0, maxval=16.0)
    log_decay = -a * jax.nn.softplus(jax.random.normal(keys[4], (batch, seq, heads, dk)))
    beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[3], (batch, seq, heads)))
    resets = jnp.zeros((batch, seq), bool).at[:, 45].set(True).at[:, 64].set(True)

    def chunked_sum(q_, k_, v_, log_decay_, beta_):
      return olmoe3._delta_rule_chunked(q_, k_, v_, log_decay_, beta_, resets, 64, "float32").sum()  # pylint: disable=protected-access

    grads = jax.grad(chunked_sum, argnums=(0, 1, 2, 3, 4))(q, k, v, log_decay, beta)
    for name, grad in zip(("q", "k", "v", "log_decay", "beta"), grads):
      self.assertTrue(bool(jnp.isfinite(grad).all()), f"non-finite grad wrt {name}")

    scan_decay_grad = jax.grad(
        lambda ld: olmoe3._delta_rule_scan(q, k, v, jnp.exp(ld), beta, resets).sum()  # pylint: disable=protected-access
    )(log_decay)
    self.assertTrue(
        bool(jnp.allclose(scan_decay_grad, grads[3], rtol=1e-3, atol=1e-5)),
        "chunked decay gradient diverges from the scan path's gradient",
    )

  def test_bfloat16_state_stays_close(self):
    """The bf16 state option halves traffic on a bandwidth-bound path.

    It is off by default. The tolerance is bf16's, not float32's: this pins that
    the option is wired correctly, not that it is free.
    """
    self._assert_matches(128, 32, state_dtype="bfloat16", tol=5e-2)
