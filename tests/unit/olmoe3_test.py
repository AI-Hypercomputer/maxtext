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

# pylint: disable=missing-function-docstring,unnecessary-lambda-assignment,use-dict-literal,unbalanced-tuple-unpacking,cell-var-from-loop,redefined-outer-name,reimported,consider-using-from-import,line-too-long

import collections
import json
import unittest

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import quantizations
import maxtext.models.models as models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR, MAXTEXT_REPO_ROOT

import os

# (model_name, total params, active params) from the reference ladder.
CANONICAL_COUNTS = {
    "olmoe3-30m": (32_323_588, 29_964_292),
    "olmoe3-3p5b": (62_864_102_080, 3_475_903_168),
    "olmoe3-3p5b-e256k8": (62_850_798_272, 3_462_599_360),
    "olmoe3-3p5b-e256k8-sh2x": (63_130_178_240, 3_741_979_328),
    "olmoe3-3p5b-e256k8-sh4x": (63_688_938_176, 4_300_739_264),
    "olmo35-tiny": (12_496_341_632, 794_233_472),
    "olmo35-tiny-e256k8": (12_492_409_472, 790_301_312),
    "olmo35-tiny-e256k8-sh2x": (12_539_595_392, 837_487_232),
    "olmo35-tiny-e256k8-sh4x": (12_633_967_232, 931_859_072),
    "olmo35-small-e256k8-sh2x": (72_580_704_064, 4_123_371_328),
    "olmo35-gagik-fixed-e256k8-sh2x": (63_130_193_648, 3_741_994_736),
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


def _sub_jaxprs(params):
  """The jaxprs nested in an equation's params (Jaxpr/ClosedJaxpr values, also inside tuples, lists, dicts)."""
  stack = list(params.values())
  while stack:
    item = stack.pop()
    if isinstance(item, (tuple, list)):
      stack.extend(item)
    elif isinstance(item, dict):
      stack.extend(item.values())
    elif hasattr(item, "eqns") and hasattr(item, "outvars"):  # Jaxpr
      yield item
    elif hasattr(item, "jaxpr") and hasattr(item, "consts"):  # ClosedJaxpr
      yield item.jaxpr


def _pallas_kernel_kinds(jaxpr):
  """Counter of the token-major fused KDA kernels in `jaxpr` (recursively), told apart by their outputs.

  fwd: first output `[B, T, H * dv]` (rank 3); bwd: six gradients; wy: X|kk and scores (two `[BH, N, C, C]`);
  states: one `[B * H, N, dk, dv]` output.
  """
  kinds = collections.Counter()

  def walk(jx):
    for eqn in jx.eqns:
      if eqn.primitive.name == "pallas_call":
        shapes = [v.aval.shape for v in eqn.outvars]
        if len(shapes) == 6:
          kinds["bwd"] += 1
        elif len(shapes[0]) == 3:
          kinds["fwd"] += 1
        else:
          kinds["wy" if len(shapes) == 2 else "states"] += 1
      for sub in _sub_jaxprs(eqn.params):
        walk(sub)

  walk(jaxpr)
  return kinds


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

  def test_3p5b_e256k8_variants_total_parameters(self):
    for name in ("olmoe3-3p5b-e256k8", "olmoe3-3p5b-e256k8-sh2x", "olmoe3-3p5b-e256k8-sh4x"):
      self._assert_total(name)

  def test_tiny_and_e256k8_variants_total_parameters(self):
    for name in ("olmo35-tiny", "olmo35-tiny-e256k8", "olmo35-tiny-e256k8-sh2x", "olmo35-tiny-e256k8-sh4x"):
      self._assert_total(name)

  def test_small_e256k8_sh2x_total_parameters(self):
    self._assert_total("olmo35-small-e256k8-sh2x")

  def test_gagik_fixed_e256k8_sh2x_total_parameters(self):
    self._assert_total("olmo35-gagik-fixed-e256k8-sh2x")

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

  def test_parameter_initialization_matches_reference(self):
    """All OLMoE3 embeddings, projections, routers, and experts use std=0.02 truncated normal."""
    cfg = _config("olmoe3-30m")
    mesh, model = _build(cfg)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    with mesh:
      params = model.init({"params": jax.random.PRNGKey(0)}, tokens, tokens)["params"]

    flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
    self.assertTrue(flat_params, "no parameters found")

    checked = set()
    for path, leaf in flat_params:
      name = "/".join(str(getattr(k, "key", k)) for k in path if str(getattr(k, "key", k)).lstrip(".") != "value")
      arr = np.asarray(leaf, dtype=np.float32)
      if name.endswith(("norm/scale", "o_norm/scale", "query_norm/scale", "key_norm/scale")):
        np.testing.assert_allclose(arr, 1.0, atol=1e-6, err_msg=name)
        checked.add("norm")
      elif name.endswith("A_log"):
        self.assertTrue(np.all((arr >= 0.0) & (arr <= np.log(16.0) + 1e-6)), f"{name} out of [0, log(16)]")
        checked.add("A_log")
      elif name.endswith(("dt_bias", "bias")):
        np.testing.assert_allclose(arr, 0.0, atol=1e-6, err_msg=name)
        checked.add("bias")
      else:
        # Every embedding, projection, conv, router, and expert kernel uses
        # truncated_normal(stddev=0.02, lower=-3, upper=3) -> values in [-0.06, 0.06].
        self.assertLessEqual(float(np.max(np.abs(arr))), 0.06 + 1e-6, f"{name} exceeds 3*0.02 truncation bound")
        if arr.size >= 1024:
          self.assertAlmostEqual(float(np.std(arr)), 0.02, delta=0.004, msg=f"{name} std={np.std(arr):.5f}")
        if "token_embedder" in name:
          checked.add("token_embedder")
        elif "mixer/attention" in name:
          checked.add("self_attention")
        elif "logits_dense" in name:
          checked.add("logits_dense")
        elif "moe_block" in name:
          checked.add("moe_block")
        elif "mixer/w_q" in name or "mixer/q_conv" in name:
          checked.add("kda")

    self.assertEqual(
        checked,
        {"norm", "A_log", "bias", "token_embedder", "self_attention", "logits_dense", "moe_block", "kda"},
    )


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
    assert self.layer.moe_block is not None
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 8, self.cfg.num_experts), jnp.float32)
    weights, _ = self.layer.moe_block.get_topk(logits, None)
    self.assertTrue(
        jnp.allclose(weights.sum(axis=-1), self.cfg.num_experts_per_tok, atol=1e-3),
        f"top-k weights sum to {weights.sum(axis=-1).ravel()[0]}, expected {self.cfg.num_experts_per_tok}",
    )

  def test_load_balance_loss_matches_reference_formula(self):
    """The LB loss is the reference's (E/K)-scaled batch-level product, not Switch-style E^2."""
    assert self.layer.moe_block is not None
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
    assert self.layer.moe_block is not None
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
    assert layer.moe_block is not None
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

  def test_emo_pool_choices_sample_from_the_set(self):
    """emo_document_expert_pool_choices: every training pool draw is one of the listed sizes.

    The 30m config samples the range [16, 32]; with choices [16, 24, 32] only
    those three sizes may appear (17..23 and 25..31 no longer can), one size
    per document, and over enough documents all three do. The pool size is
    read back as the number of experts a document's tokens are left to route
    among.
    """
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import numpy as np
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.models.olmoe3 import OLMoE3DecoderLayer

    choices = [16, 24, 32]
    cfg = _config(
        "olmoe3-30m",
        extra=("override_model_config=True", f"emo_document_expert_pool_choices={choices}"),
    )
    self.assertEqual(list(cfg.emo_document_expert_pool_choices), choices)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      layer = OLMoE3DecoderLayer(cfg, mesh, MODEL_MODE_TRAIN, layer_idx=1, rngs=nnx.Rngs(params=0))
    assert layer.moe_block is not None
    batch, docs, doc_len = 4, 16, 4
    segs = jnp.broadcast_to(jnp.repeat(jnp.arange(1, docs + 1, dtype=jnp.int32), doc_len)[None], (batch, docs * doc_len))
    logits = jax.random.normal(jax.random.PRNGKey(5), (batch, docs * doc_len, cfg.num_experts), jnp.float32)
    masked = layer.moe_block._emo_mask_logits(logits, segs, jax.random.PRNGKey(6))  # pylint: disable=protected-access
    pools = np.asarray(jnp.isfinite(masked).sum(axis=-1))  # [b, s]: the pool size each token routes within
    self.assertTrue(set(np.unique(pools).tolist()) <= set(choices), f"pool sizes {np.unique(pools)}")
    self.assertEqual(set(np.unique(pools).tolist()), set(choices), f"pool sizes {np.unique(pools)}")
    per_doc = pools.reshape(batch, docs, doc_len)
    self.assertTrue(bool((per_doc == per_doc[..., :1]).all()), "pool size must be constant within a document")
    # Eval keeps the fixed eval pool regardless of the choices.
    masked_eval = layer.moe_block._emo_mask_logits(logits, segs, None)  # pylint: disable=protected-access
    self.assertTrue(bool((jnp.isfinite(masked_eval).sum(axis=-1) == cfg.emo_eval_document_expert_pool).all()))

  def test_emo_pool_probs_weight_the_choices(self):
    """emo_document_expert_pool_probs: documents draw each listed size at the given probability.

    1024 documents with probabilities (0.25, 0.25, 0.5): three sigma on each frequency is about 0.045.
    """
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import numpy as np
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.models.olmoe3 import OLMoE3DecoderLayer

    choices, probs = [16, 24, 32], [0.25, 0.25, 0.5]
    cfg = _config(
        "olmoe3-30m",
        extra=(
            "override_model_config=True",
            f"emo_document_expert_pool_choices={choices}",
            f"emo_document_expert_pool_probs={probs}",
        ),
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      layer = OLMoE3DecoderLayer(cfg, mesh, MODEL_MODE_TRAIN, layer_idx=1, rngs=nnx.Rngs(params=0))
    assert layer.moe_block is not None
    batch, docs, doc_len = 64, 16, 4
    segs = jnp.broadcast_to(jnp.repeat(jnp.arange(1, docs + 1, dtype=jnp.int32), doc_len)[None], (batch, docs * doc_len))
    logits = jax.random.normal(jax.random.PRNGKey(5), (batch, docs * doc_len, cfg.num_experts), jnp.float32)
    masked = layer.moe_block._emo_mask_logits(logits, segs, jax.random.PRNGKey(6))  # pylint: disable=protected-access
    per_doc = np.asarray(jnp.isfinite(masked).sum(axis=-1)).reshape(batch, docs, doc_len)
    self.assertTrue(bool((per_doc == per_doc[..., :1]).all()), "pool size must be constant within a document")
    doc_pools = per_doc[..., 0].ravel()
    self.assertTrue(set(np.unique(doc_pools).tolist()) <= set(choices), f"pool sizes {np.unique(doc_pools)}")
    freqs = [float((doc_pools == c).mean()) for c in choices]
    np.testing.assert_allclose(freqs, probs, atol=0.05)

  def test_emo_pool_choices_are_validated(self):
    """A choice below top_k or above num_experts, or probabilities that don't fit the choices, are rejected."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.models.olmoe3 import OLMoE3DecoderLayer

    bad_configs = [
        ("emo_document_expert_pool_choices=[8, 32]",),
        ("emo_document_expert_pool_choices=[16, 64]",),
        ("emo_document_expert_pool_choices=[16, 32]", "emo_document_expert_pool_probs=[0.5]"),  # length
        ("emo_document_expert_pool_choices=[16, 32]", "emo_document_expert_pool_probs=[0.5, 0.4]"),  # sum
        ("emo_document_expert_pool_choices=[16, 32]", "emo_document_expert_pool_probs=[1.5, -0.5]"),  # sign
    ]
    for bad in bad_configs:
      cfg = _config("olmoe3-30m", extra=("override_model_config=True", *bad))
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      with mesh, self.assertRaises(ValueError):
        OLMoE3DecoderLayer(cfg, mesh, MODEL_MODE_TRAIN, layer_idx=1, rngs=nnx.Rngs(params=0))

  def test_emo_threshold_keep_matches_double_argsort(self):
    """The one-sort EMo mask equals ``argsort(argsort(-s)) < pool``, ties included."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import numpy as np
    from maxtext.models.olmoe3 import OLMoE3LatentRoutedMoE

    rng = np.random.default_rng(0)
    for coarse in (False, True):
      scores = rng.random((2, 32, 64)).astype(np.float32)
      if coarse:  # heavy ties, including at the threshold
        scores = np.round(scores * 4) / 4
      pool = rng.integers(8, 65, (2, 32)).astype(np.int32)
      ref = jnp.argsort(jnp.argsort(-scores, axis=-1), axis=-1) < pool[..., None]
      got = OLMoE3LatentRoutedMoE._emo_keep_by_threshold(jnp.asarray(scores), jnp.asarray(pool))  # pylint: disable=protected-access
      self.assertTrue(bool((ref == got).all()))

  def test_emo_threshold_bisection_matches_sort(self):
    """The bisected EMo threshold keeps the same experts as the sorted one, ties and negatives included."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import numpy as np
    from maxtext.models.olmoe3 import OLMoE3LatentRoutedMoE

    rng = np.random.default_rng(1)
    for coarse in (False, True):
      scores = (rng.random((2, 32, 512)) - 0.1).astype(np.float32)
      if coarse:
        scores = np.round(scores * 4) / 4
      pool = rng.integers(16, 513, (2, 32)).astype(np.int32)
      ref = jnp.argsort(jnp.argsort(-scores, axis=-1), axis=-1) < pool[..., None]
      got = OLMoE3LatentRoutedMoE._emo_keep_by_threshold(jnp.asarray(scores), jnp.asarray(pool), bisect=True)  # pylint: disable=protected-access
      self.assertTrue(bool((ref == got).all()))

  def test_lean_routing_is_exact(self):
    """``moe_lean_routing`` changes no value or gradient on the sparse-matmul path."""
    pool, seq_len = 20, 32
    outs = []
    for lean in (False, True):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "override_model_config=True",
              f"emo_min_document_expert_pool={pool}",
              f"emo_max_document_expert_pool={pool}",
              f"emo_eval_document_expert_pool={pool}",
              "sparse_matmul=True",
              "remat_policy=custom",
              "moe_routing=device",
              f"moe_lean_routing={lean}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.concatenate([jnp.ones((1, 12), jnp.int32), jnp.full((1, seq_len - 12), 2, jnp.int32)], axis=1)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    self.assertEqual(float(v0), float(v1))
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      self.assertTrue(bool(jnp.array_equal(a, b)))

  def test_a2a_token_chunks_match_unchunked(self):
    """``moe_a2a_token_chunks`` routes once over the whole sequence, so chunking changes no routing.

    The second document spans both chunks, so per-chunk EMo pools would differ; only the gradient sums over
    the chunks reassociate.
    """
    pool, seq_len = 20, 32
    outs = []
    for chunks in (1, 2):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "override_model_config=True",
              f"emo_min_document_expert_pool={pool}",
              f"emo_max_document_expert_pool={pool}",
              f"emo_eval_document_expert_pool={pool}",
              "sparse_matmul=True",
              "moe_lean_routing=True",
              f"moe_a2a_token_chunks={chunks}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.concatenate([jnp.ones((1, 12), jnp.int32), jnp.full((1, seq_len - 12), 2, jnp.int32)], axis=1)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    np.testing.assert_allclose(float(v0), float(v1), rtol=1e-6)
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)

  def test_topk_by_bisection_matches_lax_top_k(self):
    """Bisection top-k picks the same expert set as ``lax.top_k``, in ascending index order."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from maxtext.layers import moe as moe_lib

    logits = jax.random.normal(jax.random.PRNGKey(3), (4, 64, 512), jnp.float32)
    logits = jnp.where(jax.random.uniform(jax.random.PRNGKey(4), logits.shape) < 0.5, logits, -jnp.inf)
    _, ref = jax.lax.top_k(logits, 16)
    got = moe_lib.mask_to_indices(moe_lib.keep_top_by_bisection(logits, 16), 16)
    self.assertTrue(bool((jnp.sort(ref, axis=-1) == got).all()))

  def test_pallas_topk_matches_lax_top_k(self):
    """The Pallas top-k kernel returns ``lax.top_k`` indices exactly, ties and -inf included."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from maxtext.kernels import topk as topk_kernel

    logits = jax.random.normal(jax.random.PRNGKey(3), (3, 100, 512), jnp.float32)
    ties = jnp.round(logits * 2) / 2
    sparse = jnp.where(jax.random.uniform(jax.random.PRNGKey(4), logits.shape) < 0.02, logits, -jnp.inf)
    signed_zero = jnp.where(logits > 0, 0.0, -0.0)
    for x in (logits, ties, sparse, signed_zero, logits.astype(jnp.bfloat16)):
      _, ref = jax.lax.top_k(x, 16)
      got = topk_kernel.topk_indices(x, 16, block_tokens=128, interpret=True)
      self.assertTrue(bool((ref == got).all()))

  def test_pallas_topk_routing_matches_lean(self):
    """moe_topk_pallas picks the same experts in the same order, so values and grads are bit-identical."""
    seq_len = 32
    outs = []
    for pallas in (False, True):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "sparse_matmul=True",
              "moe_routing=device",
              "moe_lean_routing=True",
              f"moe_topk_pallas={pallas}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.ones((1, seq_len), jnp.int32)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    self.assertEqual(float(v0), float(v1))
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      self.assertTrue(bool(jnp.array_equal(a, b)))

  def test_bisection_routing_matches_lean(self):
    """The two bisection flags only reorder the combine sum: values and grads match lean routing closely."""
    pool, seq_len = 20, 32
    outs = []
    for bisect in (False, True):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "override_model_config=True",
              f"emo_min_document_expert_pool={pool}",
              f"emo_max_document_expert_pool={pool}",
              f"emo_eval_document_expert_pool={pool}",
              "sparse_matmul=True",
              "remat_policy=custom",
              "moe_routing=device",
              "moe_lean_routing=True",
              f"emo_threshold_by_bisection={bisect}",
              f"moe_topk_by_bisection={bisect}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.concatenate([jnp.ones((1, 12), jnp.int32), jnp.full((1, seq_len - 12), 2, jnp.int32)], axis=1)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    self.assertAlmostEqual(float(v0), float(v1), delta=1e-4 * abs(float(v0)))
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      # Tolerance scales with the leaf's largest entry: reordering the combine sum perturbs near-zero entries.
      self.assertLessEqual(float(jnp.max(jnp.abs(a - b))), 1e-3 * float(jnp.max(jnp.abs(a))) + 1e-5)


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

  def _layer_out(self, use_tokamax, x, segment_ids, dt_bias=None, extra=()):
    """Run one KDA layer, fused or unfused, and return its output."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention

    cfg = _config("olmoe3-30m", extra=(f"use_tokamax_kda={use_tokamax}",) + tuple(extra))
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh:
      layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
      if dt_bias is not None:
        layer.dt_bias.value = jnp.full_like(layer.dt_bias.value, dt_bias)
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

  def test_matches_unfused_at_large_decay(self):
    """Fused KDA stays finite and exact when steps decay past fp32 exp range.

    dt_bias=3 puts per-step log-decay near A*3 nats, up to ~48 at A=16. With the
    gate activated in-kernel this returned NaN on hardware and a 2.4e-2 error in
    interpret mode; the log-decay floor must fix it without moving the answer.
    """
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 128), jnp.float32)
    expected = self._layer_out(False, x, None, dt_bias=3.0)
    actual = self._layer_out(True, x, None, dt_bias=3.0)
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))), "tokamax KDA returned non-finite output")
    rel = jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))
    self.assertLess(float(rel), 2e-3, f"tokamax KDA relative error {rel:.2e}")

  def test_matches_unfused_unpacked(self):
    """Fused KDA matches the unfused rule on a single document."""
    self._compare(None)

  def test_matches_unfused_packed(self):
    """Fused KDA matches the unfused rule across a packed-document boundary."""
    # 1-indexed ids with the boundary mid-sequence (MaxText packing convention).
    segment_ids = jnp.concatenate([jnp.full((2, 30), 1, jnp.int32), jnp.full((2, 34), 2, jnp.int32)], axis=1)
    self._compare(segment_ids)

  def test_resets_in_gate_matches_unfused_packed(self):
    """Gate resets reproduce the exact packed rule, with the boundary mid-chunk."""
    segment_ids = jnp.concatenate([jnp.full((2, 30), 1, jnp.int32), jnp.full((2, 34), 2, jnp.int32)], axis=1)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 128), jnp.float32)
    expected = self._layer_out(False, x, segment_ids)
    # Floor 10 is the production setting and the loosest one: the old state leaks at exp(-10).
    extra = ("tokamax_kda_resets_in_gate=True", "tokamax_kda_log_decay_floor=10.0")
    actual = self._layer_out(True, x, segment_ids, extra=extra)
    rel = jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))
    self.assertLess(float(rel), 2e-3, f"gate-reset KDA relative error {rel:.2e}")

  def test_resets_in_gate_grads_match_varlen(self):
    """Gate resets give the varlen path's loss and input gradients."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention

    segment_ids = jnp.concatenate(
        [jnp.full((2, 30), 1, jnp.int32), jnp.full((2, 20), 2, jnp.int32), jnp.full((2, 14), 3, jnp.int32)], axis=1
    )
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 128), jnp.float32)
    results = []
    for in_gate in (False, True):
      extra = ("use_tokamax_kda=True", "tokamax_kda_log_decay_floor=10.0", f"tokamax_kda_resets_in_gate={in_gate}")
      cfg = _config("olmoe3-30m", extra=extra)
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      with mesh:
        layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
        loss = lambda x_, layer=layer: jnp.sum(layer(x_, segment_ids) ** 2)
        out, grad = jax.jit(jax.value_and_grad(loss))(x)
      results.append((float(out), jax.device_get(grad)))
    (out_varlen, grad_varlen), (out_gate, grad_gate) = results
    self.assertAlmostEqual(out_varlen, out_gate, delta=1e-4 * abs(out_varlen))
    rel = jnp.max(jnp.abs(grad_gate - grad_varlen)) / jnp.max(jnp.abs(grad_varlen))
    self.assertLess(float(rel), 1e-3, f"gate-reset grad relative error {rel:.2e}")

  def test_l2norm_outside_matches_in_kernel(self):
    """The MaxText-side q/k L2-norm gives the kernel's output and input grads."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 64, 128), jnp.float32)
    results = []
    for outside in (False, True):
      cfg = _config("olmoe3-30m", extra=("use_tokamax_kda=True", f"tokamax_kda_l2norm_outside={outside}"))
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      with mesh:
        layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
        out, grad = jax.jit(jax.value_and_grad(lambda x_, layer=layer: jnp.sum(layer(x_, None) ** 2)))(x)
      results.append((float(out), jax.device_get(grad)))
    (out_in, grad_in), (out_out, grad_out) = results
    self.assertAlmostEqual(out_in, out_out, delta=1e-5 * abs(out_in))
    rel = jnp.max(jnp.abs(grad_out - grad_in)) / jnp.max(jnp.abs(grad_in))
    self.assertLess(float(rel), 1e-4, f"l2norm-outside grad relative error {rel:.2e}")


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


class OLMoE3SubblockDeltaRuleTest(unittest.TestCase):
  """``kda_chunked_impl=subblock`` must be the scan too, forward and backward.

  The sub-block factorization, the block-doubling inverse with its hand-written
  VJP, and the WY products moved out of the scan are each a place where a
  plausible-looking mistake still trains, so every one is pinned to the scan.
  """

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_matmul_precision", "highest")
    self.addCleanup(jax.config.update, "jax_default_matmul_precision", None)

  def _inputs(self, seq_len, resets_at=(), strong=False):
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    batch, heads, dk, dv = 2, 3, 16, 32
    q = jax.random.normal(keys[0], (batch, seq_len, heads, dk)) * 0.5
    k = jax.random.normal(keys[1], (batch, seq_len, heads, dk)) * 0.5
    v = jax.random.normal(keys[2], (batch, seq_len, heads, dv))
    if strong:  # per-step log-decay up to ~-50 nats, the reference init's range
      log_decay = -jnp.exp(jax.random.uniform(keys[3], q.shape, minval=-3.0, maxval=3.9))
    else:
      log_decay = jnp.log(jax.nn.sigmoid(jax.random.normal(keys[3], q.shape)) * 0.4 + 0.55)
    beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq_len, heads)))
    resets = jnp.zeros((batch, seq_len), bool)
    for idx in resets_at:
      resets = resets.at[:, idx].set(True)
    return q, k, v, log_decay, beta, resets

  def _rules(self, chunk, sub_block, resets):
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.models import olmoe3

    def scan(q, k, v, log_decay, beta):
      return olmoe3._delta_rule_scan(q, k, v, jnp.exp(log_decay), beta, resets)

    def subblock(q, k, v, log_decay, beta):
      return olmoe3._delta_rule_chunked_subblock(q, k, v, log_decay, beta, resets, chunk, "float32", sub_block)

    return scan, jax.jit(subblock)

  def _assert_matches(self, seq_len, chunk, sub_block=16, resets_at=(), strong=False, grads=False):
    args = self._inputs(seq_len, resets_at, strong)
    scan, subblock = self._rules(chunk, sub_block, args[-1])
    expected, actual = scan(*args[:-1]), subblock(*args[:-1])
    self.assertTrue(bool(jnp.isfinite(actual).all()), "sub-block delta rule returned non-finite output")
    rel = jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))
    self.assertLess(float(rel), 2e-5, f"chunk={chunk} sub={sub_block}: rel err {rel:.2e}")
    if grads:
      w = jax.random.normal(jax.random.PRNGKey(9), expected.shape)
      g_ref = jax.grad(lambda *a: jnp.sum(scan(*a) * w), argnums=(0, 1, 2, 3, 4))(*args[:-1])
      g_sub = jax.grad(lambda *a: jnp.sum(subblock(*a) * w), argnums=(0, 1, 2, 3, 4))(*args[:-1])
      for name, a, b in zip(("q", "k", "v", "log_decay", "beta"), g_ref, g_sub):
        self.assertTrue(bool(jnp.isfinite(b).all()), f"non-finite d{name}")
        rel = jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))
        self.assertLess(float(rel), 1e-4, f"d{name}: rel err {rel:.2e}")

  def test_matches_scan_across_chunk_and_sub_block_sizes(self):
    for chunk, sub_block in ((16, 16), (32, 16), (64, 16), (64, 8)):
      self._assert_matches(128, chunk, sub_block, resets_at=(45, 64))

  def test_gradients_match_scan_with_packed_documents(self):
    # Boundaries mid-chunk, on a chunk edge, and at the first and last token of a chunk.
    self._assert_matches(256, 64, resets_at=(0, 5, 63, 64, 100, 128), grads=True)

  def test_reference_scale_decay_stays_finite(self):
    """Pivot factors are exp of non-positive numbers, so strong decay cannot overflow."""
    self._assert_matches(256, 64, resets_at=(17,), strong=True, grads=True)

  def test_pallas_state_scan_matches_lax_scan(self):
    """``kda_pallas_scan``: the Pallas chunk-state kernel (interpret mode here) equals the bf16 lax.scan.

    The forward runs the same bf16 arithmetic; the hand-written backward rounds its cotangents at different
    points than autodiff, so gradients agree to bf16 accuracy only (on v4 both sit 0.5-0.8% from an f32
    reference, the kernel slightly closer).
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.kernels.kda import scan as kda_scan
    from maxtext.models import olmoe3

    args = self._inputs(256, resets_at=(0, 5, 63, 64, 100, 128))
    resets = args[-1]

    def pallas_scan(uw, kc, dl):
      b, h, n = uw.shape[:3]
      flat = lambda t: t.reshape(b * h, n, *t.shape[3:])
      states, delta = kda_scan.kda_state_scan(flat(uw), flat(kc), flat(dl)[..., None], True)
      return states.reshape(b, h, n, *states.shape[2:]), delta.reshape(b, h, n, *delta.shape[2:])

    def rule(state_scan):
      def f(q, k, v, log_decay, beta):
        return olmoe3._delta_rule_chunked_subblock(
            q, k, v, log_decay, beta, resets, 64, "bfloat16", 16, state_scan=state_scan
        ).astype(jnp.float32)

      return jax.jit(f)

    ref, ker = rule(None), rule(pallas_scan)
    # Bit-identical on TPU; the CPU interpreter reorders a few f32 sums.
    np.testing.assert_allclose(np.asarray(ref(*args[:-1])), np.asarray(ker(*args[:-1])), rtol=1e-5, atol=1e-6)
    w = jax.random.normal(jax.random.PRNGKey(9), (2, 256, 3, 32))
    g_ref = jax.grad(lambda *a: jnp.sum(ref(*a) * w), argnums=(0, 1, 2, 3, 4))(*args[:-1])
    g_ker = jax.grad(lambda *a: jnp.sum(ker(*a) * w), argnums=(0, 1, 2, 3, 4))(*args[:-1])
    for name, a, b in zip(("q", "k", "v", "log_decay", "beta"), g_ref, g_ker):
      rel = jnp.linalg.norm(a - b) / jnp.linalg.norm(a)
      self.assertLess(float(rel), 2e-2, f"d{name}: rel err {rel:.2e}")

  def test_block_doubling_inverse_and_vjp(self):
    """``_invert_unit_lower`` equals the triangular solve, and so does its gradient."""
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.models import olmoe3

    for c in (16, 32, 64, 128):
      keys = jax.random.split(jax.random.PRNGKey(c), 2)
      m = jnp.tril(jax.random.normal(keys[0], (2, 3, c, c)) * 0.2, k=-1)
      eye = jnp.eye(c)

      def solve(m_):
        return jax.scipy.linalg.solve_triangular(eye + m_, jnp.broadcast_to(eye, m_.shape), lower=True)

      x = olmoe3._invert_unit_lower(m)
      rel = jnp.max(jnp.abs(x - solve(m))) / jnp.max(jnp.abs(solve(m)))
      self.assertLess(float(rel), 1e-5, f"c={c}: inverse rel err {rel:.2e}")
      w = jax.random.normal(keys[1], m.shape)
      g_ref = jnp.tril(jax.grad(lambda m_: jnp.sum(solve(m_) * w))(m), k=-1)
      g_new = jax.grad(lambda m_: jnp.sum(olmoe3._invert_unit_lower(m_) * w))(m)
      rel = jnp.max(jnp.abs(g_new - g_ref)) / jnp.max(jnp.abs(g_ref))
      self.assertLess(float(rel), 1e-4, f"c={c}: inverse VJP rel err {rel:.2e}")

  def test_block_doubling_inverse_similar_keys(self):
    """Real-key-like M (near-identical unit keys, beta up to 2): the inverse stays finite and exact.

    Here the Neumann partial sums of (-M)^i that a power-series inverse builds grow far past the answer in
    fp32; exact block doubling only ever forms exact inverses of diagonal blocks.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.models import olmoe3

    c, d = 128, 64
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    shared = jax.random.normal(keys[0], (1, 1, 1, d))
    k = shared + 0.05 * jax.random.normal(keys[1], (2, 3, c, d))
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    beta = 2.0 * jax.random.uniform(keys[2], (2, 3, c, 1), maxval=0.999)
    pos = jnp.arange(c)
    decay = jnp.exp(-0.01 * jnp.maximum(pos[:, None] - pos[None, :], 0))
    m = jnp.tril(beta * jnp.einsum("...id,...jd->...ij", k, k) * decay, k=-1)
    eye = jnp.eye(c)
    ref = jax.scipy.linalg.solve_triangular(eye + m, jnp.broadcast_to(eye, m.shape), lower=True)
    x = olmoe3._invert_unit_lower(m)
    self.assertTrue(bool(jnp.all(jnp.isfinite(x))), "inverse has non-finite entries")
    # Residual of (I + M) X = I, relative to the size of X (the inverse itself is large here).
    resid = jnp.max(jnp.abs((eye + m) @ x - eye)) / jnp.max(jnp.abs(ref))
    self.assertLess(float(resid), 1e-4, f"similar keys: residual {resid:.2e}")

  def test_fused_swiglu_gmm_matches_unfused(self):
    """``moe_fused_swiglu_gmm`` (kernels/latent_moe, Pallas interpret on CPU) matches the unfused sparse path."""
    pool, seq_len = 20, 32
    outs = []
    for fused in (False, True):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "override_model_config=True",
              f"emo_min_document_expert_pool={pool}",
              f"emo_max_document_expert_pool={pool}",
              f"emo_eval_document_expert_pool={pool}",
              "sparse_matmul=True",
              f"megablox={fused}",
              f"moe_fused_swiglu_gmm={fused}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.concatenate([jnp.ones((1, 12), jnp.int32), jnp.full((1, seq_len - 12), 2, jnp.int32)], axis=1)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    np.testing.assert_allclose(float(v0), float(v1), rtol=1e-4)
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      a, b = np.asarray(a), np.asarray(b)
      np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3 * max(float(np.max(np.abs(a))), 1e-6))

  def test_topk_by_bisection_matches_lax_top_k(self):
    """Bisection top-k picks the same expert set as ``lax.top_k``, in ascending index order."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from maxtext.layers import moe as moe_lib

    logits = jax.random.normal(jax.random.PRNGKey(3), (4, 64, 512), jnp.float32)
    logits = jnp.where(jax.random.uniform(jax.random.PRNGKey(4), logits.shape) < 0.5, logits, -jnp.inf)
    _, ref = jax.lax.top_k(logits, 16)
    got = moe_lib.mask_to_indices(moe_lib.keep_top_by_bisection(logits, 16), 16)
    self.assertTrue(bool((jnp.sort(ref, axis=-1) == got).all()))

  def test_pallas_topk_matches_lax_top_k(self):
    """The Pallas top-k kernel returns ``lax.top_k`` indices exactly, ties and -inf included."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from maxtext.kernels import topk as topk_kernel

    logits = jax.random.normal(jax.random.PRNGKey(3), (3, 100, 512), jnp.float32)
    ties = jnp.round(logits * 2) / 2
    sparse = jnp.where(jax.random.uniform(jax.random.PRNGKey(4), logits.shape) < 0.02, logits, -jnp.inf)
    signed_zero = jnp.where(logits > 0, 0.0, -0.0)
    for x in (logits, ties, sparse, signed_zero, logits.astype(jnp.bfloat16)):
      _, ref = jax.lax.top_k(x, 16)
      got = topk_kernel.topk_indices(x, 16, block_tokens=128, interpret=True)
      self.assertTrue(bool((ref == got).all()))

  def test_pallas_topk_routing_matches_lean(self):
    """moe_topk_pallas picks the same experts in the same order, so values and grads are bit-identical."""
    seq_len = 32
    outs = []
    for pallas in (False, True):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "sparse_matmul=True",
              "moe_routing=device",
              "moe_lean_routing=True",
              f"moe_topk_pallas={pallas}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.ones((1, seq_len), jnp.int32)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    self.assertEqual(float(v0), float(v1))
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      self.assertTrue(bool(jnp.array_equal(a, b)))

  def test_bisection_routing_matches_lean(self):
    """The two bisection flags only reorder the combine sum: values and grads match lean routing closely."""
    pool, seq_len = 20, 32
    outs = []
    for bisect in (False, True):
      cfg = _config(
          "olmoe3-30m",
          max_target_length=seq_len,
          extra=(
              "override_model_config=True",
              f"emo_min_document_expert_pool={pool}",
              f"emo_max_document_expert_pool={pool}",
              f"emo_eval_document_expert_pool={pool}",
              "sparse_matmul=True",
              "remat_policy=custom",
              "moe_routing=device",
              "moe_lean_routing=True",
              f"emo_threshold_by_bisection={bisect}",
              f"moe_topk_by_bisection={bisect}",
          ),
      )
      mesh, model = _build(cfg)
      tokens = (jnp.arange(seq_len, dtype=jnp.int32)[None, :] * 7) % 101 + 1
      positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      segments = jnp.concatenate([jnp.ones((1, 12), jnp.int32), jnp.full((1, seq_len - 12), 2, jnp.int32)], axis=1)
      with mesh:
        params = model.init({"params": jax.random.PRNGKey(0)}, tokens, positions, segments)

        def loss(p, model=model):
          out = model.apply(p, tokens, positions, segments, enable_dropout=False)
          logits = out[0] if isinstance(out, tuple) else out
          return jnp.sum(jnp.sin(logits))

        outs.append(jax.value_and_grad(loss)(params))
    (v0, g0), (v1, g1) = outs
    self.assertAlmostEqual(float(v0), float(v1), delta=1e-4 * abs(float(v0)))
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
      # Tolerance scales with the leaf's largest entry: reordering the combine sum perturbs near-zero entries.
      self.assertLessEqual(float(jnp.max(jnp.abs(a - b))), 1e-3 * float(jnp.max(jnp.abs(a))) + 1e-5)


try:  # pylint: disable=g-import-not-at-top
  from tokamax._src.ops.experimental.kda import api as _tokamax_kda_api  # noqa: F401  pylint: disable=unused-import

  _HAVE_TOKAMAX_KDA = True
except ImportError:
  _HAVE_TOKAMAX_KDA = False


@unittest.skipUnless(_HAVE_TOKAMAX_KDA, "tokamax build without the KDA op (openxla/tokamax PR #1103)")
class OLMoE3FusedInputProjTest(unittest.TestCase):
  """``kda_fused_input_proj`` is the six separate projections, with the same parameters."""

  def test_matches_separate_projections(self):
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 128, 128), jnp.float32)
    segments = jnp.concatenate([jnp.full((2, 50), 1, jnp.int32), jnp.full((2, 78), 2, jnp.int32)], axis=1)
    results = []
    for fused in (False, True):
      cfg = _config("olmoe3-30m", extra=("use_tokamax_kda=False", f"kda_fused_input_proj={fused}"))
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      with mesh:
        layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
        graphdef, state = nnx.split(layer)

        def loss(state_, x_, graphdef=graphdef):
          return jnp.sum(nnx.merge(graphdef, state_)(x_, segments) ** 2)

        value, grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(state, x)
      leaves = jax.tree_util.tree_leaves(nnx.to_pure_dict(grads[0])) + [grads[1]]
      results.append((float(value), leaves))
    (v_sep, g_sep), (v_fused, g_fused) = results
    self.assertAlmostEqual(v_sep, v_fused, delta=1e-6 * abs(v_sep))
    for a, b in zip(g_sep, g_fused):
      # dt_bias and A_log gradients are cancellation-heavy sums, so accumulation order shows up there first.
      rel = jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-30)
      self.assertLess(float(rel), 2e-3, f"fused projection grad rel err {rel:.2e}")


class OLMoE3NanochatInitTest(unittest.TestCase):
  """`olmo_init_method=nanochat` against OLMo-core's `InitMethod.nanochat` rules.

  The expected (fan_in, fan_out) of each parameter is written out from the reference's rules
  (OLMo-core `tanushy/parameterization-ladders`, docs/NANOCHAT_INIT_HANDOFF.md section 1),
  by name and the model's widths, not from the kernel shapes the implementation reads.
  """

  # Tiny sh2x widths, with fewer layers and experts and a smaller vocab so the init fits on CPU.
  D, KEY, VALUE, HEAD_V, HEADS, CONV = 1024, 1024, 2048, 256, 8, 4
  Q_DIM, KV_DIM, LATENT, MOE_H, SHARED_H, DENSE_H, EXPERTS, VOCAB = 1024, 512, 512, 2048, 2048, 8192, 8, 8192
  # std of a normal truncated at +/-3 std, relative to the untruncated std.
  TRUNC = 0.986534

  def _params(self, init_method):
    cfg = _config(
        "olmo35-tiny-e256k8-sh2x",
        extra=(
            "override_model_config=True",
            "base_num_decoder_layers=3",  # 0: dense KDA, 1: attention + MoE, 2: KDA + MoE
            "inhomogeneous_layer_cycle_interval=2",
            f"num_experts={self.EXPERTS}",
            "num_experts_per_tok=2",
            f"vocab_size={self.VOCAB}",
            "emo_enabled=False",
            f"olmo_init_method={init_method}",
        ),
    )
    _, model = _build(cfg)
    rng = jax.random.PRNGKey(0)
    ids = jnp.ones((1, 64), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False)["params"]
    flat = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
      name = "/".join(str(getattr(k, "key", k)) for k in path).replace("/.value", "")
      flat[name] = np.asarray(leaf, np.float64)
    return flat

  def _expected(self, name):
    """("matrix", fan_in, fan_out) | ("conv",) | ("embedding",) | ("ones",) | ("zeros",) | ("a_log",)."""
    d = self.D
    rules = {
        "mixer/w_q/kernel": (d, self.KEY),
        "mixer/w_k/kernel": (d, self.KEY),
        "mixer/w_v/kernel": (d, self.VALUE),
        "mixer/f_proj_1/kernel": (d, self.HEAD_V),
        "mixer/f_proj_2/kernel": (self.HEAD_V, self.KEY),
        "mixer/g_proj_1/kernel": (d, self.HEAD_V),
        "mixer/g_proj_2/kernel": (self.HEAD_V, self.VALUE),
        "mixer/w_b/kernel": (d, self.HEADS),
        "mixer/w_out/kernel": (self.VALUE, d),
        # The fused query kernel holds the reference's w_q and w_g, each d -> q_dim.
        "attention/query/kernel": (d, self.Q_DIM),
        "attention/key/kernel": (d, self.KV_DIM),
        "attention/value/kernel": (d, self.KV_DIM),
        "attention/out/kernel": (self.Q_DIM, d),
        "latent_down/kernel": (d, self.LATENT),
        "latent_up/kernel": (self.LATENT, d),
        "moe_block/gate/kernel": (d, self.EXPERTS),
        # Per expert; wi_0/wi_1 are the halves of the reference's fused w_up_gate.
        "moe_block/wi_0": (self.LATENT, self.MOE_H),
        "moe_block/wi_1": (self.LATENT, self.MOE_H),
        "moe_block/wo": (self.MOE_H, self.LATENT),
        "logits_dense/kernel": (d, self.VOCAB),
    }
    for suffix, (fan_in, fan_out) in rules.items():
      if name.endswith(suffix):
        return ("matrix", fan_in, fan_out)
    if "/shared_ffn/" in name:
      hidden = self.DENSE_H if name.startswith("decoder/layers_0/") else self.SHARED_H
      return ("matrix", hidden, d) if name.endswith("wo/kernel") else ("matrix", d, hidden)
    if name.endswith(("q_conv", "k_conv", "v_conv")):
      return ("conv",)
    if name == "token_embedder/embedding":
      return ("embedding",)
    if name.endswith(("/scale", "ssmax_scale")):
      return ("ones",)
    if name.endswith(("dt_bias", "g_proj_2/bias")):
      return ("zeros",)
    if name.endswith("A_log"):
      return ("a_log",)
    raise AssertionError(f"no reference rule for {name}")

  def _check_normal(self, name, w, std, truncated=True):
    rel = w.std() / (std * (self.TRUNC if truncated else 1.0))
    self.assertLess(abs(rel - 1.0), 0.05, f"{name}: std {w.std():.5f}, expected {std:.5f}")
    self.assertLess(abs(w.mean()), 0.05 * std + 3 * std / np.sqrt(w.size), name)
    if truncated:
      self.assertLessEqual(np.abs(w).max(), 3 * std * (1 + 1e-6), f"{name}: not truncated at 3 std")

  def _check_fixed(self, name, w, kind):
    if kind == "ones":
      np.testing.assert_array_equal(w, 1.0, err_msg=name)
    elif kind == "zeros":
      np.testing.assert_array_equal(w, 0.0, err_msg=name)
    else:  # A_log = log(U[1, 16))
      self.assertTrue(np.all((w >= 0.0) & (w <= np.log(16.0))), name)

  def test_nanochat_std(self):
    # OLMo-core's test_nanochat_std cases.
    for fan_in, fan_out, expected in ((256, 256, 1 / 16), (256, 1024, 1 / 16), (1024, 256, 1 / 64), (1024, 8, 8**0.5 / 1024)):
      self.assertAlmostEqual(olmoe3_module().nanochat_std(fan_in, fan_out), expected, places=12)

  def test_nanochat_init_follows_the_reference_rules(self):
    params = self._params("nanochat")
    for name, w in params.items():
      kind = self._expected(name)
      if kind[0] == "matrix":
        _, fan_in, fan_out = kind
        self._check_normal(name, w, fan_in**-0.5 * min(1.0, (fan_out / fan_in) ** 0.5))
      elif kind[0] == "conv":
        self._check_normal(name, w, self.CONV**-0.5)
      elif kind[0] == "embedding":
        self._check_normal(name, w, 1.0, truncated=False)
        self.assertGreater(np.abs(w).max(), 3.0, "the embedding init is not truncated")
      else:
        self._check_fixed(name, w, kind[0])

  def test_normal_init_is_std_0p02_everywhere(self):
    params = self._params("normal")
    for name, w in params.items():
      kind = self._expected(name)
      if kind[0] in ("matrix", "conv", "embedding"):
        self._check_normal(name, w, 0.02)
      else:
        self._check_fixed(name, w, kind[0])

  def test_nanochat_init_matches_olmo_core(self):
    """Every parameter against the std OLMo-core's InitMethod.nanochat gives it, on the same reduced hero.

    The golden file holds, per MaxText parameter (names from OLMo-core's MaxText checkpoint converter), the init
    OLMo-core's code applied; see its _source. Values can't match bitwise (different RNGs), so this checks the
    distribution: kind, std, truncation, and that both models have exactly the same parameters and shapes.
    """
    with open(os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets", "olmoe3_nanochat_init_olmo_core.json"), encoding="utf-8") as f:
      golden = json.load(f)["params"]
    cfg = _config(
        "olmo35-tiny",
        extra=(
            "override_model_config=True",
            "base_num_decoder_layers=8",
            "num_experts=16",
            "num_experts_per_tok=2",
            "vocab_size=8192",
            "emo_enabled=False",
            "olmo_init_method=nanochat",
        ),
    )
    _, model = _build(cfg)
    rng = jax.random.PRNGKey(0)
    ids = jnp.ones((1, 64), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False)["params"]
    flat = {
        "/".join(str(getattr(k, "key", k)) for k in path).replace("/.value", ""): np.asarray(leaf, np.float64)
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    }
    self.assertEqual(sorted(flat), sorted(golden))
    for name, w in flat.items():
      self.assertEqual(list(w.shape), golden[name]["shape"], name)
      kind, *args = golden[name]["init"]
      if kind == "trunc_normal":
        self._check_normal(name, w, args[0])
      elif kind == "normal":
        self._check_normal(name, w, args[0], truncated=False)
      elif kind == "log_uniform":
        self.assertEqual(tuple(args), (1, 16), name)
        self._check_fixed(name, w, "a_log")
      else:
        self._check_fixed(name, w, kind)

  def test_nanochat_query_kernel_uses_the_query_width(self):
    # With q_dim = d (the tiny widths above) the query and query+gate fan-outs give the same std. Here
    # q_dim < d, so the kernel is contracting and counting the gate half would raise the std by sqrt(2).
    olmoe3 = olmoe3_module()
    cfg = collections.namedtuple("Cfg", "olmo_init_method head_dim")("nanochat", 64)
    init = olmoe3._olmoe3_attention_init(cfg)  # pylint: disable=protected-access
    d, heads = 2048, 4  # q_dim = 256
    for shape, in_axis, out_axis, fan_in, fan_out in (
        ((d, heads, 2 * 64), (0,), (1, 2), d, heads * 64),  # query + gate
        ((d, heads, 64), (0,), (1, 2), d, heads * 64),  # key / value
        ((heads * 64, d), (0,), (1,), heads * 64, d),  # out
    ):
      w = np.asarray(init(jax.random.PRNGKey(1), shape, jnp.float32, in_axis, out_axis), np.float64)
      self._check_normal(str(shape), w, olmoe3.nanochat_std(fan_in, fan_out))

  def test_nanochat_rejects_fused_gate_and_up_kernels(self):
    for flag in ("fused_mlp=True", "prefuse_moe_weights=True"):
      cfg = _config(
          "olmo35-tiny-e256k8-sh2x",
          extra=("override_model_config=True", "base_num_decoder_layers=2", "olmo_init_method=nanochat", flag),
      )
      _, model = _build(cfg)
      with self.assertRaisesRegex(ValueError, "nanochat"):
        _abstract_params(cfg, model, 64)


def olmoe3_module():
  from maxtext.models import olmoe3  # pylint: disable=import-outside-toplevel

  return olmoe3


class OLMoE3SharedExpertWidthTest(unittest.TestCase):
  """`shared_expert_mlp_dim` decouples the shared FFN from the routed experts.

  Before this knob existed both were driven by ``moe_mlp_dim``, so every ablation
  that reshaped the routed experts silently resized the shared expert too and was
  not a single-variable change. These tests pin the two properties that makes it
  safe to rely on: unset it is exactly the old behaviour, and set it moves the
  shared FFN and nothing else.
  """

  @staticmethod
  def _shapes(cfg):
    _, model = _build(cfg)
    params = _abstract_params(cfg, model, 64)
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
      out["/".join(str(getattr(k, "key", k)) for k in path)] = leaf.shape
    return out

  def test_unset_defaults_to_routed_expert_width(self):
    """The default has to be the routed width or existing configs change silently."""
    cfg = _config("olmoe3-3p5b")
    self.assertEqual(cfg.shared_expert_mlp_dim, cfg.moe_mlp_dim)

  def test_identity_point_is_bit_identical(self):
    """Setting the knob to the value it defaults to must change nothing at all."""
    base = self._shapes(_config("olmoe3-30m"))
    cfg = _config("olmoe3-30m")
    pinned = self._shapes(
        _config(
            "olmoe3-30m",
            extra=("override_model_config=True", f"base_shared_expert_mlp_dim={cfg.base_moe_mlp_dim}"),
        )
    )
    self.assertEqual(base, pinned)

  def test_widening_moves_only_the_shared_ffn(self):
    """A wider shared expert must not touch the routed experts or the latent."""
    cfg = _config("olmoe3-30m")
    wide = _config(
        "olmoe3-30m",
        extra=("override_model_config=True", f"base_shared_expert_mlp_dim={4 * cfg.base_moe_mlp_dim}"),
    )
    self.assertEqual(wide.shared_expert_mlp_dim, 4 * cfg.moe_mlp_dim)
    self.assertEqual(wide.moe_mlp_dim, cfg.moe_mlp_dim)

    before, after = self._shapes(cfg), self._shapes(wide)
    self.assertEqual(set(before), set(after))
    changed = {k for k, shape in before.items() if shape != after[k]}
    self.assertTrue(changed, "widening the shared expert changed no parameter shapes")
    for key in changed:
      self.assertIn("shared_ffn", key, f"{key} changed but is not part of the shared FFN")
    # Layer 0 is dense and sized by mlp_dim, so it must be untouched.
    for key in changed:
      self.assertNotIn("layers_0/", key, "the dense layer 0 shared FFN must not follow this knob")

  def test_flop_accounting_follows_the_knob(self):
    """MFU's denominator is analytic, so a counter that ignores the knob inflates it."""
    cfg = _config("olmoe3-30m", max_target_length=512)
    wide = _config(
        "olmoe3-30m",
        max_target_length=512,
        extra=("override_model_config=True", f"base_shared_expert_mlp_dim={4 * cfg.base_moe_mlp_dim}"),
    )
    base_tf, _, _ = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
    wide_tf, _, _ = maxtext_utils.calculate_tflops_training_per_device(wide, log=False)
    self.assertGreater(wide_tf, base_tf, "widening the shared expert did not raise the analytic FLOP count")

    # The delta is exactly the extra SwiGLU width on the MoE layers, fwd+bwd.
    moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
    batch_seq = cfg.per_device_batch_size * cfg.max_target_length
    expected = (
        3 * 2 * batch_seq * 3 * cfg.emb_dim * (wide.shared_expert_mlp_dim - cfg.shared_expert_mlp_dim) * moe_layers
    ) / 1e12
    self.assertAlmostEqual(wide_tf - base_tf, expected, places=6)


class OLMoE3FusedKDAKernelTest(unittest.TestCase):
  """``kernels/kda.chunk_kda`` (the fused Pallas fwd/bwd, interpret mode here) against the references.

  The kernel rounds its matmul operands to bf16 (as default-precision f32 matmuls do on TPU), so it is compared
  to the f32 per-token recurrence at bf16-operand accuracy, and to itself bit for bit where only the residual
  handling differs.
  """

  def _inputs(self, seq_len=256, batch=2, heads=2, dk=16, dv=32, strong=False):
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.models import olmoe3

    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    q = olmoe3._l2_normalize(jax.random.normal(keys[0], (batch, seq_len, heads, dk))) * dk**-0.5
    k = olmoe3._l2_normalize(jax.random.normal(keys[1], (batch, seq_len, heads, dk)))
    v = jax.random.normal(keys[2], (batch, seq_len, heads, dv))
    if strong:  # per-step log-decay up to ~-50 nats, the reference init's range
      log_decay = -jnp.exp(jax.random.uniform(keys[3], q.shape, minval=-3.0, maxval=3.9))
    else:
      log_decay = jnp.log(jax.nn.sigmoid(jax.random.normal(keys[3], q.shape)) * 0.4 + 0.55)
    beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq_len, heads)))
    # Packed documents: boundaries mid-chunk, on a chunk edge and at a chunk's last token.
    seg = jnp.ones((batch, seq_len), jnp.int32)
    for idx in (5, 63, 64, 100, 191):
      if idx < seq_len:
        seg = seg.at[:, idx:].add(1)
    return q, k, v, log_decay, beta, seg

  @staticmethod
  def _rel(a, b):
    return float(jnp.linalg.norm((a - b).astype(jnp.float32)) / jnp.linalg.norm(a.astype(jnp.float32)))

  def _grads(self, fn, args, w):
    return jax.grad(lambda *a: jnp.sum(fn(*a).astype(jnp.float32) * w), argnums=(0, 1, 2, 3, 4))(*args)

  def test_matches_reference_both_layouts_with_segments(self):
    # pylint: disable=import-outside-toplevel
    from typing import Any, cast
    from maxtext.kernels import kda

    *args, seg = self._inputs()
    ref = lambda *a: kda.kda_recurrent_reference(*a, segment_ids=seg)
    expected = ref(*args)
    w = jax.random.normal(jax.random.PRNGKey(9), expected.shape)
    g_ref = self._grads(ref, args, w)
    for layout in ("token_major", "chunk_major"):
      layout_lit = cast(Any, layout)
      fn = jax.jit(lambda *a, l=layout_lit: kda.chunk_kda(*a, segment_ids=seg, chunk_size=64, layout=l))
      actual = fn(*args)
      self.assertTrue(bool(jnp.isfinite(actual).all()), layout)
      self.assertLess(self._rel(expected, actual), 1e-2, f"{layout}: output rel err")
      for name, a, b in zip(("q", "k", "v", "log_decay", "beta"), g_ref, self._grads(fn, args, w)):
        self.assertTrue(bool(jnp.isfinite(b).all()), f"{layout}: non-finite d{name}")
        self.assertLess(self._rel(a, b), 2e-2, f"{layout}: d{name} rel err")

  def test_matches_xla_subblock_strong_decay_and_bf16_state(self):
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.kernels import kda
    from maxtext.models import olmoe3

    q, k, v, log_decay, beta, seg = self._inputs(strong=True)
    args = (q, k, v, log_decay, beta)
    resets = kda.segment_resets(seg, *seg.shape)
    for state, tol in (("float32", 1e-2), ("bfloat16", 3e-2)):
      xla = olmoe3._delta_rule_chunked_subblock(q, k, v, log_decay, beta, resets, 64, state, 16).astype(jnp.float32)
      dtype = jnp.float32 if state == "float32" else jnp.bfloat16
      fused = kda.chunk_kda(*args, resets=resets, chunk_size=64, state_dtype=dtype)
      self.assertTrue(bool(jnp.isfinite(fused).all()), state)
      self.assertLess(self._rel(xla, fused), tol, f"{state}: fused vs XLA sub-block rel err")

  def test_saved_residuals_and_state_recompute_are_exact(self):
    """Naming the residuals, recomputing the states in the bwd and a bf16 output change no value."""
    # pylint: disable=import-outside-toplevel
    from typing import Any, cast
    from maxtext.kernels import kda

    *args, seg = self._inputs()
    variants: dict[str, tuple[dict[str, Any], ...]] = {
        "token_major": (
            dict(save_residuals=True),
            dict(save_residuals=True, state_stride=2),
            dict(save_residuals=True, state_stride=0),
            dict(state_stride=2),
            dict(save_residuals="wy", state_stride=2),
        ),
        "chunk_major": (dict(state_stride=0), dict(state_stride=2), dict(save_residuals="wy")),
    }
    w = None
    out = None
    for layout, kwargs_list in variants.items():
      layout_lit = cast(Any, layout)
      base = lambda *a, l=layout_lit: kda.chunk_kda(*a, segment_ids=seg, chunk_size=64, layout=l)
      ref_out = base(*args)
      if w is None:
        out, w = ref_out, jax.random.normal(jax.random.PRNGKey(9), ref_out.shape)
      g_base = self._grads(base, args, w)
      for kwargs in kwargs_list:
        fn = lambda *a, kw=kwargs, l=layout_lit: kda.chunk_kda(*a, segment_ids=seg, chunk_size=64, layout=l, **kw)
        np.testing.assert_array_equal(np.asarray(ref_out), np.asarray(fn(*args)), err_msg=f"{layout} {kwargs}")
        for a, b in zip(g_base, self._grads(fn, args, w)):
          np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"{layout} {kwargs}")
    assert out is not None
    bf16 = kda.chunk_kda(*args, segment_ids=seg, chunk_size=64, output_dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out.astype(jnp.bfloat16)), np.asarray(bf16))

  def test_pads_sequences_that_are_not_a_chunk_multiple(self):
    # pylint: disable=import-outside-toplevel
    from maxtext.kernels import kda

    *args, seg = self._inputs(seq_len=200)
    expected = kda.kda_recurrent_reference(*args, segment_ids=seg)
    actual = kda.chunk_kda(*args, segment_ids=seg, chunk_size=64)
    self.assertEqual(actual.shape, expected.shape)
    self.assertLess(self._rel(expected, actual), 1e-2)

  def test_remat_policy_saving_residuals_skips_forward_recompute(self):
    """Under remat, saving "kda_fused" leaves one forward kernel (2 without); the states kernel runs in the bwd."""
    # pylint: disable=import-outside-toplevel
    from maxtext.kernels import kda

    *args, seg = self._inputs(seq_len=128)

    def kernel_counts(save, policy, stride=0):
      def layer(*a):
        out = kda.chunk_kda(*a, segment_ids=seg, chunk_size=64, save_residuals=save, state_stride=stride)
        return jnp.sum(jnp.sin(out))

      jaxpr = jax.make_jaxpr(jax.grad(jax.checkpoint(layer, policy=policy), argnums=(0, 1, 2, 3, 4)))(*args)
      kinds = _pallas_kernel_kinds(jaxpr.jaxpr)
      return kinds["fwd"], kinds["wy"], kinds["states"], kinds["bwd"]

    save_policy = jax.checkpoint_policies.save_only_these_names(kda.RESIDUAL_NAME)
    self.assertEqual(kernel_counts(False, jax.checkpoint_policies.nothing_saveable, 1), (2, 0, 0, 1))
    self.assertEqual(kernel_counts(True, save_policy), (1, 0, 1, 1))
    self.assertEqual(kernel_counts(True, save_policy, 2), (1, 0, 0, 1))
    # 'wy': the intra/inverse kernel runs once; the recurrence + readout kernel runs again in the recompute.
    self.assertEqual(kernel_counts("wy", save_policy, 1), (2, 1, 0, 1))


class OLMoE3FusedKDAModuleTest(unittest.TestCase):
  """``kda_pallas_fused`` in the KDA layer: same values as the XLA sub-block layer; ``kda_fused=device`` remat."""

  def _run(self, extra, count_kernels=False):
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax import nnx
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 128, 128), jnp.float32)
    segments = jnp.concatenate([jnp.full((2, 50), 1, jnp.int32), jnp.full((2, 78), 2, jnp.int32)], axis=1)
    cfg = _config(
        "olmoe3-30m",
        max_target_length=128,
        extra=("use_tokamax_kda=False", "kda_chunked_impl=subblock", "gdn_state_dtype=float32", *extra),
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    policy = jax.checkpoint_policies.save_only_these_names(*(cfg.tensors_on_device or []))
    with mesh:
      layer = OLMoE3KimiDeltaAttention(cfg, mesh, None, rngs=nnx.Rngs(params=0))
      graphdef, state = nnx.split(layer)

      def loss(state_, x_, graphdef=graphdef):
        run = lambda s_, i_: nnx.merge(graphdef, s_)(i_, segments)
        return jnp.sum(jax.checkpoint(run, policy=policy)(state_, x_) ** 2)

      grad_fn = jax.value_and_grad(loss, argnums=(0, 1))
      if count_kernels:
        return _pallas_kernel_kinds(jax.make_jaxpr(grad_fn)(state, x).jaxpr)["fwd"]
      value, grads = jax.jit(grad_fn)(state, x)
    return float(value), jax.tree_util.tree_leaves(nnx.to_pure_dict(grads[0])) + [grads[1]]

  def test_fused_matches_xla_subblock(self):
    v_ref, g_ref = self._run(("kda_pallas_fused=False",))
    for tm in (True, False):
      v, g = self._run(("kda_pallas_fused=True", f"kda_fused_token_major={tm}", "kda_outer_block=32"))
      self.assertAlmostEqual(v_ref, v, delta=1e-2 * abs(v_ref))
      for a, b in zip(g_ref, g):
        rel = jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-30)
        # The kernel rounds its matmul operands to bf16; the f32 XLA path (dtype=float32 here) does not.
        self.assertLess(float(rel), 6.5e-2, f"token_major={tm}: grad rel err {rel:.2e}")

  def test_kda_fused_device_skips_forward_recompute(self):
    fused = ("kda_pallas_fused=True", "kda_fused_token_major=True", "kda_outer_block=32", "remat_policy=custom")
    self.assertEqual(self._run((*fused, "kda_fused=remat"), count_kernels=True), 2)
    self.assertEqual(self._run((*fused, "kda_fused=device"), count_kernels=True), 1)
    v_remat, g_remat = self._run((*fused, "kda_fused=remat"))
    for extra in (
        ("kda_fused=device",),
        ("kda_fused=device", "kda_fused_state_stride=1"),
        ("kda_fused=device", "kda_fused_state_stride=0"),
    ):
      v, g = self._run((*fused, *extra))
      self.assertEqual(v_remat, v)
      for a, b in zip(g_remat, g):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=str(extra))

  def test_two_stage_gathered_kernel_grad_parity(self):
    """`dot_general_with_gathered_kernel` matches standard `dot_general` grads for 2D and 3D kernels."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    from flax.core import spmd
    from maxtext.layers import linears

    devs = jax.devices()
    dp = 2 if len(devs) >= 4 else 1
    ep = len(devs) // dp
    devices = np.array(devs[: dp * ep]).reshape(dp, 1, ep)  # pylint: disable=too-many-function-args
    mesh = Mesh(devices, ("data", "fsdp", "expert"))
    rules = (("embed", ("fsdp", "expert")), ("mlp", None), ("heads", None), ("kv", None))
    save_policy = jax.checkpoint_policies.save_only_these_names("gathered_weights")

    for k_shape, norm_axis, contract_ind, k_axes in (
        ((16, 24), (2,), (0,), ("embed", "mlp")),
        ((16, 20), (2,), (0,), ("embed", None)),
        ((14 * max(ep, 2), 264), (2,), (0,), ("embed", None)),
        ((24, 16), (2,), (0,), ("mlp", "embed")),
        ((16, 2, 12), (2,), (0,), ("embed", "heads", "kv")),
        ((2, 12, 16), (2, 3), (0, 1), ("heads", "kv", "embed")),
    ):
      k = jax.random.normal(jax.random.PRNGKey(1), k_shape, jnp.float32)
      x_in = jax.random.normal(jax.random.PRNGKey(2), (8, 8, *k_shape[: len(norm_axis)]), jnp.float32)

      def ref_loss(x_, k_, na=norm_axis, ci=contract_ind):
        y = jax.lax.dot_general(x_, k_, ((na, ci), ((), ())))
        return jnp.sum(jnp.sin(y.astype(jnp.float32)))

      for save_gk in (True, False):
        for pol in (
            save_policy,
            jax.checkpoint_policies.save_only_these_names("gathered_weights", "kda_proj"),
        ):

          def two_stage_loss(x_, k_, na=norm_axis, ci=contract_ind, ka=k_axes, sgk=save_gk, p=pol):
            def body(x_b, k_b):
              return linears.dot_general_with_gathered_kernel(
                  x_b, k_b, na, ci, "high", mesh, ka, save_gathered_kernel=sgk, out_checkpoint_name="kda_proj"
              )

            y = jax.checkpoint(body, policy=p)(x_, k_)
            return jnp.sum(jnp.sin(y.astype(jnp.float32)))

          with mesh, spmd.logical_axis_rules(rules):
            grad_2s = jax.jit(jax.value_and_grad(two_stage_loss, argnums=(0, 1)))
            if dp > 1 and ep > 1:
              ir = grad_2s.lower(x_in, k).compiler_ir("hlo")
              assert ir is not None
              hlo = ir.as_hlo_text()
              self.assertIn("reduce-scatter(", hlo)
              self.assertGreaterEqual(hlo.count("all-reduce("), 1)
            v_ref, (dx_ref, dk_ref) = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1)))(x_in, k)
            v_2s, (dx_2s, dk_2s) = grad_2s(x_in, k)
          self.assertAlmostEqual(float(v_ref), float(v_2s), places=4)
          np.testing.assert_allclose(np.asarray(dx_ref, np.float32), np.asarray(dx_2s, np.float32), rtol=1e-4, atol=1e-4)
          np.testing.assert_allclose(np.asarray(dk_ref, np.float32), np.asarray(dk_2s, np.float32), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
