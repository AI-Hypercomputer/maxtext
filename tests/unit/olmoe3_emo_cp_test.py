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

"""Cross-context-parallel (``CP > 1``) EMo document-pool routing parity tests.

Verifies that ``OLMoE3LatentRoutedMoE._emo_mask_logits`` produces the same pools, masks,
and gradients when the sequence axis is sharded across context-parallel devices (``CP=2``,
``CP=4``) or all-gathered onto axis 0 under ``cp-as-ep`` ``roe_ag_and_route`` as on
``CP=1``:
- Causal (``emo_causal_pool=True``) and whole-document (``emo_causal_pool=False``) modes
- Both ``moe_lean_routing=False`` and ``moe_lean_routing=True`` (``bisect=False`` and ``True``)
- Documents spanning 2, 3, or all 4 CP shards, exact shard-boundary documents, repeated
  segment IDs separated by another document within a shard, and ``segment_ids=None``
- Training-mode random pool sampling: shared pool size across all CP shards of a document
  and independent draws for distinct documents starting at the same local shard offset
- End-to-end ``OLMoE3LatentRoutedMoE.sparse_matmul`` across ``(CP=1, EP=4)``, ``(CP=2, EP=2)``,
  and ``(CP=4, EP=1)`` under ``custom_mesh_and_rule="cp-as-ep"`` with
  ``moe_a2a_expert_major=True`` (``moe_a2a_token_chunks=1`` and ``2``), plus direct verification
  of ``EP_GATHERED_AXES`` unfolding and ``load_balance_loss(..., density_prob=...)``.

Needs 4 devices (sets host device count before importing JAX when run standalone).

    JAX_PLATFORMS=cpu pytest tests/unit/olmoe3_emo_cp_test.py
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import types  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position
from unittest import mock  # pylint: disable=wrong-import-position

from flax import linen as nn  # pylint: disable=wrong-import-position
from flax import nnx  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.layers import moe  # pylint: disable=wrong-import-position
from maxtext.models import olmoe3  # pylint: disable=wrong-import-position
from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=wrong-import-position

NUM_DEVICES = 4
NUM_EXPERTS = 16
TOP_K = 2
SEQ = 32
BATCH = 4


def _make_router(
    mesh,
    *,
    causal: bool,
    lean: bool = False,
    bisect: bool = False,
    eval_pool: int = 4,
    choices=(),
    probs=(),
    call_mode: str = "train",
):
  """Builds a lightweight OLMoE3LatentRoutedMoE stub for direct _emo_mask_logits tests."""
  cfg = types.SimpleNamespace(
      emo_causal_pool=causal,
      emo_document_expert_pool_choices=list(choices),
      emo_document_expert_pool_probs=list(probs),
      emo_min_document_expert_pool=2,
      emo_max_document_expert_pool=NUM_EXPERTS,
      emo_eval_document_expert_pool=eval_pool,
      emo_threshold_by_bisection=bisect,
      model_call_mode=call_mode,
      moe_lean_routing=lean,
  )
  return types.SimpleNamespace(config=cfg, mesh=mesh)


def _diverse_segments():
  """4 rows of length 32 covering all cross-shard boundary topologies for CP=2 and CP=4."""
  return jnp.asarray(
      [
          # Row 0: single document spanning all 4 shards.
          [1] * 32,
          # Row 1: doc 1 crosses 0->1, doc 2 inside shard 1, doc 3 spans 1->2->3, doc 4 ends shard 3.
          [1] * 12 + [2] * 2 + [3] * 15 + [4] * 3,
          # Row 2: exact shard-boundary documents (8 tokens each).
          [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8,
          # Row 3: repeated segment ID (1) separated by doc 2 inside shard 0, then crossing 0->1->2->3.
          [1] * 3 + [2] * 2 + [1] * 11 + [1] * 10 + [5] * 6,
      ],
      dtype=jnp.int32,
  )


@unittest.skipUnless(jax.device_count() >= NUM_DEVICES, f"needs {NUM_DEVICES} devices; run this file on its own")
class EmoContextParallelTest(unittest.TestCase):
  """Verifies exact EMo parity across CP shards for both sharded-seq and ep-gathered layouts."""

  def setUp(self):
    super().setUp()
    self.devices = np.array(jax.devices()[:NUM_DEVICES])
    self.mesh_cp4 = Mesh(self.devices.reshape(4), ("context",))
    self.mesh_cp2_ep2 = Mesh(self.devices.reshape(2, 2), ("context", "expert"))
    self.logits = jax.random.normal(jax.random.PRNGKey(42), (BATCH, SEQ, NUM_EXPERTS), dtype=jnp.float32)
    self.segments = _diverse_segments()

  def _run_cp1(self, logits, segments, *, causal, lean, bisect, eval_pool=4, rngs=None):
    router = _make_router(
        self.mesh_cp4,
        causal=causal,
        lean=lean,
        bisect=bisect,
        eval_pool=eval_pool,
        call_mode="inference" if rngs is None else "train",
    )
    return olmoe3.OLMoE3LatentRoutedMoE._emo_mask_logits(  # pylint: disable=protected-access
        router, logits, segments, rngs
    )

  def _run_sharded_cp(
      self,
      mesh,
      logits,
      segments,
      *,
      causal,
      lean=False,
      bisect=False,
      eval_pool=4,
      choices=(),
      probs=(),
      rngs=None,
      check_vma=False,
  ):
    """Runs _emo_mask_logits inside a jitted shard_map with sequence sharded on 'context'."""
    router = _make_router(
        mesh,
        causal=causal,
        lean=lean,
        bisect=bisect,
        eval_pool=eval_pool,
        choices=choices,
        probs=probs,
        call_mode="inference" if rngs is None else "train",
    )

    def _body(x, seg):
      return olmoe3.OLMoE3LatentRoutedMoE._emo_mask_logits(  # pylint: disable=protected-access
          router, x, seg, rngs
      )

    seg_spec = None if segments is None else P(None, "context")
    fn = jax.jit(
        jax.shard_map(
            _body,
            mesh=mesh,
            in_specs=(P(None, "context", None), seg_spec),
            out_specs=P(None, "context", None),
            check_vma=check_vma,
        )
    )
    token = moe.SEQ_SHARD_AXES.set(("context",))
    try:
      return fn(logits, segments)
    finally:
      moe.SEQ_SHARD_AXES.reset(token)

  def test_fixed_pool_parity_across_cp4_modes_and_vma(self):
    """CP=4 sharded-sequence EMo matches CP=1 across causal/whole, lean/argsort, bisect, and check_vma."""
    for causal in (True, False):
      for lean, bisect in ((False, False), (True, False), (True, True)):
        for check_vma in (False, True):
          with self.subTest(causal=causal, lean=lean, bisect=bisect, check_vma=check_vma):
            ref = self._run_cp1(self.logits, self.segments, causal=causal, lean=lean, bisect=bisect)
            got = self._run_sharded_cp(
                self.mesh_cp4,
                self.logits,
                self.segments,
                causal=causal,
                lean=lean,
                bisect=bisect,
                check_vma=check_vma,
            )
            np.testing.assert_array_equal(np.isfinite(np.asarray(got)), np.isfinite(np.asarray(ref)))
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_segment_ids_none_spans_all_cp_shards(self):
    """With segment_ids=None, the whole sequence is one document across all CP shards."""
    for causal in (True, False):
      with self.subTest(causal=causal):
        ref = self._run_cp1(self.logits, None, causal=causal, lean=True, bisect=True)
        got = self._run_sharded_cp(self.mesh_cp4, self.logits, None, causal=causal, lean=True, bisect=True)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_grad_parity_cp1_vs_cp4(self):
    """Gradients w.r.t. gate_logits match CP=1 identically (and emit no backward cross-CP errors)."""
    for causal in (True, False):
      with self.subTest(causal=causal):

        def _loss_cp1(x, causal=causal):
          masked = self._run_cp1(x, self.segments, causal=causal, lean=True, bisect=True)
          probs = jax.nn.softmax(masked, axis=-1)
          return jnp.sum(probs * jnp.arange(NUM_EXPERTS, dtype=jnp.float32))

        def _loss_cp4(x, causal=causal):
          masked = self._run_sharded_cp(
              self.mesh_cp4, x, self.segments, causal=causal, lean=True, bisect=True, check_vma=True
          )
          probs = jax.nn.softmax(masked, axis=-1)
          return jnp.sum(probs * jnp.arange(NUM_EXPERTS, dtype=jnp.float32))

        v1, g1 = jax.value_and_grad(_loss_cp1)(self.logits)
        v4, g4 = jax.jit(jax.value_and_grad(_loss_cp4))(self.logits)
        np.testing.assert_allclose(float(v4), float(v1), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(g4), np.asarray(g1), rtol=1e-5, atol=1e-6)

  def test_training_random_pools_shared_within_cross_shard_doc_and_independent_across_docs(self):
    """Training pools are identical across every CP shard of a doc, and independent across shard-local starts."""
    choices = (2, 4, 8, 16)
    probs = (0.25, 0.25, 0.25, 0.25)
    for causal in (True, False):
      with self.subTest(causal=causal):
        got = self._run_sharded_cp(
            self.mesh_cp4,
            self.logits,
            self.segments,
            causal=causal,
            lean=True,
            bisect=True,
            choices=choices,
            probs=probs,
            rngs=jax.random.PRNGKey(7),
            check_vma=True,
        )
        counts = np.sum(np.isfinite(np.asarray(got)), axis=-1)  # [B, S]
        # Row 0: single doc [0:32] must have one constant pool size across all 4 shards.
        self.assertEqual(len(set(counts[0].tolist())), 1)
        # Row 1: doc 1 [0:12], doc 2 [12:14], doc 3 [14:29] (spans shards 1,2,3), doc 4 [29:32].
        for lo, hi in ((0, 12), (12, 14), (14, 29), (29, 32)):
          self.assertEqual(len(set(counts[1, lo:hi].tolist())), 1, f"doc [{lo}:{hi}] had varying pool sizes")
        # Row 3: repeated seg 1 after doc 2 ([5:26] spans shards 0,1,2,3) shares one pool across all 4 shards.
        self.assertEqual(len(set(counts[3, 5:26].tolist())), 1)

    # Distinct documents starting at local offset 0 on different shards (Row 2) must draw independent pools.
    router = _make_router(
        self.mesh_cp4,
        causal=True,
        lean=True,
        bisect=True,
        choices=choices,
        probs=probs,
        call_mode="train",
    )

    def _body_rng(x, seg, key):
      return olmoe3.OLMoE3LatentRoutedMoE._emo_mask_logits(  # pylint: disable=protected-access
          router, x, seg, key
      )

    fn_rng = jax.jit(
        jax.shard_map(
            _body_rng,
            mesh=self.mesh_cp4,
            in_specs=(P(None, "context", None), P(None, "context"), P()),
            out_specs=P(None, "context", None),
            check_vma=True,
        )
    )
    distinct_draws = set()
    token = moe.SEQ_SHARD_AXES.set(("context",))
    try:
      for seed in range(8):
        got = fn_rng(self.logits, self.segments, jax.random.PRNGKey(100 + seed))
        row2_counts = np.sum(np.isfinite(np.asarray(got[2])), axis=-1)
        shard_pools = tuple(int(row2_counts[s]) for s in (0, 8, 16, 24))
        distinct_draws.add(len(set(shard_pools)) > 1)
    finally:
      moe.SEQ_SHARD_AXES.reset(token)
    self.assertIn(True, distinct_draws, "documents at local offset 0 on different CP shards should draw independently")

  def test_ep_gathered_cp_as_ep_unfolding_matches_cp1(self):
    """When roe_ag_and_route gathers ('context', 'expert') onto axis 0, unfolding reproduces CP=1 identically."""
    mesh = self.mesh_cp2_ep2
    # Batch sharded on 'expert' (2) and sequence sharded on 'context' (2), matching cp-as-ep.
    for causal in (True, False):
      with self.subTest(causal=causal):
        ref = self._run_cp1(self.logits, self.segments, causal=causal, lean=True, bisect=True)
        router = _make_router(mesh, causal=causal, lean=True, bisect=True, call_mode="inference")

        def _body(x_local, seg_local, router=router):
          # Simulate roe_ag_and_route's _ep_all_gather over ("context", "expert") on axis 0.
          x_ag = jax.lax.all_gather(x_local, ("context", "expert"), axis=0, tiled=True)
          seg_ag = jax.lax.all_gather(seg_local, ("context", "expert"), axis=0, tiled=True)
          ep_tok = moe.EP_GATHERED_AXES.set(("context", "expert"))
          try:
            masked_ag = olmoe3.OLMoE3LatentRoutedMoE._emo_mask_logits(  # pylint: disable=protected-access
                router, x_ag, seg_ag, None
            )
          finally:
            moe.EP_GATHERED_AXES.reset(ep_tok)
          # Extract this shard's slice back out of the gathered tensor (row-major (context, expert)).
          cp_i = jax.lax.axis_index("context")
          ep_i = jax.lax.axis_index("expert")
          b_local = x_local.shape[0]
          shard_idx = cp_i * 2 + ep_i
          return jax.lax.dynamic_slice_in_dim(masked_ag, shard_idx * b_local, b_local, axis=0)

        fn = jax.jit(
            jax.shard_map(
                _body,
                mesh=mesh,
                in_specs=(P("expert", "context", None), P("expert", "context")),
                out_specs=P("expert", "context", None),
                check_vma=False,
            )
        )
        seq_tok = moe.SEQ_SHARD_AXES.set(("context",))
        try:
          got = fn(self.logits, self.segments)
        finally:
          moe.SEQ_SHARD_AXES.reset(seq_tok)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_load_balance_loss_density_prob_matches_logits(self):
    """OLMoE3LatentRoutedMoE.load_balance_loss accepts precomputed density_prob from topk_before_ep_all_gather."""
    cfg = types.SimpleNamespace(
        moe_global_load_balancing=False,
        load_balance_loss_weight=0.01,
        olmoe3_router_metrics=False,
    )
    stub = types.SimpleNamespace(
        config=cfg,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
    )
    probs = jax.nn.softmax(self.logits, axis=-1)
    top_k = jax.lax.top_k(probs, TOP_K)[1]
    lb_logits = olmoe3.OLMoE3LatentRoutedMoE.load_balance_loss(stub, top_k, probs)
    lb_density = olmoe3.OLMoE3LatentRoutedMoE.load_balance_loss(
        stub, top_k, None, density_prob=jnp.mean(probs, axis=1)
    )
    np.testing.assert_allclose(float(lb_density), float(lb_logits), rtol=1e-6, atol=1e-7)

  def test_end_to_end_sparse_moe_cp_as_ep_parity(self):
    """End-to-end OLMoE3LatentRoutedMoE sparse_matmul matches across (CP=1,EP=4), (CP=2,EP=2), (CP=4,EP=1)."""
    base_yml = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
    emb_dim = 16
    latent_dim = 8
    mlp_dim = 16

    def _cpu_ragged_all_to_all(  # pylint: disable=too-many-positional-arguments
        x, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name
    ):
      """Pure-JAX CPU emulation of jax.lax.ragged_all_to_all (supports multi-update expert-major params)."""
      del recv_sizes
      all_x = jax.lax.all_gather(x, axis_name, axis=0)
      num_shards = all_x.shape[0]
      all_in_off = jax.lax.all_gather(input_offsets, axis_name, axis=0).reshape(num_shards, num_shards, -1)
      all_out_off = jax.lax.all_gather(output_offsets, axis_name, axis=0).reshape(num_shards, num_shards, -1)
      all_send = jax.lax.all_gather(send_sizes, axis_name, axis=0).reshape(num_shards, num_shards, -1)
      me = jax.lax.axis_index(axis_name)
      rows = jnp.arange(x.shape[0])
      out = output
      for src in range(num_shards):
        for u in range(all_in_off.shape[2]):
          in_off = all_in_off[src, me, u]
          out_off = all_out_off[src, me, u]
          sz = all_send[src, me, u]
          mask = (rows >= in_off) & (rows < in_off + sz)
          dst_idx = jnp.where(mask, out_off + (rows - in_off), out.shape[0])
          out = out.at[dst_idx].set(all_x[src], mode="drop")
      return out

    def _run_layer(cp, ep, *, causal, token_chunks, x_np, seg_np):
      cfg = pyconfig.initialize(
          [
              "",
              base_yml,
              "run_name=ut_emo_cp_e2e",
              "skip_jax_distributed_system=True",
              "enable_checkpointing=False",
              "custom_mesh_and_rule=cp-as-ep",
              "ici_data_parallelism=1",
              "ici_fsdp_parallelism=1",
              f"ici_context_parallelism={cp}",
              f"ici_expert_parallelism={ep}",
              "per_device_batch_size=1.0",
              f"max_target_length={SEQ}",
              f"base_emb_dim={emb_dim}",
              f"base_mlp_dim={mlp_dim}",
              f"base_moe_mlp_dim={mlp_dim}",
              f"moe_expert_input_dim={latent_dim}",
              f"num_experts={NUM_EXPERTS}",
              f"num_experts_per_tok={TOP_K}",
              "dtype=float32",
              "weight_dtype=float32",
              "float32_gate_logits=True",
              "sparse_matmul=True",
              "megablox=False",
              "capacity_factor=-1",
              "emo_enabled=True",
              f"emo_causal_pool={causal}",
              "emo_min_document_expert_pool=4",
              "emo_max_document_expert_pool=4",
              "emo_eval_document_expert_pool=4",
              "emo_threshold_by_bisection=True",
              "moe_lean_routing=True",
              "moe_global_load_balancing=True",
              "load_balance_loss_weight=0.01",
              "use_ring_of_experts=False",
              "moe_a2a_expert_major=True",
              f"moe_a2a_token_chunks={token_chunks}",
          ]
      )
      mesh = Mesh(self.devices.reshape((1, 1, cp, ep)), ("data", "fsdp", "context", "expert"))
      with nn.logical_axis_rules(cfg.logical_axis_rules):
        layer = olmoe3.OLMoE3LatentRoutedMoE(
            config=cfg,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            mesh=mesh,
            kernel_init=olmoe3.olmoe3_init,
            kernel_axes=("embed", None),
            intermediate_dim=mlp_dim,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
            gate_in_features=emb_dim,
            rngs=nnx.Rngs(0),
        )
        # Verify logical axis rules map exp -> ("context", "expert") and seq -> "context".
        seq_pspec = layer._logical_to_mesh_axes(  # pylint: disable=protected-access
            ("activation_batch", "activation_norm_length", None)
        )
        if cp > 1:
          self.assertIn("context", moe.mesh_axes_for_dim(seq_pspec[1]))
        # Place parameters and inputs with their mesh shardings.
        exp_sharding = NamedSharding(mesh, P(("context", "expert"), None, None))
        for attr in ("wi_0", "wi_1", "wo"):
          param = getattr(layer, attr)
          param[...] = jax.device_put(param[...], exp_sharding)
        layer.gate.kernel[...] = jax.device_put(layer.gate.kernel[...], NamedSharding(mesh, P(None, None)))
        x_sh = jax.device_put(jnp.asarray(x_np), NamedSharding(mesh, P("expert", "context", None)))
        seg_sh = jax.device_put(jnp.asarray(seg_np), NamedSharding(mesh, P("expert", "context")))
        gate_in = jax.device_put(jnp.asarray(x_np), NamedSharding(mesh, P("expert", "context", None)))
        latent_in = x_sh[..., :latent_dim]
        with mock.patch.object(jax.lax, "ragged_all_to_all", _cpu_ragged_all_to_all), olmoe3.emo_eval_pools():
          # JIT so shard_map compiles as a single executable instead of eager per-primitive compilation.
          @nnx.jit
          def _call(mod, x, gate_x, seg):
            with nn.logical_axis_rules(cfg.logical_axis_rules):
              return mod(x, gate_inputs=gate_x, input_ids=seg)

          out, lb_loss, _ = _call(layer, latent_in, gate_in, seg_sh)
      return np.asarray(out), float(lb_loss)

    x_np = np.asarray(jax.random.normal(jax.random.PRNGKey(9), (BATCH, SEQ, emb_dim), dtype=jnp.float32))
    seg_np = np.asarray(self.segments)

    # Causal mode: verify production ra2a_and_route across (CP=1,EP=4), (CP=2,EP=2) (1 & 2 chunks), (CP=4,EP=1).
    ref_causal_out, ref_causal_lb = _run_layer(1, 4, causal=True, token_chunks=1, x_np=x_np, seg_np=seg_np)
    causal_cases = [
        ("cp2_ep2_a2a_c1", 2, 2, 1),
        ("cp2_ep2_a2a_c2", 2, 2, 2),
        ("cp4_ep1_a2a_c1", 4, 1, 1),
    ]
    for label, cp, ep, chunks in causal_cases:
      with self.subTest(causal=True, case=label):
        out, lb = _run_layer(cp, ep, causal=True, token_chunks=chunks, x_np=x_np, seg_np=seg_np)
        np.testing.assert_allclose(out, ref_causal_out, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(lb, ref_causal_lb, rtol=1e-4, atol=1e-5)

    # Whole-document mode: verify end-to-end chunked a2a with forward+backward CP propagation.
    ref_whole_out, ref_whole_lb = _run_layer(1, 4, causal=False, token_chunks=1, x_np=x_np, seg_np=seg_np)
    with self.subTest(causal=False, case="cp2_ep2_a2a_c2"):
      out, lb = _run_layer(2, 2, causal=False, token_chunks=2, x_np=x_np, seg_np=seg_np)
      np.testing.assert_allclose(out, ref_whole_out, rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(lb, ref_whole_lb, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
