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

"""Unit tests for `maxtext.m3.core.sharded_embedding` and `Qwen3_8NGramEmbedding` routing."""

from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.m3.core import sharded_embedding as m3_se
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.infra import named_scopes as m3_named_scopes
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next as qwen3_8


_RUN_TPU_PLE_BENCH = flags.DEFINE_bool(
    "run_tpu_ple_bench",
    False,
    "Run physical TPUv7x ragged_all_to_all + Zipf skew + EPasCP PLE parity test.",
)


def _make_mesh(ici: dict[str, int], *, shard_mode: str = "auto") -> jax.sharding.Mesh:
  """Builds a 4-device m3 CPU mesh in either Auto or Explicit axis mode."""
  devices = jax.devices()[:4]
  if len(devices) < 4:
    raise RuntimeError(
        f"Expected >= 4 devices (set XLA_FLAGS=--xla_force_host_platform_device_count=4), got {len(devices)}."
    )
  axis_types = (
      (jax.sharding.AxisType.Explicit,) * len(m3_sharding.MESH_AXIS_NAMES)
      if shard_mode == "explicit"
      else (jax.sharding.AxisType.Auto,) * len(m3_sharding.MESH_AXIS_NAMES)
  )
  base = m3_sharding.create_mesh(ici=ici, devices=devices)
  return jax.sharding.Mesh(base.devices, base.axis_names, axis_types=axis_types)


def _reference_lookup(table: jax.Array, ids: jax.Array, out_dtype: Any) -> jax.Array:
  valid = (ids >= 0) & (ids < table.shape[0])
  safe_ids = jnp.where(valid, ids, 0)
  out = jnp.take(table, safe_ids, axis=0).astype(out_dtype)
  return jnp.where(valid[..., None], out, jnp.zeros((), dtype=out_dtype))


class ShardedEmbeddingTest(parameterized.TestCase):
  """Tests index-routed row-sharded embedding lookups and Qwen3.8 PLE integration."""

  @parameterized.named_parameters(
      ("auto", "auto"),
      ("explicit", "explicit"),
  )
  def test_row_sharded_lookup_1d_auto_and_explicit(self, shard_mode: str):
    mesh = _make_mesh({"fsdp": 4}, shard_mode=shard_mode)
    key = jax.random.PRNGKey(0)
    k_tbl, k_ids, k_w = jax.random.split(key, 3)
    table = jax.random.normal(k_tbl, (64, 16), dtype=jnp.float32)
    ids = jax.random.randint(k_ids, (4, 8), 0, 64, dtype=jnp.int32)
    weights = jax.random.normal(k_w, (4, 8, 16), dtype=jnp.float32)

    tbl_shd = NamedSharding(mesh, P("fsdp", None))
    ids_shd = NamedSharding(mesh, P("fsdp", None))
    w_shd = NamedSharding(mesh, P("fsdp", None, None))
    table_s = jax.device_put(table, tbl_shd)
    ids_s = jax.device_put(ids, ids_shd)
    w_s = jax.device_put(weights, w_shd)

    def _loss_routed(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = m3_se.row_sharded_embedding_lookup(
          t, ids_s, table_axes=("fsdp",), ids_specs=P("fsdp", None), out_dtype=jnp.float32, mesh=mesh
      )
      return jnp.sum(out * w_s), out

    def _loss_ref(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = _reference_lookup(t, ids, jnp.float32)
      return jnp.sum(out * weights), out

    with jax.sharding.set_mesh(mesh):
      (_, out_routed), grad_routed = jax.jit(jax.value_and_grad(_loss_routed, has_aux=True))(table_s)
    (_, out_ref), grad_ref = jax.value_and_grad(_loss_ref, has_aux=True)(table)

    np.testing.assert_allclose(np.asarray(out_routed), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(grad_routed), np.asarray(grad_ref), rtol=1e-5, atol=1e-5)

  def test_row_sharded_lookup_2d_tuple_axes(self):
    mesh = _make_mesh({"fsdp": 2, "expert": 2})
    k_tbl, k_ids, k_w = jax.random.split(jax.random.PRNGKey(1), 3)
    table = jax.random.normal(k_tbl, (80, 12), dtype=jnp.float32)
    ids = jax.random.randint(k_ids, (2, 12), 0, 80, dtype=jnp.int32)
    weights = jax.random.normal(k_w, (2, 12, 12), dtype=jnp.float32)

    table_s = jax.device_put(table, NamedSharding(mesh, P(("fsdp", "expert"), None)))
    ids_s = jax.device_put(ids, NamedSharding(mesh, P("fsdp", "expert")))
    w_s = jax.device_put(weights, NamedSharding(mesh, P("fsdp", "expert", None)))

    def _loss_routed(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = m3_se.row_sharded_embedding_lookup(
          t,
          ids_s,
          table_axes=("fsdp", "expert"),
          ids_specs=P("fsdp", "expert"),
          out_dtype=jnp.float32,
          mesh=mesh,
      )
      return jnp.sum(out * w_s), out

    def _loss_ref(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = _reference_lookup(t, ids, jnp.float32)
      return jnp.sum(out * weights), out

    with jax.sharding.set_mesh(mesh):
      (_, out_routed), grad_routed = jax.jit(jax.value_and_grad(_loss_routed, has_aux=True))(table_s)
    (_, out_ref), grad_ref = jax.value_and_grad(_loss_ref, has_aux=True)(table)

    np.testing.assert_allclose(np.asarray(out_routed), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(grad_routed), np.asarray(grad_ref), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("extra_data_axis_auto", {"data": 2, "fsdp": 2}, P(("data", "fsdp"), None), "auto"),
      ("extra_data_axis_explicit", {"data": 2, "fsdp": 2}, P(("data", "fsdp"), None), "explicit"),
      ("replicated_tensor_axis_auto", {"fsdp": 2, "tensor": 2}, P("fsdp", None), "auto"),
      ("replicated_tensor_axis_explicit", {"fsdp": 2, "tensor": 2}, P("fsdp", None), "explicit"),
  )
  def test_row_sharded_lookup_extra_and_unmentioned_axes(self, ici: dict[str, int], ids_spec: P, shard_mode: str):
    mesh = _make_mesh(ici, shard_mode=shard_mode)
    k_tbl, k_ids, k_w = jax.random.split(jax.random.PRNGKey(2), 3)
    table = jax.random.normal(k_tbl, (48, 8), dtype=jnp.float32)
    ids = jax.random.randint(k_ids, (4, 6), 0, 48, dtype=jnp.int32)
    weights = jax.random.normal(k_w, (4, 6, 8), dtype=jnp.float32)

    table_s = jax.device_put(table, NamedSharding(mesh, P("fsdp", None)))
    ids_s = jax.device_put(ids, NamedSharding(mesh, ids_spec))
    w_s = jax.device_put(weights, NamedSharding(mesh, P(*ids_spec, None)))

    def _loss_routed(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = m3_se.row_sharded_embedding_lookup(
          t, ids_s, table_axes=("fsdp",), ids_specs=ids_spec, out_dtype=jnp.float32, mesh=mesh
      )
      return jnp.sum(out * w_s), out

    def _loss_ref(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = _reference_lookup(t, ids, jnp.float32)
      return jnp.sum(out * weights), out

    with jax.sharding.set_mesh(mesh):
      (_, out_routed), grad_routed = jax.jit(jax.value_and_grad(_loss_routed, has_aux=True))(table_s)
    (_, out_ref), grad_ref = jax.value_and_grad(_loss_ref, has_aux=True)(table)

    np.testing.assert_allclose(np.asarray(out_routed), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(grad_routed), np.asarray(grad_ref), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("all_to_shard0_tight_buf", 1.0, 0.5),
      ("zipf_90pct_hot_shard0", 0.9, 0.75),
  )
  def test_row_sharded_lookup_zipfian_skew_multi_round_overflow(self, hot_frac: float, buffer_factor: float):
    mesh = _make_mesh({"fsdp": 4})
    k_tbl, k_hot, k_cold, k_mask, k_w = jax.random.split(jax.random.PRNGKey(3), 5)
    v_total = 64
    v_loc = v_total // 4
    table = jax.random.normal(k_tbl, (v_total, 10), dtype=jnp.float32)
    hot_ids = jax.random.randint(k_hot, (4, 16), 0, v_loc, dtype=jnp.int32)
    cold_ids = jax.random.randint(k_cold, (4, 16), v_loc, v_total, dtype=jnp.int32)
    is_hot = jax.random.uniform(k_mask, (4, 16)) < hot_frac
    ids = jnp.where(is_hot, hot_ids, cold_ids)
    weights = jax.random.normal(k_w, (4, 16, 10), dtype=jnp.float32)

    table_s = jax.device_put(table, NamedSharding(mesh, P("fsdp", None)))
    ids_s = jax.device_put(ids, NamedSharding(mesh, P("fsdp", None)))
    w_s = jax.device_put(weights, NamedSharding(mesh, P("fsdp", None, None)))

    def _loss_routed(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = m3_se.row_sharded_embedding_lookup(
          t,
          ids_s,
          table_axes=("fsdp",),
          ids_specs=P("fsdp", None),
          out_dtype=jnp.float32,
          buffer_factor=buffer_factor,
          mesh=mesh,
      )
      return jnp.sum(out * w_s), out

    def _loss_ref(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = _reference_lookup(t, ids, jnp.float32)
      return jnp.sum(out * weights), out

    with jax.sharding.set_mesh(mesh):
      (_, out_routed), grad_routed = jax.jit(jax.value_and_grad(_loss_routed, has_aux=True))(table_s)
    (_, out_ref), grad_ref = jax.value_and_grad(_loss_ref, has_aux=True)(table)

    np.testing.assert_allclose(np.asarray(out_routed), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(grad_routed), np.asarray(grad_ref), rtol=1e-5, atol=1e-5)

  def test_row_sharded_lookup_invalid_ids_and_mixed_dtypes(self):
    mesh = _make_mesh({"fsdp": 4})
    k_tbl, k_w = jax.random.split(jax.random.PRNGKey(4), 2)
    table = jax.random.normal(k_tbl, (32, 8), dtype=jnp.float32)
    ids = jnp.array(
        [
            [-1, 0, 7, 32],
            [8, 15, 100, 31],
            [16, -5, 23, 24],
            [25, 30, 33, 1],
        ],
        dtype=jnp.int32,
    )
    weights = jax.random.normal(k_w, (4, 4, 8), dtype=jnp.bfloat16)

    table_s = jax.device_put(table, NamedSharding(mesh, P("fsdp", None)))
    ids_s = jax.device_put(ids, NamedSharding(mesh, P("fsdp", None)))
    w_s = jax.device_put(weights, NamedSharding(mesh, P("fsdp", None, None)))

    def _loss_routed(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = m3_se.row_sharded_embedding_lookup(
          t, ids_s, table_axes=("fsdp",), ids_specs=P("fsdp", None), out_dtype=jnp.bfloat16, mesh=mesh
      )
      return jnp.sum(out.astype(jnp.float32) * w_s.astype(jnp.float32)), out

    def _loss_ref(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = _reference_lookup(t, ids, jnp.bfloat16)
      return jnp.sum(out.astype(jnp.float32) * weights.astype(jnp.float32)), out

    with jax.sharding.set_mesh(mesh):
      (_, out_routed), grad_routed = jax.jit(jax.value_and_grad(_loss_routed, has_aux=True))(table_s)
    (_, out_ref), grad_ref = jax.value_and_grad(_loss_ref, has_aux=True)(table)

    self.assertEqual(out_routed.dtype, jnp.bfloat16)
    self.assertEqual(grad_routed.dtype, jnp.float32)
    np.testing.assert_allclose(
        np.asarray(out_routed, dtype=np.float32), np.asarray(out_ref, dtype=np.float32), rtol=0, atol=0
    )
    np.testing.assert_allclose(np.asarray(grad_routed), np.asarray(grad_ref), rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("ecp_true_auto", True, "auto"),
      ("ecp_true_explicit", True, "explicit"),
      ("ecp_false_auto", False, "auto"),
  )
  def test_qwen3_8_ngram_embedding_parity_and_state_invariance(self, ecp: bool, shard_mode: str):
    mesh = _make_mesh({"fsdp": 2, "expert": 2}, shard_mode=shard_mode)
    sharding = m3_sharding.EPasCP() if ecp else m3_sharding.DefaultSharding()
    eos_id = 3
    vocab_size = 256
    # Include EOS boundaries right around the CP split boundary (pos 5, 6, 7, 8) to exercise 2-token left halo.
    tokens = jnp.array(
        [
            [11, 12, 13, 14, 15, eos_id, 17, 18, 19, 20, 21, 22],
            [31, 32, 33, 34, eos_id, 36, 37, 38, eos_id, 40, 41, 42],
            [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
            [eos_id, 72, 73, 74, 75, 76, eos_id, eos_id, 79, 80, 81, 82],
        ],
        dtype=jnp.int32,
    )
    if ecp:
      tokens_spec = P("fsdp", "expert")
      out_spec = P("fsdp", "expert", None)
    else:
      tokens_spec = P(("fsdp", "expert"), None)
      out_spec = P(("fsdp", "expert"), None, None)

    weights = jax.random.normal(jax.random.PRNGKey(9), (4, 12, 32), dtype=jnp.float32)

    with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
      mod_routed = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=eos_id,
          vocab_size=vocab_size,
          ple_embed_dim=32,
          ngram_size=3,
          heads_per_ngram=2,
          ngram_vocab_size_base=100,
          make_ngram_vocab_size_divisible_by=4,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          use_index_routed_lookup=True,
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )
      mod_ref = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=eos_id,
          vocab_size=vocab_size,
          ple_embed_dim=32,
          ngram_size=3,
          heads_per_ngram=2,
          ngram_vocab_size_base=100,
          make_ngram_vocab_size_divisible_by=4,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          use_index_routed_lookup=False,
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )
      # State paths, shapes, and dtypes must be identical (m3 conformance invariant).
      state_routed = jax.tree_util.tree_flatten_with_path(nnx.state(mod_routed))[0]
      state_ref = jax.tree_util.tree_flatten_with_path(nnx.state(mod_ref))[0]
      self.assertEqual([p for p, _ in state_routed], [p for p, _ in state_ref])
      for (_, v1), (_, v2) in zip(state_routed, state_ref, strict=True):
        self.assertEqual(v1[...].shape, v2[...].shape)
        self.assertEqual(v1[...].dtype, v2[...].dtype)

      tok_s = jax.device_put(tokens, NamedSharding(mesh, tokens_spec))
      w_s = jax.device_put(weights, NamedSharding(mesh, out_spec))

      # Confirm the index-routed guard actually engages on this mesh.
      self.assertIsNotNone(
          mod_routed._resolve_index_routed_axes(  # pylint: disable=protected-access
              tok_s, mesh, mod_routed.ngram_embedding.embedding[...]
          )
      )

      gdef_r, params_r = nnx.split(mod_routed, nnx.Param)
      gdef_f, params_f = nnx.split(mod_ref, nnx.Param)

      def _run_r(p):
        out = nnx.merge(gdef_r, p)(tok_s)
        return jnp.sum(out * w_s), out

      def _run_f(p):
        out = nnx.merge(gdef_f, p)(tok_s)
        return jnp.sum(out * w_s), out

      (_, out_r), grad_r = jax.jit(jax.value_and_grad(_run_r, has_aux=True))(params_r)
      (_, out_f), grad_f = jax.jit(jax.value_and_grad(_run_f, has_aux=True))(params_f)

      np.testing.assert_allclose(np.asarray(out_r), np.asarray(out_f), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(
          np.asarray(grad_r["ngram_embedding"]["embedding"][...]),
          np.asarray(grad_f["ngram_embedding"]["embedding"][...]),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_qwen3_8_ngram_embedding_host_offload_and_decode_fallbacks(self):
    mesh = _make_mesh({"fsdp": 4})
    sharding = m3_sharding.DefaultSharding()
    with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
      mod = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=3,
          vocab_size=128,
          ple_embed_dim=32,
          ngram_size=3,
          heads_per_ngram=2,
          ngram_vocab_size_base=64,
          make_ngram_vocab_size_divisible_by=4,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )
      # Batch-1 decode is indivisible across fsdp=4 -> cleanly falls back to replicated lookup.
      decode_tok = jnp.array([[10, 11, 12, 13]], dtype=jnp.int32)
      self.assertIsNone(
          mod._resolve_index_routed_axes(  # pylint: disable=protected-access
              decode_tok, mesh, mod.ngram_embedding.embedding[...]
          )
      )
      out_decode = mod(decode_tok)
      self.assertEqual(out_decode.shape, (1, 4, 32))

      # Marking the embedding table with host-offload metadata (#6406) disables device index-routing cleanly.
      mod.ngram_embedding.embedding.set_metadata("memory_kind", "pinned_host")
      train_tok = jax.device_put(jnp.ones((4, 8), dtype=jnp.int32), NamedSharding(mesh, P("fsdp", None)))
      self.assertIsNone(
          mod._resolve_index_routed_axes(  # pylint: disable=protected-access
              train_tok, mesh, mod.ngram_embedding.embedding[...]
          )
      )

  def test_qwen3_8_ngram_embedding_survives_add_named_scopes_and_rejects_custom_subclass(self):
    mesh = _make_mesh({"fsdp": 2, "expert": 2})
    sharding = m3_sharding.EPasCP()
    tokens = jnp.arange(4 * 12, dtype=jnp.int32).reshape(4, 12) % 128

    class _Container(nnx.Module):

      def __init__(self, embed_mod: Any):
        self.ple_embedding = embed_mod

    class _CustomEmbed(nnx.Embed):
      pass

    with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
      mod = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=3,
          vocab_size=128,
          ple_embed_dim=32,
          ngram_size=3,
          heads_per_ngram=2,
          ngram_vocab_size_base=64,
          make_ngram_vocab_size_divisible_by=4,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          use_index_routed_lookup=True,
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )
      ref = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=3,
          vocab_size=128,
          ple_embed_dim=32,
          ngram_size=3,
          heads_per_ngram=2,
          ngram_vocab_size_base=64,
          make_ngram_vocab_size_divisible_by=4,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          use_index_routed_lookup=False,
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )
      # `model_creation_utils.create_model` applies `add_named_scopes` to every m3 submodule,
      # swapping `ngram_embedding.__class__` to a dynamic `_m3_named_scope` subclass.
      container = _Container(mod)
      m3_named_scopes.add_named_scopes(container)
      scoped_mod = container.ple_embedding
      self.assertIsNot(type(scoped_mod.ngram_embedding), nnx.Embed)
      self.assertTrue(qwen3_8._is_plain_embed(scoped_mod.ngram_embedding))  # pylint: disable=protected-access

      tok_s = jax.device_put(tokens, NamedSharding(mesh, P("fsdp", "expert")))
      self.assertEqual(
          scoped_mod._resolve_index_routed_axes(  # pylint: disable=protected-access
              tok_s, mesh, scoped_mod.ngram_embedding.embedding[...]
          ),
          (("fsdp", "expert"), ("fsdp",), ("expert",)),
      )

      # Verify the traced forward+backward actually dispatches through shard_map (no silent fallback)
      # and matches the reference numerically.
      gdef_s, params_s = nnx.split(scoped_mod, nnx.Param)
      gdef_r, params_r = nnx.split(ref, nnx.Param)

      def _loss_s(p):
        out = nnx.merge(gdef_s, p)(tok_s)
        return jnp.sum(out**2), out

      def _loss_r(p):
        out = nnx.merge(gdef_r, p)(tok_s)
        return jnp.sum(out**2), out

      jaxpr_text = str(jax.make_jaxpr(_loss_s)(params_s))
      self.assertIn("shard_map", jaxpr_text)

      (_, out_s), grad_s = jax.jit(jax.value_and_grad(_loss_s, has_aux=True))(params_s)
      (_, out_r), grad_r = jax.jit(jax.value_and_grad(_loss_r, has_aux=True))(params_r)
      np.testing.assert_allclose(np.asarray(out_s), np.asarray(out_r), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(
          np.asarray(grad_s["ngram_embedding"]["embedding"][...]),
          np.asarray(grad_r["ngram_embedding"]["embedding"][...]),
          rtol=1e-5,
          atol=1e-5,
      )

      # A genuine custom subclass (e.g. quantized embed) must still be rejected even after add_named_scopes.
      custom = _CustomEmbed(16, 4, rngs=nnx.Rngs(0))
      self.assertFalse(qwen3_8._is_plain_embed(custom))  # pylint: disable=protected-access
      m3_named_scopes.add_named_scopes(_Container(custom))
      self.assertFalse(qwen3_8._is_plain_embed(custom))  # pylint: disable=protected-access

  def test_validation_errors(self):
    mesh = _make_mesh({"fsdp": 4})
    table = jnp.zeros((16, 4), dtype=jnp.float32)
    ids = jnp.zeros((4, 2), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "buffer_factor"):
      m3_se.row_sharded_embedding_lookup(
          table, ids, table_axes=("fsdp",), ids_specs=P("fsdp", None), buffer_factor=0.0, mesh=mesh
      )
    with self.assertRaisesRegex(ValueError, "divisible"):
      m3_se.row_sharded_embedding_lookup(
          jnp.zeros((15, 4), dtype=jnp.float32),
          ids,
          table_axes=("fsdp",),
          ids_specs=P("fsdp", None),
          mesh=mesh,
      )
    with self.assertRaisesRegex(ValueError, "must be sharded on ids_specs"):
      m3_se.row_sharded_embedding_lookup(table, ids, table_axes=("fsdp",), ids_specs=P(None, None), mesh=mesh)

  def test_tpu_ragged_all_to_all_zipf_and_ple_parity(self):
    if not _RUN_TPU_PLE_BENCH.value:
      self.skipTest("Pass --run_tpu_ple_bench=true on TPU v7x 2x2x1.")
    self.assertEqual(jax.default_backend(), "tpu")
    devices = jax.devices()
    self.assertGreaterEqual(len(devices), 8)
    mesh8 = jax.sharding.Mesh(np.array(devices[:8]).reshape(2, 4), ("fsdp", "expert"))

    # 1. 85% single-shard Zipf skew with tight `buffer_factor=0.5` on real TPU `ragged-all-to-all`.
    k_tbl, k_hot, k_cold, k_mask, k_w = jax.random.split(jax.random.PRNGKey(77), 5)
    v_total = 1024
    v_loc = v_total // 8
    table = jax.random.normal(k_tbl, (v_total, 160), dtype=jnp.bfloat16)
    hot_ids = jax.random.randint(k_hot, (2, 128), 0, v_loc, dtype=jnp.int32)
    cold_ids = jax.random.randint(k_cold, (2, 128), v_loc, v_total, dtype=jnp.int32)
    ids = jnp.where(jax.random.uniform(k_mask, (2, 128)) < 0.85, hot_ids, cold_ids)
    weights = jax.random.normal(k_w, (2, 128, 160), dtype=jnp.bfloat16)

    table_s = jax.device_put(table, NamedSharding(mesh8, P(("fsdp", "expert"), None)))
    ids_s = jax.device_put(ids, NamedSharding(mesh8, P("fsdp", "expert")))
    w_s = jax.device_put(weights, NamedSharding(mesh8, P("fsdp", "expert", None)))

    def _tpu_routed(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = m3_se.row_sharded_embedding_lookup(
          t,
          ids_s,
          table_axes=("fsdp", "expert"),
          ids_specs=P("fsdp", "expert"),
          out_dtype=jnp.bfloat16,
          buffer_factor=0.5,
          mesh=mesh8,
      )
      return jnp.sum(out.astype(jnp.float32) * w_s.astype(jnp.float32)), out

    def _tpu_ref(t: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = _reference_lookup(t, ids, jnp.bfloat16)
      return jnp.sum(out.astype(jnp.float32) * weights.astype(jnp.float32)), out

    with jax.sharding.set_mesh(mesh8):
      jitted = jax.jit(jax.value_and_grad(_tpu_routed, has_aux=True))
      compiled = jitted.lower(table_s).compile()
      self.assertIn("ragged-all-to-all", compiled.as_text())
      (_, out_r), grad_r = compiled(table_s)
    (_, out_f), grad_f = jax.jit(jax.value_and_grad(_tpu_ref, has_aux=True))(table)
    np.testing.assert_array_equal(np.asarray(out_r), np.asarray(out_f))
    np.testing.assert_allclose(
        np.asarray(grad_r, dtype=np.float32), np.asarray(grad_f, dtype=np.float32), rtol=2e-2, atol=2e-2
    )

    # 2. Full `EPasCP()` `Qwen3_8NGramEmbedding` parity across 8 TensorCores with EOS tokens at shard boundaries.
    shd_ep_cp = m3_sharding.EPasCP()
    tok = jax.random.randint(jax.random.PRNGKey(88), (2, 256), 4, 512, dtype=jnp.int32)
    tok = tok.at[:, 63].set(3).at[:, 64].set(3).at[:, 127].set(3)
    cot_ple = jax.random.normal(jax.random.PRNGKey(89), (2, 256, 2560), dtype=jnp.bfloat16)
    with jax.sharding.set_mesh(mesh8), nnx.use_eager_sharding(True):
      ple_routed = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=3,
          vocab_size=1024,
          ple_embed_dim=2560,
          ngram_size=3,
          heads_per_ngram=8,
          ngram_vocab_size_base=4096,
          make_ngram_vocab_size_divisible_by=128,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          use_index_routed_lookup=True,
          sharding=shd_ep_cp,
          rngs=nnx.Rngs(0),
      )
      ple_ref = qwen3_8.Qwen3_8NGramEmbedding(
          eos_token_id=3,
          vocab_size=1024,
          ple_embed_dim=2560,
          ngram_size=3,
          heads_per_ngram=8,
          ngram_vocab_size_base=4096,
          make_ngram_vocab_size_divisible_by=128,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          use_index_routed_lookup=False,
          sharding=shd_ep_cp,
          rngs=nnx.Rngs(0),
      )
      tok_8 = jax.device_put(tok, NamedSharding(mesh8, P("fsdp", "expert")))
      cot_8 = jax.device_put(cot_ple, NamedSharding(mesh8, P("fsdp", "expert", None)))
      gd_p, ps_p = nnx.split(ple_routed, nnx.Param)
      gd_r, ps_r = nnx.split(ple_ref, nnx.Param)

      @jax.jit
      def _step_p(p):
        y = nnx.merge(gd_p, p)(tok_8)
        return jnp.sum(y.astype(jnp.float32) * cot_8.astype(jnp.float32)), y

      @jax.jit
      def _step_r(p):
        y = nnx.merge(gd_r, p)(tok_8)
        return jnp.sum(y.astype(jnp.float32) * cot_8.astype(jnp.float32)), y

      jitted_p = jax.jit(jax.value_and_grad(_step_p, has_aux=True))
      compiled_p = jitted_p.lower(ps_p).compile()
      self.assertIn("ragged-all-to-all", compiled_p.as_text())
      (_, y_p), gp = compiled_p(ps_p)
      (_, y_r), gr = jax.jit(jax.value_and_grad(_step_r, has_aux=True))(ps_r)
      np.testing.assert_array_equal(np.asarray(y_p), np.asarray(y_r))
      np.testing.assert_allclose(
          np.asarray(gp["ngram_embedding"]["embedding"][...], dtype=np.float32),
          np.asarray(gr["ngram_embedding"]["embedding"][...], dtype=np.float32),
          rtol=2e-2,
          atol=2e-2,
      )


if __name__ == "__main__":
  absltest.main()
