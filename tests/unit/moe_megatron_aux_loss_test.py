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

"""Tests for the Megatron-LM MoE aux loss and expert-bias update of RoutedMoE.

Sigmoid routers use Megatron-LM's seq_aux_loss, computed once per full sequence,
and the loss-free-balancing bias update is one sign() over the full batch. Also
covers the moe_log_max_load_ratio debug metric. The reference functions below
are line-by-line ports of Megatron-LM (TopKRouter._apply_seq_aux_loss,
compute_routing_scores_for_aux_loss, switch_load_balancing_loss_func and
get_updated_expert_bias) for a single rank without padding.
"""

from types import SimpleNamespace
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from tests.utils.test_helpers import get_test_config_path


def reference_megatron_seq_aux_loss(logits, topk, coeff):
  """Megatron-LM seq_aux_loss for the sigmoid router; logits are (batch, seq, experts)."""
  bsz, seq_length, num_experts = logits.shape
  # Megatron tokens are laid out [seq, batch] before being flattened.
  logits = jnp.transpose(logits, (1, 0, 2)).reshape(seq_length * bsz, num_experts)
  # compute_routing_scores_for_aux_loss(score_function="sigmoid").
  scores = jax.nn.sigmoid(logits.astype(jnp.float32))
  scores = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-20)
  _, top_indices = jax.lax.top_k(scores, topk)
  routing_map = jnp.zeros(scores.shape, jnp.int32).at[jnp.arange(scores.shape[0])[:, None], top_indices].set(1)
  # _apply_seq_aux_loss: fold the batch into the expert dimension.
  scores = scores.reshape(seq_length, -1)
  routing_map = routing_map.reshape(seq_length, -1)
  tokens_per_expert = jnp.sum(routing_map, axis=0)
  total_num_tokens = seq_length
  # switch_load_balancing_loss_func(...) / bsz.
  aggregated_probs_per_expert = jnp.sum(scores, axis=0)
  aux_loss = jnp.sum(aggregated_probs_per_expert * tokens_per_expert) * (
      num_experts * coeff / (topk * total_num_tokens * total_num_tokens)
  )
  return aux_loss / bsz


def reference_updated_expert_bias(tokens_per_expert, expert_bias, rate):
  """Megatron-LM get_updated_expert_bias with tokens_per_expert already all-reduced."""
  average_tokens = jnp.sum(tokens_per_expert, axis=-1, keepdims=True) / tokens_per_expert.shape[-1]
  return expert_bias + jnp.sign(average_tokens - tokens_per_expert) * rate


class _RouterStub:
  """The RoutedMoE attributes the aux-loss methods read."""

  load_balance_loss = moe.RoutedMoE.load_balance_loss
  megatron_seq_aux_loss = moe.RoutedMoE.megatron_seq_aux_loss

  def __init__(self, num_experts, num_experts_per_tok, load_balance_loss_weight):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.config = SimpleNamespace(load_balance_loss_weight=load_balance_loss_weight)


class MegatronSeqAuxLossTest(unittest.TestCase):
  """Pure-function checks against the Megatron-LM formulas."""

  def test_loss_matches_megatron_value_and_grad(self):
    bsz, seq_length, num_experts, topk, coeff = 3, 32, 16, 4, 0.01
    logits = jax.random.normal(jax.random.PRNGKey(0), (bsz, seq_length, num_experts), jnp.float32) * 2.0
    router = _RouterStub(num_experts, topk, coeff)

    def maxtext_loss(x):
      # pre_bias_logits of the sigmoid router are the sigmoid scores before the expert bias.
      return router.megatron_seq_aux_loss(jax.nn.sigmoid(x))

    def reference_loss(x):
      return reference_megatron_seq_aux_loss(x, topk, coeff)

    np.testing.assert_allclose(maxtext_loss(logits), reference_loss(logits), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(maxtext_loss)(logits), jax.grad(reference_loss)(logits), rtol=1e-5, atol=1e-10)

  def test_inputs_ignore_bias_and_normalize(self):
    scores = jnp.array([[[0.9, 0.1, 0.5, 0.7]]], jnp.bfloat16)
    probs, top_k = moe.megatron_aux_loss_inputs(scores, 2)
    self.assertEqual(probs.dtype, jnp.float32)
    np.testing.assert_allclose(jnp.sum(probs, axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(jnp.sort(top_k, axis=-1), [[[0, 3]]])

  def test_chunked_loss_differs_from_full_sequence(self):
    """Averaging the loss of two sequence halves is not the per-sequence loss."""
    num_experts, topk, coeff = 4, 1, 0.01
    half = 8
    # Every sequence prefers expert 0 in its first half and expert 1 in its second half.
    first = jnp.tile(jnp.array([4.0, -4.0, -4.0, -4.0]), (2, half, 1))
    second = jnp.tile(jnp.array([-4.0, 4.0, -4.0, -4.0]), (2, half, 1))
    logits = jnp.concatenate([first, second], axis=1)
    router = _RouterStub(num_experts, topk, coeff)
    full = router.megatron_seq_aux_loss(jax.nn.sigmoid(logits))
    chunked = 0.5 * (
        router.megatron_seq_aux_loss(jax.nn.sigmoid(logits[:, :half]))
        + router.megatron_seq_aux_loss(jax.nn.sigmoid(logits[:, half:]))
    )
    np.testing.assert_allclose(full, reference_megatron_seq_aux_loss(logits, topk, coeff), rtol=1e-6)
    # Per half the routing is fully collapsed (loss ~ E * coeff); over the sequence it is split in two.
    self.assertGreater(float(chunked), 1.8 * float(full))

  def test_uses_megatron_seq_aux_loss(self):
    def config(routed_score_func, te_moe_block=False, moe_use_megatron_seq_aux_loss=True):
      return SimpleNamespace(
          routed_score_func=routed_score_func,
          te_moe_block=te_moe_block,
          moe_use_megatron_seq_aux_loss=moe_use_megatron_seq_aux_loss,
      )

    self.assertTrue(moe.uses_megatron_seq_aux_loss(config("sigmoid")))
    self.assertFalse(moe.uses_megatron_seq_aux_loss(config("sigmoid", moe_use_megatron_seq_aux_loss=False)))
    # Missing attribute on a stub config defaults to False.
    self.assertFalse(moe.uses_megatron_seq_aux_loss(SimpleNamespace(routed_score_func="sigmoid")))
    for score_func in ("softmax", "sqrtsoftplus", ""):
      self.assertFalse(moe.uses_megatron_seq_aux_loss(config(score_func)), score_func)
    # TransformerEngine's MoEBlock computes its own aux loss.
    self.assertFalse(moe.uses_megatron_seq_aux_loss(config("sigmoid", te_moe_block=True)))
    # Base config defaults to moe_use_megatron_seq_aux_loss=False unless enabled.
    default_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="predicate_deepseek3_default",
        enable_checkpointing=False,
        model_name="deepseek3-tiny",
    )
    self.assertFalse(default_cfg.moe_use_megatron_seq_aux_loss)
    self.assertFalse(moe.uses_megatron_seq_aux_loss(default_cfg))
    self.assertTrue(moe.uses_megatron_seq_aux_loss(_tiny_deepseek_config("predicate_deepseek3")))
    self.assertFalse(
        moe.uses_megatron_seq_aux_loss(
            _tiny_deepseek_config("predicate_deepseek3_off", moe_use_megatron_seq_aux_loss=False)
        )
    )


class ExpertBiasUpdateTest(unittest.TestCase):
  """Loss-free-balancing bias update helpers."""

  rate = 1e-3

  def test_calculate_load_balance_updates_unchanged(self):
    top_k = jax.random.randint(jax.random.PRNGKey(1), (4, 16, 2), 0, 8)
    counts = jnp.bincount(top_k.ravel(), length=8)
    expected = jnp.sign(jnp.sum(counts) / 8 - counts) * self.rate
    np.testing.assert_array_equal(moe.calculate_load_balance_updates(top_k, 8, self.rate), expected)
    np.testing.assert_array_equal(moe.calculate_expert_counts(top_k, 8), counts)

  def test_padding_is_not_counted(self):
    np.testing.assert_array_equal(moe.calculate_expert_counts(jnp.array([[0, -1], [2, -1]]), 3), [1, 0, 1])

  def test_full_batch_update_matches_megatron(self):
    """One sign of the summed chunk counts is Megatron's update; the mean of per-chunk signs is not."""
    chunk1 = jnp.array([0, 0, 0, 0, 2, 3])  # counts [4, 0, 1, 1]
    chunk2 = jnp.array([0, 1, 1, 1, 2, 3])  # counts [1, 3, 1, 1]
    num_experts = 4
    counts = moe.calculate_expert_counts(chunk1, num_experts) + moe.calculate_expert_counts(chunk2, num_experts)
    full_batch = moe.load_balance_updates_from_counts(counts, num_experts, self.rate)
    reference = reference_updated_expert_bias(counts, jnp.zeros(num_experts), self.rate)
    np.testing.assert_array_equal(full_batch, reference)
    np.testing.assert_allclose(full_batch, [-self.rate, 0.0, self.rate, self.rate])
    per_chunk_mean = 0.5 * (
        moe.calculate_load_balance_updates(chunk1, num_experts, self.rate)
        + moe.calculate_load_balance_updates(chunk2, num_experts, self.rate)
    )
    np.testing.assert_allclose(per_chunk_mean, [0.0, 0.0, self.rate, self.rate])


def _tiny_deepseek_config(run_name, **overrides):
  """Returns a tiny float32 DeepSeek-V3 config on CPU; `overrides` replace the defaults below."""
  kwargs = {
      "run_name": run_name,
      "enable_checkpointing": False,
      "model_name": "deepseek3-tiny",
      "override_model_config": True,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      # megablox runs its Pallas kernels in interpret mode off-TPU.
      "megablox": True,
      "sparse_matmul": True,
      "per_device_batch_size": 2,
      "max_target_length": 16,
      "load_balance_loss_weight": 0.01,
      "moe_use_megatron_seq_aux_loss": True,
      "routed_bias_update_rate": 1e-3,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


# Ring-of-experts ragged path, needed by moe_log_max_load_ratio.
_RAGGED_ROE = {
    "ici_expert_parallelism": 2,
    "use_ring_of_experts": True,
    "use_ragged_sort": True,
    "ragged_gather_fallback": True,
    "ragged_gather_reduce_fallback": True,
}


class MaxLoadRatioTest(unittest.TestCase):
  """The moe_log_max_load_ratio debug metric."""

  def test_max_expert_shard_load_ratio(self):
    group_sizes = jnp.array([3, 1, 0, 0, 2, 2, 4, 4], jnp.int32)  # shard loads [4, 0, 4, 8]
    ratio = moe.max_expert_shard_load_ratio(group_sizes, num_expert_shards=4, balanced_load=4)
    self.assertEqual(ratio.dtype, jnp.float32)
    self.assertAlmostEqual(float(ratio), 2.0)
    # Float balanced_load when bsz_times_seq_len < num_expert_parallelism (2 / 4 * 2 = 1.0, not 2 // 4 * 2 = 0).
    small_batch_groups = jnp.array([2, 0, 1, 0, 1, 0, 0, 0], jnp.int32)  # shard loads [2, 1, 1, 0]
    small_ratio = moe.max_expert_shard_load_ratio(small_batch_groups, num_expert_shards=4, balanced_load=(2 / 4) * 2)
    self.assertAlmostEqual(float(small_ratio), 2.0)

  def test_max_load_ratio_validation(self):
    with self.assertRaisesRegex(ValueError, "use_ring_of_experts"):
      _tiny_deepseek_config("ratio_no_roe", moe_log_max_load_ratio=True)
    cfg = _tiny_deepseek_config("ratio_roe", moe_log_max_load_ratio=True, **_RAGGED_ROE)
    self.assertTrue(cfg.moe_log_max_load_ratio)


def _run_routed_moe(cfg, params=None):
  """Runs RoutedMoE forward and backward on fixed random inputs.

  Args:
    cfg: The config to build the RoutedMoE block and its mesh from.
    params: Optional parameters; freshly initialized when None.

  Returns:
    ((output, lb_loss, bias_updates, intermediates), grads, params, inputs).
  """
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = moe.get_routed_moe(
      name="MoeBlock",
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.moe_mlp_dim,
      dtype=cfg.dtype,
  )
  rng_model, rng_inputs = jax.random.split(jax.random.PRNGKey(7))
  inputs = jax.random.normal(
      rng_inputs, (cfg.per_device_batch_size * jax.device_count(), cfg.max_target_length, cfg.base_emb_dim)
  )
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    if params is None:
      params = model.init({"params": rng_model, "dropout": rng_model}, inputs)["params"]

    def loss_fn(params):
      (out, lb_loss, bias_updates), mutated = model.apply({"params": params}, inputs, mutable=["intermediates"])
      return jnp.mean(out**2) + lb_loss, (out, lb_loss, bias_updates, mutated)

    (_, aux), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
  return aux, grads, params, inputs


def _reference_loss(cfg, params, inputs):
  """Megatron-LM seq_aux_loss of the gate logits (the routed bias is zero-initialized)."""
  gate_kernel = params["gate"]["kernel"].value
  logits = jnp.einsum("bse,ex->bsx", inputs, gate_kernel, precision=jax.lax.Precision.HIGHEST)
  return reference_megatron_seq_aux_loss(logits, cfg.num_experts_per_tok, cfg.load_balance_loss_weight)


class RoutedMoeLossTest(unittest.TestCase):
  """Runs RoutedMoE end to end on the dropless sparse and dense paths."""

  def test_sparse_and_dense_losses_match_megatron_reference(self):
    sparse_cfg = _tiny_deepseek_config("loss_sparse")
    (_, lb_sparse, _, _), _, params, inputs = _run_routed_moe(sparse_cfg)
    (_, lb_dense, _, _), _, _, _ = _run_routed_moe(_tiny_deepseek_config("loss_dense", sparse_matmul=False), params)
    reference = _reference_loss(sparse_cfg, params, inputs)
    np.testing.assert_allclose(lb_sparse, reference, rtol=1e-5)
    np.testing.assert_allclose(lb_dense, lb_sparse, rtol=1e-6)

  def test_softmax_router_keeps_switch_loss(self):
    """Non-sigmoid routers are unchanged: the Switch-style loss of the dispatch top-k."""
    cfg = _tiny_deepseek_config("loss_softmax", routed_score_func="softmax")
    (_, lb_loss, _, _), _, params, inputs = _run_routed_moe(cfg)
    self.assertFalse(moe.uses_megatron_seq_aux_loss(cfg))
    self.assertFalse(np.allclose(lb_loss, _reference_loss(cfg, params, inputs), rtol=1e-3))

  def test_sigmoid_router_can_disable_megatron_seq_aux_loss(self):
    """Setting moe_use_megatron_seq_aux_loss=False falls back to the Switch-style loss."""
    cfg = _tiny_deepseek_config("loss_sigmoid_switch", moe_use_megatron_seq_aux_loss=False)
    (_, lb_loss, _, _), _, params, inputs = _run_routed_moe(cfg)
    self.assertFalse(moe.uses_megatron_seq_aux_loss(cfg))
    self.assertFalse(np.allclose(lb_loss, _reference_loss(cfg, params, inputs), rtol=1e-3))


class _RoutedMoEDecoder(nnx.Module):
  """Minimal decoder wrapper around RoutedMoE that sows moe_lb_loss and moe_bias_updates for train.loss_fn."""

  def __init__(self, cfg, mesh, rngs):
    self.moe = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

  def __call__(self, x):
    out, lb_loss, bias_updates = self.moe(x)
    if lb_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", lb_loss)
    if bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", bias_updates)
    return out


class _TinyRoutedMoEModel(nnx.Module):
  """Minimal NNX model with a real RoutedMoE decoder compatible with train.loss_fn."""

  def __init__(self, cfg, mesh, rngs):
    self.mesh = mesh
    self.embed = nnx.Embed(cfg.vocab_size, cfg.base_emb_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
    self.decoder = _RoutedMoEDecoder(cfg, mesh, rngs=rngs)
    self.proj = nnx.Linear(cfg.base_emb_dim, cfg.vocab_size, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      encoder_images=None,
      encoder_image_masks=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks
    del enable_dropout, decoder_target_tokens, decoder_target_mask
    h = self.embed(decoder_input_tokens)
    h = self.decoder(h)
    return self.proj(h)


class RingOfExpertsIntegrationTest(unittest.TestCase):
  """Runs RoutedMoE on the ring-of-experts path with expert parallelism 2."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2 or jax.device_count() % 2:
      self.skipTest("Needs an even number (>= 2) of devices for expert parallelism 2.")

  def _run(self, params=None, **overrides):
    cfg = _tiny_deepseek_config(
        "roe_" + "_".join(f"{k}{v}" for k, v in sorted(overrides.items())),
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        **overrides,
    )
    return cfg, _run_routed_moe(cfg, params)

  def test_loss_and_bias_update_are_chunk_invariant(self):
    cfg, ((out1, lb1, bias1, _), grads1, params, inputs) = self._run(num_moe_token_chunks=1)
    _, ((out2, lb2, bias2, _), grads2, _, _) = self._run(params, num_moe_token_chunks=2)
    np.testing.assert_allclose(out2, out1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lb2, lb1, rtol=1e-6)
    np.testing.assert_allclose(lb1, _reference_loss(cfg, params, inputs), rtol=1e-5)
    np.testing.assert_array_equal(bias2, bias1)
    for g2, g1 in zip(jax.tree_util.tree_leaves(grads2), jax.tree_util.tree_leaves(grads1)):
      np.testing.assert_allclose(g2, g1, rtol=1e-4, atol=1e-6)
    # Full-batch update: every entry is exactly -rate, 0 or +rate.
    np.testing.assert_array_equal(jnp.isin(jnp.abs(bias2), jnp.array([0.0, 1e-3], jnp.float32)), True)

  def test_gradient_accumulation_matches_single_step(self):
    """GA=2 matches GA=1 for total loss, logged moe_lb_loss, full-batch bias updates, and router gradients."""

    def make_cfg(ga_steps):
      return _tiny_deepseek_config(
          f"roe_ga{ga_steps}",
          vocab_size=64,
          grad_dtype="float32",
          per_device_batch_size=2 // ga_steps,
          gradient_accumulation_steps=ga_steps,
          ici_expert_parallelism=2,
          ici_fsdp_parallelism=jax.device_count() // 2,
          use_ring_of_experts=True,
          num_moe_token_chunks=2,
      )

    cfg1 = make_cfg(1)
    cfg2 = make_cfg(2)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg1), cfg1.mesh_axes)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg1.logical_axis_rules):
      model1 = _TinyRoutedMoEModel(cfg1, mesh, rngs=nnx.Rngs(params=0))
      model2 = _TinyRoutedMoEModel(cfg2, mesh, rngs=nnx.Rngs(params=0))
      nnx.update(model2, nnx.state(model1))

    gbs, seq = cfg1.global_batch_size_to_train_on, cfg1.max_target_length
    inputs = jax.random.randint(jax.random.PRNGKey(42), (gbs, seq), 1, cfg1.vocab_size)
    targets = jax.random.randint(jax.random.PRNGKey(43), (gbs, seq), 1, cfg1.vocab_size)
    pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (gbs, seq))
    seg = jnp.ones((gbs, seq), dtype=jnp.int32)
    data = {
        "inputs": inputs,
        "targets": targets,
        "inputs_position": pos,
        "targets_position": pos,
        "inputs_segmentation": seg,
        "targets_segmentation": seg,
    }
    _, params1, _ = nnx.split(model1, nnx.Param, ...)
    pshardings = jax.tree.map(lambda v: NamedSharding(mesh, PartitionSpec()), params1)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg1.logical_axis_rules):
      (loss1, aux1), grads1 = nnx.value_and_grad(train.loss_fn, argnums=0, has_aux=True)(
          model1, cfg1, dict(data), None, None, is_train=True
      )
      loss2, aux2, grads2 = gradient_accumulation_loss_and_grad(
          train.loss_fn, cfg2, model2, None, pshardings, dict(data), None
      )

    np.testing.assert_allclose(loss2, loss1, rtol=1e-5)
    np.testing.assert_allclose(aux2["moe_lb_loss"], aux1["moe_lb_loss"], rtol=1e-5)
    bias1 = aux1["moe_bias_updates"][0]
    bias2 = aux2["moe_bias_updates"][0]
    np.testing.assert_array_equal(bias2, bias1)
    np.testing.assert_array_equal(jnp.isin(jnp.abs(bias2), jnp.array([0.0, 1e-3], jnp.float32)), True)
    for g2, g1 in zip(jax.tree_util.tree_leaves(grads2), jax.tree_util.tree_leaves(grads1)):
      np.testing.assert_allclose(g2, g1, rtol=1e-4, atol=1e-6)

  def test_load_ratio_predicts_overflow(self):
    ragged = {
        "num_moe_token_chunks": 2,
        "use_ragged_sort": True,
        "ragged_gather_fallback": True,
        "ragged_gather_reduce_fallback": True,
        "moe_log_max_load_ratio": True,
    }

    def ratio_and_overflow(ragged_buffer_factor):
      _, ((_, _, _, mutated), _, _, _) = self._run(ragged_buffer_factor=ragged_buffer_factor, **ragged)
      ratios = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_max_load_ratio")
      overflow = maxtext_utils.collect_intermediates_by_suffix(mutated, "moe_has_overflow")
      self.assertTrue(ratios, "Expected a moe_max_load_ratio intermediate to be sown.")
      return float(jnp.max(jnp.concatenate(ratios))), bool(jnp.any(jnp.concatenate(overflow)))

    ratio, overflow = ratio_and_overflow(-1.0)  # dropless: no truncated buffer
    self.assertFalse(overflow)
    self.assertGreater(ratio, 1.0)
    # The routing does not depend on the buffer, so the ratio is the same and decides the overflow.
    ratio_small, overflow_small = ratio_and_overflow(0.97 * ratio)
    ratio_large, overflow_large = ratio_and_overflow(1.03 * ratio)
    self.assertAlmostEqual(ratio_small, ratio, places=6)
    self.assertAlmostEqual(ratio_large, ratio, places=6)
    self.assertTrue(overflow_small)
    self.assertFalse(overflow_large)


if __name__ == "__main__":
  unittest.main()
