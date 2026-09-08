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

"""Unit tests for the Qwen3.5 standalone torchax converter and its rollout wiring.

The converter tests run the real `Qwen35MaxTextToVLLMConverter.convert()` on a
tiny synthetic scanned state (2 slots: GDN + full attention, 2 repetitions) and
check the emitted tensors against expectations computed directly from the
inputs with plain numpy data movement, in the three supported sharding modes
(plain TP, attention DP, expert parallelism).
"""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import jax.numpy as jnp
import pytest

from maxtext.integration.vllm import maxtext_vllm_rollout as rollout_mod
from maxtext.integration.vllm.maxtext_vllm_rollout import (
    MaxTextVllmSampler,
    _create_model_converter,
)
from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen35_moe import Qwen35MaxTextToVLLMConverter

pytestmark = pytest.mark.post_training

# Synthetic model dimensions: 4 layers in a 2-slot cycle (slot 0 GDN, slot 1
# full attention), so 2 repetitions of each slot.
D = 8  # d_model
VOCAB = 16
NUM_LAYERS = 4
NUM_SLOTS = 2
REPS = NUM_LAYERS // NUM_SLOTS
HQ, DHQ = 2, 4  # full-attention query heads (per-head dim includes the output gate)
NKV, DHKV = 1, 4  # a single KV head, so KV replication triggers at attn_shards > 1
HK, HV, DK, DV = 2, 4, 4, 4  # GDN heads/dims (V_per_K = 2)
E, F = 2, 8  # routed experts / expert inner dim
FS = 8  # shared-expert inner dim
CONV_W, CONV_C = 4, 6  # GDN short-conv width / channels
PAD = 128  # the converter pads each expert gate/up chunk to a multiple of 128


def _make_config(tp):
  return SimpleNamespace(
      base_num_decoder_layers=NUM_LAYERS,
      rollout_tensor_parallelism=tp,
      inhomogeneous_layer_cycle_interval=NUM_SLOTS,
      gdn_num_key_heads=HK,
      gdn_num_value_heads=HV,
      gdn_key_head_dim=DK,
      gdn_value_head_dim=DV,
  )


def _make_converter(tp, attn_dp=1, use_ep=False):
  return Qwen35MaxTextToVLLMConverter(_make_config(tp), mesh=None, vllm_attn_dp=attn_dp, vllm_use_ep=use_ep)


def _mlp_block(rng):
  return {
      "routed_experts": {
          "gate": {"kernel": rng.standard_normal((D, REPS, E), dtype=np.float32)},
          "wi_0": rng.standard_normal((E, REPS, D, F), dtype=np.float32),
          "wi_1": rng.standard_normal((E, REPS, D, F), dtype=np.float32),
          "wo": rng.standard_normal((E, REPS, F, D), dtype=np.float32),
      },
      "shared_expert": {
          "wi_0": {"kernel": rng.standard_normal((D, REPS, FS), dtype=np.float32)},
          "wi_1": {"kernel": rng.standard_normal((D, REPS, FS), dtype=np.float32)},
          "wo": {"kernel": rng.standard_normal((FS, REPS, D), dtype=np.float32)},
      },
      "shared_expert_gate": {"kernel": rng.standard_normal((D, REPS, 1), dtype=np.float32)},
  }


def _synthetic_state(seed=0):
  """A minimal scanned Qwen3.5 state with the tree structure convert() reads."""
  rng = np.random.default_rng(seed)
  gdn_slot = {
      "input_layernorm": {"scale": rng.standard_normal((D, REPS), dtype=np.float32)},
      "post_attention_layernorm": {"scale": rng.standard_normal((D, REPS), dtype=np.float32)},
      "attention": {
          "in_proj_qkvz": {"kernel": rng.standard_normal((D, REPS, HK * (2 * DK + 4 * DV)), dtype=np.float32)},
          "in_proj_ba": {"kernel": rng.standard_normal((D, REPS, HK * 4), dtype=np.float32)},
          "out_proj": {"kernel": rng.standard_normal((HV * DV, REPS, D), dtype=np.float32)},
          "conv1d": {"kernel": rng.standard_normal((CONV_W, REPS, 1, CONV_C), dtype=np.float32)},
          "A_log": rng.standard_normal((HV, REPS), dtype=np.float32),
          "dt_bias": rng.standard_normal((HV, REPS), dtype=np.float32),
          "norm": {"rms_norm": {"scale": rng.standard_normal((HV * DV, REPS), dtype=np.float32)}},
      },
      "mlp": _mlp_block(rng),
  }
  attn_slot = {
      "input_layernorm": {"scale": rng.standard_normal((D, REPS), dtype=np.float32)},
      "post_attention_layernorm": {"scale": rng.standard_normal((D, REPS), dtype=np.float32)},
      "attention": {
          "attention": {
              "query": {"kernel": rng.standard_normal((D, REPS, HQ, DHQ), dtype=np.float32)},
              "key": {"kernel": rng.standard_normal((D, REPS, NKV, DHKV), dtype=np.float32)},
              "value": {"kernel": rng.standard_normal((D, REPS, NKV, DHKV), dtype=np.float32)},
              "out": {"kernel": rng.standard_normal((HQ * DHQ, REPS, D), dtype=np.float32)},
              "query_norm": {"scale": rng.standard_normal((DHQ, REPS), dtype=np.float32)},
              "key_norm": {"scale": rng.standard_normal((DHKV, REPS), dtype=np.float32)},
          }
      },
      "mlp": _mlp_block(rng),
  }
  return {
      "base": {
          "token_embedder": {"embedding": rng.standard_normal((VOCAB, D), dtype=np.float32)},
          "decoder": {
              "decoder_norm": {"scale": rng.standard_normal((D,), dtype=np.float32)},
              "logits_dense": {"kernel": rng.standard_normal((D, VOCAB), dtype=np.float32)},
              "scanned_blocks": {"layers_0": gdn_slot, "layers_1": attn_slot},
          },
      }
  }


def _bf16(x):
  return jnp.asarray(x).astype(jnp.bfloat16)


def _expected_keys():
  """The full runner-state key set convert() must emit for the synthetic model."""
  keys = {
      "vllm_model.language_model.model.embed_tokens.weight",
      "vllm_model.language_model.model.norm.weight",
      "vllm_model.language_model.lm_head.weight",
  }
  for i in range(NUM_LAYERS):
    p = f"vllm_model.language_model.model.layers.{i}"
    keys |= {
        f"{p}.input_layernorm.weight",
        f"{p}.post_attention_layernorm.weight",
        f"{p}.mlp.gate.weight",
        f"{p}.mlp.experts.routed_experts.w13_weight",
        f"{p}.mlp.experts.routed_experts.w2_weight",
        f"{p}.mlp.experts.w13_weight",
        f"{p}.mlp.experts.w2_weight",
        f"{p}.mlp.shared_expert.gate_up_proj.weight",
        f"{p}.mlp.shared_expert.down_proj.weight",
        f"{p}.mlp.shared_expert_gate.weight",
    }
    if i % NUM_SLOTS == NUM_SLOTS - 1:  # full attention slot
      keys |= {
          f"{p}.self_attn.qkv_proj.weight",
          f"{p}.self_attn.o_proj.weight",
          f"{p}.self_attn.q_norm.weight",
          f"{p}.self_attn.k_norm.weight",
      }
    else:  # GDN slot
      keys |= {
          f"{p}.linear_attn.in_proj_qkvz.weight",
          f"{p}.linear_attn.in_proj_ba.weight",
          f"{p}.linear_attn.out_proj.weight",
          f"{p}.linear_attn.conv1d.weight",
          f"{p}.linear_attn.A_log",
          f"{p}.linear_attn.dt_bias",
          f"{p}.linear_attn.norm.weight",
      }
  return keys


class ConverterInitTest(unittest.TestCase):

  def test_attn_shards_derivation(self):
    self.assertEqual(_make_converter(tp=8, attn_dp=4).attn_shards, 2)
    self.assertEqual(_make_converter(tp=2).attn_shards, 2)

  def test_none_attn_dp_defaults_to_one(self):
    conv = _make_converter(tp=2, attn_dp=None)
    self.assertEqual(conv.vllm_attn_dp, 1)
    self.assertEqual(conv.attn_shards, 2)

  def test_indivisible_attn_dp_rejected(self):
    with self.assertRaises(AssertionError):
      _make_converter(tp=4, attn_dp=3)


class ReplicateKvHeadsTest(unittest.TestCase):

  def test_noop_when_shards_do_not_exceed_heads(self):
    conv = _make_converter(tp=2)  # attn_shards = 2
    kv = jnp.arange(D * 2 * DHKV, dtype=jnp.float32).reshape(D, 2, DHKV)
    self.assertIs(conv._replicate_kv_heads(kv), kv)  # pylint: disable=protected-access

  def test_consecutive_replication(self):
    conv = _make_converter(tp=4)  # attn_shards = 4
    kv = jnp.arange(D * 2 * DHKV, dtype=jnp.float32).reshape(D, 2, DHKV)
    out = conv._replicate_kv_heads(kv)  # pylint: disable=protected-access
    self.assertEqual(out.shape, (D, 4, DHKV))
    # Each head repeated consecutively: [h0, h0, h1, h1].
    self.assertTrue(jnp.array_equal(out[:, 0], kv[:, 0]))
    self.assertTrue(jnp.array_equal(out[:, 1], kv[:, 0]))
    self.assertTrue(jnp.array_equal(out[:, 2], kv[:, 1]))
    self.assertTrue(jnp.array_equal(out[:, 3], kv[:, 1]))

  def test_indivisible_head_count_rejected(self):
    conv = _make_converter(tp=4)  # attn_shards = 4
    kv = jnp.zeros((D, 3, DHKV))
    with self.assertRaises(AssertionError):
      conv._replicate_kv_heads(kv)  # pylint: disable=protected-access


class ConvertSyntheticModelTest(unittest.TestCase):
  """Runs the real convert() on the synthetic state in the three sharding modes."""

  @classmethod
  def setUpClass(cls):
    cls.state = _synthetic_state()

  def test_key_coverage_and_dtypes(self):
    out = _make_converter(tp=2).convert(self.state)
    self.assertEqual(set(out), _expected_keys())
    for key, value in out.items():
      expected_dtype = jnp.float32 if key.endswith(".A_log") else jnp.bfloat16
      self.assertEqual(value.dtype, expected_dtype, key)

  def test_expert_name_aliases_match(self):
    out = _make_converter(tp=2).convert(self.state)
    for i in range(NUM_LAYERS):
      p = f"vllm_model.language_model.model.layers.{i}"
      for name in ("w13_weight", "w2_weight"):
        self.assertTrue(jnp.array_equal(out[f"{p}.mlp.experts.routed_experts.{name}"], out[f"{p}.mlp.experts.{name}"]))

  def test_untransposed_kernels_pass_through(self):
    """o_proj / GDN out_proj / shared down_proj are the MaxText kernels unchanged."""
    out = _make_converter(tp=2).convert(self.state)
    blocks = self.state["base"]["decoder"]["scanned_blocks"]
    for rep in range(REPS):
      attn_layer = rep * NUM_SLOTS + 1
      gdn_layer = rep * NUM_SLOTS
      p_attn = f"vllm_model.language_model.model.layers.{attn_layer}"
      p_gdn = f"vllm_model.language_model.model.layers.{gdn_layer}"
      self.assertTrue(
          jnp.array_equal(
              out[f"{p_attn}.self_attn.o_proj.weight"],
              _bf16(blocks["layers_1"]["attention"]["attention"]["out"]["kernel"][:, rep, :]),
          )
      )
      self.assertTrue(
          jnp.array_equal(
              out[f"{p_gdn}.linear_attn.out_proj.weight"],
              _bf16(blocks["layers_0"]["attention"]["out_proj"]["kernel"][:, rep, :]),
          )
      )
      for slot, layer in ((0, gdn_layer), (1, attn_layer)):
        p = f"vllm_model.language_model.model.layers.{layer}"
        self.assertTrue(
            jnp.array_equal(
                out[f"{p}.mlp.shared_expert.down_proj.weight"],
                _bf16(blocks[f"layers_{slot}"]["mlp"]["shared_expert"]["wo"]["kernel"][:, rep, :]),
            )
        )

  def test_attention_dp_uses_unsharded_layout(self):
    """At attn_dp == tp the fused projections are the plain [q | k | v] concat."""
    out = _make_converter(tp=2, attn_dp=2).convert(self.state)
    attn = self.state["base"]["decoder"]["scanned_blocks"]["layers_1"]["attention"]["attention"]
    mlp = self.state["base"]["decoder"]["scanned_blocks"]["layers_1"]["mlp"]
    for rep in range(REPS):
      layer = rep * NUM_SLOTS + 1
      p = f"vllm_model.language_model.model.layers.{layer}"
      q = attn["query"]["kernel"][:, rep].transpose(1, 2, 0).reshape(HQ * DHQ, D)
      k = attn["key"]["kernel"][:, rep].transpose(1, 2, 0).reshape(NKV * DHKV, D)
      v = attn["value"]["kernel"][:, rep].transpose(1, 2, 0).reshape(NKV * DHKV, D)
      expected_qkv = np.concatenate([q, k, v], axis=0).T
      self.assertTrue(jnp.array_equal(out[f"{p}.self_attn.qkv_proj.weight"], _bf16(expected_qkv)))
      gate = mlp["shared_expert"]["wi_0"]["kernel"][:, rep].T  # [FS, D]
      up = mlp["shared_expert"]["wi_1"]["kernel"][:, rep].T
      expected_gate_up = np.concatenate([gate, up], axis=0).T
      self.assertTrue(jnp.array_equal(out[f"{p}.mlp.shared_expert.gate_up_proj.weight"], _bf16(expected_gate_up)))

  def test_kv_heads_replicated_under_plain_tp(self):
    """attn_shards=2 with one KV head: both shards carry the same K (and V) block."""
    out = _make_converter(tp=2).convert(self.state)
    attn = self.state["base"]["decoder"]["scanned_blocks"]["layers_1"]["attention"]["attention"]
    for rep in range(REPS):
      layer = rep * NUM_SLOTS + 1
      qkv = out[f"vllm_model.language_model.model.layers.{layer}.self_attn.qkv_proj.weight"]
      # Per shard: q (4 rows of Hq*dhq/2) | k (4) | v (4), concatenated over 2 shards.
      self.assertEqual(qkv.shape, (D, 2 * (HQ * DHQ // 2 + DHKV + DHKV)))
      k = _bf16(attn["key"]["kernel"][:, rep].transpose(1, 2, 0).reshape(DHKV, D).T)
      self.assertTrue(jnp.array_equal(qkv[:, 4:8], k))
      self.assertTrue(jnp.array_equal(qkv[:, 16:20], k))

  def test_routed_experts_gmm_tp_layout(self):
    out = _make_converter(tp=2).convert(self.state)
    routed = self.state["base"]["decoder"]["scanned_blocks"]["layers_1"]["mlp"]["routed_experts"]
    for rep in range(REPS):
      layer = rep * NUM_SLOTS + 1
      w13 = out[f"vllm_model.language_model.model.layers.{layer}.mlp.experts.routed_experts.w13_weight"]
      # [E, D, tp * 2 * pad128(F / tp)]: per-shard [gate | up] chunks, each padded.
      self.assertEqual(w13.shape, (E, D, 2 * 2 * PAD))
      chunk = F // 2
      for shard in range(2):
        base = shard * 2 * PAD
        gate_chunk = routed["wi_0"][:, rep][:, :, shard * chunk : (shard + 1) * chunk]
        up_chunk = routed["wi_1"][:, rep][:, :, shard * chunk : (shard + 1) * chunk]
        self.assertTrue(jnp.array_equal(w13[:, :, base : base + chunk], _bf16(gate_chunk)))
        self.assertTrue(jnp.array_equal(w13[:, :, base + PAD : base + PAD + chunk], _bf16(up_chunk)))
        self.assertTrue(not w13[:, :, base + chunk : base + PAD].any())  # padding

  def test_routed_experts_gmm_ep_layout(self):
    out = _make_converter(tp=2, use_ep=True).convert(self.state)
    routed = self.state["base"]["decoder"]["scanned_blocks"]["layers_1"]["mlp"]["routed_experts"]
    for rep in range(REPS):
      layer = rep * NUM_SLOTS + 1
      p = f"vllm_model.language_model.model.layers.{layer}"
      w13 = out[f"{p}.mlp.experts.routed_experts.w13_weight"]
      # [E, D, 2 * pad128(F)]: whole [gate | up], no per-TP interleave.
      self.assertEqual(w13.shape, (E, D, 2 * PAD))
      self.assertTrue(jnp.array_equal(w13[:, :, :F], _bf16(routed["wi_0"][:, rep])))
      self.assertTrue(jnp.array_equal(w13[:, :, PAD : PAD + F], _bf16(routed["wi_1"][:, rep])))
      w2 = out[f"{p}.mlp.experts.routed_experts.w2_weight"]
      self.assertTrue(jnp.array_equal(w2, _bf16(routed["wo"][:, rep])))

  def test_gdn_uses_attn_shards(self):
    """GDN fused projections at attn_dp == tp equal the plain [q|k|v|z] / [b|a] concat."""
    out = _make_converter(tp=2, attn_dp=2).convert(self.state)
    gdn = self.state["base"]["decoder"]["scanned_blocks"]["layers_0"]["attention"]
    for rep in range(REPS):
      layer = rep * NUM_SLOTS
      p = f"vllm_model.language_model.model.layers.{layer}"
      t_r = gdn["in_proj_qkvz"]["kernel"][:, rep].T.reshape(HK, 2 * DK + 4 * DV, D)
      v_per_k = HV // HK
      q = t_r[:, :DK, :].reshape(HK * DK, D)
      k = t_r[:, DK : 2 * DK, :].reshape(HK * DK, D)
      v = t_r[:, 2 * DK : 2 * DK + v_per_k * DV, :].reshape(HV * DV, D)
      z = t_r[:, 2 * DK + v_per_k * DV :, :].reshape(HV * DV, D)
      expected = np.concatenate([q, k, v, z], axis=0).T
      self.assertTrue(jnp.array_equal(out[f"{p}.linear_attn.in_proj_qkvz.weight"], _bf16(expected)))
      t_ba = gdn["in_proj_ba"]["kernel"][:, rep].T.reshape(HK, 2 * v_per_k, D)
      b = t_ba[:, :v_per_k, :].reshape(HV, D)
      a = t_ba[:, v_per_k:, :].reshape(HV, D)
      expected_ba = np.concatenate([b, a], axis=0).T
      self.assertTrue(jnp.array_equal(out[f"{p}.linear_attn.in_proj_ba.weight"], _bf16(expected_ba)))


class CreateModelConverterTest(unittest.TestCase):

  def test_standalone_qwen35_gets_sharding_hints(self):
    conv = _create_model_converter(
        "qwen3.5-35b-a3b",
        config=_make_config(tp=8),
        mesh=None,
        use_standalone_converter=True,
        sharding_hints={"attn_dp_size": 4, "enable_expert_parallel": True},
    )
    self.assertIsInstance(conv, Qwen35MaxTextToVLLMConverter)
    self.assertEqual(conv.vllm_attn_dp, 4)
    self.assertEqual(conv.attn_shards, 2)
    self.assertTrue(conv.vllm_use_ep)

  def test_standalone_qwen35_defaults_without_hints(self):
    conv = _create_model_converter("qwen3.5-35b-a3b", config=_make_config(tp=2), mesh=None, use_standalone_converter=True)
    self.assertEqual(conv.vllm_attn_dp, 1)
    self.assertFalse(conv.vllm_use_ep)

  def test_standalone_gemma4(self):
    config = SimpleNamespace(
        base_num_decoder_layers=6, rollout_tensor_parallelism=1, model_name="gemma4-4b", base_emb_dim=D
    )
    conv = _create_model_converter("gemma4-4b", config=config, mesh=None, use_standalone_converter=True)
    self.assertIsInstance(conv, Gemma4MaxTextToVLLMConverter)

  def test_standalone_unknown_model_rejected(self):
    with self.assertRaises(NotImplementedError):
      _create_model_converter("llama3.1-8b", config=_make_config(tp=1), mesh=None, use_standalone_converter=True)


class _DummyConverter(BaseMaxTextToVLLMConverter):
  """Minimal standalone converter emitting a canned dict."""

  def __init__(self, out):
    super().__init__(_make_config(tp=1), mesh=None)
    self._out = out

  def convert(self, model_state, **kwargs):
    return dict(self._out)

  def _convert_global(self, params):
    pass

  def _convert_attn(self, params):
    pass

  def _convert_moe(self, params):
    pass


class _FakeRunnerSampler(MaxTextVllmSampler):
  """Shadows the base class's read-only `_model_runner` property so tests can set it."""

  _model_runner = None


def _make_sampler(state, converter, chunk=None, delete_dst=False, llm=None):
  sampler = object.__new__(_FakeRunnerSampler)
  sampler._converter = converter  # pylint: disable=protected-access
  sampler.converter = converter
  sampler._model_runner = SimpleNamespace(state=state, state_leaves=None)  # pylint: disable=protected-access
  sampler.llm = llm
  sampler._driver = None  # pylint: disable=protected-access
  sampler.config = SimpleNamespace(reshard_chunk_size=chunk, delete_dst_buffers=delete_dst)
  return sampler


def _identity_reshard(src, shardings):
  del shardings
  return src


class StandaloneSyncTest(unittest.TestCase):

  def test_update_params_routes_standalone_converters(self):
    sampler = _make_sampler({}, _DummyConverter({}))
    sampler._sync_standalone_converted = mock.Mock(return_value="synced")  # pylint: disable=protected-access
    self.assertEqual(sampler.update_params({"w": 1}), "synced")
    sampler._sync_standalone_converted.assert_called_once_with({"w": 1})  # pylint: disable=protected-access

  def test_update_params_reraises_sync_failures(self):
    sampler = _make_sampler({}, _DummyConverter({}))
    sampler._sync_standalone_converted = mock.Mock(side_effect=RuntimeError("boom"))  # pylint: disable=protected-access
    with self.assertRaises(RuntimeError):
      sampler.update_params({})

  def test_sync_requires_flat_dict_state(self):
    sampler = _make_sampler(object(), _DummyConverter({}))
    with self.assertRaisesRegex(TypeError, "flat dict"):
      sampler._sync_standalone_converted({})  # pylint: disable=protected-access

  def test_sync_updates_covered_tensors_in_place(self):
    state = {
        "layers.0.qkv": jnp.zeros((2, 3), jnp.bfloat16),
        "layers.0._private": jnp.zeros((1,), jnp.bfloat16),
        "layers.0.rotary_emb.cache": jnp.zeros((1,), jnp.bfloat16),
        "layers.0.uncovered": jnp.zeros((1,), jnp.bfloat16),
    }
    new_qkv = jnp.ones((2, 3), jnp.bfloat16)
    converter = _DummyConverter({"layers.0.qkv": new_qkv, "layers.0.alias_only": jnp.ones((4,), jnp.bfloat16)})
    llm = mock.Mock()
    sampler = _make_sampler(state, converter, llm=llm)
    with mock.patch.object(rollout_mod.tunix_reshard, "reshard_pytree", _identity_reshard):
      sampler._sync_standalone_converted({})  # pylint: disable=protected-access
    self.assertTrue(jnp.array_equal(state["layers.0.qkv"], new_qkv))
    self.assertNotIn("layers.0.alias_only", state)  # version alias without a runner target is dropped
    self.assertTrue(not state["layers.0.uncovered"].any())  # left untouched (and warned about)
    self.assertIs(sampler._model_runner.state_leaves, state)  # pylint: disable=protected-access
    llm.reset_prefix_cache.assert_called_once_with()
    llm.collective_rpc.assert_has_calls([mock.call("delete_kv_cache"), mock.call("reinitialize_kv_cache")])

  def test_sync_uses_chunked_reshard_when_configured(self):
    state = {"w": jnp.zeros((2,), jnp.bfloat16)}
    converter = _DummyConverter({"w": jnp.ones((2,), jnp.bfloat16)})
    sampler = _make_sampler(state, converter, chunk=2, delete_dst=True)
    chunked = mock.Mock(side_effect=lambda src_flat, **kw: src_flat)
    with mock.patch.object(rollout_mod.tunix_gen_utils, "_reshard_in_chunks", chunked, create=True):
      sampler._sync_standalone_converted({})  # pylint: disable=protected-access
    self.assertEqual(chunked.call_args.kwargs["chunk_size"], 2)
    self.assertTrue(state["w"].all())

  def test_sync_falls_back_when_tunix_lacks_chunked_reshard(self):
    state = {"w": jnp.zeros((2,), jnp.bfloat16)}
    converter = _DummyConverter({"w": jnp.ones((2,), jnp.bfloat16)})
    sampler = _make_sampler(state, converter, chunk=2, delete_dst=False)
    with mock.patch.object(rollout_mod.tunix_gen_utils, "_reshard_in_chunks", None, create=True):
      with mock.patch.object(rollout_mod.tunix_reshard, "reshard_pytree", _identity_reshard):
        sampler._sync_standalone_converted({})  # pylint: disable=protected-access
    self.assertTrue(state["w"].all())

  def test_sync_rejects_layout_drift(self):
    state = {"w": jnp.zeros((2, 2), jnp.bfloat16)}
    converter = _DummyConverter({"w": jnp.ones((3, 3), jnp.bfloat16)})
    sampler = _make_sampler(state, converter)
    with mock.patch.object(rollout_mod.tunix_reshard, "reshard_pytree", _identity_reshard):
      with self.assertRaisesRegex(ValueError, "out of date"):
        sampler._sync_standalone_converted({})  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
