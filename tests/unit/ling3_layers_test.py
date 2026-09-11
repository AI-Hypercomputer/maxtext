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

"""Ling3 weight-copy parity against the publisher's text decoder.

Set LING3_REFERENCE_DIR to a directory containing modeling_bailing_moe_v3.py
and configuration_bailing_moe_v3.py from inclusionAI/Ling-3.0-flash revision
e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3. These reference tests use the VL
backbone dimensions with CPU fallbacks for the publisher's CUDA-only kernels.
Tests use small dimensions and preserve grouped routing and hybrid attention.
"""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import torch

from maxtext.checkpoint_conversion.utils.param_mapping import (
    LING3_MAXTEXT_TO_HF_PARAM_MAPPING,
    LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
)
from maxtext.models.ling3 import Ling3DecoderLayer
from maxtext.models.ling3 import Ling3ScannableBlock
from maxtext.models.ling3 import Ling3MLAAttention
from maxtext.layers.attention_op import AttentionOp
from maxtext.common.common_types import DecoderBlockType
from maxtext.layers.nnx_decoders import NNXDecoder
from tests.assets.ling3.hf_reference import load_reference, position_embeddings

REFERENCE = os.environ.get("LING3_REFERENCE_DIR")


class Ling3DecoderRegistryTest(unittest.TestCase):
  """Exercise the production registry, not just direct layer construction."""

  def test_ling3_and_existing_decoder_registration(self):
    for scan_layers in (False, True):
      config = SimpleNamespace(decoder_block=DecoderBlockType.LING3, scan_layers=scan_layers)
      decoder = SimpleNamespace(config=config)
      expected = Ling3ScannableBlock if scan_layers else Ling3DecoderLayer
      self.assertEqual(NNXDecoder.get_decoder_layers(decoder), [expected])
      # The registry evaluates every entry even when another model is selected.
      # A dangling Ling3 import must not break existing decoder families either.
      for block in (DecoderBlockType.GPT3, DecoderBlockType.LLAMA2):
        config.decoder_block = block
        self.assertTrue(NNXDecoder.get_decoder_layers(decoder))


def small_config():
  return SimpleNamespace(
      emb_dim=32,
      mlp_dim=64,
      moe_mlp_dim=16,
      num_query_heads=4,
      num_kv_heads=4,
      head_dim=8,
      num_experts=8,
      num_experts_per_tok=2,
      n_routing_groups=2,
      topk_routing_group=1,
      routed_scaling_factor=2.5,
      first_num_dense_layers=2,
      num_decoder_layers=6,
      inhomogeneous_layer_cycle_interval=6,
      normalization_layer_epsilon=1e-6,
      dense_init_scale=1.0,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      matmul_precision=jax.lax.Precision.HIGHEST,
      ici_context_autoregressive_parallelism=0,
      shard_mode="auto",
      qk_nope_head_dim=8,
      qk_rope_head_dim=8,
      v_head_dim=8,
      kv_lora_rank=16,
      rope_max_timescale=6000000,
      max_target_length=32,
      max_prefill_predict_length=8,
      micro_batch_size_to_train_on=1,
  )


def hf_config(reference, cfg):
  return reference.BailingMoeV3Config(
      hidden_size=cfg.emb_dim,
      intermediate_size=cfg.mlp_dim,
      num_hidden_layers=6,
      num_attention_heads=cfg.num_query_heads,
      num_key_value_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      n_group=cfg.n_routing_groups,
      topk_group=cfg.topk_routing_group,
      moe_intermediate_size=cfg.moe_mlp_dim,
      moe_shared_expert_intermediate_size=cfg.moe_mlp_dim,
      first_k_dense_replace=2,
      layer_group_size=6,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      v_head_dim=cfg.v_head_dim,
      no_kda_lora=True,
      kda_safe_gate=True,
      kda_lower_bound=-5.0,
      gated_attention_proj_granularity_type="head_wise",
      routed_scaling_factor=2.5,
      rope_theta=6000000,
      attn_implementation="eager",
  )


def copy_layer(layer, state, cfg, hfc, index):
  mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hfc.to_dict(), cfg)
  hooks = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hfc.to_dict(), cfg)
  flat = flatten_dict(nnx.to_pure_dict(nnx.state(layer, nnx.Param)))
  out = {}

  def read(key):
    return state[key.removeprefix(f"model.layers.{index}.")].detach().float().numpy()

  for path, target in flat.items():
    key = f"params-decoder-layers_{index}-" + "-".join(path)
    source = mapping[key]
    hook = hooks.get(key, lambda x, shape: x.reshape(shape))
    out[path] = jnp.asarray(
        np.stack([hook(read(k), target.shape[1:]) for k in source])
        if isinstance(source, list)
        else hook(read(source), target.shape)
    )
  nnx.update(layer, unflatten_dict(out))


@unittest.skipUnless(REFERENCE, "Set LING3_REFERENCE_DIR to the pinned upstream source directory.")
class Ling3LayerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)
    cls.ref = load_reference(
        Path(REFERENCE) / "modeling_bailing_moe_v3.py", Path(REFERENCE) / "configuration_bailing_moe_v3.py"
    )

  def test_weight_copy_and_cached_decode(self):
    cfg = small_config()
    hfc = hf_config(self.ref, cfg)
    for index in (0, 2, 5):
      with self.subTest(layer=index):
        torch.manual_seed(index)
        pt = self.ref.BailingMoeV3DecoderLayer(hfc, index).eval()
        # Upstream leaves dt_bias uninitialized; seed nontrivial, finite gate values.
        if hasattr(pt.attention, "dt_bias"):
          torch.nn.init.uniform_(pt.attention.dt_bias, -1, 1)
        if index >= 2:
          torch.nn.init.uniform_(pt.mlp.gate.expert_bias, -0.1, 0.1)
        mt = Ling3DecoderLayer(cfg, None, "train", index, rngs=nnx.Rngs(index))
        copy_layer(mt, pt.state_dict(), cfg, hfc, index)
        x = torch.randn(1, 7, cfg.emb_dim)
        offset = 1024 if index == 5 else 0
        positions = (torch.arange(7) + offset)[None]
        mask = torch.triu(torch.full((1, 1, 7, 7), -1e30), diagonal=1)
        with torch.no_grad():
          expected = pt(
              x,
              attention_mask=mask if index == 5 else None,
              position_ids=positions,
              position_embeddings=position_embeddings(hfc, positions),
          )[0].numpy()
        actual = mt(jnp.asarray(x.numpy()), decoder_positions=jnp.asarray(positions.numpy()), model_mode="train")[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        cached, cache = mt(
            jnp.asarray(x[:, :3].numpy()),
            decoder_positions=(jnp.arange(3) + offset)[None],
            kv_cache={},
            model_mode="train",
        )
        parts = [cached]
        for pos in range(3, 7):
          output, cache = mt(
              jnp.asarray(x[:, pos : pos + 1].numpy()),
              decoder_positions=jnp.array([[pos + offset]]),
              kv_cache=cache,
              model_mode="train",
          )
          parts.append(output)
        np.testing.assert_allclose(jnp.concatenate(parts, 1), expected, atol=2e-5, rtol=2e-4)

  def test_padding_and_packed_segment_reset(self):
    cfg = small_config()
    layer = Ling3DecoderLayer(cfg, None, "prefill", 0, rngs=nnx.Rngs(9))
    x = jax.random.normal(jax.random.key(1), (1, 5, cfg.emb_dim))
    first = layer(x[:, :3], decoder_positions=jnp.arange(3)[None], model_mode="train")[0]
    second = layer(x[:, 3:], decoder_positions=jnp.arange(2)[None], model_mode="train")[0]
    packed = layer(
        x,
        decoder_positions=jnp.array([[0, 1, 2, 0, 1]]),
        decoder_segment_ids=jnp.array([[1, 1, 1, 2, 2]]),
        model_mode="train",
    )[0]
    np.testing.assert_allclose(packed, jnp.concatenate((first, second), 1), atol=1e-5)
    padded = jnp.pad(x[:, :3], ((0, 0), (0, 2), (0, 0)))
    layer(
        padded,
        decoder_positions=jnp.arange(5)[None],
        decoder_segment_ids=jnp.array([[1, 1, 1, 0, 0]]),
        model_mode="prefill",
    )
    decode = layer(x[:, 3:4], decoder_positions=jnp.array([[3]]), model_mode="autoregressive")[0]
    expected = layer(x[:, :4], decoder_positions=jnp.arange(4)[None], model_mode="train")[0][:, -1:]
    np.testing.assert_allclose(decode, expected, atol=1e-5)


class Ling3MappingTest(unittest.TestCase):

  def test_mla_shared_attention_and_fixed_cache(self):
    cfg = small_config()
    layer = Ling3MLAAttention(cfg, None, "train", 5, rngs=nnx.Rngs(5))
    self.assertIsInstance(layer.attention_op, AttentionOp)
    paths = set(flatten_dict(nnx.to_pure_dict(nnx.state(layer, nnx.Param))))
    self.assertEqual(paths, {
        ("q_proj", "kernel"), ("kv_a_proj_with_mqa", "kernel"),
        ("kv_a_layernorm", "scale"), ("kv_b_proj", "kernel"),
        ("g_proj", "kernel"), ("dense", "kernel"),
    })
    x = jax.random.normal(jax.random.key(2), (1, 6, cfg.emb_dim))
    positions = jnp.array([[0, 1, 2, 0, 1, 2]])
    segments = jnp.array([[1, 1, 1, 2, 2, 0]])
    expected, _ = layer(x, decoder_positions=positions, decoder_segment_ids=segments)
    independent, _ = layer(x[:, 3:5], decoder_positions=positions[:, 3:5])
    np.testing.assert_allclose(expected[:, 3:5], independent, atol=2e-5, rtol=2e-4)
    # Fixed slots use absolute storage positions, with a nonzero RoPE offset in
    # the separate growing-cache parity test. Compare valid tokens only: padded
    # queries have no valid keys, so their output is deliberately unspecified.
    absolute = jnp.arange(6)[None]
    expected, _ = layer(x, decoder_positions=absolute, decoder_segment_ids=segments)
    cache = {
        "key": jnp.zeros((1, 8, cfg.num_query_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)),
        "value": jnp.zeros((1, 8, cfg.num_query_heads, cfg.v_head_dim)),
        "positions": jnp.zeros((1, 8), jnp.int32),
        "segments": jnp.zeros((1, 8), jnp.int32),
    }
    outputs = []
    for start, end in ((0, 3), (3, 4), (4, 5)):
      out, cache = layer(
          x[:, start:end],
          decoder_positions=absolute[:, start:end],
          decoder_segment_ids=segments[:, start:end],
          kv_cache=dict(cache, fixed=True),
          model_mode="prefill" if start == 0 else "autoregressive",
      )
      outputs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outputs, axis=1), expected[:, :5], atol=2e-5, rtol=2e-4)

  def test_linear_and_convolution_round_trip(self):
    cfg = small_config()
    hf = {"layer_group_size": 6, "first_k_dense_replace": 2}
    forward = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf, cfg)
    reverse = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf, cfg, saving_to_hf=True)
    for key, shape, mt_shape in (
        ("params-decoder-layers_0-attention-q_proj-kernel", (32, 32), (32, 32)),
        ("params-decoder-layers_5-attention-kv_a_proj_with_mqa-kernel", (24, 32), (32, 24)),
        ("params-decoder-layers_0-attention-q_conv1d-conv-kernel", (32, 1, 4), (4, 1, 32)),
        ("params-decoder-layers_0-attention-A_log", (1, 1, 4, 1), (4,)),
    ):
      x = np.arange(np.prod(shape)).reshape(shape)
      converted = forward[key](x, mt_shape)
      np.testing.assert_array_equal(reverse[key](converted, shape), x)
      if key.endswith("-kernel") and "-conv-" not in key:
        np.testing.assert_array_equal(converted, x.T)

  def test_scanned_expert_and_cycle_indices(self):
    cfg = small_config()
    hf = {"layer_group_size": 6, "first_k_dense_replace": 2}
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf, cfg, scan_layers=True)
    self.assertEqual(
        mapping["params-decoder-layers-layer_5-mlp-wi_0"][3], ["model.layers.5.mlp.experts.3.gate_proj.weight"]
    )
    self.assertEqual(
        mapping["params-decoder-layers-layer_0-attention-q_proj-kernel"], ["model.layers.0.attention.q_proj.weight"]
    )
    self.assertNotIn("params-decoder-layers-layer_5-attention-q_conv1d-conv-kernel", mapping)
    cfg.num_decoder_layers = 42
    with self.assertRaisesRegex(ValueError, "six-layer cycle"):
      LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf, cfg, scan_layers=True)

  def test_late_layer_swiglu_limits(self):
    from maxtext.models.ling3 import Ling3MoE

    cfg = small_config()
    x = jax.random.normal(jax.random.key(2), (1, 3, cfg.emb_dim)) * 10
    for index in (34, 35, 40):
      layer = Ling3MoE(cfg, None, "train", index, rngs=nnx.Rngs(3))
      ids = jnp.broadcast_to(jnp.array([0, 1]), (1, 3, 2))
      actual = layer(x, forced_routed_experts=ids)
      state = nnx.to_pure_dict(nnx.state(layer, nnx.Param))

      def tensor(value):
        return torch.from_numpy(np.array(value))

      tx = tensor(x)
      scores = torch.sigmoid(tx @ tensor(state["gate"]["kernel"]))[..., :2]
      scores = scores / scores.sum(-1, keepdim=True) * cfg.routed_scaling_factor

      def mlp(w0, w1, wo, bound):
        gate = torch.nn.functional.silu(tx @ tensor(w0))
        up = tx @ tensor(w1)
        if bound:
          gate = gate.clamp(max=bound)
          up = up.clamp(-bound, bound)
        return (gate * up) @ tensor(wo)

      limit = 4 if index >= 35 else None
      expected = sum(
          mlp(state["wi_0"][e], state["wi_1"][e], state["wo"][e], limit) * scores[..., e : e + 1] for e in range(2)
      )
      shared_limit = (7 if index >= 40 else 5) if index >= 35 else None
      expected += mlp(
          state["shared_wi_0"]["kernel"], state["shared_wi_1"]["kernel"], state["shared_wo"]["kernel"], shared_limit
      )
      np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-4)


if __name__ == "__main__":
  unittest.main()
