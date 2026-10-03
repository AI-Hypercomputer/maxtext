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

"""Small KDA recurrence checks; these do not establish HF model parity."""

# Pytest injects fixtures; tests inspect state and call Torch extension functions.
# pylint: disable=redefined-outer-name,protected-access,not-callable

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import jax
from flax import nnx
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.configs.types import DType
from maxtext.models.kimi_k3 import KimiDecoderLayer, KimiSparseMoeBlock, KimiDeltaAttention, KimiMLP
from maxtext.common.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE

from maxtext.models.kimi_k3 import jax_kda_chunk_rule, jax_kda_recurrent_step
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.moe import RoutedMoE
from torch.nn import functional
from flax.linen import logical_axis_rules
from maxtext.layers.linears import DenseGeneral
from jax.sharding import NamedSharding, PartitionSpec
from maxtext.models.kimi_k3 import round_bfloat16_logits
from maxtext.models.kimi_k3 import KimiGateProjection


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_low_rank_gate_projection_matches_torch(dtype):

  layer = KimiGateProjection(
      in_features_shape=64, out_features_shape=8, dtype=dtype, weight_dtype=dtype, rngs=nnx.Rngs(19)
  )
  random = np.random.default_rng(123)
  inputs = jnp.asarray(random.normal(size=(2, 3, 64)), dtype=dtype)
  weights = jnp.asarray(random.normal(size=(64, 8)), dtype=dtype)
  layer.kernel.set_value(weights)
  torch_dtype = torch.bfloat16 if dtype == jnp.bfloat16 else torch.float32
  expected = (
      torch.nn.functional.linear(
          torch.tensor(np.asarray(inputs, dtype=np.float32), dtype=torch_dtype),
          torch.tensor(np.asarray(weights, dtype=np.float32).T, dtype=torch_dtype),
      )
      .float()
      .numpy()
  )
  actual = jax.jit(layer)(inputs)
  np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), expected, rtol=1e-5, atol=1e-5)


def test_bfloat16_logits_preserve_hf_greedy_tie():
  # Extra accumulator bits would incorrectly rank the second vocabulary ID.
  logits = np.array([7.998, 8.010, 7.875], dtype=np.float32)
  expected = torch.tensor(logits).to(torch.bfloat16).float().numpy()
  actual = jax.jit(round_bfloat16_logits)(jnp.asarray(logits))
  np.testing.assert_array_equal(actual, expected)
  assert int(jnp.argmax(actual)) == 0


def test_float32_accumulation_preserves_multi_axis_bfloat16_projection():

  layer = DenseGeneral(
      in_features_shape=(4, 8),
      out_features_shape=6,
      axis=(-2, -1),
      dtype=jnp.bfloat16,
      weight_dtype=jnp.bfloat16,
      accumulate_in_float32=True,
      rngs=nnx.Rngs(27),
  )
  inputs = jax.random.normal(jax.random.key(28), (2, 3, 4, 8)).astype(jnp.bfloat16)
  expected = (
      torch.nn.functional.linear(
          torch.tensor(np.asarray(inputs, dtype=np.float32), dtype=torch.bfloat16).flatten(-2),
          torch.tensor(np.asarray(layer.kernel.get_value(), dtype=np.float32), dtype=torch.bfloat16).reshape(32, 6).T,
      )
      .float()
      .numpy()
  )
  actual = jax.jit(layer)(inputs)
  assert actual.dtype == jnp.bfloat16
  np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)


@pytest.mark.skipif(jax.device_count() < 4, reason="requires four devices for a split contraction")
def test_bfloat16_projection_accumulates_across_tensor_parallel_shards():

  mesh = Mesh(np.asarray(jax.devices()[:4]), ("tensor",))
  input_sharding = NamedSharding(mesh, PartitionSpec(None, None, "tensor"))
  weight_sharding = NamedSharding(mesh, PartitionSpec("tensor", None))
  output_sharding = NamedSharding(mesh, PartitionSpec())
  random = np.random.default_rng(91)
  inputs = jnp.asarray(random.normal(size=(2, 3, 1024)), dtype=jnp.bfloat16)
  weights = jnp.asarray(random.normal(size=(1024, 16)), dtype=jnp.bfloat16)

  def forward(x, weight):
    layer = DenseGeneral(
        in_features_shape=1024,
        out_features_shape=16,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        accumulate_in_float32=True,
        rngs=nnx.Rngs(29),
    )
    layer.kernel.set_value(weight)
    return layer(x)

  actual = jax.jit(forward, in_shardings=(input_sharding, weight_sharding), out_shardings=output_sharding)(
      inputs, weights
  )
  expected = (
      torch.nn.functional.linear(
          torch.tensor(np.asarray(inputs, dtype=np.float32), dtype=torch.bfloat16),
          torch.tensor(np.asarray(weights, dtype=np.float32).T, dtype=torch.bfloat16),
      )
      .float()
      .numpy()
  )
  np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)


def test_mla_bfloat16_combined_cache_matches_torch(tiny_config):

  config = pyconfig.HyperParameters(
      tiny_config._pydantic_config.model_copy(
          update={"dtype": DType.BFLOAT16, "weight_dtype": DType.BFLOAT16, "mla_naive_kvcache": True}
      )
  )
  mesh = tiny_mesh(config)
  with mesh, logical_axis_rules(config.logical_axis_rules):
    layer = KimiDecoderLayer(config, mesh, MODEL_MODE_AUTOREGRESSIVE, 3, rngs=nnx.Rngs(10)).self_attention
    query = jax.random.normal(jax.random.key(16), (1, 1, 4, 6)).astype(jnp.bfloat16)
    key = jax.random.normal(jax.random.key(17), (1, 8, 4, 6)).astype(jnp.bfloat16)
    value = jax.random.normal(jax.random.key(18), (1, 8, 4, 4)).astype(jnp.bfloat16)
    segments = jnp.array([[1, 1, 1, 0, 1, 1, 1, 0]], dtype=jnp.int32)

    def torch_tensor(array):
      return torch.tensor(np.asarray(array, dtype=np.float32), dtype=torch.bfloat16)

    scores = torch.einsum("bthd,bshd->bhts", torch_tensor(query), torch_tensor(key)) / (6**0.5)
    scores = scores.masked_fill(~torch.tensor(np.asarray(segments, dtype=bool))[:, None, None, :], float("-inf"))
    probabilities = scores.float().softmax(-1).to(torch.bfloat16)
    expected = torch.einsum("bhts,bshd->bthd", probabilities, torch_tensor(value)).float().numpy()
    cached = [(key[:, :4], value[:, :4], segments[:, :4]), (key[:, 4:], value[:, 4:], segments[:, 4:], jnp.array([3]))]
    actual = layer.attention_op(
        query,
        key[:, -1:],
        value[:, -1:],
        jnp.ones((1, 1), jnp.int32),
        jnp.array([[5]]),
        MODEL_MODE_AUTOREGRESSIVE,
        cached,
    )
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("nonzero_initial_state", [False, True])
def test_kda_recurrence_against_torch(nonzero_initial_state):
  """Check chunk and incremental evaluation using independent matrix operations."""
  rng = np.random.default_rng(42)
  batch, length, heads, key_dim, value_dim = 2, 5, 3, 4, 6
  shape = (batch, length, heads, key_dim)
  query, key = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
  value = rng.normal(size=(batch, length, heads, value_dim)).astype(np.float32)
  gate = -np.abs(rng.normal(size=shape).astype(np.float32))
  beta = rng.uniform(size=(batch, length, heads)).astype(np.float32)
  initial = rng.normal(size=(batch, heads, key_dim, value_dim)).astype(np.float32)
  if not nonzero_initial_state:
    initial.fill(0)

  state = torch.tensor(initial)
  reference = []
  for index in range(length):
    q = torch.tensor(query[:, index]) / key_dim**0.5
    k = torch.tensor(key[:, index])
    v = torch.tensor(value[:, index])
    state = state * torch.exp(torch.tensor(gate[:, index]))[..., None]
    prediction = torch.einsum("bhk,bhkv->bhv", k, state)
    state = state + torch.tensor(beta[:, index])[..., None, None] * torch.einsum("bhk,bhv->bhkv", k, v - prediction)
    reference.append(torch.einsum("bhk,bhkv->bhv", q, state))
  reference = torch.stack(reference, dim=1).numpy()

  output, final_state = jax_kda_chunk_rule(
      *map(jnp.asarray, (query, key, value, gate, beta)),
      initial_state=jnp.asarray(initial) if nonzero_initial_state else None,
  )
  np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(final_state, state.numpy(), rtol=1e-5, atol=1e-5)

  incremental_state = jnp.asarray(initial)
  incremental_outputs = []
  for index in range(length):
    output, incremental_state = jax_kda_recurrent_step(
        *map(jnp.asarray, (query[:, index], key[:, index], value[:, index], gate[:, index], beta[:, index])),
        incremental_state,
    )
    incremental_outputs.append(output)
  np.testing.assert_allclose(jnp.stack(incremental_outputs, axis=1), reference, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(incremental_state, state.numpy(), rtol=1e-5, atol=1e-5)


@pytest.fixture
def tiny_config():
  return pyconfig.initialize(
      [
          "test",
          "src/maxtext/configs/base.yml",
          "model_name=kimi-k3",
          "override_model_config=true",
          "hardware=cpu",
          "base_num_decoder_layers=4",
          "scan_layers=false",
          "base_emb_dim=16",
          "base_num_query_heads=4",
          "base_num_kv_heads=4",
          "base_mlp_dim=32",
          "base_moe_mlp_dim=8",
          "num_experts=4",
          "num_experts_per_tok=2",
          "routed_expert_hidden_size=8",
          "vocab_size=64",
          "kda_num_heads=4",
          "kda_head_dim=4",
          "head_dim=6",
          "q_lora_rank=8",
          "kv_lora_rank=4",
          "qk_nope_head_dim=4",
          "qk_rope_head_dim=2",
          "v_head_dim=4",
          "attention=dot_product",
          "per_device_batch_size=1",
          "max_prefill_predict_length=4",
          "max_target_length=20",
          "dtype=float32",
          "weight_dtype=float32",
          "skip_jax_distributed_system=true",
          "run_name=kimi_layer_test",
          "base_output_directory=/tmp/kimi_layer_test",
          "mla_naive_kvcache=false",
      ]
  )


def tiny_mesh(config):
  return Mesh(np.asarray(jax.devices()).reshape((1,) * len(config.mesh_axes)), config.mesh_axes)


def test_residual_block_boundary_removes_old_prefix(tiny_config):
  layer = KimiDecoderLayer(tiny_config, tiny_mesh(tiny_config), MODEL_MODE_TRAIN, 0, rngs=nnx.Rngs(7))
  x = jax.random.normal(jax.random.key(3), (1, 3, 16))
  attention, _ = layer.self_attention(layer.input_layernorm(x), model_mode=MODEL_MODE_TRAIN)
  residual = x.reshape(3, 1, 16)
  mlp_input = layer._apply_attn_res(attention.reshape(3, 16), residual, layer.mlp_res_proj, layer.mlp_res_norm)
  expected = attention + layer.mlp(layer.post_attention_layernorm(mlp_input.reshape(1, 3, 16)))
  actual, _, _ = layer(x, model_mode=MODEL_MODE_TRAIN, block_residual=jnp.zeros((3, 0, 16)))
  np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_latent_moe_norm_is_after_experts(tiny_config):
  layer = KimiSparseMoeBlock(tiny_config, tiny_mesh(tiny_config), rngs=nnx.Rngs(8))
  x = jax.random.normal(jax.random.key(4), (1, 3, 16))
  latent = layer.routed_expert_down_proj(x)
  routed, _, _ = layer.MoeBlock_0(latent, gate_inputs=x)
  expected = layer.routed_expert_up_proj(layer.routed_expert_norm(routed)) + layer.shared_experts(x)
  np.testing.assert_allclose(layer(x), expected, atol=1e-5, rtol=1e-5)


def test_kda_internal_cache_roundtrip_ignores_prefill_padding(tiny_config):
  layer = KimiDeltaAttention(tiny_config, tiny_mesh(tiny_config), MODEL_MODE_PREFILL, rngs=nnx.Rngs(9))
  x = jax.random.normal(jax.random.key(5), (1, 3, 16))
  expected, _ = layer(x, model_mode=MODEL_MODE_TRAIN)
  padded = jnp.concatenate([x[:, :2], jnp.full((1, 2, 16), 10.0)], axis=1)
  layer(padded, model_mode=MODEL_MODE_PREFILL, decoder_segment_ids=jnp.array([[1, 1, 0, 0]]))
  # MaxEngine reconstructs modules between prefill and autoregressive calls.
  graph, params, cache, rest = nnx.split(layer, nnx.Param, nnx.Cache, ...)
  assert jax.tree.leaves(cache), "KDA must persist state in nnx.Cache for MaxEngine"
  restored = nnx.merge(graph, params, cache, rest)
  actual, _ = restored(x[:, 2:], model_mode=MODEL_MODE_AUTOREGRESSIVE)
  np.testing.assert_allclose(actual, expected[:, 2:], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_mla_matches_torch_with_output_gate_and_unrotated_positions(tiny_config, dtype):

  tiny_config = pyconfig.HyperParameters(
      tiny_config._pydantic_config.model_copy(update={"dtype": DType(dtype), "weight_dtype": DType(dtype)})
  )
  mesh = tiny_mesh(tiny_config)
  with mesh, logical_axis_rules(tiny_config.logical_axis_rules):
    layer = KimiDecoderLayer(tiny_config, mesh, MODEL_MODE_TRAIN, 3, rngs=nnx.Rngs(10)).self_attention
    assert hasattr(layer, "g_proj"), "Kimi MLA needs the HF output gate projection"
    layer.q_norm.scale[...] = (1 + 0.2 * jax.random.normal(jax.random.key(24), (8,))).astype(getattr(jnp, dtype))
    layer.kv_norm.scale[...] = (1 + 0.2 * jax.random.normal(jax.random.key(25), (4,))).astype(getattr(jnp, dtype))
    x = jax.random.normal(jax.random.key(6), (1, 3, 16)).astype(getattr(jnp, dtype))
    tx = torch.tensor(np.asarray(x, dtype=np.float32), dtype=getattr(torch, dtype))

    def project(value, module):
      kernel = torch.tensor(np.asarray(module.kernel[...], dtype=np.float32), dtype=tx.dtype)
      return torch.einsum("bti,i...->bt...", value, kernel)

    def norm(value, module):
      weight = torch.tensor(np.asarray(module.scale[...], dtype=np.float32))
      normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
      return normalized.to(value.dtype) * weight.to(value.dtype)

    query = project(norm(project(tx, layer.wq_a), layer.q_norm), layer.wq_b)
    kv = project(tx, layer.wkv_a)
    c = norm(kv[..., :4], layer.kv_norm)
    expanded = project(c, layer.wkv_b)
    key = torch.cat([expanded[..., :4], kv[..., 4:].unsqueeze(2).expand(-1, -1, 4, -1)], dim=-1)
    value = expanded[..., 4:]
    scores = torch.einsum("bthd,bshd->bhts", query, key) / (6**0.5)
    scores = scores.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1), float("-inf"))
    out = torch.einsum("bhts,bshd->bthd", scores.float().softmax(-1).to(tx.dtype), value)
    out = out * project(tx, layer.g_proj).sigmoid()
    expected = torch.einsum(
        "bthd,hde->bte", out, torch.tensor(np.asarray(layer.out.kernel[...], dtype=np.float32), dtype=tx.dtype)
    )
    actual, _ = layer(x, x, jnp.array([[0, 1, 2]]), jnp.ones((1, 3), dtype=jnp.int32), model_mode=MODEL_MODE_TRAIN)
    tolerance = 6e-3 if dtype == "bfloat16" else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), expected.float().numpy(), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_kda_full_attention_matches_torch(tiny_config, dtype):
  """Use copied weights, causal Torch convolutions, and fp32 gate/state math."""

  cfg = tiny_config._pydantic_config.model_copy(update={"dtype": DType(dtype), "weight_dtype": DType(dtype)})
  config = pyconfig.HyperParameters(cfg)
  layer = KimiDeltaAttention(config, tiny_mesh(config), MODEL_MODE_TRAIN, rngs=nnx.Rngs(11))
  tdtype = getattr(torch, dtype)

  def tensor(value, target_dtype=tdtype):
    return torch.tensor(np.asarray(value, dtype=np.float32), dtype=target_dtype)

  def project(value, module):
    return value @ tensor(module.kernel[...])

  x = jax.random.normal(jax.random.key(12), (1, 5, 16)).astype(config.dtype)
  tx = tensor(x)
  projections = []
  for name in ("q", "k", "v"):
    raw = project(tx, getattr(layer, name + "_proj"))
    kernel = tensor(getattr(layer, name + "_conv1d").kernel[...]).permute(2, 1, 0)
    conv = functional.conv1d(raw.float().transpose(1, 2), kernel.float(), padding=3, groups=16)[..., :5].transpose(1, 2)
    projections.append(functional.silu(conv).to(tdtype).reshape(1, 5, 4, 4))
  q, k, v = projections  # pylint: disable=unbalanced-tuple-unpacking
  q, k = [value.float() / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-6) for value in (q, k)]
  q = q / 2
  gate = project(project(tx, layer.f_a_proj), layer.f_b_proj).reshape(1, 5, 4, 4).float()
  gate = -5 * torch.sigmoid(
      torch.exp(tensor(layer.A_log[...], torch.float32))[None, :]
      * (gate + tensor(layer.dt_bias[...], torch.float32).reshape(4, 4))
  )
  beta = project(tx, layer.b_proj).float().sigmoid()
  state = torch.zeros((1, 4, 4, 4))
  outputs = []
  for index in range(5):
    state = state * gate[:, index].exp()[..., None]
    error = v[:, index].float() - torch.einsum("bhk,bhkv->bhv", k[:, index], state)
    state = state + beta[:, index, :, None, None] * torch.einsum("bhk,bhv->bhkv", k[:, index], error)
    outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, index], state))
  core = torch.stack(outputs, dim=1).to(tdtype).float()
  normed = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + 1e-5)
  normed = normed * tensor(layer.o_norm.scale[...], torch.float32)
  normed = normed * project(tx, layer.g_proj).reshape(1, 5, 4, 4).float().sigmoid()
  expected = project(normed.to(tdtype).reshape(1, 5, 16), layer.o_proj).float().numpy()
  actual, _ = layer(x, model_mode=MODEL_MODE_TRAIN)
  tolerance = 1e-5 if dtype == "float32" else 0.003
  np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), expected, atol=tolerance, rtol=tolerance)


def test_latent_moe_matches_torch_routing_and_situ(tiny_config):
  layer = KimiSparseMoeBlock(tiny_config, tiny_mesh(tiny_config), rngs=nnx.Rngs(13))
  layer.MoeBlock_0.gate.bias[...] = jnp.array([0.2, -0.3, 0.1, -0.2])
  x = jax.random.normal(jax.random.key(14), (1, 3, 16)) * 3

  def tensor(value):
    return torch.tensor(np.asarray(value, dtype=np.float32))

  tx = tensor(x)
  latent = tx @ tensor(layer.routed_expert_down_proj.kernel[...])
  scores = (tx @ tensor(layer.MoeBlock_0.gate.kernel[...])).sigmoid()
  indices = (scores + tensor(layer.MoeBlock_0.gate.bias[...])).topk(2, dim=-1).indices
  weights = scores.gather(-1, indices)
  weights = weights / weights.sum(-1, keepdim=True)
  gate = torch.einsum("btl,eli->btei", latent, tensor(layer.MoeBlock_0.wi_0[...]))
  up = torch.einsum("btl,eli->btei", latent, tensor(layer.MoeBlock_0.wi_1[...]))
  activated = 4 * torch.tanh(gate / 4) * gate.sigmoid() * (25 * torch.tanh(up / 25))
  expert_outputs = torch.einsum("btei,eil->btel", activated, tensor(layer.MoeBlock_0.wo[...]))
  selected = expert_outputs.gather(2, indices[..., None].expand(-1, -1, -1, 8))
  routed = (selected * weights[..., None]).sum(2)
  routed = routed * torch.rsqrt(routed.square().mean(-1, keepdim=True) + 1e-5)
  routed = routed * tensor(layer.routed_expert_norm.scale[...])
  routed = routed @ tensor(layer.routed_expert_up_proj.kernel[...])
  shared_gate = tx @ tensor(layer.shared_experts.gate_proj.kernel[...])
  shared_up = tx @ tensor(layer.shared_experts.up_proj.kernel[...])
  shared = 4 * torch.tanh(shared_gate / 4) * shared_gate.sigmoid() * (25 * torch.tanh(shared_up / 25))
  expected = routed + shared @ tensor(layer.shared_experts.down_proj.kernel[...])
  np.testing.assert_allclose(layer(x), expected.numpy(), atol=2e-5, rtol=2e-5)


def test_dense_mlp_bfloat16_matches_torch_fused_situ(tiny_config):
  cfg = tiny_config._pydantic_config.model_copy(update={"dtype": DType.BFLOAT16, "weight_dtype": DType.BFLOAT16})
  config = pyconfig.HyperParameters(cfg)
  layer = KimiMLP(config, tiny_mesh(config), 16, 32, rngs=nnx.Rngs(15))

  def tensor(value):
    return torch.tensor(np.asarray(value, dtype=np.float32), dtype=torch.bfloat16)

  x = (jax.random.normal(jax.random.key(16), (1, 3, 16)) * 5).astype(jnp.bfloat16)
  gate = (tensor(x) @ tensor(layer.gate_proj.kernel[...])).float()
  up = (tensor(x) @ tensor(layer.up_proj.kernel[...])).float()
  activated = 4 * torch.tanh(gate / 4) * gate.sigmoid() * (25 * torch.tanh(up / 25))
  expected = activated.bfloat16() @ tensor(layer.down_proj.kernel[...])
  np.testing.assert_allclose(np.asarray(layer(x), dtype=np.float32), expected.float().numpy(), atol=0.01, rtol=0.002)


def test_non_kimi_latent_moe_router_preserves_expert_input_width(tiny_config):

  config = pyconfig.HyperParameters(
      tiny_config._pydantic_config.model_copy(
          update={"model_name": "qwen3-custom", "moe_expert_input_dim": 8, "mlp_activations": ["silu", "linear"]}
      )
  )
  layer = RoutedMoE(
      config=config,
      num_experts=4,
      num_experts_per_tok=2,
      mesh=tiny_mesh(config),
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      rngs=nnx.Rngs(7),
  )
  inputs = jnp.ones((1, 3, 8), dtype=config.dtype)
  assert layer.gate(inputs)[0].shape == (1, 3, 4)
