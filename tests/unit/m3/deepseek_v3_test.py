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

"""Tests for m3 DeepSeek-V3.

`DeepSeekV3ComparisonTest` checks parity against legacy MaxText DeepSeek-V3 on a
1-dense + 2-MoE layer `deepseek3-tiny` with group-limited routing enabled, so
MLA with YaRN, the sigmoid router with its bias, shared experts and both layer
kinds are exercised. Legacy numbers its layer kinds separately
(`dense_layers_0`, `moe_layers_0`) and nests the MoE block under Linen-era
names; `_legacy_to_m3_path` maps those onto m3's layout.

`DeepSeekV3Test` covers what has no legacy counterpart: scan, sharding, the
settings the model refuses, and the m3 loop.
"""

import math
import re

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3 import models as m3_models
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepSeekV3DenseLayer,
    DeepSeekV3Model,
    DeepSeekV3MoeLayer,
    DeepSeekV3Router,
    RouterBias,
)
from maxtext.m3.train import loop
from maxtext.utils import max_utils, model_creation_utils
from tests.utils import m3_test_utils

_MODEL = "deepseek3-tiny"
_LEGACY_LAYER = re.compile(r"(dense|moe)_layers_(\d+)")
_LEGACY_SEGMENTS = {"DeepSeekMoeBlock_0": "moe_block", "MoeBlock_0": "routed_experts"}


def _legacy_to_m3_path(path: tuple, first_num_dense_layers: int) -> tuple:
  """Maps a legacy unscanned parameter path onto m3's layout.

  `dense_layers_i` becomes `layers/layers/i` and `moe_layers_j` becomes
  `layers/layers/(first_num_dense_layers + j)`: m3 keeps both kinds in one
  `nnx.Sequential`, which holds its children in `.layers`.
  """
  out = []
  for segment in path:
    match = _LEGACY_LAYER.fullmatch(segment) if isinstance(segment, str) else None
    if match:
      index = int(match.group(2)) + (first_num_dense_layers if match.group(1) == "moe" else 0)
      out += ["layers", "layers", index]
    else:
      out.append(_LEGACY_SEGMENTS.get(segment, segment))
  return tuple(out)


def _legacy_params_as_m3(module: nnx.Module, first_num_dense_layers: int) -> dict[tuple, jax.Array]:
  """`{m3 path: value}` for every `nnx.Param` of a legacy model or of gradients with its structure."""
  return {_legacy_to_m3_path(p, first_num_dense_layers): v for p, v in m3_test_utils.flat_params(module).items()}


def _build_pair(mesh, **overrides):
  """The legacy and m3 models built from one config, with legacy's weights copied into m3.

  Returns `(legacy_model, m3_model, m3_config)`. Must run inside `jax.set_mesh(mesh)`.
  """
  cfg_legacy = m3_test_utils.tiny_config(_MODEL, use_m3=False, **overrides)
  cfg_m3 = m3_test_utils.tiny_config(_MODEL, use_m3=True, **overrides)
  legacy = model_creation_utils.create_model(cfg_legacy, mesh, rngs=nnx.Rngs(42))
  m3 = model_creation_utils.create_model(cfg_m3, mesh, rngs=nnx.Rngs(42))
  legacy_params = nnx.to_flat_state(nnx.state(legacy, nnx.Param))
  first_num_dense_layers = cfg_m3.first_num_dense_layers
  nnx.update(m3, nnx.from_flat_state([(_legacy_to_m3_path(p, first_num_dense_layers), v) for p, v in legacy_params]))
  return legacy, m3, cfg_m3


def _random_batch(config, seed: int = 0, batch: int = 2, length: int = 8) -> dict[str, jax.Array]:
  keys = jax.random.split(jax.random.key(seed), 2)
  return {
      "inputs": jax.random.randint(keys[0], (batch, length), 0, config.vocab_size),
      "targets": jax.random.randint(keys[1], (batch, length), 0, config.vocab_size),
      "positions": jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length)),
      "segment_ids": jnp.ones((batch, length), jnp.int32),
  }


def _legacy_logits(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  logits = model(
      batch["inputs"],
      batch["positions"],
      decoder_segment_ids=batch["segment_ids"],
      enable_dropout=False,
      model_mode="train",
  )
  nnx.pop(model, nnx.Intermediate)
  return logits


def _m3_logits(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  return model(batch["inputs"], batch["positions"], batch["segment_ids"])


def _cross_entropy(logits: jax.Array, batch: dict[str, jax.Array], config) -> jax.Array:
  """Legacy's one-hot cross entropy, so the two stacks' losses are directly comparable."""
  xent, _ = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(batch["targets"], config.vocab_size))
  return jnp.mean(xent)


def _one_device_legacy_mesh(config) -> jax.sharding.Mesh:
  """A one-device mesh over legacy's axis names, which include m3's."""
  return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(config.mesh_axes)), config.mesh_axes)


class DeepSeekV3ComparisonTest(absltest.TestCase):
  """Compares legacy DeepSeek-V3 (`use_m3_model=false`) with m3 DeepSeek-V3 on the same weights."""

  def setUp(self):
    super().setUp()
    self.mesh = _one_device_legacy_mesh(m3_test_utils.tiny_config(_MODEL))
    self.enter_context(jax.set_mesh(self.mesh))

  def test_create_model_routes_to_m3(self):
    cfg = m3_test_utils.tiny_config(_MODEL)
    model = model_creation_utils.create_model(cfg, self.mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(model, DeepSeekV3Model)
    logits = nnx.jit(_m3_logits)(model, _random_batch(cfg))
    self.assertEqual(logits.shape, (2, 8, cfg.vocab_size))

  def test_parameters_match(self):
    """Same parameter paths (after `_legacy_to_m3_path`), shapes and dtypes."""
    legacy, m3, cfg = _build_pair(self.mesh)
    flat_legacy = _legacy_params_as_m3(legacy, cfg.first_num_dense_layers)
    flat_m3 = m3_test_utils.flat_params(m3)
    self.assertEqual(sorted(map(str, flat_legacy)), sorted(map(str, flat_m3)))
    for path, value in flat_m3.items():
      self.assertEqual(value.shape, flat_legacy[path].shape, path)
      self.assertEqual(value.dtype, flat_legacy[path].dtype, path)
    self.assertIn(("decoder", "layers", "layers", 0, "mlp", "wi_0", "kernel"), flat_m3)
    self.assertIn(("decoder", "layers", "layers", 1, "moe_block", "routed_experts", "gate", "kernel"), flat_m3)
    self.assertIn(("decoder", "layers", "layers", 1, "moe_block", "shared_experts", "wo", "kernel"), flat_m3)
    # The routing bias is state, not a parameter.
    self.assertNotIn(("decoder", "layers", "layers", 1, "moe_block", "routed_experts", "gate", "bias"), flat_m3)
    self.assertNotEmpty(nnx.to_flat_state(nnx.state(m3, RouterBias)))

  def test_forward_logits_match(self):
    # Float32 avoids bfloat16 rounding differences from `core/moe.py`'s float32 expert combine.
    legacy, m3, cfg = _build_pair(self.mesh, dtype="float32", weight_dtype="float32")
    legacy_fn, m3_fn = nnx.jit(_legacy_logits), nnx.jit(_m3_logits)
    batch = _random_batch(cfg)
    diff = jnp.max(jnp.abs(legacy_fn(legacy, batch) - m3_fn(m3, batch)))
    self.assertLess(float(diff), 1e-4)

    # Packed sequences with multiple segments, reset positions, and trailing padding.
    packed = {
        **batch,
        "segment_ids": jnp.array([[1, 1, 1, 2, 2, 2, 0, 0], [1, 1, 2, 2, 2, 2, 2, 0]], dtype=jnp.int32),
        "positions": jnp.array([[0, 1, 2, 0, 1, 2, 0, 0], [0, 1, 0, 1, 2, 3, 4, 0]], dtype=jnp.int32),
    }
    valid = (packed["segment_ids"] != 0)[..., None]
    packed_diff = jnp.max(jnp.abs(jnp.where(valid, legacy_fn(legacy, packed) - m3_fn(m3, packed), 0.0)))
    self.assertLess(float(packed_diff), 1e-4)

    # A non-zero bias changes which experts are selected but not their weights.
    bias = jax.random.normal(jax.random.key(3), (cfg.num_experts,), jnp.float32)
    routers = [node for _, node in nnx.iter_graph(legacy) if type(node).__name__ == "GateLogit"]
    routers += [node for _, node in nnx.iter_graph(m3) if isinstance(node, DeepSeekV3Router)]
    self.assertLen(routers, 2 * (cfg.num_decoder_layers - cfg.first_num_dense_layers))
    for router in routers:
      router.bias[...] = bias.astype(router.bias[...].dtype)
    batch2 = _random_batch(cfg, seed=2)
    bias_diff = jnp.max(jnp.abs(legacy_fn(legacy, batch2) - m3_fn(m3, batch2)))
    self.assertLess(float(bias_diff), 1e-4)

  def test_loss_and_gradients_match(self):
    legacy, m3, cfg = _build_pair(self.mesh)
    batch = _random_batch(cfg, seed=1)

    @nnx.jit
    def legacy_vg(model, b):
      return nnx.value_and_grad(lambda m: _cross_entropy(_legacy_logits(m, b), b, cfg))(model)

    @nnx.jit
    def m3_vg(model, b):
      return nnx.value_and_grad(lambda m: _cross_entropy(_m3_logits(m, b), b, cfg))(model)

    loss_legacy, grads_legacy = legacy_vg(legacy, batch)
    loss_m3, grads_m3 = m3_vg(m3, batch)
    self.assertLess(abs(float(loss_legacy) - float(loss_m3)) / abs(float(loss_legacy)), 1e-3)
    flat_legacy = _legacy_params_as_m3(grads_legacy, cfg.first_num_dense_layers)
    flat_m3 = m3_test_utils.flat_params(grads_m3)
    self.assertEqual(sorted(map(str, flat_legacy)), sorted(map(str, flat_m3)))
    for path in flat_m3:
      g_legacy, g_m3 = flat_legacy[path].astype(jnp.float32), flat_m3[path].astype(jnp.float32)
      diff = float(jnp.max(jnp.abs(g_legacy - g_m3)))
      scale = float(jnp.max(jnp.abs(g_legacy)))
      self.assertTrue(diff < 0.05 or diff / (scale + 1e-6) < 0.10, f"{path}: diff={diff:.4g}, scale={scale:.4g}")


class DeepSeekV3Test(parameterized.TestCase):
  """m3 DeepSeek-V3 on its own: scan, sharding, refused settings and the loop."""

  def test_scanned_matches_unscanned(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      cfg = m3_test_utils.tiny_config(_MODEL, dtype="float32", weight_dtype="float32")
      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      # Give each MoE layer a distinct bias so scanning verifies per-layer `RouterBias` slicing.
      routers = [node for _, node in nnx.iter_graph(model) if isinstance(node, DeepSeekV3Router)]
      for i, router in enumerate(routers):
        router.bias[...] = jax.random.normal(jax.random.key(10 + i), (cfg.num_experts,), jnp.float32)
      batch = _random_batch(cfg)
      expected = _m3_logits(model, batch)
      num_params = len(m3_test_utils.flat_params(model))
      scanned = m3_scan.apply_scan(model, param_scan_axis=cfg.param_scan_axis)
      # One scanned run per layer kind, each holding one layer's parameter tree.
      runs = list(scanned.decoder.layers)
      self.assertEqual([run.num_layers for run in runs], [1, 2])
      self.assertLess(len(m3_test_utils.flat_params(scanned)), num_params)
      np.testing.assert_allclose(_m3_logits(scanned, batch), expected, rtol=1e-5, atol=1e-5)

  def test_first_num_dense_layers_clamped_to_num_decoder_layers(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      cfg = m3_test_utils.tiny_config(_MODEL, base_num_decoder_layers=2, first_num_dense_layers=3)
      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      layers = list(model.decoder.layers.layers)
      self.assertLen(layers, 2)
      self.assertTrue(all(isinstance(layer, DeepSeekV3DenseLayer) for layer in layers))
      logits = nnx.jit(_m3_logits)(model, _random_batch(cfg))
      self.assertEqual(logits.shape, (2, 8, cfg.vocab_size))
      self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))

  def test_shared_experts_zero(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      cfg = m3_test_utils.tiny_config(_MODEL, shared_experts=0)
      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      moe_layers = [layer for layer in model.decoder.layers.layers if isinstance(layer, DeepSeekV3MoeLayer)]
      self.assertNotEmpty(moe_layers)
      for layer in moe_layers:
        self.assertIsNone(layer.moe_block.shared_experts)
      logits = nnx.jit(_m3_logits)(model, _random_batch(cfg))
      self.assertEqual(logits.shape, (2, 8, cfg.vocab_size))
      self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))

  def test_every_parameter_gets_a_gradient(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      cfg = m3_test_utils.tiny_config(_MODEL)
      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      batch = _random_batch(cfg)
      grads = nnx.grad(lambda m: _cross_entropy(_m3_logits(m, batch), batch, cfg))(model)
    for path, grad in m3_test_utils.flat_params(grads).items():
      grad = np.asarray(grad.astype(jnp.float32))
      self.assertTrue(np.all(np.isfinite(grad)), path)
      self.assertGreater(np.max(np.abs(grad)), 0.0, path)

  def test_weights_shard_over_fsdp(self):
    """Under FSDP alone every weight matrix is split, MLA's up-projections included."""
    if jax.device_count() < 2:
      self.skipTest(f"needs at least 2 devices, found {jax.device_count()}")
    cfg = m3_test_utils.tiny_config(_MODEL)
    mesh = create_mesh(ici={"fsdp": 2}, devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
    params = m3_test_utils.flat_params(model)
    for path, value in params.items():
      held = math.prod(value.sharding.shard_shape(value.shape)) / value.size
      self.assertEqual(held, 0.5 if value.ndim > 1 else 1.0, path)
    attention = ("decoder", "layers", "layers", 0, "self_attention")
    for name in ("wq_b", "wkv_b"):
      kernel = params[attention + (name, "kernel")]
      self.assertEqual(kernel.sharding.shard_shape(kernel.shape)[0] * 2, kernel.shape[0], name)

  def test_weights_shard_over_tp(self):
    """Under TP alone, head/MLP/vocab projections split 2-way while latent down-projections and gate stay replicated."""
    if jax.device_count() < 2:
      self.skipTest(f"needs at least 2 devices, found {jax.device_count()}")
    cfg = m3_test_utils.tiny_config(_MODEL, use_tokamax_gmm=True, use_gmm_v2=True)
    mesh = create_mesh(ici={"tensor": 2}, devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      logits = nnx.jit(_m3_logits)(model, _random_batch(cfg))
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
    params = m3_test_utils.flat_params(model)
    sharded_leaves = {"wq_b", "wkv_b", "out", "wi_0", "wi_1", "wo", "token_embedder", "logits_dense"}
    for path, value in params.items():
      held = math.prod(value.sharding.shard_shape(value.shape)) / value.size
      expects_split = value.ndim > 1 and any(seg in sharded_leaves for seg in path)
      self.assertEqual(held, 0.5 if expects_split else 1.0, path)

  @parameterized.parameters(
      ("load_balance_loss_weight", {"load_balance_loss_weight": 0.01}),
      ("routed_bias_update_rate", {"routed_bias_update_rate": 0.001}),
      ("mtp_num_layers", {"mtp_num_layers": 1}),
      ("logits_via_embedding", {"logits_via_embedding": True}),
      ("use_indexer", {"use_indexer": True, "attention": "dot_product"}),
      ("mhc_expansion_rate", {"mhc_expansion_rate": 2}),
      ("mlp_activations_limit", {"mlp_activations_limit": 7.0}),
      (
          "engram_layers",
          {"engram_layers": "[1]", "engram_vocab_bases": "[64,64]", "hf_access_token": "test", "tokenizer_path": "test"},
      ),
      ("dropout_rate", {"enable_dropout": True, "dropout_rate": 0.1}),
      ("float32_gate_logits", {"float32_gate_logits": True}),
  )
  def test_unimplemented_settings_are_refused(self, field, overrides):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      cfg = m3_test_utils.tiny_config(_MODEL, **overrides)
      with self.assertRaisesRegex(ValueError, field):
        m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))

  def test_legacy_megablox_default_is_refused(self):
    """`base.yml` defaults to the legacy Megablox kernel, which `core/moe.py` does not run."""
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
      cfg = m3_test_utils.tiny_config(_MODEL, megablox=True)
      with self.assertRaisesRegex(ValueError, "megablox"):
        m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))

  @parameterized.named_parameters(
      ("ragged_dot", {}),
      ("dense", {"sparse_matmul": False}),
      ("gmm_v2", {"use_tokamax_gmm": True, "use_gmm_v2": True}),
      ("scanned", {"scan_layers": True}),
  )
  def test_train_loop_runs(self, overrides):
    config = m3_test_utils.tiny_config(
        _MODEL,
        steps=2,
        base_output_directory=self.create_tempdir().full_path,
        run_name="m3_deepseek_v3_loop_test",
        **overrides,
    )
    loss = loop.train_loop(config)
    self.assertTrue(math.isfinite(loss), f"loss={loss}")

  def test_train_loop_runs_with_expert_parallelism(self):
    if jax.device_count() < 2 or jax.device_count() % 2 != 0:
      self.skipTest(f"needs an even number of >= 2 devices, found {jax.device_count()}")
    config = m3_test_utils.tiny_config(
        _MODEL,
        steps=2,
        ici_expert_parallelism=2,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        base_output_directory=self.create_tempdir().full_path,
        run_name="m3_deepseek_v3_ep_loop_test",
    )
    loss = loop.train_loop(config)
    self.assertTrue(math.isfinite(loss), f"loss={loss}")


if __name__ == "__main__":
  absltest.main()
