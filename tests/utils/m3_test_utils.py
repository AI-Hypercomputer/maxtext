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

"""Helpers for m3 tests: tiny configs, and comparing m3 models against legacy MaxText."""

import re

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.utils import max_utils, maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path

_LEGACY_LAYER = re.compile(r"(dense_|moe_)?layers_(\d+)")

# Reduced shapes that keep every model under test small enough for CPU. The
# `base_*` keys are what the model YAMLs set; overriding the derived names
# (`emb_dim`, ...) is silently ignored by pyconfig.
_TINY = {
    "qwen3-0.6b": {
        "base_num_decoder_layers": 2,
        "base_emb_dim": 64,
        "base_mlp_dim": 128,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
    },
    "qwen3-30b-a3b": {
        "base_num_decoder_layers": 2,
        "base_emb_dim": 64,
        "base_mlp_dim": 32,
        "base_moe_mlp_dim": 32,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "load_balance_loss_weight": 0.01,
    },
    "deepseek3-tiny": {
        # 1 dense + 2 MoE layers, so both layer kinds and both scan runs exist.
        "base_num_decoder_layers": 3,
        "first_num_dense_layers": 1,
        "base_emb_dim": 64,
        "base_mlp_dim": 64,
        "base_moe_mlp_dim": 32,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "vocab_size": 256,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_routing_groups": 4,
        "topk_routing_group": 2,
    },
}


def tiny_config(model_name: str, use_m3: bool = True, full_size: bool = False, **overrides) -> pyconfig.HyperParameters:
  """A reduced `model_name` config. `overrides` are applied last, as `key=value` strings.

  `full_size=True` keeps the model YAML's real dimensions instead. Sets
  `sparse_matmul=false`: legacy's dense MoE path is the one m3's `dense`
  executor reproduces exactly, and it runs on CPU. Sets
  `attention=dot_product`: legacy's `autoselected` already means that below
  128 tokens, m3's means a fused kernel on TPU, and the parity tests want the
  reference arithmetic on both sides. Kernel tests override it.
  """
  args = {
      "model_name": model_name,
      "use_m3_model": use_m3,
      "scan_layers": False,
      "sparse_matmul": False,
      "attention": "dot_product",
      "weight_dtype": "bfloat16",
      "per_device_batch_size": 1,
      "max_target_length": 32,
      "max_prefill_predict_length": 16,
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      **({} if full_size else {"override_model_config": True, **_TINY[model_name]}),
      **overrides,
  }
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={_fmt(v)}" for k, v in args.items()])


def _fmt(value) -> str:
  return str(value).lower() if isinstance(value, bool) else str(value)


def legacy_to_m3_path(path: tuple, first_num_dense_layers: int = 0) -> tuple:
  """Maps a legacy unscanned decoder path onto m3's `Layers` container layout.

  `decoder/layers_3/...` becomes `decoder/layers/3/...`. DeepSeek numbers its
  two kinds separately, `dense_layers_i` and `moe_layers_j`; m3 keeps them in
  one container, so `moe_layers_j` maps to index `first_num_dense_layers + j`.
  Other paths are returned unchanged.
  """
  for i in range(len(path) - 1):
    if path[i] == "decoder" and isinstance(path[i + 1], str):
      match = _LEGACY_LAYER.fullmatch(path[i + 1])
      if match:
        index = int(match.group(2)) + (first_num_dense_layers if match.group(1) == "moe_" else 0)
        return path[: i + 1] + ("layers", index) + path[i + 2 :]
  return path


def legacy_state_to_m3(state: nnx.State, first_num_dense_layers: int = 0) -> nnx.State:
  """Returns `state` with every path mapped by `legacy_to_m3_path`."""
  return nnx.from_flat_state(
      [(legacy_to_m3_path(path, first_num_dense_layers), value) for path, value in nnx.to_flat_state(state)]
  )


def flat_params(module: nnx.Module) -> dict[tuple, jax.Array]:
  """`{path: value}` for every `nnx.Param` in `module`."""
  return {path: var[...] for path, var in nnx.to_flat_state(nnx.state(module, nnx.Param))}


# ----------------------------------------------------------------------------
# Legacy parity
# ----------------------------------------------------------------------------


def legacy_mesh(config) -> jax.sharding.Mesh:
  """A one-device mesh over legacy's axis names, which include m3's four."""
  return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(config.mesh_axes)), config.mesh_axes)


def build_pair(model_name: str, mesh, full_size: bool = False, **overrides) -> tuple[nnx.Module, nnx.Module, object]:
  """Builds the legacy and m3 versions of `model_name` from one seed, with legacy's weights copied into m3.

  Returns `(legacy_model, m3_model, m3_config)`. Must run inside `jax.set_mesh(mesh)`.
  """
  cfg_legacy = tiny_config(model_name, use_m3=False, full_size=full_size, **overrides)
  cfg_m3 = tiny_config(model_name, use_m3=True, full_size=full_size, **overrides)
  legacy = model_creation_utils.create_model(cfg_legacy, mesh, rngs=nnx.Rngs(42))
  m3 = model_creation_utils.create_model(cfg_m3, mesh, rngs=nnx.Rngs(42))
  nnx.update(m3, legacy_state_to_m3(nnx.state(legacy, nnx.Param), cfg_m3.first_num_dense_layers))
  return legacy, m3, cfg_m3


def assert_params_match(test, legacy: nnx.Module, m3: nnx.Module, first_num_dense_layers: int = 0) -> None:
  """Same parameter paths (after `legacy_to_m3_path`), shapes and dtypes."""
  flat_legacy = {legacy_to_m3_path(p, first_num_dense_layers): v for p, v in flat_params(legacy).items()}
  flat_m3 = flat_params(m3)
  test.assertEqual(sorted(flat_legacy), sorted(flat_m3))
  for path, value in flat_m3.items():
    test.assertEqual(value.shape, flat_legacy[path].shape, path)
    test.assertEqual(value.dtype, flat_legacy[path].dtype, path)


def random_batch(config, seed: int = 0, batch: int = 2, length: int = 8) -> dict[str, jax.Array]:
  keys = jax.random.split(jax.random.key(seed), 2)
  return {
      "inputs": jax.random.randint(keys[0], (batch, length), 0, config.vocab_size),
      "targets": jax.random.randint(keys[1], (batch, length), 0, config.vocab_size),
      "positions": jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length)),
      "segment_ids": jnp.ones((batch, length), jnp.int32),
  }


def legacy_logits(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  logits = model(
      batch["inputs"],
      batch["positions"],
      decoder_segment_ids=batch["segment_ids"],
      enable_dropout=False,
      model_mode="train",
  )
  nnx.pop(model, nnx.Intermediate)
  return logits


def legacy_loss(model: nnx.Module, batch: dict[str, jax.Array], config) -> jax.Array:
  """Legacy's training loss: cross entropy plus the mean of the sown MoE load-balance losses."""
  logits = model(
      batch["inputs"],
      batch["positions"],
      decoder_segment_ids=batch["segment_ids"],
      enable_dropout=False,
      model_mode="train",
  )
  xent, _ = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(batch["targets"], config.vocab_size))
  loss = jnp.mean(xent)
  intermediates = nnx.pop(model, nnx.Intermediate).to_pure_dict()
  lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediates, "moe_lb_loss")
  if lb_losses:
    loss = loss + jnp.mean(jnp.concatenate(lb_losses))
  return loss


def m3_loss(model: nnx.Module, batch: dict[str, jax.Array], config) -> jax.Array:
  """m3's loss computed the legacy way (one-hot cross entropy), so the two are directly comparable."""
  output = model(batch["inputs"], batch["positions"], batch["segment_ids"])
  xent, _ = max_utils.cross_entropy_with_logits(output.logits, jax.nn.one_hot(batch["targets"], config.vocab_size))
  loss = jnp.mean(xent)
  if output.aux_loss is not None:
    loss = loss + output.aux_loss
  return loss


def assert_losses_close(test, loss_legacy, loss_m3, rtol: float = 1e-3) -> None:
  """Losses agree to `rtol`. bf16 forward passes under autodiff fuse differently on the two stacks,
  so this is looser than the forward-logits check (which is absolute, 1e-4)."""
  legacy, m3 = float(loss_legacy), float(loss_m3)
  test.assertLess(abs(legacy - m3) / (abs(legacy) + 1e-6), rtol, f"legacy loss {legacy} vs m3 loss {m3}")


def assert_grads_close(test, grads_legacy: nnx.Module, grads_m3: nnx.Module, first_num_dense_layers: int = 0) -> None:
  """Gradients agree per parameter: absolute diff under 0.05 or relative (to the max) under 10%."""
  flat_legacy = {legacy_to_m3_path(p, first_num_dense_layers): v for p, v in flat_params(grads_legacy).items()}
  flat_m3 = flat_params(grads_m3)
  test.assertEqual(sorted(flat_legacy), sorted(flat_m3))
  for path in sorted(flat_m3):
    g_legacy, g_m3 = flat_legacy[path].astype(jnp.float32), flat_m3[path].astype(jnp.float32)
    diff = float(jnp.max(jnp.abs(g_legacy - g_m3)))
    scale = float(jnp.max(jnp.abs(g_legacy)))
    test.assertTrue(diff < 0.05 or diff / (scale + 1e-6) < 0.10, f"{path}: diff={diff:.4g}, scale={scale:.4g}")
