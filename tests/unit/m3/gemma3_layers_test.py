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

"""Unit, golden-parity, sharding, packing, and negative-control tests for Gemma 3 m3 layers."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib
import types
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.configs import pyconfig
from maxtext.m3.configs.config import load_yaml_config, process_config
from maxtext.m3.models.gemma3 import (
    Gemma3Attention,
    Gemma3Config,
    Gemma3DecoderLayer,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3Sharding,
    Gemma3TextScaledWordEmbedding,
)
from maxtext.utils import globals as maxtext_globals
import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

_GOLDEN_NPZ = pathlib.Path(__file__).resolve().parent / "testdata" / "gemma3" / "gemma3_hf_golden.npz"


def _load_layer_from_golden(
    layer: Gemma3DecoderLayer,
    data: np.lib.npyio.NpzFile,
    prefix: str,
    layer_idx: int,
) -> None:
  """Loads 1:1 HF weights from `gemma3_hf_golden.npz` into `Gemma3DecoderLayer`."""
  p = f"{prefix}__param__model.layers.{layer_idx}"
  h_dim = layer.self_attention.head_dim
  n_q = layer.self_attention.num_query_heads
  n_kv = layer.self_attention.num_kv_heads

  layer.pre_self_attention_norm.weight.set_value(jnp.asarray(np.asarray(data[f"{p}.input_layernorm.weight"])))
  layer.post_self_attention_norm.weight.set_value(jnp.asarray(np.asarray(data[f"{p}.post_attention_layernorm.weight"])))
  layer.pre_ffw_norm.weight.set_value(jnp.asarray(np.asarray(data[f"{p}.pre_feedforward_layernorm.weight"])))
  layer.post_ffw_norm.weight.set_value(jnp.asarray(np.asarray(data[f"{p}.post_feedforward_layernorm.weight"])))
  layer.self_attention.query_norm.weight.set_value(jnp.asarray(np.asarray(data[f"{p}.self_attn.q_norm.weight"])))
  layer.self_attention.key_norm.weight.set_value(jnp.asarray(np.asarray(data[f"{p}.self_attn.k_norm.weight"])))

  q_w = np.asarray(data[f"{p}.self_attn.q_proj.weight"]).T.reshape(-1, n_q, h_dim)
  k_w = np.asarray(data[f"{p}.self_attn.k_proj.weight"]).T.reshape(-1, n_kv, h_dim)
  v_w = np.asarray(data[f"{p}.self_attn.v_proj.weight"]).T.reshape(-1, n_kv, h_dim)
  o_w = np.asarray(data[f"{p}.self_attn.o_proj.weight"]).T.reshape(n_q, h_dim, -1)
  layer.self_attention.query.kernel.set_value(jnp.asarray(q_w))
  layer.self_attention.key.kernel.set_value(jnp.asarray(k_w))
  layer.self_attention.value.kernel.set_value(jnp.asarray(v_w))
  layer.self_attention.out.kernel.set_value(jnp.asarray(o_w))

  layer.mlp.wi_0.kernel.set_value(jnp.asarray(np.asarray(data[f"{p}.mlp.gate_proj.weight"]).T))
  layer.mlp.wi_1.kernel.set_value(jnp.asarray(np.asarray(data[f"{p}.mlp.up_proj.weight"]).T))
  layer.mlp.wo.kernel.set_value(jnp.asarray(np.asarray(data[f"{p}.mlp.down_proj.weight"]).T))


def _numpy_f64_attention_oracle(
    inputs: np.ndarray,
    positions: np.ndarray,
    *,
    q_kernel: np.ndarray,
    k_kernel: np.ndarray,
    v_kernel: np.ndarray,
    o_kernel: np.ndarray,
    q_norm_w: np.ndarray,
    k_norm_w: np.ndarray,
    eps: float,
    rope_theta: float,
    rope_factor: float,
    query_pre_attn_scalar: float,
    sliding_window: int | None,
) -> np.ndarray:
  """Independent float64 NumPy reference for Gemma 3 self-attention."""
  x = inputs.astype(np.float64)
  q = np.einsum("bte,ehd->bthd", x, q_kernel.astype(np.float64))
  k = np.einsum("bte,ekd->btkd", x, k_kernel.astype(np.float64))
  v = np.einsum("bte,ekd->btkd", x, v_kernel.astype(np.float64))

  def _rms(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    var = np.mean(u * u, axis=-1, keepdims=True)
    return u / np.sqrt(var + eps) * (1.0 + w.astype(np.float64))

  q = _rms(q, q_norm_w)
  k = _rms(k, k_norm_w)

  d = q.shape[-1]
  exp = np.arange(0, d, 2, dtype=np.float64) / np.float64(d)
  inv_freq = (1.0 / (np.float64(rope_theta) ** exp)) / np.float64(rope_factor)
  freqs = positions.astype(np.float64)[..., None] * inv_freq[None, None, :]
  emb = np.concatenate([freqs, freqs], axis=-1)
  cos = np.cos(emb)[..., None, :]
  sin = np.sin(emb)[..., None, :]

  def _rot(u: np.ndarray) -> np.ndarray:
    u1, u2 = u[..., : d // 2], u[..., d // 2 :]
    return u * cos + np.concatenate([-u2, u1], axis=-1) * sin

  q = _rot(q)
  k = _rot(k)

  _, _, n_q, _ = q.shape
  n_kv = k.shape[2]
  groups = n_q // n_kv
  k_rep = np.repeat(k, groups, axis=2)
  v_rep = np.repeat(v, groups, axis=2)

  logits = np.einsum("bthd,bshd->bhts", q, k_rep) * (query_pre_attn_scalar**-0.5)

  idx = np.arange(positions.shape[-1])
  delta = (idx[:, None] - idx[None, :])[None, :, :]
  mask = delta >= 0
  if sliding_window is not None:
    mask = mask & (delta < sliding_window)
  logits = np.where(mask[:, None, :, :], logits, -1e30)
  logits_max = np.max(logits, axis=-1, keepdims=True)
  probs = np.exp(logits - logits_max)
  probs = probs / np.sum(probs, axis=-1, keepdims=True)
  ctx = np.einsum("bhts,bshd->bthd", probs, v_rep)
  return np.einsum("bthd,hde->bte", ctx, o_kernel.astype(np.float64)).astype(np.float32)


def _primitive_input_dtypes(jaxpr, name: str) -> list:
  """Returns the input dtypes of every `name` primitive in `jaxpr`, including nested sub-jaxprs."""
  found = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == name:
      found.extend(v.aval.dtype for v in eqn.invars)
    for param in eqn.params.values():
      sub_jaxprs = param if isinstance(param, (list, tuple)) else [param]
      for sub_jaxpr in sub_jaxprs:
        inner = sub_jaxpr.jaxpr if hasattr(sub_jaxpr, "jaxpr") else sub_jaxpr
        if hasattr(inner, "eqns"):
          found.extend(_primitive_input_dtypes(inner, name))
  return found


class Gemma3LayersTest(unittest.TestCase):
  """Two-tier verification of Gemma 3 config, norms, embeddings, RoPE, MLP, attention, and sharding."""

  def setUp(self) -> None:
    super().setUp()
    self._stack = contextlib.ExitStack()
    self._mesh_1d = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("data", "fsdp", "tensor", "expert"),
    )
    self._stack.enter_context(jax.set_mesh(self._mesh_1d))
    self._stack.enter_context(nnx.use_eager_sharding(True))

  def tearDown(self) -> None:
    self._stack.close()
    super().tearDown()

  def test_config_presets_hf_unwrap_and_validation(self) -> None:
    """Checks 1B/4B/12B/27B presets, multimodal text_config unwrapping, v4/v5 RoPE, and validation."""
    cfg_1b = Gemma3Config.preset("gemma3-1b")
    self.assertEqual(cfg_1b.num_key_value_heads, 1)
    self.assertEqual(cfg_1b.query_pre_attn_scalar, 256.0)
    self.assertEqual(cfg_1b.sliding_window, 512)
    self.assertEqual([cfg_1b.is_sliding_layer(i) for i in range(6)], [True, True, True, True, True, False])

    cfg_27b = Gemma3Config.preset("gemma3-27b")
    self.assertEqual(cfg_27b.head_dim, 128)
    self.assertEqual(cfg_27b.query_pre_attn_scalar, 168.0)
    self.assertEqual(cfg_27b.rope_linear_scaling_factor, 8.0)

    mm_wrapped = {
        "model_type": "gemma3",
        "tie_word_embeddings": False,
        "text_config": {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_parameters": {
                "full_attention": {"rope_theta": 500_000.0, "factor": 4.0},
                "sliding_attention": {"rope_theta": 20_000.0},
            },
        },
    }
    parsed = Gemma3Config.from_hf_config_json(mm_wrapped)
    self.assertEqual(parsed.hidden_size, 64)
    self.assertFalse(parsed.tie_word_embeddings)
    self.assertEqual(parsed.rope_theta, 500_000.0)
    self.assertEqual(parsed.rope_local_base_freq, 20_000.0)
    self.assertEqual(parsed.rope_linear_scaling_factor, 4.0)

    null_tc = Gemma3Config.from_hf_config_json({"text_config": None, "vocab_size": 128})
    self.assertEqual(null_tc.vocab_size, 128)

    with self.assertRaisesRegex(ValueError, "even for RoPE"):
      Gemma3Config(head_dim=7)
    with self.assertRaisesRegex(ValueError, "attention_bias"):
      Gemma3Config(attention_bias=True)
    with self.assertRaisesRegex(ValueError, "attn_logit_softcapping"):
      Gemma3Config(attn_logit_softcapping=50.0)
    with self.assertRaisesRegex(ValueError, "remat_policy"):
      Gemma3Config(remat_policy="save_qkv_proj")
    with self.assertRaisesRegex(ValueError, "remat_policy"):
      Gemma3Config.from_hf_config_json({"remat_policy": "save_qkv_proj"})
    with self.assertRaisesRegex(ValueError, "rope_type"):
      Gemma3Config.from_hf_config_json({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}})
    with self.assertRaisesRegex(ValueError, "rope_type"):
      Gemma3Config.from_hf_config_json({"rope_parameters": {"full_attention": {"rope_type": "llama3", "factor": 4.0}}})
    with self.assertRaisesRegex(ValueError, "sliding_attention"):
      Gemma3Config.from_hf_config_json({"rope_parameters": {"sliding_attention": {"rope_type": "linear", "factor": 4.0}}})
    with self.assertRaisesRegex(ValueError, "sliding_attention"):
      Gemma3Config.from_hf_config_json(
          {"rope_parameters": {"sliding_attention": {"rope_type": "default", "factor": 2.0}}}
      )
    with self.assertRaisesRegex(ValueError, "attention_dropout"):
      Gemma3Config(attention_dropout=0.1)
    with self.assertRaisesRegex(ValueError, "bidirectional"):
      Gemma3Config.from_hf_config_json({"use_bidirectional_attention": True})

    # Diff-style HF text config (keys equal to HF defaults omitted): omitted keys take HF Gemma3TextConfig
    # defaults (8 heads, 4 KV heads, vocab 262208), never the gemma3-1b preset.
    diff_4b = Gemma3Config.from_hf_config_json(
        {
            "text_config": {
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "num_hidden_layers": 34,
                "rope_scaling": {"rope_type": "linear", "factor": 8.0},
                "sliding_window": 1024,
            }
        }
    )
    self.assertEqual(
        (diff_4b.num_attention_heads, diff_4b.num_key_value_heads, diff_4b.vocab_size, diff_4b.head_dim),
        (8, 4, 262_208, 256),
    )
    self.assertEqual(diff_4b.rope_linear_scaling_factor, 8.0)
    # HF `save_pretrained` writes `_sliding_window_pattern` and `layer_types` instead of `sliding_window_pattern`.
    six_layers = ["sliding_attention"] * 5 + ["full_attention"]
    saved = Gemma3Config.from_hf_config_json(
        {"num_hidden_layers": 6, "_sliding_window_pattern": 6, "layer_types": six_layers}
    )
    self.assertEqual(saved.sliding_window_pattern, 6)
    with self.assertRaisesRegex(ValueError, "layer_types"):
      Gemma3Config.from_hf_config_json({"num_hidden_layers": 6, "layer_types": ["full_attention"] * 6})

  def test_from_config_base_and_maxtext_precedence_and_no_silent_fallbacks(self) -> None:
    """Checks `from_config` across `Gemma3Config`, `BaseConfig`, and `MaxTextConfig` without fallbacks."""
    direct = Gemma3Config.preset("gemma3-4b")
    replaced = Gemma3Config.from_config(direct, num_hidden_layers=4)
    self.assertEqual(replaced.num_hidden_layers, 4)
    self.assertEqual(replaced.hidden_size, 2560)

    # BaseConfig: effective keys (`emb_dim`, `mlp_dim`, `num_*`) must override `base_*` keys.
    base_cfg = process_config(
        {
            "model_name": "gemma3-4b",
            "base_emb_dim": 2560,
            "emb_dim": 128,
            "base_mlp_dim": 10240,
            "mlp_dim": 256,
            "base_num_decoder_layers": 34,
            "num_decoder_layers": 6,
            "base_num_query_heads": 8,
            "num_query_heads": 4,
            "base_num_kv_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 16,
            "sliding_window_size": 64,
            "local_rope_max_timescale": 10_000.0,
            "rope_max_timescale": 1_000_000.0,
            "rope_linear_scaling_factor": 8.0,
            "param_scan_axis": 1,
            "final_logits_soft_cap": 30.0,
            "skip_jax_distributed_system": True,
        }
    )
    parsed_base = Gemma3Config.from_config(base_cfg)
    self.assertEqual(parsed_base.hidden_size, 128)
    self.assertEqual(parsed_base.intermediate_size, 256)
    self.assertEqual(parsed_base.num_hidden_layers, 6)
    self.assertEqual(parsed_base.num_attention_heads, 4)
    self.assertEqual(parsed_base.num_key_value_heads, 2)
    self.assertEqual(parsed_base.sliding_window, 64)
    self.assertEqual(parsed_base.stack_cycles, parsed_base.scan_layers)
    self.assertEqual(parsed_base.param_scan_axis, 1)
    self.assertEqual(parsed_base.final_logit_softcapping, 30.0)

    bad_base = process_config(
        {
            "model_name": "gemma3-4b",
            "sliding_window_size": 0,
            "skip_jax_distributed_system": True,
        }
    )
    with self.assertRaisesRegex(ValueError, "sliding_window"):
      Gemma3Config.from_config(bad_base)
    with self.assertRaisesRegex(ValueError, "known model_name"):
      Gemma3Config.from_config(process_config({"model_name": "llama3-8b", "skip_jax_distributed_system": True}))
    with self.assertRaisesRegex(ValueError, "BaseConfig without a known Gemma 3 model_name"):
      Gemma3Config.from_config(
          process_config(
              {
                  "emb_dim": 128,
                  "mlp_dim": 256,
                  "num_decoder_layers": 6,
                  "num_query_heads": 4,
                  "num_kv_heads": 2,
                  "head_dim": 16,
                  "skip_jax_distributed_system": True,
              }
          )
      )

    # Real MaxTextConfig from pyconfig for the stock gemma3-4b / gemma3-27b YAMLs (base.yml defaults included).
    base_yml = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
    g27_yml = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "models", "gemma3-27b.yml")
    with self.assertRaisesRegex(ValueError, "query_pre_attn_scalar"):
      Gemma3Config.from_config(process_config(load_yaml_config(g27_yml), skip_jax_distributed_system=True))
    for name, hidden, heads, kv, head_dim, q_scalar in (
        ("gemma3-4b", 2560, 8, 4, 256, 256.0),
        ("gemma3-27b", 5376, 32, 16, 128, 168.0),
    ):
      mt_cfg = pyconfig.initialize([None, base_yml], model_name=name, use_m3_model=True, skip_jax_distributed_system=True)
      got = Gemma3Config.from_config(mt_cfg)
      self.assertEqual(
          (got.hidden_size, got.num_attention_heads, got.num_key_value_heads, got.head_dim, got.query_pre_attn_scalar),
          (hidden, heads, kv, head_dim, q_scalar),
          name,
      )
      self.assertEqual((got.sliding_window, got.rope_linear_scaling_factor), (1024, 8.0), name)
      self.assertEqual((got.rope_theta, got.rope_local_base_freq), (1_000_000.0, 10_000.0), name)
      self.assertEqual(got.param_scan_axis, int(mt_cfg.param_scan_axis), name)
      self.assertEqual(got.stack_cycles, bool(mt_cfg.scan_layers), name)

    # MaxTextConfig-shaped namespace only for the <=0 sentinel and unknown-value rejections.
    def _make_mt_ns(**kw):
      defaults = {
          "use_m3_model": True,
          "model_name": "gemma3-27b",
          "vocab_size": 262_144,
          "emb_dim": 5376,
          "mlp_dim": 21504,
          "num_decoder_layers": 62,
          "num_query_heads": 32,
          "num_kv_heads": 16,
          "head_dim": 128,
          "sliding_window_size": 1024,
          "normalization_layer_epsilon": 1e-6,
          "rope_max_timescale": 1_000_000.0,
          "local_rope_max_timescale": 10_000.0,
          "rope_linear_scaling_factor": 8.0,
          "attn_logits_soft_cap": 0.0,
          "final_logits_soft_cap": 0.0,
          "logits_via_embedding": True,
          "dtype": "bfloat16",
          "weight_dtype": "float32",
          "scan_layers": True,
          "remat_policy": "minimal",
          "matmul_precision": "default",
          "param_scan_axis": 0,
          "dropout_rate": 0.0,
          "mlp_activations": ["gelu", "linear"],
      }
      defaults.update(kw)
      return types.SimpleNamespace(**defaults)

    mt_27b = Gemma3Config.from_config(_make_mt_ns())
    self.assertEqual(mt_27b.query_pre_attn_scalar, 168.0)
    self.assertEqual(mt_27b.remat_policy, "minimal")
    self.assertIsNone(mt_27b.attn_logit_softcapping)
    self.assertIsNone(mt_27b.final_logit_softcapping)
    self.assertFalse(Gemma3Config.from_config(_make_mt_ns(scan_layers=False)).stack_cycles)

    with self.assertRaisesRegex(ValueError, "use_m3_model=True"):
      Gemma3Config.from_config(_make_mt_ns(use_m3_model=False))
    with self.assertRaisesRegex(ValueError, "sliding_window_size"):
      Gemma3Config.from_config(_make_mt_ns(sliding_window_size=0))
    with self.assertRaisesRegex(ValueError, "local_rope_max_timescale"):
      Gemma3Config.from_config(_make_mt_ns(local_rope_max_timescale=-1.0))
    with self.assertRaisesRegex(ValueError, "remat_policy"):
      Gemma3Config.from_config(_make_mt_ns(remat_policy="custom"))
    with self.assertRaisesRegex(ValueError, "known model_name"):
      Gemma3Config.from_config(_make_mt_ns(model_name="llama3-8b"))

    # Dropout and activation settings are rejected, never silently dropped, on every config path.
    for bad in ({"dropout_rate": 0.1}, {"mlp_activations": ["silu", "linear"]}):
      bad_mt = pyconfig.initialize(
          [None, base_yml],
          model_name="gemma3-4b",
          use_m3_model=True,
          override_model_config=True,
          skip_jax_distributed_system=True,
          **bad,
      )
      with self.assertRaisesRegex(ValueError, "dropout_rate|mlp_activations"):
        Gemma3Config.from_config(bad_mt)
    for bad in (
        {"attention_dropout": 0.3},
        {"dropout": 0.2},
        {"dropout_rate": 0.1},
        {"dropout": 0.2, "attention_dropout": 0.0},
        {"mlp_activations": ["silu", "linear"]},
    ):
      bad_drop = process_config({"model_name": "gemma3-4b", "skip_jax_distributed_system": True, **bad})
      with self.assertRaisesRegex(ValueError, "dropout|mlp_activations"):
        Gemma3Config.from_config(bad_drop)

  def test_embedding_and_rmsnorm_against_hf_goldens_and_negctl(self) -> None:
    """Verifies scaled word embeddings, `.attend()`, and (1 + weight) RMSNorm + negative controls."""
    with np.load(_GOLDEN_NPZ) as data:
      for prefix, hidden_size in (("mini1b", 32), ("minigqa", 32)):
        cfg = Gemma3Config(
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=96,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            dtype="float32",
            weight_dtype="float32",
            matmul_precision="highest",
        )
        emb = Gemma3TextScaledWordEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            initializer_range=cfg.initializer_range,
            sharding=Gemma3Sharding.from_mesh(None, cfg),
            rngs=nnx.Rngs(0),
        )
        emb_w = np.asarray(data[f"{prefix}__param__model.embed_tokens.weight"])
        emb.weight.set_value(jnp.asarray(emb_w))
        input_ids = jnp.asarray(np.asarray(data[f"{prefix}__input_ids"]))
        emb_out = np.asarray(emb(input_ids))
        ref_emb = np.asarray(data[f"{prefix}__act__embed_tokens"])
        np.testing.assert_allclose(emb_out, ref_emb, atol=1e-6, rtol=1e-6)

        # Single-leaf embedding perturbation negative control under the exact 1e-6 predicate.
        pert_emb_w = emb_w.copy()
        pert_emb_w[int(np.asarray(input_ids)[0, 0]), 0] += 1e-3
        emb.weight.set_value(jnp.asarray(pert_emb_w))
        self.assertFalse(np.allclose(np.asarray(emb(input_ids)), ref_emb, atol=1e-6, rtol=1e-6))
        emb.weight.set_value(jnp.asarray(emb_w))

        # Tied vocabulary projection `.attend()` parity + single-leaf perturbation negative control.
        attend_logits = np.asarray(emb.attend(jnp.asarray(ref_emb), precision=jax.lax.Precision.HIGHEST))
        expected_attend = np.einsum("bte,ve->btv", ref_emb.astype(np.float64), emb_w.astype(np.float64)).astype(
            np.float32
        )
        np.testing.assert_allclose(attend_logits, expected_attend, atol=1e-5, rtol=1e-5)
        emb.weight.set_value(jnp.asarray(pert_emb_w))
        self.assertFalse(
            np.allclose(
                np.asarray(emb.attend(jnp.asarray(ref_emb), precision=jax.lax.Precision.HIGHEST)),
                expected_attend,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        emb.weight.set_value(jnp.asarray(emb_w))

        norm_kw = {"epsilon": cfg.rms_norm_eps, "param_dtype": cfg.weight_dtype, "name": "input_layernorm"}
        norm = Gemma3RMSNorm(cfg.hidden_size, sharding=Gemma3Sharding.from_mesh(None, cfg), **norm_kw)
        norm_w = np.asarray(data[f"{prefix}__param__model.layers.0.input_layernorm.weight"])
        norm.weight.set_value(jnp.asarray(norm_w))
        norm_out = np.asarray(norm(jnp.asarray(ref_emb)))
        ref_norm = np.asarray(data[f"{prefix}__act__layer0_input_layernorm"])
        np.testing.assert_allclose(norm_out, ref_norm, atol=1e-6, rtol=1e-6)
        # Single-leaf negative control under the same 1e-6 predicate.
        norm.weight.set_value(jnp.asarray(norm_w).at[0].add(1e-3))
        self.assertFalse(np.allclose(np.asarray(norm(jnp.asarray(ref_emb))), ref_norm, atol=1e-6, rtol=1e-6))
        norm.weight.set_value(jnp.asarray(norm_w))

        # Raw-HF parameter names keep legacy pre-offset `scale` / pre-scaled `embedding` checkpoints from
        # restoring into these modules by path.
        self.assertEqual(set(nnx.to_pure_dict(nnx.state(norm, nnx.Param))), {"weight"})
        self.assertEqual(set(nnx.to_pure_dict(nnx.state(emb, nnx.Param))), {"weight"})

        # Negative control: omitting the +1.0 offset (zeroing or shifting scale) breaks parity.
        norm_bad = Gemma3RMSNorm(cfg.hidden_size, sharding=Gemma3Sharding.from_mesh(None, cfg), **norm_kw)
        norm_bad.weight.set_value(jnp.asarray(norm_w - 1.0))
        bad_out = np.asarray(norm_bad(jnp.asarray(ref_emb)))
        self.assertGreater(float(np.max(np.abs(bad_out - ref_norm))), 0.5)

  def test_bfloat16_precision_boundaries(self) -> None:
    """Verifies bfloat16 precision boundaries in embeddings, `(1+scale)` RMSNorm, and attention."""
    # 1. Embedding scale boundary: `gathered.astype(bf16) * bf16(sqrt(features))` vs `f32` pre-multiply.
    emb = Gemma3TextScaledWordEmbedding(
        16,
        32,
        dtype="bfloat16",
        param_dtype="float32",
        initializer_range=0.02,
        sharding=Gemma3Sharding(),
        rngs=nnx.Rngs(5),
    )
    rng = np.random.default_rng(123)
    raw_table = rng.standard_normal((16, 32), dtype=np.float32)
    emb.weight.set_value(jnp.asarray(raw_table))
    ids_np = np.asarray([[0, 3, 7, 11]], dtype=np.int32)
    out_bf16 = emb(jnp.asarray(ids_np))
    self.assertEqual(out_bf16.dtype, jnp.bfloat16)
    gathered_bf16 = jnp.asarray(raw_table[ids_np], dtype=jnp.bfloat16)
    expected_bf16 = gathered_bf16 * jnp.asarray(32.0**0.5, dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out_bf16, dtype=np.float32), np.asarray(expected_bf16, dtype=np.float32))
    # Negative control: scaling in float32 before the bfloat16 cast is a different rounding boundary.
    f32_then_cast = jnp.asarray(raw_table[ids_np] * np.float32(32.0**0.5), dtype=jnp.bfloat16)
    self.assertFalse(np.array_equal(np.asarray(f32_then_cast, dtype=np.float32), np.asarray(out_bf16, dtype=np.float32)))
    attended_bf16 = emb.attend(out_bf16, precision=jax.lax.Precision.DEFAULT)
    self.assertEqual(attended_bf16.dtype, jnp.bfloat16)

    # 2. RMSNorm: float32 variance and `(1.0 + scale)` before the single cast back to bfloat16 (HF Gemma3RMSNorm).
    norm = Gemma3RMSNorm(32, epsilon=1e-6, param_dtype="bfloat16", sharding=Gemma3Sharding(), name="norm")
    scale_bf16 = jnp.asarray(rng.uniform(-0.05, 0.05, size=(32,)), dtype=jnp.bfloat16)
    norm.weight.set_value(scale_bf16)
    x_bf16 = jnp.asarray(rng.standard_normal((2, 8, 32)) * 3.0, dtype=jnp.bfloat16)
    norm_out = norm(x_bf16)
    self.assertEqual(norm_out.dtype, jnp.bfloat16)
    x64 = np.asarray(x_bf16, dtype=np.float64)
    s64 = np.asarray(scale_bf16, dtype=np.float64)
    ref64 = x64 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + 1e-6) * (1.0 + s64)
    np.testing.assert_allclose(np.asarray(norm_out, dtype=np.float64), ref64, rtol=2**-8, atol=0.0)
    # Negative control: a pure-bfloat16 norm (no float32 accumulation) differs from the module output.
    var_bf16 = jnp.mean(jnp.square(x_bf16), axis=-1, keepdims=True)
    pure_bf16 = x_bf16 * jax.lax.rsqrt(var_bf16 + jnp.asarray(1e-6, jnp.bfloat16)) * (1.0 + scale_bf16)
    self.assertFalse(np.array_equal(np.asarray(pure_bf16, dtype=np.float32), np.asarray(norm_out, dtype=np.float32)))

    # 3. Full Gemma3DecoderLayer in `dtype="bfloat16", matmul_precision="default"`.
    cfg_bf16 = Gemma3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dtype="bfloat16",
        weight_dtype="float32",
        matmul_precision="default",
    )
    cfg_f32 = Gemma3Config.from_config(cfg_bf16, dtype="float32", matmul_precision="highest")
    layer_bf16 = Gemma3DecoderLayer(cfg_bf16, layer_idx=0, rngs=nnx.Rngs(9))
    layer_f32 = Gemma3DecoderLayer(cfg_f32, layer_idx=0, rngs=nnx.Rngs(9))
    x_f32 = jnp.asarray(rng.standard_normal((2, 6, 32), dtype=np.float32))
    pos = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32)[None, :], (2, 6))
    y_bf16 = layer_bf16(x_f32.astype(jnp.bfloat16), decoder_positions=pos)
    y_f32 = layer_f32(x_f32, decoder_positions=pos)
    self.assertEqual(y_bf16.dtype, jnp.bfloat16)
    # bf16 keeps 8 mantissa bits: outputs near 2.5 have a 1.6e-2 ULP, and the layer chains about 6 bf16 roundings.
    np.testing.assert_allclose(
        np.asarray(y_bf16, dtype=np.float32),
        np.asarray(y_f32),
        atol=5e-2,
        rtol=2e-2,
    )

    # 4. The attention softmax always runs in float32 (HF `softmax(..., dtype=torch.float32)`), even in bf16.
    attn = Gemma3Attention(cfg_bf16, is_sliding=False, rngs=nnx.Rngs(21))
    jaxpr = jax.make_jaxpr(lambda x: attn(x, decoder_positions=pos))(x_f32.astype(jnp.bfloat16))
    exp_dtypes = _primitive_input_dtypes(jaxpr.jaxpr, "exp")
    self.assertTrue(exp_dtypes)
    self.assertEqual(set(exp_dtypes), {jnp.dtype(jnp.float32)})

  def test_mlp_attention_and_decoder_layer_against_hf_goldens_and_negctl(self) -> None:
    """Verifies GeGLU MLP, MQA/GQA local+global attention, and 4-norm layer vs HF goldens + negctls."""
    with np.load(_GOLDEN_NPZ) as data:
      cases = (
          (
              "mini1b",
              Gemma3Config(
                  vocab_size=64,
                  hidden_size=32,
                  intermediate_size=96,
                  num_hidden_layers=14,
                  num_attention_heads=4,
                  num_key_value_heads=1,
                  head_dim=8,
                  query_pre_attn_scalar=8.0,
                  sliding_window=4,
                  dtype="float32",
                  weight_dtype="float32",
                  matmul_precision="highest",
              ),
          ),
          (
              "minigqa",
              Gemma3Config(
                  vocab_size=64,
                  hidden_size=32,
                  intermediate_size=80,
                  num_hidden_layers=8,
                  num_attention_heads=4,
                  num_key_value_heads=2,
                  head_dim=8,
                  query_pre_attn_scalar=12.0,
                  sliding_window=4,
                  rope_linear_scaling_factor=8.0,
                  final_logit_softcapping=30.0,
                  tie_word_embeddings=False,
                  dtype="float32",
                  weight_dtype="float32",
                  matmul_precision="highest",
              ),
          ),
      )

      for prefix, cfg in cases:
        layer0 = Gemma3DecoderLayer(cfg, layer_idx=0, rngs=nnx.Rngs(0))
        _load_layer_from_golden(layer0, data, prefix, 0)

        emb_in = jnp.asarray(np.asarray(data[f"{prefix}__act__embed_tokens"]))
        pos_ids = jnp.asarray(np.asarray(data[f"{prefix}__position_ids"]))

        # Sub-component checks on layer 0 + single-leaf attention perturbation negative control.
        norm_in = layer0.pre_self_attention_norm(emb_in)
        ref_l0_attn = np.asarray(data[f"{prefix}__act__layer0_self_attn"])
        attn_out = np.asarray(layer0.self_attention(norm_in, decoder_positions=pos_ids))
        np.testing.assert_allclose(
            attn_out,
            ref_l0_attn,
            atol=1e-5,
            rtol=1e-5,
        )
        # Value kernel: query/key scaling is erased by QK-norm, so perturb one value-kernel element instead.
        orig_v0 = layer0.self_attention.value.kernel.get_value()
        layer0.self_attention.value.kernel.set_value(orig_v0.at[0, 0, 0].add(1e-2))
        self.assertFalse(
            np.allclose(
                np.asarray(layer0.self_attention(norm_in, decoder_positions=pos_ids)),
                ref_l0_attn,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        layer0.self_attention.value.kernel.set_value(orig_v0)

        ffw_in = jnp.asarray(np.asarray(data[f"{prefix}__act__layer0_pre_feedforward_layernorm"]))
        mlp_out = np.asarray(layer0.mlp(ffw_in))
        np.testing.assert_allclose(
            mlp_out,
            np.asarray(data[f"{prefix}__act__layer0_mlp"]),
            atol=1e-5,
            rtol=1e-5,
        )
        # Standalone Gemma3MLP + perturbation negative control.
        standalone_mlp = Gemma3MLP(cfg, rngs=nnx.Rngs(4))
        standalone_mlp.wi_0.kernel.set_value(layer0.mlp.wi_0.kernel.get_value())
        standalone_mlp.wi_1.kernel.set_value(layer0.mlp.wi_1.kernel.get_value())
        standalone_mlp.wo.kernel.set_value(layer0.mlp.wo.kernel.get_value())
        np.testing.assert_allclose(np.asarray(standalone_mlp(ffw_in)), mlp_out, atol=1e-6, rtol=1e-6)
        standalone_mlp.wi_0.kernel.set_value(layer0.mlp.wi_0.kernel.get_value().at[0, 0].add(1e-3))
        self.assertFalse(np.allclose(np.asarray(standalone_mlp(ffw_in)), mlp_out, atol=1e-6, rtol=1e-6))

        # Full 4-RMSNorm layer 0 output + exclusive sliding_window=4 boundary check.
        l0_out = np.asarray(layer0(emb_in, decoder_positions=pos_ids))
        ref_l0 = np.asarray(data[f"{prefix}__act__layer0_out"])
        np.testing.assert_allclose(l0_out, ref_l0, atol=1e-5, rtol=1e-5)
        orig_post = layer0.post_ffw_norm.weight.get_value()
        layer0.post_ffw_norm.weight.set_value(orig_post.at[0].add(1e-3))
        self.assertFalse(np.allclose(np.asarray(layer0(emb_in, decoder_positions=pos_ids)), ref_l0, atol=1e-5, rtol=1e-5))
        layer0.post_ffw_norm.weight.set_value(orig_post)

        emb_mod = Gemma3TextScaledWordEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            initializer_range=cfg.initializer_range,
            sharding=Gemma3Sharding.from_mesh(None, cfg),
            rngs=nnx.Rngs(1),
        )
        emb_mod.weight.set_value(jnp.asarray(np.asarray(data[f"{prefix}__param__model.embed_tokens.weight"])))
        pert_ids = jnp.asarray(np.asarray(data[f"{prefix}__pert_tok0_input_ids"]))
        pert_l0 = np.asarray(layer0(emb_mod(pert_ids), decoder_positions=pos_ids))
        np.testing.assert_allclose(
            pert_l0,
            np.asarray(data[f"{prefix}__pert_tok0_layer0_out"]),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_array_equal(pert_l0[:, 4:, :], l0_out[:, 4:, :])

        # Global attention layer 5 vs float64 NumPy oracle + negative controls.
        layer5 = Gemma3DecoderLayer(cfg, layer_idx=5, rngs=nnx.Rngs(2))
        _load_layer_from_golden(layer5, data, prefix, 5)
        self.assertFalse(layer5.is_sliding)
        l5_attn = np.asarray(layer5.self_attention(norm_in, decoder_positions=pos_ids))
        f64_ref = _numpy_f64_attention_oracle(
            np.asarray(norm_in),
            np.asarray(pos_ids),
            q_kernel=np.asarray(layer5.self_attention.query.kernel.get_value()),
            k_kernel=np.asarray(layer5.self_attention.key.kernel.get_value()),
            v_kernel=np.asarray(layer5.self_attention.value.kernel.get_value()),
            o_kernel=np.asarray(layer5.self_attention.out.kernel.get_value()),
            q_norm_w=np.asarray(layer5.self_attention.query_norm.weight.get_value()),
            k_norm_w=np.asarray(layer5.self_attention.key_norm.weight.get_value()),
            eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            rope_factor=cfg.rope_linear_scaling_factor,
            query_pre_attn_scalar=cfg.query_pre_attn_scalar,
            sliding_window=None,
        )
        np.testing.assert_allclose(l5_attn, f64_ref, atol=1e-5, rtol=1e-5)

        # Negative controls: single-leaf perturbation under 1e-5 predicate AND wrong sliding/RoPE mode.
        orig_qn5 = layer5.self_attention.query_norm.weight.get_value()
        layer5.self_attention.query_norm.weight.set_value(orig_qn5 + 0.25)
        self.assertFalse(
            np.allclose(
                np.asarray(layer5.self_attention(norm_in, decoder_positions=pos_ids)),
                f64_ref,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        layer5.self_attention.query_norm.weight.set_value(orig_qn5)

        wrong_sliding = Gemma3Attention(cfg, is_sliding=True, rngs=nnx.Rngs(3))
        wrong_sliding.query.kernel.set_value(layer5.self_attention.query.kernel.get_value())
        wrong_sliding.key.kernel.set_value(layer5.self_attention.key.kernel.get_value())
        wrong_sliding.value.kernel.set_value(layer5.self_attention.value.kernel.get_value())
        wrong_sliding.out.kernel.set_value(layer5.self_attention.out.kernel.get_value())
        wrong_sliding.query_norm.weight.set_value(layer5.self_attention.query_norm.weight.get_value())
        wrong_sliding.key_norm.weight.set_value(layer5.self_attention.key_norm.weight.get_value())
        wrong_out = np.asarray(wrong_sliding(norm_in, decoder_positions=pos_ids))
        self.assertGreater(float(np.max(np.abs(wrong_out - f64_ref))), 1e-3)

  def test_packed_segment_ids_and_padding_isolation_forward_and_backward(self) -> None:
    """Verifies packed document and padding isolation in both forward and backward passes."""
    cfg = Gemma3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
    )
    rng = np.random.default_rng(42)
    x_base = jnp.asarray(rng.standard_normal((2, 8, 32), dtype=np.float32))
    seg_ids = jnp.asarray(
        [[1, 1, 1, 0, 2, 2, 2, 0], [1, 1, 0, 0, 2, 2, 2, 2]],
        dtype=jnp.int32,
    )
    positions = jnp.asarray(
        [[0, 1, 2, 0, 0, 1, 2, 0], [0, 1, 0, 0, 0, 1, 2, 3]],
        dtype=jnp.int32,
    )

    for is_sliding in (True, False):
      layer = Gemma3DecoderLayer(
          cfg,
          layer_idx=0 if is_sliding else 5,
          rngs=nnx.Rngs(7),
      )
      self.assertEqual(layer.is_sliding, is_sliding)
      # Padding query rows (segment 0) attend to nothing, so attention output there is exactly zero.
      attn_rows = np.asarray(layer.self_attention(x_base, decoder_positions=positions, decoder_segment_ids=seg_ids))
      pad_rows = np.asarray(seg_ids == 0)
      np.testing.assert_array_equal(attn_rows[pad_rows], np.zeros_like(attn_rows[pad_rows]))
      out_clean = np.asarray(layer(x_base, decoder_positions=positions, decoder_segment_ids=seg_ids))

      # Perturb doc 1 (seg == 1) and pad (seg == 0) heavily; doc 2 (seg == 2) must stay identical.
      noise = jnp.asarray(rng.standard_normal((2, 8, 32), dtype=np.float32) * 5.0)
      x_pert = jnp.where((seg_ids != 2)[..., None], x_base + noise, x_base)
      out_pert = np.asarray(layer(x_pert, decoder_positions=positions, decoder_segment_ids=seg_ids))
      doc2_mask = np.asarray(seg_ids == 2)
      np.testing.assert_array_equal(out_pert[doc2_mask], out_clean[doc2_mask])

      # Negative control: omitting decoder_segment_ids leaks doc 1 / pad into doc 2.
      unmasked_clean = np.asarray(layer(x_base, decoder_positions=positions))
      unmasked_pert = np.asarray(layer(x_pert, decoder_positions=positions))
      self.assertGreater(float(np.max(np.abs(unmasked_pert[doc2_mask] - unmasked_clean[doc2_mask]))), 1e-3)

      # Backward isolation: loss on doc 2 must produce identically zero gradient on doc 1 and pad.
      def _doc2_loss(inp: jax.Array, mod: Gemma3DecoderLayer = layer) -> jax.Array:
        y = mod(inp, decoder_positions=positions, decoder_segment_ids=seg_ids)
        w = (seg_ids == 2).astype(jnp.float32)[..., None]
        return jnp.sum(jnp.square(y) * w)

      grad_x = np.asarray(jax.grad(_doc2_loss)(x_base))
      non_doc2 = np.asarray(seg_ids != 2)
      np.testing.assert_array_equal(grad_x[non_doc2], np.zeros_like(grad_x[non_doc2]))
      self.assertGreater(float(np.linalg.norm(grad_x[doc2_mask])), 1e-3)

  def test_masks_use_token_indices_not_positions(self) -> None:
    """Causality and the sliding window follow token order even when positions restart (no segment IDs)."""
    cfg = Gemma3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
    )
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal((1, 8, 32), dtype=np.float32))
    restart = jnp.asarray([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=jnp.int32)
    x_pert = x.at[0, 5].add(3.0)
    for is_sliding in (True, False):
      attn = Gemma3Attention(cfg, is_sliding=is_sliding, rngs=nnx.Rngs(8))
      base = np.asarray(attn(x, decoder_positions=restart))
      pert = np.asarray(attn(x_pert, decoder_positions=restart))
      # Token 1 precedes token 5 in index order, so it must not see the change (a positions mask would leak it).
      np.testing.assert_array_equal(pert[0, :5], base[0, :5])
      self.assertGreater(float(np.abs(pert[0, 5:] - base[0, 5:]).max()), 1e-3)

  def test_multidevice_sharding_mqa_and_gqa_jit_grad_parity(self) -> None:
    """Verifies NamedSharding placement and 1-device vs (fsdp=2, tensor=2) JIT grad parity."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 JAX devices for (fsdp=2, tensor=2) test, got {len(devices)}")

    mesh_1d = Mesh(np.asarray(devices[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "tensor", "expert"))
    mesh_4d = Mesh(np.asarray(devices[:4]).reshape(1, 2, 2, 1), ("data", "fsdp", "tensor", "expert"))

    for num_kv_heads, expected_kv_axis in ((1, None), (2, "tensor")):
      cfg = Gemma3Config(
          vocab_size=64,
          hidden_size=32,
          intermediate_size=64,
          num_hidden_layers=6,
          num_attention_heads=4,
          num_key_value_heads=num_kv_heads,
          head_dim=8,
          dtype="float32",
          weight_dtype="float32",
          matmul_precision="highest",
      )
      with jax.set_mesh(mesh_1d):
        s1 = Gemma3Sharding.from_mesh(mesh_1d, cfg)
        layer_1d = Gemma3DecoderLayer(cfg, layer_idx=0, rngs=nnx.Rngs(11), sharding=s1)
      with jax.set_mesh(mesh_4d):
        s4 = Gemma3Sharding.from_mesh(mesh_4d, cfg)
        layer_4d = Gemma3DecoderLayer(cfg, layer_idx=0, rngs=nnx.Rngs(11), sharding=s4)

      key_sharding = layer_4d.self_attention.key.kernel.get_value().sharding
      self.assertIsInstance(key_sharding, NamedSharding)
      self.assertEqual(key_sharding.spec, P(("fsdp", "expert"), expected_kv_axis, None))
      q_sharding = layer_4d.self_attention.query.kernel.get_value().sharding
      self.assertEqual(q_sharding.spec, P(("fsdp", "expert"), "tensor", None))

      rng = np.random.default_rng(99)
      x_np = rng.standard_normal((2, 8, 32), dtype=np.float32)
      pos_np = np.broadcast_to(np.arange(8, dtype=np.int32)[None, :], (2, 8))

      @nnx.jit
      def _step(mod: Gemma3DecoderLayer, x: jax.Array, pos: jax.Array):
        def _loss_fn(m: Gemma3DecoderLayer):
          return jnp.mean(jnp.square(m(x, decoder_positions=pos)))

        return nnx.value_and_grad(_loss_fn)(mod)

      with jax.set_mesh(mesh_1d):
        loss_1d, grads_1d = _step(layer_1d, jnp.asarray(x_np), jnp.asarray(pos_np))
      with jax.set_mesh(mesh_4d):
        x_4d = jax.device_put(jnp.asarray(x_np), NamedSharding(mesh_4d, P(("data", "fsdp", "expert"), None, None)))
        pos_4d = jax.device_put(jnp.asarray(pos_np), NamedSharding(mesh_4d, P(("data", "fsdp", "expert"), None)))
        loss_4d, grads_4d = _step(layer_4d, x_4d, pos_4d)

      np.testing.assert_allclose(np.asarray(loss_4d), np.asarray(loss_1d), atol=1e-6, rtol=1e-6)
      leaves_1d = jax.tree_util.tree_leaves(nnx.to_pure_dict(grads_1d))
      leaves_4d = jax.tree_util.tree_leaves(nnx.to_pure_dict(grads_4d))
      for g1, g4 in zip(leaves_1d, leaves_4d, strict=True):
        np.testing.assert_allclose(np.asarray(g4), np.asarray(g1), atol=1e-5, rtol=1e-5)

    # Also verify `param_metadata(..., eager_sharding=True)` under `nnx.use_eager_sharding(False)`
    # AND default `sharding=None` auto-replicating `kv_heads=2` across a `tensor=4` ambient mesh.
    mesh_tp4 = Mesh(np.asarray(devices[:4]).reshape(1, 1, 4, 1), ("data", "fsdp", "tensor", "expert"))
    cfg_kv2 = Gemma3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dtype="float32",
        weight_dtype="float32",
    )
    with nnx.use_eager_sharding(False), jax.set_mesh(mesh_tp4):
      layer_tp4 = Gemma3DecoderLayer(cfg_kv2, layer_idx=0, rngs=nnx.Rngs(15), sharding=None)
      emb_tp4 = Gemma3TextScaledWordEmbedding(
          64,
          32,
          dtype="float32",
          param_dtype="float32",
          initializer_range=0.02,
          sharding=Gemma3Sharding.from_mesh(None, cfg_kv2),
          rngs=nnx.Rngs(15),
      )
      self.assertEqual(emb_tp4.weight.get_value().sharding.spec, P("tensor", ("fsdp", "expert")))
      self.assertEqual(
          layer_tp4.self_attention.query.kernel.get_value().sharding.spec,
          P(("fsdp", "expert"), "tensor", None),
      )
      self.assertEqual(
          layer_tp4.self_attention.key.kernel.get_value().sharding.spec,
          P(("fsdp", "expert"), None, None),
      )
      self.assertEqual(layer_tp4.pre_self_attention_norm.weight.get_value().sharding.spec, P(None))
      self.assertTrue(layer_tp4.pre_self_attention_norm.weight.get_metadata()["preserve_unsharded"])

    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())):
      unsharded_layer = Gemma3DecoderLayer(cfg_kv2, layer_idx=0, rngs=nnx.Rngs(16), sharding=None)
      self.assertIsNone(unsharded_layer.sharding)
      unsharded_emb = Gemma3TextScaledWordEmbedding(
          64,
          32,
          dtype="float32",
          param_dtype="float32",
          initializer_range=0.02,
          sharding=None,
          rngs=nnx.Rngs(16),
      )
      out_nomesh = unsharded_layer(unsharded_emb(jnp.ones((2, 4), dtype=jnp.int32)))
      self.assertEqual(out_nomesh.shape, (2, 4, 32))

    explicit_mesh = Mesh(
        np.asarray(devices[:4]).reshape(1, 2, 2, 1),
        ("data", "fsdp", "tensor", "expert"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    with self.assertRaisesRegex(NotImplementedError, "AxisType.Auto"):
      Gemma3Sharding.from_mesh(explicit_mesh, cfg_kv2)
    with jax.set_mesh(explicit_mesh):
      with self.assertRaisesRegex(NotImplementedError, "AxisType.Auto"):
        Gemma3Sharding.from_mesh(None, cfg_kv2)
      with self.assertRaisesRegex(NotImplementedError, "AxisType.Auto"):
        Gemma3DecoderLayer(cfg_kv2, layer_idx=0, rngs=nnx.Rngs(17), sharding=None)

    abs_topo = jax.sharding.AbstractMesh((1, 64, 2, 1), ("data", "fsdp", "tensor", "expert"))
    self.assertFalse(Gemma3Sharding.from_mesh(abs_topo, Gemma3Config.preset("gemma3-4b")).replicate_kv_heads)

  @unittest.skipUnless(
      HAS_TORCH and HAS_TRANSFORMERS,
      "Requires torch and transformers for live HF Gemma3DecoderLayer VJP check.",
  )
  def test_live_hf_decoder_layer_forward_and_vjp_parity(self) -> None:
    """Verifies live forward and all-leaf backward VJP parity against HF Gemma3DecoderLayer."""
    import torch  # pylint: disable=import-outside-toplevel
    from transformers.models.gemma3 import Gemma3ForCausalLM  # pylint: disable=import-outside-toplevel
    from transformers.models.gemma3 import Gemma3TextConfig  # pylint: disable=import-outside-toplevel

    for layer_idx, n_kv, factor, q_scalar in (
        (0, 1, 1.0, 8.0),
        (5, 2, 8.0, 12.0),
    ):
      hf_cfg = Gemma3TextConfig(
          vocab_size=64,
          hidden_size=32,
          intermediate_size=80,
          num_hidden_layers=6,
          num_attention_heads=4,
          num_key_value_heads=n_kv,
          head_dim=8,
          query_pre_attn_scalar=int(q_scalar),
          sliding_window=4,
          sliding_window_pattern=6,
          rope_theta=1_000_000.0,
          rope_local_base_freq=10_000.0,
          rope_scaling={"rope_type": "linear", "factor": factor} if factor != 1.0 else None,
          rms_norm_eps=1e-6,
          hidden_activation="gelu_pytorch_tanh",
          attention_bias=False,
          attention_dropout=0.0,
          _attn_implementation="eager",
      )
      hf_model = Gemma3ForCausalLM(hf_cfg).to(dtype=torch.float32)
      hf_model.eval()
      gen = torch.Generator(device="cpu").manual_seed(2026 + layer_idx)
      with torch.no_grad():
        for p in hf_model.parameters():
          p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * (0.15 if p.ndim == 1 else 0.05))

      hf_layer = hf_model.model.layers[layer_idx]
      x_pt = (torch.randn((2, 10, 32), generator=gen, dtype=torch.float32) * 0.5).requires_grad_(True)
      dy_pt = torch.randn((2, 10, 32), generator=gen, dtype=torch.float32)
      pos_pt = torch.arange(10, dtype=torch.long).unsqueeze(0).expand(2, 10)
      pos_emb_global = hf_model.model.rotary_emb(x_pt, pos_pt)
      pos_emb_local = hf_model.model.rotary_emb_local(x_pt, pos_pt)

      q_idx = torch.arange(10)[:, None]
      k_idx = torch.arange(10)[None, :]
      delta = q_idx - k_idx
      valid = (delta >= 0) & (delta < 4) if layer_idx == 0 else (delta >= 0)
      mask_pt = torch.where(valid, 0.0, torch.finfo(torch.float32).min)[None, None, :, :].expand(2, 1, 10, 10)

      hf_model.zero_grad(set_to_none=True)
      hf_out = hf_layer(
          x_pt,
          position_embeddings_global=pos_emb_global,
          position_embeddings_local=pos_emb_local,
          attention_mask=mask_pt,
          position_ids=pos_pt,
      )[0]
      (hf_out * dy_pt).mean().backward()

      m3_cfg = Gemma3Config(
          vocab_size=64,
          hidden_size=32,
          intermediate_size=80,
          num_hidden_layers=6,
          num_attention_heads=4,
          num_key_value_heads=n_kv,
          head_dim=8,
          query_pre_attn_scalar=q_scalar,
          sliding_window=4,
          rope_linear_scaling_factor=factor,
          dtype="float32",
          weight_dtype="float32",
          matmul_precision="highest",
      )
      m3_layer = Gemma3DecoderLayer(m3_cfg, layer_idx=layer_idx, rngs=nnx.Rngs(0))
      sd = hf_layer.state_dict()
      m3_layer.pre_self_attention_norm.weight.set_value(jnp.asarray(sd["input_layernorm.weight"].numpy()))
      m3_layer.post_self_attention_norm.weight.set_value(jnp.asarray(sd["post_attention_layernorm.weight"].numpy()))
      m3_layer.pre_ffw_norm.weight.set_value(jnp.asarray(sd["pre_feedforward_layernorm.weight"].numpy()))
      m3_layer.post_ffw_norm.weight.set_value(jnp.asarray(sd["post_feedforward_layernorm.weight"].numpy()))
      m3_layer.self_attention.query_norm.weight.set_value(jnp.asarray(sd["self_attn.q_norm.weight"].numpy()))
      m3_layer.self_attention.key_norm.weight.set_value(jnp.asarray(sd["self_attn.k_norm.weight"].numpy()))
      m3_layer.self_attention.query.kernel.set_value(
          jnp.asarray(sd["self_attn.q_proj.weight"].numpy().T.reshape(32, 4, 8))
      )
      m3_layer.self_attention.key.kernel.set_value(
          jnp.asarray(sd["self_attn.k_proj.weight"].numpy().T.reshape(32, n_kv, 8))
      )
      m3_layer.self_attention.value.kernel.set_value(
          jnp.asarray(sd["self_attn.v_proj.weight"].numpy().T.reshape(32, n_kv, 8))
      )
      m3_layer.self_attention.out.kernel.set_value(jnp.asarray(sd["self_attn.o_proj.weight"].numpy().T.reshape(4, 8, 32)))
      m3_layer.mlp.wi_0.kernel.set_value(jnp.asarray(sd["mlp.gate_proj.weight"].numpy().T))
      m3_layer.mlp.wi_1.kernel.set_value(jnp.asarray(sd["mlp.up_proj.weight"].numpy().T))
      m3_layer.mlp.wo.kernel.set_value(jnp.asarray(sd["mlp.down_proj.weight"].numpy().T))

      x_jax = jnp.asarray(x_pt.detach().numpy())
      dy_jax = jnp.asarray(dy_pt.numpy())
      pos_jax = jnp.asarray(pos_pt.numpy(), dtype=jnp.int32)

      @nnx.jit
      def _fwd_vjp(mod: Gemma3DecoderLayer, inp: jax.Array, pos: jax.Array, cot: jax.Array):
        def _loss(m: Gemma3DecoderLayer, x_in: jax.Array):
          y = m(x_in, decoder_positions=pos)
          return jnp.mean(y * cot), y

        ((_, y_out), (g_mod, g_x)) = nnx.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(mod, inp)
        return y_out, g_mod, g_x

      y_jax, g_m3, gx_jax = _fwd_vjp(m3_layer, x_jax, pos_jax, dy_jax)
      np.testing.assert_allclose(np.asarray(y_jax), hf_out.detach().numpy(), atol=1e-5, rtol=1e-5)
      assert x_pt.grad is not None
      np.testing.assert_allclose(np.asarray(gx_jax), x_pt.grad.numpy(), atol=1e-5, rtol=1e-5)

      grad_pairs = (
          (g_m3.pre_self_attention_norm.weight.get_value(), hf_layer.input_layernorm.weight.grad),
          (g_m3.post_self_attention_norm.weight.get_value(), hf_layer.post_attention_layernorm.weight.grad),
          (g_m3.pre_ffw_norm.weight.get_value(), hf_layer.pre_feedforward_layernorm.weight.grad),
          (g_m3.post_ffw_norm.weight.get_value(), hf_layer.post_feedforward_layernorm.weight.grad),
          (g_m3.self_attention.query_norm.weight.get_value(), hf_layer.self_attn.q_norm.weight.grad),
          (g_m3.self_attention.key_norm.weight.get_value(), hf_layer.self_attn.k_norm.weight.grad),
          (g_m3.mlp.wi_0.kernel.get_value(), hf_layer.mlp.gate_proj.weight.grad.T),
          (g_m3.mlp.wi_1.kernel.get_value(), hf_layer.mlp.up_proj.weight.grad.T),
          (g_m3.mlp.wo.kernel.get_value(), hf_layer.mlp.down_proj.weight.grad.T),
          (
              g_m3.self_attention.query.kernel.get_value().reshape(32, 32),
              hf_layer.self_attn.q_proj.weight.grad.T,
          ),
          (
              g_m3.self_attention.key.kernel.get_value().reshape(32, n_kv * 8),
              hf_layer.self_attn.k_proj.weight.grad.T,
          ),
          (
              g_m3.self_attention.value.kernel.get_value().reshape(32, n_kv * 8),
              hf_layer.self_attn.v_proj.weight.grad.T,
          ),
          (
              g_m3.self_attention.out.kernel.get_value().reshape(32, 32),
              hf_layer.self_attn.o_proj.weight.grad.T,
          ),
      )
      for g_jax_leaf, g_pt_leaf in grad_pairs:
        self.assertIsNotNone(g_pt_leaf)
        np.testing.assert_allclose(np.asarray(g_jax_leaf), g_pt_leaf.numpy(), atol=1e-5, rtol=1e-5)

      # Negative control: one perturbed post-FFW norm entry breaks forward and gradient parity at 1e-5.
      orig_post = m3_layer.post_ffw_norm.weight.get_value()
      m3_layer.post_ffw_norm.weight.set_value(orig_post.at[0].add(0.1))
      y_bad, g_bad, _ = _fwd_vjp(m3_layer, x_jax, pos_jax, dy_jax)
      self.assertFalse(np.allclose(np.asarray(y_bad), hf_out.detach().numpy(), atol=1e-5, rtol=1e-5))
      self.assertFalse(
          np.allclose(
              np.asarray(g_bad.pre_ffw_norm.weight.get_value()),
              hf_layer.pre_feedforward_layernorm.weight.grad.numpy(),
              atol=1e-5,
              rtol=1e-5,
          )
      )


if __name__ == "__main__":
  unittest.main()
