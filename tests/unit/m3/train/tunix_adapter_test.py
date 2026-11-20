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

"""Unit tests for M3TunixAdapter across all 5 m3 causal LM architectures."""

from __future__ import annotations

import dataclasses
import inspect
import os
import pathlib
import subprocess
import sys
from typing import Any
import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax

from maxtext.m3.configs import config as m3_config
from maxtext.m3.configs import glm5_3_flash_config
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from maxtext.m3.models.granitemoe import modeling_granitemoe
from maxtext.m3.models.qwen3 import modeling_qwen3
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
from maxtext.m3.models.qwen3_vl import modeling_qwen3_vl
from maxtext.m3.train import tunix_adapter
from maxtext.m3.train.tunix_adapter import M3RolloutCache
from maxtext.m3.train.tunix_adapter import M3TunixAdapter
from maxtext.m3.train.tunix_adapter import RolloutCacheSpec
from maxtext.m3.train.tunix_adapter import segment_ids_from_attention_mask
from maxtext.m3.train.tunix_adapter import wrap_m3_for_tunix


_PAD_ID = 0


def _tunix_causal_mask(input_mask: jax.Array) -> jax.Array:
  """Builds Tunix's `[B, L, L]` causal key-validity mask `input_mask[:, None, :] & tril`."""
  seq_len = input_mask.shape[-1]
  tril = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  return jnp.asarray(input_mask, dtype=jnp.bool_)[:, None, :] & tril[None, :, :]


def _cpu_mesh() -> Mesh:
  """Builds a 4-axis m3 mesh over available host CPU devices."""
  devs = jax.devices()
  n = len(devs)
  if n >= 4:
    arr = np.asarray(devs[:4]).reshape(1, 2, 2, 1)
  elif n >= 2:
    arr = np.asarray(devs[:2]).reshape(1, 2, 1, 1)
  else:
    arr = np.asarray(devs[:1]).reshape(1, 1, 1, 1)
  return Mesh(arr, ("data", "fsdp", "tensor", "expert"))


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Qwen3MiniConfig:
  """Minimal typed Qwen3 config matching `modeling_qwen3.py` and `m3/infra/scan.py`."""

  model_name: str = "qwen3-0.6b"
  vocab_size: int = 64
  emb_dim: int = 32
  mlp_dim: int = 64
  num_decoder_layers: int = 2
  num_query_heads: int = 2
  num_kv_heads: int = 2
  head_dim: int = 16
  normalization_layer_epsilon: float = 1e-6
  rope_max_timescale: float = 10000.0
  use_qk_norm: bool = True
  float32_logits: bool = True
  logits_dot_in_fp32: bool = True
  normalize_embedding_logits: bool = False
  final_logits_soft_cap: float | None = None
  cast_logits_to_fp32: bool = True
  logits_via_embedding: bool = True
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  scan_layers: bool = False
  param_scan_axis: int = 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Qwen3_8MiniConfig:
  """Minimal typed Qwen3.8-Flash-Next config for fast fp32 CPU unit testing."""

  model_name: str = "qwen3.8-flash-next-mini"
  emb_dim: int = 32
  num_decoder_layers: int = 4
  vocab_size: int = 64
  eos_token_id: int = 63
  ple_eos_token_id: int = 63
  hc_count: int = 4
  hc_lowrank: int = 8
  scan_layers: bool = False
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  normalization_layer_epsilon: float = 1e-6
  inhomogeneous_layer_cycle_interval: int = 4
  gdn_num_key_heads: int = 2
  gdn_num_value_heads: int = 4
  gdn_key_head_dim: int = 16
  gdn_value_head_dim: int = 16
  gdn_conv_kernel_dim: int = 4
  gdn_chunk_size: int = 16
  num_query_heads: int = 4
  num_kv_heads: int = 1
  head_dim: int = 16
  partial_rotary_factor: float = 0.25
  rope_max_timescale: float = 1e4
  mrope_section: tuple[int, ...] = (1, 1, 0)
  index_n_heads: int = 2
  index_kv_heads: int = 1
  index_head_dim: int = 16
  index_select_budget: int = 16
  index_select_block_size: int = 4
  moe_mlp_dim: int = 32
  mlp_dim: int = 32
  num_experts: int = 4
  num_experts_per_tok: int = 2
  norm_topk_prob: bool = True
  routed_scaling_factor: float = 1.0
  use_ple_layer: bool = False
  heads_per_ngram: int = 2
  ple_seed: int = 0
  ple_kernel_size: int = 3
  split_ngram_parts: int = 2
  ple_hash_seed: int = 0
  ngram_vocab_size_base: int = 64
  mtp_num_layers: int = 0
  mtp_num_streams: int = 4
  mtp_loss_scaling_factor: float = 0.1
  load_balance_loss_weight: float = 0.001
  gdn_output_gate_activation: str = "silu"
  gdn_chunk_inverse: str = "block_doubling"
  qsa_indexer_mask: str = "gather"
  qsa_attention_kernel: str = "dense"
  ple_layer_ids: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Qwen3VLMiniConfig:
  """Minimal typed Qwen3-VL multimodal config for fast fp32 CPU unit testing."""

  model_name: str = "qwen3-vl-2b"
  use_multimodal: bool = True
  scan_layers: bool = False
  vocab_size: int = 64
  emb_dim: int = 32
  mlp_dim: int = 64
  num_decoder_layers: int = 2
  num_query_heads: int = 2
  num_kv_heads: int = 2
  head_dim: int = 16
  normalization_layer_epsilon: float = 1e-6
  rope_max_timescale: float = 10000.0
  use_qk_norm: bool = True
  float32_logits: bool = True
  logits_dot_in_fp32: bool = True
  normalize_embedding_logits: bool = False
  final_logits_soft_cap: float | None = None
  cast_logits_to_fp32: bool = True
  logits_via_embedding: bool = True
  mrope_section: tuple[int, ...] = (2, 3, 3)
  hidden_size_for_vit: int = 32
  intermediate_size_for_vit: int = 64
  num_attention_heads_for_vit: int = 2
  num_channels_for_vit: int = 3
  num_hidden_layers_for_vit: int = 2
  num_position_embeddings_for_vit: int = 16
  out_hidden_size_for_vit: int = 32
  patch_size_for_vit: int = 16
  rope_theta_for_vit: float = 10000.0
  spatial_merge_size_for_vit: int = 2
  temporal_patch_size_for_vit: int = 2
  deepstack_visual_indexes_for_vit: tuple[int, ...] = (1,)
  dtype: Any = jnp.float32
  dtype_mm: Any = jnp.float32
  weight_dtype: Any = jnp.float32


def _build_qwen3_mini(
    *,
    scan_layers: bool,
    logits_via_embedding: bool = True,
    mesh: Mesh | None = None,
) -> modeling_qwen3.Qwen3Model:
  """Constructs a deterministic fp32 mini dense Qwen3Model."""
  cfg = _Qwen3MiniConfig(
      scan_layers=scan_layers,
      logits_via_embedding=logits_via_embedding,
      param_scan_axis=1,
  )
  rngs = nnx.Rngs(params=0)
  active_mesh = mesh if mesh is not None else _cpu_mesh()
  with jax.set_mesh(active_mesh):
    model = modeling_qwen3.create_qwen3_model(cfg, mesh=active_mesh, rngs=rngs)
    model = m3_scan.maybe_apply_scan(model, cfg)
  return model


def _build_granitemoe_mini(
    *,
    scan_layers: bool,
    tie_word_embeddings: bool = True,
    logits_scaling: float = 2.0,
    mesh: Mesh | None = None,
) -> modeling_granitemoe.GraniteMoeModel:
  """Constructs a deterministic fp32 mini MoE GraniteMoeModel."""
  cfg = modeling_granitemoe.GraniteMoeConfig(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=48,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      num_local_experts=4,
      num_experts_per_tok=2,
      rms_norm_eps=1e-5,
      rope_theta=10000.0,
      embedding_multiplier=1.5,
      logits_scaling=logits_scaling,
      residual_multiplier=0.5,
      attention_multiplier=0.125,
      tie_word_embeddings=tie_word_embeddings,
      scan_layers=scan_layers,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
  )
  rngs = nnx.Rngs(params=7)
  if mesh is not None:
    with jax.set_mesh(mesh):
      return modeling_granitemoe.GraniteMoeModel(cfg, mesh=mesh, rngs=rngs)
  return modeling_granitemoe.GraniteMoeModel(cfg, mesh=None, rngs=rngs)


def _build_glm5_mini(*, scan_layers: bool) -> modeling_glm5_3_flash.Glm5Model:
  """Constructs a deterministic fp32 mini hybrid MoE Glm5Model for CPU execution."""
  cfg = glm5_3_flash_config.Glm5Config.mini_config(
      vocab_size=64,
      hidden_size=32,
      num_hidden_layers=5,
      first_k_dense_replace=1,
      intermediate_size=64,
      moe_intermediate_size=32,
      n_routed_experts=4,
      num_experts_per_tok=2,
      q_lora_rank=16,
      kv_lora_rank=16,
      num_attention_heads=2,
      num_key_value_heads=2,
      qk_nope_head_dim=16,
      v_head_dim=16,
      use_indexer=False,
      index_n_heads=2,
      index_head_dim=16,
      index_topk=4,
      index_kpool=2,
      linear_num_key_heads=2,
      linear_num_value_heads=2,
      linear_key_head_dim=16,
      linear_value_head_dim=16,
      mhc_expansion_rate=2,
      mhc_sinkhorn_iters=1,
      kda_implementation="xla",
      dtype="float32",
      weight_dtype="float32",
      scan_layers=scan_layers,
  )
  return modeling_glm5_3_flash.Glm5Model(
      cfg,
      mesh=None,
      rngs=nnx.Rngs(params=13),
      place_on_devices=False,
  )


def _build_qwen3_8_mini(
    *, scan_layers: bool, mesh: Mesh | None = None
) -> modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel:
  """Constructs a deterministic fp32 mini Qwen3_8FlashNextModel (1 cycle unscanned, 2 cycles scanned)."""
  cfg = _Qwen3_8MiniConfig(scan_layers=scan_layers, num_decoder_layers=8 if scan_layers else 4)
  active_mesh = mesh if mesh is not None else _cpu_mesh()
  with jax.set_mesh(active_mesh):
    return modeling_qwen3_8_flash_next.create_qwen3_8_flash_next_model(
        cfg,
        mesh=active_mesh,
        rngs=nnx.Rngs(params=19),
    )


def _build_qwen3_vl_mini(
    *,
    vocab_size: int = 64,
    model_name: str = "qwen3-vl-2b",
    mesh: Mesh | None = None,
) -> modeling_qwen3_vl.Qwen3VLModel:
  """Constructs a deterministic fp32 mini multimodal Qwen3VLModel."""
  cfg = _Qwen3VLMiniConfig(vocab_size=vocab_size, model_name=model_name)
  active_mesh = mesh if mesh is not None else _cpu_mesh()
  with jax.set_mesh(active_mesh):
    return modeling_qwen3_vl.create_qwen3_vl_model(cfg, mesh=active_mesh, rngs=nnx.Rngs(params=23))


def _build_model_by_arch(arch: str, *, scan_layers: bool, mesh: Mesh | None = None) -> nnx.Module:
  """Dispatches mini-model construction across all 5 m3 causal LM architectures."""
  if arch == "qwen3":
    return _build_qwen3_mini(scan_layers=scan_layers, mesh=mesh)
  if arch == "granitemoe":
    return _build_granitemoe_mini(scan_layers=scan_layers, mesh=mesh)
  if arch == "glm5":
    return _build_glm5_mini(scan_layers=scan_layers)
  if arch == "qwen3_8":
    return _build_qwen3_8_mini(scan_layers=scan_layers, mesh=mesh)
  if arch == "qwen3_vl":
    return _build_qwen3_vl_mini(mesh=mesh)
  raise ValueError(f"Unknown m3 architecture: {arch}")


class _RouterReplayStubModel(nnx.Module):
  """Minimal NNX model advertising `forced_routed_experts` and `skip_lm_head`."""

  def __init__(self, vocab_size: int = 16, emb_dim: int = 8):
    self.config = None
    self.embed = nnx.Embed(vocab_size, emb_dim, rngs=nnx.Rngs(0))
    self.head = nnx.Linear(emb_dim, vocab_size, use_bias=False, rngs=nnx.Rngs(1))

  def logits_from_hidden_states_for_vocab_tiling(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
    del deterministic
    return self.head(hidden_states)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      forced_routed_experts: jax.Array | None = None,
      skip_lm_head: bool = False,
      enable_dropout: bool = False,
  ) -> jax.Array:
    del decoder_positions, enable_dropout
    h = self.embed(decoder_input_tokens)
    if decoder_segment_ids is not None:
      h = h * (decoder_segment_ids != 0)[..., None].astype(h.dtype)
    if forced_routed_experts is not None:
      scale = (jnp.sum(forced_routed_experts, axis=(-2, -1), keepdims=False) + 1.0)[..., None]
      h = h * scale.astype(h.dtype)
    if skip_lm_head:
      return h
    return self.head(h)


class M3TunixAdapterTest(parameterized.TestCase):
  """Verifies M3TunixAdapter forward parity, packing isolation, gradients, KV cache, and weight sync."""

  def setUp(self):
    super().setUp()
    self.enterContext(nnx.use_eager_sharding(True))

  def test_import_preserves_jax_primitives_and_avoids_legacy_adapter(self):
    """Importing m3 tunix_adapter in a fresh Python process must not monkeypatch JAX lax or load torch."""
    sig_params = inspect.signature(M3TunixAdapter.__call__).parameters
    for required_name in ("segment_ids", "skip_lm_head", "forced_routed_experts"):
      self.assertIn(required_name, sig_params)
    self.assertIsNotNone(tunix_adapter)

    if not sys.executable:
      return

    # Run import-isolation checks in a clean child interpreter so earlier tests in the same
    # pytest-xdist worker (e.g. `model_creation_utils_test`) cannot pollute shared `jax.lax` globals.
    pkg_root = tunix_adapter.__name__.split(".m3.", 1)[0]
    legacy_mod_name = f"{pkg_root}.integration.tunix.tunix_adapter"
    top_pkg_name = tunix_adapter.__name__.split(".", 1)[0]
    top_pkg = sys.modules[top_pkg_name]

    py_paths: list[str] = []
    top_pkg_file = getattr(top_pkg, "__file__", None)
    if top_pkg_file:
      py_paths.append(str(pathlib.Path(top_pkg_file).resolve().parent.parent))
    for pkg_dir in getattr(top_pkg, "__path__", ()):
      candidate = str(pathlib.Path(pkg_dir).resolve().parent)
      if candidate not in py_paths:
        py_paths.append(candidate)
    for raw_entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
      if raw_entry:
        resolved_entry = str(pathlib.Path(raw_entry).resolve())
        if resolved_entry not in py_paths:
          py_paths.append(resolved_entry)

    child_env = dict(os.environ)
    if py_paths:
      child_env["PYTHONPATH"] = os.pathsep.join(py_paths)
    child_env["JAX_PLATFORMS"] = "cpu"
    child_env["JAX_ENABLE_COMPILATION_CACHE"] = "0"
    child_env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    probe_code = "\n".join(
        [
            "import importlib",
            "import pathlib",
            "import sys",
            "mod_name, legacy_mod, expected_file = sys.argv[1], sys.argv[2], sys.argv[3]",
            "assert mod_name not in sys.modules, f'{mod_name} preloaded'",
            "assert legacy_mod not in sys.modules, f'{legacy_mod} preloaded'",
            "mod = importlib.import_module(mod_name)",
            "import jax",
            "actual_file = str(pathlib.Path(mod.__file__).resolve())",
            "assert actual_file == expected_file, f'loaded {actual_file} != {expected_file}'",
            "wsc_name = getattr(jax.lax.with_sharding_constraint, '__name__', '')",
            "topk_name = getattr(jax.lax.top_k, '__name__', '')",
            "assert wsc_name != '_compat_wsc', f'with_sharding_constraint monkeypatched: {wsc_name}'",
            "assert topk_name != '_compat_top_k', f'top_k monkeypatched: {topk_name}'",
            "assert legacy_mod not in sys.modules, f'{legacy_mod} loaded by {mod_name}'",
            "assert sys.modules.get('torch') is None, 'torch loaded by m3 tunix_adapter'",
        ]
    )
    expected_adapter_file = str(pathlib.Path(tunix_adapter.__file__).resolve())
    proc = subprocess.run(
        [sys.executable, "-c", probe_code, tunix_adapter.__name__, legacy_mod_name, expected_adapter_file],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")

  @parameterized.named_parameters(
      ("qwen3_unscanned", "qwen3", False),
      ("qwen3_scanned", "qwen3", True),
      ("granitemoe_unscanned", "granitemoe", False),
      ("granitemoe_scanned", "granitemoe", True),
      ("glm5_unscanned", "glm5", False),
      ("glm5_scanned", "glm5", True),
      ("qwen3_8_unscanned", "qwen3_8", False),
      ("qwen3_8_scanned", "qwen3_8", True),
      ("qwen3_vl_unscanned", "qwen3_vl", False),
  )
  def test_forward_logit_exact_parity_and_packed_segment_isolation(self, arch: str, scan_layers: bool):
    """Direct m3 call and M3TunixAdapter match across all 5 m3 families and isolate segments."""
    mesh = _cpu_mesh() if arch != "glm5" else None
    model = _build_model_by_arch(arch, scan_layers=scan_layers, mesh=mesh)

    orig_cls = type(model)
    adapter = wrap_m3_for_tunix(model, pad_id=_PAD_ID)
    self.assertIs(type(adapter.base), orig_cls)
    self.assertIsNone(adapter.config)
    self.assertEqual(adapter.num_embed, 64)

    # Optimizer construction outside `with jax.set_mesh(mesh):` must succeed for every m3 family.
    opt = nnx.Optimizer(adapter, optax.adamw(1e-3), wrt=nnx.Param)
    self.assertIsNotNone(opt)

    tokens_a = jnp.array([[5, 6, 7, 11, 12, 13, 0, 0], [8, 9, 10, 14, 15, 16, 0, 0]], dtype=jnp.int32)
    tokens_b = jnp.array([[21, 22, 23, 11, 12, 13, 0, 0], [24, 25, 26, 14, 15, 16, 0, 0]], dtype=jnp.int32)
    if arch in ("glm5", "qwen3_8"):
      # GLM-5.3-Flash and Qwen3.8 GDN conv/state paths use {0, 1} padding masks on unpacked sequences.
      eval_positions = jnp.array([[0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2]], dtype=jnp.int32)
      eval_seg = jnp.array([[0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    else:
      eval_positions = jnp.array([[0, 1, 2, 0, 1, 2, 0, 0], [0, 1, 2, 0, 1, 2, 0, 0]], dtype=jnp.int32)
      eval_seg = jnp.array([[1, 1, 1, 2, 2, 2, 0, 0], [1, 1, 1, 2, 2, 2, 0, 0]], dtype=jnp.int32)

    def _run():
      direct = model(
          decoder_input_tokens=tokens_a,
          decoder_positions=eval_positions,
          decoder_segment_ids=eval_seg,
      )
      adapted, second = adapter(
          tokens_a,
          eval_positions,
          None,
          None,
          segment_ids=eval_seg,
      )
      self.assertIsNone(second)
      np.testing.assert_allclose(adapted, direct, rtol=0.0, atol=1e-6)

      # Valid tokens [3:6] must be invariant to perturbing masked/prior-segment tokens [:3].
      adapted_b, _ = adapter(
          tokens_b,
          eval_positions,
          None,
          None,
          segment_ids=eval_seg,
      )
      seg2_diff = float(jnp.max(jnp.abs(adapted[:, 3:6, :] - adapted_b[:, 3:6, :])))
      self.assertLessEqual(seg2_diff, 1e-6)

      # Negative control: dropping segment_ids leaks [:3] perturbations into [3:6].
      no_pad_adapter = M3TunixAdapter(model, pad_id=None)
      unmasked_a, _ = no_pad_adapter(tokens_a, eval_positions, None, None)
      unmasked_b, _ = no_pad_adapter(tokens_b, eval_positions, None, None)
      leak_diff = float(jnp.max(jnp.abs(unmasked_a[:, 3:6, :] - unmasked_b[:, 3:6, :])))
      self.assertGreater(
          leak_diff,
          1e-3,
          f"Negative control failed for {arch} (scan={scan_layers}): expected cross-segment leakage.",
      )

    if mesh is not None:
      with jax.set_mesh(mesh):
        _run()
    else:
      _run()

  def test_qwen3_vl_multimodal_images_and_skip_lm_head_parity(self):
    """Verifies `151655` visual token fusion, DeepStack, 3D M-RoPE `positions`, and `skip_lm_head` parity."""
    mesh = _cpu_mesh()
    mm_vocab = 151664
    qvl = _build_qwen3_vl_mini(vocab_size=mm_vocab, mesh=mesh)
    adapter = M3TunixAdapter(qvl, pad_id=_PAD_ID, expose_rollout_config=True)
    self.assertEqual(adapter.config, RolloutCacheSpec(num_layers=2, num_kv_heads=2, head_dim=16, vocab_size=mm_vocab))

    images = jax.random.normal(jax.random.PRNGKey(5), (2, 3, 2, 32, 32), dtype=jnp.float32) * 2.0
    grid_thw = jnp.array([[1, 2, 2], [1, 2, 2]], dtype=jnp.int32)
    # Include placeholder token `151655` so `merge_multimodal_embeddings` and `deepstack_process` execute.
    mm_tokens = jnp.array([[3, 4, 151655, 5, 6], [8, 151655, 9, 10, 11]], dtype=jnp.int32)
    positions_2d = jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32)[None, :], (2, 5))
    positions_3d = jnp.stack(
        [
            positions_2d,
            positions_2d + jnp.array([[0, 0, 1, 2, 2], [0, 1, 2, 2, 2]], dtype=jnp.int32),
            positions_2d + jnp.array([[0, 0, 2, 3, 3], [0, 2, 3, 3, 3]], dtype=jnp.int32),
        ],
        axis=-1,
    )
    seg = jnp.ones((2, 5), dtype=jnp.int32)

    with jax.set_mesh(mesh):
      for pos_arg in (positions_2d, positions_3d, positions_2d[..., None]):
        direct = qvl(
            decoder_input_tokens=mm_tokens,
            decoder_positions=pos_arg,
            decoder_segment_ids=seg,
            encoder_images=images,
            encoder_video_grid_thw=grid_thw,
        )
        adapted, aux = adapter(
            mm_tokens,
            pos_arg,
            None,
            None,
            segment_ids=seg,
            images=images,
            encoder_video_grid_thw=grid_thw,
        )
        self.assertIsNone(aux)
        np.testing.assert_allclose(adapted, direct, rtol=0.0, atol=1e-6)

        # Negative control: omitting `images` on `151655` tokens must materially change logits.
        text_only, _ = adapter(mm_tokens, pos_arg, None, None, segment_ids=seg)
        self.assertGreater(float(jnp.max(jnp.abs(adapted - text_only))), 1e-3)

        hidden, _ = adapter(
            mm_tokens,
            pos_arg,
            None,
            None,
            segment_ids=seg,
            skip_lm_head=True,
            images=images,
            encoder_video_grid_thw=grid_thw,
        )
        reconstructed = adapter.compute_final_logits(hidden)
        np.testing.assert_allclose(reconstructed, direct, rtol=1e-5, atol=1e-5)

      # Invalid 3D M-RoPE trailing dimension fails closed with ValueError.
      with self.assertRaisesRegex(ValueError, "positions has shape"):
        adapter(mm_tokens, jnp.zeros((2, 5, 2), dtype=jnp.int32), None, None, segment_ids=seg)

      # Explicit `image_token_id` resolves when `model_name` is unset, while missing `image_token_id` fails closed.
      qvl_unnamed = _build_qwen3_vl_mini(vocab_size=mm_vocab, model_name="", mesh=mesh)
      adapter_unnamed = M3TunixAdapter(qvl_unnamed, pad_id=_PAD_ID)
      with self.assertRaisesRegex(ValueError, "requires `image_token_id`"):
        adapter_unnamed(
            mm_tokens,
            positions_2d,
            None,
            None,
            segment_ids=seg,
            skip_lm_head=True,
            images=images,
            encoder_video_grid_thw=grid_thw,
        )
      adapter_explicit_img = M3TunixAdapter(qvl_unnamed, pad_id=_PAD_ID, image_token_id=151655)
      hidden_explicit, _ = adapter_explicit_img(
          mm_tokens,
          positions_2d,
          None,
          None,
          segment_ids=seg,
          skip_lm_head=True,
          images=images,
          encoder_video_grid_thw=grid_thw,
      )
      np.testing.assert_allclose(adapter_explicit_img.compute_final_logits(hidden_explicit), direct, rtol=1e-5, atol=1e-5)

      # Non-iterable scanned `decoder.layers` on multimodal `skip_lm_head=True` fails closed with NotImplementedError.
      orig_layers = qvl_unnamed.decoder.layers
      try:
        qvl_unnamed.decoder.layers = nnx.Module()
        with self.assertRaisesRegex(NotImplementedError, "does not support scanned layers"):
          adapter_explicit_img(
              mm_tokens,
              positions_2d,
              None,
              None,
              segment_ids=seg,
              skip_lm_head=True,
              images=images,
              encoder_video_grid_thw=grid_thw,
          )
      finally:
        qvl_unnamed.decoder.layers = orig_layers

  def test_segment_ids_precedence_and_attention_mask_validation(self):
    """Verifies `segment_ids > decoder_segment_ids > attention_mask > pad_id` and rank checks."""
    model = _build_qwen3_mini(scan_layers=False)
    adapter = M3TunixAdapter(model, pad_id=_PAD_ID)

    tokens = jnp.array(
        [[_PAD_ID, _PAD_ID, 4, 5, 6, _PAD_ID], [_PAD_ID, 7, 8, 9, _PAD_ID, _PAD_ID]],
        dtype=jnp.int32,
    )
    nonpad = jnp.array([[0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    positions = jnp.array([[0, 0, 0, 1, 2, 2], [0, 0, 1, 2, 2, 2]], dtype=jnp.int32)
    attn_mask = _tunix_causal_mask(nonpad.astype(jnp.bool_))
    packed_override = jnp.array([[1, 1, 1, 2, 2, 0], [1, 1, 2, 2, 0, 0]], dtype=jnp.int32)
    decoder_override = jnp.array([[3, 3, 3, 3, 3, 0], [4, 4, 4, 4, 0, 0]], dtype=jnp.int32)

    np.testing.assert_array_equal(
        adapter.resolve_segment_ids(
            tokens,
            segment_ids=packed_override,
            decoder_segment_ids=decoder_override,
            attention_mask=attn_mask,
        ),
        packed_override,
    )
    np.testing.assert_array_equal(
        adapter.resolve_segment_ids(
            tokens,
            decoder_segment_ids=decoder_override,
            attention_mask=attn_mask,
        ),
        decoder_override,
    )
    mask_disagree = jnp.array([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    np.testing.assert_array_equal(
        adapter.resolve_segment_ids(
            tokens,
            attention_mask=_tunix_causal_mask(mask_disagree.astype(jnp.bool_)),
        ),
        mask_disagree,
    )
    np.testing.assert_array_equal(adapter.resolve_segment_ids(tokens), nonpad)

    logits_from_mask, _ = adapter(tokens, positions, None, attn_mask)
    logits_from_seg, _ = adapter(tokens, positions, None, None, segment_ids=nonpad)
    np.testing.assert_allclose(logits_from_mask, logits_from_seg, rtol=0.0, atol=1e-6)

    tokens_alt_pad = tokens.at[0, 0].set(31)
    unmasked_adapter = M3TunixAdapter(model, pad_id=None)
    masked_0, _ = adapter(tokens, positions, None, attn_mask)
    masked_1, _ = adapter(tokens_alt_pad, positions, None, attn_mask)
    np.testing.assert_allclose(masked_0[0, 2:5], masked_1[0, 2:5], rtol=0.0, atol=1e-6)
    unmasked_0, _ = unmasked_adapter(tokens, positions, None, None)
    unmasked_1, _ = unmasked_adapter(tokens_alt_pad, positions, None, None)
    self.assertGreater(float(jnp.max(jnp.abs(unmasked_0[0, 2:5] - unmasked_1[0, 2:5]))), 1e-3)

    bad_mask = jnp.ones((2, 6), dtype=jnp.bool_)
    with self.assertRaisesRegex(ValueError, r"attention_mask has shape \(2, 6\), expected \(2, 6, 6\)"):
      segment_ids_from_attention_mask(bad_mask, tokens)
    with self.assertRaisesRegex(ValueError, "attention_mask has shape"):
      jax.jit(lambda t, p, m: adapter(t, p, None, m))(tokens, positions, bad_mask)

  @parameterized.named_parameters(
      ("qwen3_scanned", "qwen3", True, True),
      ("granitemoe_tied", "granitemoe", False, True),
      ("granitemoe_untied_scanned", "granitemoe", True, False),
      ("glm5_scanned", "glm5", True, False),
      ("qwen3_8_scanned", "qwen3_8", True, False),
      ("qwen3_vl_unscanned", "qwen3_vl", False, True),
  )
  def test_skip_lm_head_and_chunked_compute_final_logits(self, arch: str, scan_layers: bool, tie_embeddings: bool):
    """`skip_lm_head=True` + chunked `compute_final_logits` matches full forward across all 5 m3 families."""
    mesh = _cpu_mesh() if arch in ("qwen3", "qwen3_8", "qwen3_vl") else None
    if arch == "granitemoe":
      model: nnx.Module = _build_granitemoe_mini(
          scan_layers=scan_layers,
          tie_word_embeddings=tie_embeddings,
          logits_scaling=2.5,
      )
    else:
      model = _build_model_by_arch(arch, scan_layers=scan_layers, mesh=mesh)

    adapter = M3TunixAdapter(model, pad_id=_PAD_ID)
    tokens = jnp.array([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32)[None, :], (2, 6))

    def _run():
      full_logits, _ = adapter(tokens, positions, None, None, skip_lm_head=False)
      hidden, aux = adapter(tokens, positions, None, None, skip_lm_head=True)
      self.assertIsNone(aux)
      self.assertEqual(hidden.shape, (2, 6, 32))

      c0 = adapter.compute_final_logits(hidden[:, :3, :])
      c1 = adapter.compute_final_logits(hidden[:, 3:, :])
      reconstructed = jnp.concatenate([c0, c1], axis=1)
      np.testing.assert_allclose(reconstructed, full_logits, rtol=1e-5, atol=1e-5)

    if mesh is not None:
      with jax.set_mesh(mesh):
        _run()
    else:
      _run()

  @parameterized.named_parameters(
      ("qwen3_unscanned", "qwen3", False),
      ("qwen3_scanned", "qwen3", True),
      ("granitemoe_unscanned", "granitemoe", False),
      ("granitemoe_scanned", "granitemoe", True),
      ("glm5_unscanned", "glm5", False),
      ("glm5_scanned", "glm5", True),
      ("qwen3_8_unscanned", "qwen3_8", False),
      ("qwen3_vl_unscanned", "qwen3_vl", False),
  )
  def test_nnx_jit_and_value_and_grad_parity_without_tracer_leaks(self, arch: str, scan_layers: bool):
    """`nnx.jit` + `nnx.value_and_grad` through M3TunixAdapter matches direct model gradients."""
    mesh = _cpu_mesh() if arch != "glm5" else None
    model = _build_model_by_arch(arch, scan_layers=scan_layers, mesh=mesh)

    adapter = M3TunixAdapter(model, pad_id=_PAD_ID, expose_rollout_config=True)
    self.assertIsInstance(adapter.config, RolloutCacheSpec)
    self.assertEqual(hash(adapter.config), hash(adapter.config))

    tokens = jnp.array([[2, 3, 4, 0], [5, 6, 7, 8]], dtype=jnp.int32)
    positions = jnp.array([[0, 1, 2, 2], [0, 1, 2, 3]], dtype=jnp.int32)
    seg_ids = jnp.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=jnp.int32)
    targets = jnp.array([[3, 4, 5, 0], [6, 7, 8, 9]], dtype=jnp.int32)

    graphdef_before, _ = nnx.split(adapter)

    def _adapter_loss(m: M3TunixAdapter):
      logits, _ = m(tokens, positions, None, None, segment_ids=seg_ids)
      logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
      tok_lp = jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
      mask = (seg_ids != 0).astype(jnp.float32)
      return -jnp.sum(tok_lp * mask) / jnp.sum(mask)

    def _direct_loss(m: nnx.Module):
      logits = m(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg_ids,
      )
      logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
      tok_lp = jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
      mask = (seg_ids != 0).astype(jnp.float32)
      return -jnp.sum(tok_lp * mask) / jnp.sum(mask)

    def _check():
      loss_adp, grads_adp = nnx.jit(nnx.value_and_grad(_adapter_loss))(adapter)
      loss_dir, grads_dir = nnx.jit(nnx.value_and_grad(_direct_loss))(model)
      self.assertTrue(np.isfinite(float(loss_adp)))
      np.testing.assert_allclose(loss_adp, loss_dir, rtol=1e-5, atol=1e-6)

      adp_leaves = jax.tree_util.tree_leaves(grads_adp["base"])
      dir_leaves = jax.tree_util.tree_leaves(grads_dir)
      self.assertEqual(len(adp_leaves), len(dir_leaves))
      grad_norm = sum(float(jnp.sum(jnp.abs(g))) for g in adp_leaves)
      self.assertGreater(grad_norm, 0.0)
      for ga, gd in zip(adp_leaves, dir_leaves):
        np.testing.assert_allclose(ga, gd, rtol=1e-5, atol=1e-5)

      graphdef_after, _ = nnx.split(adapter)
      self.assertEqual(graphdef_before, graphdef_after)

    if mesh is not None:
      with jax.set_mesh(mesh):
        _check()
    else:
      _check()

  def test_enable_dropout_and_deterministic_forwarding_contracts(self):
    """Verifies `enable_dropout=True` maps accurately to `enable_dropout` or `deterministic=False`."""
    seen: dict[str, Any] = {}

    class _StrictDetStub(nnx.Module):
      """Model accepting `deterministic` without `**kwargs`."""

      def __call__(
          self, decoder_input_tokens, decoder_positions, decoder_segment_ids=None, *, deterministic: bool = True
      ):
        """Records `deterministic` flag."""
        del decoder_positions, decoder_segment_ids
        seen["strict_det"] = deterministic
        return jnp.zeros((*decoder_input_tokens.shape, 8), dtype=jnp.float32)

    class _VarKwDetStub(nnx.Module):
      """Model accepting `deterministic` alongside `**unused_kwargs`."""

      def __call__(
          self,
          decoder_input_tokens,
          decoder_positions,
          decoder_segment_ids=None,
          *,
          deterministic: bool = True,
          **unused_kwargs: Any,
      ):
        """Records both `deterministic` and `unused_kwargs`."""
        del decoder_positions, decoder_segment_ids
        seen["varkw_det"] = deterministic
        seen["varkw_extra"] = dict(unused_kwargs)
        return jnp.zeros((*decoder_input_tokens.shape, 8), dtype=jnp.float32)

    class _EnableDropoutStub(nnx.Module):
      """Model accepting `enable_dropout` directly."""

      def __call__(
          self, decoder_input_tokens, decoder_positions, decoder_segment_ids=None, *, enable_dropout: bool = False
      ):
        """Records `enable_dropout` flag."""
        del decoder_positions, decoder_segment_ids
        seen["enable_dropout"] = enable_dropout
        return jnp.zeros((*decoder_input_tokens.shape, 8), dtype=jnp.float32)

    toks = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    M3TunixAdapter(_StrictDetStub(), pad_id=_PAD_ID)(toks, enable_dropout=True)
    self.assertFalse(seen["strict_det"])
    M3TunixAdapter(_StrictDetStub(), pad_id=_PAD_ID)(toks, enable_dropout=False)
    self.assertTrue(seen["strict_det"])

    M3TunixAdapter(_VarKwDetStub(), pad_id=_PAD_ID)(toks, enable_dropout=True)
    self.assertFalse(seen["varkw_det"])
    self.assertNotIn("enable_dropout", seen["varkw_extra"])

    M3TunixAdapter(_EnableDropoutStub(), pad_id=_PAD_ID)(toks, enable_dropout=True)
    self.assertTrue(seen["enable_dropout"])

    # Models with neither `enable_dropout` nor `deterministic` (e.g. Glm5Model.__call__) fail closed on True.
    glm5_adapter = M3TunixAdapter(_build_glm5_mini(scan_layers=False), pad_id=_PAD_ID)
    with self.assertRaises(NotImplementedError):
      glm5_adapter(toks, enable_dropout=True)

  def test_kv_cache_rollout_plumbing_and_router_replay_contracts(self):
    """Verifies multi-step `M3RolloutCache` decode parity (`atol=1e-6`) and fail-closed KV-cache rules."""
    granite = _build_granitemoe_mini(scan_layers=False)
    adapter = M3TunixAdapter(granite, pad_id=_PAD_ID, expose_rollout_config=True)
    assert adapter.config is not None
    self.assertEqual(adapter.config.num_layers, 2)
    self.assertEqual(adapter.config.num_kv_heads, 2)
    self.assertEqual(adapter.config.head_dim, 8)

    pydantic_cfg = m3_config.BaseConfig.from_dict(
        {"base_num_decoder_layers": 2, "base_num_kv_heads": 2, "head_dim": 16, "vocab_size": 64}
    )

    class _PydanticCfgStub(nnx.Module):
      """Stub NNX module holding an m3 Pydantic `BaseConfig`."""

      def __init__(self, cfg: Any):
        self.config = cfg

    pyd_spec = RolloutCacheSpec.from_model(_PydanticCfgStub(pydantic_cfg))
    self.assertEqual(pyd_spec, RolloutCacheSpec(num_layers=2, num_kv_heads=2, head_dim=16, vocab_size=64))

    # 1. Stateful `adapter.init_cache` 4-argument rollout call (no `cache_position` kwarg) matches full forward.
    rollout_cache_0 = adapter.init_cache(batch_size=1, max_seq_len=8)
    self.assertIsInstance(rollout_cache_0, M3RolloutCache)
    self.assertEqual(int(rollout_cache_0.write_index), 0)

    seq = jnp.array([[2, 3, 4, 5, 6]], dtype=jnp.int32)
    pos = jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32)
    full_logits, _ = adapter(seq, pos, None, None)

    pf_mask = jnp.array(
        [
            [
                [True, False, False, False, False, False, False, False],
                [True, True, False, False, False, False, False, False],
                [True, True, True, False, False, False, False, False],
            ]
        ]
    )
    logits_pf, rollout_cache_1 = adapter(seq[:, :3], pos[:, :3], rollout_cache_0, pf_mask)
    self.assertIsInstance(rollout_cache_1, M3RolloutCache)
    self.assertEqual(int(rollout_cache_1.write_index), 3)
    np.testing.assert_allclose(logits_pf, full_logits[:, :3, :], rtol=1e-6, atol=1e-6)

    dec_mask_1 = jnp.array([[[True, True, True, True, False, False, False, False]]])
    logits_dec_1, rollout_cache_2 = adapter(seq[:, 3:4], pos[:, 3:4], rollout_cache_1, dec_mask_1)
    self.assertEqual(int(rollout_cache_2.write_index), 4)
    np.testing.assert_allclose(logits_dec_1[:, 0, :], full_logits[:, 3, :], rtol=1e-6, atol=1e-6)

    dec_mask_2 = jnp.array([[[True, True, True, True, True, False, False, False]]])
    logits_dec_2, rollout_cache_3 = adapter(seq[:, 4:5], pos[:, 4:5], rollout_cache_2, dec_mask_2)
    self.assertEqual(int(rollout_cache_3.write_index), 5)
    np.testing.assert_allclose(logits_dec_2[:, 0, :], full_logits[:, 4, :], rtol=1e-6, atol=1e-6)

    # 2. Raw `GraniteMoeKVCache` without `cache_position` fails closed; explicit `cache_position` works.
    raw_cache = modeling_granitemoe.init_kv_cache(granite.config, batch_size=1, max_seq_len=8)
    with self.assertRaisesRegex(ValueError, "cache_position"):
      adapter(seq[:, :3], pos[:, :3], raw_cache, None)
    with self.assertRaises(TypeError):
      adapter(seq[:, :3], pos[:, :3], {"layer_0": {}}, None)
    _, raw_pf = adapter(seq[:, :3], pos[:, :3], raw_cache, None, cache_position=0)
    logits_raw_dec, _ = adapter(seq[:, 3:4], pos[:, 3:4], raw_pf, dec_mask_1, cache_position=3)
    np.testing.assert_allclose(logits_raw_dec[:, 0, :], full_logits[:, 3, :], rtol=1e-6, atol=1e-6)

    # 3. Non-caching m3 models (including Qwen3Model's unused `cache=None` placeholder) fail closed.
    seq_b2 = jnp.broadcast_to(seq[:, :3], (2, 3))
    pos_b2 = jnp.broadcast_to(pos[:, :3], (2, 3))
    for arch in ("qwen3", "glm5", "qwen3_8", "qwen3_vl"):
      mesh = _cpu_mesh() if arch != "glm5" else None
      non_cache_adp = M3TunixAdapter(_build_model_by_arch(arch, scan_layers=False, mesh=mesh), pad_id=_PAD_ID)
      with self.assertRaises(NotImplementedError):
        non_cache_adp.init_cache(batch_size=2, max_seq_len=8)
      with self.assertRaises(NotImplementedError):
        if mesh is not None:
          with jax.set_mesh(mesh):
            non_cache_adp(seq_b2, pos_b2, {"k": jnp.zeros((2,))}, None)
        else:
          non_cache_adp(seq_b2, pos_b2, {"k": jnp.zeros((2,))}, None)

    # 4. Tuple-returning `__call__` preserves updated KV cache across decode steps.
    class _TupleCacheStub(nnx.Module):
      """Minimal causal LM returning `(logits, updated_cache)` as a tuple."""

      def __call__(
          self,
          decoder_input_tokens,
          decoder_positions,
          *,
          decoder_segment_ids=None,
          cache=None,
          deterministic=True,
      ):
        """Returns zero logits alongside an incremented KV cache dict."""
        del decoder_positions, decoder_segment_ids, deterministic
        logits = jnp.zeros((*decoder_input_tokens.shape, 16), dtype=jnp.float32)
        return logits, {"step": cache["step"] + 1}

    tuple_cache_adapter = M3TunixAdapter(_TupleCacheStub(), pad_id=_PAD_ID)
    _, tuple_cache_out = tuple_cache_adapter(
        jnp.array([[2]], dtype=jnp.int32),
        jnp.array([[3]], dtype=jnp.int32),
        {"step": jnp.asarray(1, dtype=jnp.int32)},
        dec_mask_1,
    )
    self.assertIsNotNone(tuple_cache_out)
    self.assertEqual(int(tuple_cache_out["step"]), 2)

    # 5. Router replay fail-closed and positive check.
    forced = jnp.zeros((1, 3, 2, 2), dtype=jnp.int32)
    with self.assertRaises(NotImplementedError):
      adapter(seq[:, :3], pos[:, :3], None, None, forced_routed_experts=forced)

    stub = _RouterReplayStubModel()
    stub_adapter = M3TunixAdapter(stub, pad_id=_PAD_ID)
    self.assertIsNone(stub_adapter.config)
    base_out, _ = stub_adapter(seq[:, :3], None, None, None)
    replayed_out, _ = stub_adapter(
        seq[:, :3], None, None, None, forced_routed_experts=jnp.ones((1, 3, 1, 2), dtype=jnp.int32)
    )
    self.assertGreater(float(jnp.max(jnp.abs(replayed_out - base_out))), 1e-3)

  def test_param_state_hierarchy_and_vllm_mapping_structure(self):
    """Verifies `base` prefix, real `VllmWeightMapping` scanned/unscanned expansion, and fail-closed checks."""
    mesh = _cpu_mesh()
    unscanned = _build_qwen3_mini(scan_layers=False, mesh=mesh)
    scanned = _build_qwen3_mini(scan_layers=True, mesh=mesh)

    adp_unscanned = M3TunixAdapter(unscanned, pad_id=_PAD_ID)
    adp_unscanned_m3 = M3TunixAdapter(unscanned, pad_id=_PAD_ID, unscanned_direct_sync_target="m3")
    adp_scanned = M3TunixAdapter(scanned, pad_id=_PAD_ID)

    opt = nnx.Optimizer(adp_scanned, optax.adamw(1e-3), wrt=nnx.Param)
    self.assertIsNotNone(opt)

    p_state = nnx.to_pure_dict(adp_unscanned.param_state())
    self.assertEqual(set(p_state.keys()), {"base"})
    engine_state = nnx.to_pure_dict(nnx.state(unscanned, nnx.Param))
    self.assertEqual(set(p_state["base"].keys()), set(engine_state.keys()))

    rollout_unscanned = adp_unscanned.to_rollout_pure_dict(legacy_layer_names=True)
    self.assertIn("layers_0", rollout_unscanned["base"]["decoder"])
    self.assertIn("layers_1", rollout_unscanned["base"]["decoder"])
    self.assertNotIn("layers", rollout_unscanned["base"]["decoder"])
    orig_q = unscanned.decoder.layers.layers[0].self_attention.query.kernel[...]
    self.assertIs(rollout_unscanned["base"]["decoder"]["layers_0"]["self_attention"]["query"]["kernel"], orig_q)

    rollout_scanned = adp_scanned.to_rollout_pure_dict(legacy_layer_names=True)
    self.assertIn("layers", rollout_scanned["base"]["decoder"])
    self.assertEqual(rollout_scanned["base"]["decoder"]["layers"]["self_attention"]["query"]["kernel"].shape[1], 2)

    # 1. `use_no_op_mappings=True`: scanned returns `{}`; unscanned returns 24 legacy `layers_<i>` mappings (or `{}` for m3).
    self.assertEqual(adp_scanned.to_hf_mappings(), {})
    self.assertEqual(adp_scanned.lora_to_hf_mappings(), {})
    self.assertEqual(adp_unscanned_m3.to_hf_mappings(), {})
    legacy_direct_map = adp_unscanned.to_hf_mappings("vllm_jax")
    self.assertEqual(len(legacy_direct_map), 24)
    self.assertEqual(
        legacy_direct_map["base.decoder.layers.layers.0.mlp.wi_0.kernel"][0],
        "model.decoder.layers_0.mlp.wi_0.kernel",
    )
    self.assertEqual(
        legacy_direct_map["base.decoder.layers.layers.1.self_attention.query.kernel"][0],
        "model.decoder.layers_1.self_attention.query.kernel",
    )

    # 2. Real `VllmWeightMapping` (`use_no_op_mappings=False`, standalone True & False) covers 100% of params.
    for standalone in (True, False):
      hf_scanned = M3TunixAdapter(
          scanned,
          pad_id=_PAD_ID,
          use_no_op_mappings=False,
          use_standalone_mappings=standalone,
      )
      s_map = hf_scanned.to_hf_mappings("vllm_jax")
      self.assertEqual(len(s_map), 13)
      self.assertIn("base.decoder.layers.mlp.wi_0.kernel", s_map)
      self.assertNotIn("base.decoder.logits_dense.kernel", s_map)
      self.assertIn("layer", s_map["base.decoder.layers.mlp.wi_0.kernel"][1])

      hf_unscanned = M3TunixAdapter(
          unscanned,
          pad_id=_PAD_ID,
          use_no_op_mappings=False,
          use_standalone_mappings=standalone,
      )
      u_map = hf_unscanned.to_hf_mappings("vllm_jax")
      self.assertEqual(len(u_map), 24)
      leaf_suffix = "weight" if standalone else "kernel"
      self.assertEqual(
          u_map["base.decoder.layers.layers.0.mlp.wi_0.kernel"],
          (f"model.layers.0.mlp.gate_proj.{leaf_suffix}", (None, "model")),
      )
      self.assertEqual(
          u_map["base.decoder.layers.layers.1.mlp.wi_0.kernel"],
          (f"model.layers.1.mlp.gate_proj.{leaf_suffix}", (None, "model")),
      )
      for _, (_, sharding_spec) in u_map.items():
        self.assertNotIn("layer", sharding_spec)

      if not standalone:
        lora_u = hf_unscanned.lora_to_hf_mappings("vllm_jax")
        self.assertIn("base.decoder.layers.layers.0.mlp.wi_0.kernel_lora_a", lora_u)
        self.assertIn("base.decoder.layers.layers.1.self_attention.out.kernel_lora_b", lora_u)
        for _, (_, lora_spec) in lora_u.items():
          self.assertNotIn("layer", lora_spec)

    hf_unfiltered = M3TunixAdapter(
        unscanned,
        pad_id=_PAD_ID,
        use_no_op_mappings=False,
        filter_missing_source_keys=False,
        require_complete_mappings=False,
    )
    self.assertEqual(len(hf_unfiltered.to_hf_mappings("vllm_jax")), 25)
    self.assertIn("base.decoder.logits_dense.kernel", hf_unfiltered.to_hf_mappings("vllm_jax"))

    # 3. Unmapped m3 families (including `qwen3.8-flash-next*` and `qwen3-vl*`) fail closed with `ValueError`.
    for arch in ("qwen3_8", "qwen3_vl", "granitemoe", "glm5"):
      m_arch = _build_model_by_arch(arch, scan_layers=False, mesh=(mesh if arch != "glm5" else None))
      with self.assertRaises(ValueError):
        M3TunixAdapter(m_arch, pad_id=_PAD_ID, use_no_op_mappings=False)

    # 4. Explicit partial `mapping_config` prunes absent keys cleanly and expands unscanned `to_hf_hook_fns`.
    def _scale_q_hook(w: jax.Array) -> jax.Array:
      """Scales attention query weights by 2.5 for hook expansion testing."""
      return w * 2.5

    custom_map = {
        "to_hf_mappings": {
            "base.token_embedder.embedding": ("model.embed_tokens.weight", (None, "model")),
            "base.decoder.logits_dense.kernel": ("lm_head.weight", ("model", None)),
            "base.decoder.layers.self_attention.query.kernel": ("model.layers.*.q_proj.weight", (None, "layer", "model")),
        },
        "to_hf_transpose_keys": {"base.decoder.layers.mlp.wi_0.kernel": (1, 0)},
        "to_hf_hook_fns": {
            "base.decoder.layers.self_attention.query.kernel": _scale_q_hook,
            "base.decoder.logits_dense.kernel": _scale_q_hook,
        },
    }
    mapped_scanned = M3TunixAdapter(scanned, pad_id=_PAD_ID, use_no_op_mappings=False, mapping_config=custom_map)
    mapped_unscanned = M3TunixAdapter(unscanned, pad_id=_PAD_ID, use_no_op_mappings=False, mapping_config=custom_map)
    resolved_map = mapped_unscanned.to_hf_mappings("vllm_jax")
    self.assertEqual(
        set(resolved_map.keys()),
        {
            "base.token_embedder.embedding",
            "base.decoder.layers.layers.0.self_attention.query.kernel",
            "base.decoder.layers.layers.1.self_attention.query.kernel",
        },
    )
    expanded_tr = mapped_unscanned.to_hf_transpose_keys("vllm_jax")
    self.assertIn("base.decoder.layers.layers.0.mlp.wi_0.kernel", expanded_tr)
    self.assertIn("base.decoder.layers.layers.1.mlp.wi_0.kernel", expanded_tr)

    hooks_scanned = mapped_scanned.to_hf_hook_fns("vllm_jax")
    hooks_unscanned = mapped_unscanned.to_hf_hook_fns("vllm_jax")
    self.assertEqual(set(hooks_scanned.keys()), {"base.decoder.layers.self_attention.query.kernel"})
    self.assertEqual(
        set(hooks_unscanned.keys()),
        {
            "base.decoder.layers.layers.0.self_attention.query.kernel",
            "base.decoder.layers.layers.1.self_attention.query.kernel",
        },
    )
    scanned_q = hooks_scanned["base.decoder.layers.self_attention.query.kernel"](
        scanned.decoder.layers.self_attention.query.kernel[...]
    )
    for lyr_i in (0, 1):
      unscanned_q = hooks_unscanned[f"base.decoder.layers.layers.{lyr_i}.self_attention.query.kernel"](
          unscanned.decoder.layers.layers[lyr_i].self_attention.query.kernel[...]
      )
      np.testing.assert_allclose(unscanned_q, jnp.take(scanned_q, lyr_i, axis=1), rtol=1e-6, atol=1e-6)

    # 5. `nnx.LoRAParam` (`*_lora_a` / `*_lora_b`) models succeed across scanned/unscanned & standalone True/False.
    lora_unscanned = _build_qwen3_mini(scan_layers=False, mesh=mesh)
    for lyr_i in (0, 1):
      mlp_wi0 = lora_unscanned.decoder.layers.layers[lyr_i].mlp.wi_0
      mlp_wi0.kernel_lora_a = nnx.LoRAParam(jnp.ones((32, 4), dtype=jnp.float32))
      mlp_wi0.kernel_lora_b = nnx.LoRAParam(jnp.ones((4, 64), dtype=jnp.float32))
    lora_scanned = _build_qwen3_mini(scan_layers=True, mesh=mesh)
    lora_scanned.decoder.layers.mlp.wi_0.kernel_lora_a = nnx.LoRAParam(jnp.ones((32, 2, 4), dtype=jnp.float32))
    lora_scanned.decoder.layers.mlp.wi_0.kernel_lora_b = nnx.LoRAParam(jnp.ones((4, 2, 64), dtype=jnp.float32))

    for standalone in (True, False):
      lora_adp_u = M3TunixAdapter(
          lora_unscanned,
          pad_id=_PAD_ID,
          use_no_op_mappings=False,
          use_standalone_mappings=standalone,
      )
      self.assertEqual(len(lora_adp_u.to_hf_mappings("vllm_jax")), 24)
      u_lora_map = lora_adp_u.lora_to_hf_mappings("vllm_jax")
      self.assertEqual(
          set(u_lora_map.keys()),
          {
              "base.decoder.layers.layers.0.mlp.wi_0.kernel_lora_a",
              "base.decoder.layers.layers.0.mlp.wi_0.kernel_lora_b",
              "base.decoder.layers.layers.1.mlp.wi_0.kernel_lora_a",
              "base.decoder.layers.layers.1.mlp.wi_0.kernel_lora_b",
          },
      )
      for _, (_, spec) in u_lora_map.items():
        self.assertNotIn("layer", spec)

      lora_adp_s = M3TunixAdapter(
          lora_scanned,
          pad_id=_PAD_ID,
          use_no_op_mappings=False,
          use_standalone_mappings=standalone,
      )
      self.assertEqual(len(lora_adp_s.to_hf_mappings("vllm_jax")), 13)
      self.assertEqual(
          set(lora_adp_s.lora_to_hf_mappings("vllm_jax").keys()),
          {
              "base.decoder.layers.mlp.wi_0.kernel_lora_a",
              "base.decoder.layers.mlp.wi_0.kernel_lora_b",
          },
      )

    # Unmapped LoRA parameter (`token_embedder.embedding_lora_a`) fails closed with `ValueError`.
    bad_lora = _build_qwen3_mini(scan_layers=False, mesh=mesh)
    bad_lora.token_embedder.embedding_lora_a = nnx.LoRAParam(jnp.ones((64, 4), dtype=jnp.float32))
    with self.assertRaisesRegex(ValueError, "LoRA"):
      M3TunixAdapter(bad_lora, pad_id=_PAD_ID, use_no_op_mappings=False)

    with self.assertRaises(ValueError):
      M3TunixAdapter(
          scanned,
          pad_id=_PAD_ID,
          use_no_op_mappings=False,
          mapping_config={"to_hf_mappings": {"base.nonexistent.weight": ("x", (None,))}},
      )


if __name__ == "__main__":
  unittest.main()
