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

"""Unit tests for m3 general kernel dispatch, platform auto-selection, and broadcast sharding."""

import os
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from flax import nnx
from maxtext.m3.core import kernels
from maxtext.m3.core.kernels import KernelBackend
from maxtext.m3.core.sharding import DefaultSharding, TensorType, create_mesh
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Decoder


def _make_cpu_mesh(*, explicit: bool = False) -> jax.sharding.Mesh:
  """Builds a 2D (fsdp, tensor) mesh over available host CPU devices."""
  n_dev = len(jax.devices())
  fsdp = max(1, n_dev // 2)
  tensor = max(1, n_dev // fsdp)
  return create_mesh(
      ici={"fsdp": fsdp, "tensor": tensor}, dcn={"fsdp": 1, "tensor": 1}, axis_names=("fsdp", "tensor"), explicit=explicit
  )


def _mock_platform_mesh(platform: str) -> types.SimpleNamespace:
  """Constructs a lightweight mock mesh reporting `platform` on its abstract device."""
  return types.SimpleNamespace(
      empty=False,
      axis_names=("fsdp", "tensor"),
      axis_types=(AxisType.Auto, AxisType.Auto),
      shape={"fsdp": 2, "tensor": 2},
      abstract_device=types.SimpleNamespace(platform=platform, device_kind=platform),
  )


def _numpy_attention_oracle(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool = True,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    segment_ids: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
  """Pure-NumPy float64 reference attention (`q` pre-scaled)."""
  q64 = np.asarray(q, dtype=np.float64)
  k64 = np.asarray(k, dtype=np.float64)
  v64 = np.asarray(v, dtype=np.float64)
  batch, q_len, n_q, _ = q64.shape
  kv_len = k64.shape[1]
  n_kv = k64.shape[2]
  d_v = v64.shape[3]
  groups = n_q // n_kv

  k_rep = np.repeat(k64, groups, axis=2)
  v_rep = np.repeat(v64, groups, axis=2)
  logits = np.einsum("bthd,bshd->bhts", q64, k_rep)
  if soft_cap is not None:
    logits = np.tanh(logits / float(soft_cap)) * float(soft_cap)

  q_pos = (
      np.broadcast_to(np.arange(q_len, dtype=np.int32)[None, :], (batch, q_len))
      if positions is None
      else np.broadcast_to(np.asarray(positions, dtype=np.int32), (batch, q_len))
  )
  k_pos = (
      np.broadcast_to(np.arange(kv_len, dtype=np.int32)[None, :], (batch, kv_len))
      if (positions is None or kv_len != q_len)
      else q_pos
  )
  mask = (k_pos[:, None, :, None] <= q_pos[:, None, None, :]).transpose(0, 1, 3, 2)
  if not causal:
    mask = np.ones((batch, 1, q_len, kv_len), dtype=bool)
  if sliding_window is not None:
    left = sliding_window - 1
    right = 0 if causal else (sliding_window - 1)
    q_idx = q_pos[:, None, :, None]
    k_idx = k_pos[:, None, None, :]
    mask = mask & ((q_idx - left) <= k_idx) & (k_idx <= (q_idx + right))
  if segment_ids is not None:
    seg = np.broadcast_to(np.asarray(segment_ids, dtype=np.int32), (batch, q_len))
    seg_q = seg[:, None, :, None]
    seg_k = seg[:, None, None, :]
    mask = mask & (seg_q == seg_k) & (seg_q != 0) & (seg_k != 0)

  any_valid = np.any(mask, axis=-1, keepdims=True)
  masked = np.where(mask, logits, -np.inf)
  row_max = np.where(any_valid, np.max(masked, axis=-1, keepdims=True), 0.0)
  exp_logits = np.where(mask, np.exp(masked - row_max), 0.0)
  denom = np.sum(exp_logits, axis=-1, keepdims=True)
  probs = np.where(denom > 0.0, exp_logits / np.where(denom > 0.0, denom, 1.0), 0.0)
  out = np.einsum("bhts,bshd->bthd", probs, v_rep)
  out = np.where(np.transpose(any_valid, (0, 2, 1, 3)), out, 0.0)
  if segment_ids is not None:
    seg = np.broadcast_to(np.asarray(segment_ids, dtype=np.int32), (batch, q_len))
    out = np.where((seg != 0)[:, :, None, None], out, 0.0)
  assert out.shape == (batch, q_len, n_q, d_v)
  return out


class TestLazyBackendImports:
  """Verifies importing `maxtext.m3.core.kernels` on CPU does not eagerly import heavy TPU/GPU packages."""

  def test_cpu_import_and_auto_execution_avoid_heavy_kernel_packages(self):
    code = """
import sys
import jax.numpy as jnp
import maxtext.m3.core.kernels as kernels

FORBIDDEN = (
    "tokamax",
    "maxtext.kernels.tokamax_splash_attention",
    "maxtext.kernels.tokamax_splash_attention.splash_attention_kernel",
    "maxtext.kernels.tokamax_splash_attention.splash_attention_mask",
    "jax.experimental.pallas",
    "jax.experimental.pallas.tpu",
)
loaded_at_import = [m for m in FORBIDDEN if m in sys.modules]
assert not loaded_at_import, f"Eagerly imported forbidden modules: {loaded_at_import}"

assert kernels.resolve_kernel_backend(kernels.KernelBackend.AUTO) == kernels.KernelBackend.JAX
q = jnp.ones((2, 16, 4, 32), dtype=jnp.float32)
k = jnp.ones((2, 16, 2, 32), dtype=jnp.float32)
out = kernels.dot_product_attention(q, k, k, backend=kernels.KernelBackend.AUTO)
assert out.shape == (2, 16, 4, 32)

loaded_after_cpu = [m for m in FORBIDDEN if m in sys.modules]
assert not loaded_after_cpu, f"CPU execution imported forbidden modules: {loaded_after_cpu}"

# Negative control: explicitly invoking _import_splash() loads the Splash module into sys.modules.
kernels._import_splash()
assert "maxtext.kernels.tokamax_splash_attention.splash_attention_kernel" in sys.modules
"""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0


class TestKernelBackendDispatchAndAutoSelection:
  """Covers every `KernelBackend` mode (`auto`, `jax`, `tokamax`, `splash`, `cudnn`) across platforms."""

  def test_kernel_backend_enum_values_and_parser(self):
    assert tuple(b.value for b in KernelBackend) == ("auto", "jax", "tokamax", "splash", "cudnn")
    for name in ("auto", "jax", "tokamax", "splash", "cudnn"):
      parsed = kernels.parse_kernel_backend(name)
      assert isinstance(parsed, KernelBackend)
      assert parsed == name
      assert kernels.parse_kernel_backend(name.upper()) == parsed

    assert kernels.parse_kernel_backend("xla", allow_legacy_aliases=True) == KernelBackend.JAX
    assert kernels.parse_kernel_backend("dot_product", allow_legacy_aliases=True) == KernelBackend.JAX
    assert kernels.parse_kernel_backend("flash", allow_legacy_aliases=True) == KernelBackend.SPLASH
    assert kernels.parse_kernel_backend("autoselected", allow_legacy_aliases=True) == KernelBackend.AUTO
    assert kernels.parse_kernel_backend("cudnn_flash_te", allow_legacy_aliases=True) == KernelBackend.CUDNN

  def test_auto_selection_across_cpu_tpu_and_gpu_platforms(self, monkeypatch):
    cpu_mesh = _make_cpu_mesh()
    tpu_mesh = _mock_platform_mesh("tpu")
    gpu_mesh = _mock_platform_mesh("gpu")

    assert kernels.mesh_platform(cpu_mesh) == "cpu"
    assert kernels.mesh_platform(tpu_mesh) == "tpu"
    assert kernels.mesh_platform(gpu_mesh) == "gpu"
    assert kernels.is_cpu_mesh(cpu_mesh)
    assert kernels.is_tpu_mesh(tpu_mesh)
    assert kernels.is_gpu_mesh(gpu_mesh)

    # CPU: 'auto' always selects pure JAX dot-product attention.
    assert kernels.resolve_kernel_backend("auto", mesh=cpu_mesh, seq_len=256) == KernelBackend.JAX
    assert kernels.resolve_kernel_backend(KernelBackend.JAX, mesh=cpu_mesh) == KernelBackend.JAX
    assert kernels.resolve_kernel_backend(KernelBackend.TOKAMAX, mesh=cpu_mesh) == KernelBackend.TOKAMAX

    # TPU: 'auto' selects Splash when sequence length is aligned to 128, falling back to Tokamax then JAX.
    assert kernels.resolve_kernel_backend("auto", mesh=tpu_mesh, seq_len=256) == KernelBackend.SPLASH
    assert kernels.resolve_kernel_backend("splash", mesh=tpu_mesh, seq_len=256) == KernelBackend.SPLASH
    assert kernels.resolve_kernel_backend("auto", mesh=tpu_mesh, seq_len=64) == KernelBackend.TOKAMAX

    monkeypatch.setitem(sys.modules, "tokamax", None)
    assert kernels.resolve_kernel_backend("auto", mesh=tpu_mesh, seq_len=64) == KernelBackend.JAX

    monkeypatch.setitem(
        sys.modules,
        "maxtext.kernels.tokamax_splash_attention.splash_attention_kernel",
        None,
    )
    assert kernels.resolve_kernel_backend("auto", mesh=tpu_mesh, seq_len=256) == KernelBackend.JAX

    # GPU: 'auto' selects cuDNN when available, falling back to Tokamax then JAX.
    assert kernels.resolve_kernel_backend("auto", mesh=gpu_mesh, seq_len=256) == KernelBackend.CUDNN
    assert kernels.resolve_kernel_backend("cudnn", mesh=gpu_mesh, seq_len=256) == KernelBackend.CUDNN
    monkeypatch.setattr(kernels, "cudnn_available", lambda mesh=None, config=None: False)
    assert kernels.resolve_kernel_backend("auto", mesh=gpu_mesh, seq_len=256) == KernelBackend.JAX

  def test_attention_from_config_resolves_kernel_backend_modes(self):
    cpu_mesh = _make_cpu_mesh()
    tpu_mesh = _mock_platform_mesh("tpu")
    gpu_mesh = _mock_platform_mesh("gpu")

    cfg_auto = types.SimpleNamespace(
        attention="autoselected",
        kernel_backend="auto",
        attention_type="global",
        max_target_length=256,
    )
    kern_cpu = kernels.attention_from_config(cfg_auto, DefaultSharding(), mesh=cpu_mesh)
    assert kern_cpu.backend == KernelBackend.JAX
    assert kern_cpu.kernel_backend == KernelBackend.JAX

    kern_tpu = kernels.attention_from_config(cfg_auto, DefaultSharding(), mesh=tpu_mesh)
    assert kern_tpu.backend == KernelBackend.SPLASH
    assert kern_tpu.kernel_backend == KernelBackend.SPLASH

    kern_gpu = kernels.attention_from_config(cfg_auto, DefaultSharding(), mesh=gpu_mesh)
    assert kern_gpu.backend == KernelBackend.CUDNN
    assert kern_gpu.kernel_backend == KernelBackend.CUDNN

    kern_tok = kernels.attention_from_config(
        cfg_auto,
        DefaultSharding(),
        backend=KernelBackend.TOKAMAX,
        mesh=cpu_mesh,
    )
    assert kern_tok.backend == KernelBackend.TOKAMAX
    assert kern_tok.kernel_backend == KernelBackend.TOKAMAX

  def test_cudnn_backend_dispatches_cudnn_implementation_on_gpu_mesh(self, monkeypatch):
    gpu_mesh = _mock_platform_mesh("gpu")
    captured: dict[str, object] = {}
    orig_dpa = jax.nn.dot_product_attention

    def _spy_dpa(q, k, v, *args, **kwargs):
      captured["implementation"] = kwargs.get("implementation")
      kwargs["implementation"] = "xla"
      return orig_dpa(q, k, v, *args, **kwargs)

    monkeypatch.setattr(kernels, "mesh_platform", lambda mesh=None, config=None: "gpu")
    monkeypatch.setattr(jax.nn, "dot_product_attention", _spy_dpa)

    q = jnp.ones((2, 16, 4, 32), dtype=jnp.float32)
    k = jnp.ones((2, 16, 2, 32), dtype=jnp.float32)
    kern_cudnn = kernels.AttentionKernel(backend=KernelBackend.CUDNN)
    out = kern_cudnn(q, k, k)
    assert out.shape == (2, 16, 4, 32)
    assert captured["implementation"] == "cudnn"
    assert gpu_mesh.abstract_device.platform == "gpu"


class TestBroadcastShardingNormalization:
  """Verifies singleton and replicated dims normalize cleanly across 4-CPU Auto and Explicit meshes."""

  def test_normalize_broadcast_spec_clears_singleton_and_unaligned_axes(self):
    mesh_auto = _make_cpu_mesh(explicit=False)
    mesh_exp = _make_cpu_mesh(explicit=True)

    for mesh in (mesh_auto, mesh_exp):
      # Singleton batch=1 and singleton kv_heads=1 must normalize to replicated None.
      spec = P("fsdp", "context", "tensor", None)
      norm_singleton = kernels.normalize_broadcast_spec(spec, (1, 128, 1, 64), mesh)
      assert norm_singleton == P(None, None, None, None)

      # Non-singleton aligned dims preserve active mesh axes; absent axis ('context') resolves to None.
      norm_aligned = kernels.normalize_broadcast_spec(spec, (4, 128, 4, 64), mesh)
      expected_fsdp = "fsdp" if mesh.shape["fsdp"] > 1 else None
      expected_tp = "tensor" if mesh.shape["tensor"] > 1 else None
      assert norm_aligned == P(expected_fsdp, None, expected_tp, None)

      # Unaligned dim (batch=3 across 2 shards) falls back to None when allow_unaligned_fallback=True.
      norm_unaligned = kernels.normalize_broadcast_spec(
          spec,
          (3, 128, 4, 64),
          mesh,
          allow_unaligned_fallback=True,
      )
      assert norm_unaligned == P(None, None, expected_tp, None)

  @pytest.mark.parametrize("explicit", [False, True])
  def test_singleton_batch_kv_and_mqa_heads_across_4cpu_mesh(self, explicit: bool):
    mesh = _make_cpu_mesh(explicit=explicit)
    batch, seq_len, n_q, d_qk, d_v = 4, 128, 4, 64, 64
    rng = np.random.default_rng(101)
    q_np = (rng.standard_normal((batch, seq_len, n_q, d_qk)) * 0.2).astype(np.float32)
    # Singleton batch=1 and MQA kv_heads=1!
    k_b1_np = (rng.standard_normal((1, seq_len, 1, d_qk)) * 0.2).astype(np.float32)
    v_b1_np = (rng.standard_normal((1, seq_len, 1, d_v)) * 0.2).astype(np.float32)
    k_full_np = np.broadcast_to(k_b1_np, (batch, seq_len, 1, d_qk))
    v_full_np = np.broadcast_to(v_b1_np, (batch, seq_len, 1, d_v))
    # Singleton batch=1 segment_ids and positions!
    seg_b1_np = np.ones((1, seq_len), dtype=np.int32)
    seg_b1_np[:, 64:112] = 2
    seg_b1_np[:, 112:] = 0
    pos_b1_np = np.zeros((1, seq_len), dtype=np.int32)
    pos_b1_np[:, :64] = np.arange(64, dtype=np.int32)
    pos_b1_np[:, 64:112] = np.arange(48, dtype=np.int32)

    ref = _numpy_attention_oracle(
        q_np,
        k_full_np,
        v_full_np,
        causal=True,
        segment_ids=seg_b1_np,
        positions=pos_b1_np,
    )

    with jax.set_mesh(mesh):
      q = jnp.asarray(q_np)
      k_b1 = jnp.asarray(k_b1_np)
      v_b1 = jnp.asarray(v_b1_np)
      k_full = jnp.asarray(k_full_np)
      v_full = jnp.asarray(v_full_np)
      seg_b1 = jnp.asarray(seg_b1_np)
      pos_b1 = jnp.asarray(pos_b1_np)

      for backend in (KernelBackend.AUTO, KernelBackend.JAX, KernelBackend.TOKAMAX):
        kern = kernels.AttentionKernel(
            backend=backend,
            batch="fsdp",
            heads="tensor",
            kv_heads="tensor",
            causal=True,
        )
        out_b1 = jax.jit(lambda qi, ki, vi, s, p, k_obj=kern: k_obj(qi, ki, vi, segment_ids=s, positions=p))(
            q, k_b1, v_b1, seg_b1, pos_b1
        )
        out_full = jax.jit(lambda qi, ki, vi, s, p, k_obj=kern: k_obj(qi, ki, vi, segment_ids=s, positions=p))(
            q,
            k_full,
            v_full,
            jnp.broadcast_to(seg_b1, (batch, seq_len)),
            jnp.broadcast_to(pos_b1, (batch, seq_len)),
        )
        np.testing.assert_allclose(np.asarray(out_b1), np.asarray(out_full), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(out_b1), ref, rtol=1e-5, atol=1e-5)

      # Also verify MQA (kv_heads=1 with kv_heads="tensor") and singleton segment_ids (1, S) on Splash!
      splash_kern = kernels.AttentionKernel(
          backend="splash",
          batch="fsdp",
          heads="tensor",
          kv_heads="tensor",
          causal=True,
          splash=kernels.SplashConfig(
              block_q=128,
              block_kv=128,
              block_kv_compute=128,
              block_q_dkv=128,
              block_kv_dkv=128,
              block_kv_dkv_compute=128,
              use_base2_exp=False,
          ),
      )
      out_splash = jax.jit(lambda qi, ki, vi, s: splash_kern(qi, ki, vi, segment_ids=s))(q, k_b1, v_b1, seg_b1)
      ref_splash = _numpy_attention_oracle(q_np, k_full_np, v_full_np, causal=True, segment_ids=seg_b1_np)
      np.testing.assert_allclose(np.asarray(out_splash), ref_splash, rtol=1e-5, atol=1e-5)


class TestNumericalParityAndGradients:
  """Verifies forward/backward parity across `auto`, `jax`, and `tokamax` on 4-CPU meshes."""

  def test_forward_and_backward_parity_across_backends(self):
    mesh = _make_cpu_mesh(explicit=False)
    batch, seq_len, n_q, n_kv, d_qk, d_v = 4, 64, 4, 2, 32, 24
    rng = np.random.default_rng(202)
    q_np = (rng.standard_normal((batch, seq_len, n_q, d_qk)) * 0.25).astype(np.float32)
    k_np = (rng.standard_normal((batch, seq_len, n_kv, d_qk)) * 0.25).astype(np.float32)
    v_np = (rng.standard_normal((batch, seq_len, n_kv, d_v)) * 0.25).astype(np.float32)
    cot_np = (rng.standard_normal((batch, seq_len, n_q, d_v)) * 0.25).astype(np.float32)
    seg_np = np.ones((batch, seq_len), dtype=np.int32)
    seg_np[:, 32:56] = 2
    seg_np[:, 56:] = 0

    ref_out = _numpy_attention_oracle(
        q_np,
        k_np,
        v_np,
        causal=True,
        sliding_window=16,
        soft_cap=20.0,
        segment_ids=seg_np,
    )

    with jax.set_mesh(mesh):
      q = jnp.asarray(q_np)
      k = jnp.asarray(k_np)
      v = jnp.asarray(v_np)
      cot = jnp.asarray(cot_np)
      seg = jnp.asarray(seg_np)

      results: dict[str, tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
      for backend in (KernelBackend.AUTO, KernelBackend.JAX, KernelBackend.TOKAMAX):
        kern = kernels.AttentionKernel(
            backend=backend,
            batch="fsdp",
            heads="tensor",
            kv_heads="tensor",
            causal=True,
            sliding_window=16,
            attn_logits_soft_cap=20.0,
        )

        def _loss(qi, ki, vi, k_obj=kern):
          out = k_obj(qi, ki, vi, segment_ids=seg)
          return jnp.sum(out * cot), out

        (_, out_arr), grads = jax.jit(jax.value_and_grad(_loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)
        np.testing.assert_allclose(np.asarray(out_arr), ref_out, rtol=1e-5, atol=1e-5)
        assert np.max(np.abs(np.asarray(out_arr)[:, 56:, :, :])) == 0.0
        assert np.max(np.abs(np.asarray(grads[0])[:, 56:, :, :])) == 0.0
        results[backend.value] = (np.asarray(out_arr), tuple(np.asarray(g) for g in grads))

      for other in ("auto", "tokamax"):
        np.testing.assert_allclose(results[other][0], results["jax"][0], rtol=1e-5, atol=1e-5)
        for g_other, g_jax in zip(results[other][1], results["jax"][1]):
          np.testing.assert_allclose(g_other, g_jax, rtol=1e-5, atol=1e-5)

      # Negative control: perturbing V or changing sliding_window from 16 to 4 changes outputs and gradients.
      kern_jax = kernels.AttentionKernel(
          backend=KernelBackend.JAX,
          batch="fsdp",
          heads="tensor",
          kv_heads="tensor",
          causal=True,
          sliding_window=4,
          attn_logits_soft_cap=20.0,
      )
      out_narrow = jax.jit(lambda qi, ki, vi: kern_jax(qi, ki, vi, segment_ids=seg))(q, k, v)
      assert float(np.max(np.abs(np.asarray(out_narrow) - results["jax"][0]))) > 1e-2

  def test_float32_logits_gate_preserved_across_jax_auto_and_tokamax(self):
    q = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 32), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 2, 32), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(2), (2, 16, 2, 32), dtype=jnp.bfloat16)

    for backend in (KernelBackend.JAX, KernelBackend.AUTO, KernelBackend.TOKAMAX):
      kern_bf16 = kernels.AttentionKernel(backend=backend, float32_logits=False)
      kern_fp32 = kernels.AttentionKernel(backend=backend, float32_logits=True)
      jaxpr_bf16 = str(jax.make_jaxpr(kern_bf16)(q, k, v))
      jaxpr_fp32 = str(jax.make_jaxpr(kern_fp32)(q, k, v))
      assert "BF16_BF16_F32" not in jaxpr_bf16
      if backend in (KernelBackend.JAX, KernelBackend.AUTO):
        assert "BF16_BF16_F32" in jaxpr_fp32

  def test_decode_shifted_positions_parity_and_negative_control(self):
    mesh = _make_cpu_mesh()
    batch, q_len, kv_len, n_q, n_kv, d_qk, d_v = 2, 1, 128, 4, 2, 32, 32
    rng = np.random.default_rng(42)
    q_np = (rng.standard_normal((batch, q_len, n_q, d_qk)) * 0.25).astype(np.float32)
    k_np = (rng.standard_normal((batch, kv_len, n_kv, d_qk)) * 0.25).astype(np.float32)
    v_np = (rng.standard_normal((batch, kv_len, n_kv, d_v)) * 0.25).astype(np.float32)
    pos_np = np.array([[5], [17]], dtype=np.int32)

    ref_out = _numpy_attention_oracle(q_np, k_np, v_np, causal=True, positions=pos_np)
    ref_out_none = _numpy_attention_oracle(q_np, k_np, v_np, causal=True, positions=None)
    assert float(np.max(np.abs(ref_out - ref_out_none))) > 1e-3

    q = jnp.asarray(q_np)
    k = jnp.asarray(k_np)
    v = jnp.asarray(v_np)
    pos = jnp.asarray(pos_np)

    backends = [KernelBackend.JAX, "xla"]
    if kernels.tokamax_available():
      backends.append(KernelBackend.TOKAMAX)

    with jax.set_mesh(mesh):
      for b in backends:
        kern = kernels.AttentionKernel(backend=b, causal=True)
        out = kern(q, k, v, positions=pos, segment_ids=None)
        np.testing.assert_allclose(np.asarray(out), ref_out, rtol=1e-5, atol=1e-5)

  def test_qwen3_attention_custom_positions_without_segments(self):
    cfg = types.SimpleNamespace(
        emb_dim=128,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=64,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        matmul_precision="default",
        normalization_layer_epsilon=1e-6,
        use_qk_norm=True,
        rope_max_timescale=10000.0,
        attention="dot_product",
        attention_type="global",
        float32_logits=True,
    )
    batch = 2
    attn = Qwen3Attention(config=cfg, rngs=nnx.Rngs(params=0))
    x = jax.random.normal(jax.random.PRNGKey(10), (batch, 16, 128), dtype=jnp.float32)
    custom_p = jnp.tile(jnp.arange(4, dtype=jnp.int32), (batch, 4))
    out_custom = attn(x, decoder_positions=custom_p, decoder_segment_ids=None)
    out_default = attn(x, decoder_positions=None, decoder_segment_ids=None)
    diff_qwen = float(np.max(np.abs(np.asarray(out_custom) - np.asarray(out_default))))
    assert diff_qwen > 1e-3


class TestFailClosedNegativeControls:
  """Verifies unknown backends and unsupported explicit backend/platform combinations fail closed."""

  def test_unknown_backend_names_raise_value_error(self):
    for bad_name in ("invalid_backend", "triton", "flash_v3", "", 123):
      with pytest.raises(ValueError, match="backend"):
        kernels.parse_kernel_backend(bad_name)  # type: ignore[arg-type]
      with pytest.raises(ValueError, match="backend"):
        kernels.resolve_kernel_backend(bad_name)  # type: ignore[arg-type]
      with pytest.raises(ValueError, match="backend"):
        kernels.AttentionKernel(backend=bad_name)  # type: ignore[arg-type]

    bad_cfg = types.SimpleNamespace(
        attention="autoselected",
        kernel_backend="triton",
        attention_type="global",
        max_target_length=256,
    )
    with pytest.raises(ValueError, match="Unknown kernel backend"):
      kernels.validate_attention_config(bad_cfg)

    q = jnp.ones((2, 16, 4, 32), dtype=jnp.float32)
    k = jnp.ones((2, 16, 2, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Unknown kernel backend"):
      kernels.dot_product_attention(q, k, k, backend="triton")

  def test_explicit_tpu_only_splash_on_cpu_or_gpu_fails_closed(self):
    cpu_mesh = _make_cpu_mesh()
    gpu_mesh = _mock_platform_mesh("gpu")

    for non_tpu_mesh in (cpu_mesh, gpu_mesh):
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        kernels.resolve_kernel_backend(KernelBackend.SPLASH, mesh=non_tpu_mesh, seq_len=256)
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        kernels.resolve_kernel_backend("splash", mesh=non_tpu_mesh, seq_len=256)

      cfg_splash = types.SimpleNamespace(
          attention="autoselected",
          kernel_backend="splash",
          attention_type="global",
          max_target_length=256,
      )
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        kernels.validate_attention_config(cfg_splash, mesh=non_tpu_mesh)
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        kernels.attention_from_config(cfg_splash, mesh=non_tpu_mesh)

    q = jnp.ones((4, 128, 4, 64), dtype=jnp.float32)
    k = jnp.ones((4, 128, 2, 64), dtype=jnp.float32)
    with jax.set_mesh(cpu_mesh):
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        kernels.dot_product_attention(q, k, k, backend=KernelBackend.SPLASH)
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        kernels.dot_product_attention(q, k, k, backend="splash")

      strict_splash = kernels.AttentionKernel(
          backend=KernelBackend.SPLASH,
          splash=kernels.SplashConfig(allow_cpu_interpret=False),
      )
      with pytest.raises(ValueError, match="requires a TPU mesh"):
        strict_splash(q, k, k)

  def test_explicit_gpu_only_cudnn_on_cpu_or_tpu_fails_closed(self):
    cpu_mesh = _make_cpu_mesh()
    tpu_mesh = _mock_platform_mesh("tpu")

    for non_gpu_mesh in (cpu_mesh, tpu_mesh):
      with pytest.raises(ValueError, match="requires a GPU mesh"):
        kernels.resolve_kernel_backend(KernelBackend.CUDNN, mesh=non_gpu_mesh, seq_len=256)
      with pytest.raises(ValueError, match="requires a GPU mesh"):
        kernels.resolve_kernel_backend("cudnn", mesh=non_gpu_mesh, seq_len=256)

      cfg_cudnn = types.SimpleNamespace(
          attention="autoselected",
          kernel_backend="cudnn",
          attention_type="global",
          max_target_length=256,
      )
      with pytest.raises(ValueError, match="requires a GPU mesh"):
        kernels.validate_attention_config(cfg_cudnn, mesh=non_gpu_mesh)
      with pytest.raises(ValueError, match="requires a GPU mesh"):
        kernels.attention_from_config(cfg_cudnn, mesh=non_gpu_mesh)

    q = jnp.ones((2, 16, 4, 32), dtype=jnp.float32)
    k = jnp.ones((2, 16, 2, 32), dtype=jnp.float32)
    with jax.set_mesh(cpu_mesh):
      with pytest.raises(ValueError, match="requires a GPU mesh"):
        kernels.dot_product_attention(q, k, k, backend=KernelBackend.CUDNN)
      kern_cudnn = kernels.AttentionKernel(backend=KernelBackend.CUDNN)
      with pytest.raises(ValueError, match="requires a GPU mesh"):
        kern_cudnn(q, k, k)

  def test_explicit_tokamax_when_unavailable_fails_closed(self, monkeypatch):
    monkeypatch.setitem(sys.modules, "tokamax", None)
    cpu_mesh = _make_cpu_mesh()
    with pytest.raises(ImportError, match="tokamax"):
      kernels.resolve_kernel_backend(KernelBackend.TOKAMAX, mesh=cpu_mesh)
    q = jnp.ones((2, 16, 4, 32), dtype=jnp.float32)
    k = jnp.ones((2, 16, 2, 32), dtype=jnp.float32)
    kern_tok = kernels.AttentionKernel(backend=KernelBackend.TOKAMAX)
    with pytest.raises(ImportError, match="tokamax"):
      kern_tok(q, k, k)

  def test_incompatible_broadcast_shapes_fail_closed(self):
    cpu_mesh = _make_cpu_mesh()
    q = jnp.ones((4, 16, 4, 32), dtype=jnp.float32)
    k_bad_batch = jnp.ones((3, 16, 2, 32), dtype=jnp.float32)
    v_bad_batch = jnp.ones((3, 16, 2, 32), dtype=jnp.float32)
    bad_seg = jnp.ones((3, 16), dtype=jnp.int32)

    with jax.set_mesh(cpu_mesh):
      with pytest.raises(ValueError, match="Batch size mismatch"):
        kernels.dot_product_attention(q, k_bad_batch, v_bad_batch)
      with pytest.raises(ValueError, match="segment_ids must have shape"):
        kernels.dot_product_attention(q, q[:, :, :2, :], q[:, :, :2, :], segment_ids=bad_seg)
      with pytest.raises(ValueError, match="not broadcast-compatible"):
        kernels.normalize_broadcast_sharding(k_bad_batch, P("fsdp", None, "tensor", None), cpu_mesh, target_shape=q.shape)

  def test_explicit_splash_with_custom_positions_and_no_segments_fails_closed_and_auto_falls_back(self):
    q = jnp.ones((2, 128, 4, 32), dtype=jnp.float32)
    k = jnp.ones((2, 128, 2, 32), dtype=jnp.float32)
    v = jnp.ones((2, 128, 2, 32), dtype=jnp.float32)
    pos = jnp.zeros((2, 128), dtype=jnp.int32)

    kern_strict = kernels.AttentionKernel(backend="splash", allow_xla_fallback=False)
    with pytest.raises(ValueError, match="Splash attention requires unit-step contiguous positions"):
      kern_strict(q, k, v, positions=pos, segment_ids=None)

    auto_kern = kernels.AttentionKernel(backend="auto", causal=True, allow_xla_fallback=True)
    out_auto = auto_kern(q, k, v, positions=pos, segment_ids=None)
    ref_out = _numpy_attention_oracle(np.asarray(q), np.asarray(k), np.asarray(v), causal=True, positions=np.asarray(pos))
    np.testing.assert_allclose(np.asarray(out_auto), ref_out, rtol=1e-5, atol=1e-5)

  def test_splash_mesh_layout_is_checked_at_construction(self):
    """`auto` falls back to XLA and explicit Splash fails at setup when Splash cannot shard the layout."""

    class _KvSeqSharding(DefaultSharding):
      """Shards the attention KV sequence, which Splash cannot run without ring attention."""

      def map_axis(self, axis, tensor_name, tensor_type):
        if axis == "kv_length" and tensor_type == TensorType.ACTIVATION:
          return "context"
        return super().map_axis(axis, tensor_name, tensor_type)

    tpu_mesh = _mock_platform_mesh("tpu")
    tpu_mesh.axis_names = ("fsdp", "context", "tensor")
    tpu_mesh.axis_types = (AxisType.Auto,) * 3
    tpu_mesh.shape = {"fsdp": 2, "context": 2, "tensor": 2}
    base = {"attention": "autoselected", "attention_type": "global", "max_target_length": 256}
    auto_cfg = types.SimpleNamespace(**base, kernel_backend="auto", num_query_heads=4, num_kv_heads=2)
    splash_cfg = types.SimpleNamespace(**base, kernel_backend="splash", num_query_heads=4, num_kv_heads=2)

    for cfg in (auto_cfg, splash_cfg):
      assert kernels.attention_from_config(cfg, DefaultSharding(), mesh=tpu_mesh).kernel_backend == KernelBackend.SPLASH

    fallback = kernels.attention_from_config(auto_cfg, _KvSeqSharding(), mesh=tpu_mesh)
    assert fallback.kernel_backend == KernelBackend.JAX
    assert not fallback.allow_xla_fallback
    with pytest.raises(ValueError, match="sharded KV sequence"):
      kernels.attention_from_config(splash_cfg, _KvSeqSharding(), mesh=tpu_mesh)

    odd_heads = {"num_query_heads": 3, "num_kv_heads": 3}
    auto_odd = types.SimpleNamespace(**base, kernel_backend="auto", **odd_heads)
    splash_odd = types.SimpleNamespace(**base, kernel_backend="splash", **odd_heads)
    assert kernels.attention_from_config(auto_odd, DefaultSharding(), mesh=tpu_mesh).kernel_backend == KernelBackend.JAX
    with pytest.raises(ValueError, match="not divisible by head shards"):
      kernels.attention_from_config(splash_odd, DefaultSharding(), mesh=tpu_mesh)

  def test_qwen3_modules_resolve_attention_backend_against_their_mesh(self, monkeypatch):
    """`Qwen3Decoder` -> `Qwen3DecoderLayer` -> `Qwen3Attention` pass their mesh to `attention_from_config`."""
    seen_meshes = []
    resolve = kernels.attention_from_config

    def _recording_resolve(config, sharding=None, **kwargs):
      seen_meshes.append(kwargs.get("mesh"))
      return resolve(config, sharding, **kwargs)

    monkeypatch.setattr(kernels, "attention_from_config", _recording_resolve)
    cfg = types.SimpleNamespace(
        emb_dim=128,
        mlp_dim=256,
        num_decoder_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=64,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        matmul_precision="default",
        normalization_layer_epsilon=1e-6,
        use_qk_norm=True,
        rope_max_timescale=10000.0,
        attention="dot_product",
        attention_type="global",
        float32_logits=True,
    )
    cpu_mesh = _make_cpu_mesh()
    Qwen3Attention(config=cfg, mesh=cpu_mesh, rngs=nnx.Rngs(params=0), sharding=DefaultSharding())
    Qwen3Decoder(config=cfg, mesh=cpu_mesh, rngs=nnx.Rngs(params=0), sharding=DefaultSharding())
    assert len(seen_meshes) == 1 + cfg.num_decoder_layers
    assert all(m is cpu_mesh for m in seen_meshes)
