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

"""Unit and multi-device parity tests for `maxtext.m3.core.kernels`."""

import types
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.core import kernels
from maxtext.m3.core.sharding import DefaultSharding, Sharding, TensorType, create_mesh
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Model
from maxtext.utils import maxtext_utils


def _make_test_mesh() -> jax.sharding.Mesh:
  """Builds an `(fsdp, tensor)` mesh sized dynamically from available host devices."""
  n_dev = len(jax.devices())
  tp = 2 if n_dev >= 2 and n_dev % 2 == 0 else 1
  fsdp = n_dev // tp
  return create_mesh(ici={"fsdp": fsdp, "tensor": tp})


def _splash_kernel_for_test(
    sharding: Sharding | None = None,
    *,
    causal: bool = True,
    sliding_window: int | None = None,
    attn_logits_soft_cap: float | None = None,
    gqa_head_stacking: bool = False,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
    max_logit_const: float | None = None,
    use_base2_exp: bool = False,
) -> kernels.AttentionKernel:
  s = sharding or DefaultSharding()
  q_spec = s("attention_kernel_q", ("batch", "heads", "length", "head_dim"), TensorType.ACTIVATION)
  kv_spec = s("attention_kernel_kv", ("batch", "kv_heads", "length", "head_dim"), TensorType.ACTIVATION)
  splash_cfg = kernels.SplashConfig(
      block_q=128,
      block_kv=128,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=128,
      block_kv_dkv_compute=128,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      use_base2_exp=use_base2_exp,
      max_logit_const=max_logit_const,
  )
  return kernels.AttentionKernel(
      backend="splash",
      batch=q_spec[0],
      heads=q_spec[1],
      kv_heads=kv_spec[1],
      length=q_spec[2],
      kv_length=kv_spec[2],
      causal=causal,
      sliding_window=sliding_window,
      attn_logits_soft_cap=attn_logits_soft_cap,
      float32_logits=True,
      gqa_head_stacking=gqa_head_stacking,
      splash=splash_cfg,
  )


class TestSplashVsXlaParity:
  """Forward and backward leaf-gradient parity between `xla` and `splash` backends."""

  def test_causal_gqa_seq_minor_and_max_logit_fwd_bwd_parity(self):
    mesh = _make_test_mesh()
    batch = max(4, len(jax.devices()))
    seq_len, n_q, n_kv, head_dim = 256, 8, 2, 64
    rng = jax.random.key(11)
    kq, kk, kv, kc = jax.random.split(rng, 4)
    q = jax.random.normal(kq, (batch, seq_len, n_q, head_dim), dtype=jnp.float32) * (head_dim**-0.5)
    k = jax.random.normal(kk, (batch, seq_len, n_kv, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, seq_len, n_kv, head_dim), dtype=jnp.float32)
    cotangent = jax.random.normal(kc, (batch, seq_len, n_q, head_dim), dtype=jnp.float32)

    xla_kern = kernels.AttentionKernel(backend="xla", causal=True, float32_logits=True)
    splash_const_kern = _splash_kernel_for_test(
        causal=True,
        q_layout="SEQ_MINOR",
        k_layout="SEQ_MINOR",
        v_layout="SEQ_MINOR",
        max_logit_const=25.0,
        use_base2_exp=False,
    )
    splash_dyn_kern = _splash_kernel_for_test(
        causal=True,
        q_layout="SEQ_MINOR",
        k_layout="SEQ_MINOR",
        v_layout="SEQ_MINOR",
        use_base2_exp=False,
    )
    dyn_max_logit = jnp.full((n_q,), 25.0, dtype=jnp.float32)

    with jax.set_mesh(mesh):
      with pytest.raises(ValueError, match="max_logit"):
        splash_const_kern(q, k, v, max_logit_value=25.0)

      def loss_xla(q_in, k_in, v_in):
        out = xla_kern(q_in, k_in, v_in)
        return jnp.sum(out * cotangent), out

      def loss_splash_const(q_in, k_in, v_in):
        out = splash_const_kern(q_in, k_in, v_in)
        return jnp.sum(out * cotangent), out

      def loss_splash_dyn(q_in, k_in, v_in, mlv):
        out = splash_dyn_kern(q_in, k_in, v_in, max_logit_value=mlv)
        return jnp.sum(out * cotangent), out

      (_, out_xla), grads_xla = jax.jit(jax.value_and_grad(loss_xla, argnums=(0, 1, 2), has_aux=True))(q, k, v)
      (_, out_const), grads_const = jax.jit(jax.value_and_grad(loss_splash_const, argnums=(0, 1, 2), has_aux=True))(
          q, k, v
      )
      (_, out_dyn), grads_dyn = jax.jit(jax.value_and_grad(loss_splash_dyn, argnums=(0, 1, 2), has_aux=True))(
          q, k, v, dyn_max_logit
      )

    for out_sp, grads_sp in ((out_const, grads_const), (out_dyn, grads_dyn)):
      np.testing.assert_allclose(np.asarray(out_sp), np.asarray(out_xla), rtol=1e-4, atol=1e-4)
      for g_sp, g_xl in zip(grads_sp, grads_xla):
        np.testing.assert_allclose(np.asarray(g_sp), np.asarray(g_xl), rtol=2e-4, atol=2e-4)

  def test_sliding_window_and_negative_control(self):
    mesh = _make_test_mesh()
    batch = max(4, len(jax.devices()))
    seq_len, window, n_q, n_kv, head_dim = 256, 128, 4, 2, 128
    rng = jax.random.key(23)
    kq, kk, kv, kc = jax.random.split(rng, 4)
    q = jax.random.normal(kq, (batch, seq_len, n_q, head_dim), dtype=jnp.float32) * (head_dim**-0.5)
    k = jax.random.normal(kk, (batch, seq_len, n_kv, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, seq_len, n_kv, head_dim), dtype=jnp.float32)
    cotangent = jax.random.normal(kc, (batch, seq_len, n_q, head_dim), dtype=jnp.float32)

    xla_swa = kernels.AttentionKernel(backend="xla", causal=True, sliding_window=window, float32_logits=True)
    splash_swa = _splash_kernel_for_test(causal=True, sliding_window=window, use_base2_exp=False)
    splash_global = _splash_kernel_for_test(causal=True, sliding_window=None, use_base2_exp=False)

    with jax.set_mesh(mesh):
      (_, out_xla), grads_xla = jax.jit(
          jax.value_and_grad(
              lambda qi, ki, vi: (jnp.sum(xla_swa(qi, ki, vi) * cotangent), xla_swa(qi, ki, vi)),
              argnums=(0, 1, 2),
              has_aux=True,
          )
      )(q, k, v)
      (_, out_splash), grads_splash = jax.jit(
          jax.value_and_grad(
              lambda qi, ki, vi: (jnp.sum(splash_swa(qi, ki, vi) * cotangent), splash_swa(qi, ki, vi)),
              argnums=(0, 1, 2),
              has_aux=True,
          )
      )(q, k, v)

      # Negative control: perturb early keys (index 0..15) and inspect tail queries (index 200..255),
      # which lie strictly outside the 128-token causal sliding window.
      k_perturbed = k.at[:, :16, :, :].add(5.0)
      out_swa_pert = jax.jit(splash_swa)(q, k_perturbed, v)
      out_global_base = jax.jit(splash_global)(q, k, v)
      out_global_pert = jax.jit(splash_global)(q, k_perturbed, v)

    np.testing.assert_allclose(np.asarray(out_splash), np.asarray(out_xla), rtol=1e-4, atol=1e-4)
    for g_sp, g_xl in zip(grads_splash, grads_xla):
      np.testing.assert_allclose(np.asarray(g_sp), np.asarray(g_xl), rtol=2e-4, atol=2e-4)

    np.testing.assert_allclose(np.asarray(out_swa_pert[:, 200:]), np.asarray(out_splash[:, 200:]), rtol=0.0, atol=1e-6)
    global_tail_diff = float(np.max(np.abs(np.asarray(out_global_pert[:, 200:]) - np.asarray(out_global_base[:, 200:]))))
    assert global_tail_diff > 1e-2, f"Expected global attention negative control to fire, got {global_tail_diff}"

  def test_gqa_head_stacking_packed_segments_and_padding_isolation(self):
    mesh = _make_test_mesh()
    batch = max(4, len(jax.devices()))
    seq_len, n_q, n_kv, head_dim = 256, 8, 2, 64
    rng = jax.random.key(37)
    kq, kk, kv, kc = jax.random.split(rng, 4)
    q = jax.random.normal(kq, (batch, seq_len, n_q, head_dim), dtype=jnp.float32) * (head_dim**-0.5)
    k = jax.random.normal(kk, (batch, seq_len, n_kv, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, seq_len, n_kv, head_dim), dtype=jnp.float32)
    cotangent = jax.random.normal(kc, (batch, seq_len, n_q, head_dim), dtype=jnp.float32)

    seg_row = jnp.concatenate(
        [
            jnp.ones((96,), dtype=jnp.int32),
            jnp.full((112,), 2, dtype=jnp.int32),
            jnp.zeros((48,), dtype=jnp.int32),
        ]
    )
    segment_ids = jnp.broadcast_to(seg_row[None, :], (batch, seq_len))

    xla_kern = kernels.AttentionKernel(backend="xla", causal=True, sliding_window=128, float32_logits=True)
    stacked_splash = _splash_kernel_for_test(
        causal=True,
        sliding_window=128,
        gqa_head_stacking=True,
        q_layout="SEQ_MINOR",
        k_layout="SEQ_MINOR",
        v_layout="SEQ_MINOR",
        use_base2_exp=False,
    )

    with jax.set_mesh(mesh):
      (_, out_xla), grads_xla = jax.jit(
          jax.value_and_grad(
              lambda qi, ki, vi: (
                  jnp.sum(xla_kern(qi, ki, vi, segment_ids=segment_ids) * cotangent),
                  xla_kern(qi, ki, vi, segment_ids=segment_ids),
              ),
              argnums=(0, 1, 2),
              has_aux=True,
          )
      )(q, k, v)
      (_, out_stacked), grads_stacked = jax.jit(
          jax.value_and_grad(
              lambda qi, ki, vi: (
                  jnp.sum(stacked_splash(qi, ki, vi, segment_ids=segment_ids) * cotangent),
                  stacked_splash(qi, ki, vi, segment_ids=segment_ids),
              ),
              argnums=(0, 1, 2),
              has_aux=True,
          )
      )(q, k, v)

      # Negative control: perturbing segment-1 keys must not alter segment-2 outputs when segment_ids is passed,
      # whereas omitting segment_ids changes segment-2 outputs inside the 128-token window.
      k_seg1_pert = k.at[:, 64:96, :, :].add(4.0)
      out_seg_pert = jax.jit(lambda qi, ki, vi: stacked_splash(qi, ki, vi, segment_ids=segment_ids))(q, k_seg1_pert, v)
      out_unseg_base = jax.jit(stacked_splash)(q, k, v)
      out_unseg_pert = jax.jit(stacked_splash)(q, k_seg1_pert, v)

    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads_stacked)
    np.testing.assert_allclose(np.asarray(out_stacked), np.asarray(out_xla), rtol=1e-4, atol=1e-4)
    for g_sp, g_xl in zip(grads_stacked, grads_xla):
      np.testing.assert_allclose(np.asarray(g_sp), np.asarray(g_xl), rtol=2e-4, atol=2e-4)

    # Padded positions (segment_ids == 0) must produce exact zeros in output and dQ.
    np.testing.assert_array_equal(np.asarray(out_stacked[:, 208:]), 0.0)
    np.testing.assert_array_equal(np.asarray(grads_stacked[0][:, 208:]), 0.0)

    # Segment boundary isolation + firing negative control.
    np.testing.assert_allclose(np.asarray(out_seg_pert[:, 96:208]), np.asarray(out_stacked[:, 96:208]), atol=1e-6)
    unseg_diff = float(np.max(np.abs(np.asarray(out_unseg_pert[:, 96:140]) - np.asarray(out_unseg_base[:, 96:140]))))
    assert unseg_diff > 1e-2, f"Expected unsegmented negative control to fire, got {unseg_diff}"

  def test_soft_cap_sinks_and_asymmetric_value_head_dim(self):
    mesh = _make_test_mesh()
    batch = max(4, len(jax.devices()))
    seq_len, n_q, n_kv, d_qk, d_v = 128, 4, 2, 128, 64
    rng = jax.random.key(53)
    kq, kk, kv, ks, kc = jax.random.split(rng, 5)
    q = jax.random.normal(kq, (batch, seq_len, n_q, d_qk), dtype=jnp.float32) * 0.2
    k = jax.random.normal(kk, (batch, seq_len, n_kv, d_qk), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, seq_len, n_kv, d_v), dtype=jnp.float32)
    sinks = jax.random.normal(ks, (n_q,), dtype=jnp.float32)
    cotangent = jax.random.normal(kc, (batch, seq_len, n_q, d_v), dtype=jnp.float32)

    xla_kern = kernels.AttentionKernel(backend="xla", causal=True, attn_logits_soft_cap=30.0, float32_logits=True)
    splash_kern = _splash_kernel_for_test(causal=True, attn_logits_soft_cap=30.0, use_base2_exp=False)

    with jax.set_mesh(mesh):
      (_, out_xla), grads_xla = jax.jit(
          jax.value_and_grad(
              lambda qi, ki, vi, si: (
                  jnp.sum(xla_kern(qi, ki, vi, sinks=si) * cotangent),
                  xla_kern(qi, ki, vi, sinks=si),
              ),
              argnums=(0, 1, 2, 3),
              has_aux=True,
          )
      )(q, k, v, sinks)
      (_, out_splash), grads_splash = jax.jit(
          jax.value_and_grad(
              lambda qi, ki, vi, si: (
                  jnp.sum(splash_kern(qi, ki, vi, sinks=si) * cotangent),
                  splash_kern(qi, ki, vi, sinks=si),
              ),
              argnums=(0, 1, 2, 3),
              has_aux=True,
          )
      )(q, k, v, sinks)

    assert out_splash.shape == (batch, seq_len, n_q, d_v)
    np.testing.assert_allclose(np.asarray(out_splash), np.asarray(out_xla), rtol=2e-4, atol=2e-4)
    for g_sp, g_xl in zip(grads_splash, grads_xla):
      np.testing.assert_allclose(np.asarray(g_sp), np.asarray(g_xl), rtol=3e-4, atol=3e-4)


class TestLiveQwen3AdoptionAndScan:
  """Verifies live `Qwen3Model` adoption under `ShardMode.AUTO` + `ScannedLayers` + `MODEL_FLOPS_REGISTRY`."""

  def test_qwen3_scanned_auto_mesh_xla_vs_splash_param_grad_parity(self):
    mesh = _make_test_mesh()
    base_kwargs = {
        "run_name": "qwen3_kernels_test",
        "model_name": "qwen3-0.6b",
        "override_model_config": True,
        "use_m3_model": True,
        "base_emb_dim": 128,
        "base_num_query_heads": 8,
        "base_num_kv_heads": 2,
        "head_dim": 64,
        "base_mlp_dim": 256,
        "base_num_decoder_layers": 2,
        "vocab_size": 256,
        "max_target_length": 128,
        "per_device_batch_size": 1.0,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "float32_logits": True,
        "sa_use_base2_exp": False,
        "scan_layers": True,
        "enable_checkpointing": False,
        "skip_jax_distributed_system": True,
    }
    cfg_xla = pyconfig.initialize(["", "src/maxtext/configs/base.yml"], attention="dot_product", **base_kwargs)
    cfg_splash = pyconfig.initialize(
        ["", "src/maxtext/configs/base.yml"],
        attention="flash",
        sa_gqa_head_stacking=True,
        sa_q_layout="SEQ_MINOR",
        sa_k_layout="SEQ_MINOR",
        sa_v_layout="SEQ_MINOR",
        **base_kwargs,
    )

    batch_size = max(4, len(jax.devices()))
    tokens = jax.random.randint(jax.random.key(0), (batch_size, 128), 0, 256, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32)[None, :], (batch_size, 128))
    cotangent = jax.random.normal(jax.random.key(1), (batch_size, 128, 256), dtype=jnp.float32)

    with jax.set_mesh(mesh):
      model_xla = m3_scan.maybe_apply_scan(Qwen3Model(cfg_xla, mesh=mesh, rngs=nnx.Rngs(7)), cfg_xla)
      model_splash = m3_scan.maybe_apply_scan(Qwen3Model(cfg_splash, mesh=mesh, rngs=nnx.Rngs(7)), cfg_splash)
      assert model_xla.mesh is mesh and model_splash.mesh is mesh
      assert m3_models.create_model(cfg_xla, mesh=mesh, rngs=nnx.Rngs(7)).mesh is mesh

      paths_xla = [p for p, _ in jax.tree_util.tree_leaves_with_path(nnx.state(model_xla, nnx.Param))]
      paths_splash = [p for p, _ in jax.tree_util.tree_leaves_with_path(nnx.state(model_splash, nnx.Param))]
      assert paths_xla == paths_splash

      @nnx.jit
      def step_grad(m, tok, pos, cot):
        def loss_fn(mod):
          logits = mod(tok, pos)
          return jnp.sum(logits.astype(jnp.float32) * cot), logits

        (_, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
        return logits, nnx.state(grads, nnx.Param)

      logits_xla, grads_xla = step_grad(model_xla, tokens, positions, cotangent)
      logits_splash, grads_splash = step_grad(model_splash, tokens, positions, cotangent)

    np.testing.assert_allclose(np.asarray(logits_splash), np.asarray(logits_xla), rtol=2e-4, atol=2e-4)
    for (path, g_sp), (_, g_xl) in zip(
        jax.tree_util.tree_leaves_with_path(grads_splash),
        jax.tree_util.tree_leaves_with_path(grads_xla),
    ):
      np.testing.assert_allclose(
          np.asarray(g_sp[...]),
          np.asarray(g_xl[...]),
          rtol=5e-4,
          atol=5e-4,
          err_msg=f"Gradient mismatch at leaf {path}",
      )

  @pytest.mark.parametrize("model_name", ["qwen3-0.6b", "qwen3-vl-2b"])
  def test_qwen3_flops_registry_matches_legacy(self, model_name: str):
    assert model_name in m3_models.MODEL_FLOPS_REGISTRY
    common_kw = {
        "run_name": "flops_parity",
        "model_name": model_name,
        "per_device_batch_size": 1.0,
        "max_target_length": 512,
        "scan_layers": False,
        "enable_checkpointing": False,
        "skip_jax_distributed_system": True,
    }
    cfg_m3 = pyconfig.initialize(["", "src/maxtext/configs/base.yml"], use_m3_model=True, **common_kw)
    cfg_legacy = pyconfig.initialize(["", "src/maxtext/configs/base.yml"], use_m3_model=False, **common_kw)
    m3_flops = maxtext_utils.calculate_tflops_training_per_device(cfg_m3, log=False)
    legacy_flops = maxtext_utils.calculate_tflops_training_per_device(cfg_legacy, log=False)
    np.testing.assert_allclose(m3_flops, legacy_flops, rtol=1e-12, atol=1e-12)


class TestFailClosedValidation:
  """Ensures unsupported Splash configurations fail closed rather than silently falling back to XLA."""

  def test_short_or_unaligned_sequence_raises_on_splash(self):
    bad_cfg = types.SimpleNamespace(
        attention="flash",
        attention_type="global",
        max_target_length=96,
    )
    with pytest.raises(ValueError, match="multiple of 128"):
      kernels.validate_attention_config(bad_cfg)

    mesh = _make_test_mesh()
    splash_kern = _splash_kernel_for_test()
    q = jnp.ones((4, 64, 4, 64), dtype=jnp.float32)
    k = jnp.ones((4, 64, 2, 64), dtype=jnp.float32)
    with jax.set_mesh(mesh):
      with pytest.raises(ValueError, match="multiple of 128"):
        splash_kern(q, k, k)

  def test_invalid_layout_and_invalid_gqa_stacking_raise(self):
    with pytest.raises(ValueError, match="sa_q_layout"):
      kernels.validate_attention_config(
          types.SimpleNamespace(attention="flash", attention_type="global", max_target_length=256, sa_q_layout="BAD")
      )
    with pytest.raises(ValueError, match="head_dim == 64"):
      kernels.validate_attention_config(
          types.SimpleNamespace(
              attention="flash",
              attention_type="global",
              max_target_length=256,
              sa_gqa_head_stacking=True,
              head_dim=128,
              num_query_heads=8,
              num_kv_heads=2,
          )
      )
    with pytest.raises(ValueError, match="even GQA group size"):
      kernels.validate_attention_config(
          types.SimpleNamespace(
              attention="flash",
              attention_type="global",
              max_target_length=256,
              sa_gqa_head_stacking=True,
              head_dim=64,
              num_query_heads=6,
              num_kv_heads=2,
          )
      )

  def test_sharded_kv_sequence_and_missing_mesh_raise(self):
    splash_kern = _splash_kernel_for_test()
    q = jnp.ones((4, 128, 4, 64), dtype=jnp.float32)
    k = jnp.ones((4, 128, 2, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="active mesh"):
      splash_kern(q, k, k)

    n_dev = len(jax.devices())
    if n_dev >= 2:
      mesh = create_mesh(ici={"fsdp": n_dev // 2, "tensor": 2})
      ring_kern = kernels.AttentionKernel(
          backend="splash",
          batch="fsdp",
          heads=None,
          kv_heads=None,
          length="tensor",
          kv_length="tensor",
      )
      with jax.set_mesh(mesh):
        with pytest.raises(NotImplementedError, match="ring attention"):
          ring_kern(q, k, k)

  def test_xla_validates_max_logit_value_and_preserves_bf16_fast_path(self):
    q = jnp.ones((2, 128, 4, 64), dtype=jnp.bfloat16)
    k = jnp.ones((2, 128, 2, 64), dtype=jnp.bfloat16)
    v = jnp.ones((2, 128, 2, 64), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match="max_logit_value must be > 0"):
      kernels.dot_product_attention(q, k, v, kernel=kernels.XLA_ATTENTION, max_logit_value=-1.0)
    with pytest.raises(ValueError, match="scalar or have shape"):
      kernels.dot_product_attention(
          q, k, v, kernel=kernels.XLA_ATTENTION, max_logit_value=jnp.ones((2, 4), dtype=jnp.float32)
      )
    kern_const = kernels.AttentionKernel(backend="xla", splash=kernels.SplashConfig(max_logit_const=30.0))
    with pytest.raises(ValueError, match="max_logit_const"):
      kern_const(q, k, v, max_logit_value=20.0)

    hlo_xla = jax.jit(kernels.XLA_ATTENTION).lower(q, k, v).as_text()
    assert "f32[2,128,4,64]" not in hlo_xla
    assert "f32[2,128,2,64]" not in hlo_xla
    assert "pred[2,1,128,128]" not in hlo_xla

    cfg = types.SimpleNamespace(
        emb_dim=128,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=64,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        matmul_precision="default",
        normalization_layer_epsilon=1e-6,
        use_qk_norm=True,
        rope_max_timescale=10000.0,
        attention="dot_product",
        attention_type="global",
        float32_logits=True,
    )
    attn = Qwen3Attention(config=cfg, rngs=nnx.Rngs(params=0))
    x = jnp.ones((2, 128, 128), dtype=jnp.bfloat16)
    pos = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32)[None, :], (2, 128))
    hlo_qwen = (
        nnx.jit(lambda m, inp, p: m(inp, decoder_positions=p, decoder_segment_ids=None)).lower(attn, x, pos).as_text()
    )
    assert "pred[2,1,128,128]" not in hlo_qwen
