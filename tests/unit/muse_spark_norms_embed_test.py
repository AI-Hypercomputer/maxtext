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

"""Unit tests for Muse Spark normalizations, module-owned embedding, and MLP."""

import contextlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.layers import embeddings
from maxtext.layers import normalizations
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.models import muse_spark
from tests.utils import muse_spark_test_utils as ms_utils
from tests.utils import test_helpers
import numpy as np
import torch


def _mesh() -> jax.sharding.Mesh:
  return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1, 1, 1)), ("data", "fsdp", "tensor"))


def _init_test_cfg(*args: str) -> Any:
  opts = ["", test_helpers.get_test_config_path(), "decoder_block=muse_spark", "skip_jax_distributed_system=true"]
  return pyconfig.initialize([*opts, *" ".join(args).split()])


def _embed(c: Any, m: Any) -> embeddings.Embed:
  return embeddings.Embed(c.vocab_size, c.emb_dim, config=c, mesh=m, dtype=c.dtype, rngs=nnx.Rngs(42))


def _check_module_fwd_and_grads(
    j_m: Any, r_m: Any, inps: tuple[Any, ...], in_names: tuple[str, ...], dt: Any, tdt: Any, is_fp64: bool, pfx: str
) -> Any:
  """Checks forward parity and input/parameter gradients against reference."""
  n = len(inps)
  j_in, r_in = [jnp.asarray(x, dtype=dt) for x in inps], [torch.from_numpy(x).to(tdt).requires_grad_(True) for x in inps]

  def loss(*a: Any) -> tuple[jax.Array, jax.Array]:
    out = a[-1](*a[:-1])
    return 0.5 * jnp.sum(out**2), out

  (_, out_j), g_j = nnx.value_and_grad(loss, argnums=tuple(range(n + 1)), has_aux=True)(*j_in, j_m)
  (0.5 * torch.sum((out_r := r_m(*r_in)) ** 2)).backward()
  ms_utils.check_e8_parity(np.asarray(out_j), out_r.detach().cpu().numpy(), f"{pfx}_out", is_fp64=is_fp64)
  for g, r, name in zip(g_j[:n], r_in, in_names):
    ms_utils.check_e8_parity(np.asarray(g), r.grad.cpu().numpy(), f"{pfx}_{name}", is_fp64=is_fp64)
  return g_j[n]


class MuseSparkNormsEmbedTest(parameterized.TestCase):
  """Parity and geometry unit tests for Muse Spark norms, MLP, gate, and decoder."""

  @parameterized.named_parameters(("fp32", False), ("fp64", True))
  def test_norms_e8_parity(self, is_fp64: bool) -> None:
    rng, dim, bs, slen, eps = np.random.default_rng(0), 64, 2, 16, 1e-5
    dtype, tdtype = (jnp.float64, torch.float64) if is_fp64 else (jnp.float32, torch.float32)
    with ms_utils.fp64_mode() if is_fp64 else contextlib.nullcontext():
      for j_cls, r_cls, ntype in (
          (normalizations.MuseSparkRMSNorm, ms_utils.ReferenceMuseSparkRMSNorm, "rmsnorm"),
          (normalizations.CenteredRMSNorm, ms_utils.ReferenceCenteredRMSNorm, "centered"),
          (normalizations.InputScaledRMSNorm, ms_utils.ReferenceInputScaledRMSNorm, "input_scaled"),
      ):
        for with_scale in (True, False):
          j_m = j_cls(dim, epsilon=eps, dtype=dtype, weight_dtype=dtype, with_scale=with_scale, rngs=nnx.Rngs(42))
          r_m = r_cls(dim, eps=eps, with_scale=with_scale).to(tdtype)
          if with_scale:
            self.assertEqual(j_m.scale.out_sharding, ("norm",))
            np.testing.assert_array_equal(np.asarray(j_m.scale.value), r_m.weight.detach().cpu().numpy())
            s_val = rng.uniform(0.5, 1.5, size=(dim,)).astype(np.float64 if is_fp64 else np.float32)
            j_m.scale.value, r_m.weight.data = jnp.asarray(s_val, dtype=dtype), torch.from_numpy(s_val).to(tdtype)
          x_np = rng.normal(size=(bs, slen, dim)).astype(np.float64 if is_fp64 else np.float32)
          g_m = _check_module_fwd_and_grads(j_m, r_m, (x_np,), ("gx",), dtype, tdtype, is_fp64, ntype)
          if with_scale:
            ms_utils.check_e8_parity(g_m.scale.value, r_m.weight.grad.cpu().numpy(), f"{ntype}_gscale", is_fp64=is_fp64)

  def test_norms_explicit_sharding(self) -> None:
    devs = jax.devices()[:1]
    emesh = jax.make_mesh((1, 1), ("fsdp", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2, devices=devs)
    ns, ps = jax.sharding.NamedSharding, jax.sharding.PartitionSpec
    in_s, w_s = ns(emesh, ps("fsdp", "tensor", None)), ns(emesh, ps("tensor"))
    out_s = ns(emesh, ps("fsdp", None, "tensor"))
    x_np = np.linspace(0.2, 1.8, 128, dtype=np.float32).reshape((2, 4, 16))
    s_np = np.linspace(0.5, 1.5, 16, dtype=np.float32)
    for cls in (normalizations.MuseSparkRMSNorm, normalizations.CenteredRMSNorm, normalizations.InputScaledRMSNorm):
      for ws in (True, False):
        auto_m = cls(16, with_scale=ws, shard_mode=ShardMode.AUTO, rngs=nnx.Rngs(0))
        if ws:
          auto_m.scale.value = jnp.asarray(s_np)
        ref = np.asarray(auto_m(jnp.asarray(x_np)))
        with jax.set_mesh(emesh):
          exp_m = cls(16, with_scale=ws, shard_mode=ShardMode.EXPLICIT, rngs=nnx.Rngs(0))
          if ws:
            exp_m.scale.value = jax.device_put(jnp.asarray(s_np), w_s)
          out = exp_m(jax.device_put(jnp.asarray(x_np), in_s), out_sharding=out_s)
          self.assertEqual(out.sharding, out_s)
          ms_utils.check_e8_parity(np.asarray(out), ref, f"{cls.__name__}_explicit_{ws}")

  def test_input_scaled_norm_bfloat16_rounding(self) -> None:
    dim, bs, slen, eps = 64, 2, 4, 1e-5
    s_val = np.concatenate([np.full(32, 0.002, dtype=np.float32), np.full(32, 0.5, dtype=np.float32)])
    x_np = np.linspace(0.5, 2.0, num=bs * slen * dim, dtype=np.float32).reshape((bs, slen, dim))
    x_bf16, xt = jnp.asarray(x_np, dtype=jnp.bfloat16), torch.from_numpy(x_np).to(torch.bfloat16)
    kw = {"num_features": dim, "epsilon": eps, "dtype": jnp.bfloat16, "weight_dtype": jnp.bfloat16, "rngs": nnx.Rngs(42)}
    s_bf16, s_torch = jnp.asarray(s_val, dtype=jnp.bfloat16), torch.from_numpy(s_val).to(torch.bfloat16)
    xc = jnp.asarray(x_bf16, jnp.float32)
    for j_cls, r_cls, is_c in (
        (normalizations.CenteredRMSNorm, ms_utils.ReferenceCenteredRMSNorm, True),
        (normalizations.InputScaledRMSNorm, ms_utils.ReferenceInputScaledRMSNorm, False),
    ):
      m_j, m_r = j_cls(**kw), r_cls(dim, eps=eps).to(torch.bfloat16)
      m_j.scale.value, m_r.weight.data = s_bf16, s_torch
      out_j, r_out = m_j(x_bf16), m_r(xt).detach().cpu().float().numpy()
      np.testing.assert_allclose(np.asarray(out_j, np.float32), r_out, rtol=0, atol=1e-4)
      if is_c:
        y = xc * jax.lax.rsqrt(jnp.mean(xc**2, axis=-1, keepdims=True) + eps) * (1.0 + jnp.asarray(s_bf16, jnp.float32))
        d = np.asarray(out_j, np.float32)[..., :32] - np.asarray(y, np.float32)[..., :32]
        self.assertGreater(float(np.max(np.abs(d))), 1e-4)
      else:
        un = xc * jnp.asarray(jnp.asarray(1.0, jnp.bfloat16) + s_bf16, jnp.float32)
        out_un = jnp.asarray(un * jax.lax.rsqrt(jnp.mean(un**2, axis=-1, keepdims=True) + eps), jnp.bfloat16)
        self.assertGreater(float(np.max(np.abs(np.asarray(out_j, np.float32) - np.asarray(out_un, np.float32)))), 2e-4)

  @parameterized.named_parameters(("fp32", False), ("fp64", True))
  def test_residual_gate_e8_parity(self, is_fp64: bool) -> None:
    rng, dim, bs, slen = np.random.default_rng(1), 64, 2, 16
    dtype, tdtype = (jnp.float64, torch.float64) if is_fp64 else (jnp.float32, torch.float32)
    bf16_g = muse_spark.MuseSparkResidualGate(dim, weight_dtype=jnp.bfloat16, rngs=nnx.Rngs(9))
    self.assertEqual((bf16_g.gate.value.dtype, bf16_g.gate.out_sharding), (jnp.float32, ("norm",)))
    np.testing.assert_array_equal(np.asarray(bf16_g.gate.value), np.zeros(dim, dtype=np.float32))
    with ms_utils.fp64_mode() if is_fp64 else contextlib.nullcontext():
      for zero_init in (True, False):
        j_g = muse_spark.MuseSparkResidualGate(dim, temperature=0.3, weight_dtype=dtype, rngs=nnx.Rngs(123))
        r_g = ms_utils.ReferenceMuseSparkResidualGate(dim, temperature=0.3).to(tdtype)
        if not zero_init:
          g_val = rng.uniform(-1.0, 4.0, size=(dim,)).astype(np.float64 if is_fp64 else np.float32)
          j_g.gate.value, r_g.gate.data = jnp.asarray(g_val, dtype=dtype), torch.from_numpy(g_val).to(tdtype)
        r_np, s_np = (rng.normal(size=(bs, slen, dim)).astype(np.float64 if is_fp64 else np.float32) for _ in (0, 1))
        g_m = _check_module_fwd_and_grads(j_g, r_g, (r_np, s_np), ("gres", "gsub"), dtype, tdtype, is_fp64, "gate")
        ms_utils.check_e8_parity(g_m.gate.value, r_g.gate.grad.cpu().numpy(), "gate_gleaf", is_fp64=is_fp64)

  @parameterized.named_parameters(("fp32", False), ("fp64", True))
  def test_mlp_e8_parity(self, is_fp64: bool) -> None:
    rng, dim, idim, bs, slen = np.random.default_rng(2), 64, 128, 2, 16
    dtype, tdtype = (jnp.float64, torch.float64) if is_fp64 else (jnp.float32, torch.float32)
    with ms_utils.fp64_mode() if is_fp64 else contextlib.nullcontext():
      cfg = _init_test_cfg(f"run_name=mlp dtype=float32 base_emb_dim={dim} base_mlp_dim={idim} matmul_precision=highest")
      j_mlp = muse_spark.MuseSparkTextMLP(config=cfg, rngs=nnx.Rngs(777))
      r_mlp = ms_utils.ReferenceMuseSparkTextMLP(dim, idim).to(tdtype)
      for rp, jp in ((r_mlp.gate_proj, j_mlp.wi_0), (r_mlp.up_proj, j_mlp.wi_1), (r_mlp.down_proj, j_mlp.wo)):
        rp.weight.data.copy_(torch.from_numpy(np.asarray(jp.kernel.value).T).to(tdtype))
      x_np = rng.normal(size=(bs, slen, dim)).astype(np.float64 if is_fp64 else np.float32)
      g_m = _check_module_fwd_and_grads(j_mlp, r_mlp, (x_np,), ("gx",), dtype, tdtype, is_fp64, "mlp")
      r_p = (r_mlp.gate_proj, r_mlp.up_proj, r_mlp.down_proj)
      for n, rp, jp in zip(("gwi0", "gwi1", "gwo"), r_p, (g_m.wi_0, g_m.wi_1, g_m.wo)):
        ms_utils.check_e8_parity(jp.kernel.value, rp.weight.grad.cpu().numpy().T, f"mlp_{n}", is_fp64=is_fp64)

  @parameterized.named_parameters(("fp32", False), ("fp64", True))
  def test_live_nnx_decoder_muse_spark_unscanned(self, is_fp64: bool) -> None:
    rng, mesh = np.random.default_rng(42), _mesh()
    dtype, tdtype = (jnp.float64, torch.float64) if is_fp64 else (jnp.float32, torch.float32)
    with ms_utils.fp64_mode() if is_fp64 else contextlib.nullcontext():
      cfg = _init_test_cfg(
          "run_name=t_live dtype=float32 scan_layers=false base_num_decoder_layers=2 base_emb_dim=64 base_mlp_dim=128 "
          "base_num_query_heads=4 base_num_kv_heads=2 head_dim=16 vocab_size=256 max_target_length=16 "
          "image_token_id=101 video_token_id=102 audio_token_id=103 output_multiplier=0.1767766952966369 "
          "final_logits_soft_cap=20.0 first_num_dense_layers=1 enable_dropout=false logits_dot_in_fp32=false "
          "float32_qk_product=false cast_logits_to_fp32=true matmul_precision=highest attention=dot_product "
          "normalization_layer_epsilon=1e-5 post_norm_eps=1e-8 residual_gate_temperature=0.5"
      )
      with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        decoder, shared_emb = NNXDecoder(cfg, mesh, rngs=nnx.Rngs(42)), _embed(cfg, mesh)
        a0, a1 = (decoder.layers_0.self_attention, decoder.layers_1.self_attention)
        self.assertEqual((a0.is_nope_layer, a1.is_nope_layer), (False, True))

        toks_in = jnp.array([[101, 102, 103, 0, 10]], dtype=jnp.int32)
        embedded = decoder.embed_tokens(shared_emb, toks_in, model_mode=MODEL_MODE_TRAIN)
        np.testing.assert_allclose(np.asarray(embedded[0, :3]), np.broadcast_to(np.asarray(embedded[0, 3:4]), (3, 64)))
        self.assertFalse(np.allclose(np.asarray(embedded[0, 4]), np.asarray(embedded[0, 3])))
        e0 = torch.from_numpy(np.asarray(shared_emb.embedding.value[:1])).to(tdtype)
        r_emb = ms_utils.ReferenceMuseSparkRMSNorm(cfg.emb_dim, eps=cfg.normalization_layer_epsilon, with_scale=False)(e0)
        ms_utils.check_e8_parity(embedded[0, 3:4], r_emb.detach().cpu().numpy(), "embed_row0", is_fp64=is_fp64)

        cfg_d = _init_test_cfg("run_name=t_d scan_layers=false base_num_decoder_layers=1 vocab_size=256 base_emb_dim=64")
        t_d = jnp.array([[200092, 200091, 200038, 0, 255]], dtype=jnp.int32)
        d_d, e_d = NNXDecoder(cfg_d, mesh, rngs=nnx.Rngs(42)), _embed(cfg_d, mesh)
        emb_def = d_d.embed_tokens(e_d, t_d, model_mode=MODEL_MODE_TRAIN)
        np.testing.assert_allclose(np.asarray(emb_def[0, :3]), np.broadcast_to(np.asarray(emb_def[0, 3:4]), (3, 64)))
        self.assertFalse(np.allclose(np.asarray(emb_def[0, 0]), np.asarray(emb_def[0, 4])))

        for lyr in (decoder.layers_0, decoder.layers_1):
          scales = (lyr.input_layernorm.scale, lyr.pre_ffn_norm.scale, lyr.post_ffn_norm.scale)
          for m in (*scales, lyr.post_attn_gate.gate, lyr.post_ffn_gate.gate):
            m.value = jnp.asarray(rng.uniform(0.1, 0.5, size=(cfg.emb_dim,)), dtype=dtype)
        decoder.decoder_norm.scale.value = jnp.asarray(rng.uniform(0.5, 1.2, size=(cfg.emb_dim,)), dtype=dtype)
        ref_dec = ms_utils.ReferenceMuseSparkDecoder().to(tdtype)

        def _pairs(d: Any, e: Any, r: Any) -> list[tuple[Any, Any, str]]:
          p = [(e.embedding, r.shared_emb, "g_shared_emb"), (d.logits_dense.kernel, r.logits_dense_w, "g_logits_dense")]
          p += [(d.decoder_norm.scale, r.decoder_norm.weight, "g_decoder_norm")]
          for i, (s, m) in enumerate(zip((d.layers_0, d.layers_1), (decoder.layers_0, decoder.layers_1))):
            p += [(j, q, f"l{i}_g_{n}") for j, q, n in ms_utils.get_layer_param_pairs(s, r.layers[i], m.is_dense_layer)]
          return p

        for j_p, r_p, _ in _pairs(decoder, shared_emb, ref_dec):
          r_p.data.copy_(torch.from_numpy(np.asarray(j_p.value)).to(tdtype))
        toks = jnp.array([[101, 102, 103, *range(0, 130, 10)]], dtype=jnp.int32)
        positions, segment_ids = jnp.arange(16, dtype=jnp.int32).reshape(1, 16), jnp.ones((1, 16), dtype=jnp.int32)

        def forward_loss(emb: embeddings.Embed, dec: NNXDecoder) -> tuple[jax.Array, jax.Array]:
          out = dec(emb, toks, positions, segment_ids, deterministic=True, model_mode=MODEL_MODE_TRAIN)[0]
          return 0.5 * jnp.sum(out**2), out

        (_, logits), (g_emb, g_dec) = nnx.value_and_grad(forward_loss, argnums=(0, 1), has_aux=True)(shared_emb, decoder)
        self.assertEqual((logits.dtype, logits.shape), (dtype, (1, 16, cfg.vocab_size)))
        (0.5 * torch.sum((ref_logits := ref_dec(torch.from_numpy(np.asarray(toks)))) ** 2)).backward()
        ms_utils.check_e8_parity(logits, ref_logits.detach().cpu().numpy(), "decoder_logits", is_fp64=is_fp64)
        pairs = _pairs(g_dec, g_emb, ref_dec)
        self.assertLen(jax.tree_util.tree_leaves(nnx.state((g_emb, g_dec), nnx.Param)), len(pairs))
        for j, r, n in pairs:
          ms_utils.check_e8_parity(j.value, r.grad.cpu().numpy(), n, is_fp64=is_fp64)

  @parameterized.named_parameters(("dense_1", 1), ("dense_0_fully_moe", 0))
  def test_pyconfig_initialization_dense_layers(self, dense_layers: int) -> None:
    cfg = _init_test_cfg(
        f"run_name=cfg first_num_dense_layers={dense_layers} num_experts=256 base_mlp_dim=16384 base_moe_mlp_dim=4096 "
        "interleave_moe_layer_step=1 inhomogeneous_layer_cycle_interval=1"
    )
    got = (cfg.decoder_block, cfg.first_num_dense_layers, cfg.base_mlp_dim, cfg.base_moe_mlp_dim)
    exp = (DecoderBlockType.MUSE_SPARK, dense_layers, 16384, 4096, 1e-8, 0.3)
    self.assertEqual((*got, cfg.post_norm_eps, cfg.residual_gate_temperature), exp)

  def test_scanned_layers_raises_not_implemented(self) -> None:
    cfg = _init_test_cfg("run_name=t_scan_fail scan_layers=true base_num_decoder_layers=2")
    self.assertTrue(cfg.scan_layers)
    with self.assertRaisesRegex(NotImplementedError, r"scan_layers=True.*decoder_block=muse_spark"):
      NNXDecoder(cfg, mesh=_mesh(), rngs=nnx.Rngs(0))


if __name__ == "__main__":
  absltest.main()
