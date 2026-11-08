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

"""CPU parity tests of the fused 3D-layout expert path (moe_use_fused_gmm3d).

On CPU the fused path runs its pure-JAX reference emulation of the Lineage
gmmv2 kernels (same quantization points, dense XLA ops), so these tests check
the integration: routing / index conventions, the fused gate parameter, the
SwiGLU + routing-coefficient placement, the top-k combine and every gradient
(tokens, fused gate weight, linear weight, router weights) against the unfused
megablox / ragged_dot path of RoutedMoE, unquantized (tight) and in fp8 (within
quantization noise).

The reference emulation does not read the kernel-only options (prequant_dy,
early_dwo_rs, sc_routed_gather, exact_topk_*, inv_perm_combine,
relayout_x_2d_to_3d, fused_quant_relayout, wgrad_schedule_mode), so no test
here compares module runs that differ only in them. On CPU the new Pallas
kernels are checked in interpret mode, and the kernel path is only lowered
ahead of time for TPU7x. Kernel-versus-reference numerics and the bit-identity
of the schedule modes are checked on TPU in tests/moe_fused_gmm3d_tpu_test.py.

Run with 4 CPU devices: XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""


import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers import moe_fused_gmm3d
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import qwix


_E, _K, _D, _F = 8, 2, 256, 128


def _rel_err(a, b):
  a = np.asarray(a, np.float32)
  b = np.asarray(b, np.float32)
  denom = np.linalg.norm(b)
  return float(np.linalg.norm(a - b) / denom) if denom > 0 else float(np.max(np.abs(a - b)))


def _max_abs(a, b):
  return float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))


def _random_routing(key, num_tokens, num_experts, top_k):
  """Returns (weights [N, k] f32, sort_indices [N*k] int32, group_sizes [E] int32)."""
  logits = jax.random.normal(key, (num_tokens, num_experts), jnp.float32)
  top_w, top_i = jax.lax.top_k(logits, top_k)
  weights = jax.nn.softmax(top_w, axis=-1)
  flat = top_i.reshape(-1)
  sort_indices = jnp.argsort(flat).astype(jnp.int32)
  group_sizes = jnp.bincount(flat, length=num_experts).astype(jnp.int32)
  return weights, sort_indices, group_sizes, top_i


def _make_config(fused, dtype, quantized, **overrides):
  """Builds a small Qwen3-MoE config; `fused` selects `moe_use_fused_gmm3d`, `quantized` adds fp8 Qwix."""
  kwargs = {
      "run_name": "moe_fused_gmm3d_test",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "hardware": "cpu",
      "decoder_block": "qwen3_moe",
      "base_emb_dim": _D,
      "base_mlp_dim": _F,
      "base_moe_mlp_dim": _F,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 64,
      "num_experts": _E,
      "num_experts_per_tok": _K,
      "norm_topk_prob": True,
      "mlp_activations": ["silu", "linear"],
      "max_target_length": 64,
      "per_device_batch_size": 1,
      "ici_fsdp_parallelism": 4,
      "dtype": dtype,
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "sparse_matmul": True,
      "megablox": True,
      "use_tokamax_gmm": True,
      "use_gmm_v2": True,
      "use_custom_sort_vjp": True,
      "moe_flat_fsdp_weights": True,
      "moe_use_direct_token_gather": True,
      "moe_use_fused_gmm3d": fused,
      "load_balance_loss_weight": 0.0,
      "float32_gate_logits": True,
      "capacity_factor": -1.0,
  }
  if quantized:
    kwargs.update(
        quantization="fp8_full",
        use_qwix_quantization=True,
        quantize_router_proj=False,
        weight_quantization_calibration_method="fixed,-224,224",
        act_quantization_calibration_method="fixed,-224,224",
        bwd_quantization_calibration_method="absmax",
        shard_embed_moe_on_fsdp=True,
    )
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def _fp8_rule(cfg):
  return [
      qwix.QtRule(
          module_path=".*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e5m2,
          weight_calibration_method=cfg.weight_quantization_calibration_method,
          act_calibration_method=cfg.act_quantization_calibration_method,
          bwd_calibration_method=cfg.bwd_quantization_calibration_method,
          op_names=("gmm", "ragged_dot"),
      )
  ]


def _state_dict(state):
  """Flattens an nnx State / grad pytree into {key path: array}."""
  out = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
    out[jax.tree_util.keystr(path)] = leaf
  return out


def _find(d, *names):
  for key, val in d.items():
    if all(f"'{n}'" in key or f".{n}" in key or f"/{n}" in key for n in names):
      return val
  raise KeyError(f"{names} not in {list(d)}")


class FusedGmm3dRoutedMoeTest(parameterized.TestCase):
  """RoutedMoE with moe_use_fused_gmm3d=True (CPU reference) vs the unfused sparse_matmul path.

  The unfused reference uses 3D [E, rows, cols] weights (the CPU jax.lax.ragged_dot fallback of the unfused
  path does not take the flat FSDP layout); the fused model uses the flat fused parameter, and in the fp8 case
  the (expert, fsdp) row sharding with the in-shard_map quantize + all-gather / reduce-scatter of the weights.
  """

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("needs 4 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")

  def _build(self, cfg, mesh, quantized):
    """Builds a `RoutedMoE` for `cfg`, optionally Qwix-quantized to fp8."""
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=nnx.Rngs(params=0),
      )
      if quantized:
        dummy = jnp.zeros((int(cfg.per_device_batch_size) * jax.device_count(), cfg.max_target_length, _D), cfg.dtype)
        model = qwix.quantize_model(model, qwix.QtProvider(_fp8_rule(cfg)), dummy)
    return model

  def _run(self, cfg, mesh, model, x, cot):
    """Returns `(y, param_grads, x_grad)` of the `sum(y * cot)` loss for `model`."""
    nnx.pop(model, nnx.Intermediate)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss(params_, x_):
      m = nnx.merge(graphdef, params_, rest)
      y, _, _ = m(x_)
      return jnp.sum(y.astype(jnp.float32) * cot), y

    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      (_, y), (gparams, gx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, x)
    return np.asarray(y, np.float32), _state_dict(gparams), np.asarray(gx, np.float32)

  def _compare(self, dtype, fused_quantized, tol_y, tol_grad, tag):
    """Checks the fused path against the unfused reference on outputs and gradients."""
    # Reference: unfused, unquantized, 3D weights. Fused: flat fused param (+ fp8 QAG when quantized).
    cfg_ref = _make_config(False, "float32" if fused_quantized else dtype, False, moe_flat_fsdp_weights=False)
    cfg_fused = _make_config(True, dtype, fused_quantized)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    ref = self._build(cfg_ref, mesh, False)
    fused = self._build(cfg_fused, mesh, fused_quantized)
    # Same weights: wi = concat(wi_0, wi_1) along the output dim ([E, D, F] x 2 -> flat [E * D, 2F]).
    wi_0 = np.asarray(ref.wi_0.value, np.float32).reshape(_E, _D, _F)
    wi_1 = np.asarray(ref.wi_1.value, np.float32).reshape(_E, _D, _F)
    fused.wi.value = jnp.asarray(np.concatenate([wi_0, wi_1], axis=-1).reshape(_E * _D, 2 * _F), fused.wi.value.dtype)
    fused.wo.value = jnp.asarray(np.asarray(ref.wo.value, np.float32).reshape(_E * _F, _D), fused.wo.value.dtype)
    fused.gate.kernel.value = jnp.asarray(ref.gate.kernel.value)

    batch = int(cfg_ref.per_device_batch_size) * jax.device_count()
    key_x, key_c = jax.random.split(jax.random.PRNGKey(1))
    x = jax.random.normal(key_x, (batch, cfg_ref.max_target_length, _D), jnp.float32)
    cot = jax.random.normal(key_c, x.shape, jnp.float32)

    y_ref, g_ref, gx_ref = self._run(cfg_ref, mesh, ref, x.astype(cfg_ref.dtype), cot)
    y_fused, g_fused, gx_fused = self._run(cfg_fused, mesh, fused, x.astype(cfg_fused.dtype), cot)

    dwi_ref = np.concatenate(
        [
            np.asarray(_find(g_ref, "wi_0"), np.float32).reshape(_E, _D, _F),
            np.asarray(_find(g_ref, "wi_1"), np.float32).reshape(_E, _D, _F),
        ],
        axis=-1,
    ).reshape(_E * _D, 2 * _F)
    dwo_ref = np.asarray(_find(g_ref, "wo"), np.float32).reshape(_E * _F, _D)
    comparisons = (
        ("y", y_fused, y_ref, tol_y),
        ("dx", gx_fused, gx_ref, tol_grad),
        ("dwi", np.asarray(_find(g_fused, "wi"), np.float32).reshape(_E * _D, 2 * _F), dwi_ref, tol_grad),
        ("dwo", np.asarray(_find(g_fused, "wo"), np.float32).reshape(_E * _F, _D), dwo_ref, tol_grad),
        (
            "dgate",
            np.asarray(_find(g_fused, "gate", "kernel"), np.float32),
            np.asarray(_find(g_ref, "gate", "kernel"), np.float32),
            tol_grad,
        ),
    )
    failures = []
    for name, a, b, tol in comparisons:
      rel, mx = _rel_err(a, b), _max_abs(a, b)
      print(f"[{tag}] {name}: shape={np.shape(a)} rel_l2={rel:.3e} max_abs={mx:.3e} (tol {tol})")
      if not np.isfinite(a).all() or not rel < tol:
        failures.append(f"{name}: rel_l2={rel:.3e} max_abs={mx:.3e} tol={tol}")
    self.assertFalse(failures, "\n".join(failures))

  def test_fused_param_shape(self):
    cfg = _make_config(True, "float32", False)
    cfg_unfused = _make_config(False, "float32", False)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = self._build(cfg, mesh, False)
    unfused = self._build(cfg_unfused, mesh, False)
    self.assertIsNone(model.wi_0)
    self.assertEqual(tuple(model.wi.value.shape), (_E * _D, 2 * _F))
    self.assertEqual(tuple(model.wo.value.shape), (_E * _F, _D))
    expected_wi = jnp.concatenate(
        [unfused.wi_0.value.reshape(_E, _D, _F), unfused.wi_1.value.reshape(_E, _D, _F)], axis=-1
    ).reshape(_E * _D, 2 * _F)
    np.testing.assert_array_equal(np.asarray(model.wi.value), np.asarray(expected_wi))
    np.testing.assert_array_equal(np.asarray(model.wo.value), np.asarray(unfused.wo.value))
    np.testing.assert_array_equal(np.asarray(model.gate.kernel.value), np.asarray(unfused.gate.kernel.value))

  def test_f32_unquantized_parity(self):
    self._compare("float32", False, tol_y=1e-4, tol_grad=1e-4, tag="module f32")

  def test_fp8_parity_within_quantization_noise(self):
    # Fused fp8 (flat (expert, fsdp)-sharded weights, in-shard_map QAG) vs the unquantized f32 reference.
    self._compare("bfloat16", True, tol_y=0.15, tol_grad=0.2, tag="module fp8 vs f32")

  def test_optimize_remat_nothing_saveable_exact_parity(self):
    """`fused_moe_experts` (`optimize_remat=True`) under `nothing_saveable` matches uncheckpointed VJP bit-exactly."""
    k0, k1, k2, k3, kc = jax.random.split(jax.random.PRNGKey(42), 5)
    num_tokens = 256
    dtype = jnp.bfloat16
    x = jax.random.normal(k0, (num_tokens, _D), dtype)
    wi = (0.05 * jax.random.normal(k1, (_E * _D, 2 * _F), jnp.float32)).astype(dtype)
    wo = (0.05 * jax.random.normal(k2, (_E * _F, _D), jnp.float32)).astype(dtype)
    weights, sort_indices, group_sizes, _ = _random_routing(k3, num_tokens, _E, _K)
    cot = jax.random.normal(kc, (num_tokens, _D), jnp.float32)
    params = moe_fused_gmm3d.FusedGmm3dParams(
        num_experts=_E,
        top_k=_K,
        emb_dim=_D,
        mlp_dim=_F,
        act_qtype=jnp.float8_e4m3fn,
        weight_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        act_scale=224.0 / 448.0,
        weight_scale=224.0 / 448.0,
        compute_dtype=dtype,
        grad_dtype=dtype,
        use_reference=True,
    )

    def body(x_, wi_, wo_, weights_):
      sorted_coeffs = weights_.reshape(-1, 1)[sort_indices]
      return moe_fused_gmm3d.fused_moe_experts(params, x_, wi_, wo_, sorted_coeffs, sort_indices, group_sizes)

    remat_body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)

    def loss_plain(x_, wi_, wo_, weights_):
      y = body(x_, wi_, wo_, weights_)
      return jnp.sum(y.astype(jnp.float32) * cot), y

    def loss_remat(x_, wi_, wo_, weights_):
      y = remat_body(x_, wi_, wo_, weights_)  # pylint: disable=too-many-function-args
      return jnp.sum(y.astype(jnp.float32) * cot), y

    (_, y_p), g_p = jax.jit(jax.value_and_grad(loss_plain, argnums=(0, 1, 2, 3), has_aux=True))(x, wi, wo, weights)
    (_, y_r), g_r = jax.jit(jax.value_and_grad(loss_remat, argnums=(0, 1, 2, 3), has_aux=True))(x, wi, wo, weights)
    np.testing.assert_array_equal(np.asarray(y_r), np.asarray(y_p))
    for idx, name in enumerate(("dx", "dwi", "dwo", "dweights")):
      np.testing.assert_array_equal(np.asarray(g_r[idx]), np.asarray(g_p[idx]), err_msg=name)

  def test_ragged_gather_tc_4buf_pipeline_interpret(self):
    """Verifies 4-buffer + 3-slot SMEM ring `ragged_gather_tc` bit-exactness across step counts & tail blocks."""
    from maxtext.experimental.lineage import ops as lineage_ops  # pylint: disable=import-outside-toplevel

    kx, ki = jax.random.split(jax.random.PRNGKey(123))
    x = jax.random.normal(kx, (96, 8, 128), jnp.bfloat16)
    indices = jax.random.randint(ki, (320,), 0, x.shape[0], dtype=jnp.int32)

    # Exercise 1..5 grid steps, partial tail blocks (rem > 0), non-zero start, and with_absmax.
    for max_out in (16, 90):
      for start, n in ((0, 7), (0, 16), (5, 32), (13, 45), (130, 64), (17, min(75, max_out))):
        if n > max_out:
          continue
        out = lineage_ops.ragged_gather_tc(
            x, indices, jnp.int32(start), jnp.int32(n), max_out_tokens=max_out, block_size=16, interpret=True
        )
        ref = x[indices[start : start + n]]
        np.testing.assert_array_equal(
            np.asarray(out[:n], np.float32),
            np.asarray(ref, np.float32),
            err_msg=f"max_out={max_out} start={start} n={n}",
        )
        out_a, amax = lineage_ops.ragged_gather_tc(
            x,
            indices,
            jnp.int32(start),
            jnp.int32(n),
            max_out_tokens=max_out,
            block_size=16,
            with_absmax=True,
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(out_a[:n], np.float32), np.asarray(ref, np.float32))
        ref_amax = float(np.max(np.abs(np.asarray(ref, np.float32))))
        self.assertAlmostEqual(float(np.asarray(amax)[0, 0]), ref_amax, places=5)

    # Traced zero valid tokens with `with_absmax=True` (dynamic grid clamped to 1).
    _, amax_zero = lineage_ops.ragged_gather_tc(
        x, indices, jnp.int32(10), jnp.int32(0), max_out_tokens=32, block_size=16, with_absmax=True, interpret=True
    )
    self.assertEqual(float(np.asarray(amax_zero)[0, 0]), 0.0)

  def test_params_validate_rejects_bad_configs(self):
    """`FusedGmm3dParams.validate` accepts the Qwen3-235B kernel tiles and rejects unsupported shapes / options."""
    base_kw = {
        "num_experts": 4,
        "top_k": 2,
        "emb_dim": 4096,
        "mlp_dim": 1536,
        "act_qtype": jnp.float8_e4m3fn,
        "weight_qtype": jnp.float8_e4m3fn,
        "bwd_qtype": jnp.float8_e5m2,
        "act_scale": 1.0,
        "weight_scale": 1.0,
        "dx_tile_m": 512,
        "dx_tile_k": 3072,
        "dx_tile_n": 2048,
    }
    moe_fused_gmm3d.FusedGmm3dParams(**base_kw).validate()
    for bad_overrides in (
        {"emb_dim": 100},
        {"act_qtype": None},
        {"weight_qtype": None},
        {"gate_tile_k": 512},
        {"mlp_dim": 384},
        {"dgrad_tile_n": 100},
        {"dgrad_tile_k": 100},
        {"dx_tile_n": 512},
        {"wgrad_schedule_mode": 99},
    ):
      with self.assertRaises(ValueError):
        moe_fused_gmm3d.FusedGmm3dParams(**{**base_kw, **bad_overrides}).validate()

  def test_top_k_dense_vjp_bit_exact(self):
    """Verifies `top_k_dense_vjp` matches `jax.lax.top_k` bit-exactly (values, indices and gradient)."""
    kx, kg = jax.random.split(jax.random.PRNGKey(2026))

    @jax.remat
    def _remat_topk(x_):
      return moe.top_k_dense_vjp(x_, 8, 128)

    for dtype in (jnp.float32, jnp.bfloat16):
      x = jax.random.normal(kx, (4, 64, 128), dtype)
      cot = jax.random.normal(kg, (4, 64, 8), dtype)

      def ref_fn(x_, cot_=cot):
        w, idx = jax.lax.top_k(x_, 8)
        return jnp.sum(w.astype(jnp.float32) * cot_.astype(jnp.float32)), (w, idx)

      def dense_fn(x_, cot_=cot):
        w, idx = _remat_topk(x_)
        return jnp.sum(w.astype(jnp.float32) * cot_.astype(jnp.float32)), (w, idx)

      j_ref = jax.jit(jax.value_and_grad(ref_fn, has_aux=True))
      j_dense = jax.jit(jax.value_and_grad(dense_fn, has_aux=True))
      (_, (w_ref, idx_ref)), gx_ref = j_ref(x)
      (_, (w_den, idx_den)), gx_den = j_dense(x)
      np.testing.assert_array_equal(np.asarray(w_den, np.float32), np.asarray(w_ref, np.float32))
      np.testing.assert_array_equal(np.asarray(idx_den), np.asarray(idx_ref))
      np.testing.assert_array_equal(np.asarray(gx_den, np.float32), np.asarray(gx_ref, np.float32))

  def test_exact_topk_gather_reduce_tc_interpret_and_parity(self):
    """Verifies `exact_topk_gather_reduce_tc` Pallas `interpret=True` & reference bit-exactness across block counts."""
    from maxtext.experimental.lineage import ops as lineage_ops  # pylint: disable=import-outside-toplevel

    top_k = 8
    # Exercise num_blocks = 1 (64), 3 (192) and 8 (1024). Only the last one has more blocks than output
    # buffers (4) and SMEM ring slots (4), so it is the one that runs the buffer-reuse and ring-refill steps of the
    # steady-state loop, as every production shape does.
    for idx, (num_tokens, num_blocks) in enumerate(((64, 1), (192, 3), (1024, 8))):
      for out_2d in (False, True):
        block_tokens = lineage_ops._exact_topk_block_tokens(  # pylint: disable=protected-access
            num_tokens, top_k, 16, 128, jnp.bfloat16, out_2d=out_2d
        )
        self.assertEqual(num_tokens // block_tokens, num_blocks, f"num_tokens={num_tokens} out_2d={out_2d}")
      k_x, k_r = jax.random.split(jax.random.PRNGKey(700 + idx))
      _, sort_indices, _, _ = _random_routing(k_r, num_tokens, _E, top_k)
      token_indices = (sort_indices // top_k).astype(jnp.int32)
      x = jax.random.normal(k_x, (num_tokens * top_k, 16, 128), jnp.bfloat16)

      self.assertTrue(lineage_ops.can_use_exact_topk_gather_reduce_tc(num_tokens, top_k, 16, 128, jnp.bfloat16))
      src_rows = lineage_ops.exact_topk_gather_reduce_tc_src_rows(token_indices)
      out_ref = lineage_ops.exact_topk_gather_reduce_tc(
          x, src_rows, num_out_tokens=num_tokens, top_k=top_k, interpret=False
      )
      out_interp = lineage_ops.exact_topk_gather_reduce_tc(
          x, src_rows, num_out_tokens=num_tokens, top_k=top_k, interpret=True
      )
      out_2d_ref = lineage_ops.exact_topk_gather_reduce_tc(
          x, src_rows, num_out_tokens=num_tokens, top_k=top_k, out_2d=True, interpret=False
      )
      out_2d_interp = lineage_ops.exact_topk_gather_reduce_tc(
          x, src_rows, num_out_tokens=num_tokens, top_k=top_k, out_2d=True, interpret=True
      )
      # Explicit left-to-right f32 summation per token in buffer order:
      x_f32 = np.asarray(x, np.float32)
      tok_np = np.asarray(token_indices)
      expected = np.zeros((num_tokens, 16, 128), np.float32)
      counts = np.zeros((num_tokens,), np.int32)
      for s, t in enumerate(tok_np):
        if counts[t] == 0:
          expected[t] = x_f32[s]
        else:
          expected[t] = expected[t] + x_f32[s]
        counts[t] += 1
      expected_bf16 = np.asarray(jnp.asarray(expected, jnp.bfloat16), np.float32)
      np.testing.assert_array_equal(
          np.asarray(out_ref, np.float32), expected_bf16, err_msg=f"ref num_tokens={num_tokens}"
      )
      np.testing.assert_array_equal(
          np.asarray(out_interp, np.float32), expected_bf16, err_msg=f"interp num_tokens={num_tokens}"
      )
      self.assertEqual(out_2d_interp.shape, (num_tokens, 16 * 128))
      np.testing.assert_array_equal(
          np.asarray(out_2d_ref, np.float32),
          expected_bf16.reshape(num_tokens, 16 * 128),
          err_msg=f"ref_2d num_tokens={num_tokens}",
      )
      np.testing.assert_array_equal(
          np.asarray(out_2d_interp, np.float32),
          expected_bf16.reshape(num_tokens, 16 * 128),
          err_msg=f"interp_2d num_tokens={num_tokens}",
      )

  def test_relayout_2d_to_3d_pallas_interpret(self):
    """Verifies `relayout_2d_to_3d_pallas(..., interpret=True)` bit-exactness on fp8 tokens."""
    from maxtext.experimental.lineage.gmmv2 import relayout  # pylint: disable=import-outside-toplevel

    k_fp8, _ = jax.random.split(jax.random.PRNGKey(808))
    x_fp8 = jax.random.normal(k_fp8, (512, 1024), jnp.float32).astype(jnp.float8_e4m3fn)
    for tm in (128, 512):
      out_fp8 = relayout.relayout_2d_to_3d_pallas(x_fp8, block_tokens=tm, interpret=True)
      self.assertEqual(out_fp8.shape, (512, 8, 128))
      np.testing.assert_array_equal(np.asarray(out_fp8, np.float32), np.asarray(x_fp8.reshape(512, 8, 128), np.float32))

  def test_inv_perm_src_rows_exact_topk_parity(self):
    """Verifies `_inv_perm_src_rows` groups exact token slots in canonical k=0..top_k-1 order for the exact combine."""
    from maxtext.experimental.lineage import ops as lineage_ops  # pylint: disable=import-outside-toplevel

    top_k = 8
    num_tokens = 64
    _, sort_indices, _, _ = _random_routing(jax.random.PRNGKey(809), num_tokens, _E, top_k)
    token_indices = (sort_indices // top_k).astype(jnp.int32)
    inv_rows = jax.jit(moe_fused_gmm3d._inv_perm_src_rows)(sort_indices)  # pylint: disable=protected-access
    # Every group `[t*top_k : (t+1)*top_k]` must map back to token `t` and cover all `C` slots bijectively.
    mapped_tokens = np.asarray(token_indices)[np.asarray(inv_rows)].reshape(num_tokens, top_k)
    expected_tokens = np.broadcast_to(np.arange(num_tokens, dtype=np.int32)[:, None], (num_tokens, top_k))
    np.testing.assert_array_equal(mapped_tokens, expected_tokens)
    np.testing.assert_array_equal(np.sort(np.asarray(inv_rows)), np.arange(num_tokens * top_k, dtype=np.int32))

    x_slots = jax.random.normal(jax.random.PRNGKey(810), (num_tokens * top_k, 16, 128), jnp.bfloat16)
    sort_rows = lineage_ops.exact_topk_gather_reduce_tc_src_rows(token_indices)
    y_inv = lineage_ops.exact_topk_gather_reduce_tc(
        x_slots, inv_rows, num_out_tokens=num_tokens, top_k=top_k, out_2d=True, interpret=True
    )
    y_srt = lineage_ops.exact_topk_gather_reduce_tc(
        x_slots, sort_rows, num_out_tokens=num_tokens, top_k=top_k, out_2d=True, interpret=True
    )
    self.assertLess(_rel_err(y_inv, y_srt), 1e-3)

  def test_quantize_relayout_2d_to_3d_pallas_interpret(self):
    """Verifies `quantize_relayout_2d_to_3d_pallas(..., interpret=True)` matches `static_quantize` bit for bit.

    Covers a power-of-two scale and `fixed,-256,256` (256 / 448), whose reciprocal is not exact in f32.
    """
    from maxtext.experimental.lineage.gmmv2 import relayout  # pylint: disable=import-outside-toplevel

    kx, _ = jax.random.split(jax.random.PRNGKey(1101))
    x_bf16 = (jax.random.normal(kx, (512, 1024), jnp.float32) * 8.0).astype(jnp.bfloat16)
    for scale in (0.5, 256.0 / 448.0):
      ref_static = moe_fused_gmm3d.static_quantize(x_bf16, jnp.float8_e4m3fn, scale).reshape(512, 8, 128)
      for tm in (128, 512):
        out_static = relayout.quantize_relayout_2d_to_3d_pallas(
            x_bf16, jnp.float8_e4m3fn, scale, block_tokens=tm, interpret=True
        )
        np.testing.assert_array_equal(
            np.asarray(out_static, np.float32),
            np.asarray(ref_static, np.float32),
            err_msg=f"static scale={scale} tm={tm}",
        )

  def test_fused_moe_experts_kernel_path_lowers_on_tpu7x(self):
    """Lowers `fused_moe_experts(..., use_reference=False)` fwd + bwd on TPU7x for the legacy and shipped kernels."""
    tpu7x = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    # emb_dim=2048 gives [C, 16, 128] bf16 rows: `exact_topk_combine` needs D // 128 to be a multiple of the 16 bf16
    # sublanes, and falls back to `ragged_gather_reduce_tc` otherwise.
    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=tpu7x)):
      for sched_mode, gmm_comb, split_bwd, split_dw, sc_gather, exact_comb, inv_comb, relayout_x, preq_dy in (
          # Legacy kernels: linear_combine_fwd forward and gate_bwd_fused backward.
          (0, False, False, False, False, False, False, True, False),
          # The kernel options of the Qwen3-235B v7x training config.
          (10, True, True, True, True, True, False, True, True),
      ):
        p_tpu = moe_fused_gmm3d.FusedGmm3dParams(
            num_experts=4,
            top_k=8,
            emb_dim=2048,
            mlp_dim=512,
            act_qtype=jnp.float8_e4m3fn,
            weight_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            act_scale=0.5,
            weight_scale=0.5,
            gate_tile_m=256,
            gate_tile_k=1024,
            dgrad_tile_rows=256,
            dgrad_bucket_rows=128,
            dgrad_tile_n=512,
            dgrad_tile_k=1024,
            dx_tile_m=256,
            dx_tile_k=1024,
            dx_tile_n=1024,
            dw_tile_k=1024,
            use_gmm_linear_combine=gmm_comb,
            split_gate_bwd=split_bwd,
            split_dw_gate=split_dw,
            prequant_dy=preq_dy,
            early_dwo_rs=True,
            sc_routed_gather=sc_gather,
            exact_topk_combine=exact_comb,
            exact_topk_out_2d=exact_comb,
            inv_perm_combine=inv_comb,
            relayout_x_2d_to_3d=relayout_x,
            fused_quant_relayout=True,
            overlap_dcoeffs_scatter=True,
            wgrad_schedule_mode=sched_mode,
            use_reference=False,
        )

        def moe_fn(x_, wi_, wo_, c_, s_, g_, pp=p_tpu):
          return moe_fused_gmm3d.fused_moe_experts(pp, x_, wi_, wo_, c_, s_, g_)

        aot_moe = (
            jax.jit(
                jax.value_and_grad(
                    lambda x_, wi_, wo_, c_, s_, g_, fn=moe_fn: jnp.sum(fn(x_, wi_, wo_, c_, s_, g_).astype(jnp.float32)),
                    argnums=(0, 1, 2, 3),
                )
            )
            .trace(
                jax.ShapeDtypeStruct((256, 2048), jnp.bfloat16),
                jax.ShapeDtypeStruct((4 * 2048, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((4 * 512, 2048), jnp.bfloat16),
                jax.ShapeDtypeStruct((256 * 8, 1), jnp.float32),
                jax.ShapeDtypeStruct((256 * 8,), jnp.int32),
                jax.ShapeDtypeStruct((4,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
        self.assertIn("tpu_custom_call", aot_moe, f"sched_mode={sched_mode}")

    # The kernel backward's weight-gradient reduce-scatters (`_compute_dwo`, `_rs_wi_half` + concat) only exist
    # with FSDP gather axes, so also lower the shipped options (the last row above) inside a 4-device `shard_map`
    # with the flat weights sharded on fsdp.
    fsdp_mesh = jax.sharding.AbstractMesh((4,), ("fsdp",), abstract_device=tpu7x)
    p_fsdp = dataclasses.replace(p_tpu, wi_gather_axes=("fsdp",), wo_gather_axes=("fsdp",))
    spec = jax.sharding.PartitionSpec

    def sharded(shape, dtype, s):
      return jax.ShapeDtypeStruct(shape, dtype, sharding=jax.sharding.NamedSharding(fsdp_mesh, s))

    def fsdp_loss(x_, wi_, wo_, c_, s_, g_):
      def local_loss(x_l, wi_l, wo_l, c_l, s_l, g_l):
        y = moe_fused_gmm3d.fused_moe_experts(p_fsdp, x_l, wi_l, wo_l, c_l, s_l, g_l)
        return jax.lax.psum(jnp.sum(y.astype(jnp.float32)), "fsdp")

      return jax.shard_map(
          local_loss,
          mesh=fsdp_mesh,
          in_specs=(spec("fsdp"), spec("fsdp"), spec("fsdp"), spec("fsdp"), spec("fsdp"), spec()),
          out_specs=spec(),
      )(x_, wi_, wo_, c_, s_, g_)

    with jax.sharding.use_abstract_mesh(fsdp_mesh):
      aot_fsdp = (
          jax.jit(jax.value_and_grad(fsdp_loss, argnums=(0, 1, 2, 3)))
          .trace(
              sharded((4 * 256, 2048), jnp.bfloat16, spec("fsdp")),
              sharded((4 * 2048, 1024), jnp.bfloat16, spec("fsdp")),
              sharded((4 * 512, 2048), jnp.bfloat16, spec("fsdp")),
              sharded((4 * 256 * 8, 1), jnp.float32, spec("fsdp")),
              sharded((4 * 256 * 8,), jnp.int32, spec("fsdp")),
              sharded((4,), jnp.int32, spec()),
          )
          .lower(lowering_platforms=("tpu",))
          .as_text()
      )
    self.assertIn("tpu_custom_call", aot_fsdp)
    self.assertIn("reduce_scatter", aot_fsdp)

  def test_overlap_dcoeffs_scatter_and_packed_routing_sort_parity(self):
    """`overlap_dcoeffs_scatter` + `moe_packed_routing_sort` (both read on CPU) match the default fwd + all grads."""
    # Lower actual `RoutedMoE` forward with `moe_packed_routing_sort=True` to verify packed u32 sort wiring!
    cfg_ps = _make_config(True, "bfloat16", True, moe_packed_routing_sort=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ps), cfg_ps.mesh_axes)
    m_ps = self._build(cfg_ps, mesh, True)
    nnx.pop(m_ps, nnx.Intermediate)
    gdef_ps, p_ps, rest_ps = nnx.split(m_ps, nnx.Param, ...)
    batch = int(cfg_ps.per_device_batch_size) * jax.device_count()
    x_ps = jnp.zeros((batch, cfg_ps.max_target_length, _D), jnp.bfloat16)
    with mesh, nn_partitioning.axis_rules(cfg_ps.logical_axis_rules):
      aot_packed_sort = jax.jit(lambda p_, x_: nnx.merge(gdef_ps, p_, rest_ps)(x_)[0]).lower(p_ps, x_ps).as_text()
    self.assertIn('"stablehlo.sort"', aot_packed_sort)
    self.assertIn("is_stable = false", aot_packed_sort)
    self.assertIn("ui32>", aot_packed_sort)

    # Full 4-device FSDP `RoutedMoE` forward + all gradients.
    cfg_base = _make_config(True, "bfloat16", True)
    m_base = self._build(cfg_base, mesh, True)
    kx, kc = jax.random.split(jax.random.PRNGKey(1102))
    x = jax.random.normal(kx, (batch, cfg_base.max_target_length, _D), jnp.bfloat16)
    cot = jax.random.normal(kc, x.shape, jnp.float32)
    y_base, g_base, gx_base = self._run(cfg_base, mesh, m_base, x, cot)

    cfg_w11 = _make_config(
        True,
        "bfloat16",
        True,
        moe_fused_gmm3d_overlap_dcoeffs_scatter=True,
        moe_packed_routing_sort=True,
    )
    m_w11 = self._build(cfg_w11, mesh, True)
    y_w11, g_w11, gx_w11 = self._run(cfg_w11, mesh, m_w11, x, cot)
    np.testing.assert_array_equal(y_w11, y_base)
    np.testing.assert_array_equal(gx_w11, gx_base)
    for k, g_base_k in g_base.items():
      np.testing.assert_allclose(
          np.asarray(g_w11[k], np.float32), np.asarray(g_base_k, np.float32), rtol=1e-5, atol=1e-5, err_msg=k
      )


if __name__ == "__main__":
  absltest.main()
