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

"""Tests for bwd_delayed_scaling (per-layer delayed scaling of the fp8 backward cotangent quantizers).

The model tests need 8 devices, e.g. on CPU:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
    python3 -m pytest tests/unit/bwd_delayed_scaling_test.py
"""

import sys
import unittest

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import bwd_delayed_scaling as bds
from maxtext.utils import sharding
from maxtext.utils import train_utils
from tests.utils.test_helpers import get_test_config_path


class UpdateStateTest(unittest.TestCase):
  """History, margin, unused slots and the saturation count."""

  def test_history_margin(self):
    st = jnp.zeros((3, 4), jnp.float32).at[:, 0].set(1e-3)  # H = 3, R = 1e-3
    seq = [2e-4, 5e-4, 1e-4, 1e-4, 0.0]
    for t, a in enumerate(seq):
      prev_r = float(st[0, 0])
      g = jnp.zeros_like(st).at[0, 0].set(a)
      st, sat, _ = bds.update_state(st, g, margin=2.0)
      window = seq[max(0, t - 2) : t + 1]
      want = 2.0 * max(window) if max(window) > 0 else prev_r
      np.testing.assert_allclose(float(st[0, 0]), want, rtol=1e-6)
      self.assertEqual(int(sat), int(a > prev_r))
      self.assertEqual(float(st[2, 0]), float(np.float32(1e-3)))  # unused slot keeps its scale

  def test_all_zero_history_keeps_scale(self):
    st = jnp.zeros((1, 2), jnp.float32).at[0, 0].set(3e-3)
    st, _, _ = bds.update_state(st, jnp.zeros_like(st), margin=2.0)
    self.assertEqual(float(st[0, 0]), float(np.float32(3e-3)))


class HandTest(unittest.TestCase):
  """One Qwix fp8 dot inside a tap: the stored R quantizes the cotangent exactly as `fixed,R` does."""

  def test_tap_equals_fixed(self):
    from qwix._src.core import dot_general_qt  # pylint: disable=import-outside-toplevel

    cfg = type(
        "C",
        (),
        {
            "bwd_delayed_scaling": True,
            "use_qwix_quantization": True,
            "quantization": "fp8_full",
            "gradient_accumulation_steps": 1,
            "ici_pipeline_parallelism": 1,
            "dcn_pipeline_parallelism": 1,
            "bwd_delayed_scaling_history": 1,
            "bwd_delayed_scaling_margin": 2.0,
            "bwd_delayed_scaling_init": 1e-3,
            "weight_quantization_calibration_method": "fixed,-224,224",
            "act_quantization_calibration_method": "fixed,-224,224",
        },
    )
    bds.configure(cfg)
    r = 3e-3

    def qt(method):
      return dot_general_qt.DotGeneralQtConfig(
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_qtype=jnp.float8_e4m3fn,
          dlhs_grad_qtype=jnp.float8_e5m2,
          drhs_grad_qtype=jnp.float8_e5m2,
          lhs_calibration_method="fixed,-224,224",
          rhs_calibration_method="fixed,-224,224",
          dlhs_grad_calibration_method=method,
          drhs_grad_calibration_method=method,
      )

    kx, kw, kg = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(kx, (6, 16), jnp.float32) * 3.0
    w = jax.random.normal(kw, (16, 12), jnp.float32) * 0.1
    dy = jax.random.normal(kg, (6, 12), jnp.float32) * 1e-3
    dn = (((1,), (0,)), ((), ()))
    c_dly, c_fix = qt(bds.bwd_calibration_method("absmax")), qt(f"fixed,{r}")

    def loss_tap(a, b, st):
      with bds.layer_context(st, "Hand"):
        out = bds.tap("hand", lambda p, q: dot_general_qt.dot_general_qt(p, q, dn, c_dly), a, b)
      return jnp.sum(out * dy)

    def loss_fix(a, b):
      return jnp.sum(dot_general_qt.dot_general_qt(a, b, dn, c_fix) * dy)

    def loss_abs(a, b):  # the sentinel outside a tap falls back to its method (absmax)
      return jnp.sum(dot_general_qt.dot_general_qt(a, b, dn, c_dly) * dy)

    st = jnp.zeros((bds.NSLOTS, 2), jnp.float32).at[0, 0].set(r)
    gx, gw, gs = jax.jit(jax.grad(loss_tap, argnums=(0, 1, 2)))(x, w, st)
    fx, fw = jax.jit(jax.grad(loss_fix, argnums=(0, 1)))(x, w)
    ax, _ = jax.jit(jax.grad(loss_abs, argnums=(0, 1)))(x, w)
    # XLA folds a constant scale slightly differently from a runtime one: equal to ~1 ulp.
    np.testing.assert_allclose(gx, fx, rtol=0, atol=1e-6 * float(jnp.max(jnp.abs(fx))))
    np.testing.assert_allclose(gw, fw, rtol=0, atol=1e-6 * float(jnp.max(jnp.abs(fw))))
    self.assertFalse(np.array_equal(np.asarray(ax), np.asarray(gx)))
    # amax = max|dY * s| with s = 224 / 448
    np.testing.assert_allclose(float(gs[0, 0]), 0.5 * float(jnp.max(jnp.abs(dy))), rtol=1e-6)
    self.assertTrue(bool(jnp.all(gs[1:] == 0)) and bool(jnp.all(gs[0, 1:] == 0)))


_TINY = {
    "run_name": "bwd_delayed_scaling_test",
    "model_name": "deepseek3-test",
    "override_model_config": True,
    "base_emb_dim": 64,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "n_routing_groups": -1,
    "topk_routing_group": -1,
    "base_num_decoder_layers": 3,
    "first_num_dense_layers": 1,
    "vocab_size": 256,
    "mtp_num_layers": 0,
    "per_device_batch_size": 1,
    "max_target_length": 32,
    "attention": "dot_product",
    "capacity_factor": -1,
    "dataset_type": "synthetic",
    "enable_checkpointing": False,
    "custom_mesh_and_rule": "fsdp-as-dp-for-attn",
    "ici_fsdp_parallelism": 2,
    "ici_expert_parallelism": 4,
    "scan_layers": True,
    "learning_rate": 1e-3,
    "gradient_clipping_threshold": 1.0,
    "use_iota_embed": True,
    "skip_jax_distributed_system": True,
    "sharding_tolerance": 1.0,
    "dtype": "bfloat16",
    "quantization": "fp8_full",
    "use_qwix_quantization": True,
    "weight_quantization_calibration_method": "fixed,-224,224",
    "act_quantization_calibration_method": "fixed,-224,224",
    "bwd_quantization_calibration_method": "absmax",
    "quantize_router_proj": False,
    "sparse_matmul": True,
    "megablox": True,
    "use_tokamax_gmm": False,
    "use_gmm_v2": False,
    "num_moe_token_chunks": 2,
    "use_ring_of_experts": True,
}
for _w in ("wi", "wo"):
  for _ph in ("fwd", "dlhs", "drhs"):
    for _d, _v in (("batch_seq", 16), ("embed_dim", 64), ("mlp_dim", 32)):
      _TINY[f"{_w}_tile_{_ph}_{_d}"] = _v


def _run(extra, steps):
  """Trains the tiny model; returns losses, final params and the bwd_dscale state after every step."""
  config = pyconfig.initialize([sys.argv[0], get_test_config_path()], **dict(_TINY, steps=steps, **extra))
  _, _, sms, _, mesh, _, it, _, _, _, state = train_utils.setup_train_loop(config, None)
  graphdef, state = nnx.split(state)
  ps, sms = sharding.maybe_update_params_sharding_with_opt(config, sms)
  owg = bds.owg_type()

  def dscale(s):
    return {
        "/".join(map(str, k)): np.asarray(v.get_value())
        for k, v in nnx.to_flat_state(nnx.state(nnx.merge(graphdef, s).model, owg))
        if bds.STATE_NAME in "/".join(map(str, k))
    }

  losses, states, saturated = [], [], []
  with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
    bds.configure(config)
    step, _ = train_utils.jit_train_and_eval_step(
        config, graphdef, mesh, state, sms, pre_train.train_step, params_shardings=ps
    )
    states.append(dscale(state))
    for _ in range(steps):
      state, metrics = step(state, next(it))
      losses.append(float(metrics["scalar"]["learning/loss"]))
      saturated.append(float(metrics["scalar"].get("learning/bwd_dscale_saturated", 0.0)))
      states.append(dscale(state))
    params = {
        "/".join(map(str, k)): np.asarray(jnp.asarray(v.get_value(), jnp.float32))
        for k, v in nnx.to_flat_state(nnx.state(nnx.merge(graphdef, state).model, nnx.Param))
    }
  return losses, params, states, saturated


@pytest.mark.cpu_only
class ModelTest(unittest.TestCase):
  """Tiny DeepSeek (dense + MoE layers, megablox, ring of experts, 2 token chunks) on fsdp 2 x expert 4."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 8:
      self.skipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")

  def test_step0_equals_fixed(self):
    r = 0.00125
    fixed = {"bwd_quantization_calibration_method": f"fixed,{r}"}
    delayed = {"bwd_delayed_scaling": True, "bwd_delayed_scaling_init": r, "bwd_delayed_scaling_history": 1}
    l_fix, p_fix, _, _ = _run(fixed, 1)
    l_dly, p_dly, _, _ = _run(delayed, 1)
    self.assertEqual(l_fix, l_dly)
    for k, v in p_fix.items():
      np.testing.assert_array_equal(v, p_dly[k], err_msg=k)

  def test_tracks_absmax_and_scale_is_margin_times_history_max(self):
    steps, margin = 6, 16.0
    l_abs, _, _, _ = _run({}, steps)
    l_dly, _, states, saturated = _run(
        {
            "bwd_delayed_scaling": True,
            "bwd_delayed_scaling_init": 0.00125,
            "bwd_delayed_scaling_history": 4,
            "bwd_delayed_scaling_margin": margin,
        },
        steps,
    )
    np.testing.assert_allclose(l_dly, l_abs, atol=0.02)
    self.assertEqual(saturated[1:], [0.0] * (steps - 1))  # step 0 uses the init scale
    for before, after in zip(states[:-1], states[1:]):
      for k, v in after.items():
        hmax, r_new, r_old = np.max(v[..., 1:], axis=-1), v[..., 0], before[k][..., 0]
        np.testing.assert_allclose(r_new, np.where(hmax > 0, margin * hmax, r_old), rtol=1e-6)
        self.assertGreater(int(np.sum(v[..., 1] > 0)), 0, k)


if __name__ == "__main__":
  unittest.main()
