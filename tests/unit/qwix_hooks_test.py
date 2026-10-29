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

"""Unit tests for Qwix step_context, scan_index_context in nnx_scan, and role-scoped quantization."""

import json
from types import SimpleNamespace
import unittest
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.layers import nnx_scan, quantizations
import qwix


class _ToyLayer(nnx.Module):
  """Minimal NNX layer with a single linear projection initialized to 1.25."""

  def __init__(self, features: int, *, rngs: nnx.Rngs):
    self.proj = nnx.Linear(
        features,
        features,
        use_bias=False,
        kernel_init=nnx.initializers.constant(1.25),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
    del args, kwargs
    return self.proj(x.astype(jnp.float32))


class _ToyScannedStack(nnx.Module):
  """Minimal NNX module wrapping a scanned stack of _ToyLayer instances."""

  def __init__(self, num_layers: int, features: int, *, remat: bool = False, rngs: nnx.Rngs):
    self.num_layers = num_layers
    self.remat = remat
    self.layers = nnx_scan.create_scanned_layers(
        lambda r: _ToyLayer(features, rngs=r),
        length=num_layers,
        param_scan_axis=0,
        metadata_axis_name="layers",
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return nnx_scan.apply_scanned_layers(
        self.layers,
        x,
        length=self.num_layers,
        param_scan_axis=0,
        apply_fn=lambda lyr, carry: lyr(carry),
        remat=self.remat,
    )


class QwixHooksTest(unittest.TestCase):
  """Tests nnx_scan scan_index_context, step_context, and role-scoped model quantization."""

  def test_nnx_scan_target_scan_indices_fwd_and_bwd(self):
    # 3 stacked layers of 4x4 weights initialized to 1.25.
    # Unquantized layer 0: 1.0 -> 4 * 1.0 * 1.25 = 5.0.
    # Layer 1 with 1-pass e3m0: lhs 5.0 -> 4.0, rhs 1.25 -> 1.0, output = 4 * 4.0 * 1.0 = 16.0.
    # Unquantized layer 2: 16.0 -> 4 * 16.0 * 1.25 = 80.0 (sum over 8 elements = 640.0).
    rule = qwix.QtRule(
        target_scan_indices=(1,),
        additional_qt_config={"multipass_mode": "1xe3m0"},
    )
    provider = qwix.QtProvider([rule])
    x = jnp.ones((2, 4), dtype=jnp.float32)
    for remat in (False, True):
      model = _ToyScannedStack(num_layers=3, features=4, remat=remat, rngs=nnx.Rngs(0))
      q_model = qwix.quantize_model(model, provider, x)
      out, grads = nnx.value_and_grad(lambda m, inp: jnp.sum(m(inp)))(q_model, x)
      self.assertAlmostEqual(float(out), 640.0, places=3)
      kernel_grad = grads.layers.proj.kernel[...]
      self.assertEqual(kernel_grad.shape, (3, 4, 4))

    # With 2-pass e3m0 on scan index 1, both residuals (5.0 = 4+1, 1.25 = 1+0.25) are recovered -> 1000.0.
    rule_2p = qwix.QtRule(
        target_scan_indices=(1,),
        additional_qt_config={"multipass_mode": "2xe3m0"},
    )
    model_2p = qwix.quantize_model(
        _ToyScannedStack(num_layers=3, features=4, remat=False, rngs=nnx.Rngs(0)),
        qwix.QtProvider([rule_2p]),
        x,
    )
    self.assertAlmostEqual(float(jnp.sum(model_2p(x))), 1000.0, places=3)

  def test_step_context_gating_under_jit(self):
    rule = qwix.QtRule(
        start_step=5,
        end_step=10,
        additional_qt_config={"multipass_mode": "1xe3m0"},
    )
    provider = qwix.QtProvider([rule])
    x = jnp.ones((1, 4), dtype=jnp.float32)
    model = _ToyLayer(features=4, rngs=nnx.Rngs(0))
    q_model = qwix.quantize_model(model, provider, x)
    graphdef, state = nnx.split(q_model)

    @jax.jit
    def step_fn(st, step, inp):
      m = nnx.merge(graphdef, st)
      with qwix.step_context(step):
        return jnp.sum(m(inp))

    # Step 2 (before start_step=5): unquantized -> 4 cols * 5.0 = 20.0
    self.assertAlmostEqual(float(step_fn(state, jnp.int32(2), x)), 20.0, places=3)
    # Step 5 (active): 1-pass e3m0 -> 4 cols * 4.0 = 16.0
    self.assertAlmostEqual(float(step_fn(state, jnp.int32(5), x)), 16.0, places=3)
    # Step 10 (at end_step=10, exclusive): inactive -> 20.0
    self.assertAlmostEqual(float(step_fn(state, jnp.int32(10), x)), 20.0, places=3)

  def test_role_scoped_maybe_quantize_model(self):
    rules_json = json.dumps(
        [
            {
                "module_path": ".*",
                "multipass_mode": "1xe3m0",
            }
        ]
    )
    cfg = SimpleNamespace(
        quantization="qwix",
        use_qwix_quantization=True,
        use_batch_split_schedule=False,
        micro_batch_size_to_train_on=1,
        max_target_length=4,
        mtp_num_layers=0,
        quantization_rules_json=rules_json,
        quantization_target_roles=["actor"],
        quantization_exclude_module_path="",
        quantization_start_step=0,
        quantization_end_step=-1,
        quantization_target_scan_indices="",
    )
    x = jnp.ones((1, 4), dtype=jnp.float32)

    actor_model = quantizations.maybe_quantize_model(_ToyLayer(4, rngs=nnx.Rngs(0)), cfg, role="actor")
    ref_model = quantizations.maybe_quantize_model(_ToyLayer(4, rngs=nnx.Rngs(0)), cfg, role="reference")

    self.assertAlmostEqual(float(jnp.sum(actor_model(x))), 16.0, places=3)
    self.assertAlmostEqual(float(jnp.sum(ref_model(x))), 20.0, places=3)


if __name__ == "__main__":
  absltest.main()
