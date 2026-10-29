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

"""End-to-end tests for Qwix multi-pass checkpoint transformation and training workflows."""

import json
import tempfile
from types import SimpleNamespace
import unittest
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion import quantize_checkpoint
from maxtext.layers import nnx_scan, quantizations
import orbax.checkpoint as ocp
import qwix


class _ToyLayer(nnx.Module):
  """Minimal NNX layer with a linear projection initialized to 1.25."""

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


class _ToyModel(nnx.Module):
  """Minimal NNX model with a scanned decoder stack and an unquantized output head."""

  def __init__(self, num_layers: int, features: int, *, rngs: nnx.Rngs):
    self.num_layers = num_layers
    self.layers = nnx_scan.create_scanned_layers(
        lambda r: _ToyLayer(features, rngs=r),
        length=num_layers,
        param_scan_axis=0,
        metadata_axis_name="layers",
        rngs=rngs,
    )
    self.logits_dense = nnx.Linear(
        features,
        features,
        use_bias=False,
        kernel_init=nnx.initializers.constant(1.25),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
    del args, kwargs
    h = nnx_scan.apply_scanned_layers(
        self.layers,
        x.astype(jnp.float32),
        length=self.num_layers,
        param_scan_axis=0,
        apply_fn=lambda lyr, carry: lyr(carry),
    )
    return self.logits_dense(h)


class QwixWorkflowsTest(unittest.TestCase):
  """Tests PyTree/Orbax checkpoint quantization and end-to-end Qwix multi-pass training workflows."""

  def test_apply_qwix_rules_to_pytree_scoping_and_multipass(self):
    params = {
        "decoder": {
            "layers": {
                "mlp": {"kernel": jnp.full((3, 4, 4), 1.25, dtype=jnp.float32)},
                "pre_self_attention_norm": {"scale": jnp.full((3, 4), 1.25, dtype=jnp.float32)},
            },
            "logits_dense": {"kernel": jnp.full((4, 4), 1.25, dtype=jnp.float32)},
        }
    }
    rule_1p = qwix.QtRule(
        module_path=".*decoder.*",
        exclude_module_path=".*logits_dense.*",
        target_scan_indices=(1,),
        additional_qt_config={"multipass_mode": "1xe3m0"},
    )
    out_1p = quantize_checkpoint.apply_qwix_rules_to_pytree(params, [rule_1p], param_scan_axis=0, num_scanned_layers=3)
    mlp_k = out_1p["decoder"]["layers"]["mlp"]["kernel"]
    # Layer 0 and 2 untouched (16 * 1.25 = 20.0); Layer 1 quantized with 1-pass e3m0 (16 * 1.0 = 16.0)
    self.assertAlmostEqual(float(jnp.sum(mlp_k[0])), 20.0, places=3)
    self.assertAlmostEqual(float(jnp.sum(mlp_k[1])), 16.0, places=3)
    self.assertAlmostEqual(float(jnp.sum(mlp_k[2])), 20.0, places=3)
    # Excluded logits_dense and 1D/norm scale remain untouched
    self.assertAlmostEqual(float(jnp.sum(out_1p["decoder"]["logits_dense"]["kernel"])), 20.0, places=3)
    self.assertAlmostEqual(
        float(jnp.sum(out_1p["decoder"]["layers"]["pre_self_attention_norm"]["scale"])), 15.0, places=3
    )

    # With 2-pass e3m0, Pass 1 recovers the 0.25 residual exactly (sum = 20.0)
    rule_2p = qwix.QtRule(
        module_path=".*decoder.*",
        target_scan_indices=(1,),
        additional_qt_config={"multipass_mode": "2xe3m0"},
    )
    out_2p = quantize_checkpoint.apply_qwix_rules_to_pytree(params, [rule_2p], param_scan_axis=0, num_scanned_layers=3)
    self.assertAlmostEqual(float(jnp.sum(out_2p["decoder"]["layers"]["mlp"]["kernel"][1])), 20.0, places=3)

  def test_orbax_checkpoint_roundtrip(self):
    params = {
        "decoder": {
            "layers": {"wi": jnp.full((2, 4, 4), 1.25, dtype=jnp.float32)},
            "logits_dense": {"kernel": jnp.full((4, 4), 1.25, dtype=jnp.float32)},
        }
    }
    cfg = SimpleNamespace(
        quantization="qwix",
        quantization_rules_json=json.dumps(
            [
                {
                    "module_path": ".*decoder.*",
                    "exclude_module_path": ".*logits_dense.*",
                    "target_scan_indices": "0",
                    "multipass_mode": "1xe3m0",
                }
            ]
        ),
        quantization_exclude_module_path="",
        quantization_start_step=0,
        quantization_end_step=-1,
        quantization_target_scan_indices="",
        scan_layers=True,
        num_decoder_layers=2,
        param_scan_axis=0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      in_path = f"{tmpdir}/ckpt_in"
      out_path = f"{tmpdir}/ckpt_out"
      ocp.PyTreeCheckpointer().save(in_path, params)
      quantize_checkpoint.quantize_checkpoint_from_config(in_path, out_path, cfg)
      restored = ocp.PyTreeCheckpointer().restore(out_path)
      wi = restored["decoder"]["layers"]["wi"]
      self.assertAlmostEqual(float(jnp.sum(wi[0])), 16.0, places=3)
      self.assertAlmostEqual(float(jnp.sum(wi[1])), 20.0, places=3)
      self.assertAlmostEqual(float(jnp.sum(restored["decoder"]["logits_dense"]["kernel"])), 20.0, places=3)

  def test_end_to_end_qwix_training_workflow(self):
    cfg = SimpleNamespace(
        quantization="qwix",
        use_qwix_quantization=True,
        use_batch_split_schedule=False,
        micro_batch_size_to_train_on=2,
        max_target_length=4,
        mtp_num_layers=0,
        quantization_target_roles=["actor"],
        quantization_exclude_module_path=".*logits_dense.*",
        quantization_start_step=2,
        quantization_end_step=5,
        quantization_target_scan_indices="1",
        quantization_rules_json=json.dumps(
            [
                {
                    "module_path": ".*",
                    "multipass_mode": "1xe3m0",
                    "dlhs_multipass_mode": "2xe3m0",
                    "drhs_multipass_mode": "2xe3m0",
                }
            ]
        ),
    )
    model = _ToyModel(num_layers=2, features=4, rngs=nnx.Rngs(0))
    q_model = quantizations.maybe_quantize_model(model, cfg, role="actor")
    graphdef, state = nnx.split(q_model)

    @jax.jit
    def train_step_fn(st, step, x):
      m = nnx.merge(graphdef, st)
      with qwix.step_context(step):
        loss, grads = nnx.value_and_grad(lambda mod, inp: jnp.sum(mod(inp)))(m, x)
      return loss, grads

    x = jnp.ones((2, 4), dtype=jnp.float32)
    # Step 1 (before start_step=2): all 3 projections (layer 0, layer 1, logits_dense) multiply by 5.0 -> 5^3 = 125 * 8 = 1000.0
    loss_step1, _ = train_step_fn(state, jnp.int32(1), x)
    self.assertAlmostEqual(float(loss_step1), 1000.0, places=2)

    # Step 3 (active): layer 1 (target_scan_indices="1", logits_dense excluded) uses 1-pass e3m0 (5.0 -> 4.0, 1.25 -> 1.0 -> 16.0),
    # then logits_dense multiplies by 5.0 -> 80.0 * 8 = 640.0
    loss_step3, grads_step3 = train_step_fn(state, jnp.int32(3), x)
    self.assertAlmostEqual(float(loss_step3), 640.0, places=2)
    self.assertEqual(grads_step3.layers.proj.kernel[...].shape, (2, 4, 4))

    # Step 6 (after end_step=5): inactive again -> 1000.0
    loss_step6, _ = train_step_fn(state, jnp.int32(6), x)
    self.assertAlmostEqual(float(loss_step6), 1000.0, places=2)


if __name__ == "__main__":
  absltest.main()
