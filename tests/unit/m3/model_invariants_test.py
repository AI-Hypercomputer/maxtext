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

"""Unit tests for Workstream 4: `verify_m3_model_invariants`, `remat`, and `ScannedLayers`."""

# pylint: disable=arguments-differ

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.configs.types import DType
from maxtext.m3.core.interfaces import (
    CausalLMOutput,
    M3ContractViolationError,
    RouterBias,
    verify_m3_model_invariants,
)
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra.remat import apply_remat
from maxtext.m3.infra.scan import ScannedLayers, apply_scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config


class _UnregisteredVar(nnx.Variable):  # pylint: disable=abstract-method
  """Test variable intentionally omitted from `REGISTERED_NONDIFF_VARIABLE_TYPES`."""


class ModelInvariantsAndTransformsTest(absltest.TestCase):
  """Tests `< 200ms` abstract invariant checks and `remat` + `scan` equivalence."""

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])

  def test_qwen3_passes_abstract_invariant_verification(self):
    cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)
    out = verify_m3_model_invariants(
        lambda: create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(0)),
        self.mesh,
    )
    self.assertIsInstance(out, CausalLMOutput)
    self.assertEqual(out.logits.shape, (2, 8, cfg.vocab_size))

  def test_catches_sow_mutation_and_unregistered_variable_and_missing_sharding(self):
    class _SowModel(nnx.Module):
      """Invalid test module that calls `self.sow`."""

      def __init__(self, rngs):
        del rngs
        self.w = nnx.Param(jnp.ones((4, 4)), out_sharding=P("fsdp", "tensor"))

      def __call__(self, inputs, positions, **kwargs):
        self.sow(nnx.Intermediate, "bad", jnp.sum(inputs))
        logits = jnp.zeros((*inputs.shape, 8), dtype=jnp.float32)
        if kwargs.get("return_logits", True):
          return CausalLMOutput(logits=logits, hidden_states=logits)
        return CausalLMOutput(logits=None, hidden_states=logits)

    with self.assertRaisesRegex(M3ContractViolationError, "Intermediate"):
      verify_m3_model_invariants(lambda: _SowModel(nnx.Rngs(0)), self.mesh)

    class _BadVarModel(nnx.Module):
      """Invalid test module holding an unregistered `nnx.Variable` subclass."""

      def __init__(self):
        self.w = nnx.Param(jnp.ones((4, 4)), out_sharding=P("fsdp", "tensor"))
        self.mystery = _UnregisteredVar(jnp.zeros((4,)))

      def __call__(self, inputs, positions, **kwargs):
        logits = jnp.zeros((*inputs.shape, 8), dtype=jnp.float32)
        return CausalLMOutput(logits=logits if kwargs.get("return_logits", True) else None, hidden_states=logits)

    with self.assertRaisesRegex(M3ContractViolationError, "unregistered type"):
      verify_m3_model_invariants(_BadVarModel, self.mesh)

    class _MissingShardingModel(nnx.Module):
      """Invalid test module missing `out_sharding` metadata on a 2D `nnx.Param`."""

      def __init__(self):
        self.w = nnx.Param(jnp.ones((4, 4)))

      def __call__(self, inputs, positions, **kwargs):
        logits = jnp.zeros((*inputs.shape, 8), dtype=jnp.float32)
        return CausalLMOutput(logits=logits if kwargs.get("return_logits", True) else None, hidden_states=logits)

    with self.assertRaisesRegex(M3ContractViolationError, "out_sharding"):
      verify_m3_model_invariants(_MissingShardingModel, self.mesh)

  def test_remat_and_scan_preserve_forward_and_backward_gradients(self):
    cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)
    tokens = jnp.array([[1, 5, 9, 12], [2, 6, 10, 14]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None, :], (2, 4))

    with jax.set_mesh(self.mesh):
      ref_model = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(42))
      tx_model = create_qwen3_model(cfg, self.mesh, rngs=nnx.Rngs(42))
      apply_remat(tx_model, policy="save_qkv_proj", prevent_cse=False)
      apply_scan(tx_model, param_scan_axis=1)

      self.assertIsInstance(tx_model.decoder.layers, ScannedLayers)
      # `wi_0.kernel` `[emb_dim, mlp_dim]` stacked at axis 1 has shape `[emb_dim, num_layers, mlp_dim]`
      self.assertEqual(
          tx_model.decoder.layers.mlp.wi_0.kernel[...].shape,
          (cfg.emb_dim, cfg.num_decoder_layers, cfg.mlp_dim),
      )

      ref_loss, ref_grads = nnx.value_and_grad(lambda m: jnp.mean(m(tokens, positions)))(ref_model)
      tx_loss, tx_grads = nnx.value_and_grad(lambda m: jnp.mean(m(tokens, positions)))(tx_model)

    np.testing.assert_allclose(float(tx_loss), float(ref_loss), atol=1e-5)
    # Compare stacked layer gradients against stacked per-layer reference gradients at axis 1.
    ref_wi0_grads = jnp.stack(
        [
            ref_grads["decoder"]["layers"]["layers"][i]["mlp"]["wi_0"]["kernel"][...]
            for i in range(cfg.num_decoder_layers)
        ],
        axis=1,
    )
    tx_wi0_grads = tx_grads["decoder"]["layers"]["mlp"]["wi_0"]["kernel"][...]
    np.testing.assert_allclose(np.asarray(tx_wi0_grads), np.asarray(ref_wi0_grads), atol=1e-5)

  def test_heterogeneous_sequential_scans_into_sequential_of_runs(self):
    class _LayerA(nnx.Module):
      """First heterogeneous layer variant."""

      def __init__(self, rngs):
        self.lin = nnx.Linear(8, 8, use_bias=False, rngs=rngs)
        self.bias = RouterBias(jnp.zeros((4,)))

      def __call__(self, x):
        return self.lin(x)

    class _LayerB(nnx.Module):
      """Second heterogeneous layer variant."""

      def __init__(self, rngs):
        self.proj = nnx.Linear(8, 8, use_bias=True, rngs=rngs)

      def __call__(self, x):
        return self.proj(x)

    rngs = nnx.Rngs(0)
    seq = nnx.Sequential(_LayerA(rngs), _LayerA(rngs), _LayerB(rngs), _LayerB(rngs))
    x = jnp.ones((2, 8), dtype=jnp.float32)
    expected = seq(x)

    scanned = apply_scan(seq, param_scan_axis=1)
    self.assertIsInstance(scanned, nnx.Sequential)
    self.assertLen(scanned.layers, 2)
    self.assertIsInstance(scanned.layers[0], ScannedLayers)
    self.assertIsInstance(scanned.layers[1], ScannedLayers)
    np.testing.assert_allclose(np.asarray(scanned(x)), np.asarray(expected), atol=1e-5)


if __name__ == "__main__":
  absltest.main()
