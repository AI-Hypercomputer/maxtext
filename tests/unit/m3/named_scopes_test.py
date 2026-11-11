# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `maxtext.m3.infra.named_scopes`."""

import re
from types import SimpleNamespace

from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.infra import scan
from maxtext.m3.infra.named_scopes import add_named_scopes, scope_of
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3MLP, create_qwen3_model


def tiny_config(scan_layers=False):
  """Only the configuration consumed by the dense Qwen3 model."""
  return SimpleNamespace(
      scan_layers=scan_layers,
      param_scan_axis=1,
      num_decoder_layers=3,
      emb_dim=16,
      mlp_dim=32,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=4,
      vocab_size=32,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      rope_max_timescale=1000000,
      float32_logits=True,
      logits_dot_in_fp32=True,
      normalize_embedding_logits=False,
      final_logits_soft_cap=0,
      cast_logits_to_fp32=True,
  )


TOKENS = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])
POSITIONS = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]])


def forward(model):
  return model(TOKENS, POSITIONS)


def loss(model):
  return -jnp.mean(jax.nn.log_softmax(forward(model))[..., 0])


def flat_params(model):
  return {path: value[...] for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def build(scan_layers):
  config = tiny_config(scan_layers)
  return scan.maybe_apply_scan(create_qwen3_model(config, rngs=nnx.Rngs(0)), config)


def compiled(model):
  return nnx.jit(forward).lower(model).compile().as_text()


def op_names(hlo_text):
  """The `op_name` metadata of every instruction: the name stack a profiler shows."""
  return set(re.findall(r'op_name="([^"]*)"', hlo_text))


def instructions_modulo_names(hlo_text):
  """Compiled instructions with instruction names and metadata removed, as a sorted list."""
  lines = []
  for line in hlo_text.splitlines():
    if " = " not in line:
      continue
    line = re.sub(r", metadata=\{[^}]*\}", "", line)
    line = re.sub(r"%[\w.\-/]+", "%_", line)
    lines.append(line.strip())
  return sorted(lines)


class NamedScopesTest(parameterized.TestCase):

  @parameterized.named_parameters(("unrolled", False), ("scanned", True))
  def test_scopes_name_modules_by_attribute_path(self, scan_layers):
    model = add_named_scopes(build(scan_layers))
    names = op_names(compiled(model))
    prefix = "decoder/layers/while/body/" if scan_layers else "decoder/layers/"
    layer = "" if scan_layers else "1/"
    for scope in (
        f"{layer}mlp/wi_0/",
        f"{layer}mlp/wi_1/",
        f"{layer}mlp/wo/",
        f"{layer}self_attention/query/",
        f"{layer}self_attention/out/",
        f"{layer}self_attention/query_norm/",
        f"{layer}pre_self_attention_layer_norm/",
    ):
      self.assertTrue(any(prefix in n and scope in n for n in names), msg=f"no op under {prefix}...{scope}")
    # The attention scores are inline code of the attention module; they land in its scope.
    self.assertTrue(any(f"{layer}self_attention/" in n and "->bkgts" in n for n in names))
    self.assertTrue(any("decoder/apply_output_head/decoder_norm/" in n for n in names))
    self.assertTrue(any("decoder/token_embedder/" in n for n in names))

  def test_nothing_is_scoped_without_the_transform(self):
    names = op_names(compiled(build(False)))
    self.assertFalse(any("mlp/wi_0" in n or "self_attention" in n for n in names))

  @parameterized.named_parameters(("unrolled", False), ("scanned", True))
  def test_computation_and_parameters_unchanged(self, scan_layers):
    model = build(scan_layers)
    paths = list(flat_params(model))
    out = forward(model)
    loss_value, grads = nnx.value_and_grad(loss)(model)
    jit_out = nnx.jit(forward)(model)
    program = instructions_modulo_names(compiled(model))

    add_named_scopes(model)
    self.assertEqual(list(flat_params(model)), paths)
    np.testing.assert_array_equal(forward(model), out)
    scoped_loss, scoped_grads = nnx.value_and_grad(loss)(model)
    np.testing.assert_array_equal(scoped_loss, loss_value)
    for path, value in flat_params(grads).items():
      np.testing.assert_array_equal(flat_params(scoped_grads)[path], value, err_msg=str(path))
    np.testing.assert_array_equal(nnx.jit(forward)(model), jit_out)
    self.assertEqual(instructions_modulo_names(compiled(model)), program)

  @parameterized.named_parameters(("unrolled", False), ("scanned", True))
  def test_classes_stay_recognisable_and_transform_is_idempotent(self, scan_layers):
    model = add_named_scopes(build(scan_layers))
    mlp = model.decoder.layers.mlp if scan_layers else model.decoder.layers.layers[2].mlp
    self.assertIsInstance(mlp, Qwen3MLP)
    self.assertEqual(type(mlp).__name__, "Qwen3MLP")
    self.assertIsInstance(mlp.wi_0, nnx.Linear)
    self.assertEqual(scope_of(mlp), "mlp")
    self.assertEqual(scope_of(mlp.wi_0), "wi_0")
    self.assertEqual(scope_of(model.decoder.layers), "layers")
    if not scan_layers:
      self.assertEqual(scope_of(model.decoder.layers.layers[1]), "1")
    self.assertIsNone(scope_of(model))
    classes = {id(node): type(node) for _, node in nnx.iter_graph(model) if isinstance(node, nnx.Module)}
    add_named_scopes(model)
    for _, node in nnx.iter_graph(model):
      if isinstance(node, nnx.Module):
        self.assertIs(type(node), classes[id(node)])

  def test_graphdef_is_stable_across_models(self):
    # Cached scoped classes: two models built the same way share one graphdef, so `nnx.jit` caches hit.
    a = add_named_scopes(build(False))
    b = add_named_scopes(build(False))
    self.assertEqual(nnx.graphdef(a), nnx.graphdef(b))

  def test_moved_module_takes_its_new_scope(self):
    class Holder(nnx.Module):

      def __init__(self, inner):
        self.first = inner

      def __call__(self, x):
        return (self.second if hasattr(self, "second") else self.first)(x)

    model = add_named_scopes(Holder(nnx.Linear(4, 4, rngs=nnx.Rngs(0))))
    linear = model.first
    del model.first
    model.second = linear
    add_named_scopes(model)
    self.assertEqual(scope_of(linear), "second")
    self.assertIs(type(linear).__bases__[0], nnx.Linear)
    names = op_names(nnx.jit(lambda m, x: m(x)).lower(model, jnp.ones((2, 4))).compile().as_text())
    self.assertTrue(any("second/" in n for n in names), names)
    self.assertFalse(any("first/" in n for n in names), names)

  def test_shared_module_is_scoped_once(self):
    class Tied(nnx.Module):

      def __init__(self):
        self.a = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
        self.b = self.a

    model = add_named_scopes(Tied())
    self.assertIs(model.a, model.b)
    self.assertEqual(scope_of(model.a), "a")
    self.assertIs(type(model.a).__bases__[0], nnx.Linear)
    # Documented limitation: one instance has one scope, so a call through `b` reads `a/`.
    names = op_names(nnx.jit(lambda m, x: m.b(x)).lower(model, jnp.ones((2, 4))).compile().as_text())
    self.assertTrue(any("a/dot_general" in n for n in names), names)
    self.assertFalse(any("b/" in n for n in names), names)

  def test_public_methods_nest_under_the_module_scope(self):
    class HyperConnection(nnx.Module):

      def __init__(self):
        self.proj = nnx.Linear(4, 4, rngs=nnx.Rngs(0))

      def before_block(self, x):
        return jnp.sin(self.proj(x))

      def apply_block(self, x):
        return jnp.cos(self.before_block(2 * x))  # `2 *` keeps XLA from merging it with the direct call.

    class Layer(nnx.Module):

      def __init__(self):
        self.attn_hyper_connection = HyperConnection()

      def __call__(self, x):
        return self.attn_hyper_connection.before_block(x) * self.attn_hyper_connection.apply_block(x)

    class Root(nnx.Module):

      def __init__(self):
        self.layer = Layer()

    model = add_named_scopes(Root())
    names = op_names(nnx.jit(lambda m, x: m.layer(x)).lower(model, jnp.ones((2, 4))).compile().as_text())
    hc = "layer/attn_hyper_connection"
    for scope in (
        f"{hc}/before_block/proj/dot_general",
        f"{hc}/before_block/sin",
        f"{hc}/apply_block/before_block/proj/dot_general",
        f"{hc}/apply_block/cos",
    ):
      self.assertTrue(any(scope in n for n in names), f"no op under {scope}: {names}")
    # Every method op carries the module segment, and a self-call does not repeat it.
    self.assertFalse(any("block" in n and "attn_hyper_connection/" not in n for n in names), names)
    self.assertFalse(any(f"{hc}/apply_block/attn_hyper_connection" in n for n in names), names)

  def test_iterated_list_items_carry_the_list_name(self):
    class Layer(nnx.Module):

      def __init__(self, seed):
        self.q_proj = nnx.Linear(4, 4, rngs=nnx.Rngs(seed))

      def __call__(self, x):
        return jnp.sin(self.q_proj(x))

    class Model(nnx.Module):
      """GLM-style layout: two lists of layers the model loops over, rather than an `nnx.Sequential`."""

      def __init__(self):
        self.prefix_layers = nnx.List([Layer(0), Layer(1)])
        self.suffix_layers = nnx.List([Layer(2), Layer(3)])

      def __call__(self, x):
        for layer in self.prefix_layers:
          x = layer(x)
        for layer in self.suffix_layers:
          x = layer(x)
        return x

    model = add_named_scopes(Model())
    self.assertEqual(scope_of(model.prefix_layers[0]), "prefix_layers/0")
    self.assertEqual(scope_of(model.suffix_layers[0]), "suffix_layers/0")
    names = op_names(nnx.jit(lambda m, x: m(x)).lower(model, jnp.ones((2, 4))).compile().as_text())
    for scope in ("prefix_layers/0", "prefix_layers/1", "suffix_layers/0", "suffix_layers/1"):
      self.assertTrue(any(f"{scope}/q_proj/dot_general" in n for n in names), f"no op under {scope}: {names}")
    # No layer is named by its bare index, the pre-fix `jit(<lambda>)/0/q_proj/...`.
    self.assertFalse(any(re.search(r"\)/\d+/", n) for n in names), names)
    classes = {id(node): type(node) for _, node in nnx.iter_graph(model) if isinstance(node, nnx.Module)}
    add_named_scopes(model)
    for _, node in nnx.iter_graph(model):
      if isinstance(node, nnx.Module):
        self.assertIs(type(node), classes[id(node)])


if __name__ == "__main__":
  absltest.main()
