# Copyright 2023–2026 Google LLC
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

"""Unit tests for the LearnToInitDense layer in Learn-To-Init (LTI) distillation."""

import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from absl.testing import absltest

# Import the module under test
from maxtext.layers.learn_to_init_layer import LearnToInitDense
from maxtext.trainers.post_train.distillation.lti_utils import prepare_student_weights
from unittest import mock
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.layers.learn_to_init_layer import LearnToInitDecoderLayer


# Minimal dummy models for testing
class DummyLayer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    # A simple linear kernel initialized with random normal distribution
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))


class DummyModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.layer1 = DummyLayer(rngs)
    self.layer2 = DummyLayer(rngs)


class PrepareStudentWeightsTest(unittest.TestCase):

  def test_prepare_student_weights_copy_only(self):
    """Verifies that teacher weights are correctly copied into the student model."""
    teacher = DummyModel(nnx.Rngs(0))
    student = DummyModel(nnx.Rngs(1))

    # Ensure the initialized kernels are initially different
    self.assertFalse(jnp.array_equal(teacher.layer1.kernel.value, student.layer1.kernel.value))
    self.assertFalse(jnp.array_equal(teacher.layer2.kernel.value, student.layer2.kernel.value))

    # Map teacher's layer1 to student's layer1
    copy_map = {"layer1/kernel": "layer1/kernel"}

    prepare_student_weights(
        student_model=student, teacher_model=teacher, teacher_weights_copy_map=copy_map, student_weights_share_map={}
    )

    # Verify that layer1 was copied over
    self.assertTrue(jnp.array_equal(student.layer1.kernel.value, teacher.layer1.kernel.value))

    # Verify that layer2 remained untouched
    self.assertFalse(jnp.array_equal(student.layer2.kernel.value, teacher.layer2.kernel.value))

  def test_prepare_student_weights_share_and_copy(self):
    """Verifies the behavior when using the experimental weight sharing map."""
    teacher = DummyModel(nnx.Rngs(0))
    student = DummyModel(nnx.Rngs(1))

    # We share layer1's node to layer2's path in the local dictionary.
    # This means any subsequent copy targeted at "layer2/kernel" will actually write
    # into the student's layer1 node.
    share_map = {"layer1/kernel": "layer2/kernel"}

    # We copy the teacher's layer2 into the student's layer2 path
    copy_map = {"layer2/kernel": "layer2/kernel"}

    prepare_student_weights(
        student_model=student,
        teacher_model=teacher,
        teacher_weights_copy_map=copy_map,
        student_weights_share_map=share_map,
    )

    # Since student's layer2 was "shared" from layer1, the copy operation
    # overwrites student's layer1.
    self.assertTrue(jnp.array_equal(student.layer1.kernel.value, teacher.layer2.kernel.value))

    # The actual layer2 of the student remains unchanged because the dictionary
    # reference was rerouted. We verify it still has its original initialization.
    student_original_layer2 = DummyModel(nnx.Rngs(1)).layer2.kernel.value
    self.assertTrue(jnp.array_equal(student.layer2.kernel.value, student_original_layer2))

  def test_prepare_student_weights_shape_mismatch(self):
    """Verifies that an error is raised when trying to copy misaligned shapes."""
    teacher = DummyModel(nnx.Rngs(0))
    student = DummyModel(nnx.Rngs(1))

    # Modify student shape manually to force a mismatch
    student.layer1.kernel.value = jnp.zeros((8, 8))

    copy_map = {"layer1/kernel": "layer1/kernel"}

    with self.assertRaisesRegex(AssertionError, "Shape mismatch for layer1/kernel"):
      prepare_student_weights(
          student_model=student, teacher_model=teacher, teacher_weights_copy_map=copy_map, student_weights_share_map={}
      )


class LearnToInitDenseTest(unittest.TestCase):

  def test_qkv_projection_standard_map(self):
    """Verifies parameter shapes and forward pass for QKV-like projection (is_output_projection=False)."""
    embed_dim = 16
    teacher_heads = 4
    teacher_head_dim = 8

    # C shape for Q,K,V projections: (embed_dim, teacher_heads, teacher_head_dim)
    C = jnp.ones((embed_dim, teacher_heads, teacher_head_dim))

    student_heads = 2
    student_head_dim = 16

    layer = LearnToInitDense(
        in_features_shape=(embed_dim,),
        out_features_shape=(student_heads, student_head_dim),
        C=C,
        is_output_projection=False,
        use_general_linear_map=False,
        rngs=nnx.Rngs(0),
    )

    # Verify initialized parameters
    # A maps teacher_heads -> student_heads
    self.assertEqual(layer.A.value.shape, (teacher_heads, student_heads))
    # B maps teacher_head_dim -> student_head_dim
    self.assertEqual(layer.B.value.shape, (teacher_head_dim, student_head_dim))
    self.assertEqual(layer.C.value.shape, (embed_dim, teacher_heads, teacher_head_dim))

    # Verify forward pass shape
    batch_size = 2
    seq_len = 5
    x = jnp.ones((batch_size, seq_len, embed_dim))
    out = layer(x)
    self.assertEqual(out.shape, (batch_size, seq_len, student_heads, student_head_dim))

  def test_out_projection_standard_map(self):
    """Verifies parameter shapes and forward pass for Output projection (is_output_projection=True)."""
    embed_dim = 16
    teacher_heads = 4
    teacher_head_dim = 8

    # C shape for Output projection: (teacher_heads, teacher_head_dim, embed_dim)
    C = jnp.ones((teacher_heads, teacher_head_dim, embed_dim))

    student_heads = 2
    student_head_dim = 16

    layer = LearnToInitDense(
        in_features_shape=(student_heads, student_head_dim),
        out_features_shape=(embed_dim,),
        C=C,
        axis=(-2, -1),  # Reduce over the student heads and head_dim
        is_output_projection=True,
        use_general_linear_map=False,
        rngs=nnx.Rngs(0),
    )

    # Verify initialized parameters
    # A maps teacher_heads -> student_heads
    self.assertEqual(layer.A.value.shape, (teacher_heads, student_heads))
    # B maps student_head_dim -> teacher_head_dim
    self.assertEqual(layer.B.value.shape, (student_head_dim, teacher_head_dim))

    # Verify forward pass shape
    batch_size = 2
    seq_len = 5
    x = jnp.ones((batch_size, seq_len, student_heads, student_head_dim))
    out = layer(x)
    self.assertEqual(out.shape, (batch_size, seq_len, embed_dim))

  def test_qkv_projection_general_map(self):
    """Verifies parameter shapes and forward pass for QKV-like projection with a general map (W)."""
    embed_dim = 16
    teacher_heads = 4
    teacher_head_dim = 8
    C = jnp.ones((embed_dim, teacher_heads, teacher_head_dim))

    student_heads = 2
    student_head_dim = 16

    layer = LearnToInitDense(
        in_features_shape=(embed_dim,),
        out_features_shape=(student_heads, student_head_dim),
        C=C,
        is_output_projection=False,
        use_general_linear_map=True,
        rngs=nnx.Rngs(0),
    )

    # Verify W tensor shape is correctly formatted as (x, y, u, v)
    self.assertEqual(layer.W.value.shape, (teacher_heads, teacher_head_dim, student_heads, student_head_dim))

    # Verify forward pass shape
    batch_size = 2
    seq_len = 5
    x = jnp.ones((batch_size, seq_len, embed_dim))
    out = layer(x)
    self.assertEqual(out.shape, (batch_size, seq_len, student_heads, student_head_dim))


class LearnToInitDecoderLayerTest(unittest.TestCase):

  def test_llama_lti_decoder_layer_initialization(self):
    """Verifies LearnToInitDecoderLayer initializes and modifies LlamaDecoderLayer correctly."""

    # 1. Setup mock teacher config
    mock_teacher_config = mock.MagicMock()
    mock_teacher_config.base_num_query_heads = 4
    mock_teacher_config.base_num_kv_heads = 2
    mock_teacher_config.head_dim = 16

    # 2. Setup mock student config
    mock_config = mock.MagicMock()
    mock_config.lti_use_general_linear_map = False
    mock_config.teacher_config = mock_teacher_config

    # Add attributes strictly required by LlamaDecoderLayer and Attention sub-layers
    mock_config.emb_dim = 64
    mock_config.dtype = jnp.float32
    mock_config.weight_dtype = jnp.float32
    mock_config.shard_mode = "auto"
    mock_config.normalization_layer_epsilon = 1e-6
    mock_config.num_query_heads = 4
    mock_config.num_kv_heads = 2
    mock_config.head_dim = 16
    mock_config.max_target_length = 32
    mock_config.max_prefill_predict_length = 32
    mock_config.attention = "dot_product"
    mock_config.dropout_rate = 0.0
    mock_config.float32_qk_product = False
    mock_config.float32_logits = False
    mock_config.prefill_cache_axis_order = "0,1,2,3"
    mock_config.ar_cache_axis_order = "0,1,2,3"
    mock_config.compute_axis_order = "0,1,2,3"
    mock_config.reshape_q = False
    mock_config.use_ragged_attention = False
    mock_config.ragged_block_size = 16
    mock_config.attn_logits_soft_cap = 0.0
    mock_config.mlp_dim = 128
    mock_config.mlp_activations = ["silu", "linear"]
    mock_config.debug_sharding = False
    mock_config.record_internal_nn_metrics = False
    mock_config.scan_layers = False
    mock_config.ici_context_autoregressive_parallelism = 1
    mock_config.fused_qkv = False

    # 3. Dummy Jax sharding mesh and NNX Rngs
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    rngs = nnx.Rngs(0)

    # Patch utility functions to isolate the test from deeper external dependencies
    with (
        mock.patch("maxtext.utils.max_utils.get_batch_seq_len_for_mode", return_value=(2, 32)),
        mock.patch("maxtext.layers.quantizations.configure_kv_quant", return_value=None),
    ):

      # This effectively initializes LlamaLTIDecoderLayer and implicitly calls _customize_attention_modules
      layer = LearnToInitDecoderLayer(
          base_layer_cls=LlamaDecoderLayer,
          config=mock_config,
          model_mode="train",
          mesh=mesh,
          rngs=rngs,
      )

    # 4. Verify initialization result
    self.assertIsInstance(layer.learn_to_init_wrapper, LlamaDecoderLayer)
    self.assertEqual(layer.self_attention_module_name, "self_attention")

    # 5. Verify the behavior of _customize_attention_modules
    # It should correctly replace query, key, value, and out with LearnToInitDense
    attention_module = layer.learn_to_init_wrapper.self_attention

    for proj_name in ["query", "key", "value", "out"]:
      child = getattr(attention_module, proj_name)
      self.assertIsInstance(child, LearnToInitDense, f"{proj_name} was not swapped to LearnToInitDense")

      # Validate that the dummy Teacher Tensor C is dimensioned correctly
      if proj_name == "query":
        # (emb_dim, teacher_heads, head_dim) -> (64, 4, 16)
        self.assertEqual(child.C.value.shape, (64, 4, 16))
      elif proj_name in ("key", "value"):
        # (emb_dim, teacher_kv_heads, head_dim) -> (64, 2, 16)
        self.assertEqual(child.C.value.shape, (64, 2, 16))
      elif proj_name == "out":
        # (teacher_heads, head_dim, emb_dim) -> (4, 16, 64)
        self.assertEqual(child.C.value.shape, (4, 16, 64))


if __name__ == "__main__":
  absltest.main()
