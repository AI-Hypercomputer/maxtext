
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

import jax
import jax.numpy as jnp
from flax import nnx
import unittest
import numpy as np
from maxtext.trainers.post_train.distillation import train_distill
from maxtext.utils import max_logging
import optax

# Mock max_logging to avoid failures if not initialized
max_logging.log = lambda x: print(f"LOG: {x}")

class DummyMoeBlock(nnx.Module):
  def __init__(self, num_experts, d_model, d_ff, rngs):
    self.wi_0 = nnx.Param(jax.random.normal(rngs.params(), (num_experts, d_model, d_ff)))
    self.wi_0_bias = nnx.Param(jnp.zeros((num_experts, d_ff)))
    self.wo = nnx.Param(jax.random.normal(rngs.params(), (num_experts, d_ff, d_model)))
    self.wo_bias = nnx.Param(jnp.zeros((num_experts, d_model)))

class DummyMlpBlock(nnx.Module):
  def __init__(self, d_model, d_ff, rngs):
    self.wi_0 = nnx.Module()
    self.wi_0.kernel = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)))
    self.wi_0.bias = nnx.Param(jnp.zeros((d_ff,)))
    self.wo = nnx.Module()
    self.wo.kernel = nnx.Param(jax.random.normal(rngs.params(), (d_ff, d_model)))
    self.wo.bias = nnx.Param(jnp.zeros((d_model,)))

class DummyModel(nnx.Module):
  def __init__(self, num_experts, shared_d_ff, rngs):
    # Use a simple list for moe_layers to match the path string search "moe_layers"
    self.moe_layers = nnx.List([
      nnx.Dict({
        "MoeBlock_0": DummyMoeBlock(num_experts, 8, 16, rngs),
        "shared_experts": DummyMlpBlock(8, shared_d_ff, rngs)
      })
    ])
    self.dense_layer = DummyMlpBlock(8, 16, rngs)

class SurgeryTest(unittest.TestCase):
  def test_surgery_scanned(self):
    print("\nTesting scanned models...")
    rngs = nnx.Rngs(0)
    # Teacher: 32 experts, 16 shared_d_ff
    teacher = DummyModel(32, 16, rngs)
    # Student: 16 experts, 32 shared_d_ff
    student = DummyModel(16, 32, rngs)

    config = type('Config', (), {'scan_layers': True, 'num_experts': 16})()

    # Simulate scan by adding a dimension (num_layers=2)
    def add_dim(x):
      # x might be an array (nnx.State leaf)
      return jnp.stack([x] * 2)
    
    t_state = nnx.state(teacher)
    t_state_scanned = jax.tree.map(add_dim, t_state)
    nnx.update(teacher, t_state_scanned)

    s_state = nnx.state(student)
    s_state_scanned = jax.tree.map(add_dim, s_state)
    nnx.update(student, s_state_scanned)

    # Run surgery
    train_distill.initialize_student_from_teacher(student, teacher, config)

    # Verify routed experts
    # Student 0th expert should match teacher 0th expert
    np.testing.assert_allclose(student.moe_layers[0]["MoeBlock_0"].wi_0[...][:, 0], 
                               teacher.moe_layers[0]["MoeBlock_0"].wi_0[...][:, 0])
    
    # Verify shared experts
    expected_wi_0 = jnp.concatenate([
        teacher.moe_layers[0]["shared_experts"].wi_0.kernel[...],
        teacher.moe_layers[0]["MoeBlock_0"].wi_0[...] [:, 16]
    ], axis=2)
    np.testing.assert_allclose(student.moe_layers[0]["shared_experts"].wi_0.kernel[...], expected_wi_0)

    print("Scanned model test passed!")

  def test_surgery_non_scanned(self):
    print("\nTesting non-scanned models...")
    rngs = nnx.Rngs(1)
    teacher = DummyModel(32, 16, rngs)
    student = DummyModel(16, 32, rngs)

    config = type('Config', (), {'scan_layers': False, 'num_experts': 16})()

    # Run surgery
    train_distill.initialize_student_from_teacher(student, teacher, config)

    # Verify routed experts
    np.testing.assert_allclose(student.moe_layers[0]["MoeBlock_0"].wi_0[...][0], 
                               teacher.moe_layers[0]["MoeBlock_0"].wi_0[...][0])
    
    # Verify shared experts
    expected_wi_0 = jnp.concatenate([
        teacher.moe_layers[0]["shared_experts"].wi_0.kernel[...],
        teacher.moe_layers[0]["MoeBlock_0"].wi_0[...][16]
    ], axis=1)
    np.testing.assert_allclose(student.moe_layers[0]["shared_experts"].wi_0.kernel[...], expected_wi_0)

    # Verify wo bias
    expected_wo_bias = teacher.moe_layers[0]["shared_experts"].wo.bias[...] + \
                       teacher.moe_layers[0]["MoeBlock_0"].wo_bias[...][16]
    np.testing.assert_allclose(student.moe_layers[0]["shared_experts"].wo.bias[...], expected_wo_bias)
    print("Non-scanned model test passed!")

  def test_selective_freezing(self):
    print("\nTesting selective freezing...")
    rngs = nnx.Rngs(2)
    student = DummyModel(16, 32, rngs)
    teacher = DummyModel(32, 16, rngs)
    bundle = train_distill.ModelBundle(teacher, student)
    
    optimizer = optax.adam(1e-3)
    from tunix.sft import peft_trainer
    training_config = peft_trainer.TrainingConfig(
        max_steps=10,
        eval_every_n_steps=10,
        gradient_accumulation_steps=1
    )
    
    # Mask to only train moe_layers
    mask = ['.*moe_layers.*']
    
    # Mock strategy
    strategy = type('Strategy', (), {})()
    
    trainer = train_distill.MaxTextDistillationTrainer(
        model=bundle,
        strategy=strategy,
        optimizer=optimizer,
        training_config=training_config,
        trainable_parameters_mask=mask
    )
    
    # Check what's in the optimizer state
    opt_state = nnx.state(trainer.optimizer)
    
    # Flatten the state to see which parameters are present
    from jax.tree_util import tree_flatten_with_path
    leaves, _ = tree_flatten_with_path(opt_state)
    
    found_moe = False
    found_dense = False
    
    def get_key(k):
      if hasattr(k, "key"): return str(k.key)
      if hasattr(k, "index"): return str(k.index)
      if hasattr(k, "name"): return str(k.name)
      return str(k)

    for path, leaf in leaves:
      str_path = ".".join(map(get_key, path))
      if "moe_layers" in str_path:
        found_moe = True
      if "dense_layer" in str_path:
        found_dense = True
        
    self.assertTrue(found_moe, "MoE layers should be trainable (present in optimizer state)")
    self.assertFalse(found_dense, "Dense layers should be frozen (NOT present in optimizer state)")
    print("Selective freezing test passed!")

if __name__ == "__main__":
    unittest.main()
