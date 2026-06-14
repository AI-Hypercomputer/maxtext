import unittest
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.trainers.pre_train import train as pre_train
class DeepSeekRoutedBiasTest(unittest.TestCase):
  def setUp(self):
    self.mesh = jax.sharding.Mesh(jax.devices(), ('data',))
  def _make_dummy_data(self, batch=1, seq=16):
    return {
        "inputs": jnp.zeros((batch, seq), dtype=jnp.int32),
        "inputs_position": jnp.broadcast_to(jnp.arange(seq), (batch, seq)),
        "inputs_segmentation": jnp.ones((batch, seq), dtype=jnp.int32),
        "targets": jnp.zeros((batch, seq), dtype=jnp.int32),
        "targets_segmentation": jnp.ones((batch, seq), dtype=jnp.int32),
    }
  def _create_and_run_train_step(self, config_args):
    config = pyconfig.initialize(config_args)
    rngs = jax.nnx.Rngs(0) if hasattr(jax, 'nnx') else __import__('flax.nnx', fromlist=['Rngs']).Rngs(0)
    import flax.nnx as nnx
    from maxtext.common import train_state_nnx
    rngs = nnx.Rngs(0)
    model = models.Transformer(config, self.mesh, quant=None, rngs=rngs)
    data = self._make_dummy_data(batch=config.micro_batch_size_to_train_on, seq=config.max_target_length)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    ts = train_state_nnx.TrainStateNNX(model, optimizer)
    state_graphdef, state_pure = nnx.split(ts)
    new_state, metrics = pre_train.train_step(
        state_graphdef, config, state_mesh_shardings=None, params_shardings=None, state=state_pure, data=data
    )
    return new_state, metrics
  def test_deepseek_v3_dense_routed_bias_success(self):
    """Proves that a DeepSeek V3 model with dense layers (no moe_layers attribute)
    successfully traverses the state tree and updates routed bias without crashing.
    """
    config_args = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=deepseek3-tiny",
        "decoder_block=deepseek",
        "num_decoder_layers=2",
        "per_device_batch_size=1",
        "max_target_length=16",
        "routed_bias=True",
        "routed_bias_update_rate=0.001",
        "skip_jax_distributed_system=True",
        "base_emb_dim=64",
        "base_mlp_dim=64",
        "base_moe_mlp_dim=64",
        "base_num_query_heads=1",
        "base_num_kv_heads=1",
        "num_experts=2",
        "num_experts_per_tok=2",
        "first_num_dense_layers=1",
        "sparse_matmul=False",
        "override_model_config=True",
    ]
    new_state, metrics = self._create_and_run_train_step(config_args)
    self.assertIsNotNone(new_state)
    self.assertIn("learning/loss", metrics["scalar"])
if __name__ == '__main__':
  unittest.main()
