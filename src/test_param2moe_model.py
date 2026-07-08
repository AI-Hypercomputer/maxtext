import sys
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models import models
from maxtext.utils import maxtext_utils

def test_param2moe():
  # Create a minimal config for param2moe
  config = pyconfig.initialize(
      [
          sys.argv[0],
          "src/maxtext/configs/base.yml",
          "decoder_block=param2moe",
          "base_num_decoder_layers=3",
          "first_num_dense_layers=1",
          "num_experts=4",
          "num_experts_per_tok=2",
          "shared_experts=2",

          "base_emb_dim=64",
          "base_mlp_dim=128",
          "base_moe_mlp_dim=64",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=16",
          "use_qk_norm=True",

          "max_target_length=16",
          "per_device_batch_size=1",
          "run_name=test_param2moe",
          "enable_checkpointing=False",
          "attention=dot_product",
      ]
  )

  print("Config initialized successfully.")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Instantiate model
  model = models.transformer_as_linen(config=config, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
  print("Model class instantiated.")

  # Mock inputs
  rng = jax.random.PRNGKey(0)
  s = (config.global_batch_size_to_train_on, config.max_target_length)
  ids = jax.random.randint(rng, s, 0, config.vocab_size)
  decoder_positions = jnp.stack([jnp.arange(config.max_target_length, dtype=jnp.int32) for _ in range(config.global_batch_size_to_train_on)])
  decoder_segment_ids = jnp.zeros(s)

  print("Initializing model variables...")
  variables = model.init(
      {"params": rng, "aqt": rng},
      ids,
      decoder_positions,
      model_mode=MODEL_MODE_TRAIN,
      decoder_segment_ids=decoder_segment_ids,
      enable_dropout=False,
  )
  print("Model variables initialized successfully!")
  
  # Try a forward pass
  print("Running forward pass...")
  logits = model.apply(
      variables,
      ids,
      decoder_positions,
      model_mode=MODEL_MODE_TRAIN,
      decoder_segment_ids=decoder_segment_ids,
      enable_dropout=False,
      rngs={"aqt": rng},
  )
  print("Forward pass successful! Logits shape:", logits.shape)

if __name__ == "__main__":
  test_param2moe()
