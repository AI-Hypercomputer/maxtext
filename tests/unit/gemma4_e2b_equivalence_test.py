import os
import sys
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from maxtext.configs import pyconfig
from maxtext.models.models import TransformerLinenPure

def get_model_and_config(remat_policy):
    config = pyconfig.initialize([
        "",
        "src/maxtext/configs/models/gemma4-e2b.yml",
        "compile_topology=v5e-1",
        "compile_topology_num_slices=1",
        "per_device_batch_size=1",
        "ici_fsdp_parallelism=1",
        "ici_tensor_parallelism=1",
        "max_target_length=16",
        "base_num_decoder_layers=10",
        "num_kv_shared_layers=5",
        "vocab_size=128",
        "vocab_size_per_layer_input=128",
        "dataset_type=synthetic",
        "model_name=gemma4-e2b",
        "scan_layers=False",
        "override_model_config=True",
    ], remat_policy="full")
    # Forcefully bypass read-only config
    object.__setattr__(config, 'remat_policy', remat_policy)
    mesh = jax.sharding.Mesh(jax.devices(), ('data',))
    model = TransformerLinenPure(config=config, mesh=mesh, quant=None, model_mode="train")
    return model, config

def get_loss_and_grads(model, rng_key, inputs):
    init_key, drop_key = jax.random.split(rng_key)
    positions = jnp.arange(16)[None, :]
    segment_ids = jnp.ones((1, 16), dtype=jnp.int32)
    
    # Initialize variables
    variables = model.init({'params': init_key, 'dropout': drop_key}, inputs, positions, segment_ids)
    
    def loss_fn(params):
        logits = model.apply(
            {'params': params}, 
            inputs,
            positions,
            segment_ids,
            rngs={'dropout': drop_key}
        )
        # Handle the case where TransformerLinenPure returns multiple outputs
        if isinstance(logits, tuple):
            logits = logits[0]
        return jnp.sum(logits)

    loss, grads = jax.value_and_grad(loss_fn)(variables['params'])
    return loss, grads

def test_gemma4_e2b_equivalence():
    rng = jax.random.PRNGKey(42)
    inputs = jax.random.randint(rng, (1, 16), 0, 100)
    
    # Run without remat
    model_no_remat, _ = get_model_and_config("none")
    loss_no_remat, grads_no_remat = get_loss_and_grads(model_no_remat, rng, inputs)
    
    # Run with remat
    model_remat, _ = get_model_and_config("minimal")
    loss_remat, grads_remat = get_loss_and_grads(model_remat, rng, inputs)
    
    # Assert loss matches
    assert jnp.allclose(loss_no_remat, loss_remat), "Loss does not match between remat and no-remat!"
    
    # Assert gradients match
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg="Gradients do not match!"),
        grads_no_remat, grads_remat
    )
    
    print("SUCCESS: Forward and backward pass are perfectly identical with and without remat!")

if __name__ == "__main__":
    test_gemma4_e2b_equivalence()
