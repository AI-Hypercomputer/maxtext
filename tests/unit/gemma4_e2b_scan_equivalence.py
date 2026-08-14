import os
import sys
import numpy as np

# Let it run on TPU if available or CPU
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import chex

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from maxtext.configs import pyconfig
from maxtext.models.models import transformer_as_linen

import importlib.util

def get_model_and_config():
    config = pyconfig.initialize([
        "",
        "src/maxtext/configs/base.yml",
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
        "scan_layers=True",
        "override_model_config=True",
        "pure_nnx_decoder=True",
        "enable_nnx=True",
        "skip_jax_distributed_system=True",
        "remat_policy=full",
    ])
    # The config remat policy override isn't strictly necessary if it works with 'full'
    # but the draft had object.__setattr__(config, 'remat_policy', "none")
    # I will keep it None to ensure no remat issues alter the parity equivalence.
    object.__setattr__(config, 'remat_policy', "none")
    mesh = jax.sharding.Mesh(jax.devices(), ('data',))
    model = transformer_as_linen(config=config, mesh=mesh, quant=None, model_mode="train")
    return model, config

def get_loss_and_grads(model, rng_key, inputs):
    init_key, drop_key = jax.random.split(rng_key)
    positions = jnp.arange(16)[None, :]
    segment_ids = jnp.ones((1, 16), dtype=jnp.int32)
    
    # Initialize variables
    variables = model.init({'params': init_key, 'dropout': drop_key}, inputs, positions, segment_ids)
    
    def loss_fn(params):
        call_vars = dict(variables)
        call_vars['params'] = params
        
        logits = model.apply(
            call_vars, 
            inputs,
            positions,
            segment_ids,
            rngs={'dropout': drop_key}
        )
        if isinstance(logits, tuple):
            logits = logits[0]
        return jnp.sum(logits)

    loss, grads = jax.value_and_grad(loss_fn)(variables['params'])
    return loss, grads

def assert_pytrees_all_close(tree1, tree2, rtol, atol):
    chex.assert_trees_all_close(tree1, tree2, rtol=rtol, atol=atol)

def test_gemma4_e2b_scan_equivalence():
    rng = jax.random.PRNGKey(42)
    inputs = jax.random.randint(rng, (1, 16), 0, 100)
    
    model, _ = get_model_and_config()
    from maxtext.layers import nnx_decoders
    
    import urllib.request, types, subprocess
    # Golden Execution: Load from a pinned commit before scan changes (the original unrolled loop)
    # Since the pod is not a git repo, we fetch directly from GitHub:
    url = "https://raw.githubusercontent.com/google/maxtext/1d76f5fc7853f66ce14baab26e6bdcf85a1232fc/src/maxtext/layers/nnx_decoders.py"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            orig_code = response.read().decode("utf-8")
    except Exception as e:
        print(f"Failed to fetch from GitHub ({e}), falling back to local git show...")
        # Fallback for local testing environments where GitHub egress is blocked
        import subprocess
        orig_code = subprocess.check_output(
            ["git", "show", "1d76f5fc7853f66ce14baab26e6bdcf85a1232fc:src/maxtext/layers/nnx_decoders.py"]
        ).decode("utf-8")
    
    nnx_decoders_orig = types.ModuleType("nnx_decoders_orig")
    exec(orig_code, nnx_decoders_orig.__dict__)
    unrolled_apply = nnx_decoders_orig.NNXDecoder._apply_gemma4_small_layers

    # Scanned Execution with Fix: Use our newly patched local code directly
    patched_apply = nnx_decoders.NNXDecoder._apply_gemma4_small_layers

    try:
        # Golden (Unrolled)
        nnx_decoders.NNXDecoder._apply_gemma4_small_layers = unrolled_apply
        golden_loss, golden_grads = get_loss_and_grads(model, rng, inputs)
        
        # Scanned (Patched local version)
        nnx_decoders.NNXDecoder._apply_gemma4_small_layers = patched_apply
        scanned_loss, scanned_grads = get_loss_and_grads(model, rng, inputs)
    finally:
        pass
        
    assert jnp.allclose(golden_loss, scanned_loss), f"Loss does not match! {golden_loss} vs {scanned_loss}"
    
    # 1e-2 tolerance as specified
    assert_pytrees_all_close(golden_grads, scanned_grads, rtol=1e-2, atol=1e-2)
    print("SUCCESS: Scan equivalence test passed with rtol=1e-2!")

if __name__ == "__main__":
    test_gemma4_e2b_scan_equivalence()
