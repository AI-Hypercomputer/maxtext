import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch

# 1. Add parent directory to find 'generated_code'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Add the project root 'src' to find 'MaxText'
# Path trace: tests -> integrative_rag_agent -> agent -> experimental -> MaxText -> src
# We need to go up 5 levels to reach 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.append(project_root)

# Force JAX to use strict float32 for matmuls (simulating PyTorch behavior)
jax.config.update("jax_default_matmul_precision", "float32")

from generated_code.SelfAttention.layers import SelfAttention as SelfAttentionJAX
from examples.level2_pytorch_module.self_attention import SelfAttention as SelfAttentionPT


def run_test():
    print("Initializing models...")
    EMBED_SIZE = 32
    HEADS = 4
    BATCH = 2
    SEQ_LEN = 10

    # PyTorch Init
    pt_model = SelfAttentionPT(EMBED_SIZE, HEADS)
    pt_model.eval()

    # JAX Init
    jax_model = SelfAttentionJAX(EMBED_SIZE, HEADS)
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((BATCH, SEQ_LEN, EMBED_SIZE))
    variables = jax_model.init(key, dummy_input, dummy_input, dummy_input, None)
    params = variables['params']

    print("Copying weights from PyTorch to JAX...")
    # Helper to transpose PyTorch (Out, In) -> JAX (In, Out)
    def copy_linear(pt_layer, jax_path):
        # PyTorch weights are [Out, In], JAX are [In, Out]
        # maxtext DenseGeneral uses 'kernel' for weights
        params[jax_path]['kernel'] = pt_layer.weight.detach().numpy().T
        
        # Bias handling
        if pt_layer.bias is not None:
            if 'bias' in params[jax_path]:
                params[jax_path]['bias'] = pt_layer.bias.detach().numpy()
            else:
                print(f"WARNING: {jax_path} has no bias in JAX, but PyTorch does.")

    copy_linear(pt_model.values, 'values_projection')
    copy_linear(pt_model.keys, 'keys_projection')
    copy_linear(pt_model.queries, 'queries_projection')
    copy_linear(pt_model.fc_out, 'output_projection')

    # Inputs
    np_input = np.random.randn(BATCH, SEQ_LEN, EMBED_SIZE).astype(np.float32)
    
    # Forward Pass PyTorch
    print("Running PyTorch forward pass...")
    pt_in = torch.from_numpy(np_input)
    with torch.no_grad():
        pt_out = pt_model(pt_in, pt_in, pt_in, None)

    # Forward Pass JAX
    print("Running JAX forward pass...")
    jax_out = jax_model.apply({'params': params}, np_input, np_input, np_input, None)

    # Comparison
    diff = np.abs(pt_out.numpy() - np.array(jax_out))
    max_diff = diff.max()
    print(f"\nMax absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("SUCCESS: The JAX implementation matches PyTorch!")
    else:
        print("FAILURE: The outputs diverge.")

if __name__ == "__main__":
    run_test()