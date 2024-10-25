import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import numpy as np
from typing import Sequence
import dataclasses

def create_maxtext_mesh(devices: Sequence[jax.Device]) -> Mesh:
    """
    Creates mesh with exact layout and correct dimensions:
    - 4-way tensor parallelism
    - 2-way sequence parallelism
    Dimensions: [1, 1, 1, 1, 2, 4, 1, 1]
    """
    sorted_devices = sorted(devices, key=lambda d: d.id)
    
    # exact layout we want 
    device_layout = [
        [sorted_devices[0], sorted_devices[4]],  # [0,4]
        [sorted_devices[1], sorted_devices[5]],  # [1,5]
        [sorted_devices[3], sorted_devices[7]],  # [3,7]
        [sorted_devices[2], sorted_devices[6]],  # [2,6]
    ]
    
    result = np.array(device_layout)
    print("\nInitial device layout (4x2):")
    for row in result:
        print([f"D{d.id}" for d in row])
    
    result = result.reshape(1, 1, 1, 1, 2, 4, 1, 1)
    
    mesh_axis_names = (
        'data',          
        'stage',         
        'fsdp',          
        'fsdp_transpose',
        'sequence',      
        'tensor',        
        'expert',        
        'autoregressive' 
    )
    
    print("\nMesh dimensions:")
    print("Dimension order:", " ".join(mesh_axis_names))
    
    # Create mesh and verify shape
    mesh = Mesh(result, mesh_axis_names)
    
    print("\nActual mesh shape:")
    for axis, size in mesh.shape.items():
        print(f"{axis}: {size}")
        assert (axis == 'tensor' and size == 4) or \
               (axis == 'sequence' and size == 2) or \
               size == 1, \
               f"Unexpected size for axis {axis}: {size}"
    
    # Additional validation
    print("\nValidating mesh properties:")
    print(f"Number of devices: {mesh.size}")
    print(f"Device layout shape: {result.shape}")
    print("Axis sizes:", {name: mesh.shape[name] for name in mesh_axis_names})
    
    print(f"{mesh=}")
    return mesh

def get_maxtext_logical_rules():
    """Modified logical rules to be more topology-aware."""
    return [
        # Treat mlp_batch specially to avoid complex routing
        ('mlp_batch', ['data']), # Simplified from ['data', 'fsdp', 'fsdp_transpose', 'expert']
        
        # Map sequence and tensor to topology-aware axes  
        ('mlp_sequence', 'sequence'),
        ('mlp_embedding', 'tensor'),
        
        # Keep other essential rules
        ('embed', 'tensor'),
        ('mlp', 'tensor')
    ]


class LogicallyPartitionedDense(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, inputs):
        # Simplified partitioning
        inputs = nn_partitioning.with_sharding_constraint(
            inputs, P(None, 'sequence', 'tensor')  # Map directly to physical axes
        )
        
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], self.features),
            inputs.dtype,
        )
        bias = self.param(
            'bias',
            nn.initializers.zeros,
            (self.features,),
            inputs.dtype,
        )
        
        kernel = nn_partitioning.with_sharding_constraint(
            kernel, P('tensor', None)  # Simplified kernel sharding
        )
        bias = nn_partitioning.with_sharding_constraint(
            bias, P(None)  # No sharding on bias
        )

        y = jnp.dot(inputs, kernel)
        y = y + bias
        
        return nn_partitioning.with_sharding_constraint(
            y, P(None, 'sequence', 'tensor')  # Keep consistent with input
        )

class SimplifiedMaxTextLayer(nn.Module):
    """Simplified layer using MaxText-style partitioning."""
    hidden_size: int
    
    @nn.compact
    def __call__(self, x):
        # Input sharding constraint
        x = nn_partitioning.with_sharding_constraint(
            x, P('activation_batch', 'activation_length', 'activation_embed')
        )
        
        # Partitioned dense layer
        dense = LogicallyPartitionedDense(self.hidden_size)
        y = dense(x)
        
        # Output sharding constraint
        return nn_partitioning.with_sharding_constraint(
            y, P('activation_batch', 'activation_length', 'activation_mlp')
        )

def run_sharding_test():
    print("\nStarting MaxText sharding test...")
    
    # Create mesh
    devices = jax.devices()
    mesh = create_maxtext_mesh(devices)
    print(f"\nMesh shape: {mesh.shape}")
    print(f"Mesh axes: {mesh.axis_names}")
    
    # Get logical rules
    logical_rules = get_maxtext_logical_rules()
    print("\nLogical axis rules:")
    for rule in logical_rules:
        print(f"  {rule}")
    
    # Create model
    model = SimplifiedMaxTextLayer(hidden_size=1024)
    
    # Create sample input
    batch_size = 8
    seq_len = 128
    hidden_size = 1024
    
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, hidden_size), dtype=jnp.bfloat16)
    
    with mesh:
        # Apply logical partitioning rules
        with nn_partitioning.axis_rules(logical_rules):
            print("\nInitializing parameters...")
            variables = model.init(key, x)
            
            print("\nParameter shapes:")
            jax.tree_util.tree_map(lambda x: print(f"  {x.shape}"), variables)
            
            # Define sharded compute
            @jax.jit
            def sharded_forward(params, inputs):
                return model.apply(params, inputs)
            
            print("\nRunning sharded computation...")
            output = sharded_forward(variables, x)
            jax.block_until_ready(output)
            
            print("\nOutput shape:", output.shape)
            print("\nOutput sharding spec:", output.sharding.spec)
            
            # Print sharding visualization
            # print("\nSharding visualization:")
            # print(jax.debug.visualize_array_sharding(output))
            
            return output

def visualize_2d(x):
    """Visualize array sharding by reshaping to 2D if needed"""
    if x.ndim > 2:
        x_2d = x.reshape(-1, x.shape[-1])
        return jax.debug.visualize_array_sharding(x_2d)
    return jax.debug.visualize_array_sharding(x)

if __name__ == "__main__":
    print("JAX Configuration:")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")
    
    run_sharding_test()