import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.experimental.shard_map import shard_map

MESH_DATA_AXIS = "dp"
MESH_FSDP_AXIS = "fsdp"
MESH_TENSOR_AXIS = "tp"

devices = mesh_utils.create_device_mesh((1, 1, 4))
global_mesh = Mesh(devices, (MESH_DATA_AXIS, MESH_FSDP_AXIS, MESH_TENSOR_AXIS))
print(global_mesh.shape)

batch_size = 2
seq_len = 8192
n_heads = 32
head_dim = 128
emb_dim = 4096

# gen data
def gen_data_fn():
    key = jax.random.PRNGKey(np.random.randint(0, 256))
    activations = jax.random.normal(key, shape=(batch_size, seq_len, emb_dim), dtype=jnp.bfloat16)
    weights = jax.random.normal(key, shape=(emb_dim, n_heads, head_dim), dtype=jnp.bfloat16)
    return activations, weights

data_fn = pjit(
    gen_data_fn,
    out_shardings=(P(MESH_FSDP_AXIS, None, MESH_TENSOR_AXIS), P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None)),
)

def matmul(activations, weights):
    return jnp.einsum("bsE,Ehd->bshd", activations, weights)

jit_matmul = pjit(matmul, out_shardings=P(MESH_FSDP_AXIS, None, MESH_TENSOR_AXIS, None))

@partial(
    shard_map,
    mesh=global_mesh,
    in_specs=(
        P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None),
        P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None),
    ),
    out_specs=P(MESH_FSDP_AXIS, None, MESH_TENSOR_AXIS, None),
    check_rep=False,
)
def collective_matmul(activations, weights):
    weights = jax.lax.all_gather(weights, MESH_FSDP_AXIS, axis=0, tiled=True)
    axis_size = jax.lax.psum(1, axis_name=MESH_TENSOR_AXIS)
    axis_index = jax.lax.axis_index(axis_name=MESH_TENSOR_AXIS)
    # The current sequence chunk
    chunk_size = activations.shape[1]
    mid_chunk = chunk_size // 2
    # create accum buffer
    accum = jnp.zeros(
        (
            activations.shape[0],
            activations.shape[1] * axis_size,
            weights.shape[-2],
            weights.shape[-1],
        ),
        dtype=activations.dtype,
    )

    # compute first chunk
    update = jnp.einsum("bsE,Ehd->bshd", activations, weights)
    update_index = (0, axis_index * chunk_size, 0, 0)
    accum = jax.lax.dynamic_update_slice(accum, update, update_index)
    activation_forward, activation_backward = jnp.split(activations, 2, axis=1)
    activation_forward = jax.lax.ppermute(
        activation_forward,
        axis_name=MESH_TENSOR_AXIS,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)],
    )
    activation_backward = jax.lax.ppermute(
        activation_backward,
        axis_name=MESH_TENSOR_AXIS,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
    )

    # split activations into chunks and send
    def scanned_call(i, carrys):
        accum, activation_forward, activation_backward = carrys
        update_forward = jnp.einsum("bsE,Ehd->bshd", activation_forward, weights)
        update_backward = jnp.einsum("bsE,Ehd->bshd", activation_backward, weights)

        activation_forward = jax.lax.ppermute(
            activation_forward,
            axis_name=MESH_TENSOR_AXIS,
            perm=[(j, (j + 1) % axis_size) for j in range(axis_size)],
        )
        activation_backward = jax.lax.ppermute(
            activation_backward,
            axis_name=MESH_TENSOR_AXIS,
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        forward_update_index = ((axis_index - i - 1) % axis_size) * chunk_size
        backward_update_index = ((axis_index + i + 1) % axis_size) * chunk_size + mid_chunk

        accum = jax.lax.dynamic_update_slice(accum, update_forward, (0, forward_update_index, 0, 0))
        accum = jax.lax.dynamic_update_slice(accum, update_backward, (0, backward_update_index, 0, 0))
        return (accum, activation_forward, activation_backward)

    accum, _, _ = jax.lax.fori_loop(
        0, (axis_size - 1), scanned_call, (accum, activation_forward, activation_backward)
    )
    return accum

with global_mesh:
    activations, weights = data_fn()

    activations.addressable_data(0).block_until_ready()
    weights.addressable_data(0).block_until_ready()

    def run_naive():
        with jax.named_scope("naive_matmul"):
            outputs = jit_matmul(activations, weights)
        return outputs

    def run_collective():
        with jax.named_scope("collective_matmul"):
            manual_outputs = jax.jit(collective_matmul)(activations, weights)
        return manual_outputs
    
    naive_outputs = run_naive()
    collective_outputs = run_collective()
    print(jax.jit(jnp.allclose)(naive_outputs, collective_outputs))
