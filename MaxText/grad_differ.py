import jax
from jax import numpy as jnp


def pytree_ravel(pytree):
    ravelled_tree = jax.tree.map(jnp.ravel, pytree)
    ravelled_leaves, _ = jax.tree_util.tree_flatten(ravelled_tree)
    return jnp.concatenate(ravelled_leaves)

def diff_grads(grad1, grad2):
    grad1_flat = pytree_ravel(grad1)
    grad2_flat = pytree_ravel(grad2)
    diff = grad1_flat - grad2_flat
    diff_norm = jnp.linalg.norm(diff)
    print(f"grad1 norm: {jnp.linalg.norm(grad1_flat)}")
    print(f"grad2 norm: {jnp.linalg.norm(grad2_flat)}")
    print(f"diff norm: {diff_norm}")
    return diff_norm

