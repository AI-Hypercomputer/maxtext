import jax.numpy as jnp

class TransposeAttentionOut:
    def __call__(self, tensors):
        out = tensors[0] if isinstance(tensors, list) else tensors
        if len(out.shape) == 3:
            return jnp.transpose(out.reshape(-1, out.shape[2]), (1, 0))
        return jnp.transpose(out)

op = TransposeAttentionOut()
out = jnp.ones((2048, 151936))
res = op([out])
print("res shape:", res.shape)
