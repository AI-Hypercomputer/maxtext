from flax import nnx
import jax
import jax.numpy as jnp

def check_conv_shape():
    rngs = nnx.Rngs(0)
    conv = nnx.Conv(in_features=4, out_features=8, kernel_size=(3, 3), rngs=rngs)
    print("Conv kernel shape:", conv.kernel.shape) # Wait, is it .kernel or .weight?
    # Let's print all attributes that are variables
    for name, val in nnx.iter_variables(conv):
        print(f"Variable {name} shape: {val.value.shape}")

if __name__ == "__main__":
    check_conv_shape()
