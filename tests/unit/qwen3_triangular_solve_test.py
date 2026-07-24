import unittest
import numpy as np
import jax
import jax.numpy as jnp
from maxtext.models.qwen3 import invert_unit_lower_triangular_log_depth


class TestQwen3TriangularSovle(unittest.TestCase):

  def test_invert_unit_lower_triangular_log_depth(self):
    """Test for loss at chunk_size 256."""
    jax.config.update("jax_enable_x64", True)  # Use float64 for precise testing
    chunk_size = 256

    # Generate a random matrix and make it strictly lower triangular
    key = jax.random.PRNGKey(chunk_size)
    S_random = jax.random.normal(key, (chunk_size, chunk_size), dtype=jnp.float64) / chunk_size
    S = jnp.tril(S_random, k=-1)

    # The matrix to invert is (I + S)
    identity = jnp.eye(chunk_size, dtype=jnp.float64)
    matrix_to_invert = identity + S

    # Using our custom function
    A = invert_unit_lower_triangular_log_depth(S)

    # The product A @ (I + S) should be exactly the identity matrix
    # Wait, due to numerical precision, we should check for max error (loss)
    reconstructed_identity = A @ matrix_to_invert

    # Compute loss for forward pass
    loss = jnp.max(jnp.abs(reconstructed_identity - identity))

    # We expect the loss to be very small, around numerical precision
    self.assertLess(loss, 1e-10, f"Failed for chunk_size {chunk_size} with loss {loss}")

    # Verify backward pass accuracy using jax.test_util.check_grads
    # This uses finite differences to check the correctness of the custom VJP
    try:
      from jax.test_util import check_grads

      # We check the gradients for the function.
      # `check_grads` will assert if finite difference gradients
      # don't match the custom VJP gradients.
      check_grads(invert_unit_lower_triangular_log_depth, (S,), order=1, modes=["rev"])
    except ImportError:
      # Fallback if check_grads is not available in the used jax version
      raise


if __name__ == "__main__":
  unittest.main()
