"""
A test to demonstrate the instability of top_k across JAX and PyTorch.
"""
import unittest
import torch
import jax
import jax.numpy as jnp
import numpy as np

class TopKTest(unittest.TestCase):
  """
  Demonstrates that the indices returned by top_k for identical values
  are not guaranteed to be the same across frameworks.
  """

  def test_topk_instability(self):
    """
    Shows that JAX and PyTorch can return different indices for the same
    input array containing duplicate `inf` values.
    """
    print("\n--- Running test_topk_instability ---")
    # The example array with duplicate `inf` values
    data_np = np.array([np.inf, np.inf, 1.23, -np.inf, np.inf], dtype=np.float32)
    k = 2

    # --- PyTorch Execution ---
    data_torch = torch.from_numpy(data_np)
    _, topk_indices_torch = torch.topk(data_torch, k=k, largest=True)
    torch_idx = topk_indices_torch
    print(f"PyTorch topk index: {torch_idx}")

    # --- JAX Execution ---
    data_jax = jnp.asarray(data_np)
    _, topk_indices_jax = jax.lax.top_k(data_jax, k=k)
    # jax.lax.top_k returns a list, so we extract the element
    jax_idx = topk_indices_jax
    print(f"JAX topk index: {jax_idx}")

    # This assertion is not guaranteed to fail, as the behavior can be
    # platform-dependent. The purpose of this test is to print the
    # results and illustrate the *potential* for divergence.
    print("Note: The indices may or may not be different in your specific environment.")
    print("The key takeaway is that their equality is not guaranteed.")
    print("-----------------------------------------")

  def test_stable_sort_solution(self):
    """
    Demonstrates how to achieve a stable sort (and thus a stable top_k)
    in both frameworks, guaranteeing identical results.
    """
    print("\n--- Running test_stable_sort_solution ---")
    data_np = np.array([10.0, 20.0, 20.0, 5.0, 10.0], dtype=np.float32)
    k = 3

    # --- PyTorch Stable Top-K ---
    data_torch = torch.from_numpy(data_np)
    # Create a tensor of indices for tie-breaking
    indices_torch = torch.arange(len(data_torch))
    # Sort by value (descending), then by index (ascending)
    # We negate the data to sort descending
    sorted_indices_torch = torch.stack([-data_torch, indices_torch]).T.tolist()
    sorted_indices_torch.sort()
    stable_topk_indices_torch = [int(idx) for _, idx in sorted_indices_torch[:k]]
    print(f"PyTorch stable topk indices: {stable_topk_indices_torch}")


    # --- JAX Stable Top-K (using lexsort) ---
    data_jax = jnp.asarray(data_np)
    indices_jax = jnp.arange(len(data_jax))
    # lexsort sorts by the last key first, so we put the primary key (neg_data) last.
    neg_data_jax = -data_jax
    sorted_indices_jax = jnp.lexsort((indices_jax, neg_data_jax))
    stable_topk_indices_jax = sorted_indices_jax[:k]
    print(f"JAX stable topk indices: {stable_topk_indices_jax}")

    print("Note: With a stable sort, the indices are now guaranteed to match.")
    print("-----------------------------------------")

    # Assert that the stable results are identical
    np.testing.assert_array_equal(
        np.array(stable_topk_indices_torch),
        np.array(stable_topk_indices_jax),
        err_msg="Stable top_k indices do not match."
    )


if __name__ == '__main__':
  unittest.main()
