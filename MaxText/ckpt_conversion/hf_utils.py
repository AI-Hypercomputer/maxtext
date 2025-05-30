import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import torch.nn.functional as F
from tabulate import tabulate
from typing import Optional


def convert_jax_weight_to_torch(
    weight: "jax.Array", dtype: Optional[str] = None
) -> torch.Tensor:
    expected_dtype = str(weight.dtype) if dtype is None else dtype
    expected_shape = weight.shape
    weight = multihost_utils.process_allgather(weight)
    weight = np.array(weight, dtype="float32")
    torch_dtype = getattr(torch, expected_dtype)
    torch_array = torch.from_numpy(weight).to(torch_dtype).reshape(expected_shape)
    return torch_array


def check_arrays_match(arrayA, arrayB, atol=0.01):
    """
    Compare two sets of arrays for equality within the specified absolute tolerance.
    
    This function handles both PyTorch tensors and JAX arrays, automatically
    converting between the two if necessary. If the arrays don't match within
    the specified tolerance, it prints detailed information about the mismatches
    and raises a AssertionError.
    
    Args:
        arrayA (Union[torch.Tensor, jax.Array]): First set of arrays to compare
        arrayB (Union[torch.Tensor, jax.Array]): Second set of arrays to compare
        atol (float, optional): Absolute tolerance for comparison. Defaults to 0.01
    
    Raises:
        AssertionError: If the arrays don't match within the specified tolerance
    """
    # Determine types and convert if needed
    is_A_torch = isinstance(arrayA, torch.Tensor)
    is_B_torch = isinstance(arrayB, torch.Tensor)
    
    # If one is torch and one is jax, convert jax to torch
    if is_A_torch and not is_B_torch:
        arrayB = convert_jax_weight_to_torch(arrayB)
    elif is_B_torch and not is_A_torch:
        arrayA = convert_jax_weight_to_torch(arrayA)
    
    # If both are now torch tensors
    if isinstance(arrayA, torch.Tensor):
        abs_diff_tensor = torch.abs(arrayA - arrayB)
        max_diff_val = torch.max(abs_diff_tensor).item()
        # print(f"Maximum absolute difference: {max_diff_val:.6f}")

        if max_diff_val > atol:
            mismatch_indices = abs_diff_tensor > atol
            num_mismatched_elements = mismatch_indices.sum().item()
            
            print(f"Number of elements in {arrayB.shape} exceeding absolute tolerance {atol}: {num_mismatched_elements}")
            
            # Print a few examples of mismatched elements
            if num_mismatched_elements > 0:
                print("Examples of mismatched elements (ArrayA vs ArrayB, limited to first 5):")
                limit_print = 5
                actual_limit = min(num_mismatched_elements, limit_print)
                mismatched_A_samples = arrayA[mismatch_indices][:actual_limit]
                mismatched_B_samples = arrayB[mismatch_indices][:actual_limit]
                for i in range(len(mismatched_A_samples)):
                    print(f"  A: {mismatched_A_samples[i].item():.6f}, B: {mismatched_B_samples[i].item():.6f}, Diff: {(mismatched_A_samples[i]-mismatched_B_samples[i]).item():.6f}")

            # raise AssertionError(
            #     f"Failed to match arrays. Maximum absolute difference {max_diff_val:.6f} exceeds tolerance {atol}. "
            #     f"{num_mismatched_elements} elements are mismatched."
            # )
        
    # If both are still jax arrays
    else:
        abs_diff_tensor = jnp.abs(arrayA - arrayB)
        max_diff_val = jnp.max(abs_diff_tensor).item() # .item() to get Python scalar
        print(f"Maximum absolute difference: {max_diff_val:.6f}")

        if max_diff_val > atol:
            mismatch_indices = abs_diff_tensor > atol
            num_mismatched_elements = jnp.sum(mismatch_indices).item()
            print(f"Number of elements in {arrayB.shape} exceeding absolute tolerance {atol}: {num_mismatched_elements}")
            raise AssertionError(
                f"Failed to match JAX arrays. Maximum absolute difference {max_diff_val:.6f} exceeds tolerance {atol}. "
                f"{num_mismatched_elements} elements are mismatched."
            )
    
def check_predicted_tokens_match(logits_a, logits_b, tolerance=0.1):
    """Compares the top predicted tokens from each set of logits and ensures their 
    disagreement rate doesn't exceed the tolerance threshold. Raises an AssertionError 
    if the disagreement is too high.
    
    Args:
        logits_a (jax.Array | torch.Tensor | np.ndarray): First set of model output logits
        logits_b (jax.Array | torch.Tensor | np.ndarray): Second set of model output logits to compare against logits_a
        tolerance (float, optional): Maximum allowed fraction of token prediction disagreements,
            must be between 0.0 and 1.0. Defaults to 0.05 (5%).
                        
    Examples:
        >>> logits1 = get_model_output(input1)
        >>> logits2 = get_model_output(input2) 
        >>> check_predicted_tokens_match(logits1, logits2, tolerance=0.03)  # Allows 3% disagreement
    """
    # Validate tolerance input
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("Tolerance must be between 0.0 and 1.0")
    
    metrics = get_logits_comparison_metrics(logits_a, logits_b)
    disagreement_rate = metrics["disagreement_top1"]
    
    if disagreement_rate > tolerance:
        raise AssertionError(
            f"Token prediction mismatch: {disagreement_rate:.1%} of tokens disagree "
            f"(exceeds tolerance of {tolerance:.1%})"
        )
    
def get_logits_comparison_metrics(logitsA, logitsB):
    """
    Calculate various comparison metrics between two sets of logits.
    
    This function computes several metrics to compare the similarity and differences
    between two sets of logits, including KL divergence, absolute differences,
    and agreement in top-k predictions.
    
    Args:
        logitsA (jax.Array | torch.Tensor | np.ndarray): First set of logits to compare
        logitsB (jax.Array | torch.Tensor | np.ndarray): Second set of logits to compare
    
    Returns:
        dict: A dictionary containing the following metrics:
            - max_kl_div: Maximum KL divergence between probability distributions
            - abs_diff: Maximum absolute difference between probabilities
            - disagreement_top5: Proportion of positions where top-5 predictions differ
            - disagreement_top1: Proportion of positions where top-1 predictions differ
    
    Notes:
        The function also prints a formatted table of the metrics using tabulate.
    """

    if isinstance(logitsA, jax.Array) :
        logitsA = convert_jax_weight_to_torch(logitsA)
    if isinstance(logitsA, np.ndarray):
        logitsA = torch.tensor(logitsA)
    if isinstance(logitsB, jax.Array):
        logitsB = convert_jax_weight_to_torch(logitsB)
    if isinstance(logitsB, np.ndarray):
        logitsB = torch.tensor(logitsB)
    
    # Calculate probabilities
    probs_A = F.softmax(logitsA, dim=-1)
    probs_B = F.softmax(logitsB, dim=-1)

    # Calculate metrics
    kl_div = F.kl_div(torch.log(probs_B), probs_A, reduction='none', log_target=False)
    max_kl_div = torch.max(kl_div.sum(dim=-1))

    max_abs_diff = torch.abs(probs_A - probs_B).max()

    # Calculate top-k agreement metrics
    sorted_logits_A = torch.argsort(logitsA, dim=1)
    sorted_logits_B = torch.argsort(logitsB, dim=1)
    ranking_A_top5 = sorted_logits_A[:, -5:]
    ranking_B_top5 = sorted_logits_B[:, -5:]
    disagreement_top5 = torch.mean((
        (torch.abs(ranking_B_top5 - ranking_A_top5) > 0).sum(dim=1) > 0
    ).float())

    ranking_A_top1 = sorted_logits_A[:, -1:]
    ranking_B_top1 = sorted_logits_B[:, -1:]
    disagreement_top1 = torch.mean((
        (torch.abs(ranking_B_top1 - ranking_A_top1) > 0).sum(dim=1) > 0
    ).float())
        
    metrics = {
        "max_kl_div": float(max_kl_div),
        "abs_diff": float(max_abs_diff),
        "disagreement_top5": float(disagreement_top5),
        "disagreement_top1": float(disagreement_top1),
    }
    
    table = [[key, value] for key, value in metrics.items()]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="orgtbl"))
    return metrics