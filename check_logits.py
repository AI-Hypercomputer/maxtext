import jsonlines  # pylint: disable=import-outside-toplevel
import jax
import jax.numpy as jnp
from MaxText import max_logging

clip_logits_epsilon = None


input_golden_data_path = "/home/shuningjin/deepseek3-671b/hf-671b-bf16.jsonl"
with jsonlines.open(input_golden_data_path, "r") as f:
  golden_data1 = list(f)
a = golden_data1[0]["logits"]
golden_logits_slice = jnp.array(a, dtype=jnp.float32)

input_golden_data_path = "/home/shuningjin/deepseek3-671b/hf-2025-10-31-16-41-10.jsonl"
with jsonlines.open(input_golden_data_path, "r") as f:
  golden_data2 = list(f)
b = golden_data2[0]["logits"]
train_logits_slice = jnp.array(b, dtype=jnp.float32)


# Calculate absolute and relative differences for detailed reporting
abs_diff = jnp.abs(train_logits_slice - golden_logits_slice)

# To avoid division by zero, add a small epsilon where golden_logits_slice is zero
safe_golden_logits = jnp.where(golden_logits_slice == 0, 1e-8, golden_logits_slice)
rel_diff = abs_diff / jnp.abs(safe_golden_logits)

max_abs_diff_idx = jnp.unravel_index(jnp.argmax(abs_diff), abs_diff.shape)
max_rel_diff_idx = jnp.unravel_index(jnp.argmax(rel_diff), rel_diff.shape)

max_abs_diff_val = abs_diff[max_abs_diff_idx]
max_rel_diff_val = rel_diff[max_rel_diff_idx]
msg = (
    "\n[numerical difference]\n"
    f"Max absolute difference: {max_abs_diff_val:.6f} at index {max_abs_diff_idx}\n"
    f"  (Train: {train_logits_slice[max_abs_diff_idx]:.6f}, Golden: {golden_logits_slice[max_abs_diff_idx]:.6f})\n"
    f"Max relative difference: {max_rel_diff_val:.6f} at index {max_rel_diff_idx}\n"
    f"  (Train: {train_logits_slice[max_rel_diff_idx]:.6f}, Golden: {golden_logits_slice[max_rel_diff_idx]:.6f})"
)
max_logging.log(msg)

if clip_logits_epsilon is not None:
  model_probabilities = jnp.clip(jax.nn.softmax(train_logits_slice, axis=-1), a_min=clip_logits_epsilon)
  golden_probabilities = jnp.clip(jax.nn.softmax(golden_logits_slice, axis=-1), a_min=clip_logits_epsilon)
else:
  model_probabilities = jax.nn.softmax(train_logits_slice, axis=-1)
  golden_probabilities = jax.nn.softmax(golden_logits_slice, axis=-1)

max_logging.log("\n[probability: token 1]")
max_logging.log(f"{golden_probabilities[1]=}")
max_logging.log(f"{model_probabilities[1]=}")

kl_div = jax.numpy.sum(jax.scipy.special.kl_div(golden_probabilities, model_probabilities), axis=-1)
max_kl_div_val = jax.numpy.max(kl_div)
max_kl_div_idx = jax.numpy.argmax(kl_div)
max_logging.log(
    f"\n[KL divergence]\n"
    f"KL divergence = {kl_div}, max KL divergence = {max_kl_div_val} at index {max_kl_div_idx}, "
    #  f"the corresponding token id is {ids[0, max_kl_div_idx]}"
)
