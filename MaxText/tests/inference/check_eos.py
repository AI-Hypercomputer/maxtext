import pathwaysutils
import jax
import jax.numpy as jnp
from jax import lax


# Assume these are your on-device JAX arrays
# The generated token is a JAX array, typically a single value
generated_token = jnp.array([1234]) # Example token
# The EOS IDs are also a JAX array
eos_ids = jnp.array([1, 2, 3]) # Example EOS tokens

def check_eos(generated_token, eos_ids):
  """Checks if the generated token is one of the EOS tokens, on device."""
  # The jnp.in1d function is a good way to check for membership
  # It returns a boolean array
  is_eos = jnp.in1d(generated_token, eos_ids)

  # Check if any of the booleans are True
  return jnp.any(is_eos)

# You would use this check inside your generation loop
# For example, in a while_loop or a for loop
# that you want to exit early
def generation_step(carry, x):
    # 'x' could be a placeholder, as the loop is based on a condition
    # 'carry' would hold your state, like generated_tokens, etc.
    generated_tokens, eos_ids = carry
    
    # ... your generation logic here, which produces a new_token ...
    new_token = jnp.array([5]) # This is where your model generates a token
    
    # Update the generated tokens
    updated_tokens = jnp.concatenate([generated_tokens, new_token])
    
    # Check for EOS on the device
    should_continue = lax.cond(
        check_eos(new_token, eos_ids),
        lambda: False, # if true (EOS found), set continue to False
        lambda: True   # if false (EOS not found), set continue to True
    )

    return (updated_tokens, eos_ids), should_continue

@jax.jit
def generate_until_eos(initial_tokens, eos_ids):
    # This is a simplified while_loop for demonstration
    # In a real engine, the loop would be more complex
    # and would likely be a lax.while_loop
    
    # You need to define a while loop
    def continue_while(state):
      i, tokens, eos_ids_array = state
      # Condition to continue the loop
      # You can add a max_length condition too
      is_eos_found = jnp.any(jnp.in1d(tokens[-1:], eos_ids_array))
      return jnp.logical_not(is_eos_found)
      
    def body_while(state):
      i, tokens, eos_ids_array = state
      # ... Your model inference logic ...
      new_token = jnp.array([1]) # Example output token
      
      # Append the new token
      updated_tokens = jnp.concatenate([tokens, new_token])
      
      return (i+1, updated_tokens, eos_ids_array)
      
    # Initial state for the while loop
    initial_state = (0, initial_tokens, eos_ids)
    
    # Run the while loop on the device
    _, final_tokens, _ = lax.while_loop(continue_while, body_while, initial_state)
    
    return final_tokens

pathwaysutils.initialize()

# Example usage
initial_prompt_tokens = jnp.array([10, 20, 30])
my_eos_ids = jnp.array([1, 2, 3])

# Call the jitted function
final_output = generate_until_eos(initial_prompt_tokens, my_eos_ids)
# Note: To print the final output you would need to transfer it to host,
# but the check for EOS itself happens on device.
print("Final tokens after generation:", final_output)