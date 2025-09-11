
import jax
import jax.numpy as jnp
from jax.lax import scan
import flax.linen as nn
from flax.struct import dataclass
from typing import Tuple, Any
import numpy as np
import time
from collections import deque

# --- Configuration ---
BATCH_SIZE = 8  # Max concurrent sequences
MAX_TARGET_LENGTH = 256
MAX_PREFILL_LENGTH = 128
KV_CACHE_SIZE = MAX_TARGET_LENGTH
VOCAB_SIZE = 1000
EOS_ID = 0
CHUNK_DECODE_SIZE = 16 # Number of tokens to generate in one JIT'd chunk

# --- 1. State Management: A Pytree for Decoding State ---

@dataclass
class DecodingState:
  """Holds the state for a batch of sequences being decoded."""
  # Shape: [batch, heads, seq_len, head_dim]
  kv_cache: jax.Array
  # Shape: [batch, seq_len]
  generated_tokens: jax.Array
  # Shape: [batch]
  current_positions: jax.Array
  # Shape: [batch]
  is_active: jax.Array # Boolean mask for active sequences

# --- 2. A Dummy Model for Demonstration ---

class SimpleModel(nn.Module):
  """A simple dummy model that returns logits."""
  @nn.compact
  def __call__(self, x, kv_cache, positions):
    # In a real model, you'd use the inputs to perform attention
    # and update the kv_cache. Here we just return dummy logits.
    # The shape is what matters.
    logits = jax.random.uniform(self.make_rng('random'), (x.shape[0], VOCAB_SIZE))
    # Dummy updated cache
    new_kv_entry = jnp.ones_like(kv_cache[:, :, :1, :])
    kv_cache = jax.lax.dynamic_update_slice_in_dim(kv_cache, new_kv_entry, positions[0], axis=2)
    return logits, kv_cache

# --- 3. JIT-Compiled Core Functions ---

def create_batching_functions(model: SimpleModel, params: Any):
  """Creates the JIT-compiled prefill and generation functions."""

  @jax.jit
  def prefill(state: DecodingState, prompts: jax.Array, slots: jax.Array) -> Tuple[DecodingState, jax.Array]:
    """Processes new prompts and slots them into the decoding state."""
    # 1. Run model once for the full prompt to get initial KV cache
    prompt_positions = jnp.arange(prompts.shape[1])
    dummy_cache = jnp.zeros((prompts.shape[0], 1, KV_CACHE_SIZE, 1), dtype=jnp.float32)
    logits, initial_kv = model.apply({'params': params}, prompts, dummy_cache, prompt_positions)

    # 2. Sample the first token
    next_tokens = jnp.argmax(logits, axis=-1)

    # 3. Create a new state for these specific prompts
    new_tokens = jnp.concatenate([prompts, next_tokens[:, None]], axis=1)
    new_positions = jnp.full((prompts.shape[0],), prompts.shape[1])
    new_is_active = jnp.ones((prompts.shape[0],), dtype=jnp.bool_)

    # 4. Slot the new state into the global batch state at the specified slots
    # We use a dummy loop with `vmap` for cleaner code, which JAX will optimize.
    def slot_in(current_state, update_vals):
      slot_idx, token_vals, pos_val, active_val, kv_val = update_vals
      len_t = token_vals.shape[0]
      
      # Update tokens, positions, and active status
      new_tokens = jax.lax.dynamic_update_slice(current_state.generated_tokens, token_vals[None, :], (slot_idx, 0))
      new_pos = jax.lax.dynamic_update_slice(current_state.current_positions, pos_val[None], (slot_idx,))
      new_active = jax.lax.dynamic_update_slice(current_state.is_active, active_val[None], (slot_idx,))
      
      # Update KV cache
      # Note: In a real scenario, KV cache shape would be [batch, heads, len, dim]
      # Here we use a simplified shape for clarity.
      new_kv = jax.lax.dynamic_update_slice(current_state.kv_cache, kv_val[None, ...], (slot_idx, 0, 0, 0))

      return DecodingState(
          kv_cache=new_kv,
          generated_tokens=new_tokens,
          current_positions=new_pos,
          is_active=new_active
      ), None

    final_state, _ = jax.vmap(slot_in, in_axes=(None, 0))(
        state, (slots, new_tokens, new_positions, new_is_active, initial_kv)
    )
    # Since vmap creates a new batch dimension, we take the first element.
    final_state = jax.tree_util.tree_map(lambda x: x[0], final_state)

    return final_state

  @jax.jit
  def generate_chunk(state: DecodingState) -> DecodingState:
    """Generates a chunk of tokens using jax.lax.scan."""

    def single_decode_step(carry_state: DecodingState, _):
      """Body of the scan loop for a single autoregressive step."""
      # 1. Get the last token for all sequences in the batch
      last_tokens = carry_state.generated_tokens[:, carry_state.current_positions[0]]

      # 2. Run the model forward pass
      logits, new_kv_cache = model.apply(
          {'params': params},
          last_tokens[:, None], # Model expects a sequence dim
          carry_state.kv_cache,
          carry_state.current_positions
      )

      # 3. Sample the next token
      next_tokens = jnp.argmax(logits, axis=-1)

      # 4. Update the generated sequences, but only for active sequences
      # This `where` is crucial for continuous batching.
      updated_tokens_slice = jnp.where(
          carry_state.is_active,
          next_tokens,
          carry_state.generated_tokens[:, carry_state.current_positions[0] + 1]
      )
      updated_tokens = jax.lax.dynamic_update_slice_in_dim(
          carry_state.generated_tokens,
          updated_tokens_slice,
          carry_state.current_positions[0] + 1,
          axis=1
      )

      # 5. Update positions and active status
      new_positions = carry_state.current_positions + 1
      new_is_active = carry_state.is_active & (next_tokens != EOS_ID)

      # 6. Create the state for the next iteration
      next_state = DecodingState(
          kv_cache=new_kv_cache,
          generated_tokens=updated_tokens,
          current_positions=new_positions,
          is_active=new_is_active
      )
      return next_state, None

    # Use scan to run the decode step for a fixed number of iterations
    final_state, _ = scan(single_decode_step, state, None, length=CHUNK_DECODE_SIZE)
    return final_state

  return prefill, generate_chunk


# --- 4. Host-Side Orchestrator ---

class ContinuousBatcher:
  """Manages the continuous batching loop."""

  def __init__(self):
    print("Initializing Continuous Batcher...")
    self.model = SimpleModel()
    self.key = jax.random.PRNGKey(0)
    self.params = self.model.init(self.key, jnp.ones((1,1), dtype=jnp.int32), jnp.zeros((1, 1, KV_CACHE_SIZE, 1)), jnp.ones(1, dtype=jnp.int32))['params']

    # Create the core JIT'd functions
    self.prefill, self.generate_chunk = create_batching_functions(self.model, self.params)

    # Initialize empty state on host
    self.host_state = DecodingState(
        kv_cache=np.zeros((BATCH_SIZE, 1, KV_CACHE_SIZE, 1), dtype=np.float32),
        generated_tokens=np.zeros((BATCH_SIZE, MAX_TARGET_LENGTH), dtype=np.int32),
        current_positions=np.zeros((BATCH_SIZE,), dtype=np.int32),
        is_active=np.zeros((BATCH_SIZE,), dtype=np.bool_)
    )

    self.requests = deque()
    self.results = {}
    self.next_request_id = 0
    print("Initialization complete.")

  def add_request(self, prompt: list[int]):
    """Add a new request to the queue."""
    if len(prompt) > MAX_PREFILL_LENGTH:
        raise ValueError(f"Prompt too long! Max length is {MAX_PREFILL_LENGTH}")
    self.requests.append({'id': self.next_request_id, 'prompt': prompt})
    self.results[self.next_request_id] = None
    print(f"Request {self.next_request_id} added to queue.")
    self.next_request_id += 1

  def _find_free_slots(self) -> np.ndarray:
    """Find indices of inactive slots in the batch."""
    return np.where(self.host_state.is_active == False)[0]

  def _prefill_new_requests(self):
    """Fill any free slots with requests from the queue."""
    free_slots = self._find_free_slots()
    num_to_prefill = min(len(free_slots), len(self.requests))

    if num_to_prefill == 0:
      return

    prompts_to_process = []
    slots_to_fill = free_slots[:num_to_prefill]

    for i in range(num_to_prefill):
      request = self.requests.popleft()
      prompt = request['prompt']
      
      # Pad prompt to a uniform length for batching
      padded_prompt = np.pad(prompt, (0, MAX_PREFILL_LENGTH - len(prompt)), 'constant')
      prompts_to_process.append(padded_prompt)
      
      # Mark slot as active on host immediately
      self.host_state.is_active[slots_to_fill[i]] = True
      print(f"Prefilling request {request['id']} in slot {slots_to_fill[i]}")


    # Run the JIT'd prefill function
    device_state = jax.device_put(self.host_state)
    prompts_arr = jnp.array(prompts_to_process, dtype=jnp.int32)
    slots_arr = jnp.array(slots_to_fill, dtype=jnp.int32)
    
    updated_device_state = self.prefill(device_state, prompts_arr, slots_arr)
    self.host_state = jax.device_get(updated_device_state)

  def _check_for_completion(self):
    """Check for finished sequences and store results."""
    newly_finished_mask = (self.host_state.is_active == False) & (self.host_state.current_positions > 0)
    finished_slots = np.where(newly_finished_mask)[0]

    for slot in finished_slots:
      pos = self.host_state.current_positions[slot]
      result_tokens = self.host_state.generated_tokens[slot, :pos]
      
      # Find the request ID for this slot (this part is a bit tricky)
      # In a real server, you'd maintain a slot -> request_id mapping
      # For this example, we'll just print it.
      print(f"Slot {slot} finished. Result: {result_tokens}")
      
      # Reset the slot on the host
      self.host_state.current_positions[slot] = 0
      # The `is_active` is already False from the device computation

  def step(self):
    """Perform one iteration of the batching loop."""
    print("\n--- Batcher Step ---")
    # 1. Fill empty slots with new requests
    self._prefill_new_requests()

    # 2. If there are any active sequences, generate a chunk of tokens
    if np.any(self.host_state.is_active):
      print(f"Generating a chunk of {CHUNK_DECODE_SIZE} tokens for active slots...")
      device_state = jax.device_put(self.host_state)
      updated_device_state = self.generate_chunk(device_state)
      self.host_state = jax.device_get(updated_device_state)
      print("Chunk generation complete.")
    else:
      print("No active sequences. Skipping generation.")

    # 3. Check for and handle any sequences that just finished
    self._check_for_completion()

  def run(self):
    """Run the batcher until all requests are processed."""
    while len(self.requests) > 0 or np.any(self.host_state.is_active):
      self.step()
      time.sleep(0.5) # Simulate time between steps
    print("\nAll requests processed. Batcher finished.")


if __name__ == "__main__":
  batcher = ContinuousBatcher()

  # Add some initial requests
  batcher.add_request([1, 2, 3, 4])
  batcher.add_request([5, 6, 7])
  batcher.add_request([8, 9, 10, 11, 12])

  # Start the main loop
  batcher.run()

  # We can add more requests while it's running (in a real threaded server)
  # For this example, we'll just show the initial batch is cleared.
  # Example of adding another request mid-flight:
  # (This would require running batcher.run() in a separate thread)
  # time.sleep(2)
  # batcher.add_request([100, 101])
