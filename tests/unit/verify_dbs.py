import jax
import jax.numpy as jnp
from maxtext.inference import inference_utils

def verify_dbs_logic():
    # 1. Setup: 2 groups, 1 beam per group, Vocab size 10
    num_beams = 2
    num_groups = 2
    diversity_penalty = 5.0 # Penalty > logit gap
    
    # Simulating a batch of 1 user, which is 2 beams total
    logits = jnp.zeros((2, 1, 10))
    # Token 5 is best (10.0), Token 3 is runner up (8.0)
    # The gap is 2.0.
    logits = logits.at[:, :, 5].set(10.0) 
    logits = logits.at[:, :, 3].set(8.0)
    
    # Original scores are all 0
    cumulative_logprobs = jnp.zeros((2, 1))
    
    print("--- Running DBS Step ---")
    jit_sampling = jax.jit(inference_utils.sampling_dbs, static_argnums=(2,3,4))
    
    new_tokens, new_scores, parent_indices = jit_sampling(
        logits, cumulative_logprobs, num_beams, num_groups, diversity_penalty
    )
    
    print(f"Selected Tokens: {new_tokens.flatten()}")
    print(f"Parent Indices:  {parent_indices.flatten()}")
    print(f"New Scores:      {new_scores.flatten()}")
    
    # VERIFICATION LOGIC
    # Group 0 picks Token 5. 
    # Logprob of 5 is close to 0. Logprob of 3 is ~ -2.0.
    # Group 1: Token 5 score = 0.0 - 5.0 = -5.0
    #          Token 3 score = -2.0 (no penalty)
    # Result: -2.0 > -5.0, so Group 1 MUST pick Token 3.
    
    t0, t1 = new_tokens.flatten()
    if t0 == 5 and t1 == 3:
        print("\n✅ SUCCESS: Group 1 avoided Group 0's token due to diversity penalty!")
    else:
        print(f"\n❌ FAILURE: Expected tokens [5, 3], but got [{t0}, {t1}]")

if __name__ == "__main__":
    verify_dbs_logic()
