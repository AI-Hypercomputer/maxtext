
import jax
from jax import random

init_rng = random.PRNGKey(0)

BATCH = 4
SEQ = 128
EMB = 512
FF = 2048
NUM_EXPERTS = 8
EXPERTS_PER_TOKEN = 2

CAPACITY_FACTOR = 1.5
TOKENS_PER_EXPERT = (BATCH*SEQ)/NUM_EXPERTS * CAPACITY_FACTOR * EXPERTS_PER_TOKEN


experts_matrix = jax.numpy.zeros( (EMB, NUM_EXPERTS, FF))

input_activations = jax.numpy.zeros( (BATCH, SEQ, EMB) )
expert_choices = jax.random.uniform( init_rng, (BATCH, SEQ, NUM_EXPERTS) )

expert_values, expert_indices = jax.lax.top_k(expert_choices, EXPERTS_PER_TOKEN) # this is the two experts for each (BATCH, SEQ, EXPERTS_PER_TOKEN)

#[NUMBER_EXPERTS, TOKENS_PER_EXPERT, EMB] #output into 


breakpoint()