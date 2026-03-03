"""Utility for comparing logits between Linen and NNX decoder implementations."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.layers.decoders import Decoder
from maxtext.layers.nnx_decoders import NNXDecoder


def test_linen_vs_nnx_logits(config, mesh):
  # 1. Setup Dummy Inputs
  batch_size, seq_len = 2, 128
  rng = jax.random.PRNGKey(0)

  decoder_input_tokens = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)
  decoder_positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
  decoder_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  # 2. Instantiate Both Models
  # Note: Using your wrapper for Linen or native Linen classes
  linen_model = Decoder(config=config, mesh=mesh, model_mode=MODEL_MODE_PREFILL)

  nnx_rngs = nnx.Rngs(0)
  nnx_model = NNXDecoder(config=config, mesh=mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx_rngs)

  # Dummy shared embedding (assuming you have a simple embedding table for tests)
  dummy_embedding = nn.Embed(num_embeddings=config.vocab_size, features=config.emb_dim)
  dummy_embedding.init(rng, decoder_input_tokens)

  # 3. Initialize Linen Model and Extract Weights
  init_rng, _ = jax.random.split(rng)
  linen_variables = linen_model.init(
      init_rng,
      shared_embedding=dummy_embedding,
      decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=True,
      model_mode=MODEL_MODE_PREFILL,
  )
  linen_params = linen_variables["params"]

  # 4. Inject Linen Weights into NNX
  # (Assuming your NNX model's state tree perfectly matches the Linen param tree)
  # You may need to write a small helper to reshape/map keys if the PyTree structure differs slightly.
  nnx.update(nnx_model, linen_params)

  # 5. Run Forward Passes
  # Linen forward pass
  logits_linen, hidden_linen, _ = linen_model.apply(
      linen_variables,
      shared_embedding=dummy_embedding,  # or a bound dummy_embedding object
      decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=True,
      model_mode=MODEL_MODE_PREFILL,
  )

  # NNX forward pass
  logits_nnx, hidden_nnx, _ = nnx_model(
      shared_embedding=dummy_embedding,  # Pass an NNX equivalent if required
      decoder_input_tokens=decoder_input_tokens,
      decoder_positions=decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=True,
      model_mode=MODEL_MODE_PREFILL,
  )

  # 6. The Verdict: Compare Outputs
  print("=== Verification Results ===")

  # Check max difference in raw hidden states (pre-logits)
  hidden_max_diff = jnp.max(jnp.abs(hidden_linen - hidden_nnx))
  print(f"Hidden State Max Diff: {hidden_max_diff}")

  # Check max difference in logits
  logit_max_diff = jnp.max(jnp.abs(logits_linen - logits_nnx))
  print(f"Logit Max Diff:        {logit_max_diff}")

  # Check Tolerance (Adjust atol based on dtype. 1e-5 for fp32, 1e-2 for bf16)
  tolerance_met = jnp.allclose(logits_linen, logits_nnx, atol=1e-4, rtol=1e-4)
  print(f"Passes allclose (1e-4): {tolerance_met}")

  # Check Token Predictions (Argmax)
  tokens_linen = jnp.argmax(logits_linen, axis=-1)
  tokens_nnx = jnp.argmax(logits_nnx, axis=-1)
  tokens_match = jnp.array_equal(tokens_linen, tokens_nnx)
  print(f"Argmax Tokens Match:   {tokens_match}")

  assert tolerance_met and tokens_match, "Migrations do not match!"
