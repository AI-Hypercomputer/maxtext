# Copyright 2026 Google LLC
"""Fast JIT generation verification for GLM-5.2."""

import sys
import maxtext  # Ensures Flax/JAX compatibility hooks are applied first
from flax import nnx
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


@nnx.jit
def forward_step(model, tokens, positions, segment_ids):
  return model(
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      enable_dropout=False,
  )


def main():
  cfg = pyconfig.initialize_pydantic(sys.argv)
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  
  if jax.process_index() == 0:
    print("=== Loading GLM-5.2 Tokenizer ===", flush=True)
  tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True)

  if jax.process_index() == 0:
    print(f"=== Restoring GLM-5.2 (744B) Model from {cfg.load_parameters_path} ===", flush=True)
  model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode="train")

  if jax.process_index() == 0:
    print("=== Model restored successfully! Running prompt generation test ===", flush=True)

  user_prompts = [
      "what is the capital of france",
      "The biggest planet in the solar system is",
  ]

  tests = []
  for p in user_prompts:
    # 1. Chat format (recommended for GLM-5.2)
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chat_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    tests.append((f"[CHAT TEMPLATE] {p}", chat_ids, 25))

    # 2. Base completion format with GLM prefix
    base_text = f"[gMASK]<sop>{p}"
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    tests.append((f"[RAW COMPLETION] {p}", base_ids, 20))

  for title, prompt_ids, gen_tokens in tests:
    generated_ids = list(prompt_ids)

    if jax.process_index() == 0:
      prompt_decoded = tokenizer.decode(prompt_ids)
      print(f"\n{'='*70}\n>>> {title}\n[INPUT TOKENS]: {prompt_decoded!r}\nGenerating ({gen_tokens} tokens): ", end="", flush=True)

    # Autoregressive greedy generation with JIT
    for step in range(gen_tokens):
      curr_len = len(generated_ids)
      if curr_len >= cfg.max_target_length:
        break
      padded_tokens = np.zeros((cfg.global_batch_size_to_train_on, cfg.max_target_length), dtype=np.int32)
      padded_tokens[0, :curr_len] = generated_ids
      positions = np.arange(cfg.max_target_length, dtype=np.int32)[None, :]
      segment_ids = (positions < curr_len).astype(np.int32)
      segment_ids = np.repeat(segment_ids, cfg.global_batch_size_to_train_on, axis=0)
      positions = np.repeat(positions, cfg.global_batch_size_to_train_on, axis=0)

      logits = forward_step(
          model,
          jnp.array(padded_tokens),
          jnp.array(positions),
          jnp.array(segment_ids),
      )

      # Gather logits across hosts
      logits = jax.experimental.multihost_utils.process_allgather(logits, tiled=True)
      if logits.ndim == 4:
        logits = jnp.reshape(logits, (-1, cfg.max_target_length, cfg.vocab_size))

      next_token = int(jnp.argmax(logits[0, curr_len - 1, :]))
      generated_ids.append(next_token)
      if jax.process_index() == 0:
        token_str = tokenizer.decode([next_token])
        print(token_str, end="", flush=True)

    if jax.process_index() == 0:
      output_text = tokenizer.decode(generated_ids)
      print(f"\n\n[FULL OUTPUT]:\n{output_text}\n{'='*70}", flush=True)


if __name__ == "__main__":
  main()

