# Copyright 2026 Google LLC
"""Standalone generation verification for GLM-5.2."""

import sys
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def main():
  cfg = pyconfig.initialize_pydantic(sys.argv)
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  
  if jax.process_index() == 0:
    print("=== Loading GLM-5.2 Tokenizer ===")
  tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True)

  if jax.process_index() == 0:
    print(f"=== Restoring GLM-5.2 (744B) Model from {cfg.load_parameters_path} ===")
  model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode="train")

  if jax.process_index() == 0:
    print("=== Model restored successfully! Running prompt generation test ===")

  test_prompts = [
      "The capital of France is",
      "In mathematics, 2 + 2 =",
      "The largest ocean on Earth is the",
  ]

  for prompt in test_prompts:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    generated_ids = list(prompt_ids)

    # Generate 15 tokens greedily
    for step in range(15):
      curr_len = len(generated_ids)
      padded_tokens = np.zeros((cfg.global_batch_size_to_train_on, cfg.max_target_length), dtype=np.int32)
      padded_tokens[0, :curr_len] = generated_ids
      positions = np.arange(cfg.max_target_length, dtype=np.int32)[None, :]
      segment_ids = (positions < curr_len).astype(np.int32)
      segment_ids = np.repeat(segment_ids, cfg.global_batch_size_to_train_on, axis=0)
      positions = np.repeat(positions, cfg.global_batch_size_to_train_on, axis=0)

      logits = model(
          decoder_input_tokens=jnp.array(padded_tokens),
          decoder_positions=jnp.array(positions),
          decoder_segment_ids=jnp.array(segment_ids),
          enable_dropout=False,
      )

      # Gather logits across hosts
      logits = jax.experimental.multihost_utils.process_allgather(logits, tiled=True)
      if logits.ndim == 4:
        logits = jnp.reshape(logits, (-1, cfg.max_target_length, cfg.vocab_size))

      next_token = int(jnp.argmax(logits[0, curr_len - 1, :]))
      generated_ids.append(next_token)

    if jax.process_index() == 0:
      output_text = tokenizer.decode(generated_ids)
      print(f"\n[PROMPT]: {prompt!r}")
      print(f"[GENERATION]: {output_text!r}\n" + "=" * 60)


if __name__ == "__main__":
  main()
