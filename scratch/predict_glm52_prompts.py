import functools
import os
import sys
from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.utils import model_creation_utils


def get_top_k(logits_1d, tokenizer, k=10):
  probs = jax.nn.softmax(logits_1d, axis=-1)
  top_indices = np.argsort(np.asarray(logits_1d))[-k:][::-1]
  results = []
  for idx in top_indices:
    try:
      tok_str = tokenizer.decode([int(idx)])
    except Exception:
      tok_str = f"<token_{idx}>"
    results.append((int(idx), tok_str, float(logits_1d[idx]), float(probs[idx])))
  return results


def main(argv: Sequence[str]):
  import absl.logging
  absl.logging.set_verbosity(absl.logging.INFO)
  config = pyconfig.initialize(argv)
  print("Initializing JAX distributed system for GLM-5.2...", flush=True)
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  print(f"JAX Process {jax.process_index()}/{jax.process_count()} initialized. Mesh shape: {mesh.shape}", flush=True)
  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
  print(f"Loaded tokenizer from {config.tokenizer_path}", flush=True)

  print(f"Building GLM-5.2 model from checkpoint: {config.load_parameters_path}...", flush=True)
  model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  print("GLM-5.2 model created and checkpoint restored successfully!", flush=True)

  test_cases = [
      ("Raw Prompt 1", "The capital of France is"),
      ("Raw Prompt 2", "The largest planet in our solar system is"),
      ("Raw Prompt 3", "Deep learning is a subset of machine learning that focuses on"),
      ("GLM Tagged Math", "<|user|>\nWhat is 25 * 4? Give only the number.\n<|assistant|>\n"),
      ("GLM Tagged Code", "<|user|>\nWrite a Python function to check if a number is prime.\n<|assistant|>\n"),
      ("GLM Tagged QA", "<|user|>\nWhat is the boiling point of water in Celsius?\n<|assistant|>\n"),
  ]

  from flax import nnx

  @nnx.jit
  def forward_step(model, tokens, positions, segment_ids):
    return model(
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
    )

  max_len = config.max_target_length
  output_log_path = "/tmp/glm52_predictions_output.txt"
  out_file = open(output_log_path, "w")

  def log_out(msg):
    print(msg, flush=True)
    out_file.write(msg + "\n")
    out_file.flush()

  if jax.process_index() == 0:
    log_out("=" * 80)
    log_out("GLM-5.2 Cross-Layer IndexShare 744B Model Sanity Evaluation")
    log_out(f"Model: {config.model_name} | Checkpoint: {config.load_parameters_path}")
    log_out(f"IndexShare Pattern: {config.index_share_pattern} | Use Index Share: {config.use_index_share}")
    log_out("=" * 80 + "\n")

  for label, prompt_str in test_cases:
    token_ids = tokenizer.encode(prompt_str)
    seq_len = len(token_ids)
    if jax.process_index() == 0:
      log_out("\n" + "=" * 80)
      log_out(f"Test Case: [{label}]")
      log_out(f"Prompt: {repr(prompt_str)}")
      log_out(f"Prompt Tokens ({seq_len} tokens): {token_ids}")
      log_out("-" * 80)

    current_tokens = np.zeros((config.global_batch_size_to_train_on, max_len), dtype=np.int32)
    current_tokens[:, :seq_len] = np.array(token_ids, dtype=np.int32)
    positions = np.stack([np.arange(max_len, dtype=np.int32) for _ in range(config.global_batch_size_to_train_on)])
    segment_ids = np.zeros((config.global_batch_size_to_train_on, max_len), dtype=np.int32)
    segment_ids[:, :seq_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR

    # Step 1: Top Next-Token Prediction
    logits = forward_step(model, current_tokens, positions, segment_ids)
    gathered_logits = jax.experimental.multihost_utils.process_allgather(logits, tiled=True)
    if gathered_logits.ndim == 4:
      gathered_logits = jnp.reshape(gathered_logits, (-1, max_len, config.vocab_size))

    last_logits = np.asarray(gathered_logits[0, seq_len - 1, :])
    top_tokens = get_top_k(last_logits, tokenizer, k=10)

    if jax.process_index() == 0:
      log_out("\nTop 10 Predictions for Next Token:")
      log_out(f"{'Rank':<5} | {'Token ID':<10} | {'Token':<22} | {'Logit':<10} | {'Probability':<12}")
      log_out("-" * 68)
      for rank, (t_id, t_str, logit_val, prob_val) in enumerate(top_tokens, 1):
        log_out(f"{rank:<5} | {t_id:<10} | {repr(t_str):<22} | {logit_val:<10.4f} | {prob_val:<12.6f}")

    # Step 2: Greedy Autoregressive Generation
    gen_tokens = list(token_ids)
    curr_len = seq_len
    max_gen_tokens = min(40, max_len - seq_len)
    for _ in range(max_gen_tokens):
      if curr_len >= max_len:
        break
      segment_ids[:, :curr_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR
      logits = forward_step(model, current_tokens, positions, segment_ids)
      gathered_logits = jax.experimental.multihost_utils.process_allgather(logits, tiled=True)
      if gathered_logits.ndim == 4:
        gathered_logits = jnp.reshape(gathered_logits, (-1, max_len, config.vocab_size))

      next_tok = int(np.argmax(np.asarray(gathered_logits[0, curr_len - 1, :])))
      gen_tokens.append(next_tok)
      current_tokens[:, curr_len] = next_tok
      curr_len += 1

      if next_tok in [tokenizer.eos_token_id, 154820]:
        break

    if jax.process_index() == 0:
      continuation_text = tokenizer.decode(gen_tokens[seq_len:])
      full_text = tokenizer.decode(gen_tokens)
      log_out(f"\n[Generated Continuation]:\n{repr(continuation_text)}")
      log_out(f"\n[Full Generated Text]:\n{repr(full_text)}\n")

  out_file.close()
  if jax.process_index() == 0:
    gcs_dest = "gs://maxtext-glm5-europe-west4/predictions_glm52_78l.txt"
    os.system(f"gcloud storage cp {output_log_path} {gcs_dest} || true")
    log_out(f"\nSaved full predictions log to: {gcs_dest}")


if __name__ == "__main__":
  main(sys.argv[1:])
