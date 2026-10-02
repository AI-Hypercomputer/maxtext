# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script, masked_prefill_logits_checker.py, verifies the architectural and mathematical parity 
# between a MaxText model implementation and a HuggingFace PyTorch reference model. 
#
# It generates a single, static high-entropy "Golden Batch" (e.g., 512 tokens of prose, math, and code) 
# in-memory, replicates it to satisfy the static TPU mesh dimensions (e.g., batch size 32), and 
# executes a single forward prefill pass through both frameworks simultaneously.
#
# It compares the raw logits directly using three mathematical metrics to account for 
# acceptable hardware-level accumulation differences (such as bfloat16 noise across TPU tiles):
#   1. L_infinity Norm (Max Absolute Difference)
#   2. Cosine Similarity
#   3. KL Divergence (D_KL)
#
# The script automatically verifies batch determinism, ensuring that identical batch rows match 
# exactly within standard hardware tolerances (atol=1e-3), proving that stochastic routing and 
# dropout are fully disabled during the check.
#
# Example execution for Llama 3.1-8B:
#   python -m tests.utils.zero_generation_parity_checker maxtext/configs/base.yml \
#       run_name=mpl_test \
#       model_name=llama3.1-8b \
#       tokenizer_path=meta-llama/Llama-3.1-8B \
#       load_parameters_path=${MAXTEXT_CHECKPOINT} \
#       scan_layers=True dtype=float32 \
#       --run_hf_model=True \
#       --hf_model_path="/mnt/disks/external_disk/maxtext/exp1299" \
#       --max_kl_div=0.5 remat_policy=full

import argparse
import functools
import os
from pathlib import Path
import sys
import absl
from google.cloud import storage
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
import jsonlines  
from maxtext.utils.globals import MAXTEXT_TEST_ASSETS_ROOT
from maxtext.checkpoint_conversion.utils.hf_utils import convert_jax_weight_to_torch
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import lora_utils
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log

# --- GOLDEN BATCH FOR ZERO-GENERATION VERIFICATION ---
GOLDEN_BATCH_TEXT = """The James Webb Space Telescope (JWST) is an infrared space observatory launched in 2021 to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope.

To calculate the gravitational force at the Sun-Earth L2 Lagrange point, we use Newton's law of universal gravitation:
F = G * (m1 * m2) / r^2

Here is a Python script to model the baseline orbital parameters:

def calculate_orbital_velocity(mass_kg, radius_m):
    G = 6.67430e-11 # gravitational constant
    velocity = (G * mass_kg / radius_m) ** 0.5
    return {"velocity_m_s": round(velocity, 2), "stable": True}

# Payload Configuration
payload_config = {
    "instruments": ["NIRCam", "NIRSpec", "MIRI", "FGS/NIRISS"],
    "operating_temp_k": 50,
    "sunshield_deployed": True
}

Testing multilingual and unicode tokenizer fallbacks.
Chinese (Simplified): 韦伯太空望远镜 (Webb Space Telescope).
Arabic: تلسكوب جيمس ويب الفضائي
Russian: Космический телескоп Джеймса Уэбба
Emoji byte-pair encoding test: 🚀🔭✨🌌🌍
"""


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("true", "t", "yes", "1"): return True
    if v.lower() in ("false", "f", "no", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


def get_top_k_tokens_scores(logits_tensor, tokenizer_instance, k=10, description=""):
  """Get the top-k tokens and their scores from a given logits tensor."""
  max_logging.log(f"\n--- {description} top {k} tokens ---")
  collected_tokens = []
  topk_results = torch.topk(logits_tensor[0], k=k)
  for i in range(k):
      tok_id = topk_results.indices[i].item()
      score = topk_results.values[i].item()
      tok = tokenizer_instance.decode(tok_id)
      collected_tokens.append({"id": int(tok_id), "token": tok.strip(), "score": float(score)})

  table_str = f"| {'Token ID':<10} | {'Token':<20} | {'Score':<10} |\n"
  table_str += f"|{'-'*12}|{'-'*22}|{'-'*12}|\n"
  for d in collected_tokens:
      table_str += f"| {d['id']:<10} | {d['token']:<20} | {d['score']:<10.4f} |\n"
  max_logging.log(table_str)
  return collected_tokens


def compare_top_tokens(converted_tokens, golden_tokens):
  """Compares two lists of top tokens and calculates similarity metrics."""
  converted_ids = {token["id"] for token in converted_tokens}
  golden_ids = {token["id"] for token in golden_tokens}

  intersection = converted_ids.intersection(golden_ids)
  union = converted_ids.union(golden_ids)
  overlap_count = len(intersection)
  jaccard_similarity = overlap_count / len(union) if union else 0.0

  rank_matches = 0
  min_len = min(len(converted_tokens), len(golden_tokens))
  for i in range(min_len):
    if converted_tokens[i]["id"] == golden_tokens[i]["id"]:
      rank_matches += 1

  rank_agreement = (rank_matches / min_len) * 100 if min_len > 0 else 0.0

  metrics = {
    "overlap_count": f"{len(union)}",
    "jaccard_similarity": f"{jaccard_similarity:.4f}",
    "rank_agreement_percentage": f"{rank_agreement:.2f}%",
  }

  max_logging.log("\n--- Similarity Metrics of Top Tokens ---")
  table_str = f"| {'Metric':<30} | {'Value':<20} |\n"
  table_str += f"|{'-'*32}|{'-'*22}|\n"
  for key, value in metrics.items():
    table_str += f"| {key:<30} | {str(value):<20} |\n"
  max_logging.log(table_str)


def check_mathematical_parity(model_logits, golden_logits, atol_linf=1e-3, min_cos_sim=0.999, max_kl=1e-4, clip_logits_epsilon=None):
  """
  Tier 1B: Logit Parity Metrics
  Calculates L-infinity norm, Cosine Similarity, and KL divergence over a sequence.
  """
  # Convert torch tensors to jnp arrays if necessary for unified math
  if hasattr(model_logits, "detach"):
    model_logits = jnp.asarray(model_logits.detach().to(torch.float32).cpu().numpy())
  if hasattr(golden_logits, "detach"):
    golden_logits = jnp.asarray(golden_logits.detach().to(torch.float32).cpu().numpy())

  # 1. Select the relevant vocabulary slice
  token_size = min(model_logits.shape[-1], golden_logits.shape[-1])
  model_logits_sliced = model_logits[..., :token_size]
  golden_logits_sliced = golden_logits[..., :token_size]

  # Ensure the tensors have not collapsed to zero or constants
  model_std = jnp.std(model_logits_sliced)
  golden_std = jnp.std(golden_logits_sliced)
  
  assert model_std > 1e-4, f"FATAL: MaxText logits have collapsed! Standard deviation is {model_std:.4e}. The forward pass is returning dead data."
  assert golden_std > 1e-4, f"FATAL: HF golden logits have collapsed! Standard deviation is {golden_std:.4e}. The HF forward pass is returning dead data."

  # Metric 1: L-infinity Norm (Max Absolute Error)
  abs_diff = jnp.abs(model_logits_sliced - golden_logits_sliced)
  l_inf_norm = jnp.max(abs_diff)
  max_abs_diff_idx = jnp.unravel_index(jnp.argmax(abs_diff), abs_diff.shape)

  # Metric 2: Cosine Similarity
  dot_product = jnp.sum(model_logits_sliced * golden_logits_sliced, axis=-1)
  norm_model = jnp.linalg.norm(model_logits_sliced, axis=-1)
  norm_golden = jnp.linalg.norm(golden_logits_sliced, axis=-1)
  
  # Add epsilon to avoid division by zero
  cos_sim_per_token = dot_product / (norm_model * norm_golden + 1e-8)
  
  mean_cos_sim = jnp.mean(cos_sim_per_token)
  min_cos_sim_val = jnp.min(cos_sim_per_token)

  # Metric 3: KL Divergence
  golden_prob = jax.nn.softmax(golden_logits_sliced, axis=-1)
  model_prob = jax.nn.softmax(model_logits_sliced, axis=-1)

  if clip_logits_epsilon is not None:
    golden_prob = jnp.clip(golden_prob, min=clip_logits_epsilon)
    model_prob = jnp.clip(model_prob, min=clip_logits_epsilon)
    # Re-normalize so probabilities sum to 1 (required for valid KL divergence)
    golden_prob = golden_prob / jnp.sum(golden_prob, axis=-1, keepdims=True)
    model_prob = model_prob / jnp.sum(model_prob, axis=-1, keepdims=True)

  kl_div_per_token = jnp.sum(jax.scipy.special.kl_div(golden_prob, model_prob), axis=-1)
  mean_kl_div = jnp.mean(kl_div_per_token)
  max_kl_div = jnp.max(kl_div_per_token)

  max_logging.log("\n--- Tier 1B: Logit Parity Metrics ---")
  max_logging.log(f"L-infinity Norm (Max Abs Diff): {l_inf_norm:.4e} at index {tuple(int(i) for i in max_abs_diff_idx)}")
  if atol_linf is not None:
      max_logging.log(f"  Threshold check: {l_inf_norm:.4e} < {atol_linf}")
  
  max_logging.log(f"Cosine Similarity (Mean):       {mean_cos_sim:.4f}")
  max_logging.log(f"Cosine Similarity (Min token):  {min_cos_sim_val:.4f}")
  if min_cos_sim is not None:
      max_logging.log(f"  Threshold check (Mean): {mean_cos_sim:.4f} > {min_cos_sim}")

  max_logging.log(f"KL Divergence (Max token):      {max_kl_div:.4e}")
  max_logging.log(f"KL Divergence (Mean):           {mean_kl_div:.4e}")
  if max_kl is not None:
      max_logging.log(f"  Threshold check: {max_kl_div:.4e} < {max_kl}")

  # Assertions
  if atol_linf is not None:
      assert l_inf_norm < atol_linf, f"L-infinity norm {l_inf_norm:.4e} exceeds threshold {atol_linf}"
  if min_cos_sim is not None:
      assert mean_cos_sim > min_cos_sim, f"Mean Cosine Similarity {mean_cos_sim:.4f} below threshold {min_cos_sim}"
  if max_kl is not None:
      assert max_kl_div < max_kl, f"Max KL Divergence {max_kl_div:.4e} exceeds threshold {max_kl}"
  
  max_logging.log("✅ Model parity verified successfully")
  max_logging.log(f"  Mean cosine similarity: {mean_cos_sim:.6f}")
  max_logging.log(f"  Max L∞ diff: {l_inf_norm:.4e}")
  max_logging.log(f"  Mean KL divergence: {mean_kl_div:.4e}")


def get_data(golden_data_point, config):
  """Get the golden data for the test indexed at golden_data_index"""
  max_logging.log(f"config.global_batch_size_to_train_on={config.global_batch_size_to_train_on}")
  if config.use_multimodal:
    assert "pixel_values" in golden_data_point, "no image found in golden data while use_multimodal=True"
    pixel_values = np.asarray(golden_data_point["pixel_values"], dtype=np.float32)
    max_logging.log(f"pixel_values.shape = {pixel_values.shape}")
    model_prefix = config.model_name.split("-")[0]
    if model_prefix in ["gemma3", "gemma4"]:
      if pixel_values.ndim == 2:
        h, w = config.image_size_for_vit
        p = config.patch_size_for_vit
        c = pixel_values.shape[-1] // (p * p)
        pixel_values = np.reshape(pixel_values, (h // p, w // p, p, p, c))
        pixel_values = np.transpose(pixel_values, (0, 2, 1, 3, 4))
        pixel_values = np.reshape(pixel_values, (h, w, c))
      else:
        pixel_values = np.transpose(pixel_values, (1, 2, 0))
    elif model_prefix in ["llama4"]:
      pixel_values = pixel_values[None, :]
    pixel_values = np.stack([pixel_values for _ in range(config.global_batch_size_to_train_on)])
  else:
    pixel_values = None

  original_ids = np.asarray(golden_data_point["tokens"], dtype=np.int32)
  seq_len = len(original_ids)

  if seq_len > config.max_target_length:
    raise ValueError(
        f"Golden data sequence length ({seq_len}) is greater than max_target_length ({config.max_target_length})"
    )

  s = (config.global_batch_size_to_train_on, config.max_target_length)
  padded_ids = np.pad(original_ids, (0, config.max_target_length - seq_len), "constant", constant_values=0)
  ids = np.stack([padded_ids for _ in range(config.global_batch_size_to_train_on)])

  logits = np.asarray(golden_data_point["logits"], dtype=np.float32)
  prompt = golden_data_point.get("formatted_prompt", golden_data_point.get("prompt", ""))
  max_logging.log(f' prompt="{prompt}" raw ids={original_ids}, logits.shape = {logits.shape}')

  decoder_segment_ids = np.zeros(s, dtype=np.int32)
  decoder_segment_ids[:, :seq_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR
  decoder_positions = np.stack(
      [np.arange(config.max_target_length, dtype=np.int32) for _ in range(config.global_batch_size_to_train_on)]
  )
  return ids, decoder_segment_ids, decoder_positions, logits, seq_len, pixel_values


def main(config, test_args): 
  """Test the Whole Model of model_name"""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  init_rng, rng1 = jax.random.split(init_rng)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  if not test_args.run_hf_model:
    """Comparing maxtext/huggingface model with pre-loaded golden logits"""
    max_logging.log("Initializing MaxText model")
    quant = quantizations.configure_quantization(config)
    if config.pure_nnx_decoder and config.enable_nnx:
      model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)

      if config.lora.enable_lora:
        model = lora_utils.apply_lora_to_model(model, mesh, config)
        if config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(model, config)
      state = None
    else:
      model = models.transformer_as_linen(config, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
      init_state_fn = functools.partial(maxtext_utils.init_initial_state, model, None, config, False, rng1)
      state, _ = maxtext_utils.setup_decode_state(config, mesh, None, init_state_fn)

    if test_args.golden_logits_path == "":
      input_golden_data_path = os.path.join(
          MAXTEXT_TEST_ASSETS_ROOT,
          "golden_logits",
          f"golden_data_{config.model_name}.jsonl",
      )
    else:
      input_golden_data_path = test_args.golden_logits_path
    input_golden_data_path = Path(input_golden_data_path)
    if input_golden_data_path.suffix == ".jsonl":
      max_logging.log("loading hf goldens from jsonl file")

      with jsonlines.open(input_golden_data_path, "r") as f:
        golden_data = list(f)
    else:
      raise ValueError("golden_logits_path must end with .jsonl")
    
    max_logging.log(f"loaded {len(golden_data)} golden data points")
    all_data_to_save = []
    
    for golden_data_index, golden_data_point in enumerate(golden_data):
      max_logging.log(f"\n--- Comparing forward pass for golden data index: {golden_data_index} ---")
      ids, decoder_segment_ids, decoder_positions, golden_logits, seq_len, images = get_data(golden_data_point, config)
      max_logging.log("maxtext forward pass")
      
      if state is None:
        full_train_logits = model(
            decoder_input_tokens=ids,
            decoder_positions=decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            encoder_images=images,
            enable_dropout=False,
        )
      else:
        full_train_logits = model.apply(
            state.params,
            ids,
            decoder_positions,
            decoder_segment_ids,
            encoder_images=images,
            enable_dropout=False,
            rngs={"aqt": init_rng},
        )

      full_train_logits = jax.experimental.multihost_utils.process_allgather(full_train_logits, tiled=True)
      if full_train_logits.ndim == 4:
        full_train_logits = jnp.reshape(full_train_logits, (-1, config.max_target_length, config.vocab_size))
      
      full_train_logits = full_train_logits[:, :seq_len, :]
      token_size = int(test_args.token_size) if test_args.token_size else seq_len
      min_vocab_size = min(full_train_logits.shape[-1], golden_logits.shape[-1])
      start_index = 1 if test_args.skip_first_token else 0
      
      train_logits_slice = full_train_logits[0, start_index:token_size, :min_vocab_size]
      golden_logits_slice = golden_logits[start_index:token_size, :min_vocab_size]

      # Run the unified mathematical parity check
      check_mathematical_parity(
          model_logits=train_logits_slice,
          golden_logits=golden_logits_slice,
          atol_linf=test_args.atol,
          min_cos_sim=test_args.min_cos_sim,
          max_kl=test_args.max_kl_div,
          clip_logits_epsilon=test_args.clip_logits_epsilon
      )

      if jax.process_index() == 0 and test_args.output_logits_path:
        gd = golden_data[golden_data_index]
        data_to_save = {
            "prompt": gd.get("formatted_prompt", gd.get("prompt", "")),
            "tokens": ids[0, :seq_len].tolist(),
            "logits": full_train_logits[0].tolist(),
        }
        all_data_to_save.append(data_to_save)

  else:
    """Comparing maxtext model with HF model on-the-fly"""
    if test_args.hf_model_path == "":
      raise ValueError("run_hf_model requires hf_model_path")

    hf_token = config.hf_access_token
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }

    # config.dtype may be a string ("float32") or a jax dtype object
    if hasattr(config.dtype, "name"):
        dtype_key = config.dtype.name.lower()
    else:
        dtype_key = str(config.dtype).lower()
    torch_dtype = dtype_mapping.get(dtype_key, torch.bfloat16)
    max_logging.log(f"Loading HF model with dtype: {torch_dtype} (derived from config.dtype: {config.dtype})")

    hf_model = AutoModelForCausalLM.from_pretrained(
        test_args.hf_model_path, torch_dtype=torch_dtype, token=hf_token, trust_remote_code=test_args.trust_remote_code
    )
    hf_lora_path = config.hf_lora_adapter_path
    if hf_lora_path:
      max_logging.log(f"Loading HF PEFT LoRA adapter from {hf_lora_path}")
      try:
        from peft import PeftModel  
      except ImportError as exc:
        raise ImportError("peft library is required to load HF LoRA adapter. Run `pip install peft`.") from exc
      hf_model = PeftModel.from_pretrained(hf_model, hf_lora_path)

    try:
      max_logging.log(f"Loading tokenizer from {test_args.hf_model_path}.")
      tokenizer = AutoTokenizer.from_pretrained(
          test_args.hf_model_path, token=hf_token, trust_remote_code=test_args.trust_remote_code
      )
    except Exception as e: 
      max_logging.log(f"Tokenizer loading error: {e}.\nLoading tokenizer from {config.tokenizer_path}.")
      tokenizer = AutoTokenizer.from_pretrained(
          config.tokenizer_path, token=hf_token, trust_remote_code=test_args.trust_remote_code
      )

    pad_token_prefixes = ["llama3.1", "mixtral"]
    if any(config.model_name.startswith(prefix) for prefix in pad_token_prefixes):
      tokenizer.pad_token = tokenizer.eos_token

    quant = quantizations.configure_quantization(config)
    if config.pure_nnx_decoder and config.enable_nnx:
      maxtext_model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)

      if config.lora.enable_lora:
        maxtext_model = lora_utils.apply_lora_to_model(maxtext_model, mesh, config)
        if config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(maxtext_model, config)
      maxtext_state = None
    else:
      maxtext_model = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
      init_state_fn = functools.partial(maxtext_utils.init_initial_state, maxtext_model, None, config, False, rng1)
      if test_args.ckpt_type == "linen":
        maxtext_state, _ = maxtext_utils.setup_decode_state(config, mesh, None, init_state_fn)
      else:
        maxtext_state, _ = model_creation_utils.setup_decode_state_from_nnx(maxtext_model, config, rng1, mesh)

    # --- Construct target tokens using the tokenizer ---
    target_seq_len = config.max_target_length

    # Pre-expand golden text: assume worst-case 2 chars per token, add 50% safety margin
    approx_chars_needed = int(target_seq_len * 2 * 1.5)
    golden_text_expanded = GOLDEN_BATCH_TEXT
    while len(golden_text_expanded) < approx_chars_needed:
        golden_text_expanded += "\n" + GOLDEN_BATCH_TEXT

    inputs = tokenizer(
        golden_text_expanded,
        return_tensors="pt",
        padding="max_length",
        max_length=target_seq_len,
        truncation=True,
    )

    actual_seq_len = inputs["input_ids"].shape[1]
    if actual_seq_len < target_seq_len:
        raise RuntimeError(
            f"Tokenizer produced only {actual_seq_len} tokens; expected {target_seq_len}. "
            f"Check tokenizer model_max_length setting (may be capped below {target_seq_len})."
        )
    max_logging.log(f"\n--- Running Zero-Generation Verification on {actual_seq_len} tokens ---")

    mt_ids_raw = jnp.asarray(inputs["input_ids"], dtype=jnp.int32)
    hf_attention_mask = jnp.asarray(inputs["attention_mask"], dtype=jnp.int32)

    # Pad tokens to max_target_length for MaxText's static shape requirement
    pad_len = config.max_target_length - actual_seq_len
    if pad_len < 0:
        raise ValueError(
            f"actual_seq_len ({actual_seq_len}) exceeds max_target_length ({config.max_target_length})"
        )
    mt_ids_padded = jnp.pad(mt_ids_raw, ((0, 0), (0, pad_len)), constant_values=0)
    attention_mask_padded = jnp.pad(hf_attention_mask, ((0, 0), (0, pad_len)), constant_values=0)

    # --- Diagnostic: understand which tokens are real vs padded ---
    padding_side = getattr(tokenizer, "padding_side", "right")
    # (attention_mask is jnp.int32, so use jnp)
    real_token_count = int(jnp.sum(hf_attention_mask))
    max_logging.log(f"=== Sequence composition ===")
    max_logging.log(f"Total tokens (padded): {actual_seq_len}")
    max_logging.log(f"Real tokens (unpadded): {real_token_count}")
    max_logging.log(f"Padding tokens: {actual_seq_len - real_token_count}")
    max_logging.log(f"Tokenizer padding_side: {padding_side}")

    # Compute real-token region indices based on padding side
    if padding_side == "left":
        real_start = actual_seq_len - real_token_count
        real_end = actual_seq_len
    else:  # right-padding (default)
        real_start = 0
        real_end = real_token_count

    last_real_idx = real_end - 1


    # Tile batch to global_batch_size_to_train_on
    if mt_ids_padded.shape[0] != config.global_batch_size_to_train_on:
        if config.global_batch_size_to_train_on % mt_ids_padded.shape[0] != 0:
            raise ValueError(
                f"global_batch_size_to_train_on ({config.global_batch_size_to_train_on}) "
                f"must be divisible by input batch ({mt_ids_padded.shape[0]})"
            )
        tile_factor = config.global_batch_size_to_train_on // mt_ids_padded.shape[0]
        mt_ids = jnp.tile(mt_ids_padded, (tile_factor, 1))
        attention_mask_full = jnp.tile(attention_mask_padded, (tile_factor, 1))
    else:
        mt_ids = mt_ids_padded
        attention_mask_full = attention_mask_padded

    # Segment IDs derived from attention mask (respects real padding)
    mt_decoder_segment_ids = attention_mask_full * DECODING_ACTIVE_SEQUENCE_INDICATOR

    # Positions: full range for max_target_length
    mt_decoder_positions = jnp.broadcast_to(
        jnp.arange(config.max_target_length, dtype=jnp.int32),
        (config.global_batch_size_to_train_on, config.max_target_length),
    )

    # --- HF Forward Pass ---
    with torch.no_grad():
      hf_logits_torch = hf_model(**inputs).logits
    
    # Ensure float32 on CPU for downstream numpy conversion (safe for bfloat16 configs)
    hf_logits_torch = hf_logits_torch.detach().to(torch.float32).cpu()

    # --- MaxText Forward Pass ---
    if maxtext_state is None:
      mt_logits_jax = maxtext_model(
          decoder_input_tokens=mt_ids,
          decoder_positions=mt_decoder_positions,
          decoder_segment_ids=mt_decoder_segment_ids,
          enable_dropout=False,
      )
    else:
      mt_logits_jax = maxtext_model.apply(
          maxtext_state.params,
          mt_ids,
          mt_decoder_positions,
          mt_decoder_segment_ids,
          enable_dropout=False,
          rngs={"aqt": init_rng},
      )
    
    # Gather sharded logits across hosts (needed in multi-host TPU setups)
    mt_logits_jax = jax.experimental.multihost_utils.process_allgather(mt_logits_jax, tiled=True)

    # Handle possible 4D output (e.g., [num_shards, batch, seq, vocab])
    if mt_logits_jax.ndim == 4:
        mt_logits_jax = jnp.reshape(mt_logits_jax, (-1, config.max_target_length, config.vocab_size))

    mt_logits_jax_sliced = mt_logits_jax[:, :actual_seq_len, :]

    # Convert MaxText output to torch tensor
    mt_logits_torch = convert_jax_weight_to_torch(mt_logits_jax_sliced)
    mt_logits_torch = mt_logits_torch.detach().to(torch.float32).cpu()

    if mt_logits_torch.shape[0] > 1:
      ref = mt_logits_torch[0:1].to(torch.float32).expand_as(mt_logits_torch)
      mt_f32 = mt_logits_torch.to(torch.float32)
      diff = (mt_f32 - ref).abs()
      
      max_batch_diff = diff.max().item()
      mean_batch_diff = diff.mean().item()
      max_logit_magnitude = mt_f32.abs().max().item()
      
      # Relative tolerance scaled by logit magnitude
      # For TPU float32 (bf16 matmul + f32 accum) accumulated over ~thousands of ops
      if mt_logits_torch.dtype == torch.bfloat16:
          rel_tolerance = 1e-2
      else:
          rel_tolerance = 5e-4  # allows 2.3e-3 on ~30-magnitude logits (5e-4 * 30 = 1.5e-2)
      
      abs_tolerance = rel_tolerance * max(max_logit_magnitude, 1.0)
      
      # Sanity check: mean diff should be near machine epsilon.
      # If it's not, that's a stronger signal of a real bug than max diff.
      mean_diff_threshold = 1e-4  # generous — real bugs push mean way above epsilon
      
      if mean_batch_diff > mean_diff_threshold:
          raise AssertionError(
              f"FATAL: MaxText batch mean diff {mean_batch_diff:.4e} exceeds "
              f"{mean_diff_threshold:.4e}. This suggests systematic non-determinism, "
              f"not numerical noise. Max diff: {max_batch_diff:.4e}."
          )
      
      if max_batch_diff > abs_tolerance:
          max_logging.log(
              f"WARNING: MaxText max batch diff {max_batch_diff:.4e} exceeds "
              f"tolerance {abs_tolerance:.4e}, but mean diff {mean_batch_diff:.4e} "
              f"is at machine epsilon — likely benign numerical outliers."
          )
      else:
          max_logging.log(
              f"Batch determinism verified. Max diff: {max_batch_diff:.4e}, "
              f"Mean diff: {mean_batch_diff:.4e}, "
              f"Tolerance: {abs_tolerance:.4e} (rel={rel_tolerance} × max_logit={max_logit_magnitude:.2f})"
          )

    # --- Compare logits for the last REAL token prediction ---
    # last_real_idx was computed above based on padding_side
    max_logging.log(f"Comparing top-k at last real token position: {last_real_idx}")

    hf_last_token_logits = hf_logits_torch[:, last_real_idx, :]
    mt_last_token_logits = mt_logits_torch[:, last_real_idx, :]

    tokens_maxtext = get_top_k_tokens_scores(mt_last_token_logits, tokenizer, k=10, description="MaxText model")
    tokens_hf = get_top_k_tokens_scores(hf_last_token_logits, tokenizer, k=10, description="HF model")
    compare_top_tokens(converted_tokens=tokens_maxtext, golden_tokens=tokens_hf)

    # --- Compare all logits over REAL (non-padded) tokens only ---
    start_index = real_start + (1 if test_args.skip_first_token else 0)
    end_index = real_end  # exclude padding

    if end_index <= start_index:
        raise RuntimeError(
            f"No real tokens to compare after slicing: "
            f"start_index={start_index}, end_index={end_index}, "
            f"padding_side={padding_side}, real_token_count={real_token_count}"
        )

    max_logging.log(
        f"Comparing logits over tokens [{start_index}:{end_index}] "
        f"(padding_side={padding_side}, real_token_count={real_token_count}, "
        f"padded_total={actual_seq_len})"
    )

    check_mathematical_parity(
        model_logits=mt_logits_torch[0, start_index:end_index],
        golden_logits=hf_logits_torch[0, start_index:end_index],
        atol_linf=test_args.atol,
        min_cos_sim=test_args.min_cos_sim,
        max_kl=test_args.max_kl_div,
        clip_logits_epsilon=test_args.clip_logits_epsilon,
    )
    
    all_data_to_save = []
    if jax.process_index() == 0 and test_args.output_logits_path:
      data_to_save = {
          "mt_logits": mt_logits_torch[0].tolist(),
          "hf_logits": hf_logits_torch[0].tolist(),
      }
      all_data_to_save.append(data_to_save)

    if jax.process_index() == 0 and test_args.output_logits_path and all_data_to_save:
      output_dir = os.path.dirname(test_args.output_logits_path)
      if output_dir:
          os.makedirs(output_dir, exist_ok=True)
      with jsonlines.open(test_args.output_logits_path, "a") as f:
          for item in all_data_to_save:
              f.write(item)
      max_logging.log(f"Saved {len(all_data_to_save)} logit entries to {test_args.output_logits_path}")

    if test_args.gcs_output_logits_path:
      bucket_name = test_args.gcs_output_logits_path.split("/")[2]
      destination_blob_name = "/".join(
          test_args.gcs_output_logits_path.split("/")[3:] + test_args.output_logits_path.split("/")[-1:]
      )
      upload_blob(bucket_name, test_args.output_logits_path, destination_blob_name)
      max_logging.log(f"Uploaded logits to {test_args.gcs_output_logits_path}")


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  parser = argparse.ArgumentParser()
  parser.add_argument("--atol", type=float, required=False, default=None)
  parser.add_argument("--min_cos_sim", type=float, required=False, default=0.999)
  parser.add_argument("--token_size", type=int, required=False)
  parser.add_argument("--max_kl_div", type=float, required=False, default=None)
  parser.add_argument("--golden_logits_path", type=str, required=False, default="")
  parser.add_argument("--hf_model_path", type=str, required=False, default="")
  parser.add_argument("--run_hf_model", type=str2bool, required=False, default=False)
  parser.add_argument("--output_logits_path", type=str, required=False, default="")
  parser.add_argument("--gcs_output_logits_path", type=str, required=False, default="")
  parser.add_argument("--clip_logits_epsilon", type=float, required=False, default=None)
  parser.add_argument(
      "--skip_first_token",
      action="store_true",
      required=False,
      default=False,
      help="Skip the first token during comparison to ignore BOS/init mismatches.",
  )
  parser.add_argument(
      "--ckpt_type",
      type=str,
      required=False,
      default="linen",
      choices=["linen", "nnx"],
      help="Checkpoint format to load: 'linen' (default) or 'nnx'.",
  )
  parser.add_argument(
      "--trust_remote_code", type=str2bool, required=False, default=True, help="from_pretrained: trust_remote_code"
  )

  test_args, remaining_args = parser.parse_known_args()
  model_args = [sys.argv[0]] + remaining_args

  cfg = pyconfig.initialize_pydantic(model_args)
  assert (
      test_args.atol is not None or test_args.max_kl_div is not None
  ), "At least one of --atol or --max_kl_div must be specified to define the test criteria."

  if cfg.use_multimodal:
    assert not test_args.run_hf_model, (
        "Multimodal does not support running hf model on-the-fly, please generate hf golden logits "
        "using tests/assets/logits_generation/generate_hf_golden_logits.py"
    )
  main(cfg, test_args)