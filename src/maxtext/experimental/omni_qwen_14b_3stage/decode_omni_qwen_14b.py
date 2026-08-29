# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoregressive Decoding & Inspection for Stitched Qwen3-VL-4B + Qwen3-14B Model.

Evaluates stitched (untrained pre-SFT or post-SFT) checkpoints on:
1. Pure text-only prompts (to verify LLM backbone weights and tokenizer are intact).
2. Multimodal ChartQA validation samples / custom image prompts (with 3D MRoPE & token fusion).

Usage:
  # 1. Text-only decoding (verifies LLM backbone weights):
  python3 src/maxtext/experimental/omni_qwen_14b/decode_omni_qwen_14b.py \
    --checkpoint_path="gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items" \
    --text_only \
    --prompt="What is the capital of France?"

  # 2. Text-only default test prompts:
  python3 src/maxtext/experimental/omni_qwen_14b/decode_omni_qwen_14b.py \
    --checkpoint_path="gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items" \
    --text_only

  # 3. Multimodal decode random ChartQA validation samples:
  python3 src/maxtext/experimental/omni_qwen_14b/decode_omni_qwen_14b.py \
    --checkpoint_path="gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items" \
    --num_samples=3 \
    --max_new_tokens=128

  # 4. Multimodal decode with custom text prompt and synthetic/local image:
  python3 src/maxtext/experimental/omni_qwen_14b/decode_omni_qwen_14b.py \
    --checkpoint_path="gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items" \
    --prompt="Describe the trends shown in this chart."
"""

import maxtext
# Eagerly initialize core MaxText C++ and model dependencies
_ = (maxtext.Mesh, maxtext.pyconfig, maxtext.models, maxtext.model_creation_utils)

import functools
import os
import random
import sys
from absl import app, flags
import datasets
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import omegaconf
from PIL import Image
from transformers import AutoTokenizer

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

FLAGS = flags.FLAGS


def _define_flag(fn, name, default, help_str):
  if name not in FLAGS:
    fn(name, default, help_str)


_define_flag(
    flags.DEFINE_string,
    "config_path",
    os.path.join(
        MAXTEXT_PKG_DIR,
        "experimental",
        "omni_qwen_14b_3stage",
        "maxtext-omni-qwen3-vl-14b.yml",
    ),
    "Path to model YAML configuration.",
)
_define_flag(
    flags.DEFINE_string,
    "checkpoint_path",
    "gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_qwen3_vl_14b_3stage/0/items",
    "Path to checkpoint parameters directory.",
)
_define_flag(flags.DEFINE_boolean, "text_only", False, "Enable pure text-only decoding without visual input.")
_define_flag(flags.DEFINE_integer, "num_samples", 3, "Number of random ChartQA validation samples to evaluate.")
_define_flag(flags.DEFINE_string, "prompt", "", "Optional custom prompt for single-image or text-only decoding.")
_define_flag(flags.DEFINE_string, "image_path", "", "Optional local path to an image file.")
_define_flag(flags.DEFINE_string, "description", "Omni Qwen3-VL-4B + Qwen3-14B", "Description for evaluation logs.")
_define_flag(flags.DEFINE_integer, "max_new_tokens", 128, "Maximum number of new tokens to generate.")
_define_flag(flags.DEFINE_string, "attention", "dot_product", "Attention kernel (dot_product or autoselected).")


def initialize_model_and_weights(config, checkpoint_path, mesh):
  """Instantiates the Omni model and loads checkpoint parameters.

  Args:
    config: The MaxText Omni configuration.
    checkpoint_path: Path to the checkpoint parameters directory.
    mesh: The JAX mesh for sharding.

  Returns:
    model: The stitched model.
    params: The restored model parameters.
  """
  with jax.set_mesh(mesh):
    model = model_creation_utils.from_config(config, mesh=mesh)

    abstract_vars = maxtext_utils.get_abstract_param(model, config)
    target_params_abstract = max_utils.unbox_logicallypartioned(abstract_vars["params"])

    max_logging.log(f"Loading checkpoint parameters from: {checkpoint_path}")
    restored = checkpointing.load_params_from_path(
        checkpoint_path,
        {"params": target_params_abstract},
        config.checkpoint_storage_concurrent_gb,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    return model, restored.get("params", restored)


def load_omni_config(yaml_path, checkpoint_path):
  """Loads custom omni config YAML and sets up runtime execution settings."""
  custom_cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(yaml_path), resolve=True)
  base_yml = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
  num_devs = len(jax.devices())
  per_dev_bs = 1.0 / num_devs

  argv = [
      sys.argv[0],
      base_yml,
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      "ici_fsdp_parallelism=1",
      "ici_data_parallelism=1",
      "ici_autoregressive_parallelism=1",
      "ici_tensor_parallelism=-1",
      f"attention={FLAGS.attention}",
      "scan_layers=false",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
      f"per_device_batch_size={per_dev_bs}",
      "async_checkpointing=False",
      f"load_parameters_path={checkpoint_path}",
  ]

  if "max_prefill_predict_length" not in custom_cfg:
    argv.append("max_prefill_predict_length=1024")
  if "max_target_length" not in custom_cfg:
    argv.append("max_target_length=2048")

  omni_skip_keys = {
      "vision_load_path",
      "llm_load_path",
      "stitched_output_path",
      "vision_model_name",
      "llm_model_name",
      "base_config",
      "model_name",
      "per_device_batch_size",
      "eval_per_device_batch_size",
      "ici_fsdp_parallelism",
      "ici_data_parallelism",
      "ici_tensor_parallelism",
      "ici_autoregressive_parallelism",
      "ici_expert_parallelism",
  }
  for k, v in custom_cfg.items():
    if k in omni_skip_keys:
      continue
    if isinstance(v, str):
      argv.append(f"{k}='{v}'")
    else:
      argv.append(f"{k}={v}")

  config = pyconfig.initialize(
      argv,
      override_model_config=True,
      skip_jax_distributed_system=True,
      log_config=False,
  )
  object.__setattr__(config, "model_name", "maxtext-omni-qwen3-vl-14b")
  object.__setattr__(config, "micro_batch_size_to_train_on", 1)
  object.__setattr__(config, "global_batch_size_to_train_on", 1)
  object.__setattr__(config, "per_device_batch_size", per_dev_bs)
  return config


@functools.partial(jax.jit, static_argnums=(0,))
def _prefill_step(model, params, tokens, positions, segment_ids, images):
  """Executes initial prefill pass and initializes the KV cache."""
  logits, mutated_vars = model.apply(
      {"params": params},
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      encoder_images=images,
      enable_dropout=False,
      model_mode="prefill",
      mutable=["cache"],
  )
  return logits, mutated_vars["cache"]


@functools.partial(jax.jit, static_argnums=(0,))
def _ar_step(model, params, cache, token, position):
  """Executes a single-token autoregressive step reusing the KV cache."""
  logits, mutated_vars = model.apply(
      {"params": params, "cache": cache},
      decoder_input_tokens=token,
      decoder_positions=position,
      decoder_segment_ids=None,
      encoder_images=None,
      enable_dropout=False,
      model_mode="autoregressive",
      mutable=["cache"],
  )
  return logits, mutated_vars["cache"]


@functools.partial(jax.jit, static_argnums=(0, 5))
def _generate_loop(model, params, cache, first_token, start_pos, num_steps: int):
  """Executes remaining autoregressive steps on TPU using jax.lax.scan without host syncs."""

  def step_fn(carry, _):
    current_cache, current_token, current_pos = carry
    logits, mutated_vars = model.apply(
        {"params": params, "cache": current_cache},
        decoder_input_tokens=current_token,
        decoder_positions=current_pos,
        decoder_segment_ids=None,
        encoder_images=None,
        enable_dropout=False,
        model_mode="autoregressive",
        mutable=["cache"],
    )
    next_token = jnp.argmax(logits[:, 0:1, :], axis=-1).astype(jnp.int32)  # shape: [1, 1]
    next_pos = current_pos + 1
    new_cache = mutated_vars["cache"]
    return (new_cache, next_token, next_pos), next_token

  init_carry = (cache, first_token, start_pos)
  _, generated_tokens_seq = jax.lax.scan(step_fn, init_carry, None, length=num_steps)
  return generated_tokens_seq


def decode_text_sample(model, params, config, mesh, tokenizer, prompt_str, max_new_tokens=128):
  """Runs pure text prefill and autoregressive decoding with KV caching."""
  if "<|im_start|>" not in prompt_str:
    formatted_prompt = f"<|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n"
  else:
    formatted_prompt = prompt_str
  max_logging.log(f"  [Formatted Text Prompt]: {repr(formatted_prompt)}")

  initial_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
  true_length = len(initial_tokens)
  prefill_len = config.max_prefill_predict_length
  if true_length > prefill_len:
    raise ValueError(
        f"Prompt length ({true_length}) exceeds config.max_prefill_predict_length ({prefill_len})."
    )

  pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  padded_tokens = initial_tokens + [pad_token] * (prefill_len - true_length)
  tokens = np.array([padded_tokens[:prefill_len]], dtype=np.int32)
  positions = np.tile(np.arange(prefill_len, dtype=np.int32), (1, 1))
  segment_ids = np.zeros((1, prefill_len), dtype=np.int32)
  segment_ids[:, :true_length] = 1

  eos_token_id = tokenizer.eos_token_id
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  tokens_to_decode = []

  with jax.set_mesh(mesh):
    logits, cache = _prefill_step(model, params, tokens, positions, segment_ids, None)
    first_token_id = int(jnp.argmax(logits[0, true_length - 1, :]))
    tokens_to_decode.append(first_token_id)

    if max_new_tokens > 1 and first_token_id not in (eos_token_id, im_end_id):
      first_token_arr = jnp.array([[first_token_id]], dtype=np.int32)
      first_pos_arr = jnp.array([[true_length]], dtype=np.int32)
      logits_1, ar_cache = _ar_step(model, params, cache, first_token_arr, first_pos_arr)
      second_token_id = int(jnp.argmax(logits_1[0, 0, :]))
      tokens_to_decode.append(second_token_id)

      num_scan_steps = min(max_new_tokens - 2, config.max_target_length - config.max_prefill_predict_length - 2)
      num_scan_steps = max(0, num_scan_steps)
      if num_scan_steps > 0 and second_token_id not in (eos_token_id, im_end_id):
        second_token_arr = jnp.array([[second_token_id]], dtype=np.int32)
        second_pos_arr = jnp.array([[true_length + 1]], dtype=np.int32)
        ar_tokens_seq = _generate_loop(model, params, ar_cache, second_token_arr, second_pos_arr, num_scan_steps)
        for tok in np.array(ar_tokens_seq).reshape(-1).tolist():
          if tok in (eos_token_id, im_end_id):
            break
          tokens_to_decode.append(tok)

  decoded_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True).strip()
  return decoded_text, tokens_to_decode


def decode_omni_sample(model, params, config, mesh, tokenizer, prompt_str, pil_image, max_new_tokens=128):
  """Runs prefill and fast autoregressive decoding for a single multimodal sample with KV cache."""
  # 1. Preprocess RGB image
  image_np = np.array(pil_image.convert("RGB"), dtype=np.uint8)
  processed_image = mm_processor.preprocess_image_for_training(image_np, config)
  image_pixels = (
      processed_image.pixel_values
      if hasattr(processed_image, "pixel_values") and processed_image.pixel_values is not None
      else processed_image
  )
  image_pixels = np.asarray(image_pixels)
  if image_pixels.ndim == 4:
    image_pixels = np.expand_dims(image_pixels, axis=0)
  mock_image = jnp.array(image_pixels, dtype=jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32)

  # 2. Format prompt and expand image placeholder tokens
  formatted_prompt = mm_processor.reformat_prompt(
      prompt=prompt_str,
      image_placeholder=config.image_placeholder,
      model_name=config,
      num_images=1,
  )
  initial_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
  combined_tokens = mm_processor.prepare_text_for_image_fusion(
      tokens=initial_tokens,
      config=config,
      processor_output=processed_image,
  ).tolist()

  # 3. Construct padded sequences to fixed config.max_prefill_predict_length
  true_length = len(combined_tokens)
  prefill_len = config.max_prefill_predict_length
  if true_length > prefill_len:
    raise ValueError(
        f"The combined length of expanded prompt and vision tokens ({true_length}) "
        f"exceeds config.max_prefill_predict_length ({prefill_len}). "
        "Please increase max_prefill_predict_length in your model config."
    )

  pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  padded_tokens = combined_tokens + [pad_token] * (prefill_len - true_length)
  tokens = np.array([padded_tokens[:prefill_len]], dtype=np.int32)
  positions = np.tile(np.arange(prefill_len, dtype=np.int32), (1, 1))
  segment_ids = np.zeros((1, prefill_len), dtype=np.int32)
  segment_ids[:, :true_length] = 1

  # 4. Autoregressive decoding with KV cache
  eos_token_id = tokenizer.eos_token_id
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  tokens_to_decode = []

  with jax.set_mesh(mesh):
    # Initial prefill pass (computes ViT embeddings and initializes KV cache)
    logits, cache = _prefill_step(model, params, tokens, positions, segment_ids, mock_image)
    first_token_id = int(jnp.argmax(logits[0, true_length - 1, :]))
    tokens_to_decode.append(first_token_id)

    if max_new_tokens > 1 and first_token_id not in (eos_token_id, im_end_id):
      first_token_arr = jnp.array([[first_token_id]], dtype=np.int32)
      first_pos_arr = jnp.array([[true_length]], dtype=np.int32)
      logits_1, ar_cache = _ar_step(model, params, cache, first_token_arr, first_pos_arr)
      second_token_id = int(jnp.argmax(logits_1[0, 0, :]))
      tokens_to_decode.append(second_token_id)

      num_scan_steps = min(max_new_tokens - 2, config.max_target_length - config.max_prefill_predict_length - 2)
      num_scan_steps = max(0, num_scan_steps)
      if num_scan_steps > 0 and second_token_id not in (eos_token_id, im_end_id):
        second_token_arr = jnp.array([[second_token_id]], dtype=np.int32)
        second_pos_arr = jnp.array([[true_length + 1]], dtype=np.int32)
        ar_tokens_seq = _generate_loop(model, params, ar_cache, second_token_arr, second_pos_arr, num_scan_steps)
        for tok in np.array(ar_tokens_seq).reshape(-1).tolist():
          if tok in (eos_token_id, im_end_id):
            break
          tokens_to_decode.append(tok)

  decoded_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True).strip()
  return decoded_text, tokens_to_decode


def run_evaluation(checkpoint_path, config, num_samples=3, description="Omni Model", max_new_tokens=128):
  """Runs evaluation on text prompts or ChartQA multimodal validation samples."""
  max_logging.log("=" * 70)
  max_logging.log(f">>> [DECODE EVALUATION] {description}")
  max_logging.log(f"    Checkpoint: {checkpoint_path}")
  max_logging.log(f"    Mode:       {'Text-Only' if FLAGS.text_only else 'Multimodal (Vision + Text)'}")
  max_logging.log("=" * 70)

  if jax.local_devices()[0].platform == "cpu":
    mesh = Mesh(np.array([jax.devices("cpu")[0]]), axis_names=("data",))
  else:
    mesh = maxtext_utils.get_mesh_from_config(config)

  model, restored_params = initialize_model_and_weights(config, checkpoint_path, mesh)
  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

  # Text-only evaluation mode
  if FLAGS.text_only:
    test_prompts = (
        [FLAGS.prompt]
        if FLAGS.prompt
        else [
            "What is the capital of France?",
            "Explain why the sky is blue in one sentence.",
            "What is 2 + 2?",
        ]
    )
    for idx, prompt_str in enumerate(test_prompts):
      max_logging.log(f"\n--- [Text Sample {idx+1}/{len(test_prompts)}] ---")
      response, token_ids = decode_text_sample(
          model=model,
          params=restored_params,
          config=config,
          mesh=mesh,
          tokenizer=tokenizer,
          prompt_str=prompt_str,
          max_new_tokens=max_new_tokens,
      )
      max_logging.log(
          f"""
----------------------------------------------------------------------
  Question:       {prompt_str}
  Generated IDs:  {token_ids}
  Model Response: {response}
----------------------------------------------------------------------"""
      )
    return

  # Custom multimodal prompt mode
  if FLAGS.prompt:
    if FLAGS.image_path and os.path.exists(FLAGS.image_path):
      img = Image.open(FLAGS.image_path)
    else:
      img = Image.fromarray(np.uint8(np.random.randint(0, 255, (768, 768, 3))))

    max_logging.log(f"Multimodal Prompt: {FLAGS.prompt}")
    response, token_ids = decode_omni_sample(
        model=model,
        params=restored_params,
        config=config,
        mesh=mesh,
        tokenizer=tokenizer,
        prompt_str=FLAGS.prompt,
        pil_image=img,
        max_new_tokens=max_new_tokens,
    )
    max_logging.log(f"Model Response (Tokens: {token_ids}):\n{response}")
    return

  # ChartQA dataset evaluation mode
  try:
    ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split="val")
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Error loading ChartQA evaluation dataset: {e}")
    return

  random.seed(42)
  total_samples = len(ds)
  sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

  for idx, i in enumerate(sample_indices):
    sample = ds[i]
    response, token_ids = decode_omni_sample(
        model=model,
        params=restored_params,
        config=config,
        mesh=mesh,
        tokenizer=tokenizer,
        prompt_str=f"<|image_pad|> {sample['query']}",
        pil_image=sample["image"],
        max_new_tokens=max_new_tokens,
    )

    max_logging.log(
        f"""
----------------------------------------------------------------------
[Sample {idx+1}/{len(sample_indices)} - Index {i}]
  Question:       {sample['query']}
  Ground Truth:   {sample['label']}
  Generated IDs:  {token_ids}
  Model Response: {response}
----------------------------------------------------------------------"""
    )


def main(argv):
  config_path = FLAGS.config_path
  checkpoint_path = FLAGS.checkpoint_path
  assert checkpoint_path, "Must specify --checkpoint_path"

  config = load_omni_config(config_path, checkpoint_path)

  run_evaluation(
      checkpoint_path=checkpoint_path,
      config=config,
      num_samples=FLAGS.num_samples,
      description=FLAGS.description,
      max_new_tokens=FLAGS.max_new_tokens,
  )


if __name__ == "__main__":
  app.run(main)

