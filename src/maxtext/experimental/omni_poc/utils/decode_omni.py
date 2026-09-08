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

"""
Retrieve random examples from ChartQA dataset and evaluate the stitched model
(pre-SFT and post-SFT) checkpoints to verify if:
  1. the data flows end-to-end
  2. the response is reasonable given the random vision projector
ChartQA sample Q&A, ground truth and model responses are logged.

Example usage:

python src/maxtext/experimental/omni_poc/utils/decode_omni.py \
  --checkpoint_path="gs://YOUR_BUCKET_NAME/omni_stitched_gemma3-4b_qwen3-4b/0/items" \
  --num_samples=3 \
  --max_new_tokens=128

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


_define_flag(flags.DEFINE_string, "config_path", "", "Path to the config YAML file.")
_define_flag(flags.DEFINE_string, "checkpoint_path", "", "Path to checkpoint parameters directory.")
_define_flag(flags.DEFINE_integer, "num_samples", 5, "Number of random ChartQA validation samples to evaluate.")
_define_flag(flags.DEFINE_string, "description", "Omni Model", "Description for evaluation logs.")
_define_flag(flags.DEFINE_integer, "max_new_tokens", 128, "Maximum number of new tokens to generate.")


def initialize_model_and_weights(config, checkpoint_path, mesh):
  """Instantiates the Omni model and loads checkpoint.

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

    max_logging.log(f"Checkpoint parameters path: {checkpoint_path}")
    restored = checkpointing.load_params_from_path(
        checkpoint_path,
        {"params": target_params_abstract},
        config.checkpoint_storage_concurrent_gb,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    return model, restored.get("params", restored)


def load_omni_config(yaml_path, checkpoint_path):
  """Loads custom omni config YAML and converts overrides onto base.yml."""
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
      f"per_device_batch_size={per_dev_bs}",
      "async_checkpointing=False",
      f"load_parameters_path={checkpoint_path}",
  ]

  if "max_prefill_predict_length" not in custom_cfg:
    argv.append("max_prefill_predict_length=1024")
  if "max_target_length" not in custom_cfg:
    argv.append("max_target_length=2048")

  omni_skip_keys = {
      # Custom omni YAML keys not in base.yml
      "vision_load_path",
      "llm_load_path",
      "stitched_output_path",
      "vision_model_name",
      "llm_model_name",
      "base_config",
      "model_name",
      # Batch and parallelism keys overridden for single-sample decoding
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

  # Initialize config
  config = pyconfig.initialize(
      argv,
      override_model_config=True,
      skip_jax_distributed_system=True,
      log_config=False,
  )
  # Explicitly set model_name and single-sample batch sizes on the frozen config
  object.__setattr__(config, "model_name", "maxtext-omni-gemma3-qwen3")
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
  """Executes the remaining autoregressive generation loop on TPU using jax.lax.scan without host syncs."""

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


def decode_omni_sample(model, params, config, mesh, tokenizer, prompt_str, pil_image, max_new_tokens=128):
  """Runs prefill and autoregressive decoding for a single multimodal sample with KV caching."""
  # Preprocess image
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

  # Format prompt and expand image placeholder tokens
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

  # Pad token sequence to fixed config.max_prefill_predict_length
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

  # Autoregressive decoding with KV cache
  eos_token_id = tokenizer.eos_token_id
  tokens_to_decode = []

  with jax.set_mesh(mesh):
    # Initial prefill pass (computes ViT embeddings and initializes KV cache)
    logits, cache = _prefill_step(model, params, tokens, positions, segment_ids, mock_image)
    first_token_id = int(jnp.argmax(logits[0, true_length - 1, :]))
    tokens_to_decode.append(first_token_id)

    # Autoregressive generation
    if max_new_tokens > 1 and first_token_id != eos_token_id:
      # Step 1 of AR: transitions cache metadata from cache_batch_prefill to cache_batch
      first_token_arr = jnp.array([[first_token_id]], dtype=np.int32)
      first_pos_arr = jnp.array([[true_length]], dtype=np.int32)
      logits_1, ar_cache = _ar_step(model, params, cache, first_token_arr, first_pos_arr)
      second_token_id = int(jnp.argmax(logits_1[0, 0, :]))
      tokens_to_decode.append(second_token_id)

      # Remaining AR steps inside jax.lax.scan (all steps use cache_batch metadata)
      num_scan_steps = min(max_new_tokens - 2, config.max_target_length - config.max_prefill_predict_length - 2)
      num_scan_steps = max(0, num_scan_steps)
      if num_scan_steps > 0 and second_token_id != eos_token_id:
        second_token_arr = jnp.array([[second_token_id]], dtype=np.int32)
        second_pos_arr = jnp.array([[true_length + 1]], dtype=np.int32)
        ar_tokens_seq = _generate_loop(model, params, ar_cache, second_token_arr, second_pos_arr, num_scan_steps)
        for tok in np.array(ar_tokens_seq).reshape(-1).tolist():
          if tok == eos_token_id:
            break
          tokens_to_decode.append(tok)

  return tokenizer.decode(tokens_to_decode, skip_special_tokens=True).strip()


def run_evaluation(checkpoint_path, config, num_samples=5, description="Omni Model", max_new_tokens=128):
  """Runs evaluation on ChartQA random validation samples.
  Args:
    checkpoint_path: Path to the checkpoint parameters directory.
    config: The MaxText Omni configuration.
    num_samples: Number of random ChartQA validation samples to evaluate.
    description: Description for evaluation logs.
    max_new_tokens: Maximum number of new tokens to generate.
  """
  max_logging.log("=" * 60)
  max_logging.log(f"Running Evaluation for {description}...")
  max_logging.log(f"Loading checkpoint from: {checkpoint_path}")
  max_logging.log("=" * 60)

  # Load the dataset
  try:
    ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split="val")
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Error loading ChartQA evaluation dataset: {e}")
    return

  # Initialize the model
  if jax.local_devices()[0].platform == "cpu":
    mesh = Mesh(np.array([jax.devices("cpu")[0]]), axis_names=("data",))
  else:
    mesh = maxtext_utils.get_mesh_from_config(config)

  model, restored_params = initialize_model_and_weights(config, checkpoint_path, mesh)

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

  # Run decode on samples from the dataset
  random.seed(42)
  total_samples = len(ds)
  sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

  for idx, i in enumerate(sample_indices):
    sample = ds[i]
    model_response = decode_omni_sample(
        model=model,
        params=restored_params,
        config=config,
        mesh=mesh,
        tokenizer=tokenizer,
        prompt_str=f"<image> {sample['query']}",
        pil_image=sample["image"],
        max_new_tokens=max_new_tokens,
    )

    max_logging.log(
        f"""
Sample {idx+1}/{len(sample_indices)} (Index {i}):
  Question: {sample['query']}
  Ground Truth: {sample['label']}
  Model Response: {model_response}
"""
    )


def main(argv):
  config_path = FLAGS.config_path or os.path.join(
      MAXTEXT_PKG_DIR, "experimental", "omni_poc", "maxtext-omni-gemma3-qwen3.yml"
  )
  assert FLAGS.checkpoint_path, "Must specify --checkpoint_path"
  config = load_omni_config(config_path, FLAGS.checkpoint_path)

  run_evaluation(
      checkpoint_path=FLAGS.checkpoint_path,
      config=config,
      num_samples=FLAGS.num_samples,
      description=FLAGS.description,
      max_new_tokens=FLAGS.max_new_tokens,
  )


if __name__ == "__main__":
  app.run(main)
