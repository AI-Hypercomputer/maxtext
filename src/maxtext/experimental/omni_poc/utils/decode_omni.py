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
  argv = [
      sys.argv[0],
      base_yml,
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      "ici_fsdp_parallelism=1",
      "ici_tensor_parallelism=-1",
      f"load_parameters_path={checkpoint_path}",
  ]

  omni_skip_keys = {
      "vision_load_path",
      "llm_load_path",
      "stitched_output_path",
      "vision_model_name",
      "llm_model_name",
      "base_config",
      "model_name",
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
  # Explicitly set model_name on the frozen config for multimodal routing in processor.py.
  # This prevents pyconfig search errors or inheriting conflicting base model hyperparameters.
  # TODO: works for experimental piloting. For production, relocate model config to the
  # main folder, remove this line so pyconfig can initialize model_name directly.
  object.__setattr__(config, "model_name", "omni-gemma3-qwen3")
  return config


@functools.partial(jax.jit, static_argnums=(0,))
def _forward_step(model, params, tokens, positions, segment_ids, images):
  """Executes a single forward pass of the model and returns output logits."""
  return model.apply(
      {"params": params},
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      encoder_images=images,
      enable_dropout=False,
      model_mode="prefill",
  )


def decode_omni_sample(model, params, config, mesh, tokenizer, prompt_str, pil_image, max_new_tokens=128):
  """Runs prefill and autoregressive decoding for a single multimodal sample."""
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

  # Pad token sequence to fixed config.max_target_length
  true_length = len(combined_tokens)
  seq_len = config.max_target_length
  if true_length > seq_len:
    raise ValueError(
        f"The combined length of expanded prompt and vision tokens ({true_length}) "
        f"exceeds config.max_target_length ({seq_len}). "
        "Please increase max_target_length in your model config."
    )
  pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
  padded_tokens = combined_tokens + [pad_token] * (seq_len - true_length)

  tokens = np.array([padded_tokens[:seq_len]], dtype=np.int32)
  positions = np.tile(np.arange(seq_len, dtype=np.int32), (1, 1))
  segment_ids = np.zeros((1, seq_len), dtype=np.int32)
  segment_ids[:, :true_length] = 1

  # Autoregressive decoding loop
  generated_tokens = []
  curr_len = true_length

  with jax.set_mesh(mesh):
    # Initial prefill pass
    logits = _forward_step(model, params, tokens, positions, segment_ids, mock_image)

    for _ in range(max_new_tokens):
      next_token_id = int(jnp.argmax(logits[0, curr_len - 1, :]))
      if next_token_id == tokenizer.eos_token_id:
        break
      generated_tokens.append(next_token_id)
      if curr_len >= seq_len:
        break
      tokens[0, curr_len] = next_token_id
      curr_len += 1

      # Update segment IDs and positions and do the next autoregressive step
      segment_ids[0, :curr_len] = 1
      logits = _forward_step(model, params, tokens, positions, segment_ids, mock_image)

  return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


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
Sample {idx+1} (Index {i}):
  Question: {sample['query']}
  Ground Truth: {sample['label']}
  Model Response: {model_response}
"""
    )


def main(argv):
  config_path = FLAGS.config_path or os.path.join(MAXTEXT_PKG_DIR, "experimental", "omni_poc", "omni-gemma3-qwen3.yml")
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
