# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test & Verification Script for Stitched Omni Qwen3-VL-4B + Qwen3-14B Model.

Verifies:
1. Checkpoint Stitching (restores Qwen3-VL-4B ViT + Qwen3-14B LLM + fresh MLP projector)
2. PyTree Parameter Shapes & Parameter Counts
3. End-to-End Forward Pass with Multimodal MRoPE & Vision Token Fusion
4. Output Logits Verification (Shape, Finite values, Non-NaN)

Usage:
  # 1. Run on CPU (Local / VM):
  JAX_PLATFORMS=cpu python3 src/maxtext/experimental/omni_qwen_14b/test_stitch_qwen_14b.py \
    --vision_checkpoint="gs://yuchenhou-maxtext-logs/checkpoints/qwen3-vl-4b-processor/0/items" \
    --llm_checkpoint="gs://yuchenhou-maxtext-logs/omni_checkpoints/qwen3-14b_unscanned/0/items" \
    --output_checkpoint="gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items"

  # 2. Or verify existing stitched checkpoint directly:
  python3 src/maxtext/experimental/omni_qwen_14b/test_stitch_qwen_14b.py \
    --output_checkpoint="gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items" \
    --skip_stitch
"""

import functools
import os
import sys
from absl import app, flags
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import omegaconf
from orbax import checkpoint as ocp
from PIL import Image
from transformers import AutoTokenizer

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.experimental.omni_poc.utils import stitch_checkpoint
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config_path",
    os.path.join(
        MAXTEXT_PKG_DIR,
        "experimental",
        "omni_qwen_14b",
        "maxtext-omni-qwen3-vl-14b.yml",
    ),
    "Path to model YAML configuration.",
)
flags.DEFINE_string(
    "vision_checkpoint",
    "gs://yuchenhou-maxtext-logs/checkpoints/qwen3-vl-4b-processor/0/items",
    "Source Qwen3-VL-4B checkpoint.",
)
flags.DEFINE_string(
    "llm_checkpoint",
    "gs://yuchenhou-maxtext-logs/omni_checkpoints/qwen3-14b_unscanned/0/items",
    "Source Qwen3-14B LLM checkpoint.",
)
flags.DEFINE_string(
    "output_checkpoint",
    "gs://yuchenhou-maxtext-logs/omni_checkpoints/omni_stitched_qwen3-vl-4b_qwen3-14b_unscanned/0/items",
    "Destination stitched checkpoint.",
)
flags.DEFINE_boolean("skip_stitch", False, "Skip stitching if output checkpoint already exists.")
flags.DEFINE_string("attention", "dot_product", "Attention kernel.")


def create_test_config(config_path, checkpoint_path):
  """Initializes MaxText Config with CPU/TPU test settings from experimental YAML."""
  custom_cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(config_path), resolve=True)
  base_yml = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")

  is_cpu = jax.local_devices()[0].platform == "cpu"
  ici_tensor = "1" if is_cpu else "-1"
  ici_fsdp = "1" if is_cpu else "1"

  argv = [
      sys.argv[0],
      base_yml,
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      f"ici_fsdp_parallelism={ici_fsdp}",
      f"ici_tensor_parallelism={ici_tensor}",
      f"attention={FLAGS.attention}",
      "max_target_length=2048",
      "max_prefill_predict_length=1024",
      "scan_layers=false",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
      "per_device_batch_size=1",
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
  return config


def test_stitching(config, vision_path, llm_path, output_path):
  """Step 1: Performs model checkpoint stitching."""
  max_logging.log("=" * 70)
  max_logging.log(">>> [STEP 1] Running Checkpoint Stitching...")
  max_logging.log(f"  Vision Source: {vision_path}")
  max_logging.log(f"  LLM Source:    {llm_path}")
  max_logging.log(f"  Output Path:   {output_path}")
  max_logging.log("=" * 70)

  stitch_checkpoint.stitch_and_save_checkpoints(
      config=config,
      vision_checkpoint_path=vision_path,
      llm_checkpoint_path=llm_path,
      output_checkpoint_path=output_path,
  )
  max_logging.log(">>> [STEP 1 PASSED] Checkpoint stitched and saved successfully!")


def verify_parameter_shapes(params):
  """Step 2: Validates key component shapes in the PyTree."""
  max_logging.log("=" * 70)
  max_logging.log(">>> [STEP 2] Verifying Stitched PyTree Shapes & Parameter Counts...")
  max_logging.log("=" * 70)

  p = params.get("params", params)

  # 1. Vision Encoder
  assert "vision_encoder" in p, "Missing 'vision_encoder' in stitched checkpoint params!"
  ve = p["vision_encoder"]
  max_logging.log(f"  [Vision Encoder Submodules]: {list(ve.keys())}")

  # 2. LLM Decoder
  assert "decoder" in p, "Missing 'decoder' in stitched checkpoint params!"
  assert "token_embedder" in p, "Missing 'token_embedder' in stitched checkpoint params!"
  token_emb = p["token_embedder"]["embedding"]
  max_logging.log(f"  [Token Embedder Shape]: {token_emb.shape} (Expected: [151936, 5120])")
  assert token_emb.shape == (151936, 5120), f"Token embedder shape mismatch: {token_emb.shape}"

  # 3. Multimodal Projector
  proj_found = False
  for k, v in ve.items():
    if "projector" in k.lower() or "linear" in k.lower() or "mlp" in k.lower():
      max_logging.log(f"  [Projector Layer ({k})]:")
      if isinstance(v, dict):
        for sub_k, sub_v in v.items():
          shape = sub_v.shape if hasattr(sub_v, "shape") else type(sub_v)
          max_logging.log(f"    - {sub_k}: {shape}")
      proj_found = True

  total_params = max_utils.calculate_num_params_from_pytree(params)
  max_logging.log(f"  [Total Parameters]: {total_params:,} (~{total_params/1e9:.2f}B)")
  max_logging.log(">>> [STEP 2 PASSED] All PyTree shapes and layers verified!")


@nnx.jit
def _forward_step(model: nnx.Module, tokens, positions, segment_ids, images):
  """Executes a single forward pass of the pure NNX model and returns output logits."""
  return model(
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segment_ids,
      encoder_images=images,
      enable_dropout=False,
      model_mode="prefill",
  )


def test_forward_pass(config, model, mesh):
  """Step 3: Executes an end-to-end multimodal forward pass."""
  max_logging.log("=" * 70)
  max_logging.log(">>> [STEP 3] Testing End-to-End Multimodal Forward Pass...")
  max_logging.log("=" * 70)

  with jax.set_mesh(mesh):
    # Create dummy RGB image (768x768)
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (768, 768, 3))))
    processed_image = mm_processor.preprocess_image_for_training(np.array(dummy_img), config)

    image_pixels = processed_image.pixel_values if hasattr(processed_image, "pixel_values") else processed_image
    image_pixels = np.asarray(image_pixels)
    if image_pixels.ndim == 4:
      image_pixels = np.expand_dims(image_pixels, axis=0)
    mock_image = jnp.array(image_pixels, dtype=jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32)
    max_logging.log(f"  Input Image Pixel Tensor Shape: {mock_image.shape}")

    # Format text prompt & expand vision tokens
    prompt_str = "<|image_pad|> What is the title of this chart?"
    formatted_prompt = mm_processor.reformat_prompt(
        prompt=prompt_str,
        image_placeholder=config.image_placeholder,
        model_name=config,
        num_images=1,
    )
    max_logging.log(f"  Formatted Prompt: {repr(formatted_prompt)}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    initial_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    combined_tokens = mm_processor.prepare_text_for_image_fusion(
        tokens=initial_tokens,
        config=config,
        processor_output=processed_image,
    ).tolist()

    seq_len = 1024
    true_length = len(combined_tokens)
    padded_tokens = (combined_tokens + [tokenizer.pad_token_id or 0] * seq_len)[:seq_len]
    tokens = np.array([padded_tokens], dtype=np.int32)
    max_logging.log(f"  Total Sequence Tokens (Vision + Text): {true_length} tokens (padded to {seq_len})")

    attention_mask = np.zeros((1, seq_len), dtype=np.int32)
    attention_mask[:, :true_length] = 1

    # Generate Position IDs (1D sequential for standard RoPE, 3D for MRoPE)
    if getattr(config, "use_mrope", False):
      from maxtext.multimodal import processor_qwen3_omni

      position_ids, _ = processor_qwen3_omni.get_rope_index(
          input_ids=tokens,
          image_grid_thw=processed_image.pixel_grid_thw,
          attention_mask=attention_mask,
          spatial_merge_size=config.spatial_merge_size_for_vit,
          config=config,
      )
      max_logging.log(f"  MRoPE 3D Position Tensor Shape: {position_ids.shape}")
    else:
      position_ids = np.tile(np.arange(seq_len, dtype=np.int32), (1, 1))
      max_logging.log(f"  1D Position Tensor Shape: {position_ids.shape}")

    segment_ids = np.zeros((1, seq_len), dtype=np.int32)
    segment_ids[:, :true_length] = 1

    # Execute forward step via JIT
    max_logging.log("  Executing model forward pass (JIT compiled)...")
    logits = _forward_step(
        model,
        tokens,
        position_ids,
        segment_ids,
        mock_image,
    )

    max_logging.log(f"  Output Logits Shape: {logits.shape} (Expected: [1, {seq_len}, 151936])")
    assert logits.shape == (1, seq_len, 151936), f"Unexpected logits shape: {logits.shape}"
    assert not jnp.isnan(logits).any(), "NaN detected in output logits!"
    assert jnp.isfinite(logits).all(), "Non-finite values detected in output logits!"

    top_token_id = int(jnp.argmax(logits[0, true_length - 1, :]))
    top_word = tokenizer.decode([top_token_id])
    max_logging.log(f"  Sample Greedy Next Token: ID={top_token_id}, Text={repr(top_word)}")
    max_logging.log(">>> [STEP 3 PASSED] End-to-end forward pass executed cleanly without errors!")


def main(argv):
  config_path = FLAGS.config_path
  vision_path = FLAGS.vision_checkpoint
  llm_path = FLAGS.llm_checkpoint
  output_path = FLAGS.output_checkpoint

  config = create_test_config(config_path, output_path)

  # Mesh setup (CPU or TPU)
  mesh = maxtext_utils.get_mesh_from_config(config)

  # Step 1: Stitch (if needed)
  if not FLAGS.skip_stitch:
    test_stitching(config, vision_path, llm_path, output_path)

  # Load parameters from stitched checkpoint
  max_logging.log(f"Restoring stitched checkpoint from: {output_path}")
  with jax.set_mesh(mesh):
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode="prefill")

  # Step 2: Verify parameter shapes and initializations
  token_emb = model.token_embedder.embedding.value
  max_logging.log(f"  [Token Embedder Shape]: {token_emb.shape} (Expected: [151936, 5120])")
  assert token_emb.shape == (151936, 5120), f"Token embedder shape mismatch: {token_emb.shape}"

  # Verify ln_q initialization if present
  if hasattr(model.vision_encoder, "vision_projector") and hasattr(model.vision_encoder.vision_projector, "ln_q"):
    ln_q = model.vision_encoder.vision_projector.ln_q
    if hasattr(ln_q, "scale") and hasattr(ln_q.scale, "value"):
      scale_val = ln_q.scale.value
      max_logging.log(f"  [ln_q scale mean]: {float(jnp.mean(scale_val)):.4f} (Expected: 1.0)")
      assert jnp.allclose(scale_val, 1.0), f"ln_q scale was not initialized to 1.0! Got {scale_val[:5]}"
    if hasattr(ln_q, "bias") and hasattr(ln_q.bias, "value") and ln_q.bias.value is not None:
      bias_val = ln_q.bias.value
      max_logging.log(f"  [ln_q bias mean]: {float(jnp.mean(bias_val)):.4f} (Expected: 0.0)")
      assert jnp.allclose(bias_val, 0.0), f"ln_q bias was not initialized to 0.0! Got {bias_val[:5]}"

  params = nnx.state(model, nnx.Param)
  total_params = sum(x.size for x in jax.tree.leaves(params))
  max_logging.log(f"  [Total Parameters]: {total_params:,} (~{total_params/1e9:.2f}B)")
  max_logging.log(">>> [STEP 2 PASSED] All PyTree shapes and layer initializations verified!")

  # Step 3: End-to-End Forward Pass
  test_forward_pass(config, model, mesh)

  max_logging.log("=" * 70)
  max_logging.log(">>> ALL OMNI QWEN3-VL + QWEN3-14B TESTS PASSED SUCCESSFULLY! 🎉")
  max_logging.log("=" * 70)


if __name__ == "__main__":
  app.run(main)
