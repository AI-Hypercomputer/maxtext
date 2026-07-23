"""Tests for custom vision projector implementation in MaxText."""

import argparse
import gc
import os
import sys

from flax import nnx
import jax
import jax.numpy as jnp

# Ensure the parent directory is in sys.path so we can import maxtext
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.multimodal import processor as mm_processor


def test_load_custom_vision_projector(hf_access_token, param_path):
  base_yml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/base.yml"))

  config = pyconfig.initialize(
      base_config=base_yml_path,
      model_name="qwen3-vl-2b",
      tokenizer_path="Qwen/Qwen3-VL-2B-Instruct",
      tokenizer_type="huggingface",
      load_parameters_path=param_path,
      per_device_batch_size=1,
      scan_layers=False,
      use_multimodal=True,
      prompt="Describe this image",
      image_path="tests/assets/test_image.jpg",
      max_prefill_predict_length=512,
      max_target_length=768,
      ici_tensor_parallelism=4,
      override_model_config=True,
      attention="dot_product",
      hf_access_token=hf_access_token,
      vision_projector_type="customized_mlp",
      vision_connector_num_layers=2,
      vision_connector_activation="gelu",
      vision_connector_use_bias=True,
  )

  print("Customized model")
  print(f"Active Vision Projector Type: {config.vision_projector_type}")
  print(f"Num Layers: {config.vision_connector_num_layers}")

  engine = maxengine.MaxEngine(config)

  # Load the parameters from the checkpoint
  rng = jax.random.PRNGKey(1234)
  params = engine.load_params(rng=rng)
  nnx.update(engine.model, params)

  # Preprocess the example image using the provided image_path in the config
  processor_outputs = mm_processor.preprocess_mm_data(config)

  images = processor_outputs.pixel_values
  print(f"Image pixel_values shape: {images.shape}")

  # Extract the vision wrapper and its underlying parts
  vision_wrapper = engine.model.vision_encoder
  pure_encoder = getattr(vision_wrapper, vision_wrapper.encoder_name)
  projector = getattr(vision_wrapper, vision_wrapper.projector_name)

  # Pass the image through the pure vision tower (before projector)
  encoder_output = pure_encoder(images)

  vision_embeddings_customized = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output

  print("vision embeddings:", vision_embeddings_customized.shape)
  print(f"First 10 values:\n{vision_embeddings_customized.flatten()[:10]}")

  # Pass the image through the vision tower (including projector)
  projected_embeddings_customized = projector(vision_embeddings_customized)

  print(f"projector_embeddings: {projected_embeddings_customized.shape}")
  print(f"First 10 values:\n{projected_embeddings_customized.flatten()[:10]}")

  del engine, params, vision_wrapper, pure_encoder, projector
  gc.collect()

  print("\nOriginal model")
  config_original = pyconfig.initialize(
      base_config=base_yml_path,
      model_name="qwen3-vl-2b",
      tokenizer_path="Qwen/Qwen3-VL-2B-Instruct",
      tokenizer_type="huggingface",
      load_parameters_path=param_path,
      per_device_batch_size=1,
      scan_layers=False,
      use_multimodal=True,
      prompt="Describe this image",
      image_path="tests/assets/test_image.jpg",
      max_prefill_predict_length=512,
      max_target_length=768,
      ici_tensor_parallelism=4,
      override_model_config=True,
      attention="dot_product",
      hf_access_token=hf_access_token,
  )

  print(f"Original Vision Projector Type: {getattr(config_original, 'vision_projector_type', None)}")

  engine_original = maxengine.MaxEngine(config_original)

  params_original = engine_original.load_params(rng=rng)
  nnx.update(engine_original.model, params_original)

  processor_outputs_original = mm_processor.preprocess_mm_data(config_original)
  images_original = processor_outputs_original.pixel_values

  vision_wrapper_original = engine_original.model.vision_encoder
  pure_encoder_original = getattr(vision_wrapper_original, vision_wrapper_original.encoder_name)
  projector_original = getattr(vision_wrapper_original, vision_wrapper_original.projector_name)

  encoder_output_original = pure_encoder_original(images_original)

  vision_embeddings_original = (
      encoder_output_original[0] if isinstance(encoder_output_original, tuple) else encoder_output_original
  )

  print("Original vision embeddings:", vision_embeddings_original.shape)
  print(f"First 10 values:\n{vision_embeddings_original.flatten()[:10]}")

  projected_embeddings_original = projector_original(vision_embeddings_original)

  print(f"Original projector_embeddings: {projected_embeddings_original.shape}")
  print(f"First 10 values:\n{projected_embeddings_original.flatten()[:10]}")

  print("\n===================================")
  print("Comparing customized vs original embeddings")
  vision_embeddings_match = jnp.allclose(vision_embeddings_customized, vision_embeddings_original, atol=1e-5)
  vision_max_diff = jnp.max(jnp.abs(vision_embeddings_customized - vision_embeddings_original))
  print(f"Vision embeddings match: {vision_embeddings_match} (Max diff: {vision_max_diff})")

  projector_embeddings_match = jnp.allclose(projected_embeddings_customized, projected_embeddings_original, atol=1e-5)
  projector_max_diff = jnp.max(jnp.abs(projected_embeddings_customized - projected_embeddings_original))
  print(f"Projector embeddings match: {projector_embeddings_match} (Max diff: {projector_max_diff})")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_parameters_path", required=True)
  parser.add_argument("--hf_access_token", default=os.getenv("HF_TOKEN", ""))

  args = parser.parse_args()
  test_load_custom_vision_projector(
      hf_access_token=args.hf_access_token,
      param_path=args.load_parameters_path,
  )
