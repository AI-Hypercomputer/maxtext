#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" Tests for multimodal vision encoder. """

import unittest
import os
from collections.abc import Callable

import pytest

import jsonlines

import numpy as np

import jax
import jax.numpy as jnp

from flax.core.scope import VariableDict

from MaxText import pyconfig
from MaxText import multimodal_utils
from MaxText.layers import models
from MaxText.globals import PKG_DIR
from MaxText import maxengine


# 4b with vit
DEFAULT_LOAD_PARAMETERS_PATH = "gs://maxtext-model-checkpoints/gemma3-4b/multimodal/2025-04-25-18-06-04/checkpoints/0/items"


class VisionEncoderEmbeddingTest(unittest.TestCase):

  CONFIGS = {
      "gemma3-4b": [  # tests decode with multimodal gemma-4b
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "model_name=gemma3-4b",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.gemma3')}",
          "use_multimodal=True",
          "run_name=runner_test",
          f"load_parameters_path={DEFAULT_LOAD_PARAMETERS_PATH}",
          "steps=1",
          "enable_checkpointing=False",
          "max_target_length=16",
          "max_prefill_predict_length=8",
          "per_device_batch_size=1",
          "scan_layers=false",
          "enable_checkpointing=true",
          "prompt='Describe this image'",
          rf"image_path={os.path.join(os.path.dirname(PKG_DIR), 'MaxText', 'test_assets', 'test_image.jpg')}",
          "skip_jax_distributed_system=True",
      ],
  }

  @pytest.mark.skip(reason="until b/416335384 is fixed")
  @pytest.mark.tpu_only
  def test_image_embedding_gemma3_4b_tpu(self):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    """Correctness test for the gemma3-4b image embedding."""
    # Load weights from reference checkpoint
    config = pyconfig.initialize(VisionEncoderEmbeddingTest.CONFIGS["gemma3-4b"])
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng_load_params)

    # Load and preprocess the image
    images = multimodal_utils.load_image_from_path(config.image_path)
    images = multimodal_utils.pre_process_image(images, model_name=config.model_name)
    input_images = images[jnp.newaxis, jnp.newaxis, ...]  # pytype: disable=unsupported-operands

    # Initialize only the vision encoder part and extract the corresponding params
    vision_encoder_model = models.VisionEncoder(config)
    vision_encoder_params = params["params"]["vision_encoder"]

    # Apply the vision encoder to get the image embeddings
    def apply_vision_encoder_fn(params, images_input):
      return vision_encoder_model.apply({"params": params}, images_input)

    jitted_apply_vision_encoder_fn: Callable[[VariableDict, tuple[dict, ...]], np.ndarray] = jax.jit(apply_vision_encoder_fn)
    image_embeddings = jitted_apply_vision_encoder_fn(vision_encoder_params, input_images)  # pylint: disable=not-callable

    # Load golden image embeddings generated from HuggingFace Gemma3-4b
    input_golden_data_path = os.path.join(PKG_DIR, "test_assets", "golden_data_gemma3_vit.jsonl")
    with jsonlines.open(input_golden_data_path, mode="r") as reader:
      loaded_data = next(iter(reader))
    golden_image_embeddings = np.asarray(loaded_data["soft_embeddings"], dtype=np.float32)

    # Compare the image embeddings with golden data
    mse = np.mean((image_embeddings - golden_image_embeddings) ** 2)
    print(f"MSE between image_embedding and golden data: {mse}")
    self.assertLess(mse, 1e-2, f"Image embedding mismatch with golden data, MSE {mse} exceeds threshold 1e-2")


if __name__ == "__main__":
  unittest.main()
