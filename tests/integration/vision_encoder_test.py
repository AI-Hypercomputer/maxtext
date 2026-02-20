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

""" Tests for multimodal vision encoder. """

from collections.abc import Callable
import os
import unittest

from flax.core.scope import VariableDict
import jax
import jax.numpy as jnp
import jsonlines
from MaxText import maxengine
from MaxText import pyconfig
from MaxText.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_TEST_ASSETS_ROOT
from maxtext.models import models
from maxtext.multimodal import processor_gemma3
from maxtext.multimodal import utils as mm_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest

pytestmark = [pytest.mark.external_serving, pytest.mark.integration_test]

# 4b with vit
DEFAULT_LOAD_PARAMETERS_PATH = (
    "gs://maxtext-model-checkpoints/gemma3-4b/multimodal/2025-04-25-18-06-04/checkpoints/0/items"
)


class VisionEncoderEmbeddingTest(unittest.TestCase):

  CONFIGS = {
      "gemma3-4b": [  # tests decode with multimodal gemma-4b
          None,
          get_test_config_path(),
          "model_name=gemma3-4b",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.gemma3')}",
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
          rf"image_path={os.path.join(MAXTEXT_TEST_ASSETS_ROOT, 'test_image.jpg')}",
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
    images = mm_utils.load_image_from_path(config.image_path)
    images = processor_gemma3.preprocess_mm_data_gemma3(images).pixel_values
    input_images = images[jnp.newaxis, jnp.newaxis, ...]  # pytype: disable=unsupported-operands

    # Initialize only the vision encoder part and extract the corresponding params
    vision_encoder_model = models.VisionEncoder(config, engine.mesh, rngs=engine.rng)
    vision_encoder_params = params["params"]["vision_encoder"]

    # Apply the vision encoder to get the image embeddings
    def apply_vision_encoder_fn(params, images_input):
      return vision_encoder_model.apply({"params": params}, images_input)

    jitted_apply_vision_encoder_fn: Callable[[VariableDict, tuple[dict, ...]], np.ndarray] = jax.jit(
        apply_vision_encoder_fn
    )
    image_embeddings = jitted_apply_vision_encoder_fn(vision_encoder_params, input_images)  # pylint: disable=not-callable

    # Load golden image embeddings generated from HuggingFace Gemma3-4b
    input_golden_data_path = os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "golden_logits", "golden_data_gemma3_vit.jsonl")
    with jsonlines.open(input_golden_data_path, mode="r") as reader:
      loaded_data = next(iter(reader))
    golden_image_embeddings = np.asarray(loaded_data["soft_embeddings"], dtype=np.float32)

    # Compare the image embeddings with golden data
    mse = np.mean((image_embeddings - golden_image_embeddings) ** 2)
    print(f"MSE between image_embedding and golden data: {mse}")
    self.assertLess(mse, 1e-2, f"Image embedding mismatch with golden data, MSE {mse} exceeds threshold 1e-2")


if __name__ == "__main__":
  unittest.main()
