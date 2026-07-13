"""Unit tests for checkpoint stitching of Vision and LLM backbones.

This test suite verifies the correctness of `stitch.py` by
stitching separate Vision (e.g., Gemma 3) and LLM (e.g., Qwen 3) checkpoints
into a combined multimodal model checkpoint. 

Two test cases:

1. Model layer count matching:
   Validates that the stitched checkpoint's structural layer counts for
   both the restored vision encoder blocks and LLM blocks exactly
   match the counts from the original source checkpoints.

2. Forward logits comparison:
   Performs forward passes on individual model components (Vision Tower,
   Token Embedder, and Projector) and compares output representations between
   weights restored from the original checkpoints vs. the stitched checkpoint.
   It asserts that:
   - Vision encoder outputs match exactly;
   - Text embedder outputs match exactly;
   - Projector outputs differ (since the new projector is
     intentionally initialized with random weights).
"""

import gc
import os
import unittest
import warnings

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import omegaconf
from orbax import checkpoint as ocp

from maxtext.configs import pyconfig
from maxtext.experimental.omni_poc.checkpoint_stitcher import stitch
from maxtext.layers.embeddings import Embed
from maxtext.models import gemma3
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _unwrap_params(restored_dict):
  """Fully unwrap any top-level 'params' / 'base' dictionary wrappers."""
  while isinstance(restored_dict, dict) and len(restored_dict) == 1:
    if "params" in restored_dict:
      restored_dict = restored_dict["params"]
    elif "base" in restored_dict:
      restored_dict = restored_dict["base"]
    else:
      break
  return restored_dict


def _get_vision_layer_count(vision_state):
  """Counts unique encoder block names within the vision encoder state."""
  blocks = set()

  def recurse(d):
    if isinstance(d, dict):
      for k, v in d.items():
        if k.startswith("encoderblock_"):
          blocks.add(k)
        recurse(v)

  recurse(vision_state)
  return len(blocks)


class TestOmniCheckpointStitcher(unittest.TestCase):
  """Unit tests for Omni checkpoint stitching logic.

  Verifies correct weight assembly and projector alignment on full models.
  """

  def setUp(self):
    super().setUp()
    # Read paths from environment variables or default to standard GCS bucket
    default_dir = "gs://yuchenhou-maxtext-logs/omni_checkpoints"
    base_dir = os.environ.get("OMNI_TEST_BASE_DIR", default_dir)
    self.vision_ckpt_dir = f"{base_dir}/gemma3-4b_converted/0/items"
    self.llm_ckpt_dir = f"{base_dir}/qwen3-4b_converted/0/items"
    self.output_ckpt_dir = f"{base_dir}/omni_stitched_gemma3-4b_qwen3-4b/0/items"

    # Run tests on CPU mesh to avoid running out of TPU device memory
    cpu_device = jax.devices("cpu")[0]
    self.test_mesh = Mesh(np.array([cpu_device]), axis_names=("data",))

  def tearDown(self):
    super().tearDown()
    gc.collect()

  def _create_config(self):
    """Creates a MaxText config from the Omni config file and defaults."""
    omni_config_path = os.path.join(
        MAXTEXT_REPO_ROOT,
        "src",
        "maxtext",
        "experimental",
        "omni_poc",
        "omni-gemma3-qwen3.yml",
    )
    custom_cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(omni_config_path), resolve=True)

    # Filter out safeguard/stitching-specific keys
    omni_keys = {
        "vision_load_path",
        "llm_load_path",
        "stitched_output_path",
        "vision_model_name",
        "llm_model_name",
    }
    yaml_overrides = {k: v for k, v in custom_cfg.items() if k not in omni_keys}

    base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    config = pyconfig.initialize(
        ["", base_config_path],
        override_model_config=True,
        skip_jax_distributed_system=True,
        log_config=False,
        pure_nnx=False,
        attention="dot_product",
        dtype="float32",
        **yaml_overrides,
    )
    # Use object.__setattr__ to bypass the read-only check on _HyperParameters
    object.__setattr__(config, "vision_model_name", "gemma3-4b")
    object.__setattr__(config, "llm_model_name", "qwen3-4b")
    object.__setattr__(config, "ici_context_autoregressive_parallelism", 1)
    return config

  def _ensure_stitched_checkpoint(self):
    """Runs stitching if the stitched output checkpoint does not exist."""
    if not epath.Path(self.output_ckpt_dir).exists():
      print(f"Stitched checkpoint not found at {self.output_ckpt_dir}. " "Running stitching process...")
      config = self._create_config()
      stitch.stitch_and_save_checkpoints(
          config=config,
          vision_checkpoint_path=self.vision_ckpt_dir,
          llm_checkpoint_path=self.llm_ckpt_dir,
          output_checkpoint_path=self.output_ckpt_dir,
      )

  def _restore_checkpoints(self, config):
    """Restores stitched, vision, and LLM checkpoints under the mesh."""
    # pylint: disable=protected-access
    ckptr = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
            save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
        )
    )

    with jax.set_mesh(self.test_mesh):
      # Create the target model structure
      Linen_model = model_creation_utils.from_config(config, mesh=self.test_mesh)

      # Get model info (shapes, data types, and sharding layouts)
      abstract_vars = maxtext_utils.get_abstract_param(Linen_model, config)
      target_params_abstract = abstract_vars["params"]

      # Extract the raw PyTree of ShapeDtypeStruct nodes
      target_params_abstract = max_utils.unbox_logicallypartioned(target_params_abstract)

      # Create a concrete "zero-filled template"
      # Orbax checkpointer requires target JAX arrays to know what shapes and
      # memory allocations to load checkpoints into. We map the abstract
      # shape/dtype structs to concrete zero arrays on our mesh.
      def to_concrete_zeros(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
          return jnp.zeros(leaf.shape, dtype=leaf.dtype)
        return leaf

      concrete_template = jax.tree.map(to_concrete_zeros, target_params_abstract)

      # Load weights from the STITCHED checkpoint path
      stitched_inner = stitch._restore_subtrees_from_path(
          self.output_ckpt_dir, concrete_template, ckptr
      )

      # Load weights from the ORIGINAL Vision checkpoint
      vision_abstract = concrete_template["vision_encoder"]
      vision_restored = stitch._restore_subtrees_from_path(
          self.vision_ckpt_dir, {"vision_encoder": vision_abstract}, ckptr
      )

      # Load weights from the ORIGINAL LLM checkpoint
      llm_abstract = {
          "decoder": concrete_template["decoder"],
          "token_embedder": concrete_template["token_embedder"],
      }
      llm_restored = stitch._restore_subtrees_from_path(self.llm_ckpt_dir, llm_abstract, ckptr)
    return (
        stitched_inner,
        vision_restored,
        llm_restored,
        Linen_model,
        concrete_template,
    )

  def test_1_stitch_and_assemble_correctness(self):
    """Test 1: Verifies model component layer counts match."""
    self._ensure_stitched_checkpoint()
    config = self._create_config()

    (
        stitched_inner,
        vision_restored,
        llm_restored,
        Linen_model,
        concrete_template,
    ) = self._restore_checkpoints(config)

    # Verify layer counts
    orig_decoder_layers = llm_restored["decoder"]["layers"]["mlp"]["wo"]["kernel"].shape[1]
    stitched_decoder_layers = stitched_inner["decoder"]["layers"]["mlp"]["wo"]["kernel"].shape[1]
    self.assertEqual(stitched_decoder_layers, orig_decoder_layers)

    orig_vision_layers = _get_vision_layer_count(vision_restored["vision_encoder"]["Gemma3VisionEncoderLayer_0"])
    stitched_vision_layers = _get_vision_layer_count(stitched_inner["vision_encoder"]["Gemma3VisionEncoderLayer_0"])
    self.assertEqual(stitched_vision_layers, orig_vision_layers)

    print("\n" + "=" * 80)
    print("TEST 1: MODEL LAYER COUNT MATCHING")
    print("=" * 80)
    print(
        f"  - Verified LLM layers count ({config.llm_model_name}): "
        f"orig={orig_decoder_layers} and stitched={stitched_decoder_layers}"
    )
    print(
        f"  - Verified Vision layers count ({config.vision_model_name}): "
        f"orig={orig_vision_layers} and stitched={stitched_vision_layers}"
    )
    print("=" * 80 + "\n")

    del (
        stitched_inner,
        vision_restored,
        llm_restored,
        Linen_model,
        concrete_template,
    )
    gc.collect()

  def test_2_stage_outputs_original_vs_stitched(self):
    """Test 2: Verifies output logits match original components."""
    self._ensure_stitched_checkpoint()
    config = self._create_config()
    rngs = nnx.Rngs(0)

    (
        stitched_inner,
        vision_restored,
        llm_restored,
        Linen_model,
        concrete_template,
    ) = self._restore_checkpoints(config)

    print("\n" + "=" * 80)
    print("TEST 2: FORWARD LOGITS COMPARISON")
    print("=" * 80)

    v_layer = vision_restored["vision_encoder"]["Gemma3VisionEncoderLayer_0"]
    s_layer = stitched_inner["vision_encoder"]["Gemma3VisionEncoderLayer_0"]

    print(
        "orig vision embedding kernel shape:",
        v_layer["embedding"]["kernel"].shape,
    )
    print(
        "orig vision embedding kernel first 10 values:",
        v_layer["embedding"]["kernel"].reshape(-1)[:10],
    )
    print(
        "stitched vision embedding kernel shape:",
        s_layer["embedding"]["kernel"].shape,
    )
    print(
        "stitched vision embedding kernel first 10 values:",
        s_layer["embedding"]["kernel"].reshape(-1)[:10],
    )

    print(
        "orig llm token embedder shape:",
        llm_restored["token_embedder"]["embedding"].shape,
    )
    print(
        "orig llm token embedder first 10 values:",
        llm_restored["token_embedder"]["embedding"].reshape(-1)[:10],
    )
    print(
        "stitched llm token embedder shape:",
        stitched_inner["token_embedder"]["embedding"].shape,
    )
    print(
        "stitched llm token embedder first 10 values:",
        stitched_inner["token_embedder"]["embedding"].reshape(-1)[:10],
    )
    print(
        "orig vision projector weights first 10 values:",
        vision_restored["vision_encoder"]["VisionEmbedder_0"]["mm_input_projection"]["w"].reshape(-1)[:10],
    )
    print(
        "stitched vision projector weights first 10 values:",
        stitched_inner["vision_encoder"]["VisionEmbedder_0"]["mm_input_projection"]["w"].reshape(-1)[:10],
    )

    # Instantiate and update original vision & token embedder modules
    orig_vision_tower = gemma3.Gemma3VisionEncoderLayer(config, self.test_mesh, rngs=rngs)
    orig_projector = gemma3.VisionEmbedder(config, self.test_mesh, rngs=rngs)
    # Manually define qwen3's token embedder to match MaxText's structures
    orig_token_embedder = Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        config=config,
        mesh=self.test_mesh,
        rngs=rngs,
    )

    nnx.update(
        orig_vision_tower,
        vision_restored["vision_encoder"]["Gemma3VisionEncoderLayer_0"],
    )
    nnx.update(
        orig_projector,
        vision_restored["vision_encoder"]["VisionEmbedder_0"],
    )
    nnx.update(
        orig_token_embedder,
        {"embedding": llm_restored["token_embedder"]["embedding"]},
    )

    # Instantiate and update stitched vision & token embedder modules
    stitched_vision_tower = gemma3.Gemma3VisionEncoderLayer(config, self.test_mesh, rngs=rngs)
    stitched_projector = gemma3.VisionEmbedder(config, self.test_mesh, rngs=rngs)
    stitched_token_embedder = Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        config=config,
        mesh=self.test_mesh,
        rngs=rngs,
    )

    nnx.update(
        stitched_vision_tower,
        stitched_inner["vision_encoder"]["Gemma3VisionEncoderLayer_0"],
    )
    nnx.update(
        stitched_projector,
        stitched_inner["vision_encoder"]["VisionEmbedder_0"],
    )
    nnx.update(
        stitched_token_embedder,
        {"embedding": stitched_inner["token_embedder"]["embedding"]},
    )

    # Inputs
    batch_size, prompt_len = 2, 128
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    image_size = config.image_size_for_vit
    input_images = jax.random.normal(key1, (batch_size, image_size, image_size, 3), dtype=jnp.float32)
    input_tokens = jax.random.randint(key2, (batch_size, prompt_len), 0, config.vocab_size)

    with jax.set_mesh(self.test_mesh):
      # Forward pass: Vision tower only (without projector)
      orig_vision_features = orig_vision_tower(input_images, deterministic=True)
      stitched_vision_features = stitched_vision_tower(input_images, deterministic=True)

      # Forward pass: Projector
      orig_projected_tokens = orig_projector(orig_vision_features)
      if orig_projected_tokens.ndim == 4 and orig_projected_tokens.shape[1] == 1:
        orig_projected_tokens = jnp.squeeze(orig_projected_tokens, axis=1)

      stitched_projected_tokens = stitched_projector(stitched_vision_features)
      if stitched_projected_tokens.ndim == 4 and stitched_projected_tokens.shape[1] == 1:
        stitched_projected_tokens = jnp.squeeze(stitched_projected_tokens, axis=1)

      # Forward pass: Text embedder
      orig_text_embeddings = orig_token_embedder(input_tokens)
      stitched_text_embeddings = stitched_token_embedder(input_tokens)

    # Vision output logits comparison
    np.testing.assert_allclose(orig_vision_features, stitched_vision_features, atol=1e-5)
    # Text embeddings output logits comparison
    np.testing.assert_allclose(orig_text_embeddings, stitched_text_embeddings, atol=1e-5)
    # Projector outputs logits comparison
    diff_stage3 = jnp.max(jnp.abs(orig_projected_tokens - stitched_projected_tokens))
    self.assertGreater(diff_stage3, 1.0)

    # Calculate statistics for logging
    def get_stats(orig, stitched):
      abs_diff = jnp.abs(orig - stitched)
      return (
          jnp.mean(jnp.abs(orig)),
          jnp.mean(jnp.abs(stitched)),
          jnp.mean(abs_diff),
          jnp.max(abs_diff),
      )

    orig_v_mean, stitched_v_mean, mean_v_diff, max_v_diff = get_stats(orig_vision_features, stitched_vision_features)
    orig_emb_mean, stitched_emb_mean, mean_emb_diff, max_emb_diff = get_stats(
        orig_text_embeddings, stitched_text_embeddings
    )
    orig_proj_mean, stitched_proj_mean, mean_proj_diff, max_proj_diff = get_stats(
        orig_projected_tokens, stitched_projected_tokens
    )

    print(
        f"  - Vision encoder output logits mean of abs: orig={orig_v_mean:.2f}, "
        f"stitched={stitched_v_mean:.2f} (Mean abs diff = {mean_v_diff:.2e}, max = {max_v_diff:.2e})"
    )
    print(
        f"  - Token embedder output logits mean of abs: "
        f"orig={orig_emb_mean:.2f}, stitched={stitched_emb_mean:.2f} "
        f"(Mean abs diff = {mean_emb_diff:.2e}, max = {max_emb_diff:.2e})"
    )
    print(
        f"  - Projector output logits mean of abs:     "
        f"orig={orig_proj_mean:.2f}, stitched={stitched_proj_mean:.2f} "
        f"(Mean abs diff = {mean_proj_diff:.2e}, max = {max_proj_diff:.2e})"
    )
    print("=" * 80 + "\n")

    del (
        stitched_inner,
        vision_restored,
        llm_restored,
        Linen_model,
        concrete_template,
    )
    del orig_vision_tower, orig_projector, orig_token_embedder
    del stitched_vision_tower, stitched_projector, stitched_token_embedder
    gc.collect()


if __name__ == "__main__":
  unittest.main()
