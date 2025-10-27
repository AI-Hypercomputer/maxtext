import jax
import pathlib
import os

from flax import nnx
from jax.sharding import Mesh
from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_AUTOREGRESSIVE
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.utils import gcs_utils

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from vllm.config import VllmConfig


def generate_maxtext_config(vllm_config: VllmConfig) -> pyconfig.HyperParameters:
    """Generate MaxText config from vLLM config."""

    def _path_exists(path: str) -> bool:
        if not path:
            return False
        return os.path.exists(path) or gcs_utils.gcs_path_exists(path)

    if "maxtext_config" in vllm_config.additional_config:
        overrides = vllm_config.additional_config["maxtext_config"]
    else:
        overrides = {}
        load_path = None
        if _path_exists(vllm_config.load.download_dir):
            load_path = vllm_config.load.download_dir
        elif _path_exists(vllm_config.model.model):
            load_path = vllm_config.model.model

        if load_path:
            overrides["load_parameters_path"] = load_path
        elif vllm_config.model.model:
            overrides["model_name"] = vllm_config.model.model

    # Add base config path to positional args
    base_config_path = pathlib.Path(MAXTEXT_PKG_DIR) / "configs" / "base.yml"
    argv_list = ["", str(base_config_path)]

    maxtext_config = pyconfig.initialize(argv_list, **overrides)
    return maxtext_config


class MaxTextDecoderModel(nnx.Module):
    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh) -> None:
        """Initialize the model on the provided mesh using model-defined sharding."""
        self.vllm_config = vllm_config
        self.maxtext_config = generate_maxtext_config(vllm_config)

        # Model configuration
        self.rngs = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model_mode = MODEL_MODE_AUTOREGRESSIVE

        # Model creation
        self.model, _ = model_creation_utils.create_nnx_model(self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rngs=self.rngs)
        self.logits: jax.Array | None = None

        if not isinstance(self.model, nnx.Module):
            raise ValueError("Model must be an instance of nnx.Module. Please set enable_nnx=True in the MaxText base config.")

    def __call__(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        **kwargs,
    ) -> tuple[list[jax.Array], jax.Array]:
        """Return (updated_kv_caches, hidden[Q, d_model])."""
        # Delegate to the decoder; decoder threads rpa_* into each Attention layer.
        logits, hidden, kv_caches = self.model(
            decoder_input_tokens=input_ids, 
            decoder_positions=attention_metadata.input_positions, 
            kv_caches=kv_caches, 
            attention_metadata=attention_metadata, 
            **kwargs
        )
        self.logits = logits  # Store logits for later use if needed
        return kv_caches, hidden

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Return logits[Q, vocab]."""
        if not self.logits:
            raise ValueError("Logits have not been computed yet. Please call the model first.")
        
        return self.logits

    def load_weights(self, rng_key: jax.Array) -> None:
        """Load params on the provided mesh using model-defined sharding."""
        pass


class MaxTextForCausalLM(nnx.Module):
    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
        self.cfg = vllm_config.model_config
        self.rngs = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model = MaxTextDecoderModel(vllm_config, rng_key, mesh)
        self.is_text_generation_model = True

    def __call__(self,
               kv_caches: list[jax.Array],
               input_ids: jax.Array,
               attention_metadata: AttentionMetadata,
               **kwargs) -> tuple[list[jax.Array], jax.Array]:
        # Delegate to the decoder; decoder threads rpa_* into each Attention layer.
        kv_caches, hidden = self.model(kv_caches, input_ids, attention_metadata, **kwargs)
        return kv_caches, hidden

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def get_input_embeddings(self):
        return None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.model.compute_logits(hidden_states)

    def load_weights(self, rng_key: jax.Array) -> None:
        self.model.load_weights(rng_key)
