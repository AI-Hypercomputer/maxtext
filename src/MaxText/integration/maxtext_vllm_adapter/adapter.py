import jax
import pathlib
import os
import jax.numpy as jnp

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
    base_config_path = pathlib.Path(MAXTEXT_PKG_DIR) / "configs" / "vllm.yml"
    argv_list = ["", str(base_config_path)]

    maxtext_config = pyconfig.initialize(argv_list, **overrides)
    return maxtext_config


class MaxTextDecoderModel(nnx.Module):
    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh) -> None:
        """Initialize the model on the provided mesh using model-defined sharding."""
        self.vllm_config = vllm_config
        self.maxtext_config = generate_maxtext_config(vllm_config)

        # Model configuration
        self.mesh = mesh
        self.model_mode = MODEL_MODE_AUTOREGRESSIVE

        # Model creation
        self.model: nnx.Module | None = None
        self.logits: jax.Array | None = None

    def __call__(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
        **kwargs,
    ) -> tuple[list[jax.Array], jax.Array]:
        """Return (updated_kv_caches, hidden[Q, d_model])."""
        if not isinstance(self.model, nnx.Module):
            raise ValueError("Model must be an instance of nnx.Module. Call load_weights() first.")

        if input_ids.ndim < 2:
            input_ids = jnp.expand_dims(input_ids, axis=0)
        
        input_positions = attention_metadata.input_positions
        if input_positions.ndim < 2:
            input_positions = jnp.expand_dims(input_positions, axis=0)

        aux_hidden_states = []
        logits, hidden, kv_caches = self.model(
            decoder_input_tokens=input_ids, 
            decoder_positions=input_positions, 
            kv_caches=kv_caches, 
            attention_metadata=attention_metadata,
            model_mode=self.model_mode, 
            **kwargs
        )
        if hidden.ndim > 1:
            hidden = jnp.squeeze(hidden, axis=0)
            logits = jnp.squeeze(logits, axis=0)
        
        self.logits = logits  # Store logits for later use if needed

        return kv_caches, hidden, aux_hidden_states

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Return logits[Q, voc ab]."""
        if self.logits is not None:
            return self.logits

        embeddings = self.model.token_embedder
        return self.model.decoder._apply_output_head(embeddings, hidden_states, True, self.model_mode)

    def load_weights(self, rng_key: jax.Array) -> None:
        """Load params on the provided mesh using model-defined sharding."""
        self.model, _ = model_creation_utils.create_nnx_model(
            self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
        )

class MaxTextForCausalLM(nnx.Module):
    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
        self.cfg = vllm_config.model_config
        self.mesh = mesh
        self.model = MaxTextDecoderModel(vllm_config, rng_key, mesh)
        self.is_text_generation_model = True

    def __call__(self,
               kv_caches: list[jax.Array],
               input_ids: jax.Array,
               attention_metadata: AttentionMetadata,
               *args,
               **kwargs) -> tuple[list[jax.Array], jax.Array]:
        kv_caches, hidden, aux_hidden_states = self.model(kv_caches, input_ids, attention_metadata, *args, **kwargs)
        return kv_caches, hidden, aux_hidden_states

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def get_input_embeddings(self) -> jax.Array:
        return self.model.token_embedder.embedding

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.model.compute_logits(hidden_states)

    def load_weights(self, rng_key: jax.Array) -> None:
        self.model.load_weights(rng_key)
