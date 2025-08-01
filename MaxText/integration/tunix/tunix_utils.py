# MaxText/integrations/tunix/utils.py
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from MaxText.common_types import Config
# from .maxtext_nnx_llama import LlamaTransformerNNX
from MaxText.layers.models import LlamaTransformerNNX
from MaxText.integrations.tunix.tunix_adaptor import TunixMaxTextLlama
from MaxText import maxtext_utils
from jax.sharding import Mesh
from collections.abc import Sequence

#Anisha: this is just like current mt.from_pretrained that we have
def build_maxtext_nnx_llama(
    config: Config,
    mesh: Mesh,
    rngs: nnx.Rngs,
    *,
    enable_dropout: bool = False,
    init_batch_size: int = 1,
    init_seq_len: int = 1,
) -> LlamaTransformerNNX:
    """Instantiate & initialize the raw LlamaTransformerNNX model."""
    model = LlamaTransformerNNX(config, mesh=mesh, rngs=rngs, enable_dropout=enable_dropout)

    # Lazy variable materialization using dummy shapes; real weights loaded later.
    dummy_tokens = jnp.zeros((init_batch_size, init_seq_len), dtype=jnp.int32)
    dummy_pos    = jnp.zeros_like(dummy_tokens)

    # This call will create params in submodules
    _ = model.forward_train(
        decoder_input_tokens=dummy_tokens,
        decoder_positions=dummy_pos,
        decoder_segment_ids=None,
        attention_mask=None,
    )
    return model

#Anisha: this is the mt.from_pretrained that we need
def build_tunix_wrapper(
    config: Config,
    rngs: nnx.Rngs,
    *,
    enable_dropout: bool = False,
    init_batch_size: int = 1,
    init_seq_len: int = 1,
    use_attention_mask: bool = False,
    devices: Sequence[jax.Device] | None = None,
):
    """Construct Tunix-ready wrapper (NNX) around a MaxText model."""
    devices_array = maxtext_utils.create_device_mesh(config, devices)
    mesh = Mesh(devices_array, config.mesh_axes)
    #TODO: Anisha: update train_utils.create_model to return NNX model
    # by copying over this logic
    base = build_maxtext_nnx_llama(
        config,
        mesh,
        rngs,
        enable_dropout=enable_dropout,
        init_batch_size=init_batch_size,
        init_seq_len=init_seq_len,
    )
    wrapper = TunixMaxTextLlama(base, use_attention_mask=use_attention_mask)
    return wrapper