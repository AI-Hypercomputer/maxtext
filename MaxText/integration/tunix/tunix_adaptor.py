# MaxText/integrations/tunix/tunix_adapter.py
from __future__ import annotations

from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from MaxText.common_types import MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE, Config
# from .cache_types import TunixDecodeCache, empty_cache
# from MaxText.layers.models import LlamaTransformerNNX
from MaxText.layers.models import Transformer

Array = jax.Array  # Alias for clarity


class TunixMaxTextLlama(nnx.Module):
    """Adapter exposing Tunix SFT call signature over a LlamaTransformerNNX model."""

    def __init__(
        self,
        base_model: Transformer,
        *,
        use_attention_mask: bool = False,
    ):
        super().__init__()
        self.base = base_model
        self.use_attention_mask = use_attention_mask

    # ------------------------------------------------------------------ #
    # Tunix call signature (SFT)
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        input_tokens: Array,  # [B, L]
        positions: Array,  # [B, L]
        cache: Optional[Any],  # Tunix passes None in SFT
        attention_mask: Optional[Array],  # [B, L, L] or None
        output_hidden_states: bool = False,  # ignored
    ) -> Tuple[Array, None]:
        """Forward compatible with Tunix PeftTrainer default loss.
        Returns logits, None (cache unused in SFT).
        """
        attn = attention_mask if self.use_attention_mask else None
        # logits = self.base.forward_train(
        logits = self.base(
            decoder_input_tokens=input_tokens,
            decoder_positions=positions,
            decoder_segment_ids=None,
            # attention_mask=attn,
        )
        return logits, None

    # ------------------------------------------------------------------ #
    # Decode bridge (optional outside SFT)
    # ------------------------------------------------------------------ #
    def generate_step(
        self,
        last_tokens: Array,  # [B, S] (S typically =1)
        positions: Array,  # [B, S]
        cache: TunixDecodeCache | None,
    ) -> Tuple[Array, TunixDecodeCache]:
        """Autoregressive decode wrapper."""
        if cache is None:
            cache = empty_cache()
        # logits = self.base.forward_decode(
        logits = self.base(
            last_tokens,
            positions,
            page_state=cache.page_state,
            slot=cache.slot,
            true_length=cache.true_length,
            previous_chunk=cache.previous_chunk,
        )
        # Update lightweight cache; underlying page_state persists
        new_cache = TunixDecodeCache(
            step=cache.step + positions.shape[1],
            slot=cache.slot,
            true_length=cache.true_length,
            page_state=cache.page_state,
            previous_chunk=None,  # if base returns one, plumb here
        )
        return logits, new_cache

    # ------------------------------------------------------------------ #
    # Convenience toggles
    # ------------------------------------------------------------------ #
    def set_dropout(self, enabled: bool):
        self.base.enable_dropout = enabled

    def to_hf_mappings(self):
        return {
            # Token embeddings - shard vocab dimension for TP
            "base.token_embedder.embedding": (
                "embed.embedding",
                ("model", None),
            ),
            # Final layer norm - no sharding needed
            "base.decoder.decoder_norm.scale": (
                "model.norm.scale",
                (None,),
            ),
            # LM head (logits projection) - shard vocab dimension for TP
            "base.decoder.logits_dense.kernel": (
                "lm_head",
                (None, "model"),
            ),
            # Layer-specific mappings (scanned -> unscanned)
            # MLP components - shard hidden dimensions for TP
            "base.decoder.layers.mlp.wi_0.kernel": (
                "model.layers.*.mlp.gate_proj.kernel",
                (None, "layer", "model"),
            ),  # gate_proj: (4096, 14336) - shard output
            "base.decoder.layers.mlp.wi_1.kernel": (
                "model.layers.*.mlp.up_proj.kernel",
                (None, "layer", "model"),
            ),  # up_proj: (4096, 14336) - shard output
            "base.decoder.layers.mlp.wo.kernel": (
                "model.layers.*.mlp.down_proj.kernel",
                ("model", "layer", None),
            ),  # down_proj: (14336, 4096) - shard input
            # Layer norms - no sharding needed
            "base.decoder.layers.pre_self_attention_layer_norm.scale": (
                "model.layers.*.input_layernorm.scale",
                (None, "layer"),
            ),
            "base.decoder.layers.post_self_attention_layer_norm.scale": (
                "model.layers.*.post_attention_layernorm.scale",
                (None, "layer"),
            ),
            # Attention components - shard head dimensions for TP
            "base.decoder.layers.self_attention.query.kernel": (
                "model.layers.*.self_attn.q_proj.kernel",
                (None, "layer", "model", None),
            ),  # q_proj: shard num_heads
            "base.decoder.layers.self_attention.key.kernel": (
                "model.layers.*.self_attn.k_proj.kernel",
                (None, "layer", "model", None),
            ),  # k_proj: shard num_kv_heads
            "base.decoder.layers.self_attention.value.kernel": (
                "model.layers.*.self_attn.v_proj.kernel",
                (None, "layer", "model", None),
            ),  # v_proj: shard num_kv_heads
            "base.decoder.layers.self_attention.out.kernel": (
                "model.layers.*.self_attn.o_proj.kernel",
                ("model", "layer", None, None),
            ),  # o_proj: shard input heads
        }

    def to_hf_transpose_keys(self):
        return {}

    def lora_to_hf_mappings(self):
        return None

    def to_hf_hook_fns(self):

        def reorder_rope(arr):
            evens = arr[..., ::2]
            odds = arr[..., 1::2]
            return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

        def transform_query_kernel(arr):
            head_dim = arr.shape[-1]
            assert head_dim == 128  # hard coded for now
            depth_scale = np.dtype("float32").type(np.sqrt(head_dim))
            arr = arr * depth_scale
            return reorder_rope(arr)

        def transform_key_kernel(arr):
            return reorder_rope(arr)

        return {
            "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
            "base.decoder.layers.self_attention.key.kernel": transform_key_kernel,
        }
