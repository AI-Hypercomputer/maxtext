# MaxText/integrations/tunix/tunix_adapter.py
from __future__ import annotations

from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
from flax import nnx

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
        input_tokens: Array,             # [B, L]
        positions: Array,                # [B, L]
        cache: Optional[Any],            # Tunix passes None in SFT
        attention_mask: Optional[Array], # [B, L, L] or None
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
        last_tokens: Array,          # [B, S] (S typically =1)
        positions: Array,            # [B, S]
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