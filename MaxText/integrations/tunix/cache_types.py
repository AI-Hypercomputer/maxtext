from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TunixDecodeCache:
    """Minimal decode cache bridge between Tunix and MaxText.

    Attributes
    ----------
    step : int
        Decode step offset beyond prefill length.
    slot : Optional[int]
        MaxText per-request slot index (paged KV decoding).
    true_length : Optional[int]
        True prompt length (w/out padding) for paged KV allocation.
    page_state : Optional[Any]
        MaxText page_manager.PageState (global KV memory pool).
    previous_chunk : Optional[Any]
        Engine-side decode chunk handle (returned by MaxText decode kernels).
    """
    step: int = 0
    slot: Optional[int] = None
    true_length: Optional[int] = None
    page_state: Optional[Any] = None
    previous_chunk: Optional[Any] = None


def empty_cache() -> TunixDecodeCache:
    return TunixDecodeCache()
