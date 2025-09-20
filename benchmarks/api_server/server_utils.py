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


import os
import bisect
import math
import yaml
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any

from fastapi import HTTPException

from benchmarks.api_server.maxtext_generator import MaxTextGenerator
from benchmarks.api_server.server_models import LogProbsPayload

# ----------------------------
# Debugging
# ----------------------------

DEBUG_MODE = os.environ.get("MAXTEXT_SERVER_DEBUG", "0") == "1"
DEBUG_LOG_FILE = os.environ.get("MAXTEXT_DEBUG_LOG_FILE", "benchmarks/api_server/server_debug_log.jsonl")
logger = logging.getLogger(__name__)


def log_debug_event(request_id: str, event_type: str, content: dict):
    """
    Helper to write a structured debug log entry if DEBUG_MODE is on.

    Args:
        request_id: The unique identifier for the request.
        event_type: A string describing the type of event being logged (e.g., 'request', 'response').
        content: A dictionary containing the data to be logged.
    """
    if not DEBUG_MODE:
        return
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "event": event_type,
            "content": content,
        }
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        # Use logger for errors
        logger.error(f"Error writing to debug log file '{DEBUG_LOG_FILE}': {e}")


# ----------------------------
# Request/Response Helpers
# ----------------------------


def decode_one_prompt(p: Union[str, List[int]], llm: MaxTextGenerator) -> str:
    """
    Decodes a single prompt element, which can be a string or a list of token IDs.

    Args:
        p: The prompt element to decode.
        llm: The MaxTextGenerator instance, used for its tokenizer.

    Returns:
        The decoded prompt as a string.

    Raises:
        ValueError: If the prompt item has an unsupported type.
    """
    if isinstance(p, str):
        return p
    if isinstance(p, list) and (len(p) == 0 or isinstance(p[0], int)):
        try:
            return llm.tokenizer.decode(p)
        except Exception:
            print("Return empty string on decoding error")
            return ""
    raise ValueError("Unsupported prompt item type")


def get_prompts_for_request(req: any, llm: MaxTextGenerator) -> List[str]:
    """
    Extracts and formats a list of prompts from a request object.

    This function handles both standard `CompletionRequest` and `ChatCompletionRequest`
    types, converting them into a unified list of string prompts that the model
    can process.

    Args:
        req: The request object.
        llm: The MaxTextGenerator instance.

    Returns:
        A list of string prompts.
    """
    if hasattr(req, 'messages'):  # ChatCompletionRequest
        messages = [m.model_dump() for m in req.messages]
        formatted_prompt = llm.tokenizer.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return [formatted_prompt]
    else:  # CompletionRequest
        return normalize_prompts(req.prompt, llm)


def normalize_prompts(prompt: Union[str, List[str], List[int], List[List[int]]], llm: MaxTextGenerator) -> List[str]:
    """
    Normalizes the highly flexible 'prompt' field from an OpenAI-style request
    into a simple list of strings.

    The 'prompt' field can be a single string, a list of strings, a list of
    token IDs, or a list of lists of token IDs. This function handles all
    these cases and returns a flat list of string prompts.

    Args:
        prompt: The prompt data from the request.
        llm: The MaxTextGenerator instance for decoding token IDs.

    Returns:
        A list of normalized string prompts.

    Raises:
        HTTPException: If the prompt type is not supported.
    """
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            return []
        # Prompts can be a list of strings, a single list of ints, or a list of lists of ints.
        first = prompt[0]
        if isinstance(first, str):
            return [str(x) for x in prompt]
        if isinstance(first, int):
            return [decode_one_prompt(prompt, llm)]
        if isinstance(first, list):
            return [decode_one_prompt(x, llm) for x in prompt]
    raise HTTPException(status_code=400, detail="Unsupported prompt type for this API.")


def decode_token_id(token_id: int, llm: MaxTextGenerator) -> str:
    """
    Decodes a single token ID into its string representation.

    Args:
        token_id: The integer token ID to decode.
        llm: The MaxTextGenerator instance.

    Returns:
        The decoded string.
    """
    return llm.tokenizer.decode([int(token_id)])


def finite_or_none(v: Optional[float]) -> Optional[float]:
    """
    Returns the float if it's finite (i.e., not NaN or infinity), otherwise None.

    Args:
        v: The float value to check.

    Returns:
        The original float if it is finite, otherwise None.
    """
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def to_openai_logprobs(lp_obj: Any, llm: MaxTextGenerator, want_top: bool = True) -> Optional[LogProbsPayload]:
    """
    Converts the internal logprobs object to the OpenAI-compatible format.

    Args:
        lp_obj: The internal logprobs object from the generation result.
        llm: The MaxTextGenerator instance for decoding tokens.
        want_top: Whether to populate the `top_logprobs` field.

    Returns:
        A `LogProbsPayload` object compatible with the OpenAI API, or None.
    """
    if lp_obj is None:
        return None

    token_strings = [decode_token_id(tid, llm) for tid in lp_obj.tokens]
    token_logprobs = [finite_or_none(v) for v in lp_obj.token_logprobs]
    text_offset = list(lp_obj.text_offset)

    # Ensure all lists are of the same length to avoid errors.
    min_len = min(len(token_strings), len(token_logprobs), len(text_offset))
    token_strings = token_strings[:min_len]
    token_logprobs = token_logprobs[:min_len]
    text_offset = text_offset[:min_len]

    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
    if want_top:
        # The current implementation only returns the logprob of the single sampled token.
        # This structure is a placeholder for a future feature where the model might
        # return the logprobs of multiple top tokens at each step.
        top_logprobs = [
            ({tok: lp} if lp is not None else None)
            for tok, lp in zip(token_strings, token_logprobs)
        ]

    return LogProbsPayload(
        tokens=token_strings,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=text_offset,
    )


def count_tokens(s: str, llm: MaxTextGenerator) -> int:
    """
    Counts the number of tokens in a string.

    Args:
        s: The string to tokenize.
        llm: The MaxTextGenerator instance.

    Returns:
        The number of tokens in the string.
    """
    try:
        # Use the underlying tokenizer to avoid the jetstream wrapper's
        # padding issues with single-token sequences.
        ids = llm.tokenizer.tokenizer.encode(s, add_special_tokens=False)
        return len(ids)
    except Exception as e:
        logger.warning(f"Could not count tokens for string '{s[:50]}...': {e}")
        return 0


def apply_stops_to_text_and_logprobs(
    text: str,
    logprobs_payload: Optional[LogProbsPayload],
    stop: Optional[Union[str, List[str]]],
) -> tuple[str, Optional[LogProbsPayload], Optional[str]]:
    """
    Truncates the generated text and corresponding logprobs at the first occurrence
    of any of the specified stop sequences.

    Args:
        text: The generated text.
        logprobs_payload: The corresponding logprobs payload.
        stop: The stop sequence(s) to search for.

    Returns:
        A tuple containing the truncated text, the truncated logprobs payload,
        and the reason for stopping ('stop' if a sequence was found, otherwise None).
    """
    if not stop:
        return text, logprobs_payload, None

    stops = [stop] if isinstance(stop, str) else list(stop)
    
    # Find the earliest stop sequence
    first_stop_index = -1
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1:
            first_stop_index = i if first_stop_index == -1 else min(first_stop_index, i)
    
    if first_stop_index == -1:
        return text, logprobs_payload, None

    # Truncate text
    new_text = text[:first_stop_index]

    # Truncate logprobs payload if it exists
    if logprobs_payload is not None:
        truncate_at_index = bisect.bisect_left(
            logprobs_payload.text_offset, first_stop_index
        )

        new_logprobs = LogProbsPayload(
            tokens=logprobs_payload.tokens[:truncate_at_index],
            token_logprobs=logprobs_payload.token_logprobs[:truncate_at_index],
            top_logprobs=logprobs_payload.top_logprobs[:truncate_at_index] if logprobs_payload.top_logprobs is not None else None,
            text_offset=logprobs_payload.text_offset[:truncate_at_index],
        )
        return new_text, new_logprobs, "stop"

    return new_text, logprobs_payload, "stop"
