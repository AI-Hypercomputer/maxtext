import os
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
    """Helper to write a structured debug log entry if DEBUG_MODE is on."""
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
# Argument & Config Parsing
# ----------------------------

def get_maxtext_args() -> List[str]:
    """
    Constructs MaxText arguments from a YAML config file specified by an environment variable.
    """
    config_path = os.environ.get("MAXTEXT_MODEL_CONFIG")
    if not config_path:
        raise ValueError("MAXTEXT_MODEL_CONFIG environment variable not set. This should point to a YAML config file.")

    logger.info(f"Loading model configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}") from None
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file '{config_path}': {e}") from e

    args = ["maxtext_server.py"]

    if "base_config" in config_data:
        args.append(config_data["base_config"])

    if "args" in config_data:
        for key, value in config_data["args"].items():
            if isinstance(value, str):
                args.append(f'{key}="{value}"')
            else:
                args.append(f'{key}={value}')
    logger.info(f"Constructed MaxText args: {args}")
    return args

# ----------------------------
# Request/Response Helpers
# ----------------------------

def decode_one_prompt(p: Union[str, List[int]], llm: MaxTextGenerator) -> str:
    """Decodes a single prompt element, which can be a string or a list of token IDs."""
    if isinstance(p, str):
        return p
    if isinstance(p, list) and (len(p) == 0 or isinstance(p[0], int)):
        try:
            return llm.tokenizer.decode(p)
        except Exception:
            # Return empty string on decoding error
            return ""
    raise ValueError("Unsupported prompt item type")

def normalize_prompts(prompt: Union[str, List[str], List[int], List[List[int]]], llm: MaxTextGenerator) -> List[str]:
    """
    Normalizes the highly flexible 'prompt' field from an OpenAI-style request
    into a simple list of strings.
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
    """Decodes a single token ID into its string representation."""
    return llm.tokenizer.decode([int(token_id)])

def finite_or_none(v: Optional[float]) -> Optional[float]:
    """Returns the float if it's finite, otherwise None."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None

def to_openai_logprobs(lp_obj: Any, llm: MaxTextGenerator, want_top: bool = True) -> Optional[LogProbsPayload]:
    """Converts the internal logprobs object to the OpenAI-compatible format."""
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
    """Counts the number of tokens in a string."""
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
        truncate_at_index = 0
        for off in logprobs_payload.text_offset:
            if off < first_stop_index:
                truncate_at_index += 1
            else:
                break
        
        new_logprobs = LogProbsPayload(
            tokens=logprobs_payload.tokens[:truncate_at_index],
            token_logprobs=logprobs_payload.token_logprobs[:truncate_at_index],
            top_logprobs=logprobs_payload.top_logprobs[:truncate_at_index] if logprobs_payload.top_logprobs is not None else None,
            text_offset=logprobs_payload.text_offset[:truncate_at_index],
        )
        return new_text, new_logprobs, "stop"

    return new_text, logprobs_payload, "stop"

    