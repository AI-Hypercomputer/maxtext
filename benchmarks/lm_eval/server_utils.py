import os
import math
import yaml
import json
from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any

from fastapi import HTTPException

from benchmarks.lm_eval.maxtext_generator import MaxTextGenerator
from benchmarks.lm_eval.server_models import LogProbsPayload

# ----------------------------
# Debugging
# ----------------------------

DEBUG_MODE = os.environ.get("MAXTEXT_SERVER_DEBUG", "0") == "1"
DEBUG_LOG_FILE = os.environ.get("MAXTEXT_DEBUG_LOG_FILE", "benchmarks/lm_eval/server_debug_log.jsonl")

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
        print(f"Error writing to debug log file '{DEBUG_LOG_FILE}': {e}")

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

    print(f"Loading model configuration from: {config_path}")
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
    print(f"Constructed MaxText args: {args}")
    return args

# ----------------------------
# Request/Response Helpers
# ----------------------------

def decode_one_prompt(p: Union[str, List[int]], llm: MaxTextGenerator) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, list) and (len(p) == 0 or isinstance(p[0], int)):
        try:
            return llm.tokenizer.decode(p)
        except Exception:
            return ""
    raise ValueError("Unsupported prompt item type")

def normalize_prompts(prompt: Union[str, List[str], List[int], List[List[int]]], llm: MaxTextGenerator) -> List[str]:
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            return []
        first = prompt[0]
        if isinstance(first, str):
            return [str(x) for x in prompt]
        if isinstance(first, int):
            return [decode_one_prompt(prompt, llm)]
        if isinstance(first, list):
            return [decode_one_prompt(x, llm) for x in prompt]
    raise HTTPException(status_code=400, detail="Unsupported prompt type for this API.")

def decode_token_id(token_id: int, llm: MaxTextGenerator) -> str:
    return llm.tokenizer.decode([int(token_id)])

def finite_or_none(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None

def to_openai_logprobs(lp_obj: Any, llm: MaxTextGenerator, want_top: bool = True) -> Optional[LogProbsPayload]:
    if lp_obj is None:
        return None

    token_strings = [decode_token_id(tid, llm) for tid in lp_obj.tokens]
    token_logprobs = [finite_or_none(v) for v in lp_obj.token_logprobs]
    text_offset = list(lp_obj.text_offset)

    L = min(len(token_strings), len(token_logprobs), len(text_offset))
    token_strings = token_strings[:L]
    token_logprobs = token_logprobs[:L]
    text_offset = text_offset[:L]

    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
    if want_top:
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
    try:
        ids, _ = llm.tokenizer.encode(s, is_bos=False, prefill_lengths=None)
        return len(ids)
    except Exception:
        return 0

def apply_stops_to_text_and_logprobs(
    text: str,
    lp: Optional[LogProbsPayload],
    stop: Optional[Union[str, List[str]]],
) -> tuple[str, Optional[LogProbsPayload], Optional[str]]:
    if not stop:
        return text, lp, None
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = -1
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1:
            cut = i if cut == -1 else min(cut, i)
    if cut == -1:
        return text, lp, None

    new_text = text[:cut]

    if lp is not None:
        k = 0
        for off in lp.text_offset:
            if off < cut:
                k += 1
            else:
                break
        lp = LogProbsPayload(
            tokens=lp.tokens[:k],
            token_logprobs=lp.token_logprobs[:k],
            top_logprobs=lp.top_logprobs[:k] if lp.top_logprobs is not None else None,
            text_offset=lp.text_offset[:k],
        )
    return new_text, lp, "stop"
    