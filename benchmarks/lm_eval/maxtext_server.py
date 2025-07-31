import os
import time
import uuid
import math
from typing import List, Optional, Union, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from benchmarks.lm_eval.maxtext_generator import MaxTextGenerator


# ----------------------------
# Init
# ----------------------------

def get_maxtext_args_from_env() -> List[str]:
    return [
        "maxtext_server.py",
        "MaxText/configs/base_gemma3_4b.yml",
    ]

print("Starting server and initializing MaxTextGenerator...")
llm = MaxTextGenerator(get_maxtext_args_from_env())
app = FastAPI()


# ----------------------------
# Models (OpenAI compat-ish)
# ----------------------------

class CompletionRequest(BaseModel):
    model: str
    # Accept strings, list[str], list[int], or list[list[int]] (tokenized requests)
    prompt: Union[str, List[str], List[int], List[List[int]]]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    echo: Optional[bool] = False
    logprobs: Optional[int] = None  # count of top alternatives requested
    stop: Optional[Union[str, List[str]]] = None  # support harness "until"→"stop"
    seed: Optional[int] = None  # accepted (ignored by backend for now)

    @field_validator("logprobs")
    def validate_logprobs(cls, v):
        if v is not None and v < 0:
            raise ValueError("logprobs must be a non-negative integer if provided.")
        return v


class LogProbsPayload(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: Optional[List[Dict[str, float]]] = None
    text_offset: List[int]


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[LogProbsPayload] = None
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# ----------------------------
# Helpers
# ----------------------------

def _decode_one_prompt(p: Union[str, List[int]]) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, list) and (len(p) == 0 or isinstance(p[0], int)):
        # a single tokenized prompt (list[int]) or empty -> decode to string
        try:
            return llm.tokenizer.decode(p)
        except Exception:
            # last resort: treat as empty string
            return ""
    raise ValueError("Unsupported prompt item type")

def _normalize_prompts(prompt: Union[str, List[str], List[int], List[List[int]]]) -> List[str]:
    # cases:
    #  - str
    #  - list[str]
    #  - list[int]
    #  - list[list[int]]
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            return []
        first = prompt[0]
        if isinstance(first, str):
            # list[str]
            return [str(x) for x in prompt]
        if isinstance(first, int):
            # single tokenized prompt (list[int])
            return [_decode_one_prompt(prompt)]
        if isinstance(first, list):
            # list[list[int]]
            return [_decode_one_prompt(x) for x in prompt]
    raise HTTPException(status_code=400, detail="Unsupported prompt type for this API.")

def _decode_token_id(token_id: int) -> str:
    return llm.tokenizer.decode([int(token_id)])

def _finite_or_clamp(v: Optional[float]) -> float:
    if v is None:
        return -1e10
    f = float(v)
    return f if math.isfinite(f) else -1e10

def _to_openai_logprobs(lp_obj: Any, want_top: bool = True) -> Optional[LogProbsPayload]:
    if lp_obj is None:
        return None

    token_strings = [_decode_token_id(tid) for tid in lp_obj.tokens]
    token_logprobs = [_finite_or_clamp(v) for v in lp_obj.token_logprobs]
    text_offset = list(lp_obj.text_offset)

    # ensure equal lengths
    L = min(len(token_strings), len(token_logprobs), len(text_offset))
    token_strings = token_strings[:L]
    token_logprobs = token_logprobs[:L]
    text_offset = text_offset[:L]

    top_logprobs = None
    if want_top:
        # Provide a dict per token with at least the chosen token → its logprob.
        top_logprobs = [{tok: lp} for tok, lp in zip(token_strings, token_logprobs)]

    return LogProbsPayload(
        tokens=token_strings,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=text_offset,
    )

def _count_tokens(s: str) -> int:
    try:
        ids, _ = llm.tokenizer.encode(s, is_bos=False, prefill_lengths=None)
        return len(ids)
    except Exception:
        return 0

def _apply_stops_to_text_and_logprobs(
    text: str,
    lp: Optional[LogProbsPayload],
    stop: Optional[Union[str, List[str]]],
) -> tuple[str, Optional[LogProbsPayload], Optional[str]]:
    if not stop:
        return text, lp, None
    stops = [stop] if isinstance(stop, str) else list(stop)
    # find earliest stop occurrence
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

    # truncate logprobs arrays to tokens whose offsets start before `cut`
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


# ----------------------------
# Endpoint
# ----------------------------

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this server.")

    prompts = _normalize_prompts(request.prompt)
    if len(prompts) == 0:
        raise HTTPException(status_code=400, detail="Empty prompt list.")

    start = time.time()
    try:
        completions = llm.generate_batch(
            prompts=prompts,
            image_paths=None,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,  # Non-None triggers logprob collection
            echo=request.echo,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    print(f"/v1/completions processed {len(prompts)} prompt(s) in {time.time()-start:.2f}s")

    choices: List[CompletionChoice] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    # normalize to indexable list of "Completion-like"
    for idx, prompt in enumerate(prompts):
        item = completions[idx]
        if hasattr(item, "text"):
            text_out = item.text if isinstance(item.text, str) else str(item.text)
            lp_payload = _to_openai_logprobs(getattr(item, "logprobs", None), want_top=(request.logprobs is not None))
            finish_reason = getattr(item, "finish_reason", "stop")
        else:
            text_out = str(item)
            lp_payload = None
            finish_reason = "stop"

        # apply stop sequences (truncate text + logprobs)
        text_out, lp_payload, stop_reason = _apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
        if stop_reason is not None:
            finish_reason = stop_reason

        # usage
        prompt_token_count = _count_tokens(prompt)
        completion_token_count = _count_tokens(text_out)
        prompt_tokens_total += prompt_token_count
        completion_tokens_total += completion_token_count

        choices.append(CompletionChoice(
            text=text_out,
            index=idx,
            logprobs=lp_payload,
            finish_reason=finish_reason,
        ))

    usage = Usage(
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens_total,
        total_tokens=prompt_tokens_total + completion_tokens_total,
    )

    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=usage,
    )


@app.get("/")
def health_check():
    return {"status": "ok", "message": "MaxText API server is running."}


if __name__ == "__main__":
    uvicorn.run("maxtext_server:app", host="0.0.0.0", port=8000, reload=False)
