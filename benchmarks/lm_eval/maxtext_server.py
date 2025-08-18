import os
import time
import uuid
import math
from typing import List, Optional, Union, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from benchmarks.lm_eval.maxtext_generator import MaxTextGenerator

import json
from datetime import datetime, timezone


# ----------------------------
# Init
# ----------------------------

def get_maxtext_args_from_env() -> List[str]:
    return [
        "maxtext_server.py",
        "MaxText/configs/base.yml",
        'model_name="mixtral-8x7b"',
        'load_parameters_path="gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_stable-2025-08-15-01-00-23//unscanned_ckpt/checkpoints/0/items"',
        'tokenizer_path="mistralai/Mixtral-8x7B-Instruct-v0.1"',
        'tokenizer_type="huggingface"',
        'per_device_batch_size=4',
        'ici_tensor_parallelism=4',
        'max_prefill_predict_length=4096',
        'max_target_length=8192',
        'scan_layers=false',
        'attention="dot_product"',
        'return_log_prob=True',
        'async_checkpointing=false'
    ]

print("Starting server and initializing MaxTextGenerator...")
llm = MaxTextGenerator(get_maxtext_args_from_env())
app = FastAPI()

DEBUG_MODE = os.environ.get("MAXTEXT_SERVER_DEBUG", "0") == "1"
DEBUG_LOG_FILE = "server_debug_log.jsonl"

if DEBUG_MODE:
    print(f"DEBUG MODE IS ENABLED. Requests and responses will be logged to {DEBUG_LOG_FILE}")


# ----------------------------
# Models (OpenAI compat-ish)
# ----------------------------

class CompletionRequest(BaseModel):
    model: str
    # Accept strings, list[str], list[int], or list[list[int]] (tokenized requests)
    prompt: Union[str, List[str], List[int], List[List[int]]]
    max_tokens: Optional[int] = None
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
    token_logprobs: List[Optional[float]]
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
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


# ---- Chat Completions Models ----

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    logprobs: Optional[bool] = False  # Note: Different from legacy `completions`
    top_logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: Optional[LogProbsPayload] = None # Not fully supported yet, but good for structure

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
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

def _finite_or_none(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _to_openai_logprobs(lp_obj: Any, want_top: bool = True) -> Optional[LogProbsPayload]:
    if lp_obj is None:
        return None

    token_strings = [_decode_token_id(tid) for tid in lp_obj.tokens]
    token_logprobs = [_finite_or_none(v) for v in lp_obj.token_logprobs]
    text_offset = list(lp_obj.text_offset)

    # ensure equal lengths
    L = min(len(token_strings), len(token_logprobs), len(text_offset))
    token_strings = token_strings[:L]
    token_logprobs = token_logprobs[:L]
    text_offset = text_offset[:L]

    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
    if want_top:
        # If lp is None for a position, top_logprobs[pos] must be None too.
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
    request_id = f"req_{uuid.uuid4().hex}"

    # First, normalize the prompts so we can use them for both logging and processing.
    prompts = _normalize_prompts(request.prompt)

    if DEBUG_MODE:
        try:
            # Create a dictionary from the request model.
            loggable_request = request.model_dump()
            # OVERWRITE the 'prompt' field with our newly normalized string version.
            loggable_request["prompt"] = prompts
            
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "event": "request_received",
                "request": loggable_request,  # Use the modified dictionary here
            }
            with open(DEBUG_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error writing request to debug log file: {e}")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this server.")

    if len(prompts) == 0:
        raise HTTPException(status_code=400, detail="Empty prompt list.")

    start = time.time()
    try:
        # The `prompts` variable is already calculated and ready to use.
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
            want_top = (request.logprobs or 0) > 0
            lp_payload = _to_openai_logprobs(getattr(item, "logprobs", None), want_top=want_top)
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

    response = CompletionResponse(
        model=request.model,
        choices=choices,
        usage=usage,
    )

    if DEBUG_MODE:
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "event": "response_generated",
                "response": response.model_dump(),
            }
            with open(DEBUG_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            # Non-fatal: log to console and continue
            print(f"Error writing response to debug log file: {e}")

    return response


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Handles OpenAI-compatible chat completion requests.
    This is the standard endpoint for instruction-tuned models.
    """
    request_id = f"req_{uuid.uuid4().hex}"
    if not llm.has_chat_template:
        raise HTTPException(
            status_code=400,
            detail="This model does not have a chat template. Please use the legacy /v1/completions endpoint."
        )

    # Convert Pydantic models to a list of dictionaries for the tokenizer
    messages_for_template = [msg.model_dump() for msg in request.messages]

    try:
        # This is the key step: applying the model-specific chat template.
        formatted_prompt = llm.tokenizer.tokenizer.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True  # Ensures the model knows to generate a response
        )
        print("\n--- Applied Chat Template ---")
        print(formatted_prompt)
        print("---------------------------\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply chat template: {e}")

    if DEBUG_MODE:
        # Log the request, including the formatted prompt for easy debugging
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "event": "chat_request_received",
            "request": request.model_dump(),
            "formatted_prompt": formatted_prompt,
        }
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this server.")

    # Use the existing generator to get the model's reply
    completions = llm.generate_batch(
        prompts=[formatted_prompt],
        image_paths=None,
        max_tokens=request.max_tokens,
        logprobs=request.top_logprobs,
        echo=False, # Echo is not used in chat APIs
    )
    result = completions[0]

    # Apply stop sequences if any
    text_out, _, finish_reason = _apply_stops_to_text_and_logprobs(result.text, None, request.stop)
    if finish_reason is None:
        finish_reason = result.finish_reason

    # Calculate token usage
    prompt_tokens = _count_tokens(formatted_prompt)
    completion_tokens = _count_tokens(text_out)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    # Build the final response object
    response = ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text_out),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )

    if DEBUG_MODE:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "event": "chat_response_generated",
            "response": response.model_dump(),
        }
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return response


@app.get("/")
def health_check():
    return {"status": "ok", "message": "MaxText API server is running."}


if __name__ == "__main__":
    uvicorn.run("maxtext_server:app", host="0.0.0.0", port=8000, reload=False)
