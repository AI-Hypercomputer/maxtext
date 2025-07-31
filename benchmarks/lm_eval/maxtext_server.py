import os
import time
import uuid
import json
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException

from benchmarks.lm_eval.maxtext_generator import MaxTextGenerator
from benchmarks.lm_eval.server_models import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    Usage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
)
from benchmarks.lm_eval import server_utils

# ----------------------------
# Init
# ----------------------------

print("Starting server and initializing MaxTextGenerator...")
llm = MaxTextGenerator(server_utils.get_maxtext_args())
app = FastAPI()

if server_utils.DEBUG_MODE:
    os.makedirs(os.path.dirname(server_utils.DEBUG_LOG_FILE), exist_ok=True)
    print(f"DEBUG MODE IS ENABLED. Requests and responses will be logged to {server_utils.DEBUG_LOG_FILE}")

# ----------------------------
# Endpoint
# ----------------------------

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    request_id = f"req_{uuid.uuid4().hex}"
    prompts = server_utils.normalize_prompts(request.prompt, llm)

    # Improved debug logging
    loggable_request = request.model_dump()
    loggable_request["prompt"] = prompts
    server_utils.log_debug_event(request_id, "request_received", {"endpoint": "/v1/completions", "request_data": loggable_request})

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this server.")

    if len(prompts) == 0:
        raise HTTPException(status_code=400, detail="Empty prompt list.")

    start = time.time()
    try:
        completions = llm.generate_batch(
            prompts=prompts,
            image_paths=None,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            echo=request.echo,
            stop=request.stop,
            temperature=request.temperature,
            seed=request.seed,
        )
    except Exception as e:
        server_utils.log_debug_event(request_id, "inference_error", {"detail": str(e)})
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    
    duration = time.time() - start
    print(f"/v1/completions processed {len(prompts)} prompt(s) in {duration:.2f}s")

    choices: list[CompletionChoice] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for idx, prompt in enumerate(prompts):
        item = completions[idx]
        if hasattr(item, "text"):
            text_out = item.text if isinstance(item.text, str) else str(item.text)
            want_top = (request.logprobs or 0) > 0
            lp_payload = server_utils.to_openai_logprobs(getattr(item, "logprobs", None), llm, want_top=want_top)
            finish_reason = getattr(item, "finish_reason", "stop")
        else:
            text_out = str(item)
            lp_payload = None
            finish_reason = "stop"

        text_out, lp_payload, stop_reason = server_utils.apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
        if stop_reason is not None:
            finish_reason = stop_reason

        prompt_token_count = server_utils.count_tokens(prompt, llm)
        completion_token_count = server_utils.count_tokens(text_out, llm)
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

    server_utils.log_debug_event(request_id, "response_generated", {"response_data": response.model_dump()})

    return response


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    request_id = f"req_{uuid.uuid4().hex}"
    if not llm.has_chat_template:
        raise HTTPException(
            status_code=400,
            detail="This model does not have a chat template. Please use the legacy /v1/completions endpoint."
        )

    messages_for_template = [msg.model_dump() for msg in request.messages]

    try:
        formatted_prompt = llm.tokenizer.tokenizer.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        server_utils.log_debug_event(request_id, "template_error", {"detail": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to apply chat template: {e}")

    # Improved debug logging
    server_utils.log_debug_event(
        request_id,
        "request_received",
        {
            "endpoint": "/v1/chat/completions",
            "request_data": request.model_dump(),
            "formatted_prompt": formatted_prompt,
        },
    )

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this server.")

    completions = llm.generate_batch(
        prompts=[formatted_prompt],
        image_paths=None,
        max_tokens=request.max_tokens,
        logprobs=request.top_logprobs,
        echo=False,
        stop=request.stop,
        temperature=request.temperature,
        seed=request.seed,
    )
    result = completions[0]

    text_out, _, finish_reason = server_utils.apply_stops_to_text_and_logprobs(result.text, None, request.stop)
    if finish_reason is None:
        finish_reason = result.finish_reason

    prompt_tokens = server_utils.count_tokens(formatted_prompt, llm)
    completion_tokens = server_utils.count_tokens(text_out, llm)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

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

    server_utils.log_debug_event(request_id, "response_generated", {"response_data": response.model_dump()})

    return response


@app.get("/")
def health_check():
    return {"status": "ok", "message": "MaxText API server is running."}
