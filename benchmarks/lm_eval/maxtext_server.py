import sys
import time
import uuid
from typing import List, Optional, Union, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from maxtext_generator import MaxTextGenerator


# --- 1. Initialize FastAPI App and MaxTextGenerator ---

print("Starting server and initializing MaxTextGenerator...")
llm = MaxTextGenerator(sys.argv)
app = FastAPI()


# --- 2. Define OpenAI-compatible Pydantic Models ---

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False
    # You can add other OpenAI parameters as needed (e.g., top_p, logprobs, stop, etc.)

class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]


# --- 3. Utility Functions ---

def get_prompt_lengths(prompts: List[str], tokenizer) -> List[int]:
    """Tokenize prompts and get their lengths (excluding any BOS token added by tokenizer)."""
    lengths = []
    for prompt in prompts:
        # returns (tokens, length)
        tokens, true_length = tokenizer.encode(prompt, is_bos=False)
        lengths.append(true_length)
    return lengths


# --- 4. Create the API Endpoint ---

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Handles requests to the /v1/completions endpoint (OpenAI-compatible).
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not currently supported by this server.",
        )

    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    print(f"\nReceived completion request for {len(prompts)} prompt(s).")
    start_time = time.time()

    # Compute per-prompt max_target_length: prompt length + max_tokens
    try:
        prompt_lengths = get_prompt_lengths(prompts, llm.tokenizer)
        max_target_lengths = [pl + request.max_tokens for pl in prompt_lengths]
    except Exception as e:
        print(f"Tokenization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tokenization failed: {e}"
        )

    # Run batch inference (handle chunking inside generate_batch)
    try:
        generated_texts = llm.generate_batch(
            prompts=prompts,
            max_target_length=max(max_target_lengths)  # Use max; all outputs will be trimmed to requested length
        )
    except Exception as e:
        print(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}"
        )

    # For each output, truncate to requested max_tokens if over-generated
    # (Since batch must run with a single max_target_length)
    outputs = []
    for i, (text, prompt_len) in enumerate(zip(generated_texts, prompt_lengths)):
        # Re-tokenize output and trim if needed (robust to tokenizer differences)
        out_tokens, _ = llm.tokenizer.encode(text, is_bos=False)
        if len(out_tokens) > request.max_tokens:
            # Decode only up to max_tokens
            out_tokens = out_tokens[:request.max_tokens]
            trimmed_text = llm.tokenizer.decode(out_tokens)
        else:
            trimmed_text = text
        outputs.append(trimmed_text.strip())

    end_time = time.time()
    print(f"Request processed in {end_time - start_time:.2f} seconds.")

    # Format as OpenAI CompletionResponse
    response_choices = []
    for i, text in enumerate(outputs):
        choice = CompletionChoice(
            text=text,
            index=i,
            finish_reason="stop",  # If you add EOS/max_length detection, change this accordingly
        )
        response_choices.append(choice)

    return CompletionResponse(model=request.model, choices=response_choices)


@app.get("/")
def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok", "message": "MaxText API server is running."}


# --- 5. Instructions for Running the Server ---

if __name__ == "__main__":
    print("\nTo run the server, use the following command (replace ... with your model args):")
    print("python -m uvicorn maxtext_server:app --host 0.0.0.0 --port 8000 -- run_name=... [maxtext_args]")
    print("Or, for basic testing:")
    print("uvicorn maxtext_server:app --reload --host 127.0.0.1 --port 8000 -- run_name=... [maxtext_args]")

