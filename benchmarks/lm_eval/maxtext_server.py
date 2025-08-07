import sys
import os
import time
import uuid
from typing import List, Optional, Union, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Assuming the MaxTextGenerator class is in a file named maxtext_generator.py
from benchmarks.lm_eval.maxtext_generator import MaxTextGenerator

def get_maxtext_args_from_env():
    """
    Provides a base list of arguments for MaxText's pyconfig.
    pyconfig expects an argv-style list where the first element is a placeholder
    for the script name. All other configs are loaded from the YAML file.
    """
    args = [
        "maxtext_server.py",         # Placeholder script name
        "MaxText/configs/base.yml",
    ]
    return args

# --- 1. Initialize FastAPI App and MaxTextGenerator ---

print("Starting server and initializing MaxTextGenerator...")
llm = MaxTextGenerator(get_maxtext_args_from_env())
app = FastAPI()


# --- 2. Define OpenAI-compatible Pydantic Models ---

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    # max_tokens is accepted for API compatibility but will be ignored by the server.
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False

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


# --- 3. Create the API Endpoint ---

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

    # Run batch inference. The max generation length is now controlled
    # entirely by the MaxTextGenerator's internal configuration.
    try:
        generated_texts = llm.generate_batch(prompts=prompts)
    except Exception as e:
        print(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}"
        )

    # Clean up whitespace from the outputs.
    outputs = [text.strip() for text in generated_texts]

    end_time = time.time()
    print(f"Request processed in {end_time - start_time:.2f} seconds.")

    # Format as OpenAI CompletionResponse
    response_choices = []
    for i, text in enumerate(outputs):
        choice = CompletionChoice(
            text=text,
            index=i,
            # This is now more accurate, as the backend handles stopping.
            finish_reason="stop",
        )
        response_choices.append(choice)

    return CompletionResponse(model=request.model, choices=response_choices)


@app.get("/")
def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok", "message": "MaxText API server is running."}


# --- 4. Instructions for Running the Server ---

if __name__ == "__main__":
    print("\nTo run the server, use the following command (replace ... with your model args):")
    print("python -m uvicorn maxtext_server:app --host 0.0.0.0 --port 8000 -- run_name=... [maxtext_args]")