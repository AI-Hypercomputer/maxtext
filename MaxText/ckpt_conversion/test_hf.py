from huggingface_hub import login
from typing import Sequence
import torch
import torch_xla.core.xla_model as xm # Import XLA model
import torch_xla.debug.metrics as mx
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse # Import argparse


def main(argv: Sequence[str]) -> None:

    hf_checkpoint_path = None
    for arg in argv:
        if arg.startswith("checkpoint_path="):
            hf_checkpoint_path = arg.split("=", 1)[1]
            break
    
    if hf_checkpoint_path is None:
        raise ValueError("checkpoint_path argument not found in argv. Expected format: 'checkpoint_path=/your/path'")


    device = xm.xla_device()
    print(f"Using device: {device}")

    # model_id = "google/gemma-2-2b-it"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    # model = AutoModelForCausalLM.from_pretrained(model_id, 
    #                                          torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_path) # Or use original_model_id if tokenizer files aren't in local_checkpoint_path
    model = AutoModelForCausalLM.from_pretrained(hf_checkpoint_path,
                                            torch_dtype=torch.bfloat16)


    model.to(device)
    input_text = "I love to eat"
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    # --- Print XLA metrics before generation ---
    # print("\n--- XLA Operation Metrics Report (Before Generation) ---")
    # print(mx.metrics_report())


    # --- Generate Output ---
    with torch.no_grad(): 
        outputs = model.generate(**input_ids, max_new_tokens=6)

    # print("\n--- XLA Operation Metrics Report (After Generation) ---")
    # print(mx.metrics_report())

    # --- Decode and Print ---
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
