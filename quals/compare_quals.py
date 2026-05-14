import re
import sys

def parse_responses(log_path):
    responses = {}
    current_prompt = None
    
    # Regular expression to extract prompt and its generated response
    # Match: Input `PROMPT` -> `RESPONSE`
    pattern = re.compile(r"Input `([^`]+)` -> `([^`]+)`", re.DOTALL)
    
    with open(log_path, "r") as f:
        content = f.read()
        
    matches = pattern.findall(content)
    for prompt, response in matches:
        prompt = prompt.strip()
        # Format output nicely (remove double newlines, escape vertical bars for markdown tables)
        response = response.strip().replace("\n", "<br>").replace("|", "\\|")
        responses[prompt] = response
        
    return responses

def main():
    sft_file = "quals/logs/sft_responses.log"
    dpo_file = "quals/logs/dpo_responses.log"
    
    sft_res = parse_responses(sft_file)
    dpo_res = parse_responses(dpo_file)
    
    print("# Qualitative Decoding Comparison")
    print("\n| Prompt | SFT Baseline Response | DPO Fine-Tuned Response |")
    print("| --- | --- | --- |")
    
    # Prompts in order of execution
    prompts = [
        "Explain the concept of Direct Preference Optimization in simple terms.",
        "Write a short story about a robot learning to cook.",
        "What are the pros and cons of using JAX for machine learning?",
        "How do I optimize a MaxText training run on TPU v4-8?",
        "Give me a recipe for a healthy vegetarian dinner."
    ]
    
    for prompt in prompts:
        sft_out = sft_res.get(prompt, "N/A")
        dpo_out = dpo_res.get(prompt, "N/A")
        # Escape vertical bars in prompt
        p_esc = prompt.replace("|", "\\|")
        print(f"| {p_esc} | {sft_out} | {dpo_out} |")

if __name__ == "__main__":
    main()
