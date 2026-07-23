import concurrent.futures
import requests
import time
import json

URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

PROMPTS = [
    "What is attention in transformers?",
    "Explain the concept of backpropagation.",
    "How does a Convolutional Neural Network work?",
    "What are the benefits of using a learning rate scheduler?"
]

def send_request(req_id, prompt):
    print(f"[{req_id}] Sending request: '{prompt}'")
    start_time = time.time()
    
    payload = {
        "model": "deepseekv4",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "thinking_mode": "chat",
        "max_tokens": 20,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(URL, json=payload, headers=HEADERS)
        duration = time.time() - start_time
        print(f"[{req_id}] Received response in {duration:.2f}s. Status: {response.status_code}")
        return req_id, prompt, response.json()
    except Exception as e:
        print(f"[{req_id}] Request failed: {e}")
        return req_id, prompt, None

def test_batching():
    num_requests = len(PROMPTS)
    print(f"Testing batching by firing {num_requests} concurrent requests with different prompts...")
    
    # Use a ThreadPoolExecutor to fire requests simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Submit all tasks at once
        futures = {executor.submit(send_request, i, prompt): i for i, prompt in enumerate(PROMPTS)}
        
        # Wait for all of them to finish
        for future in concurrent.futures.as_completed(futures):
            req_id = futures[future]
            try:
                ret_req_id, prompt, data = future.result()
                if data and "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")
                    
                    print(f"\n--- Result for Request [{ret_req_id}] ---")
                    print(f"Prompt: {prompt}")
                    if reasoning:
                        print(f"Thinking: {reasoning.strip()}")
                    print(f"Output: {content.strip()}")
            except Exception as exc:
                print(f"[{req_id}] Generated an exception: {exc}")

if __name__ == "__main__":
    test_batching()
