import concurrent.futures
import requests
import time

URL = "http://localhost:8000/v1/completions"
HEADERS = {"Content-Type": "application/json"}

PROMPTS = [
    """The following are multiple choice questions about history.

Question: In what year did the French Revolution begin?
A. 1776
B. 1789
C. 1793
D. 1804
Answer: B

Question: Which French monarch was known as the "Sun King"?
A. Louis XIV
B. Louis XVI
C. Henry IV
D. Francis I
Answer: A

Question: Napoleon Bonaparte was exiled to which island first?
A. Elba
B. Corsica
C. Malta
D. Sicily
Answer:""",

    """The following are multiple choice questions about science.

Question: What is the chemical symbol for gold?
A. Au
B. Ag
C. Go
D. Gd
Answer: A

Question: What planet is known as the Red Planet?
A. Venus
B. Mars
C. Jupiter
D. Saturn
Answer: B

Question: What is the hardest natural substance on Earth?
A. Platinum
B. Diamond
C. Graphene
D. Quartz
Answer:""",

    """The following are multiple choice questions about geography.

Question: Which is the longest river in the world?
A. Amazon
B. Nile
C. Yangtze
D. Mississippi
Answer: B

Question: What is the capital of Japan?
A. Kyoto
B. Osaka
C. Tokyo
D. Seoul
Answer: C

Question: Which desert is the largest hot desert in the world?
A. Gobi
B. Mojave
C. Sahara
D. Kalahari
Answer:""",

    """The following are multiple choice questions about literature.

Question: Who wrote "To Kill a Mockingbird"?
A. Mark Twain
B. Harper Lee
C. John Steinbeck
D. F. Scott Fitzgerald
Answer: B

Question: In which play does the character Hamlet appear?
A. Macbeth
B. Othello
C. Hamlet
D. King Lear
Answer: C

Question: Who is the author of "1984"?
A. George Orwell
B. Aldous Huxley
C. Ray Bradbury
D. H.G. Wells
Answer:"""
]

def send_request(req_id, prompt):
    print(f"[{req_id}] Sending completion request...")
    start_time = time.time()
    
    payload = {
        "model": "deepseekv4",
        "prompt": prompt,
        "max_tokens": 5,
        "temperature": 0.0,
        "top_p": 1.0
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
    print(f"Testing batching by firing {num_requests} concurrent requests for text completions...")
    
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
                    text_out = choice.get("text", "")
                    
                    print(f"\n--- Result for Request [{ret_req_id}] ---")
                    # Print the last few lines of the prompt for context
                    prompt_tail = "\n".join(prompt.strip().split("\n")[-5:])
                    print(f"Prompt (tail):\n...\n{prompt_tail}")
                    print(f"Model Output: {text_out.strip()}")
            except Exception as exc:
                print(f"[{req_id}] Generated an exception: {exc}")

if __name__ == "__main__":
    test_batching()
