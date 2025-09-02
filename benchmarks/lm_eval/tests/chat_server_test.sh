curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
    -d '{
    "model": "maxtext-model",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "max_tokens": 50,
    "temperature": 0.7
}'