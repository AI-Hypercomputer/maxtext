# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file defines the `EmbeddingAgent` class, which acts as an interface for interacting
with the Google Gemini model. It handles model initialization, configuration, and robust
API calls with retry mechanisms. Please add your API key and chosen model name in .env file.

Example Invocation:

```python
from integrative_rag_agent.llm_rag_agent import EmbeddingAgent

# Initialize the EmbeddingAgent
embedding_agent = EmbeddingAgent()

# Generate an embedding for a given text
text_to_embed = "This is a sample text for embedding."
embedding = embedding_agent(text_to_embed)
print(f"Generated embedding: {embedding}")
```
"""

from dotenv import load_dotenv  # If this is not available, try ``pip install python-dotenv``

load_dotenv()
import logging
import backoff  # If this is not available, try ``pip install backoff``
from google import genai as google_genai  # If this is not available, try ``pip install google-genai==1.28.0``
from google.api_core.exceptions import DeadlineExceeded, InternalServerError, RetryError


logger = logging.getLogger("__name__")


class EmbeddingAgent:
  """
  A class to manage interactions with a Google Gemini model.

  This agent handles model configuration, API calls with retry logic,
  and response processing. It is designed to be instantiated with a
  system instruction and then called with a list of memory parts.
  """

  def __init__(self, system_instruction=None):
    """
    Initializes the GeminiAgent with a specific system instruction.

    Args:
        system_instruction (str): The system prompt to guide the model's behavior.
    """
    self.client = google_genai.Client()
    self.model = "gemini-embedding-001"

  @backoff.on_exception(
    backoff.expo, (DeadlineExceeded, InternalServerError, RetryError, ValueError, TypeError), max_tries=10
  )
  def __call__(self, text):
    return (
      self.client.models.embed_content(
        model=self.model,
        contents=text,
      )
      .embeddings[0]
      .values
    )
