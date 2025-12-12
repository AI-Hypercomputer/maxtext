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
This file defines the `GemeiniAgent` class, which acts as an interface for interacting 
with the Google Gemini model. It handles model initialization, configuration, and robust 
API calls with retry mechanisms. Please add your API key and chosen model name in .env file. 
"""
import logging
import os
import time

from dotenv import load_dotenv  # If this is not available, try ``pip install python-dotenv``

load_dotenv()

import backoff  # If this is not available, try ``pip install backoff``

from google.api_core.exceptions import DeadlineExceeded, InternalServerError, RetryError
from google import genai
from google.genai import types


logger = logging.getLogger("__name__")


class GeminiAgent:
  TEMPERATURE = 0.1
  MAX_OUTPUT_TOKENS = 50000

  def __init__(self, system_instruction):
    # 1. ENTERPRISE AUTH
    # The SDK will use your Service Account or gcloud login automatically.
    self.client = genai.Client(
        vertexai=True,
        project=os.environ.get("GCP_PROJECT_ID"),
        location=os.environ.get("REGION", "us-central1")
    )
    
    # Ensure this matches the full string: "gemini-2.5-flash-lite"
    self.model_name = os.environ["Model"]
    
    # System instruction is now part of the GenerateContentConfig
    self.config = types.GenerateContentConfig(
        temperature=self.TEMPERATURE,
        max_output_tokens=self.MAX_OUTPUT_TOKENS,
        system_instruction=system_instruction 
    )

  @backoff.on_exception(
      backoff.expo, (DeadlineExceeded, InternalServerError, RetryError, ValueError, TypeError), max_tries=5
  )
  def __call__(self, memory_list):
    """
    Generates content from the model based on the provided memory list.

    This method includes a retry mechanism to handle transient API errors.
    It also handles converting a single string message to the correct
    chat dictionary format.

    Args:
        memory_list (str | list): A single message string or a list of
                                        message dictionaries in the required
                                        model format.

    Returns:
        google.generativeai.types.GenerateContentResponse: The response from the model.
    """
    if isinstance(memory_list, str):
      contents = [types.Content(role="user", parts=[types.Part(text=memory_list)])]
    elif isinstance(memory_list, dict):
      # Convert old dict style to SDK types
      parts = memory_list.get("parts", [])
      if isinstance(parts, str):
          parts = [types.Part(text=parts)] # Wrap string in a list of Parts
      contents = [types.Content(role=memory_list.get("role", "user"), parts=parts)]
    else:
      contents = memory_list
    for _ in range(10):
      resp = self.client.models.generate_content(
          model=self.model_name,
          contents=memory_list,
          config=self.config
          )
      if hasattr(resp, "text"):
        return resp
      logger.error("Response not have text %s", resp)
      time.sleep(2)
    logger.error("Failed to get a valid response with 'text' attribute after multiple retries.")
    return None
