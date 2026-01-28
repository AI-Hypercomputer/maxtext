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
This file defines the `GemeiniAgent` class, which acts as an interface for
interacting with the Google Gemini model. It handles model initialization,
configuration, and robust API calls with retry mechanisms. Please add your API
key and chosen model name in .env file.
"""
import logging
import os
import time

from dotenv import load_dotenv  # If this is not available, try ``pip install python-dotenv``

load_dotenv()

import backoff  # If this is not available, try ``pip install backoff``

import google.generativeai as genai  # If this is not available, try ``pip install google-generativeai``
from google.api_core.exceptions import DeadlineExceeded, InternalServerError, RetryError


logger = logging.getLogger("__name__")


class GeminiAgent:
  """
  A class to manage interactions with a Google Gemini model.

  This agent handles model configuration, API calls with retry logic,
  and response processing. It is designed to be instantiated with a
  system instruction and then called with a list of memory parts.
  """

  TEMPERATURE = 0.1
  MAX_OUTPUT_TOKENS = 50000
  generation_config = genai.GenerationConfig(
      temperature=TEMPERATURE,
      max_output_tokens=MAX_OUTPUT_TOKENS,
  )

  def __init__(self, system_instruction):
    """
    Initializes the GeminiAgent with a specific system instruction.

    Args:
        system_instruction (str): The system prompt to guide the model's
          behavior.
    """
    self.client = genai.GenerativeModel(
        os.environ["Model"], generation_config=self.generation_config, system_instruction=system_instruction
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
      memory_list (str | list): A single message string or a list of message
        dictionaries in the required model format.

    Returns:
      google.generativeai.types.GenerateContentResponse
        The response from the model.
    """
    if isinstance(memory_list, str):
      memory_list = {"role": "user", "parts": memory_list}
    for _ in range(10):
      resp = self.client.generate_content(memory_list, stream=False)
      if hasattr(resp, "text"):
        return resp
      logger.error("Response not have text %s", resp)
      time.sleep(2)
    logger.error("Failed to get a valid response with 'text' attribute after multiple retries.")
    return None
