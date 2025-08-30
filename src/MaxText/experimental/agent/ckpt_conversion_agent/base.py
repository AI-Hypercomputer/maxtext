# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A base class for other agents
"""

from google import genai
from google.genai.types import GenerateContentConfig

MODEL_ID = "gemini-2.5-pro"


class BaseAgent:
  """A base class for agents that provides text generation capabilities."""

  def __init__(self, api_key, model_id=MODEL_ID):
    """
    Initializes the BaseAgent with a genai client.

    Args:
        client: An initialized genai.Client object.
    """
    if not api_key:
      raise ValueError("A valid api_key must be provided.")
    client = genai.Client(api_key=api_key)
    self.client = client
    self.model_id = model_id

  def generate_text(self, prompt, tools=None):
    """
    Generates text using the provided prompt and tools.

    Args:
        prompt (str): The input prompt for the model.
        tools: Optional tools for function calling.

    Returns:
        The generated text as a string.
    """
    # Prepare the configuration for the API call
    config = GenerateContentConfig(tools=tools) if tools else None

    # Call the model to generate content
    response = self.client.models.generate_content(model=self.model_id, contents=prompt, config=config)
    return response.text
